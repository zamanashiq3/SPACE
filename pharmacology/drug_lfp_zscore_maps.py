# -*- coding: utf-8 -*-
"""
drug_lfp_zscore_maps.py -- SPACE framework: drug-normalized spatial LFP maps (robust z relative to the
same slice's 4-AP baseline) and the network-state deviation per animal and drug.

Manuscript Methods section: "Drug-normalized spatial maps".
Manuscript figure(s): Fig. 6A (per-drug robust-Z maps), Fig. 6B (network-state deviation).

Inputs:
    data/processed_2ms/<group>/<name>.h5   (schema of preprocessing/brw_to_2ms_h5.py)
        --map_key (default /lfp/metrics/rms_map): the stored per-electrode LFP amplitude summary
        [64, 64]; with --compute_from_ts the map is instead the mean |LFP| per electrode, streamed
        in time chunks from /lfp/ts_2ms_full.
    data/masks/<slice_id>__contact_mask_64x64.npy   (matched by slice prefix in the file name,
                                                     else /mask in the H5, else --mask override)
Outputs (under data/results/drug_lfp_zscore_maps/):
    zmaps/<group>__<slice_id>__<condition>_zmap.npy          robust-Z map [H, W], NaN outside the mask
    zmaps/<group>__<slice_id>__<condition>_cleaned.npy       cleaned amplitude map X (input to Z)
    figures/<group>__<slice_id>__<condition>_zmap.(png|svg)  coolwarm, symmetric +/- zlim, mask outline
    figures/group_median_zmaps.(png|svg)                     groups x drugs panel of electrode-wise
                                                             median Z maps across the slices of a group
    group_median_zmaps/<group>__<condition>_median_zmap.npy
    network_state_deviation.csv   one row per (group, slice_id, condition): median_z, mean_z,
                                  n_electrodes, is_baseline  -- the animal-level table for the LMM
    figures/network_state_deviation.(png|svg)                bar (group mean) + strip (slices) per drug
    manifest.json                 every parameter used

Algorithm (as in the Methods):
    1. For each slice, the per-electrode map of LFP amplitude of each drug recording, X_d(x, y).
    2. Pre-cleaning of every summary map (and only of these maps):
       a) residual row and column striping: subtract the row-wise median (over tissue-contact
          electrodes of that row), then the column-wise median (over tissue-contact electrodes of
          that column, recomputed on the row-corrected map);
       b) large-scale background: subtract a mask-aware Gaussian-smoothed estimate with
          sigma = 8 electrodes (normalized convolution G(mask * X) / G(mask), so electrodes outside
          the tissue never leak into the estimate).
    3. Robust z relative to the same slice's 4-AP baseline map:
           Z_d(x, y) = (X_d(x, y) - median_Omega(X_4AP)) / (1.4826 * MAD_Omega(X_4AP)),
       with median and MAD scalars over the tissue-contact electrodes Omega of the cleaned 4-AP map;
       the 4-AP map itself therefore has median Z = 0 (computed and stored as a sanity check).
    4. Network-state deviation = median of Z_d over Omega for each slice (animal) and drug.

Implementation notes:
    - "Mean LFP amplitude" (Methods) is read from the stored per-electrode summary map
      /lfp/metrics/rms_map (RMS of the 2 ms-smoothed LFP over the whole recording, the amplitude
      summary written by preprocessing/brw_to_2ms_h5.py). --map_key selects another stored map
      (e.g. /lfp/metrics/mean_map, p2p_map) and --compute_from_ts computes the literal time-mean of
      |LFP| per electrode by streaming /lfp/ts_2ms_full in time chunks (never the whole array).
    - Striping removal uses plain medians, not a smoothed row baseline as in the original prototype;
      the Methods specify medians. Rows / columns without any tissue-contact electrode get a
      correction of 0 (nothing to estimate). Row medians are removed first, then column medians.
    - The background estimate is evaluated at every electrode (also outside the mask, where
      G(mask) > 0) but the cleaned map is kept only on the mask; outside-mask electrodes are NaN
      and never enter medians, MADs or figures. Boundary mode of the Gaussian is "constant" (zero
      padding), which the normalized convolution corrects for.
    - Level preservation (default on, --no_preserve_level to disable): both cleaning steps are
      level-removing (row/column medians and a sigma = 8 background both absorb the tissue-wide
      amplitude), so a literally cleaned map has median ~0 whatever the drug, and the network-state
      deviation (median Z of the drug map) would be ~0 by construction. Cleaning is therefore
      treated as flat-fielding: after the two steps the tissue-wide median of the raw map
      (over Omega) is added back, so only spatial structure (stripes, slow background) is removed
      and the drug-induced change of overall amplitude relative to 4-AP is retained in Z. The 4-AP
      map still has median Z = 0 exactly. Set --bg_sigma 0 to skip the background step.
    - MAD is the raw median absolute deviation; the 1.4826 factor is applied as in the Methods. A
      MAD < 1e-12 (degenerate baseline map) yields NaN Z maps with a warning rather than inf.
    - Group-level "median Z maps" are electrode-wise nanmedians across the slices of a group; this
      assumes the arrays are aligned (same 64 x 64 grid, no registration). An electrode is shown only
      where at least one slice has a value.
    - Non-finite values in the raw map are treated as missing (excluded from the medians / MADs and
      from the Gaussian numerator and denominator, i.e. the local background ignores them).
    - --drugs lists the conditions plotted in the group panel and the deviation figure (default the
      four drugs); the 4-AP baseline is always processed and written to the CSV (is_baseline = 1,
      median_z == 0 by construction) so the table is self-checking.
    - A slice without a 4-AP recording is skipped with a message; a drug recording without a slice
      match (no mask prefix in the file name) falls back to the slice id parsed from the file name.
"""

import re
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
IN_ROOT = REPO_ROOT / "data" / "processed_2ms"
MASK_DIR = REPO_ROOT / "data" / "masks"
OUT_DIR = REPO_ROOT / "data" / "results" / "drug_lfp_zscore_maps"
MAP_KEY = "/lfp/metrics/rms_map"      # stored per-electrode LFP amplitude summary ("mean LFP amplitude")
BASELINE_TOKEN = "4AP"
DRUGS = ["AFDX", "BIC", "CNQX", "ML297"]
BG_SIGMA = 8.0                        # electrodes; large-scale background (Methods: sigma = 8)
ZLIM = 3.0                            # symmetric colour limits of the Z maps
MAD_SCALE = 1.4826
MAD_EPS = 1e-12
TIME_CHUNK = 5000                     # samples per streamed chunk for --compute_from_ts
FIG_DPI = 300
# --- END EDIT ---

LFP_TS_KEY = "/lfp/ts_2ms_full"
CONDITION_ORDER = ["4AP", "AFDX", "BIC", "CNQX", "ML297"]


# =============================================================================
# HELPERS (I/O)
# =============================================================================

def read_meta(f: h5py.File) -> Tuple[float, int, int]:
    m = f["/meta"].attrs
    fs = float(m.get("fs", 10000.0)); H = int(m.get("H", 64)); W = int(m.get("W", 64))
    return fs, H, W


def parse_condition(name: str) -> str:
    """Drug token in a file name -> one of 4AP, AFDX, BIC, CNQX, ML297, unknown."""
    n = name.lower()
    for tok, out in [("af-dx", "AFDX"), ("afdx", "AFDX"), ("4ap", "4AP"), ("bic", "BIC"), ("cnqx", "CNQX"), ("ml297", "ML297"),
                     ("baseline", "4AP"), ("base", "4AP"), ("veh", "4AP")]:   # 4-AP baseline recordings
        if tok in n:
            return out
    return "unknown"


def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    plt.close(fig)


def find_mask_file(h5_name: str, mask_dir: Path) -> Optional[Tuple[str, Path]]:
    """Longest slice prefix (<slice>__contact_mask_64x64.npy) contained in the H5 file name."""
    best = None
    for p in sorted(mask_dir.glob("*__contact_mask_64x64.npy")) if mask_dir.is_dir() else []:
        sid = p.name.split("__")[0]
        if sid and sid.lower() in h5_name.lower() and (best is None or len(sid) > len(best[0])):
            best = (sid, p)
    return best


def slice_id_of(h5_path: Path, mask_dir: Path) -> str:
    hit = find_mask_file(h5_path.name, mask_dir)
    return hit[0] if hit else re.split(r"_cor_|_4ap|_afdx|_af-dx|_bic|_cnqx|_ml297", h5_path.stem, flags=re.I)[0]


def load_mask(f: h5py.File, h5_path: Path, mask_dir: Path, mask_override: Optional[Path], H: int, W: int) -> Tuple[np.ndarray, str]:
    """Return (mask bool [H,W], source)."""
    hit = find_mask_file(h5_path.name, mask_dir)
    if mask_override is not None:
        m, src = np.load(mask_override), str(mask_override)
    elif hit is not None:
        m, src = np.load(hit[1]), str(hit[1])
    elif "/mask" in f:
        m, src = f["/mask"][...], "h5:/mask"
    else:
        warnings.warn(f"{h5_path.name}: no mask found -> all-True")
        m, src = np.ones((H, W)), "all-True"
    m = np.asarray(m).astype(bool)
    if m.shape != (H, W):
        m = m.reshape(H, W)
    return m, src


def load_amplitude_map(f: h5py.File, map_key: str, compute_from_ts: bool, H: int, W: int,
                       time_chunk: int = TIME_CHUNK) -> np.ndarray:
    """Per-electrode LFP amplitude map [H, W]: the stored summary map, or the streamed time-mean of |LFP|."""
    if not compute_from_ts and map_key in f:
        return np.asarray(f[map_key][...], dtype=np.float64).reshape(H, W)
    if not compute_from_ts:
        warnings.warn(f"{map_key} not in file -> computing mean |LFP| from {LFP_TS_KEY}")
    d = f[LFP_TS_KEY]
    n_ch, T = d.shape
    acc = np.zeros(n_ch, dtype=np.float64); cnt = np.zeros(n_ch, dtype=np.float64)
    for t0 in range(0, T, time_chunk):
        X = d[:, t0:min(t0 + time_chunk, T)].astype(np.float64)
        ok = np.isfinite(X)
        acc += np.where(ok, np.abs(X), 0.0).sum(axis=1); cnt += ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (acc / cnt).reshape(H, W)


# =============================================================================
# MAP CLEANING + ROBUST Z
# =============================================================================

def remove_striping(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract row-wise medians, then column-wise medians (medians over tissue-contact electrodes; NaN-safe)."""
    Y = np.where(mask & np.isfinite(X), X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN rows/cols -> NaN -> 0 correction
        row_med = np.nanmedian(Y, axis=1)
        Y = Y - np.nan_to_num(row_med, nan=0.0)[:, None]
        col_med = np.nanmedian(Y, axis=0)
        Y = Y - np.nan_to_num(col_med, nan=0.0)[None, :]
    return Y


def masked_gaussian(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Mask-aware Gaussian smoothing by normalized convolution: G(mask*img) / G(mask); NaN where G(mask) ~ 0."""
    valid = mask & np.isfinite(img)
    w = valid.astype(np.float64)
    num = gaussian_filter(np.where(valid, img, 0.0), sigma=sigma, mode="constant", cval=0.0)
    den = gaussian_filter(w, sigma=sigma, mode="constant", cval=0.0)
    out = np.full(img.shape, np.nan)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out


def remove_background(X: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Subtract the mask-aware Gaussian-smoothed (sigma electrodes) large-scale background; NaN outside the mask."""
    Y = X - masked_gaussian(X, mask, sigma)
    Y[~mask] = np.nan
    return Y


def clean_map(X: np.ndarray, mask: np.ndarray, sigma: float, preserve_level: bool = True) -> np.ndarray:
    """Striping + background removal; with preserve_level the tissue-wide median of the raw map is restored."""
    Y = remove_striping(X, mask)
    if sigma > 0:
        Y = remove_background(Y, mask, sigma)
    if preserve_level:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Y = Y + (np.nanmedian(np.where(mask, X, np.nan)) - np.nanmedian(Y))
    return Y


def robust_baseline_stats(X_base: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """(median, raw MAD) of the cleaned baseline map over tissue-contact electrodes."""
    v = X_base[mask & np.isfinite(X_base)]
    if v.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)))


def robust_z(X: np.ndarray, med: float, mad: float, mask: np.ndarray, scale: float = MAD_SCALE) -> np.ndarray:
    """Z = (X - med) / (scale * mad); NaN outside the mask (and everywhere if the MAD is degenerate)."""
    Z = np.full(X.shape, np.nan)
    if not (np.isfinite(mad) and mad > MAD_EPS):
        warnings.warn("degenerate baseline MAD -> Z map is NaN")
        return Z
    Z[mask] = (X[mask] - med) / (scale * mad)
    return Z


# =============================================================================
# FIGURES
# =============================================================================

def draw_zmap(ax, Z: np.ndarray, mask: np.ndarray, zlim: float, title: str = ""):
    Zp = np.ma.masked_invalid(np.where(mask, Z, np.nan))
    cmap = matplotlib.colormaps["coolwarm"].copy()
    cmap.set_bad((1, 1, 1, 0))
    im = ax.imshow(Zp, cmap=cmap, vmin=-zlim, vmax=zlim, interpolation="nearest", origin="upper")
    ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8)
    return im


def slice_zmap_figure(Z: np.ndarray, mask: np.ndarray, zlim: float, title: str, out_base: Path):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    im = draw_zmap(ax, Z, mask, zlim, title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="robust Z vs 4-AP")
    fig.tight_layout()
    save_fig(fig, out_base)


def group_panel_figure(median_maps: Dict[Tuple[str, str], np.ndarray], groups: List[str], drugs: List[str],
                       n_slices: Dict[Tuple[str, str], int], zlim: float, out_base: Path):
    fig, axes = plt.subplots(len(groups), len(drugs), figsize=(2.4 * len(drugs) + 0.8, 2.4 * len(groups)), squeeze=False)
    im = None
    for i, g in enumerate(groups):
        for j, d in enumerate(drugs):
            ax = axes[i, j]
            M = median_maps.get((g, d))
            if M is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.5")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.set_title(f"{g} | {d} (n=0)", fontsize=8)
                continue
            im = draw_zmap(ax, M, np.isfinite(M), zlim, f"{g} | {d} (n={n_slices[(g, d)]})")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="median robust Z vs 4-AP")
    fig.suptitle("Group median Z maps (electrode-wise median across slices)", fontsize=10)
    save_fig(fig, out_base)


def deviation_figure(df: pd.DataFrame, drugs: List[str], out_base: Path):
    """Bar = group mean of the per-slice median Z, points = slices, per drug."""
    groups = sorted(df["group"].unique())
    sub = df[df["condition"].isin(drugs)]
    fig, ax = plt.subplots(figsize=(1.4 * len(drugs) + 2.0, 3.2))
    width = 0.8 / max(len(groups), 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for gi, g in enumerate(groups):
        xs, means, sems = [], [], []
        for di, d in enumerate(drugs):
            v = sub[(sub["group"] == g) & (sub["condition"] == d)]["median_z"].dropna().to_numpy()
            x = di + (gi - (len(groups) - 1) / 2) * width
            xs.append(x); means.append(v.mean() if v.size else np.nan)
            sems.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0)
            if v.size:
                jit = (np.random.rand(v.size) - 0.5) * width * 0.5
                ax.scatter(np.full(v.size, x) + jit, v, s=14, color="k", zorder=3, alpha=0.8)
        ax.bar(xs, means, width=width * 0.9, yerr=sems, color=colors[gi % len(colors)], alpha=0.6, capsize=2,
               label=g, zorder=2)
    ax.axhline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(range(len(drugs))); ax.set_xticklabels(drugs)
    ax.set_ylabel("Network-state deviation\n(median robust Z vs 4-AP)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_base)


# =============================================================================
# PER SLICE
# =============================================================================

def collect_slices(in_root: Path, mask_dir: Path) -> Dict[Tuple[str, str], Dict[str, Path]]:
    """{(group, slice_id): {condition: h5_path}} over all recordings under in_root."""
    slices: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for p in sorted(in_root.rglob("*.h5")):
        group = p.parent.name if p.parent != in_root else in_root.name
        cond = parse_condition(p.name)
        if cond == "unknown":
            print(f"[SKIP] {p.name}: no drug token in the file name")
            continue
        key = (group, slice_id_of(p, mask_dir))
        if cond in slices.setdefault(key, {}):
            warnings.warn(f"{key}: duplicate recording for {cond}; keeping {slices[key][cond].name}")
            continue
        slices[key][cond] = p
    return slices


def process_slice(group: str, slice_id: str, recs: Dict[str, Path], args, mask_dir: Path, out_dir: Path):
    """Clean every map of the slice, z-score against its 4-AP map; returns (rows, {cond: Z}, mask)."""
    base_tok = args.baseline_token.upper()
    if base_tok not in recs:
        print(f"[SKIP] {group}/{slice_id}: no {base_tok} baseline recording (have {sorted(recs)})")
        return [], {}, None
    mask = None
    cleaned: Dict[str, np.ndarray] = {}
    for cond, p in recs.items():
        with h5py.File(p, "r") as f:
            fs, H, W = read_meta(f)
            if mask is None:
                mask, mask_src = load_mask(f, p, mask_dir, args.mask, H, W)
                print(f"[SLICE] {group}/{slice_id}: mask={mask.sum()} el ({mask_src}); conditions {sorted(recs)}")
            X = load_amplitude_map(f, args.map_key, args.compute_from_ts, H, W)
        cleaned[cond] = clean_map(X, mask, args.bg_sigma, not args.no_preserve_level)
        np.save(out_dir / "zmaps" / f"{group}__{slice_id}__{cond}_cleaned.npy", cleaned[cond].astype(np.float32))
    med, mad = robust_baseline_stats(cleaned[base_tok], mask)
    rows, zmaps = [], {}
    for cond in [c for c in CONDITION_ORDER if c in cleaned]:
        Z = robust_z(cleaned[cond], med, mad, mask)
        zmaps[cond] = Z
        v = Z[mask & np.isfinite(Z)]
        rows.append(dict(group=group, slice_id=slice_id, condition=cond, file=recs[cond].name,
                         is_baseline=int(cond == base_tok), median_z=float(np.median(v)) if v.size else np.nan,
                         mean_z=float(np.mean(v)) if v.size else np.nan, n_electrodes=int(v.size),
                         baseline_median=med, baseline_mad=mad))
        np.save(out_dir / "zmaps" / f"{group}__{slice_id}__{cond}_zmap.npy", Z.astype(np.float32))
        slice_zmap_figure(Z, mask, args.zlim, f"{group} | {slice_id} | {cond}\nmedian Z = {rows[-1]['median_z']:.3f}",
                          out_dir / "figures" / f"{group}__{slice_id}__{cond}_zmap")
    print("        " + "; ".join(f"{r['condition']}: medZ={r['median_z']:.3f} meanZ={r['mean_z']:.3f}" for r in rows)
          + f"  (baseline med={med:.4g}, MAD={mad:.4g})")
    return rows, zmaps, mask


# =============================================================================
# CLI / MAIN
# =============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Drug-normalized spatial LFP maps: robust Z of each drug's per-electrode "
                                            "amplitude map relative to the same slice's 4-AP map, and the "
                                            "network-state deviation (median Z) per animal and drug.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in_root", default=str(IN_ROOT), help="processed_2ms root (recursive; group = parent folder)")
    p.add_argument("--mask_dir", default=str(MASK_DIR), help="folder of <slice_id>__contact_mask_64x64.npy")
    p.add_argument("--out_dir", default=str(OUT_DIR), help="output folder")
    p.add_argument("--mask", default=None, help="override: one mask .npy used for every recording")
    p.add_argument("--map_key", default=MAP_KEY, help="stored per-electrode LFP amplitude map (H5 dataset)")
    p.add_argument("--compute_from_ts", action="store_true",
                   help=f"ignore --map_key and compute the time-mean of |LFP| per electrode by streaming {LFP_TS_KEY}")
    p.add_argument("--baseline_token", default=BASELINE_TOKEN, help="condition used as the baseline map")
    p.add_argument("--drugs", nargs="+", default=DRUGS, help="drug conditions shown in the group panel / deviation figure")
    p.add_argument("--bg_sigma", type=float, default=BG_SIGMA, help="Gaussian sigma (electrodes) of the background estimate")
    p.add_argument("--zlim", type=float, default=ZLIM, help="symmetric colour limit of the Z maps")
    p.add_argument("--no_preserve_level", action="store_true",
                   help="do NOT restore the tissue-wide median of the raw map after cleaning (then every cleaned map "
                        "has median ~0 and the network-state deviation is ~0 by construction)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (only the strip-plot jitter is random)")
    return p


def main():
    args = build_argparser().parse_args()
    np.random.seed(args.seed)
    args.mask = Path(args.mask) if args.mask else None
    args.drugs = [d.upper() for d in args.drugs]
    in_root, mask_dir, out_dir = Path(args.in_root), Path(args.mask_dir), Path(args.out_dir)
    slices = collect_slices(in_root, mask_dir)
    if not slices:
        raise SystemExit(f"no recordings with a drug token under {in_root}")
    for sub in ("zmaps", "figures", "group_median_zmaps"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows, all_z = [], {}          # all_z[(group, slice_id)] = {cond: Z}
    for (group, sid), recs in slices.items():
        try:
            r, z, _ = process_slice(group, sid, recs, args, mask_dir, out_dir)
            rows += r
            if z:
                all_z[(group, sid)] = z
        except Exception as e:
            print(f"[ERROR] {group}/{sid}: {type(e).__name__}: {e}")

    cols = ["group", "slice_id", "condition", "file", "is_baseline", "median_z", "mean_z", "n_electrodes",
            "baseline_median", "baseline_mad"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_dir / "network_state_deviation.csv", index=False)
    base_rows = df[df["is_baseline"] == 1]
    if len(base_rows):
        worst = float(np.nanmax(np.abs(base_rows["median_z"])))
        print(f"[CHECK] max |median Z| of the {args.baseline_token} maps = {worst:.2e} (expected 0)")

    # group-level electrode-wise median Z maps
    groups = sorted({g for g, _ in all_z})
    median_maps, n_slices = {}, {}
    for g in groups:
        for d in args.drugs:
            stack = [z[d] for (gg, _), z in all_z.items() if gg == g and d in z]
            if not stack:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                M = np.nanmedian(np.stack(stack), axis=0)
            median_maps[(g, d)] = M; n_slices[(g, d)] = len(stack)
            np.save(out_dir / "group_median_zmaps" / f"{g}__{d}_median_zmap.npy", M.astype(np.float32))
    if groups:
        group_panel_figure(median_maps, groups, args.drugs, n_slices, args.zlim, out_dir / "figures" / "group_median_zmaps")
    if len(df):
        deviation_figure(df, args.drugs, out_dir / "figures" / "network_state_deviation")

    manifest = dict(
        script="pharmacology/drug_lfp_zscore_maps.py", in_root=str(in_root), mask_dir=str(mask_dir), out_dir=str(out_dir),
        mask_override=str(args.mask) if args.mask else None,
        map_key=args.map_key, compute_from_ts=bool(args.compute_from_ts), time_chunk=TIME_CHUNK,
        amplitude_map=("time-mean of |LFP| per electrode streamed from " + LFP_TS_KEY) if args.compute_from_ts
                      else f"stored summary map {args.map_key} (Methods: 'mean LFP amplitude')",
        baseline_token=args.baseline_token.upper(), drugs=args.drugs, condition_order=CONDITION_ORDER,
        cleaning=dict(step1="subtract row-wise medians then column-wise medians over tissue-contact electrodes (NaN-safe)",
                      step2=f"subtract mask-aware Gaussian background, sigma={args.bg_sigma} electrodes "
                            "(normalized convolution G(mask*X)/G(mask), zero-padded boundary)",
                      applied_to="summary maps only", preserve_level=not args.no_preserve_level),
        robust_z=f"(X_drug - median_Omega(X_4AP)) / ({MAD_SCALE} * MAD_Omega(X_4AP)); scalar median/MAD over the mask",
        mad_scale=MAD_SCALE, mad_eps=MAD_EPS,
        network_state_deviation="median of Z over tissue-contact electrodes per slice x condition (mean_z also reported)",
        group_median_maps="electrode-wise nanmedian of the per-slice Z maps of a group (grids assumed aligned)",
        zlim=args.zlim, colormap="coolwarm", fig_dpi=FIG_DPI, seed=args.seed,
        n_slices=len(slices), n_slices_done=len(all_z), n_rows=len(df),
        slices={f"{g}/{s}": {c: p.name for c, p in recs.items()} for (g, s), recs in slices.items()},
        versions=dict(numpy=np.__version__, pandas=pd.__version__, h5py=h5py.__version__, matplotlib=matplotlib.__version__),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[DONE] {len(df)} slice x condition rows, {len(all_z)} slices -> {out_dir}")


if __name__ == "__main__":
    main()
