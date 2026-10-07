# -*- coding: utf-8 -*-
"""
condition_stereotypy.py -- SPACE framework: similarity of drug-evoked CSD patterns across conditions
(per-slice 5 x 5 Pearson correlation matrix of spatial CSD maps) and the stereotypy index.

Manuscript Methods section: "CSD correlation and inter-regional coupling" (paragraph "The similarity of
drug-evoked CSD patterns across conditions was assessed for each animal ...").
Manuscript figure(s): Fig. 7B (condition x condition correlation matrices and stereotypy index).

Inputs:
    data/processed_2ms/<group>/<name>.h5   (schema of preprocessing/brw_to_2ms_h5.py)
        --map_key (default /csd/metrics/std_map): one stored spatial CSD map [64, 64] per recording
        (alternatives /csd/metrics/max_map, /csd/metrics/min_map, /csd/metrics/mean_map)
    data/masks/<slice_id>__contact_mask_64x64.npy   (matched by slice prefix in the file name,
                                                     else /mask in the H5, else --mask override)
Outputs (under data/results/condition_stereotypy/):
    matrices/<group>__<slice_id>__corr_matrix.csv            5 x 5 Pearson r (NaN for missing conditions)
    matrices/<group>__<slice_id>__fisher_z_matrix.csv        atanh(r)
    figures/<group>__<slice_id>__corr_matrix.(png|svg)       annotated heatmap of r
    group_matrices/<group>__corr_matrix.csv                  group mean: mean of Fisher z over slices,
                                                             back-transformed with tanh (+ _n.csv counts)
    figures/group_corr_matrices.(png|svg)                    one annotated heatmap per group
    stereotypy_index.csv      one row per (group, slice_id): stereotypy_z_mean (mean Fisher z over the
                              available condition pairs), stereotypy_r (tanh of it), n_pairs,
                              n_conditions, conditions -- the animal-level table
    figures/stereotypy_index.(png|svg)                       bar (group mean) + strip (slices)
    manifest.json             every parameter used

Algorithm (as in the Methods):
    1. For each slice and each condition (4-AP, AFDX, BIC, CNQX, ML297) one spatial CSD map.
    2. Electrode set = tissue-contact electrodes excluding the outermost ring of the array (the CSD
       there is 0 by construction); electrodes non-finite in either map of a pair are dropped.
    3. Pearson r across those electrodes for every pair of conditions -> 5 x 5 matrix; conditions
       absent for a slice give NaN rows / columns.
    4. Stereotypy index = mean of the Fisher z-transformed correlations over the ten condition pairs
       (NaN pairs ignored; n_pairs reported); tanh(mean z) is reported alongside as a correlation.
    5. Group matrices: cell-wise mean of Fisher z over the slices of the group, back-transformed.

Implementation notes:
    - "Spatial CSD maps" (Methods) are the stored per-recording CSD summary maps; the default is the
      temporal standard deviation of the CSD per electrode (/csd/metrics/std_map), a sign-free measure
      of how strongly each electrode participates in the drug-evoked activity. --map_key selects the
      peak source (max_map) or sink (min_map) map instead; the choice is recorded in the manifest.
    - Pairs with fewer than 3 common electrodes or zero variance in either map give r = NaN.
    - Fisher z uses atanh with r clipped to +/-(1 - 1e-7) so an exact r = 1 cannot produce inf.
    - The stereotypy index is NaN for a slice with fewer than two conditions (no pair); the row is
      still written (n_pairs = 0) so the slice is visible in the table.
    - Heatmap annotations show r (2 decimals) and a "--" for NaN; colour range is -1..1 (coolwarm).
    - --conditions controls the matrix order and which conditions are considered (default the five
      of the Methods); a recording with another condition token is ignored with a message.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
IN_ROOT = REPO_ROOT / "data" / "processed_2ms"
MASK_DIR = REPO_ROOT / "data" / "masks"
OUT_DIR = REPO_ROOT / "data" / "results" / "condition_stereotypy"
MAP_KEY = "/csd/metrics/std_map"           # spatial CSD map per recording
CONDITIONS = ["4AP", "AFDX", "BIC", "CNQX", "ML297"]
MIN_ELECTRODES = 3
R_CLIP = 1.0 - 1e-7
FIG_DPI = 300
# --- END EDIT ---


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
    for tok, out in [("af-dx", "AFDX"), ("afdx", "AFDX"), ("4ap", "4AP"), ("bic", "BIC"), ("cnqx", "CNQX"), ("ml297", "ML297")]:
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


def inner_mask(mask: np.ndarray) -> np.ndarray:
    """Tissue-contact electrodes excluding the outermost ring of the array (CSD-based quantities)."""
    m = mask.copy()
    m[0, :] = m[-1, :] = False; m[:, 0] = m[:, -1] = False
    return m


def collect_slices(in_root: Path, mask_dir: Path, conditions: List[str]) -> Dict[Tuple[str, str], Dict[str, Path]]:
    """{(group, slice_id): {condition: h5_path}} over all recordings under in_root."""
    slices: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for p in sorted(in_root.rglob("*.h5")):
        group = p.parent.name if p.parent != in_root else in_root.name
        cond = parse_condition(p.name)
        if cond not in conditions:
            print(f"[SKIP] {p.name}: condition '{cond}' not in {conditions}")
            continue
        key = (group, slice_id_of(p, mask_dir))
        if cond in slices.setdefault(key, {}):
            warnings.warn(f"{key}: duplicate recording for {cond}; keeping {slices[key][cond].name}")
            continue
        slices[key][cond] = p
    return slices


# =============================================================================
# CORRELATION / STEREOTYPY
# =============================================================================

def pearson(a: np.ndarray, b: np.ndarray, min_n: int = MIN_ELECTRODES) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def fisher_z(r):
    r = np.clip(np.asarray(r, dtype=np.float64), -R_CLIP, R_CLIP)
    return np.arctanh(r)


def corr_matrix(maps: Dict[str, np.ndarray], electrodes: np.ndarray, conditions: List[str]) -> np.ndarray:
    """Pearson r between the maps of every pair of conditions over `electrodes` (flat indices); NaN if absent."""
    n = len(conditions)
    R = np.full((n, n), np.nan)
    vec = {c: maps[c].ravel()[electrodes] for c in conditions if c in maps}
    for i, ci in enumerate(conditions):
        for j, cj in enumerate(conditions):
            if ci in vec and cj in vec:
                R[i, j] = 1.0 if i == j else (R[j, i] if j < i else pearson(vec[ci], vec[cj]))
    return R


def stereotypy_index(R: np.ndarray) -> Tuple[float, float, int]:
    """(mean Fisher z over the off-diagonal pairs with a value, tanh of it, number of pairs)."""
    iu = np.triu_indices(R.shape[0], k=1)
    z = fisher_z(R[iu])
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan"), float("nan"), 0
    zm = float(z.mean())
    return zm, float(np.tanh(zm)), int(z.size)


# =============================================================================
# FIGURES
# =============================================================================

def draw_matrix(ax, R: np.ndarray, conditions: List[str], title: str, N: Optional[np.ndarray] = None):
    cmap = matplotlib.colormaps["coolwarm"].copy(); cmap.set_bad("0.92")
    im = ax.imshow(np.ma.masked_invalid(R), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    n = len(conditions)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(conditions, fontsize=8, rotation=45, ha="right"); ax.set_yticklabels(conditions, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            txt = "--" if not np.isfinite(v) else f"{v:.2f}"
            if N is not None and np.isfinite(v) and i != j:
                txt += f"\n(n={int(N[i, j])})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(v) and abs(v) > 0.6 else "black")
    ax.set_title(title, fontsize=9)
    return im


def slice_matrix_figure(R: np.ndarray, conditions: List[str], title: str, out_base: Path):
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = draw_matrix(ax, R, conditions, title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson r")
    fig.tight_layout()
    save_fig(fig, out_base)


def group_matrices_figure(mats: Dict[str, np.ndarray], counts: Dict[str, np.ndarray], conditions: List[str], out_base: Path):
    groups = sorted(mats)
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 0.8, 3.4), squeeze=False)
    im = None
    for ax, g in zip(axes[0], groups):
        im = draw_matrix(ax, mats[g], conditions, f"{g} (mean Fisher z, back-transformed)", counts[g])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="Pearson r")
    save_fig(fig, out_base)


def stereotypy_figure(df: pd.DataFrame, out_base: Path):
    """Bar = group mean of the per-slice stereotypy index (Fisher z), points = slices."""
    groups = sorted(df["group"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    for ax, col, lab in zip(axes, ["stereotypy_z_mean", "stereotypy_r"], ["Stereotypy index (mean Fisher z)", "tanh(index) (r scale)"]):
        for gi, g in enumerate(groups):
            v = df[df["group"] == g][col].dropna().to_numpy()
            mu = v.mean() if v.size else np.nan
            sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
            ax.bar(gi, mu, yerr=sem, width=0.6, color=colors[gi % len(colors)], alpha=0.6, capsize=3, zorder=2)
            if v.size:
                ax.scatter(gi + (np.random.rand(v.size) - 0.5) * 0.3, v, s=16, color="k", zorder=3, alpha=0.8)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
        ax.set_ylabel(lab); ax.axhline(0, color="0.6", lw=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Condition stereotypy per slice (n = slices per group)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_base)


# =============================================================================
# PER SLICE
# =============================================================================

def process_slice(group: str, slice_id: str, recs: Dict[str, Path], args, mask_dir: Path, out_dir: Path) -> Tuple[dict, np.ndarray]:
    maps, mask = {}, None
    for cond, p in recs.items():
        with h5py.File(p, "r") as f:
            fs, H, W = read_meta(f)
            if mask is None:
                mask, mask_src = load_mask(f, p, mask_dir, args.mask, H, W)
            if args.map_key not in f:
                raise KeyError(f"{p.name}: {args.map_key} not found")
            maps[cond] = np.asarray(f[args.map_key][...], dtype=np.float64).reshape(H, W)
    el = np.flatnonzero(inner_mask(mask).ravel())
    print(f"[SLICE] {group}/{slice_id}: {el.size} electrodes (mask {mask_src}, outer ring excluded); conditions {sorted(maps)}")
    R = corr_matrix(maps, el, args.conditions)
    zm, rm, n_pairs = stereotypy_index(R)
    tag = f"{group}__{slice_id}"
    pd.DataFrame(R, index=args.conditions, columns=args.conditions).to_csv(out_dir / "matrices" / f"{tag}__corr_matrix.csv")
    pd.DataFrame(fisher_z(R), index=args.conditions, columns=args.conditions).to_csv(out_dir / "matrices" / f"{tag}__fisher_z_matrix.csv")
    slice_matrix_figure(R, args.conditions, f"{group} | {slice_id}\nstereotypy z = {zm:.3f} (r = {rm:.3f}, {n_pairs} pairs)",
                        out_dir / "figures" / f"{tag}__corr_matrix")
    print(f"        stereotypy: mean z = {zm:.4f}, r = {rm:.4f}, {n_pairs} pairs")
    row = dict(group=group, slice_id=slice_id, condition="all", stereotypy_z_mean=zm, stereotypy_r=rm, n_pairs=n_pairs,
               n_conditions=len(maps), conditions=",".join(c for c in args.conditions if c in maps), n_electrodes=int(el.size))
    for i, ci in enumerate(args.conditions):
        for j in range(i + 1, len(args.conditions)):
            row[f"r_{ci}_{args.conditions[j]}"] = R[i, j]
    return row, R


# =============================================================================
# CLI / MAIN
# =============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Per-slice condition x condition Pearson correlation of spatial CSD maps "
                                            "and the stereotypy index (mean Fisher z over condition pairs).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in_root", default=str(IN_ROOT), help="processed_2ms root (recursive; group = parent folder)")
    p.add_argument("--mask_dir", default=str(MASK_DIR), help="folder of <slice_id>__contact_mask_64x64.npy")
    p.add_argument("--out_dir", default=str(OUT_DIR), help="output folder")
    p.add_argument("--mask", default=None, help="override: one mask .npy used for every recording")
    p.add_argument("--map_key", default=MAP_KEY, help="spatial CSD map dataset (e.g. /csd/metrics/std_map, max_map, min_map)")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, help="condition order of the matrix")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (only the strip-plot jitter is random)")
    return p


def main():
    args = build_argparser().parse_args()
    np.random.seed(args.seed)
    args.mask = Path(args.mask) if args.mask else None
    args.conditions = [c.upper() for c in args.conditions]
    in_root, mask_dir, out_dir = Path(args.in_root), Path(args.mask_dir), Path(args.out_dir)
    slices = collect_slices(in_root, mask_dir, args.conditions)
    if not slices:
        raise SystemExit(f"no recordings with a known condition under {in_root}")
    for sub in ("matrices", "figures", "group_matrices"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    rows, mats = [], {}
    for (group, sid), recs in slices.items():
        try:
            row, R = process_slice(group, sid, recs, args, mask_dir, out_dir)
            rows.append(row); mats[(group, sid)] = R
        except Exception as e:
            print(f"[ERROR] {group}/{sid}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stereotypy_index.csv", index=False)

    # group matrices: cell-wise mean of Fisher z over slices, back-transformed
    gmats, gcounts = {}, {}
    for g in sorted({g for g, _ in mats}):
        Zs = np.stack([fisher_z(R) for (gg, _), R in mats.items() if gg == g])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zmean = np.nanmean(Zs, axis=0)
        gmats[g] = np.tanh(zmean); gcounts[g] = np.isfinite(Zs).sum(axis=0)
        pd.DataFrame(gmats[g], index=args.conditions, columns=args.conditions).to_csv(out_dir / "group_matrices" / f"{g}__corr_matrix.csv")
        pd.DataFrame(gcounts[g], index=args.conditions, columns=args.conditions).to_csv(out_dir / "group_matrices" / f"{g}__corr_matrix_n.csv")
    if gmats:
        group_matrices_figure(gmats, gcounts, args.conditions, out_dir / "figures" / "group_corr_matrices")
    if len(df):
        stereotypy_figure(df, out_dir / "figures" / "stereotypy_index")

    manifest = dict(
        script="core/condition_stereotypy.py", in_root=str(in_root), mask_dir=str(mask_dir), out_dir=str(out_dir),
        mask_override=str(args.mask) if args.mask else None, map_key=args.map_key,
        spatial_csd_map=f"stored per-recording map {args.map_key} (Methods: 'spatial CSD maps')",
        conditions=args.conditions, electrodes="tissue-contact mask excluding the outermost ring; non-finite electrodes dropped per pair",
        correlation="Pearson r across electrodes per condition pair; NaN if a condition is missing, < %d electrodes or zero variance" % MIN_ELECTRODES,
        stereotypy_index="mean of atanh(r) over the available off-diagonal pairs (max 10); stereotypy_r = tanh(mean z)",
        r_clip=R_CLIP, group_matrices="cell-wise nanmean of Fisher z over slices, tanh back-transformed",
        fig_dpi=FIG_DPI, seed=args.seed, n_slices=len(slices), n_slices_done=len(mats),
        slices={f"{g}/{s}": {c: p.name for c, p in recs.items()} for (g, s), recs in slices.items()},
        versions=dict(numpy=np.__version__, pandas=pd.__version__, h5py=h5py.__version__, matplotlib=matplotlib.__version__),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[DONE] {len(df)} slices -> {out_dir}")


if __name__ == "__main__":
    main()
