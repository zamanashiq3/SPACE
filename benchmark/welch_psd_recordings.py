# -*- coding: utf-8 -*-
"""
welch_psd_recordings.py -- SPACE framework: conventional per-electrode Welch PSD tables of whole
recordings (the "psd_tables" consumed by benchmark/spectral_sensitivity_bh_fdr.py).

Manuscript Methods section: "Power spectral density" (Welch estimator, band definitions) and the
Statistics section's "Benjamini-Hochberg FDR across spectral comparisons" (conventional PSD
features of the sensitivity benchmark).
Manuscript figure(s): Supplementary Figure 8 (conventional-feature arm, via
benchmark/spectral_sensitivity_bh_fdr.py); per-recording mean PSD / band-power overview figures.

Inputs:
    data/processed_2ms/<group>/<name>.h5   (schema of preprocessing/brw_to_2ms_h5.py; stored fs)
    data/masks/<slice_id>__contact_mask_64x64.npy   (matched by slice prefix in the file name,
                                                     else /mask in the H5, else --mask override)
Outputs (flat, under data/psd_tables/ by default -- the loader of spectral_sensitivity_bh_fdr.py
globs ``<psd_root>/*_psd.npz`` non-recursively):
    <stem>_psd.npz             freqs (F,), psd (n_mask_electrodes, F) in uV^2/Hz, chan_idx (flat
                               electrode indices r*W + c), fs_used, fs_stored, nperseg, noverlap,
                               group, slice_id, condition, source_file
    <stem>_bandpower.csv       one row per electrode: absolute and relative band powers, total
                               power, peak frequency
    psd_bandpower_summary.csv  one row per recording: group, slice_id, condition, file, n_electrodes,
                               fs_used, nperseg, mean absolute / relative band powers
    manifest.json              every parameter used (also copied to --figs_dir)
Figures (under data/results/welch_psd/ by default, PNG 300 dpi + SVG):
    <group>__<stem>__psd.(png|svg)        mean +/- s.e.m. and median PSD over electrodes (log y)
                                          next to the median band-power bars of that recording
    folder_psd_overlay.(png|svg)          mean PSD of every recording in one axis
    folder_bandpower_relative.(png|svg)   per-recording relative band power, grouped by condition

Algorithm (as in the Methods):
    1. Tissue-contact electrodes (mask) of /lfp/ts_2ms_full are streamed in channel blocks whose
       size is chosen from the record length so that one block stays below --block_mb; the full
       4096 x T array is never loaded.
    2. Optionally each block is decimated to --decimate_to_hz (integer factor, zero-phase FIR
       anti-aliasing, scipy.signal.decimate); fs_used is the rate after decimation.
    3. Welch PSD per electrode: Hann window, segment length nperseg = round(--nfft_sec * fs_used),
       --overlap fractional overlap, removal of the segment mean, density scaling
       (scipy.signal.welch). Frequencies up to --fmax Hz are retained.
    4. Band power = trapezoidal integral of the PSD over delta 1-4, theta 4-8, alpha 8-12,
       beta 12-30, gamma 30-80 and high-gamma 80-200 Hz (lower limit inclusive, upper exclusive).
       Relative power = band power / total power over the retained range (1 Hz .. --fmax).

Implementation notes:
    - nfft_sec = 2 s is the whole-recording default (0.5 Hz resolution). When a recording is
      shorter than one segment (e.g. the synthetic examples, 1.2 s) nperseg is clamped to the
      record length with a warning, so the estimate degenerates to a single Hann periodogram.
      nperseg is not rounded to a power of two (scipy handles any length; the requested window
      length is kept exactly).
    - Decimation uses scipy.signal.decimate (Hamming-window FIR, zero phase) instead of the
      block-mean downsampling of the original prototype, so no aliasing enters the PSD. The
      factor fs / decimate_to_hz must be an integer (fs = 10000 -> 500 Hz is factor 20).
    - Power-line harmonics (--line_hz, default 60 Hz, +/- 1 Hz) are interpolated over for the
      figures only; the stored PSD and every band power are left untouched.
    - --max_ch subsamples the mask electrodes uniformly at random (seeded by --seed) when a
      recording has more mask electrodes than that; chan_idx records which ones were used.
    - NPZ files are written flat (no group sub-folders) because the downstream loader does not
      recurse; if two groups contain the same file stem the second one is written as
      <group>__<stem>_psd.npz and a warning is printed.
    - Units: /lfp/ts_2ms_full is in uV, so psd is in uV^2/Hz and band powers in uV^2.
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
from scipy.signal import welch, decimate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
IN_ROOT = REPO_ROOT / "data" / "processed_2ms"
MASK_DIR = REPO_ROOT / "data" / "masks"
OUT_DIR = REPO_ROOT / "data" / "psd_tables"
FIGS_DIR = REPO_ROOT / "data" / "results" / "welch_psd"
NFFT_SEC = 2.0             # Welch segment length (s)
OVERLAP = 0.5              # fractional segment overlap
FMAX_HZ = 250.0            # retained frequency range
LINE_HZ = 60.0             # power-line fundamental (plot-only suppression; 0 = off)
BLOCK_MB = 256.0           # upper bound of one streamed channel block (float32) in MB
BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0), "gamma": (30.0, 80.0), "high_gamma": (80.0, 200.0),
}
TOTAL_FMIN = 1.0           # total power for relative band power: TOTAL_FMIN .. FMAX_HZ
FIG_DPI = 300
# --- END EDIT ---

LFP_KEY = "/lfp/ts_2ms_full"


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


def load_mask(f: h5py.File, h5_path: Path, mask_dir: Path, mask_override: Optional[Path], H: int, W: int) -> Tuple[np.ndarray, str, str]:
    """Return (mask bool [H,W], slice_id, source)."""
    hit = find_mask_file(h5_path.name, mask_dir)
    slice_id = hit[0] if hit else re.split(r"_cor_|_4ap|_afdx|_bic|_cnqx|_ml297", h5_path.stem, flags=re.I)[0]
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
    return m, slice_id, src


# =============================================================================
# PSD
# =============================================================================

def integrate_band(freqs: np.ndarray, P: np.ndarray, lo: float, hi: float) -> float:
    """Trapezoidal integral of P over lo <= f < hi (NaN if no bin falls inside)."""
    m = (freqs >= lo) & (freqs < hi)
    if m.sum() < 2:
        return float(np.nan)
    return float(np.trapezoid(P[m], freqs[m]))


def band_table(freqs: np.ndarray, psd: np.ndarray, chan_idx: np.ndarray, W: int, fmax: float) -> pd.DataFrame:
    """Per-electrode absolute / relative band powers, total power and peak frequency."""
    rows = []
    for k, ch in enumerate(chan_idx):
        P = psd[k]
        total = integrate_band(freqs, P, TOTAL_FMIN, fmax + 1e-9)
        row = {"chan_idx": int(ch), "r": int(ch // W), "c": int(ch % W),
               "total_power": total, "peak_freq_hz": float(freqs[int(np.nanargmax(P))])}
        for b, (lo, hi) in BANDS.items():
            bp = integrate_band(freqs, P, lo, hi)
            row[f"{b}_abs"] = bp
            row[f"{b}_rel"] = bp / total if total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def welch_recording(dset: h5py.Dataset, chan_idx: np.ndarray, fs: float, args, block_mb: float) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Stream mask rows in channel blocks, optionally decimate, Welch PSD per row. Returns (freqs, psd, info)."""
    T = dset.shape[1]
    q = 1
    if args.decimate_to_hz:
        q = fs / args.decimate_to_hz
        if abs(q - round(q)) > 1e-6 or q < 1:
            raise ValueError(f"fs / decimate_to_hz = {q:.4f} is not an integer >= 1")
        q = int(round(q))
    fs_used = fs / q
    T_used = T // q if q > 1 else T
    nperseg = int(round(args.nfft_sec * fs_used))
    if nperseg > T_used:
        warnings.warn(f"record length {T_used} samples ({T_used / fs_used:.3f} s) < nfft_sec * fs "
                      f"({nperseg} samples) -> nperseg clamped to {T_used} (single-segment periodogram)")
        nperseg = T_used
    noverlap = int(round(args.overlap * nperseg))
    noverlap = min(noverlap, nperseg - 1)
    rows_per_block = max(1, int(block_mb * 1024 ** 2 / (T * 4)))

    freqs, psd = None, []
    for s in range(0, len(chan_idx), rows_per_block):
        idx = chan_idx[s:s + rows_per_block]
        X = dset[idx, :].astype(np.float32)          # h5py fancy row read, sorted indices
        if q > 1:
            X = decimate(X.astype(np.float64), q, ftype="fir", axis=-1, zero_phase=True)
        f, P = welch(X, fs=fs_used, window="hann", nperseg=nperseg, noverlap=noverlap,
                     detrend="constant", scaling="density", axis=-1)
        keep = f <= args.fmax
        if freqs is None:
            freqs = f[keep]
        psd.append(P[:, keep].astype(np.float64))
    psd = np.concatenate(psd, axis=0)
    info = {"fs_used": fs_used, "decimate_factor": q, "nperseg": nperseg, "noverlap": noverlap,
            "n_samples_used": T_used, "rows_per_block": rows_per_block}
    return freqs, psd, info


# =============================================================================
# FIGURES
# =============================================================================

def set_style():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "svg.fonttype": "none"})


def suppress_line_noise(freqs: np.ndarray, y: np.ndarray, base: float, width: float = 1.0) -> np.ndarray:
    """Plot-only: interpolate over +/- width Hz around every harmonic of base (0 = off)."""
    if not base or base <= 0:
        return y
    yy = y.copy()
    bad = np.zeros_like(freqs, dtype=bool)
    for h in range(1, int(freqs.max() // base) + 1):
        bad |= np.abs(freqs - h * base) <= width
    if bad.any() and (~bad).sum() > 2:
        yy[bad] = np.interp(freqs[bad], freqs[~bad], yy[~bad])
    return yy


def recording_figure(freqs: np.ndarray, psd: np.ndarray, bt: pd.DataFrame, title: str, line_hz: float, out_base: Path):
    set_style()
    n = psd.shape[0]
    mean = suppress_line_noise(freqs, np.nanmean(psd, 0), line_hz)
    med = suppress_line_noise(freqs, np.nanmedian(psd, 0), line_hz)
    sem = np.nanstd(psd, 0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    pos = freqs > 0
    ax.plot(freqs[pos], mean[pos], lw=1.2, color="0.15", label="mean")
    ax.fill_between(freqs[pos], np.maximum(mean - sem, 1e-300)[pos], (mean + sem)[pos], color="0.15", alpha=0.2, lw=0)
    ax.plot(freqs[pos], med[pos], lw=1.0, color="C0", ls="--", label="median")
    ax.set_yscale("log"); ax.set_xlim(0, freqs[-1])
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (uV$^2$/Hz)")
    ax.legend(frameon=False, fontsize=7); ax.set_title(f"{title}  (n = {n} electrodes)", fontsize=8)
    names = list(BANDS)
    vals = [np.nanmedian(bt[f"{b}_rel"]) for b in names]
    q1 = [np.nanpercentile(bt[f"{b}_rel"], 25) for b in names]
    q3 = [np.nanpercentile(bt[f"{b}_rel"], 75) for b in names]
    x = np.arange(len(names))
    ax2.bar(x, vals, color="0.6", width=0.7)
    ax2.errorbar(x, vals, yerr=[np.array(vals) - np.array(q1), np.array(q3) - np.array(vals)], fmt="none", ecolor="0.15", lw=0.8, capsize=2)
    ax2.set_xticks(x); ax2.set_xticklabels([b.replace("_", "-") for b in names], rotation=35, ha="right")
    ax2.set_ylabel("Relative band power"); ax2.set_title("median [IQR] over electrodes", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_base)


def folder_figures(recs: List[dict], line_hz: float, figs_dir: Path):
    set_style()
    conds = sorted({r["condition"] for r in recs})
    colors = {c: f"C{i}" for i, c in enumerate(conds)}
    # overlay of mean PSDs (interpolated to the first frequency axis if they differ)
    f0 = recs[0]["freqs"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for r in recs:
        m = suppress_line_noise(r["freqs"], r["mean_psd"], line_hz)
        if len(r["freqs"]) != len(f0) or not np.allclose(r["freqs"], f0):
            m = np.interp(f0, r["freqs"], m)
        pos = f0 > 0
        ax.plot(f0[pos], m[pos], lw=0.9, alpha=0.8, color=colors[r["condition"]], label=f"{r['group']} {r['file']}")
    ax.set_yscale("log"); ax.set_xlim(0, f0[-1]); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Mean PSD (uV$^2$/Hz)")
    ax.legend(frameon=False, fontsize=5, ncol=1 if len(recs) <= 12 else 2)
    ax.set_title("Mean PSD per recording", fontsize=8)
    save_fig(fig, figs_dir / "folder_psd_overlay")
    # relative band power per recording grouped by condition
    names = list(BANDS); x = np.arange(len(names)); w = 0.8 / max(1, len(conds))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for i, c in enumerate(conds):
        sel = [r for r in recs if r["condition"] == c]
        M = np.array([[r["summary"][f"{b}_rel_mean"] for b in names] for r in sel])
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, np.nanmean(M, 0), width=w, color=colors[c], alpha=0.6, label=f"{c} (n={len(sel)})")
        for row in M:
            ax.scatter(x + (i - (len(conds) - 1) / 2) * w, row, s=8, color=colors[c], edgecolor="0.15", lw=0.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([b.replace("_", "-") for b in names], rotation=35, ha="right")
    ax.set_ylabel("Relative band power (mean over electrodes)"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("Per-recording relative band power", fontsize=8)
    save_fig(fig, figs_dir / "folder_bandpower_relative")


# =============================================================================
# PER-RECORDING
# =============================================================================

def process_recording(h5_path: Path, group: str, args, mask_dir: Path, out_dir: Path, figs_dir: Path,
                      rng: np.random.Generator, used_stems: set) -> dict:
    with h5py.File(h5_path, "r") as f:
        fs, H, W = read_meta(f)
        mask, slice_id, mask_src = load_mask(f, h5_path, mask_dir, args.mask, H, W)
        chan_idx = np.flatnonzero(mask.reshape(-1))
        if args.max_ch and len(chan_idx) > args.max_ch:
            chan_idx = np.sort(rng.choice(chan_idx, size=args.max_ch, replace=False))
        if len(chan_idx) == 0:
            raise ValueError("mask selects no electrode")
        freqs, psd, info = welch_recording(f[LFP_KEY], chan_idx, fs, args, args.block_mb)
    cond = parse_condition(h5_path.name)
    bt = band_table(freqs, psd, chan_idx, W, args.fmax)

    stem = h5_path.stem
    if stem in used_stems:
        warnings.warn(f"duplicate file stem {stem} across groups -> writing as {group}__{stem}")
        stem = f"{group}__{stem}"
    used_stems.add(stem)
    np.savez_compressed(out_dir / f"{stem}_psd.npz", freqs=freqs, psd=psd, chan_idx=chan_idx,
                        fs_used=info["fs_used"], fs_stored=fs, nperseg=info["nperseg"], noverlap=info["noverlap"],
                        group=group, slice_id=slice_id, condition=cond, source_file=h5_path.name)
    bt.insert(0, "file", h5_path.stem); bt.insert(0, "condition", cond); bt.insert(0, "slice_id", slice_id); bt.insert(0, "group", group)
    bt.to_csv(out_dir / f"{stem}_bandpower.csv", index=False)

    summary = {"group": group, "slice_id": slice_id, "condition": cond, "file": h5_path.stem, "n_electrodes": int(len(chan_idx)),
               "fs_stored": fs, "fs_used": info["fs_used"], "nperseg": info["nperseg"], "n_samples_used": info["n_samples_used"],
               "mask_source": mask_src, "total_power_mean": float(np.nanmean(bt["total_power"])),
               "peak_freq_hz_median": float(np.nanmedian(bt["peak_freq_hz"]))}
    for b in BANDS:
        summary[f"{b}_abs_mean"] = float(np.nanmean(bt[f"{b}_abs"]))
    for b in BANDS:
        summary[f"{b}_rel_mean"] = float(np.nanmean(bt[f"{b}_rel"]))
    recording_figure(freqs, psd, bt, f"{group} {h5_path.stem}", args.line_hz, figs_dir / f"{group}__{h5_path.stem}__psd")
    return {"group": group, "file": h5_path.stem, "condition": cond, "freqs": freqs,
            "mean_psd": np.nanmean(psd, 0), "summary": summary}


# =============================================================================
# CLI
# =============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Per-electrode Welch PSD of every processed recording (tissue-contact "
                                            "electrodes, streamed in channel blocks) -> *_psd.npz tables, band-power "
                                            "CSVs and overview figures.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in_root", default=str(IN_ROOT), help="processed_2ms root (recursive; group = parent folder)")
    p.add_argument("--mask_dir", default=str(MASK_DIR), help="folder of <slice_id>__contact_mask_64x64.npy")
    p.add_argument("--mask", default=None, help="override: one mask .npy used for every recording")
    p.add_argument("--out_dir", default=str(OUT_DIR), help="output folder for *_psd.npz / CSV tables (flat)")
    p.add_argument("--figs_dir", default=str(FIGS_DIR), help="output folder for figures")
    p.add_argument("--only_contains", default=None, help="process only files whose name contains this substring")
    p.add_argument("--nfft_sec", type=float, default=NFFT_SEC, help="Welch segment length (s); clamped to the record length")
    p.add_argument("--overlap", type=float, default=OVERLAP, help="fractional segment overlap in [0, 1)")
    p.add_argument("--fmax", type=float, default=FMAX_HZ, help="retain frequencies <= fmax (Hz)")
    p.add_argument("--max_ch", type=int, default=0, help="random subsample of mask electrodes per recording (0 = all)")
    p.add_argument("--decimate_to_hz", type=float, default=None, help="decimate to this rate before Welch (integer factor of fs)")
    p.add_argument("--line_hz", type=float, default=LINE_HZ, help="power-line fundamental interpolated over in figures only (0 = off)")
    p.add_argument("--block_mb", type=float, default=BLOCK_MB, help="max size of one streamed channel block (MB)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --max_ch subsampling")
    return p


def main():
    args = build_argparser().parse_args()
    if not 0 <= args.overlap < 1:
        raise SystemExit("--overlap must be in [0, 1)")
    rng = np.random.default_rng(args.seed)
    args.mask = Path(args.mask) if args.mask else None
    in_root, mask_dir, out_dir, figs_dir = Path(args.in_root), Path(args.mask_dir), Path(args.out_dir), Path(args.figs_dir)
    files = sorted(p for p in in_root.rglob("*.h5") if not args.only_contains or args.only_contains in p.name)
    if not files:
        raise SystemExit(f"no .h5 files under {in_root}")
    out_dir.mkdir(parents=True, exist_ok=True); figs_dir.mkdir(parents=True, exist_ok=True)

    recs, used = [], set()
    for p in tqdm(files, desc="recordings"):
        group = p.parent.name if p.parent != in_root else in_root.name
        try:
            recs.append(process_recording(p, group, args, mask_dir, out_dir, figs_dir, rng, used))
        except Exception as e:
            print(f"[ERROR] {p}: {type(e).__name__}: {e}")

    cols = ["group", "slice_id", "condition", "file", "n_electrodes", "fs_stored", "fs_used", "nperseg", "n_samples_used",
            "mask_source", "total_power_mean", "peak_freq_hz_median"] + [f"{b}_abs_mean" for b in BANDS] + [f"{b}_rel_mean" for b in BANDS]
    pd.DataFrame([r["summary"] for r in recs], columns=cols).to_csv(out_dir / "psd_bandpower_summary.csv", index=False)
    if recs:
        folder_figures(recs, args.line_hz, figs_dir)

    manifest = {"script": Path(__file__).name, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "bands_hz": BANDS, "band_limits": "lower inclusive, upper exclusive; trapezoidal integration",
                "total_power_range_hz": [TOTAL_FMIN, args.fmax], "welch": {"window": "hann", "detrend": "constant", "scaling": "density"},
                "lfp_dataset": LFP_KEY, "n_recordings": len(recs), "n_files_found": len(files),
                "recordings": [{k: v for k, v in r["summary"].items()} for r in recs]}
    for d in (out_dir, figs_dir):
        with open(d / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
    print(f"[DONE] {len(recs)}/{len(files)} recordings -> {out_dir} (tables), {figs_dir} (figures)")


if __name__ == "__main__":
    main()
