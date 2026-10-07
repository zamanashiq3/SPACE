# -*- coding: utf-8 -*-
"""
mask_noise_reference.py -- compare tissue-contact (Omega) and non-contact electrodes.

Manuscript Methods section: "Anatomical registration, tissue-contact mask and regions of
interest" (last sentences: electrodes outside Omega used as an empirical noise reference).
Manuscript figure(s): Supplementary Fig. 1A-C (RMS amplitude and representative traces of
tissue-contact vs. background electrodes).

Inputs:  data/processed_2ms/<group>/*.h5 (uses /lfp/metrics/rms_map and, for the example
         traces, a short window of /lfp/ts_2ms_full) and the matching
         data/masks/<slice>__contact_mask_64x64.npy.
Outputs: data/results/mask_noise_reference/
           rms_inside_vs_outside.csv   per recording: n, median/IQR RMS inside and outside Omega,
                                       ratio, Mann-Whitney U p-value (descriptive)
           <group>__<file>__rms_map.(png|svg)     RMS map with mask outline
           <group>__<file>__traces.(png|svg)      5 random inside vs 5 random outside traces
           manifest.json

Implementation notes:
    The RMS map is the whole-recording RMS of the 2 ms-smoothed LFP written by
    preprocessing/brw_to_2ms_h5.py. Traces are shown for --trace_window_s seconds starting at
    --trace_start_s (default 0.5 s, 1.0 s). Electrodes on the outer ring are kept (this is an
    LFP, not a CSD, comparison).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
# --- EDIT THIS ---
IN_ROOT = REPO_ROOT / "data" / "processed_2ms"
MASK_DIR = REPO_ROOT / "data" / "masks"
OUT_DIR = REPO_ROOT / "data" / "results" / "mask_noise_reference"
# -----------------


def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    plt.close(fig)


def find_mask(h5_path: Path, mask_dir: Path, override: Optional[Path]):
    if override is not None:
        return np.load(override).astype(bool)
    for m in sorted(mask_dir.glob("*__contact_mask_64x64.npy")):
        if m.name.split("__")[0].lower() in h5_path.name.lower():
            return np.load(m).astype(bool)
    with h5py.File(h5_path, "r") as f:
        if "/mask" in f:
            return f["/mask"][:].astype(bool)
    raise FileNotFoundError(f"No mask for {h5_path.name}")


def process(h5_path: Path, group: str, mask: np.ndarray, args, out_dir: Path, rng) -> dict:
    with h5py.File(h5_path, "r") as f:
        fs = float(f["/meta"].attrs.get("fs", 10000.0))
        rms = f["/lfp/metrics/rms_map"][:].astype(float)
        H, W = rms.shape
        s0 = int(args.trace_start_s * fs)
        s1 = min(int((args.trace_start_s + args.trace_window_s) * fs), f["/lfp/ts_2ms_full"].shape[1])
        inside = np.flatnonzero(mask.ravel())
        outside = np.flatnonzero(~mask.ravel())
        pick_in = np.sort(rng.choice(inside, size=min(5, inside.size), replace=False))
        pick_out = np.sort(rng.choice(outside, size=min(5, outside.size), replace=False)) if outside.size else np.array([], int)
        tr_in = f["/lfp/ts_2ms_full"][pick_in, s0:s1]
        tr_out = f["/lfp/ts_2ms_full"][pick_out, s0:s1] if pick_out.size else np.zeros((0, s1 - s0))

    r_in, r_out = rms[mask], rms[~mask]
    p = mannwhitneyu(r_in, r_out, alternative="two-sided").pvalue if r_out.size and r_in.size else np.nan
    row = dict(group=group, file=h5_path.name, n_inside=int(r_in.size), n_outside=int(r_out.size),
               rms_inside_median=float(np.median(r_in)), rms_inside_iqr=float(np.subtract(*np.percentile(r_in, [75, 25]))),
               rms_outside_median=float(np.median(r_out)) if r_out.size else np.nan,
               rms_outside_iqr=float(np.subtract(*np.percentile(r_out, [75, 25]))) if r_out.size else np.nan,
               rms_ratio_in_over_out=float(np.median(r_in) / np.median(r_out)) if r_out.size else np.nan,
               mannwhitney_p=float(p))

    stem = f"{group}__{h5_path.stem}"
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(rms, cmap="magma", vmin=0, vmax=np.nanpercentile(rms, 99))
    ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(f"{h5_path.stem}\nRMS (uV)", fontsize=7); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save_fig(fig, out_dir / f"{stem}__rms_map")

    t = np.arange(s1 - s0) / fs * 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True, sharey=True)
    off = np.nanpercentile(np.abs(tr_in), 99) * 1.2 if tr_in.size else 1.0
    for k, y in enumerate(tr_in):
        axes[0].plot(t, y + k * off, lw=0.5, color="k")
    for k, y in enumerate(tr_out):
        axes[1].plot(t, y + k * off, lw=0.5, color="0.5")
    axes[0].set_title("tissue-contact electrodes (Omega)", fontsize=8)
    axes[1].set_title("non-contact electrodes (noise reference)", fontsize=8)
    axes[1].set_xlabel("time (ms)"); axes[0].set_ylabel("LFP (uV, offset)")
    save_fig(fig, out_dir / f"{stem}__traces")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in_root", type=Path, default=IN_ROOT)
    ap.add_argument("--mask_dir", type=Path, default=MASK_DIR)
    ap.add_argument("--mask", type=Path, default=None, help="override: use this mask for all files")
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--trace_start_s", type=float, default=0.5)
    ap.add_argument("--trace_window_s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h5_path in sorted(args.in_root.rglob("*.h5")):
        group = h5_path.parent.name
        try:
            mask = find_mask(h5_path, args.mask_dir, args.mask)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}"); continue
        rows.append(process(h5_path, group, mask, args, args.out_dir, rng))
        print(f"[OK] {h5_path.name}: RMS in/out = {rows[-1]['rms_ratio_in_over_out']:.2f}")
    pd.DataFrame(rows).to_csv(args.out_dir / "rms_inside_vs_outside.csv", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))


if __name__ == "__main__":
    main()
