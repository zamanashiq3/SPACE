# -*- coding: utf-8 -*-
"""
recording_overview.py -- whole-recording activity profile of each processed H5 (QC).

Streams /lfp/ts_2ms_full and /csd/ts_2ms_full over the tissue-contact electrodes and writes,
per recording, the mean |LFP| and mean |CSD| per --bin_ms bin for the entire recording, a
PNG/SVG figure of both traces (log y) with the onset candidates marked:
    * "first 2 s" rule (max mean |CSD| in SETTLE..2 s)  -- used by extract_event_epochs.py
    * whole-recording maximum of summed |LFP|            -- used by regional_csd_dynamics.py /
                                                            roi_lfp_csd_metrics.py
    * the top --n_peaks population events (local maxima of mean |LFP|, >= 1 s apart)
and a CSV (time_s, mean_abs_lfp, mean_abs_csd). Run this first on a new dataset to decide the
onset-search window (--t0_search_s of extract_event_epochs.py) and to spot start/end artefacts.

Outputs: data/results/recording_overview/<group>__<file>.{png,svg,csv}, overview_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
# --- EDIT THIS ---
IN_ROOT = REPO_ROOT / "data" / "processed_2ms"
MASK_DIR = REPO_ROOT / "data" / "masks"
OUT_DIR = REPO_ROOT / "data" / "results" / "recording_overview"
# -----------------


def find_mask(h5_path: Path, mask_dir: Path):
    for m in sorted(mask_dir.glob("*__contact_mask_64x64.npy")):
        if m.name.split("__")[0].lower() in h5_path.name.lower():
            return np.load(m).astype(bool)
    with h5py.File(h5_path, "r") as f:
        if "/mask" in f:
            return f["/mask"][:].astype(bool)
    return None


def profile(dset, idx, fs, bin_ms, ch_block=512, time_chunk=20000):
    T = dset.shape[1]
    nb = int(round(bin_ms * fs / 1000.0))
    n_bins = T // nb
    acc = np.zeros(n_bins * nb, np.float64)
    for t0 in range(0, n_bins * nb, time_chunk):
        t1 = min(t0 + time_chunk, n_bins * nb)
        for c0 in range(0, dset.shape[0], ch_block):
            c1 = min(c0 + ch_block, dset.shape[0])
            sel = idx[(idx >= c0) & (idx < c1)] - c0
            if sel.size:
                acc[t0:t1] += np.abs(dset[c0:c1, t0:t1][sel]).sum(axis=0)
    acc /= max(1, idx.size)
    return acc.reshape(n_bins, nb).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in_root", type=Path, default=IN_ROOT)
    ap.add_argument("--mask_dir", type=Path, default=MASK_DIR)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--bin_ms", type=float, default=10.0)
    ap.add_argument("--settle_ms", type=float, default=200.0)
    ap.add_argument("--n_peaks", type=int, default=5)
    ap.add_argument("--ch_stride", type=int, default=4, help="use every k-th tissue electrode (speed; 1 = all)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h5_path in sorted(args.in_root.rglob("*.h5")):
        group = h5_path.parent.name
        mask = find_mask(h5_path, args.mask_dir)
        with h5py.File(h5_path, "r") as f:
            fs = float(f["/meta"].attrs.get("fs", 10000.0))
            H = int(f["/meta"].attrs.get("H", 64)); W = int(f["/meta"].attrs.get("W", 64))
            if mask is None:
                mask = np.ones((H, W), bool)
            inner = mask.copy(); inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
            lfp = profile(f["/lfp/ts_2ms_full"], np.flatnonzero(mask.ravel())[::args.ch_stride], fs, args.bin_ms)
            csd = profile(f["/csd/ts_2ms_full"], np.flatnonzero(inner.ravel())[::args.ch_stride], fs, args.bin_ms)
        t = (np.arange(lfp.size) + 0.5) * args.bin_ms / 1000.0
        # onset candidates
        s0 = int(args.settle_ms / args.bin_ms); s2 = int(2000.0 / args.bin_ms)
        t0_first2s = t[s0 + int(np.argmax(csd[s0:s2]))] if s2 > s0 else np.nan
        t0_global = t[s0 + int(np.argmax(lfp[s0:]))]
        pk, _ = find_peaks(lfp, distance=int(1000.0 / args.bin_ms), prominence=np.nanstd(lfp))
        pk = pk[np.argsort(lfp[pk])[::-1][:args.n_peaks]]
        stem = f"{group}__{h5_path.stem}"
        pd.DataFrame({"time_s": t, "mean_abs_lfp": lfp, "mean_abs_csd": csd}).to_csv(args.out_dir / f"{stem}.csv", index=False)
        fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
        for ax, y, lab in zip(axes, (lfp, csd), ("mean |LFP| (uV)", "mean |CSD| (a.u.)")):
            ax.plot(t, y, lw=0.5, color="k"); ax.set_yscale("log"); ax.set_ylabel(lab)
            ax.axvline(t0_first2s, color="tab:red", lw=1, ls="--", label=f"first-2 s rule: {t0_first2s:.2f} s")
            ax.axvline(t0_global, color="tab:blue", lw=1, ls="--", label=f"whole-recording max |LFP|: {t0_global:.1f} s")
            for j, p in enumerate(pk):
                ax.axvline(t[p], color="tab:green", lw=0.8, alpha=0.6, label="top population events" if j == 0 else None)
        axes[0].legend(fontsize=7, loc="upper right"); axes[1].set_xlabel("time (s)")
        axes[0].set_title(f"{h5_path.name}  (fs={fs:.1f} Hz, {mask.sum()} tissue electrodes)", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{stem}.png", dpi=200); fig.savefig(args.out_dir / f"{stem}.svg"); plt.close(fig)
        rows.append(dict(group=group, file=h5_path.name, duration_s=float(t[-1]), t0_first2s_rule_s=t0_first2s,
                         t0_global_max_lfp_s=t0_global, top_event_times_s=";".join(f"{t[p]:.2f}" for p in pk),
                         lfp_first_200ms=float(lfp[:s0].mean()) if s0 else np.nan, lfp_median=float(np.median(lfp)),
                         lfp_last_1s=float(lfp[-int(1000 / args.bin_ms):].mean())))
        print(f"[OK] {stem}: first-2s rule t0={t0_first2s:.2f}s, global max {t0_global:.1f}s, "
              f"top events at {rows[-1]['top_event_times_s']} s; |LFP| first 200 ms / median / last 1 s = "
              f"{rows[-1]['lfp_first_200ms']:.1f} / {rows[-1]['lfp_median']:.1f} / {rows[-1]['lfp_last_1s']:.1f}")
    pd.DataFrame(rows).to_csv(args.out_dir / "overview_summary.csv", index=False)


if __name__ == "__main__":
    main()
