"""
pick_representative_recording.py -- SPACE framework: QC pick of a clean representative CSD image.

Manuscript figure(s): used to select which recording appears as the representative
example in Figures 1B/3 (QC/figure-prep utility, not a statistical result).

What it computes:
    Scores every CSD snapshot (one representative frame per recording, taken as the
    median around the event t0) in a genotype/group folder for vertical-stripe artifact
    contamination (a MAD-z outlier score over column-wise median |CSD| energy), ranks all
    recordings by that score, and exports the cleanest one as the paper's representative
    example image, along with a stripe-score ranking plot and a per-file TSV report. This
    does not edit or modify any pipeline outputs; it only selects and reports.

Inputs / outputs (see data/README.md for the full schema):
    Input:  ``<slice_id>_<condition>_2ms.h5`` files under a single
            ``data/processed_2ms/<group>/`` folder, and an optional 64x64 contact mask
            from ``data/masks/``.
    Output: ``BEST_clean_<base>_CSD.png/.svg`` (the selected representative frame),
            ``QC_stripe_scores.png/.svg``, and ``_stripe_report.tsv``, written under a
            results directory, e.g. ``data/results/qc_representative/``.
"""

import os, glob, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PATHS - EDIT THESE, or override via CLI flags (see build_argparser)
# ============================================================
IN_DIR   = "data/processed_2ms/5xFAD"     # folder containing 2ms .h5 files for one group
OUT_DIR  = "data/results/qc_representative"
PATTERN  = "*_2ms.h5"

MASK_NPY = "data/masks/contact_mask_64x64.npy"   # optional. set "" to disable
USE_MASK = True

# ============================================================
# STREAMING CHUNK
# ============================================================
CHUNK_T = 2000   # time chunk in samples
T0_SETTLE_MS = 200.0   # skip the first N ms of the onset search (causal-filter start-up transient)

# ============================================================
# Representative frame extraction
# ============================================================
SEARCH_MS = 2000.0        # search window for t0 (ms)
FRAME_WINDOW_MS = 20.0    # median +/- window around t0 (ms)

# ============================================================
# Stripe detection thresholds
# ============================================================
Z_THRESH = 6.0            # strong outlier threshold (MAD-z)
MAX_FILES = 0             # 0 = all

# ============================================================
# H5 KEYS
# ============================================================
CSD_KEY  = "/csd/ts_2ms_full"
META_KEY = "/meta"

# -----------------------------
# Nature-ish style
# -----------------------------
def set_nature_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.15",
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def save_png_svg(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", bbox_inches="tight", dpi=600)
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)

def load_meta(f):
    meta = f[META_KEY].attrs
    H  = int(meta["H"])
    W  = int(meta["W"])
    fs = float(meta["fs"])  # Hz
    T2 = int(meta["T2"])
    return H, W, fs, T2

def load_mask_optional(mask_npy, H, W):
    if not mask_npy:
        return None
    m = np.load(mask_npy).astype(bool)
    if m.shape != (H, W):
        raise ValueError(f"mask shape {m.shape} != (H,W)=({H},{W})")
    return m

def mask_flat_indices(mask2d):
    return np.flatnonzero(mask2d.reshape(-1))

def robust_mad_z(x, eps=1e-12):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    z = (x - med) / (1.4826 * mad)
    return z, med, mad

def find_t0_argmax_stream(dset, t_end, chunk_T=2000, use_ch_idx=None, t_start=0):
    """
    t0 = argmax_t mean(|CSD|) within [0, t_end)
    Streaming to avoid big loads.
    """
    N, T = dset.shape
    t_end = min(T, int(t_end))
    best_t = 0
    best_v = -np.inf

    for a in range(int(t_start), t_end, chunk_T):
        b = min(t_end, a + chunk_T)
        X = dset[:, a:b] if use_ch_idx is None else dset[use_ch_idx, a:b]
        score = np.mean(np.abs(X), axis=0)
        j = int(np.argmax(score))
        v = float(score[j])
        if v > best_v:
            best_v = v
            best_t = a + j

    return best_t, best_v

def extract_median_frame(dset, t0, dt_ms, window_ms, use_ch_idx=None):
    """
    Returns (N,) channel vector = median over time window around t0.
    """
    N, T = dset.shape
    halfw = max(0, int(round((window_ms / dt_ms) / 2.0)))
    a = max(0, t0 - halfw)
    b = min(T, t0 + halfw + 1)

    if use_ch_idx is None:
        X = dset[:, a:b]
        vec = np.median(X, axis=1).astype(np.float32)
        return vec
    else:
        X = dset[use_ch_idx, a:b]
        vec_small = np.median(X, axis=1).astype(np.float32)
        vec = np.full((N,), np.nan, dtype=np.float32)
        vec[use_ch_idx] = vec_small
        return vec

def compute_stripe_score(frame_hw, mask2d=None):
    """
    frame_hw: (H,W) with possible NaNs outside mask.
    Computes column-wise robust outlier z-scores of median(|frame|).
    Returns: stripe_score, n_bad_cols, z_cols, col_energy
    """
    if mask2d is not None:
        data = np.where(mask2d, frame_hw, np.nan)
    else:
        data = frame_hw

    col_energy = np.nanmedian(np.abs(data), axis=0)  # (W,)
    col_energy = np.where(np.isfinite(col_energy), col_energy, 0.0)

    z, med, mad = robust_mad_z(col_energy)
    stripe_score = float(np.max(np.abs(z)))
    n_bad = int(np.sum(np.abs(z) > Z_THRESH))
    return stripe_score, n_bad, z, col_energy

def plot_clean_frame(frame_hw, title, vlim, cmap_name="coolwarm"):
    set_nature_style()
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(3.6, 3.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(frame_hw, vmin=-vlim, vmax=vlim, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9, pad=4)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=8, width=0.6, length=2.5)
    cb.set_label("CSD (a.u.)", fontsize=9)
    return fig

def plot_qc_scores(records, out_base):
    set_nature_style()
    records = sorted(records, key=lambda r: r["stripe_score"])
    names = [r["base"] for r in records]
    scores = [r["stripe_score"] for r in records]

    fig = plt.figure(figsize=(8.5, max(3.0, 0.18 * len(records))))
    ax = fig.add_subplot(111)
    y = np.arange(len(records))
    ax.barh(y, scores)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Stripe score (max |MAD-z| across columns)")
    ax.axvline(Z_THRESH, linestyle="--", linewidth=1.0)
    ax.set_title("Vertical-stripe artifact ranking (lower is cleaner)", fontsize=10, pad=8)
    fig.tight_layout()
    save_png_svg(fig, out_base)

def build_argparser():
    p = argparse.ArgumentParser(description="Pick the cleanest representative CSD snapshot from a folder of 2 ms H5 files")
    p.add_argument("--in_dir", default=IN_DIR, help="Folder of <slice_id>_<condition>_2ms.h5 files, one group at a time (default: %(default)s)")
    p.add_argument("--out_dir", default=OUT_DIR, help="Output folder (default: %(default)s)")
    p.add_argument("--pattern", default=PATTERN, help="Glob pattern for H5 files (default: %(default)s)")
    p.add_argument("--mask", default=MASK_NPY, help="Optional 64x64 contact mask .npy, empty string to disable (default: %(default)s)")
    return p


def main():
    global IN_DIR, OUT_DIR, PATTERN, MASK_NPY
    args = build_argparser().parse_args()
    IN_DIR, OUT_DIR, PATTERN, MASK_NPY = args.in_dir, args.out_dir, args.pattern, args.mask

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(IN_DIR, PATTERN)))
    if MAX_FILES and MAX_FILES > 0:
        paths = paths[:MAX_FILES]
    print(f"[INFO] Found {len(paths)} files in: {IN_DIR}")

    if not paths:
        print("[DONE] no files")
        return

    # Load mask once
    mask2d = None
    with h5py.File(paths[0], "r") as f0:
        H0, W0, fs0, _ = load_meta(f0)
    if MASK_NPY and USE_MASK:
        mask2d = load_mask_optional(MASK_NPY, H0, W0)

    records = []
    best = None  # (stripe_score, base, frame_hw, vlim, t0)

    report_path = os.path.join(OUT_DIR, "_stripe_report.tsv")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("base\tstripe_score\tbad_cols\tfs\tdt_ms\tt0_idx\n")

        for h5_path in paths:
            base = os.path.splitext(os.path.basename(h5_path))[0]
            try:
                with h5py.File(h5_path, "r") as f:
                    if META_KEY not in f or CSD_KEY not in f:
                        print(f"[SKIP] {base}: missing keys")
                        continue

                    H, W, fs, T2 = load_meta(f)
                    dt_ms = 1000.0 / fs
                    d = f[CSD_KEY]  # (N, T)

                    use_idx = mask_flat_indices(mask2d) if mask2d is not None else None

                    t_end = int(round(min(d.shape[1], SEARCH_MS / dt_ms)))
                    t0, _ = find_t0_argmax_stream(d, t_end=t_end, chunk_T=CHUNK_T, use_ch_idx=use_idx, t_start=int(round(T0_SETTLE_MS / dt_ms)))

                    vec = extract_median_frame(d, t0=t0, dt_ms=dt_ms, window_ms=FRAME_WINDOW_MS, use_ch_idx=use_idx)
                    frame_hw = vec.reshape(H, W).astype(np.float32)

                    if mask2d is not None:
                        frame_hw = np.where(mask2d, frame_hw, np.nan)

                    stripe_score, n_bad, z_cols, col_energy = compute_stripe_score(frame_hw, mask2d=mask2d)

                    vlim = float(np.nanpercentile(np.abs(frame_hw), 99.0))
                    if not np.isfinite(vlim) or vlim <= 0:
                        vlim = 1.0

                    fp.write(f"{base}\t{stripe_score:.6g}\t{n_bad}\t{fs:.6g}\t{dt_ms:.6g}\t{t0}\n")

                    rec = {
                        "base": base,
                        "stripe_score": stripe_score,
                        "n_bad": n_bad,
                        "t0": t0,
                        "vlim": vlim,
                        "h5_path": h5_path,
                    }
                    records.append(rec)

                    if best is None or stripe_score < best[0]:
                        best = (stripe_score, base, frame_hw, vlim, t0)

                    print(f"[OK] {base}: stripe_score={stripe_score:.3f} bad_cols={n_bad} t0={t0}")

            except Exception as e:
                print(f"[ERROR] {base}: {e}")

    print(f"[INFO] wrote: {report_path}")

    if not records or best is None:
        print("[DONE] no records")
        return

    # QC plot
    plot_qc_scores(records, out_base=os.path.join(OUT_DIR, "QC_stripe_scores"))

    # Save best clean representative
    stripe_score, base, frame_hw, vlim, t0 = best
    title = f"BEST clean: {base} (t0={t0}) | stripe={stripe_score:.2f}"
    fig = plot_clean_frame(frame_hw, title=title, vlim=vlim, cmap_name="coolwarm")
    save_png_svg(fig, os.path.join(OUT_DIR, f"BEST_clean_{base}_CSD"))

    print(f"[DONE] Best clean file: {base}  stripe_score={stripe_score:.3f}")

if __name__ == "__main__":
    main()
