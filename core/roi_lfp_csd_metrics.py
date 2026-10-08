# -*- coding: utf-8 -*-
"""
roi_lfp_csd_metrics.py -- SPACE framework: ROI-based LFP/CSD snapshots, kymographs, correlation, entropy.

Manuscript figure(s): Figure 2 (LFP amplitude/energy metrics, entropy), Figure 1B/3C-F
(representative ROI traces, as a byproduct of its plotting), Supplementary Figure 2
(baseline LFP/CSD event-aligned dynamics).

What it computes:
    The most complete version of the ROI-based LFP/CSD analysis: a TIFF-aligned manual
    ROI picker (click boxes onto the 64x64 grid over a slice photo/map), per-ROI LFP and
    CSD snapshots/kymographs/raw stacked traces, ROI-level LFP/CSD/LFP-CSD Pearson
    correlation matrices, baseline-corrected LFP amplitude/energy/slope metrics with
    genotype-tagged barplots, and ROI CSD spatial Shannon-entropy over time. This
    supersedes two earlier drafts that were strict subsets of this script's outputs.
    The ROI boxes picked here are written to ``roi_definitions.csv`` and are the input
    consumed by ``core/roi_phase_gradient_pgd.py``.

Inputs / outputs (see data/README.md for the full schema):
    Input:  ``<slice_id>_<condition>_2ms.h5`` files under ``data/processed_2ms/``, a
            64x64 contact mask from ``data/masks/``, and a slice reference TIFF/photo for
            ROI-picker alignment.
    Output: per-ROI snapshots/kymographs/traces, correlation matrices, LFP metric CSVs
            and barplots, and CSD entropy tables/heatmaps under a results directory
            (e.g. ``data/results/roi_lfp_csd/``); also writes
            ``data/roi_definitions/<slice_id>_roi_definitions.csv`` for downstream use.

Purpose
-------
Clean ROI figure generator based on your finalized ROI picker / extraction logic.
NO PCA, NO KMeans, NO clustering, NO archetypes.

For every H5 file and every named ROI:
1) Uses the SAME TIFF-aligned ROI picker style from your ROI pipeline.
2) Finds t0 with chunked full-slice activity.
3) Extracts cleaned full-slice LFP and CSD snapshots.
4) Saves every snapshot separately as a 2D full-slice map:
   - whole slice shown
   - black tissue/mask outline
   - orange ROI box, NO padding
5) Saves ROI raw stacked traces for LFP and CSD.
6) Saves ROI kymographs for LFP and CSD.
7) Saves one combined raw+kymograph panel like your reference image:
   - left: raw stacked traces
   - right: kymograph
   - top: LFP
   - bottom: CSD
8) NEW 260505 update:
   - unsmoothed baseline-corrected ROI raw traces for display/CSV
   - all ROI LFP traces saved as CSV for later amplitude/energy/slope plots
   - ROI CSD time-snapshot entropy in bits saved in a separate folder
   - entropy barplots and summary CSVs
   - raw-data correlation matrices saved separately for later groupwise comparison

Expected H5 keys
----------------
/meta attrs: H, W, fs, optional T2
/lfp/ts_2ms_full : (Nch, T)
/csd/ts_2ms_full : (Nch, T)
optional /mask : (H, W), or external MASK_PATH
"""

import os
import sys
import re
import glob
import csv
import argparse
import numpy as np
import h5py
import matplotlib
if "--roi_csv" in sys.argv or (sys.platform.startswith("linux") and not os.environ.get("DISPLAY")):
    matplotlib.use("Agg")   # non-interactive run (ROI picker not used / headless Linux)
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, medfilt

# ============================================================
# CONFIG - EDIT THESE, or override via CLI flags (see build_argparser)
# ============================================================
DATA_DIR = "data/processed_2ms"
FILE_PATTERN = "*_2ms.h5"
OUT_DIR = "data/results/roi_lfp_csd"

MASK_PATH = "data/masks/contact_mask_64x64.npy"
TIF_PATH = "data/masks/<slice_id>__rectified_grid.png"   # rectified slice image from tissue_registration_mask.py

# Snapshot offsets relative to t0
SNAP_MS = [0, 50, 100, 300, 400, 500]

# ROI time window for raw traces/kymographs
PRE_MS = 50
POST_MS = 500

# ROI selection
ROI_SIZE = 4                 # 4x4 ROI
ROI_COLOR = "orange"         # ROI outline color
MASK_OUTLINE_COLOR = "black" # full slice outline color

# ------------------------------------------------------------
# GLOBAL COLOR LIMITS - EDIT HERE
# If USE_FIXED_* = True, the fixed value is used.
# If False, script computes global values from all files/ROIs.
# ------------------------------------------------------------
USE_FIXED_LFP_SNAP_VLIM = True
FIXED_LFP_SNAP_VLIM = 500.0

USE_FIXED_CSD_SNAP_VLIM = True
FIXED_CSD_SNAP_VLIM = .2

USE_FIXED_LFP_KYMO_VLIM = True
FIXED_LFP_KYMO_VLIM = 400.0

USE_FIXED_CSD_KYMO_VLIM = True
FIXED_CSD_KYMO_VLIM = .08

# Colormaps
LFP_CMAP = "seismic"
CSD_CMAP = "turbo_r"

# Raw trace display window inside extracted PRE/POST window
DISPLAY_PRE_MS = 0
DISPLAY_POST_MS = 500
RAW_TRACE_SPACING_SCALE = 1.20
RAW_TRACE_LINEWIDTH = 1.0

# ------------------------------------------------------------
# RAW TRACE FIX - 260505 UPDATE
# ------------------------------------------------------------
# The previous trace panel used cleaned/low-pass traces, which can look too
# flat/smooth. These options keep an unsmoothed baseline-corrected copy for
# raw-trace plotting, CSV export, LFP metric extraction, and raw correlation.
SAVE_UNSMOOTHED_RAW_TRACES = True
RAW_BASELINE_CORRECT = True
RAW_BASELINE_WINDOW_MS = (-50, 0)       # subtract median over this pre-t0 window
RAW_TRACE_DISPLAY_GAIN = 1.0            # increase to 2-5 only for visual amplification
RAW_TRACE_SPACING_MODE = "robust"       # "robust" or "max"
RAW_TRACE_SPACING_PCT = 95.0            # robust spacing percentile for stacked traces
RAW_TRACE_MIN_SPACING = 1e-9
SAVE_WIDE_TRACE_CSV = True              # rows=time, columns=ROI electrodes
SAVE_LONG_TRACE_CSV = False             # optional larger file: one row per time×electrode

# ------------------------------------------------------------
# LFP METRIC EXPORT FOR LATER BARPLOTS - 260505 UPDATE
# ------------------------------------------------------------
SAVE_LFP_METRIC_CSV = True
SAVE_LFP_METRIC_BARPLOTS = True
GENOTYPE_LABEL = "WT"   # default; override with --genotype-label (must match one of your data/processed_2ms/ group folder names)
LFP_METRIC_POST_START_MS = 0
LFP_METRIC_POST_END_MS = 250
LFP_ENERGY_WINDOW_MS = (0, 250)
LFP_SLOPE_WINDOW_MS = (0, 80)           # Methods: maximal |slope| within the first 80 ms after onset
LFP_LOCAL_NORM_SLOPE_EPS = 1e-12

# ------------------------------------------------------------
# ROI CSD TIME-SNAPSHOT ENTROPY - 260505 UPDATE
# ------------------------------------------------------------
SAVE_CSD_TIME_ENTROPY = True
CSD_ENTROPY_FOLDER = "ROI_CSD_time_entropy_bits"
CSD_ENTROPY_POWER = 2.0   # p_i from |CSD_i|**power; 2.0 = energy distribution
CSD_ENTROPY_EPS = 1e-12
SAVE_CSD_ENTROPY_HEATMAP = True

# ------------------------------------------------------------
# RAW-DATA CORRELATION OUTPUTS - 260505 UPDATE
# Existing cleaned correlation outputs are preserved. These add a separate
# raw/unsmoothed correlation folder and summary for later groupwise comparison.
# ------------------------------------------------------------
DO_RAW_CORRELATION_ANALYSIS = True
SAVE_RAW_DRUG_ROI_SIMILARITY = True
RAW_CORR_FOLDER_NAME = "correlation_raw_unsmoothed"

# Kymograph smoothing. Set to None or (0, 0) to disable.
KYMOGRAPH_SMOOTH_SIGMA = (0.2, 0.4)

# Signal cleaning / chunking from your pipeline
CHUNK_SIZE = 64
SNAP_CONTEXT_MS = 200
SNAP_HALF_WIN = 5
CLEAN_CUTOFF_HZ = 150
MEDIAN_KERNEL = 5

FIG_DPI = 600
SAVE_SVG = True

# ------------------------------------------------------------
# CORRELATION ANALYSIS SETTINGS - ADDED, OLD OUTPUTS PRESERVED
# ------------------------------------------------------------
DO_CORRELATION_ANALYSIS = True
CORR_CMAP = "coolwarm"
CORR_VLIM = 1.0

# Save ROI-level channel x channel matrices
SAVE_ROI_CORR_MATRICES = True
SAVE_ROI_CORR_FIGURES = True

# Save across-drug / across-ROI similarity matrices after all files are processed.
# These use the ROI subunit correlation pattern as the feature vector.
SAVE_DRUG_ROI_SIMILARITY = True

# Feature vector used for drug/ROI similarity:
#   lfp_upper = upper triangle of LFP channel-channel correlation
#   csd_upper = upper triangle of CSD channel-channel correlation
#   lfp_csd_flat = full LFP x CSD cross-correlation matrix
USE_LFP_CORR_FOR_FEATURES = True
USE_CSD_CORR_FOR_FEATURES = True
USE_LFP_CSD_CORR_FOR_FEATURES = True

# ============================================================
# STYLE / SAVE
# ============================================================
def set_nature_style():
    plt.rcParams.update({
        "figure.dpi": 600,
        "savefig.dpi": FIG_DPI,
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


def save_fig(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def sanitize_name(x):
    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9_\-]+", "_", x)
    return x if x else "ROI"

# ============================================================
# FILE / META
# ============================================================
def get_h5_files():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
    if not paths:
        raise FileNotFoundError(f"No H5 files found in {DATA_DIR} with pattern {FILE_PATTERN}")
    print(f"[INFO] Found {len(paths)} files")
    for p in paths:
        print("  ", os.path.basename(p))
    return paths


def get_drug_name(path):
    base = os.path.basename(path)
    for k in ["4AP", "Bic", "CNQX", "ML297", "AF-DX", "AFDX", "baseline", "base", "veh"]:
        if k.lower() in base.lower():
            if k.lower() in ["af-dx", "afdx"]:
                return "AF-DX"
            if k.lower() in ["baseline", "base", "veh"]:
                return "4AP"          # 4-AP baseline recording
            return k
    return os.path.splitext(base)[0]


def load_meta(f):
    m = f["/meta"].attrs
    return int(m["H"]), int(m["W"]), float(m["fs"])


def load_mask_from_h5_or_npy(f, H, W):
    if MASK_PATH and os.path.exists(MASK_PATH):
        m = np.load(MASK_PATH).astype(bool)
        if m.shape != (H, W):
            raise ValueError(f"MASK_PATH shape {m.shape} != ({H},{W})")
        return m

    if "/mask" in f:
        m = f["/mask"][:].astype(bool)
        if m.shape != (H, W):
            if m.size == H * W:
                m = m.reshape(H, W)
            else:
                raise ValueError(f"/mask shape {m.shape} incompatible with ({H},{W})")
        return m

    return np.ones((H, W), dtype=bool)

# ============================================================
# TIFF-ALIGNED ROI PICKER - PRESERVED FROM YOUR PIPELINE
# ============================================================
def align_tif_to_mask(mask, tif_path):
    """Load the slice image that is already rectified to the electrode field (the
    ``<slice>__rectified_grid.png`` written by preprocessing/tissue_registration_mask.py, or a
    TIFF/PNG/JPG whose full extent equals the 64x64 array) and align it to the mask bbox."""
    from skimage.transform import resize

    H, W = mask.shape
    if str(tif_path).lower().endswith((".tif", ".tiff")):
        import tifffile
        img = tifffile.imread(tif_path)
        if img.ndim == 3 and img.shape[0] < 10:          # stack: first page
            img = img[0]
        elif img.ndim == 3 and img.shape[0] >= 10:        # z-stack: middle page / or RGB
            img = img[img.shape[0] // 2] if img.shape[-1] > 4 else img.mean(axis=-1)
        elif img.ndim == 4:
            img = img[0, img.shape[1] // 2]
    else:
        from PIL import Image
        img = np.asarray(Image.open(tif_path).convert("L"))
    print("[DEBUG] raw image shape:", img.shape)

    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)

    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    print("[DEBUG] mask bbox:", y0, y1, x0, x1)

    h_img, w_img = img.shape
    y_scale = h_img / H
    x_scale = w_img / W

    iy0 = int(y0 * y_scale)
    iy1 = int(y1 * y_scale)
    ix0 = int(x0 * x_scale)
    ix1 = int(x1 * x_scale)

    img_crop = img[iy0:iy1, ix0:ix1]
    img_resized = resize(img_crop, (H, W), anti_aliasing=True)
    return img_resized


def roi_from_center(r, c, H, W, size=4):
    """
    Fixed-size ROI centered on clicked point.
    Keeps ROI_SIZE x ROI_SIZE even near boundaries by shifting inside grid.
    """
    half_low = size // 2
    half_high = size - half_low

    r0 = r - half_low
    r1 = r + half_high
    c0 = c - half_low
    c1 = c + half_high

    if r0 < 0:
        r1 += -r0
        r0 = 0
    if c0 < 0:
        c1 += -c0
        c0 = 0
    if r1 > H:
        shift = r1 - H
        r0 -= shift
        r1 = H
    if c1 > W:
        shift = c1 - W
        c0 -= shift
        c1 = W

    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(H, r1)
    c1 = min(W, c1)

    return slice(r0, r1), slice(c0, c1)


def load_rois_from_csv(roi_csv):
    """Non-interactive alternative to pick_rois(): read ROI boxes from a roi_definitions.csv
    (columns roi_index, roi_name, r0, r1, c0, c1; r1/c1 exclusive) and return the same
    (rois, roi_names) structure the picker returns."""
    if not os.path.exists(roi_csv):
        raise FileNotFoundError(f"ROI definitions not found: {roi_csv}")
    rois, roi_names = [], []
    with open(roi_csv, newline="") as f:
        for row in csv.DictReader(f):
            r0, r1, c0, c1 = int(row["r0"]), int(row["r1"]), int(row["c0"]), int(row["c1"])
            name = sanitize_name(row["roi_name"])
            rois.append((slice(r0, r1), slice(c0, c1)))
            roi_names.append(name)
            print(f"  [ROI] {name}: rows {r0}:{r1}, cols {c0}:{c1}  (from {os.path.basename(roi_csv)})")
    if not rois:
        raise ValueError(f"No ROI rows found in {roi_csv}")
    return rois, roi_names


def pick_rois(H, W, mask):
    """
    SAME ROI picker style as your finalized pipeline:
    - aligns TIF to 64x64 mask if TIF exists
    - overlays mask
    - shows grid
    - clicks ROI centers once
    - asks ROI names
    """
    N = int(input("Enter number of ROIs: "))

    fig, ax = plt.subplots(figsize=(6, 6))

    if os.path.exists(TIF_PATH):
        try:
            aligned_img = align_tif_to_mask(mask, TIF_PATH)
            ax.imshow(aligned_img, cmap="gray")
        except Exception as e:
            print("[ERROR] TIFF alignment failed:", e)
            ax.imshow(np.zeros((H, W)), cmap="gray")
    else:
        print("[WARN] no TIF found. Showing mask only.")
        ax.imshow(np.zeros((H, W)), cmap="gray")

    ax.imshow(mask.astype(float), cmap="gray", alpha=0.35)

    for i in range(H):
        ax.axhline(i, color="white", alpha=0.05, linewidth=0.5)
    for j in range(W):
        ax.axvline(j, color="white", alpha=0.05, linewidth=0.5)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f"Click {N} ROI centers")
    pts = plt.ginput(N, timeout=-1)
    plt.close(fig)

    rois = []
    roi_names = []
    for i, (x, y) in enumerate(pts):
        r = int(round(y))
        c = int(round(x))
        roi = roi_from_center(r, c, H, W, ROI_SIZE)
        default_name = f"ROI_{i}_{roi[0].start}_{roi[1].start}"
        name = input(f"Name for ROI {i} at center(row={r}, col={c}) [{default_name}]: ").strip()
        if not name:
            name = default_name
        name = sanitize_name(name)
        rois.append(roi)
        roi_names.append(name)
        print(f"  [ROI] {name}: rows {roi[0].start}:{roi[0].stop}, cols {roi[1].start}:{roi[1].stop}")

    return rois, roi_names

# ============================================================
# SIGNAL CLEANING
# ============================================================
def clean_signal_single(x, fs, cutoff=CLEAN_CUTOFF_HZ):
    x = np.asarray(x, dtype=np.float32)
    if x.size < 12:
        return x - np.median(x)

    if MEDIAN_KERNEL > 1 and x.size >= MEDIAN_KERNEL:
        x = medfilt(x, kernel_size=MEDIAN_KERNEL)

    nyq = 0.5 * fs
    norm = cutoff / nyq
    if norm >= 1.0:
        return (x - np.median(x)).astype(np.float32)

    b, a = butter(3, norm, btype="low")
    padlen = 3 * max(len(a), len(b))

    if x.size > padlen:
        x = filtfilt(b, a, x)
    else:
        x = np.convolve(x, b, mode="same")

    x = x - np.median(x)
    return x.astype(np.float32)


def clean_channel_matrix(X, fs):
    X = np.asarray(X, dtype=np.float32)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = clean_signal_single(X[i], fs)
    return out

# ============================================================
# EXTRACTION
# ============================================================
def find_t0_chunked(dset, chunk_size=CHUNK_SIZE, mask_flat_idx=None):
    n_ch, T = dset.shape
    score = np.zeros(T, dtype=np.float64)

    if mask_flat_idx is not None:
        idx = np.asarray(mask_flat_idx, dtype=int)
        for j0 in range(0, len(idx), chunk_size):
            j1 = min(j0 + chunk_size, len(idx))
            X = dset[idx[j0:j1], :].astype(np.float32)
            score += np.sum(np.abs(X), axis=0)
    else:
        for i0 in range(0, n_ch, chunk_size):
            i1 = min(i0 + chunk_size, n_ch)
            X = dset[i0:i1, :].astype(np.float32)
            score += np.sum(np.abs(X), axis=0)

    return int(np.argmax(score))


def extract_snapshots_cleaned_chunked(dset, H, W, fs, t0,
                                      snap_ms_list=SNAP_MS,
                                      half_win=SNAP_HALF_WIN,
                                      chunk_size=CHUNK_SIZE,
                                      context_ms=SNAP_CONTEXT_MS):
    dt = 1000.0 / fs
    n_ch, T = dset.shape
    frames = []
    ctx = int(round(context_ms / dt))

    for ms in snap_ms_list:
        t = int(round(t0 + ms / dt))
        t = max(0, min(T - 1, t))
        a = max(0, t - half_win)
        b = min(T, t + half_win + 1)

        ca = max(0, a - ctx)
        cb = min(T, b + ctx)
        vals = np.zeros(n_ch, dtype=np.float32)

        for i0 in range(0, n_ch, chunk_size):
            i1 = min(i0 + chunk_size, n_ch)
            X = dset[i0:i1, ca:cb].astype(np.float32)
            Xc = clean_channel_matrix(X, fs)
            aa = a - ca
            bb = b - ca
            vals[i0:i1] = np.median(Xc[:, aa:bb], axis=1)

        frames.append(vals)

    return np.stack(frames, axis=0)


def roi_channel_indices(roi, W):
    idx = []
    for r in range(roi[0].start, roi[0].stop):
        for c in range(roi[1].start, roi[1].stop):
            idx.append(r * W + c)
    return np.asarray(idx, dtype=int)


def extract_roi_waveforms_cleaned(dset, roi, H, W, t0, fs):
    dt = 1000.0 / fs
    pre = int(round(PRE_MS / dt))
    post = int(round(POST_MS / dt))
    target_len = pre + post

    start_req = t0 - pre
    end_req = t0 + post

    pad_left = max(0, -start_req)
    pad_right = max(0, end_req - dset.shape[1])

    start = max(0, start_req)
    end = min(dset.shape[1], end_req)

    idx = roi_channel_indices(roi, W)
    X = dset[idx, start:end].astype(np.float32)

    if pad_left > 0 or pad_right > 0:
        X = np.pad(X, ((0, 0), (pad_left, pad_right)), mode="constant")

    if X.shape[1] < target_len:
        X = np.pad(X, ((0, 0), (0, target_len - X.shape[1])), mode="constant")
    elif X.shape[1] > target_len:
        X = X[:, :target_len]

    X = clean_channel_matrix(X, fs)
    return X



# ============================================================
# RAW UNSMOOTHED ROI TRACE / CSV / LFP METRICS / CSD ENTROPY
# 260505 UPDATE - OLD CLEANED EXTRACTION ABOVE IS PRESERVED
# ============================================================
def get_roi_time_axis_ms(fs):
    """Time axis for ROI waveforms, matching PRE_MS/POST_MS and the real dt."""
    dt = 1000.0 / float(fs)
    pre = int(round(PRE_MS / dt))
    post = int(round(POST_MS / dt))
    return (np.arange(pre + post, dtype=np.float32) - pre) * dt


def baseline_correct_waveform(X, time_ms, baseline_window_ms=RAW_BASELINE_WINDOW_MS):
    """Subtract per-channel median baseline from an already-extracted waveform."""
    X = np.asarray(X, dtype=np.float32).copy()
    if not RAW_BASELINE_CORRECT:
        return X

    b0, b1 = baseline_window_ms
    bm = (time_ms >= b0) & (time_ms < b1)
    if not np.any(bm):
        bm = time_ms < 0
    if not np.any(bm):
        base = np.nanmedian(X, axis=1, keepdims=True)
    else:
        base = np.nanmedian(X[:, bm], axis=1, keepdims=True)
    return (X - base).astype(np.float32)


def extract_roi_waveforms_raw_unsmoothed(dset, roi, H, W, t0, fs):
    """
    Extract ROI traces with NO median filter and NO low-pass filter.
    Only optional baseline correction is applied. This fixes overly smoothed/flat
    raw-trace panels and provides the raw analysis copy for CSV/correlation.
    """
    dt = 1000.0 / fs
    pre = int(round(PRE_MS / dt))
    post = int(round(POST_MS / dt))
    target_len = pre + post

    start_req = t0 - pre
    end_req = t0 + post

    pad_left = max(0, -start_req)
    pad_right = max(0, end_req - dset.shape[1])

    start = max(0, start_req)
    end = min(dset.shape[1], end_req)

    idx = roi_channel_indices(roi, W)
    X = dset[idx, start:end].astype(np.float32)

    if pad_left > 0 or pad_right > 0:
        X = np.pad(X, ((0, 0), (pad_left, pad_right)), mode="constant")

    if X.shape[1] < target_len:
        X = np.pad(X, ((0, 0), (0, target_len - X.shape[1])), mode="constant")
    elif X.shape[1] > target_len:
        X = X[:, :target_len]

    time_ms = get_roi_time_axis_ms(fs)
    X = baseline_correct_waveform(X, time_ms)
    return X.astype(np.float32)


def save_waveform_csv_wide(wave, time_ms, path, prefix="ch"):
    """Save ROI waveform as CSV: time_ms + one column per ROI electrode."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wave = np.asarray(wave, dtype=np.float32)
    header = ["time_ms"] + [f"{prefix}_{i:02d}" for i in range(wave.shape[0])]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for j, t in enumerate(time_ms):
            w.writerow([float(t)] + [float(wave[i, j]) for i in range(wave.shape[0])])


def save_waveform_csv_long(wave, time_ms, path, signal_name, drug, file_name, roi_i, roi_name):
    """Optional long-format CSV: one row per electrode×time."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wave = np.asarray(wave, dtype=np.float32)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genotype", "drug", "file", "roi_index", "roi_name", "signal", "electrode", "time_ms", "value"])
        for ch in range(wave.shape[0]):
            for j, t in enumerate(time_ms):
                w.writerow([GENOTYPE_LABEL, drug, file_name, roi_i, roi_name, signal_name, ch, float(t), float(wave[ch, j])])


def save_rows_csv_dict(rows, path, fieldnames=None):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _mask_between(time_ms, start_ms, end_ms):
    return (time_ms >= start_ms) & (time_ms <= end_ms)


def compute_lfp_metrics_from_trace(trace, time_ms):
    """
    Metrics saved for later WT/5xFAD barplots:
    - amplitude: absolute peak after t0 in configured post window
    - energy: integral(trace^2 dt) and mean power over configured window
    - normalized slope: local slope normalized by local peak amplitude
      Later genotype-level normalization can be done from the exported CSV.
    """
    trace = np.asarray(trace, dtype=np.float32)
    post = _mask_between(time_ms, LFP_METRIC_POST_START_MS, LFP_METRIC_POST_END_MS)
    if not np.any(post):
        post = time_ms >= 0
    y_post = trace[post]

    if y_post.size == 0:
        peak_pos = peak_neg = amp_abs = p2p = np.nan
    else:
        peak_pos = float(np.nanmax(y_post))
        peak_neg = float(np.nanmin(y_post))
        amp_abs = float(np.nanmax(np.abs(y_post)))
        p2p = float(peak_pos - peak_neg)

    em = _mask_between(time_ms, LFP_ENERGY_WINDOW_MS[0], LFP_ENERGY_WINDOW_MS[1])
    if np.sum(em) >= 2:
        energy = float(np.trapezoid(trace[em] ** 2, time_ms[em]))
        mean_power = float(np.nanmean(trace[em] ** 2))
    else:
        energy = np.nan
        mean_power = np.nan

    sm = _mask_between(time_ms, LFP_SLOPE_WINDOW_MS[0], LFP_SLOPE_WINDOW_MS[1])
    if np.sum(sm) >= 2:
        coef = np.polyfit(time_ms[sm].astype(float), trace[sm].astype(float), 1)
        slope = float(coef[0])
        slope_abs = float(abs(slope))
    else:
        slope = np.nan
        slope_abs = np.nan

    slope_dur = float(LFP_SLOPE_WINDOW_MS[1] - LFP_SLOPE_WINDOW_MS[0])
    if np.isfinite(amp_abs) and amp_abs > LFP_LOCAL_NORM_SLOPE_EPS and np.isfinite(slope_abs):
        norm_slope_local_pct = float(100.0 * slope_abs * slope_dur / (amp_abs + LFP_LOCAL_NORM_SLOPE_EPS))
    else:
        norm_slope_local_pct = np.nan

    return {
        "lfp_peak_positive": peak_pos,
        "lfp_peak_negative": peak_neg,
        "lfp_amplitude_abs_peak": amp_abs,
        "lfp_peak_to_peak": p2p,
        "lfp_energy_integral_value2_ms": energy,
        "lfp_mean_power_value2": mean_power,
        "lfp_slope_value_per_ms": slope,
        "lfp_abs_slope_value_per_ms": slope_abs,
        "lfp_local_norm_slope_pct": norm_slope_local_pct,
    }


def make_lfp_metric_rows(lfp_wave_raw, time_ms, drug, file_name, t0, roi_i, roi_name):
    rows = []
    lfp_wave_raw = np.asarray(lfp_wave_raw, dtype=np.float32)

    for ch in range(lfp_wave_raw.shape[0]):
        m = compute_lfp_metrics_from_trace(lfp_wave_raw[ch], time_ms)
        row = {
            "genotype": GENOTYPE_LABEL,
            "drug": drug,
            "file": file_name,
            "t0": int(t0),
            "roi_index": int(roi_i),
            "roi_name": roi_name,
            "trace_id": f"ch_{ch:02d}",
            "n_channels_in_roi": int(lfp_wave_raw.shape[0]),
            "metric_post_window_ms": f"{LFP_METRIC_POST_START_MS}-{LFP_METRIC_POST_END_MS}",
            "energy_window_ms": f"{LFP_ENERGY_WINDOW_MS[0]}-{LFP_ENERGY_WINDOW_MS[1]}",
            "slope_window_ms": f"{LFP_SLOPE_WINDOW_MS[0]}-{LFP_SLOPE_WINDOW_MS[1]}",
        }
        row.update(m)
        rows.append(row)

    roi_mean = np.nanmean(lfp_wave_raw, axis=0)
    m = compute_lfp_metrics_from_trace(roi_mean, time_ms)
    row = {
        "genotype": GENOTYPE_LABEL,
        "drug": drug,
        "file": file_name,
        "t0": int(t0),
        "roi_index": int(roi_i),
        "roi_name": roi_name,
        "trace_id": "roi_mean",
        "n_channels_in_roi": int(lfp_wave_raw.shape[0]),
        "metric_post_window_ms": f"{LFP_METRIC_POST_START_MS}-{LFP_METRIC_POST_END_MS}",
        "energy_window_ms": f"{LFP_ENERGY_WINDOW_MS[0]}-{LFP_ENERGY_WINDOW_MS[1]}",
        "slope_window_ms": f"{LFP_SLOPE_WINDOW_MS[0]}-{LFP_SLOPE_WINDOW_MS[1]}",
    }
    row.update(m)
    rows.append(row)
    return rows


def save_lfp_metric_barplots(metric_rows, out_dir):
    """Quick QC barplots with scatter; final WT/5xFAD stats can use exported CSV."""
    if not metric_rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    rows = [r for r in metric_rows if r.get("trace_id") == "roi_mean"]
    if not rows:
        return

    metrics = [
        ("lfp_amplitude_abs_peak", "Amplitude [LFP]"),
        ("lfp_energy_integral_value2_ms", "Energy (value²·ms)"),
        ("lfp_local_norm_slope_pct", "Normalized EPSC/LFP slope (%)"),
        ("lfp_abs_slope_value_per_ms", "Absolute slope (value/ms)"),
    ]
    drugs = sorted(set(r["drug"] for r in rows))

    for metric, ylabel in metrics:
        set_nature_style()
        fig, ax = plt.subplots(figsize=(max(3.0, 0.7 * len(drugs) + 1.4), 3.2))
        xs = np.arange(len(drugs), dtype=float)
        means, sems, all_vals = [], [], []
        for d in drugs:
            vals = np.array([float(r[metric]) for r in rows if r["drug"] == d], dtype=float)
            vals = vals[np.isfinite(vals)]
            all_vals.append(vals)
            means.append(float(np.nanmean(vals)) if vals.size else np.nan)
            sems.append(float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)

        ax.bar(xs, means, yerr=sems, width=0.55, color="0.82", edgecolor="black", linewidth=1.0, capsize=3)
        for x, vals in zip(xs, all_vals):
            if vals.size:
                jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(np.full(vals.size, x) + jitter, vals, s=18, color="black", zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(drugs, rotation=0)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        save_fig(fig, os.path.join(out_dir, f"{metric}_barplot"))


def shannon_entropy_bits(values, power=CSD_ENTROPY_POWER, eps=CSD_ENTROPY_EPS):
    """Entropy of spatial CSD distribution inside ROI at one time snapshot."""
    v = np.asarray(values, dtype=np.float64).ravel()
    weights = np.abs(v) ** float(power)
    total = float(np.nansum(weights))
    if not np.isfinite(total) or total <= eps:
        n = int(v.size)
        return 0.0, 0.0, 0.0, total

    p = weights / (total + eps)
    p = p[np.isfinite(p) & (p > eps)]
    H = float(-np.sum(p * np.log2(p)))
    Hmax = float(np.log2(v.size)) if v.size > 1 else 1.0
    Hnorm = float(H / Hmax) if Hmax > 0 else 0.0
    Neff = float(2.0 ** H)
    return H, Hnorm, Neff, total


def save_csd_entropy_heatmap(roi_csd_values, snap_ms, out_base, title):
    set_nature_style()
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    vlim = robust_vlim(roi_csd_values, pct=99.0)
    im = ax.imshow(
        roi_csd_values,
        aspect="auto",
        interpolation="nearest",
        cmap=CSD_CMAP,
        vmin=-vlim,
        vmax=vlim,
        extent=[-0.5, roi_csd_values.shape[1] - 0.5, snap_ms[-1], snap_ms[0]],
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("ROI electrode")
    ax.set_ylabel("Snapshot time (ms)")
    ax.set_xticks(np.arange(roi_csd_values.shape[1]))
    ax.set_yticks(snap_ms)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("CSD")
    save_fig(fig, out_base)


def save_csd_time_entropy_outputs(csd_snaps, H, W, roi, drug, file_name, t0, roi_i, roi_name, entropy_root):
    """
    Separate-folder output for ROI CSD time snapshots + entropy in bits.
    Returns rows for global summary CSV.
    """
    roi_out = os.path.join(entropy_root, drug, f"ROI_{roi_i:02d}_{roi_name}")
    os.makedirs(roi_out, exist_ok=True)

    idx = roi_channel_indices(roi, W)
    roi_csd_values = np.asarray(csd_snaps[:, idx], dtype=np.float32)  # snapshots × ROI electrodes
    np.save(os.path.join(roi_out, "roi_csd_time_snap_values.npy"), roi_csd_values)

    rows = []
    csv_path = os.path.join(roi_out, "roi_csd_time_snap_entropy_bits.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        ch_cols = [f"csd_ch_{i:02d}" for i in range(roi_csd_values.shape[1])]
        w.writerow([
            "genotype", "drug", "file", "t0", "roi_index", "roi_name", "snap_index", "snap_ms",
            "entropy_bits", "normalized_entropy_0to1", "effective_electrodes_2powH",
            "total_abs_csd", "total_power_csd", "entropy_power"
        ] + ch_cols)

        for si, ms in enumerate(SNAP_MS):
            vals = roi_csd_values[si]
            Hbits, Hnorm, Neff, total_power = shannon_entropy_bits(vals)
            total_abs = float(np.nansum(np.abs(vals)))
            row = {
                "genotype": GENOTYPE_LABEL,
                "drug": drug,
                "file": file_name,
                "t0": int(t0),
                "roi_index": int(roi_i),
                "roi_name": roi_name,
                "snap_index": int(si),
                "snap_ms": float(ms),
                "entropy_bits": float(Hbits),
                "normalized_entropy_0to1": float(Hnorm),
                "effective_electrodes_2powH": float(Neff),
                "total_abs_csd": total_abs,
                "total_power_csd": float(total_power),
                "entropy_power": float(CSD_ENTROPY_POWER),
            }
            rows.append(row)
            w.writerow([
                GENOTYPE_LABEL, drug, file_name, int(t0), int(roi_i), roi_name, int(si), float(ms),
                float(Hbits), float(Hnorm), float(Neff), total_abs, float(total_power), float(CSD_ENTROPY_POWER)
            ] + [float(x) for x in vals])

    # Barplot of entropy across snapshot time points
    set_nature_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    x = np.arange(len(SNAP_MS))
    y = np.array([r["entropy_bits"] for r in rows], dtype=float)
    ax.bar(x, y, width=0.62, color="0.78", edgecolor="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(ms) for ms in SNAP_MS])
    ax.set_xlabel("CSD snapshot time from t0 (ms)")
    ax.set_ylabel("CSD spatial entropy (bits)")
    ax.set_title(f"{drug} | {roi_name}", fontsize=9, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, os.path.join(roi_out, "roi_csd_entropy_bits_barplot"))

    if SAVE_CSD_ENTROPY_HEATMAP:
        save_csd_entropy_heatmap(
            roi_csd_values,
            np.asarray(SNAP_MS, dtype=float),
            os.path.join(roi_out, "roi_csd_time_snap_heatmap"),
            f"{drug} | {roi_name} | CSD time snapshots",
        )

    return rows


def save_csd_entropy_summary_plots(entropy_rows, out_dir):
    if not entropy_rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    drugs = sorted(set(r["drug"] for r in entropy_rows))
    snap_vals = list(SNAP_MS)

    # Line plot: entropy vs snapshot time, one curve per drug.
    set_nature_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for d in drugs:
        means, sems = [], []
        for ms in snap_vals:
            vals = np.array([float(r["entropy_bits"]) for r in entropy_rows if r["drug"] == d and float(r["snap_ms"]) == float(ms)], dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.nanmean(vals)) if vals.size else np.nan)
            sems.append(float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
        ax.errorbar(snap_vals, means, yerr=sems, marker="o", linewidth=1.3, capsize=3, label=d)
    ax.set_xlabel("CSD snapshot time from t0 (ms)")
    ax.set_ylabel("CSD spatial entropy (bits)")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, os.path.join(out_dir, "summary_csd_entropy_bits_over_time"))

    # Barplot: mean entropy collapsed across snapshots.
    set_nature_style()
    fig, ax = plt.subplots(figsize=(max(3.0, 0.7 * len(drugs) + 1.4), 3.2))
    xs = np.arange(len(drugs), dtype=float)
    means, sems, all_vals = [], [], []
    for d in drugs:
        vals = np.array([float(r["entropy_bits"]) for r in entropy_rows if r["drug"] == d], dtype=float)
        vals = vals[np.isfinite(vals)]
        all_vals.append(vals)
        means.append(float(np.nanmean(vals)) if vals.size else np.nan)
        sems.append(float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
    ax.bar(xs, means, yerr=sems, width=0.55, color="0.78", edgecolor="black", linewidth=1.0, capsize=3)
    for x, vals in zip(xs, all_vals):
        if vals.size:
            jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(np.full(vals.size, x) + jitter, vals, s=14, color="black", zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(drugs)
    ax.set_ylabel("CSD spatial entropy (bits)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, os.path.join(out_dir, "summary_csd_entropy_bits_barplot"))


def make_kymograph(wave):
    if KYMOGRAPH_SMOOTH_SIGMA is not None and tuple(KYMOGRAPH_SMOOTH_SIGMA) != (0, 0):
        return gaussian_filter(wave, sigma=KYMOGRAPH_SMOOTH_SIGMA)
    return wave


def robust_vlim(arr, pct=99.0):
    arr = np.asarray(arr)
    v = np.nanpercentile(np.abs(arr), pct)
    if not np.isfinite(v) or v <= 0:
        v = np.nanmax(np.abs(arr))
    if not np.isfinite(v) or v <= 0:
        v = 1.0
    return float(v)

# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_mask_outline(ax, mask):
    ax.contour(mask.astype(float), levels=[0.5], colors=MASK_OUTLINE_COLOR, linewidths=1.0)


def draw_roi_box(ax, roi, label=None):
    r0, r1 = roi[0].start, roi[0].stop
    c0, c1 = roi[1].start, roi[1].stop
    rect = plt.Rectangle(
        (c0 - 0.5, r0 - 0.5),
        c1 - c0,
        r1 - r0,
        fill=False,
        edgecolor=ROI_COLOR,
        linewidth=2.0,
        zorder=5,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            c0 - 0.3,
            max(0, r0 - 1.2),
            label,
            color=ROI_COLOR,
            fontsize=7,
            weight="bold",
            ha="left",
            va="bottom",
            zorder=6,
        )

# ============================================================
# PLOTS
# ============================================================
def save_individual_snapshot(frame_vec, H, W, mask, roi, roi_name, ms,
                             vlim, cmap_name, title, out_base):
    set_nature_style()
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    frame = frame_vec.reshape(H, W).astype(np.float32)
    frame = np.where(mask, frame, np.nan)
    im = ax.imshow(frame, cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")
    draw_mask_outline(ax, mask)
    draw_roi_box(ax, roi, label=roi_name)
    ax.set_title(f"{title} | {ms} ms", fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.0)
    save_fig(fig, out_base)


def save_snapshot_grid_for_roi(frames, H, W, mask, roi, roi_name,
                               vlim, cmap_name, title, out_base):
    set_nature_style()
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    K = len(SNAP_MS)
    ncols = 3
    nrows = int(np.ceil(K / ncols))
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1, 1, 1, 0.055], wspace=0.08, hspace=0.15)

    im = None
    for i, ms in enumerate(SNAP_MS):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        frame = frames[i].reshape(H, W).astype(np.float32)
        frame = np.where(mask, frame, np.nan)
        im = ax.imshow(frame, cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")
        draw_mask_outline(ax, mask)
        draw_roi_box(ax, roi, label=roi_name if i == 0 else None)
        ax.set_title(f"{ms} ms", fontsize=9, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for j in range(K, nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=8, width=0.6, length=2.5)
    fig.suptitle(title, fontsize=10, y=0.995)
    save_fig(fig, out_base)


def plot_stacked_raw_traces(ax, wave, title, ylabel="Channel"):
    n_ch, n_t = wave.shape
    t = np.linspace(-PRE_MS, POST_MS, n_t)
    mask_t = (t >= DISPLAY_PRE_MS) & (t <= DISPLAY_POST_MS)
    tt = t[mask_t]
    X = wave[:, mask_t].astype(np.float32)

    X = X * float(RAW_TRACE_DISPLAY_GAIN)

    if RAW_TRACE_SPACING_MODE.lower() == "robust":
        amp = np.nanpercentile(np.abs(X), RAW_TRACE_SPACING_PCT)
    else:
        amp = np.nanmax(np.abs(X))
    spacing = float(amp) * RAW_TRACE_SPACING_SCALE
    if not np.isfinite(spacing) or spacing <= RAW_TRACE_MIN_SPACING:
        spacing = 1.0

    # Unsmoothed, baseline-corrected trace display. Robust spacing prevents one
    # large channel from making all other ROI traces look flat.
    for i in range(n_ch):
        offset = i * spacing
        ax.plot(tt, X[i] + offset, color="black", lw=RAW_TRACE_LINEWIDTH, alpha=0.95)

    ax.axvline(0, color="0.65", lw=0.7)
    ax.set_ylim(-0.75 * spacing, (n_ch - 0.25) * spacing)
    ax.set_xlim(tt[0], tt[-1])
    ax.set_title(title, pad=4, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(1, n_ch + 1, 2) * spacing)
    ax.set_yticklabels([str(v) for v in range(2, n_ch + 1, 2)])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def plot_kymograph(ax, kymo, vlim, cmap_name, title, cbar_label):
    im = ax.imshow(
        kymo,
        aspect="auto",
        origin="lower",
        cmap=cmap_name,
        vmin=-vlim,
        vmax=vlim,
        extent=[0, POST_MS, 0, kymo.shape[0]],
        interpolation="nearest",
    )
    ax.set_xlim(0, POST_MS)
    ax.set_title(title, pad=4, weight="bold")
    ax.set_xlabel("Time relative to t0 (ms)")
    ax.set_ylabel("ROI electrode")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label)
    return im


def save_raw_kymo_panel(lfp_wave, csd_wave, lfp_kymo, csd_kymo,
                        lfp_vlim, csd_vlim, drug, roi_name, out_base):
    set_nature_style()
    fig = plt.figure(figsize=(10.8, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.8], height_ratios=[1, 1], wspace=0.22, hspace=0.28)

    ax_raw_lfp = fig.add_subplot(gs[0, 0])
    plot_stacked_raw_traces(ax_raw_lfp, lfp_wave, "LFP raw traces", ylabel="Channel")

    ax_k_lfp = fig.add_subplot(gs[0, 1])
    plot_kymograph(ax_k_lfp, lfp_kymo, lfp_vlim, LFP_CMAP, "LFP kymograph", "LFP")

    ax_raw_csd = fig.add_subplot(gs[1, 0])
    plot_stacked_raw_traces(ax_raw_csd, csd_wave, "CSD raw traces", ylabel="Channel")

    ax_k_csd = fig.add_subplot(gs[1, 1])
    plot_kymograph(ax_k_csd, csd_kymo, csd_vlim, CSD_CMAP, "CSD kymograph", "CSD")

    fig.suptitle(f"{drug} | {roi_name} | raw traces + kymographs", fontsize=10, y=0.995)
    save_fig(fig, out_base)


def save_raw_traces_only(wave, title, out_base):
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.2, 4.2))
    plot_stacked_raw_traces(ax, wave, title, ylabel="Channel")
    save_fig(fig, out_base)


def save_kymo_only(kymo, vlim, cmap_name, title, cbar_label, out_base):
    set_nature_style()
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    plot_kymograph(ax, kymo, vlim, cmap_name, title, cbar_label)
    save_fig(fig, out_base)


# ============================================================
# CORRELATION ANALYSIS HELPERS - ADDED, OLD FUNCTIONS PRESERVED
# ============================================================
def safe_corrcoef(X):
    """
    X shape: channels x time.
    Returns channels x channels Pearson correlation matrix.
    Handles zero-variance traces safely.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"safe_corrcoef expects 2D array, got shape {X.shape}")

    X = X - np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    std[std == 0] = np.nan

    Z = X / std
    C = np.nanmean(Z[:, None, :] * Z[None, :, :], axis=2)
    C = np.clip(C, -1, 1)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    if C.shape[0] == C.shape[1]:
        np.fill_diagonal(C, 1.0)

    return C.astype(np.float32)


def corr_between_matrices(A, B):
    """
    A shape: channels_A x time
    B shape: channels_B x time
    Returns channels_A x channels_B Pearson correlation matrix.
    This is used for LFP channel i versus CSD channel j inside the same ROI.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"corr_between_matrices expects 2D arrays, got {A.shape} and {B.shape}")

    T = min(A.shape[1], B.shape[1])
    A = A[:, :T]
    B = B[:, :T]

    A = A - np.nanmean(A, axis=1, keepdims=True)
    B = B - np.nanmean(B, axis=1, keepdims=True)

    A_std = np.nanstd(A, axis=1, keepdims=True)
    B_std = np.nanstd(B, axis=1, keepdims=True)

    A_std[A_std == 0] = np.nan
    B_std[B_std == 0] = np.nan

    Az = A / A_std
    Bz = B / B_std

    C = np.nanmean(Az[:, None, :] * Bz[None, :, :], axis=2)
    C = np.clip(C, -1, 1)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    return C.astype(np.float32)


def upper_triangle_values(C):
    """Upper triangle excluding diagonal for square correlation matrices."""
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        return C.ravel()
    iu = np.triu_indices_from(C, k=1)
    return C[iu]


def corr_summary_stats(C):
    """Compact statistics from a correlation matrix."""
    C = np.asarray(C, dtype=np.float32)
    vals = upper_triangle_values(C) if C.shape[0] == C.shape[1] else C.ravel()

    if vals.size == 0:
        vals = np.asarray([np.nan], dtype=np.float32)

    return {
        "mean_r": float(np.nanmean(vals)),
        "median_r": float(np.nanmedian(vals)),
        "std_r": float(np.nanstd(vals)),
        "min_r": float(np.nanmin(vals)),
        "max_r": float(np.nanmax(vals)),
    }


def save_corr_matrix(C, title, out_base, cmap_name=CORR_CMAP, vlim=CORR_VLIM,
                     xlabel="ROI electrode", ylabel="ROI electrode"):
    """Save a correlation heatmap as PNG/SVG."""
    set_nature_style()

    C = np.asarray(C, dtype=np.float32)

    fig_w = max(3.6, min(7.0, 0.22 * C.shape[1] + 2.0))
    fig_h = max(3.2, min(7.0, 0.22 * C.shape[0] + 2.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        C,
        cmap=cmap_name,
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if C.shape[1] <= 32:
        ax.set_xticks(np.arange(C.shape[1]))
        ax.tick_params(axis="x", labelsize=5, rotation=90)
    else:
        ax.set_xticks([])

    if C.shape[0] <= 32:
        ax.set_yticks(np.arange(C.shape[0]))
        ax.tick_params(axis="y", labelsize=5)
    else:
        ax.set_yticks([])

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson r")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.0)

    save_fig(fig, out_base)


def make_roi_correlation_feature_vector(lfp_corr, csd_corr, lfp_csd_corr):
    """
    Convert ROI subunit correlation matrices into one feature vector.
    This vector is later used to compare ROIs/drugs.
    """
    parts = []

    if USE_LFP_CORR_FOR_FEATURES:
        parts.append(upper_triangle_values(lfp_corr).astype(np.float32))

    if USE_CSD_CORR_FOR_FEATURES:
        parts.append(upper_triangle_values(csd_corr).astype(np.float32))

    if USE_LFP_CSD_CORR_FOR_FEATURES:
        parts.append(np.asarray(lfp_csd_corr, dtype=np.float32).ravel())

    if not parts:
        return np.asarray([], dtype=np.float32)

    feat = np.concatenate(parts).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def save_matrix_csv(M, labels, path):
    """Save square labeled matrix as CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"] + list(labels))
        for lab, row in zip(labels, M):
            w.writerow([lab] + [float(x) for x in row])


def save_similarity_heatmap(M, labels, title, out_base):
    """Save a labeled similarity matrix. Labels are shown when not too many."""
    set_nature_style()

    M = np.asarray(M, dtype=np.float32)
    n = M.shape[0]

    fig_size = max(4.2, min(12.0, 0.28 * n + 2.2))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(M, cmap=CORR_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=6)

    if n <= 45:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Drug / ROI unit")
        ax.set_ylabel("Drug / ROI unit")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Feature similarity, Pearson r")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.0)

    save_fig(fig, out_base)


def feature_similarity_matrix(features):
    """
    Pearson correlation between feature vectors.
    features shape: observations x features.
    """
    X = np.asarray(features, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"feature_similarity_matrix expects 2D array, got {X.shape}")

    if X.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float32)

    X = X - np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    std[std == 0] = np.nan
    Z = X / std
    M = np.nanmean(Z[:, None, :] * Z[None, :, :], axis=2)
    M = np.clip(M, -1, 1)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(M, 1.0)
    return M.astype(np.float32)


def average_feature_records(records, key_fields):
    """
    Average feature vectors across records sharing key_fields.
    key_fields example: ["drug", "roi_name"] or ["drug"].
    """
    grouped = {}
    for rec in records:
        key = tuple(rec[k] for k in key_fields)
        grouped.setdefault(key, []).append(rec["feature"])

    out = []
    for key, feats in grouped.items():
        F = np.vstack(feats).astype(np.float32)
        mean_feat = np.nanmean(F, axis=0).astype(np.float32)
        label = " | ".join(str(x) for x in key)
        d = {k: v for k, v in zip(key_fields, key)}
        d["label"] = label
        d["feature"] = mean_feat
        d["n_records"] = len(feats)
        out.append(d)

    out = sorted(out, key=lambda r: r["label"])
    return out


def save_drug_roi_similarity_outputs(records, out_dir, subfolder="drug_roi_correlation_summary", title_prefix=""):
    """
    Creates cross-file, cross-drug, and cross-drug+ROI similarity matrices
    based on ROI subunit correlation feature vectors.
    """
    if not records:
        print("[WARN] No correlation feature records available for drug/ROI similarity.")
        return

    corr_out = os.path.join(out_dir, subfolder)
    os.makedirs(corr_out, exist_ok=True)

    # --------------------------------------------------------
    # 1) Every file x ROI observation
    # --------------------------------------------------------
    labels = [r["label"] for r in records]
    features = np.vstack([r["feature"] for r in records]).astype(np.float32)
    M = feature_similarity_matrix(features)

    np.save(os.path.join(corr_out, "all_file_roi_feature_vectors.npy"), features)
    np.save(os.path.join(corr_out, "all_file_roi_similarity_matrix.npy"), M)
    save_matrix_csv(M, labels, os.path.join(corr_out, "all_file_roi_similarity_matrix.csv"))
    save_similarity_heatmap(
        M,
        labels,
        f"{title_prefix}All file × ROI subunit-correlation similarity",
        os.path.join(corr_out, "all_file_roi_similarity_matrix"),
    )

    with open(os.path.join(corr_out, "all_file_roi_feature_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "drug", "file", "roi_index", "roi_name"])
        for r in records:
            w.writerow([r["label"], r["drug"], r["file"], r["roi_index"], r["roi_name"]])

    # --------------------------------------------------------
    # 2) Average by drug + ROI name
    # --------------------------------------------------------
    drug_roi_records = average_feature_records(records, ["drug", "roi_name"])
    labels_drug_roi = [r["label"] for r in drug_roi_records]
    features_drug_roi = np.vstack([r["feature"] for r in drug_roi_records]).astype(np.float32)
    M_drug_roi = feature_similarity_matrix(features_drug_roi)

    np.save(os.path.join(corr_out, "drug_roi_avg_feature_vectors.npy"), features_drug_roi)
    np.save(os.path.join(corr_out, "drug_roi_avg_similarity_matrix.npy"), M_drug_roi)
    save_matrix_csv(M_drug_roi, labels_drug_roi, os.path.join(corr_out, "drug_roi_avg_similarity_matrix.csv"))
    save_similarity_heatmap(
        M_drug_roi,
        labels_drug_roi,
        f"{title_prefix}Drug × ROI average subunit-correlation similarity",
        os.path.join(corr_out, "drug_roi_avg_similarity_matrix"),
    )

    with open(os.path.join(corr_out, "drug_roi_avg_feature_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "drug", "roi_name", "n_records"])
        for r in drug_roi_records:
            w.writerow([r["label"], r["drug"], r["roi_name"], r["n_records"]])

    # --------------------------------------------------------
    # 3) Average by drug only
    # --------------------------------------------------------
    drug_records = average_feature_records(records, ["drug"])
    labels_drug = [r["label"] for r in drug_records]
    features_drug = np.vstack([r["feature"] for r in drug_records]).astype(np.float32)
    M_drug = feature_similarity_matrix(features_drug)

    np.save(os.path.join(corr_out, "drug_avg_feature_vectors.npy"), features_drug)
    np.save(os.path.join(corr_out, "drug_avg_similarity_matrix.npy"), M_drug)
    save_matrix_csv(M_drug, labels_drug, os.path.join(corr_out, "drug_avg_similarity_matrix.csv"))
    save_similarity_heatmap(
        M_drug,
        labels_drug,
        f"{title_prefix}Drug-average subunit-correlation similarity",
        os.path.join(corr_out, "drug_avg_similarity_matrix"),
    )

    with open(os.path.join(corr_out, "drug_avg_feature_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "drug", "n_records"])
        for r in drug_records:
            w.writerow([r["label"], r["drug"], r["n_records"]])

    print(f"[INFO] Drug/ROI correlation similarity outputs saved to: {corr_out}")

# ============================================================
# GLOBAL SCALE PREPASS
# ============================================================
def compute_global_scales(files, rois, mask):
    need_lfp_snap = not USE_FIXED_LFP_SNAP_VLIM
    need_csd_snap = not USE_FIXED_CSD_SNAP_VLIM
    need_lfp_kymo = not USE_FIXED_LFP_KYMO_VLIM
    need_csd_kymo = not USE_FIXED_CSD_KYMO_VLIM

    vals_lfp_snap = []
    vals_csd_snap = []
    vals_lfp_kymo = []
    vals_csd_kymo = []

    if not any([need_lfp_snap, need_csd_snap, need_lfp_kymo, need_csd_kymo]):
        return {
            "lfp_snap": float(FIXED_LFP_SNAP_VLIM),
            "csd_snap": float(FIXED_CSD_SNAP_VLIM),
            "lfp_kymo": float(FIXED_LFP_KYMO_VLIM),
            "csd_kymo": float(FIXED_CSD_KYMO_VLIM),
        }

    print("[INFO] Computing global vlims because at least one USE_FIXED_* is False...")

    for p in files:
        drug = get_drug_name(p)
        print(f"  [VLIM] {drug}")
        with h5py.File(p, "r") as f:
            H, W, fs = load_meta(f)
            mask2d = load_mask_from_h5_or_npy(f, H, W)
            mask_idx = np.flatnonzero(mask2d.reshape(-1))
            d_lfp = f["/lfp/ts_2ms_full"]
            d_csd = f["/csd/ts_2ms_full"]
            t0 = find_t0_chunked(d_lfp, mask_flat_idx=mask_idx)

            if need_lfp_snap:
                lfp_snaps = extract_snapshots_cleaned_chunked(d_lfp, H, W, fs, t0)
                vals_lfp_snap.append(lfp_snaps[:, mask2d.reshape(-1)].ravel())
            if need_csd_snap:
                csd_snaps = extract_snapshots_cleaned_chunked(d_csd, H, W, fs, t0)
                vals_csd_snap.append(csd_snaps[:, mask2d.reshape(-1)].ravel())

            if need_lfp_kymo or need_csd_kymo:
                for roi in rois:
                    if need_lfp_kymo:
                        lw = extract_roi_waveforms_cleaned(d_lfp, roi, H, W, t0, fs)
                        vals_lfp_kymo.append(make_kymograph(lw).ravel())
                    if need_csd_kymo:
                        cw = extract_roi_waveforms_cleaned(d_csd, roi, H, W, t0, fs)
                        vals_csd_kymo.append(make_kymograph(cw).ravel())

    return {
        "lfp_snap": float(FIXED_LFP_SNAP_VLIM) if USE_FIXED_LFP_SNAP_VLIM else robust_vlim(np.concatenate(vals_lfp_snap)),
        "csd_snap": float(FIXED_CSD_SNAP_VLIM) if USE_FIXED_CSD_SNAP_VLIM else robust_vlim(np.concatenate(vals_csd_snap)),
        "lfp_kymo": float(FIXED_LFP_KYMO_VLIM) if USE_FIXED_LFP_KYMO_VLIM else robust_vlim(np.concatenate(vals_lfp_kymo)),
        "csd_kymo": float(FIXED_CSD_KYMO_VLIM) if USE_FIXED_CSD_KYMO_VLIM else robust_vlim(np.concatenate(vals_csd_kymo)),
    }

# ============================================================
# MAIN
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="ROI-based LFP/CSD snapshots, kymographs, correlation matrices, and entropy")
    p.add_argument("--data_dir", default=DATA_DIR, help="Folder of <slice_id>_<condition>_2ms.h5 files (default: %(default)s)")
    p.add_argument("--file_pattern", default=FILE_PATTERN, help="Glob pattern for H5 files (default: %(default)s)")
    p.add_argument("--out_dir", default=OUT_DIR, help="Output folder (default: %(default)s)")
    p.add_argument("--mask", default=MASK_PATH, help="64x64 contact mask .npy (default: %(default)s)")
    p.add_argument("--tif", default=TIF_PATH, help="Slice image rectified to the electrode field for the ROI picker: use data/masks/<slice>__rectified_grid.png from tissue_registration_mask.py (TIFF/PNG/JPG accepted) (default: %(default)s)")
    p.add_argument("--roi_csv", default=None,
                   help="Existing ROI definitions CSV (roi_index,roi_name,r0,r1,c0,c1; Python slice bounds). "
                        "When given, the interactive TIFF ROI picker is skipped (non-interactive run). "
                        "(default: %(default)s = run the picker)")
    p.add_argument("--genotype-label", default=GENOTYPE_LABEL,
                   help="Genotype/group label written into metric tables and barplots; must match one of your "
                        "data/processed_2ms/ group folder names (default: %(default)s)")
    return p


def main():
    global DATA_DIR, FILE_PATTERN, OUT_DIR, MASK_PATH, TIF_PATH, GENOTYPE_LABEL
    args = build_argparser().parse_args()
    DATA_DIR, FILE_PATTERN, OUT_DIR = args.data_dir, args.file_pattern, args.out_dir
    MASK_PATH, TIF_PATH = args.mask, args.tif
    GENOTYPE_LABEL = args.genotype_label

    os.makedirs(OUT_DIR, exist_ok=True)
    files = get_h5_files()

    with h5py.File(files[0], "r") as f:
        H, W, fs = load_meta(f)
        mask = load_mask_from_h5_or_npy(f, H, W)

    # PRESERVED ROI PICKER (skipped when --roi_csv is given)
    if args.roi_csv:
        rois, roi_names = load_rois_from_csv(args.roi_csv)
    else:
        rois, roi_names = pick_rois(H, W, mask)

    # Save ROI definitions
    with open(os.path.join(OUT_DIR, "roi_definitions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_index", "roi_name", "r0", "r1", "c0", "c1"])
        for i, (roi, name) in enumerate(zip(rois, roi_names)):
            w.writerow([i, name, roi[0].start, roi[0].stop, roi[1].start, roi[1].stop])

    scales = compute_global_scales(files, rois, mask)
    print("[INFO] Global/editable vlims:", scales)

    summary_rows = []
    correlation_feature_records = []
    raw_correlation_feature_records = []
    lfp_metric_rows_all = []
    csd_entropy_rows_all = []

    for p in files:
        drug = get_drug_name(p)
        print("\n============================================================")
        print("Processing:", drug)
        drug_dir = os.path.join(OUT_DIR, drug)
        os.makedirs(drug_dir, exist_ok=True)

        with h5py.File(p, "r") as f:
            H, W, fs = load_meta(f)
            mask2d = load_mask_from_h5_or_npy(f, H, W)
            mask_idx = np.flatnonzero(mask2d.reshape(-1))

            d_lfp = f["/lfp/ts_2ms_full"]
            d_csd = f["/csd/ts_2ms_full"]

            # t0 from LFP, matching your ROI pipeline. Change to d_csd if desired.
            t0 = find_t0_chunked(d_lfp, chunk_size=CHUNK_SIZE, mask_flat_idx=mask_idx)
            print(f"[INFO] t0={t0}")

            lfp_snaps = extract_snapshots_cleaned_chunked(d_lfp, H, W, fs, t0)
            csd_snaps = extract_snapshots_cleaned_chunked(d_csd, H, W, fs, t0)

            np.save(os.path.join(drug_dir, "lfp_snapshots.npy"), lfp_snaps)
            np.save(os.path.join(drug_dir, "csd_snapshots.npy"), csd_snaps)

            for roi_i, (roi, roi_name) in enumerate(zip(rois, roi_names)):
                print(f"  [ROI] {roi_i}: {roi_name}")
                roi_dir = os.path.join(drug_dir, f"ROI_{roi_i:02d}_{roi_name}")
                snap_dir = os.path.join(roi_dir, "individual_snapshots")
                os.makedirs(snap_dir, exist_ok=True)

                # Save individual full-slice 2D snapshots for each time point
                for snap_i, ms in enumerate(SNAP_MS):
                    save_individual_snapshot(
                        lfp_snaps[snap_i], H, W, mask2d, roi, roi_name, ms,
                        scales["lfp_snap"], LFP_CMAP,
                        title=f"{drug} {roi_name} LFP",
                        out_base=os.path.join(snap_dir, f"LFP_{roi_name}_{ms:04d}ms"),
                    )
                    save_individual_snapshot(
                        csd_snaps[snap_i], H, W, mask2d, roi, roi_name, ms,
                        scales["csd_snap"], CSD_CMAP,
                        title=f"{drug} {roi_name} CSD",
                        out_base=os.path.join(snap_dir, f"CSD_{roi_name}_{ms:04d}ms"),
                    )

                # Save snapshot grids per ROI
                save_snapshot_grid_for_roi(
                    lfp_snaps, H, W, mask2d, roi, roi_name,
                    scales["lfp_snap"], LFP_CMAP,
                    title=f"{drug} | {roi_name} | LFP snapshots",
                    out_base=os.path.join(roi_dir, f"{drug}_{roi_name}_LFP_snapshot_grid"),
                )
                save_snapshot_grid_for_roi(
                    csd_snaps, H, W, mask2d, roi, roi_name,
                    scales["csd_snap"], CSD_CMAP,
                    title=f"{drug} | {roi_name} | CSD snapshots",
                    out_base=os.path.join(roi_dir, f"{drug}_{roi_name}_CSD_snapshot_grid"),
                )

                # Extract ROI traces + kymographs
                # OLD cleaned extraction is preserved for snapshots/kymographs and the original correlation folder.
                lfp_wave = extract_roi_waveforms_cleaned(d_lfp, roi, H, W, t0, fs)
                csd_wave = extract_roi_waveforms_cleaned(d_csd, roi, H, W, t0, fs)
                lfp_kymo = make_kymograph(lfp_wave)
                csd_kymo = make_kymograph(csd_wave)

                # NEW unsmoothed baseline-corrected copy for raw trace display, CSV, metrics, and raw correlation.
                roi_time_ms = get_roi_time_axis_ms(fs)
                lfp_wave_raw = extract_roi_waveforms_raw_unsmoothed(d_lfp, roi, H, W, t0, fs)
                csd_wave_raw = extract_roi_waveforms_raw_unsmoothed(d_csd, roi, H, W, t0, fs)

                np.save(os.path.join(roi_dir, "lfp_raw_traces_cleaned.npy"), lfp_wave)
                np.save(os.path.join(roi_dir, "csd_raw_traces_cleaned.npy"), csd_wave)
                np.save(os.path.join(roi_dir, "lfp_raw_traces_unsmoothed_baseline.npy"), lfp_wave_raw)
                np.save(os.path.join(roi_dir, "csd_raw_traces_unsmoothed_baseline.npy"), csd_wave_raw)
                np.save(os.path.join(roi_dir, "roi_time_axis_ms.npy"), roi_time_ms)
                np.save(os.path.join(roi_dir, "lfp_kymo.npy"), lfp_kymo)
                np.save(os.path.join(roi_dir, "csd_kymo.npy"), csd_kymo)

                if SAVE_WIDE_TRACE_CSV:
                    save_waveform_csv_wide(lfp_wave_raw, roi_time_ms, os.path.join(roi_dir, "lfp_raw_traces_unsmoothed_baseline.csv"), prefix="lfp_ch")
                    save_waveform_csv_wide(csd_wave_raw, roi_time_ms, os.path.join(roi_dir, "csd_raw_traces_unsmoothed_baseline.csv"), prefix="csd_ch")
                    # Also save the cleaned arrays as CSV so old processed values remain accessible.
                    save_waveform_csv_wide(lfp_wave, roi_time_ms, os.path.join(roi_dir, "lfp_raw_traces_cleaned.csv"), prefix="lfp_clean_ch")
                    save_waveform_csv_wide(csd_wave, roi_time_ms, os.path.join(roi_dir, "csd_raw_traces_cleaned.csv"), prefix="csd_clean_ch")

                if SAVE_LONG_TRACE_CSV:
                    file_name = os.path.basename(p)
                    save_waveform_csv_long(lfp_wave_raw, roi_time_ms, os.path.join(roi_dir, "lfp_raw_traces_unsmoothed_baseline_LONG.csv"), "LFP", drug, file_name, roi_i, roi_name)
                    save_waveform_csv_long(csd_wave_raw, roi_time_ms, os.path.join(roi_dir, "csd_raw_traces_unsmoothed_baseline_LONG.csv"), "CSD", drug, file_name, roi_i, roi_name)

                if SAVE_LFP_METRIC_CSV:
                    file_name = os.path.basename(p)
                    lfp_metric_rows = make_lfp_metric_rows(lfp_wave_raw, roi_time_ms, drug, file_name, t0, roi_i, roi_name)
                    lfp_metric_rows_all.extend(lfp_metric_rows)
                    save_rows_csv_dict(lfp_metric_rows, os.path.join(roi_dir, "lfp_amplitude_energy_slope_metrics.csv"))

                if SAVE_CSD_TIME_ENTROPY:
                    entropy_root = os.path.join(OUT_DIR, CSD_ENTROPY_FOLDER)
                    entropy_rows = save_csd_time_entropy_outputs(
                        csd_snaps, H, W, roi, drug, os.path.basename(p), t0, roi_i, roi_name, entropy_root
                    )
                    csd_entropy_rows_all.extend(entropy_rows)

                # ------------------------------------------------------------
                # ADDED: ROI correlation matrices from the already-saved raw traces
                # ------------------------------------------------------------
                lfp_corr_stats = {"mean_r": np.nan, "median_r": np.nan, "std_r": np.nan, "min_r": np.nan, "max_r": np.nan}
                csd_corr_stats = {"mean_r": np.nan, "median_r": np.nan, "std_r": np.nan, "min_r": np.nan, "max_r": np.nan}
                lfp_csd_stats = {"mean_r": np.nan, "median_r": np.nan, "std_r": np.nan, "min_r": np.nan, "max_r": np.nan}

                if DO_CORRELATION_ANALYSIS:
                    corr_dir = os.path.join(roi_dir, "correlation")
                    os.makedirs(corr_dir, exist_ok=True)

                    lfp_corr = safe_corrcoef(lfp_wave)
                    csd_corr = safe_corrcoef(csd_wave)
                    lfp_csd_corr = corr_between_matrices(lfp_wave, csd_wave)

                    if SAVE_ROI_CORR_MATRICES:
                        np.save(os.path.join(corr_dir, "lfp_roi_corr.npy"), lfp_corr)
                        np.save(os.path.join(corr_dir, "csd_roi_corr.npy"), csd_corr)
                        np.save(os.path.join(corr_dir, "lfp_csd_cross_corr.npy"), lfp_csd_corr)

                    if SAVE_ROI_CORR_FIGURES:
                        save_corr_matrix(
                            lfp_corr,
                            f"{drug} | {roi_name} | LFP ROI correlation",
                            os.path.join(corr_dir, f"{drug}_{roi_name}_LFP_corr"),
                            xlabel="LFP ROI electrode",
                            ylabel="LFP ROI electrode",
                        )

                        save_corr_matrix(
                            csd_corr,
                            f"{drug} | {roi_name} | CSD ROI correlation",
                            os.path.join(corr_dir, f"{drug}_{roi_name}_CSD_corr"),
                            xlabel="CSD ROI electrode",
                            ylabel="CSD ROI electrode",
                        )

                        save_corr_matrix(
                            lfp_csd_corr,
                            f"{drug} | {roi_name} | LFP-CSD cross-correlation",
                            os.path.join(corr_dir, f"{drug}_{roi_name}_LFP_CSD_cross_corr"),
                            xlabel="CSD ROI electrode",
                            ylabel="LFP ROI electrode",
                        )

                    lfp_corr_stats = corr_summary_stats(lfp_corr)
                    csd_corr_stats = corr_summary_stats(csd_corr)
                    lfp_csd_stats = corr_summary_stats(lfp_csd_corr)

                    feature = make_roi_correlation_feature_vector(lfp_corr, csd_corr, lfp_csd_corr)
                    if feature.size > 0:
                        correlation_feature_records.append({
                            "label": f"{drug} | ROI{roi_i:02d}_{roi_name} | {os.path.basename(p)}",
                            "drug": drug,
                            "file": os.path.basename(p),
                            "roi_index": roi_i,
                            "roi_name": roi_name,
                            "feature": feature,
                        })

                    with open(os.path.join(corr_dir, "correlation_summary.csv"), "w", newline="") as cf:
                        cw = csv.writer(cf)
                        cw.writerow(["matrix", "mean_r", "median_r", "std_r", "min_r", "max_r"])
                        cw.writerow(["lfp_roi_corr", lfp_corr_stats["mean_r"], lfp_corr_stats["median_r"], lfp_corr_stats["std_r"], lfp_corr_stats["min_r"], lfp_corr_stats["max_r"]])
                        cw.writerow(["csd_roi_corr", csd_corr_stats["mean_r"], csd_corr_stats["median_r"], csd_corr_stats["std_r"], csd_corr_stats["min_r"], csd_corr_stats["max_r"]])
                        cw.writerow(["lfp_csd_cross_corr", lfp_csd_stats["mean_r"], lfp_csd_stats["median_r"], lfp_csd_stats["std_r"], lfp_csd_stats["min_r"], lfp_csd_stats["max_r"]])

                # ------------------------------------------------------------
                # NEW: raw/unsmoothed correlation matrices saved separately
                # ------------------------------------------------------------
                lfp_raw_corr_stats = {"mean_r": np.nan, "median_r": np.nan, "std_r": np.nan, "min_r": np.nan, "max_r": np.nan}
                csd_raw_corr_stats = {"mean_r": np.nan, "median_r": np.nan, "std_r": np.nan, "min_r": np.nan, "max_r": np.nan}
                lfp_csd_raw_stats = {"mean_r": np.nan, "median_r": np.nan, "std_r": np.nan, "min_r": np.nan, "max_r": np.nan}

                if DO_RAW_CORRELATION_ANALYSIS:
                    raw_corr_dir = os.path.join(roi_dir, RAW_CORR_FOLDER_NAME)
                    os.makedirs(raw_corr_dir, exist_ok=True)

                    lfp_corr_raw = safe_corrcoef(lfp_wave_raw)
                    csd_corr_raw = safe_corrcoef(csd_wave_raw)
                    lfp_csd_corr_raw = corr_between_matrices(lfp_wave_raw, csd_wave_raw)

                    np.save(os.path.join(raw_corr_dir, "lfp_roi_corr_RAW_unsmoothed.npy"), lfp_corr_raw)
                    np.save(os.path.join(raw_corr_dir, "csd_roi_corr_RAW_unsmoothed.npy"), csd_corr_raw)
                    np.save(os.path.join(raw_corr_dir, "lfp_csd_cross_corr_RAW_unsmoothed.npy"), lfp_csd_corr_raw)

                    if SAVE_ROI_CORR_FIGURES:
                        save_corr_matrix(
                            lfp_corr_raw,
                            f"{drug} | {roi_name} | RAW LFP ROI correlation",
                            os.path.join(raw_corr_dir, f"{drug}_{roi_name}_RAW_LFP_corr"),
                            xlabel="RAW LFP ROI electrode",
                            ylabel="RAW LFP ROI electrode",
                        )
                        save_corr_matrix(
                            csd_corr_raw,
                            f"{drug} | {roi_name} | RAW CSD ROI correlation",
                            os.path.join(raw_corr_dir, f"{drug}_{roi_name}_RAW_CSD_corr"),
                            xlabel="RAW CSD ROI electrode",
                            ylabel="RAW CSD ROI electrode",
                        )
                        save_corr_matrix(
                            lfp_csd_corr_raw,
                            f"{drug} | {roi_name} | RAW LFP-CSD cross-correlation",
                            os.path.join(raw_corr_dir, f"{drug}_{roi_name}_RAW_LFP_CSD_cross_corr"),
                            xlabel="RAW CSD ROI electrode",
                            ylabel="RAW LFP ROI electrode",
                        )

                    lfp_raw_corr_stats = corr_summary_stats(lfp_corr_raw)
                    csd_raw_corr_stats = corr_summary_stats(csd_corr_raw)
                    lfp_csd_raw_stats = corr_summary_stats(lfp_csd_corr_raw)

                    raw_feature = make_roi_correlation_feature_vector(lfp_corr_raw, csd_corr_raw, lfp_csd_corr_raw)
                    if raw_feature.size > 0:
                        raw_correlation_feature_records.append({
                            "label": f"{drug} | ROI{roi_i:02d}_{roi_name} | RAW | {os.path.basename(p)}",
                            "drug": drug,
                            "file": os.path.basename(p),
                            "roi_index": roi_i,
                            "roi_name": roi_name,
                            "feature": raw_feature,
                        })

                    with open(os.path.join(raw_corr_dir, "correlation_summary_RAW_unsmoothed.csv"), "w", newline="") as cf:
                        cw = csv.writer(cf)
                        cw.writerow(["matrix", "mean_r", "median_r", "std_r", "min_r", "max_r"])
                        cw.writerow(["lfp_roi_corr_RAW", lfp_raw_corr_stats["mean_r"], lfp_raw_corr_stats["median_r"], lfp_raw_corr_stats["std_r"], lfp_raw_corr_stats["min_r"], lfp_raw_corr_stats["max_r"]])
                        cw.writerow(["csd_roi_corr_RAW", csd_raw_corr_stats["mean_r"], csd_raw_corr_stats["median_r"], csd_raw_corr_stats["std_r"], csd_raw_corr_stats["min_r"], csd_raw_corr_stats["max_r"]])
                        cw.writerow(["lfp_csd_cross_corr_RAW", lfp_csd_raw_stats["mean_r"], lfp_csd_raw_stats["median_r"], lfp_csd_raw_stats["std_r"], lfp_csd_raw_stats["min_r"], lfp_csd_raw_stats["max_r"]])

                # Save individual raw traces and kymographs
                # Main raw trace figures now use the unsmoothed baseline-corrected copy.
                save_raw_traces_only(lfp_wave_raw, f"{drug} | {roi_name} | LFP raw traces", os.path.join(roi_dir, f"{drug}_{roi_name}_LFP_raw_traces"))
                save_raw_traces_only(csd_wave_raw, f"{drug} | {roi_name} | CSD raw traces", os.path.join(roi_dir, f"{drug}_{roi_name}_CSD_raw_traces"))
                # Preserve cleaned trace figures too, with explicit suffix.
                save_raw_traces_only(lfp_wave, f"{drug} | {roi_name} | LFP cleaned traces", os.path.join(roi_dir, f"{drug}_{roi_name}_LFP_CLEANED_traces"))
                save_raw_traces_only(csd_wave, f"{drug} | {roi_name} | CSD cleaned traces", os.path.join(roi_dir, f"{drug}_{roi_name}_CSD_CLEANED_traces"))

                save_kymo_only(lfp_kymo, scales["lfp_kymo"], LFP_CMAP, f"{drug} | {roi_name} | LFP kymograph", "LFP", os.path.join(roi_dir, f"{drug}_{roi_name}_LFP_kymograph"))
                save_kymo_only(csd_kymo, scales["csd_kymo"], CSD_CMAP, f"{drug} | {roi_name} | CSD kymograph", "CSD", os.path.join(roi_dir, f"{drug}_{roi_name}_CSD_kymograph"))

                # Save combined panel like your example. Left panels are raw/unsmoothed; kymographs remain cleaned.
                save_raw_kymo_panel(
                    lfp_wave_raw, csd_wave_raw, lfp_kymo, csd_kymo,
                    scales["lfp_kymo"], scales["csd_kymo"],
                    drug, roi_name,
                    os.path.join(roi_dir, f"{drug}_{roi_name}_RAW_LFP_CSD_KYMO_panel"),
                )

                summary_rows.append([
                    drug,
                    os.path.basename(p),
                    int(t0),
                    roi_i,
                    roi_name,
                    roi[0].start,
                    roi[0].stop,
                    roi[1].start,
                    roi[1].stop,
                    scales["lfp_snap"],
                    scales["csd_snap"],
                    scales["lfp_kymo"],
                    scales["csd_kymo"],
                    float(np.nanmean(lfp_wave)),
                    float(np.nanstd(lfp_wave)),
                    float(np.sqrt(np.nanmean(lfp_wave ** 2))),
                    float(np.nanmean(csd_wave)),
                    float(np.nanstd(csd_wave)),
                    float(np.sqrt(np.nanmean(csd_wave ** 2))),

                    lfp_corr_stats["mean_r"],
                    lfp_corr_stats["median_r"],
                    lfp_corr_stats["std_r"],
                    lfp_corr_stats["min_r"],
                    lfp_corr_stats["max_r"],

                    csd_corr_stats["mean_r"],
                    csd_corr_stats["median_r"],
                    csd_corr_stats["std_r"],
                    csd_corr_stats["min_r"],
                    csd_corr_stats["max_r"],

                    lfp_csd_stats["mean_r"],
                    lfp_csd_stats["median_r"],
                    lfp_csd_stats["std_r"],
                    lfp_csd_stats["min_r"],
                    lfp_csd_stats["max_r"],

                    float(np.nanmean(lfp_wave_raw)),
                    float(np.nanstd(lfp_wave_raw)),
                    float(np.sqrt(np.nanmean(lfp_wave_raw ** 2))),
                    float(np.nanmean(csd_wave_raw)),
                    float(np.nanstd(csd_wave_raw)),
                    float(np.sqrt(np.nanmean(csd_wave_raw ** 2))),

                    lfp_raw_corr_stats["mean_r"],
                    lfp_raw_corr_stats["median_r"],
                    lfp_raw_corr_stats["std_r"],
                    lfp_raw_corr_stats["min_r"],
                    lfp_raw_corr_stats["max_r"],

                    csd_raw_corr_stats["mean_r"],
                    csd_raw_corr_stats["median_r"],
                    csd_raw_corr_stats["std_r"],
                    csd_raw_corr_stats["min_r"],
                    csd_raw_corr_stats["max_r"],

                    lfp_csd_raw_stats["mean_r"],
                    lfp_csd_raw_stats["median_r"],
                    lfp_csd_raw_stats["std_r"],
                    lfp_csd_raw_stats["min_r"],
                    lfp_csd_raw_stats["max_r"],
                ])

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "drug", "file", "t0", "roi_index", "roi_name",
            "r0", "r1", "c0", "c1",
            "global_lfp_snap_vlim", "global_csd_snap_vlim",
            "global_lfp_kymo_vlim", "global_csd_kymo_vlim",
            "lfp_mean", "lfp_std", "lfp_rms",
            "csd_mean", "csd_std", "csd_rms",

            "lfp_corr_mean_r", "lfp_corr_median_r", "lfp_corr_std_r", "lfp_corr_min_r", "lfp_corr_max_r",
            "csd_corr_mean_r", "csd_corr_median_r", "csd_corr_std_r", "csd_corr_min_r", "csd_corr_max_r",
            "lfp_csd_corr_mean_r", "lfp_csd_corr_median_r", "lfp_csd_corr_std_r", "lfp_csd_corr_min_r", "lfp_csd_corr_max_r",

            "lfp_raw_unsmoothed_mean", "lfp_raw_unsmoothed_std", "lfp_raw_unsmoothed_rms",
            "csd_raw_unsmoothed_mean", "csd_raw_unsmoothed_std", "csd_raw_unsmoothed_rms",

            "lfp_raw_corr_mean_r", "lfp_raw_corr_median_r", "lfp_raw_corr_std_r", "lfp_raw_corr_min_r", "lfp_raw_corr_max_r",
            "csd_raw_corr_mean_r", "csd_raw_corr_median_r", "csd_raw_corr_std_r", "csd_raw_corr_min_r", "csd_raw_corr_max_r",
            "lfp_csd_raw_corr_mean_r", "lfp_csd_raw_corr_median_r", "lfp_csd_raw_corr_std_r", "lfp_csd_raw_corr_min_r", "lfp_csd_raw_corr_max_r",
        ])
        w.writerows(summary_rows)

    if SAVE_LFP_METRIC_CSV and lfp_metric_rows_all:
        save_rows_csv_dict(lfp_metric_rows_all, os.path.join(OUT_DIR, "all_lfp_amplitude_energy_slope_metrics.csv"))
        if SAVE_LFP_METRIC_BARPLOTS:
            save_lfp_metric_barplots(lfp_metric_rows_all, os.path.join(OUT_DIR, "lfp_metric_barplots"))

    if SAVE_CSD_TIME_ENTROPY and csd_entropy_rows_all:
        entropy_root = os.path.join(OUT_DIR, CSD_ENTROPY_FOLDER)
        save_rows_csv_dict(csd_entropy_rows_all, os.path.join(entropy_root, "summary_all_roi_csd_entropy_bits.csv"))
        save_csd_entropy_summary_plots(csd_entropy_rows_all, os.path.join(entropy_root, "summary_barplots"))

    if DO_CORRELATION_ANALYSIS and SAVE_DRUG_ROI_SIMILARITY:
        save_drug_roi_similarity_outputs(correlation_feature_records, OUT_DIR)

    if DO_RAW_CORRELATION_ANALYSIS and SAVE_RAW_DRUG_ROI_SIMILARITY:
        save_drug_roi_similarity_outputs(
            raw_correlation_feature_records,
            OUT_DIR,
            subfolder="drug_roi_correlation_summary_RAW_unsmoothed",
            title_prefix="RAW unsmoothed ",
        )

    print("\nDONE")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
