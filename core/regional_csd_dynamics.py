# -*- coding: utf-8 -*-
"""
regional_csd_dynamics.py -- SPACE framework: regional CSD source/sink balance, activity, asymmetry
and temporal dynamics (lag-1 autocorrelation, sample entropy, mean crossing rate) for the largest
network event of each recording.

Manuscript Methods section: "Current source density" (paragraphs "For a set of electrodes R (an ROI
or a pooled region) ..." through the balance-index description), "Regional CSD activity and
asymmetry" and "Temporal dynamics of regional CSD".
Manuscript figure(s): balance-index time courses and time-averaged BI / source-sink AUC bar data,
regional CSD activity, RAI, lag-1 autocorrelation, sample entropy and MCR comparisons.

Inputs:
    data/processed_2ms/<group>/<name>.h5   (schema of preprocessing/brw_to_2ms_h5.py; stored fs)
    data/masks/<slice_id>__contact_mask_64x64.npy   (matched by slice prefix in the file name,
                                                     else /mask in the H5, else --mask override)
    data/roi_definitions/<slice_id>_roi_definitions.csv   (roi_index, roi_name, r0, r1, c0, c1;
                                                           Python slice bounds; else --roi_csv)
Outputs (under data/results/regional_csd_dynamics/):
    roi_metrics.csv       one row per recording x ROI (M1, M2, S1, CA1, CA3, DG)
    region_metrics.csv    one row per recording x pooled region (Ctx, Hippo): activity metrics,
                          BI metrics of the pooled electrode set, lag-1 autocorrelation, sample
                          entropy, mean crossing rate and the recording's RAI
    animal_level.csv      ROI values averaged within region per (group, slice_id, condition,
                          region) -- the inferential unit -- joined with the region-level dynamics
    timecourses/<group>__<file>.npz          BI, Source, Sink, A_R(t) time courses + time axes
    figures/<group>__<file>__bi_activity.(png|svg)   2 rows (Ctx, Hippo) x 2 cols (BI cond., A_R)
    figures/group_mean_bi__<condition>.(png|svg)     mean +/- s.e.m. over slices per group
    manifest.json         every parameter used

Algorithm (as in the Methods):
    1. Onset t0 = sample of maximal summed |LFP| across tissue-contact electrodes (the peak of the
       largest event; mask rows read in channel blocks, time streamed).
    2. Epoch -50..+500 ms around t0 (only ROI rows read). Each electrode trace is baseline-corrected
       by its median over -50..0 ms (limits inclusive); NaNs are linearly interpolated.
    3. Source_R(t) = mean_R max(CSD,0), Sink_R(t) = mean_R max(-CSD,0),
       BI_R(t) = (Source - Sink)/(Source + Sink + eps) in [-1, 1].
       BI time courses for display / time-averaged BI / AUCs: traces conditioned with a 5-sample
       median filter and a zero-phase 150 Hz low-pass (3rd-order Butterworth, filtfilt), window
       -50..250 ms; time-averaged BI = mean of BI_R(t) for t >= 0; SourceAUC, SinkAUC = trapezoidal
       integrals over 0..250 ms; ratio = SourceAUC/(SinkAUC + eps).
    4. Regional CSD activity A_R(t) = mean_R |CSD| (unfiltered); CSD peak = max A_R over 0..250 ms;
       total current I_R = trapezoid of A_R over 0..500 ms; RAI = (I_Ctx - I_Hippo)/(I_Ctx + I_Hippo).
    5. Pooled region, 0..500 ms, unfiltered: lag-1 autocorrelation of the linearly detrended BI
       series (statsmodels acf, numpy fallback); sample entropy of A_R(t) (m = 2, r = 0.2 s.d.,
       Chebyshev distance, self-matches excluded, nats); mean crossing rate of A_R(t) through its
       temporal mean divided by the duration (Hz).

Implementation notes:
    - "Largest event" = the single global peak of summed |LFP| over the mask electrodes of the whole
      recording (no 2 s restriction, unlike the whole-slice onset of extract_event_epochs.py). The
      search is restricted to samples where the full -50..+500 ms epoch fits inside the recording
      (as core/roi_phase_gradient_pgd.py does), so no edge padding is ever needed.
    - ROI electrodes outside the tissue-contact mask or on the outermost ring of the array are
      excluded from every CSD quantity (common.md); an ROI/region with no remaining electrode
      yields NaN metrics. A pooled region is the union of the electrodes of its ROIs (Ctx = M1, M2,
      S1; Hippo = CA1, CA3, DG), so Source/Sink/A_R of a region are means over the pooled electrode
      set (not means of ROI means).
    - Conditioning (median filter, then filtfilt) is applied to the baseline-corrected traces over
      the whole -50..+500 ms epoch and the result is cropped to -50..250 ms afterwards, so the
      filter transient does not sit at the window edges.
    - Integrals are trapezoidal over the time axis in ms (units a.u.*ms); windows are inclusive.
    - Sample entropy: N - m template vectors are used for both the m and m + 1 counts (Richman &
      Moorman), pairs i != j, r = 0.2 * population s.d. (ddof = 0); NaN if A or B is 0.
    - Mean crossing rate: a crossing is a sign change of (a_k - mean a) between consecutive samples;
      samples exactly on the mean are dropped before counting; T = (K - 1) * dt = 0.5 s.
    - lag-1 autocorrelation: numpy fallback is the standard biased ACF estimator used by
      statsmodels.tsa.stattools.acf (adjusted=False).
    - animal_level.csv averages ROI metrics within region per (group, slice_id, condition); with
      one recording per (slice, condition) this is the recording's ROI mean. Its roi_name column
      is "ROI_mean" and event_index is 0 (one event, the largest, per recording). Group-mean BI
      figures use these per-animal, within-region ROI-averaged conditioned BI traces (Methods:
      "ROI values were averaged within region for each animal, and group traces are averages
      across animals"); the pooled-region BI traces are also stored in the NPZ.
    - Recordings with different fs are interpolated onto the time axis of the first recording of a
      condition for the group-mean figure.
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
from scipy.signal import butter, filtfilt, medfilt, detrend
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.stattools import acf as _sm_acf
    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    _sm_acf = None
    HAVE_STATSMODELS = False
    print("[INFO] statsmodels not available -> numpy lag-1 autocorrelation fallback")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
IN_ROOT = REPO_ROOT / "data" / "processed_2ms"
MASK_DIR = REPO_ROOT / "data" / "masks"
ROI_DIR = REPO_ROOT / "data" / "roi_definitions"
OUT_DIR = REPO_ROOT / "data" / "results" / "regional_csd_dynamics"
EPOCH_PRE_MS, EPOCH_POST_MS = -50.0, 500.0      # epoch extent relative to t0
BASELINE_MS = (-50.0, 0.0)                      # baseline window (inclusive)
ANALYSIS_MS = (0.0, 250.0)                      # CSD peak, SourceAUC/SinkAUC, BI time courses
BI_DISPLAY_MS = (-50.0, 250.0)                  # BI time-course window
TOTAL_MS = (0.0, 500.0)                         # total current, lag-1, SampEn, MCR
MEDIAN_KERNEL = 5                               # conditioning: median filter (samples)
LOWPASS_HZ = 150.0                              # conditioning: zero-phase Butterworth low-pass
BUTTER_ORDER = 3
SAMPEN_M = 2
SAMPEN_R_FRAC = 0.2
EPS = 1e-12
REGIONS = {"Ctx": ["M1", "M2", "S1"], "Hippo": ["CA1", "CA3", "DG"]}
ROI_ORDER = ["M1", "M2", "S1", "CA1", "CA3", "DG"]
TIME_CHUNK = 5000          # samples per streamed chunk for the t0 search
CH_BLOCK = 512             # channels per block read
SAMPEN_BLOCK = 256         # template rows per block in the sample-entropy pair count
FIG_DPI = 300
# --- END EDIT ---

LFP_KEY, CSD_KEY = "/lfp/ts_2ms_full", "/csd/ts_2ms_full"


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


def load_roi_definitions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = ["roi_name", "r0", "r1", "c0", "c1"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    if "roi_index" not in df.columns:
        df["roi_index"] = np.arange(len(df))
    df["roi_name"] = df["roi_name"].astype(str).str.strip().replace({"SSP": "S1", "ssp": "S1"})
    df["_ord"] = [ROI_ORDER.index(n) if n in ROI_ORDER else 99 for n in df["roi_name"]]
    return df.sort_values(["_ord", "roi_index"]).drop(columns="_ord").reset_index(drop=True)


def region_of(roi_name: str) -> str:
    for reg, names in REGIONS.items():
        if roi_name in names:
            return reg
    return "other"


def roi_electrodes(box: Tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    """Flat indices (row-major) of the ROI electrodes that are in the mask and not on the outer ring."""
    H, W = mask.shape
    r0, r1, c0, c1 = box
    idx = []
    for r in range(max(r0, 1), min(r1, H - 1)):
        for c in range(max(c0, 1), min(c1, W - 1)):
            if mask[r, c]:
                idx.append(r * W + c)
    return np.asarray(idx, dtype=int)


def find_t0_chunked(dset: h5py.Dataset, idx: np.ndarray, pre: int, post: int,
                    time_chunk: int = TIME_CHUNK, ch_block: int = CH_BLOCK) -> Tuple[int, np.ndarray]:
    """argmax_t sum |LFP| over electrodes `idx` (streamed), restricted so the -pre..+post epoch fits."""
    n_ch, T = dset.shape
    score = np.zeros(T, dtype=np.float64)
    for t0 in range(0, T, time_chunk):
        t1 = min(t0 + time_chunk, T)
        for c0 in range(0, n_ch, ch_block):
            c1 = min(c0 + ch_block, n_ch)
            sel = idx[(idx >= c0) & (idx < c1)] - c0
            if sel.size == 0:
                continue
            X = dset[c0:c1, t0:t1]
            score[t0:t1] += np.abs(X[sel]).sum(axis=0)
    lo, hi = pre, T - post
    if hi <= lo:
        warnings.warn("recording shorter than the epoch: t0 search unguarded")
        return int(np.argmax(score)), score
    guarded = np.full(T, -np.inf); guarded[lo:hi] = score[lo:hi]
    return int(np.argmax(guarded)), score


def read_rows(dset: h5py.Dataset, idx: np.ndarray, s0: int, s1: int) -> np.ndarray:
    """dset[idx, s0:s1] for sorted idx, read as one contiguous row block then sub-selected."""
    if idx.size == 0:
        return np.empty((0, s1 - s0), np.float32)
    a, b = int(idx.min()), int(idx.max()) + 1
    return dset[a:b, s0:s1][idx - a].astype(np.float32)


# =============================================================================
# SIGNAL
# =============================================================================

def fill_nan_linear(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=True)
    t = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        bad = ~np.isfinite(X[i])
        if bad.any():
            good = ~bad
            X[i, bad] = np.interp(t[bad], t[good], X[i, good]) if good.sum() >= 2 else (X[i, good][0] if good.any() else 0.0)
    return X


def baseline_correct(X: np.ndarray, t_ms: np.ndarray, win=BASELINE_MS) -> np.ndarray:
    X = fill_nan_linear(X)
    bm = (t_ms >= win[0]) & (t_ms <= win[1])
    if not bm.any():
        bm = t_ms <= 0
    return X - np.median(X[:, bm], axis=1, keepdims=True)


def condition_traces(X: np.ndarray, fs: float) -> np.ndarray:
    """5-sample median filter + zero-phase 3rd-order Butterworth 150 Hz low-pass (per electrode)."""
    Y = np.empty_like(X)
    wn = LOWPASS_HZ / (0.5 * fs)
    b, a = butter(BUTTER_ORDER, wn, btype="low") if wn < 1.0 else (None, None)
    for i in range(X.shape[0]):
        y = medfilt(X[i], kernel_size=MEDIAN_KERNEL) if X.shape[1] >= MEDIAN_KERNEL else X[i]
        Y[i] = filtfilt(b, a, y) if b is not None and y.size > 3 * max(len(a), len(b)) else y
    return Y


def source_sink_bi(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean rectified positive / negative parts over electrodes and the balance index."""
    if C.shape[0] == 0:
        n = np.full(C.shape[1], np.nan)
        return n, n.copy(), n.copy()
    src = np.maximum(C, 0.0).mean(axis=0)
    snk = np.maximum(-C, 0.0).mean(axis=0)
    bi = (src - snk) / (src + snk + EPS)
    return src, snk, bi


def activity(C: np.ndarray) -> np.ndarray:
    return np.abs(C).mean(axis=0) if C.shape[0] else np.full(C.shape[1], np.nan)


_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy >= 2.0 renamed trapz


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(_np_trapz(y, x)) if y.size >= 2 and np.all(np.isfinite(y)) else float("nan")


def lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size < 3 or not np.all(np.isfinite(x)):
        return float("nan")
    d = detrend(x, type="linear")
    if np.allclose(d, 0):
        return float("nan")
    if HAVE_STATSMODELS:
        return float(_sm_acf(d, nlags=1, fft=True)[1])
    d = d - d.mean()
    return float(np.dot(d[:-1], d[1:]) / np.dot(d, d))


def sample_entropy(x: np.ndarray, m: int = SAMPEN_M, r_frac: float = SAMPEN_R_FRAC, block: int = SAMPEN_BLOCK) -> float:
    """SampEn(m, r) in nats, Chebyshev distance, self-matches excluded, N-m templates for m and m+1."""
    x = np.asarray(x, np.float64)
    N = x.size
    if N <= m + 1 or not np.all(np.isfinite(x)):
        return float("nan")
    r = r_frac * np.std(x)
    if r <= 0:
        return float("nan")
    n_t = N - m
    Xm = sliding_window_view(x, m)[:n_t].astype(np.float32)        # (n_t, m)
    xm1 = x[m:m + n_t].astype(np.float32)                          # (m+1)-th element of each template
    A = B = 0
    for i0 in range(0, n_t, block):
        i1 = min(i0 + block, n_t)
        dm = np.max(np.abs(Xm[i0:i1, None, :] - Xm[None, :, :]), axis=2)       # (b, n_t)
        dm1 = np.maximum(dm, np.abs(xm1[i0:i1, None] - xm1[None, :]))
        B += int((dm <= r).sum()) - (i1 - i0)      # remove self-matches (distance 0 on the diagonal)
        A += int((dm1 <= r).sum()) - (i1 - i0)
    if A <= 0 or B <= 0:
        return float("nan")
    return float(-np.log(A / B))


def mean_crossing_rate(a: np.ndarray, dt_s: float) -> float:
    a = np.asarray(a, float)
    if a.size < 2 or not np.all(np.isfinite(a)):
        return float("nan")
    d = a - a.mean()
    s = np.sign(d[d != 0.0])
    n_cross = int((s[1:] * s[:-1] < 0).sum()) if s.size > 1 else 0
    T = (a.size - 1) * dt_s
    return n_cross / T if T > 0 else float("nan")


# =============================================================================
# PER-SET METRICS
# =============================================================================

def set_metrics(C: np.ndarray, Cc: np.ndarray, t_ms: np.ndarray, fs: float) -> Tuple[dict, dict]:
    """Metrics + time courses for one electrode set (ROI or pooled region).

    C  : baseline-corrected unfiltered CSD [n_el, T];  Cc : conditioned copy.
    """
    an = (t_ms >= ANALYSIS_MS[0]) & (t_ms <= ANALYSIS_MS[1])
    tot = (t_ms >= TOTAL_MS[0]) & (t_ms <= TOTAL_MS[1])
    disp = (t_ms >= BI_DISPLAY_MS[0]) & (t_ms <= BI_DISPLAY_MS[1])
    src_c, snk_c, bi_c = source_sink_bi(Cc)
    src_r, snk_r, bi_r = source_sink_bi(C)
    A = activity(C)
    post = disp & (t_ms >= 0)
    m = dict(
        n_electrodes=int(C.shape[0]),
        csd_peak=float(np.max(A[an])) if C.shape[0] else np.nan,
        total_current_0_500=trapz(A[tot], t_ms[tot]),
        bi_time_avg=float(np.mean(bi_c[post])) if C.shape[0] else np.nan,
        source_auc=trapz(src_c[an], t_ms[an]),
        sink_auc=trapz(snk_c[an], t_ms[an]),
    )
    m["source_sink_ratio"] = m["source_auc"] / (m["sink_auc"] + EPS) if np.isfinite(m["source_auc"]) else np.nan
    tc = dict(bi_cond=bi_c[disp], source_cond=src_c[disp], sink_cond=snk_c[disp],
              bi_raw=bi_r, source_raw=src_r, sink_raw=snk_r, activity=A)
    return m, tc


def dynamics_metrics(tc: dict, t_ms: np.ndarray, fs: float) -> dict:
    tot = (t_ms >= TOTAL_MS[0]) & (t_ms <= TOTAL_MS[1])
    return dict(lag1_autocorr=lag1_autocorr(tc["bi_raw"][tot]),
                sample_entropy=sample_entropy(tc["activity"][tot]),
                mean_crossing_rate_hz=mean_crossing_rate(tc["activity"][tot], 1.0 / fs))


# =============================================================================
# FIGURES
# =============================================================================

def recording_figure(tcs: Dict[str, dict], t_ms: np.ndarray, t_bi: np.ndarray, region_rois: Dict[str, List[str]],
                     title: str, out_base: Path):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2), sharex="col")
    for i, reg in enumerate(["Ctx", "Hippo"]):
        ax_bi, ax_a = axes[i]
        for roi in region_rois.get(reg, []):
            if roi in tcs:
                ax_bi.plot(t_bi, tcs[roi]["bi_cond"], lw=0.8, alpha=0.55, label=roi)
                ax_a.plot(t_ms, tcs[roi]["activity"], lw=0.8, alpha=0.55, label=roi)
        if reg in tcs:
            ax_bi.plot(t_bi, tcs[reg]["bi_cond"], color="k", lw=1.6, label=f"{reg} pooled")
            ax_a.plot(t_ms, tcs[reg]["activity"], color="k", lw=1.6, label=f"{reg} pooled")
        for ax in (ax_bi, ax_a):
            ax.axvline(0, color="0.5", lw=0.8, ls="--")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax_bi.axhline(0, color="0.8", lw=0.6)
        ax_bi.set_ylim(-1.05, 1.05); ax_bi.set_ylabel(f"{reg}\nBI (conditioned)")
        ax_a.set_ylabel("A_R(t)  mean |CSD| (a.u.)")
        ax_bi.legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 0].set_xlabel("Time from t0 (ms)"); axes[1, 1].set_xlabel("Time from t0 (ms)")
    axes[0, 0].set_title("Source-sink balance index"); axes[0, 1].set_title("Regional CSD activity")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_base)


def group_mean_figure(traces: List[dict], cond: str, out_base: Path):
    """traces: dicts with group, slice_id, region, t (common axis), bi (per-animal ROI-averaged BI)."""
    groups = sorted({d["group"] for d in traces})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, reg in zip(axes, ["Ctx", "Hippo"]):
        for g in groups:
            sel = [d for d in traces if d["group"] == g and d["region"] == reg]
            if not sel:
                continue
            t = sel[0]["t"]
            Y = np.vstack([d["bi"] for d in sel])
            mu = np.nanmean(Y, axis=0)
            sem = np.nanstd(Y, axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mu)
            ax.plot(t, mu, lw=1.4, label=f"{g} (n={Y.shape[0]})")
            ax.fill_between(t, mu - sem, mu + sem, alpha=0.25, lw=0)
        ax.axvline(0, color="0.5", lw=0.8, ls="--"); ax.axhline(0, color="0.8", lw=0.6)
        ax.set_ylim(-1.05, 1.05); ax.set_title(reg); ax.set_xlabel("Time from t0 (ms)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Balance index (mean +/- s.e.m. over slices)")
    fig.suptitle(f"Group-mean BI traces -- {cond}", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_base)


# =============================================================================
# PER RECORDING
# =============================================================================

def process_recording(h5_path: Path, group: str, args, mask_dir: Path, roi_dir: Path, out_dir: Path):
    roi_rows, reg_rows, bi_traces = [], [], []
    with h5py.File(h5_path, "r") as f:
        fs, H, W = read_meta(f)
        lfp, csd = f[LFP_KEY], f[CSD_KEY]
        mask, slice_id, mask_src = load_mask(f, h5_path, mask_dir, args.mask, H, W)
        roi_csv = Path(args.roi_csv) if args.roi_csv else roi_dir / f"{slice_id}_roi_definitions.csv"
        if not roi_csv.exists():
            print(f"[SKIP] {h5_path.name}: ROI CSV not found: {roi_csv}")
            return roi_rows, reg_rows, bi_traces
        rois = load_roi_definitions(roi_csv)
        cond = parse_condition(h5_path.name)
        dt = 1000.0 / fs
        pre, post = int(round(-args.epoch_pre_ms / dt)), int(round(args.epoch_post_ms / dt))
        t_ms = (np.arange(pre + post + 1) - pre) * dt
        idx_mask = np.flatnonzero(mask.ravel())
        print(f"[REC] {group}/{h5_path.name}: fs={fs:g} Hz, T={lfp.shape[1]}, slice={slice_id}, cond={cond}, "
              f"mask={mask.sum()} el ({mask_src}), {len(rois)} ROIs")
        t0, _ = find_t0_chunked(lfp, idx_mask, pre, post)
        s0, s1 = t0 - pre, t0 + post + 1
        t0_ms = t0 * 1000.0 / fs
        print(f"       t0 = {t0} samples ({t0_ms:.1f} ms); epoch {t_ms[0]:g}..{t_ms[-1]:g} ms ({t_ms.size} samples)")

        # ROI traces (unfiltered + conditioned), pooled regions = union of ROI electrodes
        C_roi: Dict[str, np.ndarray] = {}
        for r in rois.itertuples():
            idx = roi_electrodes((int(r.r0), int(r.r1), int(r.c0), int(r.c1)), mask)
            X = read_rows(csd, idx, max(s0, 0), min(s1, csd.shape[1]))
            if X.shape[1] != t_ms.size:   # unguarded short recording: pad with NaN, then interpolate
                padl, padr = max(0, -s0), max(0, s1 - csd.shape[1])
                X = np.pad(X, ((0, 0), (padl, padr)), constant_values=np.nan)
            C_roi[r.roi_name] = baseline_correct(X, t_ms)
        region_rois = {reg: [n for n in names if n in C_roi] for reg, names in REGIONS.items()}
        sets = dict(C_roi)
        for reg, names in region_rois.items():
            sets[reg] = np.vstack([C_roi[n] for n in names]) if names else np.empty((0, t_ms.size))

        base = dict(group=group, slice_id=slice_id, condition=cond, file=h5_path.name,
                    t0_sample=int(t0), t0_ms=t0_ms, fs=fs)
        tcs: Dict[str, dict] = {}
        for name, C in sets.items():
            Cc = condition_traces(C, fs) if C.shape[0] else C
            m, tc = set_metrics(C, Cc, t_ms, fs)
            tcs[name] = tc
            if name in REGIONS:
                m.update(dynamics_metrics(tc, t_ms, fs))
                reg_rows.append(dict(base, region=name, rois=",".join(region_rois[name]), **m))
            else:
                roi_rows.append(dict(base, roi_name=name, region=region_of(name), event_index=0, **m))
        I = {r["region"]: r["total_current_0_500"] for r in reg_rows}
        rai = (I["Ctx"] - I["Hippo"]) / (I["Ctx"] + I["Hippo"]) if np.isfinite(I.get("Ctx", np.nan)) and np.isfinite(I.get("Hippo", np.nan)) and (I["Ctx"] + I["Hippo"]) > 0 else np.nan
        for r in reg_rows:
            r["rai"] = rai
        print(f"       RAI = {rai:.3f}; " + "; ".join(f"{r['region']}: BIavg={r['bi_time_avg']:.3f} lag1={r['lag1_autocorr']:.4f} "
              f"SampEn={r['sample_entropy']:.3f} MCR={r['mean_crossing_rate_hz']:.1f} Hz" for r in reg_rows))

        # time-course NPZ
        disp = (t_ms >= BI_DISPLAY_MS[0]) & (t_ms <= BI_DISPLAY_MS[1])
        names = list(tcs.keys())
        npz = dict(time_ms=t_ms, time_ms_bi=t_ms[disp], set_names=np.array(names),
                   is_region=np.array([n in REGIONS for n in names]), t0_sample=t0, t0_ms=t0_ms, fs=fs,
                   bi_cond=np.vstack([tcs[n]["bi_cond"] for n in names]),
                   source_cond=np.vstack([tcs[n]["source_cond"] for n in names]),
                   sink_cond=np.vstack([tcs[n]["sink_cond"] for n in names]),
                   bi_raw=np.vstack([tcs[n]["bi_raw"] for n in names]),
                   source_raw=np.vstack([tcs[n]["source_raw"] for n in names]),
                   sink_raw=np.vstack([tcs[n]["sink_raw"] for n in names]),
                   activity=np.vstack([tcs[n]["activity"] for n in names]))
        tc_dir = out_dir / "timecourses"; tc_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tc_dir / f"{group}__{h5_path.stem}.npz", **npz)
        recording_figure(tcs, t_ms, t_ms[disp], region_rois, f"{group} | {h5_path.stem} | t0 = {t0_ms:.1f} ms",
                         out_dir / "figures" / f"{group}__{h5_path.stem}__bi_activity")
        for reg, rn in region_rois.items():
            if rn:
                bi_traces.append(dict(group=group, slice_id=slice_id, condition=cond, region=reg, t=t_ms[disp],
                                      bi=np.nanmean(np.vstack([tcs[n]["bi_cond"] for n in rn]), axis=0)))
    return roi_rows, reg_rows, bi_traces


# =============================================================================
# CLI / MAIN
# =============================================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Regional CSD source/sink balance, activity, asymmetry and temporal "
                                            "dynamics for the largest event of each recording.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in_root", default=str(IN_ROOT), help="processed_2ms root (recursive; group = parent folder)")
    p.add_argument("--mask_dir", default=str(MASK_DIR), help="folder of <slice_id>__contact_mask_64x64.npy")
    p.add_argument("--roi_dir", default=str(ROI_DIR), help="folder of <slice_id>_roi_definitions.csv")
    p.add_argument("--out_dir", default=str(OUT_DIR), help="output folder")
    p.add_argument("--mask", default=None, help="override: one mask .npy used for every recording")
    p.add_argument("--roi_csv", default=None, help="override: one ROI CSV used for every recording")
    p.add_argument("--only_contains", default=None, help="process only files whose name contains this substring")
    p.add_argument("--epoch_pre_ms", type=float, default=EPOCH_PRE_MS, help="epoch start relative to t0 (negative)")
    p.add_argument("--epoch_post_ms", type=float, default=EPOCH_POST_MS, help="epoch end relative to t0")
    p.add_argument("--seed", type=int, default=0, help="unused (no randomness); kept for house-style consistency")
    return p


def main():
    args = build_argparser().parse_args()
    np.random.seed(args.seed)
    args.mask = Path(args.mask) if args.mask else None
    if args.epoch_pre_ms > BASELINE_MS[0] or args.epoch_post_ms < TOTAL_MS[1]:
        raise SystemExit(f"epoch must cover {BASELINE_MS[0]}..{TOTAL_MS[1]} ms")
    in_root, mask_dir, roi_dir, out_dir = Path(args.in_root), Path(args.mask_dir), Path(args.roi_dir), Path(args.out_dir)
    files = sorted(p for p in in_root.rglob("*.h5") if not args.only_contains or args.only_contains in p.name)
    if not files:
        raise SystemExit(f"no .h5 files under {in_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_rows, reg_rows, bi_traces = [], [], []
    for p in tqdm(files, desc="recordings"):
        group = p.parent.name if p.parent != in_root else in_root.name
        try:
            a, b, c = process_recording(p, group, args, mask_dir, roi_dir, out_dir)
            roi_rows += a; reg_rows += b; bi_traces += c
        except Exception as e:
            print(f"[ERROR] {p}: {type(e).__name__}: {e}")

    roi_cols = ["group", "slice_id", "condition", "file", "roi_name", "region", "event_index", "t0_sample", "t0_ms", "fs",
                "n_electrodes", "csd_peak", "total_current_0_500", "bi_time_avg", "source_auc", "sink_auc", "source_sink_ratio"]
    roi_df = pd.DataFrame(roi_rows, columns=roi_cols)
    roi_df.to_csv(out_dir / "roi_metrics.csv", index=False)
    reg_cols = ["group", "slice_id", "condition", "file", "region", "rois", "t0_sample", "t0_ms", "fs", "n_electrodes",
                "csd_peak", "total_current_0_500", "rai", "bi_time_avg", "source_auc", "sink_auc", "source_sink_ratio",
                "lag1_autocorr", "sample_entropy", "mean_crossing_rate_hz"]
    reg_df = pd.DataFrame(reg_rows, columns=reg_cols)
    reg_df.to_csv(out_dir / "region_metrics.csv", index=False)

    # animal level: ROI values averaged within region per (group, slice_id, condition)
    metric_cols = ["csd_peak", "total_current_0_500", "bi_time_avg", "source_auc", "sink_auc", "source_sink_ratio"]
    if len(roi_df):
        keys = ["group", "slice_id", "condition", "region"]
        an = roi_df.groupby(keys, as_index=False).agg(**{c: (c, "mean") for c in metric_cols}, n_rois=("roi_name", "count"))
        an.insert(4, "roi_name", "ROI_mean"); an.insert(5, "event_index", 0)
        dyn = reg_df[keys + ["rai", "lag1_autocorr", "sample_entropy", "mean_crossing_rate_hz"]].groupby(keys, as_index=False).mean()
        an = an.merge(dyn, on=keys, how="left")
    else:
        an = pd.DataFrame()
    an.to_csv(out_dir / "animal_level.csv", index=False)

    # group-mean BI figures per condition (per-animal traces = ROI-averaged within region)
    for cond in sorted({d["condition"] for d in bi_traces}):
        sel = [d for d in bi_traces if d["condition"] == cond]
        t_ref = sel[0]["t"]
        for d in sel:
            if d["t"].size != t_ref.size or not np.allclose(d["t"], t_ref):
                d["bi"] = np.interp(t_ref, d["t"], d["bi"]); d["t"] = t_ref
        group_mean_figure(sel, cond, out_dir / "figures" / f"group_mean_bi__{cond}")

    manifest = dict(
        script="core/regional_csd_dynamics.py", in_root=str(in_root), mask_dir=str(mask_dir), roi_dir=str(roi_dir),
        out_dir=str(out_dir), mask_override=str(args.mask) if args.mask else None, roi_csv_override=args.roi_csv,
        only_contains=args.only_contains, n_files=len(files), n_recordings_done=int(reg_df["file"].nunique()) if len(reg_df) else 0,
        t0="argmax over the recording of summed |LFP| across tissue-contact electrodes (largest event), guarded so the epoch fits",
        epoch_ms=[args.epoch_pre_ms, args.epoch_post_ms], baseline_ms=list(BASELINE_MS), baseline="median per electrode, inclusive limits",
        nan_fill="linear interpolation", analysis_ms=list(ANALYSIS_MS), bi_display_ms=list(BI_DISPLAY_MS), total_ms=list(TOTAL_MS),
        bi_time_avg="mean of conditioned BI_R(t) for t >= 0 within bi_display_ms",
        conditioning=dict(median_kernel=MEDIAN_KERNEL, lowpass_hz=LOWPASS_HZ, butter_order=BUTTER_ORDER, zero_phase="scipy.signal.filtfilt"),
        electrodes="ROI electrodes inside the contact mask and not on the outer ring; region = union of its ROI electrodes",
        regions=REGIONS, eps=EPS, integration="trapezoidal over time in ms",
        sample_entropy=dict(m=SAMPEN_M, r="%.2f * population s.d." % SAMPEN_R_FRAC, distance="Chebyshev", self_matches="excluded",
                            templates="N-m for both m and m+1", units="nats"),
        lag1_autocorr=dict(series="unfiltered BI of the pooled region, 0..500 ms, linearly detrended", estimator="statsmodels.tsa.stattools.acf" if HAVE_STATSMODELS else "numpy biased ACF"),
        mean_crossing_rate="sign changes of A_R(t) - mean across consecutive samples / ((K-1)*dt)",
        rai="(I_Ctx - I_Hippo)/(I_Ctx + I_Hippo), I = trapezoid of A_R over 0..500 ms",
        animal_level="ROI metrics averaged within region per (group, slice_id, condition); region dynamics joined",
        group_mean_figure="per-animal ROI-averaged conditioned BI within region; mean +/- s.e.m. over slices",
        fig_dpi=FIG_DPI, seed=args.seed, statsmodels=HAVE_STATSMODELS,
        versions=dict(numpy=np.__version__, pandas=pd.__version__, h5py=h5py.__version__),
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[DONE] {len(roi_df)} ROI rows, {len(reg_df)} region rows, {len(an)} animal-level rows -> {out_dir}")


if __name__ == "__main__":
    main()
