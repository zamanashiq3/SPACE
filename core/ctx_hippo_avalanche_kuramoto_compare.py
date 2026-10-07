# -*- coding: utf-8 -*-
"""
ctx_hippo_avalanche_kuramoto_compare.py -- SPACE framework: Cortex vs Hippocampus synchrony/avalanche comparison.

Manuscript figure(s): Figure 5H,I (Ctx vs Hippo Kuramoto R / phase velocity comparison).

What it computes:
    Pools ROI-level avalanche power-law statistics and Kuramoto/PLV/phase-gradient
    synchrony metrics into a Cortex (M1, M2, S1/SSP) vs Hippocampus (CA1, CA3, DG)
    comparison across phenotypes/conditions. This is what produces the CTX vs HIPPO
    grouped comparisons in the paper's synchrony figure.

    NOTE: this script's group-dict configuration (``PHENOTYPE_ROOTS``) uses generic
    "WT"/"5xFAD" keys as the pattern for defining comparison groups -- edit the paths
    (and, if needed, the phenotype keys/labels) to match your own genotype folder names.

Inputs / outputs (see data/README.md for the full schema):
    Input:  per-ROI, per-condition event-trace folders under
            ``data/event_traces/<group>/<region>/`` containing ``lfp_event_traces.npy``,
            ``csd_event_traces.npy``, and ``event_time_axis_ms.npy`` (one folder per group
            as listed in ``PHENOTYPE_ROOTS``).
    Output: CTX-vs-HIPPO avalanche/Kuramoto comparison tables and figures under
            ``data/results/ctx_hippo_compare/``, plus per-figure source-data tables.

ROI-based avalanche-stream + Kuramoto synchrony analysis for the same event-folder
structure used by the APP/5xFAD LFP/CSD/wavelet scripts.

GOAL
----
For each phenotype, ROI, region, and event:
    1) Load ROI event traces:
           lfp_event_traces.npy      # electrodes x time
           event_time_axis_ms.npy
       Optional:
           csd_event_traces.npy

    2) Run ROI-level LFP avalanche analysis:
           - baseline-correct electrode traces
           - detect negative LFP events per electrode
           - bin activity
           - extract avalanches as consecutive non-empty bins
           - save avalanche events, avalanche size/duration tables, summary metrics
           - fit size/duration power-law exponent if possible

    3) Run ROI-level Kuramoto analysis:
           - Hilbert phase per electrode
           - Kuramoto order parameter R(t)
           - circular mean phase psi(t)
           - phase dispersion / phase coherence
           - PLV matrix
           - optional 4x4 spatial phase-gradient metrics if ROI has 16 electrodes
           - save time series, PLV matrices, summary metrics, and figures

    4) Pool ROIs into:
           CTX   = M1, M2, SSP/S1
           HIPPO = CA1, CA3, DG

    5) Compare phenotypes:
           - CTX/HIPPO metric bar plots
           - normalized/group time-series plots
           - phenotype-difference plots
           - source data tables for every figure

EXPECTED INPUT
--------------
The same folder layout as your ROI event/archetype analysis:

    WT/WT_event_dbscan_archetype/
        M1/base/event_000_...ms/lfp_event_traces.npy
        M1/base/event_000_...ms/event_time_axis_ms.npy
        ...

    5xFAD/AD_event_dbscan_archetype/
        CA1/base/event_000_...ms/lfp_event_traces.npy
        CA1/base/event_000_...ms/event_time_axis_ms.npy
        ...

IMPORTANT
---------
This script does NOT delete or overwrite your previous LFP/CSD/wavelet outputs.
It creates a new output directory:
    BASELINE_ROI_AVALANCHE_KURAMOTO_CTX_HIPPO/

Author: SPACE project -- ROI-based avalanche + Kuramoto analysis.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None

try:
    from scipy.signal import hilbert, butter, sosfiltfilt, find_peaks
except Exception:
    hilbert = None
    butter = None
    sosfiltfilt = None
    find_peaks = None

try:
    import powerlaw
    HAVE_POWERLAW = True
except Exception:
    HAVE_POWERLAW = False


# =============================================================================
# CONFIG
# =============================================================================

# --- EDIT THIS --- (or override via --root_dir / --out_dir CLI flags, see build_argparser)
ROOT_DIR = Path("data/event_traces")
OUT_DIR = Path("data/results/ctx_hippo_compare")

# --- EDIT THIS --- group-folder dict: keys are your genotype/group labels, values are the
# per-group event-trace root folder(s) to search (relative to ROOT_DIR unless absolute).
# Point these at your own data/event_traces/<group>/ folders.
PHENOTYPE_ROOTS: Dict[str, List[Path]] = {
    "WT":    [Path("WT")],
    "5xFAD": [Path("5xFAD")],
}

PHENOTYPE_ORDER = ["WT", "5xFAD"]
PHENOTYPE_DISPLAY = {
    "WT": r"C57BL6",
    "5xFAD": r"5xFAD",
}
PHENOTYPE_COLORS = {
    "WT": "black",
    "5xFAD": "red",
}

BASELINE_ALIASES = {"base", "baseline", "basal", "control", "4ap", "4AP"}

ROI_ORDER = ["M1", "M2", "SSP", "S1", "CA1", "CA3", "DG"]
ROI_CANON = {
    "M1": "M1",
    "M2": "M2",
    "S1": "SSP",
    "SSP": "SSP",
    "SOMATOSENSORY": "SSP",
    "CA1": "CA1",
    "CA3": "CA3",
    "DG": "DG",
}
ROI_REGION_MAP = {
    "M1": "CTX",
    "M2": "CTX",
    "SSP": "CTX",
    "S1": "CTX",
    "CA1": "HIPPO",
    "CA3": "HIPPO",
    "DG": "HIPPO",
}
REGION_ORDER = ["CTX", "HIPPO"]
REGION_DISPLAY = {"CTX": "Ctx", "HIPPO": "Hippo"}
REGION_COLORS = {"CTX": "royalblue", "HIPPO": "darkorange"}

FIG_DPI = 600
SAVE_PNG = True
SAVE_SVG = True
SAVE_EXCEL = True

TIME_XLIM_MS = (-50, 250)
FEATURE_WINDOW_MS = (0, 250)
BASELINE_WINDOW_MS = (-50, 0)
EPS = 1e-12

BAR_WIDTH = 0.32
POINT_SIZE = 20
POINT_JITTER = 0.055

# ---------------- Avalanche settings ----------------
RUN_AVALANCHE = True

# LFP event detection uses negative peaks after baseline subtraction.
AVALANCHE_DETECT_WINDOW_MS = (0, 250)
AVALANCHE_BASELINE_WINDOW_MS = (-50, 0)
AVALANCHE_NEG_THRESH_MAD = 4.0       # threshold = median - MAD multiplier
AVALANCHE_MIN_PEAK_DISTANCE_MS = 4.0
AVALANCHE_BIN_WIDTH_MS_LIST = [2.0, 4.0, 8.0]
AVALANCHE_WEIGHTED_BY_AMPLITUDE = False
AVALANCHE_MIN_AVALANCHES_FOR_FIT = 10

# ---------------- Kuramoto settings ----------------
RUN_KURAMOTO = True

# Optional band-pass before Hilbert. For already event-filtered LFP, this can be off.
KURAMOTO_BANDPASS = False
KURAMOTO_BANDPASS_HZ = (4.0, 80.0)
KURAMOTO_FILTER_ORDER = 3

KURAMOTO_ANALYSIS_WINDOW_MS = (0, 250)
KURAMOTO_BASELINE_WINDOW_MS = (-50, 0)

# If ROI has 16 electrodes, assume 4x4 layout for spatial phase-gradient metrics.
KURAMOTO_GRID_SHAPE = (4, 4)

# Save larger per-event matrices? True gives detailed reproducibility but larger output.
SAVE_PER_EVENT_PLV_MATRICES = True
SAVE_PER_EVENT_TIME_SERIES = True


# =============================================================================
# STYLE / IO
# =============================================================================

def set_style():
    plt.rcParams.update({
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, out_base: Path):
    out_base = Path(out_base)
    ensure_dir(out_base.parent)
    try:
        if not fig.get_constrained_layout():
            fig.tight_layout(pad=1.2)
    except Exception:
        pass
    if SAVE_PNG:
        fig.savefig(str(out_base) + ".png", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.12)
    if SAVE_SVG:
        fig.savefig(str(out_base) + ".svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_df(df: pd.DataFrame, out_base: Path):
    out_base = Path(out_base)
    ensure_dir(out_base.parent)
    df.to_csv(str(out_base) + ".csv", index=False)
    if SAVE_EXCEL:
        try:
            df.to_excel(str(out_base) + ".xlsx", index=False)
        except Exception as e:
            print(f"[WARN] Excel save failed for {out_base}: {e}")


def sanitize(x: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-.]+", "_", str(x))
    return s.strip("_") or "NA"


def canon_roi(x: Any) -> str:
    s = str(x).upper().replace("-", "").replace("_", "")
    return ROI_CANON.get(s, str(x).upper())


def roi_region(roi: Any) -> str:
    return ROI_REGION_MAP.get(canon_roi(roi), "OTHER")


def is_baseline_dir_name(name: str) -> bool:
    s = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    return s in {re.sub(r"[^a-z0-9]+", "", a.lower()) for a in BASELINE_ALIASES}


def time_mask(t, window):
    t = np.asarray(t, dtype=float)
    m = (t >= float(window[0])) & (t <= float(window[1]))
    return m if np.any(m) else np.ones(t.size, dtype=bool)


def dt_from_time_axis(t):
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return 2.0
    d = np.diff(t)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 2.0
    return float(np.nanmedian(np.abs(d)))


def sampling_rate_from_time_ms(t):
    dt = dt_from_time_axis(t)
    return 1000.0 / dt if dt > 0 else 500.0


def fill_nan_1d(y):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    ok = np.isfinite(y)
    if np.all(ok):
        return y
    if not np.any(ok):
        return np.zeros_like(y)
    x = np.arange(y.size)
    out = y.copy()
    out[~ok] = np.interp(x[~ok], x[ok], y[ok])
    return out


def robust_mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def interpolate_to_reference(y, t, t_ref):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if np.sum(ok) < 2:
        return np.full_like(t_ref, np.nan, dtype=float)
    return np.interp(t_ref, t[ok], y[ok])


def mean_sem(Y):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    m = np.nanmean(Y, axis=0)
    n = np.sum(np.isfinite(Y), axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(n), out=np.zeros_like(sd), where=n > 1)
    return m, sem, n


def welch_pvalue(a, b):
    if stats is None:
        return np.nan
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        return np.nan


def p_to_label(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_sig_bar(ax, x1, x2, y, p):
    if not np.isfinite(y):
        return
    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1.0
    h = 0.035 * yr
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black", lw=0.8, clip_on=False)
    ax.text((x1+x2)/2, y+h*1.1, p_to_label(p), ha="center", va="bottom", fontsize=8)


# =============================================================================
# DATA DISCOVERY / LOAD
# =============================================================================

class DatasetRoot:
    def __init__(self, phenotype: str, path: Path):
        self.phenotype = phenotype
        self.path = Path(path)
        self.name = self.path.name

    def __repr__(self):
        return f"DatasetRoot(phenotype={self.phenotype}, path={self.path})"


def find_roi_dir(root: Path, roi: str) -> Optional[Path]:
    roi = canon_roi(roi)
    candidates = [roi, roi.lower()]
    if roi == "SSP":
        candidates += ["S1", "s1", "ssp"]
    for c in candidates:
        q = root / c
        if q.exists() and q.is_dir():
            return q
    for child in root.iterdir():
        if child.is_dir() and canon_roi(child.name) == roi:
            return child
    return None


def find_baseline_dir(roi_dir: Path) -> Optional[Path]:
    for child in roi_dir.iterdir():
        if child.is_dir() and is_baseline_dir_name(child.name):
            return child
    return None


def parse_event_folder_name(p: Path):
    m = re.search(r"event_(\d+)_(.+?)ms", p.name)
    if not m:
        return None, None
    try:
        return int(m.group(1)), float(m.group(2))
    except Exception:
        return None, None


def discover_dataset_roots() -> List[DatasetRoot]:
    roots = []
    for phenotype, paths in PHENOTYPE_ROOTS.items():
        for p in paths:
            p = ROOT_DIR / p if not p.is_absolute() else p
            if p.exists() and p.is_dir():
                roots.append(DatasetRoot(phenotype, p))
            else:
                print(f"[WARN] missing root: {phenotype} | {p}")
    if not roots:
        raise RuntimeError("No phenotype roots found. Edit PHENOTYPE_ROOTS at top of script.")
    return roots


def load_roi_events(datasets: List[DatasetRoot]) -> Tuple[List[dict], pd.DataFrame]:
    events = []
    rows = []
    load_rois = ["M1", "M2", "SSP", "CA1", "CA3", "DG"]

    for ds in datasets:
        for roi_name in load_rois:
            rdir = find_roi_dir(ds.path, roi_name)
            if rdir is None:
                print(f"[WARN] missing ROI dir: {ds.name} | {roi_name}")
                continue
            bdir = find_baseline_dir(rdir)
            if bdir is None:
                print(f"[WARN] missing baseline dir: {ds.name} | {roi_name}")
                continue

            edirs = sorted(
                bdir.glob("event_*_*ms"),
                key=lambda p: parse_event_folder_name(p)[0] if parse_event_folder_name(p)[0] is not None else 999999
            )
            print(f"[LOAD] {ds.phenotype:8s} | {ds.name} | {roi_name:4s} | {len(edirs)} events")

            for ev_dir in edirs:
                lp = ev_dir / "lfp_event_traces.npy"
                tp = ev_dir / "event_time_axis_ms.npy"
                cp = ev_dir / "csd_event_traces.npy"
                if not lp.exists() or not tp.exists():
                    continue

                try:
                    L = np.load(lp).astype(np.float32)
                    T = np.load(tp).astype(float)
                    C = np.load(cp).astype(np.float32) if cp.exists() else None
                except Exception as e:
                    print(f"[WARN] failed loading {ev_dir}: {e}")
                    continue

                if L.ndim == 1:
                    L = L[None, :]
                if C is not None and C.ndim == 1:
                    C = C[None, :]

                n = min(L.shape[1], len(T))
                if C is not None:
                    n = min(n, C.shape[1])
                L = L[:, :n]
                T = T[:n]
                if C is not None:
                    C = C[:, :n]

                ev_idx, ev_ms = parse_event_folder_name(ev_dir)
                ev_idx = int(ev_idx) if ev_idx is not None else len(events)
                ev_ms = float(ev_ms) if ev_ms is not None else np.nan
                roi_c = canon_roi(roi_name)
                region = roi_region(roi_c)
                event_key = f"{sanitize(ds.name)}__{roi_c}__base__event_{ev_idx:03d}"

                ev = dict(
                    phenotype=ds.phenotype,
                    phenotype_display=PHENOTYPE_DISPLAY.get(ds.phenotype, ds.phenotype),
                    dataset=ds.name,
                    root=str(ds.path),
                    roi_name=roi_c,
                    region=region,
                    drug="base",
                    event_index=ev_idx,
                    event_ms=ev_ms,
                    event_dir=str(ev_dir),
                    event_key=event_key,
                    time_ms=T,
                    lfp=L,
                    csd=C,
                    n_electrodes=int(L.shape[0]),
                )
                events.append(ev)
                rows.append({k: ev[k] for k in [
                    "phenotype", "dataset", "roi_name", "region", "drug",
                    "event_index", "event_ms", "event_dir", "event_key", "n_electrodes"
                ]})

    if not events:
        raise RuntimeError("No ROI event traces loaded.")
    return events, pd.DataFrame(rows)


# =============================================================================
# AVALANCHE ANALYSIS
# =============================================================================

def baseline_correct_electrodes(X: np.ndarray, t: np.ndarray, baseline_window=BASELINE_WINDOW_MS) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    bm = time_mask(t, baseline_window)
    base = np.nanmedian(X[:, bm], axis=1, keepdims=True)
    return X - base


def detect_negative_lfp_events_roi(L: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    """
    Detect negative LFP events per electrode in an ROI event snippet.

    Returns columns:
        elec_idx0, t_idx, time_ms, amp_abs, threshold
    """
    X = baseline_correct_electrodes(L, t, AVALANCHE_BASELINE_WINDOW_MS)
    wm = time_mask(t, AVALANCHE_DETECT_WINDOW_MS)
    idx_global = np.where(wm)[0]
    if idx_global.size < 3:
        return pd.DataFrame(columns=["elec_idx0", "t_idx", "time_ms", "amp_abs", "threshold"])

    dt_ms = dt_from_time_axis(t)
    min_dist = max(1, int(round(AVALANCHE_MIN_PEAK_DISTANCE_MS / max(dt_ms, EPS))))
    rows = []

    for ch in range(X.shape[0]):
        y = fill_nan_1d(X[ch])
        base_vals = y[time_mask(t, AVALANCHE_BASELINE_WINDOW_MS)]
        med = np.nanmedian(base_vals) if base_vals.size else np.nanmedian(y)
        mad = robust_mad(base_vals) if base_vals.size else robust_mad(y)
        if not np.isfinite(mad) or mad <= 0:
            mad = robust_mad(y)
        if not np.isfinite(mad) or mad <= 0:
            mad = np.nanstd(y) if np.isfinite(np.nanstd(y)) and np.nanstd(y) > 0 else 1.0
        thr = med - AVALANCHE_NEG_THRESH_MAD * mad

        yw = y[idx_global]
        # Detect negative peaks by finding peaks in -y above -thr.
        if find_peaks is not None:
            try:
                peaks_local, props = find_peaks(-yw, height=-thr, distance=min_dist)
            except Exception:
                peaks_local = np.array([], dtype=int)
        else:
            peaks_local = np.where((yw[1:-1] < yw[:-2]) & (yw[1:-1] < yw[2:]) & (yw[1:-1] <= thr))[0] + 1

        for pl in peaks_local:
            gi = int(idx_global[int(pl)])
            rows.append({
                "elec_idx0": int(ch),
                "t_idx": gi,
                "time_ms": float(t[gi]),
                "amp_abs": float(abs(y[gi])),
                "threshold": float(thr),
            })

    if not rows:
        return pd.DataFrame(columns=["elec_idx0", "t_idx", "time_ms", "amp_abs", "threshold"])
    return pd.DataFrame(rows).sort_values(["t_idx", "elec_idx0"]).reset_index(drop=True)


def aggregate_activity(events_df: pd.DataFrame, t: np.ndarray, bin_width_ms: float, weighted=False) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    wm = time_mask(t, AVALANCHE_DETECT_WINDOW_MS)
    tw = t[wm]
    if tw.size < 2:
        return np.array([]), np.array([])
    start, stop = float(tw[0]), float(tw[-1])
    edges = np.arange(start, stop + bin_width_ms + EPS, bin_width_ms)
    if edges.size < 2:
        return np.array([]), np.array([])
    activity = np.zeros(edges.size - 1, dtype=float)
    if events_df.empty:
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, activity

    vals = events_df["amp_abs"].to_numpy(dtype=float) if weighted else np.ones(len(events_df), dtype=float)
    inds = np.searchsorted(edges, events_df["time_ms"].to_numpy(dtype=float), side="right") - 1
    valid = (inds >= 0) & (inds < activity.size)
    np.add.at(activity, inds[valid], vals[valid])
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, activity


def extract_avalanches(activity: np.ndarray) -> pd.DataFrame:
    sizes, durations, starts, ends = [], [], [], []
    i = 0
    n = len(activity)
    while i < n:
        if activity[i] <= 0:
            i += 1
            continue
        s = i
        total = 0.0
        dur = 0
        while i < n and activity[i] > 0:
            total += float(activity[i])
            dur += 1
            i += 1
        e = i - 1
        sizes.append(total)
        durations.append(dur)
        starts.append(s)
        ends.append(e)

    return pd.DataFrame({
        "avalanche_id": np.arange(len(sizes), dtype=int),
        "size": np.asarray(sizes, dtype=float),
        "duration_bins": np.asarray(durations, dtype=int),
        "start_activity_bin": np.asarray(starts, dtype=int),
        "end_activity_bin": np.asarray(ends, dtype=int),
    })


def fit_powerlaw_metric(x: np.ndarray, min_n=AVALANCHE_MIN_AVALANCHES_FOR_FIT) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 1]
    if x.size < min_n:
        return {"alpha": np.nan, "xmin": np.nan, "sigma": np.nan, "fit_ok": 0, "fit_method": "insufficient_n"}

    xi = np.round(x).astype(int)

    if HAVE_POWERLAW:
        try:
            fit = powerlaw.Fit(xi, discrete=True, verbose=False)
            return {
                "alpha": float(fit.alpha),
                "xmin": float(fit.xmin),
                "sigma": float(getattr(fit, "sigma", np.nan)),
                "fit_ok": 1,
                "fit_method": "powerlaw",
            }
        except Exception:
            pass

    # fallback discrete MLE approximation xmin=1
    try:
        xmin = 1
        xx = xi[xi >= xmin]
        alpha = 1.0 + len(xx) / np.sum(np.log(xx / (xmin - 0.5)))
        return {
            "alpha": float(alpha),
            "xmin": float(xmin),
            "sigma": np.nan,
            "fit_ok": 1,
            "fit_method": "fallback_discrete_mle",
        }
    except Exception:
        return {"alpha": np.nan, "xmin": np.nan, "sigma": np.nan, "fit_ok": 0, "fit_method": "fallback_failed"}


def avalanche_event_analysis(ev: dict, out_event_dir: Path) -> Tuple[List[dict], pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        summary_rows, event_point_rows, avalanche_rows
    """
    if not RUN_AVALANCHE:
        return [], pd.DataFrame(), pd.DataFrame()

    L = np.asarray(ev["lfp"], dtype=float)
    t = np.asarray(ev["time_ms"], dtype=float)
    event_points = detect_negative_lfp_events_roi(L, t)

    base_cols = {k: ev[k] for k in [
        "phenotype", "dataset", "roi_name", "region", "drug",
        "event_index", "event_ms", "event_dir", "event_key", "n_electrodes"
    ]}

    if not event_points.empty:
        for k, v in base_cols.items():
            event_points[k] = v

    summary_rows = []
    avalanche_all = []

    for bw in AVALANCHE_BIN_WIDTH_MS_LIST:
        centers, activity = aggregate_activity(event_points, t, bw, weighted=AVALANCHE_WEIGHTED_BY_AMPLITUDE)
        aval = extract_avalanches(activity)

        if not aval.empty:
            aval["activity_bin_width_ms"] = float(bw)
            aval["start_time_ms"] = [float(centers[int(i)]) if 0 <= int(i) < len(centers) else np.nan for i in aval["start_activity_bin"]]
            aval["end_time_ms"] = [float(centers[int(i)]) if 0 <= int(i) < len(centers) else np.nan for i in aval["end_activity_bin"]]
            for k, v in base_cols.items():
                aval[k] = v
            avalanche_all.append(aval)

        n_bins = int(len(activity))
        active_bins = int(np.sum(activity > 0))
        n_events = int(len(event_points))
        n_aval = int(len(aval))
        size_vals = aval["size"].to_numpy(dtype=float) if not aval.empty else np.array([])
        dur_vals = aval["duration_bins"].to_numpy(dtype=float) if not aval.empty else np.array([])

        size_fit = fit_powerlaw_metric(size_vals)
        dur_fit = fit_powerlaw_metric(dur_vals)

        if dur_vals.size > 0 and size_vals.size > 0 and np.nanmax(dur_vals) > 1:
            try:
                logd = np.log10(dur_vals[dur_vals > 0])
                logs = np.log10(size_vals[dur_vals > 0])
                sd_slope = float(np.polyfit(logd, logs, 1)[0]) if logd.size >= 2 else np.nan
            except Exception:
                sd_slope = np.nan
        else:
            sd_slope = np.nan

        row = dict(base_cols)
        row.update({
            "activity_bin_width_ms": float(bw),
            "n_detected_lfp_events": n_events,
            "n_activity_bins": n_bins,
            "n_active_bins": active_bins,
            "active_bin_fraction": float(active_bins / max(n_bins, 1)),
            "activity_mean_per_bin": float(np.nanmean(activity)) if activity.size else np.nan,
            "activity_mean_per_active_bin": float(np.nanmean(activity[activity > 0])) if np.any(activity > 0) else 0.0,
            "n_avalanches": n_aval,
            "avalanche_rate_per_s": float(n_aval / ((AVALANCHE_DETECT_WINDOW_MS[1] - AVALANCHE_DETECT_WINDOW_MS[0]) / 1000.0 + EPS)),
            "avalanche_size_mean": float(np.nanmean(size_vals)) if size_vals.size else 0.0,
            "avalanche_size_median": float(np.nanmedian(size_vals)) if size_vals.size else 0.0,
            "avalanche_size_max": float(np.nanmax(size_vals)) if size_vals.size else 0.0,
            "avalanche_duration_mean_bins": float(np.nanmean(dur_vals)) if dur_vals.size else 0.0,
            "avalanche_duration_median_bins": float(np.nanmedian(dur_vals)) if dur_vals.size else 0.0,
            "avalanche_duration_max_bins": float(np.nanmax(dur_vals)) if dur_vals.size else 0.0,
            "avalanche_size_alpha": size_fit["alpha"],
            "avalanche_size_xmin": size_fit["xmin"],
            "avalanche_size_fit_ok": size_fit["fit_ok"],
            "avalanche_size_fit_method": size_fit["fit_method"],
            "avalanche_duration_alpha": dur_fit["alpha"],
            "avalanche_duration_xmin": dur_fit["xmin"],
            "avalanche_duration_fit_ok": dur_fit["fit_ok"],
            "avalanche_duration_fit_method": dur_fit["fit_method"],
            "avalanche_size_duration_loglog_slope": sd_slope,
        })
        summary_rows.append(row)

        if SAVE_PER_EVENT_TIME_SERIES:
            ensure_dir(out_event_dir)
            np.savez_compressed(
                out_event_dir / f"avalanche_activity_bw{sanitize(bw)}ms.npz",
                time_bin_centers_ms=centers.astype(np.float32),
                activity=activity.astype(np.float32),
                bin_width_ms=np.float32(bw),
            )

    aval_df = pd.concat(avalanche_all, ignore_index=True) if avalanche_all else pd.DataFrame()
    return summary_rows, event_points, aval_df


def ccdf_xy(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    uniq = np.unique(xs)
    y = np.array([np.mean(xs >= u) for u in uniq], dtype=float)
    return uniq, y


# =============================================================================
# KURAMOTO ANALYSIS
# =============================================================================

def maybe_bandpass(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if not KURAMOTO_BANDPASS:
        return X
    if butter is None or sosfiltfilt is None:
        return X
    fs = sampling_rate_from_time_ms(t)
    lo, hi = KURAMOTO_BANDPASS_HZ
    try:
        sos = butter(KURAMOTO_FILTER_ORDER, [lo, hi], btype="bandpass", fs=fs, output="sos")
        return sosfiltfilt(sos, X, axis=1)
    except Exception:
        return X


def circular_mean_phase(theta: np.ndarray, axis=0):
    z = np.nanmean(np.exp(1j * theta), axis=axis)
    return np.angle(z)


def kuramoto_order_parameter(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    theta: electrodes x time
    Returns:
        R(t), psi(t)
    """
    z = np.nanmean(np.exp(1j * theta), axis=0)
    return np.abs(z), np.angle(z)


def compute_plv_matrix(theta: np.ndarray, mask_t: np.ndarray) -> np.ndarray:
    th = theta[:, mask_t]
    n = th.shape[0]
    M = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = np.abs(np.nanmean(np.exp(1j * (th[i] - th[j]))))
            M[i, j] = M[j, i] = float(val)
    return M


def phase_gradient_metrics(theta: np.ndarray, t: np.ndarray, mask_t: np.ndarray) -> Dict[str, float]:
    """
    If ROI has exactly 16 electrodes, interpret it as 4x4.
    Compute phase spatial gradient and a wave-speed proxy.
    """
    n_ch = theta.shape[0]
    gh, gw = KURAMOTO_GRID_SHAPE
    if n_ch != gh * gw:
        return {
            "phase_grad_mag_mean": np.nan,
            "phase_grad_mag_peak": np.nan,
            "phase_singularity_count_mean": np.nan,
            "phase_singularity_count_peak": np.nan,
            "wave_speed_proxy_mean": np.nan,
            "wave_speed_proxy_peak": np.nan,
        }

    sel = np.where(mask_t)[0]
    if sel.size == 0:
        return {}

    grad_vals = []
    sing_counts = []
    speed_vals = []

    for ti in sel:
        ph = theta[:, ti].reshape(gh, gw)
        # unwrap local phase along axes for smoother gradients
        ph_u = np.unwrap(np.unwrap(ph, axis=0), axis=1)
        gy, gx = np.gradient(ph_u)
        mag = np.sqrt(gx * gx + gy * gy)
        grad_vals.append(np.nanmean(mag))
        speed_vals.append(np.sqrt(np.nanmean(mag)))  # same proxy style as your Kuramoto plotting script

        count = 0
        for y in range(gh - 1):
            for x in range(gw - 1):
                block = ph[y:y+2, x:x+2]
                diffs = [
                    np.angle(np.exp(1j * (block[0,1] - block[0,0]))),
                    np.angle(np.exp(1j * (block[1,1] - block[0,1]))),
                    np.angle(np.exp(1j * (block[1,0] - block[1,1]))),
                    np.angle(np.exp(1j * (block[0,0] - block[1,0]))),
                ]
                winding = np.sum(diffs)
                if abs(winding) > np.pi:
                    count += 1
        sing_counts.append(count)

    grad_vals = np.asarray(grad_vals, dtype=float)
    sing_counts = np.asarray(sing_counts, dtype=float)
    speed_vals = np.asarray(speed_vals, dtype=float)

    return {
        "phase_grad_mag_mean": float(np.nanmean(grad_vals)) if grad_vals.size else np.nan,
        "phase_grad_mag_peak": float(np.nanmax(grad_vals)) if grad_vals.size else np.nan,
        "phase_singularity_count_mean": float(np.nanmean(sing_counts)) if sing_counts.size else np.nan,
        "phase_singularity_count_peak": float(np.nanmax(sing_counts)) if sing_counts.size else np.nan,
        "wave_speed_proxy_mean": float(np.nanmean(speed_vals)) if speed_vals.size else np.nan,
        "wave_speed_proxy_peak": float(np.nanmax(speed_vals)) if speed_vals.size else np.nan,
    }


def kuramoto_event_analysis(ev: dict, out_event_dir: Path) -> Tuple[dict, pd.DataFrame]:
    if not RUN_KURAMOTO:
        return {}, pd.DataFrame()

    if hilbert is None:
        raise RuntimeError("scipy.signal.hilbert is required for Kuramoto analysis.")

    L = np.asarray(ev["lfp"], dtype=float)
    t = np.asarray(ev["time_ms"], dtype=float)
    X = baseline_correct_electrodes(L, t, KURAMOTO_BASELINE_WINDOW_MS)
    X = np.vstack([fill_nan_1d(x) for x in X])
    X = maybe_bandpass(X, t)

    analytic = hilbert(X, axis=1)
    amp = np.abs(analytic)
    theta = np.angle(analytic)

    R, psi = kuramoto_order_parameter(theta)
    wm = time_mask(t, KURAMOTO_ANALYSIS_WINDOW_MS)
    bm = time_mask(t, KURAMOTO_BASELINE_WINDOW_MS)

    plv = compute_plv_matrix(theta, wm)
    iu = np.triu_indices_from(plv, 1)
    plv_vals = plv[iu]
    plv_vals = plv_vals[np.isfinite(plv_vals)]

    phase_disp = 1.0 - R
    amp_mean = np.nanmean(amp, axis=0)

    grad = phase_gradient_metrics(theta, t, wm)

    row = {k: ev[k] for k in [
        "phenotype", "dataset", "roi_name", "region", "drug",
        "event_index", "event_ms", "event_dir", "event_key", "n_electrodes"
    ]}
    row.update({
        "kuramoto_R_mean": float(np.nanmean(R[wm])) if np.any(wm) else np.nan,
        "kuramoto_R_peak": float(np.nanmax(R[wm])) if np.any(wm) else np.nan,
        "kuramoto_R_min": float(np.nanmin(R[wm])) if np.any(wm) else np.nan,
        "kuramoto_R_auc": float(np.trapezoid(R[wm], t[wm])) if np.any(wm) else np.nan,
        "kuramoto_R_baseline_mean": float(np.nanmean(R[bm])) if np.any(bm) else np.nan,
        "kuramoto_R_evoked_over_baseline": float(np.nanmean(R[wm]) / (np.nanmean(R[bm]) + EPS)) if np.any(wm) and np.any(bm) else np.nan,
        "phase_dispersion_mean": float(np.nanmean(phase_disp[wm])) if np.any(wm) else np.nan,
        "phase_dispersion_min": float(np.nanmin(phase_disp[wm])) if np.any(wm) else np.nan,
        "plv_mean": float(np.nanmean(plv_vals)) if plv_vals.size else np.nan,
        "plv_median": float(np.nanmedian(plv_vals)) if plv_vals.size else np.nan,
        "plv_std": float(np.nanstd(plv_vals)) if plv_vals.size else np.nan,
        "analytic_amp_mean": float(np.nanmean(amp_mean[wm])) if np.any(wm) else np.nan,
        "analytic_amp_peak": float(np.nanmax(amp_mean[wm])) if np.any(wm) else np.nan,
    })
    row.update(grad)

    ts_df = pd.DataFrame({
        "time_ms": t,
        "kuramoto_R": R,
        "mean_phase_rad": psi,
        "phase_dispersion": phase_disp,
        "analytic_amp_mean": amp_mean,
    })
    for k in ["phenotype", "dataset", "roi_name", "region", "event_index", "event_key"]:
        ts_df[k] = ev[k]

    if SAVE_PER_EVENT_TIME_SERIES or SAVE_PER_EVENT_PLV_MATRICES:
        ensure_dir(out_event_dir)

    if SAVE_PER_EVENT_TIME_SERIES:
        np.savez_compressed(
            out_event_dir / "kuramoto_timeseries.npz",
            time_ms=t.astype(np.float32),
            R=R.astype(np.float32),
            psi=psi.astype(np.float32),
            phase_dispersion=phase_disp.astype(np.float32),
            analytic_amp_mean=amp_mean.astype(np.float32),
        )

    if SAVE_PER_EVENT_PLV_MATRICES:
        np.save(out_event_dir / "kuramoto_plv_matrix.npy", plv.astype(np.float32))

    return row, ts_df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_metric_bars(df: pd.DataFrame, metrics: List[Tuple[str, str]], out_base: Path, title: str):
    if df.empty:
        return
    set_style()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(max(3.2 * n, 7.2), 3.6), squeeze=False, constrained_layout=True)
    axes = axes[0]
    stat_rows = []

    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in df.columns:
            ax.axis("off")
            continue
        sub = df.copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        xs = np.arange(len(PHENOTYPE_ORDER), dtype=float)

        for i, phenotype in enumerate(PHENOTYPE_ORDER):
            for j, region in enumerate(REGION_ORDER):
                vals = sub.loc[(sub["phenotype"] == phenotype) & (sub["region"] == region), metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                x = xs[i] + (j - 0.5) * BAR_WIDTH
                mean = float(np.nanmean(vals)) if vals.size else np.nan
                sem = float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
                ax.bar(x, mean, yerr=sem, width=BAR_WIDTH,
                       facecolor=REGION_COLORS[region], alpha=0.25,
                       edgecolor=REGION_COLORS[region], linewidth=1.4, capsize=2)
                if vals.size:
                    jitter = np.linspace(-POINT_JITTER, POINT_JITTER, vals.size) if vals.size > 1 else np.array([0.0])
                    ax.scatter(np.full(vals.size, x) + jitter, vals, s=POINT_SIZE,
                               color=REGION_COLORS[region], alpha=0.85, zorder=3)
                stat_rows.append({
                    "metric": metric, "phenotype": phenotype, "region": region,
                    "n": int(vals.size), "mean": mean, "sem": sem,
                })

        for j, region in enumerate(REGION_ORDER):
            a = sub.loc[(sub["phenotype"] == PHENOTYPE_ORDER[0]) & (sub["region"] == region), metric].to_numpy(dtype=float)
            b = sub.loc[(sub["phenotype"] == PHENOTYPE_ORDER[1]) & (sub["region"] == region), metric].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            p = welch_pvalue(a, b)
            stat_rows.append({
                "metric": metric,
                "comparison": f"{PHENOTYPE_ORDER[0]}_vs_{PHENOTYPE_ORDER[1]}",
                "region": region,
                "p_value": p,
                "n_a": int(a.size),
                "n_b": int(b.size),
                "test": "Welch t-test",
            })
            both = np.concatenate([a, b]) if a.size and b.size else np.array([])
            if both.size:
                y = np.nanmax(both)
                if np.isfinite(y):
                    x1 = xs[0] + (j - 0.5) * BAR_WIDTH
                    x2 = xs[1] + (j - 0.5) * BAR_WIDTH
                    ax.set_ylim(top=max(ax.get_ylim()[1], y * 1.28 if y > 0 else y + 1))
                    add_sig_bar(ax, x1, x2, y * 1.07 if y > 0 else y + 0.1, p)

        ax.set_xticks(xs)
        ax.set_xticklabels([PHENOTYPE_DISPLAY[p] for p in PHENOTYPE_ORDER])
        ax.set_title(ylabel, fontsize=10, pad=6)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [plt.Line2D([0], [0], color=REGION_COLORS[r], lw=6, alpha=0.35) for r in REGION_ORDER]
    labels = [REGION_DISPLAY[r] for r in REGION_ORDER]
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, fontsize=11)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(stat_rows), Path(str(out_base) + "_summary"))


def plot_kuramoto_R_lines(ts_df: pd.DataFrame, out_base: Path):
    if ts_df.empty:
        return
    set_style()
    fig, axes = plt.subplots(1, len(PHENOTYPE_ORDER), figsize=(7.4, 3.2), sharey=True, constrained_layout=True)
    if len(PHENOTYPE_ORDER) == 1:
        axes = [axes]
    source_rows = []

    for ax, phenotype in zip(axes, PHENOTYPE_ORDER):
        for region in REGION_ORDER:
            sub = ts_df[(ts_df["phenotype"] == phenotype) & (ts_df["region"] == region)]
            if sub.empty:
                continue
            traces = []
            t_ref = None
            for _, g in sub.groupby("event_key"):
                t = g["time_ms"].to_numpy(dtype=float)
                y = g["kuramoto_R"].to_numpy(dtype=float)
                if t_ref is None:
                    m = (t >= TIME_XLIM_MS[0]) & (t <= TIME_XLIM_MS[1])
                    t_ref = t[m]
                traces.append(interpolate_to_reference(y, t, t_ref))
            if not traces or t_ref is None:
                continue
            Y = np.vstack(traces)
            m, s, _ = mean_sem(Y)
            color = REGION_COLORS[region]
            ax.fill_between(t_ref, m - s, m + s, color=color, alpha=0.16, linewidth=0)
            ax.plot(t_ref, m, color=color, lw=1.2, label=REGION_DISPLAY[region])
            for i, tm in enumerate(t_ref):
                source_rows.append({
                    "phenotype": phenotype, "region": region,
                    "time_ms": float(tm), "kuramoto_R_mean": float(m[i]), "kuramoto_R_sem": float(s[i])
                })

        ax.axvline(0, color="0.45", ls="--", lw=0.9)
        ax.set_xlim(TIME_XLIM_MS)
        ax.set_ylim(0, 1.02)
        ax.set_title(PHENOTYPE_DISPLAY.get(phenotype, phenotype), fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Kuramoto order R")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("ROI Kuramoto synchrony time course", fontsize=11)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(source_rows), Path(str(out_base) + "_source_data"))


def plot_avalanche_ccdf_by_group(aval_df: pd.DataFrame, out_base: Path, bin_width_ms: float = 2.0):
    if aval_df.empty:
        return
    sub = aval_df[np.isclose(pd.to_numeric(aval_df["activity_bin_width_ms"], errors="coerce"), bin_width_ms)]
    if sub.empty:
        return

    set_style()
    fig, axes = plt.subplots(1, len(REGION_ORDER), figsize=(7.2, 3.4), constrained_layout=True)
    if len(REGION_ORDER) == 1:
        axes = [axes]
    source_rows = []

    for ax, region in zip(axes, REGION_ORDER):
        for phenotype in PHENOTYPE_ORDER:
            vals = sub.loc[(sub["region"] == region) & (sub["phenotype"] == phenotype), "size"].to_numpy(dtype=float)
            x, y = ccdf_xy(vals)
            if x.size:
                ax.scatter(x, y, s=14, alpha=0.8, label=PHENOTYPE_DISPLAY[phenotype], color=PHENOTYPE_COLORS[phenotype])
                for xi, yi in zip(x, y):
                    source_rows.append({"phenotype": phenotype, "region": region, "size": float(xi), "ccdf": float(yi)})
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Avalanche size")
        ax.set_ylabel("CCDF")
        ax.set_title(f"{REGION_DISPLAY[region]} | bw={bin_width_ms:g} ms")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("ROI avalanche size distributions", fontsize=11)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(source_rows), Path(str(out_base) + "_source_data"))


def plot_avalanche_activity_examples(activity_dir: Path, out_base: Path):
    # Optional lightweight placeholder; activity traces are saved per event as NPZ.
    pass



# =============================================================================
# ENHANCED KURAMOTO COMPARISON / VISUALIZATION
# =============================================================================
# These functions add richer, more interpretable comparisons without removing any
# of the original Kuramoto outputs above.
# Main idea: simple mean R bars are weak because R(t) is dynamic.  We therefore
# quantify event dynamics: baseline-to-evoked change, late recruitment, slope,
# peak latency, CTX-HIPPO contrast, effect-size ranking, ROI heatmaps, and
# phenotype-difference time courses.

KUR_ENHANCED_WINDOWS_MS = {
    "baseline": (-50, 0),
    "early": (0, 50),
    "mid": (50, 150),
    "late": (150, 250),
    "full": (0, 250),
}

KUR_ENHANCED_METRICS = [
    ("R_delta_late_minus_base", "ΔR late-baseline"),
    ("R_late_over_base", "Late/base R"),
    ("R_slope_full", "R slope 0-250 ms"),
    ("R_peak_latency_ms", "R peak latency"),
    ("R_auc_late", "Late R AUC"),
    ("R_late_mean", "Late mean R"),
    ("phase_dispersion_late_mean", "Late phase dispersion"),
    ("analytic_amp_late_mean", "Late analytic amp."),
    ("plv_mean", "Mean PLV"),
    ("phase_grad_mag_mean", "Phase-gradient mag."),
    ("wave_speed_proxy_mean", "Wave-speed proxy"),
    ("phase_singularity_count_mean", "Phase singularities"),
]


def _window_mean_from_ts(g: pd.DataFrame, col: str, window: Tuple[float, float]) -> float:
    if g.empty or col not in g.columns:
        return np.nan
    t = g["time_ms"].to_numpy(dtype=float)
    y = g[col].to_numpy(dtype=float)
    m = (t >= float(window[0])) & (t <= float(window[1])) & np.isfinite(y)
    return float(np.nanmean(y[m])) if np.any(m) else np.nan


def _window_auc_from_ts(g: pd.DataFrame, col: str, window: Tuple[float, float]) -> float:
    if g.empty or col not in g.columns:
        return np.nan
    t = g["time_ms"].to_numpy(dtype=float)
    y = g[col].to_numpy(dtype=float)
    m = (t >= float(window[0])) & (t <= float(window[1])) & np.isfinite(y)
    if np.sum(m) < 2:
        return np.nan
    return float(np.trapezoid(y[m], t[m]))


def _window_slope_from_ts(g: pd.DataFrame, col: str, window: Tuple[float, float]) -> float:
    if g.empty or col not in g.columns:
        return np.nan
    t = g["time_ms"].to_numpy(dtype=float)
    y = g[col].to_numpy(dtype=float)
    m = (t >= float(window[0])) & (t <= float(window[1])) & np.isfinite(y) & np.isfinite(t)
    if np.sum(m) < 3:
        return np.nan
    try:
        return float(np.polyfit(t[m], y[m], 1)[0])
    except Exception:
        return np.nan


def _peak_latency_from_ts(g: pd.DataFrame, col: str, window: Tuple[float, float]) -> float:
    if g.empty or col not in g.columns:
        return np.nan
    t = g["time_ms"].to_numpy(dtype=float)
    y = g[col].to_numpy(dtype=float)
    m = (t >= float(window[0])) & (t <= float(window[1])) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    tt = t[m]
    yy = y[m]
    try:
        return float(tt[int(np.nanargmax(yy))])
    except Exception:
        return np.nan


def add_kuramoto_timeseries_features(kuramoto_summary: pd.DataFrame, kuramoto_ts: pd.DataFrame) -> pd.DataFrame:
    """Add dynamic Kuramoto features computed from R(t), phase dispersion, and analytic amplitude."""
    if kuramoto_summary.empty or kuramoto_ts.empty:
        return kuramoto_summary.copy()
    rows = []
    for event_key, g in kuramoto_ts.groupby("event_key"):
        row = {"event_key": event_key}
        # R windows
        for wname, win in KUR_ENHANCED_WINDOWS_MS.items():
            row[f"R_{wname}_mean"] = _window_mean_from_ts(g, "kuramoto_R", win)
            row[f"R_{wname}_auc"] = _window_auc_from_ts(g, "kuramoto_R", win)
            row[f"phase_dispersion_{wname}_mean"] = _window_mean_from_ts(g, "phase_dispersion", win)
            row[f"analytic_amp_{wname}_mean"] = _window_mean_from_ts(g, "analytic_amp_mean", win)
        row["R_delta_early_minus_base"] = row.get("R_early_mean", np.nan) - row.get("R_baseline_mean", np.nan)
        row["R_delta_mid_minus_base"] = row.get("R_mid_mean", np.nan) - row.get("R_baseline_mean", np.nan)
        row["R_delta_late_minus_base"] = row.get("R_late_mean", np.nan) - row.get("R_baseline_mean", np.nan)
        row["R_late_over_base"] = row.get("R_late_mean", np.nan) / (row.get("R_baseline_mean", np.nan) + EPS)
        row["R_mid_over_base"] = row.get("R_mid_mean", np.nan) / (row.get("R_baseline_mean", np.nan) + EPS)
        row["R_slope_full"] = _window_slope_from_ts(g, "kuramoto_R", KUR_ENHANCED_WINDOWS_MS["full"])
        row["R_slope_late"] = _window_slope_from_ts(g, "kuramoto_R", KUR_ENHANCED_WINDOWS_MS["late"])
        row["R_peak_latency_ms"] = _peak_latency_from_ts(g, "kuramoto_R", KUR_ENHANCED_WINDOWS_MS["full"])
        row["phase_dispersion_delta_late_minus_base"] = row.get("phase_dispersion_late_mean", np.nan) - row.get("phase_dispersion_baseline_mean", np.nan)
        row["analytic_amp_delta_late_minus_base"] = row.get("analytic_amp_late_mean", np.nan) - row.get("analytic_amp_baseline_mean", np.nan)
        rows.append(row)
    dyn = pd.DataFrame(rows)
    out = pd.merge(kuramoto_summary.copy(), dyn, on="event_key", how="left")
    return out


def aggregate_event_to_dataset_roi(df: pd.DataFrame, metrics: List[Tuple[str, str]]) -> pd.DataFrame:
    """Aggregate events to dataset x ROI so stats are less pseudo-replicated than event-level bars."""
    if df.empty:
        return df.copy()
    metric_cols = [m for m, _ in metrics if m in df.columns]
    group_cols = ["phenotype", "dataset", "roi_name", "region"]
    keep = [c for c in group_cols if c in df.columns] + metric_cols
    if not metric_cols:
        return pd.DataFrame()
    out = df[keep].copy()
    for c in metric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    agg = out.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    return agg


def cohen_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    va = np.nanvar(a, ddof=1); vb = np.nanvar(b, ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1))
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return float((np.nanmean(b) - np.nanmean(a)) / pooled)


def build_kuramoto_sensitivity_table(df: pd.DataFrame, metrics: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        for region in list(REGION_ORDER) + ["ALL"]:
            sub = df if region == "ALL" else df[df["region"] == region]
            a = pd.to_numeric(sub.loc[sub["phenotype"] == PHENOTYPE_ORDER[0], metric], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(sub.loc[sub["phenotype"] == PHENOTYPE_ORDER[1], metric], errors="coerce").to_numpy(dtype=float)
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            p = welch_pvalue(a, b)
            rows.append({
                "metric": metric,
                "label": label,
                "region": region,
                "n_a": int(a.size),
                "n_b": int(b.size),
                "mean_a": float(np.nanmean(a)) if a.size else np.nan,
                "mean_b": float(np.nanmean(b)) if b.size else np.nan,
                "cohen_d_5xFAD_minus_WT": cohen_d(a, b),
                "p_value": p,
                "neglog10_p": float(-np.log10(p + EPS)) if np.isfinite(p) else np.nan,
                "test": "Welch t-test on dataset-ROI means",
            })
    return pd.DataFrame(rows)


def plot_kuramoto_effect_rank(stats_df: pd.DataFrame, out_base: Path, top_n: int = 14):
    if stats_df.empty:
        return
    sub = stats_df[stats_df["region"] == "ALL"].copy()
    sub = sub[np.isfinite(pd.to_numeric(sub["neglog10_p"], errors="coerce"))]
    if sub.empty:
        return
    sub = sub.sort_values("neglog10_p", ascending=True).tail(top_n)
    set_style()
    fig, ax = plt.subplots(figsize=(5.8, max(3.4, 0.34 * len(sub))), constrained_layout=True)
    ax.barh(sub["label"], sub["neglog10_p"])
    for xref, lab in [(1.301, "p=0.05"), (2.0, "p=0.01"), (3.0, "p=0.001")]:
        ax.axvline(xref, color="0.55", ls="--", lw=0.8)
        ax.text(xref, 1.01, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title("Kuramoto feature sensitivity ranking")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save_fig(fig, out_base)
    save_df(sub, Path(str(out_base) + "_source_data"))


def plot_kuramoto_effect_size_rank(stats_df: pd.DataFrame, out_base: Path, top_n: int = 14):
    if stats_df.empty:
        return
    sub = stats_df[stats_df["region"] == "ALL"].copy()
    sub = sub[np.isfinite(pd.to_numeric(sub["cohen_d_5xFAD_minus_WT"], errors="coerce"))]
    if sub.empty:
        return
    sub["abs_d"] = np.abs(sub["cohen_d_5xFAD_minus_WT"])
    sub = sub.sort_values("abs_d", ascending=True).tail(top_n)
    set_style()
    fig, ax = plt.subplots(figsize=(5.8, max(3.4, 0.34 * len(sub))), constrained_layout=True)
    ax.barh(sub["label"], sub["cohen_d_5xFAD_minus_WT"])
    ax.axvline(0, color="0.2", lw=0.8)
    ax.axvline(0.8, color="0.6", ls="--", lw=0.8)
    ax.axvline(-0.8, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("Cohen's d (5xFAD − C57BL6)")
    ax.set_title("Kuramoto effect-size ranking")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save_fig(fig, out_base)
    save_df(sub, Path(str(out_base) + "_source_data"))


def plot_kuramoto_roi_heatmap(df_roi: pd.DataFrame, metric: str, label: str, out_base: Path):
    if df_roi.empty or metric not in df_roi.columns:
        return
    rois = [r for r in ["M1", "M2", "SSP", "CA1", "CA3", "DG"] if r in set(df_roi["roi_name"])]
    if not rois:
        return
    mat = np.full((len(PHENOTYPE_ORDER), len(rois)), np.nan, dtype=float)
    rows = []
    for i, pheno in enumerate(PHENOTYPE_ORDER):
        for j, roi in enumerate(rois):
            vals = pd.to_numeric(df_roi.loc[(df_roi["phenotype"] == pheno) & (df_roi["roi_name"] == roi), metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            mat[i, j] = float(np.nanmean(vals)) if vals.size else np.nan
            rows.append({"phenotype": pheno, "roi_name": roi, "metric": metric, "mean": mat[i, j], "n": int(vals.size)})
    v = np.nanmax(np.abs(mat - np.nanmean(mat))) if np.any(np.isfinite(mat)) else 1.0
    if not np.isfinite(v) or v <= 0:
        v = np.nanmax(np.abs(mat)) if np.any(np.isfinite(mat)) else 1.0
    set_style()
    fig, ax = plt.subplots(figsize=(5.7, 2.2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=np.nanmean(mat)-v, vmax=np.nanmean(mat)+v)
    ax.set_xticks(np.arange(len(rois))); ax.set_xticklabels(rois)
    ax.set_yticks(np.arange(len(PHENOTYPE_ORDER))); ax.set_yticklabels([PHENOTYPE_DISPLAY[p] for p in PHENOTYPE_ORDER])
    ax.set_title(f"ROI heatmap | {label}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2g}", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(rows), Path(str(out_base) + "_source_data"))


def plot_kuramoto_roi_lines(df_roi: pd.DataFrame, metric: str, label: str, out_base: Path):
    if df_roi.empty or metric not in df_roi.columns:
        return
    rois = [r for r in ["M1", "M2", "SSP", "CA1", "CA3", "DG"] if r in set(df_roi["roi_name"])]
    if not rois:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.3), constrained_layout=True)
    source_rows = []
    x = np.arange(len(rois))
    for pheno in PHENOTYPE_ORDER:
        means, sems = [], []
        for roi in rois:
            vals = pd.to_numeric(df_roi.loc[(df_roi["phenotype"] == pheno) & (df_roi["roi_name"] == roi), metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.nanmean(vals)) if vals.size else np.nan)
            sems.append(float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
            source_rows.append({"phenotype": pheno, "roi_name": roi, "metric": metric, "mean": means[-1], "sem": sems[-1], "n": int(vals.size)})
        ax.errorbar(x, means, yerr=sems, marker="o", lw=1.2, capsize=2, label=PHENOTYPE_DISPLAY[pheno], color=PHENOTYPE_COLORS[pheno])
    ax.set_xticks(x); ax.set_xticklabels(rois)
    ax.set_ylabel(label)
    ax.set_title(f"ROI-resolved Kuramoto comparison | {label}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(source_rows), Path(str(out_base) + "_source_data"))


def plot_kuramoto_region_difference_timecourse(ts_df: pd.DataFrame, out_base: Path):
    """Plot 5xFAD - C57BL6 difference in R(t) for CTX and HIPPO."""
    if ts_df.empty:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    source_rows = []
    for region in REGION_ORDER:
        region_data = {}
        t_ref = None
        for pheno in PHENOTYPE_ORDER:
            sub = ts_df[(ts_df["phenotype"] == pheno) & (ts_df["region"] == region)]
            traces = []
            for _, g in sub.groupby("event_key"):
                t = g["time_ms"].to_numpy(dtype=float)
                y = g["kuramoto_R"].to_numpy(dtype=float)
                if t_ref is None:
                    m = (t >= TIME_XLIM_MS[0]) & (t <= TIME_XLIM_MS[1])
                    t_ref = t[m]
                if t_ref is not None and t_ref.size:
                    traces.append(interpolate_to_reference(y, t, t_ref))
            if traces:
                region_data[pheno] = np.nanmean(np.vstack(traces), axis=0)
        if t_ref is not None and all(p in region_data for p in PHENOTYPE_ORDER):
            diff = region_data[PHENOTYPE_ORDER[1]] - region_data[PHENOTYPE_ORDER[0]]
            ax.plot(t_ref, diff, lw=1.4, color=REGION_COLORS[region], label=f"{REGION_DISPLAY[region]} ΔR")
            for tm, di in zip(t_ref, diff):
                source_rows.append({"region": region, "time_ms": float(tm), "delta_R_5xFAD_minus_C57BL6": float(di)})
    ax.axhline(0, color="0.45", lw=0.8)
    ax.axvline(0, color="0.45", ls="--", lw=0.9)
    ax.set_xlim(TIME_XLIM_MS)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Δ Kuramoto R (5xFAD − C57BL6)")
    ax.set_title("Phenotype-difference Kuramoto time course")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(source_rows), Path(str(out_base) + "_source_data"))


def plot_kuramoto_metric_correlation(df_roi: pd.DataFrame, xmetric: str, ymetric: str, xlabel: str, ylabel: str, out_base: Path):
    if df_roi.empty or xmetric not in df_roi.columns or ymetric not in df_roi.columns:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(4.1, 3.5), constrained_layout=True)
    source_rows = []
    for pheno in PHENOTYPE_ORDER:
        sub = df_roi[df_roi["phenotype"] == pheno]
        x = pd.to_numeric(sub[xmetric], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[ymetric], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            ax.scatter(x[m], y[m], s=34, alpha=0.8, color=PHENOTYPE_COLORS[pheno], label=PHENOTYPE_DISPLAY[pheno])
            for xi, yi, roi in zip(x[m], y[m], sub.loc[m, "roi_name"]):
                source_rows.append({"phenotype": pheno, "roi_name": roi, xmetric: float(xi), ymetric: float(yi)})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Kuramoto relationship: {ylabel} vs {xlabel}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(source_rows), Path(str(out_base) + "_source_data"))


def make_enhanced_kuramoto_outputs(kuramoto_enhanced: pd.DataFrame, kuramoto_ts: pd.DataFrame):
    """Create additional Kuramoto comparison panels that are more sensitive than only mean R bars."""
    out = ensure_dir(OUT_DIR / "02_KURAMOTO_ENHANCED")
    if kuramoto_enhanced.empty:
        return

    # Aggregate to dataset-ROI means for more interpretable statistics.
    roi_df = aggregate_event_to_dataset_roi(kuramoto_enhanced, KUR_ENHANCED_METRICS)
    if not roi_df.empty:
        save_df(roi_df, OUT_DIR / "tables" / "roi_kuramoto_dataset_roi_mean_features")

    # Event-level table is still saved for traceability.
    save_df(kuramoto_enhanced, OUT_DIR / "tables" / "roi_kuramoto_event_summary_enhanced")

    # New dynamic metric bars. These are usually better than raw mean R.
    if not roi_df.empty:
        plot_metric_bars(
            roi_df,
            [("R_delta_late_minus_base", "ΔR late-baseline"),
             ("R_late_over_base", "Late/base R"),
             ("R_slope_full", "R slope 0-250 ms"),
             ("R_peak_latency_ms", "R peak latency"),
             ("R_auc_late", "Late R AUC"),
             ("phase_dispersion_delta_late_minus_base", "Δ phase dispersion"),
             ("analytic_amp_delta_late_minus_base", "Δ analytic amp."),
             ("plv_mean", "Mean PLV")],
            out / "baseline_05_KUR_dynamic_metric_bars_datasetROI",
            "Kuramoto dynamic features | dataset-ROI means",
        )

        # Sensitivity/effect-size ranks tell you which Kuramoto metrics are worth showing.
        stat_df = build_kuramoto_sensitivity_table(roi_df, KUR_ENHANCED_METRICS)
        if not stat_df.empty:
            save_df(stat_df, OUT_DIR / "tables" / "roi_kuramoto_enhanced_sensitivity_stats")
            plot_kuramoto_effect_rank(stat_df, out / "baseline_06_KUR_sensitivity_rank_neglog10p")
            plot_kuramoto_effect_size_rank(stat_df, out / "baseline_07_KUR_effect_size_rank_cohend")

        # ROI-resolved views reveal where the effect comes from instead of collapsing CTX/HIPPO too early.
        for metric, label in [
            ("R_delta_late_minus_base", "ΔR late-baseline"),
            ("R_slope_full", "R slope 0-250 ms"),
            ("R_peak_latency_ms", "R peak latency"),
            ("plv_mean", "Mean PLV"),
            ("phase_grad_mag_mean", "Phase-gradient magnitude"),
            ("wave_speed_proxy_mean", "Wave-speed proxy"),
        ]:
            if metric in roi_df.columns:
                plot_kuramoto_roi_heatmap(roi_df, metric, label, out / f"baseline_08_KUR_ROI_heatmap_{sanitize(metric)}")
                plot_kuramoto_roi_lines(roi_df, metric, label, out / f"baseline_09_KUR_ROI_lines_{sanitize(metric)}")

        # Relationship plots: sometimes disease changes coordination mode, not one scalar alone.
        plot_kuramoto_metric_correlation(
            roi_df,
            "R_delta_late_minus_base", "phase_grad_mag_mean",
            "ΔR late-baseline", "Phase-gradient magnitude",
            out / "baseline_10_KUR_correlation_deltaR_vs_phaseGradient",
        )
        plot_kuramoto_metric_correlation(
            roi_df,
            "plv_mean", "wave_speed_proxy_mean",
            "Mean PLV", "Wave-speed proxy",
            out / "baseline_11_KUR_correlation_PLV_vs_waveSpeed",
        )

    # Time-course difference: visually stronger than two separated mean curves.
    if not kuramoto_ts.empty:
        plot_kuramoto_region_difference_timecourse(
            kuramoto_ts,
            out / "baseline_12_KUR_phenotype_difference_timecourse_deltaR",
        )



# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def save_manifest(datasets: List[DatasetRoot], event_meta: pd.DataFrame):
    manifest = {
        "target": "ROI-based avalanche-stream and Kuramoto CTX/HIPPO phenotype comparison",
        "root_dir": str(ROOT_DIR.resolve()),
        "out_dir": str(OUT_DIR.resolve()),
        "phenotype_order": PHENOTYPE_ORDER,
        "phenotype_display": PHENOTYPE_DISPLAY,
        "baseline_aliases": sorted(list(BASELINE_ALIASES)),
        "roi_region_map": ROI_REGION_MAP,
        "region_order": REGION_ORDER,
        "time_xlim_ms": TIME_XLIM_MS,
        "feature_window_ms": FEATURE_WINDOW_MS,
        "avalanche_detect_window_ms": AVALANCHE_DETECT_WINDOW_MS,
        "avalanche_baseline_window_ms": AVALANCHE_BASELINE_WINDOW_MS,
        "avalanche_neg_thresh_mad": AVALANCHE_NEG_THRESH_MAD,
        "avalanche_bin_width_ms_list": AVALANCHE_BIN_WIDTH_MS_LIST,
        "kuramoto_analysis_window_ms": KURAMOTO_ANALYSIS_WINDOW_MS,
        "kuramoto_baseline_window_ms": KURAMOTO_BASELINE_WINDOW_MS,
        "kuramoto_bandpass": KURAMOTO_BANDPASS,
        "kuramoto_bandpass_hz": KURAMOTO_BANDPASS_HZ,
        "save_per_event_plv_matrices": SAVE_PER_EVENT_PLV_MATRICES,
        "datasets": [{"phenotype": d.phenotype, "name": d.name, "path": str(d.path)} for d in datasets],
        "n_events_loaded": int(len(event_meta)),
    }
    ensure_dir(OUT_DIR)
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Cortex vs Hippocampus avalanche + Kuramoto synchrony comparison")
    p.add_argument("--root_dir", default=str(ROOT_DIR), help="Root folder containing per-group event-trace subfolders (default: %(default)s)")
    p.add_argument("--out_dir", default=str(OUT_DIR), help="Output folder (default: %(default)s)")
    p.add_argument("--groups", default=None,
                   help="Comma-separated group labels = subfolder names under root_dir, first = reference "
                        "(e.g. 'CTL,5xFAD'). Overrides PHENOTYPE_ROOTS/PHENOTYPE_ORDER (default: %(default)s)")
    p.add_argument("--condition", default=None,
                   help="Condition folder token (e.g. '4AP') treated as the baseline condition; added to "
                        "BASELINE_ALIASES (default: %(default)s = use BASELINE_ALIASES as-is)")
    return p


def apply_group_override(groups: Optional[str], condition: Optional[str]) -> None:
    """Plumbing only: re-point the group dict / order / labels at CLI-given folder names."""
    global PHENOTYPE_ROOTS, PHENOTYPE_ORDER, PHENOTYPE_DISPLAY, PHENOTYPE_COLORS, BASELINE_ALIASES
    if groups:
        names = [g.strip() for g in groups.split(",") if g.strip()]
        if len(names) < 2:
            raise SystemExit("--groups needs at least two comma-separated group names")
        PHENOTYPE_ROOTS = {g: [Path(g)] for g in names}
        PHENOTYPE_ORDER = list(names)
        default_cols = ["black", "red", "royalblue", "darkorange", "green", "purple"]
        PHENOTYPE_DISPLAY = {g: PHENOTYPE_DISPLAY.get(g, g) for g in names}
        PHENOTYPE_COLORS = {g: PHENOTYPE_COLORS.get(g, default_cols[i % len(default_cols)]) for i, g in enumerate(names)}
    if condition:
        BASELINE_ALIASES = set(BASELINE_ALIASES) | {re.sub(r"[^a-z0-9]+", "", condition.lower())}


def main():
    global ROOT_DIR, OUT_DIR
    args = build_argparser().parse_args()
    ROOT_DIR = Path(args.root_dir)
    OUT_DIR = Path(args.out_dir)
    apply_group_override(args.groups, args.condition)

    set_style()
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "tables")
    ensure_dir(OUT_DIR / "01_AVALANCHE")
    ensure_dir(OUT_DIR / "02_KURAMOTO")
    ensure_dir(OUT_DIR / "03_COMBINED")

    datasets = discover_dataset_roots()
    print("[INFO] Phenotype roots:")
    for ds in datasets:
        print(f"  {ds.phenotype:8s} | {ds.path}")

    events, event_meta = load_roi_events(datasets)
    save_manifest(datasets, event_meta)
    save_df(event_meta, OUT_DIR / "tables" / "loaded_roi_event_manifest")

    avalanche_summary_rows = []
    avalanche_event_points_all = []
    avalanche_all = []

    kuramoto_summary_rows = []
    kuramoto_ts_all = []

    for ev in events:
        ev_out = ensure_dir(OUT_DIR / "per_event" / sanitize(ev["phenotype"]) / sanitize(ev["roi_name"]) / sanitize(ev["event_key"]))

        if RUN_AVALANCHE:
            try:
                rows, ev_points, aval = avalanche_event_analysis(ev, ev_out)
                avalanche_summary_rows.extend(rows)
                if ev_points is not None and not ev_points.empty:
                    avalanche_event_points_all.append(ev_points)
                if aval is not None and not aval.empty:
                    avalanche_all.append(aval)
            except Exception as e:
                print(f"[WARN] Avalanche failed {ev['event_key']}: {e}")

        if RUN_KURAMOTO:
            try:
                row, ts = kuramoto_event_analysis(ev, ev_out)
                if row:
                    kuramoto_summary_rows.append(row)
                if ts is not None and not ts.empty:
                    kuramoto_ts_all.append(ts)
            except Exception as e:
                print(f"[WARN] Kuramoto failed {ev['event_key']}: {e}")

    # Save full tables.
    avalanche_summary = pd.DataFrame(avalanche_summary_rows)
    avalanche_event_points = pd.concat(avalanche_event_points_all, ignore_index=True) if avalanche_event_points_all else pd.DataFrame()
    avalanche_df = pd.concat(avalanche_all, ignore_index=True) if avalanche_all else pd.DataFrame()

    kuramoto_summary = pd.DataFrame(kuramoto_summary_rows)
    kuramoto_ts = pd.concat(kuramoto_ts_all, ignore_index=True) if kuramoto_ts_all else pd.DataFrame()

    if not avalanche_summary.empty:
        save_df(avalanche_summary, OUT_DIR / "tables" / "roi_avalanche_event_summary")
    if not avalanche_event_points.empty:
        save_df(avalanche_event_points, OUT_DIR / "tables" / "roi_avalanche_detected_lfp_events")
    if not avalanche_df.empty:
        save_df(avalanche_df, OUT_DIR / "tables" / "roi_avalanche_all_avalanches")

    if not kuramoto_summary.empty:
        save_df(kuramoto_summary, OUT_DIR / "tables" / "roi_kuramoto_event_summary")
    if not kuramoto_ts.empty:
        save_df(kuramoto_ts, OUT_DIR / "tables" / "roi_kuramoto_timeseries_long")

    # Avalanche plots.
    if not avalanche_summary.empty:
        aval_plot_df = avalanche_summary[np.isclose(pd.to_numeric(avalanche_summary["activity_bin_width_ms"], errors="coerce"), AVALANCHE_BIN_WIDTH_MS_LIST[0])]
        aval_metrics = [
            ("n_detected_lfp_events", "Detected nLFP events"),
            ("n_avalanches", "Avalanche count"),
            ("active_bin_fraction", "Active-bin fraction"),
            ("avalanche_rate_per_s", "Avalanche rate (/s)"),
            ("avalanche_size_mean", "Mean avalanche size"),
            ("avalanche_size_max", "Max avalanche size"),
            ("avalanche_duration_mean_bins", "Mean duration (bins)"),
            ("avalanche_size_alpha", "Size power-law alpha"),
            ("avalanche_duration_alpha", "Duration alpha"),
            ("avalanche_size_duration_loglog_slope", "Size-duration slope"),
        ]
        plot_metric_bars(
            aval_plot_df,
            aval_metrics,
            OUT_DIR / "01_AVALANCHE" / "baseline_01_ROI_avalanche_metric_bars",
            "ROI avalanche metrics | CTX vs HIPPO",
        )

    if not avalanche_df.empty:
        plot_avalanche_ccdf_by_group(
            avalanche_df,
            OUT_DIR / "01_AVALANCHE" / "baseline_02_ROI_avalanche_size_CCDF",
            bin_width_ms=AVALANCHE_BIN_WIDTH_MS_LIST[0],
        )

    # Kuramoto plots.
    if not kuramoto_summary.empty:
        kur_metrics = [
            ("kuramoto_R_mean", "Mean Kuramoto R"),
            ("kuramoto_R_peak", "Peak Kuramoto R"),
            ("kuramoto_R_auc", "Kuramoto R AUC"),
            ("kuramoto_R_evoked_over_baseline", "Evoked/base R"),
            ("phase_dispersion_mean", "Phase dispersion"),
            ("plv_mean", "Mean PLV"),
            ("plv_std", "PLV SD"),
            ("phase_grad_mag_mean", "Phase-gradient mag."),
            ("phase_singularity_count_mean", "Phase singularities"),
            ("wave_speed_proxy_mean", "Wave-speed proxy"),
        ]
        plot_metric_bars(
            kuramoto_summary,
            kur_metrics,
            OUT_DIR / "02_KURAMOTO" / "baseline_03_ROI_kuramoto_metric_bars",
            "ROI Kuramoto synchrony/phase metrics | CTX vs HIPPO",
        )

    if not kuramoto_ts.empty:
        plot_kuramoto_R_lines(
            kuramoto_ts,
            OUT_DIR / "02_KURAMOTO" / "baseline_04_ROI_kuramoto_R_timecourse",
        )

    # Enhanced Kuramoto comparisons / visualizations.
    # Keep all original figures above; add dynamic features that are usually more sensitive
    # than only mean R, plus ROI-resolved heatmaps and effect-size ranking.
    if not kuramoto_summary.empty:
        kuramoto_enhanced = add_kuramoto_timeseries_features(kuramoto_summary, kuramoto_ts)
        make_enhanced_kuramoto_outputs(kuramoto_enhanced, kuramoto_ts)

    # Combined merged table for downstream statistics / ML.
    if not avalanche_summary.empty or not kuramoto_summary.empty:
        merge_keys = [
            "phenotype", "dataset", "roi_name", "region", "drug",
            "event_index", "event_ms", "event_key", "n_electrodes"
        ]
        if not avalanche_summary.empty:
            aval_first = avalanche_summary[np.isclose(pd.to_numeric(avalanche_summary["activity_bin_width_ms"], errors="coerce"), AVALANCHE_BIN_WIDTH_MS_LIST[0])].copy()
        else:
            aval_first = pd.DataFrame(columns=merge_keys)
        if kuramoto_summary.empty:
            kuramoto_summary = pd.DataFrame(columns=merge_keys)
        combined = pd.merge(aval_first, kuramoto_summary, on=merge_keys, how="outer", suffixes=("_aval", "_kur"))
        save_df(combined, OUT_DIR / "tables" / "roi_avalanche_kuramoto_combined_features")

    print("\nDONE")
    print(f"Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
