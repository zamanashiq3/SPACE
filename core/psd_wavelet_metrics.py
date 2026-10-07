# -*- coding: utf-8 -*-
"""
psd_wavelet_metrics.py -- SPACE framework: LFP/CSD event kymographs, PSD/wavelet bands, 1/f exponent.

Manuscript figure(s): Figure 2 (LFP amplitude/energy/complexity, CSD time-integrated,
balance index), Figure 4 (PSD bands, wavelet power, 1/f exponent), Figure 6
(pharmacology PSD/band power), Supplementary Figure 3 (source/sink balance),
Supplementary Figure 5 (pharmacology LFP amplitude/complexity).

What it computes:
    Per-event LFP/CSD kymographs, amplitude/energy/AUC/complexity-D features, a CSD
    sink/source balance index, Morlet-wavelet band power, Welch-PSD band power/fractions,
    and a specparam/FOOOF aperiodic (1/f) exponent fit (falling back to a transparent
    log-log linear fit when neither package is installed). This is the final (v3) version
    of this analysis, which adds the 1/f exponent fit missing from an earlier v2 draft.

    NOTE: this script's group-dict configuration (``PHENOTYPE_ROOTS``) uses generic
    "WT"/"5xFAD" keys as the pattern for defining comparison groups -- edit the paths
    (and, if needed, the phenotype keys/labels) to match your own genotype folder names.

Inputs / outputs (see data/README.md for the full schema):
    Input:  per-ROI, per-condition event-trace folders under
            ``data/event_traces/<group>/<region>/`` containing ``lfp_event_traces.npy``,
            ``csd_event_traces.npy``, and ``event_time_axis_ms.npy`` (one folder per group
            as listed in ``PHENOTYPE_ROOTS``).
    Output: figures and tables (LFP/CSD kymographs, PSD/wavelet band power, 1/f exponent,
            source/sink balance) under ``data/results/psd_wavelet/``, plus a
            ``manifest.json`` describing the run.

Baseline-only phenotype comparison for ROI event folders.

TARGET
------
Compare only BASELINE / base condition between:
    APP^NL
    APP^NL-G-F

Pool ROIs into two anatomical groups:
    CTX   = M1, M2, S1/SSP
    HIPPO = CA1, CA3, DG

Input is the already-generated ROI event/archetype folder structure:
    appnl_event_dbscan_archetype/
        M1/base/event_000_...ms/lfp_event_traces.npy
        M1/base/event_000_...ms/csd_event_traces.npy
        M1/base/event_000_...ms/event_time_axis_ms.npy
        ...
    appnl-gf_event_dbscan_archetype/
        CA1/base/event_.../

This script intentionally ignores 4AP and Bic.

Main outputs:
    BASELINE_APPNL_APPNLGF_CTX_HIPPO_LFP_CSD_WAVELET/
        tables/
        01_LFP/
        02_CSD/
        03_SOURCE_SINK/
        04_WAVELET/
        05_FREQUENCY_PSD/
        manifest.json

Figures follow the style/layout logic of the CTX/HIPPO master script:
    - two-genotype panel plots with shared y-axis
    - normalized-together plots:
          APP^NL     = black
          APP^NL-G-F = red
          CTX        = solid
          HIPPO      = dotted
    - CTX/HIPPO kymograph grids
    - source/sink and balance plots
    - wavelet heatmaps and summary bars
    - LFP complexity D and LFP firing/event-pattern metrics
    - Welch/FFT PSD frequency-band analysis
    - phenotype-difference traces and kymographs: APP^NL-G-F minus APP^NL

Notes:
    LFP units are assumed to be uV if the upstream extraction saved uV.
    CSD units are assumed to be uV/um^2 if the upstream extraction saved CSD in uV/um^2.
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
    from scipy.signal import find_peaks, welch
except Exception:
    find_peaks = None
    welch = None

# Optional standard aperiodic/1-f fitting packages.
# The script still runs without these packages by falling back to a transparent
# log-log linear PSD fit: log10(PSD) = offset - exponent*log10(f).
try:
    from specparam import SpectralModel
except Exception:
    SpectralModel = None

try:
    from fooof import FOOOF
except Exception:
    FOOOF = None

# =============================================================================
# CONFIG
# =============================================================================

# --- EDIT THIS --- (or override via --root_dir / --out_dir CLI flags, see build_argparser)
ROOT_DIR = Path("data/event_traces")
OUT_DIR = Path("data/results/psd_wavelet")

# --- EDIT THIS --- group-folder dict: keys are your genotype/group labels, values are the
# per-group event-trace root folder(s) to search (relative to ROOT_DIR unless absolute).
# Point these at your own data/event_traces/<group>/ folders.
PHENOTYPE_ROOTS: Dict[str, List[Path]] = {
    "WT": [Path("WT")],
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

BASELINE_ALIASES = {"base", "4ap", "basal", "control","4AP"}

ROI_ORDER = ["M1", "M2", "SSP", "S1", "CA1", "CA3", "DG"]
ROI_CANON = {
    "M1": "M1",
    "M2": "M2",
    "S1": "SSP",
    "SSP": "SSP",
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
KYMO_TIME_XLIM_MS = (-50, 250)
FEATURE_WINDOW_MS = (0, 250)
BASELINE_WINDOW_MS = (-50, 0)
SLOPE_WINDOW_MS = (0, 80)

LFP_TRACE_YLABEL = "LFP amplitude (uV)"
CSD_TRACE_YLABEL = "Mean |CSD| (uV/um^2)"
CSD_SIGNED_YLABEL = "Mean CSD (uV/um^2)"
LFP_KYMO_CMAP = "bwr"
CSD_KYMO_CMAP = "PiYG"

BAR_WIDTH = 0.32
POINT_SIZE = 20
POINT_JITTER = 0.055
EPS = 1e-12

# Wavelet settings.
RUN_WAVELET = True
WAVELET_FREQS_HZ = [2, 4, 8, 12, 20, 30, 45, 70, 100, 150, 200]
WAVELET_NAME = "cmor1.5-1.0"
WAVELET_ANALYSIS_WINDOW_MS = (0, 250)
WAVELET_BASELINE_WINDOW_MS = (-50, 0)

# LFP complexity / firing-pattern settings.
# Complexity D = (Tr C)^2 / Tr(C^2), computed from electrode-by-time covariance.
LFP_COMPLEXITY_SMOOTH_WIDTH = 10
LFP_COMPLEXITY_TIME_WINDOW_MS = (0, 250)
FIRING_ACTIVE_THRESHOLD_MAD = 4.0
FIRING_PEAK_PROMINENCE_MAD = 3.0
FIRING_MIN_DISTANCE_MS = 20.0

# Frequency / PSD settings.
RUN_FREQUENCY_ANALYSIS = True
PSD_ANALYSIS_WINDOW_MS = (0, 250)
PSD_NPERSEG_MAX = 256
PSD_FREQ_MAX_HZ = 250.0

# Additional spectral benchmark metrics.
# These are added without removing the original PSD/wavelet outputs.
PSD_BANDS = {
    "delta_1_4": (1.0, 4.0),
    "theta_4_8": (4.0, 8.0),
    "alpha_8_12": (8.0, 12.0),
    "beta_12_30": (12.0, 30.0),
    "gamma_30_80": (30.0, 80.0),
    "high_80_200": (80.0, 200.0),
}
THETA_GAMMA_RATIO_BANDS = ("theta_4_8", "gamma_30_80")
APERIODIC_FIT_RANGE_HZ = (2.0, 200.0)
APERIODIC_IGNORE_BANDS_HZ = [(48.0, 52.0), (58.0, 62.0), (98.0, 102.0), (118.0, 122.0), (178.0, 182.0)]
USE_SPECPARAM_IF_AVAILABLE = True

# CSD source/sink convention:
# source = positive CSD; sink = negative CSD magnitude.
CSD_SOURCE_IS_POSITIVE = True

# =============================================================================
# BASIC HELPERS
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
            fig.tight_layout(pad=1.25)
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
    if s in {"S1", "SSP", "SOMATOSENSORY"}:
        return "SSP"
    if s in {"M1", "M2", "CA1", "CA3", "DG"}:
        return s
    return str(x)


def roi_region(roi: Any) -> str:
    return ROI_REGION_MAP.get(canon_roi(roi), "OTHER")


def is_baseline_dir_name(name: str) -> bool:
    s = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    return s in BASELINE_ALIASES or "base" == s or "baseline" == s


def time_mask(t, window):
    t = np.asarray(t, dtype=float)
    m = (t >= window[0]) & (t <= window[1])
    return m if np.any(m) else np.ones(t.size, dtype=bool)


def dt_from_time_axis(t):
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return 1.0
    d = np.diff(t)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0
    return float(np.nanmedian(np.abs(d)))


def robust_vlim(arrays, pct=99.0, fallback=1.0):
    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]
    vals = []
    for a in arrays:
        if a is None:
            continue
        x = np.asarray(a, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    if not vals:
        return float(fallback)
    x = np.concatenate(vals)
    v = np.nanpercentile(np.abs(x), pct)
    if not np.isfinite(v) or v <= 0:
        v = np.nanmax(np.abs(x))
    if not np.isfinite(v) or v <= 0:
        v = fallback
    return float(v)


def mean_sem(Y):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    m = np.nanmean(Y, axis=0)
    n = np.sum(np.isfinite(Y), axis=0)
    sd = np.nanstd(Y, axis=0, ddof=1)
    sem = np.divide(sd, np.sqrt(n), out=np.zeros_like(sd), where=n > 1)
    return m, sem, n


def normalize_trace(y):
    y = np.asarray(y, dtype=float)
    mx = np.nanmax(np.abs(y))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(y)
    return y / mx


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


def interpolate_to_reference(y, t, t_ref):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if np.sum(ok) < 2:
        return np.full_like(t_ref, np.nan, dtype=float)
    return np.interp(t_ref, t[ok], y[ok])


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
    ax.text((x1+x2)/2, y+h*1.1, f"{p_to_label(p)}\np={p:.2g}" if np.isfinite(p) else "n.s.",
            ha="center", va="bottom", fontsize=6)


def sync_y_limits(axes, pad_frac=0.08):
    axes = np.asarray(axes).ravel()
    lows, highs = [], []
    for ax in axes:
        if ax is None or not ax.has_data():
            continue
        y0, y1 = ax.get_ylim()
        if np.isfinite(y0) and np.isfinite(y1):
            lows.append(y0)
            highs.append(y1)
    if not lows:
        return
    lo = float(np.nanmin(lows))
    hi = float(np.nanmax(highs))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_frac
    for ax in axes:
        if ax is not None and ax.has_data():
            ax.set_ylim(lo - pad, hi + pad)

# =============================================================================
# DISCOVERY
# =============================================================================

class DatasetRoot:
    def __init__(self, phenotype: str, path: Path):
        self.phenotype = phenotype
        self.path = Path(path)
        self.name = self.path.name

    def __repr__(self):
        return f"DatasetRoot(phenotype={self.phenotype}, path={self.path})"


def classify_phenotype_from_name(name: str) -> Optional[str]:
    n = name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", n)

    # APP^NL-G-F / APPNL-GF / appnl-gf
    if ("app" in compact and ("gf" in compact or "gfp" in compact)) or "appnl-g" in n or "appnl_g" in n:
        return "APPNL_GF"

    # APP^NL only
    if "appnl" in compact or ("app" in compact and "nl" in compact):
        return "APPNL"

    return None


def looks_like_event_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    for roi in ROI_ORDER:
        for rname in {roi, canon_roi(roi)}:
            q = p / rname
            if q.exists() and q.is_dir():
                for child in q.iterdir():
                    if child.is_dir() and is_baseline_dir_name(child.name):
                        if list(child.glob("event_*_*ms")):
                            return True
    return False


def discover_dataset_roots() -> List[DatasetRoot]:
    roots: List[DatasetRoot] = []

    manual = any(len(v) > 0 for v in PHENOTYPE_ROOTS.values())
    if manual:
        for pheno, paths in PHENOTYPE_ROOTS.items():
            for p in paths:
                p = ROOT_DIR / p if not Path(p).is_absolute() else Path(p)
                if p.exists():
                    roots.append(DatasetRoot(pheno, p))
                else:
                    print(f"[WARN] missing manual root: {pheno} | {p}")
        if roots:
            return roots

    for child in sorted(ROOT_DIR.iterdir()):
        if not child.is_dir():
            continue
        pheno = classify_phenotype_from_name(child.name)
        if pheno is None:
            continue
        if looks_like_event_root(child):
            roots.append(DatasetRoot(pheno, child))

    if not roots:
        raise RuntimeError(
            "No APPNL / APPNL-GF baseline ROI event roots found.\n"
            "Expected folders like appnl_event_dbscan_archetype and appnl-gf_event_dbscan_archetype.\n"
            "If your names are different, edit PHENOTYPE_ROOTS at the top of this script."
        )

    # Make APPNL first, APPNL_GF second.
    roots = sorted(roots, key=lambda d: (PHENOTYPE_ORDER.index(d.phenotype), d.name))
    return roots


def find_roi_dir(root: Path, roi: str) -> Optional[Path]:
    roi = canon_roi(roi)
    candidates = [roi]
    if roi == "SSP":
        candidates += ["S1", "ssp", "s1"]
    for c in candidates:
        q = root / c
        if q.exists() and q.is_dir():
            return q
    # case-insensitive fallback
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

# =============================================================================
# LOAD EVENTS
# =============================================================================

def load_baseline_events(datasets: List[DatasetRoot]) -> Tuple[List[dict], pd.DataFrame]:
    events = []
    rows = []

    for ds in datasets:
        for roi_name in ["M1", "M2", "SSP", "CA1", "CA3", "DG"]:
            rdir = find_roi_dir(ds.path, roi_name)
            if rdir is None:
                print(f"[WARN] missing ROI dir: {ds.name} | {roi_name}")
                continue
            bdir = find_baseline_dir(rdir)
            if bdir is None:
                print(f"[WARN] missing baseline dir: {ds.name} | {roi_name}")
                continue

            edirs = sorted(bdir.glob("event_*_*ms"), key=lambda p: parse_event_folder_name(p)[0] if parse_event_folder_name(p)[0] is not None else 999999)
            print(f"[LOAD] {ds.phenotype:8s} | {ds.name} | {roi_name:4s} | base | {len(edirs)} events")

            for ev_dir in edirs:
                lp = ev_dir / "lfp_event_traces.npy"
                cp = ev_dir / "csd_event_traces.npy"
                tp = ev_dir / "event_time_axis_ms.npy"
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
        raise RuntimeError("No baseline event traces were loaded.")
    return events, pd.DataFrame(rows)


def choose_reference_time(events: List[dict]) -> np.ndarray:
    # Most common length wins.
    lengths = {}
    for ev in events:
        lengths.setdefault(len(ev["time_ms"]), 0)
        lengths[len(ev["time_ms"])] += 1
    n_ref = sorted(lengths.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    for ev in events:
        if len(ev["time_ms"]) == n_ref:
            return np.asarray(ev["time_ms"], dtype=float)
    return np.asarray(events[0]["time_ms"], dtype=float)


def event_mean_signal(ev: dict, kind: str, signed_csd: bool = False) -> Optional[np.ndarray]:
    if kind == "lfp":
        arr = ev["lfp"]
        return np.nanmean(arr, axis=0)
    if kind == "csd":
        arr = ev.get("csd")
        if arr is None:
            return None
        if signed_csd:
            return np.nanmean(arr, axis=0)
        return np.nanmean(np.abs(arr), axis=0)
    raise ValueError(kind)


def event_kymo(ev: dict, kind: str, normalized: bool = False) -> Optional[np.ndarray]:
    arr = ev["lfp"] if kind == "lfp" else ev.get("csd")
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if normalized:
        mx = np.nanmax(np.abs(arr))
        if np.isfinite(mx) and mx > 0:
            arr = arr / mx
    return arr

# =============================================================================
# TABLES / FEATURES
# =============================================================================

def compute_lfp_features(y: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    bm = time_mask(t, BASELINE_WINDOW_MS)
    yy = y - np.nanmedian(y[bm])
    m = time_mask(t, FEATURE_WINDOW_MS)
    x = yy[m]
    tx = t[m]
    if x.size < 2:
        return {}
    pos = float(np.nanmax(x))
    neg = float(np.nanmin(x))
    abs_peak = float(np.nanmax(np.abs(x)))
    p2p = float(pos - neg)
    energy = float(np.trapezoid(x * x, tx))
    rms = float(np.sqrt(np.nanmean(x * x)))
    area_abs = float(np.trapezoid(np.abs(x), tx))
    pk = int(np.nanargmax(np.abs(x)))
    sm = time_mask(tx, SLOPE_WINDOW_MS)
    if np.sum(sm) >= 2:
        dx = np.gradient(x[sm], tx[sm])
    else:
        dx = np.gradient(x, tx)
    return dict(
        lfp_pos_peak_uV=pos,
        lfp_neg_peak_uV=neg,
        lfp_abs_peak_uV=abs_peak,
        lfp_p2p_uV=p2p,
        lfp_energy_uV2_ms=energy,
        lfp_rms_uV=rms,
        lfp_area_abs_uV_ms=area_abs,
        lfp_latency_abs_peak_ms=float(tx[pk]),
        lfp_max_abs_slope_uV_per_ms=float(np.nanmax(np.abs(dx))),
        lfp_norm_slope_pct=float(100.0 * np.nanmax(np.abs(dx)) / (abs_peak + EPS)),
    )


def compute_csd_features(C: Optional[np.ndarray], t: np.ndarray) -> Dict[str, float]:
    if C is None:
        return {}
    C = np.asarray(C, dtype=float)
    m = time_mask(t, FEATURE_WINDOW_MS)
    X = C[:, m]
    tx = t[m]
    if X.size == 0:
        return {}
    absX = np.abs(X)
    mean_abs = np.nanmean(absX, axis=0)

    if CSD_SOURCE_IS_POSITIVE:
        src = np.where(X > 0, X, 0.0)
        snk = np.where(X < 0, -X, 0.0)
    else:
        src = np.where(X < 0, -X, 0.0)
        snk = np.where(X > 0, X, 0.0)

    source_ts = np.nanmean(src, axis=0)
    sink_ts = np.nanmean(snk, axis=0)
    source_auc = float(np.trapezoid(source_ts, tx))
    sink_auc = float(np.trapezoid(sink_ts, tx))
    bal = (source_ts - sink_ts) / (source_ts + sink_ts + EPS)

    return dict(
        csd_abs_peak_uV_per_um2=float(np.nanmax(absX)),
        csd_mean_abs_peak_uV_per_um2=float(np.nanmax(mean_abs)),
        csd_mean_abs_auc_uV_per_um2_ms=float(np.trapezoid(mean_abs, tx)),
        csd_source_auc=float(source_auc),
        csd_sink_auc=float(sink_auc),
        csd_source_sink_ratio=float(source_auc / (sink_auc + EPS)),
        csd_balance_index_mean=float(np.nanmean(bal)),
        csd_balance_index_peak_abs=float(np.nanmax(np.abs(bal))),
    )



def box_smooth_1d(y, width=10):
    y = fill_nan_1d(y)
    width = int(width)
    if width <= 1:
        return y
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(y, kernel, mode="same")


def compute_lfp_complexity_D(X: np.ndarray) -> float:
    """
    LFP spatial complexity / dimensionality:
        D = (Tr C)^2 / Tr(C^2)
    X is electrodes x time.
    Higher D means activity is distributed across more independent spatial modes.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return np.nan
    C = np.cov(X)
    if C.ndim != 2:
        return np.nan
    trC = float(np.trace(C))
    trC2 = float(np.trace(C @ C))
    if not np.isfinite(trC) or not np.isfinite(trC2) or trC2 <= 0:
        return np.nan
    return float((trC ** 2) / trC2)


def compute_lfp_complexity_features(L: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    L = np.asarray(L, dtype=float)
    t = np.asarray(t, dtype=float)
    if L.ndim != 2 or L.shape[0] < 3:
        return {}
    bm = time_mask(t, BASELINE_WINDOW_MS)
    X = L - np.nanmedian(L[:, bm], axis=1, keepdims=True)
    wm = time_mask(t, LFP_COMPLEXITY_TIME_WINDOW_MS)
    X = X[:, wm]
    if X.shape[1] < 2:
        return {}
    Xs = np.vstack([box_smooth_1d(X[ch], width=LFP_COMPLEXITY_SMOOTH_WIDTH) for ch in range(X.shape[0])])
    D = compute_lfp_complexity_D(Xs)

    jk = []
    for omit in range(Xs.shape[0]):
        keep = [i for i in range(Xs.shape[0]) if i != omit]
        if len(keep) >= 2:
            jk.append(compute_lfp_complexity_D(Xs[keep, :]))
    jk = np.asarray(jk, dtype=float)
    jk = jk[np.isfinite(jk)]
    return dict(
        lfp_complexity_D=float(D) if np.isfinite(D) else np.nan,
        lfp_complexity_D_jackknife_mean=float(np.nanmean(jk)) if jk.size else np.nan,
        lfp_complexity_D_jackknife_sem=float(np.nanstd(jk, ddof=1) / np.sqrt(jk.size)) if jk.size > 1 else np.nan,
        lfp_complexity_n_modes_possible=int(Xs.shape[0]),
        lfp_complexity_smooth_width=int(LFP_COMPLEXITY_SMOOTH_WIDTH),
    )


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def compute_lfp_firing_pattern_features(L: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """
    LFP event-pattern features from threshold crossings/peaks.
    This is not spike sorting; it quantifies event-like LFP activation within ROI electrodes.
    """
    L = np.asarray(L, dtype=float)
    t = np.asarray(t, dtype=float)
    if L.ndim != 2 or L.shape[1] < 3:
        return {}

    bm = time_mask(t, BASELINE_WINDOW_MS)
    wm = time_mask(t, FEATURE_WINDOW_MS)
    tw = t[wm]
    if tw.size < 3:
        return {}
    X = L - np.nanmedian(L[:, bm], axis=1, keepdims=True)
    Xw = X[:, wm]
    base_abs = np.abs(X[:, bm]).ravel() if np.any(bm) else np.abs(X).ravel()
    mad = robust_mad(base_abs)
    base_med = np.nanmedian(base_abs)
    if not np.isfinite(mad) or mad <= 0:
        mad = robust_mad(np.abs(Xw).ravel())
    threshold = float(base_med + FIRING_ACTIVE_THRESHOLD_MAD * mad) if np.isfinite(mad) else float(np.nanpercentile(np.abs(Xw), 95))

    dt = dt_from_time_axis(tw)
    min_dist = max(1, int(round(FIRING_MIN_DISTANCE_MS / max(dt, EPS))))
    duration_s = max((tw[-1] - tw[0]) / 1000.0, EPS)

    active = []
    first_lat = []
    peak_counts = []
    peak_amp = []
    for ch in range(Xw.shape[0]):
        y = np.abs(fill_nan_1d(Xw[ch]))
        active_mask = y >= threshold
        active.append(bool(np.any(active_mask)))
        if np.any(active_mask):
            first_lat.append(float(tw[np.argmax(active_mask)]))
        prom = FIRING_PEAK_PROMINENCE_MAD * mad if np.isfinite(mad) and mad > 0 else None
        if find_peaks is not None:
            try:
                peaks, _ = find_peaks(y, height=threshold, prominence=prom, distance=min_dist)
            except Exception:
                peaks = np.array([], dtype=int)
        else:
            peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] >= threshold))[0] + 1
        peak_counts.append(int(len(peaks)))
        peak_amp.append(float(np.nanmax(y)) if y.size else np.nan)

    active = np.asarray(active, dtype=bool)
    first_lat = np.asarray(first_lat, dtype=float)
    peak_counts = np.asarray(peak_counts, dtype=float)

    corr_mean = np.nan
    if Xw.shape[0] >= 2 and Xw.shape[1] >= 3:
        try:
            C = np.corrcoef(Xw)
            iu = np.triu_indices_from(C, 1)
            vals = C[iu]
            vals = vals[np.isfinite(vals)]
            corr_mean = float(np.nanmean(vals)) if vals.size else np.nan
        except Exception:
            corr_mean = np.nan

    return dict(
        lfp_firing_threshold_uV=threshold,
        lfp_active_electrode_fraction=float(np.mean(active)) if active.size else np.nan,
        lfp_active_electrode_count=int(np.sum(active)),
        lfp_peak_count_sum=int(np.nansum(peak_counts)),
        lfp_peak_count_mean_per_electrode=float(np.nanmean(peak_counts)) if peak_counts.size else np.nan,
        lfp_event_rate_Hz_per_electrode=float(np.nanmean(peak_counts) / duration_s) if peak_counts.size else np.nan,
        lfp_onset_latency_mean_ms=float(np.nanmean(first_lat)) if first_lat.size else np.nan,
        lfp_onset_latency_std_ms=float(np.nanstd(first_lat)) if first_lat.size else np.nan,
        lfp_onset_latency_range_ms=float(np.nanmax(first_lat) - np.nanmin(first_lat)) if first_lat.size else np.nan,
        lfp_interelectrode_corr_mean=corr_mean,
    )

def build_event_feature_table(events: List[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        t = np.asarray(ev["time_ms"], dtype=float)
        y = event_mean_signal(ev, "lfp")
        row = {k: ev[k] for k in [
            "phenotype", "dataset", "roi_name", "region", "drug",
            "event_index", "event_ms", "event_dir", "event_key", "n_electrodes"
        ]}
        row.update(compute_lfp_features(y, t))
        row.update(compute_lfp_complexity_features(ev.get("lfp"), t))
        row.update(compute_lfp_firing_pattern_features(ev.get("lfp"), t))
        row.update(compute_csd_features(ev.get("csd"), t))
        rows.append(row)
    return pd.DataFrame(rows)

# =============================================================================
# GROUPING
# =============================================================================

def records_for_signal(events: List[dict], kind: str, region: str, normalized: bool = False, signed_csd: bool = False) -> List[dict]:
    recs = []
    for ev in events:
        if ev["region"] != region:
            continue
        y = event_mean_signal(ev, kind, signed_csd=signed_csd)
        if y is None:
            continue
        if normalized:
            y = normalize_trace(y)
        recs.append({
            "phenotype": ev["phenotype"],
            "dataset": ev["dataset"],
            "roi_name": ev["roi_name"],
            "region": ev["region"],
            "event_index": ev["event_index"],
            "time_ms": np.asarray(ev["time_ms"], dtype=float),
            "trace": np.asarray(y, dtype=float),
        })
    return recs


def group_records_by_phenotype(records: List[dict], xlim=None):
    if not records:
        return None, {}, pd.DataFrame()
    t0 = np.asarray(records[0]["time_ms"], dtype=float)
    if xlim is not None:
        m = (t0 >= xlim[0]) & (t0 <= xlim[1])
        t_ref = t0[m]
    else:
        t_ref = t0
    grouped = {p: [] for p in PHENOTYPE_ORDER}
    src_rows = []
    for r in records:
        yy = interpolate_to_reference(r["trace"], r["time_ms"], t_ref)
        grouped.setdefault(r["phenotype"], []).append(yy)
        for i, tm in enumerate(t_ref):
            src_rows.append({
                "phenotype": r["phenotype"],
                "dataset": r["dataset"],
                "roi_name": r["roi_name"],
                "region": r["region"],
                "event_index": r["event_index"],
                "time_ms": float(tm),
                "value": float(yy[i]) if i < len(yy) and np.isfinite(yy[i]) else np.nan,
            })
    return t_ref, grouped, pd.DataFrame(src_rows)


def average_kymographs(events: List[dict], kind: str, region: str, normalized: bool = False):
    out = {}
    for phenotype in PHENOTYPE_ORDER:
        selected = [ev for ev in events if ev["phenotype"] == phenotype and ev["region"] == region]
        if not selected:
            continue
        # Use first valid kymograph as reference.
        ref_ev = None
        for ev in selected:
            arr = event_kymo(ev, kind, normalized=normalized)
            if arr is not None:
                ref_ev = ev
                break
        if ref_ev is None:
            continue
        t_ref = np.asarray(ref_ev["time_ms"], dtype=float)
        arr_ref = event_kymo(ref_ev, kind, normalized=normalized)
        nch = arr_ref.shape[0]
        stack = []
        for ev in selected:
            arr = event_kymo(ev, kind, normalized=normalized)
            if arr is None or arr.shape[0] != nch:
                continue
            interp_arr = np.zeros((nch, len(t_ref)), dtype=float)
            for ch in range(nch):
                interp_arr[ch] = interpolate_to_reference(arr[ch], ev["time_ms"], t_ref)
            stack.append(interp_arr)
        if stack:
            out[phenotype] = {
                "arr": np.nanmean(np.stack(stack, axis=0), axis=0),
                "t": t_ref,
                "n_events": len(stack),
            }
    return out

# =============================================================================
# PLOTS
# =============================================================================

def plot_overlay_two_region_by_phenotype(t_ref, grouped_by_region, ylabel, title, out_base, source_df=None):
    set_style()
    fig, axes = plt.subplots(1, len(PHENOTYPE_ORDER), figsize=(7.4, 3.15), squeeze=False,
                             sharey=True, constrained_layout=True)
    axes = axes[0]
    for ai, phenotype in enumerate(PHENOTYPE_ORDER):
        ax = axes[ai]
        for region in REGION_ORDER:
            grouped = grouped_by_region.get(region, {})
            if phenotype not in grouped or len(grouped[phenotype]) == 0:
                continue
            Y = np.vstack(grouped[phenotype])
            m, s, _ = mean_sem(Y)
            color = REGION_COLORS[region]
            for y in Y:
                ax.plot(t_ref, y, color=color, alpha=0.10, lw=0.55)
            ax.fill_between(t_ref, m - s, m + s, color=color, alpha=0.18, linewidth=0)
            ax.plot(t_ref, m, color=color, lw=1.1, label=REGION_DISPLAY[region])
        ax.axvline(0, color="0.4", ls="--", lw=0.9)
        ax.set_xlim(TIME_XLIM_MS)
        ax.set_title(PHENOTYPE_DISPLAY.get(phenotype, phenotype), fontsize=10, pad=6)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(ylabel if ai == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
    sync_y_limits(axes)
    fig.suptitle(title, fontsize=11)
    save_fig(fig, out_base)
    if source_df is not None and not source_df.empty:
        save_df(source_df, Path(str(out_base) + "_source_data"))


def plot_normalized_together_line(t_ref, grouped_by_region, ylabel, title, out_base, source_df=None):
    if t_ref is None:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(5.35, 3.45), constrained_layout=True)
    style_map = {"CTX": "-", "HIPPO": ":"}
    any_data = False
    for phenotype in PHENOTYPE_ORDER:
        color = PHENOTYPE_COLORS.get(phenotype, "black")
        for region in REGION_ORDER:
            grouped = grouped_by_region.get(region, {})
            if phenotype not in grouped or len(grouped[phenotype]) == 0:
                continue
            Y = np.vstack(grouped[phenotype])
            m, s, _ = mean_sem(Y)
            ax.fill_between(t_ref, m - s, m + s, color=color, alpha=0.10, linewidth=0)
            ax.plot(t_ref, m, color=color, linestyle=style_map[region], lw=1.15,
                    label=f"{PHENOTYPE_DISPLAY.get(phenotype, phenotype)} {REGION_DISPLAY[region]}")
            any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.axvline(0, color="0.45", ls="--", lw=0.9)
    ax.set_xlim(TIME_XLIM_MS)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best", ncol=2)
    save_fig(fig, out_base)
    if source_df is not None and not source_df.empty:
        save_df(source_df, Path(str(out_base) + "_source_data"))


def plot_region_kymograph_grid(avg_by_region, cmap, cbar_label, title, out_base, pct=99.0):
    arrays = []
    for region in REGION_ORDER:
        for phenotype in PHENOTYPE_ORDER:
            rec = avg_by_region.get(region, {}).get(phenotype)
            if rec is not None:
                arrays.append(rec["arr"])
    if not arrays:
        return
    vlim = robust_vlim(arrays, pct=pct, fallback=1.0)
    set_style()
    fig, axes = plt.subplots(len(PHENOTYPE_ORDER), len(REGION_ORDER), figsize=(7.4, 4.85),
                             squeeze=False, constrained_layout=True)
    last_im = None
    for ri, phenotype in enumerate(PHENOTYPE_ORDER):
        for ci, region in enumerate(REGION_ORDER):
            ax = axes[ri, ci]
            rec = avg_by_region.get(region, {}).get(phenotype)
            if rec is None:
                ax.axis("off")
                continue
            arr = rec["arr"]
            t = rec["t"]
            mask = (t >= KYMO_TIME_XLIM_MS[0]) & (t <= KYMO_TIME_XLIM_MS[1])
            if np.any(mask):
                arr2 = arr[:, mask]
                t2 = t[mask]
            else:
                arr2, t2 = arr, t
            last_im = ax.imshow(arr2, aspect="auto", origin="lower", interpolation="nearest",
                                cmap=cmap, vmin=-vlim, vmax=vlim,
                                extent=[t2[0], t2[-1], 0, arr2.shape[0]])
            ax.set_title(f"{PHENOTYPE_DISPLAY.get(phenotype, phenotype)} | {REGION_DISPLAY[region]}", fontsize=9)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("ROI electrode")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    if last_im is not None:
        cb = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
        cb.set_label(cbar_label)
    fig.suptitle(title, fontsize=11)
    save_fig(fig, out_base)



def plot_phenotype_difference_line(t_ref, grouped_by_region, ylabel, title, out_base: Path):
    """Plot APPNL_GF - APPNL for CTX and HIPPO; useful for emphasizing phenotype separation."""
    if t_ref is None:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(5.35, 3.35), constrained_layout=True)
    any_data = False
    source_rows = []
    for region in REGION_ORDER:
        grouped = grouped_by_region.get(region, {})
        if PHENOTYPE_ORDER[0] not in grouped or PHENOTYPE_ORDER[1] not in grouped:
            continue
        if len(grouped[PHENOTYPE_ORDER[0]]) == 0 or len(grouped[PHENOTYPE_ORDER[1]]) == 0:
            continue
        A = np.vstack(grouped[PHENOTYPE_ORDER[0]])
        B = np.vstack(grouped[PHENOTYPE_ORDER[1]])
        ma, _, _ = mean_sem(A)
        mb, _, _ = mean_sem(B)
        diff = mb - ma
        ax.plot(t_ref, diff, color=REGION_COLORS[region], lw=1.35, label=f"{REGION_DISPLAY[region]} Δ")
        for i, tm in enumerate(t_ref):
            source_rows.append({"region": region, "time_ms": float(tm), "APPNL_GF_minus_APPNL": float(diff[i])})
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.45", ls="--", lw=0.9)
    ax.set_xlim(TIME_XLIM_MS)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_fig(fig, out_base)
    save_df(pd.DataFrame(source_rows), Path(str(out_base) + "_source_data"))


def plot_kymograph_difference_grid(avg_by_region, cmap, cbar_label, title, out_base: Path, pct=99.0):
    """Plot APPNL_GF - APPNL mean kymograph for CTX and HIPPO."""
    diffs = {}
    arrays = []
    for region in REGION_ORDER:
        rec_a = avg_by_region.get(region, {}).get(PHENOTYPE_ORDER[0])
        rec_b = avg_by_region.get(region, {}).get(PHENOTYPE_ORDER[1])
        if rec_a is None or rec_b is None:
            continue
        A, ta = rec_a["arr"], rec_a["t"]
        B, tb = rec_b["arr"], rec_b["t"]
        nch = min(A.shape[0], B.shape[0])
        t_ref = ta
        B_interp = np.zeros((nch, len(t_ref)), dtype=float)
        for ch in range(nch):
            B_interp[ch] = interpolate_to_reference(B[ch], tb, t_ref)
        D = B_interp - A[:nch, :len(t_ref)]
        diffs[region] = {"arr": D, "t": t_ref}
        arrays.append(D)
    if not arrays:
        return
    vlim = robust_vlim(arrays, pct=pct, fallback=1.0)
    set_style()
    fig, axes = plt.subplots(1, len(REGION_ORDER), figsize=(7.2, 2.85), squeeze=False, constrained_layout=True)
    axes = axes[0]
    last_im = None
    for ci, region in enumerate(REGION_ORDER):
        ax = axes[ci]
        rec = diffs.get(region)
        if rec is None:
            ax.axis("off")
            continue
        arr = rec["arr"]
        t = rec["t"]
        mask = (t >= KYMO_TIME_XLIM_MS[0]) & (t <= KYMO_TIME_XLIM_MS[1])
        if np.any(mask):
            arr2 = arr[:, mask]
            t2 = t[mask]
        else:
            arr2, t2 = arr, t
        last_im = ax.imshow(arr2, aspect="auto", origin="lower", interpolation="nearest",
                            cmap=cmap, vmin=-vlim, vmax=vlim,
                            extent=[t2[0], t2[-1], 0, arr2.shape[0]])
        ax.axvline(0, color="0.4", ls="--", lw=0.8)
        ax.set_title(f"{REGION_DISPLAY[region]} Δ kymograph", fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("ROI electrode")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if last_im is not None:
        cb = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cb.set_label(cbar_label)
    fig.suptitle(title, fontsize=11)
    save_fig(fig, out_base)

def plot_metric_bars(feature_df: pd.DataFrame, metrics: List[Tuple[str, str]], out_base: Path, title: str):
    if feature_df.empty:
        return
    set_style()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(max(3.25*n, 7.2), 3.45),
                             squeeze=False, constrained_layout=True)
    axes = axes[0]
    stat_rows = []

    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in feature_df.columns:
            ax.axis("off")
            continue
        sub = feature_df.copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        xs = np.arange(len(PHENOTYPE_ORDER), dtype=float)

        # Bars: CTX/HIPPO inside each phenotype.
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

        # Compare APPNL vs APPNL-GF separately for CTX and HIPPO.
        for j, region in enumerate(REGION_ORDER):
            a = sub.loc[(sub["phenotype"] == PHENOTYPE_ORDER[0]) & (sub["region"] == region), metric].to_numpy(dtype=float)
            b = sub.loc[(sub["phenotype"] == PHENOTYPE_ORDER[1]) & (sub["region"] == region), metric].to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            p = welch_pvalue(a, b)
            stat_rows.append({
                "metric": metric, "comparison": f"{PHENOTYPE_ORDER[0]}_vs_{PHENOTYPE_ORDER[1]}",
                "region": region, "p_value": p, "n_a": int(a.size), "n_b": int(b.size),
                "test": "Welch t-test",
            })
            both = np.concatenate([a, b]) if a.size and b.size else np.array([])
            if both.size:
                y = np.nanmax(both)
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

# =============================================================================
# CSD SOURCE / SINK
# =============================================================================

def source_sink_records(events: List[dict], region: str) -> List[dict]:
    recs = []
    for ev in events:
        if ev["region"] != region or ev.get("csd") is None:
            continue
        C = np.asarray(ev["csd"], dtype=float)
        t = np.asarray(ev["time_ms"], dtype=float)
        if CSD_SOURCE_IS_POSITIVE:
            source = np.nanmean(np.where(C > 0, C, 0.0), axis=0)
            sink = np.nanmean(np.where(C < 0, -C, 0.0), axis=0)
        else:
            source = np.nanmean(np.where(C < 0, -C, 0.0), axis=0)
            sink = np.nanmean(np.where(C > 0, C, 0.0), axis=0)
        ratio = source / (sink + EPS)
        balance = (source - sink) / (source + sink + EPS)
        dt = dt_from_time_axis(t)
        cum_source = np.cumsum(np.nan_to_num(source, nan=0.0)) * dt
        cum_sink = np.cumsum(np.nan_to_num(sink, nan=0.0)) * dt
        recs.append({
            "phenotype": ev["phenotype"],
            "dataset": ev["dataset"],
            "roi_name": ev["roi_name"],
            "region": ev["region"],
            "event_index": ev["event_index"],
            "time_ms": t,
            "source": source,
            "sink": sink,
            "ratio": ratio,
            "balance_index": balance,
            "cum_source": cum_source,
            "cum_sink": cum_sink,
        })
    return recs


def group_multichannel_records(records: List[dict], value_key: str, xlim=None):
    if not records:
        return None, {}
    t0 = np.asarray(records[0]["time_ms"], dtype=float)
    if xlim is not None:
        m = (t0 >= xlim[0]) & (t0 <= xlim[1])
        t_ref = t0[m]
    else:
        t_ref = t0
    grouped = {p: [] for p in PHENOTYPE_ORDER}
    for r in records:
        y = np.asarray(r[value_key], dtype=float)
        yy = interpolate_to_reference(y, r["time_ms"], t_ref)
        grouped.setdefault(r["phenotype"], []).append(yy)
    return t_ref, grouped


def flatten_source_sink_records(records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for i, tm in enumerate(r["time_ms"]):
            rows.append({
                "phenotype": r["phenotype"],
                "dataset": r["dataset"],
                "roi_name": r["roi_name"],
                "region": r["region"],
                "event_index": r["event_index"],
                "time_ms": float(tm),
                "source": float(r["source"][i]),
                "sink": float(r["sink"][i]),
                "ratio": float(r["ratio"][i]),
                "balance_index": float(r["balance_index"][i]),
                "cum_source": float(r["cum_source"][i]),
                "cum_sink": float(r["cum_sink"][i]),
            })
    return pd.DataFrame(rows)


def plot_source_sink_lines(ss_by_region, out_base: Path, source_df=None):
    set_style()
    fig, axes = plt.subplots(len(PHENOTYPE_ORDER), len(REGION_ORDER), figsize=(7.4, 5.05),
                             squeeze=False, sharey=True, constrained_layout=True)
    for ri, phenotype in enumerate(PHENOTYPE_ORDER):
        for ci, region in enumerate(REGION_ORDER):
            ax = axes[ri, ci]
            recs = ss_by_region.get(region, [])
            t_src, g_source = group_multichannel_records(recs, "source", TIME_XLIM_MS)
            t_sink, g_sink = group_multichannel_records(recs, "sink", TIME_XLIM_MS)
            if t_src is None or phenotype not in g_source or len(g_source[phenotype]) == 0:
                ax.axis("off")
                continue
            Ysrc = np.vstack(g_source[phenotype])
            Ysnk = np.vstack(g_sink[phenotype])
            m1, s1, _ = mean_sem(Ysrc)
            m2, s2, _ = mean_sem(Ysnk)
            ax.fill_between(t_src, m1 - s1, m1 + s1, color="forestgreen", alpha=0.15, linewidth=0)
            ax.fill_between(t_sink, m2 - s2, m2 + s2, color="magenta", alpha=0.15, linewidth=0)
            ax.plot(t_src, m1, color="forestgreen", lw=1.0, label="Source")
            ax.plot(t_sink, m2, color="magenta", lw=1.0, label="Sink")
            ax.axvline(0, color="0.4", ls="--", lw=0.9)
            ax.set_xlim(TIME_XLIM_MS)
            ax.set_title(f"{PHENOTYPE_DISPLAY.get(phenotype, phenotype)} | {REGION_DISPLAY[region]}", fontsize=9)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Mean source / sink")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, fontsize=7, loc="best")
    sync_y_limits(axes)
    fig.suptitle("Baseline | CSD source and sink", fontsize=11)
    save_fig(fig, out_base)
    if source_df is not None and not source_df.empty:
        save_df(source_df, Path(str(out_base) + "_source_data"))


def plot_source_sink_balance(ss_by_region, out_base: Path):
    set_style()
    fig, axes = plt.subplots(len(PHENOTYPE_ORDER), 2, figsize=(8.6, 5.85),
                             squeeze=False, constrained_layout=True)
    for ri, phenotype in enumerate(PHENOTYPE_ORDER):
        ax = axes[ri, 0]
        for region in REGION_ORDER:
            recs = ss_by_region.get(region, [])
            t_ref, grouped = group_multichannel_records(recs, "balance_index", TIME_XLIM_MS)
            if t_ref is None or phenotype not in grouped or len(grouped[phenotype]) == 0:
                continue
            Y = np.vstack(grouped[phenotype])
            m, s, _ = mean_sem(Y)
            color = REGION_COLORS[region]
            ax.fill_between(t_ref, m - s, m + s, color=color, alpha=0.16, linewidth=0)
            ax.plot(t_ref, m, color=color, lw=1.0, label=REGION_DISPLAY[region])
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.4", ls="--", lw=0.9)
        ax.set_xlim(TIME_XLIM_MS)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Balance index\n(source-sink)/(source+sink)")
        ax.set_title(f"{PHENOTYPE_DISPLAY.get(phenotype, phenotype)} | Balance", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="best")

        ax2 = axes[ri, 1]
        for region in REGION_ORDER:
            recs = ss_by_region.get(region, [])
            t1, g1 = group_multichannel_records(recs, "cum_source", TIME_XLIM_MS)
            t2, g2 = group_multichannel_records(recs, "cum_sink", TIME_XLIM_MS)
            if t1 is None or phenotype not in g1 or len(g1[phenotype]) == 0:
                continue
            Y1 = np.vstack(g1[phenotype])
            Y2 = np.vstack(g2[phenotype])
            m1, _, _ = mean_sem(Y1)
            m2, _, _ = mean_sem(Y2)
            c = REGION_COLORS[region]
            ax2.plot(t1, m1, color=c, lw=1.0, label=f"{REGION_DISPLAY[region]} source")
            ax2.plot(t2, m2, color=c, lw=1.0, ls="--", label=f"{REGION_DISPLAY[region]} sink")
        ax2.axvline(0, color="0.4", ls="--", lw=0.9)
        ax2.set_xlim(TIME_XLIM_MS)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Cumulative magnitude")
        ax2.set_title(f"{PHENOTYPE_DISPLAY.get(phenotype, phenotype)} | Accumulation", fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(frameon=False, fontsize=7, loc="best")
    sync_y_limits(axes[:, 0])
    sync_y_limits(axes[:, 1])
    fig.suptitle("Baseline | CSD source-sink balance and accumulation", fontsize=11)
    save_fig(fig, out_base)

# =============================================================================
# WAVELET
# =============================================================================

def sampling_rate_from_time_ms(t):
    dt_ms = dt_from_time_axis(t)
    return 1000.0 / dt_ms if dt_ms > 0 else 500.0


def morlet_cwt_fallback(y, fs, freqs_hz, w=6.0):
    y = fill_nan_1d(y)
    out = []
    for f in freqs_hz:
        f = float(f)
        if f <= 0:
            out.append(np.zeros_like(y, dtype=float))
            continue
        sigma_t = w / (2.0 * np.pi * f)
        half = max(int(np.ceil(4.0 * sigma_t * fs)), 8)
        tt = np.arange(-half, half + 1) / fs
        wave = np.exp(2j * np.pi * f * tt) * np.exp(-(tt ** 2) / (2 * sigma_t ** 2))
        wave -= np.mean(wave)
        norm = np.sqrt(np.sum(np.abs(wave) ** 2))
        if norm > 0:
            wave = wave / norm
        conv = np.convolve(y, np.conj(wave[::-1]), mode="same")
        out.append(np.abs(conv) ** 2)
    return np.asarray(out, dtype=np.float32)


def compute_wavelet_power(y, time_ms, freqs_hz=WAVELET_FREQS_HZ):
    y = fill_nan_1d(np.asarray(y, dtype=float))
    fs = sampling_rate_from_time_ms(time_ms)
    try:
        import pywt
        dt_s = 1.0 / fs
        central = pywt.central_frequency(WAVELET_NAME)
        scales = central / (np.asarray(freqs_hz, dtype=float) * dt_s)
        coef, _ = pywt.cwt(y, scales, WAVELET_NAME, sampling_period=dt_s)
        power = np.abs(coef) ** 2
        return power.astype(np.float32)
    except Exception:
        return morlet_cwt_fallback(y, fs, freqs_hz)


def wavelet_band_features(power, time_ms, freqs_hz=WAVELET_FREQS_HZ):
    power = np.asarray(power, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float)
    tm = time_mask(time_ms, WAVELET_ANALYSIS_WINDOW_MS)
    bm = time_mask(time_ms, WAVELET_BASELINE_WINDOW_MS)

    total = np.nanmean(power[:, tm]) if np.any(tm) else np.nanmean(power)
    base = np.nanmean(power[:, bm]) if np.any(bm) else np.nan
    out = {
        "wavelet_total_power": float(total),
        "wavelet_baseline_power": float(base) if np.isfinite(base) else np.nan,
        "wavelet_evoked_over_baseline": float(total / (base + EPS)) if np.isfinite(base) else np.nan,
    }

    bands = {
        "delta_1_4": (1, 4),
        "theta_4_8": (4, 8),
        "alpha_8_12": (8, 12),
        "beta_12_30": (12, 30),
        "gamma_30_80": (30, 80),
        "high_80_200": (80, 200),
    }
    for name, (f0, f1) in bands.items():
        fm = (freqs >= f0) & (freqs < f1)
        if not np.any(fm):
            out[f"{name}_power"] = np.nan
            out[f"{name}_fraction"] = np.nan
            continue
        val = np.nanmean(power[fm][:, tm]) if np.any(tm) else np.nanmean(power[fm])
        out[f"{name}_power"] = float(val)
        out[f"{name}_fraction"] = float(val / (total + EPS)) if np.isfinite(total) else np.nan

    Pwin = power[:, tm] if np.any(tm) else power
    twin = np.asarray(time_ms)[tm] if np.any(tm) else np.asarray(time_ms)
    if Pwin.size and np.any(np.isfinite(Pwin)):
        ij = np.unravel_index(np.nanargmax(Pwin), Pwin.shape)
        out["wavelet_peak_freq_hz"] = float(freqs[ij[0]])
        out["wavelet_peak_time_ms"] = float(twin[ij[1]])
        out["wavelet_peak_power"] = float(Pwin[ij])
    return out


def build_wavelet_tables(events: List[dict], out_dir: Path):
    rows = []
    group_power = {}

    if not RUN_WAVELET:
        return pd.DataFrame(), {}

    ensure_dir(out_dir)
    print("[INFO] Running baseline LFP wavelet analysis...")

    for ev in events:
        t = np.asarray(ev["time_ms"], dtype=float)
        y = event_mean_signal(ev, "lfp")
        bm = time_mask(t, BASELINE_WINDOW_MS)
        y = y - np.nanmedian(y[bm])
        P = compute_wavelet_power(y, t, WAVELET_FREQS_HZ)

        row = {k: ev[k] for k in [
            "phenotype", "dataset", "roi_name", "region", "drug",
            "event_index", "event_ms", "event_dir", "event_key", "n_electrodes"
        ]}
        row.update(wavelet_band_features(P, t, WAVELET_FREQS_HZ))
        rows.append(row)

        key = (ev["phenotype"], ev["region"])
        group_power.setdefault(key, []).append((P, t))

    df = pd.DataFrame(rows)
    save_df(df, out_dir / "baseline_wavelet_event_features")
    return df, group_power


def average_wavelet_power(group_power):
    out = {}
    for (phenotype, region), lst in group_power.items():
        if not lst:
            continue
        # use first time axis as reference; if same length stack directly.
        t_ref = lst[0][1]
        powers = []
        for P, t in lst:
            P = np.asarray(P, dtype=float)
            if P.shape[1] == len(t_ref) and np.allclose(t, t_ref):
                powers.append(P)
            else:
                interpP = np.zeros((P.shape[0], len(t_ref)), dtype=float)
                for fi in range(P.shape[0]):
                    interpP[fi] = interpolate_to_reference(P[fi], t, t_ref)
                powers.append(interpP)
        if powers:
            out.setdefault(region, {})[phenotype] = {
                "power": np.nanmean(np.stack(powers, axis=0), axis=0),
                "t": np.asarray(t_ref, dtype=float),
                "n_events": len(powers),
            }
    return out


def plot_wavelet_grid(avg_power_by_region, out_base: Path):
    arrays = []
    for region in REGION_ORDER:
        for phenotype in PHENOTYPE_ORDER:
            rec = avg_power_by_region.get(region, {}).get(phenotype)
            if rec is not None:
                arrays.append(np.log10(rec["power"] + EPS))
    if not arrays:
        return
    all_vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if np.any(np.isfinite(a))])
    vmin = np.nanpercentile(all_vals, 5) if all_vals.size else None
    vmax = np.nanpercentile(all_vals, 99) if all_vals.size else None

    set_style()
    fig, axes = plt.subplots(len(PHENOTYPE_ORDER), len(REGION_ORDER), figsize=(7.4, 4.85),
                             squeeze=False, constrained_layout=True)
    last_im = None
    freqs = np.asarray(WAVELET_FREQS_HZ, dtype=float)
    for ri, phenotype in enumerate(PHENOTYPE_ORDER):
        for ci, region in enumerate(REGION_ORDER):
            ax = axes[ri, ci]
            rec = avg_power_by_region.get(region, {}).get(phenotype)
            if rec is None:
                ax.axis("off")
                continue
            P = np.log10(rec["power"] + EPS)
            t = rec["t"]
            m = (t >= TIME_XLIM_MS[0]) & (t <= TIME_XLIM_MS[1])
            if np.any(m):
                P2 = P[:, m]
                t2 = t[m]
            else:
                P2, t2 = P, t
            last_im = ax.imshow(P2, aspect="auto", origin="lower", interpolation="nearest",
                                extent=[t2[0], t2[-1], freqs[0], freqs[-1]],
                                cmap="magma", vmin=vmin, vmax=vmax)
            ax.axvline(0, color="cyan", ls="--", lw=0.8)
            ax.set_title(f"{PHENOTYPE_DISPLAY.get(phenotype, phenotype)} | {REGION_DISPLAY[region]}", fontsize=9)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (Hz)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    if last_im is not None:
        cb = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.026, pad=0.02)
        cb.set_label("log10 wavelet power")
    fig.suptitle("Baseline | mean LFP wavelet power", fontsize=11)
    save_fig(fig, out_base)



# =============================================================================
# FREQUENCY / PSD ANALYSIS
# =============================================================================

def compute_psd(y: np.ndarray, t: np.ndarray):
    y = fill_nan_1d(np.asarray(y, dtype=float))
    t = np.asarray(t, dtype=float)
    bm = time_mask(t, BASELINE_WINDOW_MS)
    y = y - np.nanmedian(y[bm])
    wm = time_mask(t, PSD_ANALYSIS_WINDOW_MS)
    y = y[wm]
    if y.size < 4:
        return np.array([]), np.array([])
    fs = sampling_rate_from_time_ms(t)
    if welch is not None:
        nperseg = min(int(PSD_NPERSEG_MAX), int(y.size))
        try:
            f, pxx = welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2 if nperseg >= 8 else None)
        except Exception:
            f = np.fft.rfftfreq(y.size, d=1.0 / fs)
            pxx = (np.abs(np.fft.rfft(y)) ** 2) / max(y.size, 1)
    else:
        f = np.fft.rfftfreq(y.size, d=1.0 / fs)
        pxx = (np.abs(np.fft.rfft(y)) ** 2) / max(y.size, 1)
    keep = np.isfinite(f) & np.isfinite(pxx) & (f <= PSD_FREQ_MAX_HZ)
    return f[keep], pxx[keep]


def _mask_ignore_frequency_bands(f: np.ndarray, ignore_bands) -> np.ndarray:
    """Return True for frequencies that should be kept for aperiodic 1/f fitting."""
    f = np.asarray(f, dtype=float)
    keep = np.ones(f.shape, dtype=bool)
    for f0, f1 in ignore_bands:
        keep &= ~((f >= float(f0)) & (f <= float(f1)))
    return keep


def fit_aperiodic_exponent_loglog(f, pxx):
    """
    Fallback 1/f fit when specparam/fooof is unavailable.

    Model:
        log10(PSD) = offset - exponent * log10(f)

    Therefore:
        exponent = -slope

    This is not a full FOOOF/specparam peak-corrected model, but it provides a
    stable benchmark column and keeps the analysis runnable on machines where
    specparam is not installed.
    """
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    m = (
        np.isfinite(f) & np.isfinite(pxx) &
        (f >= APERIODIC_FIT_RANGE_HZ[0]) & (f <= APERIODIC_FIT_RANGE_HZ[1]) &
        (f > 0) & (pxx > 0)
    )
    m &= _mask_ignore_frequency_bands(f, APERIODIC_IGNORE_BANDS_HZ)
    if np.sum(m) < 5:
        return {
            "psd_aperiodic_offset": np.nan,
            "psd_aperiodic_exponent": np.nan,
            "psd_aperiodic_r2": np.nan,
            "psd_aperiodic_fit_method": "loglog_linear_insufficient_points",
        }
    x = np.log10(f[m])
    y = np.log10(pxx[m])
    try:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.nansum((y - yhat) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + EPS)
        return {
            "psd_aperiodic_offset": float(intercept),
            "psd_aperiodic_exponent": float(-slope),
            "psd_aperiodic_r2": float(r2),
            "psd_aperiodic_fit_method": "loglog_linear",
        }
    except Exception:
        return {
            "psd_aperiodic_offset": np.nan,
            "psd_aperiodic_exponent": np.nan,
            "psd_aperiodic_r2": np.nan,
            "psd_aperiodic_fit_method": "loglog_linear_failed",
        }


def fit_aperiodic_exponent(f, pxx):
    """
    Estimate the aperiodic 1/f exponent.

    Preferred: specparam / FOOOF if installed.
    Fallback: log-log linear PSD fit.
    """
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    m = (
        np.isfinite(f) & np.isfinite(pxx) &
        (f >= APERIODIC_FIT_RANGE_HZ[0]) & (f <= APERIODIC_FIT_RANGE_HZ[1]) &
        (f > 0) & (pxx > 0)
    )
    if np.sum(m) < 5:
        return fit_aperiodic_exponent_loglog(f, pxx)

    ff = f[m]
    pp = pxx[m]

    if USE_SPECPARAM_IF_AVAILABLE and SpectralModel is not None:
        try:
            sm = SpectralModel(verbose=False, peak_width_limits=[1.0, 20.0], max_n_peaks=6)
            sm.fit(ff, pp, APERIODIC_FIT_RANGE_HZ)
            ap = np.asarray(sm.get_params("aperiodic_params"), dtype=float)
            r2 = getattr(sm, "r_squared_", np.nan)
            return {
                "psd_aperiodic_offset": float(ap[0]) if ap.size > 0 else np.nan,
                "psd_aperiodic_exponent": float(ap[-1]) if ap.size > 1 else np.nan,
                "psd_aperiodic_r2": float(r2) if np.isfinite(r2) else np.nan,
                "psd_aperiodic_fit_method": "specparam",
            }
        except Exception:
            pass

    if USE_SPECPARAM_IF_AVAILABLE and FOOOF is not None:
        try:
            fm = FOOOF(verbose=False, peak_width_limits=[1.0, 20.0], max_n_peaks=6)
            fm.fit(ff, pp, APERIODIC_FIT_RANGE_HZ)
            ap = np.asarray(fm.aperiodic_params_, dtype=float)
            r2 = getattr(fm, "r_squared_", np.nan)
            return {
                "psd_aperiodic_offset": float(ap[0]) if ap.size > 0 else np.nan,
                "psd_aperiodic_exponent": float(ap[-1]) if ap.size > 1 else np.nan,
                "psd_aperiodic_r2": float(r2) if np.isfinite(r2) else np.nan,
                "psd_aperiodic_fit_method": "fooof",
            }
        except Exception:
            pass

    return fit_aperiodic_exponent_loglog(f, pxx)


def psd_band_features(f, pxx):
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    if f.size == 0 or pxx.size == 0:
        return {}

    total = float(np.trapezoid(pxx, f)) if f.size > 1 else float(np.nanmean(pxx))
    out = {"psd_total_power": total}

    # Band-resolved PSD benchmark: absolute band power, relative fraction, and mean PSD.
    # Keeps the original output names while adding explicit benchmark-friendly columns.
    band_power_cache = {}
    for short_name, (f0, f1) in PSD_BANDS.items():
        m = (f >= float(f0)) & (f < float(f1))
        val = float(np.trapezoid(pxx[m], f[m])) if np.sum(m) > 1 else np.nan
        mean_psd = float(np.nanmean(pxx[m])) if np.any(m) else np.nan
        band_power_cache[short_name] = val

        old_prefix = f"psd_{short_name}"
        out[f"{old_prefix}_power"] = val
        out[f"{old_prefix}_fraction"] = float(val / (total + EPS)) if np.isfinite(val) else np.nan
        out[f"{old_prefix}_mean_psd"] = mean_psd

        # Redundant but explicit names for downstream benchmark tables.
        out[f"benchmark_{old_prefix}_power"] = val
        out[f"benchmark_{old_prefix}_fraction"] = out[f"{old_prefix}_fraction"]
        out[f"benchmark_{old_prefix}_mean_psd"] = mean_psd

    # Theta/gamma ratio benchmark.
    theta_key, gamma_key = THETA_GAMMA_RATIO_BANDS
    theta_power = band_power_cache.get(theta_key, np.nan)
    gamma_power = band_power_cache.get(gamma_key, np.nan)
    out["psd_theta_gamma_ratio"] = float(theta_power / (gamma_power + EPS)) if np.isfinite(theta_power) and np.isfinite(gamma_power) else np.nan
    out["benchmark_psd_theta_gamma_ratio"] = out["psd_theta_gamma_ratio"]

    # Peak frequency benchmark: both global and band-restricted peak frequencies.
    out["psd_peak_freq_hz"] = float(f[int(np.nanargmax(pxx))]) if np.any(np.isfinite(pxx)) else np.nan
    out["benchmark_psd_peak_freq_hz"] = out["psd_peak_freq_hz"]
    for short_name, (f0, f1) in PSD_BANDS.items():
        m = (f >= float(f0)) & (f < float(f1)) & np.isfinite(pxx)
        if np.any(m):
            idx = np.where(m)[0][int(np.nanargmax(pxx[m]))]
            out[f"psd_{short_name}_peak_freq_hz"] = float(f[idx])
            out[f"benchmark_psd_{short_name}_peak_freq_hz"] = float(f[idx])
        else:
            out[f"psd_{short_name}_peak_freq_hz"] = np.nan
            out[f"benchmark_psd_{short_name}_peak_freq_hz"] = np.nan

    # Spectral centroid retained from original script.
    out["psd_spectral_centroid_hz"] = float(np.nansum(f * pxx) / (np.nansum(pxx) + EPS))

    # Aperiodic exponent / 1-f slope benchmark.
    ap = fit_aperiodic_exponent(f, pxx)
    out.update(ap)
    out["benchmark_psd_aperiodic_exponent"] = out.get("psd_aperiodic_exponent", np.nan)
    out["benchmark_psd_1f_slope"] = -out.get("psd_aperiodic_exponent", np.nan) if np.isfinite(out.get("psd_aperiodic_exponent", np.nan)) else np.nan
    out["benchmark_psd_aperiodic_r2"] = out.get("psd_aperiodic_r2", np.nan)

    return out


def build_frequency_tables(events: List[dict], out_dir: Path):
    if not RUN_FREQUENCY_ANALYSIS:
        return pd.DataFrame(), {}
    ensure_dir(out_dir)
    rows = []
    spectra = {}
    print("[INFO] Running baseline LFP frequency/PSD analysis...")
    for ev in events:
        t = np.asarray(ev["time_ms"], dtype=float)
        y = event_mean_signal(ev, "lfp")
        f, pxx = compute_psd(y, t)
        row = {k: ev[k] for k in [
            "phenotype", "dataset", "roi_name", "region", "drug",
            "event_index", "event_ms", "event_dir", "event_key", "n_electrodes"
        ]}
        row.update(psd_band_features(f, pxx))
        rows.append(row)
        if f.size and pxx.size:
            key = (ev["phenotype"], ev["region"])
            spectra.setdefault(key, []).append((f, pxx))
    df = pd.DataFrame(rows)
    save_df(df, out_dir / "baseline_frequency_psd_event_features")
    return df, spectra


def average_spectra(spectra):
    out = {}
    f_ref = np.linspace(0, PSD_FREQ_MAX_HZ, 400)
    for (phenotype, region), lst in spectra.items():
        vals = []
        for f, pxx in lst:
            vals.append(np.interp(f_ref, f, pxx, left=np.nan, right=np.nan))
        if vals:
            out.setdefault(region, {})[phenotype] = {
                "freq_hz": f_ref,
                "psd": np.nanmean(np.vstack(vals), axis=0),
                "n_events": len(vals),
            }
    return out


def plot_psd_spectra(avg_spectra_by_region, out_base: Path):
    set_style()
    fig, axes = plt.subplots(1, len(REGION_ORDER), figsize=(7.2, 3.25), squeeze=False, constrained_layout=True)
    axes = axes[0]
    any_data = False
    for ci, region in enumerate(REGION_ORDER):
        ax = axes[ci]
        for phenotype in PHENOTYPE_ORDER:
            rec = avg_spectra_by_region.get(region, {}).get(phenotype)
            if rec is None:
                continue
            f = rec["freq_hz"]
            y = rec["psd"]
            ax.plot(f, y, color=PHENOTYPE_COLORS[phenotype], lw=1.25,
                    label=f"{PHENOTYPE_DISPLAY[phenotype]} n={rec['n_events']}")
            any_data = True
        ax.set_xlim(0, PSD_FREQ_MAX_HZ)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("LFP PSD (uV^2/Hz)")
        ax.set_title(f"{REGION_DISPLAY[region]} spectrum", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    if not any_data:
        plt.close(fig)
        return
    fig.suptitle("Baseline | LFP frequency spectrum", fontsize=11)
    save_fig(fig, out_base)

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def make_lfp_outputs(events: List[dict], feature_df: pd.DataFrame):
    out = ensure_dir(OUT_DIR / "01_LFP")

    grouped_by_region = {}
    src_rows = []
    t_ref_main = None
    for region in REGION_ORDER:
        recs = records_for_signal(events, "lfp", region, normalized=False)
        t_ref, grouped, src_df = group_records_by_phenotype(recs, TIME_XLIM_MS)
        grouped_by_region[region] = grouped
        if t_ref is not None and t_ref_main is None:
            t_ref_main = t_ref
        if not src_df.empty:
            src_rows.append(src_df)
    if t_ref_main is not None:
        plot_overlay_two_region_by_phenotype(
            t_ref_main,
            grouped_by_region,
            LFP_TRACE_YLABEL,
            "Baseline | ALL LFP CTX/HIPPO line plot",
            out / "baseline_01_LFP_CTX_HIPPO_line",
            source_df=pd.concat(src_rows, ignore_index=True) if src_rows else None,
        )

    norm_grouped_by_region = {}
    norm_src_rows = []
    t_ref_norm = None
    for region in REGION_ORDER:
        recs = records_for_signal(events, "lfp", region, normalized=True)
        t_ref, grouped, src_df = group_records_by_phenotype(recs, TIME_XLIM_MS)
        norm_grouped_by_region[region] = grouped
        if t_ref is not None and t_ref_norm is None:
            t_ref_norm = t_ref
        if not src_df.empty:
            norm_src_rows.append(src_df)
    if t_ref_norm is not None:
        plot_normalized_together_line(
            t_ref_norm,
            norm_grouped_by_region,
            "Normalized LFP amplitude",
            "Baseline | LFP normalized together",
            out / "baseline_01b_LFP_CTX_HIPPO_normalized_together",
            source_df=pd.concat(norm_src_rows, ignore_index=True) if norm_src_rows else None,
        )
        plot_phenotype_difference_line(
            t_ref_norm,
            norm_grouped_by_region,
            "Δ normalized LFP (APPNL-GF - APPNL)",
            "Baseline | LFP phenotype-difference line",
            out / "baseline_01c_LFP_APPNLGF_minus_APPNL_difference_line",
        )

    kymo = {region: average_kymographs(events, "lfp", region, normalized=False) for region in REGION_ORDER}
    plot_region_kymograph_grid(
        kymo,
        cmap=LFP_KYMO_CMAP,
        cbar_label="LFP (uV)",
        title="Baseline | ALL LFP CTX/HIPPO kymograph",
        out_base=out / "baseline_02_LFP_CTX_HIPPO_kymograph",
        pct=99.0,
    )
    plot_kymograph_difference_grid(
        kymo,
        cmap=LFP_KYMO_CMAP,
        cbar_label="Δ LFP (uV)",
        title="Baseline | LFP kymograph difference: APPNL-GF - APPNL",
        out_base=out / "baseline_02b_LFP_APPNLGF_minus_APPNL_kymograph_difference",
        pct=99.0,
    )

    metrics = [
        ("lfp_abs_peak_uV", "LFP amplitude (uV)"),
        ("lfp_energy_uV2_ms", "LFP energy (uV^2 ms)"),
        ("lfp_norm_slope_pct", "Normalized LFP slope (%)"),
        ("lfp_latency_abs_peak_ms", "Peak latency (ms)"),
        ("lfp_complexity_D", "LFP complexity D"),
        ("lfp_active_electrode_fraction", "Active electrode fraction"),
        ("lfp_event_rate_Hz_per_electrode", "LFP event rate/electrode (Hz)"),
        ("lfp_interelectrode_corr_mean", "Inter-electrode corr."),
    ]
    plot_metric_bars(feature_df, metrics, out / "baseline_03_LFP_CTX_HIPPO_metric_bars",
                     "Baseline | CTX vs HIPPO LFP metrics")


def make_csd_outputs(events: List[dict], feature_df: pd.DataFrame):
    out = ensure_dir(OUT_DIR / "02_CSD")

    grouped_by_region = {}
    src_rows = []
    t_ref_main = None
    for region in REGION_ORDER:
        recs = records_for_signal(events, "csd", region, normalized=False, signed_csd=False)
        t_ref, grouped, src_df = group_records_by_phenotype(recs, TIME_XLIM_MS)
        grouped_by_region[region] = grouped
        if t_ref is not None and t_ref_main is None:
            t_ref_main = t_ref
        if not src_df.empty:
            src_rows.append(src_df)
    if t_ref_main is not None:
        plot_overlay_two_region_by_phenotype(
            t_ref_main,
            grouped_by_region,
            CSD_TRACE_YLABEL,
            "Baseline | ALL CSD CTX/HIPPO line plot",
            out / "baseline_04_CSD_CTX_HIPPO_line",
            source_df=pd.concat(src_rows, ignore_index=True) if src_rows else None,
        )

    norm_grouped_by_region = {}
    norm_src_rows = []
    t_ref_norm = None
    for region in REGION_ORDER:
        recs = records_for_signal(events, "csd", region, normalized=True, signed_csd=False)
        t_ref, grouped, src_df = group_records_by_phenotype(recs, TIME_XLIM_MS)
        norm_grouped_by_region[region] = grouped
        if t_ref is not None and t_ref_norm is None:
            t_ref_norm = t_ref
        if not src_df.empty:
            norm_src_rows.append(src_df)
    if t_ref_norm is not None:
        plot_normalized_together_line(
            t_ref_norm,
            norm_grouped_by_region,
            "Normalized mean |CSD|",
            "Baseline | CSD normalized together",
            out / "baseline_04b_CSD_CTX_HIPPO_normalized_together",
            source_df=pd.concat(norm_src_rows, ignore_index=True) if norm_src_rows else None,
        )
        plot_phenotype_difference_line(
            t_ref_norm,
            norm_grouped_by_region,
            "Δ normalized |CSD| (APPNL-GF - APPNL)",
            "Baseline | CSD phenotype-difference line",
            out / "baseline_04c_CSD_APPNLGF_minus_APPNL_difference_line",
        )

    kymo = {region: average_kymographs(events, "csd", region, normalized=False) for region in REGION_ORDER}
    plot_region_kymograph_grid(
        kymo,
        cmap=CSD_KYMO_CMAP,
        cbar_label="CSD (uV/um^2)",
        title="Baseline | ALL CSD CTX/HIPPO kymograph",
        out_base=out / "baseline_05_CSD_CTX_HIPPO_kymograph",
        pct=99.0,
    )
    plot_kymograph_difference_grid(
        kymo,
        cmap=CSD_KYMO_CMAP,
        cbar_label="Δ CSD (uV/um^2)",
        title="Baseline | CSD kymograph difference: APPNL-GF - APPNL",
        out_base=out / "baseline_05b_CSD_APPNLGF_minus_APPNL_kymograph_difference",
        pct=99.0,
    )

    metrics = [
        ("csd_abs_peak_uV_per_um2", "CSD abs peak (uV/um^2)"),
        ("csd_mean_abs_auc_uV_per_um2_ms", "CSD |activity| AUC"),
        ("csd_source_sink_ratio", "Source/sink ratio"),
        ("csd_balance_index_mean", "Balance index"),
    ]
    plot_metric_bars(feature_df, metrics, out / "baseline_06_CSD_CTX_HIPPO_metric_bars",
                     "Baseline | CTX vs HIPPO CSD metrics")


def make_source_sink_outputs(events: List[dict]):
    out = ensure_dir(OUT_DIR / "03_SOURCE_SINK")
    ss_by_region = {}
    flat_rows = []
    for region in REGION_ORDER:
        recs = source_sink_records(events, region)
        ss_by_region[region] = recs
        if recs:
            flat_rows.append(flatten_source_sink_records(recs))
    ss_df = pd.concat(flat_rows, ignore_index=True) if flat_rows else pd.DataFrame()
    if not ss_df.empty:
        save_df(ss_df, OUT_DIR / "tables" / "baseline_csd_source_sink_time_rows")
    plot_source_sink_lines(ss_by_region, out / "baseline_07_CSD_CTX_HIPPO_source_sink_lines", source_df=ss_df)
    plot_source_sink_balance(ss_by_region, out / "baseline_08_CSD_CTX_HIPPO_source_sink_balance_accumulation")


def make_wavelet_outputs(events: List[dict]):
    out = ensure_dir(OUT_DIR / "04_WAVELET")
    wdf, group_power = build_wavelet_tables(events, out)
    if wdf.empty:
        return
    avg = average_wavelet_power(group_power)
    plot_wavelet_grid(avg, out / "baseline_09_LFP_wavelet_power_heatmaps")

    metrics = [
        ("wavelet_total_power", "Total wavelet power"),
        ("wavelet_evoked_over_baseline", "Evoked/base power"),
        ("theta_4_8_fraction", "Theta fraction"),
        ("beta_12_30_fraction", "Beta fraction"),
        ("gamma_30_80_fraction", "Gamma fraction"),
        ("wavelet_peak_freq_hz", "Peak frequency (Hz)"),
    ]
    plot_metric_bars(wdf, metrics, out / "baseline_10_LFP_wavelet_metric_bars",
                     "Baseline | CTX vs HIPPO LFP wavelet metrics")



def make_frequency_outputs(events: List[dict]):
    out = ensure_dir(OUT_DIR / "05_FREQUENCY_PSD")
    fdf, spectra = build_frequency_tables(events, out)
    if fdf.empty:
        return
    avg = average_spectra(spectra)
    plot_psd_spectra(avg, out / "baseline_11_LFP_PSD_spectra")
    metrics = [
        ("psd_total_power", "PSD total power"),
        ("psd_delta_1_4_power", "Delta power (1-4 Hz)"),
        ("psd_theta_4_8_power", "Theta power (4-8 Hz)"),
        ("psd_gamma_30_80_power", "Gamma power (30-80 Hz)"),
        ("psd_theta_gamma_ratio", "Theta/gamma ratio"),
        ("psd_aperiodic_exponent", "1/f exponent"),
        ("benchmark_psd_1f_slope", "1/f slope"),
        ("psd_peak_freq_hz", "PSD peak frequency (Hz)"),
        ("psd_spectral_centroid_hz", "Spectral centroid (Hz)"),
        ("psd_theta_4_8_fraction", "Theta fraction"),
        ("psd_beta_12_30_fraction", "Beta fraction"),
        ("psd_gamma_30_80_fraction", "Gamma fraction"),
        ("psd_high_80_200_fraction", "High freq. fraction"),
    ]
    plot_metric_bars(fdf, metrics, out / "baseline_12_LFP_PSD_metric_bars",
                     "Baseline | CTX vs HIPPO LFP frequency metrics")


def save_manifest(datasets: List[DatasetRoot], event_meta: pd.DataFrame):
    ensure_dir(OUT_DIR)
    manifest = {
        "target": "baseline-only APPNL vs APPNL-GF CTX/HIPPO comparison",
        "root_dir": str(ROOT_DIR.resolve()),
        "out_dir": str(OUT_DIR.resolve()),
        "phenotype_order": PHENOTYPE_ORDER,
        "phenotype_display": PHENOTYPE_DISPLAY,
        "baseline_aliases": sorted(BASELINE_ALIASES),
        "roi_region_map": ROI_REGION_MAP,
        "region_order": REGION_ORDER,
        "time_xlim_ms": TIME_XLIM_MS,
        "feature_window_ms": FEATURE_WINDOW_MS,
        "wavelet_freqs_hz": WAVELET_FREQS_HZ,
        "lfp_complexity_smooth_width": LFP_COMPLEXITY_SMOOTH_WIDTH,
        "lfp_complexity_time_window_ms": LFP_COMPLEXITY_TIME_WINDOW_MS,
        "psd_analysis_window_ms": PSD_ANALYSIS_WINDOW_MS,
        "psd_freq_max_hz": PSD_FREQ_MAX_HZ,
        "psd_bands": PSD_BANDS,
        "theta_gamma_ratio_bands": THETA_GAMMA_RATIO_BANDS,
        "aperiodic_fit_range_hz": APERIODIC_FIT_RANGE_HZ,
        "aperiodic_ignore_bands_hz": APERIODIC_IGNORE_BANDS_HZ,
        "use_specparam_if_available": USE_SPECPARAM_IF_AVAILABLE,
        "datasets": [{"phenotype": d.phenotype, "name": d.name, "path": str(d.path)} for d in datasets],
        "n_events_loaded": int(len(event_meta)),
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Per-event LFP/CSD kymographs, PSD/wavelet band power, and 1/f exponent fit")
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

    datasets = discover_dataset_roots()
    print("[INFO] Phenotype roots found:")
    for ds in datasets:
        print(f"  {ds.phenotype:8s} | {ds.path}")

    events, event_meta = load_baseline_events(datasets)
    save_manifest(datasets, event_meta)
    save_df(event_meta, OUT_DIR / "tables" / "baseline_loaded_event_manifest")

    feature_df = build_event_feature_table(events)
    save_df(feature_df, OUT_DIR / "tables" / "baseline_event_lfp_csd_features")

    make_lfp_outputs(events, feature_df)
    make_csd_outputs(events, feature_df)
    make_source_sink_outputs(events)
    make_wavelet_outputs(events)
    make_frequency_outputs(events)

    print("\nDONE")
    print(f"Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
