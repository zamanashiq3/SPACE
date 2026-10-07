# -*- coding: utf-8 -*-
"""
electrode_connectivity.py -- SPACE framework: per-electrode functional connectivity.

Manuscript figure(s): Figure 3G-J (per-electrode Pearson correlation matrices per ROI,
anti-correlated-pair %).

What it computes:
    Single-electrode-resolution functional connectivity: for each anatomical ROI (16
    electrodes per ROI, up to a 96-node graph across all ROIs), Pearson correlation
    matrices between electrodes, the percentage of anti-correlated electrode pairs,
    thresholded adjacency graphs, and graph-theory metrics (degree, clustering, global
    efficiency) computed via the optional ``networkx`` package. This is the final (v2)
    version of this analysis, which upgrades an earlier draft's 6 ROI-mean-signal nodes
    to full per-electrode (16 electrodes/ROI) resolution.

    NOTE: this script's group-dict configuration (``PHENOTYPE_ROOTS``) uses generic
    "WT"/"5xFAD" keys as the pattern for defining comparison groups -- edit the paths
    (and, if needed, the phenotype keys/labels) to match your own genotype folder names.

Inputs / outputs (see data/README.md for the full schema):
    Input:  per-ROI, per-condition event-trace folders under
            ``data/event_traces/<group>/<region>/`` containing ``lfp_event_traces.npy``,
            ``csd_event_traces.npy``, and ``event_time_axis_ms.npy`` (one folder per group
            as listed in ``PHENOTYPE_ROOTS``).
    Output: per-electrode correlation matrices, graph-metric tables, and connectivity
            figures under ``data/results/electrode_connectivity/``, plus a
            ``manifest.json`` describing the run.
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
from matplotlib.lines import Line2D

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import networkx as nx
except Exception:
    nx = None

# =============================================================================
# CONFIG
# =============================================================================

# --- EDIT THIS --- (or override via --root_dir / --out_dir CLI flags, see build_argparser)
ROOT_DIR = Path("data/event_traces")
OUT_DIR = Path("data/results/electrode_connectivity")

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

ROI_ORDER = ["M1", "M2", "SSP", "CA1", "CA3", "DG"]
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
ROI_COLORS = {
    "M1": "#4C78A8",
    "M2": "#72B7B2",
    "SSP": "#54A24B",
    "CA1": "#E45756",
    "CA3": "#F58518",
    "DG": "#B279A2",
}

# Electrode-level node settings.
# Your ROI event traces are expected to be electrodes x time. This keeps only the first 16 electrodes.
ELECTRODES_PER_ROI = 16
REQUIRE_FULL_16_ELECTRODES = False     # if True, ROIs with <16 electrodes are skipped
ELECTRODE_GRID_SHAPE = (4, 4)          # for graph layout only

# Correlation preprocessing.
BASELINE_WINDOW_MS = (-50, 0)
ANALYSIS_WINDOW_MS = (0, 250)
USE_ANALYSIS_WINDOW_ONLY = True
SIGNALS_TO_RUN = ["lfp", "csd_abs", "csd_signed"]

# Correlation/graph settings.
CORR_KIND = "pearson"
GRAPH_EDGE_THRESHOLD = 0.30
# For 96 nodes, 25% is usually too dense. Top 5% gives a readable graph.
# Set to None to use GRAPH_EDGE_THRESHOLD only.
GRAPH_TOP_PERCENT: Optional[float] = 5.0
MIN_COMMON_EVENTS_FOR_PAIR = 1
MIN_POINTS_FOR_CORR = 3

# Optional physical coordinate CSV.
# For electrode-level distance curves, use columns: node,x_um,y_um
# or ROI-level columns: roi,x_um,y_um. ROI-level coords will be expanded to mini 4x4 grids.
NODE_COORD_CSV: Optional[Path] = None
ROI_COORD_CSV: Optional[Path] = None
DISTANCE_BIN_UM = 300.0
ELECTRODE_PITCH_UM_FOR_LAYOUT = 20.0

# Figure settings.
FIG_DPI = 600
SAVE_PNG = True
SAVE_SVG = True
SAVE_EXCEL = True
MATRIX_CMAP = "Spectral"
DIFF_CMAP = "coolwarm"
POINT_SIZE = 28
POINT_JITTER = 0.045
EPS = 1e-12

# =============================================================================
# STYLE / IO HELPERS
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
            fig.tight_layout(pad=1.15)
    except Exception:
        pass
    if SAVE_PNG:
        fig.savefig(str(out_base) + ".png", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.10)
    if SAVE_SVG:
        fig.savefig(str(out_base) + ".svg", bbox_inches="tight", pad_inches=0.10)
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
    return s in BASELINE_ALIASES or s in {"base", "baseline", "basal", "control"}


def time_mask(t, window):
    t = np.asarray(t, dtype=float)
    m = (t >= window[0]) & (t <= window[1])
    return m if np.any(m) else np.ones(t.size, dtype=bool)


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


def preprocess_trace(y, t):
    """Baseline subtract and optionally restrict to analysis window."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.size != t.size:
        n = min(y.size, t.size)
        y, t = y[:n], t[:n]
    bm = time_mask(t, BASELINE_WINDOW_MS)
    y = y - np.nanmedian(y[bm])
    if USE_ANALYSIS_WINDOW_ONLY:
        wm = time_mask(t, ANALYSIS_WINDOW_MS)
        y = y[wm]
        t = t[wm]
    return fill_nan_1d(y), t


def pearson_corr(a, b):
    a = fill_nan_1d(np.asarray(a, dtype=float))
    b = fill_nan_1d(np.asarray(b, dtype=float))
    n = min(a.size, b.size)
    if n < MIN_POINTS_FOR_CORR:
        return np.nan, n
    a = a[:n]
    b = b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if np.sum(ok) < MIN_POINTS_FOR_CORR:
        return np.nan, int(np.sum(ok))
    a = a[ok]
    b = b[ok]
    if np.nanstd(a) <= EPS or np.nanstd(b) <= EPS:
        return np.nan, int(ok.sum())
    r = float(np.corrcoef(a, b)[0, 1])
    return r, int(ok.sum())


def fisher_z(r):
    r = np.asarray(r, dtype=float)
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def inv_fisher_z(z):
    return np.tanh(z)


def mean_corr_fisher(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(inv_fisher_z(np.nanmean(fisher_z(vals))))


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan if vals.size == 0 else 0.0
    return float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size))


def welch_pvalue(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if stats is None or a.size < 2 or b.size < 2:
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
# ELECTRODE NODE HELPERS
# =============================================================================

def node_name(roi: str, elec_idx0: int) -> str:
    return f"{canon_roi(roi)}_e{elec_idx0 + 1:02d}"


def parse_node(node: str) -> Tuple[str, int]:
    m = re.match(r"(.+)_e(\d+)$", str(node))
    if not m:
        raise ValueError(f"Bad node name: {node}")
    return canon_roi(m.group(1)), int(m.group(2)) - 1


def node_roi(node: str) -> str:
    return parse_node(node)[0]


def node_region(node: str) -> str:
    return roi_region(node_roi(node))


def make_node_order() -> List[str]:
    nodes = []
    for roi in ROI_ORDER:
        for e in range(ELECTRODES_PER_ROI):
            nodes.append(node_name(roi, e))
    return nodes


def classify_pair_type_from_nodes(node_i: str, node_j: str) -> str:
    roi_i, _ = parse_node(node_i)
    roi_j, _ = parse_node(node_j)
    if roi_i == roi_j:
        return "same_ROI"
    ri = roi_region(roi_i)
    rj = roi_region(roi_j)
    if ri == rj == "CTX":
        return "CTX-CTX"
    if ri == rj == "HIPPO":
        return "HIPPO-HIPPO"
    if {ri, rj} == {"CTX", "HIPPO"}:
        return "CTX-HIPPO"
    return "OTHER"

# =============================================================================
# DATA DISCOVERY / LOADING
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
    if ("app" in compact and ("gf" in compact or "gfp" in compact)) or "appnl-g" in n or "appnl_g" in n:
        return "APPNL_GF"
    if "appnl" in compact or ("app" in compact and "nl" in compact):
        return "APPNL"
    return None


def looks_like_event_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    for roi in ROI_ORDER:
        qnames = {roi, roi.lower()}
        if roi == "SSP":
            qnames.update({"S1", "s1", "ssp"})
        for rname in qnames:
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
            return sorted(roots, key=lambda d: (PHENOTYPE_ORDER.index(d.phenotype), d.name))

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
    return sorted(roots, key=lambda d: (PHENOTYPE_ORDER.index(d.phenotype), d.name))


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


def load_events(datasets: List[DatasetRoot]) -> Tuple[List[dict], pd.DataFrame]:
    events = []
    rows = []
    for ds in datasets:
        for roi_name in ROI_ORDER:
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
                L, T = L[:, :n], T[:n]
                if C is not None:
                    C = C[:, :n]
                if REQUIRE_FULL_16_ELECTRODES and L.shape[0] < ELECTRODES_PER_ROI:
                    print(f"[WARN] skip {ev_dir}: only {L.shape[0]} electrodes, need {ELECTRODES_PER_ROI}")
                    continue
                ev_idx, ev_ms = parse_event_folder_name(ev_dir)
                ev_idx = int(ev_idx) if ev_idx is not None else len(events)
                roi_c = canon_roi(roi_name)
                ev = dict(
                    phenotype=ds.phenotype,
                    dataset=ds.name,
                    root=str(ds.path),
                    roi_name=roi_c,
                    region=roi_region(roi_c),
                    drug="base",
                    event_index=ev_idx,
                    event_ms=float(ev_ms) if ev_ms is not None else np.nan,
                    event_dir=str(ev_dir),
                    event_key=f"{sanitize(ds.name)}__{roi_c}__base__event_{ev_idx:03d}",
                    time_ms=T,
                    lfp=L,
                    csd=C,
                    n_electrodes=int(L.shape[0]),
                    n_electrodes_used=int(min(ELECTRODES_PER_ROI, L.shape[0])),
                )
                events.append(ev)
                rows.append({k: ev[k] for k in [
                    "phenotype", "dataset", "roi_name", "region", "event_index",
                    "event_ms", "event_dir", "event_key", "n_electrodes", "n_electrodes_used"
                ]})
    if not events:
        raise RuntimeError("No events loaded.")
    return events, pd.DataFrame(rows)

# =============================================================================
# ELECTRODE-LEVEL FC CORE
# =============================================================================

def event_electrode_signal(ev: dict, signal: str, elec_idx0: int) -> Optional[np.ndarray]:
    """Return one electrode-level time series for one event."""
    if signal == "lfp":
        L = np.asarray(ev["lfp"], dtype=float)
        if elec_idx0 >= L.shape[0]:
            return None
        return L[elec_idx0]
    if signal == "csd_abs":
        C = ev.get("csd")
        if C is None:
            return None
        C = np.asarray(C, dtype=float)
        if elec_idx0 >= C.shape[0]:
            return None
        return np.abs(C[elec_idx0])
    if signal == "csd_signed":
        C = ev.get("csd")
        if C is None:
            return None
        C = np.asarray(C, dtype=float)
        if elec_idx0 >= C.shape[0]:
            return None
        return C[elec_idx0]
    raise ValueError(signal)


def build_preprocessed_node_event_signals(events: List[dict], signal: str) -> Dict[Tuple[str, str, str], Dict[int, np.ndarray]]:
    """
    Returns:
        data[(phenotype, dataset, node)][event_index] = preprocessed electrode trace
    """
    data: Dict[Tuple[str, str, str], Dict[int, np.ndarray]] = {}
    for ev in events:
        roi = ev["roi_name"]
        n_use = min(ELECTRODES_PER_ROI, int(ev["n_electrodes"]))
        for e in range(n_use):
            y = event_electrode_signal(ev, signal, e)
            if y is None:
                continue
            yy, _ = preprocess_trace(y, ev["time_ms"])
            if yy.size < MIN_POINTS_FOR_CORR:
                continue
            node = node_name(roi, e)
            key = (ev["phenotype"], ev["dataset"], node)
            data.setdefault(key, {})[int(ev["event_index"])] = yy
    return data


def compute_pair_corr_for_dataset_electrodes(events: List[dict], signal: str, nodes: List[str]) -> pd.DataFrame:
    """
    Pairwise electrode-node correlation per dataset.

    For each dataset and electrode-node pair:
      - identify common event indices
      - concatenate matched, baseline-subtracted event snippets
      - compute Pearson correlation rho_ij
    """
    rows = []
    pre = build_preprocessed_node_event_signals(events, signal)
    datasets = sorted({(ev["phenotype"], ev["dataset"]) for ev in events}, key=lambda x: (PHENOTYPE_ORDER.index(x[0]), x[1]))

    for phenotype, dataset in datasets:
        print(f"[FC] {signal} | {phenotype} | {dataset} | {len(nodes)} electrode nodes")
        for i, ni in enumerate(nodes):
            roi_i, elec_i = parse_node(ni)
            evs_i = pre.get((phenotype, dataset, ni), {})
            if not evs_i:
                continue
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]
                roi_j, elec_j = parse_node(nj)
                evs_j = pre.get((phenotype, dataset, nj), {})
                if not evs_j:
                    continue
                common = sorted(set(evs_i.keys()) & set(evs_j.keys()))
                if len(common) < MIN_COMMON_EVENTS_FOR_PAIR:
                    r, n_points, used = np.nan, 0, 0
                else:
                    yi_all, yj_all = [], []
                    used = 0
                    for ev_idx in common:
                        yi = evs_i[ev_idx]
                        yj = evs_j[ev_idx]
                        n = min(yi.size, yj.size)
                        if n >= MIN_POINTS_FOR_CORR:
                            yi_all.append(yi[:n])
                            yj_all.append(yj[:n])
                            used += 1
                    if used >= MIN_COMMON_EVENTS_FOR_PAIR:
                        r, n_points = pearson_corr(np.concatenate(yi_all), np.concatenate(yj_all))
                    else:
                        r, n_points = np.nan, 0
                rows.append({
                    "signal": signal,
                    "phenotype": phenotype,
                    "dataset": dataset,
                    "node_i": ni,
                    "node_j": nj,
                    "roi_i": roi_i,
                    "roi_j": roi_j,
                    "elec_i": int(elec_i + 1),
                    "elec_j": int(elec_j + 1),
                    "region_i": roi_region(roi_i),
                    "region_j": roi_region(roi_j),
                    "pair_type": classify_pair_type_from_nodes(ni, nj),
                    "roi_pair": f"{roi_i}-{roi_j}" if roi_i <= roi_j else f"{roi_j}-{roi_i}",
                    "pcc": r,
                    "abs_pcc": abs(r) if np.isfinite(r) else np.nan,
                    "positive_pcc": max(r, 0) if np.isfinite(r) else np.nan,
                    "n_common_events": len(common),
                    "n_used_events": used,
                    "n_points": n_points,
                })
    return pd.DataFrame(rows)


def pair_table_to_matrix(pair_df: pd.DataFrame, phenotype: str, signal: str, nodes: List[str]) -> np.ndarray:
    M = np.eye(len(nodes), dtype=float)
    M[:] = np.nan
    np.fill_diagonal(M, 1.0)
    sub = pair_df[(pair_df["phenotype"] == phenotype) & (pair_df["signal"] == signal)]
    node_to_i = {n: i for i, n in enumerate(nodes)}
    for _, r in sub.iterrows():
        ni, nj = r["node_i"], r["node_j"]
        if ni not in node_to_i or nj not in node_to_i:
            continue
        i, j = node_to_i[ni], node_to_i[nj]
        # one value per dataset in sub; phenotype-level matrix = Fisher mean across datasets
        # handled later by collecting values here from all rows matching node pair.
    for i, ni in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            nj = nodes[j]
            vals = sub.loc[(sub["node_i"] == ni) & (sub["node_j"] == nj), "pcc"].to_numpy(dtype=float)
            val = mean_corr_fisher(vals)
            M[i, j] = M[j, i] = val
    return M


def build_matrix_summary(pair_df: pd.DataFrame, nodes: List[str]) -> pd.DataFrame:
    rows = []
    for signal in sorted(pair_df["signal"].unique()):
        for phenotype in PHENOTYPE_ORDER:
            sub = pair_df[(pair_df["signal"] == signal) & (pair_df["phenotype"] == phenotype)]
            for i, ni in enumerate(nodes):
                for j in range(i + 1, len(nodes)):
                    nj = nodes[j]
                    vals = sub.loc[(sub["node_i"] == ni) & (sub["node_j"] == nj), "pcc"].to_numpy(dtype=float)
                    if not np.any(np.isfinite(vals)):
                        continue
                    roi_i, ei = parse_node(ni)
                    roi_j, ej = parse_node(nj)
                    rows.append({
                        "signal": signal,
                        "phenotype": phenotype,
                        "node_i": ni,
                        "node_j": nj,
                        "roi_i": roi_i,
                        "roi_j": roi_j,
                        "elec_i": ei + 1,
                        "elec_j": ej + 1,
                        "pair_type": classify_pair_type_from_nodes(ni, nj),
                        "mean_pcc_fisher": mean_corr_fisher(vals),
                        "mean_abs_pcc": float(np.nanmean(np.abs(vals))) if np.any(np.isfinite(vals)) else np.nan,
                        "n_datasets": int(np.sum(np.isfinite(vals))),
                    })
    return pd.DataFrame(rows)

# =============================================================================
# GRAPH METRICS
# =============================================================================

def adjacency_from_matrix(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return binary adjacency A, weighted W, and threshold used."""
    R = np.asarray(M, dtype=float).copy()
    np.fill_diagonal(R, np.nan)
    P = np.where(np.isfinite(R), np.maximum(R, 0), np.nan)

    if GRAPH_TOP_PERCENT is not None:
        vals = P[np.isfinite(P)]
        vals = vals[vals > 0]
        if vals.size:
            thr = float(np.nanpercentile(vals, 100.0 - GRAPH_TOP_PERCENT))
            thr = max(thr, GRAPH_EDGE_THRESHOLD)
        else:
            thr = GRAPH_EDGE_THRESHOLD
    else:
        thr = GRAPH_EDGE_THRESHOLD

    W = np.where(np.isfinite(P) & (P >= thr), P, 0.0)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    A = (W > 0).astype(int)
    return A, W, thr


def clustering_coefficients_binary(A: np.ndarray) -> np.ndarray:
    """Manual local clustering coefficient: C_i = 2 e_i / [k_i(k_i-1)]."""
    A = np.asarray(A, dtype=int)
    n = A.shape[0]
    C = np.zeros(n, dtype=float)
    for i in range(n):
        nbr = np.where(A[i] > 0)[0]
        k = nbr.size
        if k < 2:
            C[i] = 0.0
            continue
        sub = A[np.ix_(nbr, nbr)]
        e = np.sum(np.triu(sub, 1))
        C[i] = (2.0 * e) / (k * (k - 1))
    return C


def global_efficiency_binary(A: np.ndarray) -> float:
    if nx is not None:
        try:
            G = nx.from_numpy_array(np.asarray(A, dtype=int))
            return float(nx.global_efficiency(G))
        except Exception:
            pass
    A = np.asarray(A, dtype=int)
    n = A.shape[0]
    dist = np.full((n, n), np.inf)
    dist[A > 0] = 1.0
    np.fill_diagonal(dist, 0.0)
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    inv = np.where((dist > 0) & np.isfinite(dist), 1.0 / dist, 0.0)
    return float(np.sum(inv) / (n * (n - 1))) if n > 1 else np.nan


def dataset_matrix(pair_df: pd.DataFrame, phenotype: str, dataset: str, signal: str, nodes: List[str]) -> np.ndarray:
    M = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
    np.fill_diagonal(M, 1.0)
    node_to_i = {n: i for i, n in enumerate(nodes)}
    sub = pair_df[(pair_df["signal"] == signal) & (pair_df["phenotype"] == phenotype) & (pair_df["dataset"] == dataset)]
    for _, r in sub.iterrows():
        ni, nj = r["node_i"], r["node_j"]
        if ni in node_to_i and nj in node_to_i:
            i, j = node_to_i[ni], node_to_i[nj]
            val = float(r["pcc"]) if np.isfinite(r["pcc"]) else np.nan
            M[i, j] = M[j, i] = val
    return M


def graph_metrics_from_pair_df(pair_df: pd.DataFrame, nodes: List[str]) -> pd.DataFrame:
    rows = []
    for signal in sorted(pair_df["signal"].unique()):
        for phenotype in PHENOTYPE_ORDER:
            M = pair_table_to_matrix(pair_df, phenotype, signal, nodes)
            A, W, thr = adjacency_from_matrix(M)
            deg = np.sum(A, axis=1)
            strength = np.sum(W, axis=1)
            cc = clustering_coefficients_binary(A)
            density = float(np.sum(np.triu(A, 1)) / (A.shape[0] * (A.shape[0] - 1) / 2.0))
            ge = global_efficiency_binary(A)
            for i, node in enumerate(nodes):
                roi, elec = parse_node(node)
                rows.append({
                    "scope": "phenotype_mean_graph",
                    "signal": signal,
                    "phenotype": phenotype,
                    "dataset": "PHENOTYPE_MEAN",
                    "node": node,
                    "roi": roi,
                    "electrode": elec + 1,
                    "region": roi_region(roi),
                    "threshold_used": thr,
                    "degree_k": float(deg[i]),
                    "node_strength": float(strength[i]),
                    "clustering_coefficient": float(cc[i]),
                    "network_density": density,
                    "global_efficiency": ge,
                })

            for dataset in sorted(pair_df.loc[(pair_df["signal"] == signal) & (pair_df["phenotype"] == phenotype), "dataset"].unique()):
                Mds = dataset_matrix(pair_df, phenotype, dataset, signal, nodes)
                Ads, Wds, thrds = adjacency_from_matrix(Mds)
                degds = np.sum(Ads, axis=1)
                strds = np.sum(Wds, axis=1)
                ccds = clustering_coefficients_binary(Ads)
                densityds = float(np.sum(np.triu(Ads, 1)) / (Ads.shape[0] * (Ads.shape[0] - 1) / 2.0))
                geds = global_efficiency_binary(Ads)
                for i, node in enumerate(nodes):
                    roi, elec = parse_node(node)
                    rows.append({
                        "scope": "dataset_graph",
                        "signal": signal,
                        "phenotype": phenotype,
                        "dataset": dataset,
                        "node": node,
                        "roi": roi,
                        "electrode": elec + 1,
                        "region": roi_region(roi),
                        "threshold_used": thrds,
                        "degree_k": float(degds[i]),
                        "node_strength": float(strds[i]),
                        "clustering_coefficient": float(ccds[i]),
                        "network_density": densityds,
                        "global_efficiency": geds,
                    })
    return pd.DataFrame(rows)

# =============================================================================
# PLOTTING
# =============================================================================

def nice_signal_name(signal: str) -> str:
    return {
        "lfp": "LFP",
        "csd_abs": "|CSD|",
        "csd_signed": "signed CSD",
    }.get(signal, signal)


def roi_boundaries(nodes: List[str]) -> Tuple[List[int], List[float], List[str]]:
    centers = []
    bounds = []
    labels = []
    start = 0
    rois = [node_roi(n) for n in nodes]
    for roi in ROI_ORDER:
        idx = [i for i, r in enumerate(rois) if r == roi]
        if not idx:
            continue
        centers.append((min(idx) + max(idx)) / 2)
        labels.append(roi)
        bounds.append(max(idx) + 0.5)
    return list(range(len(nodes))), centers, labels


def draw_roi_grid_lines(ax, nodes: List[str], color="white", lw=0.7):
    _, _, _ = roi_boundaries(nodes)
    for b in range(ELECTRODES_PER_ROI, len(nodes), ELECTRODES_PER_ROI):
        ax.axhline(b - 0.5, color=color, lw=lw)
        ax.axvline(b - 0.5, color=color, lw=lw)


def plot_fc_matrices(pair_df: pd.DataFrame, signal: str, nodes: List[str], out_base: Path):
    matrices = {p: pair_table_to_matrix(pair_df, p, signal, nodes) for p in PHENOTYPE_ORDER}
    vals = []
    for M in matrices.values():
        x = M[np.triu_indices_from(M, 1)]
        vals.extend(x[np.isfinite(x)].tolist())
    vmin = 0.0 if vals and np.nanmin(vals) >= 0 else -1.0
    vmax = 1.0
    _, centers, labels = roi_boundaries(nodes)

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.55), constrained_layout=True)
    ims = []
    for ax, phenotype in zip(axes, PHENOTYPE_ORDER):
        M = matrices[phenotype]
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=MATRIX_CMAP, interpolation="nearest", rasterized=True)
        ims.append(im)
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(f"{PHENOTYPE_DISPLAY[phenotype]}\n{nice_signal_name(signal)} electrode-level FC", fontsize=10)
        draw_roi_grid_lines(ax, nodes, color="white", lw=0.65)
        ax.set_xlabel("16 electrodes per ROI")
        ax.set_ylabel("16 electrodes per ROI")
    cb = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cb.set_label("Pearson correlation coefficient")
    fig.suptitle(f"Event-centered baseline functional connectivity | {nice_signal_name(signal)} | 16 electrodes/ROI", fontsize=11)
    save_fig(fig, out_base)


def plot_difference_matrix(pair_df: pd.DataFrame, signal: str, nodes: List[str], out_base: Path):
    A = pair_table_to_matrix(pair_df, PHENOTYPE_ORDER[0], signal, nodes)
    B = pair_table_to_matrix(pair_df, PHENOTYPE_ORDER[1], signal, nodes)
    D = B - A
    np.fill_diagonal(D, 0.0)
    vmax = np.nanpercentile(np.abs(D[np.isfinite(D)]), 95) if np.any(np.isfinite(D)) else 1.0
    vmax = max(float(vmax), 0.05)
    _, centers, labels = roi_boundaries(nodes)
    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.45), constrained_layout=True)
    im = ax.imshow(D, vmin=-vmax, vmax=vmax, cmap=DIFF_CMAP, interpolation="nearest", rasterized=True)
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    draw_roi_grid_lines(ax, nodes, color="black", lw=0.45)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label(r"Δ PCC: APP$^{NL-G-F}$ - APP$^{NL}$")
    ax.set_title(f"Phenotype-difference electrode FC | {nice_signal_name(signal)}", fontsize=10)
    save_fig(fig, out_base)


def dataset_pairtype_summary(pair_df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Animal/session-level means for valid stats without electrode-pair pseudoreplication."""
    sub = pair_df[pair_df["signal"] == signal].copy()
    rows = []
    for (phenotype, dataset, pt), g in sub.groupby(["phenotype", "dataset", "pair_type"]):
        vals = g["pcc"].to_numpy(dtype=float)
        rows.append({
            "signal": signal,
            "phenotype": phenotype,
            "dataset": dataset,
            "pair_type": pt,
            "n_electrode_pairs": int(np.sum(np.isfinite(vals))),
            "mean_pcc_fisher": mean_corr_fisher(vals),
            "mean_abs_pcc": float(np.nanmean(np.abs(vals))) if np.any(np.isfinite(vals)) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_pair_type_boxplots(pair_df: pd.DataFrame, signal: str, out_base: Path):
    summ = dataset_pairtype_summary(pair_df, signal)
    pair_types = ["same_ROI", "CTX-CTX", "HIPPO-HIPPO", "CTX-HIPPO"]
    set_style()
    fig, ax = plt.subplots(figsize=(6.9, 3.55), constrained_layout=True)
    width = 0.32
    stat_rows = []
    for pi, pt in enumerate(pair_types):
        vals_by_p = []
        for k, phenotype in enumerate(PHENOTYPE_ORDER):
            vals = summ.loc[(summ["pair_type"] == pt) & (summ["phenotype"] == phenotype), "mean_pcc_fisher"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            x = pi + (k - 0.5) * width
            bp = ax.boxplot(vals, positions=[x], widths=width*0.82, patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set_facecolor(PHENOTYPE_COLORS[phenotype])
                box.set_alpha(0.28)
                box.set_edgecolor(PHENOTYPE_COLORS[phenotype])
            for key in ["whiskers", "caps", "medians"]:
                for item in bp[key]:
                    item.set_color(PHENOTYPE_COLORS[phenotype])
                    item.set_linewidth(1.0)
            if vals.size:
                jitter = np.linspace(-POINT_JITTER, POINT_JITTER, vals.size) if vals.size > 1 else np.array([0])
                ax.scatter(np.full(vals.size, x)+jitter, vals, s=POINT_SIZE,
                           color=PHENOTYPE_COLORS[phenotype], alpha=0.8, zorder=3)
            vals_by_p.append(vals)
            stat_rows.append({
                "signal": signal, "pair_type": pt, "phenotype": phenotype,
                "n_datasets": int(vals.size), "mean_of_dataset_mean_pcc": float(np.nanmean(vals)) if vals.size else np.nan,
                "sem": sem(vals),
            })
        pval = welch_pvalue(vals_by_p[0], vals_by_p[1])
        stat_rows.append({
            "signal": signal, "pair_type": pt,
            "comparison": f"{PHENOTYPE_ORDER[0]}_vs_{PHENOTYPE_ORDER[1]}",
            "p_value": pval, "test": "Welch t-test on dataset-level mean electrode-pair PCC",
        })
        allv = np.concatenate([v for v in vals_by_p if v.size]) if any(v.size for v in vals_by_p) else np.array([])
        if allv.size:
            y = np.nanmax(allv)
            ax.set_ylim(top=max(ax.get_ylim()[1], y + 0.25 * max(0.1, abs(y))))
            add_sig_bar(ax, pi - width/2, pi + width/2, y + 0.07 * max(0.1, abs(y)), pval)
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xticks(range(len(pair_types)))
    ax.set_xticklabels(["same ROI", "CTX-CTX", "HIPPO-HIPPO", "CTX-HIPPO"])
    ax.set_ylabel("Dataset mean electrode-pair PCC")
    ax.set_title(f"Electrode-level region-pair FC summary | {nice_signal_name(signal)}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [Line2D([0], [0], color=PHENOTYPE_COLORS[p], lw=6, alpha=0.35) for p in PHENOTYPE_ORDER]
    ax.legend(handles, [PHENOTYPE_DISPLAY[p] for p in PHENOTYPE_ORDER], frameon=False, loc="best")
    save_fig(fig, out_base)
    save_df(pd.DataFrame(stat_rows), Path(str(out_base) + "_summary"))
    save_df(summ, Path(str(out_base) + "_dataset_pairtype_values"))


def plot_roi_pair_delta_heat(pair_df: pd.DataFrame, signal: str, out_base: Path):
    """6x6 ROI-pair summary averaged from all 16x16 electrode-pair correlations."""
    sub = pair_df[pair_df["signal"] == signal].copy()
    D = np.full((len(ROI_ORDER), len(ROI_ORDER)), np.nan, dtype=float)
    P = np.full_like(D, np.nan)
    rows = []
    for i, roi_i in enumerate(ROI_ORDER):
        for j, roi_j in enumerate(ROI_ORDER):
            if j < i:
                continue
            if i == j:
                mask_pair = (sub["roi_i"] == roi_i) & (sub["roi_j"] == roi_j)
            else:
                mask_pair = ((sub["roi_i"] == roi_i) & (sub["roi_j"] == roi_j)) | ((sub["roi_i"] == roi_j) & (sub["roi_j"] == roi_i))
            # dataset-level means per ROI pair
            vals_a, vals_b = [], []
            for phenotype, vals_out in [(PHENOTYPE_ORDER[0], vals_a), (PHENOTYPE_ORDER[1], vals_b)]:
                for dataset, g in sub[(sub["phenotype"] == phenotype) & mask_pair].groupby("dataset"):
                    vals_out.append(mean_corr_fisher(g["pcc"].to_numpy(dtype=float)))
            vals_a = np.asarray(vals_a, dtype=float)
            vals_b = np.asarray(vals_b, dtype=float)
            ma = float(np.nanmean(vals_a)) if np.any(np.isfinite(vals_a)) else np.nan
            mb = float(np.nanmean(vals_b)) if np.any(np.isfinite(vals_b)) else np.nan
            pval = welch_pvalue(vals_a, vals_b)
            delta = mb - ma if np.isfinite(ma) and np.isfinite(mb) else np.nan
            D[i, j] = D[j, i] = delta
            P[i, j] = P[j, i] = pval
            rows.append({
                "signal": signal, "roi_i": roi_i, "roi_j": roi_j,
                "pair_type": "same_ROI" if roi_i == roi_j else classify_pair_type_from_nodes(node_name(roi_i, 0), node_name(roi_j, 0)),
                "APPNL_dataset_mean_pcc": ma,
                "APPNL_GF_dataset_mean_pcc": mb,
                "delta_APPNL_GF_minus_APPNL": delta,
                "p_value": pval,
                "p_label": p_to_label(pval),
                "n_APPNL_datasets": int(np.sum(np.isfinite(vals_a))),
                "n_APPNL_GF_datasets": int(np.sum(np.isfinite(vals_b))),
            })
    save_df(pd.DataFrame(rows), Path(str(out_base) + "_roi_pair_stats"))

    vmax = np.nanpercentile(np.abs(D[np.isfinite(D)]), 95) if np.any(np.isfinite(D)) else 1.0
    vmax = max(float(vmax), 0.05)
    set_style()
    fig, ax = plt.subplots(figsize=(3.95, 3.45), constrained_layout=True)
    im = ax.imshow(D, vmin=-vmax, vmax=vmax, cmap=DIFF_CMAP, interpolation="nearest")
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_yticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(ROI_ORDER)
    for i in range(len(ROI_ORDER)):
        for j in range(len(ROI_ORDER)):
            val = D[i, j]
            if np.isfinite(val):
                lab = p_to_label(P[i, j]) if np.isfinite(P[i, j]) and P[i, j] < 0.05 else ""
                ax.text(j, i, f"{val:+.2f}\n{lab}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.5*vmax else "black")
    ax.axhline(2.5, color="black", lw=0.8)
    ax.axvline(2.5, color="black", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cb.set_label(r"Δ mean electrode PCC")
    ax.set_title(f"ROI-pair ΔFC from 16x16 electrode links | {nice_signal_name(signal)}", fontsize=10)
    save_fig(fig, out_base)


def roi_graph_centers():
    return {
        "M1": (-1.2, 1.0),
        "M2": (0.0, 1.18),
        "SSP": (1.2, 1.0),
        "CA1": (-1.2, 0.0),
        "CA3": (0.0, -0.18),
        "DG": (1.2, 0.0),
    }


def graph_positions_16(nodes: List[str]) -> Dict[str, Tuple[float, float]]:
    centers = roi_graph_centers()
    nrow, ncol = ELECTRODE_GRID_SHAPE
    dx, dy = 0.075, 0.075
    pos = {}
    for node in nodes:
        roi, e = parse_node(node)
        cx, cy = centers[roi]
        rr = e // ncol
        cc = e % ncol
        x = cx + (cc - (ncol - 1)/2) * dx
        y = cy - (rr - (nrow - 1)/2) * dy
        pos[node] = (x, y)
    return pos


def plot_network_graphs(pair_df: pd.DataFrame, signal: str, nodes: List[str], out_base: Path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)
    pos = graph_positions_16(nodes)
    for ax, phenotype in zip(axes, PHENOTYPE_ORDER):
        M = pair_table_to_matrix(pair_df, phenotype, signal, nodes)
        A, W, thr = adjacency_from_matrix(M)
        ax.set_title(f"{PHENOTYPE_DISPLAY[phenotype]}\n{nice_signal_name(signal)} 16-electrode graph", fontsize=10)
        max_w = np.nanmax(W) if np.any(W > 0) else 1.0
        # Draw edges first. For readability only retained high-confidence positive links are drawn.
        for i, ni in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                if A[i, j] == 0:
                    continue
                nj = nodes[j]
                x1, y1 = pos[ni]
                x2, y2 = pos[nj]
                w = W[i, j]
                lw = 0.15 + 1.8 * (w / max_w)
                alpha = 0.04 + 0.30 * (w / max_w)
                ri = node_region(ni)
                rj = node_region(nj)
                color = "0.25" if ri == rj else "#8c510a"
                ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=1)
        # ROI hull labels and electrode points.
        for roi, (cx, cy) in roi_graph_centers().items():
            ax.text(cx, cy + 0.23, roi, ha="center", va="bottom", fontsize=8, weight="bold")
        for node in nodes:
            roi, e = parse_node(node)
            x, y = pos[node]
            ax.scatter([x], [y], s=28, color=ROI_COLORS[roi], edgecolor="black", lw=0.25, zorder=3)
        ax.text(0, -0.72,
                f"nodes={len(nodes)} | edge threshold≥{thr:.2f} | top={GRAPH_TOP_PERCENT}% positive links" if GRAPH_TOP_PERCENT else f"nodes={len(nodes)} | threshold≥{thr:.2f}",
                ha="center", va="center", fontsize=7)
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-0.9, 1.55)
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ROI_COLORS[r],
                      markeredgecolor="black", markersize=7, label=r) for r in ROI_ORDER]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Thresholded electrode-level functional network | {nice_signal_name(signal)}", fontsize=11)
    save_fig(fig, out_base)


def plot_graph_metric_bars(metric_df: pd.DataFrame, signal: str, out_base: Path):
    sub = metric_df[(metric_df["signal"] == signal) & (metric_df["scope"] == "dataset_graph")].copy()
    if sub.empty:
        return
    metrics = [
        ("degree_k", "Degree k"),
        ("node_strength", "Node strength"),
        ("clustering_coefficient", "Clustering coefficient"),
        ("network_density", "Network density"),
        ("global_efficiency", "Global efficiency"),
    ]
    set_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(3.0*len(metrics), 10), 3.45), constrained_layout=True)
    stat_rows = []
    for ax, (metric, label) in zip(axes, metrics):
        vals_for_test = []
        for pi, phenotype in enumerate(PHENOTYPE_ORDER):
            # For node metrics, first average across all 96 electrode nodes within each dataset.
            vals = sub[sub["phenotype"] == phenotype].groupby("dataset")[metric].mean().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            vals_for_test.append(vals)
            x = pi
            mean = float(np.nanmean(vals)) if vals.size else np.nan
            s = sem(vals)
            ax.bar(x, mean, yerr=s, width=0.55, color=PHENOTYPE_COLORS[phenotype], alpha=0.28,
                   edgecolor=PHENOTYPE_COLORS[phenotype], linewidth=1.2, capsize=2)
            if vals.size:
                jitter = np.linspace(-POINT_JITTER, POINT_JITTER, vals.size) if vals.size > 1 else np.array([0])
                ax.scatter(np.full(vals.size, x)+jitter, vals, s=POINT_SIZE,
                           color=PHENOTYPE_COLORS[phenotype], alpha=0.8, zorder=3)
            stat_rows.append({
                "signal": signal, "metric": metric, "phenotype": phenotype,
                "n_datasets": int(vals.size), "mean": mean, "sem": s,
                "note": "dataset mean across electrode nodes",
            })
        pval = welch_pvalue(vals_for_test[0], vals_for_test[1])
        stat_rows.append({"signal": signal, "metric": metric, "comparison": "APPNL_vs_APPNL_GF", "p_value": pval, "test": "Welch t-test on dataset-level graph metric"})
        allv = np.concatenate([v for v in vals_for_test if v.size]) if any(v.size for v in vals_for_test) else np.array([])
        if allv.size:
            y = np.nanmax(allv)
            ax.set_ylim(top=max(ax.get_ylim()[1], y + 0.25 * max(0.1, abs(y))))
            add_sig_bar(ax, 0, 1, y + 0.07 * max(0.1, abs(y)), pval)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([PHENOTYPE_DISPLAY[p] for p in PHENOTYPE_ORDER], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Electrode-level graph topology metrics | {nice_signal_name(signal)}", fontsize=11)
    save_fig(fig, out_base)
    save_df(pd.DataFrame(stat_rows), Path(str(out_base) + "_summary"))


def plot_hierarchy_scale(pair_df: pd.DataFrame, signal: str, out_base: Path):
    summ = dataset_pairtype_summary(pair_df, signal)
    order = ["same_ROI", "CTX-CTX", "HIPPO-HIPPO", "CTX-HIPPO"]
    set_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    for phenotype in PHENOTYPE_ORDER:
        means, sems = [], []
        for pt in order:
            vals = summ[(summ["phenotype"] == phenotype) & (summ["pair_type"] == pt)]["mean_pcc_fisher"].to_numpy(dtype=float)
            means.append(float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan)
            sems.append(sem(vals))
        ax.errorbar(range(len(order)), means, yerr=sems, marker="o", lw=1.5, capsize=2,
                    color=PHENOTYPE_COLORS[phenotype], label=PHENOTYPE_DISPLAY[phenotype])
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["same ROI", "Intra-CTX", "Intra-HIPPO", "CTX-HIPPO"])
    ax.set_ylabel("Dataset mean electrode-pair PCC")
    ax.set_title(f"Circuit-scale interaction profile | {nice_signal_name(signal)}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    save_fig(fig, out_base)

# =============================================================================
# OPTIONAL DISTANCE ANALYSIS
# =============================================================================

def load_node_coords(nodes: List[str]) -> Optional[pd.DataFrame]:
    if NODE_COORD_CSV is not None and Path(NODE_COORD_CSV).exists():
        df = pd.read_csv(NODE_COORD_CSV)
        required = {"node", "x_um", "y_um"}
        if required.issubset(set(df.columns)):
            return df[df["node"].isin(nodes)].copy()
        print(f"[WARN] NODE_COORD_CSV must contain {required}; trying ROI_COORD_CSV.")

    if ROI_COORD_CSV is not None and Path(ROI_COORD_CSV).exists():
        rdf = pd.read_csv(ROI_COORD_CSV)
        required = {"roi", "x_um", "y_um"}
        if not required.issubset(set(rdf.columns)):
            print(f"[WARN] ROI_COORD_CSV must contain {required}; skipping physical distance analysis.")
            return None
        rdf["roi"] = rdf["roi"].map(canon_roi)
        centers = {r["roi"]: (float(r["x_um"]), float(r["y_um"])) for _, r in rdf.iterrows()}
        nrow, ncol = ELECTRODE_GRID_SHAPE
        rows = []
        for node in nodes:
            roi, e = parse_node(node)
            if roi not in centers:
                continue
            cx, cy = centers[roi]
            rr = e // ncol
            cc = e % ncol
            x = cx + (cc - (ncol - 1)/2) * ELECTRODE_PITCH_UM_FOR_LAYOUT
            y = cy - (rr - (nrow - 1)/2) * ELECTRODE_PITCH_UM_FOR_LAYOUT
            rows.append({"node": node, "x_um": x, "y_um": y})
        return pd.DataFrame(rows) if rows else None

    return None


def make_distance_table(pair_df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    xy = {r["node"]: (float(r["x_um"]), float(r["y_um"])) for _, r in coords.iterrows()}
    rows = []
    for _, r in pair_df.iterrows():
        if r["node_i"] not in xy or r["node_j"] not in xy:
            continue
        x1, y1 = xy[r["node_i"]]
        x2, y2 = xy[r["node_j"]]
        d = math.hypot(x2-x1, y2-y1)
        rr = r.to_dict()
        rr["distance_um"] = d
        rr["distance_bin_um"] = DISTANCE_BIN_UM * math.floor(d / DISTANCE_BIN_UM)
        rows.append(rr)
    return pd.DataFrame(rows)


def plot_distance_decay(dist_df: pd.DataFrame, signal: str, out_base: Path):
    sub = dist_df[dist_df["signal"] == signal].copy()
    if sub.empty:
        return
    # dataset-level bin means first
    rows = []
    for (phenotype, dataset, b), g in sub.groupby(["phenotype", "dataset", "distance_bin_um"]):
        rows.append({
            "phenotype": phenotype,
            "dataset": dataset,
            "distance_bin_um": float(b),
            "mean_pcc_fisher": mean_corr_fisher(g["pcc"].to_numpy(dtype=float)),
        })
    s = pd.DataFrame(rows)
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for phenotype in PHENOTYPE_ORDER:
        dsub = s[s["phenotype"] == phenotype]
        bins = sorted(dsub["distance_bin_um"].dropna().unique())
        xs, ys, es = [], [], []
        for b in bins:
            vals = dsub[dsub["distance_bin_um"] == b]["mean_pcc_fisher"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            xs.append(float(b + DISTANCE_BIN_UM/2))
            ys.append(float(np.nanmean(vals)))
            es.append(sem(vals))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", lw=1.5, capsize=2,
                        color=PHENOTYPE_COLORS[phenotype], label=PHENOTYPE_DISPLAY[phenotype])
    ax.axhline(0, color="0.65", lw=0.8)
    ax.set_xlabel("Electrode/node distance (um)")
    ax.set_ylabel("Dataset mean PCC")
    ax.set_title(f"Distance-dependent electrode FC | {nice_signal_name(signal)}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    save_fig(fig, out_base)

# =============================================================================
# METHOD / PLAN OUTPUT
# =============================================================================

def write_methods_and_plan(nodes: List[str]):
    out = ensure_dir(OUT_DIR / "05_methods_and_plan")
    text = f"""Electrode-level functional-network-connectivity analysis plan
================================================================

Biological comparison
---------------------
Phenotypes:
  1. APPNL
  2. APPNL-GF

Major anatomical areas:
  CTX   = M1, M2, SSP/S1
  HIPPO = CA1, CA3, DG

Node definition
---------------
Each ROI is represented by up to {ELECTRODES_PER_ROI} individual electrode traces, not by one ROI average.
Total expected nodes:
  {len(ROI_ORDER)} ROIs x {ELECTRODES_PER_ROI} electrodes = {len(nodes)} nodes

Node names:
  M1_e01 ... M1_e16, M2_e01 ... M2_e16, SSP_e01 ... SSP_e16,
  CA1_e01 ... CA1_e16, CA3_e01 ... CA3_e16, DG_e01 ... DG_e16

Core analysis logic copied from the manuscript-style framework
--------------------------------------------------------------
1. Functional connectivity matrix
   For each signal type and electrode-node pair, compute:

       rho_ij = corr(x_i, x_j)

   where x_i and x_j are event-centered baseline-subtracted electrode time series.
   Matched events are concatenated before correlation.

2. Phenotype-level matrix
   Dataset/session-level correlations are averaged using Fisher-z transform:

       z = atanh(rho)
       mean_rho = tanh(mean(z))

3. Network construction
   A positive weighted graph is built from the phenotype matrix.
   Edge rule:
       top percent = {GRAPH_TOP_PERCENT}
       minimum PCC threshold = {GRAPH_EDGE_THRESHOLD}

4. Graph metrics
   Degree:
       k_i = sum_j a_ij

   Weighted strength:
       s_i = sum_j w_ij

   Local clustering coefficient:
       C_i = 2 e_i / [k_i (k_i - 1)]

   Density and global efficiency are also saved.

5. Circuit-scale organization
   Electrode pairs are grouped into:
       same ROI
       CTX-CTX
       HIPPO-HIPPO
       CTX-HIPPO

Figure plan
-----------
Fig. FC1. LFP-based 96-node electrode functional-connectivity matrix
  a. APPNL matrix
  b. APPNL-GF matrix
  c. APPNL-GF minus APPNL difference matrix

Fig. FC2. CSD-based 96-node electrode functional-connectivity matrix
  a. |CSD| matrix
  b. signed CSD matrix
  c. phenotype-difference matrices

Fig. FC3. Region-pair quantification from electrode pairs
  a. same-ROI local coupling
  b. intra-cortical coupling
  c. intra-hippocampal coupling
  d. cortico-hippocampal coupling

Fig. FC4. Electrode-level functional network topology
  a. APPNL 96-node graph
  b. APPNL-GF 96-node graph
  c. degree / strength / clustering / density / efficiency

Fig. FC5. Circuit-scale interaction profile
  a. Mean PCC across same-ROI, intra-CTX, intra-HIPPO, CTX-HIPPO pair classes
  b. Optional physical distance-dependent PCC decay if coordinates are supplied

Recommended interpretation wording
----------------------------------
Use: "event-centered baseline functional connectivity".
Do not call it whole-recording spontaneous FC unless the input traces are continuous raw baseline recordings.

Suggested main title
--------------------
APPNL-GF reshapes electrode-level cortico-hippocampal functional connectivity

Suggested subtitle
------------------
Sixteen-electrode ROI networks reveal altered local, intra-area, and cortico-hippocampal baseline coupling.
"""
    with open(out / "functional_connectivity_16electrode_methods_and_figure_plan.txt", "w", encoding="utf-8") as f:
        f.write(text)

# =============================================================================
# MAIN
# =============================================================================

def save_manifest(datasets: List[DatasetRoot], event_meta: pd.DataFrame, nodes: List[str]):
    manifest = {
        "target": "APPNL vs APPNL-GF event-centered baseline electrode-level functional connectivity",
        "root_dir": str(ROOT_DIR.resolve()),
        "out_dir": str(OUT_DIR.resolve()),
        "phenotype_order": PHENOTYPE_ORDER,
        "roi_order": ROI_ORDER,
        "electrodes_per_roi": ELECTRODES_PER_ROI,
        "n_nodes_expected": len(nodes),
        "node_order": nodes,
        "roi_region_map": ROI_REGION_MAP,
        "signals": SIGNALS_TO_RUN,
        "baseline_window_ms": BASELINE_WINDOW_MS,
        "analysis_window_ms": ANALYSIS_WINDOW_MS,
        "use_analysis_window_only": USE_ANALYSIS_WINDOW_ONLY,
        "corr_kind": CORR_KIND,
        "graph_edge_threshold": GRAPH_EDGE_THRESHOLD,
        "graph_top_percent": GRAPH_TOP_PERCENT,
        "min_common_events_for_pair": MIN_COMMON_EVENTS_FOR_PAIR,
        "datasets": [{"phenotype": d.phenotype, "name": d.name, "path": str(d.path)} for d in datasets],
        "n_events_loaded": int(len(event_meta)),
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Per-electrode functional connectivity matrices and graph metrics per ROI")
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
    ensure_dir(OUT_DIR / "01_matrices")
    ensure_dir(OUT_DIR / "02_region_pair_stats")
    ensure_dir(OUT_DIR / "03_graphs")
    ensure_dir(OUT_DIR / "04_hierarchy_distance")

    nodes = make_node_order()
    datasets = discover_dataset_roots()
    print("[INFO] Phenotype roots found:")
    for ds in datasets:
        print(f"  {ds.phenotype:8s} | {ds.path}")

    events, event_meta = load_events(datasets)
    save_df(event_meta, OUT_DIR / "tables" / "fc16_loaded_event_manifest")
    save_manifest(datasets, event_meta, nodes)

    all_pair_dfs = []
    for signal in SIGNALS_TO_RUN:
        print(f"[INFO] Computing ELECTRODE-level FC for signal: {signal}")
        df = compute_pair_corr_for_dataset_electrodes(events, signal, nodes)
        all_pair_dfs.append(df)
    pair_df = pd.concat(all_pair_dfs, ignore_index=True)
    save_df(pair_df, OUT_DIR / "tables" / "fc16_dataset_level_electrode_pair_pcc")

    matrix_summary = build_matrix_summary(pair_df, nodes)
    save_df(matrix_summary, OUT_DIR / "tables" / "fc16_phenotype_mean_electrode_matrix_summary")

    graph_metric_df = graph_metrics_from_pair_df(pair_df, nodes)
    save_df(graph_metric_df, OUT_DIR / "tables" / "fc16_graph_metrics")

    coords = load_node_coords(nodes)
    if coords is not None:
        save_df(coords, OUT_DIR / "tables" / "fc16_node_coordinates_used")
        dist_df = make_distance_table(pair_df, coords)
        save_df(dist_df, OUT_DIR / "tables" / "fc16_distance_pair_table")
    else:
        dist_df = pd.DataFrame()

    for signal in SIGNALS_TO_RUN:
        print(f"[INFO] Plotting outputs for {signal}")
        sname = sanitize(signal)
        plot_fc_matrices(pair_df, signal, nodes, OUT_DIR / "01_matrices" / f"FC16_01_{sname}_APPNL_vs_APPNLGF_electrode_matrices")
        plot_difference_matrix(pair_df, signal, nodes, OUT_DIR / "01_matrices" / f"FC16_02_{sname}_APPNLGF_minus_APPNL_electrode_difference_matrix")
        plot_pair_type_boxplots(pair_df, signal, OUT_DIR / "02_region_pair_stats" / f"FC16_03_{sname}_electrode_region_pair_boxplots")
        plot_roi_pair_delta_heat(pair_df, signal, OUT_DIR / "02_region_pair_stats" / f"FC16_04_{sname}_roi_pair_delta_from_16x16_links")
        plot_network_graphs(pair_df, signal, nodes, OUT_DIR / "03_graphs" / f"FC16_05_{sname}_thresholded_96node_network_graphs")
        plot_graph_metric_bars(graph_metric_df, signal, OUT_DIR / "03_graphs" / f"FC16_06_{sname}_graph_metric_bars")
        plot_hierarchy_scale(pair_df, signal, OUT_DIR / "04_hierarchy_distance" / f"FC16_07_{sname}_circuit_scale_profile")
        if not dist_df.empty:
            plot_distance_decay(dist_df, signal, OUT_DIR / "04_hierarchy_distance" / f"FC16_08_{sname}_distance_decay")

    write_methods_and_plan(nodes)

    print("\nDONE")
    print(f"Saved to: {OUT_DIR.resolve()}")
    print("Main tables:")
    print(f"  {OUT_DIR / 'tables' / 'fc16_dataset_level_electrode_pair_pcc.csv'}")
    print(f"  {OUT_DIR / 'tables' / 'fc16_graph_metrics.csv'}")
    print("Figure plan:")
    print(f"  {OUT_DIR / '05_methods_and_plan' / 'functional_connectivity_16electrode_methods_and_figure_plan.txt'}")


if __name__ == "__main__":
    main()
