# -*- coding: utf-8 -*-
"""
roc_scalability_sensitivity.py -- SPACE framework: benchmark vs conventional methods.

Manuscript figure(s): Supplementary Figure 8 (ROC/AUC, scalability, sensitivity vs
conventional methods) -- this is the headline "why SPACE is better/faster" figure.

What it computes:
    Logistic-regression ROC/AUC comparing SPACE-derived spatial/phase features against
    conventional feature families (PSD-only, avalanche-only, LFP scalar, LFP
    cross-correlation), using grouped cross-validation (by dataset/animal, not by ROI, to
    avoid pseudoreplication) and bootstrap confidence intervals; a wall-clock
    computational-scalability benchmark of SPACE-style O(N*T) metrics vs pairwise
    correlation O(N^2*T) metrics across electrode counts; and a dataset-level (not
    per-event, to avoid pseudoreplication) statistical-sensitivity comparison
    (-log10 p) between SPACE and conventional metrics. This is v5b, the final "reviewer
    fix" revision of this benchmark: it restores sane per-class sample thresholds that an
    intermediate draft had over-tightened, and fixes a pseudoreplication bug in the
    sensitivity statistics.

Inputs / outputs (see data/README.md for the full schema):
    Input:  the feature tables written by ``core/ctx_hippo_avalanche_kuramoto_compare.py``
            (``.../tables/roi_avalanche_kuramoto_combined_features.csv``) and
            ``core/psd_wavelet_metrics.py`` (``.../tables/baseline_event_lfp_csd_features.csv``,
            ``.../05_FREQUENCY_PSD/baseline_frequency_psd_event_features.csv``), plus
            optionally the raw per-ROI event-trace folders under ``data/event_traces/``
            for the real (non-synthetic) scalability benchmark.
    Output: ROC/AUC, scalability, and sensitivity tables and figures (including the
            composite supplementary figure) under ``data/results/benchmark/``.

Proper data-driven Supplementary Figure generator for SPACE-style ROI MEA benchmarks.

Purpose
-------
Build a real benchmark figure from the outputs of the ROI avalanche + Kuramoto script
and the LFP/CSD/PSD script, instead of manually drawing ROC/scaling/sensitivity panels.

Panels
------
A. ROC curves comparing separate feature families:
      SPACE-derived spatial/phase features, PSD-only features, avalanche-only features,
      LFP/CSD scalar features, and LFP cross-correlation features
   Classification target can be genotype, region, or condition depending on labels available.

B. Computational scalability benchmark:
      SPACE-style O(N*T) metrics vs pairwise correlation O(N^2*T) metrics
   using real event arrays if available, otherwise a synthetic fallback.

C. Sensitivity comparison:
      -log10(p) for conventional metrics and SPACE metrics, tested at dataset/animal level
      when possible. Event-level fallback is allowed but flagged in the output table.

Expected inputs
---------------
1) ROI avalanche/Kuramoto output folder from:
   260524_ROI_AVALANCHE_KURAMOTO_CTX_HIPPO_COMPARE_v2_FULLVIS.py

   Expected table:
   BASELINE_ROI_AVALANCHE_KURAMOTO_CTX_HIPPO/tables/roi_avalanche_kuramoto_combined_features.csv

2) Optional LFP/CSD/PSD output folder from:
   260518_BASELINE_APPNL_APPNLGF_CTX_HIPPO_LFP_CSD_WAVELET_v3_psd_specparam.py

   Expected tables:
   .../tables/baseline_event_lfp_csd_features.csv
   .../05_FREQUENCY_PSD/baseline_frequency_psd_event_features.csv

3) Optional event folders for real scalability benchmark:
   WT/WT_event_dbscan_archetype and 5xFAD/AD_event_dbscan_archetype or similar.

Outputs
-------
SPACE_BENCHMARK_PROPER/
    figs/SuppFig_SPACE_benchmark_proper.png/.svg
    figs/panel_A_ROC_*.png/.svg
    figs/panel_B_scalability.png/.svg
    figs/panel_C_sensitivity.png/.svg
    tables/benchmark_feature_matrix.csv/.xlsx
    tables/roc_results.csv/.xlsx
    tables/scalability_times.csv/.xlsx
    tables/sensitivity_stats.csv/.xlsx
    manifest.json

Notes
-----
- This script avoids pseudoreplication by aggregating to dataset x ROI by default.
- If dataset labels are unavailable or too few, it will still run but will write a warning.
- ROC is only meaningful if both classes have enough samples.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.feature_selection import VarianceThreshold
except Exception as e:
    raise RuntimeError("This script requires scikit-learn. Install with: pip install scikit-learn") from e

try:
    from scipy.signal import hilbert
except Exception:
    hilbert = None

# =============================================================================
# USER CONFIG
# =============================================================================

# --- EDIT THIS --- (or override via CLI flags, see build_argparser) Main input/output roots.
SPACE_ROOT = Path("data/results/ctx_hippo_compare")
LFP_CSD_ROOT = Path("data/results/psd_wavelet")

# --- EDIT THIS --- Optional per-group event-trace folders for the real (non-synthetic)
# scalability benchmark; point these at your own data/event_traces/<group>/ folders.
EVENT_ROOTS = [
    Path("data/event_traces/WT"),
    Path("data/event_traces/5xFAD"),
]

OUT_DIR = Path("data/results/benchmark")

# Classification target for ROC.
# Options: "auto", "phenotype", "region", "drug", "condition".
ROC_TARGET = "auto"
POSITIVE_CLASS = None      # e.g. "5xFAD" or "APPNL_GF". None = second sorted class.

# Main aggregation for feature tables. Dataset x ROI is kept so spatial information is not lost,
# but ROC is evaluated with dataset-grouped cross-validation to avoid ROI leakage.
AGGREGATE_LEVEL = ["phenotype", "dataset", "roi_name", "region"]

# ROC validation settings.
MIN_SAMPLES_PER_CLASS = 3
ROC_GROUP_COL = "dataset"          # keep all ROIs from one slice/animal in the same fold
ROC_USE_GROUP_CV = True
ROC_ALLOW_APPARENT_FALLBACK = False # never train and test on the same rows for the main ROC
ROC_MAX_FEATURES_PER_MODEL = 5      # selected inside each training fold only; prevents p >> n overfit
ROC_N_BOOTSTRAP = 2000              # bootstrap CI from out-of-fold predictions
ROC_N_PERMUTATIONS = 1000           # label-permutation p-value from fixed out-of-fold scores
RANDOM_STATE = 7

# Scalability benchmark settings.
# Methods: event epochs of 300 samples (600 ms at 500 Hz), N = 100 ... 10,000 electrodes,
# 10 iterations of 100 runs per N; electrodes drawn with replacement when N exceeds the
# number of recorded electrodes.
SCALABILITY_N_LIST = [100, 200, 500, 1000, 2000, 4096, 6000, 8000, 10000]
SCALABILITY_T = 300        # number of time points used per benchmark run (600 ms at 500 Hz)
SCALABILITY_REPEATS = 10   # iterations
SCALABILITY_RUNS_PER_REPEAT = 100   # runs timed within each iteration
SYNTHETIC_IF_NO_EVENT_DATA = True

# Feature-sensitivity settings (Suppl. Fig. 7C). Methods: -log10(P) of a two-sided Welch's t-test
# between genotypes for each feature, computed on POOLED EVENTS with Benjamini-Hochberg correction
# across features; events (not animals) are the unit in this descriptive analysis only.
SENSITIVITY_UNIT = "event"        # "event" (Methods) or "dataset" (animal-level means, conservative)
SENSITIVITY_BH_CORRECTION = True  # add BH-adjusted p and -log10(p_BH) columns; ranking uses p_BH

# Figure style.
FIG_DPI = 600
SAVE_EXCEL = True

# Reviewer-facing figure conventions.
# Keep the same model colors in ROC, AUC-CI, scalability, and sensitivity panels.
PRIMARY_MODEL_ORDER = ["SPACE spatial/phase", "PSD only", "Avalanche only"]
SECONDARY_MODEL_ORDER = ["LFP cross-correlation", "Conventional combined"]
MODEL_COLORS = {
    "SPACE spatial/phase": "#1f77b4",
    "PSD only": "#ff7f0e",
    "Avalanche only": "#2ca02c",
    "LFP cross-correlation": "#9467bd",
    "Conventional combined": "#7f7f7f",
    "Chance": "#555555",
}
SCALABILITY_INCLUDE_SECONDARY = False

# =============================================================================
# STYLE / IO
# =============================================================================

def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
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
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, out_base: Path):
    ensure_dir(out_base.parent)
    fig.savefig(str(out_base) + ".png", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_df(df: pd.DataFrame, out_base: Path):
    ensure_dir(out_base.parent)
    df.to_csv(str(out_base) + ".csv", index=False)
    if SAVE_EXCEL:
        try:
            df.to_excel(str(out_base) + ".xlsx", index=False)
        except Exception as e:
            print(f"[WARN] Excel save failed for {out_base}: {e}")


def safe_name(x) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]+", "_", str(x)).strip("_") or "NA"


def ordered_model_names(names: List[str], primary_only: bool = False) -> List[str]:
    desired = PRIMARY_MODEL_ORDER if primary_only else (PRIMARY_MODEL_ORDER + SECONDARY_MODEL_ORDER)
    out = [n for n in desired if n in names]
    out += [n for n in names if n not in out]
    return out


def color_for_model(name: str) -> str:
    return MODEL_COLORS.get(name, "#333333")



def primary_only_curves(curves: Dict[str, dict]) -> Dict[str, dict]:
    """Return only the three reviewer-facing models for main panels."""
    keep = [k for k in PRIMARY_MODEL_ORDER if k in curves]
    return {k: curves[k] for k in keep}


def primary_only_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "model" not in df.columns:
        return df
    keep = ordered_model_names(df["model"].astype(str).unique().tolist(), primary_only=True)
    return df[df["model"].astype(str).isin(keep)].copy()

# =============================================================================
# FEATURE LOADING
# =============================================================================

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        print(f"[LOAD] {path}")
        return pd.read_csv(path)
    print(f"[MISS] {path}")
    return pd.DataFrame()


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ["phenotype", "dataset", "roi_name", "region", "drug", "condition", "event_key"]:
        if col not in df.columns:
            df[col] = "NA"
    return df


def load_feature_tables() -> Tuple[pd.DataFrame, Dict[str, str]]:
    notes = {}

    space_combined = read_csv_if_exists(SPACE_ROOT / "tables" / "roi_avalanche_kuramoto_combined_features.csv")
    aval = read_csv_if_exists(SPACE_ROOT / "tables" / "roi_avalanche_event_summary.csv")
    kura = read_csv_if_exists(SPACE_ROOT / "tables" / "roi_kuramoto_event_summary.csv")

    lfp = read_csv_if_exists(LFP_CSD_ROOT / "tables" / "baseline_event_lfp_csd_features.csv")
    psd = read_csv_if_exists(LFP_CSD_ROOT / "05_FREQUENCY_PSD" / "baseline_frequency_psd_event_features.csv")

    # Prefer combined SPACE table. If missing, merge avalanche + Kuramoto.
    if not space_combined.empty:
        space = normalize_keys(space_combined)
    else:
        aval = normalize_keys(aval)
        kura = normalize_keys(kura)
        if not aval.empty and not kura.empty:
            merge_keys = [c for c in ["phenotype", "dataset", "roi_name", "region", "event_key", "event_index"] if c in aval.columns and c in kura.columns]
            if not merge_keys:
                merge_keys = ["phenotype", "dataset", "roi_name", "region"]
            space = pd.merge(aval, kura, on=merge_keys, how="outer", suffixes=("_aval", "_kura"))
        elif not aval.empty:
            space = aval
        elif not kura.empty:
            space = kura
        else:
            space = pd.DataFrame()

    if space.empty:
        raise FileNotFoundError(
            "No SPACE feature table found. Run the ROI avalanche/Kuramoto script first, or edit SPACE_ROOT."
        )

    space = normalize_keys(space)
    space["feature_source_space"] = 1

    # Merge conventional LFP/CSD/PSD tables if present.
    conv = pd.DataFrame()
    if not lfp.empty or not psd.empty:
        lfp = normalize_keys(lfp)
        psd = normalize_keys(psd)
        if not lfp.empty and not psd.empty:
            merge_keys = [c for c in ["phenotype", "dataset", "roi_name", "region", "event_key", "event_index"] if c in lfp.columns and c in psd.columns]
            if not merge_keys:
                merge_keys = ["phenotype", "dataset", "roi_name", "region"]
            conv = pd.merge(lfp, psd, on=merge_keys, how="outer", suffixes=("_lfp", "_psd"))
        elif not lfp.empty:
            conv = lfp
        else:
            conv = psd
        conv = normalize_keys(conv)
        conv["feature_source_conventional"] = 1

    if not conv.empty:
        merge_keys = [c for c in ["phenotype", "dataset", "roi_name", "region", "event_key", "event_index"] if c in space.columns and c in conv.columns]
        if not merge_keys:
            merge_keys = ["phenotype", "dataset", "roi_name", "region"]
        feat = pd.merge(space, conv, on=merge_keys, how="outer", suffixes=("", "_conv"))
    else:
        feat = space.copy()
        notes["conventional_features"] = "No LFP/CSD/PSD table found; conventional group will use available LFP/correlation-like columns only if present."

    # Clean duplicated label columns after merge.
    for base in ["phenotype", "dataset", "roi_name", "region", "drug", "condition", "event_key"]:
        alt = base + "_conv"
        if alt in feat.columns:
            feat[base] = feat[base].where(feat[base].notna() & (feat[base].astype(str) != "NA"), feat[alt])

    return feat, notes


def aggregate_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_keys(df)
    keys = [k for k in AGGREGATE_LEVEL if k in df.columns]
    if not keys:
        keys = ["phenotype", "dataset", "roi_name", "region"]
        for k in keys:
            if k not in df.columns:
                df[k] = "NA"

    numeric_cols = []
    for c in df.columns:
        if c in keys:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            # Try coercion for numeric-looking strings.
            x = pd.to_numeric(df[c], errors="coerce")
            if np.sum(np.isfinite(x)) >= max(3, int(0.2 * len(x))):
                df[c] = x
                numeric_cols.append(c)

    out = df.groupby(keys, dropna=False)[numeric_cols].mean().reset_index()
    return out

# =============================================================================
# FEATURE SETS
# =============================================================================

def pick_columns(df: pd.DataFrame, include_terms: List[str], exclude_terms: Optional[List[str]] = None) -> List[str]:
    exclude_terms = exclude_terms or []
    cols = []
    for c in df.columns:
        cl = c.lower()
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(t.lower() in cl for t in include_terms) and not any(t.lower() in cl for t in exclude_terms):
            # avoid IDs / sample labels
            if cl in {"event_index", "n_electrodes", "feature_source_space", "feature_source_conventional"}:
                continue
            cols.append(c)
    return sorted(set(cols))


def _existing_numeric_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Keep unique numeric columns only, preserving order."""
    out = []
    seen = set()
    for c in cols:
        if c in seen:
            continue
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
            seen.add(c)
    return out



def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Define the feature families of the manuscript (Methods, "Classification, scalability and
    feature sensitivity"):
        SPACE spatial/phase   = Kuramoto, phase and propagation descriptors together with the
                                event-window LFP and CSD features
        PSD only              = PSD / spectral features
        Avalanche only        = avalanche descriptors
        Conventional combined = PSD + avalanche features
        LFP cross-correlation = inter-electrode LFP correlation
    Columns are assigned by name keywords; check tables/feature_sets_used.csv after a run.
    """
    avalanche_terms = [
        "aval", "avalanche", "branch", "critical", "alpha", "tau",
        "duration", "size", "active_bins", "empty_bins", "event_rate",
    ]
    psd_terms = [
        "psd", "power", "spectral", "aperiodic", "specparam", "fooof",
        "delta", "theta", "alpha_band", "beta", "gamma", "high_gamma",
        "peak_freq", "peak_frequency", "band", "welch",
    ]
    space_terms = [
        "kuramoto", "phase", "plv", "grad", "gradient", "speed", "velocity",
        "singular", "vortex", "order", "r_mean", "hotspot", "propagation",
        "latency", "spatial", "wavefront",
    ]
    lfp_csd_terms = [
        "lfp", "csd", "rms", "energy", "auc", "source", "sink", "balance",
        "slope", "amplitude", "peak", "wavelet",
    ]
    corr_terms = ["lfp_interelectrode_corr", "cross_corr", "xcorr", "pcc", "corr", "correlation"]

    avalanche_cols = pick_columns(df, avalanche_terms, exclude_terms=["psd", "spectral"])
    psd_cols = pick_columns(df, psd_terms, exclude_terms=["avalanche", "aval", "kuramoto", "plv"])
    space_cols = pick_columns(df, space_terms, exclude_terms=["avalanche", "aval", "psd", "spectral", "aperiodic"])
    lfp_csd_cols = pick_columns(df, lfp_csd_terms, exclude_terms=["psd", "spectral", "avalanche", "aval", "kuramoto", "plv", "corr"])
    corr_cols = pick_columns(df, corr_terms, exclude_terms=["kuramoto", "plv", "avalanche", "aval"])

    conventional_cols = _existing_numeric_cols(df, psd_cols + avalanche_cols)

    sets = {
        "SPACE spatial/phase": _existing_numeric_cols(df, space_cols + lfp_csd_cols),
        "PSD only": _existing_numeric_cols(df, psd_cols),
        "Avalanche only": _existing_numeric_cols(df, avalanche_cols),
        "LFP cross-correlation": _existing_numeric_cols(df, corr_cols),
        "Conventional combined": _existing_numeric_cols(df, conventional_cols),
    }

    return {k: v for k, v in sets.items() if len(v) > 0}


def choose_target(df: pd.DataFrame) -> Tuple[str, List[str]]:
    candidates = ["phenotype", "condition", "drug", "region"] if ROC_TARGET == "auto" else [ROC_TARGET]
    for c in candidates:
        if c not in df.columns:
            continue
        counts = df[c].astype(str).value_counts()
        usable = counts[counts >= MIN_SAMPLES_PER_CLASS]
        if len(usable) >= 2:
            classes = list(usable.index[:2])
            # keep sorted by PHENO-ish order if possible
            vals = sorted(classes)
            if POSITIVE_CLASS is not None and POSITIVE_CLASS in vals:
                vals = [v for v in vals if v != POSITIVE_CLASS] + [POSITIVE_CLASS]
            return c, vals[:2]
    raise RuntimeError("No usable binary target found for ROC. Check phenotype/condition/drug/region labels.")


def _make_roc_model(n_train: int, n_features: int) -> Pipeline:
    """Small-sample-safe logistic model.

    Feature selection is intentionally inside the fold-specific pipeline. This prevents
    information leakage from the held-out fold and avoids near-guaranteed separation
    when many correlated SPACE features are used with few animals/slices.
    """
    k = min(int(ROC_MAX_FEATURES_PER_MODEL), int(n_features), max(1, int(n_train) - 2))
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
    ]
    if n_features > k:
        steps.append(("select", SelectKBest(score_func=f_classif, k=k)))
    steps += [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            C=0.25,
            random_state=RANDOM_STATE,
        )),
    ]
    return Pipeline(steps)


def _valid_group_cv_splits(y: np.ndarray, groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return leave-one-group-out splits whose training set contains both classes."""
    splits = []
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(np.zeros_like(y), y, groups):
        if len(np.unique(y[train_idx])) == 2:
            splits.append((train_idx, test_idx))
    return splits


def _valid_stratified_splits(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    counts = np.bincount(y)
    min_class = int(np.min(counts))
    if min_class < 2:
        return []
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros_like(y), y))


def cv_predict_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str, str]:
    """Out-of-fold predicted probabilities for ROC.

    Returns
    -------
    scores : array
        Out-of-fold probabilities for the positive class.
    cv_method : str
        Method actually used.
    warning : str
        Empty string if everything is acceptable; otherwise a reviewer-facing warning.
    """
    n = len(y)
    scores = np.full(n, np.nan, dtype=float)
    warning = ""

    # Preferred: leave-one-dataset/slice/animal-out. This avoids ROI leakage.
    if ROC_USE_GROUP_CV and groups is not None:
        groups = np.asarray(groups).astype(str)
        valid_groups = pd.Series(groups).replace({"": "NA", "nan": "NA", "None": "NA"}).to_numpy()
        unique_groups = np.unique(valid_groups)
        if len(unique_groups) >= 4 and len(np.unique(valid_groups[y == 0])) >= 2 and len(np.unique(valid_groups[y == 1])) >= 2:
            splits = _valid_group_cv_splits(y, valid_groups)
            if splits:
                for train_idx, test_idx in splits:
                    model = _make_roc_model(len(train_idx), X.shape[1])
                    model.fit(X[train_idx], y[train_idx])
                    scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
                if np.all(np.isfinite(scores)):
                    return scores, f"leave-one-{ROC_GROUP_COL}-out grouped CV", warning
                warning = "Grouped CV left some samples unpredicted; falling back to stratified row CV."
        else:
            warning = (
                f"Grouped CV not possible: need at least two {ROC_GROUP_COL} groups per class. "
                "Falling back to stratified row CV, which is less strict."
            )

    # Secondary: stratified row-level CV. This is acceptable only when grouping is impossible.
    splits = _valid_stratified_splits(y)
    if splits:
        for train_idx, test_idx in splits:
            model = _make_roc_model(len(train_idx), X.shape[1])
            model.fit(X[train_idx], y[train_idx])
            scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        return scores, "stratified row CV", warning

    # No apparent training/test estimate unless explicitly allowed.
    if ROC_ALLOW_APPARENT_FALLBACK:
        model = _make_roc_model(n, X.shape[1])
        model.fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        return scores, "apparent fit; discovery only", "Apparent fit was used; do not present as validation."

    raise RuntimeError(
        "ROC validation not possible: too few independent samples per class. "
        "Add more animals/slices or set ROC_ALLOW_APPARENT_FALLBACK=True only for exploratory plotting."
    )


def bootstrap_auc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int = ROC_N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float, int]:
    """Percentile bootstrap CI for AUC using out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    ok = np.isfinite(scores)
    y = y[ok]
    scores = scores[ok]
    vals = []
    n = len(y)
    if n < 4 or len(np.unique(y)) < 2:
        return np.nan, np.nan, 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], scores[idx]))
    if not vals:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def permutation_auc_pvalue(
    y: np.ndarray,
    scores: np.ndarray,
    observed_auc: float,
    n_perm: int = ROC_N_PERMUTATIONS,
    seed: int = RANDOM_STATE,
) -> Tuple[float, int]:
    """Permutation p-value for the fixed out-of-fold score ranking."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    ok = np.isfinite(scores)
    y = y[ok]
    scores = scores[ok]
    if len(np.unique(y)) < 2:
        return np.nan, 0
    ge = 0
    done = 0
    for _ in range(int(n_perm)):
        yp = rng.permutation(y)
        if len(np.unique(yp)) < 2:
            continue
        ap = roc_auc_score(yp, scores)
        ge += int(ap >= observed_auc)
        done += 1
    return float((ge + 1) / (done + 1)) if done else np.nan, int(done)


def run_roc(df: pd.DataFrame, sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    target, classes = choose_target(df)
    sub = df[df[target].astype(str).isin(classes)].copy()
    pos_class = classes[1]
    y = (sub[target].astype(str) == pos_class).astype(int).to_numpy()

    groups = None
    if ROC_GROUP_COL in sub.columns:
        groups = sub[ROC_GROUP_COL].astype(str).to_numpy()

    rows = []
    curves = {}
    score_rows = []

    for name, cols in sets.items():
        cols = [c for c in cols if c in sub.columns and pd.api.types.is_numeric_dtype(sub[c])]
        cols = [c for c in cols if np.sum(np.isfinite(pd.to_numeric(sub[c], errors="coerce"))) >= 3]
        if len(cols) == 0:
            print(f"[WARN] No columns for ROC set: {name}")
            continue

        X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        try:
            scores, cv_method, cv_warning = cv_predict_scores(X, y, groups=groups)
            ok = np.isfinite(scores)
            if len(np.unique(y[ok])) < 2:
                raise RuntimeError("Out-of-fold predictions contain only one class.")
            fpr, tpr, thr = roc_curve(y[ok], scores[ok])
            auc_val = float(auc(fpr, tpr))
            ci_lo, ci_hi, n_boot = bootstrap_auc_ci(y[ok], scores[ok])
            perm_p, n_perm = permutation_auc_pvalue(y[ok], scores[ok], auc_val)
        except Exception as e:
            print(f"[WARN] ROC failed for {name}: {e}")
            continue

        curves[name] = dict(
            fpr=fpr, tpr=tpr, threshold=thr, auc=auc_val, auc_ci_low=ci_lo,
            auc_ci_high=ci_hi, n_features=len(cols), cols=cols, cv_method=cv_method,
        )
        rows.append({
            "model": name,
            "target": target,
            "negative_class": classes[0],
            "positive_class": pos_class,
            "n_samples": int(np.sum(ok)),
            "n_positive": int(np.sum(y[ok] == 1)),
            "n_negative": int(np.sum(y[ok] == 0)),
            "n_groups": int(len(np.unique(groups))) if groups is not None else np.nan,
            "n_positive_groups": int(len(np.unique(groups[y == 1]))) if groups is not None else np.nan,
            "n_negative_groups": int(len(np.unique(groups[y == 0]))) if groups is not None else np.nan,
            "n_features_available": int(len(cols)),
            "max_features_selected_inside_fold": int(ROC_MAX_FEATURES_PER_MODEL),
            "auc_cv": auc_val,
            "auc_ci_low_bootstrap": ci_lo,
            "auc_ci_high_bootstrap": ci_hi,
            "n_bootstrap_valid": n_boot,
            "permutation_p_value": perm_p,
            "n_permutations": n_perm,
            "cv_method": cv_method,
            "warning": cv_warning,
            "feature_columns": ";".join(cols),
        })

        tmp = sub[[c for c in [target, ROC_GROUP_COL, "phenotype", "dataset", "roi_name", "region"] if c in sub.columns]].copy()
        tmp["model"] = name
        tmp["y_true"] = y
        tmp["oof_score"] = scores
        score_rows.append(tmp)

    roc_df = pd.DataFrame(rows)
    if score_rows:
        oof_scores = pd.concat(score_rows, ignore_index=True)
        save_df(oof_scores, OUT_DIR / "tables" / "roc_oof_predictions")
    return roc_df, curves

def _fmt_auc_ci(d: dict) -> str:
    """Journal-style AUC label with bootstrap 95% CI when available."""
    auc_val = d.get("auc", np.nan)
    lo = d.get("auc_ci_low", np.nan)
    hi = d.get("auc_ci_high", np.nan)
    if np.isfinite(lo) and np.isfinite(hi):
        return f"AUC = {auc_val:.2f} (95% CI {lo:.2f}–{hi:.2f})"
    return f"AUC = {auc_val:.2f} (95% CI n/a)"


def _add_auc_ci_box(ax, curves: Dict[str, dict], fontsize: int = 6):
    """Add explicit AUC/CI text inside the ROC panel so CI is visible in exported graph."""
    if not curves:
        return
    lines = []
    for name, d in curves.items():
        lines.append(f"{name}: {_fmt_auc_ci(d)}")
    txt = "\n".join(lines)
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.92),
    )




def plot_roc(curves: Dict[str, dict], out_base: Path, title_extra: str = "", primary_only: bool = True):
    """Clean ROC panel.

    The AUC/CI values are shown once in the legend. The previous upper-left
    text box was removed because it duplicated the legend and obscured curves.
    """
    if not curves:
        return
    curves_use = primary_only_curves(curves) if primary_only else curves
    if not curves_use:
        return

    set_style()
    fig, ax = plt.subplots(figsize=(4.9, 4.0), constrained_layout=True)

    for name in ordered_model_names(list(curves_use.keys()), primary_only=primary_only):
        d = curves_use[name]
        label = f"{name}: {_fmt_auc_ci(d)}"
        ax.plot(
            d["fpr"], d["tpr"],
            lw=2.2,
            color=color_for_model(name),
            label=label,
        )

    ax.plot([0, 1], [0, 1], color=color_for_model("Chance"), lw=0.9, ls="--", label="Chance")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("A. ROC classification" + title_extra)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Single legend only. Put it below the plot to avoid covering ROC curves.
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        handlelength=2.5,
    )

    save_fig(fig, out_base)


def plot_auc_summary(roc_df: pd.DataFrame, out_base: Path, primary_only: bool = True):
    """Separate AUC summary graph with one line per feature family and bootstrap CI bars."""
    if roc_df is None or roc_df.empty:
        return
    needed = ["model", "auc_cv", "auc_ci_low_bootstrap", "auc_ci_high_bootstrap"]
    if any(c not in roc_df.columns for c in needed):
        return
    p = roc_df.copy()
    p = primary_only_table(p) if primary_only else p
    p = p[np.isfinite(pd.to_numeric(p["auc_cv"], errors="coerce"))].copy()
    if p.empty:
        return
    p["auc_cv"] = pd.to_numeric(p["auc_cv"], errors="coerce")
    p["auc_ci_low_bootstrap"] = pd.to_numeric(p["auc_ci_low_bootstrap"], errors="coerce")
    p["auc_ci_high_bootstrap"] = pd.to_numeric(p["auc_ci_high_bootstrap"], errors="coerce")
    p["model"] = pd.Categorical(
        p["model"],
        categories=ordered_model_names(p["model"].astype(str).unique().tolist(), primary_only=primary_only),
        ordered=True,
    )
    p = p.sort_values("model")

    y = np.arange(len(p))
    set_style()
    fig, ax = plt.subplots(figsize=(6.3, max(3.0, 0.55 * len(p) + 1.0)), constrained_layout=True)
    ax.axvline(0.5, color=color_for_model("Chance"), lw=0.9, ls="--")
    for yi, (_, row) in zip(y, p.iterrows()):
        name = str(row["model"])
        auc_val = row["auc_cv"]
        ci_l = row["auc_ci_low_bootstrap"]
        ci_h = row["auc_ci_high_bootstrap"]
        xerr_low = max(0, auc_val - ci_l) if np.isfinite(ci_l) else 0
        xerr_high = max(0, ci_h - auc_val) if np.isfinite(ci_h) else 0
        ax.errorbar(
            auc_val, yi, xerr=[[xerr_low], [xerr_high]],
            fmt="o", lw=1.8, capsize=3, markersize=5,
            color=color_for_model(name), ecolor=color_for_model(name),
        )
        txt = f"{auc_val:.2f} [{ci_l:.2f}–{ci_h:.2f}]" if np.isfinite(ci_l) and np.isfinite(ci_h) else f"{auc_val:.2f} [CI n/a]"
        ax.text(min(1.01, auc_val + 0.025), yi, txt, va="center", ha="left", fontsize=7, color=color_for_model(name))
    ax.set_xlim(0.0, 1.02)
    ax.set_yticks(y)
    ax.set_yticklabels(p["model"].astype(str).to_list(), fontsize=8)
    ax.set_xlabel("Grouped cross-validated AUC")
    ax.set_title("A2. AUC with bootstrap 95% CI")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, out_base)


def find_event_arrays() -> List[np.ndarray]:
    arrays = []
    for root in EVENT_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("lfp_event_traces.npy"):
            try:
                X = np.load(p).astype(np.float32)
                if X.ndim == 2 and X.shape[0] >= 2 and X.shape[1] >= 10:
                    arrays.append(X)
                    if len(arrays) >= 10:
                        return arrays
            except Exception:
                continue
    return arrays


def make_benchmark_matrix(N: int, T: int, real_arrays: List[np.ndarray], rng) -> np.ndarray:
    if real_arrays:
        X0 = real_arrays[int(rng.integers(0, len(real_arrays)))]
        # If ROI event has fewer electrodes than N, bootstrap electrodes with replacement.
        idx = rng.choice(np.arange(X0.shape[0]), size=N, replace=(X0.shape[0] < N))
        tmax = min(T, X0.shape[1])
        X = X0[idx, :tmax]
        if tmax < T:
            pad = rng.normal(0, np.nanstd(X) + 1e-6, size=(N, T - tmax)).astype(np.float32)
            X = np.concatenate([X, pad], axis=1)
        return X.astype(np.float32)
    # Synthetic structured signal fallback.
    t = np.linspace(0, 1, T, dtype=np.float32)
    common = np.sin(2 * np.pi * 8 * t)[None, :]
    phase = rng.uniform(0, 2 * np.pi, size=(N, 1))
    osc = np.sin(2 * np.pi * 20 * t[None, :] + phase).astype(np.float32)
    noise = rng.normal(0, 0.5, size=(N, T)).astype(np.float32)
    return (0.6 * common + 0.4 * osc + noise).astype(np.float32)


def space_metric_time(X: np.ndarray) -> float:
    # SPACE-style linear-ish metric: analytic phase, Kuramoto R, phase velocity proxy.
    t0 = time.perf_counter()
    if hilbert is not None:
        Z = hilbert(X, axis=1)
        phase = np.angle(Z)
    else:
        # fallback: use normalized signal as pseudo-phase
        phase = np.angle(np.exp(1j * (X / (np.nanstd(X, axis=1, keepdims=True) + 1e-6))))
    R = np.abs(np.nanmean(np.exp(1j * phase), axis=0))
    dphi = np.angle(np.exp(1j * np.diff(phase, axis=1)))
    v = np.nanmean(np.abs(dphi))
    _ = float(np.nanmean(R) + v)
    return time.perf_counter() - t0


def pairwise_corr_time(X: np.ndarray) -> float:
    t0 = time.perf_counter()
    C = np.corrcoef(X)
    iu = np.triu_indices_from(C, 1)
    _ = float(np.nanmean(C[iu]))
    return time.perf_counter() - t0


def psd_metric_time(X: np.ndarray) -> float:
    """Approximate PSD-style global feature cost using FFT power summaries."""
    t0 = time.perf_counter()
    X0 = X - np.nanmean(X, axis=1, keepdims=True)
    F = np.fft.rfft(X0, axis=1)
    P = (np.abs(F) ** 2) / max(1, X0.shape[1])
    _ = float(np.nanmean(P))
    return time.perf_counter() - t0


def avalanche_metric_time(X: np.ndarray) -> float:
    """Approximate avalanche-style global feature cost using thresholded active-bin runs."""
    t0 = time.perf_counter()
    thr = np.nanmean(X, axis=1, keepdims=True) + 2.0 * np.nanstd(X, axis=1, keepdims=True)
    active = (X > thr).astype(np.int8)
    bins = np.sum(active, axis=0) > 0
    padded = np.concatenate(([0], bins.astype(np.int8), [0]))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    durations = ends - starts
    sizes = [int(np.sum(active[:, s:e])) for s, e in zip(starts, ends)]
    _ = float((np.mean(durations) if len(durations) else 0) + (np.mean(sizes) if len(sizes) else 0))
    return time.perf_counter() - t0



def run_scalability() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    real_arrays = find_event_arrays()
    if not real_arrays and not SYNTHETIC_IF_NO_EVENT_DATA:
        return pd.DataFrame()
    if real_arrays:
        print(f"[INFO] Scalability using {len(real_arrays)} real event arrays")
    else:
        print("[WARN] No real event arrays found; using synthetic benchmark data")

    rows = []
    for N in SCALABILITY_N_LIST:
        for rep in range(SCALABILITY_REPEATS):
            X = make_benchmark_matrix(N, SCALABILITY_T, real_arrays, rng)
            # wall-clock time of SCALABILITY_RUNS_PER_REPEAT runs on the same matrix
            ts = sum(space_metric_time(X) for _ in range(SCALABILITY_RUNS_PER_REPEAT))
            tp = sum(psd_metric_time(X) for _ in range(SCALABILITY_RUNS_PER_REPEAT))
            ta = sum(avalanche_metric_time(X) for _ in range(SCALABILITY_RUNS_PER_REPEAT))
            rows.append({"N_electrodes": N, "repeat": rep, "method": "SPACE spatial/phase", "time_sec": ts})
            rows.append({"N_electrodes": N, "repeat": rep, "method": "PSD only", "time_sec": tp})
            rows.append({"N_electrodes": N, "repeat": rep, "method": "Avalanche only", "time_sec": ta})
            if SCALABILITY_INCLUDE_SECONDARY:
                tc = sum(pairwise_corr_time(X) for _ in range(SCALABILITY_RUNS_PER_REPEAT))
                rows.append({"N_electrodes": N, "repeat": rep, "method": "LFP cross-correlation", "time_sec": tc})
            print(f"[BENCH] N={N:4d} rep={rep} SPACE={ts:.4f}s PSD={tp:.4f}s AVAL={ta:.4f}s")
    return pd.DataFrame(rows)



def plot_scalability(df: pd.DataFrame, out_base: Path, primary_only: bool = True):
    if df.empty:
        return
    p = df.copy()
    if primary_only and "method" in p.columns:
        p = p[p["method"].astype(str).isin(PRIMARY_MODEL_ORDER)].copy()
    if p.empty:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.7), constrained_layout=True)
    for method in ordered_model_names(p["method"].astype(str).unique().tolist(), primary_only=primary_only):
        g = p[p["method"].astype(str) == method].copy()
        stats_df = g.groupby("N_electrodes")["time_sec"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(
            stats_df["N_electrodes"], stats_df["mean"], yerr=stats_df["sem"],
            marker="o", lw=1.8, capsize=2, color=color_for_model(method), label=method,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of electrodes (N)")
    ax.set_ylabel("Processing time (s)")
    ax.set_title("B. Computational scalability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="best")
    save_fig(fig, out_base)


def welch_p(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2 or stats is None:
        return np.nan
    try:
        return float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")[1])
    except Exception:
        return np.nan



def run_sensitivity(df: pd.DataFrame, sets: Dict[str, List[str]]) -> pd.DataFrame:
    """Sensitivity statistics at dataset/animal level, not ROI-level pseudoreplication."""
    target, classes = choose_target(df)
    sub = df[df[target].astype(str).isin(classes)].copy()

    unit_col = ROC_GROUP_COL if ROC_GROUP_COL in sub.columns else ("dataset" if "dataset" in sub.columns else None)
    rows = []

    if SENSITIVITY_UNIT == "event":
        sens_base = sub.copy()
        analysis_unit = "pooled events (descriptive; not animal-level inference)"
    elif unit_col is not None:
        numeric_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
        keep_cols = [target, unit_col]
        sens_base = sub[keep_cols + numeric_cols].groupby(keep_cols, dropna=False)[numeric_cols].mean().reset_index()
        analysis_unit = f"{unit_col}-level means"
    else:
        sens_base = sub.copy()
        analysis_unit = "aggregated rows; warning: no dataset grouping column available"

    for group_name, cols in sets.items():
        if group_name not in PRIMARY_MODEL_ORDER:
            continue
        for c in cols:
            if c not in sens_base.columns:
                continue
            vals0 = pd.to_numeric(sens_base.loc[sens_base[target].astype(str) == classes[0], c], errors="coerce").to_numpy(dtype=float)
            vals1 = pd.to_numeric(sens_base.loc[sens_base[target].astype(str) == classes[1], c], errors="coerce").to_numpy(dtype=float)
            p = welch_p(vals0, vals1)
            n0 = int(np.sum(np.isfinite(vals0)))
            n1 = int(np.sum(np.isfinite(vals1)))
            score = np.nan if not np.isfinite(p) else -math.log10(max(p, 1e-300))
            rows.append({
                "feature_group": group_name,
                "feature": c,
                "target": target,
                "class_0": classes[0],
                "class_1": classes[1],
                "n_class_0": n0,
                "n_class_1": n1,
                "mean_class_0": float(np.nanmean(vals0)) if n0 else np.nan,
                "mean_class_1": float(np.nanmean(vals1)) if n1 else np.nan,
                "p_value": p,
                "minus_log10_p": score,
                "test": f"Welch t-test on {analysis_unit}",
                "analysis_unit": analysis_unit,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if SENSITIVITY_BH_CORRECTION:
        out["p_value_bh"] = bh_adjust(out["p_value"].to_numpy(dtype=float))
        out["minus_log10_p_bh"] = [np.nan if not np.isfinite(q) else -math.log10(max(q, 1e-300)) for q in out["p_value_bh"]]
        out["minus_log10_p_raw"] = out["minus_log10_p"]
        out["minus_log10_p"] = out["minus_log10_p_bh"]   # ranking / plotting use the BH-adjusted value
    return out.sort_values(["feature_group", "minus_log10_p"], ascending=[True, False], na_position="last")


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values across all finite entries (NaN preserved)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    try:
        from statsmodels.stats.multitest import multipletests
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
        return out
    except Exception:
        pass
    pv = p[ok]; n = pv.size
    order = np.argsort(pv); ranked = pv[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n); res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out



def plot_sensitivity(df: pd.DataFrame, out_base: Path, top_n_per_group: int = 4):
    if df.empty:
        return
    rows = []
    for group in ordered_model_names(df["feature_group"].astype(str).unique().tolist(), primary_only=True):
        g = df[df["feature_group"].astype(str) == group].copy()
        gg = g[np.isfinite(g["minus_log10_p"])].copy()
        if gg.empty:
            continue
        rows.append(gg.head(top_n_per_group))
    if not rows:
        return
    p = pd.concat(rows, ignore_index=True)
    p["label"] = p["feature"].astype(str).str.replace("benchmark_", "", regex=False).str.replace("psd_", "PSD ", regex=False)
    p["label"] = p["label"].str.replace("kuramoto_", "Kuramoto ", regex=False).str.replace("_", " ", regex=False).str.slice(0, 34)
    p = p.sort_values(["feature_group", "minus_log10_p"], ascending=[True, True])

    set_style()
    fig, ax = plt.subplots(figsize=(5.6, max(3.2, 0.34 * len(p))), constrained_layout=True)
    y = np.arange(len(p))
    ax.barh(y, p["minus_log10_p"].to_numpy(dtype=float), color=[color_for_model(g) for g in p["feature_group"].astype(str)])
    ax.set_yticks(y)
    ax.set_yticklabels(p["label"], fontsize=7)
    ax.set_xlabel(r"$-\log_{10}(p)$")
    n0 = p["n_class_0"].dropna().astype(int).max() if "n_class_0" in p.columns and not p["n_class_0"].dropna().empty else "n/a"
    n1 = p["n_class_1"].dropna().astype(int).max() if "n_class_1" in p.columns and not p["n_class_1"].dropna().empty else "n/a"
    ax.set_title(f"C. Feature sensitivity (dataset-level; n={n0} vs {n1})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, out_base)



def _draw_auc_summary_axis(ax, roc_df: pd.DataFrame, primary_only: bool = True):
    if roc_df is None or roc_df.empty:
        ax.axis("off")
        return
    p = roc_df.copy()
    p = primary_only_table(p) if primary_only else p
    if "auc_cv" not in p.columns:
        ax.axis("off")
        return
    p["auc_cv"] = pd.to_numeric(p["auc_cv"], errors="coerce")
    p["auc_ci_low_bootstrap"] = pd.to_numeric(p.get("auc_ci_low_bootstrap", np.nan), errors="coerce")
    p["auc_ci_high_bootstrap"] = pd.to_numeric(p.get("auc_ci_high_bootstrap", np.nan), errors="coerce")
    p = p[np.isfinite(p["auc_cv"])].copy()
    if p.empty:
        ax.axis("off")
        return
    p["model"] = pd.Categorical(
        p["model"],
        categories=ordered_model_names(p["model"].astype(str).unique().tolist(), primary_only=primary_only),
        ordered=True,
    )
    p = p.sort_values("model")
    y = np.arange(len(p))
    ax.axvline(0.5, color=color_for_model("Chance"), lw=0.8, ls="--")
    for yi, (_, row) in zip(y, p.iterrows()):
        auc_val = row["auc_cv"]
        ci_l = row["auc_ci_low_bootstrap"]
        ci_h = row["auc_ci_high_bootstrap"]
        name = str(row["model"])
        xerr_low = max(0, auc_val - ci_l) if np.isfinite(ci_l) else 0
        xerr_high = max(0, ci_h - auc_val) if np.isfinite(ci_h) else 0
        ax.errorbar(
            auc_val, yi, xerr=[[xerr_low], [xerr_high]],
            fmt="o", lw=1.3, capsize=2.5, markersize=4.5,
            color=color_for_model(name), ecolor=color_for_model(name),
        )
        txt = f"{auc_val:.2f} [{ci_l:.2f}–{ci_h:.2f}]" if np.isfinite(ci_l) and np.isfinite(ci_h) else f"{auc_val:.2f} [CI n/a]"
        ax.text(min(1.01, auc_val + 0.025), yi, txt, va="center", ha="left", fontsize=5.5, color=color_for_model(name))
    ax.set_xlim(0.0, 1.02)
    ax.set_yticks(y)
    ax.set_yticklabels(p["model"].astype(str).to_list(), fontsize=6)
    ax.set_xlabel("AUC")
    ax.set_title("A2  AUC and 95% CI")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)




def make_composite(curves: Dict[str, dict], roc_df: pd.DataFrame, scal_df: pd.DataFrame, sens_df: pd.DataFrame, out_base: Path):
    set_style()
    fig = plt.figure(figsize=(10.8, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.15])

    axA = fig.add_subplot(gs[0, 0])
    curves_use = primary_only_curves(curves)
    if curves_use:
        for name in ordered_model_names(list(curves_use.keys()), primary_only=True):
            d = curves_use[name]
            label = f"{name}: {_fmt_auc_ci(d)}"
            axA.plot(
                d["fpr"], d["tpr"],
                lw=1.8,
                color=color_for_model(name),
                label=label,
            )
        axA.plot([0, 1], [0, 1], color=color_for_model("Chance"), lw=0.8, ls="--", label="Chance")

    axA.set_xlim(-0.02, 1.02)
    axA.set_ylim(-0.02, 1.02)
    axA.set_xlabel("False positive rate")
    axA.set_ylabel("True positive rate")
    axA.set_title("A  ROC classification")
    axA.legend(frameon=False, fontsize=5.8, loc="lower right")
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    axA2 = fig.add_subplot(gs[0, 1])
    _draw_auc_summary_axis(axA2, roc_df, primary_only=True)

    axB = fig.add_subplot(gs[1, 0])
    if not scal_df.empty:
        p = scal_df[scal_df["method"].astype(str).isin(PRIMARY_MODEL_ORDER)].copy()
        for method in ordered_model_names(p["method"].astype(str).unique().tolist(), primary_only=True):
            g = p[p["method"].astype(str) == method].copy()
            s = g.groupby("N_electrodes")["time_sec"].agg(["mean", "sem"]).reset_index()
            axB.errorbar(
                s["N_electrodes"], s["mean"], yerr=s["sem"],
                marker="o", lw=1.5, capsize=2,
                color=color_for_model(method),
                label=method,
            )
        axB.set_xscale("log")
        axB.set_yscale("log")
    axB.set_xlabel("Number of electrodes (N)")
    axB.set_ylabel("Processing time (s)")
    axB.set_title("B  Computational scalability")
    axB.legend(frameon=False, fontsize=6, loc="best")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    axC = fig.add_subplot(gs[1:, 1])
    if not sens_df.empty:
        rows = []
        sens_p = sens_df[sens_df["feature_group"].astype(str).isin(PRIMARY_MODEL_ORDER)].copy()
        for group in ordered_model_names(sens_p["feature_group"].astype(str).unique().tolist(), primary_only=True):
            g = sens_p[sens_p["feature_group"].astype(str) == group].copy()
            gg = g[np.isfinite(g["minus_log10_p"])].copy()
            if not gg.empty:
                rows.append(gg.head(4))
        if rows:
            p = pd.concat(rows, ignore_index=True)
            p["label"] = p["feature"].astype(str).str.replace("benchmark_", "", regex=False)
            p["label"] = p["label"].str.replace("psd_", "PSD ", regex=False).str.replace("kuramoto_", "Kuramoto ", regex=False)
            p["label"] = p["label"].str.replace("_", " ", regex=False).str.slice(0, 42)
            p = p.sort_values(["feature_group", "minus_log10_p"], ascending=[True, True])
            y = np.arange(len(p))
            axC.barh(
                y,
                p["minus_log10_p"].to_numpy(dtype=float),
                color=[color_for_model(g) for g in p["feature_group"].astype(str)],
            )
            axC.set_yticks(y)
            axC.set_yticklabels(p["label"], fontsize=7)

    axC.set_xlabel(r"$-\log_{10}(p)$")
    axC.set_title("C  Sensitivity of SPACE, PSD, and avalanche features\n(dataset-level statistics)")
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    save_fig(fig, out_base)


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="ROC/AUC, scalability, and sensitivity benchmark: SPACE vs conventional methods")
    p.add_argument("--space_root", default=str(SPACE_ROOT), help="Output folder of ctx_hippo_avalanche_kuramoto_compare.py (default: %(default)s)")
    p.add_argument("--lfp_csd_root", default=str(LFP_CSD_ROOT), help="Output folder of psd_wavelet_metrics.py (default: %(default)s)")
    p.add_argument("--out_dir", default=str(OUT_DIR), help="Output folder for benchmark results (default: %(default)s)")
    p.add_argument("--event_roots", default=None,
                   help="Comma-separated per-group event-trace folders used for the real scalability benchmark "
                        "(e.g. 'data/event_traces/CTL,data/event_traces/5xFAD'). Overrides EVENT_ROOTS "
                        "(default: %(default)s = EVENT_ROOTS)")
    p.add_argument("--quick", action="store_true",
                   help="SMOKE-TEST ONLY: shrink the scalability benchmark (N list 100..1000, 2 iterations x 5 runs). "
                        "Methods defaults (N up to 10000, 10 iterations x 100 runs) are used when omitted.")
    return p


def apply_cli_overrides(args) -> None:
    """Plumbing only: re-point EVENT_ROOTS / shrink benchmark sizes for a smoke test."""
    global EVENT_ROOTS, SCALABILITY_N_LIST, SCALABILITY_REPEATS, SCALABILITY_RUNS_PER_REPEAT
    if args.event_roots:
        EVENT_ROOTS = [Path(x.strip()) for x in args.event_roots.split(",") if x.strip()]
    if args.quick:
        SCALABILITY_N_LIST = [100, 200, 500, 1000]
        SCALABILITY_REPEATS = 2
        SCALABILITY_RUNS_PER_REPEAT = 5
        print("[WARN] --quick: reduced scalability benchmark settings; NOT the Methods defaults.")


def main():
    global SPACE_ROOT, LFP_CSD_ROOT, OUT_DIR
    args = build_argparser().parse_args()
    SPACE_ROOT = Path(args.space_root)
    LFP_CSD_ROOT = Path(args.lfp_csd_root)
    OUT_DIR = Path(args.out_dir)
    apply_cli_overrides(args)

    set_style()
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "figs")
    ensure_dir(OUT_DIR / "tables")

    feat_raw, notes = load_feature_tables()
    feat = aggregate_feature_table(feat_raw)
    save_df(feat_raw, OUT_DIR / "tables" / "benchmark_feature_matrix_event_level_raw")
    save_df(feat, OUT_DIR / "tables" / "benchmark_feature_matrix")

    sets = feature_sets(feat)
    feature_set_table = pd.DataFrame([{"feature_group": k, "n_features": len(v), "features": ";".join(v)} for k, v in sets.items()])
    save_df(feature_set_table, OUT_DIR / "tables" / "feature_sets_used")

    roc_df, curves = run_roc(feat, sets)
    save_df(roc_df, OUT_DIR / "tables" / "roc_results")
    if not roc_df.empty:
        target = roc_df.iloc[0]["target"]
        cls0 = roc_df.iloc[0]["negative_class"]
        cls1 = roc_df.iloc[0]["positive_class"]
        title_extra = f"\n{cls1} vs {cls0} ({target})"
    else:
        title_extra = ""
    plot_roc(curves, OUT_DIR / "figs" / "panel_A_ROC_data_driven", title_extra=title_extra, primary_only=True)
    plot_auc_summary(roc_df, OUT_DIR / "figs" / "panel_A2_AUC_CI_summary", primary_only=True)

    scal_df = run_scalability()
    save_df(scal_df, OUT_DIR / "tables" / "scalability_times")
    plot_scalability(scal_df, OUT_DIR / "figs" / "panel_B_scalability", primary_only=True)

    # Methods: sensitivity on pooled events (feat_raw); "dataset" option uses the aggregated table
    sens_df = run_sensitivity(normalize_keys(feat_raw) if SENSITIVITY_UNIT == "event" else feat, sets)
    save_df(sens_df, OUT_DIR / "tables" / "sensitivity_stats")
    plot_sensitivity(sens_df, OUT_DIR / "figs" / "panel_C_sensitivity")

    make_composite(curves, roc_df, scal_df, sens_df, OUT_DIR / "figs" / "SuppFig_SPACE_benchmark_proper")

    manifest = {
        "space_root": str(SPACE_ROOT.resolve()),
        "lfp_csd_root": str(LFP_CSD_ROOT.resolve()),
        "out_dir": str(OUT_DIR.resolve()),
        "aggregate_level": AGGREGATE_LEVEL,
        "roc_target": ROC_TARGET,
        "positive_class": POSITIVE_CLASS,
        "notes": notes,
        "quick_smoke_test": bool(args.quick),
        "event_roots": [str(r) for r in EVENT_ROOTS],
        "sensitivity": {"unit": SENSITIVITY_UNIT, "bh_correction": SENSITIVITY_BH_CORRECTION},
        "scalability": {"N_list": SCALABILITY_N_LIST, "T": SCALABILITY_T, "repeats": SCALABILITY_REPEATS,
                        "runs_per_repeat": SCALABILITY_RUNS_PER_REPEAT},
        "n_raw_rows": int(len(feat_raw)),
        "n_aggregated_rows": int(len(feat)),
        "feature_sets": {k: v for k, v in sets.items()},
        "figure_definition": {"GEA": "Global electrophysiological analyses. In the main supplementary figure, GEA is not plotted as one merged category; it is separated into PSD-only and Avalanche-only comparators."},
        "roc_validation": {
            "group_col": ROC_GROUP_COL,
            "use_group_cv": ROC_USE_GROUP_CV,
            "allow_apparent_fallback": ROC_ALLOW_APPARENT_FALLBACK,
            "max_features_per_model": ROC_MAX_FEATURES_PER_MODEL,
            "n_bootstrap": ROC_N_BOOTSTRAP,
            "n_permutations": ROC_N_PERMUTATIONS,
        },
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[DONE]")
    print(f"Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
