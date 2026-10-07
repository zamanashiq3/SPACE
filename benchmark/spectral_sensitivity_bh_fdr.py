# -*- coding: utf-8 -*-
"""
spectral_sensitivity_bh_fdr.py -- SPACE framework: BH-FDR spectral sensitivity comparison.

Manuscript figure(s): supports the Statistics section's "Benjamini-Hochberg FDR across
spectral comparisons" claim; feeds Supplementary Figure 8.

What it computes:
    Pairwise sensitivity of SPACE-derived synchrony features versus conventional
    channel-level PSD features, across every pair of pharmacology conditions (4-AP,
    AFDX, BIC, CNQX, ML297 -- vehicle-labeled recordings are excluded by default):
        - SPACE samples: sliding time-window means/medians of Kuramoto R and phase
          velocity (``dPhi_dt_rad_per_s``) read directly from the ``kuramoto_ALL.csv``
          tables written by ``core/phase_kuramoto.py`` (no recomputation).
        - Conventional samples: per-channel total power, peak frequency, and per-band
          power (delta/theta/alpha/beta/gamma) integrated from Welch PSD arrays stored
          in ``*_psd.npz`` files (one row per electrode).
        - Statistics: Mann-Whitney U + Cliff's delta for every condition pair per
          metric, with a Benjamini-Hochberg FDR correction applied within each metric
          across all pairs -- this is the BH-FDR correction referenced in the
          manuscript's Statistics section.
        - Visualizations: an "all metrics" bar plot for one highlighted condition pair,
          best-metric-per-pair heatmaps (one per feature family), and a per-pair
          scatter of the best SPACE vs. best conventional -log10(q).

    Relationship to ``benchmark/roc_scalability_sensitivity.py``: that script answers a
    different question -- a dataset-level (animal/slice-level) Welch's t-test sensitivity
    comparison feeding the ROC/AUC/scalability composite of Supplementary Figure 8, i.e.
    "how does SPACE's overall classification/sensitivity compare to conventional
    methods". This script instead answers "which metric best separates each specific
    pair of drug conditions, after correcting for the many pairwise comparisons",
    using a nonparametric test (Mann-Whitney U) and an effect-size measure (Cliff's
    delta) at the time-window/channel-sample level rather than the dataset level. Both
    scripts are complementary and both are needed to reproduce the manuscript's
    reported statistics; neither replaces the other.

Inputs / outputs (see data/README.md for the full schema):
    Input:  ``data/kuramoto_tables/<condition>/tables/kuramoto_ALL.csv`` (written by
            ``core/phase_kuramoto.py``) and ``data/psd_tables/*_psd.npz`` (or
            ``*_extracted_psd.npz``; a Welch-PSD pass over ``data/processed_2ms/``).
    Output: long-format sample table, pairwise Mann-Whitney/Cliff's-delta/BH-FDR table,
            best-metric-per-pair tables, and heatmap/bar/scatter figures under
            ``data/results/spectral_sensitivity_bh_fdr/`` by default.
"""

from __future__ import annotations

import argparse
import os
import glob
import json
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ===========================
# PATHS (override via CLI flags; see build_argparser)
# ===========================
SPACE_ROOT = "data/kuramoto_tables"    # contains */tables/kuramoto_ALL.csv
PSD_ROOT = "data/psd_tables"           # folder containing *_psd.npz or *_extracted_psd.npz
OUT_DIR = "data/results/spectral_sensitivity_bh_fdr"

# Optional: pick a pair to highlight with "all metrics" bar plot
PAIR_A = "4ap"
PAIR_B = "cnqx"

# ===========================
# SPACE sampling settings
# ===========================
SPACE_DT_SEC = 0.002     # kuramoto t_bin is 2 ms bins
SPACE_WINDOW_SEC = 5.0   # window length
SPACE_STEP_SEC = 2.5     # step size

# ===========================
# PSD settings (existing PSD only)
# ===========================
PSD_FMAX = 200.0
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 80),
}

# Optional: subsample channels per file (0 = all)
SUBSAMPLE_CH = 0
RNG_SEED = 0

# ===========================
# Condition handling
# ===========================
# Only these are valid "conditions" for this figure.
COND_DRUGS = ["4ap", "afdx", "bic", "cnqx", "ml297"]

# Strings that indicate vehicle. Excluded by default (see EXCLUDE_VEH).
VEH_TOKENS = ["veh", "vehicle"]

EXCLUDE_VEH = True

# ===========================
# Plot controls (avoid unreadable axis scales from extreme -log10(q))
# ===========================
# For bar plots / heatmaps, allow a cap so figures stay readable.
# The TRUE value is still annotated as q=...
CAP_NEGLOG10Q_BAR = 30.0   # set None to disable cap
CAP_NEGLOG10Q_HM = 40.0    # set None to disable cap

# ===========================
# Plot style
# ===========================
def set_nature_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "0.15",
        "xtick.color": "0.15",
        "ytick.color": "0.15",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_png_svg(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", bbox_inches="tight", dpi=600)
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)

# ===========================
# Robust parsing
# ===========================
def _norm(s: str) -> str:
    s = s.lower()
    s = s.replace("af-dx", "afdx").replace("af_dx", "afdx").replace("afd-x", "afdx")
    s = re.sub(r"[\s]+", "_", s)
    return s


def detect_drug_condition(s: str):
    """
    Return one of COND_DRUGS if present. If multiple match, choose the
    one that appears LAST in the string (fixes 'veh ... 4ap ...' cases).
    """
    ss = _norm(s)
    hits = []
    for tok in COND_DRUGS:
        idx = ss.rfind(tok)
        if idx != -1:
            hits.append((idx, tok))
    if not hits:
        return None
    hits.sort()
    return hits[-1][1]


def looks_like_vehicle(s: str) -> bool:
    ss = _norm(s)
    return any(v in ss for v in VEH_TOKENS)

# ===========================
# Small utilities
# ===========================
def integrate_band(freqs, Pxx, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if not np.any(m):
        return np.nan
    return float(np.trapezoid(Pxx[m], freqs[m]))


def neglog10_safe(p):
    if p is None or (not np.isfinite(p)) or p <= 0:
        return np.nan
    return float(-np.log10(p))


def cap_value(x, cap):
    if cap is None or (not np.isfinite(x)):
        return x
    return float(min(x, cap))

# ===========================
# Stats helpers
# ===========================
def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        return np.nan
    # O(nx*ny) but OK for typical per-metric sizes
    gt = 0; lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (nx * ny)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * m / (np.arange(1, m + 1))
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    tmp = np.empty_like(pv)
    tmp[order] = qv
    q[ok] = tmp
    return q


PAIRWISE_COLS = ["metric", "cond_A", "cond_B", "n_A", "n_B", "p_mwu", "q_bh", "cliffs_delta"]


def pairwise_stats(samples_long: pd.DataFrame):
    rows = []
    if samples_long.empty:
        return pd.DataFrame(columns=PAIRWISE_COLS)

    metrics = sorted(samples_long["metric"].dropna().unique())
    conds = sorted(samples_long["condition"].dropna().unique())

    if len(conds) < 2 or len(metrics) == 0:
        return pd.DataFrame(columns=PAIRWISE_COLS)

    for metric in metrics:
        sub = samples_long[samples_long["metric"] == metric]
        pairs = []
        pvals = []

        for a, b in combinations(conds, 2):
            xa = sub[sub["condition"] == a]["value"].to_numpy(float)
            xb = sub[sub["condition"] == b]["value"].to_numpy(float)
            xa = xa[np.isfinite(xa)]
            xb = xb[np.isfinite(xb)]

            if len(xa) < 2 or len(xb) < 2:
                p = np.nan
                d = np.nan
            else:
                p = mannwhitneyu(xa, xb, alternative="two-sided").pvalue
                d = cliffs_delta(xa, xb)

            pairs.append((a, b, len(xa), len(xb), d, p))
            pvals.append(p if np.isfinite(p) else np.nan)

        qvals = bh_fdr(np.asarray(pvals, float))

        for (a, b, na, nb, d, p), q in zip(pairs, qvals):
            rows.append({
                "metric": metric,
                "cond_A": a,
                "cond_B": b,
                "n_A": na,
                "n_B": nb,
                "p_mwu": p,
                "q_bh": q,
                "cliffs_delta": d,
            })

    return pd.DataFrame(rows, columns=PAIRWISE_COLS) if rows else pd.DataFrame(columns=PAIRWISE_COLS)

# ===========================
# SPACE loader (kuramoto_ALL.csv)
# ===========================
def extract_space_windows(kur_csv_path, cond, rec_id):
    df = pd.read_csv(kur_csv_path)
    need = {"t_bin", "R", "dPhi_dt_rad_per_s"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns in {kur_csv_path}, need {need}")

    # Time axis: kuramoto_ALL.csv t_bin is a sample index at fs_eff (phase_kuramoto.py writes
    # fs_eff into the sibling kuramoto_ALL_summary.json). Use it when present; otherwise fall
    # back to the legacy SPACE_DT_SEC (2 ms bins).
    dt_sec = SPACE_DT_SEC
    summ_path = os.path.join(os.path.dirname(kur_csv_path), "kuramoto_ALL_summary.json")
    if os.path.exists(summ_path):
        try:
            with open(summ_path, "r", encoding="utf-8") as fp:
                fs_eff = float(json.load(fp).get("fs_eff", 0.0))
            if np.isfinite(fs_eff) and fs_eff > 0:
                dt_sec = 1.0 / fs_eff
        except Exception as e:
            print(f"[WARN] could not read fs_eff from {summ_path}: {e}; using SPACE_DT_SEC")
    t = df["t_bin"].to_numpy(float) * dt_sec
    R = df["R"].to_numpy(float)
    dphi = df["dPhi_dt_rad_per_s"].to_numpy(float)

    t0 = float(np.nanmin(t))
    t1 = float(np.nanmax(t))
    win = float(SPACE_WINDOW_SEC)
    step = float(SPACE_STEP_SEC)

    out = []
    wstart = t0
    widx = 0
    while (wstart + win) <= t1:
        wend = wstart + win
        m = (t >= wstart) & (t < wend)
        if np.sum(m) >= 10:
            sid = f"{rec_id}::spacewin{widx:05d}"
            out.append({
                "family": "SPACE",
                "condition": cond,
                "recording": rec_id,
                "sample_id": sid,
                "metric": "Kuramoto_Rmean",
                "value": float(np.nanmean(R[m])),
            })
            out.append({
                "family": "SPACE",
                "condition": cond,
                "recording": rec_id,
                "sample_id": sid,
                "metric": "PhaseVel_abs_median",
                "value": float(np.nanmedian(np.abs(dphi[m]))),
            })
        widx += 1
        wstart += step

    return out


def traverse_space(space_root):
    hits = glob.glob(os.path.join(space_root, "*", "tables", "kuramoto_ALL.csv"))
    out = []
    for p in sorted(hits):
        rec_id = os.path.basename(os.path.dirname(os.path.dirname(p)))
        cond = detect_drug_condition(rec_id) or detect_drug_condition(p)

        # Exclude vehicle-like records unless user explicitly wants them
        if EXCLUDE_VEH and looks_like_vehicle(rec_id) and (cond is None):
            continue

        if cond is None:
            print(f"[WARN] SPACE: condition not found for: {rec_id}")
            continue

        out.append((cond, rec_id, p))
    return out

# ===========================
# PSD loader (NPZ)
# ===========================
def _npz_to_2d_psd(arr):
    """
    Normalize possible PSD shapes into (N, F):
      - (F,) -> (1,F)
      - (N,F) -> (N,F)
      - (N,?,F) -> flatten leading dims: (N*?,F)
    """
    a = np.asarray(arr, float)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim == 2:
        return a
    if a.ndim >= 3:
        # flatten all leading dims except last
        return a.reshape(-1, a.shape[-1])
    raise ValueError("Unsupported PSD array shape")


def extract_psd_channel_samples(npz_path, cond, rec_id, rng):
    z = np.load(npz_path, allow_pickle=True)
    if "freqs" not in z.files or "psd" not in z.files:
        raise ValueError(f"NPZ missing freqs/psd: {npz_path}")

    freqs = np.asarray(z["freqs"], float)
    psd_nf = _npz_to_2d_psd(z["psd"])  # (N, F)

    mmax = freqs <= PSD_FMAX
    freqs = freqs[mmax]
    psd_nf = psd_nf[:, mmax]

    N = psd_nf.shape[0]
    idx = np.arange(N)
    if SUBSAMPLE_CH and SUBSAMPLE_CH < N:
        idx = rng.choice(idx, size=SUBSAMPLE_CH, replace=False)

    out = []
    for i in idx:
        P = psd_nf[i]
        if not np.any(np.isfinite(P)):
            continue

        total = integrate_band(freqs, P, 1.0, PSD_FMAX)
        peak_f = float(freqs[int(np.nanargmax(P))])

        sid = f"{rec_id}::psdrow{int(i):05d}"

        out.append({
            "family": "Conventional",
            "condition": cond,
            "recording": rec_id,
            "sample_id": sid,
            "metric": "PSD_totalpower_1to200Hz",
            "value": total,
        })
        out.append({
            "family": "Conventional",
            "condition": cond,
            "recording": rec_id,
            "sample_id": sid,
            "metric": "PSD_peakfreq_1to200Hz",
            "value": peak_f,
        })
        for bn, (lo, hi) in BANDS.items():
            out.append({
                "family": "Conventional",
                "condition": cond,
                "recording": rec_id,
                "sample_id": sid,
                "metric": f"BandPower_{bn}",
                "value": integrate_band(freqs, P, lo, hi),
            })

    return out


def traverse_psd_npz(psd_root):
    pats = [
        os.path.join(psd_root, "*_psd.npz"),
        os.path.join(psd_root, "*_extracted_psd.npz"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    hits = sorted(set(hits))

    out = []
    for p in hits:
        base = os.path.basename(p)
        # Prefer drug token even if "veh" appears in name
        cond = detect_drug_condition(base) or detect_drug_condition(p)

        if cond is None:
            # if truly vehicle-only, skip by default
            if EXCLUDE_VEH and looks_like_vehicle(base):
                continue
            print(f"[WARN] PSD: condition not found for: {base}")
            continue

        out.append((cond, os.path.splitext(base)[0], p))
    return out

# ===========================
# Visualization: per-pair ALL metrics (bar)
# ===========================
def plot_pair_allmetrics_bar(stats_df, a, b, out_base):
    set_nature_style()
    if stats_df.empty:
        print("[WARN] stats_df empty; skipping pair bar.")
        return

    a = detect_drug_condition(a) or a
    b = detect_drug_condition(b) or b

    s = stats_df[((stats_df["cond_A"] == a) & (stats_df["cond_B"] == b)) |
                 ((stats_df["cond_A"] == b) & (stats_df["cond_B"] == a))].copy()
    if s.empty:
        print(f"[WARN] No stats for {a} vs {b}; skipping pair bar.")
        return

    # compute -log10(q)
    s["score_raw"] = s["q_bh"].apply(neglog10_safe)
    s["score"] = s["score_raw"].apply(lambda x: cap_value(x, CAP_NEGLOG10Q_BAR))

    # Order: Conventional first, then SPACE metrics
    def fam(metric):
        return "SPACE" if ("Kuramoto" in metric or "PhaseVel" in metric) else "Conventional"
    s["family"] = s["metric"].map(fam)
    s = s.sort_values(["family", "metric"], ascending=[True, True])

    labels = s["metric"].tolist()
    y = s["score"].to_numpy(float)
    qvals = s["q_bh"].to_numpy(float)

    fig = plt.figure(figsize=(10.8, 4.2))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(labels)), y)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"$-\log_{10}(q)$ (BH-FDR)" + (f", capped at {CAP_NEGLOG10Q_BAR}" if CAP_NEGLOG10Q_BAR else ""))
    ax.set_title(f"All metrics sensitivity: {a} vs {b}", pad=10)

    # annotate q on top (true q, not capped)
    for i, (yy, q) in enumerate(zip(y, qvals)):
        if np.isfinite(yy) and np.isfinite(q):
            txt = f"q={q:.1e}"
            ax.text(i, yy + 0.05, txt, ha="center", va="bottom", fontsize=8, rotation=90)

    save_png_svg(fig, out_base)

# ===========================
# Visualization: best metric per pair heatmap (by family)
# ===========================
def best_metric_per_pair(stats_df, family):
    """
    Return df with one row per pair (a,b):
      cond_A, cond_B, best_metric, q_bh, score=-log10(q)
    family in {"SPACE","Conventional"}.
    """
    if stats_df.empty:
        return pd.DataFrame(columns=["cond_A", "cond_B", "family", "best_metric", "q_bh", "score"])

    def fam(metric):
        return "SPACE" if ("Kuramoto" in metric or "PhaseVel" in metric) else "Conventional"

    s = stats_df.copy()
    s["family"] = s["metric"].map(fam)
    s = s[s["family"] == family].copy()
    if s.empty:
        return pd.DataFrame(columns=["cond_A", "cond_B", "family", "best_metric", "q_bh", "score"])

    s["score_raw"] = s["q_bh"].apply(neglog10_safe)
    # pick best (min q) per unordered pair
    s["pair_key"] = s.apply(lambda r: "||".join(sorted([r["cond_A"], r["cond_B"]])), axis=1)
    s = s.sort_values("q_bh").groupby("pair_key", as_index=False).head(1)

    out = pd.DataFrame({
        "cond_A": s["cond_A"].values,
        "cond_B": s["cond_B"].values,
        "family": family,
        "best_metric": s["metric"].values,
        "q_bh": s["q_bh"].values,
        "score": s["score_raw"].values,
    })
    return out


def plot_bestmetric_heatmap(best_df, cond_order, out_base, title):
    set_nature_style()
    if best_df.empty:
        print("[WARN] best_df empty; skipping best-metric heatmap.")
        return

    # restrict to available conditions
    conds = [c for c in cond_order if c in set(best_df["cond_A"]).union(set(best_df["cond_B"]))]
    if len(conds) < 2:
        print("[WARN] <2 conditions for heatmap.")
        return

    ci = {c: i for i, c in enumerate(conds)}
    M = np.full((len(conds), len(conds)), np.nan, float)

    for _, r in best_df.iterrows():
        a, b = r["cond_A"], r["cond_B"]
        if a not in ci or b not in ci:
            continue
        val = r["score"]
        val = cap_value(val, CAP_NEGLOG10Q_HM)
        ia, ib = ci[a], ci[b]
        M[ia, ib] = val
        M[ib, ia] = val

    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, interpolation="nearest")
    ax.set_xticks(range(len(conds))); ax.set_yticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right")
    ax.set_yticklabels(conds)
    ax.set_title(title + (f" (cap {CAP_NEGLOG10Q_HM})" if CAP_NEGLOG10Q_HM else ""), pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label(r"$-\log_{10}(q)$")
    save_png_svg(fig, out_base)

# ===========================
# Visualization: per-pair sensitivity scatter (best SPACE vs best Conventional)
# ===========================
def plot_bestpair_scatter(best_conv, best_space, out_base):
    set_nature_style()
    if best_conv.empty or best_space.empty:
        print("[WARN] missing best_conv or best_space; skipping scatter.")
        return

    # match by pair_key
    def keyrow(df):
        k = []
        for a, b in zip(df["cond_A"], df["cond_B"]):
            k.append("||".join(sorted([a, b])))
        return np.array(k)

    bc = best_conv.copy()
    bs = best_space.copy()
    bc["pair_key"] = keyrow(bc)
    bs["pair_key"] = keyrow(bs)

    merged = pd.merge(bc, bs, on="pair_key", suffixes=("_conv", "_space"))
    if merged.empty:
        print("[WARN] no overlapping pairs between families; skipping scatter.")
        return

    x = merged["score_conv"].to_numpy(float)
    y = merged["score_space"].to_numpy(float)

    # cap only for display; keep annotation "true"
    x_plot = np.array([cap_value(v, CAP_NEGLOG10Q_HM) for v in x], float)
    y_plot = np.array([cap_value(v, CAP_NEGLOG10Q_HM) for v in y], float)

    fig = plt.figure(figsize=(5.6, 5.2))
    ax = fig.add_subplot(111)
    ax.scatter(x_plot, y_plot)

    # y=x line (within plotted range)
    lo = np.nanmin([np.nanmin(x_plot), np.nanmin(y_plot)])
    hi = np.nanmax([np.nanmax(x_plot), np.nanmax(y_plot)])
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_xlabel(r"Best Conventional per pair  $-\log_{10}(q)$")
    ax.set_ylabel(r"Best SPACE per pair  $-\log_{10}(q)$")
    ax.set_title("Sensitivity across all drug pairs")

    # label points by pair
    for _, r in merged.iterrows():
        a = r["cond_A_conv"]; b = r["cond_B_conv"]
        px = cap_value(r["score_conv"], CAP_NEGLOG10Q_HM)
        py = cap_value(r["score_space"], CAP_NEGLOG10Q_HM)
        ax.text(px, py, f"{a}-{b}", fontsize=7, ha="left", va="bottom")

    save_png_svg(fig, out_base)

# ===========================
# MAIN
# ===========================
def build_argparser():
    p = argparse.ArgumentParser(
        description="Mann-Whitney U + Cliff's delta + BH-FDR pairwise sensitivity comparison "
                    "of SPACE (Kuramoto R / phase velocity) vs conventional PSD features, "
                    "across all pairs of pharmacology conditions."
    )
    p.add_argument("--space_root", default=SPACE_ROOT,
                    help="Root folder of */tables/kuramoto_ALL.csv, written by "
                         "core/phase_kuramoto.py (default: %(default)s)")
    p.add_argument("--psd_root", default=PSD_ROOT,
                    help="Folder of *_psd.npz / *_extracted_psd.npz Welch-PSD tables "
                         "(default: %(default)s)")
    p.add_argument("--out_dir", default=OUT_DIR,
                    help="Output folder for tables/figures (default: %(default)s)")
    p.add_argument("--pair_a", default=PAIR_A,
                    help="First condition of the highlighted all-metrics bar plot (default: %(default)s)")
    p.add_argument("--pair_b", default=PAIR_B,
                    help="Second condition of the highlighted all-metrics bar plot (default: %(default)s)")
    return p


def main():
    global SPACE_ROOT, PSD_ROOT, OUT_DIR, PAIR_A, PAIR_B
    args = build_argparser().parse_args()
    SPACE_ROOT = args.space_root
    PSD_ROOT = args.psd_root
    OUT_DIR = args.out_dir
    PAIR_A = args.pair_a
    PAIR_B = args.pair_b

    rng = np.random.default_rng(RNG_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)

    # -------- SPACE
    space = traverse_space(SPACE_ROOT)
    print(f"[INFO] SPACE records: {len(space)}")
    samples = []
    for cond, rec_id, csvp in space:
        try:
            samples += extract_space_windows(csvp, cond, rec_id)
        except Exception as e:
            print(f"[WARN] SPACE fail {rec_id}: {e}")

    # -------- PSD
    psd = traverse_psd_npz(PSD_ROOT)
    print(f"[INFO] PSD npz: {len(psd)}")
    for cond, rec_id, npzp in psd:
        try:
            samples += extract_psd_channel_samples(npzp, cond, rec_id, rng)
        except Exception as e:
            print(f"[WARN] PSD fail {rec_id}: {e}")

    if len(samples) == 0:
        print("[ERROR] No samples collected. Check --space_root / --psd_root.")
        return

    df = pd.DataFrame(samples)

    # Enforce: conditions must be in COND_DRUGS only (no veh)
    df = df[df["condition"].isin(COND_DRUGS)].copy()

    # Debug counts
    print("[INFO] condition counts (post-filter):")
    for k, v in df["condition"].value_counts().items():
        print(f"  {k}: {v}")

    if df["condition"].nunique() < 2:
        print("[ERROR] <2 conditions detected after filtering. Fix parsing or check files.")
        return

    out_samples = os.path.join(OUT_DIR, "tables", "_samples_long.tsv")
    df.to_csv(out_samples, sep="\t", index=False)
    print(f"[INFO] wrote {out_samples}")

    stats = pairwise_stats(df)
    out_stats = os.path.join(OUT_DIR, "tables", "_pairwise_stats_all_metrics.tsv")
    stats.to_csv(out_stats, sep="\t", index=False)
    print(f"[INFO] wrote {out_stats}")

    # Condition order for figures
    cond_order = ["4ap", "afdx", "bic", "cnqx", "ml297"]

    # 1) Per-pair ALL metrics bar (for selected pair)
    plot_pair_allmetrics_bar(
        stats, PAIR_A, PAIR_B,
        os.path.join(OUT_DIR, "figs", f"Fig_pairwise_allmetrics_bars_{PAIR_A}_vs_{PAIR_B}")
    )

    # 2) Best metric per pair heatmaps
    best_conv = best_metric_per_pair(stats, "Conventional")
    best_space = best_metric_per_pair(stats, "SPACE")

    best_conv.to_csv(os.path.join(OUT_DIR, "tables", "_best_metric_per_family_per_pair_CONVENTIONAL.tsv"),
                     sep="\t", index=False)
    best_space.to_csv(os.path.join(OUT_DIR, "tables", "_best_metric_per_family_per_pair_SPACE.tsv"),
                      sep="\t", index=False)

    plot_bestmetric_heatmap(
        best_conv, cond_order,
        os.path.join(OUT_DIR, "figs", "Fig_pairwise_bestmetric_heatmap_Conventional"),
        title="Best metric per pair | Conventional (-log10(q))"
    )
    plot_bestmetric_heatmap(
        best_space, cond_order,
        os.path.join(OUT_DIR, "figs", "Fig_pairwise_bestmetric_heatmap_SPACE"),
        title="Best metric per pair | SPACE (-log10(q))"
    )

    # 3) Sensitivity scatter across all pairs
    plot_bestpair_scatter(
        best_conv, best_space,
        os.path.join(OUT_DIR, "figs", "Fig_bestpair_scatter_SPACE_vs_Conventional")
    )

    print(f"[DONE]\n[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()
