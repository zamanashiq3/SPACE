"""
avalanche_statistics.py -- SPACE framework: LFP avalanche detection and power-law fits.

Manuscript figure(s): Figure 1A (conventional-methods avalanche box),
Supplementary Figure 4 (avalanche exponents, CCDFs).

What it computes:
    Streaming (memory-safe, channel-chunked) detection of negative-LFP-events (nLFP) per
    electrode, aggregation into a population activity trace over configurable
    baseline/incident/post/all time windows relative to the event t0, extraction of
    avalanches (size = summed activity, duration = consecutive active bins), and
    power-law fits (via the optional ``powerlaw`` package, with an MLE fallback when it
    is unavailable) compared against lognormal/exponential/truncated-power-law
    alternatives. Produces per-file and group-level CCDF/size-duration figures and
    summary tables.

Inputs / outputs (see data/README.md for the full schema):
    Input:  ``<slice_id>_<condition>_2ms.h5`` files under ``data/processed_2ms/`` and an
            optional 64x64 contact mask from ``data/masks/``.
    Output: per-file figures/tables (event CCDFs, avalanche size/duration summaries,
            fit parameters) and group-level summary CSVs/figures, under a results
            directory, e.g. ``data/results/avalanche/``.
"""

# ============================================================
# STREAMING LFP AVALALANCHE + POWER-LAW FACTORY
# ------------------------------------------------------------
# Memory-safe: does NOT load full (channels x time) LFP into RAM.
#
# Per file:
#   1) load meta + mask
#   2) find incident t0 from mask-only |CSD| mean in first T0_SEARCH_MS
#   3) pass-1 baseline noise estimate from LFP (masked channels, chunked)
#   4) pass-2 detect negative LFP peaks (nLFP events), chunked
#   5) build sparse activity trace for selected windows
#   6) extract avalanches from consecutive non-empty bins
#   7) fit power-law / compare with alternatives (if powerlaw pkg available)
#   8) save figures + CSV + NPZ + JSON
#
# Output structure:
#   <OUT_ROOT>/<base>/
#       figs/
#           AVAL_activity_windows.png/.svg
#           AVAL_size_ccdf_<window>.png/.svg
#           AVAL_duration_ccdf_<window>.png/.svg
#           AVAL_size_vs_duration_<window>.png/.svg
#           AVAL_event_hist_<window>.png/.svg
#       tables/
#           avalanche_window_summary.csv
#           avalanche_fit_summary.csv
#           avalanche_events.csv
#           avalanche_all_sizes.csv
#           avalanche_all_durations.csv
#           avalanche_activity_windows.npz
#           avalanche_meta.json
#
# Group-level:
#   <OUT_ROOT>/_group/
#       avalanche_fit_summary_all.csv
#       avalanche_window_summary_all.csv
#       AVAL_group_alpha_size.png/.svg
#       AVAL_group_event_rate.png/.svg
#
# Requirements:
#   numpy, pandas, h5py, matplotlib
#   optional: powerlaw
# ============================================================

import os
import re
import csv
import glob
import json
import math
import argparse
import warnings

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import powerlaw
    HAVE_POWERLAW = True
except Exception:
    HAVE_POWERLAW = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PATHS - EDIT THESE, or override via CLI flags (see build_argparser)
# ============================================================
IN_DIR   = "data/processed_2ms"
OUT_ROOT = "data/results/avalanche"
PATTERN  = "*_2ms.h5"
MASK_NPY = "data/masks/contact_mask_64x64.npy"   # set "" to disable

# ============================================================
# H5 KEYS
# ============================================================
META_KEY = "/meta"
LFP_KEY  = "/lfp/ts_2ms_full"
CSD_KEY  = "/csd/ts_2ms_full"
MASK_KEY = "/mask"

# ============================================================
# STREAM / EVENT SETTINGS
# ============================================================
CHUNK_T             = 2000   # 2000 bins = 4 sec at 2 ms
MAX_FILES           = 0      # 0 = all
USE_MASK_FOR_T0     = True
T0_SEARCH_MS        = 2000.0
BASELINE_NOISE_MS   = 500.0
REFRACTORY_MS       = 4.0
NEG_THRESH_SIGMA    = 3.0
LOCAL_MIN_ONLY      = True

# ============================================================
# AVALANCHE WINDOW SETTINGS (relative to t0)
# ============================================================
WINDOWS_MS = {
    "baseline": (-1000.0, -100.0),
    "incident": (-100.0,  400.0),
    "post":     ( 400.0, 1500.0),
    "all":      (None, None),
}

# Binning (in native bins; 1 bin = dt_ms)
# Run all to check robustness.
BIN_WIDTH_BINS = [1, 2]   # 2 ms and 4 ms if dt_ms=2 ms

# ============================================================
# FIT / PLOT SETTINGS
# ============================================================
MIN_EVENTS_FOR_ANALYSIS = 20
MIN_AVAL_FOR_FIT        = 20
CCDF_MARKER_SIZE        = 12
FIG_DPI                 = 600

# ============================================================
# STYLE
# ============================================================
def set_nature_style():
    plt.rcParams.update({
        "figure.dpi": 150,
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


def save_png_svg(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", bbox_inches="tight", dpi=FIG_DPI)
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# HELPERS
# ============================================================
def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def detect_condition_from_name(name):
    s = str(name).lower()
    if "4ap" in s:
        return "4AP"
    if "bic" in s:
        return "BIC"
    if "cnqx" in s:
        return "CNQX"
    if "ml297" in s:
        return "ML297"
    if "af-dx" in s or "afdx" in s:
        return "AFDX"
    if "veh" in s or "vehicle" in s:
        return "VEH"
    return "OTHER"


def strip_condition_from_name(name):
    s = str(name)
    s2 = re.sub(r"(?i)(?:_|-|\s)*(4ap|bic|cnqx|ml297|af-dx|afdx|veh|vehicle)(?:_|-|\s)*", "_", s)
    s2 = re.sub(r"_+", "_", s2).strip("_")
    return s2 if s2 else s


def load_meta(f):
    meta = f[META_KEY].attrs
    H  = int(meta["H"])
    W  = int(meta["W"])
    fs = float(meta["fs"])
    T2 = int(meta["T2"])
    return H, W, fs, T2


def load_mask_optional(f, H, W):
    if MASK_NPY and os.path.exists(MASK_NPY):
        m = np.load(MASK_NPY).astype(bool)
        if m.shape == (H, W):
            return m
        if m.size == H * W:
            return m.reshape(H, W)
        raise ValueError(f"MASK_NPY shape {m.shape} incompatible with {(H, W)}")

    if MASK_KEY in f:
        m = np.asarray(f[MASK_KEY][:]).astype(bool)
        if m.shape == (H, W):
            return m
        if m.size == H * W:
            return m.reshape(H, W)

    return np.ones((H, W), dtype=bool)


def ensure_chxt_shape(dset, H, W):
    if len(dset.shape) != 2:
        raise ValueError(f"Dataset must be 2D, got {dset.shape}")
    if dset.shape[0] == H * W:
        return dset, False
    if dset.shape[1] == H * W:
        return dset, True
    raise ValueError(f"Dataset shape {dset.shape} not compatible with H*W={H*W}")


def read_dset_slice_chxt(dset, ch_idx, a, b, transposed=False):
    if not transposed:
        return np.asarray(dset[ch_idx, a:b], dtype=np.float32)
    return np.asarray(dset[a:b, ch_idx], dtype=np.float32).T


def robust_sigma_from_chunk(X, axis=1):
    med = np.median(X, axis=axis, keepdims=True)
    mad = np.median(np.abs(X - med), axis=axis, keepdims=True)
    sigma = mad / 0.6745
    sigma = np.maximum(sigma, 1e-8)
    return med.astype(np.float32), sigma.astype(np.float32)


def find_t0_streaming(dset_chxt, t_start, t_end, chunk_T=2000, use_ch_idx=None, transposed=False):
    if use_ch_idx is None:
        raise ValueError("use_ch_idx should be provided for memory-safe t0 search")

    t_start = max(0, int(t_start))
    t_end   = int(t_end)
    if t_end <= t_start:
        raise ValueError("Invalid time range for t0 search")

    best_t = t_start
    best_v = -np.inf

    for a in range(t_start, t_end, chunk_T):
        b = min(t_end, a + chunk_T)
        X = read_dset_slice_chxt(dset_chxt, use_ch_idx, a, b, transposed=transposed)
        score = np.mean(np.abs(X), axis=0)
        j = int(np.argmax(score))
        v = float(score[j])
        if v > best_v:
            best_v = v
            best_t = a + j

    return best_t, best_v


def estimate_channel_noise_streaming(dset_lfp, mask_idx, T, base_end, chunk_T=2000, transposed=False):
    # Estimate one robust sigma per channel from baseline period only.
    n_mask = len(mask_idx)
    sigmas = np.full(n_mask, np.nan, dtype=np.float32)

    # Because baseline window is small, read it in one shot if possible.
    base_end = max(1, min(T, int(base_end)))
    X = read_dset_slice_chxt(dset_lfp, mask_idx, 0, base_end, transposed=transposed)
    _, sigma = robust_sigma_from_chunk(X, axis=1)
    sigmas[:] = sigma[:, 0]
    sigmas = np.maximum(sigmas, 1e-8)
    return sigmas


def detect_nlfp_events_streaming(dset_lfp, mask_idx, T, sigma_by_ch, dt_ms, chunk_T=2000, transposed=False):
    # Returns sparse event table: [t_bin, ch_flat, amp]
    refractory_bins = max(1, int(round(REFRACTORY_MS / dt_ms)))
    overlap = max(2, refractory_bins + 2)
    n_mask = len(mask_idx)

    event_t = []
    event_ch = []
    event_amp = []

    last_keep_global = np.full(n_mask, -10**9, dtype=np.int64)

    start = 0
    while start < T:
        stop = min(T, start + chunk_T)
        ext_a = max(0, start - 1)
        ext_b = min(T, stop + 1)

        X = read_dset_slice_chxt(dset_lfp, mask_idx, ext_a, ext_b, transposed=transposed)
        med = np.median(X, axis=1, keepdims=True)
        thr = med - NEG_THRESH_SIGMA * sigma_by_ch[:, None]
        below = X < thr

        if LOCAL_MIN_ONLY:
            left  = np.roll(X, 1, axis=1)
            right = np.roll(X, -1, axis=1)
            is_min = (X <= left) & (X < right)
            peaks = below & is_min
            peaks[:, 0] = False
            peaks[:, -1] = False
        else:
            peaks = below

        # Keep only events in true chunk [start, stop)
        cols = np.where(peaks)
        for i_ch, col_local in zip(cols[0], cols[1]):
            t_global = ext_a + int(col_local)
            if not (start <= t_global < stop):
                continue
            if (t_global - last_keep_global[i_ch]) < refractory_bins:
                continue
            last_keep_global[i_ch] = t_global
            event_t.append(t_global)
            event_ch.append(int(mask_idx[i_ch]))
            event_amp.append(float(abs(X[i_ch, col_local])))

        start = stop

    if len(event_t) == 0:
        return pd.DataFrame(columns=["t_bin", "ch_flat", "amp_abs"])

    ev = pd.DataFrame({
        "t_bin": np.asarray(event_t, dtype=np.int64),
        "ch_flat": np.asarray(event_ch, dtype=np.int32),
        "amp_abs": np.asarray(event_amp, dtype=np.float32),
    })
    ev = ev.sort_values(["t_bin", "ch_flat"]).reset_index(drop=True)
    return ev


def aggregate_activity_from_events(ev_t, ev_amp, t_start, t_end, bin_width_bins=1, weighted=False):
    if t_end <= t_start:
        return np.zeros(0, dtype=np.float32)

    n_bins = int(math.ceil((t_end - t_start) / float(bin_width_bins)))
    act = np.zeros(n_bins, dtype=np.float32)

    keep = (ev_t >= t_start) & (ev_t < t_end)
    if not np.any(keep):
        return act

    idx = ((ev_t[keep] - t_start) // bin_width_bins).astype(np.int64)
    vals = ev_amp[keep].astype(np.float32) if weighted else np.ones(np.sum(keep), dtype=np.float32)
    np.add.at(act, idx, vals)
    return act


def extract_avalanches(activity):
    sizes = []
    durations = []
    starts = []
    ends = []

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

    return {
        "size": np.asarray(sizes, dtype=np.float32),
        "duration": np.asarray(durations, dtype=np.int32),
        "start_bin": np.asarray(starts, dtype=np.int32),
        "end_bin": np.asarray(ends, dtype=np.int32),
    }


def fit_heavytail_discrete(x):
    out = {
        "n": int(len(x)),
        "alpha": np.nan,
        "xmin": np.nan,
        "sigma": np.nan,
        "R_lognormal": np.nan,
        "p_lognormal": np.nan,
        "R_exponential": np.nan,
        "p_exponential": np.nan,
        "R_truncated": np.nan,
        "p_truncated": np.nan,
        "fit_ok": 0,
        "pkg": "none",
    }

    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = x[x >= 1]
    x = np.round(x).astype(int)

    if len(x) < MIN_AVAL_FOR_FIT:
        return out

    if HAVE_POWERLAW:
        try:
            fit = powerlaw.Fit(x, discrete=True, verbose=False)
            out.update({
                "alpha": float(fit.alpha),
                "xmin": float(fit.xmin),
                "sigma": float(getattr(fit, "sigma", np.nan)),
                "fit_ok": 1,
                "pkg": "powerlaw",
            })
            try:
                R, p = fit.distribution_compare("power_law", "lognormal")
                out["R_lognormal"] = float(R)
                out["p_lognormal"] = float(p)
            except Exception:
                pass
            try:
                R, p = fit.distribution_compare("power_law", "exponential")
                out["R_exponential"] = float(R)
                out["p_exponential"] = float(p)
            except Exception:
                pass
            try:
                R, p = fit.distribution_compare("power_law", "truncated_power_law")
                out["R_truncated"] = float(R)
                out["p_truncated"] = float(p)
            except Exception:
                pass
            return out
        except Exception:
            pass

    # Fallback: fixed xmin=1 discrete MLE approximation
    xmin = 1
    xx = x[x >= xmin]
    if len(xx) >= MIN_AVAL_FOR_FIT:
        alpha = 1.0 + len(xx) / np.sum(np.log(xx / (xmin - 0.5)))
        out.update({
            "alpha": float(alpha),
            "xmin": float(xmin),
            "fit_ok": 1,
            "pkg": "fallback_mle",
        })
    return out


def ccdf_xy(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    uniq = np.unique(xs)
    y = np.array([np.mean(xs >= u) for u in uniq], dtype=np.float64)
    return uniq, y


def mean_size_by_duration(sizes, durations):
    sizes = np.asarray(sizes, dtype=float)
    durations = np.asarray(durations, dtype=int)
    if len(sizes) == 0:
        return np.array([]), np.array([]), np.array([])
    u = np.unique(durations)
    m = np.array([np.mean(sizes[durations == d]) for d in u], dtype=float)
    n = np.array([np.sum(durations == d) for d in u], dtype=int)
    return u, m, n


def plot_activity_windows(base, windows_payload, dt_ms, out_base):
    set_nature_style()
    keys = list(windows_payload.keys())
    fig, axs = plt.subplots(len(keys), 1, figsize=(8.0, 2.1 * len(keys)), sharex=False)
    if len(keys) == 1:
        axs = [axs]
    for ax, key in zip(axs, keys):
        item = windows_payload[key]
        act = item["activity"]
        t_ms = np.arange(len(act), dtype=np.float64) * item["bin_width_bins"] * dt_ms + item["window_ms_start"]
        ax.plot(t_ms, act, linewidth=0.8)
        ax.set_ylabel("Activity")
        ax.set_title(f"{key} | bin={item['bin_width_bins']} x dt", fontsize=9)
    axs[-1].set_xlabel("Time relative to t0 (ms)")
    fig.suptitle(f"{base} | Avalanche activity windows", fontsize=10, y=0.995)
    save_png_svg(fig, out_base)


def plot_event_hist(activity, base, window_name, dt_ms, bin_width_bins, out_base):
    set_nature_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    t_ms = np.arange(len(activity), dtype=np.float64) * bin_width_bins * dt_ms
    ax.plot(t_ms, activity, linewidth=0.8)
    ax.set_xlabel("Window time (ms)")
    ax.set_ylabel("Event count/bin")
    ax.set_title(f"{base} | {window_name} event activity")
    save_png_svg(fig, out_base)


def plot_ccdf(data, fit_info, xlabel, title, out_base):
    set_nature_style()
    x, y = ccdf_xy(data)
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    if len(x):
        ax.scatter(x, y, s=CCDF_MARKER_SIZE)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ttl = title
    if np.isfinite(fit_info.get("alpha", np.nan)):
        ttl += f"\nalpha={fit_info['alpha']:.3f}, xmin={fit_info['xmin']:.1f}, pkg={fit_info['pkg']}"
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CCDF")
    save_png_svg(fig, out_base)


def plot_size_vs_duration(sizes, durations, base, window_name, out_base):
    set_nature_style()
    u, m, n = mean_size_by_duration(sizes, durations)
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    if len(u):
        ax.scatter(u, m, s=15)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Duration (bins)")
    ax.set_ylabel("Mean size")
    ax.set_title(f"{base} | {window_name} size-duration")
    save_png_svg(fig, out_base)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, ensure_ascii=False)


# ============================================================
# CORE PER-FILE PROCESSOR
# ============================================================
def process_one_file(h5_path):
    base = os.path.splitext(os.path.basename(h5_path))[0]
    cond = detect_condition_from_name(base)
    group_key = strip_condition_from_name(base)

    out_dir  = os.path.join(OUT_ROOT, safe_name(base))
    figs_dir = os.path.join(out_dir, "figs")
    tabs_dir = os.path.join(out_dir, "tables")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tabs_dir, exist_ok=True)

    print(f"[INFO] Processing {base}")

    with h5py.File(h5_path, "r") as f:
        H, W, fs, T2 = load_meta(f)
        dt_ms = 1000.0 / float(fs)

        lfp_dset, lfp_transposed = ensure_chxt_shape(f[LFP_KEY], H, W)
        csd_dset, csd_transposed = ensure_chxt_shape(f[CSD_KEY], H, W)

        mask2d = load_mask_optional(f, H, W)
        mask_idx = np.flatnonzero(mask2d.reshape(-1))
        if mask_idx.size == 0:
            raise ValueError("Mask has zero valid contacts")

        # t0 from CSD, same event-aligned logic as old pipeline
        t0_end = min(T2, int(round(T0_SEARCH_MS / dt_ms)))
        ch_idx_for_t0 = mask_idx if USE_MASK_FOR_T0 else np.arange(H * W, dtype=int)
        t0, t0_score = find_t0_streaming(
            csd_dset,
            t_start=0,
            t_end=t0_end,
            chunk_T=CHUNK_T,
            use_ch_idx=ch_idx_for_t0,
            transposed=csd_transposed,
        )

        # pass-1: baseline sigma per channel
        base_end = min(T2, int(round(BASELINE_NOISE_MS / dt_ms)))
        sigma_by_ch = estimate_channel_noise_streaming(
            lfp_dset,
            mask_idx,
            T2,
            base_end=base_end,
            chunk_T=CHUNK_T,
            transposed=lfp_transposed,
        )

        # pass-2: event detection
        ev = detect_nlfp_events_streaming(
            lfp_dset,
            mask_idx,
            T2,
            sigma_by_ch,
            dt_ms=dt_ms,
            chunk_T=CHUNK_T,
            transposed=lfp_transposed,
        )

    # Save raw events
    if len(ev):
        ev["time_ms"] = ev["t_bin"].astype(np.float64) * dt_ms
        ev["y"] = (ev["ch_flat"] // W).astype(np.int32)
        ev["x"] = (ev["ch_flat"] % W).astype(np.int32)
        ev.to_csv(os.path.join(tabs_dir, "avalanche_events.csv"), index=False)
    else:
        ev = pd.DataFrame(columns=["t_bin", "ch_flat", "amp_abs", "time_ms", "y", "x"])
        ev.to_csv(os.path.join(tabs_dir, "avalanche_events.csv"), index=False)

    meta = {
        "base": base,
        "condition": cond,
        "group": group_key,
        "H": int(H),
        "W": int(W),
        "fs": float(fs),
        "dt_ms": float(dt_ms),
        "T2": int(T2),
        "n_mask": int(mask_idx.size),
        "t0_bin": int(t0),
        "t0_ms": float(t0 * dt_ms),
        "t0_score": float(t0_score),
        "threshold_sigma": float(NEG_THRESH_SIGMA),
        "refractory_ms": float(REFRACTORY_MS),
        "local_min_only": bool(LOCAL_MIN_ONLY),
        "have_powerlaw_pkg": bool(HAVE_POWERLAW),
    }
    save_json(os.path.join(tabs_dir, "avalanche_meta.json"), meta)

    ev_t = ev["t_bin"].to_numpy(dtype=np.int64) if len(ev) else np.zeros(0, dtype=np.int64)
    ev_amp = ev["amp_abs"].to_numpy(dtype=np.float32) if len(ev) else np.zeros(0, dtype=np.float32)

    window_rows = []
    fit_rows = []
    size_rows = []
    dur_rows = []
    activity_npz = {}
    first_payload_for_plot = {}

    for bw in BIN_WIDTH_BINS:
        for wname, (ms_a, ms_b) in WINDOWS_MS.items():
            if ms_a is None:
                t_a = 0
            else:
                t_a = max(0, int(round(t0 + ms_a / dt_ms)))
            if ms_b is None:
                t_b = T2
            else:
                t_b = min(T2, int(round(t0 + ms_b / dt_ms)))
            if t_b <= t_a:
                continue

            act = aggregate_activity_from_events(ev_t, ev_amp, t_a, t_b, bin_width_bins=bw, weighted=False)
            ava = extract_avalanches(act)

            sizes = np.round(ava["size"]).astype(int)
            durations = ava["duration"].astype(int)

            # Save detailed rows
            for i in range(len(sizes)):
                size_rows.append({
                    "file": base,
                    "condition": cond,
                    "group": group_key,
                    "window": wname,
                    "bin_width_bins": bw,
                    "avalanche_id": i,
                    "size": int(sizes[i]),
                    "start_bin": int(ava["start_bin"][i]),
                    "end_bin": int(ava["end_bin"][i]),
                })
                dur_rows.append({
                    "file": base,
                    "condition": cond,
                    "group": group_key,
                    "window": wname,
                    "bin_width_bins": bw,
                    "avalanche_id": i,
                    "duration": int(durations[i]),
                })

            fit_size = fit_heavytail_discrete(sizes)
            fit_dur = fit_heavytail_discrete(durations)

            fit_rows.append({
                "file": base,
                "condition": cond,
                "group": group_key,
                "window": wname,
                "bin_width_bins": bw,
                "metric": "size",
                **fit_size,
            })
            fit_rows.append({
                "file": base,
                "condition": cond,
                "group": group_key,
                "window": wname,
                "bin_width_bins": bw,
                "metric": "duration",
                **fit_dur,
            })

            nonzero = act[act > 0]
            window_rows.append({
                "file": base,
                "condition": cond,
                "group": group_key,
                "window": wname,
                "bin_width_bins": bw,
                "n_events": int(len(ev_t[(ev_t >= t_a) & (ev_t < t_b)])),
                "n_active_bins": int(np.sum(act > 0)),
                "n_total_bins": int(len(act)),
                "event_rate_per_bin_active_mean": float(np.mean(nonzero)) if len(nonzero) else 0.0,
                "event_rate_per_bin_all_mean": float(np.mean(act)) if len(act) else 0.0,
                "n_avalanches": int(len(sizes)),
                "size_mean": float(np.mean(sizes)) if len(sizes) else 0.0,
                "size_median": float(np.median(sizes)) if len(sizes) else 0.0,
                "size_max": float(np.max(sizes)) if len(sizes) else 0.0,
                "duration_mean_bins": float(np.mean(durations)) if len(durations) else 0.0,
                "duration_median_bins": float(np.median(durations)) if len(durations) else 0.0,
                "duration_max_bins": float(np.max(durations)) if len(durations) else 0.0,
                "window_start_ms_rel_t0": float(0.0 if ms_a is None else ms_a),
                "window_end_ms_rel_t0": float((T2 - t0) * dt_ms if ms_b is None else ms_b),
            })

            npz_key = f"{wname}_bw{bw}"
            activity_npz[f"activity_{npz_key}"] = act.astype(np.float32)
            activity_npz[f"sizes_{npz_key}"] = sizes.astype(np.int32)
            activity_npz[f"durations_{npz_key}"] = durations.astype(np.int32)
            activity_npz[f"window_start_ms_{npz_key}"] = np.float32(0.0 if ms_a is None else ms_a)
            activity_npz[f"window_end_ms_{npz_key}"] = np.float32((T2 - t0) * dt_ms if ms_b is None else ms_b)
            activity_npz[f"bin_width_bins_{npz_key}"] = np.int32(bw)

            # Single-file figs for every window/bin-width
            suffix = f"{wname}_bw{bw}"
            plot_event_hist(act, base, suffix, dt_ms, bw, os.path.join(figs_dir, f"AVAL_event_hist_{suffix}"))
            plot_ccdf(sizes, fit_size, "Avalanche size", f"{base} | {suffix} size CCDF", os.path.join(figs_dir, f"AVAL_size_ccdf_{suffix}"))
            plot_ccdf(durations, fit_dur, "Avalanche duration (bins)", f"{base} | {suffix} duration CCDF", os.path.join(figs_dir, f"AVAL_duration_ccdf_{suffix}"))
            plot_size_vs_duration(sizes, durations, base, suffix, os.path.join(figs_dir, f"AVAL_size_vs_duration_{suffix}"))

            if bw == BIN_WIDTH_BINS[0]:
                first_payload_for_plot[suffix] = {
                    "activity": act,
                    "bin_width_bins": bw,
                    "window_ms_start": 0.0 if ms_a is None else ms_a,
                }

    # save per-file tables
    pd.DataFrame(window_rows).to_csv(os.path.join(tabs_dir, "avalanche_window_summary.csv"), index=False)
    pd.DataFrame(fit_rows).to_csv(os.path.join(tabs_dir, "avalanche_fit_summary.csv"), index=False)
    pd.DataFrame(size_rows).to_csv(os.path.join(tabs_dir, "avalanche_all_sizes.csv"), index=False)
    pd.DataFrame(dur_rows).to_csv(os.path.join(tabs_dir, "avalanche_all_durations.csv"), index=False)
    np.savez_compressed(os.path.join(tabs_dir, "avalanche_activity_windows.npz"), **activity_npz)

    if first_payload_for_plot:
        plot_activity_windows(base, first_payload_for_plot, dt_ms, os.path.join(figs_dir, "AVAL_activity_windows"))

    return {
        "base": base,
        "condition": cond,
        "group": group_key,
        "meta": meta,
        "window_rows": window_rows,
        "fit_rows": fit_rows,
    }


# ============================================================
# GROUP PLOTS
# ============================================================
def group_barplot(df, value_col, title, out_base):
    if df.empty or value_col not in df.columns:
        return
    set_nature_style()
    stats = df.groupby("condition", dropna=False)[value_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.bar(stats["condition"], stats[value_col])
    ax.set_ylabel(value_col)
    ax.set_title(title)
    save_png_svg(fig, out_base)


# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Streaming LFP avalanche detection + power-law fits")
    p.add_argument("--in_dir", default=IN_DIR, help="Folder of 2ms H5 files (default: %(default)s)")
    p.add_argument("--out_root", default=OUT_ROOT, help="Output root folder (default: %(default)s)")
    p.add_argument("--pattern", default=PATTERN, help="Glob pattern for H5 files (default: %(default)s)")
    p.add_argument("--mask", default=MASK_NPY, help="Optional 64x64 contact mask .npy, empty string to disable (default: %(default)s)")
    return p


# ============================================================
# MAIN
# ============================================================
def main():
    global IN_DIR, OUT_ROOT, PATTERN, MASK_NPY
    args = build_argparser().parse_args()
    IN_DIR, OUT_ROOT, PATTERN, MASK_NPY = args.in_dir, args.out_root, args.pattern, args.mask

    os.makedirs(OUT_ROOT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IN_DIR, PATTERN)))
    if MAX_FILES > 0:
        files = files[:MAX_FILES]

    print(f"[INFO] Found {len(files)} files in {IN_DIR}")
    all_window_rows = []
    all_fit_rows = []

    for p in files:
        try:
            res = process_one_file(p)
            all_window_rows.extend(res["window_rows"])
            all_fit_rows.extend(res["fit_rows"])
            print(f"[OK] {res['base']}")
        except Exception as e:
            print(f"[ERROR] {os.path.basename(p)}: {e}")

    group_dir = os.path.join(OUT_ROOT, "_group")
    os.makedirs(group_dir, exist_ok=True)

    dfw = pd.DataFrame(all_window_rows)
    dff = pd.DataFrame(all_fit_rows)
    dfw.to_csv(os.path.join(group_dir, "avalanche_window_summary_all.csv"), index=False)
    dff.to_csv(os.path.join(group_dir, "avalanche_fit_summary_all.csv"), index=False)

    if not dff.empty:
        tmp = dff[(dff["metric"] == "size") & (dff["window"] == "incident") & (dff["bin_width_bins"] == BIN_WIDTH_BINS[0])]
        group_barplot(tmp, "alpha", "Group alpha (size, incident)", os.path.join(group_dir, "AVAL_group_alpha_size"))

    if not dfw.empty:
        tmp = dfw[(dfw["window"] == "incident") & (dfw["bin_width_bins"] == BIN_WIDTH_BINS[0])]
        group_barplot(tmp, "event_rate_per_bin_all_mean", "Group mean event rate (incident)", os.path.join(group_dir, "AVAL_group_event_rate"))

    print("[DONE]")


if __name__ == "__main__":
    main()
