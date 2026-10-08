# -*- coding: utf-8 -*-
"""
roi_phase_gradient_pgd.py -- SPACE framework: per-ROI phase-gradient-directionality (PGD).

Manuscript figure(s): Figure 5A-C (per-ROI vector fields, polar histograms),
Figure 7A,E (pharmacology CSD kymographs, polar direction plots).

What it computes:
    Per-anatomical-ROI (M1/M2/S1/CA1/CA3/DG) generalized phase (Davis/Muller-style),
    phase-gradient-directionality (PGD, a traveling-wave metric; Rubino 2006, Townsend &
    Gong 2018) and Kuramoto order parameter R(t)/mean-phase psi(t) over an event window,
    plus electrode-level vector-grid figures, a whole-slice ROI-overview QC figure, and
    an Excel export of all per-ROI time series and vectors. This is the final (v4)
    version of this analysis: it fixes a t0-window truncation bug present in an earlier
    draft where the largest transient near the end of a recording could truncate the
    post-t0 window (e.g. the AF-DX condition), by guarding the t0 search so a full
    [PRE_MS, POST_MS] window always fits.

Inputs / outputs (see data/README.md for the full schema):
    Input:  ``<slice_id>_<condition>_2ms.h5`` files under ``data/processed_2ms/<group>/``,
            a 64x64 contact mask from ``data/masks/``, and a per-slice ROI-definition CSV
            (r0/r1/c0/c1 electrode-grid boxes for M1/M2/S1[SSP]/CA1/CA3/DG) from
            ``data/roi_definitions/`` (as produced by ``core/roi_lfp_csd_metrics.py``).
    Output: per-ROI phase kymographs, PGD/R/psi time series (CSV + .npz + Excel),
            vector-grid and ROI-overview figures, written under a results directory,
            e.g. ``data/results/roi_phase_pgd/``.

Method refs: Generalized phase (Davis, Muller et al. 2020);
PGD / travelling waves (Rubino 2006; Townsend & Gong 2018).
"""

import os
import re
import csv
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

# ============================================================
# CONFIG - EDIT THESE, or override via CLI flags (see build_argparser)
# ============================================================
DATA_DIR = "data/processed_2ms"
FILE_PATTERN = "*_2ms.h5"
OUT_DIR = "data/results/roi_phase_pgd"

MASK_PATH = "data/masks/contact_mask_64x64.npy"
ROI_CSV = "data/roi_definitions/roi_definitions.csv"

ROI_ORDER = ["M1", "M2", "SSP", "CA1", "CA3", "DG"]   # NOTE: older csvs use "SSP"; "S1" is accepted as an alias
ROI_ALIASES = {"S1": "SSP"}                            # csv name -> canonical ROI_ORDER name

# Window relative to t0  (REQUESTED: pre 10, post 500)
PRE_MS = 10
POST_MS = 500
PAD_MS = 100                 # read extra each side for phase edges, then trim
GUARD_T0 = True              # restrict t0 search so a full PRE/POST window fits
MIN_ROI_ELECTRODES = 4       # ROIs with fewer contact electrodes are skipped (reported)

GP_BAND = None               # None = full band; e.g. (2.0, 40.0) wideband

# v3 additions
SNAP_MS_GRID = [0, 100, 200, 300, 400]        # times for the 4x4 grid snapshots
SHOW_GRID_QUIVER = False                       # phase-only grid snapshots (vectors are in the vector-grid figure)
SAVE_EXCEL = True                              # write R + phase values to .xlsx per drug

# v4 additions: per-electrode vectors + mean (resultant) + Total, per the Fig-4 sketch
VEC_SNAP_MS = [0, 100, 200, 300, 400]          # times for the vector-grid figure
VECTOR_KIND = "phase"                          # "phase" = unit phase vectors (mean = R,psi)
#                                                "gradient" = phase-gradient (propagation)
VEC_THIN_COLOR = "0.25"                         # individual per-electrode arrows
VEC_MEAN_COLOR = "red"                          # thick mean/resultant arrow
SAVE_VECTOR_GRID = True
MEAN_UNIT_CELLS = 1.0                            # grid cells drawn per 1.0 of mean-vector magnitude
SCALE_TICKS = [1, 2]                            # scale-bar ticks (magnitude, a.u.)

# appearance
PHASE_CMAP = "twilight_shifted"
PGD_THRESHOLD = 0.7
PGD_COLOR = "#1d3557"
ORDER_ALONG_PROPAGATION = True
OVERVIEW_BG_CMAP = "Greys"
ROI_BOX_COLOR = "orange"
MASK_OUTLINE_COLOR = "black"

CHUNK_SIZE = 64
FIG_DPI = 600
SAVE_SVG = True

# ============================================================
# STYLE / SAVE
# ============================================================
def set_nature_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": FIG_DPI, "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
        "axes.linewidth": 0.8, "axes.edgecolor": "0.15",
        "xtick.color": "0.15", "ytick.color": "0.15",
        "figure.facecolor": "white", "axes.facecolor": "white"})


def save_fig(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=FIG_DPI, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def sanitize_name(x):
    x = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(x).strip())
    return x if x else "ROI"


def wrap_phase(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ============================================================
# FILE / META / ROI
# ============================================================
def get_h5_files():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
    if not paths:
        raise FileNotFoundError(f"No H5 files in {DATA_DIR} matching {FILE_PATTERN}")
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
        return m.reshape(H, W) if m.size == H * W else m
    return np.ones((H, W), bool)


def load_roi_definitions(roi_csv):
    if not os.path.exists(roi_csv):
        raise FileNotFoundError(f"ROI definitions not found: {roi_csv}")
    table = {}
    with open(roi_csv) as f:
        for row in csv.DictReader(f):
            table[sanitize_name(row["roi_name"])] = (
                int(row["r0"]), int(row["r1"]), int(row["c0"]), int(row["c1"]))
    # map aliases (e.g. S1 -> SSP) so the canonical ROI_ORDER is found
    for alias, canon in ROI_ALIASES.items():
        if alias in table and canon not in table:
            table[canon] = table.pop(alias)
    order = [n for n in ROI_ORDER if n in table] + [n for n in table if n not in ROI_ORDER]
    missing = [n for n in ROI_ORDER if n not in table]
    if missing:
        print(f"[MISSING] ROI_ORDER names NOT in {os.path.basename(roi_csv)}: {missing}")
        print("          -> re-run your ROI picker so roi_definitions.csv has all 6 ROIs.")
    return [(n, table[n]) for n in order], missing

# ============================================================
# SIGNAL
# ============================================================
def find_t0_chunked(dset, mask_idx, pre_s, post_s, chunk=CHUNK_SIZE, guard=GUARD_T0):
    n_ch, T = dset.shape
    score = np.zeros(T)
    for j0 in range(0, len(mask_idx), chunk):
        X = dset[mask_idx[j0:j0 + chunk], :].astype(np.float32)
        score += np.abs(X).sum(0)
    if guard:                       # only allow t0 where a full window fits
        lo, hi = pre_s, T - post_s
        if hi > lo:
            m = np.full(T, -np.inf)
            m[lo:hi] = score[lo:hi]
            score = m
    return int(np.argmax(score))


def read_window_chunked(dset, a, b, chunk=CHUNK_SIZE):
    n_ch = dset.shape[0]
    out = np.empty((n_ch, b - a), np.float32)
    for i0 in range(0, n_ch, chunk):
        i1 = min(i0 + chunk, n_ch)
        out[i0:i1] = dset[i0:i1, a:b].astype(np.float32)
    return out


def generalized_phase(x, fs, band=GP_BAND):
    """Generalized phase (Davis/Muller 2020) on a small channel set. (Nc,T)->(Nc,T)."""
    X = np.asarray(x, np.float64)
    X = X - X.mean(1, keepdims=True)
    nyq = 0.5 * fs
    if band is not None:
        b, a = butter(3, [band[0] / nyq, band[1] / nyq], btype="band")
        X = filtfilt(b, a, X, axis=1)
    phu = np.unwrap(np.angle(hilbert(X, axis=1)), axis=1)
    T = phu.shape[1]; idx = np.arange(T); out = np.empty_like(phu)
    for c in range(phu.shape[0]):
        p = phu[c]; good = np.concatenate([[True], np.diff(p) >= 0])
        out[c] = np.interp(idx, idx[good], p[good]) if (not good.all() and good.sum() >= 2) else p
    return wrap_phase(out).astype(np.float32)

# ============================================================
# ROI GEOMETRY / PGD (vectorized over time)
# ============================================================
def roi_grid_channels(box, W, mask):
    r0, r1, c0, c1 = box
    nr, nc = r1 - r0, c1 - c0
    ci, lr, lc = [], [], []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if mask[r, c]:
                ci.append(r * W + c); lr.append(r - r0); lc.append(c - c0)
    return np.asarray(ci, int), np.asarray(lr, int), np.asarray(lc, int), nr, nc


def roi_pgd_and_dir(phase_roi, lr, lc, nr, nc):
    """Vectorized over time. Returns PGD(t), dominant (gx,gy)."""
    T = phase_roi.shape[1]
    grid = np.full((nr, nc, T), np.nan, np.float64)
    grid[lr, lc, :] = phase_roi
    u = np.exp(1j * grid)
    gy = np.imag(np.conj(u) * np.gradient(u, axis=0)).reshape(nr * nc, T)
    gx = np.imag(np.conj(u) * np.gradient(u, axis=1)).reshape(nr * nc, T)
    mag = np.hypot(gx, gy)
    mean_mag = np.nanmean(mag, axis=0)
    mgx = np.nanmean(gx, axis=0); mgy = np.nanmean(gy, axis=0)
    pgd = np.hypot(mgx, mgy) / (mean_mag + 1e-12)
    return pgd, float(np.nanmean(gx)), float(np.nanmean(gy)), mgx, mgy


# Methods ("Phase gradients and propagation direction"): a wave travels from earlier to later
# phase, so the instantaneous propagation direction is opposite to the ROI-mean gradient,
# theta_k = atan2(-<gy>, -<gx>). Over the K time points of a window the dominant angle and the
# directional concentration are the circular mean direction and the resultant vector length.
DIRECTION_WINDOWS_MS = {"full_0_500": (0.0, 500.0), "early_0_250": (0.0, 250.0), "late_250_500": (250.0, 500.0)}
DIR_COLS = [f"{m}_{k}" for k in DIRECTION_WINDOWS_MS for m in ("dir_angle_deg", "dir_resultant")]


def propagation_direction(mgx, mgy):
    """Instantaneous propagation direction theta_k (rad) from the ROI-mean gradient."""
    return np.arctan2(-mgy, -mgx)


def circular_stats(theta):
    """Circular mean direction (deg, [0,360)) and resultant vector length of angles (rad)."""
    th = np.asarray(theta, float); th = th[np.isfinite(th)]
    if th.size == 0:
        return np.nan, np.nan
    z = np.exp(1j * th).mean()
    return float(np.degrees(np.angle(z)) % 360.0), float(np.abs(z))


def direction_window_stats(theta, t_ms, windows=DIRECTION_WINDOWS_MS):
    out = {}
    for key, (lo, hi) in windows.items():
        sel = (t_ms >= lo) & (t_ms <= hi)
        ang, rl = circular_stats(theta[sel])
        out[f"dir_angle_deg_{key}"] = ang
        out[f"dir_resultant_{key}"] = rl
    return out


def save_direction_polar(theta_by_roi, t_ms, drug, out_base):
    """Fig. 5C-style polar summary: per-ROI dominant direction (arrow length = resultant length)
    and the pooled all-ROI direction, plus the early/late trajectory (Fig. 5B)."""
    fig = plt.figure(figsize=(6.4, 3.0))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    allth = []
    for i, (name, th) in enumerate(theta_by_roi.items()):
        sel = (t_ms >= 0) & (t_ms <= 500)
        ang, rl = circular_stats(th[sel]); allth.append(th[sel])
        if np.isfinite(ang):
            ax.annotate("", xy=(np.radians(ang), rl), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=cols[i % 10], lw=1.4))
            ax.plot([], [], color=cols[i % 10], label=f"{name} {ang:.0f}deg R={rl:.2f}")
    if allth:
        ang, rl = circular_stats(np.concatenate(allth))
        ax.annotate("", xy=(np.radians(ang), rl), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="k", lw=2.2))
        ax.plot([], [], color="k", lw=2, label=f"all ROIs {ang:.0f}deg R={rl:.2f}")
    ax.set_ylim(0, 1); ax.set_yticks([0.5, 1.0]); ax.set_theta_zero_location("E")
    ax.set_title(f"{drug}: propagation direction (0-500 ms)", fontsize=7)
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    for i, (name, th) in enumerate(theta_by_roi.items()):
        ax2.plot(t_ms, np.degrees(th) % 360, ".", ms=1.2, color=cols[i % 10], label=name)
    ax2.axvline(250, color="0.5", lw=0.7, ls="--")
    ax2.set_xlim(t_ms.min(), t_ms.max()); ax2.set_ylim(0, 360)
    ax2.set_xlabel("time from onset (ms)"); ax2.set_ylabel("propagation angle (deg)")
    ax2.set_title("angle trajectory (early | late)", fontsize=7)
    fig.tight_layout()
    save_fig(fig, out_base)


def order_electrodes(lr, lc, gx, gy):
    if not ORDER_ALONG_PROPAGATION or (abs(gx) < 1e-9 and abs(gy) < 1e-9):
        return np.lexsort((lc, lr))
    return np.argsort(lc * gx + lr * gy)


def roi_order_parameter(phase_roi):
    """Kuramoto order parameter R(t) = |mean_e exp(i*phase)| across ROI electrodes."""
    return np.abs(np.exp(1j * phase_roi).mean(axis=0))


def grid_at_time(phase_roi, lr, lc, nr, nc, ci):
    """Return the (nr,nc) phase grid and its gradient (gx,gy) at column index ci."""
    grid = np.full((nr, nc), np.nan)
    grid[lr, lc] = phase_roi[:, ci]
    u = np.exp(1j * grid)
    gy = np.imag(np.conj(u) * np.gradient(u, axis=0))
    gx = np.imag(np.conj(u) * np.gradient(u, axis=1))
    return grid, gx, gy

# ============================================================
# PLOTS
# ============================================================
def save_overview(mask, bg, rois_all, present_scores, skipped, missing, drug, out_base):
    """Whole-slice ROI selection view: mask + all ROI boxes + labels + PGD score."""
    set_nature_style()
    H, W = mask.shape
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    bgm = np.where(mask, bg, np.nan)
    ax.imshow(bgm, cmap=OVERVIEW_BG_CMAP, interpolation="nearest")
    ax.contour(mask.astype(float), levels=[0.5], colors=MASK_OUTLINE_COLOR, linewidths=1.0)
    for name, box in rois_all:
        r0, r1, c0, c1 = box
        present = name in present_scores
        col = ROI_BOX_COLOR if present else "red"
        ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False,
                                   edgecolor=col, linewidth=2.0, zorder=5,
                                   linestyle="-" if present else ":"))
        if present:
            lab = f"{name}\nPGD {present_scores[name]:.2f}"
        elif name in skipped:
            lab = f"{name}\n(skip:{skipped[name]}e)"
        else:
            lab = f"{name}\n(off-mask)"
        ax.text(c0 - 0.3, r0 - 1.0, lab, color=col, fontsize=6.5, weight="bold",
                ha="left", va="bottom", zorder=6)
    ax.set_title(f"{drug} | ROI selection on slice", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    note = []
    if missing:
        note.append("missing (not in csv): " + ",".join(missing))
    if skipped:
        note.append("skipped (<%de): " % MIN_ROI_ELECTRODES + ",".join(skipped))
    if note:
        ax.text(0.0, -0.04, " | ".join(note), transform=ax.transAxes, fontsize=6,
                color="red", va="top")
    save_fig(fig, out_base)


def save_roi_phase_kymo(kymo, pgd, t_ms, drug, roi_name, dom_angle, out_base):
    set_nature_style()
    fig, (axk, axp) = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.15})
    im = axk.imshow(kymo, aspect="auto", origin="lower", cmap=PHASE_CMAP,
                    vmin=-np.pi, vmax=np.pi, interpolation="nearest",
                    extent=[t_ms[0], t_ms[-1], 0, kymo.shape[0]])
    axk.axvline(0, color="k", lw=0.7, ls="--")
    axk.set_ylabel("electrode\n(propagation order)")
    axk.set_title(f"{drug} | {roi_name} | phase kymograph  (dominant {dom_angle:.0f}°)",
                  pad=4, weight="bold")
    cb = fig.colorbar(im, ax=axk, fraction=0.046, pad=0.02)
    cb.set_label("Phase (rad)"); cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(["-pi", "0", "pi"])
    axp.plot(t_ms, pgd, color=PGD_COLOR, lw=1.4)
    axp.axhline(PGD_THRESHOLD, ls="--", color="0.5", lw=0.8)
    axp.axvline(0, color="k", lw=0.7, ls="--")
    axp.set_ylim(0, 1); axp.set_ylabel("PGD"); axp.set_xlabel("Time relative to t0 (ms)")
    pos_k, pos_p = axk.get_position(), axp.get_position()
    axp.set_position([pos_k.x0, pos_p.y0, pos_k.width, pos_p.height])
    save_fig(fig, out_base)


def electrode_vectors_at(phase_roi, lr, lc, nr, nc, ci, kind=VECTOR_KIND):
    """Per-electrode vector at time index ci. Returns ux, uy arrays (per electrode)."""
    if kind == "gradient":
        _, gx, gy = grid_at_time(phase_roi, lr, lc, nr, nc, ci)
        return gx[lr, lc], gy[lr, lc]
    ph = phase_roi[:, ci]                    # unit phase vectors (cos, sin)
    return np.cos(ph), np.sin(ph)


def _draw_vec_panel(ax, lr, lc, nr, nc, ux, uy, mux, muy, mean_scale, title=None):
    # light 4x4 lattice
    for gx_ in range(nc + 1):
        ax.axvline(gx_ - 0.5, color="0.85", lw=0.6, zorder=0)
    for gy_ in range(nr + 1):
        ax.axhline(gy_ - 0.5, color="0.85", lw=0.6, zorder=0)
    # individual vectors (fixed display length, direction only)
    L = 0.42
    for x, y, u, v in zip(lc, lr, ux, uy):
        n = np.hypot(u, v) + 1e-12
        ax.arrow(x, y, u / n * L, v / n * L, head_width=0.12, head_length=0.12,
                 fc=VEC_THIN_COLOR, ec=VEC_THIN_COLOR, lw=0.9, length_includes_head=True, zorder=3)
    # mean / resultant vector (length ∝ magnitude), from grid centre
    cx, cy = (nc - 1) / 2.0, (nr - 1) / 2.0
    ax.arrow(cx, cy, mux * mean_scale, muy * mean_scale, head_width=0.28, head_length=0.28,
             fc=VEC_MEAN_COLOR, ec=VEC_MEAN_COLOR, lw=2.4, length_includes_head=True, zorder=5)
    ax.set_xlim(-1.0, nc + 0.5); ax.set_ylim(nr + 0.5, -1.0)  # origin top-left, square + headroom
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)


def _add_vector_scale_bar(fig, ref_ax, unit_cells=MEAN_UNIT_CELLS, ticks=SCALE_TICKS,
                          x0=0.14, y0=0.045):
    """Ruler-style scale bar drawn in FIGURE coords so its length per magnitude unit
    exactly equals the panels' red mean/Total arrows. Ticks at 0..max(ticks)."""
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    pos = ref_ax.get_position()
    span = abs(ref_ax.get_xlim()[1] - ref_ax.get_xlim()[0])
    unit_frac = unit_cells * (pos.width / span)   # figure-fraction per unit magnitude
    tmax = max(ticks)
    fig.add_artist(Line2D([x0, x0 + tmax * unit_frac], [y0, y0],
                          color=VEC_MEAN_COLOR, lw=3.0, transform=fig.transFigure))
    for m in [0] + list(ticks):
        x = x0 + m * unit_frac
        fig.add_artist(Line2D([x, x], [y0 - 0.008, y0 + 0.008],
                              color="k", lw=0.9, transform=fig.transFigure))
        fig.text(x, y0 - 0.016, str(m), ha="center", va="top", fontsize=8)
    fig.text(x0, y0 + 0.014, "mean / Total vector magnitude (a.u.)",
             ha="left", va="bottom", fontsize=7)


def save_vector_grid_single(name, d, drug, out_base):
    """Per-ROI vector grid: one row = [VEC_SNAP_MS..., Total], with scale bar."""
    set_nature_style()
    cols = VEC_SNAP_MS + ["Total"]
    fig, axes = plt.subplots(1, len(cols), figsize=(1.7 * len(cols), 2.0), squeeze=False)
    lr, lc, nr, nc = d["lr"], d["lc"], d["nr"], d["nc"]
    for j, ms in enumerate(VEC_SNAP_MS):
        ux, uy = d["vecs"][ms]; mux, muy = d["mean"][ms]
        _draw_vec_panel(axes[0][j], lr, lc, nr, nc, ux, uy, mux, muy, MEAN_UNIT_CELLS,
                        title=f"{ms} ms")
    mux, muy = d["total"]
    _draw_vec_panel(axes[0][-1], lr, lc, nr, nc, np.array([]), np.array([]),
                    mux, muy, MEAN_UNIT_CELLS, title="Total")
    fig.suptitle(f"{drug} | {name} | phase vectors + mean (red) + Total  [{VECTOR_KIND}]",
                 fontsize=9, y=1.06)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _add_vector_scale_bar(fig, axes[0][0])
    save_fig(fig, out_base)


def save_vector_grid(vec_store, drug, out_base):
    """Rows = ROI; cols = VEC_SNAP_MS then Total. Thin = individual vectors,
    thick red = mean vector; Total = summed mean vector across the times."""
    set_nature_style()
    names = list(vec_store)
    cols = VEC_SNAP_MS + ["Total"]
    ncol, nrow = len(cols), len(names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.7 * nrow), squeeze=False)
    for i, n in enumerate(names):
        d = vec_store[n]; lr, lc, nr, nc = d["lr"], d["lc"], d["nr"], d["nc"]
        for j, ms in enumerate(VEC_SNAP_MS):
            ux, uy = d["vecs"][ms]; mux, muy = d["mean"][ms]
            _draw_vec_panel(axes[i][j], lr, lc, nr, nc, ux, uy, mux, muy, MEAN_UNIT_CELLS,
                            title=(f"{ms} ms" if i == 0 else None))
            if j == 0:
                axes[i][j].set_ylabel(n, fontsize=9, rotation=0, ha="right", va="center")
        mux, muy = d["total"]
        _draw_vec_panel(axes[i][-1], lr, lc, nr, nc, np.array([]), np.array([]),
                        mux, muy, MEAN_UNIT_CELLS, title=("Total" if i == 0 else None))
    fig.suptitle(f"{drug} | per-electrode phase vectors + mean (red) + Total  [{VECTOR_KIND}]",
                 fontsize=10, y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _add_vector_scale_bar(fig, axes[0][0])
    save_fig(fig, out_base)


def write_vectors_excel(vec_store, drug, out):
    """Per-electrode + mean + total vectors -> Excel (components saved for circular stats)."""
    erows, mrows = [], []
    for n, d in vec_store.items():
        lr, lc = d["lr"], d["lc"]
        for ms in VEC_SNAP_MS:
            ux, uy = d["vecs"][ms]
            for e in range(len(ux)):
                erows.append(dict(roi=n, time_ms=ms, electrode=e,
                                  local_r=int(lr[e]), local_c=int(lc[e]),
                                  ux=float(ux[e]), uy=float(uy[e]),
                                  mag=float(np.hypot(ux[e], uy[e])),
                                  angle_deg=float(np.degrees(np.arctan2(uy[e], ux[e])))))
            mux, muy = d["mean"][ms]
            mrows.append(dict(roi=n, time_ms=ms, mean_ux=mux, mean_uy=muy,
                              magnitude=float(np.hypot(mux, muy)),
                              angle_deg=float(np.degrees(np.arctan2(muy, mux)))))
        tux, tuy = d["total"]
        mrows.append(dict(roi=n, time_ms="Total", mean_ux=tux, mean_uy=tuy,
                          magnitude=float(np.hypot(tux, tuy)),
                          angle_deg=float(np.degrees(np.arctan2(tuy, tux)))))
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame(erows).to_excel(xw, sheet_name="electrode_vectors", index=False)
        pd.DataFrame(mrows).to_excel(xw, sheet_name="mean_and_total_vectors", index=False)
    print(f"  [xlsx] {os.path.basename(out)}")


def save_grid_snapshots(phase_roi, lr, lc, nr, nc, t_ms, drug, roi_name, out_base):
    """4x4 grid phase snapshots at SNAP_MS_GRID, with phase-gradient vectors."""
    set_nature_style()
    snaps = SNAP_MS_GRID
    K = len(snaps)
    cmap = plt.get_cmap(PHASE_CMAP)
    fig, axes = plt.subplots(1, K, figsize=(1.6 * K, 2.0))
    if K == 1:
        axes = [axes]
    Y, X = np.mgrid[0:nr, 0:nc]
    im = None
    for k, ms in enumerate(snaps):
        ci = int(np.argmin(np.abs(t_ms - ms)))
        grid, gx, gy = grid_at_time(phase_roi, lr, lc, nr, nc, ci)
        ax = axes[k]
        im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=-np.pi, vmax=np.pi,
                       interpolation="nearest")
        if SHOW_GRID_QUIVER:
            ax.quiver(X, Y, gx, gy, color="k", scale=None, width=0.012,
                      pivot="mid", headwidth=3.5)
        ax.set_title(f"{ms} ms", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Phase (rad)"); cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(["-pi", "0", "pi"])
    fig.suptitle(f"{drug} | {roi_name} | 4x4 phase snapshots", fontsize=9, y=1.05)
    save_fig(fig, out_base)


def save_combined_snapshots(snap_store, drug, out_base):
    """Combined ROIs (rows) x times (cols) grid of 4x4 phase snapshots."""
    set_nature_style()
    names = list(snap_store)
    snaps = SNAP_MS_GRID
    nrow, ncol = len(names), len(snaps)
    cmap = plt.get_cmap(PHASE_CMAP)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.4 * ncol, 1.4 * nrow), squeeze=False)
    im = None
    for i, name in enumerate(names):
        for j, ms in enumerate(snaps):
            grid, gx, gy = snap_store[name][ms]
            ax = axes[i][j]
            im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=-np.pi, vmax=np.pi,
                           interpolation="nearest")
            if SHOW_GRID_QUIVER:
                Y, X = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
                ax.quiver(X, Y, gx, gy, color="k", width=0.02, pivot="mid", headwidth=3)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"{ms} ms", fontsize=8)
            if j == 0:
                ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Phase (rad)"); cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(["-pi", "0", "pi"])
    fig.suptitle(f"{drug} | 4x4 phase snapshots (rows=ROI, cols=time)", fontsize=10, y=1.01)
    save_fig(fig, out_base)


def write_excel(drug_dir, drug, t_ms, R_by_roi, psi_by_roi, pgd_by_roi, snap_store, summary_by_roi):
    """R, psi (mean phase angle) + PGD time series, phase snapshots, summary -> .xlsx."""
    out = os.path.join(drug_dir, f"{drug}_phase_R_psi_PGD.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        # R + psi (mean phase angle) + PGD time series (wide)
        df = pd.DataFrame({"time_ms": t_ms})
        for n in R_by_roi:
            df[f"R_{n}"] = R_by_roi[n]
            df[f"psi_deg_{n}"] = np.degrees(psi_by_roi[n])
            df[f"PGD_{n}"] = pgd_by_roi[n]
        df.to_excel(xw, sheet_name="R_psi_PGD_timeseries", index=False)
        # summary
        pd.DataFrame(summary_by_roi).to_excel(xw, sheet_name="summary", index=False)
        # phase snapshots per ROI (rows=time, cols=grid cell r{c})
        for n, store in snap_store.items():
            rows = []
            for ms in SNAP_MS_GRID:
                grid = store[ms][0]
                rec = {"time_ms": ms}
                for r in range(grid.shape[0]):
                    for c in range(grid.shape[1]):
                        rec[f"r{r}c{c}"] = grid[r, c]
                rows.append(rec)
            pd.DataFrame(rows).to_excel(xw, sheet_name=f"phase_{n}"[:31], index=False)
    print(f"  [xlsx] {os.path.basename(out)}")


def save_combined_grid(kymos, t_ms, drug, out_base):
    set_nature_style()
    names = list(kymos); ncol = 3; nrow = int(np.ceil(len(names) / ncol))
    fig = plt.figure(figsize=(3.4 * ncol, 2.4 * nrow))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [0.05], wspace=0.25, hspace=0.4)
    im = None
    for i, n in enumerate(names):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        im = ax.imshow(kymos[n], aspect="auto", origin="lower", cmap=PHASE_CMAP,
                       vmin=-np.pi, vmax=np.pi, interpolation="nearest",
                       extent=[t_ms[0], t_ms[-1], 0, kymos[n].shape[0]])
        ax.axvline(0, color="k", lw=0.6, ls="--"); ax.set_title(n, fontsize=9)
        if i % ncol == 0: ax.set_ylabel("electrode")
        if i // ncol == nrow - 1: ax.set_xlabel("time (ms)")
    cax = fig.add_subplot(gs[:, -1]); cb = fig.colorbar(im, cax=cax)
    cb.set_label("Phase (rad)"); cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(["-pi", "0", "pi"])
    fig.suptitle(f"{drug} | per-ROI phase kymographs", y=1.0, fontsize=10)
    save_fig(fig, out_base)

# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Per-ROI phase-gradient-directionality (PGD) and Kuramoto R/psi analysis")
    p.add_argument("--data_dir", default=DATA_DIR, help="Folder of <slice_id>_<condition>_2ms.h5 files (default: %(default)s)")
    p.add_argument("--file_pattern", default=FILE_PATTERN, help="Glob pattern for H5 files within data_dir (default: %(default)s)")
    p.add_argument("--out_dir", default=OUT_DIR, help="Output folder (default: %(default)s)")
    p.add_argument("--mask", default=MASK_PATH, help="64x64 contact mask .npy (default: %(default)s)")
    p.add_argument("--roi_csv", default=ROI_CSV, help="ROI definitions CSV with roi_name,r0,r1,c0,c1 (default: %(default)s)")
    return p


# ============================================================
# MAIN
# ============================================================
def main():
    global DATA_DIR, FILE_PATTERN, OUT_DIR, MASK_PATH, ROI_CSV
    args = build_argparser().parse_args()
    DATA_DIR, FILE_PATTERN, OUT_DIR = args.data_dir, args.file_pattern, args.out_dir
    MASK_PATH, ROI_CSV = args.mask, args.roi_csv

    os.makedirs(OUT_DIR, exist_ok=True)
    files = get_h5_files()
    rois_all, missing = load_roi_definitions(ROI_CSV)
    print(f"[INFO] ROIs in csv: {[n for n, _ in rois_all]}")

    summary_rows = []
    for p in files:
        drug = get_drug_name(p)
        print("\n============================================================\nProcessing:", drug)
        drug_dir = os.path.join(OUT_DIR, drug); os.makedirs(drug_dir, exist_ok=True)

        with h5py.File(p, "r") as f:
            H, W, fs = load_meta(f)
            mask2d = load_mask_from_h5_or_npy(f, H, W)
            mask_idx = np.flatnonzero(mask2d.reshape(-1))
            d_lfp = f["/lfp/ts_2ms_full"]
            dt = 1000.0 / fs
            T = d_lfp.shape[1]
            pre_s, post_s, pad = int(round(PRE_MS/dt)), int(round(POST_MS/dt)), int(round(PAD_MS/dt))
            t0 = find_t0_chunked(d_lfp, mask_idx, pre_s + pad, post_s + pad)
            w0 = max(0, t0 - pre_s - pad); w1 = min(T, t0 + post_s + pad + 1)
            win = read_window_chunked(d_lfp, w0, w1, CHUNK_SIZE)
            print(f"[INFO] fs={fs:.0f}Hz t0={t0} window[{w0}:{w1}] ({w1-w0} samp) of T={T}")

        t0_loc = t0 - w0
        a = max(0, t0_loc - pre_s); b = min(win.shape[1], t0_loc + post_s + 1)
        t_ms = (np.arange(a, b) - t0_loc) * dt
        if (b - a) < post_s:        # warn if still short (recording genuinely too short)
            print(f"[WARN] POST window only {(b-a)*dt:.0f} ms (recording ends early)")

        # whole-slice activity background for the overview (mean |LFP| over window)
        bg = np.abs(win).mean(1).reshape(H, W)

        # ----- collect ROI electrodes; compute generalized phase ONLY on them -----
        roi_info = []; all_ci = []
        skipped = {}
        for name, box in rois_all:
            ci, lr, lc, nr, nc = roi_grid_channels(box, W, mask2d)
            if ci.size < MIN_ROI_ELECTRODES:
                skipped[name] = int(ci.size)
                print(f"  [SKIP] {name}: only {ci.size} contact electrodes (<{MIN_ROI_ELECTRODES})")
                continue
            roi_info.append((name, ci, lr, lc, nr, nc)); all_ci.append(ci)
        if not roi_info:
            print("  [WARN] no usable ROIs for this file"); continue
        uniq = np.unique(np.concatenate(all_ci))
        pos = {c: i for i, c in enumerate(uniq)}
        phase_sub = generalized_phase(win[uniq], fs, GP_BAND)   # (n_uniq, Twin)
        del win

        kymos = {}; scores = {}
        R_by_roi = {}; pgd_by_roi = {}; psi_by_roi = {}; snap_store = {}; excel_summary = []
        vec_store = {}; theta_by_roi = {}
        for name, ci, lr, lc, nr, nc in roi_info:
            rows = np.array([pos[c] for c in ci])
            phase_roi = phase_sub[rows, a:b]
            pgd, gx, gy, mgx_t, mgy_t = roi_pgd_and_dir(phase_roi, lr, lc, nr, nc)
            theta_t = propagation_direction(mgx_t, mgy_t)       # instantaneous direction (rad)
            dir_stats = direction_window_stats(theta_t, t_ms)
            theta_by_roi[name] = theta_t
            Zc = np.exp(1j * phase_roi).mean(axis=0)               # complex order parameter
            Rt = np.abs(Zc)                                        # R(t)  = coherence
            psi = np.angle(Zc)                                     # psi(t) = mean phase ANGLE
            dom = np.degrees(np.arctan2(gy, gx)) % 360
            order = order_electrodes(lr, lc, gx, gy)
            kymo = phase_roi[order]; kymos[name] = kymo
            scores[name] = float(np.nanmean(pgd))
            R_by_roi[name] = Rt; pgd_by_roi[name] = pgd; psi_by_roi[name] = psi

            rd = os.path.join(drug_dir, sanitize_name(name))
            save_roi_phase_kymo(kymo, pgd, t_ms, drug, name, dom,
                                os.path.join(rd, f"{drug}_{name}_phase_kymo_PGD"))
            # 4x4 phase grid snapshots (+ vectors)
            save_grid_snapshots(phase_roi, lr, lc, nr, nc, t_ms, drug, name,
                                os.path.join(rd, f"{drug}_{name}_grid_snapshots"))
            snap_store[name] = {ms: grid_at_time(phase_roi, lr, lc, nr, nc,
                                                 int(np.argmin(np.abs(t_ms - ms))))
                                for ms in SNAP_MS_GRID}
            # v4: per-electrode vectors + mean (resultant) at VEC_SNAP_MS + Total
            vecs, means = {}, {}
            for ms in VEC_SNAP_MS:
                cti = int(np.argmin(np.abs(t_ms - ms)))
                ux, uy = electrode_vectors_at(phase_roi, lr, lc, nr, nc, cti, VECTOR_KIND)
                vecs[ms] = (ux, uy)
                means[ms] = (float(np.nanmean(ux)), float(np.nanmean(uy)))
            tot = (float(np.sum([means[m][0] for m in VEC_SNAP_MS])),
                   float(np.sum([means[m][1] for m in VEC_SNAP_MS])))
            vec_store[name] = dict(lr=lr, lc=lc, nr=nr, nc=nc, vecs=vecs, mean=means, total=tot)
            if SAVE_VECTOR_GRID:      # per-ROI vector grid in the ROI subfolder
                save_vector_grid_single(name, vec_store[name], drug,
                                        os.path.join(rd, f"{drug}_{name}_vector_grid"))
            np.savez(os.path.join(rd, "phase_kymo_data.npz"), kymo=kymo, pgd=pgd, R=Rt, psi=psi,
                     t_ms=t_ms, electrode_order=order, chan_idx=ci, dominant_angle_deg=dom,
                     mean_grad_x=mgx_t, mean_grad_y=mgy_t, propagation_angle_rad=theta_t)
            # order-parameter time series: R, psi (mean phase angle), PGD
            with open(os.path.join(rd, f"{drug}_{name}_R_psi_PGD.csv"), "w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["drug", "roi_name", "time_ms", "R", "psi_rad", "psi_deg", "PGD"])
                for k in range(len(t_ms)):
                    w.writerow([drug, name, float(t_ms[k]), float(Rt[k]), float(psi[k]),
                                float(np.degrees(psi[k])), float(pgd[k])])
            # full per-electrode phase values (propagation order) over time
            with open(os.path.join(rd, f"{drug}_{name}_phase_full.csv"), "w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["time_ms"] + [f"e{e:02d}" for e in range(kymo.shape[0])])
                for k in range(len(t_ms)):
                    w.writerow([float(t_ms[k])] + [float(v) for v in kymo[:, k]])
            frac = float(np.nanmean(pgd >= PGD_THRESHOLD))
            mean_R = float(np.nanmean(Rt))
            summary_rows.append([drug, os.path.basename(p), name, int(ci.size),
                                 float(dom), scores[name], mean_R, frac]
                                + [dir_stats[k] for k in DIR_COLS])
            excel_summary.append(dict(roi_name=name, n_electrodes=int(ci.size),
                                      dominant_angle_deg=float(dom), mean_PGD=scores[name],
                                      mean_R=mean_R, wave_fraction=frac, **dir_stats))
            print(f"  [ROI] {name:4s} dom {dom:5.0f}deg meanPGD {scores[name]:.3f} "
                  f"meanR {mean_R:.3f} wavefrac {frac:.2f}")

        save_overview(mask2d, bg, rois_all, scores, skipped, missing, drug,
                      os.path.join(drug_dir, f"{drug}_ROI_overview"))
        if theta_by_roi:
            save_direction_polar(theta_by_roi, t_ms, drug,
                                 os.path.join(drug_dir, f"{drug}_propagation_direction_polar"))
            pooled = np.concatenate([th[(t_ms >= 0) & (t_ms <= 500)] for th in theta_by_roi.values()])
            pooled_stats = direction_window_stats(
                np.concatenate(list(theta_by_roi.values())), np.tile(t_ms, len(theta_by_roi)))
            summary_rows.append([drug, os.path.basename(p), "ALL_ROIS", int(len(theta_by_roi)),
                                 np.nan, np.nan, np.nan, np.nan] + [pooled_stats[k] for k in DIR_COLS])
        if kymos:
            save_combined_grid(kymos, t_ms, drug,
                               os.path.join(drug_dir, f"{drug}_ALL_ROI_phase_kymographs"))
            save_combined_snapshots(snap_store, drug,
                                    os.path.join(drug_dir, f"{drug}_ALL_ROI_grid_snapshots"))
        if SAVE_EXCEL and R_by_roi:
            write_excel(drug_dir, drug, t_ms, R_by_roi, psi_by_roi, pgd_by_roi, snap_store, excel_summary)
        if SAVE_VECTOR_GRID and vec_store:
            save_vector_grid(vec_store, drug,
                             os.path.join(drug_dir, f"{drug}_ALL_ROI_vector_grid"))
            write_vectors_excel(vec_store, drug,
                                os.path.join(drug_dir, f"{drug}_vectors.xlsx"))

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug", "file", "roi_name", "n_electrodes",
                    "dominant_angle_deg", "mean_PGD", "mean_R", "wave_fraction_PGDge0.7"] + DIR_COLS)
        w.writerows(summary_rows)
    print(f"\nDONE. Outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()