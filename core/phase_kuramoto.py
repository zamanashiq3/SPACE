"""
phase_kuramoto.py -- SPACE framework: Hilbert phase, Kuramoto synchrony, traveling waves.

Manuscript figure(s): Figure 1A (SPACE box: Hilbert transform -> Kuramoto order parameter),
Figure 5 (Kuramoto R, phase velocity), Figure 7F-I (pharmacology Kuramoto R/phase velocity).

What it computes:
    The core synchrony engine of the SPACE framework. From a 2 ms-binned LFP time series
    it derives a wideband (or multi-band-fusion) instantaneous phase per electrode via the
    Hilbert transform, then computes the Kuramoto order parameter R(t)/mean phase Phi(t),
    local coherence maps, correlation/PLV/circular-phase-correlation matrices, phase-
    gradient traveling-wave velocity fields, phase kymographs, sliding-window coherograms,
    and synchronization-episode statistics. Robust to NaN/masked-out electrodes throughout
    (memory-safe channel-blocked filtering + time decimation for large recordings).

Inputs / outputs (see data/README.md for the full schema):
    Input:  one or more ``<slice_id>_<condition>_2ms.h5`` files under
            ``data/processed_2ms/<group>/`` (as written by ``preprocessing/brw_to_2ms_h5.py``),
            and optionally the matching 64x64 contact mask from ``data/masks/``.
    Output: per-file tables (Kuramoto R/Phi/dPhi time series, summary JSON, sync-episode
            stats) and figures (coherence maps, wave-velocity fields, kymographs,
            coherograms) written under a results directory, e.g. ``data/results/kuramoto/``.

# Robust spatiotemporal Kuramoto analysis (ALL-phase or multi-band fusion)
# - memory-safe (channel-blocked filtering + time decimation)
# - tqdm progress bars
# - NaN/empty-slice safe
# - mask-aware spatial ops (no color outside contacts)
# - presentation polish: perceptual cmaps, consistent styling
# - extra visuals: distributions, episode stats, rose plot, distance-decay,
#   kymograph, coherogram summary, correlation/PLV/phi-corr matrices
"""

import os, json, argparse, h5py, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks

# ================== CONFIG ==================
# --- EDIT THIS --- (or override via CLI flags below)
IN_DIR          = "data/processed_2ms"
OUT_ROOT        = "data/results/kuramoto"
MASK_NPY        = None   # optional path to a data/masks/<slice_id>__contact_mask_64x64.npy; set None if not using

MODE            = "wideband"                          # "wideband" or "fusion"
WIDEBAND_HZ     = (5.0, 1500.0)                       # Methods: zero-phase 3rd-order Butterworth 5-1500 Hz
                                                      # (acts as a 5 Hz high-pass; upper edge is clipped to Nyquist for 500 Hz files)
BANDS           = ("theta","alpha","beta","gamma")    # used if MODE="fusion"
POWER_EXP       = 0.5                                  # fusion weight exponent on band power
EXPORT_SECONDS  = True

# Memory / progress
FILTER_BLOCK_CH         = 8        # reduce if RAM is tight (e.g., 4)
FILTFILT_METHOD         = "pad"    # safe fallback if SciPy lacks 'method'
TIME_DECIMATE_WIDEBAND  = 4        # Methods: down-sample by 4 (10 kHz -> 2.5 kHz) before Hilbert
PBAR_CHANNEL_WIDEBAND   = False

# Corr/PLV (subsample to keep matrices readable)
CORR_STEP_XY        = 8
CORR_TIME_STEP      = 10
PLV_STEP_XY         = 8
PLV_CHUNK_T         = 4000

# ϕ(t) circular-correlation matrix
PHI_CORR_STEP_XY    = 8
PHI_CORR_CHUNK_T    = 4000

# Pretty traces & snapshots
MAX_PTS_PLOT        = 6000
SMOOTH_SEC          = 0.5
SNAP_PKS            = 6
SNAP_PAD_SEC        = 0.5

# Wave/gradient overlays
QUIVER_STEP         = 4
VEL_REF_PCTL        = 95

# Coherogram window
COH_WIN_SEC         = 1.0

# ================== THEME / COLOR HELPERS ==================
def set_plot_theme():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

def masked_cmap(base="viridis"):
    cm = plt.get_cmap(base).copy()
    cm.set_bad((0,0,0,0))  # transparent for NaNs
    return cm

def ensure(p): os.makedirs(p, exist_ok=True); return p

def save_fig(fig, path_no_ext):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)

# ================== H5 / META / MASK ==================
def get_meta(f):
    H = int(f["/meta"].attrs["H"]); W = int(f["/meta"].attrs["W"])
    fs = float(f["/meta"].attrs.get("fs", 10000.0))
    # Schema (data/README.md / _spec/common.md): /meta.fs is the sampling rate of the STORED
    # series. Only legacy files carry an explicit decimate_Q that still has to be applied.
    q  = int(f["/meta"].attrs.get("decimate_Q", 1))
    fs_dec = fs / max(1, q)  # sampling rate of the stored 2 ms stream
    return H, W, fs_dec

def load_contact_mask_npy(H, W):
    if MASK_NPY and os.path.exists(MASK_NPY):
        m = np.load(MASK_NPY)
        if m.shape == (H, W): return m.astype(bool)
        if m.size == H*W:     return m.reshape(H, W).astype(bool)
    return None

def get_mask2d(f, H, W):
    m = load_contact_mask_npy(H, W)
    if m is not None: return m
    if "/mask" in f:
        mm = f["/mask"][:].astype(bool)
        if mm.shape == (H, W): return mm
    if "/lfp/metrics/rms_map" in f:
        ok = np.isfinite(f["/lfp/metrics/rms_map"][:])
        if ok.shape == (H, W): return ok
    return np.ones((H, W), bool)

def masked_nan(P, mask1d):  # P: (N,T)
    P = P.astype(float, copy=True)
    if mask1d is not None and mask1d.size == P.shape[0]:
        P[~mask1d, :] = np.nan
    return P

# SciPy compat for sosfiltfilt(method=)
def _sosfiltfilt_compat(sos, x, axis=-1, method=None):
    try:
        if method is not None:
            return sosfiltfilt(sos, x, axis=axis, method=method)
        else:
            return sosfiltfilt(sos, x, axis=axis)
    except TypeError:
        return sosfiltfilt(sos, x, axis=axis)

# ================== KURAMOTO CORE ==================
def kuramoto_R_Phi(P):  # P: (N,T) radians, NaN allowed
    valid = np.isfinite(P)
    E = np.exp(1j * np.where(valid, P, 0.0))
    cnt = valid.sum(axis=0).astype(float); cnt[cnt < 1] = np.nan
    mean_vec = E.sum(axis=0) / cnt
    return np.abs(mean_vec), np.angle(mean_vec)

def phi_velocity(Phi, fs_eff):
    Phi_u = np.unwrap(Phi)
    return np.gradient(Phi_u) * fs_eff  # rad/s

def local_coherence_map(P, Phi):
    """
    R_i = |(1/K_i) * sum_t exp(i(θ_i - Φ))|; K_i = #valid times for channel i.
    NaN for channels with K_i == 0.
    """
    Phi = np.asarray(Phi)
    vt = np.isfinite(Phi)
    if vt.ndim != 1: vt = vt.ravel()
    if vt.sum() == 0:
        return np.full(P.shape[0], np.nan, dtype=np.float32)

    P_sub = P[:, vt]                        # (N, T_valid)
    Phi_sub = Phi[vt][None, :]              # (1, T_valid)
    V = np.isfinite(P_sub)                  # (N, T_valid)
    Zsum = np.zeros(P_sub.shape, dtype=np.complex64)
    Zsum[V] = np.exp(1j * (P_sub[V] - Phi_sub.repeat(P_sub.shape[0], 0)[V]))
    K = V.sum(axis=1).astype(np.float32)    # per-channel counts

    out = np.full(P.shape[0], np.nan, dtype=np.float32)
    ok = K > 0
    if np.any(ok):
        out[ok] = np.abs(Zsum.sum(axis=1)[ok] / K[ok]).astype(np.float32)
    return out

# ================== ALL-PHASE (WIDEBAND) ==================
def wideband_phase_from_ts(
        TS, fs_dec, f_lo=2.0, f_hi=120.0,
        show_channel_pbar=False,
        block_ch=FILTER_BLOCK_CH,
        filtfilt_method=FILTFILT_METHOD,
        time_decimate=TIME_DECIMATE_WIDEBAND
    ):
    TS = np.asarray(TS, dtype=np.float32, order="C")
    nyq = fs_dec * 0.5
    lo = max(0.1, f_lo) / nyq
    hi = min(f_hi, nyq - 1e-6) / nyq
    sos = butter(3, [lo, hi], btype="band", output="sos")

    N, T = TS.shape
    dec = max(1, int(time_decimate))
    out_phase = None

    if show_channel_pbar:
        ch_iter = tqdm(range(N), desc="Filtering+Hilbert (wideband)", leave=False)
        for i in ch_iter:
            x = TS[i, :]
            x_f = _sosfiltfilt_compat(sos, x, axis=0, method=filtfilt_method).astype(np.float32, copy=False)
            if dec > 1: x_f = x_f[::dec]
            z = hilbert(x_f)
            ph = np.angle(z).astype(np.float32, copy=False)
            if out_phase is None:
                T2 = ph.shape[0]
                out_phase = np.empty((N, T2), dtype=np.float32)
            out_phase[i, :] = ph
    else:
        ch_iter = tqdm(range(0, N, block_ch), desc="Filtering+Hilbert (wideband)", leave=False)
        for start in ch_iter:
            end = min(start + block_ch, N)
            X = TS[start:end, :]
            X_f = _sosfiltfilt_compat(sos, X, axis=1, method=filtfilt_method).astype(np.float32, copy=False)
            if dec > 1: X_f = X_f[:, ::dec]
            Z  = hilbert(X_f, axis=1)
            PH = np.angle(Z).astype(np.float32, copy=False)
            if out_phase is None:
                T2 = PH.shape[1]
                out_phase = np.empty((N, T2), dtype=np.float32)
            out_phase[start:end, :] = PH

    eff_fs = fs_dec / dec
    return out_phase, eff_fs

# ================== MULTI-BAND FUSION (optional) ==================
def composite_phase_from_bands(band_phases, band_powers=None, power_exponent=0.5):
    bands = list(band_phases.keys())
    N, T = next(iter(band_phases.values())).shape
    Zsum = np.zeros((N, T), dtype=np.complex128)
    wsum = np.zeros((N, T), dtype=float)
    for b in tqdm(bands, desc="Fusing bands", leave=False):
        Th = band_phases[b]
        if band_powers and b in band_powers:
            W = np.maximum(band_powers[b], 0.0) ** power_exponent
        else:
            W = np.ones_like(Th, dtype=float)
        valid = np.isfinite(Th) & np.isfinite(W)
        Zsum += np.exp(1j * np.where(valid, Th, 0.0)) * np.where(valid, W, 0.0)
        wsum += np.where(valid, W, 0.0)
    wsum[wsum <= 0] = np.nan
    return np.angle(Zsum / wsum).astype(np.float32, copy=False)

# ================== PLOTTING HELPERS ==================
def _ds_plot(x, y, max_pts=6000, smooth_win=0):
    n = y.size
    if smooth_win > 1:
        k = int(smooth_win)
        y = np.convolve(y, np.ones(k)/k, mode="same")
    if n <= max_pts: return x, y
    idx = np.linspace(0, n-1, max_pts, dtype=int)
    return x[idx], y[idx]

def plot_R_Phi_pretty(t, R, Phi, out_png, fs_eff, max_pts=MAX_PTS_PLOT):
    x = t / fs_eff
    sw = int(max(1, fs_eff * SMOOTH_SEC))
    R_safe   = np.where(np.isfinite(R), R, np.nan if np.any(np.isfinite(R)) else 0.0)
    R_fill   = np.nan_to_num(R_safe, nan=np.nanmedian(R_safe) if np.any(np.isfinite(R_safe)) else 0.0)
    Phi_fill = np.nan_to_num(Phi, nan=0.0)
    x1, R1   = _ds_plot(x, R_fill,   max_pts=max_pts, smooth_win=sw)
    x2, Phi1 = _ds_plot(x, Phi_fill, max_pts=max_pts, smooth_win=sw)
    fig, ax = plt.subplots(2,1, figsize=(9,4.6), sharex=True)
    ax[0].plot(x1, R1, lw=0.8);  ax[0].set_ylim(0,1); ax[0].set_ylabel("R(t)")
    ax[0].set_title("Kuramoto R(t) and Φ(t) — ALL (smoothed/decimated)")
    ax[1].plot(x2, Phi1, lw=0.6); ax[1].set_ylim(-1,1); ax[1].set_ylabel("Φ (rad)")
    ax[1].set_xlabel("time (s)")
    save_fig(fig, out_png)

def plot_phi_velocity(t, dphi, out_png, fs_eff):
    x = t / fs_eff
    x2, y2 = _ds_plot(x, np.nan_to_num(dphi, nan=0.0), max_pts=MAX_PTS_PLOT,
                      smooth_win=int(max(1, fs_eff*SMOOTH_SEC)))
    fig, ax = plt.subplots(figsize=(9,2.4))
    ax.plot(x2, y2, lw=0.8); ax.set_ylabel("dΦ/dt (rad/s)"); ax.set_xlabel("time (s)")
    ax.set_title("Mean-phase angular velocity — ALL (smoothed/decimated)")
    save_fig(fig, out_png)

def plot_scalar_map(M, title, out_png, vmin=None, vmax=None, clabel=None):
    fig, ax = plt.subplots(figsize=(3.9,3.6))
    cm = masked_cmap("YlGn")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor((0.08,0.06,0.2,1.0))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if clabel: cbar.set_label(clabel)
    save_fig(fig, out_png)

def plot_matrix(M, title, out_png, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    im = ax.imshow(M, cmap=masked_cmap(cmap), vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, out_png)

def plot_phase_snapshot_with_grad(img, gx_t, gy_t, title, out_png, step=QUIVER_STEP):
    fig, ax = plt.subplots(figsize=(4.0,3.7))
    cm = masked_cmap("coolwarm")
    im = ax.imshow(np.ma.masked_invalid(img), cmap=cm, vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("phase (rad)")
    H_, W_ = img.shape; Y,X = np.mgrid[0:H_, 0:W_]
    q = ax.quiver(X[::step,::step], Y[::step,::step],
                  gx_t[::step,::step], gy_t[::step,::step],
                  color="k", scale_units="xy", scale=4.0, width=0.0025)
    ax.quiverkey(q, X=0.88, Y=1.05, U=1.0, label="direction only", labelpos='E', coordinates='axes')
    ax.set_xlim(-0.5, W_-0.5); ax.set_ylim(H_-0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    save_fig(fig, out_png)

# ================== SPATIOTEMPORAL HELPERS ==================
def _grid_subsample_indices(H, W, step):
    ys = np.arange(0, H, max(1, step))
    xs = np.arange(0, W, max(1, step))
    return np.array([y*W + x for y in ys for x in xs], dtype=np.int64)

def export_corr_matrix(f, figs_dir, H, W):
    if "/lfp/ts_2ms_full" not in f: return
    TS = f["/lfp/ts_2ms_full"]
    idx = _grid_subsample_indices(H, W, CORR_STEP_XY)
    if idx.size < 2: return
    X = TS[idx, ::max(1, CORR_TIME_STEP)].astype(np.float32)
    X -= X.mean(axis=1, keepdims=True); X /= (X.std(axis=1, keepdims=True)+1e-12)
    Tsub = X.shape[1]
    C = (X @ X.T) / max(1, Tsub)
    plot_matrix(C, f"LFP Correlation (subsample {CORR_STEP_XY}×, t step={CORR_TIME_STEP})",
                os.path.join(figs_dir, "corr_subsample"), vmin=-1, vmax=1, cmap="RdBu")

def export_plv_matrix_from_phase(P_all, figs_dir, H, W):
    if P_all is None or P_all.size == 0: return
    idx = _grid_subsample_indices(H, W, PLV_STEP_XY)
    if idx.size < 2: return
    Nsub = idx.size; T = P_all.shape[1]
    C = np.zeros((Nsub, Nsub), dtype=np.complex64)
    for t0 in tqdm(range(0, T, PLV_CHUNK_T), desc="PLV accumulate", leave=False):
        t1 = min(t0 + PLV_CHUNK_T, T)
        E = np.exp(1j * P_all[idx, t0:t1]).astype(np.complex64, copy=False)
        C += E @ E.conj().T
    PLV = np.abs(C) / max(1, T)
    plot_matrix(PLV, f"PLV (subsample {PLV_STEP_XY}×, chunk={PLV_CHUNK_T})",
                os.path.join(figs_dir, "plv_subsample"), vmin=0, vmax=1, cmap="BuGn")

def export_phi_circ_corr_matrix(P_all, figs_dir, H, W):
    if P_all is None or P_all.size == 0: return
    idx = _grid_subsample_indices(H, W, PHI_CORR_STEP_XY)
    if idx.size < 2: return
    Th = P_all[idx, :]
    Nsub, T = Th.shape
    # circular mean per channel (chunked)
    sum_sin = np.zeros(Nsub, dtype=np.float64)
    sum_cos = np.zeros(Nsub, dtype=np.float64)
    for t0 in range(0, T, PHI_CORR_CHUNK_T):
        t1 = min(t0 + PHI_CORR_CHUNK_T, T)
        block = Th[:, t0:t1]
        valid = np.isfinite(block)
        sum_sin += np.where(valid, np.sin(block), 0.0).sum(axis=1)
        sum_cos += np.where(valid, np.cos(block), 0.0).sum(axis=1)
    mu = np.arctan2(sum_sin, sum_cos).astype(np.float32)

    Sxx = np.zeros(Nsub, dtype=np.float64)
    Sxy = np.zeros((Nsub, Nsub), dtype=np.float64)
    for t0 in tqdm(range(0, T, PHI_CORR_CHUNK_T), desc="ϕ-corr accumulate", leave=False):
        t1 = min(t0 + PHI_CORR_CHUNK_T, T)
        block = Th[:, t0:t1]
        x = np.sin(block - mu[:, None]).astype(np.float32, copy=False)
        x = np.where(np.isfinite(x), x, 0.0)
        Sxx += (x * x).sum(axis=1, dtype=np.float64)
        Sxy += x @ x.T
    denom = np.sqrt(np.maximum(Sxx, 1e-12))
    D = denom[:, None] * denom[None, :]
    R = np.divide(Sxy, D, out=np.zeros_like(Sxy), where=(D > 0))
    R = np.clip(R, -1.0, 1.0).astype(np.float32)
    plot_matrix(R, f"ϕ(t) circular correlation (subsample {PHI_CORR_STEP_XY}×, chunk={PHI_CORR_CHUNK_T})",
                os.path.join(figs_dir, "phi_corr_subsample"), vmin=-1, vmax=1, cmap="RdBu")

# ---- Phase gradient, velocity, coherence (mask-aware) ----
def phase_gradient_maps(P_all, H, W, mask2d=None):
    P = P_all.reshape(H, W, -1).transpose(2,0,1)  # (T,H,W)
    T, H_, W_ = P.shape
    gx = np.full_like(P, np.nan, dtype=np.float32)
    gy = np.full_like(P, np.nan, dtype=np.float32)

    gx[:, :, 1:] = np.angle(np.exp(1j*(P[:, :, 1:] - P[:, :, :-1])))
    gy[:, 1:, :] = np.angle(np.exp(1j*(P[:, 1:, :] - P[:, :-1, :])))

    if mask2d is not None:
        m = mask2d.astype(bool)
        gx_mask = np.zeros((H_, W_), bool); gx_mask[:, 1:] = (m[:, 1:] & m[:, :-1])
        gy_mask = np.zeros((H_, W_), bool); gy_mask[1:, :] = (m[1:, :] & m[:-1, :])
        gx[:, ~gx_mask] = np.nan
        gy[:, ~gy_mask] = np.nan

    return gx, gy

def phase_velocity_field(Phi, gx, gy, fs_eff, eps=1e-6):
    dt = np.gradient(np.unwrap(Phi)) * fs_eff   # (T,)
    T,H,W = gx.shape
    dt3 = dt[:,None,None]
    grad2 = gx*gx + gy*gy + eps
    vx = -(dt3 * gx) / grad2
    vy = -(dt3 * gy) / grad2
    return vx, vy

def plot_wave_maps(vx, vy, title, out_png, step=QUIVER_STEP, ref_percentile=VEL_REF_PCTL):
    if not np.isfinite(vx).any() or not np.isfinite(vy).any():
        fig, ax = plt.subplots(figsize=(4.2,3.9))
        ax.text(0.5, 0.5, "no valid velocity", ha="center", va="center")
        ax.axis("off"); save_fig(fig, out_png); return

    Vx = np.nanmedian(vx, axis=0)
    Vy = np.nanmedian(vy, axis=0)
    mag = np.hypot(Vx, Vy)
    ref = np.nanpercentile(mag, ref_percentile)
    if not np.isfinite(ref) or ref <= 0:
        ref = np.nanmax(mag) if np.isfinite(np.nanmax(mag)) else 1.0

    H,W = Vx.shape; Y,X = np.mgrid[0:H, 0:W]
    fig, ax = plt.subplots(figsize=(4.6,4.2))
    im = ax.imshow(np.ma.masked_invalid(mag), cmap=masked_cmap("viridis"))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("|v| (a.u.)")
    q = ax.quiver(X[::step,::step], Y[::step,::step], (Vx/ref)[::step,::step], (Vy/ref)[::step,::step],
                  color="w", scale_units="xy", scale=4.0, width=0.0025)
    ax.quiverkey(q, X=0.88, Y=1.05, U=1.0, label=f"dir, mag scaled / {ref:.2g}", labelpos='E', coordinates='axes')
    ax.set_xlim(-0.5,W-0.5); ax.set_ylim(H-0.5,-0.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title); save_fig(fig, out_png)

def plot_phase_gradient_coherence(gx, gy, out_png):
    mag = np.hypot(gx, gy).astype(np.float32)
    valid = np.isfinite(mag) & (mag > 0)
    ux = np.zeros_like(gx, dtype=np.float32); uy = np.zeros_like(gy, dtype=np.float32)
    ux[valid] = gx[valid] / mag[valid]; uy[valid] = gy[valid] / mag[valid]
    K = valid.sum(axis=0).astype(np.float32)
    Ux = ux.sum(axis=0); Uy = uy.sum(axis=0)
    CI = np.full_like(Ux, np.nan, dtype=np.float32)
    ok = K > 0
    if np.any(ok):
        CI[ok] = np.sqrt((Ux[ok]/K[ok])**2 + (Uy[ok]/K[ok])**2)
        CI[ok] = np.clip(CI[ok], 0.0, 1.0)
    plot_scalar_map(CI, "Phase-gradient orientation coherence", out_png, vmin=0, vmax=1, clabel="coherence")

# ---- Distance-decay for phi-corr & PLV ----
def plot_distance_decay_phi_corr(P_all, H, W, out_dir, max_pairs=20000):
    coords = np.array([(y,x) for y in range(H) for x in range(W)], dtype=float)
    Th = P_all
    mu = np.arctan2(np.sin(Th).mean(axis=1), np.cos(Th).mean(axis=1))
    X = np.sin(Th - mu[:,None]).astype(np.float32)
    N,_ = X.shape
    rng = np.random.default_rng(0)
    total_pairs = N*(N-1)//2
    if total_pairs == 0: return
    pick = min(max_pairs, total_pairs)
    idx_i = rng.integers(0, N-1, size=pick)
    idx_j = idx_i + 1 + rng.integers(0, N - idx_i - 1)
    ds = np.linalg.norm(coords[idx_i]-coords[idx_j], axis=1)
    num = np.sum(X[idx_i]*X[idx_j], axis=1)
    den = np.sqrt(np.sum(X[idx_i]*X[idx_i],axis=1) * np.sum(X[idx_j]*X[idx_j],axis=1) + 1e-12)
    rs = np.where(den>0, num/den, 0.0)
    bins = np.linspace(0, np.sqrt(H*H+W*W), 30)
    digit = np.digitize(ds, bins)
    m = np.array([np.mean(rs[digit==k]) if np.any(digit==k) else np.nan for k in range(1,len(bins))])
    centers = 0.5*(bins[:-1]+bins[1:])
    fig, ax = plt.subplots(figsize=(6.0,3.1))
    ax.plot(centers, m, lw=1.2); ax.set_ylim(-1,1)
    ax.set_xlabel("inter-electrode distance (grid units)"); ax.set_ylabel("ϕ-corr (circular)")
    ax.set_title("Spatial decay of phase correlation")
    save_fig(fig, os.path.join(out_dir, "phi_corr_decay"))

def plot_plv_distance_decay(P_all, figs_dir, H, W, step=8, max_pairs=20000):
    idx = np.array([y*W+x for y in range(0,H,step) for x in range(0,W,step)], dtype=int)
    if idx.size < 3: return
    P = P_all[idx,:]
    T = P.shape[1]; chunk = 4000
    C = np.zeros((idx.size, idx.size), dtype=np.complex64)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        E = np.exp(1j*P[:, t0:t1]).astype(np.complex64)
        C += E @ E.conj().T
    PLV = np.abs(C) / max(1, T)
    coords = np.array([(i//W, i%W) for i in idx], float)
    rng = np.random.default_rng(0)
    total = PLV.shape[0]*(PLV.shape[0]-1)//2
    n = min(max_pairs, total)
    ii = rng.integers(0, PLV.shape[0]-1, n)
    jj = ii + 1 + rng.integers(0, PLV.shape[0]-ii-1)
    d = np.linalg.norm(coords[ii]-coords[jj], axis=1)
    v = PLV[ii, jj]
    bins = np.linspace(0, np.sqrt(H*H+W*W), 30)
    digit = np.digitize(d,bins)
    m = np.array([np.mean(v[digit==k]) if np.any(digit==k) else np.nan for k in range(1,len(bins))])
    centers = 0.5*(bins[:-1]+bins[1:])
    fig,ax=plt.subplots(figsize=(6.0,3.1))
    ax.plot(centers, m); ax.set_ylim(0,1); ax.set_xlabel("distance"); ax.set_ylabel("PLV")
    ax.set_title("Spatial decay of PLV")
    save_fig(fig, os.path.join(figs_dir,"plv_decay"))

# ---- Accumulation visuals ----
def plot_accum_distributions(R, dphi, vx, vy, figs_dir):
    import matplotlib.pyplot as plt
    def _safe_flat(a):
        v = a[np.isfinite(a)]
        return v if v.size else np.array([np.nan])

    Vmag = np.hypot(np.nanmedian(vx, axis=0), np.nanmedian(vy, axis=0)).ravel()
    vals = {
        "R(t)": _safe_flat(R),
        "|dΦ/dt| (rad/s)": np.abs(_safe_flat(dphi)),
        "|v| (a.u., median)": _safe_flat(Vmag),
    }

    fig, ax = plt.subplots(1,3, figsize=(10,2.8))
    for k,(axk) in zip(vals.keys(), ax):
        x = vals[k]
        x = x[np.isfinite(x)]
        if x.size == 0:
            axk.text(0.5,0.5,"no data",ha="center",va="center"); axk.axis("off"); continue
        axk.hist(x, bins=40, density=True)
        axk.set_title(k)
    save_fig(fig, os.path.join(figs_dir, "accum_distributions"))

    fig, ax = plt.subplots(1,3, figsize=(10,2.8))
    for k,(axk) in zip(vals.keys(), ax):
        xx = vals[k]; xx = xx[np.isfinite(xx)]
        if xx.size == 0:
            axk.axis("off"); continue
        x = np.sort(xx); y = np.linspace(0,1,x.size,endpoint=False)
        axk.plot(x,y)
        axk.set_title(k+" — CDF")
    save_fig(fig, os.path.join(figs_dir, "accum_cdfs"))

def plot_wave_direction_rose(gx, gy, figs_dir, nbins=36):
    ang = np.arctan2(np.nanmedian(gy,axis=0), np.nanmedian(gx,axis=0))
    a = ang[np.isfinite(ang)]
    if a.size == 0:
        fig,ax=plt.subplots(); ax.axis("off")
        ax.text(0.5,0.5,"no valid directions",ha="center",va="center")
        save_fig(fig, os.path.join(figs_dir,"wave_dir_rose")); return
    hist, edges = np.histogram(a, bins=nbins, range=(-np.pi,np.pi), density=True)
    theta = 0.5*(edges[:-1]+edges[1:])
    fig = plt.figure(figsize=(3.0,3.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(theta, hist, width=(2*np.pi)/nbins)
    ax.set_title("")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    save_fig(fig, os.path.join(figs_dir, "wave_dir_rose"))

def detect_sync_episodes(R, fs_eff, tau=0.65, min_len_s=0.5):
    r = np.nan_to_num(R, nan=0.0)
    above = r >= tau
    starts = np.where(np.diff(np.pad(above.astype(int),(1,1)))==1)[0]
    ends   = np.where(np.diff(np.pad(above.astype(int),(1,1)))==-1)[0]
    dur = (ends - starts) / fs_eff
    keep = dur >= min_len_s
    episodes = [(int(s), int(e)) for s,e,k in zip(starts, ends, keep) if k]
    return episodes, dict(
        threshold=tau,
        n=len(episodes),
        mean_duration_s=float(np.mean(dur[keep])) if np.any(keep) else 0.0,
        occupancy_pct=float(100.0*np.sum(ends[keep]-starts[keep])/(len(r)+1e-9))
    )

def plot_R_with_episodes(t, R, fs_eff, episodes, figs_dir):
    x = t/fs_eff
    sw = int(max(1, fs_eff*0.4))
    x1, y1 = _ds_plot(x, np.convolve(np.nan_to_num(R), np.ones(sw)/sw, mode="same"),
                      max_pts=MAX_PTS_PLOT)
    fig, ax = plt.subplots(figsize=(9,2.6))
    ax.plot(x1, y1, lw=0.9); ax.set_ylim(0,1); ax.set_ylabel("R(t)"); ax.set_xlabel("time (s)")
    for s,e in episodes:
        ax.axvspan(s/fs_eff, e/fs_eff, color="tab:orange", alpha=0.2, lw=0)
    ax.set_title("R(t) with synchronization episodes")
    save_fig(fig, os.path.join(figs_dir, "R_with_episodes"))

def plot_phase_kymograph(P_all, H, W, mask2d, fs_eff, figs_dir):
    ys, xs = np.where(mask2d)
    if ys.size == 0: return
    y0,y1 = ys.min(), ys.max(); x0,x1 = xs.min(), xs.max()
    diag = np.array([((1-t)*y0 + t*y1, (1-t)*x0 + t*x1) for t in np.linspace(0,1,200)])
    diag = np.round(diag).astype(int)
    diag = np.unique(diag, axis=0)
    ridx = np.array([y*W+x for y,x in diag], int)
    Pline = P_all[ridx,:]  # (L,T)
    fig, ax = plt.subplots(figsize=(9,3.4))
    cm = masked_cmap("twilight")
    im = ax.imshow(np.ma.masked_invalid(Pline), aspect="auto", cmap=cm, vmin=-np.pi, vmax=np.pi,
                   extent=[0, Pline.shape[1]/fs_eff, 0, Pline.shape[0]])
    ax.set_xlabel("time (s)"); ax.set_ylabel("path index")
    ax.set_title("Phase kymograph (diagonal path)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase (rad)")
    save_fig(fig, os.path.join(figs_dir, "phase_kymograph"))

# ================== PER-FILE DRIVER ==================
def process_h5_allphase(h5_path, out_root,
                        mode="wideband", bands=BANDS,
                        power_exponent=POWER_EXP, wideband=WIDEBAND_HZ,
                        export_seconds=True):
    set_plot_theme()

    base = os.path.splitext(os.path.basename(h5_path))[0]
    out_dir = ensure(os.path.join(out_root, base))
    tabs = ensure(os.path.join(out_dir, "tables"))
    figs = ensure(os.path.join(out_dir, "figs"))

    steps = ["open","meta+mask","build_phase","clean_time","kuramoto","tables","figs","spatial","snapshots","coherogram","accum"]
    with tqdm(total=len(steps), desc=f"{base}", leave=False) as pbar:
        f = h5py.File(h5_path, "r"); pbar.update(1)

        H, W, fs_dec = get_meta(f)
        mask2d = get_mask2d(f, H, W); mask1d = mask2d.reshape(-1); pbar.update(1)

        # --- Build ALL-phase ---
        if mode == "wideband":
            if "/lfp/ts_2ms_full" not in f:
                f.close(); pbar.update(len(steps)-2); return
            TS = f["/lfp/ts_2ms_full"][:].astype(np.float32)  # (N,T)
            P_all, fs_eff = wideband_phase_from_ts(
                TS, fs_dec, f_lo=wideband[0], f_hi=wideband[1],
                show_channel_pbar=PBAR_CHANNEL_WIDEBAND,
                block_ch=FILTER_BLOCK_CH, filtfilt_method=FILTFILT_METHOD,
                time_decimate=TIME_DECIMATE_WIDEBAND
            )
            P_all = masked_nan(P_all, mask1d)
        elif mode == "fusion":
            if "/lfp/phase_full" not in f:
                f.close(); pbar.update(len(steps)-2); return
            band_phases = {}
            for b in bands:
                k = f"/lfp/phase_full/phase_{b}"
                if k in f["/lfp/phase_full"]:
                    band_phases[b] = masked_nan(f[k][:], mask1d)
            if not band_phases:
                f.close(); pbar.update(len(steps)-2); return
            band_powers = {}
            if "/lfp/bands" in f:
                for b in bands:
                    k = f"/lfp/bands/bp_{b}_ts"
                    if k in f["/lfp/bands"]:
                        band_powers[b] = np.maximum(f[k][:], 0.0).astype(np.float32)
            P_all = composite_phase_from_bands(band_phases, band_powers or None, power_exponent=power_exponent)
            fs_eff = fs_dec
        else:
            f.close(); raise ValueError("mode must be 'wideband' or 'fusion'")
        pbar.update(1)

        # --- Drop all-NaN time bins (robust) ---
        time_ok = np.isfinite(P_all).any(axis=0)
        dropped = int((~time_ok).sum())
        if dropped > 0:
            P_all = P_all[:, time_ok]
            tqdm.write(f"[INFO] {base}: dropped {dropped} all-NaN time bins (of {time_ok.size})")
        if P_all.shape[1] == 0:
            tqdm.write(f"[WARN] {base}: no valid time bins after filtering; skipping.")
            f.close(); return
        pbar.update(1)

        # --- Kuramoto ---
        N, T = P_all.shape
        t = np.arange(T, dtype=float)
        R, Phi = kuramoto_R_Phi(P_all)
        dphi = phi_velocity(Phi, fs_eff)
        Ri = local_coherence_map(P_all, Phi).reshape(H, W); Ri[~mask2d] = np.nan
        pbar.update(1)

        # --- Tables ---
        pd.DataFrame({"t_bin": t, "R": R, "Phi_rad": Phi, "dPhi_dt_rad_per_s": dphi}) \
            .to_csv(os.path.join(tabs, "kuramoto_ALL.csv"), index=False)
        summary = dict(
            mode=mode, bands=list(bands),
            R_mean=float(np.nanmean(R)) if np.any(np.isfinite(R)) else float("nan"),
            R_std=float(np.nanstd(R)) if np.any(np.isfinite(R)) else float("nan"),
            dPhi_dt_abs_mean=float(np.nanmean(np.abs(dphi))) if np.any(np.isfinite(dphi)) else float("nan"),
            N_channels=int(np.sum(mask2d)),
            fs_eff=float(fs_eff)
        )
        with open(os.path.join(tabs, "kuramoto_ALL_summary.json"), "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        pbar.update(1)

        # --- Global figs (clean) ---
        plot_R_Phi_pretty(t, R, Phi, os.path.join(figs, "R_Phi_ALL_pretty"), fs_eff)
        plot_phi_velocity(t, dphi, os.path.join(figs, "dPhi_dt_ALL_pretty"), fs_eff)
        plot_scalar_map(Ri, "Local coherence R_i — ALL", os.path.join(figs, "coherence_map_ALL"),
                        vmin=0, vmax=1, clabel="R_i")
        pbar.update(1)

        # --- Spatial matrices & wave maps ---
        try: export_corr_matrix(f, figs, H, W)
        except Exception as e: tqdm.write(f"[WARN] corr matrix skipped: {e}")
        try: export_plv_matrix_from_phase(P_all, figs, H, W)
        except Exception as e: tqdm.write(f"[WARN] PLV matrix skipped: {e}")
        try: export_phi_circ_corr_matrix(P_all, figs, H, W)
        except Exception as e: tqdm.write(f"[WARN] phi-corr matrix skipped: {e}")

        gx, gy = phase_gradient_maps(P_all, H, W, mask2d=mask2d)       # (T,H,W)
        vx, vy = phase_velocity_field(Phi, gx, gy, fs_eff)             # (T,H,W)
        plot_wave_maps(vx, vy, "Traveling-wave velocity (time-median)",
                       os.path.join(figs, "wave_velocity_median"))
        plot_phase_gradient_coherence(gx, gy, os.path.join(figs, "phase_grad_coherence"))
        plot_distance_decay_phi_corr(P_all, H, W, figs)
        plot_plv_distance_decay(P_all, figs, H, W)
        pbar.update(1)

        # --- Event-picked snapshots ---
        frames = pick_event_frames(R, fs_eff, n_peaks=SNAP_PKS, pad_s=SNAP_PAD_SEC)
        for ti in tqdm(frames, desc="snapshots (event-picked)", leave=False):
            img = P_all[:, ti].reshape(H, W); img[~mask2d] = np.nan
            plot_phase_snapshot_with_grad(img, gx[ti], gy[ti],
                f"Phase + gradient t={ti}", os.path.join(figs, f"phase_event_t{ti:06d}"))
        pbar.update(1)

        # --- Coherogram ---
        Ctime = local_circular_corr_over_time(P_all, H, W, win_s=COH_WIN_SEC, fs_eff=fs_eff)
        if np.isfinite(Ctime).any():
            C_med = np.nanmedian(Ctime, axis=0)
            C_mean_trace = np.nanmean(Ctime.reshape(Ctime.shape[0], -1), axis=1)
        else:
            C_med = np.zeros((H, W), np.float32)
            C_mean_trace = np.zeros(Ctime.shape[0], np.float32)

        plot_scalar_map(C_med, f"Local circular ϕ-corr (median, win={COH_WIN_SEC}s)",
                        os.path.join(figs, "phi_local_corr_median"), vmin=0, vmax=1, clabel="corr")
        tx = np.arange(len(C_mean_trace))/fs_eff
        x_plot, y_plot = _ds_plot(tx, C_mean_trace, max_pts=MAX_PTS_PLOT,
                                  smooth_win=int(max(1, fs_eff*SMOOTH_SEC)))
        fig, ax = plt.subplots(figsize=(9,2.3))
        ax.plot(x_plot, y_plot, lw=0.8); ax.set_ylim(0,1)
        ax.set_xlabel("time (s)"); ax.set_ylabel("⟨local ϕ-corr⟩")
        ax.set_title(f"Coherogram summary (win={COH_WIN_SEC}s)")
        save_fig(fig, os.path.join(figs, "phi_local_corr_trace"))
        pbar.update(1)

        # --- Accumulations & episodes & kymograph ---
        plot_accum_distributions(R, dphi, vx, vy, figs)
        plot_wave_direction_rose(gx, gy, figs)
        episodes, epi_stats = detect_sync_episodes(R, fs_eff, tau=0.65, min_len_s=0.5)
        plot_R_with_episodes(t, R, fs_eff, episodes, figs)
        with open(os.path.join(tabs, "sync_episode_stats.json"), "w", encoding="utf-8") as fp:
            json.dump(epi_stats, fp, indent=2)
        plot_phase_kymograph(P_all, H, W, mask2d, fs_eff, figs)
        pbar.update(1)

        f.close()

# ================== Coherogram (sliding local ϕ-corr) ==================
def local_circular_corr_over_time(P_all, H, W, win_s, fs_eff):
    T = P_all.shape[1]
    Wn = max(1, int(win_s * fs_eff))
    if Wn > T: Wn = T
    P = P_all.reshape(H, W, T)
    out = np.zeros((T, H, W), dtype=np.float32)
    step = max(1, Wn // 4)

    for t0 in tqdm(range(0, T - Wn + 1, step), desc="local ϕ-corr(time)", leave=False):
        t1 = t0 + Wn
        block = P[:, :, t0:t1]
        if not np.isfinite(block).any():
            continue
        s = np.sin(block); c = np.cos(block)
        s[~np.isfinite(s)] = 0.0; c[~np.isfinite(c)] = 0.0
        mu = np.arctan2(s.mean(axis=2), c.mean(axis=2))
        X = np.sin(block - mu[..., None]).astype(np.float32)

        num = np.zeros((H, W), np.float32); den = np.zeros((H, W), np.float32)
        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            X2 = np.roll(X, shift=(dy, dx), axis=(0,1))
            if dy == 1:  X2[0, :, :] = 0
            if dy == -1: X2[-1, :, :] = 0
            if dx == 1:  X2[:, 0, :] = 0
            if dx == -1: X2[:, -1, :] = 0
            n = np.sum(X * X2, axis=2)
            d = np.sqrt(np.sum(X * X, axis=2) * np.sum(X2 * X2, axis=2) + 1e-12)
            num += n; den += d

        C = np.zeros((H, W), np.float32)
        ok = den > 0
        C[ok] = num[ok] / den[ok]
        out[t0:t1, :, :] = C[None, ...]
    return out  # (T,H,W)

# ================== SNAPSHOTS PICKER ==================
def pick_event_frames(R, fs_eff, n_peaks=SNAP_PKS, pad_s=SNAP_PAD_SEC):
    r = np.nan_to_num(R, nan=0.0)
    peaks, _ = find_peaks(r, distance=int(max(1, fs_eff*0.8)))
    if peaks.size == 0:
        return np.linspace(0, len(r)-1, n_peaks, dtype=int)
    idx = np.argsort(r[peaks])[::-1][:n_peaks]
    core = peaks[idx]
    pad  = int(max(1, pad_s * fs_eff))
    frames = []
    for p in core:
        frames.extend([max(0,p-pad), p, min(len(r)-1, p+pad)])
    return np.unique(np.clip(frames, 0, len(r)-1)).astype(int)

# ================== FOLDER WRAPPER ==================
def run_folder(in_dir=IN_DIR, out_root=OUT_ROOT, pattern="*.h5", **kwargs):
    import fnmatch
    ensure(out_root)
    files = sorted(os.path.join(in_dir, f) for f in os.listdir(in_dir)
                   if f.endswith(".h5") and fnmatch.fnmatch(f, pattern))
    for p in tqdm(files, desc="Files", unit="file"):
        try:
            process_h5_allphase(p, out_root, **kwargs)
        except Exception as e:
            tqdm.write(f"[WARN] {os.path.basename(p)} failed: {e}")

# ================== CLI ==================
def build_argparser():
    p = argparse.ArgumentParser(description="Hilbert phase / Kuramoto synchrony analysis over a folder of 2 ms H5 files")
    p.add_argument("--in_dir", default=IN_DIR, help="Folder of <slice_id>_<condition>_2ms.h5 files (default: %(default)s)")
    p.add_argument("--out_root", default=OUT_ROOT, help="Output root for tables/figures (default: %(default)s)")
    p.add_argument("--mask", default=MASK_NPY, help="Optional 64x64 contact mask .npy (default: %(default)s)")
    p.add_argument("--mode", default=MODE, choices=["wideband", "fusion"], help="Phase estimation mode")
    p.add_argument("--pattern", default="*.h5", help="Glob pattern for H5 files within in_dir (default: %(default)s)")
    return p


# ================== MAIN ==================
if __name__ == "__main__":
    args = build_argparser().parse_args()
    MASK_NPY = args.mask
    run_folder(
        in_dir=args.in_dir,
        out_root=args.out_root,
        pattern=args.pattern,
        mode=args.mode,
        bands=BANDS,
        power_exponent=POWER_EXP,
        wideband=WIDEBAND_HZ,
        export_seconds=EXPORT_SECONDS,
    )