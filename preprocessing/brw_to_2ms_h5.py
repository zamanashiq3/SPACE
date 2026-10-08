# -*- coding: utf-8 -*-
"""
brw_to_2ms_h5.py -- SPACE framework: raw BRW ingestion, denoising, 2 ms binning, CSD.

Manuscript figure(s): Figure 1A (raw HD-MEA recording -> denoising -> 2 ms binning -> CSD).

What it computes:
    One-step, RAM-safe conversion from a raw 3Brain BioCAM ``.brw`` HD-MEA recording
    (4096 channels, 64x64 grid) to a compact 2 ms-resolution HDF5 file. Per channel it
    applies a causal streaming 60 Hz notch (Q = 30) + 1-350 Hz Butterworth band-pass
    (order 3, SOS, scipy.signal.sosfilt with filter state carried across 4000-sample
    chunks), smooths the filtered LFP with a 2 ms moving average advanced one sample at a
    time (so the 10 kHz rate is retained; ``bin_stride_mode="rolling"``), and computes a
    vectorized planar CSD (Gaussian sigma = 1 electrode, reflecting boundaries, then the
    five-point Laplacian; sign: negative = sink) from the smoothed LFP. It also derives per-channel summary maps
    (RMS/peak-to-peak/line-length/AUC for LFP; min/max/std for CSD), optional event
    onset/count/integral summaries, and an optional passive-conductivity snapshot.

    This is the entry point for every downstream SPACE analysis: it replaces an older
    two-step workflow that first wrote a full raw-resolution LFP/CSD H5 and then a second
    analysis H5; this script skips the intermediate raw-resolution file and writes only
    the useful 2 ms output. The default ``rolling`` mode reproduces the manuscript
    (2 ms moving average at every sample, 10 kHz output). ``--bin_stride_mode nonoverlap``
    writes non-overlapping 2 ms bins (500 Hz, ~20x smaller) for quick inspection only;
    the ROI event epochs used for statistics are produced at 500 Hz downstream by
    ``preprocessing/extract_event_epochs.py``.

Inputs / outputs (see data/README.md for the full schema):
    Input:  one or more ``data/raw_brw/*.brw`` files, and a 64x64 boolean contact mask
            produced by ``preprocessing/tissue_registration_mask.py``
            (``data/masks/<slice_id>__contact_mask_64x64.npy``).
    Output: one HDF5 file per recording under ``data/processed_2ms/<group>/``, named
            ``<slice_id>_<condition>_2ms.h5`` (this script's own default suffix is
            ``_2ms_optimized.h5``; rename/organize into genotype subfolders such as
            ``WT/`` and ``5xFAD/`` as needed for the downstream ROI/phenotype scripts).

Default output per BRW:
    /lfp/ts_2ms_full              [4096, T2] float32, optional but enabled by default
    /csd/ts_2ms_full              [4096, T2] float32, optional but enabled by default
    /lfp/metrics/*                RMS, p2p, line, auc maps
    /csd/metrics/*                min, max, std maps
    /events/*                     optional onsets/counts/integrals; rasters disabled by default
    /conductivity/*               optional median-snapshot passive conductivity maps
    /meta attrs                   fs, win, stride, genotype/drug parsing, etc.

Important:
    - This uses causal streaming IIR filtering (sosfilt), not filtfilt.
      It is much faster and chunk-safe, but introduces small filter phase delay.
      For publication waveforms, event alignment usually removes most practical impact.
    - Set SAVE_TSERIES=False if you only want scalar maps and event summaries.
    - Set SAVE_RASTERS=False unless you truly need full binary rasters; they consume space.

Requirements:
    pip install numpy h5py scipy tqdm
"""

import os
import re
import math
import argparse
from dataclasses import dataclass

import h5py
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, iirnotch, tf2sos, sosfilt
from scipy.ndimage import gaussian_filter


# ------------------------- CONFIG -------------------------
@dataclass
class Config:
    # geometry / signal
    H: int = 64
    W: int = 64
    dx_um: float = 21.0
    lfp_low_hz: float = 1.0
    lfp_high_hz: float = 350.0
    filter_order: int = 3
    notch_hz: float = 60.0       # set to 50.0 if needed; set <=0 to disable
    notch_q: float = 30.0
    gaussian_sigma: float = 1.0

    # 2 ms binning
    bin_ms: float = 2.0
    bin_stride_mode: str = "rolling"     # Methods: 2 ms moving average advanced one sample at a time (keeps 10 kHz).
                                         # "nonoverlap" = non-overlapping 2 ms bins (500 Hz, ~20x smaller; for quick looks only)

    # memory / IO
    raw_chunk_frames: int = 4000   # lower this if RAM is tight; 2000-6000 is usually safe
    h5_time_chunk: int = 2048
    h5_channel_chunk: int = 256
    compression: str = "lzf"       # "lzf" is fast; "gzip" smaller but slower; None disables compression

    # output switches
    save_tseries: bool = True      # save /lfp/ts_2ms_full and /csd/ts_2ms_full
    save_events: bool = True       # event summaries from 2 ms saved arrays
    save_rasters: bool = False     # full rasters are big; keep False by default
    save_conductivity: bool = True

    # thresholds / event settings
    lfp_z_thr: float = 3.0
    csd_k_std: float = 2.0

    # conductivity
    sigma_min: float = 0.0
    sigma_max: float = 3.0
    conductivity_sample_cap: int = 64


# ------------------------- BASIC HELPERS -------------------------
def ensure_group(h5, path):
    g = h5
    for p in path.strip("/").split("/"):
        if p:
            g = g.require_group(p)
    return g


def create_dataset(h5, path, shape, dtype, chunks=None, compression="lzf"):
    parent = ensure_group(h5, os.path.dirname(path))
    name = os.path.basename(path)
    if name in parent:
        del parent[name]
    kwargs = dict(shape=shape, dtype=dtype)
    if chunks is not None:
        kwargs["chunks"] = chunks
    if compression not in (None, "none", "None", ""):
        kwargs["compression"] = compression
    return parent.create_dataset(name, **kwargs)


def extract_drug_name(filename):
    """Condition token in the file name -> 4AP, AFDX, BIC, CNQX, ML297 or unknown.
    ``base``/``baseline``/``veh`` denote the 4-AP baseline recording."""
    b = os.path.basename(filename).lower()
    for tok, out in [("af-dx", "AFDX"), ("afdx", "AFDX"), ("4ap", "4AP"), ("bic", "BIC"), ("cnqx", "CNQX"),
                     ("ml297", "ML297"), ("baseline", "4AP"), ("base", "4AP"), ("veh", "4AP")]:
        if tok in b:
            return out
    return "unknown"


def extract_genotype(filename):
    b = os.path.basename(filename).lower()
    if re.search(r"(^|[_\-])wt([_\-]|$)", b):
        return "WT"
    if any(x in b for x in ["5xfad", "app", "ad"]):
        return "5xFAD"
    if any(x in b for x in ["c57", "ctl", "ctrl", "control"]):
        return "CTL"
    return "unknown"


def read_brw_metadata(f):
    nframes = int(f["/3BRecInfo/3BRecVars/NRecFrames"][()].squeeze())
    fs = float(f["/3BRecInfo/3BRecVars/SamplingRate"][()].squeeze())
    bit_depth = int(f["/3BRecInfo/3BRecVars/BitDepth"][()].squeeze())
    maxv = float(f["/3BRecInfo/3BRecVars/MaxVolt"][()].squeeze())
    return nframes, fs, bit_depth, maxv


def design_streaming_filters(fs, cfg: Config, n_ch):
    bp_sos = butter(
        cfg.filter_order,
        [cfg.lfp_low_hz, cfg.lfp_high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    sos_list = []
    if cfg.notch_hz and cfg.notch_hz > 0:
        b, a = iirnotch(cfg.notch_hz, cfg.notch_q, fs)
        sos_list.append(tf2sos(b, a))
    sos_list.append(bp_sos)
    sos = np.vstack(sos_list).astype(np.float64)
    zi = np.zeros((sos.shape[0], 2, n_ch), dtype=np.float64)
    return sos, zi


def vectorized_csd(lfp_tyx, dx_um=21.0, sigma=1.0):
    """CSD = -Laplacian(gaussian-smoothed LFP) / dx^2, vectorized over time."""
    sm = gaussian_filter(lfp_tyx, sigma=(0, sigma, sigma)).astype(np.float32)
    lap = np.zeros_like(sm, dtype=np.float32)
    lap[:, 1:-1, 1:-1] = (
        sm[:, :-2, 1:-1] + sm[:, 2:, 1:-1] +
        sm[:, 1:-1, :-2] + sm[:, 1:-1, 2:] -
        4.0 * sm[:, 1:-1, 1:-1]
    )
    return (-lap / (dx_um ** 2)).astype(np.float32)


def bin_nonoverlap(chunk_tyx, tail_tyx, win):
    """Non-overlapping bin mean. Saves ~win-fold space vs rolling mode."""
    if tail_tyx is not None and tail_tyx.shape[0] > 0:
        x = np.concatenate([tail_tyx, chunk_tyx], axis=0)
    else:
        x = chunk_tyx
    n_bins = x.shape[0] // win
    if n_bins <= 0:
        return np.empty((0,) + x.shape[1:], dtype=np.float32), x
    used = n_bins * win
    y = x[:used].reshape(n_bins, win, x.shape[1], x.shape[2]).mean(axis=1).astype(np.float32)
    tail = x[used:].astype(np.float32, copy=False)
    return y, tail


def bin_rolling(chunk_tyx, tail_tyx, win):
    """Old behavior: rolling 2 ms average at every raw sample. Large output."""
    if tail_tyx is not None and tail_tyx.shape[0] > 0:
        x = np.concatenate([tail_tyx, chunk_tyx], axis=0)
    else:
        x = chunk_tyx
    if x.shape[0] < win:
        return np.empty((0,) + x.shape[1:], dtype=np.float32), x
    cs = np.cumsum(x, axis=0, dtype=np.float64)
    # prepend zero frame so rolling window is correct for first window
    z = np.zeros((1,) + x.shape[1:], dtype=np.float64)
    cs = np.concatenate([z, cs], axis=0)
    y = (cs[win:] - cs[:-win]) / win
    # return only bins whose window ended inside the new chunk
    n_take = min(chunk_tyx.shape[0], y.shape[0])
    y = y[-n_take:].astype(np.float32)
    tail = x[-(win - 1):].astype(np.float32, copy=False)
    return y, tail


def update_stats(block_ch_t, state, prev_vec=None):
    """Update scalar stats for [N_CH, T] block."""
    if block_ch_t.size == 0:
        return prev_vec
    B = block_ch_t.astype(np.float64, copy=False)
    state["sum"] += B.sum(axis=1)
    state["sumsq"] += (B * B).sum(axis=1)
    state["min"] = np.minimum(state["min"], B.min(axis=1))
    state["max"] = np.maximum(state["max"], B.max(axis=1))
    state["auc"] += np.abs(B).sum(axis=1)
    if prev_vec is None:
        state["line"] += np.abs(np.diff(B, axis=1)).sum(axis=1) if B.shape[1] > 1 else 0
    else:
        cat = np.concatenate([prev_vec[:, None], B], axis=1)
        state["line"] += np.abs(np.diff(cat, axis=1)).sum(axis=1)
    return B[:, -1].copy()


def init_stats(n_ch):
    return {
        "sum": np.zeros(n_ch, dtype=np.float64),
        "sumsq": np.zeros(n_ch, dtype=np.float64),
        "min": np.full(n_ch, np.inf, dtype=np.float64),
        "max": np.full(n_ch, -np.inf, dtype=np.float64),
        "auc": np.zeros(n_ch, dtype=np.float64),
        "line": np.zeros(n_ch, dtype=np.float64),
    }


# ------------------------- CONDUCTIVITY HELPERS -------------------------
def laplacian2d(V):
    L = np.zeros_like(V, dtype=np.float32)
    L[1:-1, 1:-1] = (
        V[:-2, 1:-1] + V[2:, 1:-1] + V[1:-1, :-2] + V[1:-1, 2:] - 4 * V[1:-1, 1:-1]
    )
    return L


def central_diff_x(A):
    D = np.zeros_like(A, dtype=np.float32)
    D[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / 2.0
    return D


def central_diff_y(A):
    D = np.zeros_like(A, dtype=np.float32)
    D[1:-1, :] = (A[2:, :] - A[:-2, :]) / 2.0
    return D


def simple_conductivity_snapshot(h5, cfg: Config, H, W, T2, mask=None):
    if T2 <= 0 or "/lfp/ts_2ms_full" not in h5 or "/csd/ts_2ms_full" not in h5:
        return
    n_samp = min(cfg.conductivity_sample_cap, T2)
    idx = np.linspace(0, T2 - 1, n_samp).astype(int)
    V = np.median(h5["/lfp/ts_2ms_full"][:, idx], axis=1).reshape(H, W).astype(np.float32)
    C = np.median(h5["/csd/ts_2ms_full"][:, idx], axis=1).reshape(H, W).astype(np.float32)

    L = laplacian2d(V)
    floor = max(1e-8, float(np.nanpercentile(np.abs(L), 5)) * 1e-2)
    valid = np.abs(L) > floor
    sigma = np.zeros_like(V, dtype=np.float32)
    sigma[valid] = -C[valid] / L[valid]
    sigma = gaussian_filter(sigma, sigma=1.0).astype(np.float32)
    sigma = np.clip(sigma, cfg.sigma_min, cfg.sigma_max).astype(np.float32)

    Ex, Ey = -central_diff_x(V), -central_diff_y(V)
    Jx, Jy = sigma * Ex, sigma * Ey
    p_map = (sigma * (Ex * Ex + Ey * Ey)).astype(np.float32)

    create_dataset(h5, "/conductivity/sigma_map", (H, W), "float32", chunks=(H, W), compression=cfg.compression)[:, :] = sigma
    create_dataset(h5, "/conductivity/E_field", (2, H, W), "float32", chunks=(2, H, W), compression=cfg.compression)[:, :, :] = np.stack([Ex, Ey])
    create_dataset(h5, "/conductivity/J_field", (2, H, W), "float32", chunks=(2, H, W), compression=cfg.compression)[:, :, :] = np.stack([Jx, Jy])
    create_dataset(h5, "/conductivity/p_map", (H, W), "float32", chunks=(H, W), compression=cfg.compression)[:, :] = p_map

    roi = mask.astype(bool) if mask is not None else np.ones((H, W), dtype=bool)
    g = ensure_group(h5, "/conductivity")
    g.attrs["median_sigma"] = float(np.nanmedian(sigma[roi]))
    g.attrs["p90_sigma"] = float(np.nanpercentile(sigma[roi], 90))
    g.attrs["p_mean"] = float(np.nanmean(p_map[roi]))
    g.attrs["sigma_invalid_frac"] = float(1.0 - valid.mean())


# ------------------------- EVENT SUMMARY FROM SAVED 2MS ARRAYS -------------------------
def compute_events_from_saved_2ms(h5, cfg: Config, H, W, T2):
    if T2 <= 0 or "/lfp/ts_2ms_full" not in h5 or "/csd/ts_2ms_full" not in h5:
        return
    N_CH = H * W
    lfp_mean = h5["/lfp/metrics/mean_map"][:].reshape(-1).astype(np.float32)
    lfp_rms = h5["/lfp/metrics/rms_map"][:].reshape(-1).astype(np.float32)
    csd_std = h5["/csd/metrics/std_map"][:].reshape(-1).astype(np.float32)
    sink_thr = (-cfg.csd_k_std * csd_std).astype(np.float32)
    src_thr = (cfg.csd_k_std * csd_std).astype(np.float32)

    burst_onset = np.full(N_CH, -1, dtype=np.int64)
    sink_onset = np.full(N_CH, -1, dtype=np.int64)
    source_onset = np.full(N_CH, -1, dtype=np.int64)
    sink_counts = np.zeros(N_CH, dtype=np.int64)
    source_counts = np.zeros(N_CH, dtype=np.int64)
    sink_integral = np.zeros(N_CH, dtype=np.float64)
    source_integral = np.zeros(N_CH, dtype=np.float64)

    if cfg.save_rasters:
        d_burst = create_dataset(h5, "/lfp/events/burst_raster", (N_CH, T2), "uint8",
                                 chunks=(min(cfg.h5_channel_chunk, N_CH), min(cfg.h5_time_chunk, T2)), compression=cfg.compression)
        d_sink = create_dataset(h5, "/csd/events/sink_raster", (N_CH, T2), "uint8",
                                chunks=(min(cfg.h5_channel_chunk, N_CH), min(cfg.h5_time_chunk, T2)), compression=cfg.compression)
        d_source = create_dataset(h5, "/csd/events/source_raster", (N_CH, T2), "uint8",
                                  chunks=(min(cfg.h5_channel_chunk, N_CH), min(cfg.h5_time_chunk, T2)), compression=cfg.compression)
    else:
        d_burst = d_sink = d_source = None

    for t0 in tqdm(range(0, T2, cfg.h5_time_chunk), desc="events-2ms", leave=False):
        t1 = min(t0 + cfg.h5_time_chunk, T2)
        L = h5["/lfp/ts_2ms_full"][:, t0:t1].astype(np.float32)  # [ch, t]
        C = h5["/csd/ts_2ms_full"][:, t0:t1].astype(np.float32)

        burst = ((L - lfp_mean[:, None]) / (lfp_rms[:, None] + 1e-12)) > cfg.lfp_z_thr
        sink = C < sink_thr[:, None]
        source = C > src_thr[:, None]

        if cfg.save_rasters:
            d_burst[:, t0:t1] = burst.astype(np.uint8)
            d_sink[:, t0:t1] = sink.astype(np.uint8)
            d_source[:, t0:t1] = source.astype(np.uint8)

        for arr, onset in [(burst, burst_onset), (sink, sink_onset), (source, source_onset)]:
            seen = arr.any(axis=1)
            first = arr.argmax(axis=1)
            m = (onset < 0) & seen
            onset[m] = t0 + first[m]

        neg = np.minimum(C, 0.0)
        pos = np.maximum(C, 0.0)
        sink_counts += sink.sum(axis=1)
        source_counts += source.sum(axis=1)
        # integral in 2 ms-bin units; multiply by bin duration seconds
        dt = cfg.bin_ms / 1000.0
        sink_integral += np.abs(neg).sum(axis=1) * dt
        source_integral += pos.sum(axis=1) * dt

    create_dataset(h5, "/lfp/events/burst_onset_idx", (N_CH, 1), "int64", chunks=(min(N_CH, 4096), 1), compression=cfg.compression)[:, 0] = burst_onset
    create_dataset(h5, "/csd/events/sink_onset_idx", (N_CH, 1), "int64", chunks=(min(N_CH, 4096), 1), compression=cfg.compression)[:, 0] = sink_onset
    create_dataset(h5, "/csd/events/source_onset_idx", (N_CH, 1), "int64", chunks=(min(N_CH, 4096), 1), compression=cfg.compression)[:, 0] = source_onset
    create_dataset(h5, "/csd/events/sink_counts", (N_CH, 1), "int64", chunks=(min(N_CH, 4096), 1), compression=cfg.compression)[:, 0] = sink_counts
    create_dataset(h5, "/csd/events/source_counts", (N_CH, 1), "int64", chunks=(min(N_CH, 4096), 1), compression=cfg.compression)[:, 0] = source_counts
    create_dataset(h5, "/csd/events/sink_integral", (N_CH, 1), "float32", chunks=(min(N_CH, 4096), 1), compression=cfg.compression)[:, 0] = sink_integral.astype(np.float32)
    create_dataset(h5, "/csd/events/source_integral", (N_CH, 1), "float32", chunks=(min(N_CH, 4096), 1), compression=cfg.compression)[:, 0] = source_integral.astype(np.float32)


# ------------------------- CORE PROCESSOR -------------------------
def process_one_brw(brw_path, mask_path, out_dir, cfg: Config):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(brw_path))[0]
    out_h5 = os.path.join(out_dir, f"{base}_2ms_optimized.h5")

    mask = np.load(mask_path).astype(bool)
    if mask.shape != (cfg.H, cfg.W):
        raise ValueError(f"Mask shape must be {(cfg.H, cfg.W)}, got {mask.shape}")
    active_idx = np.where(mask.reshape(-1))[0]
    N_CH = cfg.H * cfg.W

    with h5py.File(brw_path, "r") as f_in:
        nframes, fs, bit_depth, maxv = read_brw_metadata(f_in)
        raw = f_in["/3BData/Raw"]
        frame_size = len(active_idx)
        win = int(round((cfg.bin_ms / 1000.0) * fs))
        if win < 1:
            raise ValueError("bin_ms too small for sampling rate")
        if cfg.bin_stride_mode == "nonoverlap":
            T2 = nframes // win
        elif cfg.bin_stride_mode == "rolling":
            T2 = max(nframes - win + 1, 0)
        else:
            raise ValueError("bin_stride_mode must be 'nonoverlap' or 'rolling'")

        print(f"\n[BRW] {base}")
        print(f"  frames={nframes:,}, fs={fs:g} Hz, active_ch={frame_size}, bin={win} samples ({cfg.bin_ms} ms), T2={T2:,}")
        print(f"  output={out_h5}")

        # filters are applied to all 4096 channels; inactive channels stay zero
        sos, zi = design_streaming_filters(fs, cfg, N_CH)

        lfp_stats = init_stats(N_CH)
        csd_stats = init_stats(N_CH)
        lfp_prev = None
        csd_prev = None
        lfp_tail = None
        csd_tail = None
        t2_cursor = 0

        with h5py.File(out_h5, "w") as f_out:
            # metadata
            f_out.create_dataset("/mask", data=mask.astype(np.uint8), compression=cfg.compression)
            gmeta = ensure_group(f_out, "/meta")
            gmeta.attrs["source_brw"] = os.path.abspath(brw_path)
            gmeta.attrs["fs"] = fs
            gmeta.attrs["win_samples"] = win
            gmeta.attrs["bin_ms"] = cfg.bin_ms
            gmeta.attrs["bin_stride_mode"] = cfg.bin_stride_mode
            gmeta.attrs["T_raw"] = nframes
            gmeta.attrs["T2"] = T2
            gmeta.attrs["H"] = cfg.H
            gmeta.attrs["W"] = cfg.W
            gmeta.attrs["dx_um"] = cfg.dx_um
            gmeta.attrs["drug"] = extract_drug_name(base)
            gmeta.attrs["genotype"] = extract_genotype(base)
            gmeta.attrs["filter"] = "streaming causal sosfilt: notch + bandpass"
            gmeta.attrs["notch_hz"] = cfg.notch_hz
            gmeta.attrs["bandpass_hz"] = (cfg.lfp_low_hz, cfg.lfp_high_hz)

            # optional time-series datasets
            if cfg.save_tseries:
                d_lfp = create_dataset(
                    f_out, "/lfp/ts_2ms_full", (N_CH, T2), "float32",
                    chunks=(min(cfg.h5_channel_chunk, N_CH), min(cfg.h5_time_chunk, max(T2, 1))),
                    compression=cfg.compression,
                )
                d_csd = create_dataset(
                    f_out, "/csd/ts_2ms_full", (N_CH, T2), "float32",
                    chunks=(min(cfg.h5_channel_chunk, N_CH), min(cfg.h5_time_chunk, max(T2, 1))),
                    compression=cfg.compression,
                )
            else:
                d_lfp = d_csd = None

            n_chunks = math.ceil(nframes / cfg.raw_chunk_frames)
            pbar = tqdm(range(n_chunks), desc="BRW->2ms")
            for i in pbar:
                start = i * cfg.raw_chunk_frames
                end = min((i + 1) * cfg.raw_chunk_frames, nframes)
                n = end - start

                flat = raw[start * frame_size: end * frame_size]
                active = flat.reshape(n, frame_size).astype(np.float32)

                full = np.zeros((n, N_CH), dtype=np.float32)
                full[:, active_idx] = active
                full *= (maxv / (2 ** bit_depth))

                # streaming filter: [time, channel]
                filt, zi = sosfilt(sos, full, axis=0, zi=zi)
                lfp_tyx = filt.astype(np.float32).reshape(n, cfg.H, cfg.W)
                csd_tyx = vectorized_csd(lfp_tyx, dx_um=cfg.dx_um, sigma=cfg.gaussian_sigma)

                if cfg.bin_stride_mode == "nonoverlap":
                    lfp_2ms, lfp_tail = bin_nonoverlap(lfp_tyx, lfp_tail, win)
                    csd_2ms, csd_tail = bin_nonoverlap(csd_tyx, csd_tail, win)
                else:
                    lfp_2ms, lfp_tail = bin_rolling(lfp_tyx, lfp_tail, win)
                    csd_2ms, csd_tail = bin_rolling(csd_tyx, csd_tail, win)

                if lfp_2ms.shape[0] == 0:
                    continue

                # guard in case rolling length lands slightly different at final block
                nb = min(lfp_2ms.shape[0], csd_2ms.shape[0], T2 - t2_cursor)
                if nb <= 0:
                    continue
                lfp_ch_t = lfp_2ms[:nb].reshape(nb, N_CH).T.astype(np.float32)
                csd_ch_t = csd_2ms[:nb].reshape(nb, N_CH).T.astype(np.float32)

                if cfg.save_tseries:
                    d_lfp[:, t2_cursor:t2_cursor + nb] = lfp_ch_t
                    d_csd[:, t2_cursor:t2_cursor + nb] = csd_ch_t

                lfp_prev = update_stats(lfp_ch_t, lfp_stats, lfp_prev)
                csd_prev = update_stats(csd_ch_t, csd_stats, csd_prev)
                t2_cursor += nb
                pbar.set_postfix({"T2_written": t2_cursor})

            # finalize maps
            denom = max(t2_cursor, 1)
            lfp_mean = lfp_stats["sum"] / denom
            lfp_rms = np.sqrt(np.maximum(lfp_stats["sumsq"] / denom, 0))
            lfp_p2p = lfp_stats["max"] - lfp_stats["min"]
            csd_mean = csd_stats["sum"] / denom
            csd_var = csd_stats["sumsq"] / denom - csd_mean ** 2
            csd_std = np.sqrt(np.maximum(csd_var, 1e-12))

            for path, arr in [
                ("/lfp/metrics/mean_map", lfp_mean),
                ("/lfp/metrics/rms_map", lfp_rms),
                ("/lfp/metrics/p2p_map", lfp_p2p),
                ("/lfp/metrics/line_map", lfp_stats["line"]),
                ("/lfp/metrics/auc_map", lfp_stats["auc"]),
                ("/csd/metrics/mean_map", csd_mean),
                ("/csd/metrics/min_map", csd_stats["min"]),
                ("/csd/metrics/max_map", csd_stats["max"]),
                ("/csd/metrics/std_map", csd_std),
            ]:
                create_dataset(f_out, path, (cfg.H, cfg.W), "float32", chunks=(cfg.H, cfg.W), compression=cfg.compression)[:, :] = arr.reshape(cfg.H, cfg.W).astype(np.float32)

            gmeta.attrs["T2_written"] = t2_cursor

            if cfg.save_events and cfg.save_tseries:
                compute_events_from_saved_2ms(f_out, cfg, cfg.H, cfg.W, t2_cursor)

            if cfg.save_conductivity and cfg.save_tseries:
                simple_conductivity_snapshot(f_out, cfg, cfg.H, cfg.W, t2_cursor, mask=mask)

    print(f"[DONE] {out_h5}")
    return out_h5


# ------------------------- FOLDER / CLI -------------------------
def process_folder(brw_dir, mask_path, out_dir, cfg: Config, only_contains=None):
    files = [f for f in os.listdir(brw_dir) if f.lower().endswith(".brw")]
    files.sort()
    if only_contains:
        key = only_contains.lower()
        files = [f for f in files if key in f.lower()]
    if not files:
        print(f"No BRW files found in {brw_dir}")
        return []
    outs = []
    for fname in files:
        outs.append(process_one_brw(os.path.join(brw_dir, fname), mask_path, out_dir, cfg))
    return outs


def build_argparser():
    p = argparse.ArgumentParser(description="Optimized one-step BRW -> 2 ms LFP/CSD H5 processor")
    p.add_argument("--brw_dir", required=True, help="Folder containing .brw files")
    p.add_argument("--mask", required=True, help="64x64 contact mask .npy")
    p.add_argument("--out", required=True, help="Output folder")
    p.add_argument("--only", default=None, help="Optional filename substring filter, e.g. 4ap")
    p.add_argument("--chunk", type=int, default=4000, help="Raw frames per chunk. Lower if RAM constrained.")
    p.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"], help="HDF5 compression")
    p.add_argument("--bin_stride_mode", default="rolling", choices=["rolling", "nonoverlap"],
                   help="rolling (default, as in the manuscript): 2 ms moving average advanced one sample "
                        "at a time, output stays at the native rate (large files). nonoverlap: "
                        "non-overlapping 2 ms bins (500 Hz, ~20x smaller) for quick inspection.")
    p.add_argument("--no_tseries", action="store_true", help="Do not save /lfp/ts_2ms_full or /csd/ts_2ms_full")
    p.add_argument("--no_events", action="store_true", help="Skip event summaries")
    p.add_argument("--rasters", action="store_true", help="Save full event rasters; large")
    p.add_argument("--no_conductivity", action="store_true", help="Skip conductivity maps")
    p.add_argument("--notch", type=float, default=60.0, help="Notch frequency in Hz (e.g. 60 for North America/parts of Asia, 50 for most of Europe/Asia); <=0 disables.")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config(
        raw_chunk_frames=args.chunk,
        compression=None if args.compression == "none" else args.compression,
        bin_stride_mode=args.bin_stride_mode,
        save_tseries=not args.no_tseries,
        save_events=not args.no_events,
        save_rasters=args.rasters,
        save_conductivity=not args.no_conductivity,
        notch_hz=args.notch,
    )
    process_folder(args.brw_dir, args.mask, args.out, cfg, only_contains=args.only)


if __name__ == "__main__":
    main()
