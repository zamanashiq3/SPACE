# -*- coding: utf-8 -*-
"""
extract_event_epochs.py -- SPACE framework: ROI-level sampling of three network events per recording.

Manuscript Methods section: "Event alignment and analysis windows" (paragraph "For ROI-level
analyses, three network events were sampled from each recording ...").
Manuscript figure(s): produces the event epochs consumed by Figs 2-6 and Supplementary Fig. 2
(core/psd_wavelet_metrics.py, core/roi_phase_gradient_pgd.py, core/electrode_connectivity.py,
core/ctx_hippo_avalanche_kuramoto_compare.py, core/avalanche_statistics.py).

Inputs:
    data/processed_2ms/<group>/<name>.h5   (schema of preprocessing/brw_to_2ms_h5.py)
    data/masks/<slice_id>__contact_mask_64x64.npy   (matched by slice prefix in the file name,
                                                     else /mask in the H5, else --mask override)
    data/roi_definitions/<slice_id>_roi_definitions.csv   (roi_index, roi_name, r0, r1, c0, c1;
                                                           Python slice bounds; else --roi_csv)
Outputs:
    data/event_traces/<group>/<ROI>/<condition>/event_<k>_<t0ms>ms/
        lfp_event_traces.npy    float32 [16, 251]   ROI electrodes (row-major) x time, 500 Hz
        csd_event_traces.npy    float32 [16, 251]
        event_time_axis_ms.npy  float64 [251]       -100, -98, ..., 400 ms relative to event centre
        event_meta.json
    data/results/event_epochs/event_catalog.csv   one row per recording x event
    data/results/event_epochs/manifest.json
    data/results/event_epochs/qc/<group>__<file>__population_counts.(png|svg)

Algorithm (exactly as in the Methods):
    1. t0 (whole-slice onset) = argmax over the first 2 s of the mean |CSD| across tissue-contact
       electrodes (outer ring excluded), streamed in time chunks / channel blocks.
    2. On every tissue-contact electrode, negative LFP deflections = local minima of the LFP more
       than 3 robust s.d. below the electrode median, robust s.d. = MAD/0.6745 from the first
       500 ms; refractory period 4 ms per electrode (scipy.signal.find_peaks on -x, distance).
       Detection is restricted to t0-100 .. t0+400 ms.
    3. Deflections are counted per sample across electrodes; contiguous runs of non-zero counts
       are population events; size = total count in the run. The three runs whose sizes are
       closest to the 20th, 50th and 80th percentiles of the size distribution are selected
       (ties -> earlier run); each event is centred on the midpoint of its run.
    4. For every ROI and selected event, LFP and CSD [16, T] for -100..+400 ms around the centre
       are read at the stored rate and stored at 500 Hz (2 ms).

Implementation notes:
    - The electrode median used for the 3-SD criterion is taken from the same first 500 ms as the
      MAD (the Methods do not say otherwise).
    - The 500 Hz epoch is obtained by block-averaging non-overlapping q = fs/500 samples centred
      on each 2 ms tick (q = 20 at 10 kHz); at fs = 500 the samples are copied as is. fs must be
      an integer multiple of 500 Hz. Samples that fall outside the recording are NaN (downstream
      scripts fill NaNs by linear interpolation).
    - Percentile targets use numpy's default linear interpolation; when fewer than three runs
      exist the same run may serve several targets (a warning is printed). With >= 3 runs the
      three events are forced to be distinct (targets are assigned in the order 20/50/80).
    - Event k within the output folder (event_{k:03d}) is a running index per
      (group, ROI, condition) across recordings, because the spec layout has no slice level;
      slice_id / recording are recorded in event_meta.json and in event_catalog.csv, where
      `event_index` is this k and `percentile_target` identifies the 20/50/80 rank.
    - If the recording is shorter than 2 s the t0 search uses the whole recording; the detection
      window is clipped to the recording.
"""

import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
IN_ROOT = REPO_ROOT / "data" / "processed_2ms"
MASK_DIR = REPO_ROOT / "data" / "masks"
ROI_DIR = REPO_ROOT / "data" / "roi_definitions"
OUT_ROOT = REPO_ROOT / "data" / "event_traces"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "event_epochs"
T0_SEARCH_S = 2.0          # whole-slice onset searched in the first 2 s
NOISE_WINDOW_S = 0.5       # robust SD / median from the first 500 ms
THRESH_SD = 3.0            # deflection threshold (robust SDs below the median)
REFRACTORY_MS = 4.0        # per-electrode refractory period
DETECT_PRE_MS, DETECT_POST_MS = 100.0, 400.0    # detection window relative to t0
EPOCH_PRE_MS, EPOCH_POST_MS = 100.0, 400.0      # epoch extent relative to event centre
FS_EPOCH = 500.0
PERCENTILES = (20.0, 50.0, 80.0)
TIME_CHUNK = 5000          # samples per streamed chunk for the t0 search
CH_BLOCK = 512             # channels per block read
FIG_DPI = 300
# --- END EDIT ---

LFP_KEY, CSD_KEY = "/lfp/ts_2ms_full", "/csd/ts_2ms_full"


# =============================================================================
# HELPERS
# =============================================================================

def read_meta(f: h5py.File) -> Tuple[float, int, int]:
    m = f["/meta"].attrs
    fs = float(m.get("fs", 10000.0)); H = int(m.get("H", 64)); W = int(m.get("W", 64))
    return fs, H, W


def parse_condition(name: str) -> str:
    """Drug token in a file name -> one of 4AP, AFDX, BIC, CNQX, ML297, unknown."""
    n = name.lower()
    for tok, out in [("af-dx", "AFDX"), ("afdx", "AFDX"), ("4ap", "4AP"), ("bic", "BIC"), ("cnqx", "CNQX"), ("ml297", "ML297")]:
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
    return df


def inner_mask_idx(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    m = mask.copy()
    m[0, :] = m[-1, :] = False; m[:, 0] = m[:, -1] = False
    return np.flatnonzero(m.ravel())


# =============================================================================
# STEP 1: whole-slice onset t0
# =============================================================================

def find_t0_chunked(dset: h5py.Dataset, idx: np.ndarray, fs: float, search_s: float = T0_SEARCH_S,
                    time_chunk: int = TIME_CHUNK, ch_block: int = CH_BLOCK) -> Tuple[int, np.ndarray]:
    """argmax_t mean |CSD| over electrodes `idx` within the first `search_s` seconds (streamed)."""
    n_ch, T = dset.shape
    T_s = min(T, int(round(search_s * fs)))
    score = np.zeros(T_s, dtype=np.float64)
    for t0 in range(0, T_s, time_chunk):
        t1 = min(t0 + time_chunk, T_s)
        for c0 in range(0, n_ch, ch_block):
            c1 = min(c0 + ch_block, n_ch)
            sel = idx[(idx >= c0) & (idx < c1)] - c0
            if sel.size == 0:
                continue
            X = dset[c0:c1, t0:t1]
            score[t0:t1] += np.abs(X[sel]).sum(axis=0)
    score /= max(len(idx), 1)
    return int(np.argmax(score)), score


# =============================================================================
# STEP 2-3: deflection detection and population events
# =============================================================================

def read_rows_blocked(dset: h5py.Dataset, idx: np.ndarray, s0: int, s1: int, ch_block: int = CH_BLOCK) -> np.ndarray:
    """dset[idx, s0:s1] read in contiguous channel blocks (idx sorted)."""
    n_ch = dset.shape[0]
    out = np.empty((len(idx), s1 - s0), dtype=np.float32)
    k = 0
    for c0 in range(0, n_ch, ch_block):
        c1 = min(c0 + ch_block, n_ch)
        sel = idx[(idx >= c0) & (idx < c1)] - c0
        if sel.size == 0:
            continue
        out[k:k + sel.size] = dset[c0:c1, s0:s1][sel]
        k += sel.size
    return out


def detect_population_events(lfp: h5py.Dataset, idx: np.ndarray, fs: float, t0: int) -> dict:
    """Per-sample deflection counts in t0-100..t0+400 ms and contiguous-run events."""
    T = lfp.shape[1]
    n_noise = min(T, int(round(NOISE_WINDOW_S * fs)))
    base = read_rows_blocked(lfp, idx, 0, n_noise)                       # [n_el, n_noise]
    med = np.median(base, axis=1)
    rsd = np.median(np.abs(base - med[:, None]), axis=1) / 0.6745
    thr = med - THRESH_SD * rsd                                            # minima must be below this
    dist = max(1, int(round(REFRACTORY_MS * fs / 1000.0)))
    w0 = int(round(t0 - DETECT_PRE_MS * fs / 1000.0)); w1 = int(round(t0 + DETECT_POST_MS * fs / 1000.0)) + 1
    w0c, w1c = max(0, w0), min(T, w1)
    margin = dist
    r0, r1 = max(0, w0c - margin), min(T, w1c + margin)
    X = read_rows_blocked(lfp, idx, r0, r1)
    counts = np.zeros(w1c - w0c, dtype=np.int64)
    n_defl_per_el = np.zeros(len(idx), dtype=np.int64)
    for i in range(len(idx)):
        if not np.isfinite(rsd[i]) or rsd[i] <= 0:
            continue
        pk, _ = find_peaks(-X[i].astype(np.float64), height=-thr[i], distance=dist)
        pk = pk + r0
        pk = pk[(pk >= w0c) & (pk < w1c)]
        n_defl_per_el[i] = pk.size
        counts[pk - w0c] += 1
    # contiguous runs of non-zero counts
    nz = counts > 0
    edges = np.diff(np.concatenate([[0], nz.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1); ends = np.flatnonzero(edges == -1) - 1   # inclusive, window-relative
    runs = [dict(start=int(s + w0c), end=int(e + w0c), size=int(counts[s:e + 1].sum()),
                 centre=int((s + e) // 2 + w0c)) for s, e in zip(starts, ends)]
    return dict(counts=counts, win_start=w0c, win_end=w1c, runs=runs, med=med, rsd=rsd,
                n_defl_per_el=n_defl_per_el, window_clipped=(w0c != w0 or w1c != w1))


def select_events(runs: List[dict], percentiles=PERCENTILES) -> List[dict]:
    """Runs closest in size to the 20/50/80th percentiles (ties -> earlier run)."""
    if not runs:
        return []
    sizes = np.array([r["size"] for r in runs], dtype=float)
    targets = np.percentile(sizes, percentiles)
    chosen, used = [], set()
    distinct = len(runs) >= len(percentiles)
    for p, tgt in zip(percentiles, targets):
        d = np.abs(sizes - tgt)
        if distinct:
            d = d.copy(); d[list(used)] = np.inf
        j = int(np.argmin(d))         # argmin returns the first (earliest) minimum
        used.add(j)
        ev = dict(runs[j]); ev.update(percentile_target=float(p), size_target=float(tgt), run_index=j)
        chosen.append(ev)
    if not distinct:
        warnings.warn(f"only {len(runs)} population event(s): some events serve several percentile targets")
    return chosen


# =============================================================================
# STEP 4: epoch extraction
# =============================================================================

def epoch_time_axis() -> np.ndarray:
    n = int(round((EPOCH_PRE_MS + EPOCH_POST_MS) * FS_EPOCH / 1000.0)) + 1
    return -EPOCH_PRE_MS + np.arange(n) * (1000.0 / FS_EPOCH)


def read_roi_rows(dset: h5py.Dataset, box: Tuple[int, int, int, int], W: int, s0: int, s1: int) -> np.ndarray:
    """[16, s1-s0] for the 4x4 box (row-major), NaN outside the recording."""
    r0, r1, c0, c1 = box
    T = dset.shape[1]
    out = np.full(((r1 - r0) * (c1 - c0), s1 - s0), np.nan, dtype=np.float32)
    a, b = max(0, s0), min(T, s1)
    if b > a:
        for k, r in enumerate(range(r0, r1)):
            out[k * (c1 - c0):(k + 1) * (c1 - c0), a - s0:b - s0] = dset[r * W + c0:r * W + c1, a:b]
    return out


def extract_epoch(dset: h5py.Dataset, box, W: int, centre: int, fs: float) -> np.ndarray:
    """ROI x time epoch at FS_EPOCH: block-average of q = fs/FS_EPOCH samples centred on each tick."""
    q = fs / FS_EPOCH
    if abs(q - round(q)) > 1e-6 or q < 1:
        raise ValueError(f"fs={fs} is not an integer multiple of {FS_EPOCH} Hz")
    q = int(round(q))
    n = len(epoch_time_axis())
    s0 = centre - int(round(EPOCH_PRE_MS * fs / 1000.0)) - q // 2
    X = read_roi_rows(dset, box, W, s0, s0 + n * q)
    if q == 1:
        return X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(X.reshape(X.shape[0], n, q), axis=2).astype(np.float32)


# =============================================================================
# QC FIGURE
# =============================================================================

def qc_figure(det: dict, events: List[dict], fs: float, t0: int, title: str, out_base: Path):
    t_ms = (np.arange(det["win_start"], det["win_end"]) ) * 1000.0 / fs
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(t_ms, det["counts"], color="0.25", lw=0.6, label="deflections per sample")
    cols = {20.0: "tab:blue", 50.0: "tab:green", 80.0: "tab:red"}
    for ev in events:
        c = cols.get(ev["percentile_target"], "tab:orange")
        ax.axvspan(ev["start"] * 1000.0 / fs, (ev["end"] + 1) * 1000.0 / fs, color=c, alpha=0.25, lw=0)
        ax.axvline(ev["centre"] * 1000.0 / fs, color=c, lw=1.0,
                   label=f"p{ev['percentile_target']:.0f}: size {ev['size']} (target {ev['size_target']:.1f})")
    ax.axvline(t0 * 1000.0 / fs, color="k", ls="--", lw=0.8, label="t0 (max mean |CSD|)")
    ax.set_xlabel("time from recording start (ms)"); ax.set_ylabel("count")
    ax.set_title(f"{title}: {len(det['runs'])} population events", fontsize=9)
    ax.legend(fontsize=6, loc="upper right", frameon=False)
    save_fig(fig, out_base)


# =============================================================================
# MAIN
# =============================================================================

def process_recording(h5_path: Path, group: str, args, mask_dir: Path, roi_dir: Path, out_root: Path,
                      results_dir: Path, counters: Dict[Tuple[str, str, str], int]) -> List[dict]:
    rows = []
    with h5py.File(h5_path, "r") as f:
        fs, H, W = read_meta(f)
        lfp, csd = f[LFP_KEY], f[CSD_KEY]
        mask, slice_id, mask_src = load_mask(f, h5_path, mask_dir, args.mask, H, W)
        roi_csv = Path(args.roi_csv) if args.roi_csv else roi_dir / f"{slice_id}_roi_definitions.csv"
        if not roi_csv.exists():
            print(f"[SKIP] {h5_path.name}: ROI CSV not found: {roi_csv}")
            return rows
        rois = load_roi_definitions(roi_csv)
        cond = parse_condition(h5_path.name)
        idx_inner = inner_mask_idx(mask)
        idx_mask = np.flatnonzero(mask.ravel())
        print(f"[REC] {group}/{h5_path.name}: fs={fs:g} Hz, T={lfp.shape[1]}, slice={slice_id}, cond={cond}, "
              f"mask={mask.sum()} el ({mask_src}), {len(rois)} ROIs")
        if args.dry_run:
            for r in rois.itertuples():
                print(f"       would write {out_root / group / r.roi_name / cond}/event_<k>_<t0ms>ms/ x3")
            return rows

        t0, _ = find_t0_chunked(csd, idx_inner, fs)
        det = detect_population_events(lfp, idx_mask, fs, t0)
        events = select_events(det["runs"])
        print(f"       t0 = {t0} samples ({t0 * 1000 / fs:.1f} ms); {len(det['runs'])} population events; "
              f"{det['n_defl_per_el'].sum()} deflections on {int((det['n_defl_per_el'] > 0).sum())} electrodes"
              + ("; detection window clipped to recording" if det["window_clipped"] else ""))
        if not events:
            print(f"[WARN] {h5_path.name}: no deflections -> no events written")
            return rows
        qc_figure(det, events, fs, t0, f"{group} {h5_path.stem}", results_dir / "qc" / f"{group}__{h5_path.stem}__population_counts")
        t_axis = epoch_time_axis()

        for ev in events:
            c_ms = ev["centre"] * 1000.0 / fs
            for r in rois.itertuples():
                box = (int(r.r0), int(r.r1), int(r.c0), int(r.c1))
                key = (group, r.roi_name, cond)
                k = counters.get(key, 0); counters[key] = k + 1
                ev_dir = out_root / group / r.roi_name / cond / f"event_{k:03d}_{c_ms:.1f}ms"
                ev_dir.mkdir(parents=True, exist_ok=True)
                L = extract_epoch(lfp, box, W, ev["centre"], fs)
                C = extract_epoch(csd, box, W, ev["centre"], fs)
                np.save(ev_dir / "lfp_event_traces.npy", L.astype(np.float32))
                np.save(ev_dir / "csd_event_traces.npy", C.astype(np.float32))
                np.save(ev_dir / "event_time_axis_ms.npy", t_axis.astype(np.float64))
                meta = dict(recording=h5_path.name, slice_id=slice_id, group=group, condition=cond,
                            roi_name=r.roi_name, roi_box=list(box), event_index=k, event_size=int(ev["size"]),
                            percentile_target=ev["percentile_target"], t0_sample=int(ev["centre"]), t0_ms=c_ms,
                            fs_source=fs, fs_epoch=FS_EPOCH, slice_t0_sample=int(t0), slice_t0_ms=t0 * 1000.0 / fs,
                            run_start_sample=int(ev["start"]), run_end_sample=int(ev["end"]), roi_index=int(r.roi_index))
                (ev_dir / "event_meta.json").write_text(json.dumps(meta, indent=2))
                rows.append(dict(group=group, slice_id=slice_id, condition=cond, file=h5_path.name,
                                 t0_sample=int(t0), t0_ms=t0 * 1000.0 / fs, roi_name=r.roi_name,
                                 event_index=k, event_centre_sample=int(ev["centre"]), event_centre_ms=c_ms,
                                 size=int(ev["size"]), percentile_target=ev["percentile_target"],
                                 size_target=ev["size_target"], run_start_ms=ev["start"] * 1000.0 / fs,
                                 run_end_ms=(ev["end"] + 1) * 1000.0 / fs, n_population_events=len(det["runs"]),
                                 event_dir=str(ev_dir.relative_to(out_root))))
    return rows


def build_argparser():
    p = argparse.ArgumentParser(description="Sample three network events (p20/p50/p80 of population-event size) per "
                                            "recording and extract ROI LFP/CSD epochs at 500 Hz.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--in_root", default=str(IN_ROOT), help="processed_2ms root (recursive; group = parent folder)")
    p.add_argument("--mask_dir", default=str(MASK_DIR), help="folder of <slice_id>__contact_mask_64x64.npy")
    p.add_argument("--roi_dir", default=str(ROI_DIR), help="folder of <slice_id>_roi_definitions.csv")
    p.add_argument("--out_root", default=str(OUT_ROOT), help="event_traces output root")
    p.add_argument("--results_dir", default=str(RESULTS_DIR), help="catalog, manifest and QC figures")
    p.add_argument("--mask", default=None, help="override: one mask .npy used for every recording")
    p.add_argument("--roi_csv", default=None, help="override: one ROI CSV used for every recording")
    p.add_argument("--only_contains", default=None, help="process only files whose name contains this substring")
    p.add_argument("--dry_run", action="store_true", help="list what would be produced, write nothing")
    p.add_argument("--seed", type=int, default=0, help="unused (no randomness); kept for house-style consistency")
    return p


def main():
    args = build_argparser().parse_args()
    args.mask = Path(args.mask) if args.mask else None
    in_root, mask_dir, roi_dir = Path(args.in_root), Path(args.mask_dir), Path(args.roi_dir)
    out_root, results_dir = Path(args.out_root), Path(args.results_dir)
    files = sorted(p for p in in_root.rglob("*.h5") if not args.only_contains or args.only_contains in p.name)
    if not files:
        raise SystemExit(f"no .h5 files under {in_root}")
    if not args.dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
    counters: Dict[Tuple[str, str, str], int] = {}
    # existing event folders continue the running index per (group, ROI, condition)
    if out_root.is_dir() and not args.dry_run:
        for d in out_root.glob("*/*/*/event_*_*ms"):
            m = re.search(r"event_(\d+)_(.+?)ms", d.name)
            if m:
                key = (d.parents[2].name, d.parents[1].name, d.parents[0].name)
                counters[key] = max(counters.get(key, 0), int(m.group(1)) + 1)
    rows: List[dict] = []
    for p in files:
        group = p.parent.name if p.parent != in_root else in_root.name
        try:
            rows += process_recording(p, group, args, mask_dir, roi_dir, out_root, results_dir, counters)
        except Exception as e:
            print(f"[ERROR] {p}: {type(e).__name__}: {e}")
    if args.dry_run:
        print(f"[DRY RUN] {len(files)} recording(s) listed; nothing written")
        return
    cat = pd.DataFrame(rows)
    cat.to_csv(results_dir / "event_catalog.csv", index=False)
    params = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple))}
    manifest = dict(script="preprocessing/extract_event_epochs.py", command=" ".join(sys.argv),
                    args={k: str(v) for k, v in vars(args).items()}, parameters=params,
                    n_recordings=len(files), n_event_folders=len(rows),
                    recordings=[str(p) for p in files])
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[DONE] {len(rows)} event folders from {len(files)} recordings -> {out_root}; catalog in {results_dir}")


if __name__ == "__main__":
    main()
