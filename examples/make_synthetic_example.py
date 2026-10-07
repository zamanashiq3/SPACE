# -*- coding: utf-8 -*-
"""
make_synthetic_example.py -- SPACE framework: generate a small SYNTHETIC example dataset.

Manuscript Methods section: "Signal preprocessing", "Current source density" (schema only;
no biological content is modelled).
Manuscript figure(s): none (test fixture / smoke-test data for the pipeline).

!!! The data written by this script are ENTIRELY SYNTHETIC. They exist only so that the
!!! repository can be exercised end to end without raw HD-MEA recordings. Nothing in them
!!! supports any biological claim; group/drug differences are arbitrary scalar factors.

Inputs:
    none (everything is generated from a seeded RNG).

Outputs (under --out_root, default data/example_synthetic/):
    masks/<slice>__contact_mask_64x64.npy            bool [64,64] blob covering ~60% of grid
    roi_definitions/<slice>_roi_definitions.csv      6 x 4x4 ROIs (M1, M2, S1, CA1, CA3, DG)
    processed_2ms/<group>/<slice>_cor_<drug>_2ms.h5  exact schema of preprocessing/brw_to_2ms_h5.py
    SYNTHETIC_README.txt, manifest.json

Signal model (per recording):
    LFP(t, r, c) = background + event(t, r, c), where background is AR(1) "1/f-ish" noise
    (soft-clipped at 2.5 SD so that noise alone rarely crosses a 3-robust-SD threshold, then
    2 ms moving-averaged like the real processed recordings) plus
    a weak 20 Hz oscillation on every electrode, and the event is a Gaussian-shaped negative
    deflection (plus a burst of 20 Hz) that starts at ~0.4 s and travels across the mask with
    a slice-specific direction; it is followed by a few smaller after-deflections so that the
    population-event size distribution has small, typical and large events. Electrodes outside
    the mask carry noise only. CSD is computed from the LFP with ``vectorized_csd`` imported
    from preprocessing/brw_to_2ms_h5.py (Gaussian sigma = 1 electrode, 5-point Laplacian,
    -lap/dx^2, outer ring = 0), and the /lfp/metrics and /csd/metrics maps with the same
    streaming ``init_stats``/``update_stats`` helpers of that script.

Implementation notes:
    - Data are generated and written in time chunks (default 2,000 samples) so that RAM stays
      well below the size of the file; the AR(1) noise state is carried across chunks with
      scipy.signal.lfilter so the background is continuous.
    - Default --fs 10000 (rolling mode, 0.1 ms sampling) and --duration_s 1.5 give
      4096 x 15000 float32 = ~245 MB per dataset, x2 (LFP+CSD) per file (less after lzf).
      ``--fs 500 --duration_s 20`` is the light option (non-overlap mode, ~330 MB per file; a 3 s
      recording is ~50 MB). lzf barely compresses the noise, so expect close to the raw size.
    - /mask is written as uint8 (as brw_to_2ms_h5.py does); readers cast to bool.
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import h5py
from scipy.signal import lfilter

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
OUT_ROOT = REPO_ROOT / "data" / "example_synthetic"
GROUPS: Dict[str, List[str]] = {"CTL": ["ctl_s1", "ctl_s2"], "5xFAD": ["fad_s1", "fad_s2"]}
DRUGS = ["4ap", "afdx", "bic", "cnqx", "ml297"]
FS = 10000.0          # Hz; 10000 -> "rolling" mode, 500 -> "nonoverlap" mode
DURATION_S = 1.5
H, W = 64, 64
DX_UM = 21.0
TIME_CHUNK = 2000
SEED = 0
ROI_NAMES = ["M1", "M2", "S1", "CA1", "CA3", "DG"]
# --- END EDIT ---

# arbitrary scalar factors (SYNTHETIC; no biological meaning)
GROUP_AMP = {"CTL": 1.0, "5xFAD": 0.8}
DRUG_AMP = {"4ap": 1.0, "afdx": 0.9, "bic": 1.15, "cnqx": 0.5, "ml297": 0.75}
EVENT_START_S = 0.4


def load_brw_helpers():
    """Import vectorized_csd / init_stats / update_stats from preprocessing/brw_to_2ms_h5.py."""
    src = REPO_ROOT / "preprocessing" / "brw_to_2ms_h5.py"
    spec = importlib.util.spec_from_file_location("brw_to_2ms_h5", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:  # e.g. tqdm missing
        raise SystemExit(f"Cannot import {src}: {e}. Install requirements.txt first.")
    return mod.vectorized_csd, mod.init_stats, mod.update_stats


# =============================================================================
# MASK / ROI
# =============================================================================

def make_blob_mask(rng: np.random.Generator, H: int = 64, W: int = 64, target_frac: float = 0.6) -> np.ndarray:
    """Irregular ellipse-like blob covering ~target_frac of the grid (bool [H,W])."""
    rr, cc = np.mgrid[0:H, 0:W]
    cy, cx = H / 2 - 0.5 + rng.uniform(-2, 2), W / 2 - 0.5 + rng.uniform(-2, 2)
    theta = np.arctan2(rr - cy, cc - cx)
    wobble = 1.0 + 0.08 * np.sin(3 * theta + rng.uniform(0, 2 * np.pi)) + 0.05 * np.cos(5 * theta + rng.uniform(0, 2 * np.pi))
    ar = rng.uniform(0.8, 1.25)
    # scale radius until the area fraction matches
    lo, hi = 10.0, 60.0
    for _ in range(40):
        rad = 0.5 * (lo + hi)
        m = ((rr - cy) / (rad * ar)) ** 2 + ((cc - cx) / (rad / ar)) ** 2 <= wobble ** 2
        if m.mean() < target_frac:
            lo = rad
        else:
            hi = rad
    return m


def pick_rois(mask: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Six 4x4 ROI boxes (M1, M2, S1 upper half; CA1, CA3, DG lower half) fully inside the mask."""
    H_, W_ = mask.shape
    rows_in = np.where(mask.any(axis=1))[0]
    cols_in = np.where(mask.any(axis=0))[0]
    r_lo, r_hi = rows_in.min(), rows_in.max()
    c_lo, c_hi = cols_in.min(), cols_in.max()
    rc = 0.5 * (r_lo + r_hi)
    cands = {
        "M1": (rc - 0.5 * (rc - r_lo), c_lo + 0.3 * (c_hi - c_lo)),
        "M2": (rc - 0.5 * (rc - r_lo), c_lo + 0.5 * (c_hi - c_lo)),
        "S1": (rc - 0.5 * (rc - r_lo), c_lo + 0.7 * (c_hi - c_lo)),
        "CA1": (rc + 0.5 * (r_hi - rc), c_lo + 0.3 * (c_hi - c_lo)),
        "CA3": (rc + 0.5 * (r_hi - rc), c_lo + 0.5 * (c_hi - c_lo)),
        "DG": (rc + 0.5 * (r_hi - rc), c_lo + 0.7 * (c_hi - c_lo)),
    }
    rows = []
    cen_r, cen_c = rc, 0.5 * (c_lo + c_hi)
    for i, name in enumerate(ROI_NAMES):
        r, c = cands[name]
        r += rng.uniform(-1, 1); c += rng.uniform(-1, 1)
        # walk toward the mask centre until the 4x4 box is fully inside the mask (and off the outer ring)
        for _ in range(80):
            r0, c0 = int(round(r)) - 2, int(round(c)) - 2
            r0 = min(max(r0, 1), H_ - 5); c0 = min(max(c0, 1), W_ - 5)
            if mask[r0:r0 + 4, c0:c0 + 4].all():
                break
            r += 0.5 * np.sign(cen_r - r); c += 0.5 * np.sign(cen_c - c)
        else:
            raise RuntimeError(f"could not place ROI {name} inside mask")
        rows.append(dict(roi_index=i, roi_name=name, r0=r0, r1=r0 + 4, c0=c0, c1=c0 + 4))
    return pd.DataFrame(rows)


# =============================================================================
# SIGNAL MODEL
# =============================================================================

class SyntheticRecording:
    """Chunk-wise generator of one synthetic LFP recording [T, H, W] (uV)."""

    def __init__(self, mask: np.ndarray, fs: float, amp: float, seed: int, H: int = 64, W: int = 64):
        self.mask = mask.astype(bool)
        self.fs = float(fs)
        self.H, self.W = H, W
        self.rng = np.random.default_rng(seed)
        rr, cc = np.mgrid[0:H, 0:W]
        # propagation direction (slice-specific) and arrival time of the wavefront per electrode
        ang = self.rng.uniform(0, 2 * np.pi)
        proj = (rr * np.sin(ang) + cc * np.cos(ang)).astype(np.float64)
        proj_in = proj[self.mask]
        frac = (proj - proj_in.min()) / max(proj_in.max() - proj_in.min(), 1e-9)
        travel_s = 0.12 + 0.04 * self.rng.uniform()      # time to cross the mask
        self.t_arr = EVENT_START_S + frac * travel_s        # [H, W] s
        self.amp = amp
        # spatial amplitude profile (smooth bump, 1 at centre of mass)
        cm_r, cm_c = rr[self.mask].mean(), cc[self.mask].mean()
        self.profile = np.exp(-(((rr - cm_r) / 28.0) ** 2 + ((cc - cm_c) / 28.0) ** 2)) * self.mask
        # a few smaller after-deflections with partial spatial extent (small/typical/large events)
        self.after = []
        n_after = 5
        for k in range(n_after):
            t_k = EVENT_START_S + travel_s + 0.10 + 0.09 * k + self.rng.uniform(-0.01, 0.01)
            ar, ac = rr[self.mask][self.rng.integers(self.mask.sum())], cc[self.mask][self.rng.integers(self.mask.sum())]
            rad = self.rng.uniform(6, 20)
            sub = np.exp(-(((rr - ar) / rad) ** 2 + ((cc - ac) / rad) ** 2)) * self.mask
            a_k = self.rng.uniform(0.25, 0.7)
            self.after.append((t_k, sub, a_k))
        # AR(1) background state per electrode (1/f-ish)
        self.ar_coef = np.exp(-1.0 / (0.004 * self.fs))      # ~4 ms correlation time
        self.zi = np.zeros((1, H * W))
        self.noise_sd = 0.8                                   # uV, after soft clipping at 2.5 SD
        # 2 ms moving average, as applied to the real recordings (keeps the stored rate)
        self.win = max(1, int(round(0.002 * self.fs)))
        self.zi_ma = np.zeros((self.win - 1, H * W)) if self.win > 1 else None
        self.osc_phase = self.rng.uniform(0, 2 * np.pi, size=(H, W))

    def chunk(self, i0: int, i1: int) -> np.ndarray:
        """Return LFP [i1-i0, H, W] float32 for samples i0..i1 (continuous across calls)."""
        n = i1 - i0
        t = (np.arange(i0, i1) / self.fs)[:, None, None]
        white = self.rng.standard_normal((n, self.H * self.W)) * np.sqrt(1 - self.ar_coef ** 2)
        bg, self.zi = lfilter([1.0], [1.0, -self.ar_coef], white, axis=0, zi=self.zi)   # unit-variance AR(1)
        # soft clip (light tails) so that noise alone does not produce 3-robust-SD minima
        bg = self.noise_sd * 2.5 * np.tanh(bg / 2.5)
        if self.win > 1:
            bg, self.zi_ma = lfilter(np.ones(self.win) / self.win, [1.0], bg, axis=0, zi=self.zi_ma)
        bg = bg.reshape(n, self.H, self.W)
        osc = 4.0 * np.sin(2 * np.pi * 20.0 * t + self.osc_phase[None]) * self.mask[None]
        # main travelling event: Gaussian negative deflection + 20 Hz burst in its wake
        dt = t - self.t_arr[None]
        # asymmetric Gaussian deflection: 8 ms rise, 35 ms decay
        env = np.exp(-0.5 * (dt / np.where(dt < 0, 0.008, 0.035)) ** 2)
        # 20 Hz burst in the wake of the wavefront; phase common to the whole slice (network rhythm)
        burst = np.exp(-np.clip(dt, 0, None) / 0.15) * (dt > 0) * np.sin(2 * np.pi * 20.0 * (t - EVENT_START_S))
        ev = -self.amp * self.profile[None] * (180.0 * env + 40.0 * burst)
        for t_k, sub, a_k in self.after:
            ev += -self.amp * a_k * 120.0 * sub[None] * np.exp(-0.5 * ((t - t_k) / 0.012) ** 2)
        return (bg + osc + ev).astype(np.float32)


# =============================================================================
# H5 WRITER (schema of preprocessing/brw_to_2ms_h5.py)
# =============================================================================

def write_h5(path: Path, rec: SyntheticRecording, mask: np.ndarray, fs: float, T: int, drug: str,
             genotype: str, helpers, time_chunk: int = TIME_CHUNK) -> Tuple[float, float]:
    vectorized_csd, init_stats, update_stats = helpers
    Hh, Ww = mask.shape
    N_CH = Hh * Ww
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "rolling" if fs >= 1000 else "nonoverlap"
    ch_chunk, t_chunk = 256, min(2048, T)
    with h5py.File(path, "w") as f:
        f.create_dataset("/mask", data=mask.astype(np.uint8), compression="lzf")
        g = f.require_group("meta")
        g.attrs.update({
            "source_brw": "SYNTHETIC (examples/make_synthetic_example.py)", "fs": float(fs),
            "win_samples": int(round(0.002 * fs)) if mode == "rolling" else 20, "bin_ms": 2.0,
            "bin_stride_mode": mode, "T_raw": int(T if mode == "rolling" else T * 20), "T2": int(T),
            "H": Hh, "W": Ww, "dx_um": DX_UM, "drug": drug, "genotype": genotype,
            "filter": "none (synthetic data)", "notch_hz": 60.0, "bandpass_hz": (1.0, 350.0),
            "synthetic": True,
        })
        d_lfp = f.create_dataset("/lfp/ts_2ms_full", (N_CH, T), "float32", chunks=(ch_chunk, t_chunk), compression="lzf")
        d_csd = f.create_dataset("/csd/ts_2ms_full", (N_CH, T), "float32", chunks=(ch_chunk, t_chunk), compression="lzf")
        ls, cs = init_stats(N_CH), init_stats(N_CH)
        lp = cp = None
        for i0 in range(0, T, time_chunk):
            i1 = min(i0 + time_chunk, T)
            lfp_tyx = rec.chunk(i0, i1)
            csd_tyx = vectorized_csd(lfp_tyx, dx_um=DX_UM, sigma=1.0)
            lfp_ch_t = lfp_tyx.reshape(i1 - i0, N_CH).T
            csd_ch_t = csd_tyx.reshape(i1 - i0, N_CH).T
            d_lfp[:, i0:i1] = lfp_ch_t
            d_csd[:, i0:i1] = csd_ch_t
            lp = update_stats(lfp_ch_t, ls, lp)
            cp = update_stats(csd_ch_t, cs, cp)
        denom = max(T, 1)
        maps = {
            "/lfp/metrics/mean_map": ls["sum"] / denom,
            "/lfp/metrics/rms_map": np.sqrt(np.maximum(ls["sumsq"] / denom, 0)),
            "/lfp/metrics/p2p_map": ls["max"] - ls["min"],
            "/lfp/metrics/line_map": ls["line"],
            "/lfp/metrics/auc_map": ls["auc"],
            "/csd/metrics/mean_map": cs["sum"] / denom,
            "/csd/metrics/min_map": cs["min"],
            "/csd/metrics/max_map": cs["max"],
            "/csd/metrics/std_map": np.sqrt(np.maximum(cs["sumsq"] / denom - (cs["sum"] / denom) ** 2, 1e-12)),
        }
        for k, v in maps.items():
            f.create_dataset(k, data=v.reshape(Hh, Ww).astype(np.float32), chunks=(Hh, Ww), compression="lzf")
        g.attrs["T2_written"] = int(T)
    return path.stat().st_size / 1e6, float(maps["/lfp/metrics/p2p_map"][mask.ravel()].mean())


# =============================================================================
# MAIN
# =============================================================================

def build_argparser():
    p = argparse.ArgumentParser(
        description="Generate a small SYNTHETIC SPACE example dataset (masks, ROI CSVs, processed_2ms H5). "
                    "Sizes: 4096 x T x 4 B x 2 (LFP+CSD); default 10 kHz x 1.5 s = ~490 MB per file. "
                    "Light option: --fs 500 --duration_s 20 (non-overlap mode, ~330 MB per file; 3 s = ~50 MB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_root", default=str(OUT_ROOT), help="output root")
    p.add_argument("--fs", type=float, default=FS, help="stored sampling rate (Hz): 10000 = rolling mode, 500 = non-overlap")
    p.add_argument("--duration_s", type=float, default=DURATION_S, help="recording duration (s)")
    p.add_argument("--n_slices_per_group", type=int, default=2, help="slices per group (max 2 with the built-in ids)")
    p.add_argument("--drugs", default=",".join(DRUGS), help="comma-separated drug tokens")
    p.add_argument("--groups", default=",".join(GROUPS.keys()), help="comma-separated groups (CTL,5xFAD)")
    p.add_argument("--time_chunk", type=int, default=TIME_CHUNK, help="samples per generation/write chunk")
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main():
    args = build_argparser().parse_args()
    out = Path(args.out_root)
    if not out.is_absolute():
        out = REPO_ROOT / out
    helpers = load_brw_helpers()
    fs = float(args.fs)
    T = int(round(args.duration_s * fs))
    drugs = [d.strip().lower() for d in args.drugs.split(",") if d.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}; known: {list(GROUPS)}")
    if args.n_slices_per_group > 2:
        for g in groups:
            GROUPS[g] = GROUPS[g] + [f"{GROUPS[g][0][:-3]}_s{k}" for k in range(3, args.n_slices_per_group + 1)]

    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "roi_definitions").mkdir(parents=True, exist_ok=True)
    written: List[dict] = []
    print(f"[SYNTHETIC] out_root={out}  fs={fs:g} Hz  T={T} samples ({args.duration_s:g} s)  seed={args.seed}")
    for gi, group in enumerate(groups):
        for si, slice_id in enumerate(GROUPS[group][:args.n_slices_per_group]):
            rng = np.random.default_rng([args.seed, gi, si])
            mask = make_blob_mask(rng, H, W, 0.6)
            np.save(out / "masks" / f"{slice_id}__contact_mask_64x64.npy", mask)
            rois = pick_rois(mask, rng)
            rois.to_csv(out / "roi_definitions" / f"{slice_id}_roi_definitions.csv", index=False)
            print(f"  [{group}/{slice_id}] mask {mask.mean()*100:.1f}% of grid; ROIs: " +
                  ", ".join(f"{r.roi_name}[{r.r0}:{r.r1},{r.c0}:{r.c1}]" for r in rois.itertuples()))
            for di, drug in enumerate(drugs):
                amp = GROUP_AMP.get(group, 1.0) * DRUG_AMP.get(drug, 1.0) * rng.uniform(0.95, 1.05)
                rec = SyntheticRecording(mask, fs, amp, seed=int(rng.integers(2**31)), H=H, W=W)
                h5_path = out / "processed_2ms" / group / f"{slice_id}_cor_{drug}_2ms.h5"
                mb, p2p = write_h5(h5_path, rec, mask, fs, T, drug, group, helpers, args.time_chunk)
                written.append(dict(group=group, slice_id=slice_id, drug=drug, file=str(h5_path), size_MB=round(mb, 1), mean_p2p_uV=round(p2p, 1)))
                print(f"    wrote {h5_path.relative_to(out)}  ({mb:.1f} MB, mean in-mask p2p {p2p:.0f} uV)")

    (out / "SYNTHETIC_README.txt").write_text(
        "All files in this folder are SYNTHETIC, generated by examples/make_synthetic_example.py.\n"
        "They contain no biological data and support no biological claim; they exist only to\n"
        "exercise the SPACE pipeline. Signal model and parameters: see the script docstring\n"
        "and manifest.json.\n")
    manifest = dict(synthetic=True, script="examples/make_synthetic_example.py", command=" ".join(sys.argv),
                    params=vars(args), fs=fs, T=T, groups=groups, drugs=drugs, event_start_s=EVENT_START_S,
                    group_amp=GROUP_AMP, drug_amp=DRUG_AMP, dx_um=DX_UM, files=written)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    tot = sum(w["size_MB"] for w in written)
    print(f"\n[SUMMARY] {len(written)} H5 files ({tot:.0f} MB), {len(list((out/'masks').glob('*.npy')))} masks, "
          f"{len(list((out/'roi_definitions').glob('*.csv')))} ROI CSVs under {out}")
    print("[SUMMARY] Next: python preprocessing/extract_event_epochs.py "
          f"--in_root {out/'processed_2ms'} --mask_dir {out/'masks'} --roi_dir {out/'roi_definitions'} "
          f"--out_root {out/'event_traces'} --results_dir {out/'results'/'event_epochs'}")


if __name__ == "__main__":
    main()
