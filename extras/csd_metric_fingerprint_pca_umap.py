# -*- coding: utf-8 -*-
"""
drug_zscore_fingerprint.py -- SPACE framework: drug-normalized Z-score fingerprint maps.

Manuscript figure(s): Figure 6A ("Median Z (LFP)" per-drug spatial fingerprint maps,
CTL vs 5xFAD), plus the underlying per-electrode t/p-maps used to compare each
pharmacology condition against the 4-AP baseline.

What it computes:
    A mask-safe pipeline over the per-electrode CSD and current-flow metric maps
    (``csd_std_map``, ``csd_min_map``, ``csd_max_map``, ``sink_thr_map``,
    ``source_thr_map``, ``ci_map``, ``ai_map``) already stored in each recording's 2 ms
    HDF5 file:
        1. clean_map  -- NaN-fill, percentile clip within the tissue mask, remove
           row/column line artifacts, mask before and after Gaussian smoothing so
           smoothing never bleeds signal across the tissue-contact boundary.
        2. normalize_map -- per-metric Z-score (mean/std computed over the in-mask
           pixels only, NaN outside the mask).
        3. combined_Z -- the composite per-electrode Z-score map used for Figure 6A,
           built by averaging the per-metric Z-scores at each electrode (the
           manuscript figure calls this composite map "Median Z (LFP)").
        4. Per-electrode Welch's t-test (``scipy.stats.ttest_ind``) of each
           pharmacology condition's per-metric Z-score vector against the 4-AP
           baseline's, producing a t-map and p-map per condition per slice.
        5. Group-level summaries across all slices/conditions: PCA, UMAP
           (``umap-learn``), and a drug x drug correlation matrix of the combined-Z
           feature vectors (restricted to electrodes in the mask shared by every slice).

    Run once per genotype/phenotype group (e.g. once for CTL, once for 5xFAD); running
    it for both groups and placing the resulting per-drug "Median Z (LFP)" maps side by
    side is what produces the CTL-vs-5xFAD comparison in Figure 6A. This is otherwise a
    straight port of the original analysis script -- no changes to the cleaning,
    normalization, combination, statistics, or group-summary math.

Inputs / outputs (see data/README.md for the full schema):
    Input:  one or more genotype/group folders of 2 ms HDF5 recordings under
            ``data/processed_2ms/<group>/`` (named ``<slice_id>_<condition>_2ms.h5``,
            one file per pharmacology condition per slice), plus one tissue-contact mask
            per slice under ``data/masks/`` (named
            ``<slice_id>__contact_mask_64x64.npy``).
    Output: per-slice RAW/Z/COMBINED/DELTA (t-map, p-map) PNGs and a group-level
            PCA/UMAP/correlation summary, written under
            ``data/results/drug_zscore_fingerprint/<group>/`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as nd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from scipy.stats import ttest_ind

# =============================================================================
# CONFIG
# =============================================================================

# --- EDIT THIS --- (or override via CLI flags, see build_argparser) group label ->
# folder of that group's 2 ms HDF5 recordings, following data/processed_2ms/<group>/.
GROUP_DIRS = {
    "CTL": Path("data/processed_2ms/CTL"),
    "5xFAD": Path("data/processed_2ms/5xFAD"),
}

# --- EDIT THIS --- shared folder of per-slice tissue-contact masks
# (<slice_id>__contact_mask_64x64.npy), following data/masks/.
MASK_DIR = Path("data/masks")

OUT_DIR = Path("data/results/drug_zscore_fingerprint")

DRUGS = ["4AP", "Bic", "AFDX", "CNQX", "ML297"]
BASELINE_DRUG = "4AP"

CSD_METRICS = [
    "csd_std_map",
    "csd_min_map",
    "csd_max_map",
    "sink_thr_map",
    "source_thr_map",
]

FLOW_METRICS = [
    "ci_map",
    "ai_map",
]

ALL_METRICS = CSD_METRICS + FLOW_METRICS

RAW_CMAP = "pink_r"
Z_CMAP = "coolwarm"

VMIN, VMAX = -3, 3

# ============================================================
# CSD-specific colormaps (each metric gets its own)
# ============================================================

CMAP_CSD_SINK = "OrRd"     # warm scale for outward current
CMAP_CSD_SOURCE = "PuBu"   # cool scale for inward current
CMAP_CSD_STD = "Purples"   # magnitude-only variability


# ============================================================
# UTILS
# ============================================================

def ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def print_section(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def save_map(arr, cmap, title, out_png, vmin=None, vmax=None):
    """Mask-safe plotting (NaN = white)."""
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')   # critical fix

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="none")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def write_csv(arr, out_csv):
    H, W = arr.shape
    with open(out_csv, "w") as f:
        f.write("i,j,val\n")
        for i in range(H):
            for j in range(W):
                f.write(f"{i},{j},{arr[i, j]}\n")

# ============================================================
# LOADING
# ============================================================

def load_mask(mask_path):
    return np.load(mask_path).astype(bool)


def load_maps_h5(path):
    import h5py
    maps = {}
    with h5py.File(path, "r") as f:
        csd = f["csd"]["metrics"]
        maps["csd_std_map"] = csd["std_map"][:]
        maps["csd_min_map"] = csd["min_map"][:]
        maps["csd_max_map"] = csd["max_map"][:]
        maps["sink_thr_map"] = csd["sink_thr_map"][:]
        maps["source_thr_map"] = csd["source_thr_map"][:]

        flow = f["flow"]
        maps["ci_map"] = flow["ci_map"][:]
        maps["ai_map"] = flow["ai_map"][:]

    return maps


def discover_slices(mask_dir: Path, group_dir: Path):
    """Slice IDs present both as a mask file and as at least one H5 in group_dir."""
    mask_dir = Path(mask_dir)
    group_dir = Path(group_dir)
    h5_names = [p.name.lower() for p in group_dir.glob("*.h5")] if group_dir.exists() else []
    slice_ids = []
    for mp in sorted(mask_dir.glob("*__contact_mask_64x64.npy")):
        slice_id = mp.name.split("__contact_mask_64x64.npy")[0]
        if any(slice_id.lower() in fn for fn in h5_names):
            slice_ids.append(slice_id)
    return slice_ids


def find_h5_for_drug(group_dir: Path, slice_id: str, drug: str):
    """Find the recording for one slice/condition under the flat <group_dir>/*.h5 layout
    (files named <slice_id>_<condition>_2ms.h5, per data/README.md)."""
    for p in sorted(Path(group_dir).glob("*.h5")):
        fn = p.name.lower()
        if slice_id.lower() in fn and drug.lower() in fn:
            return p
    return None

# ============================================================
# ARTIFACT REMOVAL
# ============================================================

def remove_line_artifact(arr):
    arr2 = arr.astype(float)
    # Horizontal streaks
    row_med = np.median(arr2, axis=1)
    arr2 = arr2 - row_med[:, None]
    # Vertical streaks
    col_med = np.median(arr2, axis=0)
    arr2 = arr2 - col_med[None, :]
    return arr2


def gaussian3(arr, sigma=0.8):
    return nd.gaussian_filter(arr, sigma=sigma)


def clean_map(arr, mask):
    """
    Safe cleaning:
        - NaN->0
        - Clip inside mask
        - Remove line artifacts
        - Mask BEFORE smoothing (critical)
        - Smooth
        - Mask AGAIN after smoothing
        - Outside = NaN (white)
    """
    arr = np.where(np.isnan(arr), 0, arr)

    valid = arr[mask]
    if valid.size > 5:
        lo, hi = np.percentile(valid, [0.5, 99.5])
        arr = np.clip(arr, lo, hi)

    arr = remove_line_artifact(arr)

    # ---- critical: prevent smoothing bleed
    arr = arr * mask

    arr_smooth = gaussian3(arr, sigma=0.8)

    # ---- critical: NaN outside mask
    arr_smooth = np.where(mask, arr_smooth, np.nan)

    return arr_smooth

# ============================================================
# NORMALIZATION (mean / std within mask)
# ============================================================

def normalize_map(arr, mask):
    valid = arr[mask]
    valid = valid[~np.isnan(valid)]

    if valid.size < 10:
        return np.where(mask, np.nan, np.nan)

    mu = np.mean(valid)
    sd = np.std(valid)
    sd = max(sd, 1e-9)

    Z = (arr - mu) / sd
    return np.where(mask, Z, np.nan)


def normalize_all(maps, mask):
    Z = {}
    for m in ALL_METRICS:
        Z[m] = normalize_map(maps[m], mask)
    return Z

# ============================================================
# COMBINED Z
# ============================================================

def combined_Z(Z, mask):
    stack = np.stack([Z[m] for m in ALL_METRICS], axis=0)
    comb = np.nanmean(stack, axis=0)
    return np.where(mask, comb, np.nan)

# ============================================================
# PROCESS SINGLE SLICE
# ============================================================

def process_slice(slice_id, group_dir, mask_path, out_root,
                   FV, labels, global_mask, drugs, baseline_drug):

    print_section(f"Processing slice: {slice_id}")

    # dirs
    SL = ensure(Path(out_root) / slice_id)
    RAW_DIR = ensure(SL / "RAW")
    Z_DIR = ensure(SL / "Z")
    COMB_DIR = ensure(SL / "COMBINED")
    DELTA_DIR = ensure(SL / "DELTA")

    mask = load_mask(mask_path)

    baseline_Z = None
    baseline_comb = None

    for drug in drugs:

        # find file
        h5p = find_h5_for_drug(group_dir, slice_id, drug)
        if h5p is None:
            print(f"[WARN] H5 not found for {slice_id} / {drug}")
            continue

        # ---- LOAD MAPS ----
        maps_raw = load_maps_h5(h5p)

        # ---- CLEAN ----
        maps_clean = {}
        for m in ALL_METRICS:
            maps_clean[m] = clean_map(maps_raw[m], mask)

        # ---- SAVE RAW CLEANED ----
        for m in ALL_METRICS:
            # ---- SAVE RAW CSD MAPS WITH SEPARATE COLORMAPS ----
            if "source_thr_map" in maps_clean:
                save_map(
                    maps_clean["source_thr_map"],
                    CMAP_CSD_SOURCE,
                    "",
                    RAW_DIR / f"{drug}_csd_source.png"
                )

            if "sink_thr_map" in maps_clean:
                save_map(
                    maps_clean["sink_thr_map"],
                    CMAP_CSD_SINK,
                    "",
                    RAW_DIR / f"{drug}_csd_sink.png"
                )

            if "std_map" in maps_clean:
                save_map(
                    maps_clean["std_map"],
                    CMAP_CSD_STD,
                    "",
                    RAW_DIR / f"{drug}_csd_std.png"
                )
            else:
                save_map(
                    maps_clean[m],
                    RAW_CMAP,
                    '',
                    RAW_DIR / f"{drug}_{m}.png"
                )

        # ---- NORMALIZE ----
        Z = normalize_all(maps_clean, mask)

        for m in ALL_METRICS:
            save_map(
                Z[m],
                Z_CMAP,
                '',
                Z_DIR / f"{drug}_{m}_Z.png",
                vmin=VMIN, vmax=VMAX
            )

        # ---- COMBINED Z ----
        comb = combined_Z(Z, mask)
        save_map(
            comb, Z_CMAP,
            '',
            COMB_DIR / f"{drug}_combinedZ.png",
            vmin=VMIN, vmax=VMAX
        )

        # ---- BASELINE ----
        if drug == baseline_drug:
            baseline_Z = Z
            baseline_comb = comb
        else:
            if baseline_comb is not None:
                # dZ
                dZ = comb - baseline_comb
                dZ = np.where(mask, dZ, np.nan)

                save_map(
                    dZ, Z_CMAP,
                    '',
                    DELTA_DIR / f"{drug}_deltaZ.png",
                    vmin=VMIN, vmax=VMAX
                )

                # t-map / p-map
                tmap = np.full(mask.shape, np.nan)
                pmap = np.full(mask.shape, np.nan)

                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if not mask[i, j]:
                            continue
                        A = np.array([baseline_Z[m][i, j] for m in ALL_METRICS])
                        B = np.array([Z[m][i, j] for m in ALL_METRICS])
                        valid = (~np.isnan(A)) & (~np.isnan(B))
                        if valid.sum() < 3:
                            continue
                        t, p = ttest_ind(A[valid], B[valid], equal_var=False)
                        tmap[i, j] = t
                        pmap[i, j] = p

                save_map(tmap, "coolwarm", '', DELTA_DIR / f"{drug}_tmap.png")
                save_map(pmap, "viridis", '', DELTA_DIR / f"{drug}_pmap.png")

        # ---- for global PCA ----
        FV.append(comb[global_mask].flatten())
        labels.append(f"{slice_id}_{drug}")

    return FV, labels

# ============================================================
# GLOBAL ANALYSIS
# ============================================================

def global_PCA(FV, labels, out, drugs):
    ensure(out)
    X = np.array(FV)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    PCs = pca.fit_transform(Xs)

    # 2D
    fig, ax = plt.subplots(figsize=(7, 6))
    for d in drugs:
        idx = [i for i, l in enumerate(labels) if d in l]
        ax.scatter(PCs[idx, 0], PCs[idx, 1], s=60, label=d)
    ax.legend()
    fig.savefig(Path(out) / "PCA_2D.png", dpi=300)
    plt.close(fig)


def global_UMAP(FV, labels, out, drugs):
    ensure(out)
    X = np.array(FV)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
    U = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for d in drugs:
        idx = [i for i, l in enumerate(labels) if d in l]
        ax.scatter(U[idx, 0], U[idx, 1], s=60, label=d)
    ax.legend()
    fig.savefig(Path(out) / "UMAP.png", dpi=300)
    plt.close(fig)


def global_corr(FV, labels, out, drugs):
    ensure(out)
    drug_vec = {d: [] for d in drugs}

    for fv, lab in zip(FV, labels):
        for d in drugs:
            if d in lab:
                drug_vec[d].append(fv)

    avg = {d: np.mean(np.stack(v), axis=0) for d, v in drug_vec.items() if len(v) > 0}

    C = np.zeros((len(drugs), len(drugs)))
    for i, d1 in enumerate(drugs):
        for j, d2 in enumerate(drugs):
            if d1 in avg and d2 in avg:
                C[i, j] = np.corrcoef(avg[d1], avg[d2])[0, 1]
            else:
                C[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(C, annot=False, cmap="vlag",
                xticklabels=drugs, yticklabels=drugs,
                vmin=-1, vmax=1, ax=ax)
    fig.savefig(Path(out) / "corr.png", dpi=300)
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================

def run_group(group_label, group_dir, mask_dir, out_root, drugs, baseline_drug):
    group_dir = Path(group_dir)
    out_group = ensure(Path(out_root) / group_label)
    print_section(f"Building global mask for group: {group_label}")

    slice_ids = discover_slices(mask_dir, group_dir)
    if not slice_ids:
        print(f"[WARN] No slices found for group {group_label} in {group_dir} "
              f"(masks searched in {mask_dir})")
        return

    mask_paths = {sid: Path(mask_dir) / f"{sid}__contact_mask_64x64.npy" for sid in slice_ids}
    all_masks = [load_mask(mask_paths[sid]) for sid in slice_ids]
    global_mask = all_masks[0].copy()
    for m in all_masks[1:]:
        global_mask &= m

    FV, labels = [], []

    for slice_id in slice_ids:
        FV, labels = process_slice(
            slice_id, group_dir, mask_paths[slice_id], out_group,
            FV, labels, global_mask, drugs, baseline_drug,
        )

    if len(FV) >= 2:
        G = ensure(out_group / "GLOBAL")
        global_PCA(FV, labels, G / "PCA", drugs)
        global_UMAP(FV, labels, G / "UMAP", drugs)
        global_corr(FV, labels, G / "CORR", drugs)
    else:
        print(f"[WARN] Not enough slice/drug feature vectors for group-level PCA/UMAP/corr "
              f"in group {group_label} (need >= 2, got {len(FV)})")

    print_section(f"Group {group_label} done.")


def build_argparser():
    p = argparse.ArgumentParser(
        description="Drug-normalized Z-score fingerprint maps (Figure 6A) and per-condition "
                    "t/p-maps vs the 4-AP baseline."
    )
    p.add_argument(
        "--group_dirs", nargs="+", default=None,
        help="One or more 'label=path' pairs, each a folder of that group's "
             "data/processed_2ms/<group>/ HDF5 recordings (default: the GROUP_DIRS config "
             "at the top of this file, i.e. %s)" % (
                 ", ".join(f"{k}={v}" for k, v in GROUP_DIRS.items())
             ),
    )
    p.add_argument("--mask_dir", default=str(MASK_DIR),
                    help="Folder of <slice_id>__contact_mask_64x64.npy masks (default: %(default)s)")
    p.add_argument("--out_dir", default=str(OUT_DIR),
                    help="Output root folder (default: %(default)s)")
    p.add_argument("--drugs", nargs="+", default=DRUGS,
                    help="Pharmacology condition tokens matched against H5 filenames "
                         "(default: %(default)s)")
    p.add_argument("--baseline_drug", default=BASELINE_DRUG,
                    help="Baseline condition used for the delta-Z / t-map / p-map comparisons "
                         "(default: %(default)s)")
    return p


def main():
    args = build_argparser().parse_args()

    if args.group_dirs:
        group_dirs = {}
        for item in args.group_dirs:
            if "=" not in item:
                raise SystemExit(f"--group_dirs entries must be 'label=path', got: {item}")
            label, path = item.split("=", 1)
            group_dirs[label] = Path(path)
    else:
        group_dirs = GROUP_DIRS

    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    ensure(out_dir)

    for group_label, group_dir in group_dirs.items():
        run_group(group_label, group_dir, mask_dir, out_dir, args.drugs, args.baseline_drug)

    print_section("ALL DONE.")


if __name__ == "__main__":
    main()
