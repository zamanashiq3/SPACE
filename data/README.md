# Data folder layout

This folder is not checked into the repository (see `.gitignore`) except for this
README and `.gitkeep` placeholders that keep the empty subfolder structure under git.
Populate the subfolders below with your own recordings before running the scripts in
`preprocessing/`, `core/`, and `benchmark/`. All scripts accept an override path via a
CLI flag (or a clearly marked `# --- EDIT THIS ---` constant near the top of the file)
if you'd rather keep your data somewhere else.

## `slices.csv`

Written by `run_baseline.py init`: one row per slice — `slice_id, group, brw, photo`. The slice id
is the photo file stem and must be contained in the `.brw` file name; `group` is the genotype
label used as the `processed_2ms/<group>/` subfolder (edit it if the automatic guess is wrong).

## `raw_brw/`

Original 3Brain BioCAM HD-MEA recordings, one `.brw` file per slice/condition (e.g. one
file per pharmacology condition: 4-AP, AFDX, BIC, CNQX, ML297). This is the input to
`preprocessing/brw_to_2ms_h5.py`. File naming is up to you, but the script's built-in
genotype/drug-name parsing looks for substrings like `wt`, `5xfad`/`app`/`ad`, and
`cor_<condition>` in the filename, so names such as `cor_4ap_WT_slice03.brw` are
recognized automatically (anything else still works, just with `genotype="unknown"` /
`drug="unknown"` metadata unless you adjust the parsing).

## `slice_photos/`

Bright-field photos of each slice, taken so the electrode grid is visible in-frame. This
is the input to `preprocessing/tissue_registration_mask.py`, which has you click the 4
corners of the MEA field to register the photo onto the 64x64 electrode grid.

## `masks/`

Output of `preprocessing/tissue_registration_mask.py`: for each slice,
`<slice_id>__contact_mask_64x64.npy` (the boolean 64x64 tissue-contact mask consumed by
essentially every downstream script), plus `<slice_id>__rectified_grid.png`,
`<slice_id>__contact_overlay.png`, `<slice_id>__contact_mask_64x64.png`, and
`<slice_id>__contact_manifest.json` (click coordinates, homography, thresholds used).

## `processed_2ms/`

Output of `preprocessing/brw_to_2ms_h5.py`: one HDF5 file per recording, named
`<slice_id>_<condition>_2ms.h5`, containing the 2 ms-binned LFP (`/lfp/ts_2ms_full`) and
CSD (`/csd/ts_2ms_full`) time series plus summary metric maps and metadata attributes.
Several downstream scripts (`core/roi_lfp_csd_metrics.py`,
`core/psd_wavelet_metrics.py`'s siblings, `utils/pick_representative_recording.py`)
expect these files organized into **per-group subfolders**, e.g.:

```
processed_2ms/
  WT/
    cor_4ap_WT_slice03_2ms.h5
    ...
  5xFAD/
    cor_4ap_5xFAD_slice07_2ms.h5
    ...
```

Use whatever group labels match your study design (e.g. `WT` / `5xFAD`, or
`APPNL` / `APPNLGF`) -- the group-dict configs in the phenotype-comparison scripts
(`PHENOTYPE_ROOTS` in `core/psd_wavelet_metrics.py`, `core/electrode_connectivity.py`,
and `core/ctx_hippo_avalanche_kuramoto_compare.py`) are keyed by these same labels, so
keep the folder names and the dict keys consistent.

## `roi_definitions/`

Per-slice CSV files defining anatomical ROI boxes on the 64x64 electrode grid, with
columns `roi_index, roi_name, r0, r1, c0, c1` (row/column bounds of each ROI box). ROI
names follow the cortex/hippocampus convention used throughout the pipeline: `M1`, `M2`,
`S1` (also written `SSP` in some tables), `CA1`, `CA3`, `DG`. These files are produced by
the manual TIFF-aligned ROI picker in `core/roi_lfp_csd_metrics.py` (which writes
`roi_definitions.csv` into its output folder) and consumed by
`core/roi_phase_gradient_pgd.py`. Copy or symlink the CSV(s) you want to reuse into this
folder, one per slice (or a shared one if your ROI boxes are consistent across slices).

## `event_traces/`

Per-ROI, per-condition event epochs **produced by `preprocessing/extract_event_epochs.py`**
(Methods, "Event alignment and analysis windows": three events per recording at the 20th/50th/80th
size percentiles, −100..400 ms, stored at 500 Hz) and consumed by `core/psd_wavelet_metrics.py`,
`core/electrode_connectivity.py`, `core/ctx_hippo_avalanche_kuramoto_compare.py` and
`benchmark/roc_scalability_sensitivity.py`:

```
event_traces/
  <group>/                      # genotype label = processed_2ms/<group>
    <ROI>/                      # M1, M2, S1, CA1, CA3, DG
      <condition>/              # 4AP, AFDX, BIC, CNQX, ML297
        event_000_412.3ms/
          lfp_event_traces.npy       # float32 [16, 251]  ROI electrodes (row-major) x time
          csd_event_traces.npy       # float32 [16, 251]
          event_time_axis_ms.npy     # -100, -98, ..., 400 ms relative to the event centre
          event_meta.json            # recording, slice_id, event size, percentile target, t0 ...
```

`data/results/event_epochs/event_catalog.csv` lists every selected event (slice, condition,
size, percentile target, onset) and is the key for tracing a feature row back to a recording.

## `kuramoto_tables/`

Per-recording/condition Kuramoto summary tables, produced by `core/phase_kuramoto.py`
and consumed by `benchmark/spectral_sensitivity_bh_fdr.py` (and by the companion
`lfp-csd-explorer` repo's `kuramoto_polar_movie.py`, which documents this same folder).
Expected layout per condition:

```
kuramoto_tables/<condition_name>/tables/kuramoto_ALL.csv
kuramoto_tables/<condition_name>/tables/kuramoto_ALL_summary.json
```

`kuramoto_ALL.csv` has columns `t_bin, R, Phi_rad, dPhi_dt_rad_per_s` (one row per
Kuramoto time bin, 2 ms per bin); the summary JSON carries at least `fs_eff`, the
effective sampling rate of that table.

## `psd_tables/`

Per-recording Welch power-spectral-density tables, produced by
`benchmark/welch_psd_recordings.py` (a Welch-PSD pass over the `/lfp/ts_2ms_full`
dataset of every recording under `data/processed_2ms/`, tissue-contact electrodes only;
any other `scipy.signal.welch`-based extraction producing the layout below will also do)
and consumed by `benchmark/spectral_sensitivity_bh_fdr.py`.
Expected file naming: `*_psd.npz` or `*_extracted_psd.npz`, one file per recording, each
containing at least:

```
freqs   -- 1D array of frequency bins (Hz)
psd     -- array of shape (n_electrodes, n_freqs), or (n_freqs,) for a single channel
```

## `example_synthetic/`

Created by `examples/make_synthetic_example.py` / `examples/run_example.sh`: a self-contained
copy of this whole layout (masks, roi_definitions, processed_2ms, event_traces, psd_tables,
kuramoto_tables, results) filled with **synthetic** signals, for smoke-testing the pipeline.
Gitignored; delete it at any time.

## `results/`

All pipeline outputs -- figures, tables, CSVs, NPZ/HDF5 intermediate files -- land here
by default. This entire folder is gitignored: everything in it is regenerated by
re-running the scripts, so nothing here needs to be (or should be) committed.
