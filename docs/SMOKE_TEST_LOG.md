# Smoke-test log: pre-existing SPACE / lfp-csd-explorer scripts on `data/example_synthetic/`

Date: 2026-10-07. Environment: Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib (Agg), 2 CPU / 7 GB RAM.
Repo root for all SPACE commands: `<repo>` (run from there).
Explorer commands run from `<explorer repo>` with `S=<repo>/data/example_synthetic`, `O=data/example_results`.

Rule followed: **plumbing fixes only** (paths, CLI flags, group/condition labels, schema/time-axis reading, backend, optional imports, numpy-2 deprecations). No analysis parameter, window, estimator or statistic was changed. Every new CLI flag keeps the pre-existing constant as its default.

## Packages installed (`pip install --break-system-packages`)
`powerlaw`, `specparam`, `umap-learn`, `PyWavelets` (needed by `psd_wavelet_metrics.py` Morlet CWT; it silently fell back to a hand-rolled CWT without it), `imageio-ffmpeg` (MP4 encoder for `csd_event_movie.py`; system ffmpeg also present).
Already present: statsmodels, networkx, tifffile, openpyxl, imageio, scikit-image, opencv-python-headless, scikit-learn.

## Synthetic data layout facts that matter
- Groups `CTL`, `5xFAD`; one slice per group (`ctl_s1`, `fad_s1`); conditions `4AP`, `AFDX`.
- H5: fs = 10000 (rolling 2 ms bins), T = 12000 (1.2 s), `/meta.bin_ms = 2.0`, no `decimate_Q` attr.
- Event traces: 3 events per ROI per condition, 16 x 251 @ 500 Hz (-100 .. +400 ms).
- Mask / ROI csv are per slice -> whole-slice scripts that take ONE mask / ONE roi csv are run once per group folder.

---

## Change log (file | what | why)

### SPACE/core/roi_phase_gradient_pgd.py
- Added `matplotlib.use("Agg")` before `pyplot` import. (headless run)
- `ROI_ORDER` keeps `SSP`; added `ROI_ALIASES = {"S1": "SSP"}` and alias mapping in `load_roi_definitions()` so a spec csv with `S1` is recognised as the canonical ROI instead of being reported `[MISSING] SSP`. Box coordinates untouched.

### SPACE/core/psd_wavelet_metrics.py
- Added CLI `--groups CTL,5xFAD` (first = reference) and `--condition <token>`; new `apply_group_override()` re-points `PHENOTYPE_ROOTS / PHENOTYPE_ORDER / PHENOTYPE_DISPLAY / PHENOTYPE_COLORS` and extends `BASELINE_ALIASES`. Defaults (`WT`/`5xFAD`) unchanged when flags are omitted.
- Condition folders: the loader already normalises folder names to lowercase alnum, so the spec's uppercase `4AP` matches `BASELINE_ALIASES` (`{"base","4ap","basal","control","4AP"}`). **This script is baseline-only by design**: it uses only the `4AP` folder and ignores `AFDX` (not changed; see conflicts).

### SPACE/core/electrode_connectivity.py
- Same `--groups` / `--condition` + `apply_group_override()` as above.

### SPACE/core/ctx_hippo_avalanche_kuramoto_compare.py
- Same `--groups` / `--condition` + `apply_group_override()`. Note: some output column names are literally `cohen_d_5xFAD_minus_WT` / `delta_R_5xFAD_minus_C57BL6`; they mean `PHENOTYPE_ORDER[1] - PHENOTYPE_ORDER[0]` (labels only, not changed).

### SPACE/core/avalanche_statistics.py
- Added `matplotlib.use("Agg")`. No other change (CLI already had `--in_dir/--out_root/--mask`).

### SPACE/core/phase_kuramoto.py
- Added `matplotlib.use("Agg")`.
- `get_meta()`: `decimate_Q` default changed `5 -> 1`. Per the schema `/meta.fs` is already the rate of the stored series; the old default silently divided fs by 5 (-> 2 kHz) for files without a `decimate_Q` attr, giving a wrong time axis / Nyquist. Legacy files that carry `decimate_Q` behave as before. With fs = 10 kHz and the Methods decimation by 4, `fs_eff = 2500 Hz` is written to `kuramoto_ALL_summary.json`.
- No subsampling needed: a full 4096 x 12000 file runs in ~10 s.

### SPACE/core/roi_lfp_csd_metrics.py
- Added `--roi_csv <path>`: new `load_rois_from_csv()` reads `roi_index,roi_name,r0,r1,c0,c1` (slice bounds) and returns the same `(rois, roi_names)` structure as the interactive picker, which is then skipped. Without the flag the TIFF picker runs exactly as before.
- Backend: `matplotlib.use("Agg")` when `--roi_csv` is given or no `$DISPLAY` (the picker needs a GUI backend otherwise).
- `np.trapz -> np.trapezoid` (numpy 2.x removal).

### SPACE/utils/pick_representative_recording.py
- Added `matplotlib.use("Agg")`.

### SPACE/pharmacology/spectral_sensitivity_bh_fdr.py
- `extract_space_windows()`: time axis now uses `fs_eff` from the sibling `kuramoto_ALL_summary.json` (`dt = 1/fs_eff`), falling back to the legacy `SPACE_DT_SEC = 0.002` when the json is absent. `kuramoto_ALL.csv` `t_bin` is a sample index at `fs_eff`, not a 2 ms bin, for files produced by the current `phase_kuramoto.py`. Added `import json`. Window/step (5 s / 2.5 s) untouched.

### SPACE/benchmark/roc_scalability_sensitivity.py
- Added `--event_roots a,b` (overrides the `EVENT_ROOTS` list, which still defaults to `data/event_traces/WT`, `.../5xFAD`).
- Added `--quick` (smoke test only): scalability N list `[100,200,500,1000]`, 2 iterations x 5 runs. Without the flag the Methods defaults (`N` up to 10000, 10 x 100) are used. ROC bootstrap / permutation counts are **not** touched by `--quick` (they are bound as function defaults; ROC is fast anyway). The manifest now records `quick_smoke_test`, `event_roots` and the scalability settings used.

### lfp-csd-explorer/csd_event_movie.py
- Graceful degradation: if `imageio.get_writer(... libx264)` raises (no ffmpeg backend), frames are written as `<base>_CSD_event_frames/frame_NNNNNN.png` instead of crashing (verified by simulating a writer failure).
- Reads the event window once (`dset[:, start:end]`) instead of one H5 column per frame (per-column reads on the chunked dataset took > 10 min for 6000 frames).
- Added `--frame_ms` (ms of recording per movie frame). Default `None` = one frame per stored sample, i.e. unchanged behaviour. With 10 kHz rolling files pass `--frame_ms 2` to get the 2 ms frame step the script was written for.

### lfp-csd-explorer/full_slice_3x2_movie.py
- `read_dt_ms()`: prefer `1000/fs` over `bin_ms`. In the current schema `bin_ms = 2.0` is the smoothing width while the sample step is `1000/fs` (0.1 ms for rolling files); the old order produced a 20x-too-long time axis.
- `cv2` import made optional; MP4 encoding wrapped in try/except, PNG frames (always written first) are kept and a warning printed if encoding is unavailable.

### Not modified
`benchmark/welch_psd_recordings.py` (already worked; outputs in `data/example_synthetic/psd_tables` + `results/welch_psd`), `lfp-csd-explorer/csd_snapshot_montage.py`, `roi_hotspot_kymograph.py`, `kuramoto_polar_movie.py`, `inspect_recording.py`.

### Data staging (no code)
Copied the four `kuramoto_ALL.csv` + `kuramoto_ALL_summary.json` from step 6 into
`data/example_synthetic/kuramoto_tables/<recording>/tables/` (folder name = recording base name, which carries the drug token the loaders parse). This is the `data/kuramoto_tables/<condition>/tables/kuramoto_ALL.csv` layout expected by `spectral_sensitivity_bh_fdr.py` and `kuramoto_polar_movie.py`; the real `data/kuramoto_tables/` was left empty.

---

## Working commands (all verified to run to completion)

```bash
cd <repo>
R=data/example_synthetic/results; S=data/example_synthetic

# 1 roi_phase_gradient_pgd.py  (~1 min / group)
for g in CTL 5xFAD; do s=$([ $g = CTL ] && echo ctl_s1 || echo fad_s1)
python core/roi_phase_gradient_pgd.py --data_dir $S/processed_2ms/$g --mask $S/masks/${s}__contact_mask_64x64.npy \
  --roi_csv $S/roi_definitions/${s}_roi_definitions.csv --out_dir $R/roi_phase_gradient_pgd/$g; done

# 2 psd_wavelet_metrics.py  (~40 s)
python core/psd_wavelet_metrics.py --root_dir $S/event_traces --out_dir $R/psd_wavelet_metrics --groups CTL,5xFAD

# 3 electrode_connectivity.py  (~3 min)
python core/electrode_connectivity.py --root_dir $S/event_traces --out_dir $R/electrode_connectivity --groups CTL,5xFAD

# 4 ctx_hippo_avalanche_kuramoto_compare.py  (~30 s)
python core/ctx_hippo_avalanche_kuramoto_compare.py --root_dir $S/event_traces --out_dir $R/ctx_hippo_avalanche_kuramoto_compare --groups CTL,5xFAD

# 5 avalanche_statistics.py  (~1 min / group)
for g in CTL 5xFAD; do s=$([ $g = CTL ] && echo ctl_s1 || echo fad_s1)
python core/avalanche_statistics.py --in_dir $S/processed_2ms/$g --mask $S/masks/${s}__contact_mask_64x64.npy --out_root $R/avalanche_statistics/$g; done

# 6 phase_kuramoto.py  (~10 s / file)
for g in CTL 5xFAD; do s=$([ $g = CTL ] && echo ctl_s1 || echo fad_s1)
python core/phase_kuramoto.py --in_dir $S/processed_2ms/$g --mask $S/masks/${s}__contact_mask_64x64.npy --out_root $R/phase_kuramoto/$g; done
#   stage tables for step 9 / explorer:
for d in $R/phase_kuramoto/*/*/; do n=$(basename $d); mkdir -p $S/kuramoto_tables/$n/tables
  cp $d/tables/kuramoto_ALL.csv $d/tables/kuramoto_ALL_summary.json $S/kuramoto_tables/$n/tables/; done

# 7 roi_lfp_csd_metrics.py  (~7 min / group, non-interactive)
for g in CTL 5xFAD; do s=$([ $g = CTL ] && echo ctl_s1 || echo fad_s1)
python core/roi_lfp_csd_metrics.py --data_dir $S/processed_2ms/$g --mask $S/masks/${s}__contact_mask_64x64.npy \
  --roi_csv $S/roi_definitions/${s}_roi_definitions.csv --genotype-label $g --out_dir $R/roi_lfp_csd_metrics/$g; done

# 8 pick_representative_recording.py  (seconds)
for g in CTL 5xFAD; do s=$([ $g = CTL ] && echo ctl_s1 || echo fad_s1)
python utils/pick_representative_recording.py --in_dir $S/processed_2ms/$g --mask $S/masks/${s}__contact_mask_64x64.npy --out_dir $R/pick_representative_recording/$g; done

# 9a welch_psd_recordings.py (already worked; outputs were present)
python benchmark/welch_psd_recordings.py --in_root $S/processed_2ms --mask_dir $S/masks --out_dir $S/psd_tables --figs_dir $R/welch_psd
# 9b spectral_sensitivity_bh_fdr.py  (seconds)
python pharmacology/spectral_sensitivity_bh_fdr.py --space_root $S/kuramoto_tables --psd_root $S/psd_tables \
  --out_dir $R/spectral_sensitivity_bh_fdr --pair_a 4ap --pair_b afdx

# 10 roc_scalability_sensitivity.py  (~30 s with --quick; drop --quick for the Methods benchmark)
python benchmark/roc_scalability_sensitivity.py --space_root $R/ctx_hippo_avalanche_kuramoto_compare \
  --lfp_csd_root $R/psd_wavelet_metrics --event_roots $S/event_traces/CTL,$S/event_traces/5xFAD \
  --out_dir $R/roc_scalability_sensitivity --quick

# 11 explorer
cd <explorer repo>
S=<repo>/data/example_synthetic; O=data/example_results
for g in CTL 5xFAD; do s=$([ $g = CTL ] && echo ctl_s1 || echo fad_s1)
python csd_snapshot_montage.py --in_dir $S/processed_2ms/$g --out_dir $O/csd_snapshot_montage/$g --mask $S/masks/${s}__contact_mask_64x64.npy
python roi_hotspot_kymograph.py --in_dir $S/processed_2ms/$g --out_dir $O/roi_hotspot_kymograph/$g --mask $S/masks/${s}__contact_mask_64x64.npy
python csd_event_movie.py --in_dir $S/processed_2ms/$g --out_dir $O/csd_event_movie/$g --mask $S/masks/${s}__contact_mask_64x64.npy --frame_ms 2
done
python kuramoto_polar_movie.py --root_dir $S/kuramoto_tables --out_root $O/kuramoto_polar_movie
python inspect_recording.py $S/processed_2ms/CTL/ctl_s1_cor_4ap_2ms.h5 > $O/inspect_recording/ctl_s1_cor_4ap_2ms.txt
for f in CTL/ctl_s1_cor_4ap_2ms CTL/ctl_s1_cor_afdx_2ms 5xFAD/fad_s1_cor_4ap_2ms 5xFAD/fad_s1_cor_afdx_2ms; do
  g=${f%%/*}; n=${f##*/}; s=$([ $g = CTL ] && echo ctl_s1 || echo fad_s1)
  python full_slice_3x2_movie.py --input $S/processed_2ms/$f.h5 --out $O/full_slice_3x2_movie/$n \
    --mask $S/masks/${s}__contact_mask_64x64.npy --start_sec 0.3 --duration_sec 0.5 --fps 10 --baseline_sec 0.1; done
```

---

## Status table

| script | status | notes |
|---|---|---|
| core/roi_phase_gradient_pgd.py | fixed | Agg backend; `S1`->`SSP` alias. Runs per group (one mask/roi csv). |
| core/psd_wavelet_metrics.py | fixed | `--groups/--condition`; PyWavelets installed. Baseline(4AP)-only by design. |
| core/electrode_connectivity.py | fixed | `--groups/--condition`. Cosmetic "axes collapsed" layout warnings on tiny N. |
| core/ctx_hippo_avalanche_kuramoto_compare.py | fixed | `--groups/--condition`. Polyfit RankWarnings (3 events/ROI). |
| core/avalanche_statistics.py | fixed | Agg backend only. baseline/post windows partly outside the 1.2 s file -> few avalanches, powerlaw edge warnings. |
| core/phase_kuramoto.py | fixed | Agg; `decimate_Q` default 1 (use stored fs). fs_eff=2500 Hz. ~10 s/file, no subsampling needed. |
| core/roi_lfp_csd_metrics.py | fixed | `--roi_csv` skips TIFF picker; `np.trapz`->`np.trapezoid`; Agg. ~7 min/group. |
| utils/pick_representative_recording.py | fixed | Agg backend only. |
| benchmark/welch_psd_recordings.py | ok | unchanged. |
| pharmacology/spectral_sensitivity_bh_fdr.py | fixed | time axis from `fs_eff`; SPACE family yields 0 windows on 1.2 s files (see conflicts); Conventional stats + bar plot produced. |
| benchmark/roc_scalability_sensitivity.py | fixed | `--event_roots`, `--quick`. Only 1 dataset/group -> grouped CV falls back to stratified row CV (warning written in `roc_results.csv`). |
| explorer/csd_snapshot_montage.py | ok | unchanged. |
| explorer/roi_hotspot_kymograph.py | ok | unchanged. |
| explorer/csd_event_movie.py | fixed | PNG fallback; block read; `--frame_ms`. |
| explorer/kuramoto_polar_movie.py | ok | unchanged; MP4 via matplotlib FFMpegWriter (ffmpeg present). 19 frames/recording after auto onset. |
| explorer/inspect_recording.py | ok | unchanged. |
| explorer/full_slice_3x2_movie.py | fixed | dt from `fs`; cv2 optional / MP4 failure tolerated. |

## Data-assumption conflicts with the spec layout (to document)

1. **`spectral_sensitivity_bh_fdr.py` SPACE windows (5 s window / 2.5 s step)** need recordings >= 5 s; the 1.2 s synthetic files give 0 SPACE samples, so the SPACE best-metric heatmap / scatter are skipped (only Conventional PSD stats are produced). Also, `kuramoto_ALL.csv` `t_bin` is a sample index at `fs_eff` (2500 Hz for 10 kHz input); the script previously assumed 2 ms bins (now read from the summary json).
2. **`psd_wavelet_metrics.py`, `electrode_connectivity.py`, `ctx_hippo_avalanche_kuramoto_compare.py` are baseline-only**: they load exactly one condition folder per ROI, the first whose name is in `BASELINE_ALIASES` (`4AP`). `AFDX/BIC/CNQX/ML297` event folders are ignored. `--condition` can swap which folder is treated as baseline, but there is no per-condition loop.
3. **One mask / one ROI csv per run** (`roi_phase_gradient_pgd.py`, `roi_lfp_csd_metrics.py`, `avalanche_statistics.py`, `phase_kuramoto.py`, `pick_representative_recording.py`): they take a single `--mask` (and `--roi_csv`) for a whole folder, so with per-slice masks they must be run per slice/group folder; they do not auto-match `<slice_id>__contact_mask_64x64.npy` by filename prefix as the spec suggests. Output sub-folders are keyed by drug name only (`4AP`, `AF-DX`), so running two slices into one `--out_dir` would overwrite.
4. **Drug label spelling**: `roi_phase_gradient_pgd.py` / `roi_lfp_csd_metrics.py` write `AF-DX` folders/labels, the benchmark scripts use lowercase `afdx`, the spec token is `AFDX`.
5. **`avalanche_statistics.py` windows** (`baseline -1000..-100 ms`, `post 400..1500 ms` relative to t0, `T0_SEARCH_MS = 2000`) and `pick_representative_recording.py` / explorer `SEARCH_MS = 2000` exceed the 1.2 s synthetic recordings (they clip gracefully but the baseline/post windows are mostly empty).
6. **`roc_scalability_sensitivity.py`** needs >= 2 datasets per class for grouped CV and >= 2 dataset-level means per class for the sensitivity t-tests; with one slice per group it falls back to row-level CV (flagged) and the sensitivity p-values are NaN.
7. **`phase_kuramoto.py` legacy `decimate_Q`**: files written by the old preprocessing carried `decimate_Q`; spec files store the final `fs` directly. The script now trusts `fs` unless `decimate_Q` is present.
8. **`full_slice_3x2_movie.py` / `csd_event_movie.py`** were written for 2 ms-step streams; on 10 kHz rolling files the native frame step is 0.1 ms (use `--frame_ms 2`, `--avg_ms`, short `--duration_sec`).
9. `psd_wavelet_metrics.py` prefers `pywt` for the Morlet CWT and silently falls back to its own implementation when PyWavelets is missing — install it to match the intended estimator.
