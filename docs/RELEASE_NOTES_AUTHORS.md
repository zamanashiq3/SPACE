# Author checklist before publishing the repositories

Items found while reconciling the code with `SPACE_Methods_v1` and the Results text.
Each is either a code change made in this release (**done**, please confirm it matches what
was actually run for the paper) or a manuscript/code inconsistency that only you can resolve
(**decide**). Delete this file before the public release.

## A. Code changed to match the Methods — please confirm

| # | Methods statement | What the original code did | Release |
|---|---|---|---|
| A1 | 2 ms moving average advanced one sample at a time; 10 kHz retained; sampling interval 0.1 ms | `260512_optimized_brw_to_2ms_h5.py` defaulted to non-overlapping 2 ms bins (500 Hz); rolling was an option. The older `main_runner.py` + `analysis_writer_2ms_streaming.py` workflow (the `*_analysis_out_full_2ms` folders) was rolling at 10 kHz. | `--bin_stride_mode rolling` is the default (**done**). If some figures were produced from 500 Hz non-overlap files, say so in the Methods or re-run. |
| A2 | Whole-slice Kuramoto: zero-phase Butterworth 5–1,500 Hz, down-sampled ×4 to 2.5 kHz | `kuranoto_all_in_one.py`: `WIDEBAND_HZ = (1.0, 300.0)`, decimate ×4 | `WIDEBAND_HZ = (5.0, 1500.0)` (**done**). Confirm which band the Fig. 5/7 Kuramoto values came from. |
| A3 | Scalability: N = 100 … 10,000, 10 iterations of 100 runs, 300-sample epochs | N = 16 … 4,096, 3 repeats, single runs | Constants and loop updated (**done**); `--quick` for smoke tests. Re-run for the Suppl. Fig. 7B numbers if needed. |
| A4 | Feature sensitivity on **pooled events** with BH correction | Welch's t on animal-level means, no BH | `SENSITIVITY_UNIT = "event"` + BH-adjusted column; ranking uses −log10(P_BH) (**done**). The Methods say "−log10(P) … with BH correction" — state whether the plotted value is the raw or adjusted P. |
| A5 | SPACE features = Kuramoto/phase/propagation + event-window LFP & CSD; conventional = PSD + avalanche; correlation separate | "Avalanche only" set also included the LFP/CSD scalar features; "Conventional combined" included correlation | Families now follow the Methods (**done**). AUC values may change; recheck Suppl. Fig. 7A. |
| A6 | Normalized LFP slope within the first 80 ms | `roi_lfp_csd_metrics.py` used 0–50 ms (its own display CSV); `psd_wavelet_metrics.py` already used 0–80 | Both 0–80 ms (**done**). |
| A7 | Display maps at 0, 250, 500 ms | `roi_lfp_csd_metrics.py` `SNAP_MS = [0, 50, 100, 300, 400, 500]` | Left unchanged (display only); set `SNAP_MS = [0, 250, 500]` on the command line / constant if you want the script to match the figure exactly. |

## B. Written for the release from the Methods (no original script was provided)

| # | Module | Choice that was not fixed by the Methods — please check |
|---|---|---|
| B1 | `preprocessing/extract_event_epochs.py` | Electrode median for the 3-SD rule taken from the same first 500 ms as the MAD; the three picks are forced to be distinct runs; epochs block-averaged (20 samples) to 500 Hz. If you have the original event-extraction script (the one that produced the `*_event_dbscan_archetype` folders), replace this module with it. |
| B2 | `core/regional_csd_dynamics.py` | "Largest event" = global peak of summed \|LFP\| over the whole recording; pooled-region traces are means over the union of ROI electrodes; SampEn uses population SD; MCR counts sign changes of (a − mean). |
| B3 | `core/drug_lfp_zscore_maps.py` | Taken literally, row/column-median and σ = 8 background subtraction remove each map's overall level, so the median Z of every drug map would be ≈ 0 and Fig. 6B could not show drug effects. The release therefore restores the tissue-wide median after cleaning (flat-fielding; `--no_preserve_level` gives the literal version). **Please confirm this is what was done, and consider adding one sentence to the Methods.** "Mean LFP amplitude" is taken as the stored per-electrode RMS map (`--compute_from_ts` gives time-mean \|LFP\|). |
| B4 | `core/condition_stereotypy.py` | "Spatial CSD map" of a condition = per-electrode CSD SD map (`/csd/metrics/std_map`); `--map_key` selects max/min maps instead. State the map used in the Methods. |
| B5 | `core/roi_phase_gradient_pgd.py` (added block) | Propagation direction θ_k = atan2(−⟨g_y⟩, −⟨g_x⟩), circular mean/resultant over 0–500, 0–250, 250–500 ms per ROI and pooled over ROIs ("all ROIs" row = the whole-slice values quoted in Results, e.g. 333.4°, R = 0.98). Confirm the whole-slice value in the paper was pooled this way (over all ROI electrodes and time points) rather than, e.g., over ROI mean directions. |

## C. Manuscript statements with no corresponding code — decide

| # | Where | Issue |
|---|---|---|
| C1 | Abstract, Introduction, Results ("An anatomically registered framework…"), Fig. 1A | The text says non-tissue electrodes are used as a biological null whose **dominant principal components are projected out** of the tissue signal ("image-based artifact suppression", "common-mode noise subtraction"). No script in the code base does this; the Methods only use non-contact electrodes as a *noise reference for comparison* (Suppl. Fig. 1), which `utils/mask_noise_reference.py` now implements. Either add the PCA-projection step to `preprocessing/brw_to_2ms_h5.py` (and the Methods) or soften the text to "tissue segmentation restricts analysis to contact electrodes; background electrodes serve as an empirical noise reference". |
| C2 | Methods, Current source density | Code divides the Laplacian by `dx_um² = 21²` (electrode size) whereas the pitch is 42 µm; harmless constant since σ_c = 1 and CSD is in a.u., but the README now states it. |
| C3 | Methods, Statistics | "GraphPad Prism 23" — Prism versions are 10.x in 2025–26; check. |
| C4 | Code availability | Fill in the two GitHub URLs; the Methods say "an example dataset is distributed with the code" — the release ships a synthetic generator; if you want a *real* example (one slice, one condition), the 10 kHz rolling file is ~1 GB per 2-min recording (500 Hz non-overlap ≈ 50 MB), so a non-overlap example plus the masks/ROI CSV and the derived `event_traces/` (a few MB) would fit a Zenodo record. |
| C5 | Methods, Software | "package versions are listed in the repository" → `requirements-pinned.txt` was built from your conda export (Windows); h5py, opencv, scikit-image, PyWavelets, specparam, powerlaw, statsmodels were not in that export and carry minimum versions only — run `pip freeze` in the analysis environment and paste the exact lines. |
| C6 | Fig. 3H / Fig. 7C–D | Inter-regional coupling in the Methods uses \|CSD\| epochs; `electrode_connectivity.py` computes `lfp`, `csd_abs` and `csd_signed`; make sure the figure used `csd_abs`. |
| C7 | Suppl. Fig. 7 numbering | The earlier README said Suppl. Fig. 8; the manuscript's benchmark figure is Suppl. Fig. 7. Updated. |

## D. Smoke test

`examples/run_example.sh` runs every script on synthetic data (log in `_spec/run_example.log`
of the delivery bundle). Pre-existing scripts needed only plumbing fixes (CLI overrides for
group names, Agg backend, numpy 2 `trapz`, one wrong time-axis default in
`full_slice_3x2_movie.py`, `decimate_Q` default in `phase_kuramoto.py`); the full change log is
in the bundle's `_spec/smoke_test_log.md`.
