# Methods → code crosswalk

Each subsection of the manuscript Methods, the script/function that implements it, and the
constants that hold the stated parameters. Line numbers drift; search for the symbol names.
"Animal-level table" = the CSV that was imported into GraphPad Prism for the inferential
statistics (Welch's *t*, BH-FDR, mixed-effects models), which are not performed in this code.

## Anatomical registration, tissue-contact mask and regions of interest

| Methods statement | Implementation |
|---|---|
| Four corners clicked; planar homography onto a 1024 × 1024 canvas (16 × 16 px per electrode); OpenCV `findHomography` / `warpPerspective` | `preprocessing/tissue_registration_mask.py` → `pick_four_corners()`, `main()` (`--canvas 1024`) |
| Greyscale, contrast stretch 1st–99th percentile | `_robust_gray()` |
| Otsu threshold; morphological opening 7 × 7, closing 9 × 9; largest 8-connected component | `_detect_contact_mask()` (`cv2.THRESH_OTSU`, `MORPH_OPEN np.ones((7,7))`, `MORPH_CLOSE np.ones((9,9))`, `connectedComponentsWithStats(m, 8)`) |
| Electrode is tissue-contact when ≥ 30 % of its cell is tissue → Ω stored as 64 × 64 binary array | `_bin_to_64x64(coverage=0.30)` → `<slice>__contact_mask_64x64.npy` (+ manifest JSON with homography and thresholds) |
| Manual refinement of masks | edit the `.npy` (any array editor); the manifest records the automatic version |
| Electrodes outside Ω as empirical noise reference (Suppl. Fig. 1) | `utils/mask_noise_reference.py` (RMS inside vs outside Ω, example traces) |
| Six 4 × 4 ROIs (M1, M2, S1, CA1, CA3, DG) centred on manually selected positions, stored as row/column bounds | `preprocessing/pick_rois.py` (click once per slice on the rectified photo; `--size 4`) → `data/roi_definitions/<slice>_roi_definitions.csv` (`roi_index, roi_name, r0, r1, c0, c1`, Python slice bounds); the same CSV is passed to every ROI-level script (`core/roi_lfp_csd_metrics.py` keeps its own legacy picker) |
| Pooling Ctx = M1, M2, S1; Hippo = CA1, CA3, DG | `REGIONS` / `ROI_REGION_MAP` constants in every ROI-level script |

## Signal preprocessing

| Methods statement | Implementation (`preprocessing/brw_to_2ms_h5.py`) |
|---|---|
| Raw digital → voltage from bit depth and full-scale voltage in the file | `read_brw_metadata()` |
| 4,000-sample chunks, filter states carried across chunks | `Config.raw_chunk_frames = 4000`; `sosfilt(..., zi=zi)` state carried in the streaming loop |
| 60 Hz IIR notch, Q = 30 | `Config.notch_hz = 60.0`, `notch_q = 30.0`; `iirnotch` → `tf2sos` |
| Butterworth band-pass 1–350 Hz, order 3, SOS, causal single pass | `Config.lfp_low_hz = 1.0`, `lfp_high_hz = 350.0`, `filter_order = 3`; `butter(..., output="sos")`; `design_streaming_filters()` stacks notch + band-pass; `scipy.signal.sosfilt` |
| 2 ms moving average advanced one sample (10 kHz retained) | `Config.bin_ms = 2.0`, `bin_stride_mode = "rolling"` → `bin_rolling()` (`--bin_stride_mode nonoverlap` gives 500 Hz bins for quick looks only) |
| Electrode × time array V ∈ ℝ^4096×T stored in HDF5 with mask and metadata | `/lfp/ts_2ms_full` [4096, T], `/mask`, `/meta` attrs (`fs`, `bin_ms`, `bin_stride_mode`, `notch_hz`, `bandpass_hz`, `drug`, `genotype`) |

## Current source density

| Methods statement | Implementation |
|---|---|
| Gaussian smoothing across the grid, σ = 1 electrode, reflecting boundaries, truncation 4σ | `vectorized_csd()`: `scipy.ndimage.gaussian_filter(sigma=(0,1,1))` (defaults `mode="reflect"`, `truncate=4.0`) |
| Five-point Laplacian; CSD = −∇²V (σ_c = 1, a.u.); negative = sink | `vectorized_csd()`: `lap = up+down+left+right − 4·centre`; returns `−lap/dx²` (dx = `Config.dx_um`, a constant scale) |
| Outer ring undefined → excluded; electrodes outside Ω excluded | outer ring written as 0 and skipped via `inner_mask_idx()` in every CSD script |
| CSD smoothed with the same 2 ms average | same `bin_rolling()` applied to the CSD stream (`/csd/ts_2ms_full`) |
| Source_R(t), Sink_R(t) = mean rectified positive/negative CSD over the ROI; BI_R(t) = (Source − Sink)/(Source + Sink) | `core/regional_csd_dynamics.py` → `source_sink_bi()`; also `core/psd_wavelet_metrics.py` (`balance = (source − sink)/(source + sink + EPS)`) on the 500 Hz epochs |
| BI time courses over −50..250 ms from traces conditioned with a 5-sample median filter and zero-phase 150 Hz low-pass; time-averaged BI = mean for t ≥ 0 | `regional_csd_dynamics.py`: `MEDIAN_KERNEL = 5`, `LOWPASS_HZ = 150`, `BUTTER_ORDER = 3`, `condition_traces()` (`medfilt` + `filtfilt`), `BI_DISPLAY_MS = (−50, 250)`, `set_metrics()` → `bi_time_avg` |
| Source/Sink AUC by trapezoidal integration; ratio SourceAUC/(SinkAUC + ε) | `set_metrics()` → `source_auc`, `sink_auc`, `source_sink_ratio` (`ANALYSIS_MS = (0, 250)`) |
| ROI values averaged within region per animal | `animal_level.csv` |

## Event alignment and analysis windows

| Methods statement | Implementation |
|---|---|
| Whole-slice onset t₀ = time of maximal mean \|CSD\| over Ω within the first 2 s | `preprocessing/extract_event_epochs.py` → `find_t0_chunked()` (`T0_SEARCH_S = 2.0`); same rule in `core/avalanche_statistics.py`, `core/phase_kuramoto.py`, `utils/pick_representative_recording.py` and the visualizer |
| ROI-level: negative LFP deflections = local minima > 3 robust SD below the electrode median; robust SD = MAD/0.6745 from the first 500 ms; refractory 4 ms | `extract_event_epochs.py`: `THRESH_SD = 3.0`, `NOISE_WINDOW_S = 0.5`, `REFRACTORY_MS = 4.0`, `detect_population_events()` (`find_peaks(−x, height, distance)`) |
| Count per sample across electrodes within −100..400 ms of t₀; contiguous non-zero runs = population events; size = total count | `detect_population_events()` (`DETECT_WINDOW_MS = (−100, 400)`) |
| Three events closest to the 20th/50th/80th size percentiles, centred on the run midpoint | `select_events()` (`PERCENTILES = (20, 50, 80)`) |
| Epochs = ROI LFP and CSD × time, stored at 2 ms resolution (500 Hz) | `extract_epoch()` (block average of fs/500 samples) → `data/event_traces/<group>/<ROI>/<cond>/event_k_<t0>ms/{lfp,csd}_event_traces.npy [16, 251]`, `event_time_axis_ms.npy` (−100..400 ms) |
| Baseline −50..0 ms, analysis 0..250 ms, inclusive; baseline-correct by baseline median; linear NaN interpolation | `BASELINE_WINDOW_MS`, `FEATURE_WINDOW_MS` / `ANALYSIS_WINDOW_MS`, `preprocess_trace()` / `baseline_correct()` + `fill_nan_1d()` in every epoch-level script |
| Display maps at 0, 250, 500 ms; median-5 + zero-phase 3rd-order Butterworth 150 Hz + segment-median subtraction; median over ±0.5 ms (11 samples); electrodes outside Ω blanked | `core/roi_lfp_csd_metrics.py` (`SNAP_MS`, `extract_snapshots_cleaned_chunked()`, `SNAP_HALF_WIN`) |
| ROI kymographs (16 electrodes row-major vs time), Gaussian 0.2 electrodes × 0.4 samples | `roi_lfp_csd_metrics.py` kymograph functions (`gaussian_filter(sigma=(0.2, 0.4))`) |
| Hotspot ROI: mean \|CSD\| over 11 samples at t₀, Gaussian σ = 1, 99th percentile within Ω, largest component, intensity-weighted centroid, 4 × 4 block | `lfp-csd-explorer/roi_hotspot_kymograph.py` (`HOTSPOT_PCT = 99.0`, `GAUSS_SIGMA = 1.0`, `largest_hotspot_centroid()`) |
| Representative recording = smallest max \|z_c\| of per-column median \|CSD\| at peak | `utils/pick_representative_recording.py` (`robust z = (x − med)/(1.4826·MAD)`) |

## LFP amplitude, energy, slope and spatial complexity

| Methods statement | Implementation |
|---|---|
| Regional waveform = ROI mean, baseline-corrected | `core/psd_wavelet_metrics.py` event loader; `core/roi_lfp_csd_metrics.py` `trace_id = roi_mean` rows |
| Amplitude = peak \|deflection\| in 0..250 ms; energy = ∫x² dt (trapezoid, µV²·ms) | `psd_wavelet_metrics.py` LFP feature block (`FEATURE_WINDOW_MS = (0, 250)`); `roi_lfp_csd_metrics.py` (`LFP_METRIC_POST_*`, `LFP_ENERGY_WINDOW_MS`) |
| Normalized slope = max \|dx/dt\| (central differences) in the first 80 ms, relative to peak (% ms⁻¹) | `psd_wavelet_metrics.py` `SLOPE_WINDOW_MS = (0, 80)`; `roi_lfp_csd_metrics.py` `LFP_SLOPE_WINDOW_MS = (0, 80)` |
| Group waveforms interpolated to −50..250 ms; shape-only overlays divided by own peak (display only) | `psd_wavelet_metrics.py` average-trace plots (`TIME_XLIM_MS = (−50, 250)`) |
| Complexity D = (Σλ)²/Σλ² of the electrode covariance (unbiased) of baseline-corrected traces in 0..250 ms smoothed with a 10-sample moving average; leave-one-electrode-out stability | `psd_wavelet_metrics.py` → `compute_lfp_complexity_D()`, `compute_lfp_complexity_features()` (`LFP_COMPLEXITY_SMOOTH_WIDTH = 10`, `LFP_COMPLEXITY_TIME_WINDOW_MS = (0, 250)`) |

## Regional CSD activity and asymmetry; Temporal dynamics of regional CSD

All in `core/regional_csd_dynamics.py`, computed at the stored sampling rate from the
baseline-corrected, otherwise unfiltered CSD of the largest event (onset = peak of summed
\|LFP\| over Ω, `find_t0_chunked()`).

| Methods statement | Implementation |
|---|---|
| A_R(t) = mean \|CSD\| over region; peak in 0..250 ms; total current = ∫₀⁵⁰⁰ A_R dt | `activity()`, `set_metrics()` → `csd_peak`, `total_current_0_500` (`TOTAL_MS = (0, 500)`) |
| RAI = (I_Ctx − I_Hippo)/(I_Ctx + I_Hippo) | `region_metrics.csv` → `rai` |
| Lag-1 autocorrelation of the linearly detrended BI series, 0..500 ms | `lag1_autocorr()` (`scipy.signal.detrend` + `statsmodels.tsa.stattools.acf(nlags=1)`, numpy fallback) |
| Sample entropy of A_R(t), m = 2, r = 0.2 · SD, Chebyshev, no self-matches, nats | `sample_entropy()` (`SAMPEN_M = 2`, `SAMPEN_R_FRAC = 0.2`) |
| Mean crossing rate = crossings of the temporal mean / duration (Hz) | `mean_crossing_rate()` |
| Animal-level: ROI means within region | `animal_level.csv` |

## Phase extraction and Kuramoto synchrony

| Methods statement | Implementation |
|---|---|
| φ_i(t) = arg of the analytic signal (Hilbert); R(t) e^{jΦ(t)} = (1/N) Σ e^{jφ_i} | `core/phase_kuramoto.py` → `wideband_phase_from_ts()` (`scipy.signal.hilbert`), `kuramoto_R_Phi()` |
| Whole slice: zero-phase 3rd-order Butterworth 5–1,500 Hz (SOS, `sosfiltfilt`), down-sample ×4 to 2.5 kHz, Hilbert; N = tissue-contact electrodes | `WIDEBAND_HZ = (5.0, 1500.0)` (upper edge clipped to Nyquist), `TIME_DECIMATE_WIDEBAND = 4`, `masked_nan()` with the contact mask |
| Recording summaries: mean R, SD of R, mean phase velocity = mean \|dΦ/dt\| of the unwrapped mean phase | `kuramoto_ALL.csv` (`t_bin, R, Phi_rad, dPhi_dt_rad_per_s`), `kuramoto_ALL_summary.json`; `phi_velocity()` |
| Coherence map R_i = \|⟨e^{j(φ_i − Φ)}⟩_t\| | `local_coherence_map()` |
| Regional: Hilbert of the baseline-corrected broadband epoch of the 16 ROI electrodes; R(t) across them; regional R = mean over 0..250 ms; ROI values averaged to Ctx/Hippo; regional MPV; PLV_ij | `core/ctx_hippo_avalanche_kuramoto_compare.py` Kuramoto block (`KURAMOTO_BANDPASS = False`, `KURAMOTO_ANALYSIS_WINDOW_MS = (0, 250)`, PLV matrices) |

## Phase gradients and propagation direction

All in `core/roi_phase_gradient_pgd.py`.

| Methods statement | Implementation |
|---|---|
| Epoch −10..500 ms with 100 ms padding discarded | `PRE_MS = 10`, `POST_MS = 500`, `PAD_MS = 100` |
| Generalized phase: mean-centre, Hilbert, unwrap, replace negative increments by linear interpolation, wrap to (−π, π] | `generalized_phase()` |
| Gradient from the unit phasor u = e^{jφ} (central differences interior, one-sided edges; rad/electrode) | `roi_pgd_and_dir()` |
| PGD = \|⟨∇φ⟩\| / ⟨\|∇φ\|⟩; wave fraction = fraction of time with PGD ≥ 0.7 | `roi_pgd_and_dir()`, `PGD_THRESHOLD = 0.7` → `wave_fraction_PGDge0.7` |
| θ_dom = atan2(⟨g_y⟩, ⟨g_x⟩); instantaneous direction opposite to the mean gradient; circular mean and resultant length over 0..500, 0..250, 250..500 ms | `roi_pgd_and_dir()`, summary rows (`dominant_angle_deg`, resultant length), early/late windows |
| Vector maps at 0, 250, 500 ms with the ROI mean vector | `_draw_vec_panel()` (`VEC_SNAP_MS`) |

## Spectral analysis

All in `core/psd_wavelet_metrics.py` on the 500 Hz ROI epochs.

| Methods statement | Implementation |
|---|---|
| Welch PSD of the baseline-corrected regional waveform, Hann, nperseg = min(256, N), 50 % overlap, mean removal, density scaling, ≤ 250 Hz | `PSD_NPERSEG_MAX = 256`, `PSD_FREQ_MAX_HZ = 250`, `scipy.signal.welch(nperseg=min(256, N), noverlap=nperseg//2)` |
| Bands δ 1–4, θ 4–8, α 8–12, β 12–30, γ 30–80, high-γ 80–200 (lower inclusive, upper exclusive); relative power; peak frequency | `PSD_BANDS`, band-integration helper (trapezoid) |
| specparam over 2–200 Hz, fixed mode, peak width 1–20 Hz, ≤ 6 peaks; fallback OLS log–log slope excluding line-noise bands; method recorded | `APERIODIC_FIT_RANGE_HZ = (2, 200)`, `SpectralModel(peak_width_limits=[1, 20], max_n_peaks=6)` (legacy `FOOOF` accepted), `APERIODIC_IGNORE_BANDS_HZ`, `aperiodic_fit_method` column |
| Complex Morlet `cmor1.5-1.0` at 2, 4, 8, 12, 20, 30, 45, 70, 100, 150, 200 Hz; power = \|W\|²; total, baseline, evoked/baseline ratio, band fractions, wavelet peak frequency | `WAVELET_NAME`, `WAVELET_FREQS_HZ`, `compute_wavelet_power()`, `wavelet_band_features()` |
| Recording-level Welch PSD per electrode (conventional comparison) | `benchmark/welch_psd_recordings.py` → `data/psd_tables/*_psd.npz` |

## Neuronal avalanche analysis

| Methods statement | Implementation |
|---|---|
| ROI epochs at 500 Hz; noise = 1.4826·MAD over −50..0 ms; nLFP = local minimum in 0..250 ms below median − 4·σ̂; ≥ 4 ms apart (`find_peaks`) | `core/ctx_hippo_avalanche_kuramoto_compare.py`: `AVALANCHE_BASELINE_WINDOW_MS`, `AVALANCHE_DETECT_WINDOW_MS`, `AVALANCHE_NEG_THRESH_MAD = 4.0`, `AVALANCHE_MIN_PEAK_DISTANCE_MS = 4.0` |
| Bins of 2, 4, 8 ms; avalanche = maximal run of non-empty bins; size S, duration T; CCDFs | `AVALANCHE_BIN_WIDTH_MS_LIST = [2, 4, 8]` |
| Discrete power-law fits (≥ 10 avalanches) with `powerlaw` (KS-optimal x_min, MLE); γ from log S vs log T | `AVALANCHE_MIN_AVALANCHES_FOR_FIT = 10`, `fit_powerlaw_metric()` (fallback discrete MLE) |
| Whole-slice streaming version (t₀-aligned windows, alternative-distribution comparison) | `core/avalanche_statistics.py` |

## CSD correlation and inter-regional coupling

| Methods statement | Implementation |
|---|---|
| Zero-lag Pearson r between electrode CSD series; within-ROI 16 × 16 over −50..500 ms | `core/roi_lfp_csd_metrics.py` (`lfp_roi_corr`, `csd_roi_corr` matrices); `core/electrode_connectivity.py` within-ROI block |
| 96-node inter-regional: \|CSD\| epochs baseline-corrected, 0..250 ms, concatenated across matching event indices; pairs with < 3 samples or zero variance excluded; Fisher z; ROI value = mean over pairs to the other five ROIs | `electrode_connectivity.py` (`SIGNALS_TO_RUN` includes `csd_abs`, `ANALYSIS_WINDOW_MS = (0, 250)`, `MIN_POINTS_FOR_CORR = 3`, `fisher_z()`, `mean_corr_fisher()`) |
| Stereotypy: Pearson r across Ω between spatial CSD maps of each condition pair (5 × 5); index = mean Fisher z over the 10 pairs | `pharmacology/condition_stereotypy.py` → `stereotypy_index.csv` |

## Drug-normalized spatial maps

`pharmacology/drug_lfp_zscore_maps.py`.

| Methods statement | Implementation |
|---|---|
| Per-electrode mean LFP amplitude map per drug | `MAP_KEY = /lfp/metrics/rms_map` (or `--compute_from_ts`) |
| Row- and column-median striping removal; Gaussian σ = 8 background removal (summary maps only) | `remove_striping()`, `remove_background()` (`BG_SIGMA = 8`, mask-aware normalized convolution) |
| Robust z against the slice's 4-AP map: (X − median)/(1.4826·MAD) over Ω | `robust_baseline_stats()`, `robust_z()` (`MAD_SCALE = 1.4826`) |
| Network-state deviation = median Z over Ω per animal and drug (4-AP = 0) | `network_state_deviation.csv` (animal-level table) |

## Classification, scalability and feature sensitivity

`benchmark/roc_scalability_sensitivity.py` on the per-event feature tables of
`psd_wavelet_metrics.py`, `electrode_connectivity.py` and `ctx_hippo_avalanche_kuramoto_compare.py`.

| Methods statement | Implementation |
|---|---|
| Event tables averaged to one row per recording × ROI; SPACE / conventional / correlation feature sets | `aggregate_feature_table()`, `feature_sets()` |
| L2 logistic regression, liblinear, C = 0.25, balanced; median imputation, zero-variance removal, ANOVA-F selection of ≤ 5 features, standardization, all fitted inside folds; leave-one-animal-out | `_make_roc_model()` (`LogisticRegression(solver="liblinear", C=0.25, class_weight="balanced")`, `SimpleImputer(median)`, `VarianceThreshold`, `SelectKBest(f_classif, k≤5)`, `StandardScaler`), `LeaveOneGroupOut` on `ROC_GROUP_COL = "dataset"` |
| 95 % CI from 2,000 bootstraps; one-sided P from 1,000 label permutations | `ROC_N_BOOTSTRAP = 2000`, `ROC_N_PERMUTATIONS = 1000`, `bootstrap_auc_ci()`, `permutation_auc_pvalue()` |
| Scaling: 300-sample epochs, N = 100…10,000 (with replacement above the recorded count), 10 iterations × 100 runs; SPACE = Hilbert + R(t) + MPV; conventional = periodogram + threshold avalanches; pairwise Pearson | `SCALABILITY_T = 300`, `SCALABILITY_N_LIST`, `SCALABILITY_REPEATS = 10`, `SCALABILITY_RUNS_PER_REPEAT = 100`, `make_benchmark_matrix()`, `space_metric_time()`, `psd_metric_time()`, `avalanche_metric_time()`, `pairwise_corr_time()` |
| Sensitivity = −log₁₀ P of two-sided Welch's t on pooled events, BH-corrected | `welch_p()`, sensitivity block (`statsmodels` `multipletests(method="fdr_bh")`) |
| Supporting pairwise Mann–Whitney / Cliff's δ / BH-FDR of SPACE (R, MPV) vs PSD features across condition pairs | `pharmacology/spectral_sensitivity_bh_fdr.py` |

## Statistics and reproducibility

`core/genotype_roi_comparison.py` pools every module's output into one animal-level table
(events → ROI → region means per slice), runs two-tailed Welch's *t*-tests per metric × ROI /
region / slice with Benjamini–Hochberg correction within each feature family (`comparison_stats.csv`,
`REPORT.md`, figures). `run_baseline.py` chains the whole baseline workflow.

Animal-level tables produced here (`animal_level.csv`, `network_state_deviation.csv`,
`stereotypy_index.csv`, per-ROI summary CSVs and `.xlsx` exports) were analysed in GraphPad
Prism (Welch's *t*, BH-FDR within the stated families, linear mixed-effects models with animal as
random intercept, Šidák-adjusted contrasts). Correlations are Fisher z-transformed before
averaging in every script (`fisher_z()`).

## Software

See `requirements.txt` (minimal) and `requirements-pinned.txt` (versions of the analysis
environment). All scripts stream HDF5 in blocks; peak memory is independent of recording length.
