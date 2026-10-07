# Provenance: original working scripts → release modules

The analysis was developed as dated, hard-coded scripts. This table records where each one
went in the release, so that results produced with an original script can be traced to a
module here. "Superseded" means a later version of the same analysis replaced it; "not in
manuscript" means the analysis was exploratory and is not reported. The original files are
kept privately by the authors and are not part of the repository.

| Original script | Release module | Status |
|---|---|---|
| `main_runner.py` + `analysis_writer_2ms_streaming.py` | `preprocessing/brw_to_2ms_h5.py` | superseded (two-step raw-resolution workflow; the rolling 2 ms average at every sample is reproduced by `--bin_stride_mode rolling`) |
| `260512_optimized_brw_to_2ms_h5.py` | `preprocessing/brw_to_2ms_h5.py` | included |
| `image_helper_and_manifest.py` | `preprocessing/tissue_registration_mask.py` | included |
| — (event sampling at 20/50/80th percentiles) | `preprocessing/extract_event_epochs.py` | written for the release from the Methods |
| `260505_roi_lfp_csd_snapshot_raw_kymo_FULL_CORR_ENTROPY_METRICS.py` | `core/roi_lfp_csd_metrics.py` | included (final version of the 260419 → 260428 → 260503 → 260505 series) |
| `260419_lfp_roi_snapshot_archtype_raw.py`, `260428_…FULL.py`, `260503_…CORR_UPDATED.py` | `core/roi_lfp_csd_metrics.py` | superseded |
| — (regional CSD activity, RAI, BI dynamics: lag-1 AC, SampEn, MCR) | `core/regional_csd_dynamics.py` | written for the release from the Methods |
| `260726_roi_phase_kymo_PGD_v4.py` | `core/roi_phase_gradient_pgd.py` | included (+ time-resolved propagation direction / circular statistics added for Fig. 5B–C, 7E) |
| `260501_roi_phase_kymograph_PGD_FULL.py` | `core/roi_phase_gradient_pgd.py` | superseded |
| `260430_roi_phase_region_vector_kymo_FULL.py`, `260726_roi_phase_region_vector_kymo_FULL.py` | — | not in manuscript (regional phase-vector kymographs; regional R(t)/PLV are covered by `ctx_hippo_avalanche_kuramoto_compare.py`) |
| `kuranoto_all_in_one.py` | `core/phase_kuramoto.py` | included (band set to 5–1,500 Hz as in the Methods) |
| `kuranoto_new.py` | — | superseded by `kuranoto_all_in_one.py` |
| `260309_phase_map.py`, `260310_kuramoto_phase_visualization_factory.py` | — | superseded (phase snapshots now in `core/phase_kuramoto.py` and `core/roi_phase_gradient_pgd.py`) |
| `260304_make_csd_kymograph_and_kuramoto_direction.py` | `core/roi_phase_gradient_pgd.py` (direction), `lfp-csd-explorer/roi_hotspot_kymograph.py` (kymographs) | superseded |
| `260322_LFP_avalanche_streaming_factory.py` | `core/avalanche_statistics.py` | included |
| — (ROI-epoch avalanches, 2/4/8 ms bins; regional Kuramoto/PLV) | `core/ctx_hippo_avalanche_kuramoto_compare.py` | included (from the event-epoch working folder) |
| — (PSD/wavelet/specparam/complexity on ROI epochs) | `core/psd_wavelet_metrics.py` | included (from the event-epoch working folder) |
| — (96-node inter-regional coupling) | `core/electrode_connectivity.py` | included (from the event-epoch working folder) |
| `roboustz_v1-remove_articact.py` | `core/drug_lfp_zscore_maps.py` | rewritten to the Methods (robust z against the 4-AP map, σ = 8 background) |
| `roboustz_v1.py`, `roboustz_v3.py`, `csd-only_vis.py` | `extras/csd_metric_fingerprint_pca_umap.py` | exploratory CSD-metric fingerprint with PCA/UMAP; not in manuscript |
| — (5 × 5 condition stereotypy) | `core/condition_stereotypy.py` | written for the release from the Methods |
| `261221_4_psd_welch_streaming_traverse.py` | `benchmark/welch_psd_recordings.py` | included |
| `261221_5_sensitivity_psd_vs_space_pvalues.py` | `benchmark/spectral_sensitivity_bh_fdr.py` | included (supporting) |
| `261221_p7_roc_space_vs_corr.py`, `261221_6_benchmark_spcace.py` | `benchmark/roc_scalability_sensitivity.py` | superseded by the combined benchmark (leave-one-animal-out, bootstrap CI, permutation P, 10 × 100-run scaling) |
| `261221_2_pick_clean_csd_representative.py` | `utils/pick_representative_recording.py` | included |
| `261221_CSD_PSD_welch_compare_like_PSD.py`, `261221_PSD_2D_projection_maps.py` | — | not in manuscript (CSD spectra; PSD projection maps) |
| `260305_moran.py` | — | not in manuscript (Moran's I, fragmentation, PCI) |
| `260409_lfp_cluster_panels_analysis.py`, `260409_lfp_event_analysis_pipeline_v2.py`, `ltp_cluster_panels.py` | — | not in manuscript (archetype clustering, spike-latency analysis) |
| `viz_save_all_2ms.py`, `visulization_csd_lfp_FINAL.py` | — | superseded by `lfp-csd-explorer` |
| `260121_1_make_csd_snapshots_streaming_hardcoded.py` | `lfp-csd-explorer/csd_snapshot_montage.py` | included |
| `260408_make_csd_kymograph_roi.py`, `260415_lfp_snapshot_kymo.py` | `lfp-csd-explorer/roi_hotspot_kymograph.py` | included / superseded |
| `261221_make_CSD_event_movie.py` | `lfp-csd-explorer/csd_event_movie.py` | included |
| `260624_make_FULL_SLICE_SPACE_3x2_movie_v5_CSD_H_ONLY.py` | `lfp-csd-explorer/full_slice_3x2_movie.py` | included |
| `261221_3_polar_kuramoto_movie.py` | `lfp-csd-explorer/kuramoto_polar_movie.py` | included |
| `261221_3a_polar_kuramoto_density.py` | — | not in manuscript |
| `inspect_ts2ms.py` | `lfp-csd-explorer/inspect_recording.py` | included |
