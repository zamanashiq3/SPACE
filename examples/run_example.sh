#!/usr/bin/env bash
# End-to-end run of the SPACE pipeline on a SYNTHETIC example dataset.
# Usage:  bash examples/run_example.sh [--light]
#   default : 10 kHz rolling files, 1.5 s, 1 slice per group, 4-AP + AFDX   (~2 GB disk, ~30 min on 2 cores)
#   --light : 500 Hz non-overlap files, 20 s (faster, smaller; the whole-slice scripts still run)
# Every command below is the same one you would run on real data, with data/example_synthetic/
# substituted for data/. Outputs: data/example_synthetic/results/<module>/
set -euo pipefail
cd "$(dirname "$0")/.."
S=data/example_synthetic; R=$S/results; mkdir -p "$R"

GEN_ARGS="--fs 10000 --duration_s 1.5 --n_slices_per_group 1 --drugs 4ap,afdx"
[[ "${1:-}" == "--light" ]] && GEN_ARGS="--fs 500 --duration_s 20 --n_slices_per_group 1 --drugs 4ap,afdx"

echo "### 0  synthetic data";        python examples/make_synthetic_example.py --out_root $S $GEN_ARGS
echo "### 1  noise reference";       python utils/mask_noise_reference.py --in_root $S/processed_2ms --mask_dir $S/masks --out_dir $R/mask_noise_reference
echo "### 2  event epochs";          python preprocessing/extract_event_epochs.py --in_root $S/processed_2ms --mask_dir $S/masks --roi_dir $S/roi_definitions --out_root $S/event_traces --results_dir $R/event_epochs
echo "### 3  regional CSD dynamics"; python core/regional_csd_dynamics.py --in_root $S/processed_2ms --mask_dir $S/masks --roi_dir $S/roi_definitions --out_dir $R/regional_csd_dynamics
echo "### 4  LFP/PSD/wavelet";       python core/psd_wavelet_metrics.py --root_dir $S/event_traces --out_dir $R/psd_wavelet_metrics --groups CTL,5xFAD
echo "### 5  connectivity";          python core/electrode_connectivity.py --root_dir $S/event_traces --out_dir $R/electrode_connectivity --groups CTL,5xFAD
echo "### 6  regional Kuramoto / avalanches"; python core/ctx_hippo_avalanche_kuramoto_compare.py --root_dir $S/event_traces --out_dir $R/ctx_hippo_avalanche_kuramoto_compare --groups CTL,5xFAD
echo "### 7  drug z-maps";           python pharmacology/drug_lfp_zscore_maps.py --in_root $S/processed_2ms --mask_dir $S/masks --out_dir $R/drug_lfp_zscore_maps
echo "### 8  stereotypy";            python pharmacology/condition_stereotypy.py --in_root $S/processed_2ms --mask_dir $S/masks --out_dir $R/condition_stereotypy
echo "### 9  Welch PSD tables";      python benchmark/welch_psd_recordings.py --in_root $S/processed_2ms --mask_dir $S/masks --out_dir $S/psd_tables --figs_dir $R/welch_psd

# whole-slice scripts take one mask / one ROI csv per call -> loop over slices
for g in CTL 5xFAD; do
  for m in $S/masks/*__contact_mask_64x64.npy; do
    sid=$(basename "$m" | cut -d_ -f1-2); grp_files=$(ls $S/processed_2ms/$g/${sid}_*.h5 2>/dev/null || true)
    [[ -z "$grp_files" ]] && continue
    roi=$S/roi_definitions/${sid}_roi_definitions.csv
    echo "### 10 PGD $g/$sid";        python core/roi_phase_gradient_pgd.py --data_dir $S/processed_2ms/$g --mask $m --roi_csv $roi --out_dir $R/roi_phase_gradient_pgd/$g
    echo "### 11 ROI maps $g/$sid";   python core/roi_lfp_csd_metrics.py --data_dir $S/processed_2ms/$g --mask $m --roi_csv $roi --genotype-label $g --out_dir $R/roi_lfp_csd_metrics/$g
    echo "### 12 Kuramoto $g/$sid";   python core/phase_kuramoto.py --in_dir $S/processed_2ms/$g --mask $m --out_root $R/phase_kuramoto/$g
    echo "### 13 avalanches $g/$sid"; python core/avalanche_statistics.py --in_dir $S/processed_2ms/$g --mask $m --out_root $R/avalanche_statistics/$g
    echo "### 14 representative $g";  python utils/pick_representative_recording.py --in_dir $S/processed_2ms/$g --mask $m --out_dir $R/pick_representative_recording/$g
  done
done
# stage Kuramoto tables in the data/kuramoto_tables layout
for d in $R/phase_kuramoto/*/*/; do n=$(basename "$d"); mkdir -p $S/kuramoto_tables/$n/tables
  cp "$d"/tables/kuramoto_ALL.csv "$d"/tables/kuramoto_ALL_summary.json $S/kuramoto_tables/$n/tables/; done

echo "### 15 spectral sensitivity"; python pharmacology/spectral_sensitivity_bh_fdr.py --space_root $S/kuramoto_tables --psd_root $S/psd_tables --out_dir $R/spectral_sensitivity_bh_fdr --pair_a 4ap --pair_b afdx
echo "### 16 ROC / scalability / sensitivity (quick)"; python benchmark/roc_scalability_sensitivity.py --space_root $R/ctx_hippo_avalanche_kuramoto_compare --lfp_csd_root $R/psd_wavelet_metrics --event_roots $S/event_traces/CTL,$S/event_traces/5xFAD --out_dir $R/roc_scalability_sensitivity --quick
echo "DONE -> $R"
