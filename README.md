# SPACE — Spatial Phase, Activity and Current Estimation

Spike‑sorting‑free, anatomically registered analysis of 4,096‑channel HD‑MEA recordings
(3Brain BioCAM, 64 × 64 grid) from acute brain slices. This repository is the analysis code of

> *SPACE: A mesoscale computational framework for characterizing network coherence and
> pharmacological profiling in high-density MEAs* (2026, in review).

The core workflow is a **genotype × ROI comparison of 4‑AP baseline recordings** (one slice per
animal; here CTL vs 5xFAD), run with a single command. The pharmacological extension used for
Fig. 6–7 of the paper lives in `pharmacology/` and is optional. The companion repository
[lfp-csd-explorer](../lfp-csd-explorer) renders the CSD/LFP maps, kymographs and movies.

Raw `.brw` recordings are not distributed. `examples/make_synthetic_example.py` generates a
synthetic dataset that exercises every script; the real data are available from the
corresponding author on reasonable request.

- **[METHODS_TO_CODE.md](METHODS_TO_CODE.md)** — every Methods paragraph → script / function / parameter.
- **[data/README.md](data/README.md)** — folder and file schemas.
- **[docs/](docs/)** — author checklist, provenance of the original scripts, smoke‑test log.

---

## Quick start (your own recordings)

```
data/raw_brw/      c57bl6_cor_6a_base.brw   5xfad_cor_2a_base.brw   ...   (one 4‑AP recording per slice)
data/slice_photos/ c57bl6_cor_6a.jpg        5xfad_cor_2a.jpg        ...   (photo stem = slice id, contained in the brw name)
```

```bash
pip install -r requirements.txt            # Python >= 3.10
python run_baseline.py init                # writes data/slices.csv (slice_id, group, brw, photo) — check the group column
python run_baseline.py mask                # interactive: click the 4 corners of the array on each photo
python run_baseline.py convert             # .brw -> data/processed_2ms/<group>/<slice>_base_2ms.h5  (~15 min, ~20 GB per 2‑min file)
python run_baseline.py rois                # interactive: click the 6 ROI centres (M1 M2 S1 CA1 CA3 DG) on each rectified photo
python run_baseline.py analyze             # every analysis module, ~1–2 h for two slices
python run_baseline.py compare             # CTL vs 5xFAD per ROI / region / slice -> data/results/genotype_roi_comparison/
```
`python run_baseline.py all` runs the six stages in order; steps whose outputs exist are skipped
(`--force` reruns, `--only`/`--skip` select steps, `--dry_run` prints the commands). Works from
Windows `cmd.exe`, PowerShell or bash. Logs: `data/results/logs/`.

The final report is `data/results/genotype_roi_comparison/REPORT.md` with
`comparison_stats.csv` (group means, SEM, Welch *t*, BH‑adjusted *q*, Cohen's *d* per metric ×
ROI/region) and figures per feature family plus a Fig. 2‑style overview. With one animal per
group the tables still report the values and differences, but *p*/*q* are undefined (the
manuscript used n = 3 per genotype).

## What `analyze` runs

```
data/processed_2ms/<group>/<slice>_base_2ms.h5      (preprocessing/brw_to_2ms_h5.py: notch 60 Hz, 1–350 Hz, 2 ms moving average, planar CSD)
   │
   ├─ utils/mask_noise_reference.py                 tissue vs background electrodes               Suppl. 1
   ├─ preprocessing/extract_event_epochs.py         3 events/recording (20/50/80th pct) -> data/event_traces/   (500 Hz ROI epochs)
   │     ├─ core/psd_wavelet_metrics.py             LFP amplitude/energy/slope, complexity D, Welch bands, 1/f, wavelet   Fig. 2B–E, 4
   │     ├─ core/electrode_connectivity.py          16×16 within‑ROI and 96‑node inter‑regional CSD coupling              Fig. 3G–H
   │     ├─ core/ctx_hippo_avalanche_kuramoto_compare.py   regional Kuramoto R / MPV / PLV; ROI avalanches               Fig. 5H–I, Suppl. 3
   │     └─ benchmark/roc_scalability_sensitivity.py       ROC, scalability, feature sensitivity                         Suppl. 7
   ├─ core/regional_csd_dynamics.py                 |CSD|, total current, RAI, balance index, lag‑1 AC, SampEn, MCR       Fig. 2F–M
   ├─ core/roi_lfp_csd_metrics.py                   maps at 0/250/500 ms, raw traces, kymographs, ROI correlation         Fig. 2A,C; 3A–G
   ├─ core/roi_phase_gradient_pgd.py                generalized phase, gradient vectors, PGD, propagation direction       Fig. 5A–C
   ├─ core/phase_kuramoto.py                        whole‑slice Hilbert phase, Kuramoto R(t), Φ(t), MPV, coherence map   Fig. 5D–G
   ├─ core/avalanche_statistics.py                  whole‑slice streaming nLFP avalanches, power‑law fits                 Suppl. 3
   ├─ benchmark/welch_psd_recordings.py             per‑electrode Welch PSD tables (conventional comparison)
   └─ utils/pick_representative_recording.py        objective representative recording (column‑artefact z)
                                                     │
core/genotype_roi_comparison.py  ◄───────────────────┘   pools all animal‑level tables -> CTL vs 5xFAD per ROI / Ctx / Hippo
```

Every script can also be run on its own (`python <script> --help`); `examples/run_example.sh`
shows the full command line of each on the synthetic dataset.

## Pharmacology extension (optional, `pharmacology/`)

For slices recorded under several conditions (4‑AP baseline, AFDX, BIC, CNQX, ML297; condition
token in the file name), convert each recording as above and run:

| Script | Methods section | Figures |
|---|---|---|
| `pharmacology/drug_lfp_zscore_maps.py` | Drug‑normalized spatial maps (robust z vs 4‑AP, network‑state deviation) | 6A–B |
| `pharmacology/condition_stereotypy.py` | 5 × 5 condition correlation, stereotypy index | 7B |
| `pharmacology/spectral_sensitivity_bh_fdr.py` | pairwise SPACE‑vs‑PSD sensitivity across conditions (supporting) | Suppl. 7C |
| `core/psd_wavelet_metrics.py`, `electrode_connectivity.py`, `ctx_hippo_…compare.py` with `--condition AFDX` etc. | per‑condition spectral, coupling and Kuramoto metrics | 6C–D, 7C–G, Suppl. 4–6 |

## Repository layout

```
run_baseline.py            one‑command pipeline (init / mask / convert / rois / analyze / compare)
preprocessing/             tissue_registration_mask.py, brw_to_2ms_h5.py, pick_rois.py, extract_event_epochs.py
core/                      analysis modules (see above) + genotype_roi_comparison.py
benchmark/                 welch_psd_recordings.py, roc_scalability_sensitivity.py
pharmacology/              drug‑condition analyses (optional)
utils/                     mask_noise_reference.py, pick_representative_recording.py
extras/                    exploratory CSD‑metric fingerprint (not in the manuscript)
examples/                  make_synthetic_example.py, run_example.sh
data/                      schema only (README + .gitkeep); recordings and results are gitignored
```

## Installation notes

`powerlaw`, `specparam`, `networkx`, `statsmodels`, `umap-learn` are optional (documented
fallbacks). The two interactive steps (`mask`, `rois`) need a GUI matplotlib backend
(`pip install PyQt5` if no window appears). Versions used for the manuscript:
`requirements-pinned.txt`.

## Conventions

- Electrode flat index = `row * 64 + col`; ROI electrodes row‑major within the 4 × 4 box;
  ROI CSV columns `roi_index, roi_name, r0, r1, c0, c1` are Python slice bounds.
- CSD: negative = sink; arbitrary units (σ_c = 1; Laplacian / pitch²).
- Windows: baseline −50..0 ms, analysis 0..250 ms (inclusive); traces baseline‑corrected by
  their baseline median. Regions: Ctx = M1, M2, S1; Hippo = CA1, CA3, DG. The animal is the
  unit: events → ROI → region means per slice before any test.
- The stored sampling rate (`/meta` attr `fs`, 9043.34 Hz for BioCAM X) is used everywhere; the
  2 ms window is therefore 18 samples.

## Citation

```bibtex
@article{space_hdmea_2026,
  title   = {SPACE: a mesoscale computational framework for characterizing network coherence and pharmacological profiling in high-density MEAs},
  author  = {<author list>},
  journal = {<journal>},
  year    = {2026},
  doi     = {<doi>}
}
```

MIT License — see [LICENSE](LICENSE).
