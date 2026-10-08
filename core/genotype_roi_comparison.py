# -*- coding: utf-8 -*-
"""
genotype_roi_comparison.py -- SPACE framework: pooled genotype comparison (CTL vs 5xFAD) of every
baseline metric at ROI, region and whole-slice level, with Welch t-tests and BH-FDR per family.

Manuscript Methods section: "Statistics and reproducibility" and "Genotype differences at baseline".
Manuscript figure(s): Fig. 2 (LFP / CSD genotype comparisons), Fig. 3H (six ROIs), Fig. 4B-H
(spectral bands and features), Fig. 5F-G (whole-slice Kuramoto R and mean phase velocity),
inter-regional correlation and avalanche comparisons.

Inputs (all read-only; every table is discovered recursively under --results_root, so both the flat
``results/<module>/<group>/...`` layout and the ``--per_slice_dirs`` layout
``results/<module>/<group>/<slice_id>/...`` of run_baseline.py are handled):
    regional_csd_dynamics/{roi_metrics,region_metrics}.csv
        CSD peak, total current, time-averaged BI, Source/Sink AUC (ROI level);
        lag-1 autocorrelation, sample entropy, mean crossing rate (region level); RAI (slice level)
    psd_wavelet_metrics/**/baseline_event_lfp_csd_features.csv       LFP amplitude / energy / slope /
                                                                      complexity D (+ event CSD peak, BI)
    psd_wavelet_metrics/**/baseline_wavelet_event_features.csv       wavelet band fractions, total /
                                                                      baseline power, peak frequency
    psd_wavelet_metrics/**/baseline_frequency_psd_event_features.csv Welch PSD aperiodic exponent, peak
                                                                      frequency, total power
    roi_lfp_csd_metrics/**/all_lfp_amplitude_energy_slope_metrics.csv  LFP amplitude / energy / slope of
                                                                        the largest event (t0), per channel
    electrode_connectivity/**/fc16_dataset_level_electrode_pair_pcc.csv  electrode-pair Pearson r ->
                                                                          Fisher z inter-/intra-regional
                                                                          correlation per ROI
    ctx_hippo_avalanche_kuramoto_compare/**/roi_kuramoto_event_summary.csv   ROI Kuramoto R, PLV, phase
                                                                              gradient, wave speed
    ctx_hippo_avalanche_kuramoto_compare/**/roi_avalanche_event_summary.csv  ROI avalanche tau / alpha /
                                                                              gamma / count / rate
    roi_phase_gradient_pgd/**/summary.csv                      mean PGD, PGD-window R, wave fraction,
                                                               direction resultant per ROI
    phase_kuramoto/**/kuramoto_ALL_summary.json                whole-slice Kuramoto R mean / s.d. and
                                                               mean phase velocity (MPV)
    avalanche_statistics/**/avalanche_fit_summary.csv          whole-slice avalanche size / duration
                                                               exponents
    mask_noise_reference/**/rms_inside_vs_outside.csv          QC: RMS ratio inside / outside the mask
Outputs (under data/results/genotype_roi_comparison/):
    tidy_metrics_long.csv       one row per slice x level (roi | region | slice) x unit x metric
    animal_level_wide.csv       one row per slice x level x unit, every metric as a column
    comparison_stats.csv        metric, family, level, unit, group means / SEM / n, diff, Cohen's d,
                                standardized difference, Welch t, df, p, BH q
    figures/<family>__regions.(png|svg)     bars (mean +/- SEM, slice points) per metric, x = Ctx / Hippo
    figures/<family>__rois.(png|svg)        same with x = the six ROIs
    figures/effect_size_heatmap.(png|svg)   rows = metrics, cols = regions / ROIs / slice, colour = d
    figures/overview_headline.(png|svg)     Fig. 2-style overview of the headline metrics
    REPORT.md                   per-family results, provenance, caveats; manifest.json

Statistics (as in the Methods):
    The animal (slice) is the unit. Event / channel values are averaged within slice x ROI; ROI values
    are averaged within slice x region (Ctx = M1, M2, S1; Hippo = CA1, CA3, DG) and over all ROIs for
    the whole-slice value. Metrics native to the region or slice level (lag-1, SampEn, MCR, RAI,
    whole-slice R / MPV, avalanche exponents) are taken as computed by their module. Each metric x
    unit is compared between genotypes with a two-tailed unpaired Welch t-test (scipy ttest_ind,
    equal_var=False) on slice-level values; Cohen's d uses the pooled s.d. Benjamini-Hochberg FDR is
    applied within each pre-defined family x level set (statsmodels fdr_bh, numpy fallback); q < 0.05
    is marked. Correlation coefficients are Fisher z-transformed before averaging and testing.

Implementation notes:
    - Families: lfp, csd, spectral_bands (relative alpha / beta / gamma / high-gamma power),
      spectral_features (total / baseline wavelet power, peak frequency, beta-gamma fraction, aperiodic
      exponent), phase (Kuramoto R, MPV, PGD, wave fraction, direction resultant, PLV), coupling
      (inter- / intra-regional Fisher z), avalanche (tau, alpha, gamma, count, rate) and qc.
    - "beta_gamma_fraction" = (beta + gamma band power) / (sum of all wavelet band powers).
    - Inter-regional correlation of an ROI = mean Fisher z of all electrode pairs linking that ROI to
      electrodes of the other region (CTX-HIPPO pairs); intra-regional = pairs within the same region
      but different ROIs. Reported in z units.
    - Slice id: event-level tables carry only the group name as ``dataset``; the slice is taken from
      ``event_meta.json`` of each event folder when it is reachable, else from the recording file
      name (prefix before the ``_cor_`` / drug token), else -- when the group has exactly one known
      slice -- that slice. Known slices are collected from data/masks and from every file-level table.
    - With n = 1 slice per group (the distributed example data) the Welch test and Cohen's d are
      undefined; p, q, d are written as NaN, the group values and their difference are still
      reported and the heatmap falls back to the standardized difference
      (mean_case - mean_control) / mean(|mean_control|, |mean_case|). The manuscript used n = 3.
    - Only the baseline condition (--condition, default 4AP; event tables label it "base") is used.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    HAVE_SM = True
except Exception:  # pragma: no cover
    HAVE_SM = False

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
RESULTS_ROOT = REPO_ROOT / "data" / "results"
MASK_DIR = REPO_ROOT / "data" / "masks"
CONDITION = "4AP"                       # baseline condition compared
CONTROL_PATTERNS = ("ctl", "wt", "c57", "control")
AVALANCHE_WINDOW = "all"                # window of avalanche_statistics fit summary used for whole-slice exponents
Q_ALPHA = 0.05
FIG_DPI = 300
COLORS = {"control": "#4d4d4d", "case": "#c0392b"}
ROI_ORDER = ["M1", "M2", "S1", "CA1", "CA3", "DG"]
REGION_ORDER = ["Ctx", "Hippo"]
REGION_OF = {"M1": "Ctx", "M2": "Ctx", "S1": "Ctx", "CA1": "Hippo", "CA3": "Hippo", "DG": "Hippo"}
ROI_ALIASES = {"SSP": "S1", "SS1": "S1"}
FAMILY_ORDER = ["lfp", "csd", "spectral_bands", "spectral_features", "phase", "coupling", "avalanche", "qc"]
HEADLINE = [  # Fig. 2-style overview (metric key, preferred level)
    ("lfp_amplitude_uV", "region"), ("lfp_slope_uV_per_ms", "region"), ("lfp_complexity_D", "region"),
    ("csd_total_current", "region"), ("bi_time_avg", "region"), ("rai", "slice"),
    ("lag1_autocorr", "region"), ("sample_entropy", "region"), ("mean_crossing_rate_hz", "region"),
    ("wholeslice_kuramoto_R", "slice"), ("wholeslice_mpv_rad_per_s", "slice"), ("interregional_corr_z_lfp", "region"),
]
# --- END EDIT ---

METRIC_LABELS = {
    "lfp_amplitude_uV": "LFP amplitude (uV)", "lfp_energy_uV2_ms": "LFP energy (uV^2 ms)",
    "lfp_slope_uV_per_ms": "LFP slope (uV/ms)", "lfp_complexity_D": "Complexity D",
    "lfp_amplitude_t0_uV": "LFP amplitude, t0 event (uV)", "lfp_energy_t0_uV2_ms": "LFP energy, t0 event",
    "lfp_slope_t0_uV_per_ms": "LFP slope, t0 event (uV/ms)", "lfp_norm_slope_t0_pct": "LFP norm. slope, t0 (%)",
    "csd_peak": "CSD peak (a.u.)", "csd_total_current": "CSD total current (0-500 ms)",
    "bi_time_avg": "BI (time-avg)", "source_auc": "Source AUC", "sink_auc": "Sink AUC",
    "source_sink_ratio": "Source/Sink ratio", "rai": "RAI", "lag1_autocorr": "Lag-1 autocorr.",
    "sample_entropy": "SampEn", "mean_crossing_rate_hz": "MCR (Hz)",
    "csd_event_abs_peak": "CSD |peak|, events", "csd_event_bi_mean": "BI mean, events",
    "rel_alpha": "Rel. alpha (8-12 Hz)", "rel_beta": "Rel. beta (12-30 Hz)",
    "rel_gamma": "Rel. gamma (30-80 Hz)", "rel_high_gamma": "Rel. high-gamma (80-200 Hz)",
    "wavelet_total_power": "Wavelet total power", "wavelet_baseline_power": "Wavelet baseline power",
    "wavelet_evoked_over_baseline": "Evoked / baseline power", "wavelet_peak_freq_hz": "Wavelet peak freq (Hz)",
    "beta_gamma_fraction": "Beta-gamma fraction", "psd_aperiodic_exponent": "Aperiodic exponent",
    "psd_peak_freq_hz": "PSD peak freq (Hz)", "psd_total_power": "PSD total power",
    "roi_kuramoto_R": "Kuramoto R (ROI)", "roi_plv": "PLV (ROI)", "phase_grad_mag": "Phase gradient |grad|",
    "wave_speed_proxy": "Wave speed proxy", "pgd_mean": "PGD", "pgd_kuramoto_R": "Kuramoto R (PGD window)",
    "wave_fraction": "Wave fraction (PGD>=0.7)", "direction_resultant": "Direction resultant",
    "wholeslice_kuramoto_R": "Kuramoto R (whole slice)", "wholeslice_kuramoto_R_sd": "Kuramoto R s.d. (whole slice)",
    "wholeslice_mpv_rad_per_s": "MPV (rad/s, whole slice)",
    "interregional_corr_z_lfp": "Inter-regional r (Fisher z, LFP)", "intraregional_corr_z_lfp": "Intra-regional r (z, LFP)",
    "interregional_corr_z_csd_abs": "Inter-regional r (z, |CSD|)", "intraregional_corr_z_csd_abs": "Intra-regional r (z, |CSD|)",
    "avalanche_tau": "Avalanche tau (size)", "avalanche_alpha": "Avalanche alpha (duration)",
    "avalanche_gamma": "Avalanche gamma (size-dur.)", "avalanche_count": "Avalanche count",
    "avalanche_rate_per_s": "Avalanche rate (1/s)", "avalanche_tau_wholeslice": "Avalanche tau (whole slice)",
    "avalanche_alpha_wholeslice": "Avalanche alpha (whole slice)",
    "rms_ratio_in_over_out": "RMS in/out mask",
}
FAMILY_LABELS = {"lfp": "LFP", "csd": "CSD", "spectral_bands": "Spectral bands", "spectral_features": "Spectral features",
                 "phase": "Phase / synchrony", "coupling": "Inter-regional coupling", "avalanche": "Avalanches", "qc": "QC"}

LONG_COLS = ["group", "slice_id", "condition", "level", "unit", "roi_name", "region", "metric", "family",
             "source_module", "value", "n_averaged"]


# =============================================================================
# HELPERS
# =============================================================================

def parse_condition(name: str) -> str:
    """Drug token in a file / folder name -> one of 4AP, AFDX, BIC, CNQX, ML297, unknown."""
    n = str(name).lower()
    for tok, out in [("af-dx", "AFDX"), ("afdx", "AFDX"), ("4ap", "4AP"), ("4-ap", "4AP"), ("bic", "BIC"),
                     ("cnqx", "CNQX"), ("ml297", "ML297"), ("baseline", "4AP"), ("base", "4AP"), ("veh", "4AP")]:
        if tok in n:
            return out
    return "unknown"


def canon_roi(name) -> Optional[str]:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    s = str(name).strip()
    s = re.sub(r"^ROI_\d+_", "", s)
    s = ROI_ALIASES.get(s.upper(), s.upper())
    return s if s in ROI_ORDER else None


def canon_region(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s.startswith("ctx") or s.startswith("cor"):
        return "Ctx"
    if s.startswith("hip"):
        return "Hippo"
    return None


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(r, float), -0.999999, 0.999999))


def save_fig(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    plt.close(fig)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaNs ignored and returned as NaN)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    if HAVE_SM:
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
        return q
    pv = p[ok]; m = pv.size
    order = np.argsort(pv); ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


class SliceRegistry:
    """Known slice ids per group (from data/masks and file-level tables) + slice-id inference."""

    def __init__(self, mask_dir: Path):
        self.known: Dict[str, set] = {}
        self.all: set = set()
        if mask_dir.is_dir():
            for p in mask_dir.glob("*__contact_mask_64x64.npy"):
                self.all.add(p.name.split("__")[0])

    def register(self, group: str, slice_id: str):
        if slice_id:
            self.known.setdefault(group, set()).add(slice_id)
            self.all.add(slice_id)

    def from_file(self, group: str, fname: str) -> str:
        """Slice id from a recording name: longest known slice contained in it, else prefix before _cor_/drug."""
        n = Path(str(fname)).name.lower()
        hits = [s for s in self.all if s.lower() in n]
        if hits:
            sid = max(hits, key=len)
        else:
            sid = re.split(r"_cor_|_4ap|_afdx|_af-dx|_bic|_cnqx|_ml297", Path(str(fname)).stem, flags=re.I)[0]
        self.register(group, sid)
        return sid

    def resolve_dataset(self, group: str, token: str) -> str:
        """Dataset token of an event table (often == group name) -> slice id."""
        tok = str(token)
        if tok in self.all:
            return tok
        if re.search(r"_cor_|4ap|afdx|bic|cnqx|ml297", tok, flags=re.I):
            return self.from_file(group, tok)
        known = sorted(self.known.get(group, set()))
        if len(known) == 1:
            return known[0]
        return tok


# =============================================================================
# HARVEST
# =============================================================================

class Harvest:
    """Collects raw (pre-aggregation) rows and provenance."""

    def __init__(self, results_root: Path, registry: SliceRegistry, condition: str, avalanche_window: str):
        self.root = results_root
        self.reg = registry
        self.cond = condition
        self.aval_window = avalanche_window
        self.rows: List[dict] = []
        self.files: List[str] = []
        self.missing: List[str] = []
        self.notes: List[str] = []
        self.harvested: Dict[str, List[str]] = {}
        self._meta_cache: Dict[str, Optional[str]] = {}

    # -- utilities ------------------------------------------------------------
    def find(self, module: str, pattern: str) -> List[Path]:
        d = self.root / module
        return sorted(p for p in d.rglob(pattern) if p.is_file()) if d.is_dir() else []

    def add(self, module: str, family: str, group: str, slice_id: str, condition: str, level: str,
            roi: Optional[str], region: Optional[str], metric: str, value, src: Path):
        if value is None or not np.isfinite(float(value)):
            return
        unit = roi if level == "roi" else (region if level == "region" else "slice")
        self.rows.append(dict(group=str(group), slice_id=str(slice_id), condition=condition, level=level, unit=unit,
                              roi_name=roi, region=region, metric=metric, family=family, source_module=module,
                              value=float(value), source_file=str(src.relative_to(self.root))))
        self.harvested.setdefault(module, [])
        if metric not in self.harvested[module]:
            self.harvested[module].append(metric)

    def _mark(self, p: Path):
        s = str(p.relative_to(self.root))
        if s not in self.files:
            self.files.append(s)

    def slice_from_event_dir(self, event_dir, group: str, dataset: str) -> str:
        key = str(event_dir)
        if key not in self._meta_cache:
            sid = None
            for cand in (REPO_ROOT / key, Path(key), self.root.parent.parent / key):
                mp = cand / "event_meta.json"
                if mp.is_file():
                    try:
                        sid = json.loads(mp.read_text()).get("slice_id")
                    except Exception:
                        sid = None
                    break
            self._meta_cache[key] = sid
        sid = self._meta_cache[key]
        if sid:
            self.reg.register(group, sid)
            return sid
        return self.reg.resolve_dataset(group, dataset)

    def _event_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize an event-level table (phenotype/dataset/roi_name/region/drug/event_dir)."""
        d = df.copy()
        d["group"] = d["phenotype"].astype(str)
        cond_col = "drug" if "drug" in d.columns else None
        d["condition"] = d[cond_col].map(parse_condition) if cond_col else self.cond
        d = d[d["condition"] == self.cond]
        d["roi"] = d["roi_name"].map(canon_roi)
        d = d[d["roi"].notna()]
        d["reg"] = d["roi"].map(REGION_OF)
        ed = d["event_dir"] if "event_dir" in d.columns else pd.Series([""] * len(d), index=d.index)
        d["slice_id"] = [self.slice_from_event_dir(e, g, s) for e, g, s in zip(ed, d["group"], d["dataset"].astype(str))]
        return d

    def _emit_event_metrics(self, module: str, d: pd.DataFrame, mapping: Dict[str, Tuple[str, str]], src: Path):
        """mapping: source column -> (metric key, family). Values are per event; averaging happens later."""
        for col, (metric, fam) in mapping.items():
            if col not in d.columns:
                self.notes.append(f"{src.relative_to(self.root)}: column {col} not found")
                continue
            for _, r in d.iterrows():
                self.add(module, fam, r["group"], r["slice_id"], self.cond, "roi", r["roi"], r["reg"], metric, r[col], src)

    # -- modules --------------------------------------------------------------
    def regional_csd_dynamics(self):
        mod = "regional_csd_dynamics"
        roi_files, reg_files = self.find(mod, "roi_metrics.csv"), self.find(mod, "region_metrics.csv")
        if not roi_files and not reg_files:
            self.missing.append(mod); return
        roi_map = {"csd_peak": "csd_peak", "total_current_0_500": "csd_total_current", "bi_time_avg": "bi_time_avg",
                   "source_auc": "source_auc", "sink_auc": "sink_auc", "source_sink_ratio": "source_sink_ratio"}
        for p in roi_files:
            df = pd.read_csv(p); self._mark(p)
            df = df[df["condition"].astype(str).map(parse_condition) == self.cond]
            for _, r in df.iterrows():
                roi = canon_roi(r["roi_name"])
                if roi is None:
                    continue
                sid = r["slice_id"] if "slice_id" in df.columns else self.reg.from_file(r["group"], r["file"])
                self.reg.register(str(r["group"]), str(sid))
                for col, m in roi_map.items():
                    if col in df.columns:
                        self.add(mod, "csd", r["group"], sid, self.cond, "roi", roi, REGION_OF[roi], m, r[col], p)
        reg_map = {"lag1_autocorr": "lag1_autocorr", "sample_entropy": "sample_entropy", "mean_crossing_rate_hz": "mean_crossing_rate_hz"}
        for p in reg_files:
            df = pd.read_csv(p); self._mark(p)
            df = df[df["condition"].astype(str).map(parse_condition) == self.cond]
            for _, r in df.iterrows():
                reg = canon_region(r["region"])
                sid = r["slice_id"] if "slice_id" in df.columns else self.reg.from_file(r["group"], r["file"])
                self.reg.register(str(r["group"]), str(sid))
                for col, m in reg_map.items():
                    if col in df.columns and reg:
                        self.add(mod, "csd", r["group"], sid, self.cond, "region", None, reg, m, r[col], p)
                if "rai" in df.columns:
                    self.add(mod, "csd", r["group"], sid, self.cond, "slice", None, None, "rai", r["rai"], p)

    def psd_wavelet_metrics(self):
        mod = "psd_wavelet_metrics"
        lfp = self.find(mod, "baseline_event_lfp_csd_features.csv")
        wav = self.find(mod, "baseline_wavelet_event_features.csv")
        psd = self.find(mod, "baseline_frequency_psd_event_features.csv")
        if not (lfp or wav or psd):
            self.missing.append(mod); return
        for p in lfp:
            d = self._event_table(pd.read_csv(p)); self._mark(p)
            self._emit_event_metrics(mod, d, {
                "lfp_abs_peak_uV": ("lfp_amplitude_uV", "lfp"), "lfp_energy_uV2_ms": ("lfp_energy_uV2_ms", "lfp"),
                "lfp_max_abs_slope_uV_per_ms": ("lfp_slope_uV_per_ms", "lfp"), "lfp_complexity_D": ("lfp_complexity_D", "lfp"),
                "csd_abs_peak_uV_per_um2": ("csd_event_abs_peak", "csd"), "csd_balance_index_mean": ("csd_event_bi_mean", "csd")}, p)
        for p in wav:
            d = self._event_table(pd.read_csv(p)); self._mark(p)
            bands = [c for c in d.columns if re.fullmatch(r"[a-z]+_\d+_\d+_power", c)]
            if bands and {"beta_12_30_power", "gamma_30_80_power"} <= set(d.columns):
                tot = d[bands].sum(axis=1)
                d["beta_gamma_fraction"] = (d["beta_12_30_power"] + d["gamma_30_80_power"]) / tot.replace(0, np.nan)
            self._emit_event_metrics(mod, d, {
                "alpha_8_12_fraction": ("rel_alpha", "spectral_bands"), "beta_12_30_fraction": ("rel_beta", "spectral_bands"),
                "gamma_30_80_fraction": ("rel_gamma", "spectral_bands"), "high_80_200_fraction": ("rel_high_gamma", "spectral_bands"),
                "wavelet_total_power": ("wavelet_total_power", "spectral_features"),
                "wavelet_baseline_power": ("wavelet_baseline_power", "spectral_features"),
                "wavelet_evoked_over_baseline": ("wavelet_evoked_over_baseline", "spectral_features"),
                "wavelet_peak_freq_hz": ("wavelet_peak_freq_hz", "spectral_features"),
                "beta_gamma_fraction": ("beta_gamma_fraction", "spectral_features")}, p)
        for p in psd:
            d = self._event_table(pd.read_csv(p)); self._mark(p)
            self._emit_event_metrics(mod, d, {
                "psd_aperiodic_exponent": ("psd_aperiodic_exponent", "spectral_features"),
                "psd_peak_freq_hz": ("psd_peak_freq_hz", "spectral_features"),
                "psd_total_power": ("psd_total_power", "spectral_features")}, p)

    def roi_lfp_csd_metrics(self):
        mod = "roi_lfp_csd_metrics"
        files = self.find(mod, "all_lfp_amplitude_energy_slope_metrics.csv")
        if not files:
            self.missing.append(mod); return
        cmap = {"lfp_amplitude_abs_peak": "lfp_amplitude_t0_uV", "lfp_energy_integral_value2_ms": "lfp_energy_t0_uV2_ms",
                "lfp_abs_slope_value_per_ms": "lfp_slope_t0_uV_per_ms", "lfp_local_norm_slope_pct": "lfp_norm_slope_t0_pct"}
        for p in files:
            df = pd.read_csv(p); self._mark(p)
            group_default = p.relative_to(self.root).parts[1] if len(p.relative_to(self.root).parts) > 2 else "unknown"
            df["group"] = df["genotype"].astype(str) if "genotype" in df.columns else group_default
            df["condition"] = df["drug"].astype(str).map(parse_condition) if "drug" in df.columns else df["file"].map(parse_condition)
            df = df[df["condition"] == self.cond]
            df["roi"] = df["roi_name"].map(canon_roi)
            df = df[df["roi"].notna()]
            cols = [c for c in cmap if c in df.columns]
            agg = df.groupby(["group", "file", "roi"])[cols].mean().reset_index()   # channels -> ROI
            for _, r in agg.iterrows():
                sid = self.reg.from_file(r["group"], r["file"])
                for c in cols:
                    self.add(mod, "lfp", r["group"], sid, self.cond, "roi", r["roi"], REGION_OF[r["roi"]], cmap[c], r[c], p)

    def electrode_connectivity(self):
        mod = "electrode_connectivity"
        files = self.find(mod, "fc16_dataset_level_electrode_pair_pcc.csv")
        if not files:
            self.missing.append(mod); return
        for p in files:
            df = pd.read_csv(p); self._mark(p)
            df = df[df["signal"].isin(["lfp", "csd_abs"])].copy()
            df["z"] = fisher_z(df["pcc"].values)
            df["ri"], df["rj"] = df["roi_i"].map(canon_roi), df["roi_j"].map(canon_roi)
            df = df[df["ri"].notna() & df["rj"].notna()]
            df["inter"] = df["ri"].map(REGION_OF) != df["rj"].map(REGION_OF)
            df["intra"] = (~df["inter"]) & (df["ri"] != df["rj"])
            both = pd.concat([df.rename(columns={"ri": "roi"}), df.rename(columns={"rj": "roi"})], ignore_index=True)
            for kind in ("inter", "intra"):
                sub = both[both[kind]]
                agg = sub.groupby(["signal", "phenotype", "dataset", "roi"])["z"].mean().reset_index()
                for _, r in agg.iterrows():
                    sid = self.reg.resolve_dataset(str(r["phenotype"]), str(r["dataset"]))
                    self.add(mod, "coupling", r["phenotype"], sid, self.cond, "roi", r["roi"], REGION_OF[r["roi"]],
                             f"{kind}regional_corr_z_{r['signal']}", r["z"], p)

    def ctx_hippo(self):
        mod = "ctx_hippo_avalanche_kuramoto_compare"
        kur, ava = self.find(mod, "roi_kuramoto_event_summary.csv"), self.find(mod, "roi_avalanche_event_summary.csv")
        if not (kur or ava):
            self.missing.append(mod); return
        for p in kur:
            d = self._event_table(pd.read_csv(p)); self._mark(p)
            self._emit_event_metrics(mod, d, {
                "kuramoto_R_mean": ("roi_kuramoto_R", "phase"), "plv_mean": ("roi_plv", "phase"),
                "phase_grad_mag_mean": ("phase_grad_mag", "phase"), "wave_speed_proxy_mean": ("wave_speed_proxy", "phase")}, p)
        for p in ava:
            raw = pd.read_csv(p)
            if "activity_bin_width_ms" in raw.columns:
                raw = raw[raw["activity_bin_width_ms"] == raw["activity_bin_width_ms"].min()]
            d = self._event_table(raw); self._mark(p)
            self._emit_event_metrics(mod, d, {
                "avalanche_size_alpha": ("avalanche_tau", "avalanche"), "avalanche_duration_alpha": ("avalanche_alpha", "avalanche"),
                "avalanche_size_duration_loglog_slope": ("avalanche_gamma", "avalanche"),
                "n_avalanches": ("avalanche_count", "avalanche"), "avalanche_rate_per_s": ("avalanche_rate_per_s", "avalanche")}, p)

    def roi_phase_gradient_pgd(self):
        mod = "roi_phase_gradient_pgd"
        files = [p for p in self.find(mod, "summary.csv") if "roi_name" in pd.read_csv(p, nrows=0).columns]
        if not files:
            self.missing.append(mod); return
        cmap = {"mean_PGD": "pgd_mean", "mean_R": "pgd_kuramoto_R", "wave_fraction_PGDge0.7": "wave_fraction",
                "dir_resultant_full_0_500": "direction_resultant"}
        for p in files:
            df = pd.read_csv(p); self._mark(p)
            group = p.relative_to(self.root).parts[1]
            cond_src = df["drug"] if "drug" in df.columns else df["file"]
            df = df[cond_src.astype(str).map(parse_condition) == self.cond]
            for _, r in df.iterrows():
                roi = canon_roi(r["roi_name"])
                if roi is None:
                    continue
                sid = self.reg.from_file(group, r["file"])
                for c, m in cmap.items():
                    if c in df.columns:
                        self.add(mod, "phase", group, sid, self.cond, "roi", roi, REGION_OF[roi], m, r[c], p)

    def phase_kuramoto(self):
        mod = "phase_kuramoto"
        files = self.find(mod, "kuramoto_ALL_summary.json")
        if not files:
            self.missing.append(mod); return
        for p in files:
            rel = p.relative_to(self.root).parts   # phase_kuramoto/<group>/[<slice>/]<rec>/tables/x.json
            group, rec = rel[1], p.parents[1].name
            if parse_condition(rec) != self.cond:
                continue
            js = json.loads(p.read_text()); self._mark(p)
            sid = self.reg.from_file(group, rec)
            for k, m in [("R_mean", "wholeslice_kuramoto_R"), ("R_std", "wholeslice_kuramoto_R_sd"),
                         ("dPhi_dt_abs_mean", "wholeslice_mpv_rad_per_s")]:
                if k in js:
                    self.add(mod, "phase", group, sid, self.cond, "slice", None, None, m, js[k], p)

    def avalanche_statistics(self):
        mod = "avalanche_statistics"
        files = self.find(mod, "avalanche_fit_summary.csv")
        if not files:
            self.missing.append(mod); return
        for p in files:
            df = pd.read_csv(p); self._mark(p)
            group = p.relative_to(self.root).parts[1]
            df = df[(df["window"].astype(str) == self.aval_window) & (df["condition"].astype(str).map(parse_condition) == self.cond)]
            if "bin_width_bins" in df.columns and len(df):
                df = df[df["bin_width_bins"] == df["bin_width_bins"].min()]
            for _, r in df.iterrows():
                sid = self.reg.from_file(group, r["file"])
                m = {"size": "avalanche_tau_wholeslice", "duration": "avalanche_alpha_wholeslice"}.get(str(r["metric"]))
                if m and (("fit_ok" not in df.columns) or bool(r["fit_ok"])):
                    self.add(mod, "avalanche", group, sid, self.cond, "slice", None, None, m, r["alpha"], p)

    def mask_noise_reference(self):
        mod = "mask_noise_reference"
        files = self.find(mod, "rms_inside_vs_outside.csv")
        if not files:
            self.missing.append(mod); return
        for p in files:
            df = pd.read_csv(p); self._mark(p)
            df = df[df["file"].astype(str).map(parse_condition) == self.cond]
            for _, r in df.iterrows():
                sid = self.reg.from_file(str(r["group"]), r["file"])
                self.add(mod, "qc", r["group"], sid, self.cond, "slice", None, None, "rms_ratio_in_over_out", r["rms_ratio_in_over_out"], p)

    def run_all(self):
        # file-level tables first so the slice registry is populated before event tables are resolved
        for fn in (self.regional_csd_dynamics, self.roi_lfp_csd_metrics, self.roi_phase_gradient_pgd, self.phase_kuramoto,
                   self.avalanche_statistics, self.mask_noise_reference, self.psd_wavelet_metrics, self.electrode_connectivity,
                   self.ctx_hippo):
            try:
                fn()
            except Exception as e:  # keep going on a malformed module
                warnings.warn(f"{fn.__name__}: {type(e).__name__}: {e}")
                self.notes.append(f"{fn.__name__} failed: {type(e).__name__}: {e}")
        for m in self.missing:
            warnings.warn(f"module not found under {self.root}: {m} (skipped)")


# =============================================================================
# AGGREGATION  (events -> ROI -> region -> slice)
# =============================================================================

def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> animal-level long table with levels roi / region / slice."""
    keys = ["group", "slice_id", "condition", "metric", "family", "source_module"]
    out = []
    roi = raw[raw["level"] == "roi"]
    if len(roi):
        g = roi.groupby(keys + ["roi_name"])["value"].agg(["mean", "size"]).reset_index()
        g["level"], g["unit"], g["region"] = "roi", g["roi_name"], g["roi_name"].map(REGION_OF)
        out.append(g)
        rg = g.groupby(keys + ["region"])["mean"].agg(["mean", "size"]).reset_index()   # ROI -> region
        rg["level"], rg["unit"], rg["roi_name"] = "region", rg["region"], None
        out.append(rg)
        sl = g.groupby(keys)["mean"].agg(["mean", "size"]).reset_index()               # ROI -> whole slice
        sl["level"], sl["unit"], sl["roi_name"], sl["region"] = "slice", "slice", None, None
        out.append(sl)
    reg = raw[raw["level"] == "region"]
    if len(reg):
        g = reg.groupby(keys + ["region"])["value"].agg(["mean", "size"]).reset_index()
        g["level"], g["unit"], g["roi_name"] = "region", g["region"], None
        out.append(g)
        sl = g.groupby(keys)["mean"].agg(["mean", "size"]).reset_index()
        sl["level"], sl["unit"], sl["roi_name"], sl["region"] = "slice", "slice", None, None
        out.append(sl)
    sli = raw[raw["level"] == "slice"]
    if len(sli):
        g = sli.groupby(keys)["value"].agg(["mean", "size"]).reset_index()
        g["level"], g["unit"], g["roi_name"], g["region"] = "slice", "slice", None, None
        out.append(g)
    df = pd.concat(out, ignore_index=True).rename(columns={"mean": "value", "size": "n_averaged"})
    return df[LONG_COLS].sort_values(["family", "metric", "level", "unit", "group", "slice_id"]).reset_index(drop=True)


# =============================================================================
# STATISTICS
# =============================================================================

def compare(long: pd.DataFrame, control: str, case: str) -> pd.DataFrame:
    rows = []
    for (metric, family, level, unit), g in long.groupby(["metric", "family", "level", "unit"], sort=False):
        a = g.loc[g["group"] == control, "value"].to_numpy(float)
        b = g.loc[g["group"] == case, "value"].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        n1, n2 = a.size, b.size
        m1 = a.mean() if n1 else np.nan
        m2 = b.mean() if n2 else np.nan
        s1 = a.std(ddof=1) if n1 > 1 else np.nan
        s2 = b.std(ddof=1) if n2 > 1 else np.nan
        sem1 = s1 / np.sqrt(n1) if n1 > 1 else np.nan
        sem2 = s2 / np.sqrt(n2) if n2 > 1 else np.nan
        diff = m2 - m1
        d = t = df_w = p = np.nan
        if n1 > 1 and n2 > 1:
            sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            d = diff / sp if sp > 0 else np.nan
            if (s1 > 0 or s2 > 0):
                res = stats.ttest_ind(b, a, equal_var=False)
                t, p = float(res.statistic), float(res.pvalue)
                v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
                df_w = (v1 + v2) ** 2 / ((v1 ** 2 / (n1 - 1)) + (v2 ** 2 / (n2 - 1)))
        denom = (abs(m1) + abs(m2)) / 2.0
        std_diff = diff / denom if (np.isfinite(denom) and denom > 0) else np.nan
        rows.append(dict(metric=metric, label=METRIC_LABELS.get(metric, metric), family=family, level=level, unit=unit,
                         n_control=n1, n_case=n2, mean_control=m1, sem_control=sem1, mean_case=m2, sem_case=sem2,
                         diff_case_minus_control=diff, pct_change=100 * diff / abs(m1) if (np.isfinite(m1) and m1 != 0) else np.nan,
                         cohen_d=d, std_diff=std_diff, welch_t=t, welch_df=df_w, p=p))
    st = pd.DataFrame(rows)
    st["q"] = np.nan
    for _, idx in st.groupby(["family", "level"]).groups.items():
        st.loc[idx, "q"] = bh_adjust(st.loc[idx, "p"].to_numpy())
    st["significant_q"] = st["q"] < Q_ALPHA
    st["source_module"] = st["metric"].map(long.drop_duplicates("metric").set_index("metric")["source_module"])
    st["family_rank"] = st["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}).fillna(99)
    mrank = {m: i for i, m in enumerate(METRIC_LABELS)}
    st["metric_rank"] = st["metric"].map(mrank).fillna(999)
    urank = {u: i for i, u in enumerate(REGION_ORDER + ROI_ORDER + ["slice"])}
    st["unit_rank"] = st["unit"].map(urank).fillna(999)
    return (st.sort_values(["family_rank", "metric_rank", "metric", "unit_rank"])
              .drop(columns=["family_rank", "metric_rank", "unit_rank"]).reset_index(drop=True))


# =============================================================================
# FIGURES
# =============================================================================

def set_style():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42})


def annot_text(row: pd.Series) -> str:
    if row["n_control"] < 2 or row["n_case"] < 2:
        return f"n={int(row['n_control'])}/{int(row['n_case'])}"
    s = f"p={fmt_p(row['p'])}"
    if np.isfinite(row["q"]):
        s += "*" if row["q"] < Q_ALPHA else ""
        s += f"\nq={fmt_p(row['q'])}"
    return s


def panel_bars(ax, st: pd.DataFrame, long: pd.DataFrame, metric: str, level: str, units: Sequence[str],
               control: str, case: str, rng: np.random.Generator, title: Optional[str] = None):
    """Grouped bars (mean +/- SEM) + slice points for one metric at one level."""
    sub = st[(st["metric"] == metric) & (st["level"] == level)].set_index("unit")
    dat = long[(long["metric"] == metric) & (long["level"] == level)]
    units = [u for u in units if u in sub.index]
    w = 0.36
    for k, (grp, col, off) in enumerate([(control, COLORS["control"], -w / 2), (case, COLORS["case"], w / 2)]):
        means = [sub.loc[u, "mean_control" if k == 0 else "mean_case"] for u in units]
        sems = [sub.loc[u, "sem_control" if k == 0 else "sem_case"] for u in units]
        has_sem = any(np.isfinite(s) for s in sems)
        sems = [0 if not np.isfinite(s) else s for s in sems]
        x = np.arange(len(units)) + off
        ax.bar(x, means, w, yerr=sems if has_sem else None, color=col, alpha=0.85, edgecolor="none", capsize=1.5,
               error_kw=dict(lw=0.6), label=grp, zorder=2)
        for xi, u in zip(x, units):
            v = dat[(dat["unit"] == u) & (dat["group"] == grp)]["value"].to_numpy()
            ax.scatter(xi + rng.uniform(-w / 5, w / 5, v.size), v, s=6, color="white", edgecolor=col, linewidth=0.5, zorder=3)
    ymax = np.nanmax([ax.get_ylim()[1]] + list(dat["value"])) if len(dat) else ax.get_ylim()[1]
    ymin = min(0, ax.get_ylim()[0])
    span = (ymax - ymin) or 1.0
    for i, u in enumerate(units):
        ax.text(i, ymax + 0.04 * span, annot_text(sub.loc[u]), ha="center", va="bottom", fontsize=5.2, linespacing=0.95)
    ax.set_ylim(ymin, ymax + 0.3 * span)
    ax.set_xticks(np.arange(len(units)))
    ax.set_xticklabels(["whole slice" if u == "slice" else u for u in units])
    ax.set_xlim(-0.7, len(units) - 0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(title or METRIC_LABELS.get(metric, metric), pad=2)


def grid(n: int, ncols: int = 4) -> Tuple[int, int]:
    ncols = min(ncols, max(n, 1))
    return int(np.ceil(n / ncols)), ncols


def fig_family(st: pd.DataFrame, long: pd.DataFrame, family: str, level: str, control: str, case: str,
               out_base: Path, rng: np.random.Generator):
    metrics = list(dict.fromkeys(st[(st["family"] == family)]["metric"]))
    if not metrics:
        return False
    units = REGION_ORDER if level == "region" else ROI_ORDER
    nr, nc = grid(len(metrics), 4 if level == "region" else 3)
    fig, axes = plt.subplots(nr, nc, figsize=(1.75 * nc if level == "region" else 2.6 * nc, 1.65 * nr), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        lvl = next(l for l in (level, "region", "slice") if ((st["metric"] == m) & (st["level"] == l)).any())
        panel_bars(ax, st, long, m, lvl, {"roi": ROI_ORDER, "region": REGION_ORDER, "slice": ["slice"]}[lvl], control, case, rng)
    for ax in axes.ravel()[len(metrics):]:
        ax.axis("off")
    hdl, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(hdl, lab, frameon=False, loc="upper right", ncol=2, handlelength=0.8, bbox_to_anchor=(1.0, 1.03))
    fig.suptitle(f"{FAMILY_LABELS.get(family, family)} - {control} vs {case} ({'regions' if level == 'region' else 'ROIs'})",
                 fontsize=8, x=0.02, ha="left", y=1.02)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save_fig(fig, out_base)
    return True


def fig_heatmap(st: pd.DataFrame, control: str, case: str, out_base: Path):
    cols = REGION_ORDER + ROI_ORDER + ["slice"]
    st = st.copy()
    st["eff"] = st["cohen_d"].where(st["cohen_d"].notna(), st["std_diff"])
    use_d = st["cohen_d"].notna().any()
    metrics = list(dict.fromkeys(st["metric"]))
    M = np.full((len(metrics), len(cols)), np.nan)
    sig = np.zeros_like(M, dtype=bool)
    for i, m in enumerate(metrics):
        for j, u in enumerate(cols):
            r = st[(st["metric"] == m) & (st["unit"] == u)]
            if len(r):
                M[i, j] = r["eff"].iloc[0]; sig[i, j] = bool(r["significant_q"].iloc[0])
    vmax = np.nanpercentile(np.abs(M), 98) if np.isfinite(M).any() else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(4.2, 0.17 * len(metrics) + 1.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(["whole slice" if c == "slice" else c for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=5.5)
    fam = st.drop_duplicates("metric").set_index("metric")["family"]
    prev = None
    for i, m in enumerate(metrics):
        if fam[m] != prev and i > 0:
            ax.axhline(i - 0.5, color="k", lw=0.5)
        prev = fam[m]
    for i in range(len(metrics)):
        for j in range(len(cols)):
            if sig[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=7, color="k")
    for s in ax.spines.values():
        s.set_visible(True)
    ax.axvline(1.5, color="k", lw=0.5); ax.axvline(7.5, color="k", lw=0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"Cohen's d ({case} - {control})" if use_d else f"standardized difference ({case} - {control}) / mean|mean|", fontsize=6)
    ax.set_title(f"Effect size by metric and unit ({'* q<0.05' if use_d else 'no inferential test: n<2 per group'})", fontsize=7)
    save_fig(fig, out_base)


def fig_overview(st: pd.DataFrame, long: pd.DataFrame, control: str, case: str, out_base: Path, rng: np.random.Generator):
    items = [(m, lvl) for m, lvl in HEADLINE if (st["metric"] == m).any()]
    if not items:
        return False
    nr, nc = grid(len(items), 4)
    fig, axes = plt.subplots(nr, nc, figsize=(1.8 * nc, 1.7 * nr), squeeze=False)
    for ax, (m, lvl) in zip(axes.ravel(), items):
        has = ((st["metric"] == m) & (st["level"] == lvl)).any()
        lvl2 = lvl if has else "slice"
        panel_bars(ax, st, long, m, lvl2, REGION_ORDER if lvl2 == "region" else ["slice"], control, case, rng)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    hdl, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(hdl, lab, frameon=False, loc="upper right", ncol=2, handlelength=0.8, bbox_to_anchor=(1.0, 1.03))
    fig.suptitle(f"Baseline genotype comparison: {control} vs {case} (mean +/- SEM, points = slices)", fontsize=8,
                 x=0.02, ha="left", y=1.02)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save_fig(fig, out_base)
    return True


# =============================================================================
# REPORT
# =============================================================================

def write_report(out: Path, st: pd.DataFrame, long: pd.DataFrame, h: Harvest, control: str, case: str, args) -> None:
    n_ctl = long.loc[long["group"] == control, "slice_id"].nunique()
    n_case = long.loc[long["group"] == case, "slice_id"].nunique()
    inferential = n_ctl >= 2 and n_case >= 2
    L = [f"# Genotype comparison at baseline ({control} vs {case}, condition {args.condition})", "",
         f"Results root: `{h.root}`  ", f"Slices: {control} n={n_ctl} ({', '.join(sorted(long.loc[long['group'] == control, 'slice_id'].unique()))}); "
         f"{case} n={n_case} ({', '.join(sorted(long.loc[long['group'] == case, 'slice_id'].unique()))})", "",
         "Unit = animal (slice). Event / channel values averaged within slice x ROI, ROI values averaged within slice x region "
         "and over all ROIs (whole slice). Two-tailed Welch t-test per metric x unit; Benjamini-Hochberg q within each family x level; "
         "q < 0.05 marked with *. Reported P values are unadjusted.", ""]
    if not inferential:
        L += ["> **Caveat:** inferential statistics (Welch t, Cohen's d, BH q) require n >= 2 slices per group; "
              f"this run has n = {n_ctl} / {n_case}, so p, q and d are NaN and only group values, differences and the "
              "standardized difference are reported. The manuscript used n = 3 mice per genotype.", ""]
    for fam in FAMILY_ORDER:
        s = st[st["family"] == fam]
        if not len(s):
            continue
        L += [f"## {FAMILY_LABELS.get(fam, fam)} (`{fam}`)", "",
              f"| Metric | Unit | {control} mean +/- SEM (n) | {case} mean +/- SEM (n) | Change | d | p | q |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            if r["level"] == "roi" and r["unit"] not in ROI_ORDER:
                continue
            def ms(m, se, n):
                return "n/a" if not np.isfinite(m) else (f"{m:.4g} +/- {se:.2g} ({n})" if np.isfinite(se) else f"{m:.4g} ({n})")
            arrow = "higher" if r["diff_case_minus_control"] > 0 else ("lower" if r["diff_case_minus_control"] < 0 else "equal")
            pct = f" ({r['pct_change']:+.1f}%)" if np.isfinite(r["pct_change"]) else ""
            star = "*" if r["significant_q"] else ""
            L.append(f"| {r['label']} | {'whole slice' if r['unit'] == 'slice' else r['unit']} | {ms(r['mean_control'], r['sem_control'], r['n_control'])} | "
                     f"{ms(r['mean_case'], r['sem_case'], r['n_case'])} | {arrow} in {case}{pct} | "
                     f"{r['cohen_d']:.2f} | {fmt_p(r['p'])} | {fmt_p(r['q'])}{star} |".replace("nan", "n/a"))
        L.append("")
    L += ["## Data provenance", ""]
    for mod in sorted(h.harvested):
        L.append(f"- **{mod}**: {', '.join(h.harvested[mod])}")
    L += ["", "Files read:", ""] + [f"- `{f}`" for f in h.files]
    if h.missing:
        L += ["", "Modules not found (skipped): " + ", ".join(h.missing)]
    if h.notes:
        L += ["", "Notes / warnings:", ""] + [f"- {n}" for n in h.notes]
    L += ["", "## Caveats", "",
          f"- n per group: {control} = {n_ctl}, {case} = {n_case}. " + ("" if inferential else "No inferential test possible (n < 2)."),
          "- Event-level tables identify recordings only by group; slice ids were resolved via event_meta.json or, when a group has a "
          "single slice, by that slice (see module docstring).",
          "- Correlation metrics are Fisher z; MPV is not log-transformed here (the LMM of the pharmacology analysis uses log10).",
          "- Avalanche power-law exponents are NaN when the source module could not fit them (insufficient avalanches)."]
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def pick_groups(groups: Iterable[str], requested: Optional[List[str]], control: Optional[str]) -> Tuple[str, str]:
    groups = sorted(set(groups))
    if requested:
        groups = [g for g in requested if g in groups] or groups
    if not groups:
        raise SystemExit("no groups found in the harvested tables")
    if control and control in groups:
        ctl = control
    else:
        ctl = next((g for g in groups if any(g.lower().startswith(p) or p in g.lower() for p in CONTROL_PATTERNS)), groups[0])
        if control:
            warnings.warn(f"--control {control} not among groups {groups}; using {ctl}")
    others = [g for g in groups if g != ctl]
    if not others:
        raise SystemExit(f"only one group found ({ctl}); two groups are required")
    if len(others) > 1:
        warnings.warn(f"more than two groups found {groups}; comparing {ctl} vs {others[0]}")
    return ctl, others[0]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default=str(RESULTS_ROOT), help="folder holding the per-module result folders")
    p.add_argument("--out_dir", default=None, help="default <results_root>/genotype_roi_comparison")
    p.add_argument("--groups", nargs="+", default=None, help="groups to compare (default: auto; first = control)")
    p.add_argument("--control", default=None, help="name of the control group (default: CTL/WT/C57-like, else first)")
    p.add_argument("--condition", default=CONDITION, help="baseline condition token (default 4AP)")
    p.add_argument("--mask_dir", default=str(MASK_DIR), help="folder of <slice_id>__contact_mask_64x64.npy (slice registry)")
    p.add_argument("--avalanche_window", default=AVALANCHE_WINDOW, help="window of avalanche_fit_summary.csv for whole-slice exponents")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (point jitter)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    root = Path(args.results_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    out = Path(args.out_dir) if args.out_dir else root / "genotype_roi_comparison"
    if not out.is_absolute():
        out = REPO_ROOT / out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    set_style()

    reg = SliceRegistry(Path(args.mask_dir))
    h = Harvest(root, reg, parse_condition(args.condition), args.avalanche_window)
    h.run_all()
    if not h.rows:
        raise SystemExit(f"nothing harvested under {root}")
    raw = pd.DataFrame(h.rows)
    if args.groups:
        raw = raw[raw["group"].isin(args.groups)]
    control, case = pick_groups(raw["group"].unique(), args.groups, args.control)
    raw = raw[raw["group"].isin([control, case])]
    print(f"[GROUPS] control={control} case={case}; slices: "
          + "; ".join(f"{g}: {sorted(raw.loc[raw['group'] == g, 'slice_id'].unique())}" for g in (control, case)))

    long = aggregate(raw)
    long.to_csv(out / "tidy_metrics_long.csv", index=False)
    wide = long.pivot_table(index=["group", "slice_id", "condition", "level", "unit"], columns="metric", values="value", aggfunc="mean")
    wide = wide.reset_index().sort_values(["level", "unit", "group", "slice_id"])
    wide.to_csv(out / "animal_level_wide.csv", index=False)

    st = compare(long, control, case)
    st.to_csv(out / "comparison_stats.csv", index=False)
    n_ctl, n_case = long.loc[long["group"] == control, "slice_id"].nunique(), long.loc[long["group"] == case, "slice_id"].nunique()
    if n_ctl < 2 or n_case < 2:
        print(f"[NOTE] n per group = {control}:{n_ctl}, {case}:{n_case}. Welch t-test / Cohen's d / BH q are undefined "
              "with n < 2 per group -> written as NaN; group values and differences are still reported "
              "(the manuscript used n = 3 per genotype).")
    else:
        print(f"[STATS] {int(st['significant_q'].sum())} of {int(st['p'].notna().sum())} tests with q < {Q_ALPHA}")

    figs = []
    for fam in FAMILY_ORDER:
        if fig_family(st, long, fam, "region", control, case, out / "figures" / f"{fam}__regions", rng):
            figs.append(f"figures/{fam}__regions")
        if (st[(st["family"] == fam) & (st["level"] == "roi")]).shape[0] and \
                fig_family(st, long, fam, "roi", control, case, out / "figures" / f"{fam}__rois", rng):
            figs.append(f"figures/{fam}__rois")
    fig_heatmap(st, control, case, out / "figures" / "effect_size_heatmap"); figs.append("figures/effect_size_heatmap")
    if fig_overview(st, long, control, case, out / "figures" / "overview_headline", rng):
        figs.append("figures/overview_headline")

    write_report(out, st, long, h, control, case, args)
    manifest = dict(module="genotype_roi_comparison", args=vars(args), results_root=str(root), out_dir=str(out),
                    control=control, case=case, condition=h.cond, n_slices={control: int(n_ctl), case: int(n_case)},
                    slices={g: sorted(long.loc[long["group"] == g, "slice_id"].unique()) for g in (control, case)},
                    families=FAMILY_ORDER, metrics_by_module=h.harvested, files_read=h.files, modules_missing=h.missing,
                    notes=h.notes, figures=figs, statistics=dict(test="Welch two-tailed t-test (scipy ttest_ind, equal_var=False)",
                    effect_size="Cohen's d (pooled s.d.); std_diff = diff / mean(|means|)", fdr="Benjamini-Hochberg within family x level",
                    q_alpha=Q_ALPHA, statsmodels=HAVE_SM), n_tests=int(len(st)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] {len(long)} animal-level rows, {len(st)} comparisons, {len(figs)} figures -> {out}")
    if h.missing:
        print("[WARN] modules skipped: " + ", ".join(h.missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
