# -*- coding: utf-8 -*-
"""
run_baseline.py -- one-command SPACE pipeline for 4-AP baseline recordings (genotype x ROI comparison).

Workflow (each stage is a subcommand; ``all`` chains them; everything is driven by data/slices.csv):

    data/raw_brw/<name>.brw  +  data/slice_photos/<slice_id>.<jpg|png|tif>
        | init      scan the two folders, match photo stem inside the brw name, guess the group
        |           (c57/ctl/wt -> CTL, 5xfad/ad/app -> 5xFAD) and write data/slices.csv
        |           (columns slice_id,group,brw,photo; EDIT the group column if the guess is wrong)
        | mask      preprocessing/tissue_registration_mask.py  (interactive: click 4 array corners)
        |               -> data/masks/<slice_id>__contact_mask_64x64.npy, <slice_id>__rectified_grid.png
        | convert   preprocessing/brw_to_2ms_h5.py  -> data/processed_2ms/<group>/<brw_stem>_2ms.h5
        |               (renamed from *_2ms_optimized.h5; /meta drug='4AP' if no drug token, genotype=<group>)
        | rois      preprocessing/pick_rois.py  (interactive: click M1, M2, S1, CA1, CA3, DG)
        |               -> data/roi_definitions/<slice_id>_roi_definitions.csv
        | analyze   every analysis module, same flags as examples/run_example.sh but on data/:
        |             whole-dataset : mask_noise_reference, extract_event_epochs, regional_csd_dynamics,
        |                             psd_wavelet_metrics, electrode_connectivity, ctx_hippo_compare, welch_psd
        |             per slice     : roi_lfp_csd_metrics, roi_phase_gradient_pgd, phase_kuramoto,
        |                             avalanche_statistics, pick_representative_recording
        |             then          : stage kuramoto tables, roc_scalability_sensitivity
        |           -> data/results/<module>[/<group>] ; logs in data/results/logs/<step>.log
        | compare   core/genotype_roi_comparison.py --results_root data/results
        v
    data/results/genotype_roi_comparison/

Examples (Windows cmd.exe or bash, from the repo root):
    python run_baseline.py all
    python run_baseline.py init --force
    python run_baseline.py analyze --only mask_noise_reference,regional_csd_dynamics --force
    python run_baseline.py analyze --skip roc_scalability_sensitivity --keep_going
    python run_baseline.py --data_root data/example_synthetic analyze --quick --dry_run

Implementation notes
    * Steps are skipped when their main output already exists (``--force`` reruns); the per-slice
      scripts are filtered with ``--file_pattern``/``--pattern "<slice_id>*_2ms.h5"``.
    * roi_lfp_csd_metrics / roi_phase_gradient_pgd write per-condition folders, so with more than one
      slice per group their outputs would collide under data/results/<module>/<group>/. Pass
      ``--per_slice_dirs`` to write data/results/<module>/<group>/<slice_id>/ instead.
    * No shell=True, pathlib everywhere, subprocess.run([sys.executable, script, ...], cwd=repo root).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
PY = sys.executable
GROUP_RULES = [("5xfad", "5xFAD"), ("app", "5xFAD"), ("c57", "CTL"), ("ctl", "CTL"), ("wt", "CTL"), ("ad", "5xFAD")]
DRUG_TOKENS = ("4AP", "AFDX", "BIC", "CNQX", "ML297")
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------------- paths / slice table
class Layout:
    """All paths derived from --data_root (relative to the repo root)."""

    def __init__(self, data_root: str):
        self.root = (REPO_ROOT / data_root).resolve() if not Path(data_root).is_absolute() else Path(data_root)
        self.slices_csv = self.root / "slices.csv"
        self.raw_brw = self.root / "raw_brw"
        self.photos = self.root / "slice_photos"
        self.masks = self.root / "masks"
        self.processed = self.root / "processed_2ms"
        self.rois = self.root / "roi_definitions"
        self.events = self.root / "event_traces"
        self.psd_tables = self.root / "psd_tables"
        self.kuramoto_tables = self.root / "kuramoto_tables"
        self.results = self.root / "results"
        self.logs = self.results / "logs"

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    def mask_npy(self, sid: str) -> Path:
        return self.masks / f"{sid}__contact_mask_64x64.npy"

    def roi_csv(self, sid: str) -> Path:
        return self.rois / f"{sid}_roi_definitions.csv"

    def h5_files(self, sid: str, group: str) -> list[Path]:
        return sorted((self.processed / group).glob(f"{sid}*_2ms.h5"))


def guess_group(name: str) -> str:
    n = name.lower()
    for tok, grp in GROUP_RULES:
        if re.search(rf"(^|[^a-z]){re.escape(tok)}([^a-z]|$)", n) or (len(tok) > 2 and tok in n):
            return grp
    return "UNKNOWN"


def load_slices(lay: Layout) -> pd.DataFrame:
    if not lay.slices_csv.exists():
        sys.exit(f"[ERROR] {lay.rel(lay.slices_csv)} not found: run `python run_baseline.py init` first "
                 f"(or create it by hand with columns slice_id,group,brw,photo).")
    df = pd.read_csv(lay.slices_csv, dtype=str).fillna("")
    need = {"slice_id", "group", "brw", "photo"}
    if not need.issubset(df.columns):
        sys.exit(f"[ERROR] {lay.slices_csv} must have columns {sorted(need)}")
    if (df.group == "UNKNOWN").any():
        print(f"[WARN] some rows in {lay.rel(lay.slices_csv)} have group=UNKNOWN; edit the file before `analyze`.")
    return df


def groups_of(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(g for g in df.group if g))


# ----------------------------------------------------------------------------- step runner
@dataclass
class Step:
    name: str
    cmd: list[str]                                   # [] = python-side action only (see post)
    main_output: Path | None = None                  # skip if exists (unless --force)
    post: Callable[[], None] | None = None           # python callback after a successful run
    interactive: bool = False                        # stream to console instead of log file
    status: str = "pending"
    seconds: float = 0.0


def _exists(p: Path | None) -> bool:
    if p is None:
        return False
    if p.is_dir():
        return any(p.iterdir())
    return p.exists()


def run_steps(steps: list[Step], lay: Layout, *, dry_run: bool, force: bool = False, keep_going: bool = False,
              only: str = "", skip: str = "") -> bool:
    only_set = {s.strip() for s in only.split(",") if s.strip()}
    skip_set = {s.strip() for s in skip.split(",") if s.strip()}

    def selected(st: Step) -> bool:
        base = st.name.split("__")[0]
        if only_set and not ({st.name, base} & only_set):
            return False
        return not ({st.name, base} & skip_set)

    lay.logs.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for st in steps:
        if not selected(st):
            st.status = "deselected"
            continue
        print(f"\n### {st.name}")
        if _exists(st.main_output) and not force:
            print(f"    [SKIP] output exists: {lay.rel(st.main_output)}")
            st.status = "skipped"
            continue
        shown = subprocess.list2cmdline(["python", *st.cmd[1:]]) if st.cmd else "(python-side action)"
        print(f"    {shown}")
        if dry_run:
            st.status = "dry-run"
            continue
        t0 = time.time()
        try:
            if st.cmd:
                if st.interactive:
                    res = subprocess.run(st.cmd, cwd=REPO_ROOT)
                else:
                    log = lay.logs / f"{st.name}.log"
                    with open(log, "w", encoding="utf-8", errors="replace") as fh:
                        res = subprocess.run(st.cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)
                    if res.returncode != 0:
                        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
                        print("    ---- last log lines ----\n    " + "\n    ".join(tail))
                if res.returncode != 0:
                    raise RuntimeError(f"exit code {res.returncode}")
            if st.post:
                st.post()
            st.status = "ok"
        except Exception as e:  # noqa: BLE001
            st.status = f"FAILED ({e})"
            ok_all = False
            print(f"    [FAIL] {st.name}: {e}" + (f"  (log: {lay.rel(lay.logs / (st.name + '.log'))})" if st.cmd and not st.interactive else ""))
            if not keep_going:
                break
        finally:
            st.seconds = time.time() - t0
            if st.status == "ok":
                print(f"    [OK] {st.seconds:.1f} s")
    print_summary(steps)
    return ok_all


def print_summary(steps: list[Step]) -> None:
    rows = [(s.name, s.status, f"{s.seconds:.1f}") for s in steps]
    w = max(len(r[0]) for r in rows) if rows else 10
    print("\n" + "=" * (w + 44))
    print(f"{'step':<{w}}  {'status':<28} {'time (s)':>8}")
    print("-" * (w + 44))
    for n, s, t in rows:
        print(f"{n:<{w}}  {s[:28]:<28} {t:>8}")
    print("=" * (w + 44))


# ----------------------------------------------------------------------------- subcommands
def cmd_init(lay: Layout, args) -> None:
    if lay.slices_csv.exists() and not args.force:
        print(f"[init] {lay.rel(lay.slices_csv)} exists (use --force to rebuild). Current table:")
        print(pd.read_csv(lay.slices_csv, dtype=str).to_string(index=False))
        return
    brws = sorted(lay.raw_brw.glob("*.brw"))
    photos = sorted(p for p in lay.photos.iterdir() if p.suffix.lower() in PHOTO_EXT) if lay.photos.exists() else []
    if not brws:
        sys.exit(f"[ERROR] no .brw files in {lay.rel(lay.raw_brw)}")
    rows = []
    for brw in brws:
        cands = [p for p in photos if p.stem.lower() in brw.stem.lower()]
        photo = max(cands, key=lambda p: len(p.stem)) if cands else None   # longest matching stem wins
        sid = photo.stem if photo else brw.stem
        if not photo:
            print(f"[WARN] no photo whose stem is contained in '{brw.name}'; slice_id set to the brw stem")
        rows.append(dict(slice_id=sid, group=guess_group(brw.stem), brw=lay.rel(brw).replace("\\", "/"),
                         photo=lay.rel(photo).replace("\\", "/") if photo else ""))
    df = pd.DataFrame(rows)
    if args.dry_run:
        print(df.to_string(index=False)); return
    lay.root.mkdir(parents=True, exist_ok=True)
    df.to_csv(lay.slices_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n[init] wrote {lay.rel(lay.slices_csv)}  -- check the 'group' column (CTL / 5xFAD) and edit if needed.")


def cmd_mask(lay: Layout, args) -> bool:
    df = load_slices(lay)
    steps = []
    for r in df.itertuples():
        if not r.photo:
            print(f"[WARN] {r.slice_id}: no photo in slices.csv; cannot build a mask"); continue
        steps.append(Step(f"mask__{r.slice_id}",
                          [PY, "preprocessing/tissue_registration_mask.py", "--image", r.photo, "--out", lay.rel(lay.masks)],
                          lay.mask_npy(r.slice_id), interactive=True))
    return run_steps(steps, lay, dry_run=args.dry_run, force=args.force)


def cmd_convert(lay: Layout, args) -> bool:
    import h5py

    df = load_slices(lay)
    steps = []
    for r in df.itertuples():
        brw = REPO_ROOT / r.brw
        out_dir = lay.processed / r.group
        final_h5 = out_dir / f"{brw.stem}_2ms.h5"

        def post(brw=brw, out_dir=out_dir, final_h5=final_h5, group=r.group):
            opt = out_dir / f"{brw.stem}_2ms_optimized.h5"
            if opt.exists():
                opt.replace(final_h5)
            if not final_h5.exists():
                raise FileNotFoundError(f"expected {final_h5}")
            with h5py.File(final_h5, "a") as f:
                m = f.require_group("meta").attrs
                if str(m.get("drug", "")).upper() not in DRUG_TOKENS:   # no condition token -> 4-AP baseline
                    m["drug"] = "4AP"
                m["genotype"] = group
                drug = m["drug"]
            print(f"    -> {lay.rel(final_h5)}  (drug={drug}, genotype={group})")

        cmd = [PY, "preprocessing/brw_to_2ms_h5.py", "--brw_dir", lay.rel(brw.parent), "--only", brw.stem,
               "--mask", lay.rel(lay.mask_npy(r.slice_id)), "--out", lay.rel(out_dir)]
        if args.stride:
            cmd += ["--bin_stride_mode", args.stride]
        existing = lay.h5_files(r.slice_id, r.group)
        steps.append(Step(f"convert__{r.slice_id}", cmd, existing[0] if existing else final_h5, post=post))
    return run_steps(steps, lay, dry_run=args.dry_run, force=args.force, keep_going=args.keep_going)


def cmd_rois(lay: Layout, args) -> bool:
    df = load_slices(lay)
    steps = [Step(f"rois__{r.slice_id}",
                  [PY, "preprocessing/pick_rois.py", "--slice_id", r.slice_id, "--mask_dir", lay.rel(lay.masks),
                   "--roi_dir", lay.rel(lay.rois)] + (["--redo"] if args.force else []),
                  lay.roi_csv(r.slice_id), interactive=True) for r in df.itertuples()]
    return run_steps(steps, lay, dry_run=args.dry_run, force=args.force)


def build_analysis_steps(lay: Layout, df: pd.DataFrame, args) -> list[Step]:
    R, rel = lay.results, lay.rel
    groups = ",".join(groups_of(df))
    P, M, RO, E = rel(lay.processed), rel(lay.masks), rel(lay.rois), rel(lay.events)
    steps = [
        Step("mask_noise_reference", [PY, "utils/mask_noise_reference.py", "--in_root", P, "--mask_dir", M,
                                      "--out_dir", rel(R / "mask_noise_reference")], R / "mask_noise_reference"),
        Step("extract_event_epochs", [PY, "preprocessing/extract_event_epochs.py", "--in_root", P, "--mask_dir", M,
                                      "--roi_dir", RO, "--out_root", E, "--results_dir", rel(R / "event_epochs")]
             + (["--min_event_size", str(args.min_event_size)] if args.min_event_size is not None else []),
             R / "event_epochs" / "event_catalog.csv"),
        Step("regional_csd_dynamics", [PY, "core/regional_csd_dynamics.py", "--in_root", P, "--mask_dir", M,
                                       "--roi_dir", RO, "--out_dir", rel(R / "regional_csd_dynamics")],
             R / "regional_csd_dynamics" / "roi_metrics.csv"),
        Step("psd_wavelet_metrics", [PY, "core/psd_wavelet_metrics.py", "--root_dir", E,
                                     "--out_dir", rel(R / "psd_wavelet_metrics"), "--groups", groups],
             R / "psd_wavelet_metrics" / "manifest.json"),
        Step("electrode_connectivity", [PY, "core/electrode_connectivity.py", "--root_dir", E,
                                        "--out_dir", rel(R / "electrode_connectivity"), "--groups", groups],
             R / "electrode_connectivity" / "manifest.json"),
        Step("ctx_hippo_avalanche_kuramoto_compare", [PY, "core/ctx_hippo_avalanche_kuramoto_compare.py", "--root_dir", E,
                                                      "--out_dir", rel(R / "ctx_hippo_avalanche_kuramoto_compare"), "--groups", groups],
             R / "ctx_hippo_avalanche_kuramoto_compare" / "manifest.json"),
        Step("welch_psd_recordings", [PY, "benchmark/welch_psd_recordings.py", "--in_root", P, "--mask_dir", M,
                                      "--out_dir", rel(lay.psd_tables), "--figs_dir", rel(R / "welch_psd")],
             lay.psd_tables / "psd_bandpower_summary.csv"),
    ]
    # ---- per-slice (whole-slice scripts take one mask / one ROI csv per call)
    for r in df.itertuples():
        sid, g = r.slice_id, r.group
        gdir, mask, roi = rel(lay.processed / g), rel(lay.mask_npy(sid)), rel(lay.roi_csv(sid))
        pat = f"{sid}*_2ms.h5"
        recs = [h.stem for h in lay.h5_files(sid, g)] or [f"{sid}_2ms"]
        sub = (lambda mod: R / mod / g / sid) if args.per_slice_dirs else (lambda mod: R / mod / g)
        steps += [
            Step(f"roi_lfp_csd_metrics__{sid}", [PY, "core/roi_lfp_csd_metrics.py", "--data_dir", gdir, "--mask", mask,
                                                 "--roi_csv", roi, "--genotype-label", g, "--file_pattern", pat,
                                                 "--out_dir", rel(sub("roi_lfp_csd_metrics"))],
                 sub("roi_lfp_csd_metrics") / "all_lfp_amplitude_energy_slope_metrics.csv"),
            Step(f"roi_phase_gradient_pgd__{sid}", [PY, "core/roi_phase_gradient_pgd.py", "--data_dir", gdir, "--mask", mask,
                                                    "--roi_csv", roi, "--file_pattern", pat,
                                                    "--out_dir", rel(sub("roi_phase_gradient_pgd"))],
                 sub("roi_phase_gradient_pgd") / "summary.csv"),
            Step(f"phase_kuramoto__{sid}", [PY, "core/phase_kuramoto.py", "--in_dir", gdir, "--mask", mask, "--pattern", pat,
                                            "--out_root", rel(R / "phase_kuramoto" / g)],
                 R / "phase_kuramoto" / g / recs[0] / "tables" / "kuramoto_ALL.csv"),
            Step(f"avalanche_statistics__{sid}", [PY, "core/avalanche_statistics.py", "--in_dir", gdir, "--mask", mask,
                                                  "--pattern", pat, "--out_root", rel(R / "avalanche_statistics" / g)],
                 R / "avalanche_statistics" / g / recs[0]),
            Step(f"pick_representative_recording__{sid}", [PY, "utils/pick_representative_recording.py", "--in_dir", gdir,
                                                           "--mask", mask, "--pattern", pat,
                                                           "--out_dir", rel(R / "pick_representative_recording" / g)],
                 R / "pick_representative_recording" / g / "_stripe_report.tsv"),
        ]

    def stage_kuramoto():
        import shutil
        n = 0
        for tdir in sorted((R / "phase_kuramoto").glob("*/*/tables")):
            dst = lay.kuramoto_tables / tdir.parent.name / "tables"
            dst.mkdir(parents=True, exist_ok=True)
            for fn in ("kuramoto_ALL.csv", "kuramoto_ALL_summary.json"):
                if (tdir / fn).exists():
                    shutil.copy2(tdir / fn, dst / fn); n += 1
        print(f"    staged {n} kuramoto files -> {rel(lay.kuramoto_tables)}")

    steps.append(Step("stage_kuramoto_tables", [], lay.kuramoto_tables, post=stage_kuramoto))
    steps.append(Step("roc_scalability_sensitivity",
                      [PY, "benchmark/roc_scalability_sensitivity.py", "--space_root", rel(R / "ctx_hippo_avalanche_kuramoto_compare"),
                       "--lfp_csd_root", rel(R / "psd_wavelet_metrics"),
                       "--event_roots", ",".join(rel(lay.events / g) for g in groups_of(df)),
                       "--out_dir", rel(R / "roc_scalability_sensitivity")] + (["--quick"] if args.quick else []),
                      R / "roc_scalability_sensitivity" / "manifest.json"))
    return steps


def cmd_analyze(lay: Layout, args) -> bool:
    df = load_slices(lay)
    steps = build_analysis_steps(lay, df, args)
    return run_steps(steps, lay, dry_run=args.dry_run, force=args.force, keep_going=args.keep_going,
                     only=args.only, skip=args.skip)


def cmd_compare(lay: Layout, args) -> bool:
    script = REPO_ROOT / "core" / "genotype_roi_comparison.py"
    if not script.exists():
        print(f"[compare] {lay.rel(script)} does not exist -- nothing to run.")
        return False
    groups = sorted(load_slices(lay)["group"].unique().tolist()) if lay.slices_csv.exists() else []
    cmd = [PY, "core/genotype_roi_comparison.py", "--results_root", lay.rel(lay.results),
           "--mask_dir", lay.rel(lay.masks)]
    if groups:
        ctl = [g for g in groups if g.upper().startswith(("CTL", "WT", "C57", "CONTROL"))]
        groups = ctl + [g for g in groups if g not in ctl]      # control first
        cmd += ["--groups", *groups]
    steps = [Step("genotype_roi_comparison", cmd, lay.results / "genotype_roi_comparison" / "REPORT.md")]
    return run_steps(steps, lay, dry_run=args.dry_run, force=True)


def cmd_all(lay: Layout, args) -> bool:
    if not lay.slices_csv.exists():
        cmd_init(lay, args)
        if args.dry_run and not lay.slices_csv.exists():
            print("[all] dry run: slices.csv not written, stopping here."); return True
    for fn in (cmd_mask, cmd_convert, cmd_rois, cmd_analyze, cmd_compare):
        if not fn(lay, args) and not args.keep_going:
            print(f"\n[all] stopped at stage '{fn.__name__[4:]}' (fix the error and rerun `python run_baseline.py all`).")
            return False
    return True


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    glob_opts = argparse.ArgumentParser(add_help=False)
    # SUPPRESS so that a subcommand's copy of the option never overrides a value given before the subcommand
    glob_opts.add_argument("--data_root", default=argparse.SUPPRESS, help="data folder (relative to repo root); default: data")
    glob_opts.add_argument("--dry_run", action="store_true", default=argparse.SUPPRESS,
                           help="print the commands without running them")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], parents=[glob_opts],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Examples" + __doc__.split("Examples", 1)[1])
    sp = p.add_subparsers(dest="cmd", required=True)

    def common(q, *, analysis=False, convert=False, init=False):
        q.add_argument("--force", action="store_true", help="rerun steps whose output exists" + (" / rebuild slices.csv" if init else ""))
        q.add_argument("--keep_going", action="store_true", help="continue after a failed step")
        if convert:
            q.add_argument("--stride", choices=["rolling", "nonoverlap"], default=None, help="brw_to_2ms_h5 --bin_stride_mode")
        if analysis:
            q.add_argument("--only", default="", help="comma-separated step names to run (base name or name__slice)")
            q.add_argument("--skip", default="", help="comma-separated step names to skip")
            q.add_argument("--quick", action="store_true", help="pass --quick to roc_scalability_sensitivity (smoke test)")
            q.add_argument("--min_event_size", type=int, default=None, help="passthrough to extract_event_epochs")
            q.add_argument("--per_slice_dirs", action="store_true",
                           help="write roi_lfp_csd_metrics / roi_phase_gradient_pgd under results/<module>/<group>/<slice_id>")

    common(sp.add_parser("init", parents=[glob_opts], help="scan raw_brw/ + slice_photos/ and write slices.csv"), init=True)
    common(sp.add_parser("mask", parents=[glob_opts], help="tissue-contact mask per slice (interactive)"))
    common(sp.add_parser("convert", parents=[glob_opts], help=".brw -> processed_2ms/<group>/*_2ms.h5"), convert=True)
    common(sp.add_parser("rois", parents=[glob_opts], help="pick 6 ROIs per slice (interactive)"))
    common(sp.add_parser("analyze", parents=[glob_opts], help="run every analysis module"), analysis=True)
    common(sp.add_parser("compare", parents=[glob_opts], help="run core/genotype_roi_comparison.py"))
    common(sp.add_parser("all", parents=[glob_opts], help="init -> mask -> convert -> rois -> analyze -> compare"), analysis=True, convert=True, init=True)
    args = p.parse_args(argv)
    args.data_root = getattr(args, "data_root", "data")
    args.dry_run = getattr(args, "dry_run", False)

    lay = Layout(args.data_root)
    print(f"[run_baseline] repo={REPO_ROOT}  data_root={lay.rel(lay.root)}  {'(DRY RUN)' if args.dry_run else ''}")
    fn = {"init": cmd_init, "mask": cmd_mask, "convert": cmd_convert, "rois": cmd_rois,
          "analyze": cmd_analyze, "compare": cmd_compare, "all": cmd_all}[args.cmd]
    ok = fn(lay, args)
    return 0 if ok in (True, None) else 1


if __name__ == "__main__":
    sys.exit(main())
