"""Regenerate every result, figure, table and the manuscript of the study (rule 6).

Stages run in the order of :data:`STAGES`. The finite element stages are
resumable. A one-shot stage (cohort, verification, pilot) whose output files
all exist is skipped (``--force`` re-runs it). The long stages (cohort runs,
sensitivity, Sobol, fields, 3D length series) always start and resume inside
the stage, reusing every finished lesion or run. The
analysis stages (outcome table, metrics, consistency checks, figures, tables,
manuscript) always run, since they only read results files.

Usage::

    python run_all.py                    # everything (resume)
    python run_all.py --list             # stages, outputs and status
    python run_all.py --only metrics figures tables manuscript
    python run_all.py --from cohort_runs --force

Every run appends a record (stage, status, seconds, git commit, software
versions) to ``results/run_all_log.json``. All randomness uses
``design.seed`` (plus a per-stage offset) inside the stages.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))
import conftest  # noqa: E402,F401  (imports dtivl from this checkout's src/)

RESULTS = ROOT / "results"


# --------------------------------------------------------------------- stages
def _cohort():
    from dtivl.drivers import stage_cohort
    stage_cohort.run()


def _verification_lame():
    from dtivl.drivers import stage_verification as V
    V.lame()


def _verification_phasefield():
    from dtivl.fracture.phasefield import verification_phasefield
    verification_phasefield()


def _verification_nonlinear():
    from dtivl.solvers.nonlinear import verification_nonlinear
    verification_nonlinear()


def _mesh_convergence():
    from dtivl.drivers import stage_verification as V
    V.mesh_convergence()


def _foundation():
    from dtivl.drivers import stage_verification as V
    V.foundation_sensitivity()


def _calibration_stent():
    from dtivl.drivers import stage_pilot
    stage_pilot.identify_stent_stiffness()


def _pilot_linear_transient():
    from dtivl.drivers import stage_pilot
    stage_pilot.run_linear_and_transient()


#: part files written by parallel pilot processes (Checkpoint 2); folded in before resuming
PILOT_PARTS = ("part2", "part3", "stent365")


def _pilot_nonlinear():
    from dtivl.drivers import stage_pilot
    for lid in stage_pilot.pilot_ids():
        if (RESULTS / f"pilot_nonlinear_{lid}.json").exists():
            stage_pilot.merge_nonlinear_parts(lid, parts=PILOT_PARTS)
    stage_pilot.run_nonlinear(resume=True)
    stage_pilot.merge_nonlinear()


def _example3d():
    from dtivl.drivers import stage_pilot
    if any((RESULTS / f"{p}.json").exists() for p in ("pilot_example3d_part2", "pilot_example3d_part3",
                                                       "pilot_example3d_part4")):
        stage_pilot.merge_example3d()
    stage_pilot.run_example3d(resume=True)
    stage_pilot.merge_example3d()
    stage_pilot.run_example3d_onset()


def _diagnostics():
    from dtivl.drivers import stage_diagnostics as D
    D.localisation()
    D.ell_sensitivity()
    D.rankine_test()


def _cohort_runs():
    from dtivl.drivers import stage_cohort_runs as C
    C.run()


def _cohort_outcomes():
    from dtivl.drivers import stage_cohort_runs as C
    C.aggregate()


def _sensitivity_oat():
    from dtivl.drivers import stage_sensitivity as S
    S.one_at_a_time()
    S.collect_oat()


def _sensitivity_morris():
    from dtivl.drivers import stage_sensitivity as S
    S.morris()
    S.analyze()


def _sobol():
    from dtivl.drivers import stage_sobol
    stage_sobol.run()
    stage_sobol.analyze()


def _fields():
    from dtivl.drivers import stage_fields
    stage_fields.run()


def _length3d():
    from dtivl.drivers import stage_length3d
    stage_length3d.run()


def _metrics():
    from dtivl.drivers import stage_metrics
    stage_metrics.run()


def _modulus_window():
    from dtivl.drivers import stage_modulus_window
    stage_modulus_window.run()
    stage_modulus_window.analyze()


def _consistency():
    from dtivl.drivers import stage_consistency
    stage_consistency.run()


def _figures():
    from dtivl.drivers import figs
    print(json.dumps(figs.make_all(), indent=1))


def _tables():
    from dtivl.drivers import tables
    print(json.dumps(tables.make_all(), indent=1))


def _manuscript():
    subprocess.run([sys.executable, str(ROOT / "manuscript" / "build_manuscript.py")], check=True, cwd=ROOT)


@dataclass
class Stage:
    name: str
    fn: object
    outputs: tuple = ()          # paths relative to the project root; all present -> stage skipped
    always: bool = False         # analysis stages: always run
    note: str = ""
    extra: dict = field(default_factory=dict)

    def done(self) -> bool:
        return bool(self.outputs) and all((ROOT / p).exists() for p in self.outputs)


STAGES = [
    Stage("cohort", _cohort, ("cohort/cohort.csv", "results/cohort_summary.json"), note="lesion cohort (masks, cohort.csv)"),
    Stage("verification_lame", _verification_lame, ("results/verification_lame.json",)),
    Stage("verification_phasefield", _verification_phasefield, ("results/verification_phasefield.json",)),
    Stage("verification_nonlinear", _verification_nonlinear, ("results/verification_nonlinear.json",)),
    Stage("mesh_convergence", _mesh_convergence, ("results/verification_mesh_convergence.json",)),
    Stage("foundation", _foundation, ("results/verification_foundation_linear.json",)),
    Stage("calibration_stent", _calibration_stent, ("results/calibration_stent.json",)),
    Stage("pilot_linear_transient", _pilot_linear_transient, ("results/pilot_linear_transient.json",)),
    Stage("pilot_nonlinear", _pilot_nonlinear, ("results/pilot_nonlinear.json",), note="Checkpoint 2 pilot, E sweep"),
    Stage("example3d", _example3d, ("results/pilot_example3d.json", "results/pilot_example3d_onset.json"),
          note="D1 3D example (hours)"),
    Stage("diagnostics", _diagnostics, ("results/diag_localisation.json", "results/diag_ell.json",
                                        "results/diag_rankine.json")),
    Stage("cohort_runs", _cohort_runs, note="72 lesions x 6 protocols (days on 4 cores; resumes per protocol)"),
    Stage("cohort_outcomes", _cohort_outcomes, always=True, note="results/cohort_outcomes.csv"),
    Stage("sensitivity_oat", _sensitivity_oat, note="39 runs; resumes per run"),
    Stage("sensitivity_morris", _sensitivity_morris, note="216 runs; resumes per run"),
    Stage("sobol", _sobol, note="linear predictors; resumes per lesion"),
    Stage("fields", _fields, note="figure fields; resumes per lesion"),
    Stage("length3d", _length3d, note="D12 3D length series (hours); resumes per run"),
    Stage("metrics", _metrics, always=True),
    Stage("modulus_window", _modulus_window, always=True,
          note="D18 post hoc: core IVL runs at two moduli in the calibration window (96 runs; resumes per run)"),
    Stage("consistency", _consistency, always=True),
    Stage("figures", _figures, always=True),
    Stage("tables", _tables, always=True),
    Stage("manuscript", _manuscript, always=True),
]
_BY_NAME = {s.name: s for s in STAGES}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:          # not a git checkout
        return None


def _versions() -> dict:
    from importlib import metadata

    out = {"python": sys.version.split()[0]}
    for mod, dist in (("dolfinx", "fenics-dolfinx"), ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                      ("sklearn", "scikit-learn"), ("SALib", "SALib"), ("matplotlib", "matplotlib"), ("gmsh", "gmsh")):
        try:
            out[mod] = getattr(__import__(mod), "__version__", None) or metadata.version(dist)
        except Exception:
            out[mod] = None
    return out


def _log(rows: list[dict]) -> None:
    path = RESULTS / "run_all_log.json"
    old = json.loads(path.read_text()) if path.exists() else []
    old.append({"started": rows[0]["started"] if rows else None, "git_commit": _git_commit(),
                "versions": _versions(), "stages": rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, indent=1))


def select(only=None, skip=(), start=None) -> list[Stage]:
    for n in list(only or []) + list(skip) + ([start] if start else []):
        if n not in _BY_NAME:
            raise SystemExit(f"unknown stage {n!r}; see --list")
    names = [s.name for s in STAGES]
    if start:
        names = names[names.index(start):]
    if only:
        names = [n for n in names if n in only]
    return [_BY_NAME[n] for n in names if n not in skip]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the stages and whether their outputs exist")
    ap.add_argument("--only", nargs="+", metavar="STAGE")
    ap.add_argument("--skip", nargs="+", default=[], metavar="STAGE")
    ap.add_argument("--from", dest="start", metavar="STAGE", help="start at this stage")
    ap.add_argument("--force", action="store_true", help="re-run stages whose outputs exist")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    a = ap.parse_args(argv)
    stages = select(a.only, a.skip, a.start)
    if a.list:
        for s in STAGES:
            state = "always" if s.always else ("done" if s.done() else ("resumable" if not s.outputs else "missing"))
            print(f"{s.name:26s} {state:9s} {', '.join(s.outputs) or '-'}  {s.note}")
        return 0
    rows, rc = [], 0
    for s in stages:
        row = {"stage": s.name, "started": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        if s.done() and not (a.force or s.always):
            row["status"] = "skipped (outputs exist)"
            rows.append(row)
            print(f"[skip] {s.name}", flush=True)
            continue
        print(f"[run ] {s.name}", flush=True)
        t0 = time.perf_counter()
        try:
            s.fn()
            row["status"] = "ok"
        except Exception as exc:  # report and stop (or continue with --keep-going)
            row["status"] = f"failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()
            rc = 1
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        if rc and not a.keep_going:
            break
    _log(rows)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
