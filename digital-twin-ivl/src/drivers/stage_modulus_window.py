"""Stage: core IVL runs at two calcium moduli inside the calibration window (decision D18, post hoc).

The calibration (D8) set the calcium modulus to the stiffest pilot-sweep value at which the IVL balloon
alone did not damage the pilot lesions while IVL did; the window met by that criterion was a single
sweep point. This stage repeats, for every core lesion, the outcome model (IVL and the IVL balloon alone,
:func:`dtivl.drivers.stage_sensitivity.run_model`, with the transient transfer factor recomputed) and the
pre-treatment predictor analysis (:func:`dtivl.drivers.stage_sobol.evaluate`) at the two log-midpoints
between the baseline modulus and its neighbouring sweep moduli (:func:`window_moduli`), with every other
input at its baseline. :func:`analyze` reports, per modulus, the outcome counts and the core AUC of the
best single mechanics predictor of the main analysis and of the Fujino score, with bootstrap intervals,
in ``results/modulus_window.json``. Everything here is post hoc (added after the first draft, with the
outcomes known) and is reported as such.

Runs: one file per (lesion, modulus) in ``results/modulus_window_runs/``; each is claimed with a lock
file (:func:`dtivl.drivers.common.claim_lock`), so several worker processes can run concurrently;
complete runs are skipped. Command line: ``python -m dtivl.drivers.stage_modulus_window {run,analyze}``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from dtivl.drivers.common import RESULTS_DIR, _to_jsonable

log = logging.getLogger(__name__)

RUN_DIR = "modulus_window_runs"
OUT_JSON = "modulus_window.json"
NOTE = ("POST HOC (decision D18, added after the first draft with the outcomes known): core IVL and "
        "balloon-alone runs and predictor analyses at the log-midpoints between the calibrated calcium modulus "
        "(D8) and its neighbouring pilot-sweep moduli; all other inputs at baseline")


def window_moduli() -> list[float]:
    """Log-midpoints between the baseline calcium modulus and its neighbours in the pilot sweep (MPa)."""
    from dtivl.drivers.stage_pilot import e_sweep
    from dtivl.materials.constitutive import MaterialSet

    base = MaterialSet.baseline().ca_E
    sweep = sorted(e_sweep())
    i = min(range(len(sweep)), key=lambda k: abs(sweep[k] - base))
    out = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(sweep):
            out.append(round(math.sqrt(sweep[j] * base), 6))
    return out


def core_ids() -> list[str]:
    from dtivl.drivers.stage_cohort_runs import cohort_ids
    return cohort_ids(("core",))


def _stem(lid: str, E: float) -> str:
    return f"{lid}_E{E:.3f}"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def run_one(lid: str, E: float) -> dict:
    """Predictors and outcome model of lesion ``lid`` at calcium modulus ``E`` (MPa)."""
    from dtivl.drivers import stage_sensitivity as S
    from dtivl.drivers import stage_sobol as SB

    plm, info, md = S.lesion_mesh(lid)
    inputs = S.ModelInputs.baseline().set(ca_E=E)
    t0 = time.perf_counter()
    pred = SB.evaluate(md, plm, inputs)
    rec = S.run_model(md, plm, inputs)
    rec.update(lesion=lid, ca_E_MPa=E, predictors=pred, mesh=info, seconds=time.perf_counter() - t0,
               complete="error" not in rec)
    rec["outputs"] = S.outputs(rec)
    return rec


def run(lesions: list[str] | None = None, out_dir: str | Path | None = None, max_runs: int | None = None) -> list:
    """Run every (core lesion, window modulus) not yet complete; returns the stems run."""
    from dtivl.drivers.common import claim_lock

    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    d = out / RUN_DIR
    (d / "locks").mkdir(parents=True, exist_ok=True)
    done = []
    for E in window_moduli():
        for lid in lesions or core_ids():
            stem = _stem(lid, E)
            p = d / f"{stem}.json"
            r = _read(p)
            if r and r.get("complete"):
                continue
            lock = d / "locks" / f"{stem}.lock"
            if not claim_lock(lock):
                continue
            try:
                try:
                    rec = run_one(lid, E)
                except Exception as exc:                      # recorded; retried by the next run()
                    log.exception("%s failed", stem)
                    rec = {"lesion": lid, "ca_E_MPa": E, "complete": False, "error": repr(exc)}
                p.write_text(json.dumps(_to_jsonable(rec)))
                done.append(stem)
            finally:
                lock.unlink(missing_ok=True)
            if max_runs is not None and len(done) >= max_runs:
                return done
    return done


def n_left(out_dir: str | Path | None = None) -> int:
    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    d = out / RUN_DIR
    left = 0
    for E in window_moduli():
        for lid in core_ids():
            r = _read(d / f"{_stem(lid, E)}.json")
            left += not (r and r.get("complete"))
    return left


def analyze(out_dir: str | Path | None = None) -> dict:
    """Per modulus: counts of IVL and balloon-alone through-thickness damage and the core AUCs (bootstrap
    intervals) of the best single mechanics predictor of the main analysis and of the Fujino score."""
    import pandas as pd
    from dtivl import params as P
    from dtivl.stats import roc as R

    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    d = out / RUN_DIR
    ms_path = out / "metrics_summary.json"
    metrics = _read(ms_path) or {}
    prim = (metrics.get("outcomes") or {}).get("ivl_fracture") or {}
    best = (prim.get("fujino_comparison") or {}).get("best_single_mechanics")
    uni = prim.get("univariable") or {}
    cfg = metrics.get("config") or {}
    seed = int(cfg.get("seed", P.raw("design.seed")))
    n_boot = int(cfg.get("n_boot", P.get("metrics.bootstrap_n").value))
    level = float(cfg.get("ci_level", P.get("metrics.ci_level").value))
    from dtivl.drivers.common import COHORT_DIR
    coh = pd.read_csv(COHORT_DIR / "cohort.csv").set_index("lesion_id")
    res = {"note": NOTE, "moduli_MPa": window_moduli(), "best_single_mechanics": best, "by_modulus": {}}
    for E in window_moduli():
        rows = []
        for lid in core_ids():
            r = _read(d / f"{_stem(lid, E)}.json")
            if not (r and r.get("complete")):
                continue
            o = r.get("outputs") or {}
            rows.append({"lesion_id": lid, "ivl_fracture": o.get("ivl_fracture"),
                         "balloon_fracture": o.get("balloon_fracture"),
                         "ivl_onset_load": o.get("ivl_onset_load"),
                         **{f"pred_{k}": v for k, v in (r.get("predictors") or {}).items()},
                         "fujino_score": coh.loc[lid, "fujino_score"] if lid in coh.index else math.nan})
        e = {"n_runs_complete": len(rows), "n_runs_planned": len(core_ids())}
        if rows:
            t = pd.DataFrame(rows)
            ok = t["ivl_fracture"].notna()
            y = t.loc[ok, "ivl_fracture"].astype(bool).astype(int).to_numpy()
            e.update(n=int(ok.sum()), n_ivl_fracture=int(y.sum()),
                     n_balloon_fracture=int(t["balloon_fracture"].fillna(False).astype(bool).sum()))
            for p in [q for q in (best, "fujino_score") if q]:
                if p not in t or y.sum() < 2 or (len(y) - y.sum()) < 2:
                    continue
                s = (uni.get(p) or {}).get("direction", 1) or 1
                x = s * pd.to_numeric(t.loc[ok, p], errors="coerce").to_numpy(float)
                fin = np.isfinite(x)
                rr = R.roc_analysis(y[fin], x[fin], 1, n_boot=n_boot, rng=R.rng_for(seed, f"modulus_window|{E}|{p}"),
                                    level=level)
                e[p] = {"auc": rr["auc"], "auc_ci": [rr["auc_ci_lo"], rr["auc_ci_hi"]],
                        "auc_ci_degenerate": rr["auc_ci_degenerate"], "direction": s}
        res["by_modulus"][f"{E:g}"] = e
    (out / OUT_JSON).write_text(json.dumps(_to_jsonable(res), indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze", "left"])
    ap.add_argument("--max-runs", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if a.cmd == "run":
        print(run(max_runs=a.max_runs))
    elif a.cmd == "left":
        print(n_left())
    else:
        print(json.dumps(analyze(), indent=1)[:2000])
