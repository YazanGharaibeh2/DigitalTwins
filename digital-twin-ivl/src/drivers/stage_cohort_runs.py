"""Stage: 2D runs of the core and hold-out cohorts at the baseline calcium modulus.

Per lesion (decisions D8-D15 of Checkpoint 2):

1. pre-treatment linear predictors (:func:`dtivl.metrics.predictors.predictor_analysis`);
2. the IVL load factor T_equivalent from the transient analysis
   (:func:`dtivl.solvers.transient.transfer_factor`); an input of the outcome
   model, not a predictor;
3. the nonlinear phase-field protocols of :data:`COHORT_PROTOCOLS` (the
   outcome model), and the erosion comparator on the lesions of
   ``design.erosion_subset`` (D13).

Results go to ``results/cohort_runs/<lesion>.json``, written after every
step so a run resumes where it stopped (``resume=True``); any number of worker
processes can run :func:`run` concurrently (lesions are claimed with lock files).
:func:`aggregate` writes ``results/cohort_outcomes.csv`` (one row per lesion)
and ``results/cohort_runs_summary.json``.
"""
from __future__ import annotations

import itertools
import json

import pandas as pd

from dtivl import params as P
from dtivl.drivers.common import COHORT_DIR, RESULTS_DIR, Timer, _to_jsonable, read_json, write_json
from dtivl.drivers.stage_pilot import _protocol_record, mesh2d
from dtivl.geometry.cohort import MORPH_CODE
from dtivl.materials.constitutive import MaterialSet

OUT_DIR = RESULTS_DIR / "cohort_runs"
COHORT_PROTOCOLS = ("physiological", "balloon_only", "balloon_10atm", "ivl", "stent_no_ivl", "stent_post_ivl")
PREDICTOR_KEYS = ("sigma1_p95", "sigma1_p99", "sigma1_mean", "sigma1_peak_nl", "sed_mean", "sed_peak_nl",
                  "scf_edge_nl", "frac_eps_gt_epsf", "frac_eps_gt_half_epsf", "frac_sig_gt_sigc",
                  "frac_sig_gt_half_sigc", "min_thickness_high_stress")
TRANSIENT_KEYS = ("T", "T_equivalent", "sigma1_peak_transient", "sigma1_static", "sigma1_static_equivalent",
                  "t_peak", "peak_at_window_end")


def cohort_ids(sets=("core", "holdout")) -> list[str]:
    df = pd.read_csv(COHORT_DIR / "cohort.csv")
    return df[df["set"].isin(sets)]["lesion_id"].tolist()


def erosion_subset() -> list[str]:
    """Core lesion ids of ``design.erosion_subset`` (full factorial of the listed levels)."""
    e = P.raw("design.erosion_subset")
    ids = [f"C-A{a:g}-T{t:g}-{MORPH_CODE[m]}" for a, t, m in
           itertools.product(e["arc_deg"], e["thickness_mm"], e["morphology"])]
    have = set(cohort_ids(("core",)))
    missing = [i for i in ids if i not in have]
    if missing:
        raise ValueError(f"design.erosion_subset names lesions not in the core set: {missing}")
    return ids


def _path(lid: str):
    return OUT_DIR / f"{lid}.json"


def _save(lid: str, rec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(lid).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_to_jsonable(rec), indent=1))
    tmp.replace(_path(lid))


def run_lesion(lid: str, *, resume: bool = True, erosion: bool | None = None) -> dict:
    """Run (or complete) every analysis of one lesion; returns the record."""
    from dtivl.metrics.predictors import predictor_analysis
    from dtivl.solvers.nonlinear import NonlinearState, run_ivl, run_pressure, run_stent
    from dtivl.solvers.transient import transfer_factor

    ms = MaterialSet.baseline()
    rec = json.loads(_path(lid).read_text()) if resume and _path(lid).exists() else {}
    if rec and abs(rec.get("ca_E_MPa", ms.ca_E) - ms.ca_E) > 1e-9:
        raise ValueError(f"{lid}: stored results are for E = {rec['ca_E_MPa']} MPa, baseline is {ms.ca_E} MPa")
    erosion = (lid in erosion_subset()) if erosion is None else erosion
    plm, info, md = mesh2d(lid)
    rec.update(lesion=lid, ca_E_MPa=ms.ca_E, k_floor=ms.k_floor,
               mesh={k: info[k] for k in ("n_nodes", "n_cells", "n_cells_per_tag")})
    if "predictors" not in rec:
        with Timer() as t:
            pr = predictor_analysis(md, plm, ms)
        rec["predictors"] = {**{k: pr.get(k) for k in PREDICTOR_KEYS}, "p_lumen_MPa": pr.get("p_lumen_MPa"),
                             "seconds": t.seconds}
        _save(lid, rec)
    if "transient" not in rec:
        with Timer() as t:
            tf = transfer_factor(md, plm, ms)
        rec["transient"] = {**{k: tf[k] for k in TRANSIENT_KEYS}, "seconds": t.seconds}
        _save(lid, rec)
    T = rec["transient"]["T_equivalent"]
    K_s = read_json("calibration_stent")["K_s_MPa_per_mm"]
    p_phys = P.get("loads.physiological_pressure").value
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_poba = P.get("loads.poba_pressure_atm").value
    R_ivl = 0.5 * P.get("geometry.ivl_balloon_diameter_mm").value
    proto = rec.setdefault("protocols", {})

    def _do(key, fn):
        with Timer() as t:
            proto[key] = fn()
        proto[key]["seconds"] = t.seconds
        _save(lid, rec)

    if "physiological" not in proto:
        _do("physiological", lambda: _protocol_record(run_pressure(NonlinearState(md, ms), p_phys), md, plm))
    if "balloon_only" not in proto:
        _do("balloon_only", lambda: _protocol_record(
            run_pressure(NonlinearState(md, ms), p_bal, cap_radius=R_ivl, phase="balloon_only"), md, plm))
    if "balloon_10atm" not in proto:
        _do("balloon_10atm", lambda: _protocol_record(
            run_pressure(NonlinearState(md, ms), p_poba, cap_radius=R_ivl, phase="balloon_10atm"), md, plm))
    if "ivl" not in proto or "stent_post_ivl" not in proto:
        st_ivl = NonlinearState(md, ms)
        _do("ivl", lambda: _protocol_record(run_ivl(st_ivl, T=T), md, plm))
        if "stent_post_ivl" not in proto:
            _do("stent_post_ivl", lambda: _protocol_record(run_stent(st_ivl, K_s=K_s), md, plm))
    if "stent_no_ivl" not in proto:
        _do("stent_no_ivl", lambda: _protocol_record(run_stent(NonlinearState(md, ms), K_s=K_s), md, plm))
    if erosion and "ivl_erosion" not in proto:
        _do("ivl_erosion", lambda: _protocol_record(
            run_ivl(NonlinearState(md, ms, fracture="erosion"), T=T), md, plm))
    rec["complete"] = True
    _save(lid, rec)
    return rec


def _claim(lid: str) -> bool:
    """Atomically claim a lesion for this process (lock file with the pid); stale locks are taken over."""
    import os

    lock = OUT_DIR / "locks" / f"{lid}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
                os.kill(pid, 0)
                return False                  # held by a live process
            except (ValueError, ProcessLookupError, FileNotFoundError):
                lock.unlink(missing_ok=True)  # stale lock: retry once
    return False


def _release(lid: str) -> None:
    (OUT_DIR / "locks" / f"{lid}.lock").unlink(missing_ok=True)


def run(ids=None, resume: bool = True) -> list[str]:
    """Run every lesion of ``ids`` (default: core + hold-out) that is not complete and not claimed by
    another live worker; any number of workers can run this concurrently. Returns the ids done here."""
    import logging

    ids = list(ids or cohort_ids())
    done = []
    for lid in ids:
        if resume and _path(lid).exists() and json.loads(_path(lid).read_text()).get("complete"):
            continue
        if not _claim(lid):
            continue
        try:
            run_lesion(lid, resume=resume)
            done.append(lid)
        except Exception as exc:          # record and continue with the next lesion
            logging.getLogger(__name__).exception("%s failed: %s", lid, exc)
            rec = json.loads(_path(lid).read_text()) if _path(lid).exists() else {"lesion": lid}
            rec["error"] = f"{type(exc).__name__}: {exc}"
            _save(lid, rec)
        finally:
            _release(lid)
    return done


def _flat(rec: dict) -> dict:
    row = {"lesion_id": rec["lesion"], "ca_E_MPa": rec.get("ca_E_MPa"), "complete": bool(rec.get("complete")),
           "error": rec.get("error")}
    for k, v in (rec.get("predictors") or {}).items():
        row[f"pred_{k}"] = v
    for k, v in (rec.get("transient") or {}).items():
        row[f"ivl_load_{k}"] = v
    for p, r in (rec.get("protocols") or {}).items():
        c = r.get("cracks", {})
        row[f"{p}_status"] = r.get("status")
        row[f"{p}_fracture"] = c.get("fracture")
        row[f"{p}_onset"] = c.get("onset_load")
        row[f"{p}_onset_phase"] = c.get("onset_phase")
        row[f"{p}_damaged_fraction"] = c.get("damaged_fraction")
        row[f"{p}_diffuse"] = c.get("diffuse")
        row[f"{p}_n_fractures"] = c.get("n_fractures")
        row[f"{p}_n_partial"] = c.get("n_partial")
        row[f"{p}_max_depth_frac"] = c.get("max_depth_frac")
        row[f"{p}_unconverged_steps"] = r.get("n_steps_unconverged_stagger")
        row[f"{p}_seconds"] = r.get("seconds")
        row[f"{p}_max_lumen_area"] = r.get("max_lumen_area")
        if "p_top" in (r.get("summary") or {}):    # IVL protocols: peak load applied (censoring value of the onset)
            row[f"{p}_p_top"] = r["summary"]["p_top"]
        for k, v in (r.get("summary", {}).get("metrics") or {}).items():
            if isinstance(v, (int, float, bool)) or v is None:
                row[f"{p}_{k}"] = v
    return row


def aggregate() -> pd.DataFrame:
    """One row per lesion (cohort descriptors + predictors + outcomes) -> results/cohort_outcomes.csv."""
    desc = pd.read_csv(COHORT_DIR / "cohort.csv")
    rows = [_flat(json.loads(p.read_text())) for p in sorted(OUT_DIR.glob("*.json"))]
    out = pd.DataFrame(rows)
    keep = ["lesion_id", "set", "morphology", "ca_arc_deg", "ca_thickness", "ca_depth", "ca_arc_total_deg",
            "ca_thickness_max", "ca_area", "lumen_area", "n_deposits", "lipid_arc_deg", "plaque_ecc_index",
            "fujino_score"]
    df = desc[[c for c in keep if c in desc.columns]].merge(out, on="lesion_id", how="inner")
    df.to_csv(RESULTS_DIR / "cohort_outcomes.csv", index=False)
    summary = {"n_lesions": int(len(df)), "n_complete": int(df["complete"].sum()),
               "n_error": int(df["error"].notna().sum()), "by_set": df.groupby("set").size().to_dict(),
               "ca_E_MPa": sorted(df["ca_E_MPa"].dropna().unique().tolist())}
    for p in COHORT_PROTOCOLS + ("ivl_erosion",):
        col = f"{p}_fracture"
        if col in df:
            ok = df[f"{p}_status"] == "ok"
            summary[p] = {"n_run": int(df[f"{p}_status"].notna().sum()), "n_ok": int(ok.sum()),
                          "n_fracture": int((df[col] == True).sum()),  # noqa: E712
                          "n_unconverged_runs": int((df.get(f"{p}_unconverged_steps", 0) > 0).sum())}
    write_json("cohort_runs_summary", summary)
    return df


if __name__ == "__main__":
    run()
