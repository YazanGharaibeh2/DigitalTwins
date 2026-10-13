"""Damage-localisation diagnostics for the pilot (Checkpoint 2, decision D10).

The pilot showed damage spreading over the whole calcium deposit instead of
forming discrete cracks. Three experiments test whether that is a solution-path
artefact or a property of the model, on one pilot lesion with the IVL balloon
cap active, for each modulus in ``E_values``:

1. ``fine``: pressure ramp in steps of ``dp`` from 0 to the IVL balloon
   pressure; per step the maximum damage, damaged area fraction and crack
   classification (:func:`dtivl.fracture.cracks.crack_analysis`).
2. ``seeded``: same ramp, but at the last step before damage reaches
   ``d_crack`` anywhere, a radial through-thickness crack (d = 1 on the P1
   nodes of calcium within ``SEED_HALF_DEG`` of the deposit centre) is imposed
   as an irreversible lower bound; the ramp then continues.
3. ``heterogeneous``: G_c multiplied per cell by ``1 + cov * xi`` (clipped to
   ``1 +/- 3 cov``), ``xi`` a unit-variance Gaussian random field with
   Gaussian covariance and correlation length ``ell`` (random-phase Fourier
   sum of ``N_MODES`` modes, seed ``design.seed``), evaluated at the cell
   centroids.

If the seeded crack and the heterogeneity both end in whole-deposit damage,
the diffuse outcome is a property of the model at that modulus, not of the
solution path. Writes ``results/diag_localisation.json``.

:func:`ell_sensitivity` refines the calcium mesh and ``ell`` together
(``results/diag_ell.json``).

:func:`rankine_test` (decision D10, time-boxed test of option b) runs the IVL
protocol on every pilot lesion at the baseline calcium modulus
(``MaterialSet.baseline()``) with the damage driving force ``"energy"``
(``psi+`` of the volumetric-deviatoric split) and ``"rankine"`` (``psi_R =
(E/2) <e1>+^2``, maximum principal Green-Lagrange strain; see
:mod:`dtivl.fracture.phasefield`), with ``T`` = the lesion's
``T_equivalent`` at that modulus, and records the protocol and crack summary
of each run and its wall time. It also repeats the homogeneous-plate onset
check with the Rankine force at that modulus. Writes
``results/diag_rankine.json``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from dtivl import params as P
from dtivl.drivers.common import Timer, write_json
from dtivl.materials.constitutive import MaterialSet

SEED_HALF_DEG = 2.5      #: half-width of the seeded radial crack (deg)
N_MODES = 400            #: Fourier modes of the random field
DIAG_LESION = "P-A90-T0.2-SUP"


def gaussian_field(points: np.ndarray, corr_len: float, seed: int, n_modes: int = N_MODES) -> np.ndarray:
    """Unit-variance Gaussian random field with covariance exp(-r^2 / (2 corr_len^2)) at ``points``."""
    rng = np.random.default_rng(seed)
    dim = points.shape[1]
    k = rng.normal(0.0, 1.0 / corr_len, size=(n_modes, dim))
    phase = rng.uniform(0.0, 2.0 * math.pi, n_modes)
    return math.sqrt(2.0 / n_modes) * np.cos(points @ k.T + phase).sum(axis=1)


def _cell_centroids(md) -> np.ndarray:
    msh = md.mesh
    tdim = msh.topology.dim
    imap = msh.topology.index_map(tdim)
    n = imap.size_local + imap.num_ghosts
    return msh.geometry.x[msh.geometry.dofmaps[0][:n]].mean(axis=1)[:, : msh.geometry.dim]


def _step_record(st, md, plm, p, info, hist) -> dict:
    from dtivl.fracture.cracks import crack_analysis

    obs = st.observables()
    hist.append({"phase": "diag", "load": float(p), "max_d": obs["max_d"]})
    c = crack_analysis(md, st.d, plm, history=hist)
    return {"p": float(p), "max_d": obs["max_d"], "damaged_fraction": c["damaged_fraction"],
            "fracture": c["fracture"], "n_fractures": c["n_fractures"], "diffuse": c["diffuse"],
            "pattern": c["pattern"], "lumen_area": obs["lumen_area"],
            "stagger_its": info.get("stagger_its"), "stagger_converged": info.get("stagger_converged")}


def _run(md, plm, ms, p_steps, *, seed_at=None, gc_factor=None, h_ca=None) -> dict:
    from dtivl.solvers.nonlinear import NonlinearState, SolveError

    st = NonlinearState(md, ms, fracture="phasefield", gc_factor=gc_factor, h_ca=h_ca, name="diag")
    st.set_loads(R_cap=0.5 * P.get("geometry.ivl_balloon_diameter_mm").value)
    steps, hist, seeded = [], [], None
    for p in p_steps:
        st.set_loads(p=float(p))
        try:
            info = st.solve_step()
            st.commit()
        except SolveError as exc:
            steps.append({"p": float(p), "error": str(exc)})
            break
        steps.append(_step_record(st, md, plm, p, info, hist))
        if seed_at is not None and seeded is None and p >= seed_at - 1e-12:
            x = st.D.tabulate_dof_coordinates()
            th = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
            band = (np.abs(th) < SEED_HALF_DEG) & (st.d_ub.x.array > 0)
            st.d.x.array[band] = 1.0
            st.d_lb.x.array[band] = 1.0
            st.d.x.scatter_forward()
            st.d_lb.x.scatter_forward()
            seeded = {"p": float(p), "n_nodes": int(band.sum()), "half_width_deg": SEED_HALF_DEG}
    first = next((s for s in steps if s.get("damaged_fraction", 0.0) > 0.0), None)
    return {"steps": steps, "seed": seeded,
            "first_damaged_p": None if first is None else first["p"],
            "final_damaged_fraction": steps[-1].get("damaged_fraction") if steps else None,
            "final_diffuse": steps[-1].get("diffuse") if steps else None,
            "final_n_fractures": steps[-1].get("n_fractures") if steps else None}


def localisation(lesion: str = DIAG_LESION, E_values=None, dp: float = 0.002, cov: float = 0.10) -> dict:
    """Run the three experiments for each modulus and write ``results/diag_localisation.json``."""
    from dtivl.drivers.stage_pilot import e_sweep, mesh2d
    from dtivl.geometry.labels import TAG_CALCIUM

    base = MaterialSet.baseline().ca_E
    if E_values is None:
        E_values = [base, min(e_sweep(), key=lambda e: abs(math.log10(e) - 2.0))]
    p_end = P.get("loads.ivl_balloon_pressure_atm").value
    p_steps = np.arange(dp, p_end + 0.5 * dp, dp)
    plm, info, md = mesh2d(lesion)
    seed = int(P.raw("design.seed"))
    out = {"lesion": lesion, "dp_MPa": dp, "p_end_MPa": p_end, "cov": cov, "random_field_seed": seed,
           "seed_half_width_deg": SEED_HALF_DEG, "by_E": []}
    for E in E_values:
        ms = MaterialSet.baseline(ca_E=E)
        row = {"ca_E_MPa": E}
        with Timer() as t:
            row["fine"] = _run(md, plm, ms, p_steps)
        row["fine"]["seconds"] = t.seconds
        # seed at the last step before any cell is cracked
        pre = [s["p"] for s in row["fine"]["steps"] if s.get("damaged_fraction", 1.0) == 0.0]
        p_seed = pre[-1] if pre else float(p_steps[0])
        with Timer() as t:
            row["seeded"] = _run(md, plm, ms, p_steps, seed_at=p_seed)
        row["seeded"]["seconds"] = t.seconds
        from dtivl.solvers.nonlinear import NonlinearState

        ell = NonlinearState(md, ms, fracture="none", name="diag_ell").ell
        cen = _cell_centroids(md)
        xi = gaussian_field(cen, ell, seed)
        fac = np.ones(cen.shape[0])
        ca = md.cell_tags.find(TAG_CALCIUM)
        fac[ca] = np.clip(1.0 + cov * xi[ca], 1.0 - 3.0 * cov, 1.0 + 3.0 * cov)
        with Timer() as t:
            row["heterogeneous"] = _run(md, plm, ms, p_steps, gc_factor=fac)
        row["heterogeneous"]["seconds"] = t.seconds
        row["heterogeneous"]["gc_factor_calcium"] = {"mean": float(fac[ca].mean()), "sd": float(fac[ca].std()),
                                                     "min": float(fac[ca].min()), "max": float(fac[ca].max()),
                                                     "corr_len_mm": ell}
        out["by_E"].append(row)
        write_json("diag_localisation", out)
    return out


#: calcium mesh sizes (mm) of the regularisation-length study; ell = fracture.ell_over_h * h_ca
ELL_STUDY_H = (0.04, 0.02, 0.01)


def ell_sensitivity(lesion: str = DIAG_LESION, E_values=None, h_values=ELL_STUDY_H, dp: float = 0.002) -> dict:
    """Fine pressure ramp (experiment 1 of :func:`localisation`) with the calcium mesh size and
    the regularisation length refined together (ell / h fixed). Writes ``results/diag_ell.json``.

    Tests whether discrete (non-diffuse) cracks appear once the crack band (about 4 ell wide) is
    narrower than the calcium sheet is thick.
    """
    from dtivl.drivers.common import MESH_DIR, load_mask
    from dtivl.drivers.stage_pilot import e_sweep
    from dtivl.geometry.labels import TAG_CALCIUM
    from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap

    base = MaterialSet.baseline().ca_E
    if E_values is None:
        E_values = [base, min(e_sweep(), key=lambda e: abs(math.log10(e) - 2.0))]
    p_end = P.get("loads.ivl_balloon_pressure_atm").value
    p_steps = np.arange(dp, p_end + 0.5 * dp, dp)
    plm = load_mask(lesion)
    h_soft = float(P.raw("discretisation.h_soft_mm"))
    ell_over_h = P.get("fracture.ell_over_h").value
    out = {"lesion": lesion, "dp_MPa": dp, "p_end_MPa": p_end, "ell_over_h": ell_over_h, "h_soft_mm": h_soft,
           "by_h": []}
    for h in h_values:
        info = mesh_from_labelmap(plm, MESH_DIR / f"{lesion}_ellstudy_h{h:g}.msh", h_ca=h, h_soft=h_soft)
        md = load_mesh(info["msh"], gdim=2)
        rec = {"h_ca_mm": h, "ell_mm": ell_over_h * h, "n_cells": info["n_cells"],
               "n_cells_calcium": int(md.cell_tags.find(TAG_CALCIUM).size), "by_E": []}
        for E in E_values:
            ms = MaterialSet.baseline(ca_E=E)
            with Timer() as t:
                r = _run(md, plm, ms, p_steps, h_ca=h)
            r["seconds"] = t.seconds
            r["ca_E_MPa"] = E
            rec["by_E"].append(r)
            out_rec = [x for x in out["by_h"] if x["h_ca_mm"] != h] + [rec]
            write_json("diag_ell", {**out, "by_h": out_rec})
        out["by_h"] = [x for x in out["by_h"] if x["h_ca_mm"] != h] + [rec]
    write_json("diag_ell", out)
    return out


#: damage driving forces compared by :func:`rankine_test`
RANKINE_FORCES = ("energy", "rankine")
#: crack-summary keys copied into the compact per-run table of :func:`rankine_test`
RANKINE_TABLE_KEYS = ("fracture", "n_fractures", "n_partial", "diffuse", "n_diffuse", "damaged_fraction",
                      "max_depth_frac", "pattern", "location_frac", "onset_load", "onset_phase")


def _lesion_T_equivalent(lid: str, ms: MaterialSet, md, plm) -> tuple[float, str]:
    """``T_equivalent`` of a lesion at ``ms.ca_E`` from ``results/pilot_linear_transient.json``, else computed
    with :func:`dtivl.solvers.transient.transfer_factor`. Returns ``(T, source)``."""
    from dtivl.drivers.common import RESULTS_DIR, read_json

    if (RESULTS_DIR / "pilot_linear_transient.json").exists():
        rows = read_json("pilot_linear_transient").get("lesions", {}).get(lid, {}).get("by_E", [])
        for r in rows:
            if abs(r["ca_E_MPa"] - ms.ca_E) <= 1e-6 * ms.ca_E and "transient" in r:
                return float(r["transient"]["T_equivalent"]), "results/pilot_linear_transient.json"
    from dtivl.solvers.transient import transfer_factor

    return float(transfer_factor(md, plm, ms)["T_equivalent"]), "computed (transfer_factor)"


#: crack-analysis keys stored per step of the crack evolution (:func:`_track_crack_evolution`)
EVOLUTION_KEYS = ("fracture", "n_fractures", "n_partial", "diffuse", "n_diffuse", "damaged_fraction",
                  "max_depth_frac", "pattern")
#: crack-evolution keys of the first cracked step copied into the table row (prefixed ``first_crack_``)
FIRST_CRACK_KEYS = ("p", "phase", "n_crack_cells", "damaged_fraction", "n_fractures", "n_partial", "diffuse",
                    "pattern", "stagger_converged")


def _rankine_row(rec: dict) -> dict:
    """Compact per-run summary (status, damage onset, crack counts and patterns, wall time, and the
    crack state at the first step with cracked cells)."""
    cr = rec.get("cracks", {})
    evo = rec.get("crack_evolution") or [{}]
    return {"status": rec["status"], "seconds": rec.get("seconds"), "cpu_seconds": rec.get("cpu_seconds"),
            "n_steps_unconverged_stagger": rec.get("n_steps_unconverged_stagger"),
            "first_damage_p": rec.get("first_damage_p"), "max_d": rec.get("summary", {}).get("max_d"),
            **{k: cr.get(k) for k in RANKINE_TABLE_KEYS},
            **{f"first_crack_{k}": evo[0].get(k) for k in FIRST_CRACK_KEYS}}


def _track_crack_evolution(st, md, plm) -> tuple[list, dict]:
    """Wrap ``st.record`` so that every accepted step whose cracked-cell count changed is post-processed
    with :func:`dtivl.fracture.cracks.crack_analysis` (diagnostics only: the solution is not touched).
    Returns ``(evolution list, timing dict)``; the timing holds the post-processing wall and CPU time,
    which the caller subtracts from the protocol time."""
    from dtivl.fracture.cracks import crack_analysis

    evo, timing, last = [], {"seconds": 0.0, "cpu_seconds": 0.0}, [0]
    record = st.record

    def wrapped(*args, **kwargs):
        rec = record(*args, **kwargs)
        n = int(rec.get("n_crack_cells") or 0)
        if n != last[0]:
            last[0] = n
            w0, c0 = time.perf_counter(), time.process_time()
            ca = crack_analysis(md, st.d, plm, history=st.history)
            evo.append({"step": rec["step"], "phase": rec["phase"], "p": rec["p"], "max_d": rec["max_d"],
                        "n_crack_cells": n, "stagger_converged": rec.get("stagger_converged"),
                        "stagger_err": rec.get("stagger_err"), **{k: ca.get(k) for k in EVOLUTION_KEYS}})
            timing["seconds"] += time.perf_counter() - w0
            timing["cpu_seconds"] += time.process_time() - c0
        return rec

    st.record = wrapped
    return evo, timing


def _read_rankine_runs(names) -> dict:
    """Completed runs ``{lesion: {force: record}}`` found in the given result files."""
    from dtivl.drivers.common import RESULTS_DIR, read_json

    runs: dict = {}
    for name in names:
        if not (RESULTS_DIR / f"{name}.json").exists():
            continue
        for lid, rec in read_json(name).get("lesions", {}).items():
            for force, r in rec.get("runs", {}).items():
                runs.setdefault(lid, {})[force] = {"lesion_meta": {k: v for k, v in rec.items() if k != "runs"},
                                                   "run": r}
    return runs


def rankine_test(lesions=None, driving_forces=RANKINE_FORCES, *, part: str | None = None,
                 resume: bool = True) -> dict:
    """D10 option (b): IVL on the pilot lesions with the energy and the Rankine damage driving force.

    For each lesion (default :func:`dtivl.drivers.stage_pilot.pilot_ids`) and force, a fresh
    :class:`~dtivl.solvers.nonlinear.NonlinearState` at ``MaterialSet.baseline()`` runs
    :func:`~dtivl.solvers.nonlinear.run_ivl` with ``T = T_equivalent`` of the lesion at that modulus
    (:func:`_lesion_T_equivalent`); the record is
    :func:`dtivl.drivers.stage_pilot._protocol_record` plus the wall and process CPU time of the
    protocol (``seconds``, ``cpu_seconds``), the first pressure with any damage (``first_damage_p``),
    the compact step history (``steps``: phase, p, max_d, n_crack_cells, stagger iterations) and the
    crack evolution (``crack_evolution``: crack analysis at every step whose cracked-cell count
    changed, :func:`_track_crack_evolution`). ``table`` lists the compact rows
    (:func:`_rankine_row`). ``verification_rankine_onset`` is
    :func:`dtivl.fracture.phasefield.onset_check` (2D plate, Rankine force) at the same modulus.

    ``part``: write ``results/diag_rankine_<part>.json`` instead of ``diag_rankine.json`` (lesions
    can then run in parallel processes). ``resume``: runs already stored in ``diag_rankine.json`` or
    in any ``diag_rankine_*.json`` part file are reused, so a final call without ``part`` merges the
    parts into ``diag_rankine.json``.
    """
    from dtivl.drivers.common import RESULTS_DIR
    from dtivl.drivers.stage_pilot import _protocol_record, mesh2d, pilot_ids
    from dtivl.fracture import phasefield as PF
    from dtivl.solvers.nonlinear import NonlinearState, SolverSettings, run_ivl

    lesions = list(lesions or pilot_ids())
    out_name = "diag_rankine" + (f"_{part}" if part else "")
    old = {}
    if resume:
        names = ["diag_rankine"] + sorted(p.stem for p in RESULTS_DIR.glob("diag_rankine_*.json"))
        old = _read_rankine_runs(dict.fromkeys(names + [out_name]))
    ms = MaterialSet.baseline()
    out = {"check": "D10(b): damage driving force 'energy' (psi+) vs 'rankine' (psi_R = E/2 <e1>+^2) under IVL",
           "ca_E_MPa": ms.ca_E, "material_set": ms.provenance, "driving_forces": list(driving_forces),
           "protocol": "run_ivl with T = T_equivalent of the lesion at ca_E_MPa (default increments)",
           "lesions": {}, "table": []}
    with Timer() as t:
        v = PF.onset_check(2, ms, driving_force="rankine")
    out["verification_rankine_onset"] = {k: v.get(k) for k in (
        "passed", "eps_f", "gl_axial_onset_theory", "gl_axial_bracket", "rel_err_gl_axial", "psi_onset_theory_MPa",
        "psi_bracket_MPa", "rel_err_psi", "e1_matches_gl_axial")}
    out["verification_rankine_onset"]["seconds"] = t.seconds

    def _save():
        write_json(out_name, out)

    for lid in lesions:
        plm, info, md = mesh2d(lid)
        T, src = _lesion_T_equivalent(lid, ms, md, plm)
        rec = {"T_equivalent": T, "T_source": src,
               "mesh": {k: info[k] for k in ("n_nodes", "n_cells", "n_cells_per_tag")}, "runs": {}}
        out["lesions"][lid] = rec
        for force in driving_forces:
            prev = old.get(lid, {}).get(force)
            if prev is not None and abs(prev["lesion_meta"].get("T_equivalent", -1.0) - T) <= 1e-12:
                rec["runs"][force] = prev["run"]
            else:
                st = NonlinearState(md, ms, settings=SolverSettings(driving_force=force), name=f"rk_{force}")
                evo, post = _track_crack_evolution(st, md, plm)
                cpu0 = time.process_time()
                with Timer() as t:
                    res = run_ivl(st, T=T)
                cpu = time.process_time() - cpu0
                r = _protocol_record(res, md, plm)
                # protocol time without the crack-evolution post-processing; CPU time is the process CPU
                # time (robust to other jobs on a shared machine), wall time is not
                r["seconds"] = t.seconds - post["seconds"]
                r["cpu_seconds"] = cpu - post["cpu_seconds"]
                r["postprocessing_seconds"] = post["seconds"]
                r["driving_force"] = force
                r["first_damage_p"] = next((h["p"] for h in st.history if h["max_d"] > 0.0), None)
                r["steps"] = [{k: h.get(k) for k in ("phase", "p", "max_d", "n_crack_cells", "stagger_its",
                                                     "stagger_converged")} for h in st.history]
                r["crack_evolution"] = evo
                rec["runs"][force] = r
            out["table"] = [x for x in out["table"] if (x["lesion"], x["driving_force"]) != (lid, force)]
            out["table"].append({"lesion": lid, "driving_force": force, "T_equivalent": T,
                                 **_rankine_row(rec["runs"][force])})
            _save()
    out["total_seconds"] = sum(r.get("seconds") or 0.0 for x in out["lesions"].values() for r in x["runs"].values())
    out["total_cpu_seconds"] = sum(r.get("cpu_seconds") or 0.0 for x in out["lesions"].values()
                                   for r in x["runs"].values())
    _save()
    return out


if __name__ == "__main__":
    localisation()
    ell_sensitivity()
