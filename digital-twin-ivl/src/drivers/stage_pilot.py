"""Stage 3: Checkpoint-2 pilot (3 lesions in 2D + one 3D Fujino-4 example).

For each pilot lesion and each calcium modulus of a log-spaced sweep over the
cited sensitivity range, compute:
* the linear pre-treatment predictors (geometry + linear elasticity),
* the transient IVL transfer factor T (single pulse, linear elastodynamics),
* the nonlinear phase-field outcomes (added by :func:`run_nonlinear`).
"""
from __future__ import annotations

import numpy as np

from dtivl import params as P
from dtivl.drivers.common import MESH_DIR, ROOT, Timer, load_mask, write_json
from dtivl.geometry import cohort
from dtivl.materials.constitutive import MaterialSet
from dtivl.meshing.extrude3d import mesh3d_from_labelmap
from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap
from dtivl.metrics.predictors import predictor_analysis

#: Number of log-spaced calcium moduli between the cited sensitivity bounds.
N_E_SWEEP = 9   # log-spaced points over sens_E (was 5 before Checkpoint 2; 9 keeps the 5 old points and halves the spacing)


def e_sweep() -> list[float]:
    """Calcium moduli (MPa): log-spaced over materials.calcium.sens_E, plus the baseline."""
    lo, hi = (q.value for q in P.get_list("materials.calcium.sens_E"))
    pts = set(np.round(np.logspace(np.log10(lo), np.log10(hi), N_E_SWEEP), 6).tolist())
    pts.add(round(MaterialSet.baseline().ca_E, 6))
    return sorted(pts)


def pilot_ids() -> list[str]:
    return [s.lesion_id for s in cohort.pilot_specs()]


def example3d_id() -> str:
    return cohort.example3d_spec().lesion_id


def mesh2d(lesion_id: str):
    plm = load_mask(lesion_id)
    info = mesh_from_labelmap(plm, MESH_DIR / f"{lesion_id}.msh")
    return plm, info, load_mesh(info["msh"], gdim=2)


def mesh3d(lesion_id: str):
    """Coarse 3D mesh of the D1 example (sizes from discretisation.example3d)."""
    spec = cohort.example3d_spec()
    assert spec.lesion_id == lesion_id
    plm = load_mask(lesion_id)
    sz = P.raw("discretisation.example3d")
    info = mesh3d_from_labelmap(plm, spec.ca_length, MESH_DIR / f"{lesion_id}.msh", h_ca=float(sz["h_calcium_mm"]),
                                h_soft=float(sz["h_soft_mm"]), h_axial=float(sz["h_axial_mm"]))
    return plm, info, load_mesh(info["msh"], gdim=3)


def mesh2d_like3d(lesion_id: str):
    """2D mesh of the 3D example's cross-section with the same in-plane sizes (2D-vs-3D comparison)."""
    plm = load_mask(lesion_id)
    sz = P.raw("discretisation.example3d")
    info = mesh_from_labelmap(plm, MESH_DIR / f"{lesion_id}_2d.msh", h_ca=float(sz["h_calcium_mm"]),
                              h_soft=float(sz["h_soft_mm"]))
    return plm, info, load_mesh(info["msh"], gdim=2)


def run_example3d(e_values=None, transient=True, nonlinear=True, resume=True, models=("3d", "2d"),
                  out_name: str = "pilot_example3d") -> dict:
    """D1: test the 3D code on the Fujino-4 example and compare with the 2D cross-section model.

    ``resume``: rows of ``results/<out_name>.json`` that already hold every
    requested analysis for a (model, modulus) pair are reused. ``models`` and
    ``out_name`` let parts run in separate processes; :func:`merge_example3d`
    combines the part files.
    """
    from dtivl.drivers.common import read_json
    from dtivl.fracture.phasefield import regularisation_length
    from dtivl.solvers.nonlinear import NonlinearState, run_ivl
    from dtivl.solvers.transient import transfer_factor

    lid = example3d_id()
    base = MaterialSet.baseline().ca_E
    e_values = e_values or [base, min(e_sweep(), key=lambda e: abs(np.log10(e) - np.log10(STENT_E_TARGET_MPA)))]
    h_ca = float(P.raw("discretisation.example3d")["h_calcium_mm"])
    ell = regularisation_length(h_ca=h_ca)
    out = {"lesion": lid, "E_values_MPa": e_values, "ell_mm": ell, "models": {}}
    need = {"predictors"} | ({"transient"} if transient else set()) | ({"ivl"} if nonlinear else set())
    old = {}
    if resume and (ROOT / "results" / f"{out_name}.json").exists():
        prev = read_json(out_name)
        for key, m in prev.get("models", {}).items():
            for r in m.get("by_E", []):
                if need <= set(r):
                    old[(key, round(r["ca_E_MPa"], 6))] = r
    for dim, builder in ((3, mesh3d), (2, mesh2d_like3d)):
        if f"{dim}d" not in models:
            continue
        plm, info, md = builder(lid)
        rec = {"mesh": {k: info[k] for k in ("n_nodes", "n_cells", "n_cells_per_tag")}, "by_E": []}
        for E in e_values:
            if (f"{dim}d", round(E, 6)) in old:
                rec["by_E"].append(old[(f"{dim}d", round(E, 6))])
                out["models"][f"{dim}d"] = rec
                continue
            ms = MaterialSet.baseline(ca_E=E)
            row = {"ca_E_MPa": E}
            with Timer() as t:
                row["predictors"] = predictor_analysis(md, plm, ms)
            row["predictors_seconds"] = t.seconds
            if transient:
                with Timer() as t:
                    tf = transfer_factor(md, plm, ms)
                row["transient"] = {k: tf[k] for k in ("T", "T_equivalent", "sigma1_peak_transient",
                                                       "sigma1_static_equivalent", "t_peak", "peak_at_window_end")}
                row["transient_seconds"] = t.seconds
            if nonlinear:
                T = row["transient"]["T_equivalent"] if transient else 0.0
                st = NonlinearState(md, ms, h_ca=h_ca, ell=ell)
                with Timer() as t:
                    row["ivl"] = _protocol_record(run_ivl(st, T=T), md, plm)
                row["ivl"]["seconds"] = t.seconds
            rec["by_E"].append(row)
            out["models"][f"{dim}d"] = rec
            write_json(out_name, out)
    return out


def merge_example3d(parts=("pilot_example3d", "pilot_example3d_part2")) -> dict:
    """Combine 3D-example part files into ``results/pilot_example3d.json`` (rows keyed by model and modulus)."""
    from dtivl.drivers.common import read_json

    out, rows = None, {}
    for name in parts:
        if not (ROOT / "results" / f"{name}.json").exists():
            continue
        part = read_json(name)
        out = out or {k: v for k, v in part.items() if k != "models"}
        for key, m in part["models"].items():
            for r in m["by_E"]:
                old = rows.get((key, round(r["ca_E_MPa"], 6)), ({}, None))[0]
                if len(r) >= len(old):           # keep the most complete row
                    rows[(key, round(r["ca_E_MPa"], 6))] = (r, m["mesh"])
    out["models"] = {}
    for (key, _), (r, mesh) in sorted(rows.items(), key=lambda kv: (kv[0][0] != "3d", kv[0][1])):
        out["models"].setdefault(key, {"mesh": mesh, "by_E": []})["by_E"].append(r)
    out["E_values_MPa"] = sorted({e for (_, e) in rows})
    write_json("pilot_example3d", out)
    return out


def run_linear_and_transient(include_3d: bool = True, transient: bool = True) -> dict:
    from dtivl.solvers.transient import transfer_factor

    out = {"E_sweep_MPa": e_sweep(), "lesions": {}}
    jobs = [(lid, 2) for lid in pilot_ids()] + ([(example3d_id(), 3)] if include_3d else [])
    for lid, dim in jobs:
        plm, info, md = mesh2d(lid) if dim == 2 else mesh3d(lid)
        rec = {"dim": dim, "mesh": {k: info[k] for k in ("n_nodes", "n_cells", "n_cells_per_tag")}, "by_E": []}
        for E in out["E_sweep_MPa"]:
            ms = MaterialSet.baseline(ca_E=E)
            row = {"ca_E_MPa": E}
            with Timer() as t:
                row["predictors"] = predictor_analysis(md, plm, ms)
            row["predictors_seconds"] = t.seconds
            if transient:
                with Timer() as t:
                    tf = transfer_factor(md, plm, ms)
                row["transient"] = {k: tf[k] for k in ("T", "T_equivalent", "sigma1_peak_transient", "sigma1_static",
                                                       "sigma1_static_equivalent", "t_peak", "peak_at_window_end",
                                                       "p_peak", "width_us", "dt", "t_end")}
                row["transient"]["min_sigma1_history"] = float(np.min(tf["sigma1_max_history"]))
                row["transient_seconds"] = t.seconds
            rec["by_E"].append(row)
        out["lesions"][lid] = rec
    write_json("pilot_linear_transient", out)
    return out


# ---------------------------------------------------------------------------
# Nonlinear phase-field pilot
# ---------------------------------------------------------------------------
#: Calcium moduli (MPa) for the expensive stent runs: baseline and the sweep value nearest 1 GPa.
STENT_E_TARGET_MPA = 1000.0
#: Increments used to move the rigid cylinder from R_dep to the recoiled radius in the K_s identification.
RECOIL_RAMP_STEPS = 5


def _crack_summary(md, state, plm) -> dict:
    from dtivl.fracture.cracks import crack_analysis

    ca = crack_analysis(md, state.damage_field(), plm, history=state.history)
    keep = ("fracture", "n_fractures", "n_partial", "diffuse", "n_diffuse", "damaged_fraction", "max_depth_frac",
            "pattern", "location_frac", "onset_load", "onset_phase", "n_cracked_cells", "cracked_measure",
            "ca_measure")
    return {k: ca.get(k) for k in keep}


def _protocol_record(res, md, plm, *, with_cracks=True) -> dict:
    st = res.state
    rec = {"status": res.status, "summary": {k: v for k, v in res.summary.items()
                                             if k not in ("lumen_pre_xy", "lumen_post_xy")},
           "n_steps": len(st.history),
           # quality flag: accepted steps whose staggered (u, d) loop hit the iteration limit
           "n_steps_unconverged_stagger": sum(1 for h in res.history if h.get("stagger_converged") is False)}
    if st.history:
        last = st.history[-1]
        rec["final"] = {k: last.get(k) for k in ("p", "R_in", "R_cap", "lumen_area", "max_d", "n_crack_cells",
                                                   "p_contact_mean")}
        peak = max(st.history, key=lambda h: h.get("lumen_area") or 0.0)
        rec["max_lumen_area"] = peak.get("lumen_area")
    if with_cracks:
        rec["cracks"] = _crack_summary(md, st, plm)
    return rec


def identify_stent_stiffness(lesion_id: str | None = None) -> dict:
    """Identify K_s so a calcium-free version of a pilot lesion shows the clinical acute recoil.

    Deploy the rigid cylinder to R_target in the calcium-free lesion (calcium
    relabelled as 'other' tissue), then set the cylinder to
    R_f = R_dep (1 - recoil_target) and read the mean contact pressure
    P(R_f). The recoil equilibrium P(R_f) = K_s (R_dep - R_f) gives K_s.
    """
    from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_OTHER
    from dtivl.solvers.nonlinear import NonlinearState, run_stent

    lesion_id = lesion_id or pilot_ids()[0]
    plm = load_mask(lesion_id)
    plm.labels = plm.labels.copy()
    plm.labels[plm.labels == LABEL_CALCIUM] = LABEL_OTHER
    info = mesh_from_labelmap(plm, MESH_DIR / f"{lesion_id}_nocalcium.msh")
    md = load_mesh(info["msh"], gdim=2)
    target = P.get("loads.stent_recoil_target_pct").value / 100.0
    ms = MaterialSet.baseline()
    st = NonlinearState(md, ms, fracture="none")
    with Timer() as t:
        res = run_stent(st, K_s=1e12)          # effectively rigid stent: deployment only, no recoil
        R_dep = res.summary["R_dep"]
        R_f = R_dep * (1.0 - target)
        for R in np.linspace(R_dep, R_f, RECOIL_RAMP_STEPS + 1)[1:]:   # unload the rigid cylinder gradually
            st.set_loads(p=0.0, R_in=float(R), R_cap=None)
            st.solve_step()
            st.commit()
        P_f = st.observables()["p_contact_mean"]
    K_s = P_f / (R_dep - R_f)
    out = {"lesion_template": lesion_id, "recoil_target_pct": target * 100.0, "R_dep": R_dep, "R_f": R_f,
           "P_tissue_at_R_f": P_f, "K_s_MPa_per_mm": K_s, "reached_target": res.summary.get("reached_target"),
           "seconds": t.seconds, "material_set": ms.provenance}
    write_json("calibration_stent", out)
    return out


def _expected_protocols(E, stent_e_values, include_erosion) -> set:
    keys = {"physiological", "balloon_only", "balloon_10atm", "ivl"}
    if any(abs(E - e) < 1e-9 for e in stent_e_values):
        keys |= {"stent_post_ivl", "stent_no_ivl"} | ({"ivl_erosion"} if include_erosion else set())
    return keys


def run_nonlinear(lesions=None, e_values=None, stent_e_values=None, include_erosion=True, resume=True) -> dict:
    """Nonlinear phase-field pilot over the modulus sweep.

    For every lesion and modulus: physiological pressure, IVL balloon only
    (4 atm, cap), a plain balloon at ``loads.poba_pressure_atm`` (10 atm,
    same cap; the balloon-only arm of the cadaver comparison) and IVL with
    the lesion- and modulus-specific transient T. For ``stent_e_values``:
    stent deployment without IVL and after IVL, and the erosion comparator.
    ``resume``: protocols already stored in
    ``results/pilot_nonlinear_<lesion>.json`` are reused and only missing
    ones are run (the post-IVL stent re-runs IVL if needed), so an
    interrupted or extended run continues; the file is rewritten after
    every modulus and always holds every row known so far.
    """
    from dtivl.drivers.common import read_json
    from dtivl.solvers.nonlinear import NonlinearState, run_ivl, run_pressure, run_stent

    lt = read_json("pilot_linear_transient")
    aggregate = lesions is None
    lesions = lesions or pilot_ids()
    e_values = e_values or lt["E_sweep_MPa"]
    if stent_e_values is None:
        base = MaterialSet.baseline().ca_E
        hi = min(lt["E_sweep_MPa"], key=lambda e: abs(np.log10(e) - np.log10(STENT_E_TARGET_MPA)))
        stent_e_values = [base, hi]
    K_s = read_json("calibration_stent")["K_s_MPa_per_mm"]
    p_phys = P.get("loads.physiological_pressure").value
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_poba = P.get("loads.poba_pressure_atm").value
    R_ivl = 0.5 * P.get("geometry.ivl_balloon_diameter_mm").value
    out = {"K_s": K_s, "p_phys": p_phys, "p_bal": p_bal, "p_poba": p_poba, "R_ivl": R_ivl, "lesions": {}}
    for lid in lesions:
        old = {}
        if resume and (ROOT / "results" / f"pilot_nonlinear_{lid}.json").exists():
            old = {round(r["ca_E_MPa"], 6): r for r in read_json(f"pilot_nonlinear_{lid}").get("rows", [])}
        plm, info, md = mesh2d(lid)
        # T_equivalent = transient peak / static sigma1 of the *quasi-static* model (MaterialSet moduli),
        # i.e. the factor that reproduces the transient peak inside run_ivl (design.md section 5.4).
        Tmap = {round(r["ca_E_MPa"], 6): r["transient"]["T_equivalent"] for r in lt["lesions"][lid]["by_E"]}
        rows = []

        def _save():
            # every write holds all rows known so far (resumed rows not yet reached included)
            have = {round(x["ca_E_MPa"], 6) for x in rows}
            allrows = rows + [r for e, r in old.items() if e not in have]
            write_json(f"pilot_nonlinear_{lid}", {"lesion": lid, "K_s": K_s,
                                                 "rows": sorted(allrows, key=lambda x: x["ca_E_MPa"])})

        def _timed(row, key, fn):
            with Timer() as t:
                row[key] = fn()
            row[key]["seconds"] = t.seconds

        for E in e_values:
            prev = old.get(round(E, 6), {})
            if "skipped" in prev or _expected_protocols(E, stent_e_values, include_erosion) <= set(prev):
                rows.append(prev)
                continue
            ms = MaterialSet.baseline(ca_E=E)
            T = Tmap[round(E, 6)]
            row = {"ca_E_MPa": E, "T": T, **{k: v for k, v in prev.items() if isinstance(v, dict)}}
            try:
                row["k_floor"] = ms.k_floor
            except ValueError as exc:          # calcium not stiffer than the matrix: crack filler undefined
                row["skipped"] = str(exc)
                rows.append(row)
                _save()
                continue
            stent = any(abs(E - e) < 1e-9 for e in stent_e_values)
            if "physiological" not in row:
                _timed(row, "physiological",
                       lambda: _protocol_record(run_pressure(NonlinearState(md, ms), p_phys), md, plm))
            if "balloon_only" not in row:
                _timed(row, "balloon_only", lambda: _protocol_record(
                    run_pressure(NonlinearState(md, ms), p_bal, cap_radius=R_ivl, phase="balloon_only"), md, plm))
            if "balloon_10atm" not in row:
                _timed(row, "balloon_10atm", lambda: _protocol_record(
                    run_pressure(NonlinearState(md, ms), p_poba, cap_radius=R_ivl, phase="balloon_10atm"), md, plm))
            st_ivl = None
            if "ivl" not in row or (stent and "stent_post_ivl" not in row):
                st_ivl = NonlinearState(md, ms)
                _timed(row, "ivl", lambda: _protocol_record(run_ivl(st_ivl, T=T), md, plm))
            if stent:
                if "stent_post_ivl" not in row:
                    _timed(row, "stent_post_ivl", lambda: _protocol_record(run_stent(st_ivl, K_s=K_s), md, plm))
                if "stent_no_ivl" not in row:
                    _timed(row, "stent_no_ivl",
                           lambda: _protocol_record(run_stent(NonlinearState(md, ms), K_s=K_s), md, plm))
                if include_erosion and "ivl_erosion" not in row:
                    _timed(row, "ivl_erosion", lambda: _protocol_record(
                        run_ivl(NonlinearState(md, ms, fracture="erosion"), T=T), md, plm))
            rows.append(row)
            _save()
        out["lesions"][lid] = rows
    if aggregate:
        write_json("pilot_nonlinear", out)
    return out


def merge_nonlinear() -> dict:
    """Combine per-lesion pilot files (parallel runs) into results/pilot_nonlinear.json."""
    from dtivl.drivers.common import read_json

    out = {"lesions": {}}
    for lid in pilot_ids():
        part = read_json(f"pilot_nonlinear_{lid}")
        out["lesions"][lid] = part["rows"]
        out["K_s"] = part["K_s"]
    write_json("pilot_nonlinear", out)
    return out
