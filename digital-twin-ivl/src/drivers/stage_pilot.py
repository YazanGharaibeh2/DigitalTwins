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
N_E_SWEEP = 5


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


def run_example3d(e_values=None, transient=True, nonlinear=True, resume=True) -> dict:
    """D1: test the 3D code on the Fujino-4 example and compare with the 2D cross-section model.

    ``resume``: rows of ``results/pilot_example3d.json`` that already hold
    every requested analysis for a (model, modulus) pair are reused.
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
    if resume and (ROOT / "results" / "pilot_example3d.json").exists():
        prev = read_json("pilot_example3d")
        for key, m in prev.get("models", {}).items():
            for r in m.get("by_E", []):
                if need <= set(r):
                    old[(key, round(r["ca_E_MPa"], 6))] = r
    for dim, builder in ((3, mesh3d), (2, mesh2d_like3d)):
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
           "n_steps": len(st.history)}
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
    keys = {"physiological", "balloon_only", "ivl"}
    if any(abs(E - e) < 1e-9 for e in stent_e_values):
        keys |= {"stent_post_ivl", "stent_no_ivl"} | ({"ivl_erosion"} if include_erosion else set())
    return keys


def run_nonlinear(lesions=None, e_values=None, stent_e_values=None, include_erosion=True, resume=True) -> dict:
    """Nonlinear phase-field pilot over the modulus sweep.

    For every lesion and modulus: physiological pressure, IVL balloon only
    (4 atm, cap) and IVL with the lesion- and modulus-specific transient T.
    For ``stent_e_values``: stent deployment without IVL and after IVL.
    ``resume``: rows of ``results/pilot_nonlinear_<lesion>.json`` that are
    complete for a modulus (or skipped) are reused, so an interrupted run
    continues where it stopped; the per-lesion file is rewritten after
    every modulus.
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
    R_ivl = 0.5 * P.get("geometry.ivl_balloon_diameter_mm").value
    out = {"K_s": K_s, "p_phys": p_phys, "p_bal": p_bal, "R_ivl": R_ivl, "lesions": {}}
    for lid in lesions:
        done = {}
        if resume and (ROOT / "results" / f"pilot_nonlinear_{lid}.json").exists():
            for r in read_json(f"pilot_nonlinear_{lid}").get("rows", []):
                if "skipped" in r or _expected_protocols(r["ca_E_MPa"], stent_e_values, include_erosion) <= set(r):
                    done[round(r["ca_E_MPa"], 6)] = r
        plm, info, md = mesh2d(lid)
        # T_equivalent = transient peak / static sigma1 of the *quasi-static* model (MaterialSet moduli),
        # i.e. the factor that reproduces the transient peak inside run_ivl (design.md section 5.4).
        Tmap = {round(r["ca_E_MPa"], 6): r["transient"]["T_equivalent"] for r in lt["lesions"][lid]["by_E"]}
        rows = []
        for E in e_values:
            if round(E, 6) in done:
                rows.append(done[round(E, 6)])
                continue
            ms = MaterialSet.baseline(ca_E=E)
            T = Tmap[round(E, 6)]
            row = {"ca_E_MPa": E, "T": T}
            try:
                row["k_floor"] = ms.k_floor
            except ValueError as exc:          # calcium not stiffer than the matrix: crack filler undefined
                row["skipped"] = str(exc)
                rows.append(row)
                write_json(f"pilot_nonlinear_{lid}", {"lesion": lid, "rows": rows, "K_s": K_s})
                continue
            with Timer() as t:
                row["physiological"] = _protocol_record(run_pressure(NonlinearState(md, ms), p_phys), md, plm)
            row["physiological"]["seconds"] = t.seconds
            with Timer() as t:
                row["balloon_only"] = _protocol_record(
                    run_pressure(NonlinearState(md, ms), p_bal, cap_radius=R_ivl, phase="balloon_only"), md, plm)
            row["balloon_only"]["seconds"] = t.seconds
            st_ivl = NonlinearState(md, ms)
            with Timer() as t:
                res_ivl = run_ivl(st_ivl, T=T)
                row["ivl"] = _protocol_record(res_ivl, md, plm)
            row["ivl"]["seconds"] = t.seconds
            if any(abs(E - e) < 1e-9 for e in stent_e_values):
                with Timer() as t:
                    row["stent_post_ivl"] = _protocol_record(run_stent(st_ivl, K_s=K_s), md, plm)
                row["stent_post_ivl"]["seconds"] = t.seconds
                with Timer() as t:
                    row["stent_no_ivl"] = _protocol_record(run_stent(NonlinearState(md, ms), K_s=K_s), md, plm)
                row["stent_no_ivl"]["seconds"] = t.seconds
                if include_erosion:
                    with Timer() as t:
                        row["ivl_erosion"] = _protocol_record(run_ivl(NonlinearState(md, ms, fracture="erosion"), T=T),
                                                              md, plm)
                    row["ivl_erosion"]["seconds"] = t.seconds
            rows.append(row)
            write_json(f"pilot_nonlinear_{lid}", {"lesion": lid, "rows": rows, "K_s": K_s})
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
