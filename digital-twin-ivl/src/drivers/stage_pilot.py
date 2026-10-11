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
from dtivl.drivers.common import MESH_DIR, Timer, load_mask, write_json
from dtivl.geometry import cohort
from dtivl.materials.constitutive import MaterialSet
from dtivl.meshing.extrude3d import mesh3d_from_labelmap
from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap
from dtivl.metrics.predictors import predictor_analysis

#: Number of log-spaced calcium moduli between the cited sensitivity bounds.
N_E_SWEEP = 5
#: Coarsening of the 3D example relative to the 2D defaults (3D cost; user asked for coarse models).
COARSEN_3D = 2.0


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
    spec = cohort.example3d_spec()
    assert spec.lesion_id == lesion_id
    plm = load_mask(lesion_id)
    h_ca = P.get("discretisation.h_calcium_mm").value * COARSEN_3D
    h_soft = P.get("discretisation.h_soft_mm").value * COARSEN_3D
    h_ax = P.get("discretisation.h_axial_3d_mm").value * COARSEN_3D
    info = mesh3d_from_labelmap(plm, spec.ca_length, MESH_DIR / f"{lesion_id}.msh", h_ca=h_ca, h_soft=h_soft, h_axial=h_ax)
    return plm, info, load_mesh(info["msh"], gdim=3)


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
