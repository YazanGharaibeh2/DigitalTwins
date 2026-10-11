"""Stage 2: verification (analytical, mesh convergence, numerical sensitivities).

Only verification is done here: comparison with exact solutions and
discretisation-error estimates. Nothing in this stage uses clinical outcomes.
"""
from __future__ import annotations

import math

from dtivl import params as P
from dtivl.drivers.common import MESH_DIR, RESULTS_DIR, Timer, load_mask, write_json
from dtivl.materials.constitutive import MaterialSet
from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap
from dtivl.metrics.convergence import gci_three_grid, representative_size
from dtivl.metrics.predictors import predictor_analysis

#: Lesions used for the mesh-convergence study (open arc with edges; ring with nodule).
GCI_LESIONS = ("C-A180-T0.2-SUP", "C-A360-T0.4-NOD")
#: Refinement factors applied to the default (coarse) mesh sizes; ratio 2 between levels.
GCI_FACTORS = (2.0, 1.0, 0.5)
GCI_QUANTITIES = ("sigma1_peak", "sigma1_p95", "sigma1_mean", "sed_mean", "sed_peak")
FOUNDATION_LEVELS_KEY = "loads.perivascular_k_MPa_per_mm"


def lame() -> dict:
    from dtivl.solvers.linear import verification_lame

    return verification_lame(RESULTS_DIR / "verification_lame.json")


def _mesh(lesion_id: str, h_ca: float, h_soft: float, tag: str):
    plm = load_mask(lesion_id)
    info = mesh_from_labelmap(plm, MESH_DIR / "verification" / f"{lesion_id}_{tag}.msh", h_ca=h_ca, h_soft=h_soft)
    md = load_mesh(info["msh"], gdim=2)
    return plm, info, md


def mesh_convergence(lesions=GCI_LESIONS, factors=GCI_FACTORS) -> dict:
    """Three-grid GCI (Celik 2008) of the linear predictors on representative lesions."""
    h_ca0 = P.get("discretisation.h_calcium_mm").value
    h_soft0 = P.get("discretisation.h_soft_mm").value
    ms = MaterialSet.baseline()
    out = {"factors": list(factors), "h_ca_default": h_ca0, "h_soft_default": h_soft0, "lesions": {}}
    for lid in lesions:
        levels = []
        for f in factors:
            with Timer() as t:
                plm, info, md = _mesh(lid, h_ca0 * f, h_soft0 * f, f"f{f:g}")
                pred = predictor_analysis(md, plm, ms)
            total_area = sum(info["areas"].values())
            ca_area = info["areas"].get(4, float("nan"))
            levels.append({
                "factor": f, "h_ca": h_ca0 * f, "h_soft": h_soft0 * f,
                "n_cells": info["n_cells"], "n_cells_calcium": info["n_cells_per_tag"].get(4),
                "h_rep_all": representative_size(total_area, info["n_cells"], 2),
                "h_rep_calcium": representative_size(ca_area, info["n_cells_per_tag"].get(4), 2),
                "ndofs": pred["ndofs"], "seconds": t.seconds,
                **{q: pred[q] for q in GCI_QUANTITIES},
            })
        levels.sort(key=lambda d: d["h_rep_calcium"])  # finest first
        gci = {}
        for q in GCI_QUANTITIES:
            h = tuple(d["h_rep_calcium"] for d in levels)
            phi = tuple(d[q] for d in levels)
            gci[q] = gci_three_grid(h, phi)
            # relative difference between the default (coarse, factor 1) mesh and the finest mesh
            default = next(d for d in levels if d["factor"] == 1.0)
            gci[q]["rel_diff_default_vs_fine"] = abs(default[q] - levels[0][q]) / abs(levels[0][q])
            ext = gci[q]["phi_ext"]
            gci[q]["rel_err_default_vs_extrapolated"] = (abs(default[q] - ext) / abs(ext)
                                                         if ext not in (None, 0) and math.isfinite(ext) else None)
        out["lesions"][lid] = {"levels": levels, "gci": gci}
    write_json("verification_mesh_convergence", out)
    return out


def foundation_sensitivity(lesion: str = GCI_LESIONS[0]) -> dict:
    """Linear predictors vs perivascular foundation stiffness (rigid-body stabiliser)."""
    q = P.get(FOUNDATION_LEVELS_KEY)
    levels = (q.value,) + tuple(q.sens)
    plm, info, md = _mesh(lesion, P.get("discretisation.h_calcium_mm").value, P.get("discretisation.h_soft_mm").value, "found")
    ms = MaterialSet.baseline()
    rows = []
    for k in levels:
        pred = predictor_analysis(md, plm, ms, k_found=k)
        rows.append({"k_found": k, **{x: pred[x] for x in GCI_QUANTITIES}, "foundation_ratio": pred["foundation_ratio"]})
    base = rows[0]
    for r in rows:
        r["rel_change_sigma1_peak"] = abs(r["sigma1_peak"] - base["sigma1_peak"]) / abs(base["sigma1_peak"])
    out = {"lesion": lesion, "rows": rows}
    write_json("verification_foundation_linear", out)
    return out
