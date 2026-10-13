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
"""
from __future__ import annotations

import math

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


if __name__ == "__main__":
    localisation()
    ell_sensitivity()
