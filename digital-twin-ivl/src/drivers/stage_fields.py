"""Stage: damage and displacement fields of representative lesions (figure data).

The cohort records (:mod:`dtivl.drivers.stage_cohort_runs`) keep summaries
only. This stage re-runs, for the lesions of ``design.field_lesions``, the IVL
protocol with the phase field, the IVL protocol with the erosion comparator,
and the post-IVL stent protocol, with the same inputs as the cohort runs, and
saves the final fields of each run to ``results/fields/<lesion>_<protocol>.npz``:

* ``points`` (n, 2), ``triangles`` (m, 3): P1 triangulation of the mesh (mm);
* ``cell_tag`` (m,): tissue tag of each triangle;
* ``d`` (n,): nodal damage (phase field; zeros for erosion);
* ``d_cell`` (m,): cell damage (cell-averaged phase field, or the eroded indicator);
* ``u`` (n, 2): displacement at the points (mm).

The fields are used only for figures. ``results/fields/fields_summary.json``
records, per run, whether the outcome (fracture flag, damage-onset load)
reproduces the cohort record of the same lesion, as a determinism check.
"""
from __future__ import annotations

import json
import os

import numpy as np

from dtivl import params as P
from dtivl.drivers.common import MESH_DIR, RESULTS_DIR, Timer, load_mask, read_json, write_json
from dtivl.drivers.stage_cohort_runs import OUT_DIR as COHORT_RUNS_DIR
from dtivl.drivers.stage_pilot import _protocol_record
from dtivl.materials.constitutive import MaterialSet
from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap

FIELDS_DIR = RESULTS_DIR / "fields"
FIELD_PROTOCOLS = ("ivl", "ivl_erosion", "stent_post_ivl")


def field_lesions() -> list[str]:
    return list(P.raw("design.field_lesions")["lesions"])


def _mesh(lid: str):
    """Mesh into a private path so the stage never collides with the cohort workers' meshes."""
    plm = load_mask(lid)
    path = MESH_DIR / "fields" / f"{lid}.{os.getpid()}.msh"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = mesh_from_labelmap(plm, path)
    md = load_mesh(info["msh"], gdim=2)
    path.unlink(missing_ok=True)
    return plm, info, md


def extract_fields(state) -> dict:
    """P1 triangulation, tags, nodal and cell damage and displacement of a 2D state (serial)."""
    from dolfinx import fem

    msh = state.mesh
    V1 = fem.functionspace(msh, ("Lagrange", 1, (2,)))
    S1 = fem.functionspace(msh, ("Lagrange", 1))
    u1, d1 = fem.Function(V1), fem.Function(S1)
    u1.interpolate(state.u)
    if state.fracture == "phasefield":
        d1.interpolate(state.d)
    tri = np.asarray(S1.dofmap.list, dtype=np.int64)
    pts = S1.tabulate_dof_coordinates()[:, :2]
    # vector P1 dofs share the scalar P1 dof layout (block size 2)
    u = u1.x.array.reshape(-1, 2)
    n_cells = msh.topology.index_map(2).size_local
    tag = np.zeros(n_cells, dtype=np.int32)
    ct = state.cell_tags
    keep = ct.indices < n_cells
    tag[ct.indices[keep]] = ct.values[keep]
    d_cell = np.zeros(n_cells)
    d_cell[state.ca_cells] = state.cell_damage()
    return {"points": pts, "triangles": tri[:n_cells], "cell_tag": tag, "d": d1.x.array[:pts.shape[0]].copy(),
            "d_cell": d_cell, "u": u[:pts.shape[0]].copy()}


def _save(lid: str, protocol: str, fields: dict) -> str:
    FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    path = FIELDS_DIR / f"{lid}_{protocol}.npz"
    np.savez_compressed(path, **fields)
    return str(path.relative_to(RESULTS_DIR.parent))


def _matches(rec: dict, cohort_rec: dict | None) -> dict:
    """Outcome of the re-run against the cohort record (None: no cohort record for this protocol)."""
    if not cohort_rec:
        return {"cohort_record": False}
    a, b = rec.get("cracks", {}), cohort_rec.get("cracks", {})
    oa, ob = a.get("onset_load"), b.get("onset_load")
    same_onset = (oa is None and ob is None) or (oa is not None and ob is not None and abs(oa - ob) <= 1e-9 * max(1.0, abs(ob)))
    return {"cohort_record": True, "fracture_matches": a.get("fracture") == b.get("fracture"),
            "onset_matches": bool(same_onset), "status_matches": rec.get("status") == cohort_rec.get("status")}


def run_lesion(lid: str) -> dict:
    """Run the field protocols of one lesion; returns the per-protocol summary."""
    from dtivl.solvers.nonlinear import NonlinearState, run_ivl, run_stent

    ms = MaterialSet.baseline()
    cpath = COHORT_RUNS_DIR / f"{lid}.json"
    if not cpath.exists():
        raise FileNotFoundError(f"{lid}: no cohort record; the field stage reuses its T_equivalent")
    crec = json.loads(cpath.read_text())
    if abs(crec.get("ca_E_MPa", ms.ca_E) - ms.ca_E) > 1e-9:
        raise ValueError(f"{lid}: cohort record is for E = {crec['ca_E_MPa']} MPa, baseline is {ms.ca_E} MPa")
    T = crec["transient"]["T_equivalent"]
    K_s = read_json("calibration_stent")["K_s_MPa_per_mm"]
    plm, info, md = _mesh(lid)
    out = {}

    def _do(key, state, fn):
        with Timer() as t:
            res = fn()
        rec = _protocol_record(res, md, plm)
        out[key] = {"status": rec["status"], "fracture": rec["cracks"].get("fracture"),
                    "onset_load": rec["cracks"].get("onset_load"), "seconds": t.seconds,
                    "file": _save(lid, key, extract_fields(state)),
                    **_matches(rec, crec.get("protocols", {}).get(key))}

    st = NonlinearState(md, ms)
    _do("ivl", st, lambda: run_ivl(st, T=T))
    _do("stent_post_ivl", st, lambda: run_stent(st, K_s=K_s))
    st_e = NonlinearState(md, ms, fracture="erosion")
    _do("ivl_erosion", st_e, lambda: run_ivl(st_e, T=T))
    return out


def _done(entry: dict | None) -> bool:
    return bool(entry) and all(k in entry and (RESULTS_DIR.parent / entry[k]["file"]).exists() for k in FIELD_PROTOCOLS)


def run(ids=None, resume: bool = True) -> dict:
    """Run every lesion of ``ids`` (default ``design.field_lesions``); writes fields_summary.json.
    ``resume``: lesions whose fields are all on disk are skipped."""
    ids = list(ids or field_lesions())
    path = FIELDS_DIR / "fields_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for lid in ids:
        if resume and _done(summary.get(lid)):
            continue
        summary[lid] = run_lesion(lid)
        FIELDS_DIR.mkdir(parents=True, exist_ok=True)
        write_json("fields/fields_summary", summary)
    return summary


if __name__ == "__main__":
    run()
