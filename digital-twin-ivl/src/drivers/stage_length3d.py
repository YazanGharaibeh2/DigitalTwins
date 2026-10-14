"""Stage: D12 3D calcium-length series (``design.lengths_3d_mm``) to damage onset.

For each cross-section of ``design.length_series_3d.cross_sections`` (core
lesions chosen by the analyst after the 2D cohort runs; the criterion is in the
config basis) and each length L, the cross-section is extruded over the
calcified length L with calcium-free margins (``geometry.margin_3d_mm``) and
loaded by the lumen pressure under the IVL balloon cap. The pressure follows
the inflation and pulse ramps of the 2D IVL protocol (``DEFAULT_STEPS``
``ivl_inflate`` steps to p_bal, ``ivl_pulse`` steps to p_top), with p_top the
peak equivalent IVL pressure of the lesion's 2D cohort run. The ramp stops at
the first step with max d >= ``fracture.d_crack`` (damage onset), so the 3D
runs never enter damage propagation (D12).

The same ramp is run on the 2D cross-section meshed with the same in-plane
sizes (``discretisation.example3d``), the reference of the 3D/2D onset ratio.
Records per run: onset load and phase, the axial position of the damage
maximum relative to the calcified block (0 and 1: its ends), the crack summary
at onset, the 3D Fujino score, mesh sizes and run time. Results:
``results/length3d.json``, written after every run; finished runs are reused.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dtivl import params as P
from dtivl.drivers.common import COHORT_DIR, MESH_DIR, RESULTS_DIR, Timer, load_mask, write_json
from dtivl.drivers.stage_cohort_runs import OUT_DIR as COHORT_RUNS_DIR
from dtivl.materials.constitutive import MaterialSet
from dtivl.meshing.extrude3d import mesh3d_from_labelmap
from dtivl.meshing.mask_to_mesh import load_mesh, mesh_from_labelmap

OUT_NAME = "length3d"
CRACK_KEYS = ("fracture", "n_fractures", "n_partial", "diffuse", "damaged_fraction", "max_depth_frac", "pattern")


def design() -> dict:
    cfg = P.raw("design.length_series_3d")
    sz = P.raw("discretisation.example3d")
    return {"cross_sections": list(cfg["cross_sections"]), "lengths_mm": [float(x) for x in P.raw("design.lengths_3d_mm")],
            "h_ca": float(sz["h_calcium_mm"]), "h_soft": float(sz["h_soft_mm"]),
            "h_axial": float(cfg["h_axial_ca_mm"]), "h_axial_margin": float(cfg["h_axial_margin_mm"])}


def load_ramp(p_top: float) -> list[tuple[str, float]]:
    """(phase, lumen pressure) steps of the 2D IVL protocol up to the pulse peak."""
    from dtivl.solvers.nonlinear import DEFAULT_STEPS

    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    n1, n2 = DEFAULT_STEPS["ivl_inflate"], DEFAULT_STEPS["ivl_pulse"]
    return ([("ivl_inflate", p_bal * k / n1) for k in range(1, n1 + 1)]
            + [("ivl_pulse", p_bal + (p_top - p_bal) * k / n2) for k in range(1, n2 + 1)])


def onset_ramp(md, plm, ms, ramp, *, h_ca: float, ell: float, z_block: tuple[float, float] | None = None,
               name: str = "length3d") -> dict:
    """Run ``ramp`` with the IVL balloon cap and stop at damage onset; returns the run record.

    Each nominal increment is one adaptive ramp of the solver (increment halving on failure, as in the 2D
    protocols), so the onset load is the first recorded load with max d >= d_crack, as in
    :func:`dtivl.fracture.cracks.crack_analysis`.
    """
    from dtivl.fracture.cracks import crack_analysis
    from dtivl.solvers.nonlinear import NonlinearState, _ramp

    d_crack = float(P.raw("fracture.d_crack"))
    st = NonlinearState(md, ms, h_ca=h_ca, ell=ell, name=name)
    st.set_loads(R_cap=0.5 * P.get("geometry.ivl_balloon_diameter_mm").value)
    rec = {"status": "no onset up to p_top", "onset": None}
    p_prev = 0.0
    with Timer() as t:
        for phase, p in ramp:
            status = _ramp(st, phase=phase, load_name="p", start=p_prev, end=p, n_steps=1,
                           apply=lambda lam: st.set_loads(p=lam))
            if status != "ok":
                rec["status"] = f"{status} at p = {st.loads['p']:.6g}"
                break
            p_prev = p
            last = st.history[-1]
            if last["max_d"] >= d_crack:
                c = crack_analysis(md, st.d, plm, history=st.history)
                onset = {"p": c.get("onset_load"), "phase": c.get("onset_phase"),
                         "p_previous": st.history[-2]["load"] if len(st.history) > 1 else 0.0,
                         "cracks": {k: c.get(k) for k in CRACK_KEYS}}
                if z_block is not None:
                    x = st.D.tabulate_dof_coordinates()
                    d = st.d.x.array[:x.shape[0]]
                    z0, z1 = z_block
                    onset["z_rel_max_d"] = float((x[int(np.argmax(d)), 2] - z0) / (z1 - z0))
                    dam = d >= d_crack
                    zr = (x[dam, 2] - z0) / (z1 - z0)
                    onset["frac_damaged_dofs_end_quarters"] = float(np.mean((zr < 0.25) | (zr > 0.75)))
                rec.update(status="onset", onset=onset)
                break
    steps = [{k: h.get(k) for k in ("phase", "load", "max_d", "lumen_area", "stagger_its", "stagger_converged")}
             for h in st.history]
    rec.update(steps=steps, n_steps_unconverged_stagger=sum(1 for s in steps if s["stagger_converged"] is False),
               seconds=t.seconds)
    return rec


def _p_top(lid: str) -> float:
    path = COHORT_RUNS_DIR / f"{lid}.json"
    if not path.exists():
        raise FileNotFoundError(f"{lid}: no 2D cohort record (p_top comes from its IVL run)")
    rec = json.loads(path.read_text())
    return float(rec["protocols"]["ivl"]["summary"]["p_top"])


def _descriptors(lid: str) -> dict:
    df = pd.read_csv(COHORT_DIR / "cohort.csv").set_index("lesion_id")
    row = df.loc[lid]
    return {"ca_arc_total_deg": float(row["ca_arc_total_deg"]), "ca_thickness_max": float(row["ca_thickness_max"])}


def run(cross_sections=None, lengths=None, resume: bool = True) -> dict:
    """Run every (cross-section, model) of the design not yet in ``results/length3d.json``."""
    from dtivl.fracture.phasefield import regularisation_length
    from dtivl.geometry.fujino import fujino_score

    des = design()
    cross_sections = list(cross_sections or des["cross_sections"])
    lengths = [float(x) for x in (lengths or des["lengths_mm"])]
    path = RESULTS_DIR / f"{OUT_NAME}.json"
    out = json.loads(path.read_text()) if resume and path.exists() else {}
    out["design"] = des
    ms = MaterialSet.baseline()
    ell = regularisation_length(h_ca=des["h_ca"])
    out.update(ca_E_MPa=ms.ca_E, ell_mm=ell)
    runs = out.setdefault("runs", {})
    for lid in cross_sections:
        p_top = _p_top(lid)
        ramp = load_ramp(p_top)
        dsc = _descriptors(lid)
        plm = load_mask(lid)
        for L in [None] + lengths:                     # None: 2D cross-section at the 3D in-plane sizes
            key = f"{lid}|2d" if L is None else f"{lid}|3d|L{L:g}"
            if key in runs and runs[key].get("status") in ("onset", "no onset up to p_top"):
                continue
            if L is None:
                info = mesh_from_labelmap(plm, MESH_DIR / "length3d" / f"{lid}_2d.msh",
                                          h_ca=des["h_ca"], h_soft=des["h_soft"])
                md, z_block = load_mesh(info["msh"], gdim=2), None
            else:
                info = mesh3d_from_labelmap(plm, L, MESH_DIR / "length3d" / f"{lid}_L{L:g}.msh", h_ca=des["h_ca"],
                                            h_soft=des["h_soft"], h_axial=des["h_axial"],
                                            h_axial_margin=des["h_axial_margin"])
                md, z_block = load_mesh(info["msh"], gdim=3), (info["z_blocks"][1], info["z_blocks"][2])
            rec = onset_ramp(md, plm, ms, ramp, h_ca=des["h_ca"], ell=ell, z_block=z_block,
                             name=f"len3d_{'2d' if L is None else f'L{L:g}'}")
            rec.update(lesion=lid, model="2d" if L is None else "3d", length_mm=L, p_top=p_top,
                       fujino_score=fujino_score(dsc["ca_arc_total_deg"], dsc["ca_thickness_max"], L),
                       mesh={k: info[k] for k in ("n_nodes", "n_cells", "n_cells_per_tag")},
                       n_layers=info.get("n_layers"))
            runs[key] = rec
            out["summary"] = summary(out)
            write_json(OUT_NAME, out)
    return out


def summary(out: dict | None = None) -> dict:
    """Onset load of each 3D run relative to the 2D run of the same cross-section (same in-plane mesh)."""
    out = out or json.loads((RESULTS_DIR / f"{OUT_NAME}.json").read_text())
    rows = {}
    for key, r in out["runs"].items():
        if r["model"] != "3d":
            continue
        ref = out["runs"].get(f"{r['lesion']}|2d")
        p3 = (r.get("onset") or {}).get("p")
        p2 = ((ref or {}).get("onset") or {}).get("p")
        rows[key] = {"lesion": r["lesion"], "length_mm": r["length_mm"], "status": r["status"], "onset_3d": p3,
                     "onset_2d": p2, "ratio_3d_2d": (p3 / p2) if (p3 and p2) else None,
                     "z_rel_max_d": (r.get("onset") or {}).get("z_rel_max_d"), "fujino_score": r["fujino_score"]}
    return rows


if __name__ == "__main__":
    run()
