"""Tests for the finite-strain solver and load protocols (design.md section 5)."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl import params as P
from dtivl.fracture.cracks import crack_analysis
from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER, TAG_FIBROUS, TAG_WALL
from dtivl.geometry.polar import PolarLabelMap
from dtivl.materials.constitutive import MaterialSet
from dtivl.solvers import nonlinear as NL

# Small test lesion: lumen a, outer b, 180-degree calcium sector r1..r2 centred on +x.
A, B, R1, R2, ARC = 0.8, 1.5, 0.9, 1.1, 180.0
H, H_CA = 0.2, 0.06
CAP = 0.84          # envelope cap close to the lumen: partial damage, short run

CRACK_KEYS = {"fracture", "n_fractures", "n_partial", "max_depth_frac", "pattern", "location_frac",
              "onset_load"}
STENT_KEYS = {"lumen_area_pre", "lumen_area_post", "lumen_gain", "msa", "stent_expansion_pct",
              "acute_recoil_pct", "malapposition_area", "malapposed_frac", "reached_target",
              "max_contact_pressure_mean"}


def sector_plm(a=A, b=B, r1=R1, r2=R2, arc=ARC, n_theta=360, dr=0.01) -> PolarLabelMap:
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    r = (np.arange(int(np.ceil(b / dr))) + 0.5) * dr
    TH, R = np.meshgrid(th, r, indexing="ij")
    lab = np.full(TH.shape, LABEL_OTHER, np.uint8)
    lab[R < a] = LABEL_LUMEN
    dth = np.abs((np.degrees(TH) + 180.0) % 360.0 - 180.0)
    lab[(R >= r1) & (R < r2) & (dth <= arc / 2)] = LABEL_CALCIUM
    plm = PolarLabelMap(lab, dr, np.full(n_theta, b))
    plm.validate()
    return plm


@pytest.fixture(scope="module")
def sector_mesh():
    return NL.annulus_meshdata(A, B, H, ca_sector=(R1, R2, ARC), h_ca=H_CA)


@pytest.fixture(scope="module")
def tube_mesh():
    return NL.annulus_meshdata(1.0, 2.0, 0.15)


# ---------------------------------------------------------------------------
def test_annulus_mesh_tags(sector_mesh):
    md = sector_mesh
    assert md.cell_tags.find(4).size > 0 and md.cell_tags.find(TAG_FIBROUS).size > 0
    st = NL.NonlinearState(md, h_ca=H_CA, fracture="none")
    assert st.has_calcium
    # calcium area of the 180-degree sector
    assert st.cell_damage().size == st.ca_cells.size
    from dolfinx import fem
    area = fem.assemble_scalar(fem.form(1.0 * st.dx(4)))
    assert area == pytest.approx(np.pi * (R2 ** 2 - R1 ** 2) / 2, rel=0.01)
    # lumen polyline: closed CCW loop at r = a; Nanson area matches the polygon
    loop = st.lumen_loop()
    assert np.allclose(np.hypot(*loop.T), A, atol=A * (1 - np.cos(H / A)))
    obs = st.observables()
    from dtivl.metrics.outcomes import polygon_area
    assert obs["lumen_area"] == pytest.approx(polygon_area(loop), rel=1e-9)
    assert st.lumen_measure_ref == pytest.approx(2 * np.pi * A, rel=0.01)


@pytest.mark.parametrize("tag", [TAG_FIBROUS, TAG_WALL])
def test_hyperelastic_thick_cylinder_vs_lame(tag):
    """Small-pressure follower-load response of a single soft material vs Lame (< 1 %)."""
    res = NL.hyperelastic_lame_check(tag=tag)
    assert res["status"] == "ok"
    assert res["max_rel_err_ur_lumen"] < 0.01
    assert res["max_rel_err_ur_outer"] < 0.01
    assert res["passed"]


def test_rigid_cylinder_contact():
    """Homogeneous tube expanded by the rigid cylinder (curved mesh)."""
    res = NL.contact_check()
    assert res["status"] == "ok"
    rows = res["rows"]
    pm = [r["p_contact_mean_MPa"] for r in rows]
    assert pm[0] > 0.0 and all(np.diff(pm) > 0.0)
    for r in rows:
        assert r["frac_nodes_in_contact"] == 1.0
        # contact nodes lie at rho = R within the penalty penetration p_c / k_pen
        assert r["max_abs_R_minus_rho_mm"] <= 1.5 * r["penalty_penetration_mm"]
        assert r["rel_err_p_vs_lame"] < 0.1
    assert res["passed"]


def test_contact_linear_mesh_pressure_monotone(tube_mesh):
    """Same expansion on the (polygonal) linear mesh used by the pipeline."""
    st = NL.NonlinearState(tube_mesh, fracture="none")
    R0 = (1 - NL.STENT_START_GAP_FRAC) * st.min_lumen_radius()
    status = NL._ramp(st, phase="c", load_name="R", start=R0, end=1.04, n_steps=4,
                      apply=lambda R: st.set_loads(R_in=R))
    assert status == "ok"
    p = [r["p_contact_mean"] for r in st.history]
    assert p[-1] > 0 and all(b >= a for a, b in zip(p, p[1:]))
    X, x = st.lumen_points()
    rho = np.hypot(x[:, 0], x[:, 1])
    # On the polygonal lumen the cylinder touches the chord midpoints; every lumen node lies at
    # rho = R within the chord sag plus the largest penalty penetration.
    order = np.argsort(np.arctan2(X[:, 1], X[:, 0]))
    pts = X[order]
    chord = 2.0 * np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T).max()
    sag = 1.0 * (1.0 - np.cos(chord / 2.0))
    pen = st.history[-1]["p_contact_max"] / st.k_pen
    assert np.abs(1.04 - rho).max() <= sag + pen + 1e-12


@pytest.mark.parametrize("T", [0.05])
def test_ivl_protocol_end_to_end(sector_mesh, T):
    st = NL.NonlinearState(sector_mesh, h_ca=H_CA, settings=NL.SolverSettings(stagger_max_it=30))
    res = NL.run_ivl(st, T=T, cap_radius=CAP, n_steps=(4, 3, 3))
    state, hist = res                      # unpackable as (state, history)
    assert res.status == "ok"
    phases = [r["phase"] for r in hist]
    for ph in ("ivl_inflate", "ivl_pulse", "ivl_unload", "ivl_release"):
        assert ph in phases
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    p_top = p_bal + T * P.get("loads.ivl_peak_pressure_MPa").value * P.get("loads.ivl_pulse_scale").value
    assert max(r["p"] for r in hist) == pytest.approx(p_top)
    assert hist[-1]["p"] == 0.0 and hist[-1]["R_cap"] is None
    for key in ("load", "max_d", "n_crack_cells", "lumen_area", "p_contact_mean", "p_cap_mean",
                "E_elastic", "E_fracture", "E_contact", "E_foundation", "stagger_its"):
        assert key in hist[0]
    # damage is irreversible along the protocol
    md = [r["max_d"] for r in hist]
    assert all(b >= a - 1e-12 for a, b in zip(md, md[1:]))
    # the cap limits the lumen expansion (lumen radius <= cap + penalty penetration)
    pen = max(r["p_cap_mean"] for r in hist) / st.k_pen
    assert max(r["lumen_area"] for r in hist) <= np.pi * (CAP + 3 * pen) ** 2
    crack = crack_analysis(sector_mesh, state.d, sector_plm(), history=hist)
    assert CRACK_KEYS <= set(crack)
    assert isinstance(crack["fracture"], bool)
    assert crack["n_fractures"] + crack["n_partial"] == len(crack["pattern"]) == len(crack["location_frac"])
    assert 0.0 <= crack["max_depth_frac"] <= 1.0
    assert all(p in ("radial", "circumferential", "undetermined") for p in crack["pattern"])
    assert all(lf is None or 0.0 <= lf <= 1.0 for lf in crack["location_frac"])
    if crack["n_fractures"] + crack["n_partial"] > 0:
        assert crack["onset_load"] is not None
    json.dumps({k: crack[k] for k in CRACK_KEYS})


def test_stent_recoil_bisection(tube_mesh):
    st = NL.NonlinearState(tube_mesh)
    K_s = 0.5
    res = NL.run_stent(st, K_s=K_s, R_target=1.05, n_steps=5)
    s = res.summary
    assert res.status == "ok" and s["reached_target"]
    assert s["R_dep"] == pytest.approx(1.05)
    assert s["R_min"] <= s["R_final"] <= s["R_dep"]
    # R_final solves P_tissue(R) = K_s (R_dep - R) within the bisection tolerance
    assert abs(s["recoil_residual"]) <= K_s * 2 * NL.RECOIL_TOL_MM + 1e-3 * s["P_dep"]
    m = s["metrics"]
    assert STENT_KEYS <= set(m)
    assert m["msa"] == pytest.approx(np.pi * s["R_final"] ** 2)
    assert m["acute_recoil_pct"] == pytest.approx((s["R_dep"] - s["R_final"]) / s["R_dep"] * 100)
    assert m["lumen_gain"] > 0.0
    assert m["malapposition_area"] == pytest.approx(0.0, abs=1e-9)
    assert m["max_contact_pressure_mean"] == pytest.approx(s["P_dep"])


def test_stent_pressure_limited(tube_mesh):
    st = NL.NonlinearState(tube_mesh)
    p_rated = 0.03
    res = NL.run_stent(st, K_s=0.5, R_target=1.05, n_steps=5, p_rated=p_rated)
    s = res.summary
    assert res.status == "limited" and not s["reached_target"]
    assert s["R_dep"] < 1.05
    assert s["max_contact_pressure_mean"] <= p_rated
    assert s["R_final"] <= s["R_dep"]
    assert all(r["p_contact_mean"] <= p_rated for r in res.history if r["phase"] == "stent_expand")


def test_stent_after_ivl_keeps_damage(sector_mesh):
    """Stent after IVL starts from u = 0 with the post-IVL damage field."""
    st = NL.NonlinearState(sector_mesh, h_ca=H_CA, fracture="none")
    st.d.x.array[:] = 0.3                      # stands for a post-IVL damage field
    st.u.x.array[:] = 0.01
    NL.reset_displacement(st)
    assert np.all(st.u.x.array == 0.0) and np.all(st.d.x.array == 0.3)
    assert st.loads == {"p": 0.0, "R_in": None, "R_cap": None}


def test_run_pressure_physiological(sector_mesh):
    st = NL.NonlinearState(sector_mesh, h_ca=H_CA)
    p_phys = P.get("loads.physiological_pressure").value
    res = NL.run_pressure(st, p_phys, n_steps=2)
    assert res.status == "ok"
    assert res.history[-1]["p"] == pytest.approx(p_phys)
    assert res.history[-1]["lumen_area"] > np.pi * A ** 2 * 0.99
    assert res.history[-1]["E_foundation"] < 1e-3 * res.history[-1]["E_elastic"]


@pytest.mark.slow
def test_3d_tube_pressure_and_stent():
    md = NL.tube_meshdata(1.0, 1.5, 0.6, 0.3, ca_sector=(1.1, 1.3, 180.0))
    assert md.facet_tags.find(30).size > 0
    st = NL.NonlinearState(md, h_ca=0.3)
    assert st.z_range == pytest.approx((0.0, 0.6))
    loop0 = st.lumen_loop()
    from dtivl.metrics.outcomes import polygon_area
    assert polygon_area(loop0) == pytest.approx(np.pi, rel=0.05)
    res = NL.run_pressure(st, 0.01, n_steps=2)
    assert res.status == "ok"
    # u_z = 0 on the end faces
    from dolfinx import fem
    dofs = fem.locate_dofs_topological(st.V.sub(2), 2, md.facet_tags.find(30))
    assert np.abs(st.u.x.array[dofs]).max() < 1e-12
    r2 = NL.run_stent(st, K_s=0.5, R_target=1.03, n_steps=3)
    assert r2.status == "ok"
    s = r2.summary
    assert s["R_min"] <= s["R_final"] <= s["R_dep"]
    assert STENT_KEYS <= set(s["metrics"])


def test_verification_nonlinear_json(tmp_path):
    rep = NL.verification_nonlinear(tmp_path / "v.json")
    assert (tmp_path / "v.json").exists()
    assert rep["passed"], rep
