"""Tests for the linear predictor analysis on a synthetic calcium sector."""
import math

import numpy as np
import pytest

from dtivl import params as P
from dtivl.geometry.labels import (LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER, TAG_CALCIUM, TAG_FIBROUS,
                                   TAG_LUMEN_BND, TAG_OUTER_BND)
from dtivl.geometry.polar import PolarLabelMap
from dtivl.materials.constitutive import MaterialSet
from dtivl.metrics import predictors as PR
from dtivl.solvers import linear as L

R_LUMEN, R_OUT = 1.0, 2.0
R_CA_IN, R_CA_OUT = 1.15, 1.45
H = 0.05
N_THETA, DR = 360, 0.01


def _sector_mesh(arc_deg: float):
    """Annulus (fibrous) with a calcium annular sector centred on +x."""
    from mpi4py import MPI
    from dolfinx.io.gmsh import model_to_mesh

    with L._gmsh_model("sector") as gmsh:
        occ = gmsh.model.occ
        tissue = occ.cut([(2, occ.addDisk(0, 0, 0, R_OUT, R_OUT))],
                         [(2, occ.addDisk(0, 0, 0, R_LUMEN, R_LUMEN))])[0]
        ca = occ.cut([(2, occ.addDisk(0, 0, 0, R_CA_OUT, R_CA_OUT))],
                     [(2, occ.addDisk(0, 0, 0, R_CA_IN, R_CA_IN))])[0]
        if arc_deg < 360:
            half = math.radians(arc_deg) / 2
            ang = np.linspace(-half, half, 33)
            pts = [occ.addPoint(0, 0, 0)] + [occ.addPoint(3 * np.cos(t), 3 * np.sin(t), 0) for t in ang]
            lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
            wedge = occ.addPlaneSurface([occ.addCurveLoop(lines)])
            ca = occ.intersect(ca, [(2, wedge)])[0]
        _, mapping = occ.fragment(tissue, ca)
        occ.synchronize()
        ca_s = sorted({t for m in mapping[len(tissue):] for _, t in m})
        all_s = [t for _, t in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, ca_s, TAG_CALCIUM)
        gmsh.model.addPhysicalGroup(2, [s for s in all_s if s not in ca_s], TAG_FIBROUS)
        inner, outer = [], []
        for _, c in gmsh.model.getBoundary([(2, s) for s in all_s], combined=True, oriented=False):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
            (outer if max(x1 - x0, y1 - y0) > R_LUMEN + R_OUT else inner).append(c)
        gmsh.model.addPhysicalGroup(1, inner, TAG_LUMEN_BND)
        gmsh.model.addPhysicalGroup(1, outer, TAG_OUTER_BND)
        gmsh.option.setNumber("Mesh.MeshSizeMin", H)
        gmsh.option.setNumber("Mesh.MeshSizeMax", H)
        gmsh.model.mesh.generate(2)
        return model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)


def _sector_labelmap(arc_deg: float, center_deg: float = 0.0) -> PolarLabelMap:
    n_r = int(round(R_OUT * 1.05 / DR))
    th = 2 * np.pi * np.arange(N_THETA) / N_THETA
    r = (np.arange(n_r) + 0.5) * DR
    T, R = np.meshgrid(th, r, indexing="ij")
    lab = np.full((N_THETA, n_r), LABEL_OTHER, dtype=np.uint8)
    lab[R < R_LUMEN] = LABEL_LUMEN
    dang = np.abs((T - math.radians(center_deg) + np.pi) % (2 * np.pi) - np.pi)
    lab[(R >= R_CA_IN) & (R < R_CA_OUT) & (dang <= math.radians(arc_deg) / 2 + 1e-9)] = LABEL_CALCIUM
    plm = PolarLabelMap(lab, DR, np.full(N_THETA, R_OUT), None, {"id": f"sector{arc_deg}"})
    plm.validate()
    return plm


@pytest.fixture(scope="module")
def ms():
    return MaterialSet.baseline()


@pytest.fixture(scope="module")
def p_load():
    return P.get("loads.ivl_balloon_pressure_atm").value


@pytest.fixture(scope="module")
def open_arc(ms, p_load):
    md = _sector_mesh(120.0)
    return md, _sector_labelmap(120.0), L.solve_linear(md, ms, p_load)


@pytest.fixture(scope="module")
def ring(ms, p_load):
    md = _sector_mesh(360.0)
    return md, _sector_labelmap(360.0), L.solve_linear(md, ms, p_load)


def _check_finite(pred):
    for k in PR.PREDICTOR_KEYS:
        assert k in pred
        if k != "has_edges":
            assert np.isfinite(pred[k]), k


def test_open_arc_predictors(open_arc, ms):
    md, plm, res = open_arc
    pred = PR.predictors(res, plm, ms)
    _check_finite(pred)
    assert pred["has_edges"] is True
    assert pred["scf_edge"] > 1.0
    assert pred["n_edge_cells"] > 0
    assert pred["sigma1_peak"] >= pred["sigma1_p95"] >= pred["sigma1_mean"] * 0.5
    assert pred["sed_peak"] >= pred["sed_mean"] > 0
    for k in ("frac_eps_gt_half_epsf", "frac_eps_gt_epsf", "frac_sig_gt_half_sigc", "frac_sig_gt_sigc"):
        assert 0.0 <= pred[k] <= 1.0
    assert pred["frac_eps_gt_epsf"] <= pred["frac_eps_gt_half_epsf"]
    assert pred["frac_sig_gt_sigc"] <= pred["frac_sig_gt_half_sigc"]
    # local thickness from the label map: the sector is (R_CA_OUT - R_CA_IN) thick
    assert pred["min_thickness_high_stress"] == pytest.approx(R_CA_OUT - R_CA_IN, abs=1.5 * DR)
    # FE calcium area matches the sector area
    assert pred["ca_measure_fe"] == pytest.approx(np.pi * (R_CA_OUT ** 2 - R_CA_IN ** 2) / 3, rel=0.02)
    assert res.foundation_ratio < 1e-3


def test_ring_has_no_edges(ring, ms):
    md, plm, res = ring
    pred = PR.predictors(res, plm, ms)
    _check_finite(pred)
    assert pred["has_edges"] is False
    assert pred["scf_edge"] == 1.0


def test_open_arc_concentrates_more_than_ring(open_arc, ring, ms):
    p_arc = PR.predictors(open_arc[2], open_arc[1], ms)
    p_ring = PR.predictors(ring[2], ring[1], ms)
    assert p_arc["sigma1_peak"] / p_arc["sigma1_mean"] > p_ring["sigma1_peak"] / p_ring["sigma1_mean"]


def test_linear_scaling_with_pressure(open_arc, ms, p_load):
    md, plm, res1 = open_arc
    res2 = L.solve_linear(md, ms, 2.0 * p_load)
    np.testing.assert_allclose(res2.sigma, 2.0 * res1.sigma, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(res2.sigma1, 2.0 * res1.sigma1, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(res2.eps1, 2.0 * res1.eps1, rtol=1e-6, atol=1e-15)
    np.testing.assert_allclose(res2.sed, 4.0 * res1.sed, rtol=1e-6, atol=1e-15)
    p1, p2 = PR.predictors(res1, plm, ms), PR.predictors(res2, plm, ms)
    assert p2["sigma1_peak"] == pytest.approx(2 * p1["sigma1_peak"], rel=1e-6)
    assert p2["sigma1_p95"] == pytest.approx(2 * p1["sigma1_p95"], rel=1e-6)
    assert p2["sed_mean"] == pytest.approx(4 * p1["sed_mean"], rel=1e-6)
    assert p2["scf_edge"] == pytest.approx(p1["scf_edge"], rel=1e-6)


def test_predictor_analysis_default_load(open_arc, ms, p_load):
    md, plm, res = open_arc
    pred = PR.predictor_analysis(md, plm, ms)
    assert pred["p_lumen_MPa"] == pytest.approx(p_load)
    assert pred["sigma1_peak"] == pytest.approx(PR.predictors(res, plm, ms)["sigma1_peak"], rel=1e-8)


def test_deposits_wraparound_and_fragments():
    plm = _sector_labelmap(60.0, center_deg=0.0)       # straddles theta = 0
    deps = PR.deposits(plm)
    assert len(deps) == 1 and not deps[0]["closed"] and len(deps[0]["ends"]) == 2
    assert deps[0]["t_ca"] == pytest.approx(R_CA_OUT - R_CA_IN, abs=1.5 * DR)
    ends = sorted(np.degrees(np.mod(deps[0]["ends"], 2 * np.pi)))
    assert ends[0] == pytest.approx(30.5, abs=1.0) and ends[1] == pytest.approx(329.5, abs=1.0)
    # two separate deposits
    lab = plm.labels.copy()
    lab[_sector_labelmap(40.0, center_deg=180.0).labels == LABEL_CALCIUM] = LABEL_CALCIUM
    plm2 = PolarLabelMap(lab, DR, plm.r_out)
    assert len(PR.deposits(plm2)) == 2
    assert PR.deposits(_sector_labelmap(360.0))[0]["closed"]


def test_weighted_quantile():
    v = np.array([3.0, 1.0, 2.0, 4.0])
    assert PR.weighted_quantile(v, np.ones(4), 0.5) == 2.0
    assert PR.weighted_quantile(v, np.array([0, 0, 0, 1.0]), 0.1) == 4.0
    assert PR.weighted_quantile(v, np.ones(4), 1.0) == 4.0


def test_local_thickness_fallback():
    plm = _sector_labelmap(60.0)
    t = PR.local_thickness(plm, np.radians([0.0, 90.0]))
    assert t[0] == pytest.approx(R_CA_OUT - R_CA_IN, abs=1.5 * DR)
    assert t[1] == pytest.approx(R_CA_OUT - R_CA_IN, abs=1.5 * DR)   # nearest calcium A-line
