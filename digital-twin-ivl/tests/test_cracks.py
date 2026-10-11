"""Tests for crack post-processing (design.md section 5.5)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl.fracture.cracks import calcium_deposits, cell_values, crack_analysis
from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER, TAG_CALCIUM
from dtivl.geometry.polar import PolarLabelMap

A, B, R1, R2, ARC = 0.8, 1.5, 0.9, 1.1, 180.0


def _plm(segments, a=A, b=B, r1=R1, r2=R2, n_theta=360, dr=0.01):
    """Label map with calcium in r1..r2 over the angular segments [(lo, hi), ...] (deg)."""
    th = np.degrees(2 * np.pi * np.arange(n_theta) / n_theta)
    r = (np.arange(int(np.ceil(b / dr))) + 0.5) * dr
    TH, R = np.meshgrid(th, r, indexing="ij")
    lab = np.full(TH.shape, LABEL_OTHER, np.uint8)
    lab[R < a] = LABEL_LUMEN
    for lo, hi in segments:
        ang = (TH - lo) % 360.0
        lab[(R >= r1) & (R < r2) & (ang <= (hi - lo) % 360.0)] = LABEL_CALCIUM
    return PolarLabelMap(lab, dr, np.full(n_theta, b))


@pytest.fixture(scope="module")
def sector():
    from dtivl.solvers.nonlinear import annulus_meshdata

    return annulus_meshdata(A, B, 0.2, ca_sector=(R1, R2, ARC), h_ca=0.04)


def _cells(md):
    msh = md.mesh
    n = msh.topology.index_map(2).size_local
    cen = msh.geometry.x[msh.geometry.dofmaps[0][:n]].mean(axis=1)
    return n, np.degrees(np.arctan2(cen[:, 1], cen[:, 0])), np.hypot(cen[:, 0], cen[:, 1])


def test_calcium_deposits_wraparound():
    lab, deps = calcium_deposits(_plm([(350.0, 10.0), (90.0, 135.0)]))
    assert len(deps) == 2
    d0 = min(deps, key=lambda d: abs(((np.degrees(d["theta_center"]) + 180) % 360) - 180))
    assert abs(((np.degrees(d0["theta_center"]) + 180) % 360) - 180) < 1.0
    assert np.degrees(d0["half_span"]) == pytest.approx(10.0, abs=1.0)
    ring = calcium_deposits(_plm([(0.0, 359.99)]))[1]
    assert len(ring) == 1 and ring[0]["closed"]


def test_crack_classification_synthetic(sector):
    md = sector
    n, th, r = _cells(md)
    is_ca = np.zeros(n, bool)
    is_ca[md.cell_tags.find(TAG_CALCIUM)] = True
    dval = np.zeros(n)
    through = is_ca & (np.abs(th) < 3.0)                               # radial, full thickness
    partial = is_ca & (th > 40.0) & (th < 44.0) & (r < 1.0)            # from the luminal surface
    internal = is_ca & (th > -60.0) & (th < -30.0) & (np.abs(r - 1.0) < 0.03)  # circumferential
    dval[through | partial | internal] = 1.0
    hist = [{"phase": "x", "load": 0.1, "max_d": 0.5}, {"phase": "x", "load": 0.2, "max_d": 0.95}]
    out = crack_analysis(md, dval, _plm([(-90.0, 90.0)]), history=hist)
    for key in ("fracture", "n_fractures", "n_partial", "max_depth_frac", "pattern", "location_frac",
                "onset_load"):
        assert key in out
    assert out["fracture"] is True
    assert out["n_fractures"] == 1 and out["n_partial"] == 2
    assert out["max_depth_frac"] == 1.0
    assert out["onset_load"] == 0.2 and out["onset_phase"] == "x"
    comps = {round(c["angle_deg"]): c for c in out["components"]}
    c_thr = next(c for c in out["components"] if c["through"])
    assert c_thr["pattern"] == "radial"
    assert c_thr["location_frac"] == pytest.approx(0.0, abs=0.03)
    c_par = next(c for c in out["components"] if c["touches_luminal"] and not c["through"])
    assert not c_par["touches_abluminal"]
    assert 0.3 < c_par["depth_frac"] < 0.8
    assert c_par["location_frac"] == pytest.approx(42.0 / 90.0, abs=0.05)
    c_int = next(c for c in out["components"] if not c["touches_luminal"] and not c["touches_abluminal"])
    assert c_int["pattern"] == "circumferential"
    assert c_int["location_frac"] == pytest.approx(45.0 / 90.0, abs=0.05)
    assert len(out["pattern"]) == len(out["location_frac"]) == 3
    assert comps  # components carry angles


def test_end_damage_is_not_a_fracture(sector):
    """Damage at a deposit end touches both surfaces through the end face but does not cut the
    deposit; a band that detaches a tip fragment does."""
    md = sector
    n, th, r = _cells(md)
    is_ca = np.zeros(n, bool)
    is_ca[md.cell_tags.find(TAG_CALCIUM)] = True
    tip = np.zeros(n)
    tip[is_ca & (th > 84.0)] = 1.0                                     # the whole end region
    out = crack_analysis(md, tip, _plm([(-90.0, 90.0)]))
    assert out["n_fractures"] == 0 and out["n_partial"] == 1 and not out["fracture"]
    c = out["components"][0]
    assert c["spans_surfaces"] and not c["cuts_deposit"]
    detach = np.zeros(n)
    detach[is_ca & (th > 78.0) & (th < 82.0)] = 1.0                   # band leaving an intact tip
    out = crack_analysis(md, detach, _plm([(-90.0, 90.0)]))
    assert out["fracture"] and out["n_fractures"] == 1


def test_ring_cut_counts_as_fracture():
    """A closed 360-degree ring: one radial through band cuts it open (fracture); a partial-depth
    band does not."""
    from dtivl.solvers.nonlinear import annulus_meshdata

    md = annulus_meshdata(A, B, 0.2, ca_sector=(R1, R2, 360.0), h_ca=0.05)
    n, th, r = _cells(md)
    is_ca = np.zeros(n, bool)
    is_ca[md.cell_tags.find(TAG_CALCIUM)] = True
    plm = _plm([(0.0, 359.99)])
    band = np.zeros(n)
    band[is_ca & (np.abs(th - 120.0) < 3.0)] = 1.0
    out = crack_analysis(md, band, plm)
    assert out["fracture"] and out["n_fractures"] == 1
    assert out["location_frac"] == [None]
    two = band.copy()
    two[is_ca & (np.abs(th + 60.0) < 3.0)] = 1.0
    assert crack_analysis(md, two, plm)["n_fractures"] == 2
    part = np.zeros(n)
    part[is_ca & (np.abs(th - 120.0) < 3.0) & (r < 1.0)] = 1.0
    out = crack_analysis(md, part, plm)
    assert not out["fracture"] and out["n_partial"] == 1


def test_no_crack_and_function_input(sector):
    from dolfinx import fem

    md = sector
    D = fem.functionspace(md.mesh, ("Lagrange", 1))
    d = fem.Function(D)
    out = crack_analysis(md, d, _plm([(-90.0, 90.0)]))
    assert out["fracture"] is False and out["n_fractures"] == 0 and out["n_partial"] == 0
    assert out["max_depth_frac"] == 0.0 and out["pattern"] == [] and out["onset_load"] is None
    d.x.array[:] = 1.0
    vals = cell_values(md, d)
    assert np.allclose(vals, 1.0)
    Q0 = fem.functionspace(md.mesh, ("DG", 0))
    q = fem.Function(Q0)
    q.x.array[:] = 0.25
    assert np.allclose(cell_values(md, q), 0.25)
    full = crack_analysis(md, d, _plm([(-90.0, 90.0)]))
    assert full["fracture"] and full["n_fractures"] == 1
    assert full["cracked_measure"] == pytest.approx(full["ca_measure"])


def test_crack_analysis_3d_tube():
    """3D: a radial through-thickness band over the whole length is one radial fracture."""
    from dtivl.solvers.nonlinear import tube_meshdata

    md = tube_meshdata(A, B, 0.6, 0.12, ca_sector=(R1, R2, ARC))
    msh = md.mesh
    n = msh.topology.index_map(3).size_local
    cen = msh.geometry.x[msh.geometry.dofmaps[0][:n]].mean(axis=1)
    th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0]))
    is_ca = np.zeros(n, bool)
    is_ca[md.cell_tags.find(TAG_CALCIUM)] = True
    dval = np.where(is_ca & (np.abs(th - 30.0) < 3.0), 1.0, 0.0)
    out = crack_analysis(md, dval, _plm([(-90.0, 90.0)]))
    assert out["fracture"] and out["n_fractures"] == 1 and out["n_partial"] == 0
    assert out["pattern"] == ["radial"]
    assert out["location_frac"][0] == pytest.approx(30.0 / 90.0, abs=0.06)
