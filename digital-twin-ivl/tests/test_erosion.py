"""Tests for the element-erosion comparator (design.md section 5.6)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl.fracture import erosion as ER
from dtivl.fracture import phasefield as PF
from dtivl.fracture.cracks import crack_analysis
from dtivl.materials.constitutive import MaterialSet


def _plate_state(ms):
    from dolfinx import fem

    from dtivl.solvers.nonlinear import NonlinearState

    md = PF._box_meshdata(2)
    disp = fem.Constant(md.mesh, 0.0)
    length = PF.VERIF_PLATE[0]
    st = NonlinearState(md, ms, fracture="erosion", bc_factory=PF._uniaxial_bcs(md.mesh, length, disp))
    return st, disp, length


def test_failure_threshold():
    ms = MaterialSet.baseline()
    assert ER.failure_threshold(ms) == ("strain", pytest.approx(ms.eps_f))
    ms_s = MaterialSet.baseline(failure_mode="stress")
    assert ER.failure_threshold(ms_s) == ("stress", pytest.approx(ms_s.sigma_c))


@pytest.mark.parametrize("mode", ["strain", "stress"])
def test_homogeneous_plate_erosion_criterion(mode):
    """Uniaxial plane-strain tension: the cell-averaged measure crosses the threshold at the
    expected strain, and every (identical) cell is flagged in the same step."""
    ms = MaterialSet.baseline(failure_mode=mode)
    st, disp, length = _plate_state(ms)
    eroder = st._eroder
    n_ca = st.ca_cells.size
    steps = (np.arange(12) + 0.55) * 0.1 * ms.eps_f           # 0.1 eps_f increments, off the threshold
    first = None
    for eps in steps:
        disp.value = eps * length
        st._solve_u({"newton_its": 0})
        crit = eroder.measure()
        assert np.ptp(crit) <= 1e-9 * abs(crit).max()          # homogeneous field
        if mode == "strain":
            # homogeneous axial stretch = 1 + applied engineering strain
            assert crit == pytest.approx(np.full(n_ca, eps), rel=1e-6)
        if first is None and crit.min() >= eroder.threshold:
            first = eps
            assert eroder.find_new().size == n_ca              # all cells at once
    assert first is not None
    if mode == "strain":
        assert first - 0.1 * ms.eps_f < ms.eps_f <= first
    else:
        # small-strain plane-strain uniaxial estimate, 5 % allowance for finite strain
        eps_c = ms.sigma_c * (1 - ms.ca_nu ** 2) / ms.ca_E
        assert first - 0.1 * ms.eps_f < 1.05 * eps_c and first >= 0.95 * eps_c
    # marking (without re-solving) flags every cell; the indicator follows
    assert st._eroder.update() == n_ca
    ind = ER.eroded_indicator(st)
    assert ind[st.ca_cells].sum() == n_ca
    assert np.all(st.cell_damage() == 1.0)
    assert ER.ErosionUpdater(st).intact().sum() == 0


def test_erosion_continuation_reaches_k_floor():
    """Adaptive log-continuation ends at eta = k_floor and recovers from a failed stage.

    crack_filler="none" puts the floor at k_res, several decades below 1."""
    st, disp, length = _plate_state(MaterialSet.baseline(crack_filler="none"))
    disp.value = 0.5 * st.ms.eps_f * length
    st._solve_u({"newton_its": 0})
    er = st._eroder
    dofs = er._eta_dofs[:3]
    calls = {"n": 0}

    def flaky_solve():
        calls["n"] += 1
        if calls["n"] == 1:                    # first stage fails -> step halved
            raise RuntimeError("synthetic failure")
        st._solve_u({"newton_its": 0})

    n = er.erode_with_continuation(dofs, flaky_solve)
    assert np.all(st.eta.x.array[dofs] == st.ms.k_floor)
    assert n >= 3 and calls["n"] == n + 1


# Small test lesion (same as tests/test_nonlinear.py): lumen A, outer B, 180-degree calcium sector.
A, B, R1, R2, ARC, H, H_CA = 0.8, 1.5, 0.9, 1.1, 180.0, 0.2, 0.06
CAP_EROSION = 0.83   # tight cap: partial erosion


def sector_plm(n_theta=360, dr=0.01):
    from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER
    from dtivl.geometry.polar import PolarLabelMap

    th = 2 * np.pi * np.arange(n_theta) / n_theta
    r = (np.arange(int(np.ceil(B / dr))) + 0.5) * dr
    TH, R = np.meshgrid(th, r, indexing="ij")
    lab = np.full(TH.shape, LABEL_OTHER, np.uint8)
    lab[R < A] = LABEL_LUMEN
    dth = np.abs((np.degrees(TH) + 180.0) % 360.0 - 180.0)
    lab[(R >= R1) & (R < R2) & (dth <= ARC / 2)] = LABEL_CALCIUM
    return PolarLabelMap(lab, dr, np.full(n_theta, B))


def test_erosion_crack_postprocessing_on_sector():
    from dtivl.solvers import nonlinear as NL

    md = NL.annulus_meshdata(A, B, H, ca_sector=(R1, R2, ARC), h_ca=H_CA)
    st = NL.NonlinearState(md, h_ca=H_CA, fracture="erosion")
    res = NL.run_pressure(st, 0.4, cap_radius=CAP_EROSION, n_steps=4)
    assert res.status == "ok"
    ind = st.damage_field()
    assert ind.shape[0] >= md.mesh.topology.index_map(2).size_local
    n_er = res.history[-1]["n_eroded"]
    assert n_er == int(ind.sum())
    assert n_er > 0
    crack = crack_analysis(md, ind, sector_plm(), history=res.history)
    assert crack["n_cracked_cells"] == n_er
    assert crack["onset_load"] is not None
    assert crack["n_fractures"] + crack["n_partial"] >= 1
    # eroded cells never recover
    ne = [r["n_eroded"] for r in res.history]
    assert all(b >= a for a, b in zip(ne, ne[1:]))
