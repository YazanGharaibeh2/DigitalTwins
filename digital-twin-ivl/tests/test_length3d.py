"""Tests for the D12 3D length-series stage (src/drivers/stage_length3d.py)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from dtivl import params as P
from dtivl.drivers import stage_length3d as L3
from dtivl.geometry.labels import TAG_WALL
from dtivl.materials.constitutive import MaterialSet
from dtivl.solvers.nonlinear import DEFAULT_STEPS


def test_load_ramp_follows_the_2d_ivl_protocol():
    p_bal = P.get("loads.ivl_balloon_pressure_atm").value
    r = L3.load_ramp(1.0)
    n1, n2 = DEFAULT_STEPS["ivl_inflate"], DEFAULT_STEPS["ivl_pulse"]
    assert len(r) == n1 + n2
    assert [ph for ph, _ in r] == ["ivl_inflate"] * n1 + ["ivl_pulse"] * n2
    assert r[n1 - 1][1] == pytest.approx(p_bal) and r[-1][1] == pytest.approx(1.0)
    p = np.array([x for _, x in r])
    assert np.all(np.diff(p) > 0)


def test_design_is_from_config():
    d = L3.design()
    assert d["lengths_mm"] == [float(x) for x in P.raw("design.lengths_3d_mm")]
    assert d["h_ca"] == float(P.raw("discretisation.example3d")["h_calcium_mm"])
    from dtivl.drivers.stage_cohort_runs import cohort_ids
    assert set(d["cross_sections"]) <= set(cohort_ids(("core",)))
    # at least 2 (quadratic) axial layers over the shortest calcified length
    assert min(d["lengths_mm"]) / d["h_axial"] >= 2 - 1e-9


def _sector_plm(a=0.8, b=1.5, r1=0.9, r2=1.1, arc=180.0, n_theta=360, dr=0.01):
    from dtivl.geometry.labels import LABEL_CALCIUM, LABEL_LUMEN, LABEL_OTHER
    from dtivl.geometry.polar import PolarLabelMap

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


def test_onset_ramp_stops_at_first_onset():
    """Small 2D annulus with soft (tissue-level) calcium: the ramp stops at the first step with
    max d >= d_crack, and every earlier step is below it."""
    from dtivl.solvers.nonlinear import annulus_meshdata

    md = annulus_meshdata(0.8, 1.5, 0.2, ca_sector=(0.9, 1.1, 180.0), h_ca=0.08, tag=TAG_WALL)
    ms = MaterialSet.baseline(ca_E=P.get("materials.calcium.E_tissue_level").value)
    ramp = [("ivl_inflate", 0.05 * k) for k in range(1, 21)]
    rec = L3.onset_ramp(md, _sector_plm(), ms, ramp, h_ca=0.08, ell=0.16)
    d_crack = float(P.raw("fracture.d_crack"))
    assert rec["status"] == "onset", rec["status"]
    on = rec["onset"]
    assert on["p"] == pytest.approx(rec["steps"][-1]["load"])
    assert all(s["max_d"] < d_crack for s in rec["steps"][:-1]) and rec["steps"][-1]["max_d"] >= d_crack
    assert on["p_previous"] < on["p"] < ramp[-1][1]
    assert "z_rel_max_d" not in on                     # 2D: no axial position
