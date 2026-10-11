"""Tests for the stent outcome metrics (design.md section 5.5)."""
import math

import numpy as np
import pytest

from dtivl import params as P
from dtivl.metrics.outcomes import malapposition, polygon_area, stent_metrics, wall_radius_along_rays


def circle(R, n=720, center=(0.0, 0.0)):
    t = 2 * np.pi * np.arange(n) / n
    return np.column_stack([center[0] + R * np.cos(t), center[1] + R * np.sin(t)])


def test_polygon_area():
    sq = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float)
    assert polygon_area(sq) == pytest.approx(2.0)
    assert polygon_area(sq[::-1]) == pytest.approx(2.0)
    assert polygon_area(circle(1.0)) == pytest.approx(np.pi, rel=1e-4)


def test_malapposition_concentric():
    area, frac = malapposition(circle(1.3), 1.0, 0.1)
    assert area == pytest.approx(np.pi * (1.3 ** 2 - 1.1 ** 2), rel=1e-3)
    assert frac == 1.0
    area, frac = malapposition(circle(1.05), 1.0, 0.1)
    assert area == pytest.approx(0.0, abs=1e-9) and frac == 0.0


def test_malapposition_eccentric_lumen():
    # lumen circle of radius 1.2 shifted by 0.3 along +x around a stent of radius 1.0
    lum = circle(1.2, center=(0.3, 0.0))
    theta = np.array([0.0, np.pi])
    rw = wall_radius_along_rays(lum, theta, 1.0)
    assert rw == pytest.approx([1.5, 0.9], abs=1e-6)
    area, frac = malapposition(lum, 1.0, 0.1)
    assert 0.0 < frac < 1.0 and area > 0.0


def test_stent_metrics_values():
    D_ref = P.get("geometry.reference_diameter_mm").value
    m = stent_metrics(lumen_area_pre=2.2, lumen_area_post=3.0, lumen_post_xy=circle(1.0),
                      R_dep=1.05, R_final=1.0, reached_target=True, max_contact_pressure_mean=0.5)
    for key in ("lumen_area_pre", "lumen_area_post", "lumen_gain", "msa", "stent_expansion_pct",
                "acute_recoil_pct", "malapposition_area", "malapposed_frac", "reached_target",
                "max_contact_pressure_mean"):
        assert key in m
    assert m["lumen_gain"] == pytest.approx(0.8)
    assert m["msa"] == pytest.approx(math.pi)
    assert m["stent_expansion_pct"] == pytest.approx(100.0 / (0.5 * D_ref) ** 2)
    assert m["acute_recoil_pct"] == pytest.approx(0.05 / 1.05 * 100)
    assert m["malapposition_area"] == pytest.approx(0.0, abs=1e-9)
    assert m["malapposed_frac"] == 0.0
    assert m["malapposition_gap_mm"] == P.get("loads.malapposition_gap_mm").value
    m2 = stent_metrics(lumen_area_pre=2.2, lumen_area_post=3.0, lumen_post_xy=None, R_dep=1.0, R_final=1.0,
                       reached_target=False, max_contact_pressure_mean=1.0)
    assert math.isnan(m2["malapposition_area"]) and m2["reached_target"] is False
