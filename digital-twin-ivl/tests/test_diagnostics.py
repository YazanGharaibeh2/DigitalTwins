"""Tests for the localisation diagnostics helpers."""
import numpy as np
import pytest

from dtivl.drivers.stage_diagnostics import gaussian_field


def test_gaussian_field_statistics():
    """Unit variance, zero mean and covariance exp(-r^2 / (2 L^2)) (ensemble over seeds)."""
    L = 0.1
    x0 = np.array([[0.0, 0.0], [L, 0.0], [3 * L, 0.0]])
    vals = np.array([gaussian_field(x0, L, seed) for seed in range(2000)])
    assert vals.mean(axis=0) == pytest.approx(np.zeros(3), abs=0.08)
    assert vals.std(axis=0) == pytest.approx(np.ones(3), abs=0.06)
    c = np.corrcoef(vals.T)
    assert c[0, 1] == pytest.approx(np.exp(-0.5), abs=0.06)
    assert c[0, 2] == pytest.approx(np.exp(-4.5), abs=0.06)


def test_gaussian_field_deterministic():
    pts = np.random.default_rng(0).uniform(size=(50, 3))
    assert np.array_equal(gaussian_field(pts, 0.2, 7), gaussian_field(pts, 0.2, 7))
    assert not np.array_equal(gaussian_field(pts, 0.2, 7), gaussian_field(pts, 0.2, 8))
