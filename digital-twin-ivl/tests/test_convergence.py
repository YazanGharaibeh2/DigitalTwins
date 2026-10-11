import math

import pytest

from dtivl.metrics.convergence import gci_three_grid


def test_recovers_known_order_and_limit():
    # phi(h) = 2 + 0.5 h^2: exact limit 2, order 2
    h = (0.01, 0.02, 0.04)
    phi = tuple(2.0 + 0.5 * x ** 2 for x in h)
    r = gci_three_grid(h, phi)
    assert r["convergence"] == "monotonic"
    assert r["p"] == pytest.approx(2.0, rel=1e-6)
    assert r["phi_ext"] == pytest.approx(2.0, rel=1e-9)
    assert r["gci_fine21"] > 0


def test_non_uniform_refinement_ratio():
    h = (0.01, 0.015, 0.03)
    phi = tuple(1.0 - 0.2 * x ** 1.5 for x in h)
    r = gci_three_grid(h, phi)
    assert r["p"] == pytest.approx(1.5, rel=1e-6)
    assert r["phi_ext"] == pytest.approx(1.0, rel=1e-9)


def test_oscillatory_flagged():
    r = gci_three_grid((1.0, 2.0, 4.0), (1.0, 1.05, 0.95))  # R = e21/e32 = -0.5
    assert r["convergence"] == "oscillatory"


def test_bad_ordering_raises():
    with pytest.raises(ValueError):
        gci_three_grid((2.0, 1.0, 4.0), (1.0, 1.0, 1.0))


def test_identical_solutions():
    r = gci_three_grid((1.0, 2.0, 4.0), (3.0, 3.0, 3.0))
    assert r["convergence"] == "converged"
    assert not math.isnan(r["phi_ext"])
