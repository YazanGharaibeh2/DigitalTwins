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


def test_rankine_test_helpers():
    """D10(b) driver: T_equivalent at the baseline modulus is read from the pilot transient results,
    and the compact table row carries the reported fields."""
    from dtivl.drivers.common import RESULTS_DIR, read_json
    from dtivl.drivers.stage_diagnostics import RANKINE_TABLE_KEYS, _lesion_T_equivalent, _rankine_row
    from dtivl.drivers.stage_pilot import pilot_ids
    from dtivl.materials.constitutive import MaterialSet

    if not (RESULTS_DIR / "pilot_linear_transient.json").exists():
        pytest.skip("results/pilot_linear_transient.json not present")
    ms = MaterialSet.baseline()
    lid = pilot_ids()[0]
    T, src = _lesion_T_equivalent(lid, ms, None, None)
    row = next(r for r in read_json("pilot_linear_transient")["lesions"][lid]["by_E"]
               if abs(r["ca_E_MPa"] - ms.ca_E) < 1e-6)
    assert T == row["transient"]["T_equivalent"] and src.endswith("pilot_linear_transient.json")
    rec = {"status": "ok", "seconds": 1.0, "summary": {"max_d": 1.0}, "first_damage_p": 0.5,
           "cracks": {"n_fractures": 2, "n_partial": 1, "diffuse": False, "damaged_fraction": 0.1,
                      "pattern": ["radial", "radial", "circumferential"], "onset_load": 0.6}}
    out = _rankine_row(rec)
    assert set(RANKINE_TABLE_KEYS) <= set(out)
    assert out["status"] == "ok" and out["n_fractures"] == 2 and out["pattern"][-1] == "circumferential"
    assert out["onset_load"] == 0.6 and out["first_damage_p"] == 0.5 and out["seconds"] == 1.0
