"""Checkpoint decisions recorded in params/study_config.yaml."""
import pytest

from dtivl import params as P
from dtivl.materials.constitutive import MaterialSet


def test_d8_baseline_is_a_sweep_point():
    """D8: the calibrated calcium modulus is one of the pilot sweep points (no drift between the two)."""
    from dtivl.drivers.stage_pilot import e_sweep

    q = P.get("materials.calcium.E")
    assert q.status == "CALIBRATED"
    assert any(abs(q.value - e) < 1e-6 for e in e_sweep())
    assert MaterialSet.baseline().ca_E == pytest.approx(q.value)


def test_tissue_level_modulus_kept_in_sweep():
    from dtivl.drivers.stage_pilot import e_sweep

    t = P.get("materials.calcium.E_tissue_level")
    assert t.status == "DERIVED"
    assert t.value == pytest.approx(P.lit("calcification_fracture_stress").value
                                    / (P.lit("calcification_fracture_stretch").value - 1.0))
    assert any(abs(t.value - e) < 1e-6 for e in e_sweep())
    assert P.raw("decisions")["D8"]
