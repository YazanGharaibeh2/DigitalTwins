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


def test_erosion_subset_is_in_core():
    from dtivl.drivers.stage_cohort_runs import cohort_ids, erosion_subset

    sub = erosion_subset()
    assert len(sub) == 8 and set(sub) <= set(cohort_ids(("core",)))


def test_alt_values_parsed_only_where_marked_as_alternative():
    """D15: only a range column "number (remark marking an alternative value)" gives ``alt``; a number
    with a remark naming a subgroup or another quantity (e.g. Fujino "96 (score 0-3)") does not."""
    lit = P.load_literature()
    with_alt = {name: row.alt for name, row in lit.items() if row.alt is not None}
    assert with_alt == {"density_calcified_plaque_model_input": pytest.approx(1450)}
    assert lit["fujino_score4_stent_expansion"].alt is None
    assert P.get("materials.density.calcium").sens == (lit["density_calcified_plaque_model_input"].alt,)
