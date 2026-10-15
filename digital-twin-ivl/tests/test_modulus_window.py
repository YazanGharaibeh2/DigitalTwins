"""D18 modulus-window stage: the two moduli are the log-midpoints between the calibrated modulus and its
sweep neighbours; analyze() counts outcomes and computes AUCs from the run files."""
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from dtivl.drivers import stage_modulus_window as W
from dtivl.drivers.stage_pilot import e_sweep
from dtivl.materials.constitutive import MaterialSet


def test_window_moduli_are_log_midpoints():
    base = MaterialSet.baseline().ca_E
    sweep = sorted(e_sweep())
    i = sweep.index(min(sweep, key=lambda e: abs(e - base)))
    lo, hi = W.window_moduli()
    assert lo == round(math.sqrt(sweep[i - 1] * base), 6) and hi == round(math.sqrt(sweep[i + 1] * base), 6)
    assert sweep[i - 1] < lo < base < hi < sweep[i + 1]


def test_analyze_on_synthetic_runs(tmp_path, monkeypatch):
    lids = [f"C-L{k}" for k in range(8)]
    monkeypatch.setattr(W, "core_ids", lambda: lids)
    d = tmp_path / W.RUN_DIR
    d.mkdir()
    (tmp_path / "metrics_summary.json").write_text(json.dumps({"outcomes": {"ivl_fracture": {
        "fujino_comparison": {"best_single_mechanics": "pred_sigma1_peak_nl"},
        "univariable": {"pred_sigma1_peak_nl": {"direction": 1}, "fujino_score": {"direction": 1}}}},
        "config": {"seed": 1, "n_boot": 50, "ci_level": 0.95}}))
    import pandas as pd
    from dtivl.drivers import common
    coh = tmp_path / "cohort"
    coh.mkdir()
    pd.DataFrame({"lesion_id": lids, "fujino_score": [0, 0, 1, 1, 2, 2, 3, 3]}).to_csv(coh / "cohort.csv", index=False)
    monkeypatch.setattr(common, "COHORT_DIR", coh)
    for E in W.window_moduli():
        for k, lid in enumerate(lids):
            frac = k >= 4
            rec = {"lesion": lid, "ca_E_MPa": E, "complete": True, "predictors": {"sigma1_peak_nl": float(k)},
                   "outputs": {"ivl_fracture": frac, "balloon_fracture": False, "ivl_onset_load": 0.5 if frac else None}}
            (d / f"{W._stem(lid, E)}.json").write_text(json.dumps(rec))
    res = W.analyze(tmp_path)
    for e in res["by_modulus"].values():
        assert e["n_runs_complete"] == 8 and e["n_ivl_fracture"] == 4 and e["n_balloon_fracture"] == 0
        assert e["pred_sigma1_peak_nl"]["auc"] == 1.0
        assert 0.5 <= e["fujino_score"]["auc"] <= 1.0
    assert json.loads((tmp_path / W.OUT_JSON).read_text())["note"].startswith("POST HOC")
