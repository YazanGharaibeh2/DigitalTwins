"""Figure 7 (sensitivity): builds from results files written by the real collectors (OAT collect_oat,
Morris analyze) on synthetic run records, with Sobol and 3D records in the documented format; panels
whose inputs are missing are drawn as 'not yet computed'."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from PIL import Image

from dtivl.drivers import figstyle
from dtivl.drivers import stage_sensitivity as S
from dtivl.drivers.figs import fig7_sensitivity as F7
from test_sensitivity import _prec


@pytest.fixture
def fig_dir(tmp_path, monkeypatch):
    out = tmp_path / "figs"
    out.mkdir()
    monkeypatch.setattr(figstyle, "FIG_DIR", out)
    return out


def _write_oat(d):
    sc = [s.name for s in S.oat_scenarios()]
    run_dir = d / S.OAT_RUNS
    run_dir.mkdir()
    kinds = [_prec(onset=0.4, phase="ivl_pulse", fracture=True), _prec(onset=0.45, phase="ivl_pulse"),
             _prec(p_top=0.7), _prec(onset=0.35, phase="ivl_inflate", fracture=True)]
    for lid in ("L1", "L2"):
        for j, name in enumerate(sc[:5]):
            rec = {"lesion": lid, "scenario": name, "complete": True, "transient": {"T_equivalent": 0.05},
                   "ivl": kinds[j % len(kinds)], "balloon_only": _prec()}
            rec["outputs"] = S.outputs(rec)
            (run_dir / f"{S._oat_stem(lid, name)}.json").write_text(json.dumps(rec))
    S.collect_oat(d)


def _write_morris(d):
    design = S.morris_design()
    run_dir = d / S.MORRIS_RUNS
    run_dir.mkdir()
    (run_dir / S.MORRIS_DESIGN).write_text(json.dumps(design))
    X = np.asarray(design["X"])
    lo = np.array([b[0] for b in design["problem"]["bounds"]])
    hi = np.array([b[1] for b in design["problem"]["bounds"]])
    U = (X - lo) / (hi - lo)
    for lid in design["lesions"]:
        for i, u in enumerate(U):
            out = {"ivl_onset_load": 0.3 + 0.2 * u[0] - 0.1 * u[3], "ivl_onset_censored": False,
                   "ivl_onset_after_peak": False, "ivl_onset_step": 0.02, "ivl_onset_load_uncensored": 0.3,
                   "ivl_fracture": bool(u[1] > 0.5), "balloon_fracture": False, "ivl_status": "ok",
                   "balloon_status": "ok"}
            (run_dir / f"{S._morris_stem(lid, i)}.json").write_text(json.dumps(
                {"lesion": lid, "index": i, "complete": True, "outputs": out}))
    S.analyze(d)


def _write_sobol_l3d_metrics(d):
    names = ["log10_ca_E", "eps_f", "log10_k_perivascular", "k_over_mu_soft", "fibrous_E"]
    ind = {n: {"S1": 0.1 * k, "ST": 0.15 * k, "S1_conf": 0.01, "ST_conf": 0.01} for k, n in enumerate(names)}
    (d / F7.SOBOL_JSON).write_text(json.dumps({"lesion_results": {
        lid: {"predictors": {"sigma1_peak_nl": {"indices": ind}}} for lid in ("L1", "L2")}}))
    runs = {"X|2d": {"model": "2d", "lesion": "X", "onset": {"p": 0.6}}}
    for L, p in ((1.0, 0.7), (3.0, 0.65), (5.0, 0.62)):
        runs[f"X|3d|{L}"] = {"model": "3d", "lesion": "X", "length_mm": L, "status": "onset", "onset": {"p": p}}
    (d / F7.L3D_JSON).write_text(json.dumps({"runs": runs}))
    (d / F7.METRICS_JSON).write_text(json.dumps(
        {"outcomes": {"ivl_fracture": {"fujino_comparison": {"best_single_mechanics": "pred_sigma1_peak_nl"}}}}))


def test_fig7_all_panels(tmp_path, fig_dir):
    d = tmp_path / "results"
    d.mkdir()
    _write_oat(d)
    _write_morris(d)
    _write_sobol_l3d_metrics(d)
    png = F7.make("fig7_test", results_dir=d)
    assert png.endswith(".png") and os.path.exists(png) and os.path.exists(png[:-4] + ".pdf")
    assert Image.open(png).size[0] > 1000


def test_fig7_partial_and_missing(tmp_path, fig_dir):
    d = tmp_path / "results"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        F7.make("fig7_test", results_dir=d)
    _write_oat(d)                                     # only the one-at-a-time results exist
    assert os.path.exists(F7.make("fig7_test", results_dir=d))


def test_fig7_labels_carry_no_numbers():
    assert F7._labels_have_no_typed_numbers()
    assert F7._scenario_label("fracture.ell_over_h=1.5") == "phase-field length ℓ/h = 1.5"
