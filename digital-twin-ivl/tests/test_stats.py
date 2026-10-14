"""Tests of dtivl.stats (ROC, thresholds, concordance, logistic models) and of the Phase 5 metrics stage.

The stage is tested end to end on a synthetic per-lesion table with the
columns of ``results/cohort_outcomes.csv`` (``synthetic_cohort``); the
synthetic numbers exist only inside these tests and are never written to
``results/``.
"""
import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from dtivl import params as P
from dtivl.drivers import stage_metrics as S
from dtivl.drivers.common import RESULTS_DIR
from dtivl.stats import models as M
from dtivl.stats import roc as R

#: test-data seed (design seed plus a test-only offset)
SEED = int(P.raw("design.seed")) + 7
N_BOOT = 60
MECH = ["pred_sigma1_p95", "pred_sigma1_p99", "pred_sigma1_mean", "pred_sigma1_peak_nl", "pred_sed_mean",
        "pred_sed_peak_nl", "pred_scf_edge_nl", "pred_frac_eps_gt_epsf", "pred_frac_eps_gt_half_epsf",
        "pred_frac_sig_gt_sigc", "pred_frac_sig_gt_half_sigc", "pred_min_thickness_high_stress"]
PROTOCOLS = ("physiological", "balloon_only", "balloon_10atm", "ivl", "stent_post_ivl", "stent_no_ivl")


# =========================================================================== synthetic cohort
def synthetic_cohort(n_hold: int = 24, seed: int = SEED, leak_T: bool = True) -> pd.DataFrame:
    """Per-lesion table shaped like cohort_outcomes.csv: 48 core lesions (factorial) + ``n_hold`` hold-out lesions.

    Damage under IVL is driven by a latent stress (arc up, thickness down);
    ``ivl_load_T_equivalent`` is made a perfect separator of the IVL outcome
    when ``leak_T`` (a trap: it must never be used as a predictor).
    """
    rng = np.random.default_rng(seed)
    morphs = ["superficial_sheet", "deep_sheet", "nodule", "fragmented"]
    rows = []
    for a in (90, 180, 270, 360):
        for t in (0.2, 0.4, 0.8):
            for m in morphs:
                rows.append({"set": "core", "ca_arc_deg": a, "ca_thickness": t, "morphology": m})
    for i in range(n_hold):
        rows.append({"set": "holdout", "ca_arc_deg": float(rng.uniform(60, 360)),
                     "ca_thickness": float(rng.uniform(0.2, 0.8)), "morphology": morphs[i % 4]})
    df = pd.DataFrame(rows)
    n = len(df)
    df["lesion_id"] = [f"{s[0].upper()}-{k:03d}" for k, s in enumerate(df["set"])]
    deep = df["morphology"] == "deep_sheet"
    df["ca_depth"] = np.where(deep, 0.34, 0.16)
    df["ca_arc_total_deg"] = df["ca_arc_deg"] - 1.0
    df["ca_thickness_max"] = df["ca_thickness"]
    df["ca_area"] = np.deg2rad(df["ca_arc_deg"]) * df["ca_thickness"] * 1.0
    df["lumen_area"] = 2.2 + rng.normal(0, 0.05, n) * (df["set"] == "holdout")
    df["n_deposits"] = np.where(df["morphology"] == "fragmented", np.maximum(2, df["ca_arc_deg"] // 45), 1)
    df["lipid_arc_deg"] = np.where(df["set"] == "holdout", rng.uniform(0, 120, n), 0.0)
    df["plaque_ecc_index"] = np.where(df["morphology"] == "nodule", 0.3, 0.0)
    df["fujino_score"] = 2 * (df["ca_arc_total_deg"] > 180) + 1 * (df["ca_thickness_max"] > 0.5)
    df["ca_E_MPa"] = P.get("materials.calcium.E").value
    df["complete"], df["error"] = True, np.nan
    stress = (df["ca_arc_deg"] / 90.0) / df["ca_thickness"] * np.exp(rng.normal(0, 0.25, n))
    df["pred_sigma1_p95"] = stress
    df["pred_sigma1_p99"] = 1.1 * stress
    df["pred_sigma1_mean"] = 0.5 * stress * np.exp(rng.normal(0, 0.1, n))
    df["pred_sigma1_peak_nl"] = 1.2 * stress * np.exp(rng.normal(0, 0.1, n))
    df["pred_sed_mean"] = 1e-3 * stress ** 2
    df["pred_sed_peak_nl"] = 3e-3 * stress ** 2 * np.exp(rng.normal(0, 0.2, n))
    df["pred_scf_edge_nl"] = 1.0 + rng.uniform(0, 1, n)
    df["pred_frac_eps_gt_epsf"] = 0.0
    df["pred_frac_eps_gt_half_epsf"] = np.clip(0.02 * stress + rng.normal(0, 0.02, n), 0, 1)
    df["pred_frac_sig_gt_sigc"] = 0.0
    df["pred_frac_sig_gt_half_sigc"] = np.clip(0.03 * stress + rng.normal(0, 0.03, n), 0, 1)
    df["pred_min_thickness_high_stress"] = df["ca_thickness"]
    df["pred_p_lumen_MPa"], df["pred_seconds"] = 0.4053, rng.uniform(0.1, 0.3, n)
    latent = np.log(stress) + rng.normal(0, 0.35, n)
    frac = latent > np.quantile(latent, 0.6)
    T = rng.uniform(0.02, 0.12, n)
    if leak_T:
        T = np.where(frac, 0.2 + T, T)
    df["ivl_load_T"], df["ivl_load_T_equivalent"] = 0.9 * T, T
    for p in PROTOCOLS:
        df[f"{p}_status"] = "ok"
    df["ivl_fracture"] = frac
    df["ivl_onset"] = np.where(frac, 0.45 + 0.2 * np.exp(-latent), np.nan)
    df["physiological_fracture"] = False
    df["balloon_only_fracture"] = False                      # single class (as for the calibrated modulus)
    df["balloon_10atm_fracture"] = latent > np.quantile(latent, 0.4)
    for p in ("stent_post_ivl", "stent_no_ivl"):
        df[f"{p}_fracture"] = frac
        df[f"{p}_stent_expansion_pct"] = 85.0 - 4.0 * df["ca_thickness"] + rng.normal(0, 0.5, n)
    df.loc[3, "ivl_status"] = "failed"                       # excluded from the IVL outcomes
    df.loc[5, "stent_no_ivl_status"] = "limited"             # pressure-limited expansion: kept
    df.loc[7, "stent_post_ivl_status"] = np.nan              # not run yet
    return df


@pytest.fixture(scope="module")
def cohort_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("cohort") / "cohort_outcomes.csv"
    synthetic_cohort().to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def stage_run(cohort_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("metrics")
    summary = S.run(cohort_csv, out, n_boot=N_BOOT)
    return summary, out


# =========================================================================== roc.py
def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, 60)
    x = np.round(rng.normal(size=60) + y, 1)                 # rounding creates ties
    assert R.auc_score(y, x) == pytest.approx(roc_auc_score(y, x))
    assert R.auc_score(y, -x) == pytest.approx(1 - roc_auc_score(y, x))
    assert math.isnan(R.auc_score(np.zeros(5, int), x[:5]))
    rows = R.auc_rows(np.vstack([y, y]), np.vstack([x, -x]))
    assert rows == pytest.approx([roc_auc_score(y, x), 1 - roc_auc_score(y, x)])


def test_youden_separable_and_midpoint():
    y = np.array([0, 0, 0, 1, 1])
    x = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    yd = R.youden(y, x)
    assert yd["cut"] == pytest.approx(4.0)                   # midpoint between 3 and 5
    assert (yd["sens"], yd["spec"], yd["j"]) == (1.0, 1.0, 1.0)
    r = R.roc_analysis(y, -x, -1, n_boot=50, rng=R.rng_for(SEED, "t"), level=0.95)
    assert r["threshold"] == pytest.approx(-4.0) and r["auc"] == 1.0
    assert R.classify(-x, r["threshold"], -1).tolist() == [0, 0, 0, 1, 1]
    assert math.isnan(R.youden(y, np.ones(5))["cut"])       # constant predictor: no threshold


def test_youden_tie_breaks_toward_specific_cut():
    # cuts x>=2: sens 1, spec 1/2 (J .5); x>=3: sens 1/2, spec 1 (J .5) -> higher cut chosen
    y = np.array([0, 1, 0, 1])
    x = np.array([1.0, 2.0, 2.5, 3.0])
    yd = R.youden(y, x)
    assert yd["j"] == pytest.approx(0.5)
    assert yd["cut"] == pytest.approx(2.75) and yd["spec"] == 1.0


def test_wilson_matches_scipy():
    for k, n in ((0, 10), (3, 10), (10, 10), (17, 40)):
        ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        assert R.wilson_ci(k, n, 0.95) == pytest.approx((ci.low, ci.high), abs=1e-12)
    assert all(math.isnan(v) for v in R.wilson_ci(0, 0, 0.95))
    perf = R.threshold_performance(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]), 0.95)
    assert (perf["sens"], perf["spec"], perf["acc"]) == (0.5, 1.0, 0.75)
    perf = R.threshold_performance(np.array([0, 0]), np.array([1, 0]), 0.95)
    assert math.isnan(perf["sens"]) and perf["spec"] == 0.5


def test_bootstrap_deterministic_and_stratified():
    y = np.array([1] * 4 + [0] * 9)
    a = R.bootstrap_indices(R.rng_for(SEED, "k"), 30, y.size, y)
    b = R.bootstrap_indices(R.rng_for(SEED, "k"), 30, y.size, y)
    c = R.bootstrap_indices(R.rng_for(SEED, "other"), 30, y.size, y)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert (y[a].sum(axis=1) == 4).all()                     # class counts kept in every resample
    lo, hi = R.percentile_ci(np.arange(101.0), 0.9)
    assert (lo, hi) == pytest.approx((5.0, 95.0))


def test_harrell_c_hand_example():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 0, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])                    # perfectly concordant
    assert R.harrell_c(value, event, risk) == (1.0, 5)       # pairs (1,2),(1,3),(1,4),(2,3),(2,4); 3 is censored
    c, n = R.harrell_c(value, event, np.array([4.0, 1.0, 2.0, 1.0]))
    # (1,2) c, (1,3) c, (1,4) c, (2,3): 1<2 discordant, (2,4): tie 1/2 -> 3.5 / 5
    assert (c, n) == (pytest.approx(0.7), 5)
    assert math.isnan(R.harrell_c(value, np.zeros(4), risk)[0])


def test_loocv_threshold_rule():
    y = np.array([0] * 6 + [1] * 6)
    x = np.r_[np.arange(6.0), 10.0 + np.arange(6.0)]          # separated groups (a gap wider than the spacing)
    lo = R.loocv_threshold(y, x, 1, 0.95)
    assert lo["sens"] == 1.0 and lo["spec"] == 1.0 and lo["n_undefined"] == 0
    lo = R.loocv_threshold(y, -x, "core", 0.95)              # direction re-derived in each fold
    assert lo["acc"] == 1.0


# =========================================================================== models.py
def _toy(seed=SEED, n=40):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = (x1 + 0.5 * rng.normal(size=n) > 0).astype(int)
    X = pd.DataFrame({"x1": x1, "x1_copy": np.exp(x1), "x3": -y + 0.8 * rng.normal(size=n),
                      "noise": rng.normal(size=n), "const": 1.0, "gap": np.r_[np.nan, rng.normal(size=n - 1)]})
    return X, y


def test_selection_rule_skips_redundant_constant_and_missing():
    X, y = _toy()
    sel = M.select_predictors(X, y, list(X.columns), max_predictors=3, max_abs_spearman=0.8)
    assert sel[0] == "x1" and "x1_copy" not in sel           # rank-identical copy is redundant
    assert "const" not in sel and "gap" not in sel and len(sel) <= 3
    assert "x3" in sel
    assert M.select_predictors(X, np.zeros(len(y), int), ["x1"], max_predictors=3, max_abs_spearman=0.8) == []
    disc = max(R.auc_score(y, X["noise"]), 1 - R.auc_score(y, X["noise"]))
    strict = M.select_predictors(X, y, ["x1", "noise"], max_predictors=3, max_abs_spearman=0.8,
                                 min_discrimination=disc + 1e-9)
    assert strict == ["x1"]                                  # below the minimum discrimination: not taken


def test_fit_and_loocv_do_not_use_the_left_out_outcome():
    X, y = _toy()
    spec = M.ModelSpec("t", pool=tuple(X.columns), max_predictors=2, max_abs_spearman=0.8, C=1.0)
    lo = M.loocv_predict(spec, X, y)
    y2 = y.copy()
    y2[0] = 1 - y2[0]
    lo2 = M.loocv_predict(spec, X, y2)
    assert lo2["prob"][0] == pytest.approx(lo["prob"][0])    # row 0's prediction ignores y[0]
    fit = M.fit_spec(spec, X, y)
    co = fit.coefficients()
    z = sum(co["coef_per_sd"][k] * (X[k] - co["mean"][k]) / co["sd"][k] for k in fit.predictors) + co["intercept"]
    assert np.allclose(z, fit.decision(X))                   # the reported formula reproduces the index
    sc = M.prob_scores(y, fit.predict_proba(X))
    assert 0.5 < sc["auc"] <= 1 and sc["log_loss"] > 0 and 0 <= sc["brier"] <= 1
    const = M.fit_spec(M.ModelSpec("c", fixed=("const",)), X, y)
    assert const.lr is None and np.allclose(const.predict_proba(X), y.mean())


def test_paired_bootstraps():
    X, y = _toy()
    idx = R.bootstrap_indices(R.rng_for(SEED, "pb"), 40, len(y), y)
    d = M.paired_bootstrap_fixed(y, X["x1"], X["noise"], idx, 0.95)
    assert d["delta_auc"] == pytest.approx(R.auc_score(y, X["x1"]) - R.auc_score(y, X["noise"]))
    assert d["delta_ci_lo"] <= d["delta_auc"] <= d["delta_ci_hi"] or d["n_boot_valid"] < 40
    same = M.paired_bootstrap_fixed(y, X["x1"], X["x1"], idx, 0.95)
    assert same["delta_ci_lo"] == same["delta_ci_hi"] == 0.0
    a = M.ModelSpec("a", fixed=("noise",), pool=("x1", "x3"), max_predictors=1, max_abs_spearman=0.8)
    b = M.ModelSpec("b", fixed=("noise",))
    r = M.paired_bootstrap_refit(a, b, X, y, idx[:15], 0.95)
    assert r["delta_auc"] > 0 and set(r["selection_frequency_a"]) <= {"x1", "x3"} and r["selection_frequency_b"] == {}


# =========================================================================== configuration and guards
def test_metrics_config_entries_are_design():
    cfg = P.raw("metrics")
    for k, node in cfg.items():
        assert node.get("status") == "DESIGN" and node.get("basis"), k
    assert P.raw("design.composite_index_name")["status"] == "DESIGN"
    c = S.load_config()
    assert c.seed == int(P.raw("design.seed")) + int(P.get("metrics.seed_offset").value)
    assert c.n_boot == int(P.get("metrics.bootstrap_n").value)
    assert all(p.startswith("pred_") for p in c.groups["mechanics"])
    assert not any(p.startswith("ivl_load_") for p, _ in c.candidates)
    assert int(c.composite["max_predictors"]) <= 3
    S.check_predictors(c)


@pytest.mark.parametrize("bad", ["ivl_load_T_equivalent", "ivl_load_T", "ivl_fracture", "stent_no_ivl_msa",
                                 "pred_p_lumen_MPa"])
def test_forbidden_predictors_are_refused(bad):
    c = S.load_config()
    groups = {**c.groups, "geometry": c.groups["geometry"] + [bad]}
    with pytest.raises(ValueError, match="may not be predictors"):
        S.check_predictors(dataclasses.replace(c, groups=groups), synthetic_cohort(n_hold=0))


def test_censoring_value_is_peak_ivl_pressure():
    df = synthetic_cohort(n_hold=0)
    oc = next(o for o in S.load_config().outcomes if o["kind"] == "censored")
    t, excl = S._outcome_table(df, oc, S.load_config())
    cens = t[t["_event"] == 0]
    expect = (P.get("loads.ivl_balloon_pressure_atm").value + cens["ivl_load_T_equivalent"]
              * P.get("loads.ivl_peak_pressure_MPa").value * P.get("loads.ivl_pulse_scale").value)
    assert np.allclose(cens["_value"], expect)
    ev = t[t["_event"] == 1]
    assert np.allclose(ev["_value"], ev["ivl_onset"])
    assert excl["status_not_ok"] == {"failed": 1}


# =========================================================================== stage end to end
def test_stage_outputs(stage_run):
    summary, out = stage_run
    for f in ("metrics_roc.csv", "metrics_roc_curves.csv", "metrics_summary.json"):
        assert (out / f).exists()
    js = json.loads((out / "metrics_summary.json").read_text())
    roc = pd.read_csv(out / "metrics_roc.csv")
    assert list(roc.columns) == S.ROC_COLUMNS
    assert not roc["predictor"].astype(str).str.startswith("ivl_load_").any()
    prim = js["outcomes"]["ivl_fracture"]
    assert prim["core"]["n"] == 47 and prim["excluded"]["status_not_ok"] == {"failed": 1}
    comp = prim["composite"]
    assert comp["name"] == P.raw("design.composite_index_name")["value"]
    assert 1 <= len(comp["selected"]) <= 3 and set(comp["selected"]) <= set(MECH)
    assert 0.5 < comp["core"]["loocv_auc"] <= 1.0
    assert comp["holdout"]["n"] == 24 and comp["holdout"]["auc"] is not None
    # (a) core rows: AUC inside its bootstrap interval; thresholds and Wilson intervals in range
    core = roc[(roc.outcome == "ivl_fracture") & (roc.set == "core") & (roc.group != "model")
               & roc.auc_ci_lo.notna()]
    assert ((core.auc_ci_lo <= core.auc + 1e-12) & (core.auc <= core.auc_ci_hi + 1e-12)).all()
    hold = roc[(roc.outcome == "ivl_fracture") & (roc.set == "holdout") & roc.sens.notna()]
    assert len(hold) > 10
    for c in ("sens", "spec", "acc"):
        assert ((hold[f"{c}_ci_lo"] <= hold[c] + 1e-12) & (hold[c] <= hold[f"{c}_ci_hi"] + 1e-12)).all()
    best = prim["fujino_comparison"]["best_single_mechanics"]
    assert best in MECH
    fj = prim["fujino_comparison"]["fujino_score"]
    assert set(fj["auc_differences"]) == {"core", "holdout"}
    assert any(k.endswith("- fujino_score") for k in fj["auc_differences"]["core"])
    inc = prim["incremental_value"]
    assert {"apparent_refit_bootstrap", "loocv"} <= set(inc["core"]) and "delta_auc" in inc["holdout"]
    # secondary outcomes
    nc = js["not_computed"]
    assert any(e["outcome"] == "balloon_only_fracture" and e["analysis"] == "all analyses" for e in nc)
    assert js["outcomes"]["ivl_onset"]["core"]["n_events"] > 0
    assert js["outcomes"]["stent_no_ivl_expansion"]["excluded"]["status_not_ok"] == {}
    assert js["outcomes"]["stent_post_ivl_expansion"]["excluded"]["status_not_ok"] == {"not run": 1}
    on = roc[(roc.outcome == "ivl_onset") & (roc.set == "core") & roc.c_index.notna()]
    assert ((on.c_index >= 0) & (on.c_index <= 1)).all() and comp["name"] in set(on.predictor)
    # wording rule
    for f in ("metrics_roc.csv", "metrics_summary.json"):
        assert "validat" not in (out / f).read_text().lower()


def test_stage_is_deterministic(cohort_csv, stage_run, tmp_path):
    _, out = stage_run
    S.run(cohort_csv, tmp_path, n_boot=N_BOOT)
    a = pd.read_csv(out / "metrics_roc.csv")
    b = pd.read_csv(tmp_path / "metrics_roc.csv")
    pd.testing.assert_frame_equal(a, b)


def test_holdout_outcomes_do_not_change_core_results(stage_run, tmp_path):
    _, out = stage_run
    df = synthetic_cohort()
    h = df["set"] == "holdout"
    df.loc[h, "ivl_fracture"] = ~df.loc[h, "ivl_fracture"].astype(bool)   # flip every hold-out outcome
    p = tmp_path / "flipped.csv"
    df.to_csv(p, index=False)
    S.run(p, tmp_path, n_boot=N_BOOT)
    a, b = pd.read_csv(out / "metrics_roc.csv"), pd.read_csv(tmp_path / "metrics_roc.csv")
    key = (a.outcome == "ivl_fracture") & (a.set == "core")
    pd.testing.assert_frame_equal(a[key].reset_index(drop=True), b[key].reset_index(drop=True))
    ha, hb = a[(a.outcome == "ivl_fracture") & (a.set == "holdout")], b[(b.outcome == "ivl_fracture") & (b.set == "holdout")]
    assert not ha["sens"].equals(hb["sens"])
    ja = json.loads((out / "metrics_summary.json").read_text())["outcomes"]["ivl_fracture"]["composite"]
    jb = json.loads((tmp_path / "metrics_summary.json").read_text())["outcomes"]["ivl_fracture"]["composite"]
    assert ja["selected"] == jb["selected"] and ja["model"] == jb["model"]


def test_stage_without_holdout_rows(tmp_path):
    p = tmp_path / "core_only.csv"
    synthetic_cohort(n_hold=0).to_csv(p, index=False)
    js = S.run(p, tmp_path, n_boot=20)
    nc = js["not_computed"]
    assert any(e.get("set") == "holdout" and e["analysis"] == "hold-out evaluation" for e in nc)
    assert any(e["analysis"] == "incremental value" and e.get("set") == "holdout" for e in nc)
    prim = js["outcomes"]["ivl_fracture"]
    assert prim["composite"]["holdout"]["n"] == 0 and prim["composite"]["core"]["loocv_auc"] is not None
    roc = pd.read_csv(tmp_path / "metrics_roc.csv")
    assert roc[(roc.set == "holdout")]["auc"].isna().all()


def test_stage_single_class_primary(tmp_path):
    df = synthetic_cohort(n_hold=6)
    df["ivl_fracture"] = False
    df["ivl_onset"] = np.nan
    p = tmp_path / "no_events.csv"
    df.to_csv(p, index=False)
    js = S.run(p, tmp_path, n_boot=20)
    prim = js["outcomes"]["ivl_fracture"]
    assert "single outcome class" in prim["note"] and "composite" not in prim
    assert any(e["outcome"] == "ivl_onset" and e["analysis"] == "all analyses" for e in js["not_computed"])
    assert (tmp_path / "metrics_roc.csv").exists() and (tmp_path / "metrics_summary.json").exists()
    # secondary outcomes without the composite index still run
    assert js["outcomes"]["stent_no_ivl_expansion"]["predictors"]


def test_stage_on_existing_cohort_outcomes(tmp_path):
    """Smoke run on results/cohort_outcomes.csv when it exists (cohort runs in progress); outputs to tmp."""
    path = RESULTS_DIR / "cohort_outcomes.csv"
    if not path.exists():
        pytest.skip("results/cohort_outcomes.csv not present")
    js = S.run(path, tmp_path, n_boot=20)
    assert js["input"]["n_rows"] == len(pd.read_csv(path))
