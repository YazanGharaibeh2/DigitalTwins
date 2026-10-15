"""Tests of the consistency stage (dtivl.drivers.stage_consistency, checks C1-C12).

Every check runs on synthetic per-lesion tables shaped like
``results/cohort_outcomes.csv`` (``synthetic_outcomes``): the reader is
monkeypatched or the table is passed to :func:`evaluate` directly, outputs go
to ``tmp_path``, and the synthetic numbers never reach ``results/``. Published
values in the assertions are read through :mod:`dtivl.params`, never typed.
"""
import ast
import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from dtivl import params as P
from dtivl.drivers import stage_consistency as SC
from dtivl.drivers.common import RESULTS_DIR
from dtivl.stats import roc as R

#: test-data seed (design seed plus a test-only offset)
SEED = int(P.raw("design.seed")) + 11
N_BOOT = 200
MORPHS = ("superficial_sheet", "deep_sheet", "nodule", "fragmented")
PROTOCOLS = ("physiological", "balloon_only", "balloon_10atm", "ivl", "stent_post_ivl", "stent_no_ivl")


# =========================================================================== synthetic data
def synthetic_outcomes(n_hold: int = 12, seed: int = SEED) -> pd.DataFrame:
    """Per-lesion table shaped like cohort_outcomes.csv: 48 core lesions (factorial) + ``n_hold`` hold-out lesions.

    Fracture is driven by a latent load (arc up, thickness down, nodules down);
    ``ivl_load_T_equivalent`` separates the IVL outcome perfectly (a trap: the
    stage must never read it).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in (90, 180, 270, 360):
        for t in (0.2, 0.4, 0.8):
            for m in MORPHS:
                rows.append({"set": "core", "ca_arc_deg": float(a), "ca_thickness": t, "morphology": m})
    for i in range(n_hold):
        rows.append({"set": "holdout", "ca_arc_deg": float(rng.uniform(60, 360)),
                     "ca_thickness": float(rng.uniform(0.2, 0.8)), "morphology": MORPHS[i % 4]})
    df = pd.DataFrame(rows)
    n = len(df)
    df["lesion_id"] = [f"{s[0].upper()}-{k:03d}" for k, s in enumerate(df["set"])]
    df["ca_arc_total_deg"] = df["ca_arc_deg"] - 1.0
    df["ca_thickness_max"] = df["ca_thickness"]
    df["fujino_score"] = 2 * (df["ca_arc_total_deg"] > 180) + 1 * (df["ca_thickness_max"] > 0.5)
    df["ca_E_MPa"] = P.get("materials.calcium.E").value
    df["complete"], df["error"] = True, np.nan
    stress = (df["ca_arc_deg"] / 90.0) / df["ca_thickness"] * np.exp(rng.normal(0, 0.2, n))
    df["pred_sigma1_p99"], df["pred_sigma1_peak_nl"] = stress, 1.2 * stress
    df["pred_p_lumen_MPa"] = P.get("loads.ivl_balloon_pressure_atm").value
    latent = (df["ca_arc_deg"] / 90.0 - 2.0 * df["ca_thickness"] - 1.0 * (df["morphology"] == "nodule")
              + rng.normal(0, 0.3, n))
    frac = (latent > np.quantile(latent, 0.4)).to_numpy()
    for p in PROTOCOLS:
        df[f"{p}_status"] = "ok"
    df["ivl_fracture"] = frac
    df["balloon_10atm_fracture"] = (latent > np.quantile(latent, 0.7)).to_numpy()
    df["balloon_only_fracture"] = False
    df["physiological_fracture"] = False
    df["ivl_n_fractures"] = np.where(frac, 1 + (df["ca_arc_deg"] >= 270), 0)
    df["ivl_n_partial"] = np.where(frac, 0, (df["ca_arc_deg"] >= 180).astype(int))
    df["ivl_diffuse"] = frac & (df["ca_thickness"] == 0.2).to_numpy()
    df["ivl_load_T_equivalent"] = np.where(frac, 0.3, 0.05)          # trap (never read)
    recoil = P.lit("acute_recoil_EES_CoCr_clinical").value
    for p in ("stent_post_ivl", "stent_no_ivl"):
        df[f"{p}_stent_expansion_pct"] = 85.0 - 3.0 * df["fujino_score"] + rng.normal(0, 0.5, n)
        df[f"{p}_acute_recoil_pct"] = recoil + rng.normal(0, 0.3, n)
    return df


def synthetic_summary(*, within: bool = True, n_above: int | None = 0, mismatch: bool = False) -> dict:
    """cohort_summary.json-like dict with the CTA values copied from the literature table."""
    lo = P.lit("plaque_burden_calcified_plaque_CTA").value
    hi = P.lit("plaque_burden_calcified_nodule_CTA").value
    pb = 0.5 * (lo + hi) if within else lo - 1.0
    out = {"n_per_set": {"core": 48, "holdout": 24, "pilot": 3, "example3d": 1}, "n_total": 76,
           "plaque_burden_check": {"implied_plaque_burden_pct": pb, "cta_calcified_plaque_pct": lo + mismatch,
                                   "cta_calcified_nodule_pct": hi, "within_range": within},
           "plaque_growth": {"n_grown": {"core": 4, "holdout": 2}, "max_grown_mm": 0.3,
                             "plaque_burden_pct_range_grown": [lo, hi + 1.0]}}
    if n_above is not None:
        out["plaque_growth"]["n_grown_above_nodule_cta"] = n_above
    return out


def synthetic_cohort(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"lesion_id": df["lesion_id"], "fujino_max": 3})


CALIBRATION = {"lesion_template": "P-TEST", "recoil_target_pct": 4.3, "K_s_MPa_per_mm": 6.5, "reached_target": True}


@pytest.fixture(scope="module")
def cfg():
    return SC.load_config(n_boot=N_BOOT)


def evaluate(df, cfg, **extra) -> dict:
    """Rows of every check, keyed by (id, set)."""
    ex = {"cohort_summary": synthetic_summary(), "calibration": CALIBRATION, "cohort": synthetic_cohort(df)}
    ex.update(extra)
    return {(r["id"], r["set"]): r for r in SC.evaluate(df, cfg, ex)}


def quantity(row: dict, prefix: str) -> dict:
    return next(q for q in row["quantities"] if q["name"].startswith(prefix))


def run_stage(monkeypatch, tmp_path, df, **extra):
    """Run the stage with the reader and the side inputs monkeypatched; returns (summary, csv table)."""
    ex = {"cohort_summary": synthetic_summary(), "calibration": CALIBRATION, "cohort": synthetic_cohort(df)}
    ex.update(extra)
    monkeypatch.setattr(SC, "load_outcomes", lambda path=None: df.copy())
    monkeypatch.setattr(SC, "load_cohort_summary", lambda path=None: ex["cohort_summary"])
    monkeypatch.setattr(SC, "load_calibration", lambda path=None: ex["calibration"])
    monkeypatch.setattr(SC, "load_cohort", lambda path=None: ex["cohort"])
    summary = SC.run(tmp_path / "synthetic_outcomes.csv", tmp_path, n_boot=N_BOOT)
    return summary, pd.read_csv(tmp_path / "consistency.csv")


# =========================================================================== configuration and literature
def test_config_entries_are_design(cfg):
    node = P.raw("consistency")
    for k, v in node.items():
        entries = v.items() if k == "checks" else [(k, v)]
        for name, e in entries:
            assert e.get("status") == "DESIGN" and e.get("basis"), name
    assert cfg.seed == int(P.raw("design.seed")) + int(P.get("consistency.seed_offset").value)
    assert cfg.seed != int(P.raw("design.seed")) + int(P.get("metrics.seed_offset").value)
    assert set(cfg.checks) == set(SC.CHECKS) and "UNVERIFIED" not in cfg.allow_status
    lit = P.load_literature()
    for cid, spec in cfg.checks.items():
        for row in spec["published"] + [c["row"] for c in spec.get("context_rows", [])]:
            assert row in lit and lit[row].status != "UNVERIFIED", (cid, row)
    assumptions = pd.read_csv(P.ASSUMPTIONS_CSV)
    assert {"consistency.seed_offset", "consistency.checks.C1", "consistency.checks.C12"} <= set(assumptions.assumption)


def test_published_rows_read_through_params(cfg):
    for spec in cfg.checks.values():
        for row in spec["published"]:
            pub = SC.published(row, cfg)
            lit = P.load_literature()[row]
            if pub["ok"]:
                assert pub["value"] == P.lit(row, allow_unverified=lit.status in cfg.allow_status).value
                assert pub["sd"] == lit.sd and pub["value_cell"] == lit.value_str
            else:
                assert pub["reason"]
    stress = SC.published("insilico_postdilation_calcium_stress", cfg)       # two values in one cell
    assert not stress["ok"] and "not a plain number" in stress["reason"]
    sec = SC.published("fracture_thickness_cutoff_balloon", cfg)
    assert sec["ok"] and sec["status"] == "SECONDARY" and "SECONDARY" in SC._pub_text(sec)
    strict = dataclasses.replace(cfg, allow_status=())
    assert not SC.published("fracture_thickness_cutoff_balloon", strict)["ok"]
    unverified = [k for k, v in P.load_literature().items() if v.status == "UNVERIFIED"]
    for row in unverified[:3]:
        pub = SC.published(row, dataclasses.replace(cfg, allow_status=("UNVERIFIED",)))
        assert not pub["ok"] and "UNVERIFIED" in pub["reason"] and pub["value"] is None
    assert not SC.published("no_such_row", cfg)["ok"]


def test_ivl_load_columns_are_refused(cfg):
    for key, bad in (("column", "ivl_load_T_equivalent"), ("column", "ivl_load_T")):
        checks = {**cfg.checks, "C1": {**cfg.checks["C1"], key: bad}}
        with pytest.raises(ValueError, match="IVL load factor"):
            SC.check_columns(dataclasses.replace(cfg, checks=checks))
    checks = {**cfg.checks, "C9": {**cfg.checks["C9"], "columns": ["pred_sigma1_p99", "ivl_load_T_equivalent"]}}
    with pytest.raises(ValueError, match="IVL load factor"):
        SC.check_columns(dataclasses.replace(cfg, checks=checks))


# =========================================================================== statistics helpers
def test_direction_label_and_combine():
    d = SC.direction_label
    assert d(0.5, 0.2, 0.8, 0.0, 1, True) == "agrees"
    assert d(0.5, -0.1, 0.8, 0.0, 1, True) == "partial"
    assert d(0.5, -0.1, 0.8, 0.0, 1, False) == "agrees"
    assert d(-0.5, -0.8, -0.2, 0.0, 1, True) == "disagrees"
    assert d(0.0, -0.1, 0.1, 0.0, 1, True) == "disagrees"            # on the null value
    assert d(0.4, 0.2, 0.7, 1.0, -1, True) == "agrees"               # odds ratio below 1
    assert d(math.nan, 0, 1, 0.0, 1, True) == SC.NOT_COMPUTED
    assert SC.combine(["agrees", "agrees", SC.NOT_COMPUTED]) == "agrees"
    assert SC.combine(["disagrees", "disagrees"]) == "disagrees"
    assert SC.combine(["agrees", "disagrees"]) == "partial"
    assert SC.combine(["partial", "agrees"]) == "partial"
    assert SC.combine([SC.NOT_COMPUTED]) == SC.NOT_COMPUTED and SC.combine([]) == SC.NOT_COMPUTED


def test_odds_ratio_haldane_woolf_and_wilson():
    corr, level = 0.5, 0.95
    r = SC.odds_ratio(0, 6, 5, 3, corr, level)
    a, b, c, d = 0 + corr, 6 + corr, 5 + corr, 3 + corr
    lor = math.log(a * d / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = st.norm.ppf(0.975)
    assert r["value"] == pytest.approx(math.exp(lor))
    assert (r["ci_lo"], r["ci_hi"]) == pytest.approx((math.exp(lor - z * se), math.exp(lor + z * se)))
    assert math.isnan(SC.odds_ratio(0, 6, 5, 3, 0.0, level)["value"])   # empty cell without correction
    pr = SC.proportion(7, 20, level)
    assert pr["value"] == pytest.approx(35.0)
    assert (pr["ci_lo"], pr["ci_hi"]) == pytest.approx(tuple(100 * v for v in R.wilson_ci(7, 20, level)))


def test_spearman_interval_is_deterministic(cfg):
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=30)
    y = (x + rng.normal(0, 0.7, 30) > 0).astype(float)
    a = SC.spearman_ci(x, y, cfg, "key-a", binary_y=True)
    b = SC.spearman_ci(x, y, cfg, "key-a", binary_y=True)
    c = SC.spearman_ci(x, y, cfg, "key-b", binary_y=True)
    assert a == b and (a["ci_lo"], a["ci_hi"]) != (c["ci_lo"], c["ci_hi"])
    assert a["value"] == pytest.approx(st.spearmanr(x, y).statistic)
    assert a["ci_lo"] <= a["value"] <= a["ci_hi"] and a["n_boot_valid"] == cfg.n_boot


# =========================================================================== checks C1-C12
def test_c1_c2_direction_and_youden(cfg):
    df = synthetic_outcomes()
    for p in ("ivl", "balloon_10atm"):
        df[f"{p}_fracture"] = (df["ca_arc_deg"] >= 270) & (df["ca_thickness"] <= 0.4)
    rows = evaluate(df, cfg)
    c1, c2 = rows[("C1", "core")], rows[("C2", "core")]
    assert c1["agreement"] == "agrees" and c2["agreement"] == "agrees"
    assert quantity(c1, "rho ivl")["value"] > 0 and quantity(c2, "rho balloon_10atm")["value"] < 0
    yd = quantity(c1, "Youden cut")
    assert yd["descriptive"] and 180 < yd["value"] < 270                 # midpoint between the arc levels
    assert "Youden cut descriptive" in c1["note"] and "fracture_angle_cutoff_balloon" in c1["note"]
    assert not any(q["name"].startswith("Youden") for q in rows[("C1", "holdout")]["quantities"])
    assert "core set only" in rows[("C1", "holdout")]["note"]
    assert c1["n_lesions"] == 48
    # reversed effect: larger arc and thicker calcium fracture less
    for p in ("ivl", "balloon_10atm"):
        df[f"{p}_fracture"] = (df["ca_arc_deg"] <= 180) & (df["ca_thickness"] >= 0.4)
    rows = evaluate(df, cfg)
    assert rows[("C1", "core")]["agreement"] == "disagrees" and rows[("C2", "core")]["agreement"] == "disagrees"
    # opposite directions under the two protocols: partial
    df["balloon_10atm_fracture"] = df["ca_arc_deg"] >= 270
    df["ivl_fracture"] = df["ca_arc_deg"] <= 180
    assert evaluate(df, cfg)[("C1", "core")]["agreement"] == "partial"


def test_c3_nodules_fracture_less(cfg):
    df = synthetic_outcomes()
    nod = df["morphology"] == "nodule"
    sheet = df["morphology"].isin(["superficial_sheet", "deep_sheet"])
    df["ivl_fracture"] = (sheet & (df["ca_thickness"] <= 0.4)
                          | nod & (df["ca_arc_deg"] == 360) & (df["ca_thickness"] == 0.2))
    row = evaluate(df, cfg)[("C3", "core")]
    orr = quantity(row, "odds ratio")
    assert orr["value"] < 1 and orr["ci_hi"] < 1 and row["agreement"] == "agrees"
    assert row["n_lesions"] == int((nod | sheet)[df["set"] == "core"].sum())     # fragmented in neither group
    assert "fragmented" in row["note"] and orr["table"]["exposed_event"] == 1
    df["ivl_fracture"] = nod & (df["ca_thickness"] <= 0.4)                       # nodules fracture more
    assert evaluate(df, cfg)[("C3", "core")]["agreement"] == "disagrees"
    df["ivl_fracture"] = df["ca_arc_deg"] >= 270                                # same rate in both groups: OR = 1
    assert quantity(evaluate(df, cfg)[("C3", "core")], "odds ratio")["value"] == pytest.approx(1.0)
    assert evaluate(df, cfg)[("C3", "core")]["agreement"] == "disagrees"
    df["ivl_fracture"] = True                                                    # one outcome class only
    row = evaluate(df, cfg)[("C3", "core")]
    assert row["agreement"] == SC.NOT_COMPUTED and "outcome class" in row["note"]


def test_c4_fracture_rate_band_and_overlap(cfg):
    band = sorted(P.lit(r).value for r in cfg.checks["C4"]["published"])
    df = synthetic_outcomes()
    core = df["set"] == "core"
    n = int(core.sum())

    def with_k(k):
        d = df.copy()
        d["ivl_fracture"] = False
        d.loc[d.index[core][:k], "ivl_fracture"] = True
        return evaluate(d, cfg)[("C4", "core")]

    k_in = round(n * (band[0] + band[1]) / 200.0)
    row = with_k(k_in)
    assert row["agreement"] == "agrees" and "within the published range" in row["note"]
    assert quantity(row, "ivl fracture")["value"] == pytest.approx(100.0 * k_in / n)
    k_ov = next(k for k in range(k_in, -1, -1) if 100.0 * k / n < band[0]
                and 100.0 * R.wilson_ci(k, n, cfg.level)[1] >= band[0])
    row = with_k(k_ov)
    assert row["agreement"] == "agrees" and "only the Wilson interval overlaps" in row["note"]
    row = with_k(1)
    assert row["agreement"] == "disagrees" and row["ci_hi"] < band[0]
    assert cfg.checks["C4"]["caveat"] in row["note"]
    assert "laser_IVL_OCT_modification_rate" in row["note"]


def test_c5_c6_zero_rule(cfg):
    df = synthetic_outcomes()
    df["balloon_10atm_fracture"] = False
    rows = evaluate(df, cfg)
    assert rows[("C5", "core")]["agreement"] == "agrees"
    c6 = rows[("C6", "core")]
    assert c6["agreement"] == "agrees" and c6["kind"] == "model_internal"
    assert c6["source_rows"] is None and "model-internal" in c6["published"]
    assert "not a published observation" in c6["note"]
    assert quantity(c6, "balloon_only fracture")["ci_hi"] > 0                # Wilson interval of 0 of n
    df.loc[0, "balloon_10atm_fracture"] = True
    df.loc[1, "balloon_only_fracture"] = True
    rows = evaluate(df, cfg)
    c5 = rows[("C5", "core")]
    assert c5["agreement"] == "disagrees" and "decision D8" in c5["note"] and "1 of 48" in c5["note"]
    assert rows[("C6", "core")]["agreement"] == "disagrees"


def test_c7_c8_mean_within_sd(cfg):
    bands = {r: (P.lit(r, "value-sd").value, P.lit(r, "value+sd").value) for r in cfg.checks["C7"]["published"]}
    (lo1, hi1), (lo2, hi2) = bands.values()
    df = synthetic_outcomes()
    df.loc[2, "stent_post_ivl_status"] = "limited"                            # pressure-limited: kept
    df.loc[3, "stent_post_ivl_status"] = "failed"                             # excluded

    def c7(value):
        d = df.copy()
        d["stent_post_ivl_stent_expansion_pct"] = value + np.linspace(-0.5, 0.5, len(d))
        return evaluate(d, cfg)[("C7", "core")]

    both = 0.5 * (max(lo1, lo2) + min(hi1, hi2))
    one = 0.5 * (min(lo1, lo2) + max(lo1, lo2))                               # inside the wider-reaching band only
    row = c7(both)
    assert row["agreement"] == "agrees" and row["n_lesions"] == 47
    assert "pressure-limited" in row["note"] and "status failed 1" in row["note"]
    for text in ("maximum-calcium-site", "MSA-site", "caps the model expansion"):
        assert text in row["note"]
    assert c7(one)["agreement"] == "partial"
    assert c7(min(lo1, lo2) - 10.0)["agreement"] == "disagrees"
    c8 = evaluate(df, cfg)[("C8", "core")]
    assert c8["kind"] == "calibration_target" and c8["agreement"] == "agrees"
    assert "calibration target" in c8["note"] and SC._g(CALIBRATION["K_s_MPa_per_mm"]) in c8["note"]
    assert {q["name"] for q in c8["quantities"]} == {"mean stent_no_ivl", "mean stent_post_ivl"}
    nocal = evaluate(df, cfg, calibration=None)[("C8", "core")]
    assert "calibration_stent.json not found" in nocal["note"]


def test_c9_stress_scaling(cfg, monkeypatch):
    df = synthetic_outcomes()
    core = df[df["set"] == "core"]
    p_t = P.get(cfg.checks["C9"]["target_load"]).value
    expect = float(np.median(core["pred_sigma1_p99"] * p_t / core["pred_p_lumen_MPa"]))
    row = evaluate(df, cfg)[("C9", "core")]
    q = quantity(row, "median pred_sigma1_p99")
    assert q["value"] == pytest.approx(expect) and q["stat"] == "median"
    assert row["agreement"] == SC.NOT_COMPUTED and "not a plain number" in row["note"]   # published cell not parsed
    assert row["kind"] == "model_vs_model" and "model vs model" in row["note"]
    # criterion logic, with a plain-number published value injected by the test
    factor = float(cfg.checks["C9"]["factor"])
    orig = SC.published

    def fake(value):
        def pub(row_name, c):
            out = orig(row_name, c)
            if row_name == "insilico_postdilation_calcium_stress":
                out = {**out, "ok": True, "value": value, "reason": None}
            return out
        return pub

    peak = float(np.median(core["pred_sigma1_peak_nl"] * p_t / core["pred_p_lumen_MPa"]))
    monkeypatch.setattr(SC, "published", fake(expect))
    assert evaluate(df, cfg)[("C9", "core")]["agreement"] == "agrees"
    monkeypatch.setattr(SC, "published", fake(expect * factor * 2.0))
    assert evaluate(df, cfg)[("C9", "core")]["agreement"] == "disagrees"
    monkeypatch.setattr(SC, "published", fake(0.5 * (expect * factor + peak * factor)))   # between the two
    assert evaluate(df, cfg)[("C9", "core")]["agreement"] == "partial"


def test_c10_fracture_count_and_full_thickness_fraction(cfg):
    row_c = cfg.checks["C10"]["count"]["published"]
    c_lo, c_hi = P.lit(row_c, "value-sd").value, P.lit(row_c, "value+sd").value
    pf = P.lit(cfg.checks["C10"]["fraction"]["published"]).value
    df = synthetic_outcomes()
    row = evaluate(df, cfg)[("C10", "core")]
    core = df[(df["set"] == "core")]
    fr = core[core["ivl_fracture"]]
    cnt = quantity(row, "fractures per fractured lesion")
    assert cnt["value"] == pytest.approx(fr["ivl_n_fractures"].mean()) and cnt["n"] == len(fr)
    tot = core["ivl_n_fractures"] + core["ivl_n_partial"]
    ft = quantity(row, "full-thickness fraction")
    assert ft["value"] == pytest.approx(100.0 * core["ivl_n_fractures"].sum() / tot.sum())
    assert ft["n"] == int((tot > 0).sum()) and ft["ci_lo"] <= ft["value"] <= ft["ci_hi"]
    lab_c = "agrees" if c_lo <= cnt["value"] <= c_hi else "disagrees"
    lab_f = "agrees" if ft["ci_lo"] <= pf <= ft["ci_hi"] else "disagrees"
    assert row["agreement"] == SC.combine([lab_c, lab_f])
    assert f"{int(fr['ivl_diffuse'].sum())} of {len(fr)} fractured lesions carry diffuse damage" in row["note"]
    # one through-thickness component per fractured lesion and no partial cracks: fraction 100 %
    df["ivl_n_fractures"] = df["ivl_fracture"].astype(int)
    df["ivl_n_partial"] = 0
    row = evaluate(df, cfg)[("C10", "core")]
    ft = quantity(row, "full-thickness fraction")
    assert ft["value"] == ft["ci_lo"] == ft["ci_hi"] == pytest.approx(100.0)
    assert row["agreement"] == SC.combine(["agrees" if c_lo <= 1 <= c_hi else "disagrees",
                                           "agrees" if pf == 100.0 else "disagrees"])


def test_c11_fujino_direction(cfg):
    df = synthetic_outcomes()
    row = evaluate(df, cfg)[("C11", "core")]
    assert row["agreement"] == "agrees" and quantity(row, "rho")["value"] < 0
    assert "maximum attainable cross-sectional score 3 (cohort.csv fujino_max)" in row["note"]
    df["stent_no_ivl_stent_expansion_pct"] = 80.0 + 3.0 * df["fujino_score"]
    assert evaluate(df, cfg)[("C11", "core")]["agreement"] == "disagrees"
    row = evaluate(df, cfg, cohort=None)[("C11", "core")]                    # without cohort.csv: the package maximum
    assert "dtivl.geometry.fujino" in row["note"]


def test_c12_plaque_burden_copied(cfg):
    df = synthetic_outcomes()
    s = synthetic_summary()
    row = evaluate(df, cfg, cohort_summary=s)[("C12", "cohort")]
    assert row["agreement"] == "agrees" and row["model_value"] == s["plaque_burden_check"]["implied_plaque_burden_pct"]
    assert row["n_lesions"] == sum(s["n_per_set"][k] for k in cfg.sets)
    assert [r for r in evaluate(df, cfg) if r[0] == "C12"] == [("C12", "cohort")]
    assert evaluate(df, cfg, cohort_summary=synthetic_summary(n_above=2))[("C12", "cohort")]["agreement"] == "partial"
    outside = evaluate(df, cfg, cohort_summary=synthetic_summary(within=False))[("C12", "cohort")]
    assert outside["agreement"] == "disagrees"
    for bad in (synthetic_summary(mismatch=True), None, {"n_total": 1}):
        r = evaluate(df, cfg, cohort_summary=bad)[("C12", "cohort")]
        assert r["agreement"] == SC.NOT_COMPUTED and r["note"]


# =========================================================================== missing data
def test_missing_holdout_protocols_and_columns(cfg):
    df = synthetic_outcomes(n_hold=0)
    df.loc[:2, "ivl_status"] = "failed"
    df = df.drop(columns=[c for c in df if c.startswith("balloon_10atm_")] + ["morphology"])
    rows = evaluate(df, cfg)
    for (cid, set_), r in rows.items():
        assert r["agreement"] in SC.AGREEMENT
        if r["agreement"] == SC.NOT_COMPUTED:
            assert r["note"], (cid, set_)
        if set_ == "holdout":
            assert r["agreement"] == SC.NOT_COMPUTED and "no 'holdout' rows" in r["note"]
    assert rows[("C5", "core")]["agreement"] == SC.NOT_COMPUTED and "not run 48" in rows[("C5", "core")]["note"]
    c1 = rows[("C1", "core")]
    assert c1["agreement"] != SC.NOT_COMPUTED and [q["name"] for q in c1["quantities"]] == ["rho ivl"]
    assert "Youden cut not computed" in c1["note"]
    assert rows[("C3", "core")]["agreement"] == SC.NOT_COMPUTED and "morphology missing" in rows[("C3", "core")]["note"]
    c4 = rows[("C4", "core")]
    assert c4["n_lesions"] == 45 and "ivl status failed 3" in c4["note"]
    tiny = synthetic_outcomes(n_hold=0).iloc[:3]
    for (cid, set_), r in evaluate(tiny, cfg).items():
        if cid != "C12":
            assert r["agreement"] == SC.NOT_COMPUTED and r["note"], cid
    empty = synthetic_outcomes().iloc[0:0]
    assert all(r["agreement"] == SC.NOT_COMPUTED for k, r in evaluate(empty, cfg).items() if k[0] != "C12")


# =========================================================================== stage outputs
def test_stage_outputs_wording_and_determinism(monkeypatch, tmp_path):
    df = synthetic_outcomes()
    summary, table = run_stage(monkeypatch, tmp_path / "a", df)
    assert list(table.columns) == SC.CSV_COLUMNS
    assert set(table["agreement"]) <= set(SC.AGREEMENT)
    n_lesion_checks = len([c for c in SC.CHECKS if c not in SC.COHORT_LEVEL_CHECKS])
    assert len(table) == n_lesion_checks * len(SC.load_config().sets) + len(SC.COHORT_LEVEL_CHECKS)
    for f in ("consistency.csv", "consistency.json"):
        text = (tmp_path / "a" / f).read_text()
        assert "validat" not in text.lower() and "ivl_load_" not in text
    js = json.loads((tmp_path / "a" / "consistency.json").read_text())
    blob = (tmp_path / "a" / "consistency.json").read_text()
    assert "NaN" not in blob and "Infinity" not in blob
    assert js["config"]["seed"] == int(P.raw("design.seed")) + int(P.get("consistency.seed_offset").value)
    assert js["literature"]["insilico_postdilation_calcium_stress"]["ok"] is False
    assert js["input"]["n_rows"] == len(df) and js["input"]["by_set"] == {"core": 48, "holdout": 12}
    assert {e["id"] for e in js["not_computed"]} >= {"C9"}
    # the same input gives the same intervals
    _, again = run_stage(monkeypatch, tmp_path / "b", df)
    pd.testing.assert_frame_equal(table, again)
    # hold-out outcomes do not change core rows
    flipped = df.copy()
    h = flipped["set"] == "holdout"
    flipped.loc[h, "ivl_fracture"] = ~flipped.loc[h, "ivl_fracture"].astype(bool)
    _, t2 = run_stage(monkeypatch, tmp_path / "c", flipped)
    core_rows = table["set"] == "core"
    pd.testing.assert_frame_equal(table[core_rows].reset_index(drop=True), t2[core_rows].reset_index(drop=True))
    # rows not at the baseline calcium modulus are excluded and counted
    other = df.copy()
    other.loc[:4, "ca_E_MPa"] = other["ca_E_MPa"] * 2
    s3, _ = run_stage(monkeypatch, tmp_path / "d", other)
    assert s3["input"]["n_rows_used"] == len(df) - 5 and s3["input"]["notes"]


def test_interval_streams_do_not_depend_on_other_checks(cfg):
    df = synthetic_outcomes()
    full = evaluate(df, cfg)[("C1", "core")]
    only = {(r["id"], r["set"]): r for r in SC.evaluate(df, dataclasses.replace(cfg, checks={"C1": cfg.checks["C1"]}),
                                                         {"cohort_summary": None, "calibration": None,
                                                          "cohort": None})}[("C1", "core")]
    assert full["quantities"] == only["quantities"]


def test_wording_guard():
    with pytest.raises(ValueError, match="study rule 3"):
        SC.check_wording("the model was Validated against patients")
    SC.check_wording(SC.WORDING)


def test_stage_on_existing_cohort_outcomes(tmp_path):
    """Smoke run on results/cohort_outcomes.csv when it exists (cohort runs in progress); outputs to tmp."""
    path = RESULTS_DIR / "cohort_outcomes.csv"
    if not path.exists():
        pytest.skip("results/cohort_outcomes.csv not present")
    js = SC.run(path, tmp_path, n_boot=20)
    assert js["input"]["n_rows"] == len(pd.read_csv(path))
    assert set(pd.read_csv(tmp_path / "consistency.csv")["agreement"]) <= set(SC.AGREEMENT)


def test_no_float_literals_in_function_bodies():
    """Thresholds and published values come from the config / literature table; only named module-level
    constants (percent, null values) may carry float literals."""
    tree = ast.parse(open(SC.__file__).read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                    bad.append((sub.lineno, sub.value))
    assert not bad, f"float literals in function bodies: {bad}"
