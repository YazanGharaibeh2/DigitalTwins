"""Tests for the metrics tables (Tables 3 and 4): values are copied from the metrics outputs."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import pytest

from dtivl.drivers import stage_metrics as S
from dtivl.drivers import table3_predictors as T3
from dtivl.drivers import table4_models as T4
from test_stats import synthetic_cohort


@pytest.fixture(scope="module")
def metrics_out(tmp_path_factory):
    d = tmp_path_factory.mktemp("tables")
    p = d / "cohort_outcomes.csv"
    synthetic_cohort().to_csv(p, index=False)
    S.run(p, d, n_boot=50)
    return d


def test_formatting_helpers():
    assert T3._ci(0.5, 0.25, 0.75) == "0.50 (0.25–0.75)"
    assert T3._ci(-0.08, -0.15, -0.03, nd=2) == "−0.08 (−0.15 to −0.03)"
    assert T3._ci(1.0, 1.0, 1.0, degenerate=True).endswith(T3.DEGENERATE)
    assert T3._ci(None, None, None) == "" and T3._f(float("nan")) == ""


def test_table3_copies_metrics_roc(metrics_out):
    roc = pd.read_csv(metrics_out / "metrics_roc.csv")
    df = T3.rows(roc)
    core = roc[(roc.outcome == T3.OUTCOME) & (roc.set == "core")].set_index("predictor")
    assert set(df["key"]) == set(core.index) & set(T3.NAMES)
    for _, r in df.iterrows():
        c = core.loc[r["key"]]
        assert r["Core AUC (95% CI)"].startswith(T3._f(c["auc"]))
    aucs = [core.loc[k, "auc"] for k in df["key"]]
    assert aucs == sorted(aucs, reverse=True)


def test_table4_copies_metrics_summary(metrics_out):
    js = json.loads((metrics_out / "metrics_summary.json").read_text())
    models, diffs = T4.build(js)
    o = js["outcomes"][T4.OUTCOME]
    comp = models[models["Type"].str.startswith("composite")].iloc[0]
    assert comp["Core AUC, LOOCV (95% CI)"].startswith(T3._f(o["composite"]["core"]["loocv_auc"]))
    lo = o["incremental_value"]["core"]["loocv"]
    row = diffs[diffs["Comparison"] == "geometry + mechanics − geometry, LOOCV"].iloc[0]
    assert row["ΔAUC (95% CI)"].startswith(T3._f(lo["delta_auc"]))
    assert {"core", "holdout"} <= set(diffs["Set"])
