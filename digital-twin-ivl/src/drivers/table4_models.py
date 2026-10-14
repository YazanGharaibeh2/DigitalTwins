"""Table 4: composite index, Fujino score and the incremental value of mechanics (primary outcome, D10).

Rows: the best single mechanics predictor, the composite index (name from
design.composite_index_name), the Fujino score, the geometry-only model and the
geometry + mechanics model, with core apparent AUC, leave-one-out AUC (with its
interval), log-loss and Brier score where defined, and hold-out AUC; then the
paired differences (composite and best predictor minus Fujino; geometry +
mechanics minus geometry) with their bootstrap intervals and the share of
resamples with a difference <= 0. Values are copied from
results/metrics_summary.json; only their display format is set here. Writes
results/tables/table4_models.csv (display strings) and .md.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from dtivl.drivers.common import RESULTS_DIR
from dtivl.drivers.table3_predictors import DEGENERATE, NAMES, _ci, _f
from dtivl.drivers.tables import TABLE_DIR

OUTCOME = "ivl_fracture"


def _v(d: dict | None, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def _row(model, kind, n_core, app, loo, loo_ci, loo_deg, ll, brier, hold, hold_ci, hold_deg, n_hold):
    return {"Model": model, "Type": kind, "n core": n_core,
            "Core AUC, apparent": _f(app), "Core AUC, LOOCV (95% CI)": _ci(loo, *(loo_ci or (None, None)), degenerate=loo_deg),
            "LOOCV log-loss": _f(ll, 3), "LOOCV Brier": _f(brier, 3),
            "Hold-out AUC (95% CI)": _ci(hold, *(hold_ci or (None, None)), degenerate=hold_deg), "n hold-out": n_hold}


def _diff_row(label, d: dict | None, setname):
    if not d:
        return None
    return {"Comparison": label, "Set": setname, "AUC a": _f(d.get("auc_a")), "AUC b": _f(d.get("auc_b")),
            "ΔAUC (95% CI)": _ci(d.get("delta_auc"), d.get("delta_ci_lo"), d.get("delta_ci_hi"),
                                  degenerate=bool(d.get("delta_ci_degenerate"))),
            "Share of resamples with ΔAUC ≤ 0": _f(d.get("boot_frac_delta_le_0"), 3),
            "Δ log-loss (95% CI)": _ci(d.get("delta_log_loss"), d.get("delta_log_loss_ci_lo"),
                                       d.get("delta_log_loss_ci_hi"), nd=3) if "delta_log_loss" in d else "",
            "Δ Brier (95% CI)": _ci(d.get("delta_brier"), d.get("delta_brier_ci_lo"), d.get("delta_brier_ci_hi"),
                                    nd=3) if "delta_brier" in d else ""}


def build(js: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = js["outcomes"][OUTCOME]
    name = js["config"].get("composite_index_name", "composite")
    uni, comp, inc = o.get("univariable", {}), o.get("composite") or {}, o.get("incremental_value") or {}
    best = _v(o, "fujino_comparison", "best_single_mechanics")
    nc, nh = _v(o, "core", "n"), _v(o, "holdout", "n")
    models = []
    for p, kind in ((best, "single mechanics predictor (best core AUC)"), ("fujino_score", "imaging score")):
        if p and p in uni:
            u = uni[p]
            h = u.get("holdout") or {}
            models.append(_row(NAMES.get(p, (p,))[0], kind, nc, u.get("auc"), None, None, False, None, None,
                               h.get("auc"), (h.get("auc_ci_lo"), h.get("auc_ci_hi")) if h.get("auc_ci_lo") is not None
                               else None, bool(h.get("auc_ci_degenerate")), h.get("n")))
    cc, ch = comp.get("core") or {}, comp.get("holdout") or {}
    if cc:
        sel = ", ".join(NAMES.get(p, (p,))[0] for p in comp.get("selected", []))
        models.append(_row(f"{name} ({sel})", "composite index (penalised logistic)", cc.get("n"), cc.get("apparent_auc"),
                           cc.get("loocv_auc"), cc.get("loocv_auc_ci"), bool(cc.get("loocv_auc_ci_degenerate")),
                           cc.get("loocv_log_loss"), cc.get("loocv_brier"), ch.get("auc"),
                           (ch.get("auc_ci_lo"), ch.get("auc_ci_hi")) if ch.get("auc_ci_lo") is not None else None,
                           bool(ch.get("auc_ci_degenerate")), ch.get("n")))
    ic, ih = inc.get("core") or {}, inc.get("holdout") or {}
    lo, ap = ic.get("loocv") or {}, ic.get("apparent_refit_bootstrap") or {}
    for lbl, a_or_b in (("geometry model", "b"), ("geometry + mechanics model", "a")):
        if lo:
            models.append(_row(lbl, "penalised logistic", ic.get("n"), ap.get(f"auc_{a_or_b}"), lo.get(f"auc_{a_or_b}"),
                               None, False, lo.get(f"log_loss_{a_or_b}"), lo.get(f"brier_{a_or_b}"),
                               ih.get(f"auc_{a_or_b}"), None, False, ih.get("n")))
    diffs = []
    ad = _v(o, "fujino_comparison", "fujino_score", "auc_differences") or {}
    for setname in ("core", "holdout"):
        for key, d in (ad.get(setname) or {}).items():
            if isinstance(d, dict) and "delta_auc" in d:
                lbl = key.replace("fujino_score", "Fujino score")
                for p, (nm, _) in NAMES.items():
                    lbl = lbl.replace(p, nm)
                diffs.append(_diff_row(lbl, d, setname))
    diffs.append(_diff_row("geometry + mechanics − geometry, apparent (refit in each resample)", ap or None, "core"))
    diffs.append(_diff_row("geometry + mechanics − geometry, LOOCV", lo or None, "core"))
    if ih.get("delta_auc") is not None:
        diffs.append(_diff_row("geometry + mechanics − geometry, core-fit models", ih, "holdout"))
    return pd.DataFrame(models), pd.DataFrame([d for d in diffs if d])


def _md(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return (["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) or v is None else str(v) for v in r) + " |"
               for r in df.itertuples(index=False)])


def make() -> str:
    js = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    models, diffs = build(js)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    models.to_csv(TABLE_DIR / "table4_models.csv", index=False)
    diffs.to_csv(TABLE_DIR / "table4_differences.csv", index=False)
    md = _md(models) + [""] + _md(diffs) + ["", f"{DEGENERATE} Degenerate bootstrap interval (AUC of 0 or 1, or every "
                                                 "resample equal): not a precision statement. LOOCV intervals: "
                                                 "bootstrap of fixed out-of-fold predictions (refit and reselection "
                                                 "variability not included)."]
    (TABLE_DIR / "table4_models.md").write_text("\n".join(md) + "\n")
    return str(TABLE_DIR / "table4_models.csv")
