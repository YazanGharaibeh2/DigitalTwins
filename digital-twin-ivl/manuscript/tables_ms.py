"""Table blocks of the manuscript. Cells are copied from results/tables/*.csv (written by
src/drivers/tables.py); here only columns are selected, numeric cells formatted and literature sources
replaced by citation numbers."""
from __future__ import annotations

import math
import re

from build_manuscript import Ctx, Table


def _num(x, sig: int = 3) -> str:
    """A plain number cell to ``sig`` significant digits; anything else unchanged."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)
    if math.isnan(v):
        return ""
    s = f"{v:.{sig}g}"
    if "e" in s:                                   # no exponent notation in the table
        r = float(s)
        s = f"{r:.0f}" if abs(r) >= 1 else f"{r:.{sig + 3}f}".rstrip("0")
    return s.replace("-", "−") if s.startswith("-") else s


def _symbol(x) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    if not re.fullmatch(r"[A-Za-z\\{}_^,0-9]+", x):
        return x
    x = re.sub(r"_([A-Za-z0-9,]{2,})", r"_{\1}", x)     # D_ref -> D_{ref}
    return f"${x}$"


def _table1(ctx: Ctx, df):
    out = df.copy()

    def src(r):
        keys = [k.strip() for k in str(r["lit_key"]).replace(";", ",").split(",")] if isinstance(r["lit_key"], str) else []
        cites = ctx.cite_lit(*keys) if keys else ""
        return cites or (r["source"] if isinstance(r["source"], str) else "")
    out["Source"] = [src(r) for _, r in out.iterrows()]
    out["Value"] = [_num(v) for v in out["value"]]
    out["Symbol"] = [_symbol(v) for v in out["symbol"]]
    out["Unit"] = [str(u).replace("^2", "²").replace("^3", "³") if isinstance(u, str) else "" for u in out["unit"]]
    out["Sensitivity range"] = [s if isinstance(s, str) else "" for s in out["sensitivity"]]
    out["Parameter"] = out["parameter"]
    out["Group"] = out["group"]
    out["Status"] = out["status"]
    return out


def table1(ctx: Ctx) -> Table:
    return Table("table1_parameters",
                 "Model inputs. Literature values with their status (VERIFIED: value confirmed in the cited source; "
                 "SECONDARY: from a secondary source; DERIVED: computed from other inputs; DESIGN or ASSUMPTION: "
                 "study choice), the range used in the sensitivity analysis, and the source or basis.",
                 columns=["Group", "Parameter", "Symbol", "Value", "Unit", "Sensitivity range", "Status", "Source"],
                 prepare=_table1)


def table2(ctx: Ctx) -> list:
    return [
        Table("table2_verification_consistency",
              "(a) Verification against analytical solutions and mesh refinement.",
              file="table2_verification", columns=["Check", "Quantity", "Result", "Acceptance", "Met"]),
    ]


def table2b(ctx: Ctx) -> list:
    return [
        Table("table2_verification_consistency",
              "(continued) (b) Consistency of the simulated outcomes with published observations and model-internal "
              "criteria. Agreement criteria are stated per check in the results files; hold-out rows are evaluated "
              "with core-derived quantities only.",
              file="table2_consistency",
              columns=["Check", "Set", "Observation", "Published", "Model value (95% CI)", "n", "Agreement"]),
    ]


def table3(ctx: Ctx) -> Table:
    return Table("table3_predictors",
                 "Univariable prediction of through-thickness damage under IVL. Core AUC with bootstrap interval, "
                 "Youden threshold, sensitivity and specificity (apparent and leave-one-out), and the hold-out AUC, "
                 "sensitivity and specificity at the core threshold. 'Set on core': direction not stated a priori.",
                 columns=["Predictor", "Unit", "Damage expected for", "Core AUC (95% CI)", "Threshold (95% CI)",
                          "Sens / spec", "LOOCV sens / spec", "Hold-out AUC (95% CI)", "Hold-out sens / spec"],
                 note="† Degenerate bootstrap interval (an AUC of zero or one, or every resample equal); not a "
                      "statement of precision.")


def table4(ctx: Ctx) -> list:
    return [
        Table("table4_models",
              "(a) Composite index, best single predictor, Fujino score and the geometry models for "
              "through-thickness damage under IVL.",
              columns=["Model", "Type", "Core AUC, apparent", "Core AUC, LOOCV (95% CI)", "LOOCV log-loss",
                       "LOOCV Brier", "Hold-out AUC (95% CI)"]),
        Table("table4_models",
              "(continued) (b) Paired differences: AUC difference with its bootstrap interval, share of resamples "
              "with a difference at or below zero (descriptive), and differences in log-loss and Brier score.",
              file="table4_differences"),
    ]
