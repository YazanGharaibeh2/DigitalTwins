"""Table 3: univariable prediction of through-thickness damage under IVL (primary outcome, D10).

One row per candidate predictor (pre-treatment mechanics, geometry descriptors,
Fujino score): a priori direction, core-set AUC with its bootstrap interval,
Youden threshold, sensitivity and specificity (apparent and leave-one-out), and
the hold-out AUC, sensitivity and specificity at the core threshold. Values are
copied from results/metrics_roc.csv; only their display format is set here.
Writes results/tables/table3_predictors.csv (display strings) and .md.
"""
from __future__ import annotations

import math

import pandas as pd

from dtivl.drivers.common import RESULTS_DIR
from dtivl.drivers.tables import TABLE_DIR

OUTCOME = "ivl_fracture"
#: display name and unit of each candidate (units of the predictor definitions in metrics/predictors.py)
NAMES = {
    "pred_sigma1_p95": ("σ1, 95th percentile", "MPa"), "pred_sigma1_p99": ("σ1, 99th percentile", "MPa"),
    "pred_sigma1_mean": ("σ1, mean", "MPa"), "pred_sigma1_peak_nl": ("σ1, non-local peak", "MPa"),
    "pred_sed_mean": ("SED, mean", "MJ/m³"), "pred_sed_peak_nl": ("SED, non-local peak", "MJ/m³"),
    "pred_scf_edge_nl": ("Edge stress concentration", "–"),
    "pred_frac_eps_gt_epsf": ("Area fraction ε1 > εf", "–"), "pred_frac_eps_gt_half_epsf": ("Area fraction ε1 > εf/2", "–"),
    "pred_frac_sig_gt_sigc": ("Area fraction σ1 > σc", "–"), "pred_frac_sig_gt_half_sigc": ("Area fraction σ1 > σc/2", "–"),
    "pred_min_thickness_high_stress": ("Minimum thickness under high stress", "mm"),
    "ca_arc_total_deg": ("Calcium arc", "deg"), "ca_thickness_max": ("Calcium thickness", "mm"),
    "ca_depth": ("Calcium depth", "mm"), "ca_area": ("Calcium area", "mm²"), "n_deposits": ("Number of deposits", "–"),
    "lumen_area": ("Lumen area", "mm²"), "lipid_arc_deg": ("Lipid arc", "deg"),
    "plaque_ecc_index": ("Plaque eccentricity", "–"), "fujino_score": ("Fujino score (cross-sectional)", "points"),
}
GROUP = {"mechanics": "Mechanics", "geometry": "Geometry", "comparator": "Comparator"}
DEGENERATE = "†"


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _f(x, nd=2) -> str:
    return "" if _missing(x) else f"{x:.{nd}f}".replace("-", "\u2212")


def _g(x, sig=3) -> str:
    return "" if _missing(x) else f"{x:.{sig}g}".replace("-", "\u2212")


def _ci(v, lo, hi, nd=2, fmt=_f, degenerate=False) -> str:
    """'v (lo–hi)'; ' to ' as the separator when a bound is negative."""
    if fmt(v, nd) == "":
        return ""
    if fmt(lo, nd) and fmt(hi, nd):
        sep = " to " if (lo < 0 or hi < 0) else "\u2013"
        s = f"{fmt(v, nd)} ({fmt(lo, nd)}{sep}{fmt(hi, nd)})"
    else:
        s = fmt(v, nd)
    return s + (DEGENERATE if degenerate else "")


def _pair(a, b, nd=2) -> str:
    return "" if _f(a, nd) == "" else f"{_f(a, nd)} / {_f(b, nd)}"


def rows(roc: pd.DataFrame) -> pd.DataFrame:
    r = roc[(roc["outcome"] == OUTCOME) & (roc["predictor"].isin(NAMES))]
    core = r[r["set"] == "core"].set_index("predictor")
    hold = r[r["set"] == "holdout"].set_index("predictor")
    out = []
    for p in core.sort_values("auc", ascending=False).index:
        c = core.loc[p]
        h = hold.loc[p] if p in hold.index else None
        name, unit = NAMES[p]
        d = c["direction"]
        rule = "" if pd.isna(d) else ("higher" if d > 0 else "lower") + (" (set on core)" if c["direction_source"] == "core" else "")
        out.append({
            "Predictor": name, "Unit": unit, "Group": GROUP.get(c["group"], c["group"]),
            "Damage expected for": rule,
            "Core AUC (95% CI)": _ci(c["auc"], c["auc_ci_lo"], c["auc_ci_hi"], degenerate=bool(c.get("auc_ci_degenerate") is True or c.get("auc_ci_degenerate") == 1)),
            "Threshold (95% CI)": _ci(c["threshold"], c["threshold_ci_lo"], c["threshold_ci_hi"], nd=3, fmt=_g),
            "Sens / spec": _pair(c["sens"], c["spec"]),
            "LOOCV sens / spec": _pair(c["loocv_sens"], c["loocv_spec"]),
            "Hold-out AUC (95% CI)": "" if h is None else _ci(h["auc"], h["auc_ci_lo"], h["auc_ci_hi"]),
            "Hold-out sens / spec": "" if h is None else _pair(h["sens"], h["spec"]),
            "n core (events)": f"{int(c['n'])} ({int(c['n_pos'])})",
            "n hold-out (events)": "" if h is None or pd.isna(h["n"]) else f"{int(h['n'])} ({int(h['n_pos'])})",
            "key": p,
        })
    return pd.DataFrame(out)


def make() -> str:
    roc = pd.read_csv(RESULTS_DIR / "metrics_roc.csv")
    df = rows(roc)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / "table3_predictors.csv"
    df.to_csv(path, index=False)
    cols = [c for c in df.columns if c != "key"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in df[cols].itertuples(index=False)]
    md += ["", f"{DEGENERATE} Degenerate bootstrap interval (AUC of 0 or 1, or every resample equal): not a precision statement."]
    (TABLE_DIR / "table3_predictors.md").write_text("\n".join(md) + "\n")
    return str(path)
