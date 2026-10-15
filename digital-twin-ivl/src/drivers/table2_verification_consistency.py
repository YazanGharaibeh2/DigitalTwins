"""Table 2: verification checks and consistency with published observations.

Part A (verification): analytical and numerical checks of the solvers, from
results/verification_*.json, each as the largest error against its acceptance
limit (the limit stored in the file, or the named constant of the solver module
that the check uses). Part B (consistency): the checks C1-C12 of
results/consistency.csv (core set rows, and the cohort-level row), with the
published value, the model value, the criterion and the agreement label.
Consistency is not validation: the model is compared with observations from
other lesions and devices, never with outcomes of the modelled lesions.
Writes results/tables/table2_verification.csv, table2_consistency.csv
(display strings, read by the manuscript builder) and a Markdown review copy.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from dtivl.drivers.common import RESULTS_DIR
from dtivl.drivers.table3_predictors import _f
from dtivl.drivers.tables import TABLE_DIR


def _read(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def _pct(x: float, nd: int = 2) -> str:
    return f"{100.0 * x:.{nd}f} %"


def verification_rows() -> list[dict]:
    from dtivl.drivers.stage_verification import MESH_DEPENDENT
    from dtivl.solvers.nonlinear import VERIF_CONTACT_LAME_TOL, VERIF_CONTACT_PEN_TOL, VERIF_LAME_TOL

    rows = []
    lame = _read("verification_lame")
    err = max(max(c["max_rel_err_ur"], c["max_rel_err_hoop"]) for c in lame["cases"])
    mats = ", ".join(c["material"] for c in lame["cases"])
    rows.append({"Check": "Linear elasticity: thick cylinder under lumen pressure vs Lamé solution",
                 "Quantity": f"largest relative error of u_r and σθθ ({mats})", "Result": _pct(err, 3),
                 "Acceptance": f"≤ {_pct(lame['tolerance_rel'], 0)}", "Met": "yes" if err <= lame["tolerance_rel"] else "no",
                 "Source": "results/verification_lame.json"})
    mc = _read("verification_mesh_convergence")
    ch = {k: v for k, v in mc["max_rel_change_default_vs_fine"].items() if k not in MESH_DEPENDENT}
    worst = max(ch, key=ch.get)
    acc = mc["rel_change_accept"]
    rows.append({"Check": f"Mesh convergence of the predictors ({len(mc['lesions'])} lesions, {len(mc['factors'])} mesh levels)",
                 "Quantity": f"largest change, default vs finest mesh ({worst}); cell-level peaks "
                             f"({', '.join(MESH_DEPENDENT)}) excluded as mesh-dependent",
                 "Result": _pct(ch[worst], 4), "Acceptance": f"≤ {_pct(acc, 0)}",
                 "Met": "yes" if ch[worst] <= acc else "no (at the limit)" if ch[worst] <= acc * 1.01 else "no",
                 "Source": "results/verification_mesh_convergence.json"})
    pf = _read("verification_phasefield")
    e = max(max(c["rel_err_psi"], c["rel_err_strain"]) for c in pf["cases"])
    rows.append({"Check": "AT1 phase field: damage onset vs the analytical onset (uniform strain, "
                          + " and ".join(c["case"] for c in pf["cases"]) + ")",
                 "Quantity": "largest relative error of the onset energy and strain", "Result": _pct(e, 2),
                 "Acceptance": f"≤ {_pct(pf['tolerance_rel'], 0)}", "Met": "yes" if e <= pf["tolerance_rel"] else "no",
                 "Source": "results/verification_phasefield.json"})
    nl = _read("verification_nonlinear")
    lm = nl["lame"]
    e = max(lm["max_rel_err_ur_lumen"], lm["max_rel_err_ur_outer"])
    rows.append({"Check": "Hyperelastic solver: small-load thick cylinder vs Lamé solution",
                 "Quantity": "largest relative error of u_r", "Result": _pct(e, 3),
                 "Acceptance": f"≤ {_pct(VERIF_LAME_TOL, 0)}", "Met": "yes" if e <= VERIF_LAME_TOL else "no",
                 "Source": "results/verification_nonlinear.json"})
    cr = nl["contact"]["rows"]
    e = max(r["rel_err_p_vs_lame"] for r in cr)
    rows.append({"Check": "Penalty contact: rigid cylinder in a thick cylinder vs Lamé solution",
                 "Quantity": "largest relative error of the contact pressure", "Result": _pct(e, 2),
                 "Acceptance": f"≤ {_pct(VERIF_CONTACT_LAME_TOL, 0)}", "Met": "yes" if e <= VERIF_CONTACT_LAME_TOL else "no",
                 "Source": "results/verification_nonlinear.json"})
    ratio = max(r["max_abs_R_minus_rho_mm"] / r["penalty_penetration_mm"] for r in cr)
    rows.append({"Check": "Penalty contact: penetration",
                 "Quantity": "largest penetration / penalty penetration p/k_pen", "Result": _f(ratio, 2),
                 "Acceptance": f"≤ {1 + VERIF_CONTACT_PEN_TOL:g}", "Met": "yes" if ratio <= 1 + VERIF_CONTACT_PEN_TOL else "no",
                 "Source": "results/verification_nonlinear.json"})
    fd = _read("verification_foundation_linear")
    base = fd["rows"][0]["k_found"]
    nxt = fd["rows"][1]
    rows.append({"Check": "Elastic foundation (rigid-body constraint): sensitivity of the predictors",
                 "Quantity": f"change of the peak σ1 when the foundation stiffness is raised from {base:g} to "
                             f"{nxt['k_found']:g} MPa/mm", "Result": _pct(nxt["rel_change_sigma1_peak"], 3),
                 "Acceptance": f"≤ {_pct(lame['foundation_tolerance'], 1)}",
                 "Met": "yes" if nxt["rel_change_sigma1_peak"] <= lame["foundation_tolerance"] else "no",
                 "Source": "results/verification_foundation_linear.json"})
    return rows


def _model_text(r) -> str:
    v = r["model_value"]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = _f(v, 2)
    if not (pd.isna(r.get("ci_lo")) or pd.isna(r.get("ci_hi"))):
        sep = " to " if (r["ci_lo"] < 0 or r["ci_hi"] < 0) else "–"
        s += f" ({_f(r['ci_lo'], 2)}{sep}{_f(r['ci_hi'], 2)})"
    return s


def consistency_rows() -> list[dict]:
    df = pd.read_csv(RESULTS_DIR / "consistency.csv")
    out = []
    for _, r in df[df["set"].isin(["core", "cohort"])].iterrows():
        out.append({"Check": r["id"], "Observation": r["observation"], "Published": r["published"],
                    "Model quantity": r["model_quantity"], "Model value (95% CI)": _model_text(r),
                    "n": "" if pd.isna(r["n_lesions"]) else str(int(r["n_lesions"])),
                    "Criterion": r["criterion"], "Agreement": r["agreement"], "Set": r["set"]})
    return out


def _md(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [
        "| " + " | ".join("" if pd.isna(v) else str(v).replace("\n", " ") for v in r) + " |"
        for r in df.itertuples(index=False)]


def make() -> str:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    a, b = pd.DataFrame(verification_rows()), pd.DataFrame(consistency_rows())
    a.to_csv(TABLE_DIR / "table2_verification.csv", index=False)
    b.to_csv(TABLE_DIR / "table2_consistency.csv", index=False)
    (TABLE_DIR / "table2_verification_consistency.md").write_text(
        "\n".join(["A. Verification", ""] + _md(a) + ["", "B. Consistency with published observations (core set)", ""]
                  + _md(b)) + "\n")
    return str(TABLE_DIR / "table2_verification.csv")
