"""Figure 6: prediction of through-thickness calcium damage under IVL (primary outcome, D10).

(a) Core set: ROC curves of the best single mechanics predictor and the Fujino
    score (fixed scores, a priori direction) and of the composite index and the
    geometry / geometry + mechanics models (leave-one-out, out-of-fold).
(b) Hold-out set: the same scores and the core-fit models, unchanged.
(c) AUC with its bootstrap 95 % interval for every candidate predictor on the
    core set (filled) and the hold-out AUC (open), sorted by core AUC.
(d) Incremental value of mechanics over geometry: difference in AUC
    (geometry + mechanics minus geometry) with its paired-bootstrap interval.

Reads results/metrics_summary.json, results/metrics_roc.csv and
results/cohort_outcomes.csv only. ROC curves are drawn from the per-lesion
scores and predictions in these files; every AUC printed on the figure is the
value reported by the metrics stage, and the figure refuses to build if the
plotted curve does not reproduce it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from dtivl.drivers.common import RESULTS_DIR
from dtivl.drivers.figstyle import GRID, SERIES, TEXT, TEXT_2, apply_style, save
from dtivl.stats import roc as R

OUTCOME = "ivl_fracture"
GROUP_STYLE = {"mechanics": (SERIES[0], "o"), "geometry": (SERIES[1], "s"), "comparator": (SERIES[2], "D")}
LABELS = {
    "pred_sigma1_p95": "$\\sigma_1$ P95", "pred_sigma1_p99": "$\\sigma_1$ P99", "pred_sigma1_mean": "$\\sigma_1$ mean",
    "pred_sigma1_peak_nl": "$\\sigma_1$ peak (non-local)", "pred_sed_mean": "SED mean",
    "pred_sed_peak_nl": "SED peak (non-local)", "pred_scf_edge_nl": "edge SCF",
    "pred_frac_eps_gt_epsf": "area $\\varepsilon_1>\\varepsilon_f$", "pred_frac_eps_gt_half_epsf": "area $\\varepsilon_1>\\varepsilon_f/2$",
    "pred_frac_sig_gt_sigc": "area $\\sigma_1>\\sigma_c$", "pred_frac_sig_gt_half_sigc": "area $\\sigma_1>\\sigma_c/2$",
    "pred_min_thickness_high_stress": "min. thickness, high $\\sigma_1$", "ca_arc_total_deg": "calcium arc",
    "ca_thickness_max": "calcium thickness", "ca_depth": "calcium depth", "ca_area": "calcium area",
    "n_deposits": "deposits", "lumen_area": "lumen area", "lipid_arc_deg": "lipid arc",
    "plaque_ecc_index": "eccentricity", "fujino_score": "Fujino score",
}


def _label(p: str) -> str:
    return LABELS.get(p, p)


def _load():
    js = json.loads((RESULTS_DIR / "metrics_summary.json").read_text())
    roc = pd.read_csv(RESULTS_DIR / "metrics_roc.csv")
    df = pd.read_csv(RESULTS_DIR / "cohort_outcomes.csv")
    return js, roc, df


def _score_curve(df: pd.DataFrame, oc: dict, predictor: str, direction: int, set_name: str):
    """ROC points of a fixed score on one set (rows with the outcome and a finite score)."""
    t = df[(df["set"] == set_name) & df[f"{oc['protocol']}_status"].isin(oc.get("ok_status", ["ok"]))]
    y = pd.to_numeric(t[oc["column"]].map({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce")
    x = pd.to_numeric(t[predictor], errors="coerce")
    ok = y.notna() & x.notna()
    y, x = y[ok].to_numpy(int), x[ok].to_numpy(float)
    if len(y) == 0 or y.min() == y.max():
        return None
    pts = R.roc_points(y, direction * x)
    return pts["fpr"], pts["tpr"], R.auc_score(y, direction * x), len(y)


def _prob_curve(rows: list, key: str):
    rows = [r for r in rows if r.get(key) is not None]
    if not rows:
        return None
    y = np.array([r["outcome"] for r in rows], int)
    p = np.array([r[key] for r in rows], float)
    if y.min() == y.max():
        return None
    pts = R.roc_points(y, p)
    return pts["fpr"], pts["tpr"], R.auc_score(y, p), len(y)


def _roc_axes(ax, curves, title):
    """``curves``: ((label, color, ls, marker), curve or None, AUC reported by the metrics stage)."""
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.0, ls="--", zorder=0)
    drawn = 0
    for (label, color, ls, marker), cur, auc_ref in curves:
        if cur is None or auc_ref is None:
            continue
        fpr, tpr, auc, n = cur
        if abs(auc - auc_ref) > 1e-9:          # the curve must be the one the metrics stage scored
            raise ValueError(f"{label}: AUC of the plotted curve {auc} differs from metrics_summary {auc_ref}")
        ax.step(fpr, tpr, where="post", color=color, ls=ls, lw=1.5, label=f"{label}: AUC {auc_ref:.2f}")
        drawn += 1
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, loc="left")
    if drawn:
        ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), frameon=False, fontsize=5.5, handlelength=2.6,
                  borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", color=TEXT_2, transform=ax.transAxes)


def make(name: str = "fig6_prediction") -> str:
    import matplotlib.pyplot as plt

    apply_style()
    js, roc, df = _load()
    out = js["outcomes"][OUTCOME]
    cfg = js["config"]
    oc = out
    comp_name = cfg.get("composite_index_name", "composite")
    best = (out.get("fujino_comparison") or {}).get("best_single_mechanics")
    uni = out.get("univariable", {})
    comp = out.get("composite") or {}
    inc = out.get("incremental_value") or {}
    c_rows = comp.get("lesions", [])
    i_rows = inc.get("lesions", [])

    fig = plt.figure(figsize=(7.1, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1], wspace=0.5, hspace=0.75)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axC, axD = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    def uni_auc(p, set_name):
        u = uni[p]
        return u.get("auc") if set_name == "core" else (u.get("holdout") or {}).get("auc")

    def score_entries(set_name):
        ent = []
        for p, lbl, color, marker in ((best, _label(best) if best else "", SERIES[0], "o"),
                                      ("fujino_score", "Fujino score", SERIES[2], "D")):
            if p and p in uni:
                ent.append(((lbl, color, "-", marker), _score_curve(df, oc, p, int(uni[p]["direction"]), set_name),
                            uni_auc(p, set_name)))
        return ent

    cc, ci = comp.get("core") or {}, inc.get("core") or {}
    ch, ih = comp.get("holdout") or {}, inc.get("holdout") or {}

    core_models = [((f"{comp_name} (LOOCV)", SERIES[3], "--", "^"),
                    _prob_curve([r for r in c_rows if r["set"] == "core"], "p_loocv"), cc.get("loocv_auc")),
                   (("geometry model (LOOCV)", SERIES[1], ":", "s"),
                    _prob_curve([r for r in i_rows if r["set"] == "core"], "p_geometry_loocv"),
                    (ci.get("loocv") or {}).get("auc_b")),
                   (("geometry + mechanics (LOOCV)", SERIES[6], "-.", "v"),
                    _prob_curve([r for r in i_rows if r["set"] == "core"], "p_geometry_mechanics_loocv"),
                    (ci.get("loocv") or {}).get("auc_a"))]
    hold_models = [((f"{comp_name} (core fit)", SERIES[3], "--", "^"),
                    _prob_curve([r for r in c_rows if r["set"] == "holdout"], "p_core_fit"), ch.get("auc")),
                   (("geometry model (core fit)", SERIES[1], ":", "s"),
                    _prob_curve([r for r in i_rows if r["set"] == "holdout"], "p_geometry_core_fit"), ih.get("auc_b")),
                   (("geometry + mechanics (core fit)", SERIES[6], "-.", "v"),
                    _prob_curve([r for r in i_rows if r["set"] == "holdout"], "p_geometry_mechanics_core_fit"),
                    ih.get("auc_a"))]
    nc, nh = out["core"]["n"], out["holdout"]["n"]
    _roc_axes(axA, score_entries("core") + core_models, f"a  Core set (n = {nc})")
    _roc_axes(axB, score_entries("holdout") + hold_models, f"b  Hold-out set (n = {nh})")

    # (c) AUC forest
    r = roc[(roc["outcome"] == OUTCOME) & (roc["predictor"].isin(uni.keys()))]
    rc = r[r["set"] == "core"].set_index("predictor")
    rh = r[r["set"] == "holdout"].set_index("predictor")
    order = rc["auc"].sort_values(ascending=True).index.tolist()
    for k, p in enumerate(order):
        color, marker = GROUP_STYLE.get(rc.loc[p, "group"], (TEXT_2, "o"))
        lo, hi = rc.loc[p, "auc_ci_lo"], rc.loc[p, "auc_ci_hi"]
        axC.plot([lo, hi], [k, k], color=color, lw=1.2, solid_capstyle="round")
        axC.plot(rc.loc[p, "auc"], k, marker=marker, color=color, ms=4, mec="white", mew=0.5, ls="none")
        if p in rh.index and np.isfinite(rh.loc[p, "auc"]):
            axC.plot(rh.loc[p, "auc"], k, marker=marker, mfc="white", mec=color, ms=4, mew=1.0, ls="none")
    axC.axvline(0.5, color=GRID, lw=1.0, ls="--", zorder=0)
    axC.set_yticks(range(len(order)))
    axC.set_yticklabels([_label(p) for p in order], fontsize=5.5)
    axC.set_xlim(0, 1.02)
    axC.set_xlabel("AUC (core: filled, 95 % bootstrap interval; hold-out: open)")
    axC.set_title("c  Univariable discrimination", loc="left")
    axC.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], color=c, marker=m, ls="none", ms=4, label=g) for g, (c, m) in GROUP_STYLE.items()
            if (rc["group"] == g).any()]
    axC.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3, frameon=False, fontsize=5.5,
               borderaxespad=0.0)

    # (d) incremental value: delta AUC
    rows = []
    core_inc = inc.get("core") or {}
    if core_inc.get("loocv"):
        lv = core_inc["loocv"]
        rows.append(("core, LOOCV", lv["delta_auc"], lv["delta_ci_lo"], lv["delta_ci_hi"]))
    if core_inc.get("apparent_refit_bootstrap"):
        ap = core_inc["apparent_refit_bootstrap"]
        rows.append(("core, apparent (refit bootstrap)", ap["delta_auc"], ap["delta_ci_lo"], ap["delta_ci_hi"]))
    hold_inc = inc.get("holdout") or {}
    if hold_inc.get("delta_auc") is not None:
        rows.append(("hold-out (core-fit models)", hold_inc["delta_auc"], hold_inc["delta_ci_lo"],
                     hold_inc["delta_ci_hi"]))
    if rows:
        for k, (lbl, d, lo, hi) in enumerate(rows[::-1]):
            axD.plot([lo, hi], [k, k], color=SERIES[6], lw=1.4, solid_capstyle="round")
            axD.plot(d, k, marker="o", color=SERIES[6], ms=4.5, mec="white", mew=0.5, ls="none")
        axD.set_yticks(range(len(rows)))
        axD.set_yticklabels([r[0] for r in rows[::-1]], fontsize=5.5)
        span = max(abs(v) for rr in rows for v in rr[1:] if v is not None and math.isfinite(v)) or 0.05
        axD.set_xlim(-1.15 * span, 1.15 * span)
        axD.set_ylim(-0.7, len(rows) - 0.3)
    else:
        axD.text(0.5, 0.5, "not available", ha="center", va="center", color=TEXT_2, transform=axD.transAxes)
        axD.set_yticks([])
    axD.axvline(0.0, color=TEXT_2, lw=0.8)
    axD.set_xlabel("$\\Delta$AUC, geometry + mechanics − geometry (95 % interval)")
    axD.set_title("d  Incremental value of mechanics", loc="left")
    axD.grid(axis="y", visible=False)
    for ax in (axA, axB, axC, axD):
        ax.title.set_color(TEXT)
    return str(save(fig, name))
