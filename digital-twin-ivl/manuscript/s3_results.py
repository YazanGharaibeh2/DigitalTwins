"""Section 3, Results. Every number is read from results/*.json|csv (rule 1); wording that depends on a
result (direction of a difference, whether an interval includes zero) is chosen from the same values."""
from __future__ import annotations

import math

from build_manuscript import BuildError, Ctx, Figure, Heading, Para

import tables_ms
from s2_methods import _atm_mpa, _const
from s2_statistics import GEOMETRY_WORDS

MS = "results/metrics_summary.json"
OUT = "results/cohort_outcomes.csv"
CONS = "results/consistency.json"
LAME = "results/verification_lame.json"
NLV = "results/verification_nonlinear.json"
PFV = "results/verification_phasefield.json"
MESH = "results/verification_mesh_convergence.json"
FIELDS = "results/fields/fields_summary.json"
PRIMARY = "outcomes.ivl_fracture"

#: protocols in the order of the text, with their description
PROTOCOLS = {
    "physiological": "physiological pressure",
    "balloon_only": "the IVL balloon inflated without pulses",
    "balloon_10atm": "the plain balloon",
    "ivl": "IVL",
    "stent_no_ivl": "stenting without IVL",
    "stent_post_ivl": "stenting after IVL",
}
MORPH_WORDS = {"superficial_sheet": "superficial sheets", "deep_sheet": "deep sheets", "nodule": "nodules",
               "fragmented": "fragmented deposits"}


# ------------------------------------------------------------------ helpers
def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pname(ctx: Ctx, p: str) -> str:
    """In-text name of a candidate predictor (percentile levels from the predictors module)."""
    pc = lambda n: _const(ctx, "dtivl.metrics.predictors", n, fmt="g")  # noqa: E731
    if p == "pred_sigma1_p95":
        return f"the {pc('SIGMA1_PERCENTILE')}th percentile of $\\sigma_1$"
    if p == "pred_sigma1_p99":
        return f"the {pc('SIGMA1_PERCENTILE_HIGH')}th percentile of $\\sigma_1$"
    fixed = {
        "pred_sigma1_mean": "the mean $\\sigma_1$", "pred_sigma1_peak_nl": "the non-local peak $\\sigma_1$",
        "pred_sed_mean": "the mean SED", "pred_sed_peak_nl": "the non-local peak SED",
        "pred_scf_edge_nl": "the edge stress concentration factor",
        "pred_frac_eps_gt_epsf": "the area fraction with $\\varepsilon_1 > \\varepsilon_f$",
        "pred_frac_eps_gt_half_epsf": "the area fraction with $\\varepsilon_1 > \\varepsilon_f/2$",
        "pred_frac_sig_gt_sigc": "the area fraction with $\\sigma_1$ above the failure stress",
        "pred_frac_sig_gt_half_sigc": "the area fraction with $\\sigma_1$ above half the failure stress",
        "pred_min_thickness_high_stress": "the minimum calcium thickness under high stress",
        "fujino_score": "the Fujino score",
        **{k: "the " + v for k, v in GEOMETRY_WORDS.items()},
    }
    if p not in fixed:
        raise BuildError(f"no in-text name for predictor {p}")
    return fixed[p]


def _and(words: list[str]) -> str:
    words = list(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _interval(ctx: Ctx, lo, hi, *, src: str, key: str, fmt: str = ".2f") -> str:
    """'lo–hi', or 'lo to hi' when a bound is negative."""
    a = ctx.num(lo, src=src, key=key + "[lo]", fmt=fmt)
    b = ctx.num(hi, src=src, key=key + "[hi]", fmt=fmt)
    neg = (not _missing(lo) and lo < 0) or (not _missing(hi) and hi < 0)
    return f"{a} to {b}" if neg else f"{a}–{b}"


def _vci(ctx: Ctx, rel: str, key: str, ci_key: str | tuple, fmt: str = ".2f", degenerate_key: str | None = None,
         unit: str = "", inner: bool = False) -> str:
    """'value (95% CI lo–hi)' from a json file, or 'value; 95% CI lo–hi' (``inner``, for use inside
    parentheses); ``ci_key`` is a list key or a (lo, hi) pair of keys."""
    v = ctx.v(rel, key, fmt=fmt)
    if isinstance(ci_key, tuple):
        lo, hi = ctx.opt(rel, ci_key[0]), ctx.opt(rel, ci_key[1])
        k = ci_key[0]
    else:
        ci = ctx.opt(rel, ci_key) or [None, None]
        lo, hi = ci
        k = ci_key
    lbl = f"{ctx.param('metrics.ci_level', fmt='.0f', scale=100.0)}% CI"
    s = f"{lbl} {_interval(ctx, lo, hi, src=rel, key=k, fmt=fmt)}{unit}"
    if degenerate_key and ctx.opt(rel, degenerate_key) is True:
        s += ", degenerate"
    return f"{v}{unit}; {s}" if inner else f"{v}{unit} ({s})"


def _diff_word(lo, hi, pos: str, neg: str, none: str) -> str:
    if _missing(lo) or _missing(hi):
        return Ctx.PENDING
    if lo > 0:
        return pos
    if hi < 0:
        return neg
    return none


def _k_of_n(ctx: Ctx, k: int, n: int, *, key: str, src: str = OUT) -> str:
    if n == 0:
        if ctx.draft:
            ctx.pending.append(f"{src}:{key}")
            return f"{Ctx.PENDING} of {Ctx.PENDING}"
        raise BuildError(f"no lesions for {key}")
    return f"{ctx.count(k, src=src, key=key + ' [events]')} of {ctx.count(n, src=src, key=key + ' [n]')}"


def _ok(df, protocol: str):
    """Rows whose run of ``protocol`` is usable (the status filter of the metrics stage)."""
    ok = {"stent_no_ivl": ("ok", "limited"), "stent_post_ivl": ("ok", "limited")}.get(protocol, ("ok",))
    col = f"{protocol}_status"
    return df[df[col].isin(ok)] if col in df else df.iloc[0:0]


def _sets(ctx: Ctx):
    df = ctx.data(OUT)
    return df, df[df["set"] == "core"], df[df["set"] == "holdout"]


# ------------------------------------------------------------------ 3.1
def s31(ctx: Ctx) -> list:
    lame_cases = ctx.data(LAME)["cases"]
    i_h = max(range(len(lame_cases)), key=lambda i: lame_cases[i]["max_rel_err_hoop"])
    i_u = max(range(len(lame_cases)), key=lambda i: lame_cases[i]["max_rel_err_ur"])
    pf_cases = ctx.data(PFV)["cases"]
    i_p = max(range(len(pf_cases)), key=lambda i: pf_cases[i]["rel_err_psi"])
    i_s = max(range(len(pf_cases)), key=lambda i: pf_cases[i]["rel_err_strain"])
    mesh = ctx.data(MESH)
    acc = mesh["rel_change_accept"]
    dep = mesh["mesh_dependent_quantities"]
    ch = {k: v for k, v in mesh["max_rel_change_default_vs_fine"].items() if k not in dep}
    within = {k: v for k, v in ch.items() if v <= acc}
    above = {k: v for k, v in ch.items() if v > acc}
    kmax = max(within, key=within.get)
    pc = lambda k, rel=MESH: ctx.v(rel, f"max_rel_change_default_vs_fine.{k}", fmt=".1f", scale=100.0)  # noqa: E731
    gci = mesh["lesions"]
    if above:
        conv = sorted({gci[les]["gci"][k]["convergence"] for les in gci for k in above if k in gci[les]["gci"]})
        above_txt = (f" The exception was {_and([pname(ctx, 'pred_' + k) for k in above])}, which changed by "
                     f"{_and([pc(k) + '%' for k in above])}, just above the limit, with {_and(conv)} behaviour of "
                     f"the three-level estimate {ctx.cite('celik2008gci')}; it was kept as a candidate and is "
                     "flagged here.")
    else:
        above_txt = ""
    return [
        Heading("3. Results", 1),
        Heading("3.1. Verification", 2),
        Para(
            "The linear solver reproduced the plane-strain Lamé solution of a thick-walled cylinder with a "
            f"largest relative error of {ctx.v(LAME, f'cases.{i_h}.max_rel_err_hoop', fmt='.2g', scale=100.0)}% in "
            f"the hoop stress and {ctx.v(LAME, f'cases.{i_u}.max_rel_err_ur', fmt='.2g', scale=100.0)}% in the "
            f"radial displacement (tolerance {ctx.v(LAME, 'tolerance_rel', fmt='g', scale=100.0)}%; "
            f"{ctx.fig('fig3_verification')}a,b). The hyperelastic solver reproduced the small-load Lamé "
            f"displacement within {ctx.v(NLV, 'lame.max_rel_err_ur_lumen', fmt='.2g', scale=100.0)}%. In "
            "homogeneous tension, damage onset in the AT1 phase field occurred within "
            f"{ctx.v(PFV, f'cases.{i_p}.rel_err_psi', fmt='.2g', scale=100.0)}% of the analytical critical "
            f"energy and {ctx.v(PFV, f'cases.{i_s}.rel_err_strain', fmt='.2g', scale=100.0)}% of the analytical "
            f"onset strain (tolerance {ctx.v(PFV, 'tolerance_rel', fmt='g', scale=100.0)}%; "
            f"{ctx.fig('fig3_verification')}d)."),
        Para(
            "Between the default and the finest mesh, the pre-treatment predictors changed by at most "
            f"{pc(kmax)}% ({pname(ctx, 'pred_' + kmax)}), within the acceptance limit of "
            f"{ctx.v(MESH, 'rel_change_accept', fmt='g', scale=100.0)}% ({ctx.fig('fig3_verification')}c)."
            f"{above_txt} The raw cell peaks of $\\sigma_1$ and SED changed by {pc('sigma1_peak')}% and "
            f"{pc('sed_peak')}%, which is why only their non-local versions were used. The verification results "
            f"are summarised in {ctx.tab('table2_verification_consistency')}."),
        Figure("fig3_verification",
               "Verification. (a) Hoop stress of the linear solver against the plane-strain Lamé solution of a "
               "thick-walled cylinder. (b) Relative error of the radial displacement and the hoop stress at each "
               "sample radius, with the tolerance. (c) Pre-treatment predictors at the default and the coarse mesh "
               "relative to the finest mesh, with the acceptance band; the raw cell peaks are marked as excluded. "
               "(d) Damage onset of the AT1 phase field in homogeneous tension against the analytical onset."),
        *tables_ms.table2(ctx),
    ]


# ------------------------------------------------------------------ 3.2
def _status_sentence(ctx: Ctx, df, label: str) -> str:
    parts = []
    for p in PROTOCOLS:
        st = df[f"{p}_status"].fillna("not run")
        bad = {s: int(n) for s, n in st.value_counts().items() if s not in ("ok", "limited")}
        if bad:
            parts.append(f"{PROTOCOLS[p]} ("
                         + ", ".join(f"{ctx.count(n, src=OUT, key=f'count({label} {p}_status == {s})')} {s}"
                                     for s, n in bad.items()) + ")")
    if not parts:
        return "Every run of every protocol completed."
    return "Runs that did not complete: " + "; ".join(parts) + "."


def s32(ctx: Ctx) -> list:
    df, core, hold = _sets(ctx)
    paras = []
    # completion, convergence flags
    ok_ivl = _ok(core, "ivl")
    n_uc = int((ok_ivl["ivl_unconverged_steps"].fillna(0) > 0).sum())
    stent_lim = int((core["stent_post_ivl_status"] == "limited").sum() + (core["stent_no_ivl_status"] == "limited").sum())
    paras.append(Para(
        f"The core set comprised {ctx.count(len(core), src=OUT, key='count(set == core)')} lesions and the "
        f"hold-out set {ctx.count(len(hold), src=OUT, key='count(set == holdout)')}. "
        f"{_status_sentence(ctx, core, 'core')} In {_k_of_n(ctx, n_uc, len(ok_ivl), key='core IVL runs with unconverged_steps > 0')} "
        "completed IVL runs of the core set, at least one load step was accepted after the staggered iteration "
        "reached its iteration limit (Section 2.6). "
        + ("No stent run of the core set was pressure-limited." if stent_lim == 0 else
           f"Stent expansion was pressure-limited (status limited) in "
           f"{ctx.count(stent_lim, src=OUT, key='count(core stent_*_status == limited)')} stent runs of the core set.")))
    # outcomes per protocol (core)
    tt = []
    for p in ("physiological", "balloon_only", "balloon_10atm", "ivl", "stent_no_ivl", "stent_post_ivl"):
        t = _ok(core, p)
        k = int((t[f"{p}_fracture"] == 1).sum())
        tt.append(f"{_k_of_n(ctx, k, len(t), key=f'core {p}_fracture')} lesions under {PROTOCOLS[p]}")
    paras.append(Para(
        "Through-thickness calcium damage occurred, in the core set, in " + _and(tt) + " ("
        f"{ctx.fig('fig4_damage_fields')} shows the damage fields of four lesions and {ctx.fig('fig5_outcomes')}a "
        "the damage class of every core lesion). The plain balloon was inflated to a higher pressure than the IVL "
        f"balloon ({_atm_mpa(ctx, 'loads.poba_pressure_atm')} versus {_atm_mpa(ctx, 'loads.ivl_balloon_pressure_atm')}); "
        "the plain-balloon result is compared with a published cadaver observation in Section 3.5."))
    # IVL by design factor
    t = _ok(core, "ivl")
    by = []
    for arc in sorted(t["ca_arc_deg"].unique()):
        s = t[t["ca_arc_deg"] == arc]
        by.append(f"{_k_of_n(ctx, int(s['ivl_fracture'].sum()), len(s), key=f'core ivl_fracture | ca_arc_deg == {arc:g}')} "
                  f"at {ctx.num(float(arc), src=OUT, key='ca_arc_deg level', fmt='g')}°")
    byt = []
    for th in sorted(t["ca_thickness"].unique()):
        s = t[t["ca_thickness"] == th]
        byt.append(f"{_k_of_n(ctx, int(s['ivl_fracture'].sum()), len(s), key=f'core ivl_fracture | ca_thickness == {th:g}')} "
                   f"at {ctx.num(float(th), src=OUT, key='ca_thickness level', fmt='g')} mm")
    bym = []
    for m in MORPH_WORDS:
        s = t[t["morphology"] == m]
        if len(s):
            bym.append(f"{_k_of_n(ctx, int(s['ivl_fracture'].sum()), len(s), key=f'core ivl_fracture | morphology == {m}')} "
                       f"{MORPH_WORDS[m]}")
    paras.append(Para(
        "Under IVL, through-thickness damage occurred in " + _and(by)
        + " of arc. By thickness it occurred in " + _and(byt) + ", and by morphology in " + _and(bym) + "."))
    # damage pattern
    f = t[t["ivl_fracture"] == 1]
    n_diff = int((f["ivl_diffuse"] == 1).sum())
    er = df[df["ivl_erosion_status"] == "ok"] if "ivl_erosion_status" in df else df.iloc[0:0]
    n_er_f = int((er["ivl_erosion_fracture"] == 1).sum())
    med_cracks = float(er.loc[er["ivl_erosion_fracture"] == 1, "ivl_erosion_n_fractures"].median()) if n_er_f else math.nan
    lead = ("The phase-field damage was mostly spread over the deposit rather than concentrated in discrete "
            "cracks: in " if len(f) and n_diff > len(f) / 2 else "In ")
    paras.append(Para(
        lead + f"{_k_of_n(ctx, n_diff, len(f), key='core ivl_diffuse | ivl_fracture')} core lesions with "
        "through-thickness damage under IVL, damage covered at least half of the damaged calcium region (the "
        f"diffuse pattern of Section 2.7; {ctx.fig('fig4_damage_fields')}a). The element-erosion comparator, run on "
        f"{ctx.count(len(er), src=OUT, key='count(ivl_erosion_status == ok)')} lesions, gave through-thickness "
        f"cracks in {ctx.count(n_er_f, src=OUT, key='count(ivl_erosion_fracture == 1)')} of them, as discrete "
        f"cracks (median {ctx.num(med_cracks, src=OUT, key='median(ivl_erosion_n_fractures | ivl_erosion_fracture == 1)', fmt='g')} "
        f"per lesion; {ctx.fig('fig4_damage_fields')}b). The damage after stenting is shown in "
        f"{ctx.fig('fig4_damage_fields')}c."))
    # stent expansion (core)
    both = core[core["stent_no_ivl_status"].isin(("ok", "limited")) & core["stent_post_ivl_status"].isin(("ok", "limited"))]
    a = both["stent_no_ivl_stent_expansion_pct"].astype(float)
    b = both["stent_post_ivl_stent_expansion_pct"].astype(float)
    d = (b - a)
    n_ivl_f = int((both["ivl_fracture"] == 1).sum())
    allx = [float(x) for p in ("stent_no_ivl", "stent_post_ivl") for x in _ok(core, p)[f"{p}_stent_expansion_pct"].dropna()]
    lo_all, hi_all = (min(allx), max(allx)) if allx else (math.nan, math.nan)
    n_both_f = int(((both["ivl_fracture"] == 1) & (both["stent_no_ivl_fracture"] == 1)).sum())
    q = lambda s, x, key: ctx.num(float(s.quantile(x)) if len(s) else math.nan, src=OUT, key=key, fmt=".1f")  # noqa: E731
    paras.append(Para(
        f"In the {ctx.count(len(both), src=OUT, key='count(core, both stent runs ok or limited)')} core lesions with "
        "both stent runs, the median stent expansion was "
        f"{q(a, 0.5, 'median(stent_no_ivl_stent_expansion_pct | both)')}% "
        f"(interquartile range {q(a, 0.25, 'q25(stent_no_ivl_stent_expansion_pct | both)')}–"
        f"{q(a, 0.75, 'q75(stent_no_ivl_stent_expansion_pct | both)')}%) without IVL and "
        f"{q(b, 0.5, 'median(stent_post_ivl_stent_expansion_pct | both)')}% "
        f"({q(b, 0.25, 'q25(stent_post_ivl_stent_expansion_pct | both)')}–"
        f"{q(b, 0.75, 'q75(stent_post_ivl_stent_expansion_pct | both)')}%) after IVL. The paired difference had a "
        f"median of {q(d, 0.5, 'median(post_ivl - no_ivl expansion | both)')} percentage points "
        f"({ctx.fig('fig5_outcomes')}b). Of the "
        f"{ctx.count(n_ivl_f, src=OUT, key='count(both, ivl_fracture == 1)')} of these lesions with "
        "through-thickness damage under IVL, "
        f"{ctx.count(n_both_f, src=OUT, key='count(both, ivl_fracture == 1 and stent_no_ivl_fracture == 1)')} also "
        "had it under stenting without IVL. Across the core stent runs, the expansion ranged from "
        f"{ctx.num(lo_all, src=OUT, key='min(stent_*_stent_expansion_pct | core, ok or limited)', fmt='.1f')}% to "
        f"{ctx.num(hi_all, src=OUT, key='max(stent_*_stent_expansion_pct | core, ok or limited)', fmt='.1f')}%."),
    )
    return [Heading("3.2. Simulated Lesion Outcomes", 2), *paras,
            Figure("fig4_damage_fields",
                   "Calcium damage fields of one lesion per morphology with the same arc and thickness. (a) IVL with "
                   "the phase field: damage. (b) IVL with the element-erosion comparator: eroded calcium cells. (c) "
                   "Stent deployed after IVL: damage, with the lumen outline after stenting and the reference lumen. "
                   "Deformed configuration, final state of each run."),
            Figure("fig5_outcomes",
                   "Outcomes of the core factorial. (a) Damage class of each lesion for each damage protocol "
                   "(rows) and morphology (columns), on a grid of calcium arc by thickness: through-thickness damage, "
                   "partial damage only, no damage onset, or not run; IVL cells give the damage-onset load. (b) "
                   "Stent expansion without IVL and after IVL for each lesion; open markers: pressure-limited "
                   "expansion.")]


# ------------------------------------------------------------------ 3.3
def s33(ctx: Ctx) -> list:
    rank = ctx.opt(MS, f"{PRIMARY}.univariable_ranking") or []
    mech = [r["predictor"] for r in rank if r["group"] == "mechanics"]
    geo = [r["predictor"] for r in rank if r["group"] == "geometry"]
    top = mech[:3]
    U = f"{PRIMARY}.univariable"
    auc = lambda p, inner=True: _vci(ctx, MS, f"{U}.{p}.auc", f"{U}.{p}.auc_ci",  # noqa: E731
                                     degenerate_key=f"{U}.{p}.auc_ci_degenerate", inner=inner)
    best = ctx.opt(MS, f"{PRIMARY}.fujino_comparison.best_single_mechanics")
    fj = f"{PRIMARY}.fujino_comparison.fujino_score"
    comp = f"{PRIMARY}.composite"
    sel = ctx.opt(MS, f"{comp}.selected") or []
    conflicts = ctx.opt(MS, f"{comp}.coefficient_sign_opposes_direction") or []
    freq = ctx.opt(MS, f"{comp}.core.loocv_selection_frequency") or {}
    name = ctx.opt(MS, f"{comp}.name") or "the composite index"
    dcore = f"{fj}.auc_differences.core"
    lift_fj = f"{dcore}.{name} (LOOCV) - fujino_score"
    best_fj = f"{dcore}.{best} - fujino_score"
    lift_best = f"{dcore}.{name} (LOOCV) - {best}"
    inc = f"{PRIMARY}.incremental_value"
    incl = f"{inc}.core.loocv"
    inca = f"{inc}.core.apparent_refit_bootstrap"
    g = lambda k: ctx.opt(MS, k)  # noqa: E731

    paras = [Heading("3.3. Prediction of Through-Thickness Damage under IVL", 2)]
    paras.append(Para(
        f"Of the {ctx.v(MS, f'{PRIMARY}.core.n', fmt='d')} core lesions with a completed IVL run, "
        f"{ctx.v(MS, f'{PRIMARY}.core.n_pos', fmt='d')} had through-thickness damage; of the "
        f"{ctx.v(MS, f'{PRIMARY}.holdout.n', fmt='d')} hold-out lesions, "
        f"{ctx.v(MS, f'{PRIMARY}.holdout.n_pos', fmt='d')} had. AUC intervals are bootstrap percentile "
        "intervals. The univariable results "
        f"are listed in {ctx.tab('table3_predictors')} and shown in {ctx.fig('fig6_prediction')}."))
    if top:
        paras.append(Para(
            "On the core set, the mechanics predictors with the largest AUC were "
            + _and([f"{pname(ctx, p)} (AUC {auc(p)})" for p in top])
            + ". Among the geometric descriptors, the largest AUC was that of "
            + (f"{pname(ctx, geo[0])} (AUC {auc(geo[0])})" if geo else Ctx.PENDING)
            + f", and the Fujino score had an AUC of {auc('fujino_score', inner=False)}. With the threshold re-derived in "
            f"each leave-one-out fold, {pname(ctx, best)} classified the core lesions with a sensitivity of "
            f"{ctx.v(MS, f'{U}.{best}.loocv_sens', fmt='.2f')} and a specificity of "
            f"{ctx.v(MS, f'{U}.{best}.loocv_spec', fmt='.2f')}; the Fujino score had "
            f"{ctx.v(MS, f'{U}.fujino_score.loocv_sens', fmt='.2f')} and "
            f"{ctx.v(MS, f'{U}.fujino_score.loocv_spec', fmt='.2f')}."))
    if sel:
        conf_txt = ("No selected predictor had a coefficient opposite to its stated direction." if not conflicts else
                    "The coefficient of " + _and([pname(ctx, c["predictor"]) for c in conflicts])
                    + " had the opposite sign to its stated direction.")
        always = [p for p in sel if freq.get(p) == 1.0]
        freq_txt = ("These predictors were selected in every leave-one-out fold." if len(always) == len(sel) else
                    "Across the leave-one-out folds, the selection frequencies were "
                    + _and([f"{ctx.v(MS, f'{comp}.core.loocv_selection_frequency.{p}', fmt='.0f', scale=100.0)}% for "
                            f"{pname(ctx, p)}" for p in freq]) + ".")
        paras.append(Para(
            f"The rule of Section 2.10 selected {_and([pname(ctx, p) for p in sel])} for {name} "
            f"({ctx.tab('table4_models')}). {freq_txt} "
            f"{conf_txt} {name} had an apparent AUC of {ctx.v(MS, f'{comp}.core.apparent_auc', fmt='.2f')} and a "
            f"leave-one-out AUC of {_vci(ctx, MS, f'{comp}.core.loocv_auc', f'{comp}.core.loocv_auc_ci')}, with a "
            f"leave-one-out log-loss of {ctx.v(MS, f'{comp}.core.loocv_log_loss', fmt='.3f')} and Brier score of "
            f"{ctx.v(MS, f'{comp}.core.loocv_brier', fmt='.3f')}. Its leave-one-out AUC was "
            + _diff_word(*(g(f"{lift_best}.delta_ci_{s}") for s in ("lo", "hi")),
                         "higher than", "lower than", "not different from")
            + f" that of {pname(ctx, best)} (difference "
            f"{_vci(ctx, MS, f'{lift_best}.delta_auc', (f'{lift_best}.delta_ci_lo', f'{lift_best}.delta_ci_hi'), inner=True)})."))
    paras.append(Para(
        f"Compared with the Fujino score, the AUC of {pname(ctx, best)} was "
        + _diff_word(g(f"{best_fj}.delta_ci_lo"), g(f"{best_fj}.delta_ci_hi"), "higher", "lower",
                     "not different (the interval includes zero)")
        + f" (difference {_vci(ctx, MS, f'{best_fj}.delta_auc', (f'{best_fj}.delta_ci_lo', f'{best_fj}.delta_ci_hi'), inner=True)}), "
        f"and so was the leave-one-out AUC of {name} "
        f"(difference {_vci(ctx, MS, f'{lift_fj}.delta_auc', (f'{lift_fj}.delta_ci_lo', f'{lift_fj}.delta_ci_hi'), inner=True)}). "
        "The geometry-only model had a leave-one-out AUC of "
        f"{ctx.v(MS, f'{incl}.auc_b', fmt='.2f')}, and adding the selected mechanics predictors gave "
        f"{ctx.v(MS, f'{incl}.auc_a', fmt='.2f')} (difference "
        f"{_vci(ctx, MS, f'{incl}.delta_auc', (f'{incl}.delta_ci_lo', f'{incl}.delta_ci_hi'), inner=True)}; "
        f"{ctx.fig('fig6_prediction')}d). The apparent difference with both models refitted in every resample was "
        f"{_vci(ctx, MS, f'{inca}.delta_auc', (f'{inca}.delta_ci_lo', f'{inca}.delta_ci_hi'))}. "
        + _incremental_sentence(ctx, incl)))
    paras.append(_holdout_para(ctx, best, name))
    paras.append(Figure(
        "fig6_prediction",
        "Prediction of through-thickness calcium damage under IVL. (a) Core set: ROC curves of the best single "
        "mechanics predictor and the Fujino score (fixed scores) and of the composite index and the geometry and "
        "geometry + mechanics models (leave-one-out predictions). (b) Hold-out set: the same scores and the "
        "core-fit models, unchanged. (c) AUC with its bootstrap interval for every candidate on the core set "
        "(filled) and on the hold-out set (open). (d) Difference in AUC, geometry + mechanics minus geometry, with "
        "its paired bootstrap interval."))
    paras.append(tables_ms.table3(ctx))
    paras += tables_ms.table4(ctx)
    return paras


def _incremental_sentence(ctx: Ctx, incl: str) -> str:
    lo, hi = ctx.opt(MS, f"{incl}.delta_ci_lo"), ctx.opt(MS, f"{incl}.delta_ci_hi")
    llo, lhi = ctx.opt(MS, f"{incl}.delta_log_loss_ci_lo"), ctx.opt(MS, f"{incl}.delta_log_loss_ci_hi")
    blo, bhi = ctx.opt(MS, f"{incl}.delta_brier_ci_lo"), ctx.opt(MS, f"{incl}.delta_brier_ci_hi")
    if any(_missing(x) for x in (lo, hi, llo, lhi, blo, bhi)):
        return Ctx.PENDING
    s = []
    s.append("The interval of the AUC difference includes zero, so the AUC gives no evidence of incremental value"
             if lo <= 0 <= hi else ("The AUC difference favours the model with mechanics" if lo > 0
                                    else "The AUC difference favours the geometry-only model"))
    better_ll, better_b = lhi < 0, bhi < 0
    worse_ll, worse_b = llo > 0, blo > 0
    ll = (f"log-loss difference {_vci(ctx, MS, f'{incl}.delta_log_loss', (f'{incl}.delta_log_loss_ci_lo', f'{incl}.delta_log_loss_ci_hi'), fmt='.3f', inner=True)}; "
          f"Brier score difference {_vci(ctx, MS, f'{incl}.delta_brier', (f'{incl}.delta_brier_ci_lo', f'{incl}.delta_brier_ci_hi'), fmt='.3f', inner=True)}")
    if better_ll and better_b:
        s.append(f"whereas the out-of-fold log-loss and Brier score were lower with mechanics ({ll})")
    elif worse_ll and worse_b:
        s.append(f"and the out-of-fold log-loss and Brier score were higher with mechanics ({ll})")
    else:
        s.append(f"and the out-of-fold log-loss and Brier score differences were mixed or included zero ({ll})")
    return "; ".join(s) + "."


def _holdout_para(ctx: Ctx, best: str, name: str) -> Para:
    U = f"{PRIMARY}.univariable"
    comp = f"{PRIMARY}.composite"
    n = ctx.opt(MS, f"{PRIMARY}.holdout.n") or 0
    if n == 0 and not ctx.draft:
        raise BuildError("hold-out results missing")
    inc = f"{PRIMARY}.incremental_value.holdout"
    return Para(
        "On the hold-out set, with the core thresholds and models unchanged, "
        f"{pname(ctx, best)} had an AUC of {_vci(ctx, MS, f'{U}.{best}.holdout.auc', (f'{U}.{best}.holdout.auc_ci_lo', f'{U}.{best}.holdout.auc_ci_hi'))} "
        f"and classified the lesions with a sensitivity of {ctx.v(MS, f'{U}.{best}.holdout.sens', fmt='.2f')} and a "
        f"specificity of {ctx.v(MS, f'{U}.{best}.holdout.spec', fmt='.2f')}. {name} had an AUC of "
        f"{_vci(ctx, MS, f'{comp}.holdout.auc', (f'{comp}.holdout.auc_ci_lo', f'{comp}.holdout.auc_ci_hi'))}, "
        f"sensitivity {ctx.v(MS, f'{comp}.holdout.sens', fmt='.2f')} and specificity "
        f"{ctx.v(MS, f'{comp}.holdout.spec', fmt='.2f')}, and the Fujino score had an AUC of "
        f"{_vci(ctx, MS, f'{U}.fujino_score.holdout.auc', (f'{U}.fujino_score.holdout.auc_ci_lo', f'{U}.fujino_score.holdout.auc_ci_hi'))}. "
        "The core-fit geometry + mechanics model minus the geometry-only model had a hold-out AUC difference of "
        f"{_vci(ctx, MS, f'{inc}.delta_auc', (f'{inc}.delta_ci_lo', f'{inc}.delta_ci_hi'))} "
        f"({ctx.fig('fig6_prediction')}b).")


# ------------------------------------------------------------------ 3.4
def s34(ctx: Ctx) -> list:
    on = "outcomes.ivl_onset"
    rank = ctx.opt(MS, f"{on}.ranking") or []
    top = [r["predictor"] for r in rank if r["group"] == "mechanics"][:2]
    c = lambda p: _vci(ctx, MS, f"{on}.predictors.{p}.c_index", (f"{on}.predictors.{p}.c_index_ci_lo",  # noqa: E731
                                                                  f"{on}.predictors.{p}.c_index_ci_hi"), inner=True)
    b10 = "outcomes.balloon_10atm_fracture"
    rb = ctx.opt(MS, f"{b10}.univariable_ranking") or []
    tb = [r["predictor"] for r in rb if r["group"] == "mechanics"][:1]
    ero = "outcomes.ivl_erosion_fracture"
    re_ = ctx.opt(MS, f"{ero}.univariable_ranking") or []
    ero_txt = ("The erosion subset had too few lesions in one outcome class for a univariable analysis."
               if not re_ else
               "Against through-thickness cracks of the element-erosion comparator, the largest AUC was that of "
               f"{pname(ctx, re_[0]['predictor'])} "
               f"(AUC {_vci(ctx, MS, f'{ero}.univariable.{re_[0]['predictor']}.auc', f'{ero}.univariable.{re_[0]['predictor']}.auc_ci', inner=True)}).")
    st = "outcomes.stent_post_ivl_expansion"
    srank = ctx.opt(MS, f"{st}.ranking") or []
    s_top = srank[0]["predictor"] if srank else None
    paras = [Heading("3.4. Secondary Outcomes", 2)]
    paras.append(Para(
        f"Damage onset under IVL was recorded in {ctx.v(MS, f'{on}.core.n_events', fmt='d')} of "
        f"{ctx.v(MS, f'{on}.core.n', fmt='d')} core lesions; the others were censored at their peak equivalent IVL "
        "pressure. The concordance of the onset load with the oriented predictors was highest for "
        + (_and([f"{pname(ctx, p)} (C {c(p)})" for p in top]) if top else Ctx.PENDING)
        + ". Under the plain balloon, the largest AUC among the mechanics predictors was that of "
        + (f"{pname(ctx, tb[0])} (AUC {_vci(ctx, MS, f'{b10}.univariable.{tb[0]}.auc', f'{b10}.univariable.{tb[0]}.auc_ci', inner=True)})"
           if tb else Ctx.PENDING)
        + f". {ero_txt}"))
    if s_top:
        sk = f"{st}.predictors.{s_top}"
        sign = ctx.opt(MS, f"{sk}.observed_sign")
        paras.append(Para(
            "In the exploratory analysis of stent expansion after IVL, the strongest rank association was that of "
            f"{pname(ctx, s_top)} (Spearman correlation "
            f"{_vci(ctx, MS, f'{sk}.spearman', (f'{sk}.spearman_ci_lo', f'{sk}.spearman_ci_hi'), inner=True)}), "
            + ("larger values going with larger expansion" if sign == 1 else "larger values going with smaller expansion")
            + "."))
    return paras


# ------------------------------------------------------------------ 3.5
def s35(ctx: Ctx) -> list:
    cons = ctx.data(CONS)
    core_checks = [c for c in cons["checks"] if c["set"] == "core"]
    labels = ("agrees", "partial", "disagrees", "not computed")
    words = {"agrees": "agreed", "partial": "agreed partially", "disagrees": "disagreed", "not computed": "could not be computed"}
    counts = _and([f"{ctx.v(CONS, f'agreement_counts.core.{k}', fmt='d')} {words[k]}"
                   for k in labels if cons["agreement_counts"]["core"].get(k)])
    ids = lambda lab: _and([c["id"] for c in core_checks if c["agreement"] == lab])  # noqa: E731
    by_id = {c["id"]: c for c in core_checks}
    i4 = next(i for i, c in enumerate(cons["checks"]) if c["id"] == "C4" and c["set"] == "core")
    i5 = next(i for i, c in enumerate(cons["checks"]) if c["id"] == "C5" and c["set"] == "core")
    paras = [Heading("3.5. Consistency with Published Observations", 2)]
    paras.append(Para(
        f"Of the checks against published observations and model-internal criteria on the core set "
        f"({ctx.tab('table2_verification_consistency')}), {counts}. The checks that agreed were {ids('agrees')}; "
        f"partial agreement (point estimate on the expected side, interval including the null value) was found for "
        f"{ids('partial')}; and {ids('disagrees')} disagreed."))
    from dtivl import params as P
    c4, c5 = cons["checks"][i4], cons["checks"][i5]
    pub = sorted(P.lit(r).value for r in ("fracture_rate_DisruptCAD_III_OCT", "fracture_rate_DisruptCAD_II_OCT"))
    rel4 = ("below" if c4["model_value"] < pub[0] else "above" if c4["model_value"] > pub[-1] else "within the range of")
    why5 = ("This disagreement follows from the calibration of the calcium modulus (decision D8), which was chosen so "
            "that the IVL balloon alone does not fracture calcium, at a lower pressure than the plain balloon."
            if c5["agreement"] == "disagrees" else "")
    paras.append(Para(
        "The proportion of core lesions with through-thickness damage under IVL was "
        f"{_vci(ctx, CONS, f'checks.{i4}.model_value', (f'checks.{i4}.ci_lo', f'checks.{i4}.ci_hi'), fmt='.1f', unit='%')}, "
        f"{rel4} the lesion-level fracture rates of "
        f"{ctx.lit('fracture_rate_DisruptCAD_III_OCT', fmt='g')}% and {ctx.lit('fracture_rate_DisruptCAD_II_OCT', fmt='g')}% "
        f"seen by OCT in the Disrupt CAD III and II studies {ctx.cite('hill2020disruptcad3', 'ali2019disruptcad2')} "
        "(check C4). The core set is a designed factorial, not a clinical case mix, so this proportion follows from "
        "the design. The plain balloon produced through-thickness damage in "
        f"{_vci(ctx, CONS, f'checks.{i5}.model_value', (f'checks.{i5}.ci_lo', f'checks.{i5}.ci_hi'), fmt='.1f', unit='%')} "
        "of core lesions, whereas no lesion fractured under a plain balloon in human cadaver coronary arteries "
        f"examined by histology {ctx.cite('kawai2023cadaver')} (check C5). {why5}"))
    paras += _stent_and_pattern_checks(ctx, cons)
    if by_id.get("C6", {}).get("agreement") == "agrees":
        paras.append(Para(
            "The IVL balloon inflated without pulses produced no through-thickness damage in any core lesion (check "
            "C6). This model-internal criterion was used for the calibration on the pilot lesions, three of which "
            "share the geometry of core lesions, so it is not independent of the calibration."))
    paras += tables_ms.table2b(ctx)
    return paras


def _check_index(cons: dict, cid: str) -> int:
    return next(i for i, c in enumerate(cons["checks"]) if c["id"] == cid and c["set"] == "core")


def _stent_and_pattern_checks(ctx: Ctx, cons: dict) -> list:
    """Checks C7 (stent expansion), C10 (fracture count and depth) and C11 (Fujino score and expansion)."""
    out = []
    i7, i10, i11 = (_check_index(cons, c) for c in ("C7", "C10", "C11"))
    q7 = f"checks.{i7}.quantities.0"
    c7 = cons["checks"][i7]
    if c7["agreement"] != "not computed":
        q = cons["checks"][i7]["quantities"][0]
        pr = next(j for j, r in enumerate(c7["published_rows"]) if r["row"] == "stent_expansion_maxCa_site_DisruptCAD_III")
        sd_pub = c7["published_rows"][pr]["sd"]
        spread = (" Its spread was much narrower than the clinical one (standard deviation "
                  f"{ctx.v(CONS, f'{q7}.sd', fmt='.1f')} versus "
                  f"{ctx.v(CONS, f'checks.{i7}.published_rows.{pr}.sd', fmt='g')} percentage points)."
                  if q["sd"] < sd_pub / 5 else "")
        word = {"agrees": "lay within", "partial": "lay partly within", "disagrees": "lay outside"}[c7["agreement"]]
        out.append(Para(
            f"The mean stent expansion after IVL ({ctx.v(CONS, f'{q7}.mean', fmt='.1f')}%, range "
            f"{ctx.v(CONS, f'{q7}.min', fmt='.1f')}–{ctx.v(CONS, f'{q7}.max', fmt='.1f')}%) {word} one standard "
            "deviation of the expansion reported at the site of maximum calcium "
            f"({ctx.lit('stent_expansion_maxCa_site_DisruptCAD_III', fmt='g')}%) and at the minimum stent area "
            f"({ctx.lit('stent_expansion_MSA_site_DisruptCAD_III', fmt='g')}%) in Disrupt CAD III "
            f"{ctx.cite('hill2020disruptcad3')} (check C7).{spread}"))
    c10 = cons["checks"][i10]
    if c10["agreement"] != "not computed":
        q = c10["quantities"]
        single = q[0]["sd"] == 0 and q[0]["mean"] == 1
        q10 = f"checks.{i10}.quantities"
        model = ("every fractured core lesion had a single through-thickness damage region" if single else
                 f"fractured core lesions had a mean of {ctx.v(CONS, f'{q10}.0.mean', fmt='.1f')} through-thickness "
                 "damage regions")
        out.append(Para(
            f"Under IVL, {model}, and {ctx.v(CONS, f'{q10}.1.value', fmt='.0f')}% of the damage regions were full "
            "thickness, whereas a pooled OCT analysis of the Disrupt CAD studies reported a mean of "
            f"{ctx.lit('fractures_per_lesion_pooled_IVL_OCT', fmt='g')} fractures per lesion, "
            f"{ctx.lit('full_thickness_fracture_fraction_pooled_IVL_OCT', fmt='g')}% of them full thickness "
            f"{ctx.cite('tct120pooledoct2021')} (check C10, {c10['agreement'].replace('agrees', 'agreement').replace('partial', 'partial agreement').replace('disagrees', 'disagreement')}). "
            "The phase field spreads damage over the deposit, so it does not resolve separate cracks "
            "(Section 3.2)."))
    c11 = cons["checks"][i11]
    if c11["agreement"] != "not computed":
        q11 = f"checks.{i11}.quantities.0"
        qv = c11["quantities"][0]
        assoc = ("a negative association" if qv["ci_hi"] < 0 else
                 "a positive association, opposite to the clinical one," if qv["ci_lo"] > 0 else
                 "no clear association")
        out.append(Para(
            f"Stent expansion without IVL showed {assoc} with the cross-sectional Fujino score (Spearman correlation "
            f"{_vci(ctx, CONS, f'{q11}.value', (f'{q11}.ci_lo', f'{q11}.ci_hi'), inner=True)}; check C11). "
            "Clinically, lesions with a score of $4$ had a mean stent expansion of "
            f"{ctx.lit('fujino_score4_stent_expansion', fmt='g')}%, against "
            f"{ctx.lit('fujino_score0to3_stent_expansion', fmt='g')}% for lower scores "
            f"{ctx.cite('fujino2018score')}. The highest score requires a lesion-length criterion that a single "
            "cross-section cannot meet, and the model expansion varied little between lesions."))
    return out


# ------------------------------------------------------------------ 3.6
def s36(ctx: Ctx) -> list:
    from build_manuscript import ROOT
    have = all((ROOT / f).exists() for f in ("results/sensitivity_oat.json", "results/sensitivity_morris.json",
                                             "results/sensitivity_sobol.json"))
    if not have and not ctx.draft:
        raise BuildError("sensitivity results missing")
    return [Heading("3.6. Sensitivity and Uncertainty", 2),
            Para(f"{Ctx.PENDING} One-at-a-time, Morris and Sobol results; three-dimensional length series.")
            if not have else Para(Ctx.PENDING)]


def blocks(ctx: Ctx) -> list:
    return [*s31(ctx), *s32(ctx), *s33(ctx), *s34(ctx), *s35(ctx), *s36(ctx)]
