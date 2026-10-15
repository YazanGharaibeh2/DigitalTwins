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
    uc = ok_ivl["ivl_unconverged_steps"].fillna(0) > 0
    pos = ok_ivl["ivl_fracture"] == 1
    n_uc_pos, n_uc_neg = int((uc & pos).sum()), int((uc & ~pos).sum())
    stent_lim = int((core["stent_post_ivl_status"] == "limited").sum() + (core["stent_no_ivl_status"] == "limited").sum())
    paras.append(Para(
        f"The core set comprised {ctx.count(len(core), src=OUT, key='count(set == core)')} lesions and the "
        f"hold-out set {ctx.count(len(hold), src=OUT, key='count(set == holdout)')}. "
        f"{_status_sentence(ctx, core, 'core')} In "
        f"{_k_of_n(ctx, n_uc_pos, int(pos.sum()), key='core IVL runs with unconverged_steps > 0 | ivl_fracture')} "
        "completed IVL runs of the core set with through-thickness damage and in "
        f"{_k_of_n(ctx, n_uc_neg, int((~pos).sum()), key='core IVL runs with unconverged_steps > 0 | no ivl_fracture')} "
        "without it, at least one load step was accepted after the staggered iteration reached its iteration limit "
        "(Section 2.6). "
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
    paras.append(_ivl_load_para(ctx, core))
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
    nf = t[t["ivl_fracture"] == 0]
    dmin = float(f["ivl_damaged_fraction"].min()) if len(f) else math.nan
    n_whole = int((f["ivl_damaged_fraction"] >= 1.0).sum())
    n_none = int((nf["ivl_damaged_fraction"] <= 0.0).sum())
    er = df[df["ivl_erosion_status"] == "ok"] if "ivl_erosion_status" in df else df.iloc[0:0]
    n_er_f = int((er["ivl_erosion_fracture"] == 1).sum())
    med_cracks = float(er.loc[er["ivl_erosion_fracture"] == 1, "ivl_erosion_n_fractures"].median()) if n_er_f else math.nan
    all_none = len(f) > 0 and n_whole == len(f) and n_none == len(nf)
    paras.append(Para(
        ("The phase-field outcome was all or nothing. " if all_none else "")
        + f"In {_k_of_n(ctx, n_whole, len(f), key='core ivl_damaged_fraction == 1 | ivl_fracture')} core lesions "
        "with through-thickness damage under IVL, the damage covered the whole calcium (the smallest damaged "
        f"fraction was {ctx.num(100 * dmin, src=OUT, key='min(ivl_damaged_fraction | ivl_fracture) x 100', fmt='.0f')}%), "
        f"and in {_k_of_n(ctx, n_none, len(nf), key='core ivl_damaged_fraction == 0 | no ivl_fracture')} lesions "
        f"without it no calcium cell was damaged ({ctx.fig('fig4_damage_fields')}a)."
        + (" Through-thickness damage was therefore equivalent to damage onset before the peak IVL load."
           if all_none else "")
        + " The element-erosion comparator, run on "
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
        f"{ctx.num(hi_all, src=OUT, key='max(stent_*_stent_expansion_pct | core, ok or limited)', fmt='.1f')}%. "
        + _stent_target_sentence(ctx, core)),
    )
    return [Heading("3.2. Simulated Lesion Outcomes", 2), *paras,
            Figure("fig4_damage_fields",
                   "Calcium damage fields of one lesion per morphology with the same arc and thickness. (a) IVL with "
                   "the phase field: damage. (b) IVL with the element-erosion comparator: eroded calcium cells"
                   + _erosion_extra_note() + ". (c) "
                   "Stent deployed after IVL: damage, with the lumen outline after stenting and the reference lumen. "
                   "Deformed configuration, final state of each run."),
            Figure("fig5_outcomes",
                   "Outcomes of the core factorial. (a) Damage class of each lesion for each damage protocol "
                   "(rows) and morphology (columns), on a grid of calcium arc by thickness: through-thickness damage, "
                   "partial damage only, no damage onset, or not run; IVL cells give the damage-onset load. (b) "
                   "Stent expansion without IVL and after IVL for each lesion; open markers: pressure-limited "
                   "expansion.")]


def _ivl_load_para(ctx: Ctx, core) -> Para:
    """The equivalent IVL load against the plain-balloon pressure, and the overlap of their outcomes."""
    from dtivl import params as P
    t = core[(core["ivl_status"] == "ok") & (core["balloon_10atm_status"] == "ok")]
    p_poba = P.get("loads.poba_pressure_atm").value
    below = int((t["ivl_p_top"] < p_poba).sum())
    iv = t["ivl_fracture"] == 1
    pb = t["balloon_10atm_fracture"] == 1
    lum0 = P.get("geometry.lumen_area_mm2").value
    bo = _ok(core, "balloon_only")
    return Para(
        "In the nonlinear model, IVL acted as a quasi-static inflation to the peak equivalent pressure, which ranged "
        f"from {ctx.num(float(t['ivl_p_top'].min()), src=OUT, key='min(ivl_p_top | core)', fmt='.2f')} to "
        f"{ctx.num(float(t['ivl_p_top'].max()), src=OUT, key='max(ivl_p_top | core)', fmt='.2f')} MPa and was below the "
        f"plain-balloon pressure in {_k_of_n(ctx, below, len(t), key='count(ivl_p_top < poba pressure)')} lesions. "
        f"Of the {ctx.count(int(iv.sum()), src=OUT, key='count(ivl_fracture | both ok)')} lesions damaged under IVL, "
        f"{ctx.count(int((iv & pb).sum()), src=OUT, key='count(ivl_fracture and balloon_10atm_fracture)')} were also "
        "damaged under the plain balloon, which damaged "
        f"{ctx.count(int((pb & ~iv).sum()), src=OUT, key='count(balloon_10atm_fracture and not ivl_fracture)')} "
        f"further lesion{'' if int((pb & ~iv).sum()) == 1 else 's'}"
        + ("; in this model the two protocols therefore differed only in load magnitude. "
           if int((iv & ~pb).sum()) == 0 else ". ")
        + "The lesions "
        "did not resist dilation: under the IVL balloon pressure alone, the lumen area grew from "
        f"{ctx.param('geometry.lumen_area_mm2', fmt='.2g')} mm² to "
        f"{ctx.num(float(bo['balloon_only_max_lumen_area'].min()), src=OUT, key='min(balloon_only_max_lumen_area)', fmt='.1f')}–"
        f"{ctx.num(float(bo['balloon_only_max_lumen_area'].max()), src=OUT, key='max(balloon_only_max_lumen_area)', fmt='.1f')} mm².")


def _stent_target_sentence(ctx: Ctx, core) -> str:
    vals = [v for p in ("stent_no_ivl", "stent_post_ivl") for v in _ok(core, p)[f"{p}_reached_target"].tolist()]
    n_ok = sum(bool(v) for v in vals if v == v)
    if vals and n_ok == len(vals):
        return ("Every stent reached its target diameter, so the expansion differed between lesions only through "
                "the elastic recoil after unloading.")
    return (f"The stent reached its target diameter in {_k_of_n(ctx, n_ok, len(vals), key='count(stent_*_reached_target)')} "
            "runs.")


def _erosion_extra_note() -> str:
    """Morphologies of the figure lesions whose erosion runs are outside the comparator subset (figure-only)."""
    from dtivl import params as P
    sub = P.raw("design.erosion_subset")
    morph = set(sub.get("morphology", [])) if isinstance(sub, dict) else set()
    extra = [MORPH_WORDS.get(m, m) for m in ("superficial_sheet", "deep_sheet", "nodule", "fragmented")
             if morph and m not in morph]
    return (f"; the {_and(extra)} runs were made for this figure only and are outside the comparator subset"
            if extra else "")


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
                         "higher than", "lower than", "not clearly different from")
            + f" that of {pname(ctx, best)} (difference "
            f"{_vci(ctx, MS, f'{lift_best}.delta_auc', (f'{lift_best}.delta_ci_lo', f'{lift_best}.delta_ci_hi'), inner=True)})."))
    paras.append(Para(
        f"Compared with the Fujino score, the AUC of {pname(ctx, best)} was "
        + _diff_word(g(f"{best_fj}.delta_ci_lo"), g(f"{best_fj}.delta_ci_hi"), "higher", "lower",
                     "not clearly different (the interval includes zero)")
        + f" (difference {_vci(ctx, MS, f'{best_fj}.delta_auc', (f'{best_fj}.delta_ci_lo', f'{best_fj}.delta_ci_hi'), inner=True)}), "
        f"and the leave-one-out AUC of {name} was "
        + _diff_word(g(f"{lift_fj}.delta_ci_lo"), g(f"{lift_fj}.delta_ci_hi"), "higher", "lower", "not clearly different")
        + f" (difference {_vci(ctx, MS, f'{lift_fj}.delta_auc', (f'{lift_fj}.delta_ci_lo', f'{lift_fj}.delta_ci_hi'), inner=True)}). "
        "The geometry-only model had a leave-one-out AUC of "
        f"{ctx.v(MS, f'{incl}.auc_b', fmt='.2f')}, and adding the selected mechanics predictors gave "
        f"{ctx.v(MS, f'{incl}.auc_a', fmt='.2f')} (difference "
        f"{_vci(ctx, MS, f'{incl}.delta_auc', (f'{incl}.delta_ci_lo', f'{incl}.delta_ci_hi'), inner=True)}; "
        f"{ctx.fig('fig6_prediction')}d). The apparent difference with both models refitted in every resample was "
        f"{_vci(ctx, MS, f'{inca}.delta_auc', (f'{inca}.delta_ci_lo', f'{inca}.delta_ci_hi'))}. "
        + _incremental_sentence(ctx, incl)))
    paras.append(_fujino_artefact_para(ctx, best))
    paras.append(_selection_para(ctx, best, rank))
    paras.append(_benchmark_para(ctx, best))
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


def _fujino_artefact_para(ctx: Ctx, best: str) -> Para:
    """Post-hoc: the 180 deg design level measures just below the strict arc cut-off of the Fujino score."""
    ph = f"{PRIMARY}.post_hoc.fujino_arc_tolerance.core"
    d = f"{ph}.{best} - fujino_arc_tolerance"
    g = lambda k: ctx.opt(MS, k)  # noqa: E731
    if g(f"{ph}.auc") is None:
        return Para(f"{Ctx.PENDING} (post-hoc Fujino sensitivity)")
    return Para(
        f"The arc criterion of the Fujino score awards points above {ctx.v(MS, f'{ph}.arc_cutoff_deg', fmt='g')}°. "
        f"The {ctx.v(MS, f'{ph}.n_arc_just_below_cutoff', fmt='d')} core lesions of the "
        f"{ctx.v(MS, f'{ph}.arc_cutoff_deg', fmt='g')}° design level measured just below it on the label map and "
        f"received no arc points; {_n_below_damaged(ctx)} were damaged under IVL. In a post hoc sensitivity "
        "analysis "
        "added after the first draft, an arc within "
        f"{ctx.v(MS, f'{ph}.arc_tolerance_deg', fmt='g')}° below the cut-off was counted as meeting it. The Fujino "
        f"score then had an AUC of {_vci(ctx, MS, f'{ph}.auc', f'{ph}.auc_ci')}, and its difference from "
        f"{pname(ctx, best)} was {_vci(ctx, MS, f'{d}.delta_auc', (f'{d}.delta_ci_lo', f'{d}.delta_ci_hi'))}; the "
        "advantage of the mechanics predictor over the Fujino score therefore "
        + ("depends on where the design levels fall relative to the score's cut-offs."
           if (g(f"{d}.delta_ci_lo") or 0) <= 0 else "persisted."))


def _n_below_damaged(ctx: Ctx) -> str:
    """'k of n': core lesions with IVL outcome whose measured arc lies within the tolerance below the cut-off."""
    from dtivl.drivers.stage_metrics import FUJINO_ARC_TOL_DEG
    from dtivl.geometry.fujino import fujino_criteria
    thr = fujino_criteria()["angle"].threshold
    t = _ok(ctx.data(OUT)[ctx.data(OUT)["set"] == "core"], "ivl")
    arc = t["ca_arc_total_deg"].astype(float)
    sel = t[(arc <= thr) & (arc + FUJINO_ARC_TOL_DEG >= thr)]
    return _k_of_n(ctx, int((sel["ivl_fracture"] == 1).sum()), len(sel), key="core ivl_fracture | arc just below cut-off")


def _benchmark_para(ctx: Ctx, best: str) -> Para:
    """Post hoc (D18): outcome-criterion benchmark and penalty sensitivity of the logistic models."""
    g = lambda k: ctx.opt(MS, k)  # noqa: E731
    b = f"{PRIMARY}.post_hoc.outcome_criterion_benchmark.core"
    ps = f"{PRIMARY}.post_hoc.penalty_sensitivity"
    if g(f"{b}.auc") is None:
        return Para(f"{Ctx.PENDING} (post-hoc benchmark)")
    sens = g(f"{b}.at_index_1.sens")
    txt = ("Two further analyses were added after the first draft. In an outcome-criterion benchmark, "
           f"{pname(ctx, best)} was scaled to each lesion's peak equivalent IVL load and divided by the failure stress "
           "of the outcome model; this linear failure index uses the IVL load, an input of the outcome model, and "
           f"is a benchmark, not a predictor. It ranked the core lesions with an AUC of {_vci(ctx, MS, f'{b}.auc', f'{b}.auc_ci')}. "
           "Read as a failure criterion (index of one or more), it had a sensitivity of "
           f"{ctx.v(MS, f'{b}.at_index_1.sens', fmt='.2f')} and a specificity of "
           f"{ctx.v(MS, f'{b}.at_index_1.spec', fmt='.2f')}"
           + (": damage onset in the nonlinear model occurred below the linear failure load in most damaged lesions."
              if sens is not None and sens < 0.5 else "."))
    models = g(f"{ps}.models") or {}
    Cs = [c for c in (models.get("geometry") or {})]
    if models and Cs:
        ll = lambda m, c: (models.get(m) or {}).get(c, {}).get("loocv_log_loss")  # noqa: E731
        lower_all = all((ll("geometry+mechanics", c) or 9) < (ll("geometry", c) or 0) for c in Cs)
        flex_better = any(((models.get("flexible geometry") or {}).get(c, {}).get("loocv_auc") or 0)
                          > ((models.get("geometry") or {}).get(c, {}).get("loocv_auc") or 1) + 0.02 for c in Cs)
        cs_txt = _and([ctx.num(float(c), src=MS, key=f"{ps}.C", fmt="g") for c in Cs])
        txt += (f" With the inverse penalty of the logistic models set to {cs_txt}, the geometry + mechanics model had "
                + ("a lower" if lower_all else "not always a lower")
                + " leave-one-out log-loss than the geometry-only model at every penalty ("
                + "; ".join(f"{ctx.num(ll('geometry+mechanics', c), src=MS, key=f'{ps}.models[geometry+mechanics][{c}].loocv_log_loss', fmt='.3f')} versus "
                            f"{ctx.num(ll('geometry', c), src=MS, key=f'{ps}.models[geometry][{c}].loocv_log_loss', fmt='.3f')}" for c in Cs)
                + "), and a flexible geometry model with a quadratic arc term and morphology indicators "
                + ("discriminated better than the geometry-only model." if flex_better else
                   "did not discriminate better than the geometry-only model (leave-one-out AUC "
                   + "; ".join(ctx.num(models["flexible geometry"][c]["loocv_auc"], src=MS,
                                       key=f"{ps}.models[flexible geometry][{c}].loocv_auc", fmt=".2f") for c in Cs) + ")."))
    return Para(txt)


def _selection_para(ctx: Ctx, best: str, rank: list) -> Para:
    """Selection of the best predictor, predictors against their stated direction, calibration geometries."""
    U = f"{PRIMARY}.univariable"
    uni = ctx.opt(MS, U) or {}
    n_mech = sum(1 for r in rank if r["group"] == "mechanics")
    against = [p for p, u in uni.items() if u.get("direction_source") == "config"
               and (u.get("auc_ci") or [None, None])[1] is not None and u["auc_ci"][1] < 0.5]
    ex = f"{PRIMARY}.post_hoc.core_without_calibration_geometries"
    txt = (f"{pname(ctx, best)[0].upper() + pname(ctx, best)[1:]} had the largest core AUC of the "
           f"{ctx.count(n_mech, src=MS, key='count(univariable_ranking | mechanics)')} mechanics candidates and was "
           "selected on the same core lesions; its intervals and its differences from other scores do not include "
           "this selection and are therefore optimistic.")
    if against:
        lst = _and([f"{pname(ctx, p)} (AUC {_vci(ctx, MS, f'{U}.{p}.auc', f'{U}.{p}.auc_ci', inner=True)})"
                    for p in against])
        txt += (" " + lst[0].upper() + lst[1:] + (" was" if len(against) == 1 else " were")
                + " associated with damage in the direction opposite to the one stated before the analysis.")
    n_ex = len(ctx.opt(MS, f"{ex}.excluded") or [])
    if n_ex and ctx.opt(MS, f"{ex}.{best}.auc") is not None:
        txt += (f" Without the {ctx.count(n_ex, src=MS, key=f'len({ex}.excluded)')} core lesions that share the "
                "geometry of the pilot lesions used to calibrate the calcium modulus (post hoc), the AUC of "
                f"{pname(ctx, best)} was {_vci(ctx, MS, f'{ex}.{best}.auc', f'{ex}.{best}.auc_ci')} and that of the "
                f"Fujino score {_vci(ctx, MS, f'{ex}.fujino_score.auc', f'{ex}.fujino_score.auc_ci')}.")
    return Para(txt)


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
KIND_WORDS = {"model_internal": "the model-internal criterion", "calibration_target": "the calibration target",
              "model_vs_model": "the comparison with an earlier model", "geometry_plausibility": "the geometry check"}


def _tally_text(ctx: Ctx, cons: dict, core_checks: list) -> str:
    """Agreement counts by kind of check: published observations separately from checks that agree by
    construction (model-internal criterion, calibration target) and from model-to-model comparisons."""
    pub = [c for c in core_checks if c["kind"] == "published_observation"]
    ids = lambda cs, lab: _and([c["id"] for c in cs if c["agreement"] == lab])  # noqa: E731
    k = "agreement_counts_by_kind.core.published_observation"
    parts = []
    for lab, word in (("agrees", "agreed"), ("partial", "were inconclusive (point estimate on the expected side, "
                                                        "interval including no effect, or only one of two criteria met)"),
                      ("disagrees", "disagreed"), ("not computed", "could not be computed")):
        n = sum(c["agreement"] == lab for c in pub)
        if n:
            parts.append(f"{ctx.v(CONS, f'{k}.{lab}', fmt='d')} {word} ({ids(pub, lab)})")
    txt = (f"Of the {ctx.count(len(pub), src=CONS, key='count(core checks | kind == published_observation)')} "
           f"comparisons with published observations on the core set ({ctx.tab('table2_verification_consistency')}), "
           + _and(parts) + ".")
    other = [c for c in core_checks if c["kind"] != "published_observation"]
    for c in other:
        what = KIND_WORDS.get(c["kind"], c["kind"])
        if c["agreement"] == "agrees" and c["kind"] in ("model_internal", "calibration_target"):
            txt += f" {what[0].upper() + what[1:]} ({c['id']}) agreed, as expected by construction."
        elif c["agreement"] == "not computed":
            txt += f" {what[0].upper() + what[1:]} ({c['id']}) could not be computed."
        else:
            txt += f" {what[0].upper() + what[1:]} ({c['id']}): {c['agreement']}."
    return txt


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
    paras.append(Para(_tally_text(ctx, cons, core_checks)))
    from dtivl import params as P
    c4, c5 = cons["checks"][i4], cons["checks"][i5]
    pub = sorted(P.lit(r).value for r in ("fracture_rate_DisruptCAD_III_OCT", "fracture_rate_DisruptCAD_II_OCT"))
    rel4 = ("below" if c4["model_value"] < pub[0] else "above" if c4["model_value"] > pub[-1] else "within the range of")
    why5 = ("In the model the plain balloon is a larger static load than the equivalent IVL load (Section 3.2), so "
            "this disagreement follows from the quasi-static representation of IVL (decision D14) together with the "
            "calcium modulus calibrated so that the IVL balloon alone does not fracture calcium (decision D8)."
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
            "C6). This model-internal criterion set the calcium modulus on the three pilot lesions, all of which share "
            "the geometry of core lesions, so it is not independent of the calibration."))
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
            f"{ctx.cite('hill2020disruptcad3')} (check C7).{spread}"
            + (" Because every stent was expanded to its target diameter, this agreement mainly reflects the stent "
               "model and the wide clinical spread." if _stent_target_sentence(ctx, ctx.data(OUT)[ctx.data(OUT)["set"] == "core"]).startswith("Every")
               else "")))
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
            f"{ctx.cite('tct120pooledoct2021')} (check C10, {'one of its two criteria met' if c10['agreement'] == 'partial' else c10['agreement']}). "
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
PILOT = "results/pilot_nonlinear.json"
OAT = "results/sensitivity_oat.json"
MORRIS = "results/sensitivity_morris.json"
SOBOL = "results/sensitivity_sobol.json"
L3D = "results/length3d.json"


def _modulus_window(ctx: Ctx) -> str:
    """Pilot sweep of the calcium modulus (Section 2.4): balloon-alone and IVL damage per modulus."""
    les = ctx.data(PILOT)["lesions"]
    by_e = {}
    for lid, rows in les.items():
        for r in rows:
            if "skipped" in r or (r.get("ivl") or {}).get("status") != "ok":
                continue
            e = r["ca_E_MPa"]
            b = bool(((r.get("balloon_only") or {}).get("cracks") or {}).get("fracture"))
            v = bool(((r.get("ivl") or {}).get("cracks") or {}).get("fracture"))
            by_e.setdefault(e, []).append((b, v))
    es = sorted(by_e)
    n = len(les)
    ok = [e for e in es if len(by_e[e]) == n and not any(b for b, _ in by_e[e]) and all(v for _, v in by_e[e])]
    if not ok:
        return ("In the pilot sweep of the calcium modulus (Section 2.4), no modulus gave IVL damage in every pilot "
                "lesion without damage under the IVL balloon alone.")
    lo_i, hi_i = es.index(ok[0]), es.index(ok[-1])
    fe = lambda e: ctx.num(e, src=PILOT, key="lesions.*.ca_E_MPa", fmt=".4g")  # noqa: E731
    cnt = lambda e, j: sum(x[j] for x in by_e[e])  # noqa: E731
    win = (f"only {fe(ok[0])} MPa" if len(ok) == 1 else f"the moduli from {fe(ok[0])} to {fe(ok[-1])} MPa")
    txt = (f"In the pilot sweep of the calcium modulus (Section 2.4), {win} gave through-thickness damage under "
           f"IVL in all {ctx.count(n, src=PILOT, key='len(lesions)')} pilot lesions without damage under the IVL "
           "balloon alone.")
    if lo_i > 0:
        e = es[lo_i - 1]
        txt += (f" At the next lower modulus ({fe(e)} MPa) the balloon alone damaged "
                f"{_k_of_n(ctx, cnt(e, 0), len(by_e[e]), key=f'pilot balloon_only fracture at E={e:g}', src=PILOT)} lesions")
    if hi_i < len(es) - 1:
        e = es[hi_i + 1]
        txt += ((";" if lo_i > 0 else " At") + f" at the next higher modulus ({fe(e)} MPa) IVL damaged "
                f"{_k_of_n(ctx, cnt(e, 1), len(by_e[e]), key=f'pilot ivl fracture at E={e:g}', src=PILOT)}")
    return txt + ". The fracture outcome of the model is therefore sensitive to the calcium modulus, and the " \
                 "calibrated value lies in a narrow window of the sweep."


def _scen(ctx: Ctx, name: str) -> str:
    from dtivl.drivers.figs.fig7_sensitivity import SCENARIO_LABELS
    key, _, val = name.partition("=")
    lab = SCENARIO_LABELS.get(key, key)
    try:
        v = float(val)
    except ValueError:
        return f"{lab} = {val}" if val else lab
    return f"{lab} = {ctx.num(v, src=OAT, key=f'scenario {name}', fmt='g')}"


def _oat_text(ctx: Ctx) -> str:
    oat = ctx.data(OAT)
    runs = oat.get("runs") or {}
    lesions = sorted(runs)
    scen = []
    for lid in lesions:
        scen += [sc for sc in runs[lid] if sc != "baseline" and sc not in scen]
    changed = [sc for sc in scen if any((runs[l].get(sc) or {}).get("ivl_fracture_changed_vs_baseline") for l in lesions)]
    diffs = [((runs[l].get(sc) or {}).get("ivl_onset_load_minus_baseline"), sc) for sc in scen for l in lesions]
    diffs = [(d, sc) for d, sc in diffs if d is not None]
    txt = (f"Of the {ctx.count(len(scen), src=OAT, key='number of scenarios')} one-at-a-time scenarios, "
           + (f"{ctx.count(len(changed), src=OAT, key='scenarios with ivl_fracture_changed_vs_baseline')} changed the "
              "IVL through-thickness damage outcome of at least one pilot lesion ("
              + _and([_scen(ctx, sc) for sc in changed]) + ")" if changed else
              "none changed the IVL through-thickness damage outcome of any pilot lesion")
           + f" ({ctx.fig('fig7_sensitivity')}a).")
    if diffs:
        dmin, dmax = min(d for d, _ in diffs), max(d for d, _ in diffs)
        big = max(diffs, key=lambda x: abs(x[0]))
        txt += (" Where damage onset was observed in both runs, the onset load changed by "
                f"{_interval(ctx, dmin, dmax, src=OAT, key='ivl_onset_load_minus_baseline range', fmt='.3f')} MPa; "
                f"the largest change came from {_scen(ctx, big[1])}.")
    return txt


def _inp(name: str) -> str:
    from dtivl.drivers.figs.fig7_sensitivity import INPUT_LABELS
    return INPUT_LABELS.get(name, name).replace(" (log)", "")


def _morris_text(ctx: Ctx) -> str:
    m = ctx.data(MORRIS)
    parts = []
    for out, what, unit in (("ivl_onset_load", "the IVL damage-onset load", " MPa"),
                            ("ivl_fracture", "IVL through-thickness damage", "")):
        idx = ((m.get("pooled") or {}).get(out) or {}).get("indices") or {}
        if not idx:
            continue
        mx = max((idx[n].get("mu_star") or 0.0) for n in idx)
        if mx <= 0:
            parts.append(f"no input changed {what}")
            continue
        top = [n for n in sorted(idx, key=lambda n: -(idx[n].get("mu_star") or 0.0))[:3]
               if (idx[n].get("mu_star") or 0.0) >= 0.1 * mx]
        parts.append(f"for {what}, the input{'s' if len(top) > 1 else ''} with the largest $\\mu^*$ "
                     f"{'were' if len(top) > 1 else 'was'} "
                     + _and([f"{_inp(n)} "
                             f"({ctx.v(MORRIS, f'pooled.{out}.indices.{n}.mu_star', fmt='.3g')}{unit})"
                             for n in top]))
    if not parts:
        return ""
    return ("In the Morris screening, pooled over the screening lesions, " + "; ".join(parts)
            + f" ({ctx.fig('fig7_sensitivity')}b,c). $\\mu^*$ is the mean absolute change of the output per "
            "normalised input range; censored onset loads sit at the peak IVL load, which itself moves with the "
            "pulse scale and the transfer factor.")


def _sobol_text(ctx: Ctx, best: str | None) -> str:
    sb = ctx.data(SOBOL)
    key = best[len("pred_"):] if best and best.startswith("pred_") else best
    lr = sb.get("lesion_results") or {}
    have = {l: r["predictors"][key]["indices"] for l, r in lr.items()
            if key in (r.get("predictors") or {}) and (r["predictors"][key] or {}).get("indices")}
    if not have:
        return ""
    names = list(next(iter(have.values())))
    import numpy as np
    med = {n: float(np.median([have[l][n]["ST"] for l in have])) for n in names}
    top = max(med, key=med.get)
    rs = (sb.get("ranking_stability") or {}).get(key) or {}
    txt = (f"For {pname(ctx, best)}, the total-order Sobol index was largest for the {_inp(top)} "
           f"(median over the lesions {ctx.num(med[top], src=SOBOL, key=f'median ST {key} {top}', fmt='.2f')}; "
           f"{ctx.fig('fig7_sensitivity')}d).")
    if rs.get("tau_median") is not None:
        txt += (f" Across the samples of the uncertain inputs, the order of the "
                f"{ctx.v(SOBOL, f'ranking_stability.{key}.n_lesions', fmt='d')} lesions by this predictor had a "
                f"median Kendall $\\tau$ of {ctx.v(SOBOL, f'ranking_stability.{key}.tau_median', fmt='.2f')} with "
                "their baseline order and was unchanged in "
                f"{ctx.v(SOBOL, f'ranking_stability.{key}.fraction_order_unchanged', fmt='.0f', scale=100.0)}% of "
                "the samples.")
    return txt


def _l3d_text(ctx: Ctx) -> str:
    runs = (ctx.data(L3D).get("runs") or {})
    keys = sorted((k for k, r in runs.items() if r.get("model") == "3d"), key=lambda k: runs[k]["length_mm"])
    if not keys:
        if not ctx.draft:
            raise BuildError(f"no 3D runs in {L3D}")
        ctx.pending.append(f"{L3D}: 3D runs")
        return f"{Ctx.PENDING} (three-dimensional length series)"
    items, no_onset = [], []
    for k in keys:
        r = runs[k]
        ref = runs.get(f"{r['lesion']}|2d") or {}
        p3, p2 = (r.get("onset") or {}).get("p"), (ref.get("onset") or {}).get("p")
        L = ctx.num(r["length_mm"], src=L3D, key=f"runs.{k}.length_mm", fmt="g")
        if p3 and p2:
            items.append(f"{ctx.num(p3 / p2, src=L3D, key=f'runs.{k}.onset.p / 2d onset.p', fmt='.2f')} at {L} mm")
        else:
            no_onset.append(L)
    txt = ("In the three-dimensional length series, the damage-onset load relative to the 2D model of the same "
           "cross-section was " + (_and(items) if items else Ctx.PENDING))
    if no_onset:
        txt += "; the models of " + _and(no_onset) + " mm did not reach damage onset within the IVL ramp"
    return txt + f" ({ctx.fig('fig7_sensitivity')}e)."


MODWIN = "results/modulus_window.json"


def _modwin_text(ctx: Ctx, best: str | None) -> str:
    """Post hoc (D18): outcomes and AUCs at the two moduli inside the calibration window."""
    m = ctx.data(MODWIN)
    parts = []
    for Ek, e in m.get("by_modulus", {}).items():
        E = float(Ek)
        if not e.get("n"):
            continue
        key = f"by_modulus[{Ek}]"
        txt = (f"at {ctx.num(E, src=MODWIN, key='moduli_MPa', fmt='.0f')} MPa, IVL damaged "
               f"{_k_of_n(ctx, e['n_ivl_fracture'], e['n'], key=f'{key}.n_ivl_fracture', src=MODWIN)} core lesions "
               f"and the IVL balloon alone damaged {ctx.count(e['n_balloon_fracture'], src=MODWIN, key=f'{key}.n_balloon_fracture')}")
        aucs = [f"{pname(ctx, p)} {ctx.num(e[p]['auc'], src=MODWIN, key=f'{key}.{p}.auc', fmt='.2f')}"
                for p in (best, "fujino_score") if p and isinstance(e.get(p), dict)]
        if aucs:
            txt += " (AUC of " + _and(aucs) + ")"
        parts.append(txt)
    if not parts:
        return ""
    n_done = sum(e.get("n_runs_complete", 0) for e in m["by_modulus"].values())
    n_plan = sum(e.get("n_runs_planned", 0) for e in m["by_modulus"].values())
    if n_done < n_plan:
        if not ctx.draft:
            raise BuildError(f"modulus-window runs incomplete: {n_done} of {n_plan}")
        ctx.pending.append(f"{MODWIN}: {n_done} of {n_plan} runs")
    return ("In the post hoc runs at two calcium moduli inside the calibration window, " + "; ".join(parts) + ".")


def s36(ctx: Ctx) -> list:
    from build_manuscript import ROOT
    best = ctx.opt(MS, f"{PRIMARY}.fujino_comparison.best_single_mechanics")
    paras = [Heading("3.6. Sensitivity and Uncertainty", 2), Para(_modulus_window(ctx))]
    texts = []
    for rel, fn in ((MODWIN, lambda c: _modwin_text(c, best)), (OAT, _oat_text), (MORRIS, _morris_text),
                    (SOBOL, lambda c: _sobol_text(c, best)), (L3D, _l3d_text)):
        if (ROOT / rel).exists():
            texts.append(fn(ctx))
        elif ctx.draft:
            ctx.pending.append(rel)
            texts.append(f"{Ctx.PENDING} ({rel})")
        else:
            raise BuildError(f"sensitivity results missing: {rel}")
    paras += [Para(t) for t in texts if t]
    paras.append(Figure(
        "fig7_sensitivity",
        "Sensitivity and uncertainty. (a) One-at-a-time scenarios on the pilot lesions: change of the IVL "
        "damage-onset load from each lesion's baseline run (open marker: onset censored in either run; ring: "
        "damage outcome changed). (b, c) Morris screening pooled over the screening lesions: $\\mu^*$ of the IVL "
        "damage-onset load and of IVL through-thickness damage, with bootstrap intervals. (d) Total-order "
        "(filled) and first-order (open) Sobol indices of the best single mechanics predictor per lesion. "
        "(e) Damage-onset load of the 3D models relative to the 2D model against model length."))
    return paras


def blocks(ctx: Ctx) -> list:
    return [*s31(ctx), *s32(ctx), *s33(ctx), *s34(ctx), *s35(ctx), *s36(ctx)]
