"""Front matter: title, author placeholder, structured abstract (MDPI: Background/Objectives, Methods, Results,
Conclusions) and keywords. Abstract numbers come from the results files through the ledger (rule 1)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

from s3_results import MS, OUT, PRIMARY, _diff_word, _vci, pname

KEYWORDS = ("intravascular lithotripsy", "coronary artery calcification", "finite element analysis",
            "phase-field fracture", "computational biomechanics", "in silico", "stent expansion", "optical coherence tomography")


def blocks(ctx: Ctx) -> list:
    import sections
    g = lambda k: ctx.opt(MS, k)  # noqa: E731
    U = f"{PRIMARY}.univariable"
    best = g(f"{PRIMARY}.fujino_comparison.best_single_mechanics")
    comp = f"{PRIMARY}.composite"
    name = g(f"{comp}.name") or "the composite index"
    incl = f"{PRIMARY}.incremental_value.core.loocv"
    df = ctx.data(OUT)
    n_core = int((df["set"] == "core").sum())
    n_hold = int((df["set"] == "holdout").sum())
    inc_word = _diff_word(g(f"{incl}.delta_ci_lo"), g(f"{incl}.delta_ci_hi"),
                          "higher", "lower", "not clearly different")
    hb = f"{U}.{best}.holdout"
    dfj = f"{PRIMARY}.fujino_comparison.fujino_score.auc_differences.core.{best} - fujino_score"
    vs_fj = _diff_word(g(f"{dfj}.delta_ci_lo"), g(f"{dfj}.delta_ci_hi"), "better than", "worse than",
                       "no differently from")
    gain = _diff_word(g(f"{incl}.delta_ci_lo"), g(f"{incl}.delta_ci_hi"), "a gain", "a loss", "no clear gain")
    cons = ctx.data("results/consistency.json")
    agr = {c["id"]: c["agreement"] for c in cons["checks"] if c["set"] == "core"}
    arc = {"agrees": "reproduced", "partial": "partly reproduced"}.get(agr.get("C1"), "did not reproduce")
    rate = "and" if agr.get("C4") == "agrees" else "but not"
    ph = f"{PRIMARY}.post_hoc.fujino_arc_tolerance.core"
    dph = f"{ph}.{best} - fujino_arc_tolerance"
    fj_robust = (g(f"{dph}.delta_ci_lo") or 0) > 0
    c5 = agr.get("C5")
    abstract = (
        "**Background/Objectives:** Intravascular lithotripsy (IVL) fractures coronary calcium before stenting, "
        "but which lesions fracture is hard to predict from imaging scores. We asked whether stress computed "
        "before treatment anticipates calcium damage under IVL in *in silico* lesions. "
        "**Methods:** Calcified coronary cross-sections formed a designed core set "
        f"({ctx.count(n_core, src=OUT, key='count(set == core)')} lesions: calcium arc, thickness, morphology) and a "
        f"sampled hold-out set ({ctx.count(n_hold, src=OUT, key='count(set == holdout)')} lesions). Candidate "
        "predictors came from a linear-elastic analysis of the untreated lesion; through-thickness calcium damage "
        "came from a separate nonlinear finite element model with a phase-field damage law under a quasi-static "
        "equivalent IVL load. Thresholds and a composite index were derived on the core set, tested unchanged on "
        "the hold-out set and compared with the optical coherence tomography (OCT) calcium score of Fujino. "
        "**Results:** Damage was all or nothing. On the core set, "
        f"{pname(ctx, best)} had an apparent area under the receiver operating characteristic curve (AUC) of "
        f"{_vci(ctx, MS, f'{U}.{best}.auc', f'{U}.{best}.auc_ci')}, the cross-sectional Fujino score "
        f"{ctx.v(MS, f'{U}.fujino_score.auc', fmt='.2f')}, or {ctx.v(MS, f'{ph}.auc', fmt='.2f')} with a measurement "
        "tolerance at its arc cut-off. Leave-one-out AUCs were "
        f"{ctx.v(MS, f'{comp}.core.loocv_auc', fmt='.2f')} for the composite index {name}, "
        f"{ctx.v(MS, f'{incl}.auc_b', fmt='.2f')} for geometry alone and {ctx.v(MS, f'{incl}.auc_a', fmt='.2f')} with "
        f"mechanics added ({inc_word}). On the hold-out set, {pname(ctx, best)} had an AUC of "
        f"{_vci(ctx, MS, f'{hb}.auc', (f'{hb}.auc_ci_lo', f'{hb}.auc_ci_hi'))}. The model {arc} the published "
        "association of fracture with calcium arc; IVL acted as a static load below the plain-balloon pressure"
        + (", and the plain balloon damaged lesions that did not fracture in cadaver arteries. " if c5 == "disagrees"
           else ". ")
        + "**Conclusions:** In this model, linear stress in the untreated lesion anticipated nonlinear damage onset, "
        "partly by construction, "
        + ("and ranked lesions better than the Fujino score" if fj_robust else
           "but did not clearly outperform geometry or a tolerance-corrected Fujino score")
        + ". The model reproduces no IVL-specific effect; comparison with OCT after IVL in patient lesions is "
        "required.")
    return [
        Heading(sections.TITLE, 0),
        Para("[Author names and affiliations to be added.]", style="small"),
        Para(abstract, style="abstract"),
        Para("**Keywords:** " + "; ".join(KEYWORDS), style="keywords"),
    ]
