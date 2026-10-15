"""Front matter: title, author placeholder, structured abstract (MDPI: Background/Objectives, Methods, Results,
Conclusions) and keywords. Abstract numbers come from the results files through the ledger (rule 1)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

from s3_results import MS, OUT, PRIMARY, _diff_word, _vci, pname

KEYWORDS = ("intravascular lithotripsy", "coronary artery calcification", "finite element analysis",
            "phase-field fracture", "digital twin", "in silico", "stent expansion", "optical coherence tomography")


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
    abstract = (
        "**Background/Objectives:** Intravascular lithotripsy (IVL) fractures coronary calcium before stenting, "
        "but which lesions fracture is hard to predict from imaging scores. We asked whether mechanics computed "
        "before treatment predict calcium fracture under IVL in *in silico* lesions. "
        "**Methods:** Calcified coronary cross-sections were generated as a designed core set "
        f"({ctx.count(n_core, src=OUT, key='count(set == core)')} lesions: calcium arc, thickness and morphology) "
        f"and a sampled hold-out set ({ctx.count(n_hold, src=OUT, key='count(set == holdout)')} lesions). "
        "Candidate predictors came from a linear-elastic analysis of the untreated lesion. The outcome, "
        "through-thickness calcium damage, came from a separate nonlinear finite element model with a "
        "phase-field fracture law under an equivalent IVL load. Thresholds and a composite index were derived on "
        "the core set and tested unchanged on the hold-out set, against the optical coherence tomography (OCT) "
        "calcium score of Fujino. "
        "**Results:** On the core set, "
        f"{pname(ctx, best)} had an area under the receiver operating characteristic curve (AUC) of "
        f"{_vci(ctx, MS, f'{U}.{best}.auc', f'{U}.{best}.auc_ci')}, and the "
        f"Fujino score {_vci(ctx, MS, f'{U}.fujino_score.auc', f'{U}.fujino_score.auc_ci')}. The composite index "
        f"{name} had a leave-one-out AUC of {_vci(ctx, MS, f'{comp}.core.loocv_auc', f'{comp}.core.loocv_auc_ci')}. "
        f"A geometry-only model reached {ctx.v(MS, f'{incl}.auc_b', fmt='.2f')}, and adding mechanics gave "
        f"{ctx.v(MS, f'{incl}.auc_a', fmt='.2f')} ({inc_word}). On the hold-out set, {pname(ctx, best)} had an AUC "
        f"of {_vci(ctx, MS, f'{hb}.auc', (f'{hb}.auc_ci_lo', f'{hb}.auc_ci_hi'))}. The model {arc} the "
        f"published association of fracture with calcium arc {rate} the clinical fracture rates. "
        "**Conclusions:** In this model, pre-treatment mechanics ranked lesions for calcium fracture under IVL "
        f"{vs_fj} the Fujino score, with {gain} in discrimination over geometry alone. These are *in silico* "
        "results; they require comparison with OCT after IVL in patient lesions.")
    return [
        Heading(sections.TITLE, 0),
        Para("[Author names and affiliations to be added.]", style="small"),
        Para(abstract, style="abstract"),
        Para("**Keywords:** " + "; ".join(KEYWORDS), style="keywords"),
    ]
