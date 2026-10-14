"""Section 2.10, Statistical Analysis (implementation: src/drivers/stage_metrics.py, src/stats/)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

CFG = "params/study_config.yaml"


def _cand(ctx: Ctx, group: str) -> str:
    from dtivl import params as P
    n = len(P.raw("metrics.candidates")["value"][group])
    return ctx.count(n, src=CFG, key=f"len(metrics.candidates.{group})")


def blocks(ctx: Ctx) -> list:
    from dtivl import params as P

    comp = P.raw("metrics.composite")["value"]
    name = P.raw("design.composite_index_name")
    seed_off = int(P.get("metrics.seed_offset").value)
    n_geo_model = len(P.raw("metrics.incremental")["value"]["geometry_model"])
    ci = ctx.param("metrics.ci_level", fmt=".0f", scale=100.0)
    nb = ctx.param("metrics.bootstrap_n", fmt=".0f")
    return [
        Heading("2.10. Statistical Analysis", 2),
        Para(
            "All thresholds, predictor directions that were not fixed in advance, predictor selections and model fits "
            "were derived on the core set only. The hold-out set was used only to evaluate them unchanged. "
            f"The candidate predictors were the {_cand(ctx, 'mechanics')} pre-treatment mechanics predictors, "
            f"{_cand(ctx, 'geometry')} geometric descriptors and the cross-sectional Fujino score. The quantities "
            "of the outcome model (the IVL load factor and every outcome column) were excluded from the candidates "
            "by a check in the code. Each candidate had a direction stated before the analysis (a higher value "
            "expected with damage, or a lower one for calcium thickness, depth and the minimum thickness under high "
            "stress). For descriptors without a stated direction, the direction was set on the core set and is "
            "flagged in the results."),
        Para(
            "For the primary outcome (through-thickness damage under IVL) and the binary secondary outcomes, "
            "discrimination was summarised by the area under the receiver operating characteristic curve (AUC) of "
            "the oriented predictor. The threshold maximised the Youden index, placed midway between adjacent "
            "observed values, with ties broken towards the more specific cut. Sensitivity, specificity and accuracy "
            "at the threshold are reported with Wilson score intervals. Intervals for the AUC and the threshold come "
            f"from {nb} bootstrap resamples, stratified by outcome class, as two-sided {ci} % percentile intervals. "
            "The threshold rule was also assessed by leave-one-out cross-validation (LOOCV): each lesion was "
            "classified with the threshold derived without it. When the AUC was $0$ or $1$, or every resample gave the "
            "same value, the percentile interval has zero width; such intervals are flagged as degenerate and are "
            "not read as statements of precision. A binary analysis required at least "
            f"{ctx.param('metrics.min_per_class', fmt='.0f')} lesions in each outcome class."),
        Para(
            f"The composite index, named {name['value']} ({name['expansion']}), is a penalised logistic regression "
            "on at most "
            f"{ctx.num(comp['max_predictors'], src=CFG, key='metrics.composite.max_predictors', fmt='d')} mechanics "
            "predictors chosen by a fixed rule. Candidates were the mechanics predictors without missing or constant "
            "values whose oriented AUC on the training rows was at least "
            f"{ctx.num(comp['min_discrimination'], src=CFG, key='metrics.composite.min_discrimination', fmt='.2f')}. "
            "They were ranked by that AUC and taken in rank order, skipping any candidate whose absolute Spearman "
            "correlation with a predictor already chosen exceeded "
            f"{ctx.num(comp['max_abs_spearman'], src=CFG, key='metrics.composite.max_abs_spearman', fmt='.2f')}. "
            "The chosen predictors were standardised, and an L2-penalised logistic regression with a fixed inverse "
            f"penalty of {ctx.param('metrics.logistic_C', fmt='.1f')} was fitted. The index is the model log-odds. "
            "Selection, fitting and the Youden threshold of the predicted probability were repeated inside every "
            "LOOCV fold and every bootstrap resample, so that the left-out lesion never contributed to its own "
            "classification. Out-of-fold performance is reported as the LOOCV AUC, the log-loss and the Brier "
            "score. The bootstrap interval of the LOOCV AUC resamples the fixed out-of-fold predictions and does not "
            "include the variability of refitting and reselection. Selected predictors whose fitted coefficient has "
            "the opposite sign to their stated direction are reported."),
        Para(
            "The composite index and the best single mechanics predictor were compared with the Fujino score by the "
            "paired bootstrap of the AUC difference. The incremental value of mechanics over geometry was tested by "
            "comparing a geometry-only penalised logistic model on the "
            f"{ctx.count(n_geo_model, src=CFG, key='len(metrics.incremental.geometry_model)')} descriptors of the "
            "core design factors (calcium arc, thickness, depth and number of deposits) with the same model plus "
            "the mechanics predictors chosen by the composite rule. The two models were compared by the apparent AUC "
            "difference, with both models refitted in every resample, by the LOOCV AUC, log-loss and Brier score "
            "(paired bootstrap of the out-of-fold predictions), and on the hold-out set with the core-fit models "
            "evaluated unchanged. A difference whose interval includes zero is reported as no evidence of "
            "incremental value."),
        Para(
            "For the damage-onset load under IVL, an event was the first load step at which the maximum damage "
            "reached the crack threshold. A lesion without onset was right-censored at its own peak equivalent IVL "
            "pressure, the largest load it received. The transient load factor enters only this censoring value, "
            "never a predictor. Discrimination was summarised by Harrell's concordance index (C) of the oriented "
            "predictor, where a higher oriented value should go with an earlier onset. For stent expansion with and "
            "without prior IVL, no direction was stated in advance. Spearman's correlation and C were computed on "
            "the raw predictor, their observed signs are reported, and these analyses are labelled exploratory. "
            "A ranked analysis required at least "
            f"{ctx.param('metrics.min_rows_continuous', fmt='.0f')} lesions with the outcome."),
        Para(
            "As a cross-model check, the same predictors were scored against through-thickness damage under IVL "
            "computed with the element-erosion comparator on its lesion subset (Section 2.6), a fracture model "
            "separate from the phase field. All resampling used the study seed "
            f"({ctx.raw('design.seed', fmt='d')}) plus a stage offset "
            f"({ctx.num(seed_off, src=CFG, key='metrics.seed_offset', fmt='d')}), with an independent random "
            "stream for each analysis, so that every interval is reproducible."),
    ]
