"""Section 2.10, Statistical Analysis (implementation: src/drivers/stage_metrics.py, src/stats/)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

CFG = "params/study_config.yaml"
COH = "cohort/cohort.csv"

#: manuscript wording of the geometric candidates (checked against metrics.candidates.geometry)
GEOMETRY_WORDS = {
    "ca_arc_total_deg": "calcium arc", "ca_thickness_max": "maximum calcium thickness",
    "ca_depth_min": "minimum calcium depth", "ca_area": "calcium area", "n_deposits": "number of deposits",
    "lumen_area": "lumen area", "lipid_arc_deg": "lipid arc", "plaque_ecc_index": "plaque eccentricity index",
}
#: the one mechanics candidate with a negative a priori direction
MECH_NEGATIVE = {"pred_min_thickness_high_stress": "the minimum calcium thickness under high stress"}


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _cand(ctx: Ctx, group: str) -> str:
    from dtivl import params as P
    n = len(P.raw("metrics.candidates")["value"][group])
    return ctx.count(n, src=CFG, key=f"len(metrics.candidates.{group})")


def _directions():
    """(positive, negative, set-on-core) geometric candidates and the mechanics overrides, from the config."""
    from dtivl import params as P
    cand = P.raw("metrics.candidates")["value"]
    d = P.raw("metrics.direction")["value"]
    over, default = d["overrides"], d["default"]
    geo = cand["geometry"]
    if set(geo) != set(GEOMETRY_WORDS):
        raise ValueError(f"geometric candidates changed: {sorted(set(geo) ^ set(GEOMETRY_WORDS))}")
    mech_over = {k: v for k, v in over.items() if k in cand["mechanics"]}
    if mech_over != {k: -1 for k in MECH_NEGATIVE} or default != 1:
        raise ValueError(f"mechanics directions changed: {mech_over}, default {default}")
    pos = [g for g in geo if over.get(g, default) == 1]
    neg = [g for g in geo if over.get(g, default) == -1]
    core = [g for g in geo if over.get(g, default) == "core"]
    return pos, neg, core


def blocks(ctx: Ctx) -> list:
    import pandas as pd
    from dtivl import params as P

    comp = P.raw("metrics.composite")["value"]
    name = P.raw("design.composite_index_name")
    seed_off = int(P.get("metrics.seed_offset").value)
    geo_model = P.raw("metrics.incremental")["value"]["geometry_model"]
    ci = ctx.param("metrics.ci_level", fmt=".0f", scale=100.0)
    nb = ctx.param("metrics.bootstrap_n", fmt=".0f")
    pos, neg, core_dir = _directions()
    w = lambda keys: _and([GEOMETRY_WORDS[k] for k in keys])  # noqa: E731

    coh = pd.read_csv(COH)
    core_rows = coh[coh["set"] == "core"]
    n_core_lipid = int((core_rows["lipid_arc_deg"] > 0).sum())
    if n_core_lipid == 0:
        lipid_note = ("The lipid arc is zero in every core lesion, because the core design has no lipid pool; it is "
                      "therefore constant on the core set and has no threshold.")
    else:
        lipid_note = (f"A lipid pool is present in "
                      f"{ctx.count(n_core_lipid, src=COH, key='count(set == core and lipid_arc_deg > 0)')} core "
                      "lesions.")
    # a geometry-model descriptor that is not one of the words above would be named wrongly
    if not set(geo_model) <= set(GEOMETRY_WORDS):
        raise ValueError(f"geometry model descriptors not worded: {sorted(set(geo_model) - set(GEOMETRY_WORDS))}")

    return [
        Heading("2.10. Statistical Analysis", 2),
        Para(
            "All thresholds, data-derived predictor directions, predictor selections and model fits were derived "
            "on the core set only. The hold-out set was used only to evaluate them unchanged. The candidate "
            f"predictors were the {_cand(ctx, 'mechanics')} pre-treatment mechanics predictors of Section 2.8, "
            f"{_cand(ctx, 'geometry')} geometric descriptors ({w(list(GEOMETRY_WORDS))}) and the cross-sectional "
            f"Fujino score. {lipid_note} The quantities of the outcome model (the IVL load factor and every outcome "
            "column) were excluded from the candidates by a check in the code."),
        Para(
            "The expected direction of each association was stated before the analysis. A higher value was "
            "expected with damage for the stress, strain-energy, area-fraction and edge-concentration predictors, "
            f"the {w(pos)} and the Fujino score; a lower value was expected for {MECH_NEGATIVE[next(iter(MECH_NEGATIVE))]}, "
            f"the {w(neg)}. No direction was stated for the {w(core_dir)}. For these, the direction was set on the "
            "core set as the sign that gives an AUC of at least one half. It was re-derived inside each "
            "leave-one-out fold and kept fixed in the bootstrap, and it is flagged in the results as set on the "
            "core set, which makes these AUCs optimistic."),
        Para(
            "Each analysis used the lesions whose run for the relevant protocol had completed (status ok). For "
            "stent expansion, runs stopped because the mean stent–lesion contact pressure reached the rated "
            "pressure before the nominal diameter (status limited) were kept with the expansion they reached. Runs "
            "whose solver failed at the minimum load increment were excluded. The number of lesions excluded for each reason is recorded per outcome in the results "
            "files."),
        Para(
            "For the primary outcome (through-thickness damage under IVL) and the binary secondary outcomes, "
            "discrimination was summarised by the area under the receiver operating characteristic curve (AUC) of "
            "the oriented predictor. The threshold maximised the Youden index, placed midway between adjacent "
            "observed values, with ties broken towards the more specific cut. Sensitivity, specificity and accuracy "
            "at the threshold are reported with Wilson score intervals. Intervals for the AUC and the threshold come "
            f"from {nb} bootstrap resamples, stratified by outcome class, as two-sided {ci}% percentile intervals. "
            "The threshold rule was also assessed by LOOCV: each lesion was classified with the threshold derived "
            "without it. When the AUC was $0$ or $1$, or every resample gave the same value, the percentile interval "
            "has zero width; such intervals are flagged as degenerate and are not read as statements of precision. "
            "A binary analysis required at least "
            f"{ctx.param('metrics.min_per_class', fmt='.0f')} lesions in each outcome class."),
        Para(
            f"The composite index, named {name['value']} ({name['expansion']}), is a penalised logistic regression "
            "on at most "
            f"{ctx.num(comp['max_predictors'], src=CFG, key='metrics.composite.max_predictors', fmt='d')} mechanics "
            "predictors chosen by a fixed rule. Candidates were the mechanics predictors without missing or constant "
            "values whose oriented AUC on the training rows was at least "
            f"{ctx.num(comp['min_discrimination'], src=CFG, key='metrics.composite.min_discrimination', fmt='.2f')}, "
            "so a predictor associated with the opposite of its stated direction could not enter. "
            "They were ranked by that AUC and taken in rank order, skipping any candidate whose absolute Spearman "
            "correlation with a predictor already chosen exceeded "
            f"{ctx.num(comp['max_abs_spearman'], src=CFG, key='metrics.composite.max_abs_spearman', fmt='.2f')}. "
            "The chosen predictors were standardised, and an L2-penalised logistic regression with a fixed inverse "
            f"penalty of {ctx.param('metrics.logistic_C', fmt='.1f')} was fitted. The index is the model log-odds. "
            "Selection, fitting and the Youden threshold of the predicted probability were repeated inside every "
            "LOOCV fold, so that the left-out lesion never contributed to its own classification. Out-of-fold "
            "performance is reported as the LOOCV AUC, the log-loss and the Brier score. The bootstrap interval of "
            "the LOOCV AUC resamples the fixed out-of-fold predictions and does not include the variability of "
            "refitting and reselection. Selected predictors whose fitted coefficient has the opposite sign to their "
            "stated direction are reported. The rule was drafted during implementation, after a test run on the "
            "preliminary core results, and amended once after code review to rank the candidates by the oriented "
            "AUC."),
        Para(
            "The composite index and the best single mechanics predictor were compared with the Fujino score by the "
            "paired bootstrap of the AUC difference. The incremental value of mechanics over geometry was tested by "
            "comparing a geometry-only penalised logistic model with the same model plus the mechanics predictors "
            "chosen by the composite rule. The geometry-only model used the "
            f"{ctx.count(len(geo_model), src=CFG, key='len(metrics.incremental.geometry_model)')} descriptors that "
            f"represent the core design factors ({w(geo_model)}; the last two summarise the morphology). The two "
            "models were compared by the apparent AUC difference, with both models reselected and refitted in every "
            "bootstrap resample, by the LOOCV AUC, log-loss and Brier score (paired bootstrap of the out-of-fold "
            "predictions), and on the hold-out set with the core-fit models evaluated unchanged. A difference whose "
            "interval includes zero is reported as no evidence of incremental value."),
        Para(
            "For the damage-onset load under IVL, an event was the first load step at which the maximum damage "
            "reached the crack threshold. An onset first recorded after the peak load, during unloading, was "
            "counted as an event at the peak load. A lesion without onset was right-censored at its own peak "
            "equivalent IVL pressure, the largest load it received. The transient load factor enters only this "
            "censoring value, never a predictor. Discrimination was summarised by Harrell's concordance index (C) of "
            "the oriented predictor, where a higher oriented value should go with an earlier onset. For stent "
            "expansion with and without prior IVL, no direction was stated in advance. Spearman's correlation and C "
            "were computed on the raw predictor, their observed signs are reported, and these analyses are labelled "
            "exploratory. A ranked analysis required at least "
            f"{ctx.param('metrics.min_rows_continuous', fmt='.0f')} lesions with the outcome."),
        Para(
            "As a cross-model check, the same predictors were scored against through-thickness damage under IVL "
            "computed with the element-erosion comparator on its lesion subset (Section 2.6), a fracture model "
            "separate from the phase field. The resampling of this analysis was seeded with the study seed plus a "
            "fixed offset for the metrics stage (seed "
            f"{ctx.num(int(P.raw('design.seed')) + seed_off, src=CFG, key='design.seed + metrics.seed_offset', fmt='d')}), "
            "with an independent random stream for each analysis, so that every interval is reproducible."),
    ]
