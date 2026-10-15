"""1. Introduction.

Every number is read through ``ctx``: published values from
``params/literature_parameters.csv`` (each cited to the bibliography entry of
the row's source), study-design counts from ``params/study_config.yaml`` and
``results/cohort_summary.json``. Literature rows go through
:func:`dtivl.params.lit` first, so a row whose status is not allowed in the
main analysis (UNVERIFIED, SECONDARY) stops the build instead of being quoted.
"""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para
from dtivl import params as P
from dtivl.geometry.fujino import fujino_criteria, fujino_max_score

LIT = "params/literature_parameters.csv"
CFG = "params/study_config.yaml"
COHORT = "results/cohort_summary.json"


# ------------------------------------------------------------------ helpers
def _lit(ctx: Ctx, row: str, transform: str = "value", fmt: str = ".3g", scale: float = 1.0) -> str:
    """Literature value (base units, see dtivl.params) with a status check; ``scale`` converts units."""
    q = P.lit(row, transform)
    return ctx.num(q.value, src=LIT, key=f"{row}.{transform}", fmt=fmt, scale=scale)


def _sd(ctx: Ctx, row: str, fmt: str = ".3g") -> str:
    """Standard deviation of a literature row (base units)."""
    P.lit(row)                                   # status check
    sd = P.load_literature()[row].sd
    return ctx.num(sd, src=LIT, key=f"{row}.sd", fmt=fmt)


def _tabulated(ctx: Ctx, row: str, fmt: str = "g") -> str:
    """Value of a single-number literature row in the unit of the table (e.g. atm)."""
    P.lit(row)                                   # status check
    p = P.load_literature()[row]
    return ctx.num(float(p.value_str), src=LIT, key=f"{row}.value [{p.unit}, as tabulated]", fmt=fmt)


def _mean_sd(ctx: Ctx, row: str, fmt: str = ".3g") -> str:
    return f"{_lit(ctx, row, fmt=fmt)} ± {_sd(ctx, row, fmt=fmt)}"


def _fujino_points(ctx: Ctx, criterion: str) -> str:
    """Points of one Fujino criterion, parsed by dtivl.geometry.fujino from the row's notes."""
    c = fujino_criteria()[criterion]
    row = {"angle": "fujino_angle_threshold", "thickness": "fujino_thickness_threshold",
           "length": "fujino_length_threshold"}[criterion]
    return ctx.num(c.points, src=LIT, key=f"{row}.notes[points]", fmt="d")


def _n_levels(ctx: Ctx, factor: str) -> str:
    key = f"design.core_factorial.{factor}"
    return ctx.num(len(P.raw(key)), src=CFG, key=f"len({key})", fmt="d")


def _index_name() -> tuple[str, str]:
    """(acronym, long form) of the composite index, from design.composite_index_name."""
    node = P.raw("design.composite_index_name")
    name, expansion = node["value"], node["expansion"]
    long = expansion[:-len(" index")] if expansion.endswith(" index") else expansion
    return name, long


# ------------------------------------------------------------------ text
def blocks(ctx: Ctx) -> list:
    c = ctx.cite
    pct = " %"
    idx, idx_long = _index_name()
    n_comp = P.raw("metrics.composite")["value"]["max_predictors"]

    # (i) calcified lesions, underexpansion, calcium modification
    p1 = (
        "Severe coronary calcification resists balloon dilation and limits stent expansion, and stent "
        "underexpansion is the main mechanical consequence that lesion preparation tries to prevent "
        f"{c('mintz2015imaging', 'fujino2018score')}. Plain and cutting balloons and intravascular lithotripsy "
        "(IVL) all aim to fracture the calcium so that the vessel wall can expand "
        f"{c('fujino2018fracture', 'jcin2022cuttingballoon', 'kereiakes2021principles')}. Whether a given "
        "lesion will fracture is therefore a practical question before treatment."
    )

    # (ii) IVL mechanism and clinical OCT evidence; nodules fracture less
    p2 = (
        "Coronary IVL delivers acoustic pressure pulses from electrohydraulic emitters inside a semi-compliant "
        f"balloon, which is inflated at only {_tabulated(ctx, 'ivl_balloon_inflation_pressure')} atm "
        f"({_lit(ctx, 'ivl_balloon_inflation_pressure', fmt='.3f')} MPa) to appose the wall "
        f"{c('tovarforero2019system', 'karimigalougahi2021acoustic')}. The peak positive pressure is about "
        f"{_lit(ctx, 'ivl_peak_positive_pressure')} MPa, and the proposed fracture mechanism is the compressive "
        f"stress of the wavefront {c('kereiakes2021principles')}. Optical coherence tomography (OCT) substudies "
        f"found calcium fracture in {_lit(ctx, 'fracture_rate_DisruptCAD_II_OCT')}{pct} of lesions in "
        f"Disrupt CAD II {c('ali2019disruptcad2')} and in {_lit(ctx, 'fracture_rate_DisruptCAD_III_OCT')}{pct} "
        "in Disrupt CAD III, with multiple fractures in "
        f"{_lit(ctx, 'multiple_fracture_fraction_DisruptCAD_III')}{pct} of the fractured lesions; stent "
        f"expansion was {_mean_sd(ctx, 'stent_expansion_maxCa_site_DisruptCAD_III', fmt='.1f')}{pct} at the "
        "site of maximum calcification and "
        f"{_mean_sd(ctx, 'stent_expansion_MSA_site_DisruptCAD_III', fmt='.1f')}{pct} at the site of minimum "
        f"stent area {c('hill2020disruptcad3')}. A pooled OCT analysis of the Disrupt CAD studies reported "
        f"{_mean_sd(ctx, 'fractures_per_lesion_pooled_IVL_OCT')} fractures per lesion, "
        f"{_lit(ctx, 'full_thickness_fracture_fraction_pooled_IVL_OCT')}{pct} of them through the full "
        f"calcium thickness {c('tct120pooledoct2021')}. Fracture depends on morphology: it occurred at only "
        f"{_lit(ctx, 'fracture_rate_calcified_nodule_sites_IVL')}{pct} of calcified-nodule sites in the pooled "
        f"Disrupt CAD III and IV OCT substudies {c('eij2024calcifiednodules')}, and in a single-centre study it "
        "was associated with a larger calcium arc, a smaller minimum calcium thickness and superficial calcium, "
        "with lower odds for nodular calcium (odds ratio "
        f"{_lit(ctx, 'nodular_calcium_odds_ratio_for_fracture')}) {c('emori2023ivlfracture')}. Clinical "
        "fracture rates are probably lower bounds, since an ex vivo study found histological microfractures "
        f"that OCT did not show {c('laserivl_eurointervention')}."
    )

    # (iii) imaging-based calcium scores and their limits
    p3 = (
        "Before treatment, the expected result is estimated from imaging. The Fujino OCT calcium score awards "
        f"{_fujino_points(ctx, 'angle')} points for a maximum calcium angle above "
        f"{_lit(ctx, 'fujino_angle_threshold')}°, {_fujino_points(ctx, 'thickness')} point for a maximum "
        f"thickness above {_lit(ctx, 'fujino_thickness_threshold')} mm and {_fujino_points(ctx, 'length')} "
        f"point for a length above {_lit(ctx, 'fujino_length_threshold')} mm; lesions with the maximum score of "
        f"{ctx.num(fujino_max_score(True), src=LIT, key='sum of fujino_*_threshold.notes[points]', fmt='d')} "
        f"had a stent expansion of {_lit(ctx, 'fujino_score4_stent_expansion')}{pct} against "
        f"{_lit(ctx, 'fujino_score0to3_stent_expansion')}{pct} for lower scores "
        f"{c('fujino2018score')}. A revised score adds a minimum calcium thickness above "
        f"{_lit(ctx, 'sato_min_thickness_threshold', fmt='.2f')} mm {c('sato2025revised')}, and machine "
        "learning on intravascular OCT (IVOCT) lesion attributes predicts underexpansion "
        f"{c('gharaibeh2023underexpansion')}. Fracture cut-offs were derived for plain balloons (calcium angle "
        f"{_lit(ctx, 'fracture_angle_cutoff_balloon')}°) {c('fujino2018fracture')} and cutting balloons "
        f"({_lit(ctx, 'fracture_angle_cutoff_cutting_balloon')}° and "
        f"{_lit(ctx, 'fracture_thickness_cutoff_cutting_balloon')} mm) {c('jcin2022cuttingballoon')}, whereas "
        f"an IVL study reported associations without cut-offs {c('emori2023ivlfracture')}. All of these "
        "predictors are geometric. They contain neither the load nor the stiffness of the calcium and the "
        "surrounding tissue, nor the stress concentration at the ends of a deposit."
    )

    # (iv) computational models, fracture modelling, digital twins
    p4 = (
        "Finite element (FE) models have been used to study balloon angioplasty and stenting "
        f"{c('holzapfel2002angioplasty', 'lally2005stent')} and the effect of calcification on plaque stress "
        f"{c('huang2001calcification')}. In calcified lesions, the predicted stresses and damage depend strongly "
        f"on the constitutive representation {c('conway2017predilation')}, heavily calcified lesions carry a "
        f"larger share of the balloon load {c('dong2023loadsharing')}, and simulation-driven machine learning "
        f"has predicted stent expansion {c('dong2020simml')}. Calcium fracture under plain and cutting balloons "
        f"has been assessed through the maximum principal stress {c('zhu2021cuttingballoon')} or a brittle "
        f"calcium model {c('li2023balloontype')}. IVL studies prescribed the fracture and simulated its effect "
        f"{c('sogbadji2025defects', 'laserivl_eurointervention')}, so fracture was an input rather than a "
        "prediction. Variational phase-field models of brittle fracture can predict whether and where a crack "
        "forms without a prescribed path: they regularise the crack over a length $\\ell$ "
        f"{c('bourdin2000numerical')}, have robust staggered solvers {c('miehe2010phasefield')}, link crack "
        f"nucleation to the material strength {c('tanne2018nucleation')} and have been applied to arterial "
        f"tissue {c('gultekin2016phasefield')}. Pipelines from intravascular images to FE models "
        f"{c('kadry2021platform', 'straughan2023automated', 'gharaibeh2019spie')} bring lesion-specific "
        f"digital twins, in the sense proposed for precision cardiology {c('corralacero2020digitaltwin')}, "
        "within reach, and the in silico trial framework sets out how the credibility of such models is "
        f"established {c('viceconti2021insilico')}."
    )

    # (v) the gap
    p5 = (
        "To our knowledge, no study has tested whether lesion-level mechanics from a computationally cheap "
        "pre-treatment "
        "analysis predict IVL-induced fracture better than geometry or the Fujino score, in a controlled "
        "cohort in which the fracture comes from a separate simulation. Clinical data cannot answer this "
        "directly: morphology and device use vary together, OCT under-detects fractures, and each lesion is "
        "treated once. The inputs are also uncertain: calcium moduli range from about "
        f"{_lit(ctx, 'calcified_tissue_tangential_modulus', fmt='.2g')} MPa in tissue-level tension "
        f"{c('loree1994modulus')} to "
        f"{_lit(ctx, 'calcification_modulus_nanoindentation', transform='lower', scale=1e-3, fmt='.3g')}–"
        f"{_lit(ctx, 'calcification_modulus_nanoindentation', transform='upper', scale=1e-3, fmt='.3g')} GPa "
        f"reported from nanoindentation {c('cvet2014calcmodulus')}, and tissue-level fracture data for "
        f"calcified plaque come from a single uniaxial study {c('holzapfel2004plaque')}."
    )

    # (vi) aim and what the study does
    p6 = (
        "This study addresses that question with a structured in silico cohort. A full factorial core set of "
        f"{ctx.v(COHORT, 'n_per_set.core', fmt='d')} two-dimensional (2D) cross-sections combines "
        f"{_n_levels(ctx, 'arc_deg')} calcium arcs, {_n_levels(ctx, 'thickness_mm')} thicknesses and "
        f"{_n_levels(ctx, 'morphology')} morphologies (superficial sheet, deep sheet, calcified nodule, "
        f"fragmented) {c('sugiyama2019calcified')}; a hold-out set of "
        f"{ctx.v(COHORT, 'n_per_set.holdout', fmt='d')} lesions is sampled by Latin hypercube over "
        "continuous ranges that include lipid pools and fibrous caps. Lesions are polar label maps in the "
        f"IVOCT segmentation convention {c('gharaibeh2019spie', 'lee2020twostep')}, so segmented images can "
        "use the same pipeline. Predictors come from the geometry and from a linear-elastic analysis of the "
        "undamaged lesion under the IVL balloon pressure. Outcomes come from a separate nonlinear simulation "
        "with phase-field damage in the calcium, in which IVL is the quasi-static equivalent of a single "
        "acoustic pulse, followed by stent expansion; the primary outcome is through-thickness calcium damage "
        "under IVL. Predictors are evaluated by receiver operating characteristic (ROC) analysis with "
        "leave-one-out cross-validation (LOOCV) and bootstrap intervals, combined by penalised logistic regression "
        f"on at most {ctx.num(n_comp, src=CFG, key='metrics.composite.value.max_predictors', fmt='d')} "
        f"mechanics predictors into the {idx_long} ({idx}) index, and compared with the cross-sectional Fujino "
        "score. Geometry-only and geometry-plus-mechanics models give the incremental value of mechanics. "
        "Thresholds and models are derived on the core set and applied unchanged to the hold-out set. No "
        "patient data are used: the study offers verification against analytical solutions and under mesh "
        "refinement, consistency with published observations and sensitivity analysis of uncertain inputs, "
        "not a clinical test of the predictors."
    )

    return [Heading("1. Introduction", 1)] + [Para(t) for t in (p1, p2, p3, p4, p5, p6)]
