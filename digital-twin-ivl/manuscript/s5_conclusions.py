"""Section 5, Conclusions (statements chosen from the results files; no claim about patients, rule 3)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

from s3_results import MS, PRIMARY, _diff_word


def blocks(ctx: Ctx) -> list:
    g = lambda k: ctx.opt(MS, k)  # noqa: E731
    best = g(f"{PRIMARY}.fujino_comparison.best_single_mechanics")
    dfj = f"{PRIMARY}.fujino_comparison.fujino_score.auc_differences.core.{best} - fujino_score"
    incl = f"{PRIMARY}.incremental_value.core.loocv"
    vs_fj = _diff_word(g(f"{dfj}.delta_ci_lo"), g(f"{dfj}.delta_ci_hi"), "better than", "worse than",
                       "no differently from")
    gain = _diff_word(g(f"{incl}.delta_ci_lo"), g(f"{incl}.delta_ci_hi"),
                      "and added discrimination to geometry alone",
                      "but discriminated less well than geometry alone",
                      "but added no clear discrimination to geometry alone")
    return [
        Heading("5. Conclusions", 1),
        Para(
            "In a structured *in silico* cohort of calcified coronary lesions, stress measures from a linear-elastic "
            "analysis of the untreated lesion ranked lesions for through-thickness calcium damage under IVL "
            f"{vs_fj} the cross-sectional Fujino score, {gain}. The fracture outcome came from a separate "
            "nonlinear phase-field model, but both models share the lesion geometry and the calcium constants, and "
            "the calcium modulus was calibrated to a model-internal criterion. The results show what a mechanics "
            "digital twin can extract from pre-treatment geometry; whether this carries information for patients "
            "must be tested against OCT after IVL in patient-derived lesions."),
    ]
