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
                      "but added no clear discrimination to geometry alone on the core set")
    ph = f"{PRIMARY}.post_hoc.fujino_arc_tolerance.core"
    fj_robust = (g(f"{ph}.{best} - fujino_arc_tolerance.delta_ci_lo") or 0) > 0
    return [
        Heading("5. Conclusions", 1),
        Para(
            "In a designed *in silico* cohort of calcified coronary cross-sections, a stress measure from a "
            "linear-elastic analysis of the untreated lesion anticipated through-thickness damage onset under an "
            f"equivalent IVL load in a separate nonlinear phase-field model, {gain}. Part of this agreement is "
            "expected by construction, because the predictor approximates the damage-nucleation criterion of the "
            "outcome model and both models share the geometry and the constants. "
            + ("The predictor ranked lesions better than the cross-sectional Fujino score, also with a measurement "
               "tolerance at the score's arc cut-off. " if fj_robust else
               f"Its advantage over the cross-sectional Fujino score {vs_fj.replace('better than', 'was clear only without')} "
               "a measurement tolerance at the score's arc cut-off. "
               if vs_fj == "better than" else "")
            + "The model represents IVL as a static inflation and reproduces no effect specific to lithotripsy, and "
            "its calcium strength was set by calibration to a model-internal criterion. Whether pre-treatment "
            "mechanics carry information for patients must be tested against OCT after IVL in patient-derived "
            "lesions with a model that distinguishes the pulse from static loading."),
    ]
