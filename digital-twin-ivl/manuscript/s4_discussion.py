"""Section 4, Discussion. Numbers through the ledger; statements that depend on a result are chosen from the
results files (rule 1), and no statement claims agreement with patient outcomes (rule 3)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

from s3_results import CONS, MS, OUT, PRIMARY, _and, _diff_word, _k_of_n, _ok, _vci, pname


def _check(cons: dict, cid: str) -> dict:
    return next(c for c in cons["checks"] if c["id"] == cid and c["set"] == "core")


def blocks(ctx: Ctx) -> list:
    g = lambda k: ctx.opt(MS, k)  # noqa: E731
    U = f"{PRIMARY}.univariable"
    best = g(f"{PRIMARY}.fujino_comparison.best_single_mechanics")
    name = g(f"{PRIMARY}.composite.name") or "the composite index"
    rank = g(f"{PRIMARY}.univariable_ranking") or []
    geo = [r["predictor"] for r in rank if r["group"] == "geometry"]
    dcore = f"{PRIMARY}.fujino_comparison.fujino_score.auc_differences.core"
    best_fj = f"{dcore}.{best} - fujino_score"
    lift_best = f"{dcore}.{name} (LOOCV) - {best}"
    incl = f"{PRIMARY}.incremental_value.core.loocv"
    cons = ctx.data(CONS)
    c1, c2, c3 = (_check(cons, c)["agreement"] for c in ("C1", "C2", "C3"))
    df = ctx.data(OUT)
    core = df[df["set"] == "core"]
    t = _ok(core, "ivl")
    frag = t[t["morphology"] == "fragmented"]
    n_frag_f = int(frag["ivl_fracture"].sum())
    n_uc = int((t["ivl_unconverged_steps"].fillna(0) > 0).sum())
    hold_n = g(f"{PRIMARY}.holdout.n") or 0
    from dtivl import params as P
    i5 = next(i for i, c in enumerate(cons["checks"]) if c["id"] == "C5" and c["set"] == "core")
    c5, c6 = cons["checks"][i5], _check(cons, "C6")
    both = core[core["stent_no_ivl_status"].isin(("ok", "limited")) & core["stent_post_ivl_status"].isin(("ok", "limited"))]
    n_ivl_f = int((both["ivl_fracture"] == 1).sum())
    n_both_f = int(((both["ivl_fracture"] == 1) & (both["stent_no_ivl_fracture"] == 1)).sum())
    dexp = (both["stent_post_ivl_stent_expansion_pct"] - both["stent_no_ivl_stent_expansion_pct"]).astype(float)
    small_change = len(dexp) > 0 and abs(float(dexp.median())) < 1.0
    f = t[t["ivl_fracture"] == 1]
    mostly_diffuse = len(f) > 0 and int((f["ivl_diffuse"] == 1).sum()) > len(f) / 2
    ell = P.get("fracture.ell_over_h").value * P.raw("discretisation.h_calcium_mm")
    th_min = float(core["ca_thickness"].min())

    fj_word = _diff_word(g(f"{best_fj}.delta_ci_lo"), g(f"{best_fj}.delta_ci_hi"),
                         "discriminated better than", "discriminated worse than", "did not discriminate differently from")
    lift_word = _diff_word(g(f"{lift_best}.delta_ci_lo"), g(f"{lift_best}.delta_ci_hi"),
                           "improved on", "fell short of", "did not improve on")
    inc_auc = _diff_word(g(f"{incl}.delta_ci_lo"), g(f"{incl}.delta_ci_hi"), "raised", "lowered", "did not clearly change")
    prob_better = (g(f"{incl}.delta_log_loss_ci_hi") or 0) < 0 and (g(f"{incl}.delta_brier_ci_hi") or 0) < 0
    hold_txt = ("The hold-out evaluation (Section 3.3) tests the core thresholds and models on lesions outside the "
                "core design." if hold_n or ctx.draft else "")

    out = [Heading("4. Discussion", 1)]
    # ---- principal findings
    out.append(Para(
        "In a structured *in silico* cohort of calcified coronary cross-sections, mechanics predictors computed "
        "from a linear-elastic analysis of the untreated lesion discriminated through-thickness calcium damage "
        "under IVL, which was computed by a separate nonlinear phase-field model. The best single predictor, "
        f"{pname(ctx, best)}, reached a core-set AUC of {ctx.v(MS, f'{U}.{best}.auc', fmt='.2f')} and "
        f"{fj_word} the cross-sectional Fujino score (AUC {ctx.v(MS, f'{U}.fujino_score.auc', fmt='.2f')}). The "
        f"composite index {name} {lift_word} the best single predictor. Geometry alone already carried much of the "
        "information: "
        + (f"{pname(ctx, geo[0])} was the strongest geometric descriptor, and " if geo else "")
        + f"the geometry-only model reached a leave-one-out AUC of {ctx.v(MS, f'{incl}.auc_b', fmt='.2f')}. Adding "
        f"mechanics {inc_auc} the leave-one-out AUC"
        + (", but it lowered the out-of-fold log-loss and Brier score, so its gain lay in calibrated "
           "probabilities more than in ranking." if prob_better else ".")
        + f" {hold_txt}"))
    # ---- interpretation: why mechanics and Fujino differ
    word1 = {"agrees": "reproduced", "partial": "partly reproduced", "disagrees": "did not reproduce",
             "not computed": "could not test"}
    out.append(Para(
        "The Fujino score was built to predict stent underexpansion, and it awards points for a large calcium "
        f"arc, a thick deposit and a long lesion {ctx.cite('fujino2018score')}. For fracture, these factors do not "
        "act in the same direction. After plain balloon angioplasty, fracture was associated with a large arc and "
        f"thin calcium {ctx.cite('fujino2018fracture')}. The model {word1[c1]} the arc association (check C1) and "
        f"{word1[c2]} the thickness association (check C2). A thick deposit adds to the Fujino score, but in a "
        "thick-walled calcium shell the same lumen pressure produces lower stress. A predictor that combines arc, "
        "thickness and depth through the stress field can therefore rank lesions for fracture differently from a "
        "score that adds them. The non-local stress measures used here were also less sensitive to the mesh than "
        "raw cell peaks (Section 3.1), which matters when a single peak value is used as a threshold."))
    # ---- morphology
    out.append(Para(
        "Clinical OCT studies report fewer calcium fractures after IVL in calcified nodules than in other calcium "
        f"{ctx.cite('emori2023ivlfracture', 'eij2024calcifiednodules')}. The model {word1[c3]} this (check C3). "
        + (f"Fragmented deposits had no through-thickness damage in any of the "
           f"{ctx.count(len(frag), src=OUT, key='count(core fragmented, ivl ok)')} core lesions"
           if n_frag_f == 0 and len(frag) else
           f"Fragmented deposits had through-thickness damage in {_k_of_n(ctx, n_frag_f, len(frag), key='core ivl_fracture | fragmented')} core lesions")
        + ". Separate small deposits do not form a continuous arc that carries hoop stress, so the model predicts "
        "that they are unlikely to crack through. The model has no clinical counterpart for this result, and "
        "it should be tested with OCT."))
    # ---- calibration and absolute rates
    out.append(Para(
        "The absolute fracture rates of the model should not be compared with clinical rates. The core set is a "
        "designed factorial, not a case mix (check C4). More importantly, the calcium modulus was calibrated so "
        "that the IVL balloon inflated alone does not fracture calcium (decision D8). With this calibration the "
        "plain balloon at its higher pressure produced through-thickness damage in "
        f"{ctx.v(CONS, f'checks.{i5}.model_value', fmt='.0f')}% of the core lesions, whereas no lesion "
        f"fractured under a plain balloon in a human cadaver study {ctx.cite('kawai2023cadaver')} (check C5). "
        + ("The model's static fracture load therefore lies between the two balloon pressures, lower than the "
           "cadaver data suggest. Because IVL entered the outcome model as an equivalent static pressure, the same "
           "bias may affect IVL. " if c5["agreement"] == "disagrees" and c6["agreement"] == "agrees" else "")
        + "The predictors were scored on the ranking of lesions, which is less sensitive to such a "
        "shift than absolute rates. The sensitivity analysis to the calcium modulus (Section 3.6) quantifies this "
        "dependence."))
    # ---- stent
    if small_change:
        out.append(Para(
            "Stent expansion varied little between lesions and was nearly the same with and without prior IVL "
            "(Section 3.2). In the model, stenting alone produced through-thickness damage in "
            f"{_k_of_n(ctx, n_both_f, n_ivl_f, key='count(both stent runs, ivl_fracture == 1 and stent_no_ivl_fracture == 1)')} "
            "lesions that IVL fractured, and the stent was represented as a rigid cylinder expanded towards a "
            "target diameter. The model therefore cannot show a benefit of IVL on expansion, and the narrow spread "
            "of expansion (check C7) shows that the stent model does not reproduce the variability seen in "
            "patients. The exploratory stent analyses are reported for completeness and should not be read as "
            "clinical predictions."))
    # ---- damage pattern
    out.append(Para(
        ("The phase field spread damage over most of the deposit, whereas the element-erosion comparator "
         "produced discrete cracks. " if mostly_diffuse else "")
        + "The phase-field outcome is a statement about through-thickness loss of load-bearing "
        "capacity, not about the number or position of cracks, so the fracture counts of OCT studies (check C10) "
        f"cannot be compared with it {ctx.cite('tct120pooledoct2021')}. The width of a phase-field damage band "
        "scales with the regularisation length, and the thinnest calcium of the core set was only "
        f"{ctx.num(th_min / ell, src=OUT + ' + ' + 'params/study_config.yaml', key='min(core ca_thickness) / (fracture.ell_over_h x discretisation.h_calcium_mm)', fmt='.1f')} "
        "such lengths thick, so a single band can span the deposit."))
    # ---- circularity
    out.append(Para(
        "Predictors and outcomes were computed with separate models (Section 2.1): the predictors from a "
        "linear-elastic analysis of the undamaged lesion under the IVL balloon pressure, and the outcomes from a "
        "nonlinear phase-field analysis under the equivalent IVL load. Both models share the lesion geometry and "
        "the calcium constants, so this separation lies in the physics and the load, not in the inputs. A "
        "stress predictor that summarises the same geometry and constants is expected to discriminate well within "
        "the model. The AUCs therefore measure how much of the nonlinear outcome is already contained in the "
        "linear stress state of the untreated lesion; they are not estimates of clinical accuracy. Scoring the "
        "same predictors against the erosion comparator, a different fracture model, is a partial check "
        "(Section 3.4)."))
    # ---- limitations
    out.append(Para(
        "This study has further limitations. No patient outcome data exist for these lesions; the evidence is "
        "verification, consistency with published observations, and sensitivity analysis. The lesions are "
        "idealised two-dimensional plane-strain cross-sections; the three-dimensional length series covers one "
        "cross-section only. IVL was represented by a quasi-static equivalent load from a linear transient "
        "analysis of one pulse with an assumed shape; pulse number, fatigue and cavitation were not modelled. "
        "The calcium properties are uncertain over a wide range, and the failure criterion rests on few "
        "tissue-level data. In "
        f"{_k_of_n(ctx, n_uc, len(t), key='core IVL runs with unconverged_steps > 0')} core IVL runs, at least "
        "one load step was accepted with an unconverged staggered iteration. The non-local peak SED changed under "
        "mesh refinement by slightly more than the acceptance limit. The core set is small, so apparent AUCs are "
        "optimistic; the leave-one-out and hold-out estimates are the relevant ones. Directions of some "
        "geometric descriptors were set on the core set, and the composite rule was drafted after a test run on "
        "preliminary core results."))
    out.append(Para(
        "The next steps are lesions built from patient OCT images with segmentation-to-mesh pipelines "
        f"{ctx.cite('gharaibeh2019spie', 'kadry2021platform', 'straughan2023automated')}, comparison of the "
        "predicted fracture with OCT after IVL in the same lesions, three-dimensional models over the lesion "
        "length, and a deformable stent model. Only such a comparison can show whether the pre-treatment "
        "mechanics predictors identified here carry information for patients."))
    return out
