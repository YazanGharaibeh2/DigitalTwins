"""Section 4, Discussion. Numbers through the ledger; statements that depend on a result are chosen from the
results files (rule 1), and no statement claims agreement with patient outcomes (rule 3). Rewritten after the
review of the first draft (CHANGES.md)."""
from __future__ import annotations

from build_manuscript import Ctx, Heading, Para

from s3_results import CONS, MS, OUT, PRIMARY, _and, _diff_word, _k_of_n, _ok, _vci, pname

DIAG_ELL = "results/diag_ell.json"
DIAG_LOC = "results/diag_localisation.json"
MESH = "results/verification_mesh_convergence.json"


def _check(cons: dict, cid: str) -> dict:
    return next(c for c in cons["checks"] if c["id"] == cid and c["set"] == "core")


def _whole_deposit_in_diagnostics(ctx: Ctx) -> str:
    """Smallest phase-field length and the variants (seeded crack, heterogeneous toughness) of the Checkpoint 2
    diagnostics that still ended with whole-deposit damage; empty if any did not."""
    ell = ctx.data(DIAG_ELL)
    runs = [(h["ell_mm"], e["final_damaged_fraction"]) for h in ell["by_h"] for e in h["by_E"]]
    loc = ctx.data(DIAG_LOC)
    variants = {k for e in loc["by_E"] for k, v in e.items() if isinstance(v, dict) and "final_damaged_fraction" in v}
    whole_loc = all(v["final_damaged_fraction"] >= 1.0 for e in loc["by_E"] for v in e.values()
                    if isinstance(v, dict) and "final_damaged_fraction" in v)
    if not runs or not all(f >= 1.0 for _, f in runs) or not whole_loc:
        return ""
    ell_min = min(e for e, _ in runs)
    extra = [w for k, w in (("seeded", "a seeded initial crack"), ("heterogeneous", "heterogeneous toughness"))
             if k in variants]
    return ("In the diagnostic runs made before the cohort, whole-deposit damage persisted with the phase-field "
            f"length reduced to {ctx.num(ell_min * 1e3, src=DIAG_ELL, key='min(by_h.ell_mm) x 1000', fmt='g')} µm"
            + (", with " + _and(extra) if extra else "") + ", so its cause is not the ratio of the regularisation "
            "length to the deposit thickness and was not identified.")


def blocks(ctx: Ctx) -> list:
    from dtivl import params as P
    from dtivl.materials.constitutive import MaterialSet

    g = lambda k: ctx.opt(MS, k)  # noqa: E731
    U = f"{PRIMARY}.univariable"
    best = g(f"{PRIMARY}.fujino_comparison.best_single_mechanics")
    name = g(f"{PRIMARY}.composite.name") or "the composite index"
    dcore = f"{PRIMARY}.fujino_comparison.fujino_score.auc_differences.core"
    lift_best = f"{dcore}.{name} (LOOCV) - {best}"
    incl = f"{PRIMARY}.incremental_value.core.loocv"
    ph = f"{PRIMARY}.post_hoc.fujino_arc_tolerance.core"
    dph = f"{ph}.{best} - fujino_arc_tolerance"
    cons = ctx.data(CONS)
    agr = {c: _check(cons, c)["agreement"] for c in ("C1", "C2", "C3", "C5", "C6")}
    df = ctx.data(OUT)
    core = df[df["set"] == "core"]
    t = _ok(core, "ivl")
    f, nf = t[t["ivl_fracture"] == 1], t[t["ivl_fracture"] == 0]
    all_none = len(f) > 0 and (f["ivl_damaged_fraction"] >= 1.0).all() and (nf["ivl_damaged_fraction"] <= 0.0).all()
    frag = t[t["morphology"] == "fragmented"]
    n_frag_f = int(frag["ivl_fracture"].sum())
    uc = t["ivl_unconverged_steps"].fillna(0) > 0
    pb_ok = core[(core["ivl_status"] == "ok") & (core["balloon_10atm_status"] == "ok")]
    ivl_in_poba = len(pb_ok) > 0 and not ((pb_ok["ivl_fracture"] == 1) & (pb_ok["balloon_10atm_fracture"] == 0)).any()
    p_poba = P.get("loads.poba_pressure_atm").value
    all_below = len(pb_ok) > 0 and bool((pb_ok["ivl_p_top"] < p_poba).all())
    against = [p for p, u in (g(U) or {}).items() if u.get("direction_source") == "config"
               and (u.get("auc_ci") or [None, None])[1] is not None and u["auc_ci"][1] < 0.5]
    ms = MaterialSet.baseline()
    sig_c = P.lit("calcification_fracture_stress").value
    mesh = ctx.data(MESH)
    n_mesh_pred = len([k for k in mesh["max_rel_change_default_vs_fine"] if k not in mesh["mesh_dependent_quantities"]])
    n_mech = len(P.raw("metrics.candidates")["value"]["mechanics"])

    lift_word = _diff_word(g(f"{lift_best}.delta_ci_lo"), g(f"{lift_best}.delta_ci_hi"),
                           "improved on", "fell short of", "did not clearly improve on")
    inc_auc = _diff_word(g(f"{incl}.delta_ci_lo"), g(f"{incl}.delta_ci_hi"), "raised", "lowered",
                         "did not clearly change")
    prob_better = (g(f"{incl}.delta_log_loss_ci_hi") or 0) < 0 and (g(f"{incl}.delta_brier_ci_hi") or 0) < 0
    fujino_robust = (g(f"{dph}.delta_ci_lo") or 0) > 0
    word = {"agrees": "reproduced", "partial": "gave an inconclusive result for",
            "disagrees": "did not reproduce", "not computed": "could not test"}

    out = [Heading("4. Discussion", 1)]
    # ---- principal findings and what they mean
    out.append(Para(
        "In a designed *in silico* cohort of calcified coronary cross-sections, a stress measure from a "
        "linear-elastic analysis of the untreated lesion, "
        f"{pname(ctx, best)}, ranked lesions for through-thickness calcium damage in a separate nonlinear "
        f"phase-field model with a core-set AUC of {ctx.v(MS, f'{U}.{best}.auc', fmt='.2f')}. The composite index "
        f"{name} {lift_word} this single predictor. Three features of the model qualify the result. "
        + ("First, the damage outcome was all or nothing, so it amounts to damage onset before the peak load. "
           if all_none else "First, ")
        + "The predictor averages the maximum principal stress over a radius equal to the phase-field length, the "
        "length that governs damage nucleation in the AT1 model, and the area-fraction predictors use the outcome "
        "model's own failure strain and stress. A high AUC is therefore partly expected by construction: it "
        "measures how well the linear stress state of the untreated lesion anticipates nonlinear damage onset in "
        "the same model, not clinical accuracy. Second, all core lesions share one reference geometry and one set "
        "of material constants, so every mechanics predictor is a deterministic function of the three design "
        "factors. On the core set the comparison with the geometry-only model compares model forms, not "
        f"information content: adding mechanics {inc_auc} the leave-one-out AUC"
        + (" and lowered the out-of-fold log-loss and Brier score, at a fixed penalty" if prob_better else "")
        + ". Only the hold-out set, in which lumen area, eccentricity and lipid vary, can show added information. "
        "Third, the advantage over the Fujino score "
        + ("persisted when the arc cut-off of the score was relaxed by the measurement tolerance (Section 3.3)."
           if fujino_robust else
           f"depended on the {ctx.v(MS, f'{ph}.arc_cutoff_deg', fmt='g')}° design level measuring just below the "
           "score's arc cut-off; with a measurement "
           "tolerance at the cut-off the two scores did not differ clearly (Section 3.3).")))
    # ---- published predictors
    out.append(Para(
        "The Fujino score was built to predict stent underexpansion and awards points for a large calcium arc, a "
        f"thick deposit and a long lesion {ctx.cite('fujino2018score')}; a single cross-section cannot meet the "
        "length criterion. After plain balloon angioplasty, fracture was associated with a large arc and thin "
        f"calcium {ctx.cite('fujino2018fracture')}. The model {word[agr['C1']]} the arc association (check C1) and "
        f"{word[agr['C2']]} the thickness association (check C2)"
        + ("; " + _and([pname(ctx, p) for p in against]) + " was associated with damage opposite to its stated "
           "direction (Section 3.3)" if against else "")
        + "."
        + (f" In this model the calcium arc (AUC {ctx.v(MS, f'{U}.ca_arc_total_deg.auc', fmt='.2f')}) discriminated "
           f"damage better than the maximum calcium thickness (AUC {ctx.v(MS, f'{U}.ca_thickness_max.auc', fmt='.2f')})."
           if (g(f"{U}.ca_arc_total_deg.auc") or 0) > (g(f"{U}.ca_thickness_max.auc") or 1) else "")))
    # ---- morphology
    frag_txt = (f"Fragmented deposits had no through-thickness damage in any of the "
                f"{ctx.count(len(frag), src=OUT, key='count(core fragmented, ivl ok)')} core lesions; separate small "
                "deposits do not form a continuous arc that carries hoop load. This result has no clinical "
                "counterpart and should be tested with OCT."
                if len(frag) and n_frag_f == 0 else
                f"Fragmented deposits had through-thickness damage in "
                f"{_k_of_n(ctx, n_frag_f, len(frag), key='core ivl_fracture | fragmented')} core lesions.")
    out.append(Para(
        "Clinical OCT studies report fewer calcium fractures after IVL in calcified nodules than in other calcium "
        f"{ctx.cite('emori2023ivlfracture', 'eij2024calcifiednodules')}; the model {word[agr['C3']]} this "
        f"(check C3). {frag_txt}"))
    # ---- IVL representation and calibration
    out.append(Para(
        "The representation of IVL and the calibration limit what the model can say about lithotripsy. In the "
        "nonlinear model, IVL is a quasi-static inflation to a lesion-specific equivalent pressure (decision D14)"
        + (", which was below the plain-balloon pressure in every lesion" if all_below else "")
        + (", and every lesion damaged under IVL was also damaged under the plain balloon. The model therefore "
           "reproduces no effect specific to lithotripsy, and the plain balloon damaged lesions that did not "
           f"fracture in cadaver arteries {ctx.cite('kawai2023cadaver')} (check C5). " if ivl_in_poba else ". ")
        + "Because calcium failure is strain-controlled, the calibration of the calcium modulus (decision D8) also "
        "set the calcium strength: the implied failure stress was "
        f"{ctx.num(ms.sigma_crit, src='src/materials/constitutive.py MaterialSet.baseline().sigma_crit', key='sigma_crit (MPa)', fmt='.2g')} MPa, "
        f"against a measured calcification fracture stress of {ctx.lit('calcification_fracture_stress', fmt='.3g')} MPa "
        f"{ctx.cite_lit('calcification_fracture_stress')}, and the modulus window that met the calibration "
        "criterion was a single point of the pilot sweep (Section 3.6). The ranking of lesions may be less "
        "sensitive to these choices than the absolute fracture rates; the one-at-a-time and Morris analyses test "
        "this only on a few lesions."))
    # ---- lesion and stent mechanics
    out.append(Para(
        "The model lesions did not resist dilation: under the IVL balloon pressure alone the lumen area grew "
        "substantially (Section 3.2), and the hyperelastic soft tissue recovers on unloading, so fracture cannot "
        "produce a lasting lumen gain. In plane strain, an open calcium arc in soft tissue does not constrain "
        "dilation, so the model does not represent calcium-limited expansion, the premise of lesion preparation. "
        "The stent is a rigid cylinder expanded to a target diameter; the expansion therefore differed between "
        "lesions only through elastic recoil, its spread was far narrower than the clinical one (check C7), and the "
        "exploratory stent analyses should not be read as clinical predictions."))
    # ---- damage pattern
    out.append(Para(
        ("The phase field damaged the whole deposit rather than forming discrete cracks, whereas the "
         "element-erosion comparator produced discrete cracks. " if all_none else "")
        + _whole_deposit_in_diagnostics(ctx)
        + " Because the fracture toughness follows from the phase-field length, which is tied to the mesh size, the "
        "outcome model is in effect a strength-based damage-nucleation model; the number and position of cracks "
        f"cannot be compared with OCT fracture counts (check C10) {ctx.cite('tct120pooledoct2021')}."))
    # ---- separation of the models
    ero = g("outcomes.ivl_erosion_fracture.univariable_ranking")
    ero_txt = ("Scoring the same predictors against the erosion comparator tests the damage law only, because "
               "the comparator shares the geometry, the constants, the load and the failure strain (Section 3.4)."
               if ero else
               "The planned cross-model check against the erosion comparator could not be made, because every "
               "lesion of its subset fractured (Section 3.4); it would in any case test only the damage law, "
               "because the comparator shares the geometry, the constants, the load and the failure strain.")
    out.append(Para(
        "Predictors and outcomes came from separate models (Section 2.1): a linear-elastic analysis of the "
        "undamaged lesion under the IVL balloon pressure, and a nonlinear phase-field analysis under the "
        "equivalent IVL load. Both share the lesion geometry, all material constants and the calcium failure "
        "parameters, and the calcium modulus was calibrated by running the outcome model on three core "
        f"geometries. The separation therefore lies in the physics and the load, not in the inputs. {ero_txt}"))
    # ---- further limitations
    out.append(Para(
        "Further limitations follow. No patient outcome data exist for these lesions; the evidence consists of "
        "verification, consistency with published observations and sensitivity analysis. The lesions are "
        "idealised two-dimensional plane-strain cross-sections, and the three-dimensional length series covers "
        "one cross-section. The transfer factor of the IVL load comes from a linear transient analysis of one pulse "
        "of assumed shape; its solver was checked by unit tests, and its discretisation error on the cohort meshes "
        "was not estimated. The phase-field outcome was not checked under mesh refinement, and the mesh study of "
        f"the predictors covered {ctx.count(n_mesh_pred, src=MESH, key='count(predictor quantities in max_rel_change_default_vs_fine)')} "
        f"of the {ctx.count(n_mech, src='params/study_config.yaml', key='len(metrics.candidates.mechanics)')} "
        "mechanics candidates, with the non-local peak SED slightly above the acceptance limit. In "
        f"{_k_of_n(ctx, int((uc & (t['ivl_fracture'] == 1)).sum()), len(f), key='core IVL runs with unconverged steps | ivl_fracture')} "
        "IVL runs with through-thickness damage, and in "
        f"{_k_of_n(ctx, int((uc & (t['ivl_fracture'] == 0)).sum()), len(nf), key='core IVL runs with unconverged steps | no ivl_fracture')} "
        "without it, at least one load step was accepted with an unconverged staggered iteration. The core set is a "
        "fixed factorial: its bootstrap intervals describe resampling of design points, leave-one-out "
        "cross-validation keeps each lesion's near neighbours in the training set, and the best predictor was "
        "chosen on the same lesions. The candidate list, the a priori directions and the composite rule were "
        "fixed when preliminary core outcomes already existed, and the analyses marked post hoc were added after "
        "the first draft."))
    out.append(Para(
        "The next steps are lesions built from patient OCT images with segmentation-to-mesh pipelines "
        f"{ctx.cite('gharaibeh2019spie', 'kadry2021platform', 'straughan2023automated')}, a representation of IVL "
        "that distinguishes the pulse from a static inflation, a damage model whose outcome does not depend on the "
        "phase-field length alone, a deformable stent, and three-dimensional lesions. Only a comparison of the "
        "predicted fracture with OCT after IVL in the same lesions can show whether pre-treatment mechanics carry "
        "information for patients."))
    return out
