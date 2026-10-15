# Checkpoint 3: first manuscript draft, figures and self-review

Date: 2026-10-15. Branch `claude/exciting-bohr-4kp4x0`.

**Status in one paragraph.** A complete first draft exists and builds from the results files. The cohort is not finished: 30 of 72 lesions are done, all of them core lesions, and no hold-out lesion has run. Sensitivity runs have not started. An adversarial review of the draft (four independent reviewers) found that several headline statements did not hold up. The text has been corrected; nothing was hidden or softened. **The review also exposed limitations of the model itself, which text changes cannot fix.** They decide what this paper can claim. I need your decisions on them (Section 4) before spending the remaining compute and before finalising the manuscript.

All numbers below are copied from the files named next to them. In the manuscript itself, every number is read from a file by `manuscript/build_manuscript.py` (ledger: `manuscript/manuscript_numbers.csv`).

## 1. What exists

| Item | Location | State |
|---|---|---|
| Manuscript (MDPI order, structured abstract, Sections 1–5, back matter) | `manuscript/manuscript.md`, `manuscript/manuscript.docx` | Draft build. Values that do not exist yet print as `[pending]`: hold-out results, sensitivity, 3D series. The final build refuses to run while any value is missing. |
| Figures 1–6 | `figures/fig1_design` … `fig6_prediction` (.png 600 dpi, .pdf) | Built from partial cohort data; they will be regenerated |
| Figure 7 (sensitivity) | `src/drivers/figs/fig7_sensitivity.py` | Code and tests done; it needs the sensitivity results |
| Tables 1–4 | `results/tables/*.csv`, embedded in the manuscript | Table 1 sources mapped to citation numbers |
| Change log | `CHANGES.md` | Every change since the draft is logged; no numeric result was edited by hand |
| Post-hoc analyses added after the review | `results/metrics_summary.json` → `outcomes.ivl_fracture.post_hoc` | Labelled as post hoc in Methods 2.10 and Results 3.3 |

Compute status (`results/cohort_runs/`):
- **Cohort:** 30 of 72 lesions complete, all core; the 24 hold-out lesions have not run.
- **Not started:** the one-at-a-time runs (39), Morris (216), Sobol and the D12 3D series.

The container stops when this session is idle, and every stop discards the lesions in progress. Two stops today cost about 15 hours of wall time. At about 25 minutes per lesion on 4 cores, the cohort needs about 4–5 more hours of uninterrupted uptime, and the sensitivity runs about 8 more.

## 2. Preliminary results (core set, 32 lesions with an IVL run)

These will change when the cohort completes.

- **Damage is all or nothing.** Every damaged lesion lost its whole calcium deposit (`cohort_outcomes.csv`: `ivl_damaged_fraction` is 1.0 in all 16 positive lesions and 0 in all 16 negative ones).
- **Best predictor:** the non-local peak σ1 had a core AUC of 0.96 (95% CI 0.89–1.00) (`univariable.pred_sigma1_peak_nl`).
- **Fujino score:** AUC 0.62 as computed. With a 2° measurement tolerance at its arc cut-off it rises to 0.89, and the difference from σ1 becomes 0.07 (−0.04 to 0.21) (`post_hoc.fujino_arc_tolerance`).
- **LIFT composite:** leave-one-out AUC 0.95; no clear gain over σ1.
- **Geometry vs mechanics:** leave-one-out AUC 0.88 for geometry only and 0.92 with mechanics added, a difference of 0.04 (−0.01 to 0.12). Log-loss and Brier score improved.
- **IVL vs plain balloon:** IVL was a static load below the plain-balloon pressure in every lesion (peak equivalent pressure 0.51–0.96 MPa against 1.01 MPa). All 16 IVL-damaged lesions were also damaged by the plain balloon, which damaged one more.
- **Consistency with published observations:** of 8 comparisons, 2 agreed (C1 arc direction; C7 stent expansion, largely by construction), 3 were inconclusive and 3 disagreed (C4 rates, C5 plain balloon in cadavers, C11 Fujino vs expansion) (`consistency.json`).

## 3. Self-review: the ten main weaknesses

Ranked by how much they limit the paper. The ones marked **model** cannot be fixed by rewriting.

1. **The AUC is partly expected by construction (model, rule 4 in spirit).**
   - The outcome is all or nothing, so "through-thickness damage" equals "AT1 damage onset before the peak load".
   - The best predictor is linear σ1 averaged over a radius equal to the phase-field length, which is close to the nucleation criterion of the outcome model.
   - The area-fraction predictors use the outcome model's own failure strain and stress.
   - Both models share the geometry and all material constants.

   The separation the brief requires (cheap pre-treatment analysis vs separate damage simulation) is met in the code. Its scientific value is limited: a high AUC shows the linear model anticipates the nonlinear one, not that it predicts fracture. The Discussion now says this.

2. **IVL is a static inflation in the outcome model (model, D14).** The equivalent load is below the plain-balloon pressure in every lesion, and IVL-damaged lesions are a subset of plain-balloon-damaged ones. The model reproduces no lithotripsy-specific effect, and the plain balloon damages 53% of lesions, against 0 of 7 in the cadaver study (check C5 disagrees).

3. **The calibration sets the calcium strength and the event rate (model, D8).** Failure is strain-controlled, so calibrating E to 365 MPa sets σ_crit = 7.3 MPa, about 41 times the measured calcification fracture stress of 0.179 MPa. The admissible window is a single point of the pilot sweep:
   - at 121 MPa the balloon alone damages 1 of 3 pilot lesions;
   - at 1100 MPa IVL damages none.

   Two of the three calibration geometries are core lesions included in every AUC. Excluding them changes nothing material (post hoc).

4. **The mechanics-over-Fujino headline was an artefact.** The 180° design level measures 179° on the label map. It falls just below the score's strict ">180°" cut-off and gets no arc points, although 9 of 9 such lesions were damaged. With a 2° tolerance the difference disappears. This is now reported (post hoc) and the claims are withdrawn.

5. **On the core set, mechanics cannot add information beyond geometry (design).** All core lesions share one reference geometry and one set of constants, so every mechanics predictor is a deterministic function of arc, thickness and morphology. The incremental-value test on the core compares model forms. Only the 24-lesion hold-out set, in which lumen area, eccentricity and lipid vary, can test added information, and it has not run yet.

6. **Damage spreads over the whole deposit; the toughness is numerical (model).**
   - Whole-deposit damage persisted in the Checkpoint 2 diagnostics with ℓ reduced to 20 µm, with a seeded crack and with heterogeneous toughness. Its cause was not identified.
   - Because ℓ = 2h and G_c follows from ℓ, the model is a strength-based nucleation model.
   - The phase-field outcome was never checked under mesh refinement.

7. **The lesions do not resist dilation; the stent is rigid (model).** Under 4 atm alone the lumen grows from 2.2 to 2.7–6.0 mm². The soft tissue recovers elastically, so fracture cannot give lumen gain. Every stent reaches its target, so expansion differs only through recoil. The stent analyses are uninformative and are now labelled as such.

8. **Numerical reliability.**
   - All 10 IVL runs with an unconverged staggered step are positives (10 of the 16 positives).
   - The transient solver that sets T_eq is unit-tested only, and its mesh error on the cohort meshes was not estimated.
   - The mesh study covered 7 of 12 mechanics predictors, and the non-local peak SED is 2.0002% against a 2% limit.

9. **Statistical design.**
   - The core set is a fixed factorial, so bootstrap intervals describe resampling of design points.
   - LOOCV keeps each lesion's near neighbours in the training set.
   - The best predictor was chosen on the same lesions (winner's curse).
   - Directions, the candidate list and the composite rule were fixed when preliminary outcomes existed.

   All of this is now stated in the text.

10. **The external evidence is thin, and some sources are weak.**
    - Only C1 (direction of the arc effect) is an informative, independent agreement.
    - 12 of 66 cited references are still incomplete after the web-search completion.
    - **Eccentricity threshold:** the row used for the hold-out eccentricity bound cites a 2016 paper (Komiyama et al.) that may not define the threshold. A 1996 Circulation paper (Mintz et al.) defines it explicitly; the search to confirm it hit the session limit.
    - **Cap thickness:** the cap-thickness rows match Jang et al. 2005, but the exact values were not seen in a snippet.

## 4. Decisions needed

**P1. Scope and framing (most important).** Options:

- **(a) Keep the model and reframe the paper (recommended).** The paper becomes an honest *in silico* study of how well a cheap linear stress proxy anticipates nonlinear damage onset in a designed cohort, with IVL modelled as an equivalent static load. The current corrected draft is written this way. Effort: finish the compute, then a final editing round.
- **(b) Fix the model before finishing.** This means:
  - a nonlinear representation of the pulse, so IVL differs from static inflation;
  - a damage law that localises, or a fracture-energy calibration independent of the mesh;
  - inelastic soft tissue, so fracture can give lumen gain;
  - a deformable stent.

  Weeks of development and a full cohort re-run; it would answer the question the brief set out to answer.
- **(c) Narrow the paper to the predictor-vs-outcome agreement.** Drop the stent and expansion claims entirely, and drop the "IVL" framing in favour of "balloon-type static loading".

**P2. Title.** "In Silico Digital Twins…" (D6) implies patient-specific models. Keep it, or change it to e.g. "In Silico Models of Calcified Coronary Lesions under an Equivalent Lithotripsy Load: Linear Stress Predictors of Calcium Damage".

**P3. Compute plan.**
- **Morris:** 216 runs, about 6–8 h, on a model with the limitations above. Run as designed, reduce it (e.g. 4 trajectories), or defer until P1 is settled.
- **Cohort:** completing it is needed under every option.

**P4. Further post-hoc analyses the reviewers suggested** (cheap; each will be labelled post hoc). These are proposals only; none has been run:
- an outcome-criterion benchmark: σ1_nl scaled to each lesion's peak load over σ_crit, showing how much of the AUC is expected by construction;
- a flexible geometry comparator: morphology indicators with arc as a category;
- the core IVL runs at two more moduli inside the calibration window, about 4 CPU-hours.

**P5. Items still awaiting confirmation from earlier work:**
- the LIFT name;
- the composite-rule amendment (rank by oriented AUC);
- the onset and censoring definitions;
- the consistency criteria;
- the D12 subset and axial mesh size;
- `ca_depth_min` as the depth predictor;
- nanoindentation modulus as SECONDARY;
- the split of the Fujino 0–3 row;
- the IVL pulse width, which is SECONDARY and not covered by D15.

**P6. References.**
- **Still incomplete (12):**
  - no DOI: holzapfel2002angioplasty, huang2001calcification, rooney1979arteries, rooney1982lesions, stent_specs;
  - authors missing: cvu2016eccentricity (placeholder title), mgh_oct_cap, qmul_carotid_acoustic, umb1998acoustic, dolfinx2023;
  - no volume: density_thermal2007, bmc2022nodulecta.
- **Faster completion:** allow-listing `api.crossref.org`, `doi.org` and `eutils.ncbi.nlm.nih.gov` under the environment's network settings would let me complete them from the DOI records directly.

## 5. What happens next without further input

The pipeline resumes on every container start (`restart_all.sh` in the session scratchpad):
1. the cohort finishes;
2. an interim analysis refreshes metrics, consistency, figures and tables on the full cohort;
3. the sensitivity queue runs (OAT, Morris, Sobol, 3D series), unless P3 says otherwise.

The manuscript text updates itself from the files at each build. I will not start new modelling work (P1b) or change the title (P2) without your decision.
