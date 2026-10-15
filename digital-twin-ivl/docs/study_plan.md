# Study plan (one page)

Title (D17, replaces D6): *In Silico Models of Calcified Coronary Lesions under an Equivalent Lithotripsy Load: Linear Stress Predictors of Calcium Damage*. Decisions D1–D7 (Checkpoint 1) and D8 (Checkpoint 2) are recorded in `params/study_config.yaml`.

**D8 (Checkpoint 2, 2026-10-13): calcium modulus baseline 365 MPa**, calibrated to the model-internal criterion that the IVL balloon inflation alone (4 atm) does not fracture the pilot lesions while IVL does (`materials.calcium.E`, status CALIBRATED). It is not calibrated to a published observation: at this modulus a 10-atm plain balloon still fractures 2 of 3 pilot lesions, unlike the cadaver study. With the D5 strain criterion the implied failure stress is 7.3 MPa (measured tissue-level strength 0.179 MPa). The tissue-level secant modulus (8.95 MPa, `materials.calcium.E_tissue_level`) and the rest of the 1.47 MPa–10 GPa sweep are reported as sensitivity analysis. **D9–D15 (Checkpoint 2, 2026-10-13):** recommendations accepted (crack filler with a no-filler sensitivity; primary outcomes = through-thickness damage yes/no and damage-onset pressure, with a time-boxed Rankine-type driving-force test; 3D length series to damage onset on an analyst-chosen subset; erosion on a subset; single-pulse IVL equivalence; SECONDARY-status exceptions with a density sensitivity). D11 amended by the PI: a 0.8 mm core thickness level is added (+16 core lesions). D10(b) result (`results/diag_rankine.json`): a Rankine-type (maximum principal strain) damage driving force gives the same whole-deposit damage and the same onset load increment as the energy force on the three pilot lesions at 365 MPa; it is kept only as the sensitivity option `fracture.driving_force`, and crack number and pattern are not reported (D10).

**Question.** Can lesion-level mechanical metrics from a cheap pre-treatment analysis predict calcium fracture under IVL, and the resulting stent expansion, better than geometry alone, as summarised by the Fujino OCT calcium score [17]? The study uses a structured in silico cohort of calcified coronary cross-sections and segments. No clinical data are used.

**Cohort (Phase 2).** Lesions are generated as polar label maps in the IVOCT convention (0 = other, 1 = lumen, 4 = calcium, plus internal labels for fibrous tissue and lipid). They are meshed with Gmsh. The same mask-to-mesh function will be tested on hand-made masks so that real segmentations can plug in later.
- *Core factorial, 2D plane strain:* calcium arc {90, 180, 270, 360}° × thickness {0.2, 0.4} mm (D2) × morphology {superficial sheet, deep sheet, nodule, fragmented} (D4) = 32 lesions, with fixed lumen and wall.
- *3D:* the 3D code is first tested on one example with Fujino score 4 (D1): arc 270°, thickness 0.6 mm, length 7 mm, each one step beyond the D2/D3 series. The 3D length levels {1, 3, 5} mm (D3) are run after Checkpoint 2, with the number of 3D runs set from the measured cost of the example.
- *Note:* with thickness ≤ 0.4 mm and length ≤ 5 mm, no core or 3D-level lesion exceeds the Fujino thickness (> 0.5 mm) or length (> 5 mm) criteria. The Fujino score of the cohort therefore takes only the values 0 and 2 (arc > 180°). This is raised for decision at Checkpoint 2.
- *Space-filling hold-out:* about 30 lesions from a Latin hypercube (seeded) over continuous arc, thickness, depth, eccentricity, lumen diameter, fibrous-cap thickness and lipid inclusion. Ranges come from `params/literature_parameters.csv`. Thresholds derived on the core set are tested here.

**Models (Phase 3).** Soft tissues are modelled as follows. The wall uses Yeoh with verified media constants [30], lumped across layers. Fibrous tissue and lipid use neo-Hookean models from [28] and [34], nearly incompressible. Calcium is linear-elastic and brittle. Only rigid-body constraints are applied, with an optional elastic foundation (sensitivity). The load cases are:
1. Physiological pressure (baseline).
2. IVL: 4 atm balloon apposition plus pulse loading. (a) Quasi-static equivalent pressure *T* × 5 MPa, where the transfer factor *T* comes from (b) transient elastodynamic runs (2–5 µs pulse) on a subset. Both are reported. Pulse-count fatigue is omitted because no calcified-plaque fatigue data were found.
3. Stenting with and without prior IVL: a cylindrical envelope is expanded under displacement control to the target diameter and then released. Outputs are lumen gain, acute recoil, expansion %, and malapposition area.

**Fracture.** AT1 phase-field is applied in calcium only, with a volumetric–deviatoric split. Per D5, the failure strain is the primary quantity (stretch 1.02 ± 0.005 [27]); the critical stress is E·ε_f, and G_c is derived from the AT1 nucleation relation and ℓ [50]. A stress-controlled scenario (σ_c = 179 kPa fixed) is run as a sensitivity. On a subset, results are compared with max-principal-stress element erosion. Recorded per lesion: onset load, location, number of fractures, depth, and pattern (radial or circumferential).

**Non-circularity.** Predictors come only from geometry plus a *linear-elastic, undamaged* analysis (small-strain moduli, unit IVL load). Outcomes come only from the *nonlinear phase-field* runs. Shared inputs (geometry, calcium modulus) are acknowledged in Methods.

**Verification and plausibility (Phase 4).**
- Lamé check with < 1% error.
- Three-level mesh convergence with GCI [52] on peak calcium stress and SED.
- One-at-a-time sensitivity on the three pilot lesions: ℓ, the bulk/shear ratio K/μ, foundation stiffness, IVL pulse scale, no crack filler (D9), the Rankine-type driving force (D10), the calcium density (D15) and the stress-controlled failure criterion (D5).
- Morris screening of the phase-field outcomes (damage-onset load, through-thickness damage under IVL and under the balloon alone) on 4 core lesions spanning arc, thickness and morphology: C-A90-T0.4-SUP, C-A180-T0.8-DEEP, C-A270-T0.2-NOD and C-A360-T0.4-FRAG. Inputs: calcium modulus, failure strain, IVL pulse scale, ℓ, foundation stiffness, calcium density (the only density varied), soft-tissue K/μ and fibrous modulus. The crack filler is varied one at a time only. Sobol indices [53,54] for the linear predictors.
- Consistency checks against published observations, with agreement and disagreement reported plainly:
  - direction of the arc and thickness effects [9,18];
  - lower fracture in nodules [8,9];
  - lesion-level fracture rates of 67–79% [1,2], noting that OCT under-detects fractures [11,12];
  - expansion ranges [1];
  - calcium stress magnitudes reported for balloon loading [38].
  - Implementation (Phase 5, `src/drivers/stage_consistency.py`, config section `consistency`, status DESIGN): checks C1–C12, one row per check and set in `results/consistency.csv` and `results/consistency.json`. Besides the items above: no fracture under the plain balloon of the cadaver study (expected disagreement, D8), the D8 balloon-alone criterion (labelled model-internal), acute recoil (labelled a calibration target), fractures per lesion and the full-thickness fraction, the direction of the Fujino-score effect on stent expansion, and the implied plaque burden against CTA (copied from `results/cohort_summary.json`). Each row states the published value as read from the literature table, the criterion, the agreement label (agrees, disagrees, partial, not computed) and the number of lesions; literature cells that are not a plain number are not parsed by hand, so their check is reported as not computed.

**Metrics and statistics (Phase 5).**
- Candidate predictors: peak and P95 max-principal stress, mean and peak SED, area fraction above thresholds, edge stress-concentration factor, and minimum thickness under high stress.
- Thresholds from ROC/Youden analysis with LOOCV and 2000-sample bootstrap CIs.
- A composite index: penalised logistic regression on ≤ 3 pre-specified predictors, given an acronym name.
- Comparison with the Fujino score: cross-sectional components in 2D, the full score in 3D.
- Incremental value: geometry-only versus geometry + mechanics models, compared by LOOCV AUC, log-loss and bootstrap ΔAUC, and then confirmed on the hold-out set. A null result will be reported as such.

**Software and reproducibility.** Python 3.12 with DOLFINx and Gmsh from conda-forge (reachable from this container), plus numpy, scipy, pandas, SALib, scikit-learn, matplotlib and python-docx. scikit-fem 12.0.2 is the pip fallback and has been confirmed to install here. Versions will be pinned in `requirements.txt` and `environment.yml` in Phase 2. All randomness uses seed 20261009. A single `run_all.py` regenerates every result, figure and the manuscript. `build_manuscript.py` reads every number from `results/*.json|csv`.

**Outputs (Phases 6–7).** 7 figures (≥ 300 dpi, colour-blind-safe), 4 tables, an MDPI-format Markdown and .docx manuscript, and `CHANGES.md` after the first draft.

**Main risks.**
- The calcium modulus spans about four orders of magnitude. *Mitigation:* the strain-controlled failure scenario plus a dedicated sensitivity sweep.
- No IVL attenuation or density data were found. *Mitigation:* the transient comparison is labelled conditional, and the pulse amplitude is varied ×0.5–1.
- 3D phase-field cost on 4 CPU cores. *Mitigation:* coarse meshes and an L16 subset of 16 runs; can drop to 8 if needed.
