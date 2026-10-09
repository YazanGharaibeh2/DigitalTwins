# Study plan (one page)

**Question.** Can lesion-level mechanical metrics from a cheap pre-treatment analysis predict calcium fracture under IVL, and the resulting stent expansion, better than geometry alone, as summarised by the Fujino OCT calcium score [17]? The study uses a structured in silico cohort of calcified coronary cross-sections and segments. No clinical data are used.

**Cohort (Phase 2).** Lesions are generated as polar label maps in the IVOCT convention (0 = other, 1 = lumen, 4 = calcium, plus internal labels for fibrous tissue and lipid). They are meshed with Gmsh. The same mask-to-mesh function will be tested on hand-made masks so that real segmentations can plug in later.
- *Core factorial, 2D plane strain:* calcium arc {90, 180, 270, 360}° × thickness {4 levels; see decision D2} × morphology {superficial sheet, deep sheet, nodule, fragmented} = 64 lesions, with fixed lumen and wall.
- *3D coarse subset:* calcium length {4 levels; D3} combined with the other three factors in an L16 orthogonal array, giving 16 short segments (≥ 4 required). These test whether 2D conclusions hold and supply the full Fujino score, which needs length.
- *Space-filling hold-out:* about 30 lesions from a Latin hypercube (seeded) over continuous arc, thickness, depth, eccentricity, lumen diameter, fibrous-cap thickness and lipid inclusion. Ranges come from `params/literature_parameters.csv`. Thresholds derived on the core set are tested here.

**Models (Phase 3).** Soft tissues are modelled as follows. The wall uses Yeoh with verified media constants [30], lumped across layers. Fibrous tissue and lipid use neo-Hookean models from [28] and [34], nearly incompressible. Calcium is linear-elastic and brittle. Only rigid-body constraints are applied, with an optional elastic foundation (sensitivity). The load cases are:
1. Physiological pressure (baseline).
2. IVL: 4 atm balloon apposition plus pulse loading. (a) Quasi-static equivalent pressure *T* × 5 MPa, where the transfer factor *T* comes from (b) transient elastodynamic runs (2–5 µs pulse) on a subset. Both are reported. Pulse-count fatigue is omitted because no calcified-plaque fatigue data were found.
3. Stenting with and without prior IVL: a cylindrical envelope is expanded under displacement control to the target diameter and then released. Outputs are lumen gain, acute recoil, expansion %, and malapposition area.

**Fracture.** AT1 phase-field is applied in calcium only, with a volumetric–deviatoric split. The failure strain is the verified quantity (stretch 1.02 ± 0.005 [27]); the critical stress is E·ε_f, and G_c is derived from the AT1 nucleation relation and ℓ [50]. A stress-controlled scenario (σ_c = 179 kPa) is run as a sensitivity. On a subset, results are compared with max-principal-stress element erosion. Recorded per lesion: onset load, location, number of fractures, depth, and pattern (radial or circumferential).

**Non-circularity.** Predictors come only from geometry plus a *linear-elastic, undamaged* analysis (small-strain moduli, unit IVL load). Outcomes come only from the *nonlinear phase-field* runs. Shared inputs (geometry, calcium modulus) are acknowledged in Methods.

**Verification and plausibility (Phase 4).**
- Lamé check with < 1% error.
- Three-level mesh convergence with GCI [52] on peak calcium stress and SED.
- Sensitivity to ℓ, to the bulk/shear ratio K/μ, and to foundation stiffness.
- Morris screening of phase-field outcomes on 3 lesions; Sobol indices [53,54] for the linear predictors.
- Consistency checks against published observations, with agreement and disagreement reported plainly:
  - direction of the arc and thickness effects [9,18];
  - lower fracture in nodules [8,9];
  - lesion-level fracture rates of 67–79% [1,2], noting that OCT under-detects fractures [11,12];
  - expansion ranges [1];
  - calcium stress magnitudes reported for balloon loading [38].

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
