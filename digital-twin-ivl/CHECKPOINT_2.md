# Checkpoint 2: cohort, verification, 3-lesion pilot, 3D example

Generated 2026-10-13 by `python -m dtivl.drivers.report_checkpoint2` from `results/*.json`, `cohort/cohort.csv` and the parameter files. Every result number is read from those files; the fixed text holds only the options and recommendations.
Nothing here is validated against patients; this checkpoint covers verification, a pilot and qualitative comparisons with published observations.

## 0. Decisions needed before the full cohort run

**D8: calcium modulus regime (the main decision).** The pilot outcome depends on the calcium modulus more than on anything else (Section 4.1). Moduli (MPa) at which each protocol produces through-thickness damage (completed runs only):

| Lesion | IVL balloon alone (4.00 atm) | plain balloon (10.0 atm) | IVL (balloon + pulse) | no through-thickness damage even with IVL |
|---|---|---|---|---|
| P-A90-T0.2-SUP | 4.42, 8.95, 13.3, 40.2, 121 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 1100, 3317, 10000 |
| P-A270-T0.4-DEEP | 4.42, 8.95, 13.3, 40.2 | 4.42, 8.95, 13.3, 40.2, 121, 365, 1100 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 1100, 3317, 10000 |
| P-A360-T0.2-NOD | 4.42, 8.95, 13.3, 40.2 | 4.42, 8.95, 13.3, 40.2, 121 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 1100, 3317, 10000 |

* Baseline E = 8.95 MPa (secant modulus at failure of tissue-level calcification, failure stress 179 kPa (SD 56 (123-235)) [VERIFIED]): the IVL balloon alone (4.00 atm) damages every pilot lesion through its thickness, with damage onset between P-A90-T0.2-SUP: 0.0811 and 0.122 MPa; P-A270-T0.4-DEEP: 0.0811 and 0.122 MPa; P-A360-T0.2-NOD: 0.0811 and 0.122 MPa (fine steps on P-A90-T0.2-SUP: 0.0880 MPa = 0.868 atm). IVL then adds nothing. Every pilot lesion has the same outcome; if that holds for the cohort, the Phase 5 ROC analysis would have no non-fracture cases.
* Published contrast: `calcium_fracture_rate_POBA_10atm_cadaver_histology` = 0 % of lesions (0 of 7 lesions (POBA) vs IVL 62.5 % of 8 lesions, P = 0.01) [VERIFIED]. Measured moduli: `calcified_tissue_tangential_modulus` = 1466 kPa (SD 1284) [VERIFIED]; `calcification_modulus_nanoindentation_carotid` = 22.40 GPa (IQR 17.70-27.55) [VERIFIED].
* Moduli at which no pilot lesion has through-thickness damage under the IVL balloon alone (4.00 atm) but at least one has it with IVL: 365 MPa. With the plain balloon (10.0 atm) of the cadaver study instead: none.

Options:
  (a) keep the tissue-level baseline and report that the model then cannot separate lesions;
  (b) use a GPa-level baseline: up to the largest tested modulus (10000 MPa, below the carotid nanoindentation value, which was not tested) no pilot lesion fractures even with IVL, which leaves nothing to predict and contradicts the clinical IVL fracture rates;
  (c) calibrate the modulus to a criterion and report the whole sweep as the sensitivity analysis, keeping the strain criterion of D5 (failure stress = E x eps_f).

  For (c) the only criterion the pilot can meet is a model-internal surrogate: inflating the IVL balloon (4.00 atm) without pulses should not fracture the calcium, IVL should. It is a modelling assumption, not a published observation. Candidate: 365 MPa. There the plain balloon (10.0 atm) fractures 2 of 3 pilot lesions (P-A90-T0.2-SUP, P-A270-T0.4-DEEP), so the published cadaver contrast is not met (D14 explains why). The implied failure stress is 7.30 MPa, against the measured tissue-level strength 179 kPa (SD 56 (123-235)) [VERIFIED]. IVL damage onset at the candidate: P-A90-T0.2-SUP: 0.521 MPa; P-A270-T0.4-DEEP: 0.612 MPa; P-A360-T0.2-NOD: 1.30 MPa. Runs at the candidate with unconverged (u, d) steps: P-A270-T0.4-DEEP balloon_10atm: 2; P-A270-T0.4-DEEP ivl: 2.
  The comparisons that stay independent of this choice are the nodule effect, the arc/thickness cut-offs and the fracture rates; stent expansion and recoil are not checks (Section 4).

  Recommendation: none of the three is satisfactory. I lean to (c) with the candidate above, labelled as calibration to a model-internal surrogate, with (a) and the full sweep reported as sensitivity, because only an intermediate modulus gives outcomes that can differ between lesions. The alternative is to report the regime dependence itself as the main finding. Your call.

**D9: crack filler (new assumption).** A fully damaged calcium band keeps the shear stiffness of the fibrous matrix (degradation floor k = mu_fibrous / mu_calcium = 0.224 at the tissue-level baseline, 0.00549 at 365 MPa) instead of becoming a void. Without it the displacement solver failed in stiff-calcium runs once a deposit cracked (seen during development; those runs were not kept). G_c is rescaled so that the verified onset criterion is unchanged (Section 2.3). The filler is undefined when calcium is not stiffer than the matrix, so the moduli 1.47 MPa were skipped; this includes the measured tissue-level modulus `calcified_tissue_tangential_modulus` (1466 kPa), the lower bound of the sensitivity range. Proposed sensitivity: k = k_residual (no filler) where the solver allows. Approve, or ask for an alternative.

**D10: fracture outcomes and the damage model.** Every fractured pilot run is diffuse (60 runs; 57 cover the whole deposit; 0 have discrete cracks). Diagnostics (Section 8): finer load steps, a seeded radial crack, a toughness heterogeneity (nominal coefficient of variation 10 %, realised 12.5 %) and a regularisation length reduced from 0.0800 to 0.0200 mm. End state diffuse in all of them: yes. At E = 8.95 MPa the first cracked step has 1.53 % of the calcium damaged (circumferential x2); at E = 121 MPa the first cracked step has 100 % of the calcium damaged (diffuse). My explanation, not yet tested: the volumetric-deviatoric energy split lets damage degrade a whole high-energy layer instead of opening a crack perpendicular to the hoop tension; option (b) would test it. Sensitivity of the damage onset to the regularisation length: E = 8.95 MPa: 0.0880, 0.0860, 0.0840 MPa for ell = 0.0800, 0.0400, 0.0200 mm (change -4.5 %; unconverged (u, d) steps in these runs: 8); E = 121 MPa: 0.322, 0.302, 0.290 MPa for ell = 0.0800, 0.0400, 0.0200 mm (change -9.9 %; unconverged (u, d) steps in these runs: 1); compare the 2.0 % mesh criterion of Section 2.4.
Options:
  (a) keep the model; primary outcomes = through-thickness damage yes/no and damage-onset pressure (resolved to the load step, and sensitive to the regularisation length as shown); damaged fraction is almost always the whole deposit and carries little information; crack number and radial/circumferential pattern not reported;
  (b) replace the energy split by a principal-strain (spectral) driving force, which opens cracks perpendicular to the maximum principal strain, and re-test localisation on the pilot before the full run;
  Recommendation: (a) for the main analysis plus a time-boxed (b) on the pilot lesions; if (b) gives discrete cracks, crack number and pattern become secondary outcomes.

**D11: Fujino range.** Core-set Fujino scores: 0: 16, 1: 2, 2: 12, 3: 2. Only 4 core lesions exceed the Fujino thickness threshold (0.500 mm): nodule with thickness 0.400 mm (the protrusion counts); 2D has no length criterion. The Phase 5 comparison with Fujino is then mostly a comparison with the arc criterion. IVL-treated lesions in Disrupt CAD III had a maximum thickness of 0.96 mm (SD 0.25) [VERIFIED]. Options: keep the D2 thickness levels and state the limitation, or add a thicker level (value for you to choose; +16 core lesions).

**D12: 3D length series (D3).** I propose running the 1/3/5 mm series after this checkpoint on a subset you choose, and in 3D only up to damage onset. Run times on the Fujino-4 example: full IVL 3D: not completed; 2D section at 8.95 MPa 72 s; 2D section at 1100 MPa 7.3 s. Onset-only run: 2D at 8.95 MPa 1.6 s; 3D at 8.95 MPa 2474 s. The 3D baseline-modulus IVL run was stopped (Section 5).

**D13: erosion comparator.** It now completes. Under IVL: P-A90-T0.2-SUP E = 8.95 MPa: phase field 0.122 MPa, 100 % damaged; erosion 0.0811 MPa, 100 % damaged, 0 partial components; P-A90-T0.2-SUP E = 1100 MPa: phase field no damage, 0 % damaged; erosion no damage, 0 % damaged, 0 partial components; P-A270-T0.4-DEEP E = 8.95 MPa: phase field 0.122 MPa, 100 % damaged; erosion 0.0811 MPa, 100 % damaged, 0 partial components; P-A270-T0.4-DEEP E = 1100 MPa: phase field no damage, 0 % damaged; erosion 0.688 MPa (partial only), 3.85 % damaged, 50 partial components; P-A360-T0.2-NOD E = 8.95 MPa: phase field 0.122 MPa, 100 % damaged; erosion 0.0811 MPa, 100 % damaged, 0 partial components; P-A360-T0.2-NOD E = 1100 MPa: phase field no damage, 0 % damaged; erosion no damage, 0 % damaged, 0 partial components. Where both damage the calcium, the onsets differ by about one load increment (0.0405 MPa). Erosion uses the cell-averaged maximum principal strain, the phase field an energy criterion; they disagree at the stiff modulus for the deep sheet, where only erosion damages the calcium. Keep erosion as the subset comparator?

**D14: how IVL is represented (assumption to accept or change).** The quasi-static IVL step adds T_equivalent x peak pressure 5 MPa (approx. 50 atm) [VERIFIED] x pulse scale 1.00 [ASSUMPTION; sensitivity 0.500, 1.00] to the balloon pressure 4 atm [VERIFIED], once (single-pulse peak equivalence, an assumption). It therefore exceeds the plain balloon (10.0 atm) only where T_equivalent > 0.122. Moduli where that holds: P-A90-T0.2-SUP: 3317, 10000 MPa (max T_equivalent 0.151); P-A270-T0.4-DEEP: none (max T_equivalent 0.0854); P-A360-T0.2-NOD: 365, 1100, 3317, 10000 MPa (max T_equivalent 0.962). Where it does not hold, the model cannot reproduce 'IVL fractures what a plain balloon at that pressure does not', whatever the modulus. The model has no cumulative effect of repeated pulses (no fatigue data for calcified plaque were found; `calcium_fatigue_SN_data`). Options: (a) keep the single-pulse equivalence and state the limitation; (b) add a pulse-count damage accumulation law, an unverified assumption; (c) treat the pulse scale as a calibration parameter. Recommendation: (a).

**D15: inputs used under a SECONDARY-status exception.** Rule 2 keeps SECONDARY values out of the main analysis unless you approve an exception. Current exceptions: `materials.density.wall` <- `density_arterial_wall_model_input` = 1075 kg/m^3 [SECONDARY]; `materials.density.fibrous` <- `density_arterial_wall_model_input` = 1075 kg/m^3 [SECONDARY]; `materials.density.lipid` <- `density_plaque_model_input` = 920 kg/m^3 [SECONDARY]; `materials.density.calcium` <- `density_calcified_plaque_model_input` = 2050 kg/m^3 (1450 (other FE input)) [SECONDARY]; `loads.physiological_pressure` <- `normal_systolic_blood_pressure_upper` = 120 mmHg (diastolic <80) [SECONDARY]. The densities enter T_equivalent and therefore every IVL outcome; the physiological pressure enters the physiological protocol. Options: approve them with a density sensitivity run (for example the alternative calcium density in that row's range column), or find verified sources first.

## 1. In silico cohort (60 lesions)

| Set | Lesions | Fujino score: count |
|---|---|---|
| core | 32 | 0: 16, 1: 2, 2: 12, 3: 2 |
| holdout | 24 | 0: 10, 1: 2, 2: 8, 3: 4 |
| pilot | 3 | 0: 1, 2: 2 |
| example3d | 1 | 4: 1 |

Core reference geometry: lumen radius 0.837 mm, plaque thickness 0.848 mm, media + adventitia 0.567 mm, calcium depth 0.160 mm (superficial) and 0.340 mm (deep).
Implied plaque burden 80.6 %, between the mean CT-angiography plaque burden of calcified plaques (77.05 % (CN 85.40; P=0.005) [VERIFIED]) and of calcified nodules (85.40 % (P=0.005 vs calcified plaque without CN) [VERIFIED]): yes.

Hold-out depths now follow the morphology label (deep sheets sample the deeper half of the literature depth range, other morphologies the shallower half):

| Morphology | min depth (mm) | max depth (mm) | n |
|---|---|---|---|
| deep_sheet | 0.250 | 0.330 | 6 |
| fragmented | 0.185 | 0.221 | 6 |
| nodule | 0.165 | 0.247 | 6 |
| superficial_sheet | 0.172 | 0.235 | 6 |

Hold-out lesions with a lipid pool: 8 of 24. The LHS lipid indicator selects a fraction 0.500 of them; a pool is dropped when the calcium arc leaves no angular room opposite it.

Figure: `figures/fig2_cohort_gallery.png` (core factorial set).

## 2. Verification

### 2.1 Linear elasticity: thick-walled cylinder vs Lamé (plane strain)

| Material | h (mm) | max rel. error u_r | max rel. error hoop stress | passed |
|---|---|---|---|---|
| calcium | 0.0250 | 0.015 % | 0.036 % | yes |
| fibrous | 0.0250 | 0.018 % | 0.57 % | yes |

Tolerance 1.0 %.

### 2.2 Finite strain: hyperelastic Lamé limit and rigid-cylinder contact

Hyperelastic (fibrous) small-load limit: max rel. error of u_r 0.14 % at the lumen, 0.029 % at the outer surface; passed: yes. Contact check passed: yes.

### 2.3 Phase-field (AT1) damage onset, homogeneous bar

| Case | E (MPa) | k_floor | psi_c theory (MPa) | psi onset measured, interpolated (MPa) | rel. error (interpolated) | rel. error (first damaged step) | passed |
|---|---|---|---|---|---|---|---|
| 2D | 8.95 | 0.224 | 0.00179 | 0.001790 | 0.025 % | 0.52 % | yes |
| 3D | 8.95 | 0.224 | 0.00179 | 0.001791 | 0.072 % | 0.57 % | yes |

Tolerance 5.0 % on the first-damaged-step error (energy and strain).

### 2.4 Mesh convergence (three grids; Celik GCI)

Nominal refinement factors 2.00, 1.00, 0.500; measured grid ratios C-A180-T0.2-SUP: r21 = 1.89, r32 = 1.60; C-A360-T0.4-NOD: r21 = 1.93, r32 = 1.85.

| Quantity | max rel. change default vs fine | max GCI (fine), monotonic cases only | convergence class | status |
|---|---|---|---|---|
| sigma1_peak | 5.9 % | n/a (no monotonic case) | C-A180-T0.2-SUP: divergent, C-A360-T0.4-NOD: divergent | mesh-dependent (not a predictor) |
| sigma1_peak_nl | 0.55 % | n/a (no monotonic case) | C-A180-T0.2-SUP: divergent, C-A360-T0.4-NOD: divergent | accepted |
| sigma1_p99 | 0.11 % | 3.5e-04 % | C-A180-T0.2-SUP: monotonic, C-A360-T0.4-NOD: oscillatory | accepted |
| sigma1_p95 | 0.31 % | 0.082 % | C-A180-T0.2-SUP: oscillatory, C-A360-T0.4-NOD: monotonic | accepted |
| sigma1_mean | 0.011 % | 0.021 % | C-A180-T0.2-SUP: monotonic, C-A360-T0.4-NOD: monotonic | accepted |
| sed_mean | 0.051 % | 6.4e-04 % | C-A180-T0.2-SUP: divergent, C-A360-T0.4-NOD: monotonic | accepted |
| sed_peak | 9.8 % | n/a (no monotonic case) | C-A180-T0.2-SUP: divergent, C-A360-T0.4-NOD: divergent | mesh-dependent (not a predictor) |
| sed_peak_nl | 0.58 % | n/a (no monotonic case) | C-A180-T0.2-SUP: divergent, C-A360-T0.4-NOD: oscillatory | accepted |
| scf_edge_nl | 0.26 % | 0.046 % | C-A180-T0.2-SUP: monotonic, C-A360-T0.4-NOD: converged | accepted |

Acceptance: relative change below 2.0 % between the default and fine grids. The GCI assumes monotonic convergence (Celik et al.); for oscillatory or divergent triplets the relative change is the criterion. The raw peaks are singular at deposit corners and are not used as predictors; the nonlocal (critical-distance) peaks replace them.
Non-monotonic accepted quantities whose formal Richardson numbers exceed the criterion (shown for transparency; Richardson extrapolation is not meaningful for these triplets): sigma1_peak_nl (C-A360-T0.4-NOD, divergent): formal GCI 10 %, error vs extrapolated value 8.0 %.

## 3. Pilot (3 lesions): linear predictors and the IVL load factor

### 3.1 Linear predictors (pre-treatment analysis) at lumen pressure 0.405 MPa

The threshold-area predictor uses the same failure strain eps_f as the damage model (a shared material input, as are the geometry and the moduli); no quantity from the damage simulation enters a predictor.

| Lesion | E (MPa) | sigma1 P95 (MPa) | nonlocal peak sigma1 (MPa) | calcium area with eps1 > eps_f |
|---|---|---|---|---|
| P-A90-T0.2-SUP | 1.47 | 0.338 | 0.344 | 100 % |
| P-A90-T0.2-SUP | 4.42 | 0.804 | 0.785 | 100 % |
| P-A90-T0.2-SUP | 8.95 | 1.18 | 1.16 | 100 % |
| P-A90-T0.2-SUP | 13.3 | 1.39 | 1.36 | 100 % |
| P-A90-T0.2-SUP | 40.2 | 1.89 | 1.81 | 100 % |
| P-A90-T0.2-SUP | 121 | 2.32 | 2.17 | 0 % |
| P-A90-T0.2-SUP | 365 | 2.69 | 2.47 | 0 % |
| P-A90-T0.2-SUP | 1100 | 2.94 | 2.66 | 0 % |
| P-A90-T0.2-SUP | 3317 | 3.05 | 2.75 | 0 % |
| P-A90-T0.2-SUP | 10000 | 3.08 | 2.78 | 0 % |
| P-A270-T0.4-DEEP | 1.47 | 0.269 | 0.287 | 100 % |
| P-A270-T0.4-DEEP | 4.42 | 0.576 | 0.576 | 100 % |
| P-A270-T0.4-DEEP | 8.95 | 0.812 | 0.831 | 100 % |
| P-A270-T0.4-DEEP | 13.3 | 0.948 | 0.984 | 100 % |
| P-A270-T0.4-DEEP | 40.2 | 1.35 | 1.44 | 50 % |
| P-A270-T0.4-DEEP | 121 | 1.93 | 2.01 | 0 % |
| P-A270-T0.4-DEEP | 365 | 3.30 | 3.46 | 0 % |
| P-A270-T0.4-DEEP | 1100 | 6.20 | 6.98 | 0 % |
| P-A270-T0.4-DEEP | 3317 | 10.1 | 11.8 | 0 % |
| P-A270-T0.4-DEEP | 10000 | 13.1 | 15.7 | 0 % |
| P-A360-T0.2-NOD | 1.47 | 0.333 | 0.341 | 100 % |
| P-A360-T0.2-NOD | 4.42 | 0.830 | 0.840 | >99.5 % |
| P-A360-T0.2-NOD | 8.95 | 1.24 | 1.26 | 98 % |
| P-A360-T0.2-NOD | 13.3 | 1.47 | 1.52 | 97 % |
| P-A360-T0.2-NOD | 40.2 | 1.95 | 2.09 | 92 % |
| P-A360-T0.2-NOD | 121 | 2.21 | 2.36 | 0.46 % |
| P-A360-T0.2-NOD | 365 | 2.32 | 2.43 | 0 % |
| P-A360-T0.2-NOD | 1100 | 2.36 | 2.46 | 0 % |
| P-A360-T0.2-NOD | 3317 | 2.38 | 2.47 | 0 % |
| P-A360-T0.2-NOD | 10000 | 2.39 | 2.47 | 0 % |

### 3.2 IVL load factor (input of the outcome model, not a predictor)

T = peak transient maximum principal stress in calcium / static stress of the same (transient-model) geometry under the peak pulse pressure. T_equivalent = the same transient peak / the static stress of the quasi-static outcome model, so that the quasi-static IVL step reproduces the transient peak. T_equivalent sets the pulse load of every IVL outcome run and is excluded from the Phase 5 predictor set. It depends on tissue densities used under a SECONDARY-status exception (`density_arterial_wall_model_input`, `density_plaque_model_input`, `density_calcified_plaque_model_input`), so every IVL outcome is conditional on them (D15). T is mesh-sensitive at the default calcium mesh (notes in `src/solvers/transient.py`) and was not part of the Section 2.4 convergence study.

| Lesion | E (MPa) | T | T_equivalent | peak at window end |
|---|---|---|---|---|
| P-A90-T0.2-SUP | 1.47 | 0.0240 | 0.0264 | no |
| P-A90-T0.2-SUP | 4.42 | 0.0134 | 0.0154 | no |
| P-A90-T0.2-SUP | 8.95 | 0.0104 | 0.0116 | no |
| P-A90-T0.2-SUP | 13.3 | 0.0111 | 0.0121 | no |
| P-A90-T0.2-SUP | 40.2 | 0.0155 | 0.0166 | no |
| P-A90-T0.2-SUP | 121 | 0.0292 | 0.0312 | no |
| P-A90-T0.2-SUP | 365 | 0.0549 | 0.0578 | no |
| P-A90-T0.2-SUP | 1100 | 0.105 | 0.109 | no |
| P-A90-T0.2-SUP | 3317 | 0.147 | 0.151 | no |
| P-A90-T0.2-SUP | 10000 | 0.139 | 0.142 | no |
| P-A270-T0.4-DEEP | 1.47 | 0.0760 | 0.0854 | no |
| P-A270-T0.4-DEEP | 4.42 | 0.0401 | 0.0456 | no |
| P-A270-T0.4-DEEP | 8.95 | 0.0294 | 0.0331 | no |
| P-A270-T0.4-DEEP | 13.3 | 0.0261 | 0.0292 | no |
| P-A270-T0.4-DEEP | 40.2 | 0.0239 | 0.0264 | no |
| P-A270-T0.4-DEEP | 121 | 0.0330 | 0.0374 | no |
| P-A270-T0.4-DEEP | 365 | 0.0415 | 0.0518 | no |
| P-A270-T0.4-DEEP | 1100 | 0.0480 | 0.0565 | no |
| P-A270-T0.4-DEEP | 3317 | 0.0572 | 0.0608 | no |
| P-A270-T0.4-DEEP | 10000 | 0.0561 | 0.0533 | no |
| P-A360-T0.2-NOD | 1.47 | 0.112 | 0.119 | no |
| P-A360-T0.2-NOD | 4.42 | 0.0441 | 0.0526 | no |
| P-A360-T0.2-NOD | 8.95 | 0.0281 | 0.0326 | no |
| P-A360-T0.2-NOD | 13.3 | 0.0241 | 0.0272 | no |
| P-A360-T0.2-NOD | 40.2 | 0.0306 | 0.0321 | no |
| P-A360-T0.2-NOD | 121 | 0.0795 | 0.0811 | no |
| P-A360-T0.2-NOD | 365 | 0.195 | 0.198 | no |
| P-A360-T0.2-NOD | 1100 | 0.422 | 0.427 | no |
| P-A360-T0.2-NOD | 3317 | 0.763 | 0.774 | no |
| P-A360-T0.2-NOD | 10000 | 0.948 | 0.962 | no |

Figure: `figures/checkpoint2_transfer_factor.png`.

## 4. Pilot (3 lesions): nonlinear phase-field outcomes

Assumptions of these protocols: a balloon is a uniform follower pressure on the lumen with a rigid envelope cap at the IVL balloon radius (1.59 mm, from `geometry.ivl_balloon_diameter_mm`); the plain balloon has the same size as the IVL balloon (as in the cadaver study's balloon arm); 'damage onset' is the first load step at which a calcium cell reaches d >= 0.900; 'fracture' is a damaged component that spans the calcium thickness and cuts the deposit; a component covering at least 50 % of its calcium region is labelled diffuse (an analysis choice, decision D10) and still counts as a fracture. Model fracture is not the same quantity as fracture seen on histology or OCT.
Stent radial stiffness K_s = 6.07 MPa/mm, calibrated so that the calcium-free template (P-A90-T0.2-SUP) recoils by 4.3 % (SD 7.1) [VERIFIED]; recoil and stent expansion are therefore not consistency checks.
Onsets are resolved only to the load step: pressure ramps use 10 increments (IVL balloon alone (4.00 atm): 0.0405 MPa; plain balloon (10.0 atm): 0.101 MPa), IVL inflation 10 increments to the balloon pressure and 10 to the pulse peak, stent expansion 20 radius increments; an increment is halved only when a step fails.

### 4.1 Fracture outcome by calcium modulus

B = IVL balloon alone (4.00 atm), P = plain balloon (10.0 atm), I = IVL (balloon + pulse), S = stent without IVL (run at 2 moduli); – = no through-thickness damage; 'failed' = the solver stopped before the end of the protocol (no outcome).

| E (MPa) | P-A90-T0.2-SUP | P-A270-T0.4-DEEP | P-A360-T0.2-NOD |
|---|---|---|---|
| 1.47 | skipped | skipped | skipped |
| 4.42 | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse |
| 8.95 | B: diffuse; P: diffuse; I: diffuse; S: diffuse | B: diffuse; P: diffuse; I: diffuse; S: diffuse | B: diffuse; P: diffuse; I: diffuse; S: diffuse |
| 13.3 | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse |
| 40.2 | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse |
| 121 | B: diffuse; P: diffuse; I: diffuse | B: –; P: diffuse; I: diffuse | B: –; P: diffuse; I: diffuse |
| 365 | B: –; P: diffuse; I: diffuse | B: –; P: diffuse; I: diffuse | B: –; P: –; I: diffuse |
| 1100 | B: –; P: –; I: –; S: – | B: –; P: diffuse; I: –; S: diffuse | B: –; P: –; I: –; S: failed |
| 3317 | B: –; P: –; I: – | B: –; P: –; I: – | B: –; P: –; I: – |
| 10000 | B: –; P: –; I: – | B: –; P: –; I: – | B: –; P: –; I: – |

### 4.2 All protocol runs

| Lesion | E (MPa) | Protocol | Status | unconverged (u, d) steps | Damage onset | through-thickness / partial components | patterns | Damaged calcium area | Max lumen area (mm²) | Stent expansion / recoil / mean contact pressure at deployment |
|---|---|---|---|---|---|---|---|---|---|---|
| P-A90-T0.2-SUP | 1.47 | all | skipped |  | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A90-T0.2-SUP | 4.42 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 4.42 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 4.42 | plain balloon (10.0 atm) | ok | 0 | 0.101 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 4.42 | IVL (balloon + pulse) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 8.95 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 8.95 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 8.95 | plain balloon (10.0 atm) | ok | 0 | 0.101 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 8.95 | IVL (balloon + pulse) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 8.95 | Stent, no IVL | ok | not recorded | 0.927 mm | 1 / 0 | diffuse | 100 % | 8.08 | 83.1 % / 4.27 % / 0.432 MPa |
| P-A90-T0.2-SUP | 8.95 | Stent after IVL | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 8.08 | 83.1 % / 4.27 % / 0.432 MPa |
| P-A90-T0.2-SUP | 8.95 | IVL, erosion comparator | ok | not recorded | 0.0811 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 13.3 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 13.3 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 13.3 | plain balloon (10.0 atm) | ok | 0 | 0.101 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 13.3 | IVL (balloon + pulse) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 40.2 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 40.2 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.162 MPa | 1 / 0 | diffuse | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 40.2 | plain balloon (10.0 atm) | ok | 0 | 0.203 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 40.2 | IVL (balloon + pulse) | ok | not recorded | 0.162 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 121 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 121 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.324 MPa | 1 / 0 | diffuse | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 121 | plain balloon (10.0 atm) | ok | 0 | 0.405 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 121 | IVL (balloon + pulse) | ok | not recorded | 0.324 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 365 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 365 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 5.12 |  |
| P-A90-T0.2-SUP | 365 | plain balloon (10.0 atm) | ok | 0 | 1.01 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 365 | IVL (balloon + pulse) | ok | not recorded | 0.521 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 1100 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 1100 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 5.10 |  |
| P-A90-T0.2-SUP | 1100 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 1100 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 1100 | Stent, no IVL | ok | not recorded | no damage | 0 / 0 | – | 0 % | 8.08 | 81.6 % / 5.18 % / 0.531 MPa |
| P-A90-T0.2-SUP | 1100 | Stent after IVL | ok | not recorded | no damage | 0 / 0 | – | 0 % | 8.08 | 81.6 % / 5.18 % / 0.531 MPa |
| P-A90-T0.2-SUP | 1100 | IVL, erosion comparator | ok | not recorded | no damage | 0 / 0 | – | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 3317 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 3317 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 5.08 |  |
| P-A90-T0.2-SUP | 3317 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 3317 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 8.00 |  |
| P-A90-T0.2-SUP | 10000 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 10000 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 5.08 |  |
| P-A90-T0.2-SUP | 10000 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 10000 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 7.99 |  |
| P-A270-T0.4-DEEP | 1.47 | all | skipped |  | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A270-T0.4-DEEP | 4.42 | Physiological pressure | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.27 |  |
| P-A270-T0.4-DEEP | 4.42 | IVL balloon alone (4.00 atm) | ok | 0 | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 4.42 | plain balloon (10.0 atm) | ok | 0 | 0.203 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 4.42 | IVL (balloon + pulse) | ok | 0 | 0.122 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 8.95 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.26 |  |
| P-A270-T0.4-DEEP | 8.95 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 8.95 | plain balloon (10.0 atm) | ok | 1 | 0.203 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 8.95 | IVL (balloon + pulse) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 8.95 | Stent, no IVL | ok | not recorded | 0.947 mm | 1 / 0 | diffuse | 100 % | 8.08 | 83.4 % / 4.13 % / 0.417 MPa |
| P-A270-T0.4-DEEP | 8.95 | Stent after IVL | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 8.08 | 83.4 % / 4.13 % / 0.417 MPa |
| P-A270-T0.4-DEEP | 8.95 | IVL, erosion comparator | ok | not recorded | 0.0811 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 13.3 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.25 |  |
| P-A270-T0.4-DEEP | 13.3 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 13.3 | plain balloon (10.0 atm) | ok | 0 | 0.203 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 13.3 | IVL (balloon + pulse) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A270-T0.4-DEEP | 40.2 | Physiological pressure | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.24 |  |
| P-A270-T0.4-DEEP | 40.2 | IVL balloon alone (4.00 atm) | ok | 0 | 0.243 MPa | 1 / 0 | diffuse | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 40.2 | plain balloon (10.0 atm) | ok | 0 | 0.304 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 40.2 | IVL (balloon + pulse) | ok | 0 | 0.243 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A270-T0.4-DEEP | 121 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.24 |  |
| P-A270-T0.4-DEEP | 121 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 3.16 |  |
| P-A270-T0.4-DEEP | 121 | plain balloon (10.0 atm) | ok | 0 | 0.507 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 121 | IVL (balloon + pulse) | ok | not recorded | 0.443 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 365 | Physiological pressure | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.24 |  |
| P-A270-T0.4-DEEP | 365 | IVL balloon alone (4.00 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 3.06 |  |
| P-A270-T0.4-DEEP | 365 | plain balloon (10.0 atm) | ok | 2 | 0.608 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 365 | IVL (balloon + pulse) | ok | 2 | 0.612 MPa | 1 / 0 | diffuse | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 1100 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.23 |  |
| P-A270-T0.4-DEEP | 1100 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.94 |  |
| P-A270-T0.4-DEEP | 1100 | plain balloon (10.0 atm) | ok | 3 | 0.811 MPa | 1 / 0 | diffuse | 53.0 % | 8.00 |  |
| P-A270-T0.4-DEEP | 1100 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 3.87 |  |
| P-A270-T0.4-DEEP | 1100 | Stent, no IVL | ok | not recorded | 1.08 mm | 1 / 0 | diffuse | 62.6 % | 8.08 | 81.8 % / 5.03 % / 0.518 MPa |
| P-A270-T0.4-DEEP | 1100 | Stent after IVL | ok | not recorded | 1.08 mm | 1 / 0 | diffuse | 62.6 % | 8.08 | 81.8 % / 5.03 % / 0.518 MPa |
| P-A270-T0.4-DEEP | 1100 | IVL, erosion comparator | ok | not recorded | 0.688 MPa (partial only) | 0 / 50 | radial x6, circumferential x3, undetermined x41 | 3.85 % | 4.34 |  |
| P-A270-T0.4-DEEP | 3317 | Physiological pressure | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.23 |  |
| P-A270-T0.4-DEEP | 3317 | IVL balloon alone (4.00 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.81 |  |
| P-A270-T0.4-DEEP | 3317 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 4.64 |  |
| P-A270-T0.4-DEEP | 3317 | IVL (balloon + pulse) | ok | 0 | no damage | 0 / 0 | – | 0 % | 3.55 |  |
| P-A270-T0.4-DEEP | 10000 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.23 |  |
| P-A270-T0.4-DEEP | 10000 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.72 |  |
| P-A270-T0.4-DEEP | 10000 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 4.02 |  |
| P-A270-T0.4-DEEP | 10000 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 3.18 |  |
| P-A360-T0.2-NOD | 1.47 | all | skipped |  | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A360-T0.2-NOD | 4.42 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.11 |  |
| P-A360-T0.2-NOD | 4.42 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 4.42 | plain balloon (10.0 atm) | ok | 0 | 0.101 MPa | 1 / 0 | diffuse | 100 % | 7.99 |  |
| P-A360-T0.2-NOD | 4.42 | IVL (balloon + pulse) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.93 |  |
| P-A360-T0.2-NOD | 8.95 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 8.95 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 8.95 | plain balloon (10.0 atm) | ok | 0 | 0.101 MPa | 1 / 0 | diffuse | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 8.95 | IVL (balloon + pulse) | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 7.90 |  |
| P-A360-T0.2-NOD | 8.95 | Stent, no IVL | ok | not recorded | 0.775 mm | 1 / 0 | diffuse | 100 % | 8.09 | 83.0 % / 4.33 % / 0.437 MPa |
| P-A360-T0.2-NOD | 8.95 | Stent after IVL | ok | not recorded | 0.122 MPa | 1 / 0 | diffuse | 100 % | 8.09 | 83.0 % / 4.33 % / 0.437 MPa |
| P-A360-T0.2-NOD | 8.95 | IVL, erosion comparator | ok | not recorded | 0.0811 MPa | 1 / 0 | diffuse | 100 % | 7.90 |  |
| P-A360-T0.2-NOD | 13.3 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 13.3 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.0811 MPa | 1 / 0 | diffuse | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 13.3 | plain balloon (10.0 atm) | ok | 0 | 0.101 MPa | 1 / 0 | diffuse | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 13.3 | IVL (balloon + pulse) | ok | not recorded | 0.0811 MPa | 1 / 0 | diffuse | 100 % | 7.89 |  |
| P-A360-T0.2-NOD | 40.2 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 40.2 | IVL balloon alone (4.00 atm) | ok | not recorded | 0.162 MPa | 1 / 0 | diffuse | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 40.2 | plain balloon (10.0 atm) | ok | 0 | 0.203 MPa | 1 / 0 | diffuse | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 40.2 | IVL (balloon + pulse) | ok | not recorded | 0.162 MPa | 1 / 0 | diffuse | 100 % | 7.90 |  |
| P-A360-T0.2-NOD | 121 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 121 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.18 |  |
| P-A360-T0.2-NOD | 121 | plain balloon (10.0 atm) | ok | 0 | 0.507 MPa | 1 / 0 | diffuse | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 121 | IVL (balloon + pulse) | ok | not recorded | 0.446 MPa | 1 / 0 | diffuse | 100 % | 7.96 |  |
| P-A360-T0.2-NOD | 365 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 365 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.11 |  |
| P-A360-T0.2-NOD | 365 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.18 |  |
| P-A360-T0.2-NOD | 365 | IVL (balloon + pulse) | ok | not recorded | 1.30 MPa | 1 / 0 | diffuse | 100 % | 8.01 |  |
| P-A360-T0.2-NOD | 1100 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 1100 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.08 |  |
| P-A360-T0.2-NOD | 1100 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.12 |  |
| P-A360-T0.2-NOD | 1100 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.20 |  |
| P-A360-T0.2-NOD | 1100 | Stent, no IVL | failed | not recorded | no damage | 0 / 0 | – | 0 % | 2.12 | not valid: solver failed in stent expansion at R = 0.811 mm (mean contact pressure 1.35 MPa, 83 % of rated) |
| P-A360-T0.2-NOD | 1100 | Stent after IVL | failed | not recorded | no damage | 0 / 0 | – | 0 % | 2.20 | not valid: solver failed in stent expansion at R = 0.811 mm (mean contact pressure 1.35 MPa, 83 % of rated) |
| P-A360-T0.2-NOD | 1100 | IVL, erosion comparator | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.20 |  |
| P-A360-T0.2-NOD | 3317 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 3317 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 3317 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 3317 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.20 |  |
| P-A360-T0.2-NOD | 10000 | Physiological pressure | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 10000 | IVL balloon alone (4.00 atm) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 10000 | plain balloon (10.0 atm) | ok | 0 | no damage | 0 / 0 | – | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 10000 | IVL (balloon + pulse) | ok | not recorded | no damage | 0 / 0 | – | 0 % | 2.19 |  |

Figure: `figures/checkpoint2_pilot_onset.png`.

## 5. 3D Fujino-4 example (X3D-A270-T0.6-L7-SUP) vs its 2D mid-section

Regularisation length in this example: 0.300 mm (coarser 3D mesh). 'not run' = the analysis was not completed for that model and modulus (see the note below the table).

| Model | E (MPa) | cells | sigma1 P95 (MPa) | nonlocal peak sigma1 (MPa) | mean SED (MPa) | T_equivalent | IVL: status, damage onset | IVL damaged area | IVL run time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 3D | 8.95 | 64416 | 0.792 | 1.03 | 0.0206 | 0.0312 | not run | not run | not run |
| 3D | 1100 | 64416 | 5.22 | 7.92 | 0.00270 | not run | not run | not run | not run |
| 2D | 8.95 | 1952 | 0.765 | 0.807 | 0.0187 | 0.0399 | ok, 0.122 MPa | 100 % | 72 |
| 2D | 1100 | 1952 | 4.88 | 6.38 | 0.00222 | 0.0729 | ok, no damage | 0 % | 7.3 |

The full 3D IVL protocol at the tissue-level baseline modulus was stopped before completion because its damage-propagation steps were too slow; it was replaced by the onset run below. The 3D IVL run at the high modulus had not finished when this report was built (it runs alone because two 3D runs together exceed the memory limit); the report is regenerated when it finishes.

Balloon-pressure ramp to damage onset (10 increments up to 0.405 MPa, IVL balloon cap), 3D vs the 2D mid-section (`results/pilot_example3d_onset.json`):

| Model | E (MPa) | status | damage onset p (MPa) | damaged area at onset | through-thickness at onset | max depth at onset | pattern at onset | unconverged (u, d) steps | run time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 3D | 8.95 | onset | 0.122 | 43.2 % | no | 90.6 % | radial | 0 | 2474 |
| 2D | 8.95 | onset | 0.122 | 44.0 % | no | 90.6 % | radial | 0 | 1.6 |

## 6. What the pilot shows

* P-A90-T0.2-SUP, IVL balloon alone (4.00 atm): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2, 121 MPa; none at E = 365, 1100, 3317, 10000 MPa.
* P-A90-T0.2-SUP, plain balloon (10.0 atm): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; none at E = 1100, 3317, 10000 MPa.
* P-A90-T0.2-SUP, IVL (balloon + pulse): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; none at E = 1100, 3317, 10000 MPa.
* P-A270-T0.4-DEEP, IVL balloon alone (4.00 atm): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2 MPa; none at E = 121, 365, 1100, 3317, 10000 MPa.
* P-A270-T0.4-DEEP, plain balloon (10.0 atm): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2, 121, 365, 1100 MPa; none at E = 3317, 10000 MPa.
* P-A270-T0.4-DEEP, IVL (balloon + pulse): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; none at E = 1100, 3317, 10000 MPa.
* P-A360-T0.2-NOD, IVL balloon alone (4.00 atm): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2 MPa; none at E = 121, 365, 1100, 3317, 10000 MPa.
* P-A360-T0.2-NOD, plain balloon (10.0 atm): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2, 121 MPa; none at E = 365, 1100, 3317, 10000 MPa.
* P-A360-T0.2-NOD, IVL (balloon + pulse): through-thickness damage at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; none at E = 1100, 3317, 10000 MPa.

* Baseline calcium modulus E = 8.95 MPa (secant modulus at failure of tissue-level calcification; failure stress 179 kPa (SD 56 (123-235)) [VERIFIED]). Fractured outcomes that are diffuse: 60 (of which 57 cover the whole deposit); with discrete cracks: 0.
* Moduli skipped because the calcium is not stiffer than the fibrous matrix (fibrous modulus 2312 kPa (SD 2180) [VERIFIED]): P-A90-T0.2-SUP: 1.47 MPa; P-A270-T0.4-DEEP: 1.47 MPa; P-A360-T0.2-NOD: 1.47 MPa.
* Protocol runs that did not finish: P-A360-T0.2-NOD E = 1100 MPa stent_no_ivl (failed, in stent expansion at R = 0.811 mm); P-A360-T0.2-NOD E = 1100 MPa stent_post_ivl (failed, in stent expansion at R = 0.811 mm).
* Staggered (u, d) loop at its iteration limit (step accepted with a warning): 8 steps in 4 runs (P-A270-T0.4-DEEP E = 8.95 MPa balloon_10atm: 1; P-A270-T0.4-DEEP E = 365 MPa balloon_10atm: 2; P-A270-T0.4-DEEP E = 365 MPa ivl: 2; P-A270-T0.4-DEEP E = 1100 MPa balloon_10atm: 3). The flag was added during the pilot and is not recorded for 87 of 126 runs; it will be recorded for every run of the full cohort.

## 7. Consistency with published observations (preliminary, qualitative)

These are comparisons with published cohorts, not validation: no patient-specific outcome data exist for these idealised lesions.

| Observation | Published value [status] | Pilot | Agreement |
|---|---|---|---|
| No fracture at physiological pressure | expected; pressure 120 mmHg (diastolic <80) [SECONDARY] | 0 of 27 runs fracture | agrees |
| Fracture after high-pressure dilation without IVL predicted by arc > 225 deg and thickness < 0.240 mm (P-A90-T0.2-SUP: arc 89.0 deg, max thickness 0.200 mm, so no fracture expected) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | stent without IVL fractures at E = 8.95 MPa of 8.95, 1100 MPa | agrees at 1100 MPa; disagrees at 8.95 MPa |
| Fracture after high-pressure dilation without IVL predicted by arc > 225 deg and thickness < 0.240 mm (P-A270-T0.4-DEEP: arc 269 deg, max thickness 0.400 mm, so no fracture expected; decided by the SECONDARY thickness cut-off) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | stent without IVL fractures at E = 8.95, 1100 MPa of 8.95, 1100 MPa | agrees at none; disagrees at 8.95, 1100 MPa |
| Fracture after high-pressure dilation without IVL predicted by arc > 225 deg and thickness < 0.240 mm (P-A360-T0.2-NOD: arc 360 deg, max thickness 0.500 mm, so no fracture expected; decided by the SECONDARY thickness cut-off) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | stent without IVL fractures at E = 8.95 MPa of 8.95 MPa; failed (excluded) at 1100 MPa | agrees at none; disagrees at 8.95 MPa |
| Plain balloon at 10.0 atm does not fracture calcium; IVL does (cadaver histology) — P-A90-T0.2-SUP | 0 % of lesions (0 of 7 lesions (POBA) vs IVL 62.5 % of 8 lesions, P = 0.01) [VERIFIED] | plain balloon fractures at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; IVL at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa | met at none; not met at 4.42, 8.95, 13.3, 40.2, 121, 365, 1100, 3317, 10000 MPa |
| Plain balloon at 10.0 atm does not fracture calcium; IVL does (cadaver histology) — P-A270-T0.4-DEEP | 0 % of lesions (0 of 7 lesions (POBA) vs IVL 62.5 % of 8 lesions, P = 0.01) [VERIFIED] | plain balloon fractures at E = 4.42, 8.95, 13.3, 40.2, 121, 365, 1100 MPa; IVL at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa | met at none; not met at 4.42, 8.95, 13.3, 40.2, 121, 365, 1100, 3317, 10000 MPa |
| Plain balloon at 10.0 atm does not fracture calcium; IVL does (cadaver histology) — P-A360-T0.2-NOD | 0 % of lesions (0 of 7 lesions (POBA) vs IVL 62.5 % of 8 lesions, P = 0.01) [VERIFIED] | plain balloon fractures at E = 4.42, 8.95, 13.3, 40.2, 121 MPa; IVL at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa | met at 365 MPa; not met at 4.42, 8.95, 13.3, 40.2, 121, 1100, 3317, 10000 MPa |
| Nodular calcium fractures less often after IVL than sheets | 0.24 OR (95% CI 0.08-0.70) [VERIFIED] | P-A90-T0.2-SUP: IVL fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; P-A270-T0.4-DEEP: IVL fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; P-A360-T0.2-NOD: IVL fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa | disagrees (no nodule effect in the pilot) |
| Most IVL fractures are full thickness | 72 % of fractures [VERIFIED] | IVL protocol: 18 of 18 damaged components through-thickness, 18 of them diffuse | not comparable while damage is diffuse (D10) |
| Several fractures per lesion | 3.1 count (SD 2.7) [VERIFIED] | discrete crack count not defined when damage is diffuse | not comparable (D10) |
| Stent expansion at the minimum stent area | 78.4 % (SD 25.8) [VERIFIED] | 81.6-83.4 % (5 distinct values) | not a check: set by the target and reference diameters and the recoil calibration |
| Lumen gain after IVL alone (before stenting) | 1.77 / 2.59 / 5.22 mm^2 (SD 0.72 / 1.11 / 1.69) [VERIFIED] | P-A90-T0.2-SUP E = 8.95: 2.21 vs 2.21 mm² (after IVL vs no IVL); P-A90-T0.2-SUP E = 1100: 2.21 vs 2.21 mm² (after IVL vs no IVL); P-A270-T0.4-DEEP E = 8.95: 2.21 vs 2.21 mm² (after IVL vs no IVL); P-A270-T0.4-DEEP E = 1100: 2.21 vs 2.21 mm² (after IVL vs no IVL); P-A360-T0.2-NOD E = 8.95: 2.06 vs 2.06 mm² (after IVL vs no IVL) | disagrees: no residual gain (hyperelastic model with damage, no permanent set) |

## 8. Why the damage is diffuse (diagnostics for D10)

Lesion P-A90-T0.2-SUP, IVL balloon cap, pressure steps of 0.00200 MPa up to 0.405 MPa (`results/diag_localisation.json`). G_c heterogeneity: nominal coefficient of variation 10 %, realised 12.5 % (factor clipped to 0.721-1.30, correlation length 0.0800 mm).

| E (MPa) | Experiment | first cracked step p (MPa) | damaged area at that step | pattern at that step | damaged area at the end | diffuse at the end | steps with unconverged (u, d) loop |
|---|---|---|---|---|---|---|---|
| 8.95 | fine steps | 0.0880 | 1.53 % | circumferential x2 | 100 % | yes | 0 |
| 8.95 | seeded radial crack | 0.0880 | 14.0 % | radial, circumferential x2 | 100 % | yes | 0 |
| 8.95 | G_c heterogeneity | 0.0860 | 1.89 % | circumferential x2 | 100 % | yes | 0 |
| 121 | fine steps | 0.322 | 100 % | diffuse | 100 % | yes | 0 |
| 121 | seeded radial crack | 0.322 | 100 % | diffuse | 100 % | yes | 0 |
| 121 | G_c heterogeneity | 0.322 | 100 % | diffuse | 100 % | yes | 0 |

Regularisation length refined with the calcium mesh (ell / h fixed; `results/diag_ell.json`):

| h_Ca (mm) | ell (mm) | calcium cells | E (MPa) | first cracked p (MPa) | damaged area at the end | diffuse | through-thickness components | patterns | unconverged (u, d) steps | run time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0400 | 0.0800 | 609 | 8.95 | 0.0880 | 100 % | yes | 1 | diffuse | 0 | 150 |
| 0.0400 | 0.0800 | 609 | 121 | 0.322 | 100 % | yes | 1 | diffuse | 0 | 113 |
| 0.0200 | 0.0400 | 2105 | 8.95 | 0.0860 | 100 % | yes | 1 | diffuse | 0 | 401 |
| 0.0200 | 0.0400 | 2105 | 121 | 0.302 | 100 % | yes | 1 | diffuse | 0 | 221 |
| 0.0100 | 0.0200 | 7972 | 8.95 | 0.0840 | 100 % | yes | 1 | diffuse | 8 | 1645 |
| 0.0100 | 0.0200 | 7972 | 121 | 0.290 | 100 % | yes | 1 | diffuse | 1 | 838 |

## 9. Cost of the full cohort and plan after approval

Wall-clock time per pilot protocol run (2D, one process per run; several pilot processes shared the machine, so these are upper-side estimates; all moduli pooled):

| Protocol | runs | median (s) | max (s) |
|---|---|---|---|
| Physiological pressure | 27 | 4.7 | 12 |
| IVL balloon alone (4.00 atm) | 27 | 14 | 245 |
| plain balloon (10.0 atm) | 27 | 37 | 1112 |
| IVL (balloon + pulse) | 27 | 70 | 621 |
| Stent, no IVL | 6 | 290 | 5290 |
| Stent after IVL | 6 | 294 | 5753 |
| IVL, erosion comparator | 6 | 81 | 231 |

One lesion at one modulus with every protocol except erosion: 0.20 h (medians) to 3.6 h (maxima) of wall-clock time. The 56 core and hold-out lesions at one baseline modulus then need about 11-203 process-hours, i.e. 2.8-51 h on the 4 cores of this container, before the sensitivity analysis. Stiff-calcium stent runs dominate the upper bound.

Plan after your decisions: (1) fix the nodule stent solver failure at stiff calcium (Section 4) and re-run that case; (2) run the core and hold-out sets at the chosen baseline (2D, all protocols, with the convergence flag recorded for every run); (3) sensitivity analysis (Morris screening, then Sobol on the influential inputs) over calcium modulus, failure strain, pulse scale, regularisation length, crack filler, densities and perivascular support; (4) Phase 5 metrics (ROC/Youden with bootstrap CIs, LOOCV, composite index, comparison with Fujino); (5) 3D length series on the subset you choose. No full-cohort run starts before your approval.
