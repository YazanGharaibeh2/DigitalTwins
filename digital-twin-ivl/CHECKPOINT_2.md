# Checkpoint 2: cohort, verification, 3-lesion pilot, 3D example

Generated 2026-10-13 by `python -m dtivl.drivers.report_checkpoint2` from `results/*.json`, `cohort/cohort.csv` and the parameter files. No number below is typed by hand.
Nothing here is validated against patients; this checkpoint covers verification and a pilot.

## 0. Decisions needed before the full cohort run

**D8: calcium modulus regime (the main decision).** The pilot outcome depends on the calcium modulus more than on anything else (Section 4.1). Per lesion, moduli (MPa) at which the calcium fractures:

| Lesion | IVL balloon alone (4 atm) | plain balloon (10 atm) | IVL (balloon + pulse) | no fracture even with IVL |
|---|---|---|---|---|
| P-A90-T0.2-SUP | 4.42, 8.95, 13.3, 40.2, 121 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 1100, 3317, 10000 |
| P-A270-T0.4-DEEP | 4.42, 8.95, 13.3, 40.2 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 3317 |
| P-A360-T0.2-NOD | 4.42, 8.95, 13.3, 40.2 | 4.42, 8.95, 13.3, 40.2, 121 | 4.42, 8.95, 13.3, 40.2, 121, 365 | 1100, 3317, 10000 |

* Baseline (E = 8.95 MPa, secant modulus at failure of tissue-level calcification; failure stress 0.179 MPa): the 4-atm balloon alone fractures every pilot lesion (P-A90-T0.2-SUP at 0.122 MPa = 1.20 atm; P-A270-T0.4-DEEP at 0.122 MPa = 1.20 atm; P-A360-T0.2-NOD at 0.122 MPa = 1.20 atm) and the whole deposit is damaged. IVL then adds nothing, and every lesion would have the same outcome, so the Phase 5 ROC analysis would have no events to discriminate.
* Published contrast: in human coronary cadaver lesions a plain balloon at 10 atm fractured 0 % (0 of 7 lesions; IVL 62.5 (P=0.01)) [VERIFIED]. Measured calcium moduli span tissue-level tension tests (1466 kPa, Loree 1994) to nanoindentation (22.40 GPa, IQR 17.70-27.55) [VERIFIED].
* Moduli at which no pilot lesion fractures under the 4-atm IVL balloon alone but at least one fractures with IVL: 365 MPa (lesions also differ in IVL outcome at: none). The stricter version with the 10-atm plain balloon of the cadaver study: none.

Options:
  (a) keep the tissue-level baseline and report that the model then cannot separate lesions;
  (b) use a GPa-level baseline (nanoindentation range): at the GPa moduli tested no pilot lesion fractures even with IVL, which also leaves nothing to predict and contradicts the IVL fracture rates;
  (c) calibrate: take the baseline from the moduli listed above (balloon alone does not fracture, IVL does), keep the strain criterion of D5 (failure stress = E eps_f), and report the whole sweep as the sensitivity analysis. The cadaver balloon-vs-IVL contrast is then used for calibration and cannot also be reported as a consistency check; the other comparisons (nodule effect, arc/thickness cut-offs, fracture rates, stent expansion) stay independent.
  Recommendation: (c). Candidate baselines: 365 MPa (meets the 4-atm contrast; no tested modulus meets the 10-atm one for every pilot lesion). This calibrates an uncertain input to one qualitative observation, and the manuscript would say so.

**D9: crack filler (new assumption).** A fully damaged calcium band keeps the shear stiffness of the fibrous matrix (degradation floor k = mu_fibrous / mu_calcium = 0.224 at the tissue-level baseline, 0.00549 at 365 MPa) instead of becoming a void. Without it the stiff-calcium runs (E >= 121 MPa) did not converge once a deposit cracked. G_c is rescaled so that the verified onset criterion is unchanged (Section 2.3). Moduli where calcium is not stiffer than the matrix are skipped. Proposed sensitivity: k = k_residual (no filler) where the solver allows. Approve, or ask for an alternative.

**D10: fracture outcomes and the damage model.** In the pilot the damage covers the whole deposit instead of forming discrete cracks. Diagnostics (Section 8): finer load steps, a seeded radial crack, a 10 % toughness heterogeneity and a 4x smaller regularisation length all end with the whole deposit damaged (all diffuse: yes). Damage starts as circumferential bands along the luminal surface, where the sheet is bent open. My explanation, not yet tested: the volumetric-deviatoric energy split lets damage degrade the whole high-energy luminal layer instead of opening a crack perpendicular to the hoop tension; option (b) would test it. The onset pressure is robust to the regularisation length: E = 8.95 MPa: 0.0880, 0.0860, 0.0840 MPa for ell = 0.0800, 0.0400, 0.0200 mm; E = 121 MPa: 0.322, 0.302, 0.290 MPa for ell = 0.0800, 0.0400, 0.0200 mm.
Options:
  (a) keep the model; primary outcomes = fracture yes/no, onset pressure and damaged fraction (all robust); crack number and radial/circumferential pattern not reported;
  (b) replace the energy split by a principal-strain (spectral) driving force, which opens cracks perpendicular to the maximum principal strain, and re-test localisation on the pilot before the full run (about one more pilot cycle);
  Recommendation: (a) for the main analysis, plus a time-boxed (b) on the three pilot lesions; if (b) gives discrete cracks, crack number and pattern become secondary outcomes.

**D11: Fujino range.** Core-set Fujino scores: 0: 16, 1: 2, 2: 12, 3: 2 (no 4s; with D2 thicknesses only nodules exceed 0.5 mm, and 2D has no length). The Phase 5 comparison with Fujino is then mostly a comparison with the arc criterion. IVL-treated lesions in Disrupt CAD III had a maximum thickness of 0.96 (SD 0.25) mm [VERIFIED]. Options: keep D2 and state the limitation, or add a 0.8 mm level (+16 core lesions).

**D12: 3D length series (D3).** I propose running the 1/3/5 mm series after this checkpoint on a subset you choose, and only up to fracture onset in 3D. Cost of the full IVL protocol on the Fujino-4 example: 3D not finished vs 2D section, E = 8.95 MPa: 72 s; 2D section, E = 1100 MPa: 7.3 s. With damage spreading, a 3D step costs far more (Section 5).

**D13: erosion comparator.** It now completes. Fracture onset under IVL: P-A90-T0.2-SUP E=8.95: phase field 0.122 MPa, erosion 0.0811 MPa; P-A90-T0.2-SUP E=1100: phase field none, erosion none; P-A270-T0.4-DEEP E=8.95: phase field 0.122 MPa, erosion 0.0811 MPa; P-A360-T0.2-NOD E=8.95: phase field 0.122 MPa, erosion 0.0811 MPa; P-A360-T0.2-NOD E=1100: phase field none, erosion none. Erosion uses the cell-averaged maximum principal strain, the phase field an energy criterion, so onset can differ. Keep erosion as the subset comparator?

## 1. In silico cohort (60 lesions)

| Set | Lesions | Fujino score: count |
|---|---|---|
| core | 32 | 0: 16, 1: 2, 2: 12, 3: 2 |
| holdout | 24 | 0: 10, 1: 2, 2: 8, 3: 4 |
| pilot | 3 | 0: 1, 2: 2 |
| example3d | 1 | 4: 1 |

Core reference geometry: lumen radius 0.837 mm, plaque thickness 0.848 mm, media + adventitia 0.567 mm, calcium depth 0.160 mm (superficial) and 0.340 mm (deep).
Implied plaque burden 80.6 %, inside the CT-angiography range of calcified plaques (77.0 %) to calcified nodules (85.4 %): yes.

Hold-out depths now follow the morphology label (deep sheets sample the deeper half of the literature depth range, other morphologies the shallower half):

| Morphology | min depth (mm) | max depth (mm) | n |
|---|---|---|---|
| deep_sheet | 0.250 | 0.330 | 6 |
| fragmented | 0.185 | 0.221 | 6 |
| nodule | 0.165 | 0.247 | 6 |
| superficial_sheet | 0.172 | 0.235 | 6 |

Hold-out lesions with a lipid pool: 8 of 24. The LHS lipid indicator selects half of them; pools are dropped when the calcium arc leaves no angular room opposite it.

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

| Case | E (MPa) | k_floor | psi_c theory (MPa) | psi onset measured (MPa) | rel. error | passed |
|---|---|---|---|---|---|---|
| 2D | 8.95 | 0.224 | 0.00179 | 0.001790 | 0.52 % | yes |
| 3D | 8.95 | 0.224 | 0.00179 | 0.001791 | 0.57 % | yes |

### 2.4 Mesh convergence (three grids, refinement factors 2.00, 1.00, 0.500; Celik GCI)

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

Acceptance: relative change below 2.0 % between the default and fine grids. The GCI is defined only for monotonic convergence (Celik et al.); for oscillatory or divergent triplets the relative change is the criterion. The raw peaks are singular at deposit corners and are not used as predictors; the nonlocal (critical-distance) peaks replace them.

## 3. Pilot (3 lesions): linear predictors and IVL transfer factor

Linear predictors at lumen pressure 0.405 MPa (the IVL balloon pressure), over the calcium-modulus sweep. T_equivalent is the transient peak maximum principal stress in calcium divided by the quasi-static stress under the same peak pressure; it scales the pulse in the quasi-static IVL step.

| Lesion | E (MPa) | sigma1 P95 (MPa) | nonlocal peak sigma1 (MPa) | calcium area with eps1 > eps_f | T | T_equivalent | peak at window end |
|---|---|---|---|---|---|---|---|
| P-A90-T0.2-SUP | 1.47 | 0.338 | 0.344 | 100 % | 0.0240 | 0.0264 | no |
| P-A90-T0.2-SUP | 4.42 | 0.804 | 0.785 | 100 % | 0.0134 | 0.0154 | no |
| P-A90-T0.2-SUP | 8.95 | 1.18 | 1.16 | 100 % | 0.0104 | 0.0116 | no |
| P-A90-T0.2-SUP | 13.3 | 1.39 | 1.36 | 100 % | 0.0111 | 0.0121 | no |
| P-A90-T0.2-SUP | 40.2 | 1.89 | 1.81 | 100 % | 0.0155 | 0.0166 | no |
| P-A90-T0.2-SUP | 121 | 2.32 | 2.17 | 0 % | 0.0292 | 0.0312 | no |
| P-A90-T0.2-SUP | 365 | 2.69 | 2.47 | 0 % | 0.0549 | 0.0578 | no |
| P-A90-T0.2-SUP | 1100 | 2.94 | 2.66 | 0 % | 0.105 | 0.109 | no |
| P-A90-T0.2-SUP | 3317 | 3.05 | 2.75 | 0 % | 0.147 | 0.151 | no |
| P-A90-T0.2-SUP | 10000 | 3.08 | 2.78 | 0 % | 0.139 | 0.142 | no |
| P-A270-T0.4-DEEP | 1.47 | 0.269 | 0.287 | 100 % | 0.0760 | 0.0854 | no |
| P-A270-T0.4-DEEP | 4.42 | 0.576 | 0.576 | 100 % | 0.0401 | 0.0456 | no |
| P-A270-T0.4-DEEP | 8.95 | 0.812 | 0.831 | 100 % | 0.0294 | 0.0331 | no |
| P-A270-T0.4-DEEP | 13.3 | 0.948 | 0.984 | 100 % | 0.0261 | 0.0292 | no |
| P-A270-T0.4-DEEP | 40.2 | 1.35 | 1.44 | 50 % | 0.0239 | 0.0264 | no |
| P-A270-T0.4-DEEP | 121 | 1.93 | 2.01 | 0 % | 0.0330 | 0.0374 | no |
| P-A270-T0.4-DEEP | 365 | 3.30 | 3.46 | 0 % | 0.0415 | 0.0518 | no |
| P-A270-T0.4-DEEP | 1100 | 6.20 | 6.98 | 0 % | 0.0480 | 0.0565 | no |
| P-A270-T0.4-DEEP | 3317 | 10.1 | 11.8 | 0 % | 0.0572 | 0.0608 | no |
| P-A270-T0.4-DEEP | 10000 | 13.1 | 15.7 | 0 % | 0.0561 | 0.0533 | no |
| P-A360-T0.2-NOD | 1.47 | 0.333 | 0.341 | 100 % | 0.112 | 0.119 | no |
| P-A360-T0.2-NOD | 4.42 | 0.830 | 0.840 | 100 % | 0.0441 | 0.0526 | no |
| P-A360-T0.2-NOD | 8.95 | 1.24 | 1.26 | 98 % | 0.0281 | 0.0326 | no |
| P-A360-T0.2-NOD | 13.3 | 1.47 | 1.52 | 97 % | 0.0241 | 0.0272 | no |
| P-A360-T0.2-NOD | 40.2 | 1.95 | 2.09 | 92 % | 0.0306 | 0.0321 | no |
| P-A360-T0.2-NOD | 121 | 2.21 | 2.36 | 0.46 % | 0.0795 | 0.0811 | no |
| P-A360-T0.2-NOD | 365 | 2.32 | 2.43 | 0 % | 0.195 | 0.198 | no |
| P-A360-T0.2-NOD | 1100 | 2.36 | 2.46 | 0 % | 0.422 | 0.427 | no |
| P-A360-T0.2-NOD | 3317 | 2.38 | 2.47 | 0 % | 0.763 | 0.774 | no |
| P-A360-T0.2-NOD | 10000 | 2.39 | 2.47 | 0 % | 0.948 | 0.962 | no |

Figure: `figures/checkpoint2_transfer_factor.png`.

## 4. Pilot (3 lesions): nonlinear phase-field outcomes

Stent radial stiffness K_s = 6.07 MPa/mm, identified so that the calcium-free template (P-A90-T0.2-SUP) recoils by 4.30 % (clinical acute recoil). Onset is the first load step with max d >= d_crack; for stents the load is the cylinder radius (mm). Onset is resolved only to the load step: pressure ramps use 10 increments (4 atm: 0.0405 MPa; 10 atm: 0.101 MPa), IVL inflation 10 increments to the balloon pressure and 10 to the pulse peak; increments are halved only when a step fails.

### 4.1 Fracture outcome by calcium modulus

B = IVL balloon alone (4 atm), P = plain balloon at 10 atm (same size), I = IVL (balloon + pulse), S = stent without IVL (run at two moduli only); – = no fracture.

| E (MPa) | P-A90-T0.2-SUP | P-A270-T0.4-DEEP | P-A360-T0.2-NOD |
|---|---|---|---|
| 1.47 | skipped | skipped | skipped |
| 4.42 | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse |
| 8.95 | B: diffuse; P: diffuse; I: diffuse; S: diffuse | B: diffuse; P: diffuse; I: diffuse; S: diffuse | B: diffuse; P: diffuse; I: diffuse; S: diffuse |
| 13.3 | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse |
| 40.2 | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse | B: diffuse; P: diffuse; I: diffuse |
| 121 | B: diffuse; P: diffuse; I: diffuse | B: –; P: diffuse; I: diffuse | B: –; P: diffuse; I: diffuse |
| 365 | B: –; P: diffuse; I: diffuse | B: –; P: diffuse; I: diffuse | B: –; P: –; I: diffuse |
| 1100 | B: –; P: –; I: –; S: – | not run | B: –; P: –; I: –; S: – (failed) |
| 3317 | B: –; P: –; I: – | B: –; P: –; I: – | B: –; P: –; I: – |
| 10000 | B: –; P: –; I: – | not run | B: –; P: –; I: – |

Moduli at which IVL fractures the calcium but the balloon alone does not: P-A270-T0.4-DEEP: 121, 365 MPa; P-A360-T0.2-NOD: 121, 365 MPa; P-A90-T0.2-SUP: 365 MPa.

| Lesion | E (MPa) | Protocol | Status | Fracture onset | Through-thickness fractures (pattern) | Damaged calcium area | Max lumen area (mm²) | Stent expansion / recoil / deployment pressure |
|---|---|---|---|---|---|---|---|---|
| P-A90-T0.2-SUP | 1.47 | all | skipped | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A90-T0.2-SUP | 4.42 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 4.42 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 4.42 | Plain balloon, 10 atm | ok | 0.101 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 4.42 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 8.95 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 8.95 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 8.95 | Plain balloon, 10 atm | ok | 0.101 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 8.95 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 8.95 | Stent, no IVL | ok | 0.927 mm | 1 (diffuse) | 100 % | 8.08 | 83.1 % / 4.27 % / 0.432 MPa |
| P-A90-T0.2-SUP | 8.95 | Stent after IVL | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.08 | 83.1 % / 4.27 % / 0.432 MPa |
| P-A90-T0.2-SUP | 8.95 | IVL, erosion comparator | ok | 0.0811 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 13.3 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 13.3 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 13.3 | Plain balloon, 10 atm | ok | 0.101 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 13.3 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 40.2 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 40.2 | IVL balloon only (4 atm) | ok | 0.162 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 40.2 | Plain balloon, 10 atm | ok | 0.203 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 40.2 | IVL (balloon + pulse) | ok | 0.162 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 121 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 121 | IVL balloon only (4 atm) | ok | 0.324 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 121 | Plain balloon, 10 atm | ok | 0.405 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 121 | IVL (balloon + pulse) | ok | 0.324 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 365 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 365 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 5.12 |  |
| P-A90-T0.2-SUP | 365 | Plain balloon, 10 atm | ok | 1.01 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 365 | IVL (balloon + pulse) | ok | 0.521 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A90-T0.2-SUP | 1100 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 1100 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 5.10 |  |
| P-A90-T0.2-SUP | 1100 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 1100 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 1100 | Stent, no IVL | ok | none | 0 (–) | 0 % | 8.08 | 81.6 % / 5.18 % / 0.531 MPa |
| P-A90-T0.2-SUP | 1100 | Stent after IVL | ok | none | 0 (–) | 0 % | 8.08 | 81.6 % / 5.18 % / 0.531 MPa |
| P-A90-T0.2-SUP | 1100 | IVL, erosion comparator | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 3317 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 3317 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 5.08 |  |
| P-A90-T0.2-SUP | 3317 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 3317 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 8.00 |  |
| P-A90-T0.2-SUP | 10000 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 10000 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 5.08 |  |
| P-A90-T0.2-SUP | 10000 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 10000 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A270-T0.4-DEEP | 1.47 | all | skipped | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A270-T0.4-DEEP | 4.42 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.27 |  |
| P-A270-T0.4-DEEP | 4.42 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 4.42 | Plain balloon, 10 atm | ok | 0.203 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 4.42 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 8.95 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A270-T0.4-DEEP | 8.95 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 8.95 | Plain balloon, 10 atm | ok | 0.203 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 8.95 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 8.95 | Stent, no IVL | ok | 0.947 mm | 1 (diffuse) | 100 % | 8.08 | 83.4 % / 4.13 % / 0.417 MPa |
| P-A270-T0.4-DEEP | 8.95 | Stent after IVL | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.08 | 83.4 % / 4.13 % / 0.417 MPa |
| P-A270-T0.4-DEEP | 8.95 | IVL, erosion comparator | ok | 0.0811 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 13.3 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.25 |  |
| P-A270-T0.4-DEEP | 13.3 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 13.3 | Plain balloon, 10 atm | ok | 0.203 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 13.3 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A270-T0.4-DEEP | 40.2 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.24 |  |
| P-A270-T0.4-DEEP | 40.2 | IVL balloon only (4 atm) | ok | 0.243 MPa | 1 (diffuse) | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 40.2 | Plain balloon, 10 atm | ok | 0.304 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 40.2 | IVL (balloon + pulse) | ok | 0.243 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A270-T0.4-DEEP | 121 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.24 |  |
| P-A270-T0.4-DEEP | 121 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 3.16 |  |
| P-A270-T0.4-DEEP | 121 | Plain balloon, 10 atm | ok | 0.507 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 121 | IVL (balloon + pulse) | ok | 0.443 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 365 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.24 |  |
| P-A270-T0.4-DEEP | 365 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 3.06 |  |
| P-A270-T0.4-DEEP | 365 | Plain balloon, 10 atm | ok | 0.608 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 365 | IVL (balloon + pulse) | ok | 0.612 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 3317 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.23 |  |
| P-A270-T0.4-DEEP | 3317 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 2.81 |  |
| P-A270-T0.4-DEEP | 3317 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 4.64 |  |
| P-A270-T0.4-DEEP | 3317 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 3.55 |  |
| P-A360-T0.2-NOD | 1.47 | all | skipped | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A360-T0.2-NOD | 4.42 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.11 |  |
| P-A360-T0.2-NOD | 4.42 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 4.42 | Plain balloon, 10 atm | ok | 0.101 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A360-T0.2-NOD | 4.42 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.93 |  |
| P-A360-T0.2-NOD | 8.95 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 8.95 | IVL balloon only (4 atm) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 8.95 | Plain balloon, 10 atm | ok | 0.101 MPa | 1 (diffuse) | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 8.95 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.90 |  |
| P-A360-T0.2-NOD | 8.95 | Stent, no IVL | ok | 0.775 mm | 1 (diffuse) | 100 % | 8.09 | 83.0 % / 4.33 % / 0.437 MPa |
| P-A360-T0.2-NOD | 8.95 | Stent after IVL | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.09 | 83.0 % / 4.33 % / 0.437 MPa |
| P-A360-T0.2-NOD | 8.95 | IVL, erosion comparator | ok | 0.0811 MPa | 1 (diffuse) | 100 % | 7.90 |  |
| P-A360-T0.2-NOD | 13.3 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 13.3 | IVL balloon only (4 atm) | ok | 0.0811 MPa | 1 (diffuse) | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 13.3 | Plain balloon, 10 atm | ok | 0.101 MPa | 1 (diffuse) | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 13.3 | IVL (balloon + pulse) | ok | 0.0811 MPa | 1 (diffuse) | 100 % | 7.89 |  |
| P-A360-T0.2-NOD | 40.2 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 40.2 | IVL balloon only (4 atm) | ok | 0.162 MPa | 1 (diffuse) | 100 % | 6.77 |  |
| P-A360-T0.2-NOD | 40.2 | Plain balloon, 10 atm | ok | 0.203 MPa | 1 (diffuse) | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 40.2 | IVL (balloon + pulse) | ok | 0.162 MPa | 1 (diffuse) | 100 % | 7.90 |  |
| P-A360-T0.2-NOD | 121 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 121 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 2.18 |  |
| P-A360-T0.2-NOD | 121 | Plain balloon, 10 atm | ok | 0.507 MPa | 1 (diffuse) | 100 % | 7.98 |  |
| P-A360-T0.2-NOD | 121 | IVL (balloon + pulse) | ok | 0.446 MPa | 1 (diffuse) | 100 % | 7.96 |  |
| P-A360-T0.2-NOD | 365 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 365 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 2.11 |  |
| P-A360-T0.2-NOD | 365 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 2.18 |  |
| P-A360-T0.2-NOD | 365 | IVL (balloon + pulse) | ok | 1.30 MPa | 1 (diffuse) | 100 % | 8.01 |  |
| P-A360-T0.2-NOD | 1100 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 1100 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 2.08 |  |
| P-A360-T0.2-NOD | 1100 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 2.12 |  |
| P-A360-T0.2-NOD | 1100 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 2.20 |  |
| P-A360-T0.2-NOD | 1100 | Stent, no IVL | failed | none | 0 (–) | 0 % | 2.12 | not valid: solver failed at cylinder radius 0.811 mm, mean contact pressure 1.35 MPa (83 % of rated) |
| P-A360-T0.2-NOD | 1100 | Stent after IVL | failed | none | 0 (–) | 0 % | 2.20 | not valid: solver failed at cylinder radius 0.811 mm, mean contact pressure 1.35 MPa (83 % of rated) |
| P-A360-T0.2-NOD | 1100 | IVL, erosion comparator | ok | none | 0 (–) | 0 % | 2.20 |  |
| P-A360-T0.2-NOD | 3317 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 3317 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 3317 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 3317 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 2.20 |  |
| P-A360-T0.2-NOD | 10000 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.06 |  |
| P-A360-T0.2-NOD | 10000 | IVL balloon only (4 atm) | ok | none | 0 (–) | 0 % | 2.07 |  |
| P-A360-T0.2-NOD | 10000 | Plain balloon, 10 atm | ok | none | 0 (–) | 0 % | 2.09 |  |
| P-A360-T0.2-NOD | 10000 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 2.19 |  |

Figure: `figures/checkpoint2_pilot_onset.png`.

## 5. 3D Fujino-4 example (X3D-A270-T0.6-L7-SUP) vs its 2D mid-section

Regularisation length in this example: 0.300 mm (coarser 3D mesh).

| Model | E (MPa) | cells | sigma1 P95 (MPa) | nonlocal peak sigma1 (MPa) | mean SED (MPa) | T_equivalent | IVL: status, onset | IVL damaged area |
|---|---|---|---|---|---|---|---|---|
| 3D | 8.95 | 64416 | 0.792 | 1.03 | 0.0206 | – | – | – |
| 3D | 1100 | 64416 | 5.22 | 7.92 | 0.00270 | – | – | – |
| 2D | 8.95 | 1952 | 0.765 | 0.807 | 0.0187 | 0.0399 | ok, 0.122 MPa | 100 % |
| 2D | 1100 | 1952 | 4.88 | 6.38 | 0.00222 | 0.0729 | ok, none | 0 % |

Balloon-pressure ramp to fracture onset (10 increments up to 0.405 MPa, IVL balloon cap), 3D vs the 2D mid-section (`results/pilot_example3d_onset.json`). The full 3D IVL protocol at the tissue-level baseline modulus was stopped before completion because its damage-propagation steps were too slow, and was replaced by this onset run.

| Model | E (MPa) | status | onset p (MPa) | damaged area at onset | pattern at onset | unconverged (u, d) steps | run time (s) |
|---|---|---|---|---|---|---|---|
| 2D | 8.95 | onset | 0.122 | 44.0 % | radial | 0 | 1.6 |

## 6. What the pilot shows

* P-A90-T0.2-SUP, IVL balloon only (4 atm): fracture at E = 4.42, 8.95, 13.3, 40.2, 121 MPa; no fracture at E = 365, 1100, 3317, 10000 MPa.
* P-A90-T0.2-SUP, IVL (balloon + pulse): fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; no fracture at E = 1100, 3317, 10000 MPa.
* P-A270-T0.4-DEEP, IVL balloon only (4 atm): fracture at E = 4.42, 8.95, 13.3, 40.2 MPa; no fracture at E = 121, 365, 3317 MPa.
* P-A270-T0.4-DEEP, IVL (balloon + pulse): fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; no fracture at E = 3317 MPa.
* P-A360-T0.2-NOD, IVL balloon only (4 atm): fracture at E = 4.42, 8.95, 13.3, 40.2 MPa; no fracture at E = 121, 365, 1100, 3317, 10000 MPa.
* P-A360-T0.2-NOD, IVL (balloon + pulse): fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; no fracture at E = 1100, 3317, 10000 MPa.
  (IVL balloon pressure 0.405 MPa = 4.00 atm.)

* Baseline calcium modulus (E = 8.95 MPa, secant modulus at failure of tissue-level calcification) gives a failure stress of 0.179 MPa. Fractured outcomes with diffuse whole-deposit damage: 57; with discrete cracks: 0.
* Moduli skipped because the calcium is not stiffer than the fibrous matrix (E_fibrous = 2.31 MPa): P-A90-T0.2-SUP: 1.47 MPa; P-A270-T0.4-DEEP: 1.47 MPa; P-A360-T0.2-NOD: 1.47 MPa.
* Protocols that did not finish: P-A360-T0.2-NOD E=1100 stent_no_ivl (failed, phase failed); P-A360-T0.2-NOD E=1100 stent_post_ivl (failed, phase failed).
* Stent deployment uses the rated pressure 1.62 MPa as the expansion limit.

## 7. Consistency with published observations (preliminary, qualitative)

These are comparisons with published cohorts, not validation: no patient-specific outcome data exist for these idealised lesions.

| Observation | Published value [status] | Pilot |
|---|---|---|
| No fracture at physiological pressure | expected (no source needed) | 0 of 25 runs fracture |
| Fracture after high-pressure dilation (no IVL) predicted by arc > 225 deg and thickness < 0.240 mm (P-A90-T0.2-SUP: arc 89.0 deg, max thickness 0.200 mm -> no fracture expected) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | stent without IVL: fracture at E = 8.95 of 8.95, 1100 MPa |
| Fracture after high-pressure dilation (no IVL) predicted by arc > 225 deg and thickness < 0.240 mm (P-A270-T0.4-DEEP: arc 269 deg, max thickness 0.400 mm -> no fracture expected) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | stent without IVL: fracture at E = 8.95 of 8.95 MPa |
| Fracture after high-pressure dilation (no IVL) predicted by arc > 225 deg and thickness < 0.240 mm (P-A360-T0.2-NOD: arc 360 deg, max thickness 0.500 mm -> no fracture expected) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | stent without IVL: fracture at E = 8.95 of 8.95, 1100 MPa |
| Plain balloon at 10 atm does not fracture calcium; IVL does (cadaver histology) | 0 (0 of 7 lesions; IVL 62.5 (P=0.01)) % of lesions [VERIFIED] | P-A90-T0.2-SUP: 10-atm balloon fractures at E = 4.42, 8.95, 13.3, 40.2, 121, 365 of 4.42, 8.95, 13.3, 40.2, 121, 365, 1100, 3317, 10000 MPa; P-A270-T0.4-DEEP: 10-atm balloon fractures at E = 4.42, 8.95, 13.3, 40.2, 121, 365 of 4.42, 8.95, 13.3, 40.2, 121, 365, 3317 MPa; P-A360-T0.2-NOD: 10-atm balloon fractures at E = 4.42, 8.95, 13.3, 40.2, 121 of 4.42, 8.95, 13.3, 40.2, 121, 365, 1100, 3317, 10000 MPa (calibration target under D8 option (c)) |
| Nodular calcium fractures less often after IVL than sheets | 0.24 (95% CI 0.08-0.70) OR [VERIFIED] | P-A90-T0.2-SUP: IVL fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; P-A270-T0.4-DEEP: IVL fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa; P-A360-T0.2-NOD: IVL fracture at E = 4.42, 8.95, 13.3, 40.2, 121, 365 MPa |
| Most IVL fractures are full thickness | 72 % of fractures [VERIFIED] | 57 of 57 damaged components through-thickness (57 diffuse) |
| Several fractures per lesion | 3.1 (SD 2.7) count [VERIFIED] | discrete crack count not defined when damage is diffuse (see D10) |
| Stent expansion at the minimum stent area | 78.4 (SD 25.8) % [VERIFIED] | 81.6-83.4 % over 8 completed stent runs; 2 stent runs failed and are excluded |
| Lumen gain after IVL alone (before stenting) | 1.77 / 2.59 / 5.22 (SD 0.72 / 1.11 / 1.69) mm^2 [VERIFIED] | P-A90-T0.2-SUP E=8.95: 2.21 vs 2.21 mm² (after IVL vs no IVL); P-A90-T0.2-SUP E=1100: 2.21 vs 2.21 mm² (after IVL vs no IVL); P-A270-T0.4-DEEP E=8.95: 2.21 vs 2.21 mm² (after IVL vs no IVL); P-A360-T0.2-NOD E=8.95: 2.06 vs 2.06 mm² (after IVL vs no IVL); P-A360-T0.2-NOD E=1100: 2.06 vs 2.06 mm² (after IVL vs no IVL). No residual gain: the model is hyperelastic with damage and has no permanent set |

## 8. Why the damage is diffuse (diagnostics for D10)

Lesion P-A90-T0.2-SUP, IVL balloon cap, pressure steps of 0.00200 MPa up to 0.405 MPa (`results/diag_localisation.json`).

| E (MPa) | Experiment | first cracked step p (MPa) | pattern at that step | damaged area at the end | diffuse at the end | steps with unconverged (u, d) loop |
|---|---|---|---|---|---|---|
| 8.95 | fine steps | 0.0880 | circumferential, circumferential | 100 % | yes | 0 |
| 8.95 | seeded radial crack | 0.0880 | radial, circumferential, circumferential | 100 % | yes | 0 |
| 8.95 | G_c heterogeneity (CoV 0.10) | 0.0860 | circumferential, circumferential | 100 % | yes | 0 |
| 121 | fine steps | 0.322 | diffuse | 100 % | yes | 0 |
| 121 | seeded radial crack | 0.322 | diffuse | 100 % | yes | 0 |
| 121 | G_c heterogeneity (CoV 0.10) | 0.322 | diffuse | 100 % | yes | 0 |

Regularisation length refined with the calcium mesh (ell / h fixed; `results/diag_ell.json`):

| h_Ca (mm) | ell (mm) | calcium cells | E (MPa) | first cracked p (MPa) | damaged area at the end | diffuse | through-thickness components | patterns | unconverged (u, d) steps | run time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0400 | 0.0800 | 609 | 8.95 | 0.0880 | 100 % | yes | 1 | diffuse | 0 | 150 |
| 0.0400 | 0.0800 | 609 | 121 | 0.322 | 100 % | yes | 1 | diffuse | 0 | 113 |
| 0.0200 | 0.0400 | 2105 | 8.95 | 0.0860 | 100 % | yes | 1 | diffuse | 0 | 401 |
| 0.0200 | 0.0400 | 2105 | 121 | 0.302 | 100 % | yes | 1 | diffuse | 0 | 221 |
| 0.0100 | 0.0200 | 7972 | 8.95 | 0.0840 | 100 % | yes | 1 | diffuse | 8 | 1645 |
| 0.0100 | 0.0200 | 7972 | 121 | 0.290 | 100 % | yes | 1 | diffuse | 1 | 838 |
