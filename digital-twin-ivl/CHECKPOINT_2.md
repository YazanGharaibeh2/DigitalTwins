# Checkpoint 2: cohort, verification, 3-lesion pilot, 3D example

Generated 2026-10-13 by `python -m dtivl.drivers.report_checkpoint2` from `results/*.json`, `cohort/cohort.csv` and the parameter files. No number below is typed by hand.
Nothing here is validated against patients; this checkpoint covers verification and a pilot.

## 0. Decisions needed before the full cohort run

**D8: calcium modulus regime (the main one).** The pilot shows that the result depends on which
calcium modulus is the baseline (Section 6). At the tissue-level baseline (secant modulus at failure),
calcium is only a few times stiffer than the fibrous matrix: it fails at about 1 atm under the
balloon alone, the whole deposit fails at once, and IVL adds nothing. That does not match the
clinical picture IVL was designed for (calcium that resists high-pressure balloons). Stiffer calcium
(particle/nanoindentation range) reverses this. Options:
  (a) keep the tissue-level baseline and report the regime problem as a finding;
  (b) baseline at a GPa-level modulus (literature nanoindentation range), with the strain-controlled
      criterion of D5 unchanged, and report the tissue-level case as a sensitivity;
  (c) choose the baseline as the modulus at which the 3 pilot lesions best match published qualitative
      observations (balloon alone rarely fractures; IVL fractures thin, wide arcs). This is calibration
      and would be reported as such, and those observations could then not be reused as a consistency check.
  My recommendation: (b), with (a) as the sensitivity. It keeps the calibration separate from the
  consistency checks.

**D9: crack filler (new assumption).** A fully damaged calcium band keeps the shear stiffness of the
fibrous matrix around it (degradation floor k = mu_fibrous / mu_calcium) instead of becoming a void.
Without it, stiff calcium (E >= 121 MPa) left near-zero-stiffness bands and the solver failed. G_c is
rescaled so the verified onset criterion is unchanged (Section 2.3). Sensitivity: k = k_residual
('none') where the solver allows. Approve, or ask for an alternative.

**D10: fracture outcomes when damage is diffuse.** In thin sheets under balloon pressure, damage spreads
over the whole deposit rather than forming discrete cracks (seeded cracks and a 10 % toughness
heterogeneity did not change this at the tested moduli). I added a "diffuse" label (a damaged zone
covering at least half of its deposit). Proposed primary outcomes: fracture yes/no, onset pressure,
damaged fraction and max depth. Number of fractures and radial/circumferential pattern would be reported
only where cracks are discrete. Approve, or keep crack count/pattern as primary outcomes.

**D11: Fujino range.** With D2 thickness levels (0.2, 0.4 mm) and no length in 2D, the core set spans
Fujino scores 0–3 (mostly 0 and 2; the 3s are nodules, whose thickness includes the protrusion). The
Fujino comparison in Phase 5 will be limited to arc. The IVL-treated lesions in Disrupt CAD III had a
mean maximum thickness near 1 mm. Options: keep D2 (and say so), or add a 0.8 mm level (+16 core lesions).

**D12: 3D length series (D3).** I propose running the 1/3/5 mm length series after this checkpoint,
on the subset you choose, because one 3D nonlinear run costs much more than a 2D run (Section 5).

**D13: erosion comparator.** The erosion model now completes. It uses the cell-averaged maximum
principal strain, the phase field uses an energy criterion, so onset can differ. Keep it as the
subset comparator?

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
| P-A90-T0.2-SUP | 8.95 | 1.18 | 1.16 | 100 % | 0.0104 | 0.0116 | no |
| P-A90-T0.2-SUP | 13.3 | 1.39 | 1.36 | 100 % | 0.0111 | 0.0121 | no |
| P-A90-T0.2-SUP | 121 | 2.32 | 2.17 | 0 % | 0.0292 | 0.0312 | no |
| P-A90-T0.2-SUP | 1100 | 2.94 | 2.66 | 0 % | 0.105 | 0.109 | no |
| P-A90-T0.2-SUP | 10000 | 3.08 | 2.78 | 0 % | 0.139 | 0.142 | no |
| P-A270-T0.4-DEEP | 1.47 | 0.269 | 0.287 | 100 % | 0.0760 | 0.0854 | no |
| P-A270-T0.4-DEEP | 8.95 | 0.812 | 0.831 | 100 % | 0.0294 | 0.0331 | no |
| P-A270-T0.4-DEEP | 13.3 | 0.948 | 0.984 | 100 % | 0.0261 | 0.0292 | no |
| P-A270-T0.4-DEEP | 121 | 1.93 | 2.01 | 0 % | 0.0330 | 0.0374 | no |
| P-A270-T0.4-DEEP | 1100 | 6.20 | 6.98 | 0 % | 0.0480 | 0.0565 | no |
| P-A270-T0.4-DEEP | 10000 | 13.1 | 15.7 | 0 % | 0.0561 | 0.0533 | no |
| P-A360-T0.2-NOD | 1.47 | 0.333 | 0.341 | 100 % | 0.112 | 0.119 | no |
| P-A360-T0.2-NOD | 8.95 | 1.24 | 1.26 | 98 % | 0.0281 | 0.0326 | no |
| P-A360-T0.2-NOD | 13.3 | 1.47 | 1.52 | 97 % | 0.0241 | 0.0272 | no |
| P-A360-T0.2-NOD | 121 | 2.21 | 2.36 | 0.46 % | 0.0795 | 0.0811 | no |
| P-A360-T0.2-NOD | 1100 | 2.36 | 2.46 | 0 % | 0.422 | 0.427 | no |
| P-A360-T0.2-NOD | 10000 | 2.39 | 2.47 | 0 % | 0.948 | 0.962 | no |

Figure: `figures/checkpoint2_transfer_factor.png`.

## 4. Pilot (3 lesions): nonlinear phase-field outcomes

Stent radial stiffness K_s = 6.07 MPa/mm, identified so that the calcium-free template (P-A90-T0.2-SUP) recoils by 4.30 % (clinical acute recoil). Onset is the first load step with max d >= d_crack; for stents the load is the cylinder radius (mm).

| Lesion | E (MPa) | Protocol | Status | Fracture onset | Through-thickness fractures (pattern) | Damaged calcium area | Max lumen area (mm²) | Stent expansion / recoil / deployment pressure |
|---|---|---|---|---|---|---|---|---|
| P-A90-T0.2-SUP | 1.47 | all | skipped | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A90-T0.2-SUP | 8.95 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 8.95 | IVL balloon only | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 8.95 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 8.95 | Stent, no IVL | ok | 0.927 mm | 1 (diffuse) | 100 % | 8.08 | 83.1 % / 4.27 % / 0.432 MPa |
| P-A90-T0.2-SUP | 8.95 | Stent after IVL | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.08 | 83.1 % / 4.27 % / 0.432 MPa |
| P-A90-T0.2-SUP | 8.95 | IVL, erosion comparator | ok | 0.0811 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 13.3 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.27 |  |
| P-A90-T0.2-SUP | 13.3 | IVL balloon only | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 13.3 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 121 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 121 | IVL balloon only | ok | 0.324 MPa | 1 (diffuse) | 100 % | 7.00 |  |
| P-A90-T0.2-SUP | 121 | IVL (balloon + pulse) | ok | 0.324 MPa | 1 (diffuse) | 100 % | 7.99 |  |
| P-A90-T0.2-SUP | 1100 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 1100 | IVL balloon only | ok | none | 0 (–) | 0 % | 5.10 |  |
| P-A90-T0.2-SUP | 1100 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 1100 | Stent, no IVL | ok | none | 0 (–) | 0 % | 8.08 | 81.6 % / 5.18 % / 0.531 MPa |
| P-A90-T0.2-SUP | 1100 | Stent after IVL | ok | none | 0 (–) | 0 % | 8.08 | 81.6 % / 5.18 % / 0.531 MPa |
| P-A90-T0.2-SUP | 1100 | IVL, erosion comparator | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A90-T0.2-SUP | 10000 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A90-T0.2-SUP | 10000 | IVL balloon only | ok | none | 0 (–) | 0 % | 5.08 |  |
| P-A90-T0.2-SUP | 10000 | IVL (balloon + pulse) | ok | none | 0 (–) | 0 % | 7.99 |  |
| P-A270-T0.4-DEEP | 1.47 | all | skipped | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |
| P-A270-T0.4-DEEP | 8.95 | Physiological pressure | ok | none | 0 (–) | 0 % | 2.26 |  |
| P-A270-T0.4-DEEP | 8.95 | IVL balloon only | ok | 0.122 MPa | 1 (diffuse) | 100 % | 7.56 |  |
| P-A270-T0.4-DEEP | 8.95 | IVL (balloon + pulse) | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A270-T0.4-DEEP | 8.95 | Stent, no IVL | ok | 0.947 mm | 1 (diffuse) | 100 % | 8.08 | 83.4 % / 4.13 % / 0.417 MPa |
| P-A270-T0.4-DEEP | 8.95 | Stent after IVL | ok | 0.122 MPa | 1 (diffuse) | 100 % | 8.08 | 83.4 % / 4.13 % / 0.417 MPa |
| P-A270-T0.4-DEEP | 8.95 | IVL, erosion comparator | ok | 0.0811 MPa | 1 (diffuse) | 100 % | 8.00 |  |
| P-A360-T0.2-NOD | 1.47 | all | skipped | crack filler undefined: calcium shear modulus 0.5638 MPa is not above the fibrous matrix (0.7707 MPa) |  |  |  |  |

Figure: `figures/checkpoint2_pilot_onset.png`.

## 5. 3D Fujino-4 example (X3D-A270-T0.6-L7-SUP) vs its 2D mid-section

Regularisation length in this example: 0.300 mm (coarser 3D mesh).

| Model | E (MPa) | cells | sigma1 P95 (MPa) | nonlocal peak sigma1 (MPa) | mean SED (MPa) | T_equivalent | IVL: status, onset | IVL damaged area |
|---|---|---|---|---|---|---|---|---|
| 3D | 8.95 | 64416 | 0.792 | 1.03 | 0.0206 | – | – | – |
| 3D | 1100 | 64416 | 5.22 | 7.92 | 0.00270 | – | – | – |
| 2D | 8.95 | 1952 | 0.765 | 0.807 | 0.0187 | – | – | – |
| 2D | 1100 | 1952 | 4.88 | 6.38 | 0.00222 | – | – | – |

## 6. What the pilot shows

* P-A90-T0.2-SUP, IVL balloon only: fracture at E = 8.95, 13.3, 121 MPa; no fracture at E = 1100, 10000 MPa.
* P-A90-T0.2-SUP, IVL (balloon + pulse): fracture at E = 8.95, 13.3, 121 MPa; no fracture at E = 1100, 10000 MPa.
* P-A270-T0.4-DEEP, IVL balloon only: fracture at E = 8.95 MPa; no fracture at E = none MPa.
* P-A270-T0.4-DEEP, IVL (balloon + pulse): fracture at E = 8.95 MPa; no fracture at E = none MPa.
* P-A360-T0.2-NOD, IVL balloon only: not run yet.
* P-A360-T0.2-NOD, IVL (balloon + pulse): not run yet.
  (IVL balloon pressure 0.405 MPa = 4.00 atm.)

* Baseline calcium modulus (E = 8.95 MPa, secant modulus at failure of tissue-level calcification) gives a failure stress of 0.179 MPa. Fractured outcomes with diffuse whole-deposit damage: 14; with discrete cracks: 0.
* Moduli skipped because the calcium is not stiffer than the fibrous matrix (E_fibrous = 2.31 MPa): P-A90-T0.2-SUP: 1.47 MPa; P-A270-T0.4-DEEP: 1.47 MPa; P-A360-T0.2-NOD: 1.47 MPa.
* Protocols that did not finish: none.
* Stent deployment uses the rated pressure 1.62 MPa as the expansion limit.

## 7. Consistency with published observations (preliminary, qualitative)

These are comparisons with published cohorts, not validation: no patient-specific outcome data exist for these idealised lesions.

| Observation | Published value [status] | Pilot |
|---|---|---|
| No fracture at physiological pressure | expected (no source needed) | 0 of 6 runs fracture |
| Balloon fracture predicted by arc > 225 deg and thickness < 0.240 mm (P-A90-T0.2-SUP: arc 89.0 deg, thickness 0.200 mm -> no fracture expected) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | balloon-only fracture at E = 8.95, 13.3, 121 of 8.95, 13.3, 121, 1100, 10000 MPa |
| Balloon fracture predicted by arc > 225 deg and thickness < 0.240 mm (P-A270-T0.4-DEEP: arc 269 deg, thickness 0.400 mm -> no fracture expected) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | balloon-only fracture at E = 8.95 of 8.95 MPa |
| Balloon fracture predicted by arc > 225 deg and thickness < 0.240 mm (P-A360-T0.2-NOD: arc 360 deg, thickness 0.500 mm -> no fracture expected) | 225 deg [VERIFIED]; 0.24 mm [SECONDARY] | balloon-only fracture at E = none of none MPa |
| Nodular calcium fractures less often after IVL than sheets | 0.24 (95% CI 0.08-0.70) OR [VERIFIED] | P-A90-T0.2-SUP: IVL fracture at E = 8.95, 13.3, 121 MPa; P-A270-T0.4-DEEP: IVL fracture at E = 8.95 MPa; P-A360-T0.2-NOD: IVL fracture at E = none MPa |
| Most IVL fractures are full thickness | 72 % of fractures [VERIFIED] | 14 of 14 damaged components through-thickness (14 diffuse) |
| Several fractures per lesion | 3.1 (SD 2.7) count [VERIFIED] | discrete crack count not defined when damage is diffuse (see D10) |
| Stent expansion at the minimum stent area | 78.4 (SD 25.8) % [VERIFIED] | 81.6-83.4 % over 6 stent runs |
| Lumen gain after IVL alone (before stenting) | 1.77 / 2.59 / 5.22 (SD 0.72 / 1.11 / 1.69) mm^2 [VERIFIED] | 2.21 vs 2.21 mm² (after IVL vs no IVL); 2.21 vs 2.21 mm² (after IVL vs no IVL); 2.21 vs 2.21 mm² (after IVL vs no IVL) — no residual gain: the model is hyperelastic with damage and has no permanent set |
