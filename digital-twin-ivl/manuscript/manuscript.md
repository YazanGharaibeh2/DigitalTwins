# In Silico Digital Twins of Calcified Coronary Lesions under Intravascular Lithotripsy and Stenting

## 1. Introduction

Severe coronary calcification resists balloon dilation and limits stent expansion, and stent underexpansion is the main mechanical consequence that lesion preparation tries to prevent [1,2]. Plain and cutting balloons and intravascular lithotripsy (IVL) all aim to fracture the calcium so that the vessel wall can expand [3–5]. Whether a given lesion will fracture is therefore a practical question before treatment.

Coronary IVL delivers acoustic pressure pulses from electrohydraulic emitters inside a semi-compliant balloon, which is inflated at only 4 atm (0.405 MPa) to appose the wall [6,7]. The peak positive pressure is about 5 MPa, and the proposed fracture mechanism is the compressive stress of the wavefront [5]. Optical coherence tomography (OCT) substudies found calcium fracture in 78.7 % of lesions in Disrupt CAD II [8] and in 67.4 % in Disrupt CAD III, with multiple fractures in 67.7 % of the fractured lesions; stent expansion was 101.7 ± 28.9 % at the site of maximum calcification and 78.4 ± 25.8 % at the site of minimum stent area [9]. A pooled OCT analysis of the Disrupt CAD studies reported 3.1 ± 2.7 fractures per lesion, 72 % of them through the full calcium thickness [10]. Fracture depends on morphology: it occurred at only 23.1 % of calcified-nodule sites in the pooled Disrupt CAD III and IV OCT substudies [11], and in a single-centre study it was associated with a larger calcium arc, a smaller minimum calcium thickness and superficial calcium, with lower odds for nodular calcium (odds ratio 0.24) [12]. Clinical fracture rates are probably lower bounds, since an ex vivo study found histological microfractures that OCT did not show [13].

Before treatment, the expected result is estimated from imaging. The Fujino OCT calcium score awards 2 points for a maximum calcium angle above 180°, 1 point for a maximum thickness above 0.5 mm and 1 point for a length above 5 mm; lesions with the maximum score of 4 had a stent expansion of 78 % against 96 % for lower scores [2]. A revised score adds a minimum calcium thickness above 0.30 mm [14], and machine learning on intravascular OCT (IVOCT) lesion attributes predicts underexpansion [15]. Fracture cut-offs were derived for plain balloons (calcium angle 225°) [3] and cutting balloons (198° and 0.445 mm) [4], whereas an IVL study reported associations without cut-offs [12]. All of these predictors are geometric. They contain neither the load nor the stiffness of the calcium and the surrounding tissue, nor the stress concentration at the ends of a deposit.

Finite element (FE) models have been used to study balloon angioplasty and stenting [16,17] and the effect of calcification on plaque stress [18]. In calcified lesions, the predicted stresses and damage depend strongly on the constitutive representation [19], heavily calcified lesions carry a larger share of the balloon load [20], and simulation-driven machine learning has predicted stent expansion [21]. Calcium fracture under plain and cutting balloons has been assessed through the maximum principal stress [22] or a brittle calcium model [23]. IVL studies prescribed the fracture and simulated its effect [13,24], so fracture was an input rather than a prediction. Variational phase-field models of brittle fracture can predict whether and where a crack forms without a prescribed path: they regularise the crack over a length $\ell$ [25], have robust staggered solvers [26], link crack nucleation to the material strength [27] and have been applied to arterial tissue [28]. Pipelines from intravascular images to FE models [29–31] bring lesion-specific digital twins, in the sense proposed for precision cardiology [32], within reach, and the in silico trial framework sets out how the credibility of such models is established [33].

To our knowledge, no study has tested whether lesion-level mechanics from a computationally cheap pre-treatment analysis predict IVL-induced fracture better than geometry or the Fujino score, in a controlled cohort in which the fracture comes from a separate simulation. Clinical data cannot answer this directly: morphology and device use vary together, OCT under-detects fractures, and each lesion is treated once. The inputs are also uncertain: calcium moduli range from about 1.5 MPa in tissue-level tension [34] to 10–25 GPa reported from nanoindentation [35], and tissue-level fracture data for calcified plaque come from a single uniaxial study [36].

This study addresses that question with a structured in silico cohort. A full factorial core set of 48 two-dimensional (2D) cross-sections combines 4 calcium arcs, 3 thicknesses and 4 morphologies (superficial sheet, deep sheet, calcified nodule, fragmented) [37]; a hold-out set of 24 lesions is sampled by Latin hypercube over continuous ranges that include lipid pools and fibrous caps. Lesions are polar label maps in the IVOCT segmentation convention [31,38], so segmented images can use the same pipeline. Predictors come from the geometry and from a linear-elastic analysis of the undamaged lesion under the IVL balloon pressure. Outcomes come from a separate nonlinear simulation with phase-field damage in the calcium, in which IVL is the quasi-static equivalent of a single acoustic pulse, followed by stent expansion; the primary outcome is through-thickness calcium damage under IVL. Predictors are evaluated by receiver operating characteristic (ROC) analysis with leave-one-out cross-validation (LOOCV) and bootstrap intervals, combined by penalised logistic regression on at most 3 mechanics predictors into the Lithotripsy-Induced Fracture Tendency (LIFT) index, and compared with the cross-sectional Fujino score. Geometry-only and geometry-plus-mechanics models give the incremental value of mechanics. Thresholds and models are derived on the core set and applied unchanged to the hold-out set. No patient data are used: the study offers verification against analytical solutions and under mesh refinement, consistency with published observations and sensitivity analysis of uncertain inputs, not a clinical test of the predictors.

## 2. Materials and Methods

### 2.1. Study Design

This is an in silico study; no patient data were used. A structured cohort of calcified coronary cross-sections was generated from polar label maps, and every lesion was analysed by two separate finite element models (Figure 1). The predictor model is a cheap pre-treatment analysis: the lesion geometry and one linear-elastic solve of the undamaged lesion under the IVL balloon pressure. The outcome model is a separate nonlinear simulation of the treatment with phase-field damage in the calcium: the IVL balloon alone, a plain balloon, IVL, and stent deployment with and without prior IVL. All predictors were computed from the predictor model and the geometry only, and all outcomes from the outcome model only. The two models share the lesion geometry and the material constants, including the calcium modulus; no quantity computed by the outcome model is used as a predictor. In particular, the IVL load factor of the transient pulse analysis (Section 2.5) is an input of the outcome model and is excluded from the candidate predictors by a check in the code. Every model input is listed with its value, source and status in Table 1; inputs that could not be verified in a primary source were not used in the main analysis.

### 2.2. Lesion Cohort

The core set is a full factorial of calcium arc (90, 180, 270 and 360°), calcium thickness (0.2, 0.4 and 0.8 mm) and morphology (superficial sheet, deep sheet, calcified nodule and fragmented calcium) [1,37], giving 48 lesions (Figure 2). All core lesions share one reference geometry: a circular lumen of area 2.2e+06 mm² [9], an external elastic membrane at the reference vessel radius (diameter 3.37e+03 mm) [39], and a wall (media and adventitia) of 567 mm [40,41], which gives a plaque burden of 80.6 %. Superficial sheets, nodules and fragments start at a depth of 1.6e+02 mm below the lumen and deep sheets at 3.4e+02 mm (the mean minus and plus one standard deviation of an ex vivo OCT series) [42]. A nodule adds a protrusion of height 3e+02 mm over a base arc of 60° into the lumen; fragmented calcium is split into one deposit per 45° of arc, each filling 60 % of its pitch. Where calcium, depth and a minimum fibrous layer of 50 mm above the internal elastic lamina did not fit within the reference plaque, the plaque was grown outward with the lumen unchanged (16 core lesions; decision D11).

The hold-out set of 24 lesions was drawn by Latin hypercube sampling (seed 20261009) over continuous ranges of calcium arc (60–360°), thickness (0.2–0.8 mm), depth, lumen area and plaque eccentricity, with the four morphologies in a balanced cycle and a lipid pool with a fibrous cap in half of the lesions; the lumen-area and depth ranges are the published mean ± one standard deviation [9,42]. Thresholds and models derived on the core set were applied to the hold-out set without change.

Each lesion was stored as a polar label map in the intravascular OCT convention (A-lines × radial samples; 360 A-lines, 10 µm radial sampling) [31,38], and all geometric descriptors were computed from the label map only, as they would be from a segmented image: total and largest contiguous calcium arc, maximum calcium thickness, depth, calcium area, number of deposits, lumen area, lipid arc and plaque eccentricity. The cross-sectional Fujino score awards 2 points for an arc above 180° and 1 point for a thickness above 0.5 mm [2]; in 2D the length criterion cannot be scored, so the score is at most 3.

### 2.3. Geometry and Meshing

The label map was converted into a conforming partition of lumen, calcium, lipid, fibrous intima (inside the internal elastic lamina) and wall, and meshed with linear triangles in Gmsh [43]. The element size was 40 µm in and next to the calcium and grew to 0.15 mm in the soft tissue. Displacements were interpolated with quadratic (P2) and the damage field with linear (P1) Lagrange elements, under plane strain. For the three-dimensional length series (Section 2.9), the cross-section was extruded over the calcified length with calcium-free end segments of 2000 mm, using in-plane sizes of 0.15 mm (calcium) and 0.4 mm (soft tissue) and axial sizes of 0.5 mm (calcified block) and 1 mm (end blocks).

### 2.4. Constitutive Models

The wall (media and adventitia, lumped) was modelled as an isotropic Yeoh solid with constants from human coronary layers [44]. The fibrous intima and the lipid pool were nearly incompressible neo-Hookean solids with small-strain Young moduli of 2.31 MPa [34] and 5.5 kPa [45]; the soft tissues used a bulk-to-shear modulus ratio of 50 (an assumption, varied in the sensitivity analysis). Calcium was linear elastic and brittle, with a Poisson ratio of 0.3 [35].

The calcium modulus is the most uncertain input: published values span more than three orders of magnitude, from 1.5 MPa for tissue-level tension [34] to the gigapascal range from nanoindentation [35]. The baseline was set to 365 MPa (decision D8), the point of the pilot modulus sweep at which the IVL balloon inflated alone did not damage the three pilot lesions while IVL did. This is a calibration to a criterion internal to the model, not to a measured property. With the failure strain below, it implies a calcium failure stress of 7.3 MPa, above the tissue-level strength of 0.18 MPa measured in tension [36], and at this modulus a plain balloon at 10 atm still fractured part of the pilot lesions, unlike in cadaver lesions [46]. The modulus was therefore varied in the sensitivity analysis (Section 2.9).

### 2.5. Boundary Conditions and Load Protocols

The lumen pressure acted as a follower load on the luminal surface. Rigid-body motion was removed by a weak elastic foundation on the outer boundary (0.0001 MPa/mm; its effect on the predictors is reported in Table 2). Balloons and the stent were rigid cylinders with frictionless penalty contact (penalty 1000 MPa/mm). Five protocols were run on every lesion. (i) Physiological pressure (16 kPa) [47]. (ii) The IVL balloon alone: lumen pressure to 4 atm [6] inside a cap at the radius of the IVL balloon (diameter 3.19e+03 mm) [39]. (iii) A plain balloon at 10 atm with the same cap, the balloon arm of a cadaver comparison [46]. (iv) IVL (below). (v) Stent deployment without and (vi) after IVL: a rigid cylinder expanded from just inside the smallest lumen radius to the target stent diameter (3.21e+03 mm) [39], stopping early if the mean contact pressure reached the rated balloon pressure (16 atm) [48]; the stent then recoiled to the radius at which the tissue pressure balanced a linear stent stiffness. The stiffness (6.07 MPa/mm) was identified so that a calcium-free version of a pilot lesion showed the clinical acute recoil of 4.3 % [49]. Damage was active in every protocol; stenting after IVL started from the IVL damage field. Loads were applied in increments that were halved on non-convergence, up to 6 times.

IVL was represented by its quasi-static equivalent (decision D14). The balloon was inflated to 4 atm inside the cap, the pressure was raised by one pulse-equivalent increment to the peak $p_{top} = p_{bal} + T_{eq}\,p_{peak}\,s$, and the lesion was unloaded. Here $p_{peak}$ = 5 MPa is the peak positive pressure of the pulse [5], $s$ = 1 a pulse-amplitude scale (varied in the sensitivity analysis), and $T_{eq}$ a lesion-specific transfer factor from a linear elastodynamic analysis of one pulse. That analysis applied a half-sine lumen pressure pulse [50] to the lesion with a non-reflecting outer boundary, used the measured sound speeds of the tissues to set their bulk moduli [51–53] with densities from modelling studies [54–56] (secondary sources, decision D15), and integrated in time with the non-dissipative generalised-α scheme and a mixed displacement-pressure formulation. $T_{eq}$ is the peak tensile principal stress in the calcium during the pulse divided by the peak of the same quantity in the quasi-static model under the same static pressure, so that the pulse-equivalent increment reproduces the transient peak. Repeated pulses and fatigue were not modelled.

### 2.6. Fracture Model

Calcium damage was modelled with the AT1 variational phase-field model of brittle fracture [25,27]. The damage $d$ ($d = 0$ intact, $d = 1$ broken) lives in the calcium only and is irreversible. The calcium strain energy was split into volumetric-deviatoric tensile and compressive parts, of which only the tensile part $\psi^+$ is degraded by $g(d) = (1-k)(1-d)^2 + k$, and the fracture energy is $\frac{3 G_c}{8}\int (d/\ell + \ell |\nabla d|^2)\,dx$. The residual stiffness $k$ was set so that fully damaged calcium keeps the shear stiffness of the surrounding fibrous matrix (decision D9; $k$ = 0.0055), and $G_c = 16\,\ell\,(1-k)\,\psi_{crit}/3$ places the AT1 damage onset at the critical energy $\psi_{crit}$ of a uniaxial failure strain $\varepsilon_f$ (decision D5). The failure strain $\varepsilon_f$ = 0.02 is the measured failure stretch of calcified tissue minus one [36]. The regularisation length was $\ell$ = 80 µm, 2 times the calcium element size. Each load increment was solved by alternate minimisation [26]: a Newton solve for the displacement with damage fixed, then a bound-constrained solve for the damage with displacement fixed, repeated until the largest damage change fell below 0.001 or 100 iterations were reached; increments that reached the iteration limit were counted and are reported.

As a comparator with a different fracture model (decision D13), IVL was also simulated with element erosion: after each converged increment, calcium cells whose cell-averaged maximum principal strain reached $\varepsilon_f$ lost their tensile stiffness, and the increment was re-solved until no further cells eroded. The comparator was run on 8 core lesions (two arcs, the thinnest and thickest levels, a sheet and a nodule).

### 2.7. Outcome Definitions

Calcium was considered damaged where the cell-averaged damage exceeded 0.9. The damage-onset load of a protocol is the lumen pressure of the first load step at which the maximum damage reached this value. A damaged component that connects the luminal and abluminal calcium surfaces is a through-thickness fracture; other damaged components are partial. A component covering at least half of its calcium deposit was labelled diffuse; it is still a through-thickness fracture when it connects the two surfaces. The primary outcome (decision D10) is through-thickness damage under IVL; secondary outcomes are the damage-onset load under IVL, through-thickness damage under the IVL balloon alone and under the plain balloon, and stent expansion with and without prior IVL. Stent expansion is the stent area after recoil divided by the reference lumen area (the area of a circle of the reference vessel diameter), in per cent; acute recoil is the relative loss of stent radius between deployment and recoil.

### 2.8. Pre-Treatment Mechanics Predictors

The predictor model is a small-strain, plane-strain linear-elastic solve of the undamaged lesion under a lumen pressure of 4 atm (the IVL balloon pressure), with the material constants of Section 2.4 linearised at zero strain. From the calcium cells it gives the maximum principal stress $\sigma_1$ (including the out-of-plane stress), the maximum principal strain $\varepsilon_1$ and the strain energy density (SED). The predictors were the area-weighted 95th and 99th percentiles and mean of $\sigma_1$; the mean SED; non-local peaks of $\sigma_1$ and SED, in which each cell value is replaced by the area-weighted mean over the calcium within a radius equal to the phase-field length (80 µm) before the maximum is taken; the calcium area fractions with $\varepsilon_1$ above $\varepsilon_f$ and $\varepsilon_f/2$ and with $\sigma_1$ above the failure stress and half of it; an edge stress concentration factor (the non-local peak $\sigma_1$ near deposit ends over the mean $\sigma_1$ of the calcium, within an angular window of at least 10° around each end); and the smallest local calcium thickness among the cells above the 90th percentile of $\sigma_1$. Raw cell peaks were not used as predictors because they did not converge under mesh refinement (Table 2).

### 2.9. Verification, Consistency and Sensitivity Analyses

Verification compared the solvers with analytical solutions: linear elasticity of a thick cylinder under pressure (Lamé solution), the small-load response of the hyperelastic solver, penalty contact with a rigid cylinder, and the AT1 damage onset under uniform strain. Discretisation error of the predictors was estimated on two lesions at three mesh levels with the grid convergence index [57]. Criteria and results are listed in Table 2.

Consistency with published observations was assessed by comparing model outcomes with reported clinical and experimental observations made in other lesions: the direction of the calcium arc and thickness effects on fracture, the lower fracture rate of nodules, lesion-level fracture rates after IVL, the absence of fracture under a plain balloon in cadaver lesions, stent expansion and recoil, the number of fractures per lesion and the association of the Fujino score with expansion. Each check has a criterion fixed in the configuration and a label (agrees, partial, disagrees, not computed). These comparisons test plausibility, not the accuracy of the predictions for any lesion.

Sensitivity analysis had four parts. One-at-a-time scenarios on the three pilot lesions (39 runs) removed the crack filler (D9), changed the phase-field length, the pulse scale, the foundation stiffness and the soft-tissue bulk modulus, replaced the damage driving force by a Rankine-type principal-strain force (D10), lowered the calcium density (D15) and switched to a stress-based failure criterion (D5). Morris elementary effects [58] screened 8 inputs of the outcome model on 4 core lesions (6 trajectories, 216 runs). First-order and total Sobol indices [58,59] of the 5 inputs that enter the predictor model were computed for every predictor on the same lesions. Finally, a three-dimensional length series (decision D12) extruded 1 core cross-section over calcified lengths of 1, 3, 5 mm and loaded it with the IVL ramp up to damage onset, to compare the onset load with the 2D model at the same in-plane resolution.

### 2.10. Statistical Analysis

All thresholds, predictor directions that were not fixed in advance, predictor selections and model fits were derived on the core set only. The hold-out set was used only to evaluate them unchanged. The candidate predictors were the 12 pre-treatment mechanics predictors, 8 geometric descriptors and the cross-sectional Fujino score. The quantities of the outcome model (the IVL load factor and every outcome column) were excluded from the candidates by a check in the code. Each candidate had a direction stated before the analysis (a higher value expected with damage, or a lower one for calcium thickness, depth and the minimum thickness under high stress). For descriptors without a stated direction, the direction was set on the core set and is flagged in the results.

For the primary outcome (through-thickness damage under IVL) and the binary secondary outcomes, discrimination was summarised by the area under the receiver operating characteristic curve (AUC) of the oriented predictor. The threshold maximised the Youden index, placed midway between adjacent observed values, with ties broken towards the more specific cut. Sensitivity, specificity and accuracy at the threshold are reported with Wilson score intervals. Intervals for the AUC and the threshold come from 2000 bootstrap resamples, stratified by outcome class, as two-sided 95 % percentile intervals. The threshold rule was also assessed by leave-one-out cross-validation (LOOCV): each lesion was classified with the threshold derived without it. When the AUC was $0$ or $1$, or every resample gave the same value, the percentile interval has zero width; such intervals are flagged as degenerate and are not read as statements of precision. A binary analysis required at least 2 lesions in each outcome class.

The composite index, named LIFT (Lithotripsy-Induced Fracture Tendency index), is a penalised logistic regression on at most 3 mechanics predictors chosen by a fixed rule. Candidates were the mechanics predictors without missing or constant values whose oriented AUC on the training rows was at least 0.60. They were ranked by that AUC and taken in rank order, skipping any candidate whose absolute Spearman correlation with a predictor already chosen exceeded 0.80. The chosen predictors were standardised, and an L2-penalised logistic regression with a fixed inverse penalty of 1.0 was fitted. The index is the model log-odds. Selection, fitting and the Youden threshold of the predicted probability were repeated inside every LOOCV fold and every bootstrap resample, so that the left-out lesion never contributed to its own classification. Out-of-fold performance is reported as the LOOCV AUC, the log-loss and the Brier score. The bootstrap interval of the LOOCV AUC resamples the fixed out-of-fold predictions and does not include the variability of refitting and reselection. Selected predictors whose fitted coefficient has the opposite sign to their stated direction are reported.

The composite index and the best single mechanics predictor were compared with the Fujino score by the paired bootstrap of the AUC difference. The incremental value of mechanics over geometry was tested by comparing a geometry-only penalised logistic model on the 4 descriptors of the core design factors (calcium arc, thickness, depth and number of deposits) with the same model plus the mechanics predictors chosen by the composite rule. The two models were compared by the apparent AUC difference, with both models refitted in every resample, by the LOOCV AUC, log-loss and Brier score (paired bootstrap of the out-of-fold predictions), and on the hold-out set with the core-fit models evaluated unchanged. A difference whose interval includes zero is reported as no evidence of incremental value.

For the damage-onset load under IVL, an event was the first load step at which the maximum damage reached the crack threshold. A lesion without onset was right-censored at its own peak equivalent IVL pressure, the largest load it received. The transient load factor enters only this censoring value, never a predictor. Discrimination was summarised by Harrell's concordance index (C) of the oriented predictor, where a higher oriented value should go with an earlier onset. For stent expansion with and without prior IVL, no direction was stated in advance. Spearman's correlation and C were computed on the raw predictor, their observed signs are reported, and these analyses are labelled exploratory. A ranked analysis required at least 5 lesions with the outcome.

As a cross-model check, the same predictors were scored against through-thickness damage under IVL computed with the element-erosion comparator on its lesion subset (Section 2.6), a fracture model separate from the phase field. All resampling used the study seed (20261009) plus a stage offset (500), with an independent random stream for each analysis, so that every interval is reproducible.

### 2.11. Software and Reproducibility

The models were implemented in Python 3.12.15 with DOLFINx 0.11.0 [60] and PETSc 3.25.6 (direct solver MUMPS), meshes were generated with Gmsh 4.15.2 [43], and the statistics used scikit-learn 1.9.1, SciPy 1.18.1 and SALib 1.6.0 [58]. A single script regenerates every result, figure, table and this manuscript from the configuration and the literature parameter table; all random draws are seeded. Every number in the text is read from a results or parameter file when the manuscript is built, and the build lists each number with its source and fails if the text contains a number that was not read from a file.

## References

1. Mintz, G. S. Intravascular Imaging of Coronary Calcification and Its Clinical Implications. *JACC: Cardiovascular Imaging* **2015**, *8*, 461–471.

2. Fujino, A.; Mintz, G. S.; Matsumura, M.; et al. A New Optical Coherence Tomography-Based Calcium Scoring System to Predict Stent Underexpansion. *EuroIntervention* **2018**, *13*, e2182–e2189. https://doi.org/10.4244/EIJ-D-17-00962.

3. Fujino, A.; Mintz, G. S.; Lee, T.; et al. Predictors of Calcium Fracture Derived From Balloon Angioplasty and its Effect on Stent Expansion Assessed by Optical Coherence Tomography. *JACC: Cardiovascular Interventions* **2018**, *11*, 1015–1017. https://doi.org/10.1016/j.jcin.2018.02.004.

4. Authors to be confirmed. Predictors for Calcium Fracture With a Novel Cutting Balloon: An Optical Coherence Tomography Study. *JACC: Cardiovascular Interventions* **2022**. https://doi.org/10.1016/j.jcin.2022.02.017.

5. Kereiakes, D. J.; Virmani, R.; Hokama, J. Y.; Illindala, U.; Mena-Hurtado, C.; Holden, A.; Hill, J. M.; Lyden, S. P.; Ali, Z. A. Principles of Intravascular Lithotripsy for Calcific Plaque Modification. *JACC: Cardiovascular Interventions* **2021**, *14*, 1275–1292. https://doi.org/10.1016/j.jcin.2021.03.036.

6. Tovar Forero, M. N.; Daemen, J. The Coronary Intravascular Lithotripsy System. *Interventional Cardiology Review* **2019**, *14*, 174–181. https://doi.org/10.15420/icr.2019.18.R1.

7. Karimi Galougahi, K.; et al. Calcific Plaque Modification by Acoustic Shock Waves: Intravascular Lithotripsy in Coronary Interventions. *Circulation: Cardiovascular Interventions* **2021**. https://doi.org/10.1161/CIRCINTERVENTIONS.120.009354.

8. Ali, Z. A.; et al. Safety and Effectiveness of Coronary Intravascular Lithotripsy for Treatment of Severely Calcified Coronary Stenoses: The Disrupt CAD II Study. *Circulation: Cardiovascular Interventions* **2019**, *12*, e008434. https://doi.org/10.1161/CIRCINTERVENTIONS.119.008434.

9. Hill, J. M.; et al. Intravascular Lithotripsy for Treatment of Severely Calcified Coronary Artery Disease. *Journal of the American College of Cardiology* **2020**. https://doi.org/10.1016/j.jacc.2020.09.603.

10. Authors to be confirmed. TCT-120: Pooled Optical Coherence Tomography Analysis of the Disrupt CAD Studies. *Journal of the American College of Cardiology* **2021**. https://doi.org/10.1016/j.jacc.2021.09.970.

11. Authors to be confirmed. Outcomes of Coronary Intravascular Lithotripsy for the Treatment of Calcified Nodules: A Pooled Analysis of the Disrupt CAD Studies. *EuroIntervention* **2024**. https://doi.org/10.4244/EIJ-D-24-00282.

12. Emori, H.; Shiono, Y.; Kuriyama, N.; Honda, Y.; Ebihara, S.; Kadooka, K.; et al. Calcium Fracture After Intravascular Lithotripsy as Assessed With Optical Coherence Tomography. *Circulation Journal* **2023**, *87*, 799–805. https://doi.org/10.1253/circj.CJ-22-0695.

13. Authors to be confirmed. Intravascular Laser Lithotripsy for Calcium Fracture in Human Coronary Arteries. *EuroIntervention*.

14. Sato, ?; et al. A Revised Optical Coherence Tomography-Derived Calcium Score to Predict Stent Underexpansion in Severely Calcified Lesions. *JACC: Cardiovascular Interventions* **2025**. https://doi.org/10.1016/j.jcin.2024.12.001.

15. Gharaibeh, Y.; Wilson, D. L. Prediction of Stent Under-Expansion in Calcified Coronary Arteries Using Machine Learning on Intravascular Optical Coherence Tomography Images. *Scientific Reports* **2023**, *13*. https://doi.org/10.1038/s41598-023-44610-9.

16. Holzapfel, G. A.; Stadler, M.; Schulze-Bauer, C. A. J. A Layer-Specific Three-Dimensional Model for the Simulation of Balloon Angioplasty Using Magnetic Resonance Imaging and Mechanical Testing. *Annals of Biomedical Engineering* **2002**, *30*, 753–767.

17. Lally, C.; Dolan, F.; Prendergast, P. J. Cardiovascular Stent Design and Vessel Stresses: A Finite Element Analysis. *Journal of Biomechanics* **2005**, *38*, 1574–1581.

18. Huang, H.; Virmani, R.; Younis, H.; Burke, A. P.; Kamm, R. D.; Lee, R. T. The Impact of Calcification on the Biomechanical Stability of Atherosclerotic Plaques. *Circulation* **2001**, *103*, 1051–1056.

19. Conway, C.; McGarry, J. P.; Edelman, E. R.; McHugh, P. E. Numerical Simulation of Stent Angioplasty with Predilation: An Investigation into Lesion Constitutive Representation and Calcification Influence. *Annals of Biomedical Engineering* **2017**, *45*, 2244–2252. https://doi.org/10.1007/s10439-017-1851-3.

20. Dong, P.; et al. Load-Sharing Characteristics of Stenting and Post-Dilation in Heavily Calcified Coronary Artery. *Scientific Reports* **2023**, *13*, 16878. https://doi.org/10.1038/s41598-023-43160-4.

21. Dong, P.; Ye, G.; Kaya, M.; Gu, L. Simulation-Driven Machine Learning for Predicting Stent Expansion in Calcified Coronary Artery. *Applied Sciences* **2020**, *10*, 5820. https://doi.org/10.3390/app10175820.

22. Zhu, X.; Umezu, M.; Iwasaki, K. Finite Element Analysis of Cutting Balloon Expansion in a Calcified Artery Model of Circular Angle 180\textdegree: Effects of Balloon-to-Diameter Ratio and Number of Blades Facing Calcification on Potential Calcification Fracturing and Perforation Reduction. *PLoS ONE* **2021**. https://doi.org/10.1371/journal.pone.0251404.

23. Li, J.; Cao, H.; Li, M.; Shu, L.; Lin, C. A Study of Balloon Type on Calcified Coronary Lesion Predilation: A Finite Element Analysis. *Proceedings of the Institution of Mechanical Engineers, Part H* **2023**. https://doi.org/10.1177/09544119231157853.

24. Sogbadji, J.; Kadry, K.; Poletti, G.; Berti, F.; Edelman, E. R.; Nezami, F. R. Impact of Lesion Preparation-Induced Calcified Plaque Defects in Vascular Intervention for Atherosclerotic Disease: In Silico Assessment. *Biomechanics and Modeling in Mechanobiology* **2025**. https://doi.org/10.1007/s10237-024-01923-6.

25. Bourdin, B.; Francfort, G. A.; Marigo, J.-J. Numerical Experiments in Revisited Brittle Fracture. *Journal of the Mechanics and Physics of Solids* **2000**, *48*, 797–826.

26. Miehe, C.; Hofacker, M.; Welschinger, F. A Phase Field Model for Rate-Independent Crack Propagation: Robust Algorithmic Implementation Based on Operator Splits. *Computer Methods in Applied Mechanics and Engineering* **2010**, *199*, 2765–2778. https://doi.org/10.1016/j.cma.2010.04.011.

27. Tanne, E.; Li, T.; Bourdin, B.; Marigo, J.-J.; Maurini, C. Crack Nucleation in Variational Phase-Field Models of Brittle Fracture. *Journal of the Mechanics and Physics of Solids* **2018**.

28. Gultekin, O.; Dal, H.; Holzapfel, G. A. A Phase-Field Approach to Model Fracture of Arterial Walls: Theory and Finite Element Analysis. *Computer Methods in Applied Mechanics and Engineering* **2016**, *312*, 542–566.

29. Kadry, K.; Olender, M. L.; Marlevi, D.; Edelman, E. R.; Nezami, F. R. A Platform for High-Fidelity Patient-Specific Structural Modelling of Atherosclerotic Arteries: From Intravascular Imaging to Three-Dimensional Stress Distributions. *Journal of the Royal Society Interface* **2021**, *18*, 20210436. https://doi.org/10.1098/rsif.2021.0436.

30. Straughan, R.; Kadry, K.; Parikh, S. A.; Edelman, E. R.; Nezami, F. R. Fully Automated Construction of Three-Dimensional Finite Element Simulations from Optical Coherence Tomography. *Computers in Biology and Medicine* **2023**, *165*, 107341. https://doi.org/10.1016/j.compbiomed.2023.107341.

31. Gharaibeh, Y.; et al. Deep Learning Segmentation of Coronary Calcified Plaque from Intravascular Optical Coherence Tomography (IVOCT) Images with Application to Finite Element Modeling of Stent Deployment. *Proc. SPIE Medical Imaging* **2019**, *10951*. https://doi.org/10.1117/12.2515256.

32. Corral-Acero, J.; Lamata, P. The `Digital Twin' to Enable the Vision of Precision Cardiology. *European Heart Journal* **2020**, *41*, 4556–4564. https://doi.org/10.1093/eurheartj/ehaa159.

33. Viceconti, M.; Pappalardo, F.; Rodriguez, B.; Horner, M.; Bischoff, J.; Musuamba Tshinanu, F. In Silico Trials: Verification, Validation and Uncertainty Quantification of Predictive Models Used in the Regulatory Evaluation of Biomedical Products. *Methods* **2021**, *185*, 120–127. https://doi.org/10.1016/j.ymeth.2020.01.011.

34. Loree, H. M.; Grodzinsky, A. J.; Park, S. Y.; Gibson, L. J.; Lee, R. T. Static Circumferential Tangential Modulus of Human Atherosclerotic Tissue. *Journal of Biomechanics* **1994**, *27*, 195–204.

35. Authors to be confirmed. Effect of Calcification Modulus and Geometry on Stress in Models of Calcified Atherosclerotic Plaque. *Cardiovascular Engineering and Technology* **2014**. https://doi.org/10.1007/s13239-014-0186-6.

36. Holzapfel, G. A.; Sommer, G.; Regitnig, P. Anisotropic Mechanical Properties of Tissue Components in Human Atherosclerotic Plaques. *Journal of Biomechanical Engineering* **2004**, *126*, 657–665.

37. Sugiyama, T.; Yamamoto, E.; Fracassi, F.; et al. Calcified Plaques in Patients With Acute Coronary Syndromes. *JACC: Cardiovascular Interventions* **2019**.

38. Lee, J.; et al. Segmentation of Coronary Calcified Plaque in Intravascular OCT Images Using a Two-Step Deep Learning Approach. *IEEE Access* **2020**.

39. Yamada, T.; Taniguchi, N.; Hata, T.; Nakajima, S.; Kawakami, S.; Takahashi, A. Impact of Calcified Plaque Distribution on Symmetrical Stent Expansion After Intravascular Lithotripsy. *Journal of the Society for Cardiovascular Angiography & Interventions* **2025**. https://doi.org/10.1016/j.jscai.2025.102616.

40. Konishi, ?; et al. The Histological Analysis of the Coronary Medial Thickness: Implications for Percutaneous Coronary Intervention. *PLoS ONE* **2023**, *18*, e0283840. https://doi.org/10.1371/journal.pone.0283840.

41. Rioufol, G.; et al. Adventitia Measurement in Coronary Artery: An in Vivo Intravascular Ultrasound Study. *Heart* **2006**, *92*, 985–986.

42. Authors to be confirmed. Volumetric Characterization of Human Coronary Calcification by Frequency-Domain Optical Coherence Tomography.

43. Geuzaine, C.; Remacle, J.-F. Gmsh: A 3-D Finite Element Mesh Generator with Built-in Pre- and Post-Processing Facilities. *International Journal for Numerical Methods in Engineering* **2009**, *79*, 1309–1331.

44. Escuer, J.; Martinez, M. A.; McGinty, S.; Pena, E. Mathematical Modelling of the Restenosis Process after Stent Implantation. *Journal of the Royal Society Interface* **2019**, *16*, 20190313. https://doi.org/10.1098/rsif.2019.0313.

45. Authors to be confirmed. Mapping Elasticity Moduli of Atherosclerotic Plaque in Situ via Atomic Force Microscopy. *Journal of Structural Biology* **2011**.

46. Kawai, K.; et al. Histology, OCT, and Micro-CT Evaluation of Coronary Calcification Treated With Intravascular Lithotripsy: Atherosclerotic Cadaver Study. *JACC: Cardiovascular Interventions* **2023**, *16*, 2097–2108. https://doi.org/10.1016/j.jcin.2023.06.021.

47. Whelton, P. K.; et al. 2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults.

48. Abbott Vascular; Boston Scientific. XIENCE and SYNERGY Coronary Stent System Specifications (Nominal and Rated Burst Pressures).

49. Authors to be confirmed. Comparison of in Vivo Acute Stent Recoil between the Bioabsorbable Everolimus-Eluting Coronary Stent and the Everolimus-Eluting Cobalt Chromium Coronary Stent.

50. Brinton, T. J.; et al. Feasibility of Shockwave Coronary Intravascular Lithotripsy for the Treatment of Calcified Coronary Stenoses: First Description. *Circulation* **2019**. https://doi.org/10.1161/CIRCULATIONAHA.118.036531.

51. Rooney, ?; et al. Speed of Sound in Excised Arteries (1979) and Arterial Lesions (1982).

52. Authors to be confirmed. Speed of Sound in Human Coronary Artery Walls at 50 MHz. *Brazilian Journal of Medical and Biological Research* **2002**.

53. Authors to be confirmed. Acoustic Properties of Atherosclerosis of Human Aorta Obtained with High-Frequency Ultrasound. *Ultrasound in Medicine and Biology* **1998**.

54. Authors to be confirmed. Minimizing Thermal Damage to Vascular Nerves While Drilling of Calcified Plaque. *BMC Research Notes* **2019**. https://doi.org/10.1186/s13104-019-4381-2.

55. Authors to be confirmed. [Stent-deployment finite element study with Ogden plaque models]. **2014**. https://doi.org/10.1186/s40759-014-0002-x.

56. Authors to be confirmed. [Thermal model of atherosclerotic arteries]. *BioMedical Engineering OnLine*. https://doi.org/10.1186/1475-925X-6-8.

57. Celik, I. B.; Ghia, U.; Roache, P. J.; Freitas, C. J.; Coleman, H.; Raad, P. E. Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications. *Journal of Fluids Engineering* **2008**, *130*, 078001. https://doi.org/10.1115/1.2960953.

58. Herman, J.; Usher, W. SALib: An Open-Source Python Library for Sensitivity Analysis. *Journal of Open Source Software* **2017**, *2*, 97. https://doi.org/10.21105/joss.00097.

59. Saltelli, A.; Annoni, P.; Azzini, I.; Campolongo, F.; Ratto, M.; Tarantola, S. Variance Based Sensitivity Analysis of Model Output. Design and Estimator for the Total Sensitivity Index. *Computer Physics Communications* **2010**.

60. Authors to be confirmed. DOLFINx: The Next Generation FEniCS Problem Solving Environment. *Zenodo preprint* **2023**. https://doi.org/10.5281/zenodo.10447665.
