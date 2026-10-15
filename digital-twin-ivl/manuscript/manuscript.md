# In Silico Digital Twins of Calcified Coronary Lesions under Intravascular Lithotripsy and Stenting

## 1. Introduction

Severe coronary calcification resists balloon dilation and limits stent expansion, and stent underexpansion is the main mechanical consequence that lesion preparation tries to prevent [1,2]. Plain and cutting balloons and intravascular lithotripsy (IVL) all aim to fracture the calcium so that the vessel wall can expand [3–5]. Whether a given lesion will fracture is therefore a practical question before treatment.

Coronary IVL delivers acoustic pressure pulses from electrohydraulic emitters inside a semi-compliant balloon, which is inflated at only 4 atm (0.405 MPa) to appose the wall [6,7]. The peak positive pressure is about 5 MPa, and the proposed fracture mechanism is the compressive stress of the wavefront [5]. Optical coherence tomography (OCT) substudies found calcium fracture in 78.7 % of lesions in Disrupt CAD II [8] and in 67.4 % in Disrupt CAD III, with multiple fractures in 67.7 % of the fractured lesions; stent expansion was 101.7 ± 28.9 % at the site of maximum calcification and 78.4 ± 25.8 % at the site of minimum stent area [9]. A pooled OCT analysis of the Disrupt CAD studies reported 3.1 ± 2.7 fractures per lesion, 72 % of them through the full calcium thickness [10]. Fracture depends on morphology: it occurred at only 23.1 % of calcified-nodule sites in the pooled Disrupt CAD III and IV OCT substudies [11], and in a single-centre study it was associated with a larger calcium arc, a smaller minimum calcium thickness and superficial calcium, with lower odds for nodular calcium (odds ratio 0.24) [12]. Clinical fracture rates are probably lower bounds, since an ex vivo study found histological microfractures that OCT did not show [13].

Before treatment, the expected result is estimated from imaging. The Fujino OCT calcium score awards 2 points for a maximum calcium angle above 180°, 1 point for a maximum thickness above 0.5 mm and 1 point for a length above 5 mm; lesions with the maximum score of 4 had a stent expansion of 78 % against 96 % for lower scores [2]. A revised score adds a minimum calcium thickness above 0.30 mm [14], and machine learning on intravascular OCT (IVOCT) lesion attributes predicts underexpansion [15]. Fracture cut-offs were derived for plain balloons (calcium angle 225°) [3] and cutting balloons (198° and 0.445 mm) [4], whereas an IVL study reported associations without cut-offs [12]. All of these predictors are geometric. They contain neither the load nor the stiffness of the calcium and the surrounding tissue, nor the stress concentration at the ends of a deposit.

Finite element (FE) models have been used to study balloon angioplasty and stenting [16,17] and the effect of calcification on plaque stress [18]. In calcified lesions, the predicted stresses and damage depend strongly on the constitutive representation [19], heavily calcified lesions carry a larger share of the balloon load [20], and simulation-driven machine learning has predicted stent expansion [21]. Calcium fracture under plain and cutting balloons has been assessed through the maximum principal stress [22] or a brittle calcium model [23]. IVL studies prescribed the fracture and simulated its effect [13,24], so fracture was an input rather than a prediction. Variational phase-field models of brittle fracture can predict whether and where a crack forms without a prescribed path: they regularise the crack over a length $\ell$ [25], have robust staggered solvers [26], link crack nucleation to the material strength [27] and have been applied to arterial tissue [28]. Pipelines from intravascular images to FE models [29–31] bring lesion-specific digital twins, in the sense proposed for precision cardiology [32], within reach, and the in silico trial framework sets out how the credibility of such models is established [33].

To our knowledge, no study has tested whether lesion-level mechanics from a computationally cheap pre-treatment analysis predict IVL-induced fracture better than geometry or the Fujino score, in a controlled cohort in which the fracture comes from a separate simulation. Clinical data cannot answer this directly: morphology and device use vary together, OCT under-detects fractures, and each lesion is treated once. The inputs are also uncertain: calcium moduli range from about 1.5 MPa in tissue-level tension [34] to 10–25 GPa reported from nanoindentation [35], and tissue-level fracture data for calcified plaque come from a single uniaxial study [36].

This study addresses that question with a structured in silico cohort. A full factorial core set of 48 two-dimensional (2D) cross-sections combines 4 calcium arcs, 3 thicknesses and 4 morphologies (superficial sheet, deep sheet, calcified nodule, fragmented) [37]; a hold-out set of 24 lesions is sampled by Latin hypercube over continuous ranges that include lipid pools and fibrous caps. Lesions are polar label maps in the IVOCT segmentation convention [31,38], so segmented images can use the same pipeline. Predictors come from the geometry and from a linear-elastic analysis of the undamaged lesion under the IVL balloon pressure. Outcomes come from a separate nonlinear simulation with phase-field damage in the calcium, in which IVL is the quasi-static equivalent of a single acoustic pulse, followed by stent expansion; the primary outcome is through-thickness calcium damage under IVL. Predictors are evaluated by receiver operating characteristic (ROC) analysis with leave-one-out cross-validation (LOOCV) and bootstrap intervals, combined by penalised logistic regression on at most 3 mechanics predictors into the Lithotripsy-Induced Fracture Tendency (LIFT) index, and compared with the cross-sectional Fujino score. Geometry-only and geometry-plus-mechanics models give the incremental value of mechanics. Thresholds and models are derived on the core set and applied unchanged to the hold-out set. No patient data are used: the study offers verification against analytical solutions and under mesh refinement, consistency with published observations and sensitivity analysis of uncertain inputs, not a clinical test of the predictors.

### 2.10. Statistical Analysis

All thresholds, predictor directions that were not fixed in advance, predictor selections and model fits were derived on the core set only. The hold-out set was used only to evaluate them unchanged. The candidate predictors were the 12 pre-treatment mechanics predictors, 8 geometric descriptors and the cross-sectional Fujino score. The quantities of the outcome model (the IVL load factor and every outcome column) were excluded from the candidates by a check in the code. Each candidate had a direction stated before the analysis (a higher value expected with damage, or a lower one for calcium thickness, depth and the minimum thickness under high stress). For descriptors without a stated direction, the direction was set on the core set and is flagged in the results.

For the primary outcome (through-thickness damage under IVL) and the binary secondary outcomes, discrimination was summarised by the area under the receiver operating characteristic curve (AUC) of the oriented predictor. The threshold maximised the Youden index, placed midway between adjacent observed values, with ties broken towards the more specific cut. Sensitivity, specificity and accuracy at the threshold are reported with Wilson score intervals. Intervals for the AUC and the threshold come from 2000 bootstrap resamples, stratified by outcome class, as two-sided 95 % percentile intervals. The threshold rule was also assessed by leave-one-out cross-validation (LOOCV): each lesion was classified with the threshold derived without it. When the AUC was $0$ or $1$, or every resample gave the same value, the percentile interval has zero width; such intervals are flagged as degenerate and are not read as statements of precision. A binary analysis required at least 2 lesions in each outcome class.

The composite index, named LIFT (Lithotripsy-Induced Fracture Tendency index), is a penalised logistic regression on at most 3 mechanics predictors chosen by a fixed rule. Candidates were the mechanics predictors without missing or constant values whose oriented AUC on the training rows was at least 0.60. They were ranked by that AUC and taken in rank order, skipping any candidate whose absolute Spearman correlation with a predictor already chosen exceeded 0.80. The chosen predictors were standardised, and an L2-penalised logistic regression with a fixed inverse penalty of 1.0 was fitted. The index is the model log-odds. Selection, fitting and the Youden threshold of the predicted probability were repeated inside every LOOCV fold and every bootstrap resample, so that the left-out lesion never contributed to its own classification. Out-of-fold performance is reported as the LOOCV AUC, the log-loss and the Brier score. The bootstrap interval of the LOOCV AUC resamples the fixed out-of-fold predictions and does not include the variability of refitting and reselection. Selected predictors whose fitted coefficient has the opposite sign to their stated direction are reported.

The composite index and the best single mechanics predictor were compared with the Fujino score by the paired bootstrap of the AUC difference. The incremental value of mechanics over geometry was tested by comparing a geometry-only penalised logistic model on the 4 descriptors of the core design factors (calcium arc, thickness, depth and number of deposits) with the same model plus the mechanics predictors chosen by the composite rule. The two models were compared by the apparent AUC difference, with both models refitted in every resample, by the LOOCV AUC, log-loss and Brier score (paired bootstrap of the out-of-fold predictions), and on the hold-out set with the core-fit models evaluated unchanged. A difference whose interval includes zero is reported as no evidence of incremental value.

For the damage-onset load under IVL, an event was the first load step at which the maximum damage reached the crack threshold. A lesion without onset was right-censored at its own peak equivalent IVL pressure, the largest load it received. The transient load factor enters only this censoring value, never a predictor. Discrimination was summarised by Harrell's concordance index (C) of the oriented predictor, where a higher oriented value should go with an earlier onset. For stent expansion with and without prior IVL, no direction was stated in advance. Spearman's correlation and C were computed on the raw predictor, their observed signs are reported, and these analyses are labelled exploratory. A ranked analysis required at least 5 lesions with the outcome.

As a cross-model check, the same predictors were scored against through-thickness damage under IVL computed with the element-erosion comparator on its lesion subset (Section 2.6), a fracture model separate from the phase field. All resampling used the study seed (20261009) plus a stage offset (500), with an independent random stream for each analysis, so that every interval is reproducible.

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
