# Focused literature review: calcified coronary lesions, intravascular lithotripsy (IVL), stenting, and finite element (FE) prediction of calcium fracture

Status: Checkpoint 1 working document (2026-10-09). Reference numbers [n] refer to the list in Section 10. The same entries are in `references.bib`.

## 0. Search method and verification level

The searches covered six themes: (1) FE modelling of calcified plaque under balloon angioplasty, IVL and stenting; (2) IVL physics; (3) constitutive data for plaque components and the arterial wall; (4) calcium fracture strength and fatigue; (5) clinical OCT studies that link calcium morphology to fracture; (6) digital-twin and patient-specific stent simulation.

**Constraint.** This container's network policy blocks publisher sites, PubMed, PMC, Crossref, doi.org, OpenAlex, Semantic Scholar and mdpi.com. Every bibliographic detail and numeric value below was therefore checked against web-search result pages and snippets, not against full texts. Each reference carries a verification code:

| Code | Meaning |
|---|---|
| **D** | Title, venue and DOI all confirmed in search results |
| **C** | Title and venue confirmed with volume/pages or PMID, but the DOI did not appear in any result. The DOI is left blank on purpose and is never filled in from memory. |
| **P** | Partial: title and venue confirmed; authors and/or DOI not confirmed |

Numeric values are marked *(snippet)* when the number appeared in a search result attributed to the cited paper. Values that appeared only in a secondary source are marked *(secondary)*. Before submission, every P and C entry should be completed from the DOI record, which needs network access to doi.org or Crossref.

## 1. Clinical problem and IVL clinical evidence

Severe coronary calcification limits stent expansion, and stent underexpansion is the main mechanical consequence that lesion preparation tries to prevent [17,20]. In a 1155-lesion IVUS series, Mintz and colleagues found calcium in 73% of lesions by IVUS against 38% by angiography. Where calcium was present it was superficial only in 48%, deep only in 28%, and both in 24% *(secondary, via a 2022 Mintz editorial; the primary 1995 Circulation paper was not retrieved)*.

Coronary IVL uses a semi-compliant balloon with electrohydraulic emitters. The balloon is inflated to 4 atm for wall apposition, and pulses are delivered in sequences of 10 at 1 Hz, up to 80 pulses per catheter for the original C2 catheter [15] *(snippet)*. Newer catheters deliver pulses faster or carry more pulses, so device settings must be stated for each model run. The Disrupt CAD programme provides the reference clinical data:

- **Disrupt CAD II** (n = 120; OCT substudy n = 47) reported calcium fracture in 78.7% of lesions, with 3.4 ± 2.6 fractures per lesion and fracture length 5.5 ± 5.0 mm [2] *(snippet)*.
- **Disrupt CAD III OCT substudy** (n = 100) reported fracture in 67.4% of lesions, with multiple fractures in 67.7% of those. Stent expansion was 101.7 ± 28.9% at the pre-IVL maximum-calcium site and 78.4 ± 25.8% at the final minimum-stent-area site. Minimum stent area was 6.5 ± 2.1 mm² and pre-procedure minimum lumen area 2.2 ± 0.8 mm² [1] *(snippet)*. Minimum stent area was similar with and without OCT-visible fracture [1] *(secondary)*.
- **Patient-level pooled analyses** cover all Disrupt CAD lesions [5]; eccentric versus concentric lesions (47 eccentric, 26%, of 180; similar procedural success) [6]; quartiles of calcium arc (N = 230; calcium length and volume index rose with concentricity) [7]; and calcified nodules [8]. In the nodule analysis of Disrupt CAD III/IV OCT substudies (155 patients), fracture at calcified-nodule sites was 23.1% *(snippet)*.
- **Single-centre OCT studies.** Emori et al. found fracture in 25% of analysed segments after IVL. Fracture was associated with larger calcium arc, smaller minimum calcium thickness, superficial calcium and non-nodular calcium (nodular OR 0.24, 95% CI 0.08–0.70) [9] *(snippet)*; no cutoff values were visible in the snippet. Yamada et al. related baseline calcium angle and IVL-induced fractures to stent expansion and roundness in 49 patients [10].
- **Ex vivo and cadaver data.** A 2025 cadaver comparison confirmed fractures by micro-CT and histology in 80% of IVL-treated lesions, 66.7% with cutting balloons and 33.3% with ultra-high-pressure balloons (P = 0.35) [11] *(snippet)*. An ex vivo laser-IVL study (n = 19) found OCT-visible calcium modification in 78% of arteries and histological microfractures that OCT did not show in 57% [12] *(snippet)*. Clinical OCT fracture rates are therefore likely to undercount fractures, and a simulated "fracture" endpoint is not directly comparable to an OCT-visible fracture.

## 2. IVL physics

Kereiakes et al. give the most specific description of the pressure field. The peak positive pressure is about 5 MPa (≈50 atm) and the negative pressure is small (about −0.3 MPa). Extracorporeal shockwave lithotripsy, by comparison, uses 30–110 MPa [13] *(snippet)*. The proposed fracture mechanism is the compressive stress of the wavefront, and the small negative pressure is described as well below the threshold associated with tissue damage [13] *(snippet)*. Karimi Galougahi et al. repeat the ≈50 atm figure and the subnominal 4 atm balloon [14] *(snippet)*. Reported pulse durations differ: the first-in-human description reports peak energy lasting under 2 µs [3] *(snippet)*, and an ACC expert analysis reports about 5 µs *(secondary; authors not retrieved)*. In the first-in-human series the median IVL balloon pressure was 6 atm [3] *(snippet)*.

**Attenuation with distance.** No quantitative attenuation data for the coronary IVL field in tissue were found. Measured sound speeds put calcified plaque at about 1760–2000 m/s, against about 1526–1568 m/s for fatty and normal intimal tissue [35,36] *(snippet)*. The impedance mismatch at calcium boundaries is therefore moderate, but densities could not be verified, so impedances cannot yet be computed from cited values. Any attenuation law used in the model (for example, geometric spreading from a line source) will be labelled a modelling assumption and included in the sensitivity analysis.

**Fatigue and pulse count.** No fatigue (S–N) data for calcified plaque were found. Pulse-count damage accumulation will therefore be omitted, and the omission stated in Methods.

## 3. OCT/IVUS calcium morphology, scoring and fracture predictors

- **Fujino OCT calcium score** [17]. Points are awarded for maximum calcium angle > 180° (2 points), maximum thickness > 0.5 mm (1 point) and length > 5 mm (1 point). The score was derived in 128 patients and validated in 133. Score-4 lesions had stent expansion of 78% against 96% for scores 0–3 (p < 0.01) *(snippet)*. This score is the geometric comparator for the study.
- **Balloon-only fracture predictors** [18]. Maximum calcium angle predicted fracture with AUC 0.92 at a best cutoff of 225° (sensitivity 85.7%, specificity 90.1%). The thickness cutoff was 0.24 mm *(angle values from snippet; 0.24 mm from secondary sources)*.
- **Cutting balloon** [19]: 198° and 0.445 mm in 134 lesions *(snippet)*. **Rotational atherectomy**: a 0.67 mm thickness cutoff is attributed to Maejima et al. 2016 *(secondary; primary paper not retrieved; not used as an input)*.
- **Morphological classes.** Sugiyama et al. classified calcified culprit plaques by OCT into eruptive calcified nodules, calcified protrusions and superficial calcific sheets [21]. With the deep-versus-superficial distinction from Section 1, these classes support the morphology factor of the in silico cohort: superficial sheet, deep sheet, nodule and fragmented.
- **Automated IVOCT analysis from the author's group.** Calcium can be segmented automatically from IVOCT [24]. Segmentations have been turned into lesion-specific FE models of stent deployment [23]. Lesion attributes from pre-stent IVOCT predicted under-expansion (110 lesions, AUC 0.85 ± 0.02) [22]. These works define the label-map convention (0 = other, 1 = lumen, 4 = calcium) adopted here.
- **Reference vessel size.** In a normal-angiogram cohort of 453 patients, mean diameters were LAD 3.50 ± 0.52 mm, LCX 3.31 ± 0.57 mm and RCA 3.18 ± 0.57 mm [25] *(snippet)*.

## 4. Mechanical properties

**Arterial wall.** Holzapfel et al. tested layer-specific strips from 13 human LAD arteries. The intima:media:adventitia thickness ratio was about 27:36:40 [26] *(snippet)*. The layer-wise HGO constants are in that paper's Table 3, but the per-specimen values could not be read from snippets. Escuer et al. fitted a Yeoh model to the circumferential media data of [26]: c10 = 17.01 kPa, c20 = −73.42 kPa, c30 = 414.95 kPa [30] *(snippet)*. With this set, ∂W/∂I1 stays positive for all I1 ≥ 3; its minimum is about 12.7 kPa, by our own algebra. The HGO model with fibre dispersion [29] is the 3D option. It will be used only if verified layer constants are obtained.

**Fibrous and cellular intima.** At 25 kPa applied circumferential stress, tangential moduli were 927 ± 468 kPa (cellular), 2312 ± 2180 kPa (hypocellular) and 1466 ± 1284 kPa (calcified), with no significant difference between groups [28] *(snippet)*. The lowest intimal fracture stress was 254.8 ± 79.8 kPa, in the circumferential fibrous cap [27] *(snippet)*. Intima stiffness has a strong effect on predicted cap stress [33].

**Lipid / necrotic core.** An AFM study measured lipid-rich areas at 5.5 ± 3.5 kPa [34] *(snippet)*. The species and the full citation still need confirmation. FE studies have assumed lipid moduli between about 1 and 50 kPa with Poisson's ratio near 0.49 *(secondary)*.

**Calcium stiffness.** Calcium stiffness depends strongly on the length scale of the measurement. Tissue-level tension tests give moduli in the low-MPa range: 1466 ± 1284 kPa in [28], and an implied secant modulus of about 9 MPa from the failure point in [27] (our calculation, 179 kPa / 0.02). Nanoindentation of calcification particles gave 0.69 ± 2.3 GPa (maximum 21 GPa), and other nanoindentation studies give 10–25 GPa *(secondary, via [31,32])*. A parametric FE study varied the calcification modulus from 1 MPa to 10 GPa, with ν = 0.3. Stress concentration at the calcium–fibrous interface rose with modulus [31] *(snippet)*. The modulus must therefore be treated as a dominant uncertain input, not a fixed constant.

**Calcium fracture strength.** The only tissue-level calcification fracture data found come from Holzapfel et al. [27]. Calcified tissue behaved linearly and failed at 179 ± 56 kPa and a stretch of 1.02 ± 0.005, lower than any other plaque component *(snippet)*. The implied failure strain of about 2% is the most directly measured fracture quantity available. Brittle-fracture FE studies have modelled calcium with dense-bone properties [41] *(snippet; values not retrieved)*. No fracture-energy (G_c) measurement for calcified plaque was found.

**Fatigue.** None found (see Section 2).

## 5. FE modelling of calcified lesions under balloon, IVL and stent

- **Balloon/stent FE in calcified arteries.** In a heavily calcified lesion, stents expanded into a dog-bone shape, and 12% of the balloon load was transferred to the lesion against 0.8% in a non-calcified lesion. Raising post-dilation pressure from 10 to 20 atm increased minimum lumen diameter from 1.48 to 2.82 mm. Calcium stress rose from 1.5 to 8.4 MPa and strain energy from 1.77 to 26.5 mJ [38] *(snippet)*. In patient-specific models, a larger calcification angle limited stretchability and stent expansion *(secondary; source paper not identified)*. Simulation-driven machine learning has been applied to 120 OCT cross-sections [39]. The choice of constitutive model for the lesion strongly affects predicted damage and stresses, particularly with calcification [42].
- **Brittle fracture under balloons.** Li et al. modelled calcium as brittle with dense-bone properties. They found that a cutting balloon fractured a 180° calcium arc at about 1.2 MPa and that thickness, but not length, affected fracture [41] *(snippet)*. Zhu et al. used maximum principal stress in a 180° calcium model to compare cutting-balloon configurations [40].
- **IVL-specific in silico work.** Sogbadji et al. prescribed IVL-induced defects in idealised lesions and simulated stenting [37]. Larger defects gave more lumen gain, defects that split the calcium into two parts were most beneficial, and defect location mattered little *(snippet)*. The laser-IVL ex vivo study used FE to show that fractures alone contributed modestly to compliance gain, and that debonding and intimal tears also contributed [12] *(snippet)*. Neither study predicts where or whether fracture occurs. Fracture is an input in both.
- **Earlier balloon-angioplasty FE** includes the layer-specific 3D model of [43], the stent-design study of [44], and analyses of calcification effects on plaque stress [45].

## 6. Fracture modelling methods

Variational phase-field fracture regularises the Francfort–Marigo energy [48]. The operator-split, history-field implementation [49] is the standard robust scheme. For AT1-type models the nucleation stress depends on the regularisation length ℓ, the toughness and the modulus [50] *(secondary description)*. This property is used here: the verified strength (or failure strain) fixes the ℓ–G_c pair, so no unverified toughness value is needed. Phase-field fracture has been applied to arterial tissue [51].

## 7. Digital twins and image-to-model pipelines

Image-to-FE pipelines from intravascular imaging exist. Examples include high-fidelity 3D structural models from IVUS/OCT [46], fully automated OCT-to-FE "digital twins" [47], and the IVOCT segmentation-to-stent-FE path of the author's group [23]. Position papers define the cardiac digital twin [58]. Model credibility for regulatory use is framed by ASME V&V 40 [57], whose terms (verification, validation, uncertainty quantification, context of use) are adopted in this study.

## 8. Numerical verification and statistics

Discretisation error will be reported with the grid convergence index [52]. Global sensitivity will use Sobol indices [53] through SALib [54]. Meshing will use Gmsh [55] and the FE solver DOLFINx [56]. ROC/Youden analysis with bootstrap and leave-one-out cross-validation needs no further methodological citations beyond standard texts, which will be added at manuscript stage only if their DOIs can be confirmed.

## 9. Gap addressed by this study

1. **Clinical cutoffs are device-specific and mostly not from IVL.** Published angle/thickness cutoffs come from plain balloons (225°/0.24 mm [18]), cutting balloons (198°/0.445 mm [19]) and rotational atherectomy (0.67 mm, secondary). IVL-specific work identifies predictors but reports no cutoffs [9]. The Fujino score [17] was derived for stent underexpansion, not for IVL fracture.
2. **FE studies either study one lesion or prescribe fracture.** Patient-specific models [38,46,47] examine few lesions. IVL in silico work prescribes defect geometry [12,37]. Brittle-fracture studies address balloons, not IVL [40,41]. No study predicts IVL-induced fracture from morphology across a structured population of lesions with a fracture model, derives mechanics-based thresholds, or tests them against the Fujino score.
3. **Material uncertainty is large and rarely propagated.** Calcium modulus spans about four orders of magnitude [27,28,31,32]. Only one tissue-level calcium fracture dataset was found [27], and no fatigue data. A predictive digital twin must report how its conclusions depend on these inputs.
4. **The imaging-to-mechanics path is not linked to IVL decisions.** Segmentation-to-FE pipelines exist [23,46,47] but have not been used to produce an IVL-specific fracture or expansion prediction.

The study targets the intersection of these gaps. It builds a morphologically structured in silico cohort in the IVOCT label-map convention and runs fracture simulations under IVL loading. It then derives lesion-level mechanical predictors and an acronym-named index, and compares them with the Fujino score. All claims are limited to verification, consistency with published observations, and uncertainty analysis.

## 10. References

Codes: D = DOI confirmed; C = citation confirmed, DOI not confirmed; P = partial. "et al." marks author lists not fully confirmed.

1. Hill JM, et al. Intravascular lithotripsy for treatment of severely calcified coronary artery disease (Disrupt CAD III). *J. Am. Coll. Cardiol.* **2020**. doi:10.1016/j.jacc.2020.09.603 — D
2. Ali ZA, et al. Safety and effectiveness of coronary intravascular lithotripsy for treatment of severely calcified coronary stenoses: the Disrupt CAD II study. *Circ. Cardiovasc. Interv.* **2019**, 12, e008434. doi:10.1161/CIRCINTERVENTIONS.119.008434 — D
3. Brinton TJ, et al. Feasibility of shockwave coronary intravascular lithotripsy for the treatment of calcified coronary stenoses: first description. *Circulation* **2019**. doi:10.1161/CIRCULATIONAHA.118.036531 — D
4. Ali ZA, Brinton TJ, Hill JM, Maehara A, Matsumura M, Karimi Galougahi K, Illindala U, Götberg M, Whitbourn R, Van Mieghem N, Meredith IT, Di Mario C, Fajadet J. Optical coherence tomography characterization of coronary lithoplasty for treatment of calcified lesions: first description. *JACC Cardiovasc. Imaging* **2017**, 10, 897–906. doi:10.1016/j.jcmg.2017.05.012 — D
5. Kereiakes DJ, Di Mario C, Riley RF, Fajadet J, Shlofmitz RA, Saito S, Ali ZA, Klein AJ, Price MJ, Hill JM, Stone GW. Intravascular lithotripsy for treatment of calcified coronary lesions: patient-level pooled analysis of the Disrupt CAD studies. *JACC Cardiovasc. Interv.* **2021**, 14, 1337–1348. doi:10.1016/j.jcin.2021.04.015 — D
6. Blachutzik F, et al. Safety and effectiveness of coronary intravascular lithotripsy in eccentric calcified coronary lesions: a patient-level pooled analysis from the Disrupt CAD I and CAD II studies. *Clin. Res. Cardiol.* **2021**, 110, 228–236. doi:10.1007/s00392-020-01737-3 — D
7. Ali ZA, et al. Impact of calcium eccentricity on the safety and effectiveness of coronary intravascular lithotripsy: pooled analysis from the Disrupt CAD studies. *Circ. Cardiovasc. Interv.* **2023**, 16, e012898. doi:10.1161/CIRCINTERVENTIONS.123.012898 — D
8. [Authors to be confirmed]. Outcomes of coronary intravascular lithotripsy for the treatment of calcified nodules: a pooled analysis of the Disrupt CAD studies. *EuroIntervention* **2024**. doi:10.4244/EIJ-D-24-00282 — D (authors pending)
9. Emori H, Shiono Y, Kuriyama N, Honda Y, Ebihara S, Kadooka K, et al. Calcium fracture after intravascular lithotripsy as assessed with optical coherence tomography. *Circ. J.* **2023**, 87, 799–805. doi:10.1253/circj.CJ-22-0695 — D
10. Yamada T, Taniguchi N, Hata T, Nakajima S, Kawakami S, Takahashi A. Impact of calcified plaque distribution on symmetrical stent expansion after intravascular lithotripsy. *J. Soc. Cardiovasc. Angiogr. Interv.* **2025**. doi:10.1016/j.jscai.2025.102616 — D (journal inferred from DOI prefix)
11. [Authors to be confirmed]. Comparison of vascular injury from intravascular lithotripsy, cutting, or ultra-high-pressure balloons during coronary calcium modification. *JACC Cardiovasc. Interv.* **2025**. doi:10.1016/j.jcin.2025.06.035 — D (authors pending)
12. [Authors to be confirmed]. Intravascular laser lithotripsy for calcium fracture in human coronary arteries. *EuroIntervention* (PMID 38060282). — P
13. Kereiakes DJ, Virmani R, Hokama JY, Illindala U, Mena-Hurtado C, Holden A, Hill JM, Lyden SP, Ali ZA. Principles of intravascular lithotripsy for calcific plaque modification. *JACC Cardiovasc. Interv.* **2021**, 14, 1275–1292. doi:10.1016/j.jcin.2021.03.036 — D
14. Karimi Galougahi K, et al. Calcific plaque modification by acoustic shock waves: intravascular lithotripsy in coronary interventions. *Circ. Cardiovasc. Interv.* **2021**. doi:10.1161/CIRCINTERVENTIONS.120.009354 — D
15. Tovar Forero MN, Daemen J. The coronary intravascular lithotripsy system. *Interv. Cardiol.* **2019**, 14, 174–181. doi:10.15420/icr.2019.18.R1 — D
16. Gruslova A, et al. Review of intravascular lithotripsy for treating coronary, peripheral artery, and valve calcifications. *Catheter. Cardiovasc. Interv.* **2024**. doi:10.1002/ccd.30933 — D
17. Fujino A, Mintz GS, Matsumura M, et al. A new optical coherence tomography-based calcium scoring system to predict stent underexpansion. *EuroIntervention* **2018**, 13, e2182–e2189. doi:10.4244/EIJ-D-17-00962 — D
18. Fujino A, Mintz GS, Lee T, et al. Predictors of calcium fracture derived from balloon angioplasty and its effect on stent expansion assessed by optical coherence tomography. *JACC Cardiovasc. Interv.* **2018**, 11, 1015–1017. doi:10.1016/j.jcin.2018.02.004 — D
19. [Authors to be confirmed]. Predictors for calcium fracture with a novel cutting balloon: an optical coherence tomography study. *JACC Cardiovasc. Interv.* **2022**. doi:10.1016/j.jcin.2022.02.017 — D (authors pending)
20. Mintz GS. Intravascular imaging of coronary calcification and its clinical implications. *JACC Cardiovasc. Imaging* **2015**, 8, 461–471. PMID 25882575 — C
21. Sugiyama T, Yamamoto E, Fracassi F, et al. Calcified plaques in patients with acute coronary syndromes. *JACC Cardiovasc. Interv.* **2019**. PMID 30898249 — C
22. Gharaibeh Y, et al. (Wilson DL, senior). Prediction of stent under-expansion in calcified coronary arteries using machine learning on intravascular optical coherence tomography images. *Sci. Rep.* **2023**, 13. doi:10.1038/s41598-023-44610-9 — D
23. Gharaibeh Y, et al. Deep learning segmentation of coronary calcified plaque from intravascular optical coherence tomography (IVOCT) images with application to finite element modeling of stent deployment. *Proc. SPIE* **2019**, 10951. doi:10.1117/12.2515256 — D (author list pending)
24. Lee J, et al. Segmentation of coronary calcified plaque in intravascular OCT images using a two-step deep learning approach. *IEEE Access* **2020** (PMC7885992). — P
25. Mohan A, Gopalakrishnan A, Chandran R, Joseph S, Mathew AJ, Nair AS, Sudhakaran R. Examining the influence of gender, age, and dominance on the caliber of normal coronary arteries in the South Indian population. *Cureus* **2023**. doi:10.7759/cureus.51146 — D
26. Holzapfel GA, Sommer G, Gasser CT, Regitnig P. Determination of layer-specific mechanical properties of human coronary arteries with nonatherosclerotic intimal thickening and related constitutive modeling. *Am. J. Physiol. Heart Circ. Physiol.* **2005**, 289, H2048–H2058. doi:10.1152/ajpheart.00934.2004 — D
27. Holzapfel GA, Sommer G, Regitnig P. Anisotropic mechanical properties of tissue components in human atherosclerotic plaques. *J. Biomech. Eng.* **2004**, 126, 657–665. — C
28. Loree HM, Grodzinsky AJ, Park SY, Gibson LJ, Lee RT. Static circumferential tangential modulus of human atherosclerotic tissue. *J. Biomech.* **1994**, 27, 195–204. — C
29. Gasser TC, Ogden RW, Holzapfel GA. Hyperelastic modelling of arterial layers with distributed collagen fibre orientations. *J. R. Soc. Interface* **2006**, 3, 15–35. doi:10.1098/rsif.2005.0073 — D
30. Escuer J, Martínez MA, McGinty S, Peña E. Mathematical modelling of the restenosis process after stent implantation. *J. R. Soc. Interface* **2019**, 16, 20190313. doi:10.1098/rsif.2019.0313 — D
31. [Authors to be confirmed]. Effect of calcification modulus and geometry on stress in models of calcified atherosclerotic plaque. *Cardiovasc. Eng. Technol.* **2014**. doi:10.1007/s13239-014-0186-6 — D (authors pending)
32. Barrett HE, Van der Heiden K, Farrell E, Gijsen FJH, Akyildiz AC. Calcifications in atherosclerotic plaques and impact on plaque biomechanics. *J. Biomech.* **2019**. — P
33. Akyildiz AC, et al. Effects of intima stiffness and plaque morphology on peak cap stress. *Biomed. Eng. Online* **2011**, 10, 25. doi:10.1186/1475-925X-10-25 — D
34. [Authors to be confirmed]. Mapping elasticity moduli of atherosclerotic plaque in situ via atomic force microscopy. *J. Struct. Biol.* **2011**. — P
35. [Authors to be confirmed]. Carotid atherosclerotic plaque characterisation by measurement of ultrasound sound speed in vitro at high frequency, 20 MHz. *Ultrasonics*. — P
36. [Authors to be confirmed]. Acoustic properties of atherosclerosis of human aorta obtained with high-frequency ultrasound. *Ultrasound Med. Biol.* **1998**. — P
37. Sogbadji J, Kadry K, Poletti G, Berti F, Edelman ER, Nezami FR. Impact of lesion preparation-induced calcified plaque defects in vascular intervention for atherosclerotic disease: in silico assessment. *Biomech. Model. Mechanobiol.* **2025**. doi:10.1007/s10237-024-01923-6 — D
38. Dong P, et al. Load-sharing characteristics of stenting and post-dilation in heavily calcified coronary artery. *Sci. Rep.* **2023**, 13, 16878. doi:10.1038/s41598-023-43160-4 — D
39. Dong P, Ye G, Kaya M, Gu L. Simulation-driven machine learning for predicting stent expansion in calcified coronary artery. *Appl. Sci.* **2020**, 10, 5820. doi:10.3390/app10175820 — D
40. Zhu X, Umezu M, Iwasaki K. Finite element analysis of cutting balloon expansion in a calcified artery model of circular angle 180°: effects of balloon-to-diameter ratio and number of blades facing calcification on potential calcification fracturing and perforation reduction. *PLoS ONE* **2021**. doi:10.1371/journal.pone.0251404 — D
41. Li J, Cao H, Li M, Shu L, Lin C. A study of balloon type on calcified coronary lesion predilation: a finite element analysis. *Proc. Inst. Mech. Eng. H* **2023**. doi:10.1177/09544119231157853 — D
42. Conway C, McGarry JP, Edelman ER, McHugh PE. Numerical simulation of stent angioplasty with predilation: an investigation into lesion constitutive representation and calcification influence. *Ann. Biomed. Eng.* **2017**, 45, 2244–2252. doi:10.1007/s10439-017-1851-3 — D
43. Holzapfel GA, Stadler M, Schulze-Bauer CAJ. A layer-specific three-dimensional model for the simulation of balloon angioplasty using magnetic resonance imaging and mechanical testing. *Ann. Biomed. Eng.* **2002**, 30, 753–767. — C
44. Lally C, Dolan F, Prendergast PJ. Cardiovascular stent design and vessel stresses: a finite element analysis. *J. Biomech.* **2005**, 38, 1574–1581. — C
45. Huang H, Virmani R, Younis H, Burke AP, Kamm RD, Lee RT. The impact of calcification on the biomechanical stability of atherosclerotic plaques. *Circulation* **2001**, 103, 1051–1056. PMID 11222465 — C
46. Kadry K, Olender ML, Marlevi D, Edelman ER, Nezami FR. A platform for high-fidelity patient-specific structural modelling of atherosclerotic arteries: from intravascular imaging to three-dimensional stress distributions. *J. R. Soc. Interface* **2021**, 18, 20210436. doi:10.1098/rsif.2021.0436 — D
47. Straughan R, Kadry K, Parikh SA, Edelman ER, Nezami FR. Fully automated construction of three-dimensional finite element simulations from optical coherence tomography. *Comput. Biol. Med.* **2023**, 165, 107341. doi:10.1016/j.compbiomed.2023.107341 — D
48. Bourdin B, Francfort GA, Marigo J-J. Numerical experiments in revisited brittle fracture. *J. Mech. Phys. Solids* **2000**, 48, 797–826. — C
49. Miehe C, Hofacker M, Welschinger F. A phase field model for rate-independent crack propagation: robust algorithmic implementation based on operator splits. *Comput. Methods Appl. Mech. Eng.* **2010**, 199, 2765–2778. doi:10.1016/j.cma.2010.04.011 — D
50. Tanné E, Li T, Bourdin B, Marigo J-J, Maurini C. Crack nucleation in variational phase-field models of brittle fracture. *J. Mech. Phys. Solids* **2018**. — P
51. Gültekin O, Dal H, Holzapfel GA. A phase-field approach to model fracture of arterial walls: theory and finite element analysis. *Comput. Methods Appl. Mech. Eng.* **2016**, 312, 542–566. — C
52. Celik IB, Ghia U, Roache PJ, Freitas CJ, Coleman H, Raad PE. Procedure for estimation and reporting of uncertainty due to discretization in CFD applications. *J. Fluids Eng.* **2008**, 130, 078001. doi:10.1115/1.2960953 — D
53. Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S. Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Comput. Phys. Commun.* **2010**. — P
54. Herman J, Usher W. SALib: an open-source Python library for sensitivity analysis. *J. Open Source Softw.* **2017**, 2, 97. doi:10.21105/joss.00097 — D
55. Geuzaine C, Remacle J-F. Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing facilities. *Int. J. Numer. Methods Eng.* **2009**, 79, 1309–1331. — C
56. [Authors to be confirmed]. DOLFINx: the next generation FEniCS problem solving environment. Zenodo preprint **2023**. doi:10.5281/zenodo.10447665 — D (authors pending)
57. Viceconti M, Pappalardo F, Rodriguez B, Horner M, Bischoff J, Musuamba Tshinanu F. In silico trials: verification, validation and uncertainty quantification of predictive models used in the regulatory evaluation of biomedical products. *Methods* **2021**, 185, 120–127. doi:10.1016/j.ymeth.2020.01.011 — D
58. Corral-Acero J, et al. The 'Digital Twin' to enable the vision of precision cardiology. *Eur. Heart J.* **2020**, 41, 4556–4564. doi:10.1093/eurheartj/ehaa159 — D

Totals: 58 references. D = 40, C = 10, P = 8. Six D entries still need their author lists completed (refs 8, 11, 19, 23, 31, 56).

## 11. Searched but not verified (excluded from inputs until confirmed)

- Maejima et al., *Circ. J.* 2016: calcium thickness and crack formation after rotational atherectomy. Paper not retrieved; the 0.67 mm cutoff is known only from secondary sources.
- Ebenstein et al., *J. Biomed. Mater. Res. A* 2009: nanoindentation of calcification. Not retrieved; values cited via [31,32].
- Layer-specific HGO constants for human coronary arteries (Table 3 of [26]): not readable from snippets. Values reported in secondary stent papers had unclear units.
- Lee et al., *JACC Cardiovasc. Imaging* 2017 (calcified nodule prevalence by OCT); Räber et al. 2018 EAPCI consensus (stent-expansion criteria); Dodge et al. 1992 (normal coronary lumen diameter); Loree et al. 1992 (cap thickness and stress); Niederer et al. 2021 (digital twins). None was confirmed by search.
- A 2026 *Heart and Vessels* paper on intimal thickness over a superficial calcified sheet and IVL fracture (doi:10.1007/s00380-026-02706-y appeared only as a link; content not confirmed).
- A meta-analysis reporting a pooled post-IVL fracture rate of 68% (prediction interval 0.13–1.00), seen in one snippet; title, journal and authors not confirmed.
- Densities of calcified plaque and soft tissue, needed for the transient IVL model: no verified source found.

## 12. Additional sources verified in Phase 2 (geometry, devices, acoustics)

A second search round (four topic agents, each followed by an independent verifier) supplied the inputs needed to build lesion geometry. Codes as in §10. These entries are numbered 59 onward.

59. Konishi et al. The histological analysis of the coronary medial thickness: implications for percutaneous coronary intervention. *PLoS ONE* **2023**, 18, e0283840. doi:10.1371/journal.pone.0283840. Media 0.214 mm (IQR 0.157–0.281) in main branches. — D
60. Rioufol G, et al. Adventitia measurement in coronary artery: an in vivo intravascular ultrasound study. *Heart* **2006**, 92, 985–986. Adventitia 353 ± 75 µm in atherosclerotic LAD. The value was seen only in a review citing the paper. — C
61. Sato et al. A revised optical coherence tomography-derived calcium score to predict stent underexpansion in severely calcified lesions. *JACC Cardiovasc. Interv.* **2025**. doi:10.1016/j.jcin.2024.12.001 — D
62. [Authors to be confirmed]. A novel intravascular lithotripsy system in severely calcified coronary lesions: the COCALP study. *MedComm* **2025**. doi:10.1002/mco2.70208 — D
63. Lee T, et al. Prevalence, predictors, and clinical presentation of a calcified nodule as assessed by optical coherence tomography. *JACC Cardiovasc. Imaging* **2017**. doi:10.1016/j.jcmg.2017.05.013 — D
64. [Authors to be confirmed]. Serum marker and CT characteristics of coronary calcified nodule assessed by intravascular ultrasound. *BMC Cardiovasc. Disord.* **2022**. doi:10.1186/s12872-022-02931-z — D
65. [Authors to be confirmed]. TCT-120 abstract: pooled OCT analysis of the Disrupt CAD studies. *J. Am. Coll. Cardiol.* **2021**. doi:10.1016/j.jacc.2021.09.970 — D
66. [Authors to be confirmed]. Volumetric characterization of human coronary calcification by frequency-domain optical coherence tomography. Calcium depth 0.25 ± 0.09 mm in 30 cadaveric LAD arteries. — P
67. [Authors to be confirmed]. Cardiovascular Ultrasound **2016**: IVUS plaque eccentricity index; eccentric if max/min ≥ 3. doi:10.1186/s12947-016-0090-3 — D
68. Rooney et al. 1979 and 1982: speed of sound in excised arteries (1609 ± 8 m/s at 37 °C) and in highly calcified lesions (~2000 m/s). — P
69. [Authors to be confirmed]. Speed of sound in human coronary artery walls at 50 MHz. *Braz. J. Med. Biol. Res.* **2002**. — P
70. [Authors to be confirmed]. Comparison of in vivo acute stent recoil between the bioabsorbable everolimus-eluting coronary stent and the everolimus-eluting cobalt chromium coronary stent. Acute recoil 4.3 ± 7.1% (EES). — P
71. Abbott Vascular (XIENCE) and Boston Scientific (SYNERGY) product specifications: nominal pressure 11–12 atm; rated burst pressure 16 atm (SYNERGY 2.25–2.75 mm: 18 atm). Manufacturer documents. — P
72. Model-input densities used in earlier studies (not measurements): calcified plaque 2050 kg/m³ (doi:10.1186/s13104-019-4381-2) and 1450 kg/m³ (doi:10.1186/s40759-014-0002-x); arterial wall 1075 and plaque 920 kg/m³ (doi:10.1186/1475-925X-6-8). — D (as sources of model inputs only)
73. Whelton PK, et al. 2017 ACC/AHA high blood pressure guideline: normal systolic < 120 mmHg. Details from secondary summaries. — P

**Findings that change the study context.**
- *Calcium thickness in IVL-treated lesions.* The Disrupt CAD III OCT substudy reports mean maximum arc 292.5 ± 76.5°, maximum thickness 0.96 ± 0.25 mm and calcified length 47.9 ± 18.8 mm. The thickness levels approved for the core set (0.2, 0.4 mm) are therefore thinner than typical IVL-treated calcium. They lie near the balloon-fracture thickness cutoffs (0.24 mm [18]; 0.505 mm in a Kubo/Akasaka OCT study, secondary source).
- *Geometry construction.* No absolute intima thickness at calcified lesions was found. The plaque thickness is therefore derived from the reference lumen radius and the lesion lumen radius. The implied plaque burden (≈81%) lies within the CTA range for calcified plaques (77–85%) [64].
- *Fracture geometry after IVL.* Pooled OCT shows 3.1 ± 2.7 fractures per lesion, fracture length 5.6 ± 5.3 mm, and 72% full-thickness fractures [65]. These are the targets of the fracture-pattern consistency check.
- *Not found.* No measured density of calcified plaque, no coronary IVL pressure–distance profile (only a 1/r decay for an extracorporeal electrohydraulic lithotripter), no stent radial stiffness, and no calcified-nodule protrusion height. Each of these enters the model as a labelled assumption with a sensitivity range.
