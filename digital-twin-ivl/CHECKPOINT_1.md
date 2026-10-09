# Checkpoint 1 — scope check, literature, parameters, plan

Date: 2026-10-09. Phase 1 deliverables:

| Deliverable | File |
|---|---|
| Special-issue (SI) and author-instruction summary | this file, Sections 1–2 |
| Literature review (58 refs; 40 with search-confirmed DOIs) | `lit/review.md`, `lit/references.bib` |
| Literature parameter table (68 rows, each with a verification status) | `params/literature_parameters.csv` |
| Modelling assumptions, kept separate from literature inputs | `params/model_assumptions.csv` |
| One-page study plan | `docs/study_plan.md` |
| Acronym title options | this file, Section 4 |

**Network constraint (affects verification).** This container's egress policy blocks mdpi.com, doi.org, Crossref, PubMed/PMC, publisher sites, OpenAlex and Semantic Scholar. Every fact below was confirmed through web-search results only. Anything not confirmed is marked as such and is not used as a model input. To let me complete DOIs and author lists and download the MDPI Word template, add these hosts under the environment's *Network access → Allowed domains*: `www.mdpi.com`, `mdpi-res.com`, `api.crossref.org`, `doi.org`, `pubmed.ncbi.nlm.nih.gov`, `eutils.ncbi.nlm.nih.gov`, `pmc.ncbi.nlm.nih.gov`. Steps: https://code.claude.com/docs/en/cloud-environments#network-access

## 1. Special issue

| Item | Finding | Confidence |
|---|---|---|
| Journal | *Medical Sciences* (MDPI) | confirmed |
| SI title | **Artificial Intelligence and Computational Modeling for Next-Generation Medical Imaging and Therapy** | confirmed (search result listing on the journal's "General" section page) |
| Guest Editor | Dr. Nikolaos Dikaios | confirmed by two independent results (the SI listing, and his *Diagnostics* SI page cross-linking to it) |
| Submission deadline | **31 December 2026** | confirmed (SI listing and section page) |
| Scope keywords | Include multimodal data fusion, **predictive digital twins**, generative models for image reconstruction, and federated learning in healthcare | from one search snippet of the SI page; full scope text not retrieved |
| Status | Open; about 3 articles listed at time of search | snippet |
| Exact SI URL | **Not retrieved.** The SI appears on https://www.mdpi.com/journal/medsci/special_issues and https://www.mdpi.com/journal/medsci/sections/general; the SI's own slug did not appear in any result. | — |

**Fit.** The SI explicitly lists predictive digital twins within computational modelling for therapy. This study builds an imaging-segmentation → FE → therapy-outcome (IVL, stent) pipeline and evaluates predictive metrics, which is a direct match. Please open the SI page and confirm the URL and wording; I could not load it.

## 2. *Medical Sciences* author instructions (as retrievable)

| Item | Finding | Confidence |
|---|---|---|
| Initial submission | **Free format** (no strict formatting at first submission); journal formatting required at revision | confirmed (snippet of the medsci instructions page) |
| Required elements regardless of format | Author contributions, conflict-of-interest and other ethics statements | confirmed (snippet) |
| Article type | Research **Article** (original research) | standard MDPI type |
| Abstract | **Structured, about 250 words**, with headings Background/Objectives, Methods, Results, Conclusions | one snippet attributed this to the medsci page, a second to *JCM*; treat as likely and confirm |
| Section order | Introduction; Materials and Methods; Results; Discussion; Conclusions (optional) | confirmed (snippet) |
| Methods detail | Enough detail to replicate; give **name and version of all software** | confirmed (snippet) |
| Keywords | 3–10 | standard MDPI; not confirmed for medsci |
| Back matter (MDPI template order) | Supplementary Materials; Author Contributions (CRediT); Funding; Institutional Review Board Statement; Informed Consent Statement; Data Availability Statement; Acknowledgments; Conflicts of Interest; Abbreviations (optional); Appendix (optional) | the medsci snippet confirms contributions, COI and ethics statements; the full order is the standard MDPI template |
| Reference style | MDPI style based on ACS. Numbered in order of appearance, in square brackets placed before punctuation, e.g. "[4]." No superscripts. ISO 4 journal abbreviations. Journal format: `Author 1; Author 2. Title. Abbrev. Journal Year, Volume, First–Last page.` Any consistent style is accepted at initial (free-format) submission. | confirmed (snippets of the MDPI reference guide and medsci page) |
| Word template | Could not be downloaded (mdpi.com blocked). Fallback: a clean `.docx` (python-docx) plus Markdown source, laid out in MDPI section order. | — |

Planned back-matter placeholders: CRediT contributions; Funding ("This research received no external funding" or [grant]); IRB ("Not applicable; in silico study using no human or animal data"); Informed Consent ("Not applicable"); Data Availability (GitHub repository URL plus archived release DOI); Conflicts of Interest ("The authors declare no conflicts of interest" or [disclosure]).

## 3. Gap (summary; full text in `lit/review.md` §9)

1. Published fracture cutoffs depend on the device and mostly do not come from IVL: 225°/0.24 mm for plain balloons, 198°/0.445 mm for cutting balloons. IVL-specific OCT work names predictors but gives no cutoffs. The Fujino score targets underexpansion, not IVL fracture.
2. FE studies either model one or a few patient lesions, or **prescribe** IVL fractures. None predicts IVL fracture across a morphologically structured population, derives mechanics thresholds, and compares them with the Fujino score.
3. Calcium modulus spans about four orders of magnitude. Only one tissue-level calcium fracture dataset was found (failure at 179 ± 56 kPa and stretch 1.02), and no fatigue data. A digital twin must propagate this uncertainty.
4. IVOCT segmentation-to-FE pipelines exist but have not been linked to IVL fracture or expansion prediction.

## 4. Acronym title options

1. **CLEAVE: Coronary Lithotripsy Effects Assessed in Virtual Experiments — A Finite Element Digital-Twin Cohort for Predicting Calcium Fracture and Stent Expansion** *(recommended: the word names the endpoint, the expansion is literal, and it avoids claims beyond an in silico study)*
2. **LITHO: Lesion-specific In silico Testing of Heavily calcified Obstructions — Predicting Calcium Fracture and Stent Expansion after Intravascular Lithotripsy**
3. **SHATTER: Stress-based Hazard Analysis of calcium Thickness, Type, Extent and Rupture — In Silico Digital Twins of Calcified Coronary Lesions under Intravascular Lithotripsy and Stenting**

The composite index will get its own acronym in Phase 5, once its form is fixed.

## 5. Decisions needed before Phase 2

- **D1 — 2D versus 3D.** Your opening brief says "models should be 3D". Section 3 asks for 2D for the full cohort with a 3D subset of ≥ 4. Proposed: a 2D full factorial (arc × thickness × morphology = 64 lesions), a 3D coarse L16 subset that adds length (16 segments), and a 2D Latin-hypercube hold-out (~30). An all-3D 4 × 4 × 4 × 4 factorial (256 phase-field runs) is not feasible on this 4-core container in reasonable time.
- **D2 — Thickness levels.** Your examples (2 mm, 4 mm) lie above the OCT-measured range and above every published cutoff: 0.24, 0.445, 0.5 and 0.67 mm. OCT also cannot resolve calcium much thicker than about 1.5–2 mm. Proposed: **0.25, 0.5, 1.0, 1.5 mm**, which bracket the published cutoffs.
- **D3 — Length levels (3D only).** Proposed: **5, 10, 20, 30 mm**, with the Fujino cutoff at 5 mm. Your examples were 1 cm and 3 cm.
- **D4 — Morphologies.** Proposed: superficial sheet, deep sheet, calcified nodule (protruding), fragmented (multiple small deposits). Lipid and fibrous-cap variation would appear only in the hold-out set.
- **D5 — Calcium failure parameterisation.** Primary: strain-controlled (ε_f = 0.02 from Holzapfel 2004, with the critical stress scaling with E). Sensitivity: stress-controlled (σ_c = 179 kPa fixed).
- **D6 — Title.** Choose one of Section 4 or suggest another.
- **D7 — Network.** Optionally allow-list the hosts above so I can complete the 18 references that lack a confirmed DOI and the 6 whose author lists are incomplete.

## 6. What was not done

- No code, meshes or results exist yet. Phase 2 starts after approval.
- `requirements.txt` and `run_all.py` will be created in Phase 2, when the solver environment is built and versions can be pinned from a working install.
