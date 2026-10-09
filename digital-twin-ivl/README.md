# digital-twin-ivl

An in silico finite element study of predictive digital twins of calcified coronary lesions under intravascular lithotripsy (IVL) and stent deployment. It is the basis of a manuscript prepared for the *Medical Sciences* (MDPI) special issue "Artificial Intelligence and Computational Modeling for Next-Generation Medical Imaging and Therapy". The study uses no clinical data.

## Status

Phase 1 (scope check, literature, parameters, plan) is complete. See `CHECKPOINT_1.md`. Phase 2 (lesion cohort) starts after approval.

## Layout

```
params/      literature_parameters.csv (cited inputs, with verification status), model_assumptions.csv
lit/         review.md, references.bib
docs/        study_plan.md
src/         geometry/, meshing/, materials/, solvers/, fracture/, metrics/, stats/   (Phase 2+)
cohort/      lesion masks, meshes, cohort.csv                                          (Phase 2)
results/     *.json, *.csv                                                             (Phase 3+)
figures/                                                                               (Phase 6)
manuscript/  build_manuscript.py, manuscript.md, manuscript.docx                       (Phase 7)
tests/       analytical and regression tests                                           (Phase 2+)
run_all.py  requirements.txt  CHANGES.md                                               (Phase 2+)
```

## Rules this repository follows

1. Every number in the manuscript is read from `results/*.json|csv` by `manuscript/build_manuscript.py`. No numbers are typed by hand.
2. Every literature-derived input is listed in `params/literature_parameters.csv` with its source and verification status. Inputs marked `UNVERIFIED` are not used in the main analysis.
3. Validation is described only as verification, consistency with published observations, and sensitivity/uncertainty analysis. No patient-level validation is claimed.
4. Predictors come from geometry plus a linear-elastic, undamaged analysis. Fracture outcomes come from a separate nonlinear phase-field simulation.
5. After the first draft, every cut or substantive change is logged in `CHANGES.md`.
6. All randomness uses a fixed seed (20261009). `run_all.py` regenerates every result and figure.
