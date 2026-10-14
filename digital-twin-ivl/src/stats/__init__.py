"""dtivl.stats: statistics of the Phase 5 predictive metrics.

* :mod:`dtivl.stats.roc` - AUC, ROC points, Youden threshold, Wilson intervals,
  stratified bootstrap, leave-one-out threshold rule, Harrell's C, Spearman.
* :mod:`dtivl.stats.models` - pre-specified predictor selection rule and
  L2-penalised logistic models (composite index, geometry-only and
  geometry + mechanics models), LOOCV and paired bootstrap comparisons.

The pipeline stage that reads ``results/cohort_outcomes.csv`` and writes the
metrics results is :mod:`dtivl.drivers.stage_metrics`.
"""
