| Model | Type | n core | Core AUC, apparent | Core AUC, LOOCV (95% CI) | LOOCV log-loss | LOOCV Brier | Hold-out AUC (95% CI) | n hold-out |
|---|---|---|---|---|---|---|---|---|
| σ1, 95th percentile | single mechanics predictor (best core AUC) | 20 | 1.00 |  |  |  |  | 0 |
| Fujino score (cross-sectional) | imaging score | 20 | 0.35 |  |  |  |  | 0 |
| LIFT (σ1, 95th percentile, Area fraction ε1 > εf/2) | composite index (penalised logistic) | 20 | 1.00 | 1.00 (1.00–1.00)† | 0.237 | 0.063 |  | 0 |
| geometry model | penalised logistic | 20 | 1.00 | 0.96 | 0.287 | 0.087 |  | 0 |
| geometry + mechanics model | penalised logistic | 20 | 1.00 | 0.99 | 0.206 | 0.058 |  | 0 |

| Comparison | Set | AUC a | AUC b | ΔAUC (95% CI) | Share of resamples with ΔAUC ≤ 0 | Δ log-loss (95% CI) | Δ Brier (95% CI) |
|---|---|---|---|---|---|---|---|
| LIFT (LOOCV) - Fujino score | core | 1.00 | 0.35 | 0.65 (0.46–0.83) | 0.000 |  |  |
| σ1, 95th percentile - Fujino score | core | 1.00 | 0.35 | 0.65 (0.46–0.83) | 0.000 |  |  |
| LIFT (LOOCV) - σ1, 95th percentile | core | 1.00 | 1.00 | 0.00 (0.00–0.00)† | 1.000 |  |  |
| geometry + mechanics − geometry, apparent (refit in each resample) | core | 1.00 | 1.00 | 0.00 (0.00–0.00) | 0.981 |  |  |
| geometry + mechanics − geometry, LOOCV | core | 0.99 | 0.96 | 0.03 (0.00–0.11) | 0.264 | −0.082 (−0.151 to −0.029) | −0.030 (−0.064 to −0.006) |

† Degenerate bootstrap interval (AUC of 0 or 1, or every resample equal): not a precision statement. LOOCV intervals: bootstrap of fixed out-of-fold predictions (refit and reselection variability not included).
