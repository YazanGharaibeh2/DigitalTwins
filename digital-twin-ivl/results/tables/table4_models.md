| Model | Type | n core | Core AUC, apparent | Core AUC, LOOCV (95% CI) | LOOCV log-loss | LOOCV Brier | Hold-out AUC (95% CI) | n hold-out |
|---|---|---|---|---|---|---|---|---|
| σ1, non-local peak | single mechanics predictor (best core AUC) | 32 | 0.96 |  |  |  |  | 0 |
| Fujino score (cross-sectional) | imaging score | 32 | 0.62 |  |  |  |  | 0 |
| LIFT (σ1, non-local peak, σ1, mean, Edge stress concentration) | composite index (penalised logistic) | 32 | 0.99 | 0.95 (0.85–1.00) | 0.284 | 0.081 |  | 0 |
| geometry model | penalised logistic | 32 | 0.94 | 0.88 | 0.499 | 0.129 |  | 0 |
| geometry + mechanics model | penalised logistic | 32 | 0.98 | 0.92 | 0.384 | 0.096 |  | 0 |

| Comparison | Set | AUC a | AUC b | ΔAUC (95% CI) | Share of resamples with ΔAUC ≤ 0 | Δ log-loss (95% CI) | Δ Brier (95% CI) |
|---|---|---|---|---|---|---|---|
| LIFT (LOOCV) - Fujino score | core | 0.95 | 0.62 | 0.34 (0.18–0.52) | 0.000 |  |  |
| σ1, non-local peak - Fujino score | core | 0.96 | 0.62 | 0.35 (0.18–0.54) | 0.000 |  |  |
| LIFT (LOOCV) - σ1, non-local peak | core | 0.95 | 0.96 | −0.01 (−0.06 to 0.02) | 0.792 |  |  |
| geometry + mechanics − geometry, apparent (refit in each resample) | core | 0.98 | 0.94 | 0.04 (0.00–0.14) | 0.324 |  |  |
| geometry + mechanics − geometry, LOOCV | core | 0.92 | 0.88 | 0.04 (−0.01 to 0.12) | 0.099 | −0.115 (−0.179 to −0.052) | −0.034 (−0.059 to −0.009) |

† Degenerate bootstrap interval (AUC of 0 or 1, or every resample equal): not a precision statement. LOOCV intervals: bootstrap of fixed out-of-fold predictions (refit and reselection variability not included).
