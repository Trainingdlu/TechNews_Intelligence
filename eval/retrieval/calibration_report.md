# G2 Relevance Calibration — LLM vs Human

Labeled pairs: **100**

| Metric | Value |
|---|---|
| Raw agreement (3-class) | 48.0% |
| Cohen's kappa (nominal) | 0.269 |
| Quadratic-weighted kappa (ordinal) | 0.333 |
| Binary-relevance agreement (rel>=1) | 68.0% |
| Binary-relevance Cohen's kappa | 0.352 |

Interpretation (Landis & Koch): <0 poor, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, 0.8-1.0 almost perfect.

## Confusion matrix (rows = human, cols = LLM)

| Human \ LLM | 0 | 1 | 2 |
|---|---|---|---|
| **0** | 18 | 1 | 0 |
| **1** | 9 | 8 | 1 |
| **2** | 22 | 19 | 22 |
