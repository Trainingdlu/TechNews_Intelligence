# G2 Relevance Calibration — LLM vs Human

Labeled pairs: **100**

| Metric | Value |
|---|---|
| Raw agreement (3-class) | 57.0% |
| Cohen's kappa (nominal) | 0.356 |
| Quadratic-weighted kappa (ordinal) | 0.514 |
| Binary-relevance agreement (rel>=1) | 77.0% |
| Binary-relevance Cohen's kappa | 0.515 |

Interpretation (Landis & Koch): <0 poor, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, 0.8-1.0 almost perfect.

## Confusion matrix (rows = human, cols = LLM)

| Human \ LLM | 0 | 1 | 2 |
|---|---|---|---|
| **0** | 27 | 7 | 3 |
| **1** | 4 | 5 | 1 |
| **2** | 9 | 19 | 25 |
