# Confirmatory analysis: outcome aki, baseline window 90 days

Outer cross-validation 5 folds x 2 repeats (protocol, Section 4). Aggregate results only.

## Availability

| Model | folds | files per fold |
|---|---|---|
| M1 DE-KELM-ARD | 10/10 | 10-10 |
| M2 DE-KELM | 10/10 | 10-10 |
| M4 Grid-tuned KELM | 10/10 | 1-1 |
| M6 XGBoost | 10/10 | 1-1 |
| M8a Logistic regression | 10/10 | 1-1 |
| M8b Random forest | 10/10 | 1-1 |
| LGBM LightGBM | 10/10 | 1-1 |

## Discrimination (mean of 10 folds with corrected 95% CI; pooled per repeat)

| Model | AUPRC mean [CI] | AUROC mean [CI] | pooled AUPRC r1 / r2 | pooled AUROC r1 / r2 |
|---|---|---|---|---|
| M1 DE-KELM-ARD | 0.3647 [0.3528, 0.3765] | 0.8910 [0.8833, 0.8986] | 0.3633 / 0.3618 | 0.8909 / 0.8907 |
| M2 DE-KELM | 0.3516 [0.3355, 0.3677] | 0.8866 [0.8782, 0.8951] | 0.3498 / 0.3494 | 0.8872 / 0.8856 |
| M4 Grid-tuned KELM | 0.2844 [0.2675, 0.3012] | 0.8716 [0.8631, 0.8802] | 0.2807 / 0.2836 | 0.8720 / 0.8711 |
| M6 XGBoost | 0.3888 [0.3744, 0.4032] | 0.8909 [0.8846, 0.8972] | 0.3854 / 0.3870 | 0.8909 / 0.8906 |
| M8a Logistic regression | 0.2597 [0.2418, 0.2776] | 0.8560 [0.8471, 0.8648] | 0.2580 / 0.2566 | 0.8556 / 0.8562 |
| M8b Random forest | 0.3461 [0.3349, 0.3574] | 0.8815 [0.8746, 0.8884] | 0.3420 / 0.3446 | 0.8824 / 0.8805 |
| LGBM LightGBM | 0.3821 [0.3677, 0.3965] | 0.8888 [0.8829, 0.8947] | 0.3798 / 0.3802 | 0.8891 / 0.8882 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0803 [+0.0676, +0.0930] | 14.34 | 8.35e-08 | 1.67e-07 | yes | 0.961 (0.801) |
| H2: DE-KELM-ARD > DE-KELM | +0.0131 [+0.0032, +0.0230] | 3.01 | 0.007411 | 0.007411 | yes | 0.948 (0.649) |

Fold-level differences: H1: +0.0725, +0.0799, +0.0951, +0.0794, +0.0847, +0.0775, +0.0607, +0.0864, +0.0888, +0.0779; H2: +0.0162, +0.0122, +0.0106, +0.0257, +0.0046, +0.0064, +0.0140, +0.0034, +0.0148, +0.0231

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0826 [+0.0696, +0.0956], p = 9.999e-05 (10000 replicates)
- H1, repeat 2: +0.0782 [+0.0655, +0.0911], p = 9.999e-05 (10000 replicates)
- H2, repeat 1: +0.0135 [+0.0040, +0.0231], p = 0.0031 (10000 replicates)
- H2, repeat 2: +0.0124 [+0.0034, +0.0211], p = 0.0036 (10000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0131 [+0.0032, +0.0230] | 0.01482 | 0.03705 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0803 [+0.0676, +0.0930] | 1.67e-07 | 6.68e-07 |
| M1 - XGBoost (10 folds) | AUPRC | -0.0242 [-0.0417, -0.0066] | 0.01235 | 0.03705 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.1049 [+0.0938, +0.1161] | 5.134e-09 | 2.567e-08 |
| M1 - Random forest (10 folds) | AUPRC | +0.0185 [+0.0015, +0.0355] | 0.03583 | 0.03705 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0044 [+0.0010, +0.0077] | 0.01653 | 0.03305 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0193 [+0.0157, +0.0229] | 6.979e-07 | 2.792e-06 |
| M1 - XGBoost (10 folds) | AUROC | +0.0001 [-0.0023, +0.0024] | 0.9552 | 0.9552 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0350 [+0.0295, +0.0405] | 1.557e-07 | 7.785e-07 |
| M1 - Random forest (10 folds) | AUROC | +0.0095 [+0.0050, +0.0141] | 0.001086 | 0.003257 |
