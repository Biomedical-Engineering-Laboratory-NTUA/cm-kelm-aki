# Confirmatory analysis: outcome aki, baseline window 7 days

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
| M1 DE-KELM-ARD | 0.4089 [0.3742, 0.4437] | 0.8972 [0.8900, 0.9044] | 0.4056 / 0.4055 | 0.8980 / 0.8963 |
| M2 DE-KELM | 0.3770 [0.3336, 0.4205] | 0.8910 [0.8832, 0.8988] | 0.3719 / 0.3726 | 0.8915 / 0.8902 |
| M4 Grid-tuned KELM | 0.3174 [0.2782, 0.3566] | 0.8763 [0.8694, 0.8833] | 0.3139 / 0.3113 | 0.8771 / 0.8753 |
| M6 XGBoost | 0.4211 [0.3916, 0.4506] | 0.8953 [0.8872, 0.9034] | 0.4203 / 0.4153 | 0.8969 / 0.8934 |
| M8a Logistic regression | 0.2848 [0.2590, 0.3106] | 0.8662 [0.8600, 0.8724] | 0.2815 / 0.2813 | 0.8667 / 0.8654 |
| M8b Random forest | 0.3811 [0.3608, 0.4014] | 0.8890 [0.8821, 0.8959] | 0.3832 / 0.3759 | 0.8896 / 0.8881 |
| LGBM LightGBM | 0.4192 [0.3844, 0.4540] | 0.8941 [0.8840, 0.9042] | 0.4224 / 0.4096 | 0.8946 / 0.8932 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0916 [+0.0693, +0.1139] | 9.30 | 3.261e-06 | 6.521e-06 | yes | 0.930 (0.453) |
| H2: DE-KELM-ARD > DE-KELM | +0.0319 [+0.0107, +0.0532] | 3.40 | 0.003919 | 0.003919 | yes | 0.972 (0.905) |

Fold-level differences: H1: +0.1198, +0.0942, +0.0850, +0.0655, +0.0919, +0.0915, +0.1044, +0.0654, +0.0981, +0.1000; H2: +0.0615, +0.0373, +0.0027, +0.0345, +0.0301, +0.0448, +0.0221, +0.0173, +0.0324, +0.0366

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0917 [+0.0758, +0.1063], p = 0.0004998 (2000 replicates)
- H1, repeat 2: +0.0942 [+0.0791, +0.1094], p = 0.0004998 (2000 replicates)
- H2, repeat 1: +0.0336 [+0.0223, +0.0454], p = 0.0004998 (2000 replicates)
- H2, repeat 2: +0.0329 [+0.0221, +0.0427], p = 0.0004998 (2000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0319 [+0.0107, +0.0532] | 0.007838 | 0.02351 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0916 [+0.0693, +0.1139] | 6.521e-06 | 2.608e-05 |
| M1 - XGBoost (10 folds) | AUPRC | -0.0122 [-0.0259, +0.0016] | 0.07686 | 0.07686 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.1241 [+0.1038, +0.1444] | 2.274e-07 | 1.137e-06 |
| M1 - Random forest (10 folds) | AUPRC | +0.0278 [+0.0079, +0.0477] | 0.01157 | 0.02351 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0063 [+0.0029, +0.0096] | 0.002249 | 0.004498 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0209 [+0.0146, +0.0272] | 3.756e-05 | 0.0001502 |
| M1 - XGBoost (10 folds) | AUROC | +0.0019 [-0.0023, +0.0061] | 0.3264 | 0.3264 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0310 [+0.0225, +0.0396] | 1.785e-05 | 8.924e-05 |
| M1 - Random forest (10 folds) | AUROC | +0.0083 [+0.0046, +0.0120] | 0.0006651 | 0.001995 |
