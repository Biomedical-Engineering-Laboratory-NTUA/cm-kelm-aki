# Confirmatory analysis: outcome st2, baseline window 365 days

Outer cross-validation 5 folds x 2 repeats (protocol, Section 4). Aggregate results only.

## Availability

| Model | folds | files per fold |
|---|---|---|
| M1 DE-KELM-ARD | 10/10 | 10-10 |
| M2 DE-KELM | 10/10 | 10-10 |
| M4 Grid-tuned KELM | 10/10 | 1-1 |
| M5 Sparse GP (ARD) | 10/10 | 1-1 |
| M6 XGBoost | 10/10 | 1-1 |
| M8a Logistic regression | 10/10 | 1-1 |
| M8b Random forest | 10/10 | 1-1 |
| LGBM LightGBM | 10/10 | 1-1 |

## Discrimination (mean of 10 folds with corrected 95% CI; pooled per repeat)

| Model | AUPRC mean [CI] | AUROC mean [CI] | pooled AUPRC r1 / r2 | pooled AUROC r1 / r2 |
|---|---|---|---|---|
| M1 DE-KELM-ARD | 0.2092 [0.1765, 0.2419] | 0.9027 [0.8894, 0.9160] | 0.1980 / 0.2049 | 0.9028 / 0.9020 |
| M2 DE-KELM | 0.2042 [0.1772, 0.2313] | 0.8990 [0.8846, 0.9134] | 0.1929 / 0.2002 | 0.8984 / 0.8988 |
| M4 Grid-tuned KELM | 0.1639 [0.1527, 0.1752] | 0.8968 [0.8847, 0.9089] | 0.1569 / 0.1608 | 0.8966 / 0.8967 |
| M5 Sparse GP (ARD) | 0.2178 [0.1896, 0.2461] | 0.9005 [0.8878, 0.9131] | 0.2085 / 0.2103 | 0.9008 / 0.8996 |
| M6 XGBoost | 0.2347 [0.2029, 0.2664] | 0.8949 [0.8819, 0.9079] | 0.2306 / 0.2310 | 0.8964 / 0.8929 |
| M8a Logistic regression | 0.1392 [0.1323, 0.1461] | 0.8867 [0.8736, 0.8998] | 0.1361 / 0.1358 | 0.8897 / 0.8832 |
| M8b Random forest | 0.1814 [0.1528, 0.2100] | 0.8911 [0.8758, 0.9064] | 0.1752 / 0.1751 | 0.8911 / 0.8906 |
| LGBM LightGBM | 0.2246 [0.2002, 0.2490] | 0.8931 [0.8777, 0.9085] | 0.2193 / 0.2188 | 0.8939 / 0.8918 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0452 [+0.0172, +0.0733] | 3.64 | 0.002683 | 0.005366 | yes | 0.921 (0.362) |
| H2: DE-KELM-ARD > DE-KELM | +0.0049 [-0.0154, +0.0252] | 0.55 | 0.2976 | 0.2976 | no | 0.913 (0.306) |

Fold-level differences: H1: +0.0465, +0.0468, +0.0151, +0.0343, +0.0702, +0.0717, +0.0339, +0.0143, +0.0667, +0.0531; H2: +0.0126, -0.0161, +0.0093, -0.0033, +0.0177, +0.0168, -0.0213, -0.0031, +0.0239, +0.0130

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0411 [+0.0245, +0.0571], p = 0.0004998 (2000 replicates)
- H1, repeat 2: +0.0441 [+0.0254, +0.0598], p = 0.0004998 (2000 replicates)
- H2, repeat 1: +0.0051 [-0.0061, +0.0156], p = 0.1939 (2000 replicates)
- H2, repeat 2: +0.0047 [-0.0074, +0.0153], p = 0.2164 (2000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0049 [-0.0154, +0.0252] | 0.5952 | 0.6052 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0452 [+0.0172, +0.0733] | 0.005366 | 0.02683 |
| M1 - Sparse GP (ARD) (10 folds) | AUPRC | -0.0087 [-0.0266, +0.0093] | 0.3026 | 0.6052 |
| M1 - XGBoost (10 folds) | AUPRC | -0.0255 [-0.0561, +0.0051] | 0.09167 | 0.275 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.0700 [+0.0406, +0.0994] | 0.0004412 | 0.002647 |
| M1 - Random forest (10 folds) | AUPRC | +0.0278 [-0.0006, +0.0561] | 0.05403 | 0.2161 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0037 [+0.0002, +0.0072] | 0.03938 | 0.1023 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0059 [+0.0011, +0.0108] | 0.02138 | 0.1023 |
| M1 - Sparse GP (ARD) (10 folds) | AUROC | +0.0022 [-0.0025, +0.0070] | 0.3143 | 0.3143 |
| M1 - XGBoost (10 folds) | AUROC | +0.0078 [+0.0015, +0.0141] | 0.02046 | 0.1023 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0160 [+0.0086, +0.0235] | 0.0009028 | 0.005417 |
| M1 - Random forest (10 folds) | AUROC | +0.0116 [+0.0020, +0.0212] | 0.02272 | 0.1023 |
