# Confirmatory analysis: outcome st3, baseline window 365 days

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
| M1 DE-KELM-ARD | 0.2120 [0.1392, 0.2849] | 0.9097 [0.8869, 0.9325] | 0.1957 / 0.1974 | 0.9095 / 0.9078 |
| M2 DE-KELM | 0.2025 [0.1329, 0.2722] | 0.9086 [0.8860, 0.9312] | 0.1874 / 0.1887 | 0.9082 / 0.9075 |
| M4 Grid-tuned KELM | 0.1312 [0.0879, 0.1744] | 0.9005 [0.8787, 0.9223] | 0.1062 / 0.1250 | 0.9024 / 0.8988 |
| M5 Sparse GP (ARD) | 0.2536 [0.2159, 0.2913] | 0.9154 [0.8936, 0.9371] | 0.2387 / 0.2493 | 0.9149 / 0.9145 |
| M6 XGBoost | 0.2801 [0.2318, 0.3285] | 0.9073 [0.8896, 0.9250] | 0.2629 / 0.2685 | 0.9068 / 0.9064 |
| M8a Logistic regression | 0.0893 [0.0698, 0.1087] | 0.8858 [0.8534, 0.9182] | 0.0831 / 0.0846 | 0.8901 / 0.8791 |
| M8b Random forest | 0.2546 [0.1939, 0.3153] | 0.9038 [0.8785, 0.9290] | 0.2416 / 0.2517 | 0.9003 / 0.9062 |
| LGBM LightGBM | 0.2429 [0.1985, 0.2873] | 0.8942 [0.8728, 0.9156] | 0.2293 / 0.2311 | 0.8964 / 0.8893 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0809 [+0.0382, +0.1235] | 4.29 | 0.001009 | 0.002018 | yes | 0.931 (0.460) |
| H2: DE-KELM-ARD > DE-KELM | +0.0095 [-0.0086, +0.0276] | 1.19 | 0.133 | 0.133 | no | 0.967 (0.857) |

Fold-level differences: H1: +0.1236, +0.1052, +0.0891, +0.0241, +0.0932, +0.0674, +0.0641, +0.0375, +0.1048, +0.0996; H2: +0.0180, -0.0091, +0.0337, -0.0114, +0.0158, +0.0101, +0.0071, +0.0074, +0.0203, +0.0031

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0895 [+0.0607, +0.1218], p = 0.0004998 (2000 replicates)
- H1, repeat 2: +0.0724 [+0.0513, +0.0962], p = 0.0004998 (2000 replicates)
- H2, repeat 1: +0.0083 [-0.0018, +0.0199], p = 0.05097 (2000 replicates)
- H2, repeat 2: +0.0087 [-0.0017, +0.0214], p = 0.04748 (2000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0095 [-0.0086, +0.0276] | 0.266 | 0.266 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0809 [+0.0382, +0.1235] | 0.002018 | 0.01198 |
| M1 - Sparse GP (ARD) (10 folds) | AUPRC | -0.0415 [-0.0834, +0.0004] | 0.05167 | 0.1356 |
| M1 - XGBoost (10 folds) | AUPRC | -0.0681 [-0.1297, -0.0065] | 0.0339 | 0.1356 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.1228 [+0.0582, +0.1874] | 0.001996 | 0.01198 |
| M1 - Random forest (10 folds) | AUPRC | -0.0426 [-0.0836, -0.0015] | 0.0437 | 0.1356 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0011 [-0.0032, +0.0053] | 0.5798 | 1 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0092 [-0.0021, +0.0205] | 0.09879 | 0.494 |
| M1 - Sparse GP (ARD) (10 folds) | AUROC | -0.0057 [-0.0154, +0.0041] | 0.221 | 0.8841 |
| M1 - XGBoost (10 folds) | AUROC | +0.0024 [-0.0122, +0.0169] | 0.7185 | 1 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0239 [-0.0038, +0.0516] | 0.08233 | 0.494 |
| M1 - Random forest (10 folds) | AUROC | +0.0059 [-0.0056, +0.0175] | 0.2747 | 0.8841 |
