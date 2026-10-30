# Confirmatory analysis: outcome aki, baseline window 365 days

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
| M1 DE-KELM-ARD | 0.2867 [0.2673, 0.3061] | 0.8599 [0.8535, 0.8664] | 0.2838 / 0.2831 | 0.8594 / 0.8602 |
| M2 DE-KELM | 0.2807 [0.2656, 0.2958] | 0.8584 [0.8519, 0.8648] | 0.2767 / 0.2797 | 0.8583 / 0.8582 |
| M4 Grid-tuned KELM | 0.2494 [0.2311, 0.2678] | 0.8478 [0.8396, 0.8561] | 0.2447 / 0.2494 | 0.8469 / 0.8486 |
| M6 XGBoost | 0.2815 [0.2594, 0.3037] | 0.8586 [0.8502, 0.8671] | 0.2773 / 0.2810 | 0.8578 / 0.8591 |
| M8a Logistic regression | 0.1963 [0.1865, 0.2061] | 0.8245 [0.8168, 0.8323] | 0.1938 / 0.1943 | 0.8240 / 0.8249 |
| M8b Random forest | 0.2567 [0.2380, 0.2755] | 0.8501 [0.8384, 0.8619] | 0.2527 / 0.2558 | 0.8503 / 0.8497 |
| LGBM LightGBM | 0.2775 [0.2626, 0.2924] | 0.8571 [0.8492, 0.8650] | 0.2733 / 0.2772 | 0.8561 / 0.8578 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0373 [+0.0199, +0.0546] | 4.85 | 0.0004551 | 0.0009102 | yes | 0.930 (0.448) |
| H2: DE-KELM-ARD > DE-KELM | +0.0060 [-0.0058, +0.0178] | 1.16 | 0.1387 | 0.1387 | no | 0.956 (0.741) |

Fold-level differences: H1: +0.0231, +0.0331, +0.0527, +0.0529, +0.0327, +0.0271, +0.0542, +0.0386, +0.0175, +0.0407; H2: +0.0006, +0.0044, +0.0111, +0.0173, +0.0074, +0.0010, +0.0177, +0.0130, -0.0080, -0.0042

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0391 [+0.0292, +0.0495], p = 9.999e-05 (10000 replicates)
- H1, repeat 2: +0.0338 [+0.0238, +0.0445], p = 9.999e-05 (10000 replicates)
- H2, repeat 1: +0.0071 [+0.0003, +0.0141], p = 0.0204 (10000 replicates)
- H2, repeat 2: +0.0035 [-0.0035, +0.0105], p = 0.1563 (10000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0060 [-0.0058, +0.0178] | 0.2774 | 0.5547 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0373 [+0.0199, +0.0546] | 0.0009102 | 0.003641 |
| M1 - XGBoost (10 folds) | AUPRC | +0.0052 [-0.0148, +0.0251] | 0.573 | 0.573 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.0904 [+0.0736, +0.1071] | 6.636e-07 | 3.318e-06 |
| M1 - Random forest (10 folds) | AUPRC | +0.0300 [+0.0112, +0.0488] | 0.005714 | 0.01714 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0016 [-0.0006, +0.0038] | 0.1357 | 0.2714 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0121 [+0.0087, +0.0155] | 2.145e-05 | 8.579e-05 |
| M1 - XGBoost (10 folds) | AUROC | +0.0013 [-0.0021, +0.0047] | 0.4085 | 0.4085 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0354 [+0.0308, +0.0401] | 3.368e-08 | 1.684e-07 |
| M1 - Random forest (10 folds) | AUROC | +0.0098 [+0.0033, +0.0163] | 0.007823 | 0.02347 |
