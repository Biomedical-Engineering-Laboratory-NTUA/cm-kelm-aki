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
| M1 DE-KELM-ARD | 0.3455 [0.3218, 0.3692] | 0.8863 [0.8797, 0.8929] | 0.3439 / 0.3434 | 0.8864 / 0.8860 |
| M2 DE-KELM | 0.3348 [0.3103, 0.3593] | 0.8832 [0.8756, 0.8908] | 0.3332 / 0.3325 | 0.8837 / 0.8825 |
| M4 Grid-tuned KELM | 0.2786 [0.2578, 0.2993] | 0.8712 [0.8636, 0.8787] | 0.2766 / 0.2746 | 0.8702 / 0.8718 |
| M6 XGBoost | 0.3758 [0.3535, 0.3981] | 0.8873 [0.8823, 0.8923] | 0.3727 / 0.3732 | 0.8868 / 0.8877 |
| M8a Logistic regression | 0.2700 [0.2579, 0.2821] | 0.8583 [0.8504, 0.8663] | 0.2668 / 0.2671 | 0.8581 / 0.8583 |
| M8b Random forest | 0.3387 [0.3208, 0.3566] | 0.8804 [0.8743, 0.8866] | 0.3383 / 0.3338 | 0.8815 / 0.8792 |
| LGBM LightGBM | 0.3715 [0.3467, 0.3962] | 0.8847 [0.8780, 0.8913] | 0.3748 / 0.3627 | 0.8849 / 0.8842 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0669 [+0.0426, +0.0912] | 6.23 | 7.684e-05 | 0.0001537 | yes | 0.907 (0.261) |
| H2: DE-KELM-ARD > DE-KELM | +0.0107 [-0.0034, +0.0247] | 1.72 | 0.05989 | 0.05989 | no | 0.936 (0.513) |

Fold-level differences: H1: +0.0844, +0.0542, +0.0488, +0.0942, +0.0494, +0.0437, +0.0850, +0.0625, +0.0649, +0.0822; H2: +0.0201, +0.0074, +0.0009, +0.0264, -0.0063, +0.0028, +0.0070, +0.0200, +0.0081, +0.0203

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0673 [+0.0548, +0.0794], p = 9.999e-05 (10000 replicates)
- H1, repeat 2: +0.0688 [+0.0567, +0.0807], p = 9.999e-05 (10000 replicates)
- H2, repeat 1: +0.0107 [+0.0029, +0.0184], p = 0.0046 (10000 replicates)
- H2, repeat 2: +0.0110 [+0.0034, +0.0188], p = 0.0025 (10000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0107 [-0.0034, +0.0247] | 0.1198 | 0.2396 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0669 [+0.0426, +0.0912] | 0.0001537 | 0.0006147 |
| M1 - XGBoost (10 folds) | AUPRC | -0.0304 [-0.0436, -0.0172] | 0.0005574 | 0.001672 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.0755 [+0.0584, +0.0925] | 3.489e-06 | 1.745e-05 |
| M1 - Random forest (10 folds) | AUPRC | +0.0068 [-0.0095, +0.0230] | 0.3703 | 0.3703 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0031 [+0.0001, +0.0061] | 0.04633 | 0.139 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0152 [+0.0099, +0.0204] | 0.0001128 | 0.0004513 |
| M1 - XGBoost (10 folds) | AUROC | -0.0010 [-0.0042, +0.0022] | 0.5017 | 0.5017 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0280 [+0.0238, +0.0322] | 1.132e-07 | 5.658e-07 |
| M1 - Random forest (10 folds) | AUROC | +0.0059 [-0.0002, +0.0120] | 0.05789 | 0.139 |
