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
| M1 DE-KELM-ARD | 0.3528 [0.3249, 0.3807] | 0.8890 [0.8794, 0.8986] | 0.3486 / 0.3508 | 0.8888 / 0.8890 |
| M2 DE-KELM | 0.3448 [0.3204, 0.3692] | 0.8863 [0.8765, 0.8961] | 0.3420 / 0.3420 | 0.8863 / 0.8861 |
| M4 Grid-tuned KELM | 0.3046 [0.2797, 0.3294] | 0.8786 [0.8691, 0.8880] | 0.3053 / 0.2980 | 0.8791 / 0.8779 |
| M6 XGBoost | 0.3831 [0.3510, 0.4153] | 0.8891 [0.8799, 0.8982] | 0.3863 / 0.3752 | 0.8890 / 0.8890 |
| M8a Logistic regression | 0.2811 [0.2638, 0.2984] | 0.8618 [0.8500, 0.8737] | 0.2798 / 0.2755 | 0.8615 / 0.8619 |
| M8b Random forest | 0.3450 [0.3186, 0.3714] | 0.8823 [0.8706, 0.8939] | 0.3444 / 0.3401 | 0.8826 / 0.8816 |
| LGBM LightGBM | 0.3819 [0.3490, 0.4148] | 0.8882 [0.8762, 0.9002] | 0.3783 / 0.3801 | 0.8873 / 0.8890 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0482 [+0.0273, +0.0692] | 5.21 | 0.0002794 | 0.0005588 | yes | 0.949 (0.651) |
| H2: DE-KELM-ARD > DE-KELM | +0.0080 [-0.0010, +0.0170] | 2.02 | 0.03728 | 0.03728 | yes | 0.977 (0.948) |

Fold-level differences: H1: +0.0414, +0.0746, +0.0410, +0.0414, +0.0173, +0.0518, +0.0494, +0.0432, +0.0654, +0.0568; H2: +0.0007, +0.0193, +0.0126, +0.0067, -0.0028, +0.0080, +0.0136, +0.0019, +0.0117, +0.0083

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0433 [+0.0330, +0.0536], p = 9.999e-05 (10000 replicates)
- H1, repeat 2: +0.0528 [+0.0419, +0.0637], p = 9.999e-05 (10000 replicates)
- H2, repeat 1: +0.0067 [+0.0001, +0.0132], p = 0.0231 (10000 replicates)
- H2, repeat 2: +0.0088 [+0.0021, +0.0151], p = 0.0049 (10000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0080 [-0.0010, +0.0170] | 0.07456 | 0.1491 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0482 [+0.0273, +0.0692] | 0.0005588 | 0.002235 |
| M1 - XGBoost (10 folds) | AUPRC | -0.0303 [-0.0511, -0.0096] | 0.009119 | 0.02736 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.0717 [+0.0524, +0.0910] | 1.467e-05 | 7.335e-05 |
| M1 - Random forest (10 folds) | AUPRC | +0.0078 [-0.0202, +0.0358] | 0.5454 | 0.5454 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0027 [+0.0009, +0.0046] | 0.00947 | 0.01894 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0104 [+0.0074, +0.0135] | 2.847e-05 | 0.0001139 |
| M1 - XGBoost (10 folds) | AUROC | -0.0000 [-0.0056, +0.0055] | 0.9897 | 0.9897 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0272 [+0.0204, +0.0339] | 7.651e-06 | 3.826e-05 |
| M1 - Random forest (10 folds) | AUROC | +0.0068 [+0.0028, +0.0107] | 0.003651 | 0.01095 |
