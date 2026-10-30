# Confirmatory analysis: outcome aki, baseline window 365 days

Outer cross-validation 5 folds x 2 repeats (protocol, Section 4). Aggregate results only.

## Availability

| Model | folds | files per fold |
|---|---|---|
| M1 DE-KELM-ARD | 10/10 | 10-10 |
| M2 DE-KELM | 10/10 | 10-10 |
| M3 CM-KELM | 10/10 | 5-5 |
| M4 Grid-tuned KELM | 10/10 | 1-1 |
| M5 Sparse GP (ARD) | 10/10 | 1-1 |
| M6 XGBoost | 10/10 | 1-1 |
| M7 FT-Transformer | 10/10 | 1-1 |
| M8a Logistic regression | 10/10 | 1-1 |
| M8b Random forest | 10/10 | 1-1 |
| LGBM LightGBM | 10/10 | 1-1 |
| RS-ARD Random search, ARD weights | 5/10 | 10-10 |
| RS-mask Random search, masks | 5/10 | 10-10 |
| RS-ARD-isl Random search, ARD weights (island ensembles) | 5/10 | 10-10 |
| RS-mask-isl Random search, masks (island ensembles) | 5/10 | 10-10 |
| CM-ARD CM-KELM-ARD (top-5) | 10/10 | 10-10 |
| M3-top5 CM-KELM (top-5) | 10/10 | 5-5 |

## Discrimination (mean of 10 folds with corrected 95% CI; pooled per repeat)

| Model | AUPRC mean [CI] | AUROC mean [CI] | pooled AUPRC r1 / r2 | pooled AUROC r1 / r2 |
|---|---|---|---|---|
| M1 DE-KELM-ARD | 0.3610 [0.3389, 0.3831] | 0.8896 [0.8850, 0.8943] | 0.3607 / 0.3579 | 0.8891 / 0.8900 |
| M2 DE-KELM | 0.3484 [0.3238, 0.3730] | 0.8855 [0.8794, 0.8916] | 0.3465 / 0.3448 | 0.8858 / 0.8851 |
| M3 CM-KELM | 0.3383 [0.3147, 0.3620] | 0.8847 [0.8786, 0.8907] | 0.3334 / 0.3394 | 0.8837 / 0.8854 |
| M4 Grid-tuned KELM | 0.2811 [0.2637, 0.2985] | 0.8704 [0.8645, 0.8762] | 0.2743 / 0.2839 | 0.8698 / 0.8709 |
| M5 Sparse GP (ARD) | 0.3618 [0.3399, 0.3837] | 0.8882 [0.8823, 0.8941] | 0.3608 / 0.3582 | 0.8878 / 0.8884 |
| M6 XGBoost | 0.3869 [0.3636, 0.4102] | 0.8898 [0.8817, 0.8979] | 0.3881 / 0.3804 | 0.8902 / 0.8891 |
| M7 FT-Transformer | 0.3706 [0.3412, 0.4001] | 0.8881 [0.8797, 0.8965] | 0.3624 / 0.3654 | 0.8874 / 0.8872 |
| M8a Logistic regression | 0.2602 [0.2501, 0.2703] | 0.8564 [0.8523, 0.8606] | 0.2586 / 0.2569 | 0.8563 / 0.8566 |
| M8b Random forest | 0.3469 [0.3238, 0.3699] | 0.8822 [0.8751, 0.8893] | 0.3471 / 0.3416 | 0.8830 / 0.8812 |
| LGBM LightGBM | 0.3853 [0.3644, 0.4061] | 0.8885 [0.8814, 0.8957] | 0.3896 / 0.3765 | 0.8881 / 0.8887 |
| RS-ARD Random search, ARD weights | 0.3190 [0.2746, 0.3635] | 0.8837 [0.8753, 0.8920] | 0.3176 / - | 0.8836 / - |
| RS-mask Random search, masks | 0.3342 [0.2927, 0.3756] | 0.8839 [0.8756, 0.8922] | 0.3320 / - | 0.8838 / - |
| RS-ARD-isl Random search, ARD weights (island ensembles) | 0.3204 [0.2754, 0.3653] | 0.8840 [0.8758, 0.8921] | 0.3182 / - | 0.8839 / - |
| RS-mask-isl Random search, masks (island ensembles) | 0.3317 [0.2909, 0.3725] | 0.8836 [0.8757, 0.8916] | 0.3298 / - | 0.8836 / - |
| CM-ARD CM-KELM-ARD (top-5) | 0.3551 [0.3318, 0.3784] | 0.8895 [0.8835, 0.8954] | 0.3534 / 0.3520 | 0.8889 / 0.8900 |
| M3-top5 CM-KELM (top-5) | 0.3401 [0.3158, 0.3645] | 0.8848 [0.8791, 0.8905] | 0.3350 / 0.3417 | 0.8838 / 0.8855 |

## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)

| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |
|---|---|---|---|---|---|---|
| H1: DE-KELM-ARD > Grid-tuned KELM | +0.0799 [+0.0594, +0.1004] | 8.83 | 4.989e-06 | 9.977e-06 | yes | 0.947 (0.633) |
| H2: DE-KELM-ARD > DE-KELM | +0.0126 [+0.0030, +0.0222] | 2.98 | 0.007782 | 0.007782 | yes | 0.960 (0.788) |

Fold-level differences: H1: +0.0962, +0.0569, +0.0930, +0.0858, +0.1012, +0.0835, +0.0733, +0.0828, +0.0665, +0.0597; H2: +0.0157, +0.0030, +0.0049, +0.0259, +0.0149, +0.0061, +0.0163, +0.0101, +0.0196, +0.0097

Sensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):

- H1, repeat 1: +0.0864 [+0.0738, +0.0990], p = 9.999e-05 (10000 replicates)
- H1, repeat 2: +0.0740 [+0.0617, +0.0866], p = 9.999e-05 (10000 replicates)
- H2, repeat 1: +0.0142 [+0.0047, +0.0235], p = 0.0017 (10000 replicates)
- H2, repeat 2: +0.0131 [+0.0044, +0.0222], p = 0.0018 (10000 replicates)

## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)

| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |
|---|---|---|---|---|
| M1 - DE-KELM (10 folds) | AUPRC | +0.0126 [+0.0030, +0.0222] | 0.01556 | 0.07782 |
| M1 - CM-KELM (10 folds) | AUPRC | +0.0226 [+0.0085, +0.0368] | 0.005497 | 0.03298 |
| M1 - Grid-tuned KELM (10 folds) | AUPRC | +0.0799 [+0.0594, +0.1004] | 9.977e-06 | 7.982e-05 |
| M1 - Sparse GP (ARD) (10 folds) | AUPRC | -0.0008 [-0.0162, +0.0146] | 0.9108 | 0.9108 |
| M1 - XGBoost (10 folds) | AUPRC | -0.0259 [-0.0468, -0.0050] | 0.02046 | 0.08185 |
| M1 - FT-Transformer (10 folds) | AUPRC | -0.0096 [-0.0356, +0.0163] | 0.4225 | 0.8449 |
| M1 - Logistic regression (10 folds) | AUPRC | +0.1008 [+0.0730, +0.1285] | 1.798e-05 | 0.0001259 |
| M1 - Random forest (10 folds) | AUPRC | +0.0141 [-0.0007, +0.0289] | 0.05956 | 0.1787 |
| M1 - DE-KELM (10 folds) | AUROC | +0.0041 [+0.0013, +0.0069] | 0.008767 | 0.04383 |
| M1 - CM-KELM (10 folds) | AUROC | +0.0050 [+0.0030, +0.0069] | 0.0002684 | 0.00161 |
| M1 - Grid-tuned KELM (10 folds) | AUROC | +0.0193 [+0.0133, +0.0253] | 4.708e-05 | 0.0003296 |
| M1 - Sparse GP (ARD) (10 folds) | AUROC | +0.0014 [-0.0013, +0.0042] | 0.2735 | 0.8204 |
| M1 - XGBoost (10 folds) | AUROC | -0.0001 [-0.0051, +0.0048] | 0.9476 | 0.9977 |
| M1 - FT-Transformer (10 folds) | AUROC | +0.0015 [-0.0033, +0.0063] | 0.4989 | 0.9977 |
| M1 - Logistic regression (10 folds) | AUROC | +0.0332 [+0.0266, +0.0397] | 1.173e-06 | 9.382e-06 |
| M1 - Random forest (10 folds) | AUROC | +0.0074 [+0.0015, +0.0133] | 0.01937 | 0.07748 |
