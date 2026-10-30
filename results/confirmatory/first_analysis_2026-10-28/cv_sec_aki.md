# Secondary analyses: outcome aki, baseline window 365 days

Outer CV 5x2 (protocol Sections 7-9). Recalibration: Platt scaling cross-fitted inside each test fold (5 internal splits). Aggregate results only.

## Availability (order check passed: predictions align with fold_ids)

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

## Calibration (mean of folds [corrected 95% CI])

| Model | Brier raw | Brier recal. | intercept raw | slope raw | ICI raw | intercept recal. | slope recal. | ICI recal. |
|---|---|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 0.1650 [0.1633, 0.1666] | 0.0372 [0.0366, 0.0379] | -2.671 | 3.121 | 0.352 | -0.000 | 0.998 | 0.007 |
| M2 DE-KELM | 0.1650 [0.1632, 0.1668] | 0.0377 [0.0370, 0.0383] | -2.671 | 3.012 | 0.352 | -0.000 | 0.998 | 0.007 |
| M3 CM-KELM | 0.1652 [0.1633, 0.1671] | 0.0379 [0.0372, 0.0386] | -2.672 | 2.972 | 0.352 | -0.000 | 0.998 | 0.007 |
| M4 Grid-tuned KELM | 0.1652 [0.1643, 0.1661] | 0.0397 [0.0391, 0.0403] | -2.680 | 2.513 | 0.349 | -0.000 | 0.997 | 0.006 |
| M5 Sparse GP (ARD) | 0.0373 [0.0368, 0.0378] | 0.0372 [0.0367, 0.0378] | -0.032 | 1.095 | 0.008 | -0.000 | 0.998 | 0.006 |
| M6 XGBoost | 0.0365 [0.0358, 0.0373] | 0.0365 [0.0358, 0.0372] | 0.101 | 0.969 | 0.006 | -0.000 | 0.998 | 0.005 |
| M7 FT-Transformer | 0.1302 [0.1159, 0.1445] | 0.0373 [0.0366, 0.0379] | -2.907 | 0.957 | 0.221 | -0.000 | 0.998 | 0.006 |
| M8a Logistic regression | 0.1481 [0.1462, 0.1500] | 0.0410 [0.0406, 0.0414] | -3.076 | 0.802 | 0.270 | -0.000 | 0.998 | 0.011 |
| M8b Random forest | 0.0403 [0.0396, 0.0411] | 0.0377 [0.0369, 0.0384] | -0.768 | 1.222 | 0.041 | -0.000 | 0.999 | 0.006 |
| LGBM LightGBM | 0.0366 [0.0358, 0.0374] | 0.0365 [0.0357, 0.0373] | 0.228 | 0.922 | 0.009 | -0.000 | 0.998 | 0.006 |
| RS-ARD Random search, ARD weights | 0.1635 [0.1610, 0.1660] | 0.0383 [0.0370, 0.0397] | -2.658 | 3.023 | 0.348 | -0.000 | 0.998 | 0.006 |
| RS-mask Random search, masks | 0.1656 [0.1637, 0.1674] | 0.0380 [0.0368, 0.0392] | -2.673 | 2.966 | 0.352 | -0.000 | 0.998 | 0.007 |

Raw outputs of the kernel models are not probabilities (KELM regression scores, clipped to [0, 1] for the raw Brier); compare models after recalibration.

## Risk thresholds (recalibrated; mean of folds)

| Model | threshold | flagged | sensitivity | specificity | PPV | NPV | net benefit [corrected 95% CI] |
|---|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 5% | 0.216 | 0.802 | 0.814 | 0.182 | 0.988 | 0.0299 [0.0293, 0.0304] |
| M1 DE-KELM-ARD | 10% | 0.145 | 0.698 | 0.883 | 0.235 | 0.983 | 0.0218 [0.0206, 0.0229] |
| M1 DE-KELM-ARD | 20% | 0.075 | 0.523 | 0.948 | 0.341 | 0.975 | 0.0132 [0.0120, 0.0143] |
| M2 DE-KELM | 5% | 0.213 | 0.791 | 0.817 | 0.182 | 0.987 | 0.0295 [0.0288, 0.0302] |
| M2 DE-KELM | 10% | 0.144 | 0.686 | 0.883 | 0.232 | 0.982 | 0.0212 [0.0199, 0.0225] |
| M2 DE-KELM | 20% | 0.075 | 0.513 | 0.947 | 0.332 | 0.974 | 0.0125 [0.0113, 0.0137] |
| M3 CM-KELM | 5% | 0.210 | 0.792 | 0.820 | 0.184 | 0.987 | 0.0296 [0.0291, 0.0302] |
| M3 CM-KELM | 10% | 0.144 | 0.686 | 0.884 | 0.233 | 0.982 | 0.0212 [0.0198, 0.0226] |
| M3 CM-KELM | 20% | 0.075 | 0.510 | 0.947 | 0.331 | 0.974 | 0.0123 [0.0109, 0.0137] |
| M4 Grid-tuned KELM | 5% | 0.222 | 0.791 | 0.807 | 0.174 | 0.987 | 0.0290 [0.0279, 0.0301] |
| M4 Grid-tuned KELM | 10% | 0.143 | 0.662 | 0.883 | 0.225 | 0.981 | 0.0200 [0.0187, 0.0213] |
| M4 Grid-tuned KELM | 20% | 0.064 | 0.428 | 0.954 | 0.325 | 0.970 | 0.0100 [0.0091, 0.0109] |
| M5 Sparse GP (ARD) | 5% | 0.227 | 0.816 | 0.804 | 0.176 | 0.988 | 0.0300 [0.0291, 0.0309] |
| M5 Sparse GP (ARD) | 10% | 0.144 | 0.700 | 0.884 | 0.237 | 0.983 | 0.0219 [0.0210, 0.0229] |
| M5 Sparse GP (ARD) | 20% | 0.072 | 0.511 | 0.950 | 0.344 | 0.974 | 0.0131 [0.0118, 0.0144] |
| M6 XGBoost | 5% | 0.210 | 0.792 | 0.820 | 0.185 | 0.987 | 0.0297 [0.0291, 0.0302] |
| M6 XGBoost | 10% | 0.121 | 0.660 | 0.907 | 0.267 | 0.981 | 0.0224 [0.0208, 0.0241] |
| M6 XGBoost | 20% | 0.062 | 0.483 | 0.960 | 0.383 | 0.973 | 0.0141 [0.0130, 0.0152] |
| M7 FT-Transformer | 5% | 0.233 | 0.821 | 0.797 | 0.172 | 0.989 | 0.0299 [0.0289, 0.0310] |
| M7 FT-Transformer | 10% | 0.140 | 0.693 | 0.888 | 0.242 | 0.983 | 0.0220 [0.0208, 0.0233] |
| M7 FT-Transformer | 20% | 0.073 | 0.505 | 0.949 | 0.336 | 0.974 | 0.0125 [0.0110, 0.0140] |
| M8a Logistic regression | 5% | 0.230 | 0.779 | 0.798 | 0.165 | 0.986 | 0.0279 [0.0274, 0.0284] |
| M8a Logistic regression | 10% | 0.126 | 0.589 | 0.898 | 0.229 | 0.977 | 0.0180 [0.0166, 0.0194] |
| M8a Logistic regression | 20% | 0.045 | 0.291 | 0.968 | 0.318 | 0.964 | 0.0066 [0.0056, 0.0076] |
| M8b Random forest | 5% | 0.240 | 0.814 | 0.790 | 0.166 | 0.988 | 0.0292 [0.0285, 0.0299] |
| M8b Random forest | 10% | 0.134 | 0.668 | 0.893 | 0.243 | 0.981 | 0.0213 [0.0199, 0.0228] |
| M8b Random forest | 20% | 0.057 | 0.431 | 0.962 | 0.370 | 0.971 | 0.0121 [0.0104, 0.0138] |
| LGBM LightGBM | 5% | 0.210 | 0.789 | 0.819 | 0.183 | 0.987 | 0.0295 [0.0286, 0.0303] |
| LGBM LightGBM | 10% | 0.122 | 0.655 | 0.906 | 0.263 | 0.981 | 0.0220 [0.0206, 0.0235] |
| LGBM LightGBM | 20% | 0.062 | 0.480 | 0.959 | 0.377 | 0.973 | 0.0138 [0.0121, 0.0155] |
| RS-ARD Random search, ARD weights | 5% | 0.219 | 0.802 | 0.810 | 0.178 | 0.988 | 0.0297 [0.0281, 0.0312] |
| RS-ARD Random search, ARD weights | 10% | 0.149 | 0.696 | 0.879 | 0.228 | 0.983 | 0.0212 [0.0187, 0.0238] |
| RS-ARD Random search, ARD weights | 20% | 0.075 | 0.495 | 0.946 | 0.321 | 0.973 | 0.0114 [0.0087, 0.0140] |
| RS-mask Random search, masks | 5% | 0.212 | 0.785 | 0.818 | 0.181 | 0.987 | 0.0292 [0.0280, 0.0305] |
| RS-mask Random search, masks | 10% | 0.146 | 0.688 | 0.882 | 0.230 | 0.982 | 0.0211 [0.0187, 0.0234] |
| RS-mask Random search, masks | 20% | 0.075 | 0.503 | 0.947 | 0.327 | 0.974 | 0.0119 [0.0103, 0.0136] |

Net benefit, M1 minus comparator (corrected t-test, two-sided):

| Comparator | 5% | 10% | 20% |
|---|---|---|---|
| M2 DE-KELM | +0.0004 (p=0.198) | +0.0006 (p=0.106) | +0.0007 (p=0.0752) |
| M3 CM-KELM | +0.0002 (p=0.45) | +0.0005 (p=0.0995) | +0.0009 (p=0.0173) |
| M4 Grid-tuned KELM | +0.0009 (p=0.0606) | +0.0018 (p=0.000185) | +0.0032 (p=0.000457) |
| M5 Sparse GP (ARD) | -0.0001 (p=0.582) | -0.0002 (p=0.637) | +0.0001 (p=0.735) |
| M6 XGBoost | +0.0002 (p=0.463) | -0.0007 (p=0.2) | -0.0009 (p=0.0942) |
| M7 FT-Transformer | -0.0001 (p=0.885) | -0.0003 (p=0.581) | +0.0007 (p=0.103) |
| M8a Logistic regression | +0.0019 (p=0.000256) | +0.0038 (p=0.00155) | +0.0066 (p=7.72e-07) |
| M8b Random forest | +0.0006 (p=0.135) | +0.0004 (p=0.387) | +0.0011 (p=0.0894) |
| LGBM LightGBM | +0.0004 (p=0.338) | -0.0003 (p=0.51) | -0.0006 (p=0.46) |

## Fixed-sensitivity operating points (threshold from one half of the test fold, applied to the other half; mean of folds)

| Model | target | achieved sensitivity | specificity | PPV | NPV | flagged |
|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 80% | 0.799 | 0.815 | 0.182 | 0.988 | 0.215 |
| M1 DE-KELM-ARD | 90% | 0.903 | 0.692 | 0.132 | 0.993 | 0.337 |
| M2 DE-KELM | 80% | 0.797 | 0.811 | 0.178 | 0.987 | 0.219 |
| M2 DE-KELM | 90% | 0.901 | 0.688 | 0.130 | 0.993 | 0.341 |
| M3 CM-KELM | 80% | 0.799 | 0.811 | 0.179 | 0.987 | 0.218 |
| M3 CM-KELM | 90% | 0.900 | 0.680 | 0.127 | 0.993 | 0.349 |
| M4 Grid-tuned KELM | 80% | 0.799 | 0.799 | 0.170 | 0.987 | 0.230 |
| M4 Grid-tuned KELM | 90% | 0.898 | 0.650 | 0.117 | 0.992 | 0.377 |
| M5 Sparse GP (ARD) | 80% | 0.800 | 0.818 | 0.185 | 0.988 | 0.212 |
| M5 Sparse GP (ARD) | 90% | 0.900 | 0.693 | 0.132 | 0.993 | 0.336 |
| M6 XGBoost | 80% | 0.798 | 0.816 | 0.182 | 0.987 | 0.214 |
| M6 XGBoost | 90% | 0.902 | 0.690 | 0.130 | 0.993 | 0.339 |
| M7 FT-Transformer | 80% | 0.801 | 0.812 | 0.180 | 0.988 | 0.218 |
| M7 FT-Transformer | 90% | 0.897 | 0.691 | 0.131 | 0.992 | 0.338 |
| M8a Logistic regression | 80% | 0.798 | 0.780 | 0.157 | 0.987 | 0.249 |
| M8a Logistic regression | 90% | 0.899 | 0.613 | 0.107 | 0.992 | 0.412 |
| M8b Random forest | 80% | 0.803 | 0.803 | 0.174 | 0.988 | 0.226 |
| M8b Random forest | 90% | 0.902 | 0.667 | 0.123 | 0.993 | 0.361 |
| LGBM LightGBM | 80% | 0.798 | 0.813 | 0.181 | 0.987 | 0.217 |
| LGBM LightGBM | 90% | 0.900 | 0.695 | 0.132 | 0.993 | 0.334 |
| RS-ARD Random search, ARD weights | 80% | 0.794 | 0.816 | 0.181 | 0.987 | 0.214 |
| RS-ARD Random search, ARD weights | 90% | 0.903 | 0.682 | 0.128 | 0.993 | 0.347 |
| RS-mask Random search, masks | 80% | 0.797 | 0.808 | 0.177 | 0.987 | 0.221 |
| RS-mask Random search, masks | 90% | 0.899 | 0.684 | 0.128 | 0.993 | 0.345 |

## Decision curves

See cv_sec_aki_dca.png (pooled recalibrated predictions, repeat 1 solid, repeat 2 dashed).

## Subgroups (discrimination; mean of folds, pooled AUPRC per repeat in the csv)

| Subgroup | n | events | Model | AUROC [CI] | AUPRC [CI] | O/E recal. |
|---|---|---|---|---|---|---|
| Cardiac surgery with CPB | 2212 | 500 | M1 DE-KELM-ARD | 0.787 [0.753, 0.820] | 0.526 [0.471, 0.580] | 1.00 |
| Cardiac surgery with CPB | 2212 | 500 | M2 DE-KELM | 0.768 [0.731, 0.805] | 0.488 [0.444, 0.532] | 1.02 |
| Cardiac surgery with CPB | 2212 | 500 | M3 CM-KELM | 0.762 [0.719, 0.805] | 0.469 [0.421, 0.517] | 1.04 |
| Cardiac surgery with CPB | 2212 | 500 | M4 Grid-tuned KELM | 0.726 [0.693, 0.759] | 0.401 [0.359, 0.443] | 1.15 |
| Cardiac surgery with CPB | 2212 | 500 | M5 Sparse GP (ARD) | 0.784 [0.754, 0.815] | 0.505 [0.449, 0.562] | 1.00 |
| Cardiac surgery with CPB | 2212 | 500 | M6 XGBoost | 0.778 [0.740, 0.817] | 0.521 [0.449, 0.593] | 1.03 |
| Cardiac surgery with CPB | 2212 | 500 | M7 FT-Transformer | 0.775 [0.735, 0.814] | 0.514 [0.435, 0.594] | 1.03 |
| Cardiac surgery with CPB | 2212 | 500 | M8a Logistic regression | 0.731 [0.699, 0.764] | 0.417 [0.369, 0.465] | 1.08 |
| Cardiac surgery with CPB | 2212 | 500 | M8b Random forest | 0.753 [0.714, 0.791] | 0.462 [0.394, 0.530] | 0.91 |
| Cardiac surgery with CPB | 2212 | 500 | LGBM LightGBM | 0.782 [0.744, 0.820] | 0.517 [0.444, 0.589] | 1.02 |
| Cardiac surgery with CPB | 2212 | 500 | RS-ARD Random search, ARD weights | 0.732 [0.662, 0.802] | 0.437 [0.365, 0.508] | 1.04 |
| Cardiac surgery with CPB | 2212 | 500 | RS-mask Random search, masks | 0.763 [0.699, 0.827] | 0.463 [0.416, 0.509] | 1.03 |
| Urology | 5083 | 777 | M1 DE-KELM-ARD | 0.767 [0.727, 0.807] | 0.398 [0.328, 0.468] | 0.85 |
| Urology | 5083 | 777 | M2 DE-KELM | 0.765 [0.729, 0.801] | 0.403 [0.341, 0.464] | 0.85 |
| Urology | 5083 | 777 | M3 CM-KELM | 0.758 [0.724, 0.792] | 0.387 [0.329, 0.445] | 0.85 |
| Urology | 5083 | 777 | M4 Grid-tuned KELM | 0.729 [0.694, 0.765] | 0.330 [0.272, 0.387] | 0.95 |
| Urology | 5083 | 777 | M5 Sparse GP (ARD) | 0.759 [0.724, 0.793] | 0.399 [0.338, 0.460] | 0.98 |
| Urology | 5083 | 777 | M6 XGBoost | 0.792 [0.755, 0.829] | 0.463 [0.413, 0.512] | 1.02 |
| Urology | 5083 | 777 | M7 FT-Transformer | 0.774 [0.740, 0.808] | 0.432 [0.365, 0.500] | 0.99 |
| Urology | 5083 | 777 | M8a Logistic regression | 0.618 [0.573, 0.663] | 0.224 [0.188, 0.261] | 1.09 |
| Urology | 5083 | 777 | M8b Random forest | 0.767 [0.726, 0.807] | 0.431 [0.355, 0.506] | 1.05 |
| Urology | 5083 | 777 | LGBM LightGBM | 0.791 [0.749, 0.832] | 0.466 [0.400, 0.531] | 1.01 |
| Urology | 5083 | 777 | RS-ARD Random search, ARD weights | 0.759 [0.702, 0.815] | 0.378 [0.262, 0.494] | 0.88 |
| Urology | 5083 | 777 | RS-mask Random search, masks | 0.756 [0.715, 0.796] | 0.382 [0.291, 0.472] | 0.84 |
| Other surgery | 43189 | 1188 | M1 DE-KELM-ARD | 0.875 [0.866, 0.883] | 0.267 [0.235, 0.300] | 1.13 |
| Other surgery | 43189 | 1188 | M2 DE-KELM | 0.869 [0.858, 0.880] | 0.252 [0.226, 0.279] | 1.13 |
| Other surgery | 43189 | 1188 | M3 CM-KELM | 0.868 [0.859, 0.878] | 0.249 [0.219, 0.279] | 1.11 |
| Other surgery | 43189 | 1188 | M4 Grid-tuned KELM | 0.846 [0.834, 0.859] | 0.201 [0.173, 0.229] | 0.98 |
| Other surgery | 43189 | 1188 | M5 Sparse GP (ARD) | 0.870 [0.859, 0.882] | 0.272 [0.241, 0.303] | 1.01 |
| Other surgery | 43189 | 1188 | M6 XGBoost | 0.870 [0.854, 0.885] | 0.275 [0.240, 0.309] | 0.98 |
| Other surgery | 43189 | 1188 | M7 FT-Transformer | 0.871 [0.855, 0.886] | 0.265 [0.231, 0.299] | 1.00 |
| Other surgery | 43189 | 1188 | M8a Logistic regression | 0.829 [0.817, 0.841] | 0.198 [0.178, 0.218] | 0.92 |
| Other surgery | 43189 | 1188 | M8b Random forest | 0.862 [0.847, 0.876] | 0.242 [0.215, 0.270] | 1.01 |
| Other surgery | 43189 | 1188 | LGBM LightGBM | 0.867 [0.852, 0.882] | 0.272 [0.230, 0.313] | 0.99 |
| Other surgery | 43189 | 1188 | RS-ARD Random search, ARD weights | 0.869 [0.858, 0.881] | 0.231 [0.180, 0.283] | 1.08 |
| Other surgery | 43189 | 1188 | RS-mask Random search, masks | 0.868 [0.857, 0.879] | 0.244 [0.192, 0.295] | 1.13 |
| Age >= 65 | 21896 | 1185 | M1 DE-KELM-ARD | 0.864 [0.849, 0.878] | 0.340 [0.322, 0.359] | 0.97 |
| Age >= 65 | 21896 | 1185 | M2 DE-KELM | 0.859 [0.841, 0.877] | 0.320 [0.297, 0.342] | 0.97 |
| Age >= 65 | 21896 | 1185 | M3 CM-KELM | 0.858 [0.841, 0.876] | 0.310 [0.283, 0.338] | 0.97 |
| Age >= 65 | 21896 | 1185 | M4 Grid-tuned KELM | 0.843 [0.827, 0.860] | 0.253 [0.228, 0.279] | 0.96 |
| Age >= 65 | 21896 | 1185 | M5 Sparse GP (ARD) | 0.861 [0.845, 0.877] | 0.339 [0.307, 0.372] | 0.98 |
| Age >= 65 | 21896 | 1185 | M6 XGBoost | 0.863 [0.847, 0.879] | 0.349 [0.315, 0.383] | 0.99 |
| Age >= 65 | 21896 | 1185 | M7 FT-Transformer | 0.861 [0.844, 0.878] | 0.338 [0.300, 0.376] | 1.00 |
| Age >= 65 | 21896 | 1185 | M8a Logistic regression | 0.832 [0.820, 0.845] | 0.257 [0.235, 0.278] | 0.96 |
| Age >= 65 | 21896 | 1185 | M8b Random forest | 0.856 [0.839, 0.873] | 0.311 [0.272, 0.350] | 1.00 |
| Age >= 65 | 21896 | 1185 | LGBM LightGBM | 0.860 [0.847, 0.873] | 0.341 [0.306, 0.376] | 0.99 |
| Age >= 65 | 21896 | 1185 | RS-ARD Random search, ARD weights | 0.857 [0.829, 0.886] | 0.294 [0.244, 0.343] | 0.96 |
| Age >= 65 | 21896 | 1185 | RS-mask Random search, masks | 0.857 [0.826, 0.888] | 0.307 [0.266, 0.347] | 0.97 |
| Emergency surgery | 5067 | 358 | M1 DE-KELM-ARD | 0.869 [0.837, 0.901] | 0.385 [0.312, 0.457] | 1.07 |
| Emergency surgery | 5067 | 358 | M2 DE-KELM | 0.860 [0.830, 0.890] | 0.369 [0.297, 0.441] | 1.11 |
| Emergency surgery | 5067 | 358 | M3 CM-KELM | 0.858 [0.827, 0.888] | 0.355 [0.285, 0.426] | 1.11 |
| Emergency surgery | 5067 | 358 | M4 Grid-tuned KELM | 0.829 [0.797, 0.860] | 0.285 [0.235, 0.335] | 1.05 |
| Emergency surgery | 5067 | 358 | M5 Sparse GP (ARD) | 0.868 [0.840, 0.897] | 0.400 [0.331, 0.468] | 0.98 |
| Emergency surgery | 5067 | 358 | M6 XGBoost | 0.865 [0.829, 0.900] | 0.401 [0.329, 0.473] | 1.02 |
| Emergency surgery | 5067 | 358 | M7 FT-Transformer | 0.865 [0.833, 0.896] | 0.417 [0.349, 0.484] | 0.98 |
| Emergency surgery | 5067 | 358 | M8a Logistic regression | 0.834 [0.798, 0.870] | 0.315 [0.274, 0.357] | 0.92 |
| Emergency surgery | 5067 | 358 | M8b Random forest | 0.860 [0.833, 0.886] | 0.366 [0.278, 0.453] | 1.06 |
| Emergency surgery | 5067 | 358 | LGBM LightGBM | 0.864 [0.831, 0.898] | 0.402 [0.345, 0.460] | 1.01 |
| Emergency surgery | 5067 | 358 | RS-ARD Random search, ARD weights | 0.851 [0.804, 0.898] | 0.310 [0.208, 0.411] | 1.13 |
| Emergency surgery | 5067 | 358 | RS-mask Random search, masks | 0.855 [0.801, 0.908] | 0.344 [0.221, 0.468] | 1.10 |
| CKD (eGFR < 60) | 3279 | 446 | M1 DE-KELM-ARD | 0.843 [0.820, 0.865] | 0.458 [0.394, 0.521] | 1.04 |
| CKD (eGFR < 60) | 3279 | 446 | M2 DE-KELM | 0.818 [0.790, 0.845] | 0.401 [0.324, 0.477] | 1.13 |
| CKD (eGFR < 60) | 3279 | 446 | M3 CM-KELM | 0.814 [0.783, 0.846] | 0.391 [0.307, 0.476] | 1.14 |
| CKD (eGFR < 60) | 3279 | 446 | M4 Grid-tuned KELM | 0.774 [0.740, 0.807] | 0.297 [0.236, 0.359] | 1.19 |
| CKD (eGFR < 60) | 3279 | 446 | M5 Sparse GP (ARD) | 0.840 [0.815, 0.865] | 0.462 [0.384, 0.540] | 0.98 |
| CKD (eGFR < 60) | 3279 | 446 | M6 XGBoost | 0.830 [0.801, 0.858] | 0.434 [0.366, 0.501] | 1.04 |
| CKD (eGFR < 60) | 3279 | 446 | M7 FT-Transformer | 0.807 [0.775, 0.839] | 0.374 [0.313, 0.435] | 0.98 |
| CKD (eGFR < 60) | 3279 | 446 | M8a Logistic regression | 0.757 [0.735, 0.779] | 0.291 [0.244, 0.337] | 1.03 |
| CKD (eGFR < 60) | 3279 | 446 | M8b Random forest | 0.813 [0.778, 0.847] | 0.397 [0.311, 0.484] | 1.16 |
| CKD (eGFR < 60) | 3279 | 446 | LGBM LightGBM | 0.828 [0.799, 0.858] | 0.427 [0.359, 0.496] | 1.05 |
| CKD (eGFR < 60) | 3279 | 446 | RS-ARD Random search, ARD weights | 0.814 [0.756, 0.872] | 0.393 [0.233, 0.553] | 1.12 |
| CKD (eGFR < 60) | 3279 | 446 | RS-mask Random search, masks | 0.810 [0.763, 0.856] | 0.375 [0.252, 0.499] | 1.14 |

M1 minus comparator within subgroups (corrected t-test, two-sided; Holm within each subgroup and metric; descriptive, protocol Section 8):

| Subgroup | Comparator | metric | difference [CI] | p | Holm p |
|---|---|---|---|---|---|
| Cardiac surgery with CPB | M2 DE-KELM | AUROC | +0.0189 [+0.0061, +0.0318] | 0.00866 | 0.0519 |
| Cardiac surgery with CPB | M2 DE-KELM | AUPRC | +0.0378 [+0.0088, +0.0667] | 0.0162 | 0.114 |
| Cardiac surgery with CPB | M3 CM-KELM | AUROC | +0.0244 [+0.0069, +0.0420] | 0.0117 | 0.0587 |
| Cardiac surgery with CPB | M3 CM-KELM | AUPRC | +0.0569 [+0.0105, +0.1033] | 0.0216 | 0.13 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUROC | +0.0607 [+0.0419, +0.0794] | 4.45e-05 | 0.000401 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUPRC | +0.1250 [+0.0792, +0.1708] | 0.000163 | 0.00147 |
| Cardiac surgery with CPB | M5 Sparse GP (ARD) | AUROC | +0.0023 [-0.0077, +0.0123] | 0.618 | 1 |
| Cardiac surgery with CPB | M5 Sparse GP (ARD) | AUPRC | +0.0207 [-0.0098, +0.0512] | 0.16 | 0.638 |
| Cardiac surgery with CPB | M6 XGBoost | AUROC | +0.0083 [-0.0129, +0.0294] | 0.399 | 1 |
| Cardiac surgery with CPB | M6 XGBoost | AUPRC | +0.0047 [-0.0522, +0.0617] | 0.855 | 1 |
| Cardiac surgery with CPB | M7 FT-Transformer | AUROC | +0.0118 [-0.0109, +0.0345] | 0.271 | 1 |
| Cardiac surgery with CPB | M7 FT-Transformer | AUPRC | +0.0116 [-0.0419, +0.0651] | 0.636 | 1 |
| Cardiac surgery with CPB | M8a Logistic regression | AUROC | +0.0553 [+0.0285, +0.0820] | 0.00117 | 0.00937 |
| Cardiac surgery with CPB | M8a Logistic regression | AUPRC | +0.1087 [+0.0561, +0.1613] | 0.00116 | 0.00929 |
| Cardiac surgery with CPB | M8b Random forest | AUROC | +0.0339 [+0.0155, +0.0523] | 0.00242 | 0.017 |
| Cardiac surgery with CPB | M8b Random forest | AUPRC | +0.0639 [+0.0042, +0.1236] | 0.0384 | 0.192 |
| Cardiac surgery with CPB | LGBM LightGBM | AUROC | +0.0045 [-0.0137, +0.0227] | 0.589 | 1 |
| Cardiac surgery with CPB | LGBM LightGBM | AUPRC | +0.0090 [-0.0489, +0.0670] | 0.732 | 1 |
| Urology | M2 DE-KELM | AUROC | +0.0025 [-0.0136, +0.0185] | 0.733 | 1 |
| Urology | M2 DE-KELM | AUPRC | -0.0046 [-0.0384, +0.0291] | 0.764 | 1 |
| Urology | M3 CM-KELM | AUROC | +0.0092 [-0.0080, +0.0265] | 0.256 | 1 |
| Urology | M3 CM-KELM | AUPRC | +0.0116 [-0.0199, +0.0432] | 0.427 | 1 |
| Urology | M4 Grid-tuned KELM | AUROC | +0.0376 [+0.0180, +0.0573] | 0.00188 | 0.0124 |
| Urology | M4 Grid-tuned KELM | AUPRC | +0.0687 [+0.0402, +0.0972] | 0.000401 | 0.0028 |
| Urology | M5 Sparse GP (ARD) | AUROC | +0.0084 [-0.0002, +0.0171] | 0.0556 | 0.278 |
| Urology | M5 Sparse GP (ARD) | AUPRC | -0.0009 [-0.0263, +0.0245] | 0.937 | 1 |
| Urology | M6 XGBoost | AUROC | -0.0248 [-0.0364, -0.0132] | 0.000939 | 0.00751 |
| Urology | M6 XGBoost | AUPRC | -0.0644 [-0.0996, -0.0291] | 0.00257 | 0.0154 |
| Urology | M7 FT-Transformer | AUROC | -0.0067 [-0.0193, +0.0059] | 0.258 | 1 |
| Urology | M7 FT-Transformer | AUPRC | -0.0341 [-0.0732, +0.0051] | 0.0804 | 0.402 |
| Urology | M8a Logistic regression | AUROC | +0.1489 [+0.1149, +0.1829] | 3.84e-06 | 3.46e-05 |
| Urology | M8a Logistic regression | AUPRC | +0.1740 [+0.1186, +0.2294] | 5.64e-05 | 0.000508 |
| Urology | M8b Random forest | AUROC | +0.0003 [-0.0223, +0.0230] | 0.974 | 1 |
| Urology | M8b Random forest | AUPRC | -0.0325 [-0.0821, +0.0172] | 0.173 | 0.693 |
| Urology | LGBM LightGBM | AUROC | -0.0237 [-0.0360, -0.0115] | 0.00177 | 0.0124 |
| Urology | LGBM LightGBM | AUPRC | -0.0675 [-0.0918, -0.0431] | 0.000146 | 0.00117 |
| Other surgery | M2 DE-KELM | AUROC | +0.0058 [+0.0015, +0.0101] | 0.0143 | 0.0856 |
| Other surgery | M2 DE-KELM | AUPRC | +0.0152 [-0.0043, +0.0346] | 0.111 | 0.667 |
| Other surgery | M3 CM-KELM | AUROC | +0.0065 [+0.0036, +0.0094] | 0.000642 | 0.00514 |
| Other surgery | M3 CM-KELM | AUPRC | +0.0187 [-0.0058, +0.0431] | 0.118 | 0.667 |
| Other surgery | M4 Grid-tuned KELM | AUROC | +0.0285 [+0.0142, +0.0428] | 0.00144 | 0.0101 |
| Other surgery | M4 Grid-tuned KELM | AUPRC | +0.0661 [+0.0437, +0.0885] | 9.04e-05 | 0.000724 |
| Other surgery | M5 Sparse GP (ARD) | AUROC | +0.0043 [-0.0009, +0.0095] | 0.0949 | 0.38 |
| Other surgery | M5 Sparse GP (ARD) | AUPRC | -0.0044 [-0.0164, +0.0077] | 0.433 | 1 |
| Other surgery | M6 XGBoost | AUROC | +0.0050 [-0.0053, +0.0152] | 0.301 | 0.602 |
| Other surgery | M6 XGBoost | AUPRC | -0.0071 [-0.0171, +0.0029] | 0.142 | 0.667 |
| Other surgery | M7 FT-Transformer | AUROC | +0.0042 [-0.0052, +0.0136] | 0.34 | 0.602 |
| Other surgery | M7 FT-Transformer | AUPRC | +0.0026 [-0.0182, +0.0235] | 0.782 | 1 |
| Other surgery | M8a Logistic regression | AUROC | +0.0460 [+0.0336, +0.0584] | 1.51e-05 | 0.000136 |
| Other surgery | M8a Logistic regression | AUPRC | +0.0692 [+0.0473, +0.0910] | 5.26e-05 | 0.000473 |
| Other surgery | M8b Random forest | AUROC | +0.0132 [+0.0018, +0.0245] | 0.0277 | 0.138 |
| Other surgery | M8b Random forest | AUPRC | +0.0252 [+0.0113, +0.0392] | 0.00273 | 0.0191 |
| Other surgery | LGBM LightGBM | AUROC | +0.0073 [-0.0030, +0.0177] | 0.142 | 0.427 |
| Other surgery | LGBM LightGBM | AUPRC | -0.0042 [-0.0234, +0.0151] | 0.635 | 1 |
| Age >= 65 | M2 DE-KELM | AUROC | +0.0046 [+0.0002, +0.0090] | 0.0438 | 0.263 |
| Age >= 65 | M2 DE-KELM | AUPRC | +0.0209 [+0.0018, +0.0400] | 0.0351 | 0.198 |
| Age >= 65 | M3 CM-KELM | AUROC | +0.0055 [+0.0008, +0.0102] | 0.0255 | 0.179 |
| Age >= 65 | M3 CM-KELM | AUPRC | +0.0300 [+0.0068, +0.0532] | 0.0169 | 0.119 |
| Age >= 65 | M4 Grid-tuned KELM | AUROC | +0.0203 [+0.0125, +0.0281] | 0.000233 | 0.00186 |
| Age >= 65 | M4 Grid-tuned KELM | AUPRC | +0.0874 [+0.0627, +0.1120] | 2.18e-05 | 0.000175 |
| Age >= 65 | M5 Sparse GP (ARD) | AUROC | +0.0024 [-0.0002, +0.0050] | 0.0645 | 0.323 |
| Age >= 65 | M5 Sparse GP (ARD) | AUPRC | +0.0013 [-0.0161, +0.0188] | 0.867 | 1 |
| Age >= 65 | M6 XGBoost | AUROC | +0.0008 [-0.0056, +0.0071] | 0.789 | 0.806 |
| Age >= 65 | M6 XGBoost | AUPRC | -0.0084 [-0.0318, +0.0149] | 0.436 | 1 |
| Age >= 65 | M7 FT-Transformer | AUROC | +0.0024 [-0.0022, +0.0070] | 0.269 | 0.806 |
| Age >= 65 | M7 FT-Transformer | AUPRC | +0.0023 [-0.0263, +0.0310] | 0.858 | 1 |
| Age >= 65 | M8a Logistic regression | AUROC | +0.0312 [+0.0227, +0.0396] | 1.58e-05 | 0.000142 |
| Age >= 65 | M8a Logistic regression | AUPRC | +0.0837 [+0.0661, +0.1013] | 1.91e-06 | 1.72e-05 |
| Age >= 65 | M8b Random forest | AUROC | +0.0074 [-0.0021, +0.0169] | 0.113 | 0.454 |
| Age >= 65 | M8b Random forest | AUPRC | +0.0294 [+0.0030, +0.0558] | 0.033 | 0.198 |
| Age >= 65 | LGBM LightGBM | AUROC | +0.0033 [-0.0044, +0.0110] | 0.357 | 0.806 |
| Age >= 65 | LGBM LightGBM | AUPRC | -0.0007 [-0.0201, +0.0187] | 0.935 | 1 |
| Emergency surgery | M2 DE-KELM | AUROC | +0.0094 [+0.0019, +0.0169] | 0.0193 | 0.116 |
| Emergency surgery | M2 DE-KELM | AUPRC | +0.0161 [-0.0116, +0.0438] | 0.222 | 0.856 |
| Emergency surgery | M3 CM-KELM | AUROC | +0.0114 [+0.0031, +0.0197] | 0.0126 | 0.0883 |
| Emergency surgery | M3 CM-KELM | AUPRC | +0.0293 [-0.0027, +0.0612] | 0.068 | 0.476 |
| Emergency surgery | M4 Grid-tuned KELM | AUROC | +0.0404 [+0.0279, +0.0529] | 4.54e-05 | 0.000409 |
| Emergency surgery | M4 Grid-tuned KELM | AUPRC | +0.0998 [+0.0363, +0.1633] | 0.00617 | 0.0494 |
| Emergency surgery | M5 Sparse GP (ARD) | AUROC | +0.0007 [-0.0098, +0.0113] | 0.877 | 1 |
| Emergency surgery | M5 Sparse GP (ARD) | AUPRC | -0.0151 [-0.0351, +0.0050] | 0.123 | 0.739 |
| Emergency surgery | M6 XGBoost | AUROC | +0.0041 [-0.0129, +0.0210] | 0.599 | 1 |
| Emergency surgery | M6 XGBoost | AUPRC | -0.0162 [-0.0510, +0.0185] | 0.318 | 0.856 |
| Emergency surgery | M7 FT-Transformer | AUROC | +0.0044 [-0.0064, +0.0152] | 0.382 | 1 |
| Emergency surgery | M7 FT-Transformer | AUPRC | -0.0318 [-0.0803, +0.0166] | 0.171 | 0.856 |
| Emergency surgery | M8a Logistic regression | AUROC | +0.0351 [+0.0202, +0.0501] | 0.00049 | 0.00392 |
| Emergency surgery | M8a Logistic regression | AUPRC | +0.0695 [+0.0281, +0.1109] | 0.00421 | 0.0379 |
| Emergency surgery | M8b Random forest | AUROC | +0.0094 [-0.0062, +0.0251] | 0.206 | 1 |
| Emergency surgery | M8b Random forest | AUPRC | +0.0189 [-0.0347, +0.0725] | 0.445 | 0.856 |
| Emergency surgery | LGBM LightGBM | AUROC | +0.0045 [-0.0194, +0.0285] | 0.678 | 1 |
| Emergency surgery | LGBM LightGBM | AUPRC | -0.0175 [-0.0456, +0.0105] | 0.191 | 0.856 |
| CKD (eGFR < 60) | M2 DE-KELM | AUROC | +0.0250 [+0.0130, +0.0370] | 0.00109 | 0.00655 |
| CKD (eGFR < 60) | M2 DE-KELM | AUPRC | +0.0568 [+0.0112, +0.1024] | 0.0201 | 0.101 |
| CKD (eGFR < 60) | M3 CM-KELM | AUROC | +0.0285 [+0.0132, +0.0437] | 0.00219 | 0.011 |
| CKD (eGFR < 60) | M3 CM-KELM | AUPRC | +0.0662 [+0.0160, +0.1164] | 0.0154 | 0.0925 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUROC | +0.0691 [+0.0518, +0.0864] | 8.19e-06 | 6.55e-05 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUPRC | +0.1603 [+0.1226, +0.1979] | 4.91e-06 | 3.93e-05 |
| CKD (eGFR < 60) | M5 Sparse GP (ARD) | AUROC | +0.0027 [-0.0033, +0.0087] | 0.335 | 0.337 |
| CKD (eGFR < 60) | M5 Sparse GP (ARD) | AUPRC | -0.0043 [-0.0500, +0.0415] | 0.837 | 0.837 |
| CKD (eGFR < 60) | M6 XGBoost | AUROC | +0.0131 [-0.0058, +0.0320] | 0.152 | 0.337 |
| CKD (eGFR < 60) | M6 XGBoost | AUPRC | +0.0238 [+0.0018, +0.0459] | 0.0372 | 0.112 |
| CKD (eGFR < 60) | M7 FT-Transformer | AUROC | +0.0358 [+0.0194, +0.0523] | 0.000828 | 0.0058 |
| CKD (eGFR < 60) | M7 FT-Transformer | AUPRC | +0.0833 [+0.0572, +0.1094] | 5e-05 | 0.00035 |
| CKD (eGFR < 60) | M8a Logistic regression | AUROC | +0.0855 [+0.0647, +0.1063] | 6.57e-06 | 5.91e-05 |
| CKD (eGFR < 60) | M8a Logistic regression | AUPRC | +0.1667 [+0.1391, +0.1943] | 2.5e-07 | 2.25e-06 |
| CKD (eGFR < 60) | M8b Random forest | AUROC | +0.0301 [+0.0111, +0.0490] | 0.00586 | 0.0234 |
| CKD (eGFR < 60) | M8b Random forest | AUPRC | +0.0603 [-0.0024, +0.1230] | 0.0575 | 0.115 |
| CKD (eGFR < 60) | LGBM LightGBM | AUROC | +0.0145 [-0.0041, +0.0330] | 0.112 | 0.337 |
| CKD (eGFR < 60) | LGBM LightGBM | AUPRC | +0.0303 [+0.0048, +0.0557] | 0.0248 | 0.101 |

## Interpretability: ARD relevance

Stability across the 10 outer folds: Kendall's W (M1) = 0.595 (p = 5.5e-66); Kendall's W (GP) = 0.931 (p = 2.4e-124).

Agreement M1 vs GP per fold: Spearman rho mean 0.506 (range 0.437-0.568); top-10 overlap mean 5.8/10 (range 5-7). Overall rankings: Spearman 0.656, top-10 overlap 7/10.

| rank | feature | M1 relevance (mean of folds) | SD | folds active | rank in top-10 (folds) | GP relevance | GP rank |
|---|---|---|---|---|---|---|---|
| 1 | preop_creatinine | 0.0680 | 0.0025 | 10/10 | 10/10 | 0.0560 | 1 |
| 2 | department_UR | 0.0440 | 0.0098 | 10/10 | 9/10 | 0.0360 | 2 |
| 3 | art_line | 0.0292 | 0.0046 | 10/10 | 9/10 | 0.0223 | 5 |
| 4 | andur | 0.0287 | 0.0076 | 10/10 | 9/10 | 0.0244 | 4 |
| 5 | preop_bun | 0.0252 | 0.0049 | 10/10 | 8/10 | 0.0152 | 10 |
| 6 | department_GS | 0.0238 | 0.0087 | 10/10 | 7/10 | 0.0260 | 3 |
| 7 | asa | 0.0216 | 0.0046 | 10/10 | 6/10 | 0.0152 | 11 |
| 8 | uo_ml_kg_h | 0.0216 | 0.0076 | 10/10 | 7/10 | 0.0157 | 9 |
| 9 | antype_Neuraxial | 0.0205 | 0.0064 | 10/10 | 4/10 | 0.0115 | 22 |
| 10 | department_OG | 0.0192 | 0.0062 | 10/10 | 5/10 | 0.0085 | 47 |
| 11 | department_IM | 0.0189 | 0.0058 | 10/10 | 4/10 | 0.0130 | 13 |
| 12 | cvp_line | 0.0184 | 0.0059 | 10/10 | 3/10 | 0.0127 | 15 |
| 13 | uo_sum | 0.0164 | 0.0089 | 10/10 | 3/10 | 0.0101 | 32 |
| 14 | department_OS | 0.0162 | 0.0057 | 10/10 | 1/10 | 0.0114 | 23 |
| 15 | pre_antiplatelet | 0.0150 | 0.0053 | 10/10 | 1/10 | 0.0087 | 44 |
| 16 | hr_max | 0.0149 | 0.0030 | 10/10 | 0/10 | 0.0095 | 38 |
| 17 | epi_sum | 0.0144 | 0.0067 | 10/10 | 2/10 | 0.0112 | 24 |
| 18 | antype_Regional | 0.0143 | 0.0060 | 10/10 | 1/10 | 0.0076 | 70 |
| 19 | emop | 0.0143 | 0.0061 | 10/10 | 1/10 | 0.0084 | 49 |
| 20 | department_NS | 0.0139 | 0.0062 | 10/10 | 1/10 | 0.0171 | 7 |

