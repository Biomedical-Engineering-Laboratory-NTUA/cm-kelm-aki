# Secondary analyses: outcome st2, baseline window 365 days

Outer CV 5x2 (protocol Sections 7-9). Recalibration: Platt scaling cross-fitted inside each test fold (5 internal splits). Aggregate results only.

## Availability (order check passed: predictions align with fold_ids)

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

## Calibration (mean of folds [corrected 95% CI])

| Model | Brier raw | Brier recal. | intercept raw | slope raw | ICI raw | intercept recal. | slope recal. | ICI recal. |
|---|---|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 0.1567 [0.1547, 0.1587] | 0.0132 [0.0129, 0.0136] | -3.803 | 3.332 | 0.371 | -0.001 | 0.995 | 0.003 |
| M2 DE-KELM | 0.1587 [0.1575, 0.1600] | 0.0132 [0.0130, 0.0135] | -3.813 | 3.258 | 0.374 | -0.001 | 0.995 | 0.003 |
| M4 Grid-tuned KELM | 0.1538 [0.1521, 0.1554] | 0.0136 [0.0135, 0.0138] | -3.800 | 2.791 | 0.363 | -0.001 | 0.995 | 0.003 |
| M5 Sparse GP (ARD) | 0.0133 [0.0129, 0.0136] | 0.0133 [0.0129, 0.0136] | -0.145 | 1.047 | 0.004 | -0.000 | 0.994 | 0.003 |
| M6 XGBoost | 0.0130 [0.0127, 0.0134] | 0.0130 [0.0127, 0.0133] | 0.398 | 0.884 | 0.005 | -0.001 | 0.995 | 0.003 |
| M8a Logistic regression | 0.1218 [0.1184, 0.1251] | 0.0143 [0.0142, 0.0144] | -4.658 | 0.620 | 0.239 | -0.002 | 0.994 | 0.005 |
| M8b Random forest | 0.0140 [0.0136, 0.0143] | 0.0134 [0.0131, 0.0137] | -0.767 | 1.123 | 0.024 | -0.001 | 1.001 | 0.003 |
| LGBM LightGBM | 0.0134 [0.0132, 0.0137] | 0.0131 [0.0129, 0.0134] | 1.132 | 0.742 | 0.009 | -0.002 | 0.993 | 0.003 |

Raw outputs of the kernel models are not probabilities (KELM regression scores, clipped to [0, 1] for the raw Brier); compare models after recalibration.

## Risk thresholds (recalibrated; mean of folds)

| Model | threshold | flagged | sensitivity | specificity | PPV | NPV | net benefit [corrected 95% CI] |
|---|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 5% | 0.067 | 0.606 | 0.941 | 0.136 | 0.994 | 0.0061 [0.0052, 0.0070] |
| M1 DE-KELM-ARD | 10% | 0.039 | 0.481 | 0.968 | 0.186 | 0.992 | 0.0037 [0.0029, 0.0045] |
| M1 DE-KELM-ARD | 20% | 0.015 | 0.259 | 0.989 | 0.265 | 0.989 | 0.0012 [0.0005, 0.0019] |
| M2 DE-KELM | 5% | 0.065 | 0.593 | 0.944 | 0.139 | 0.993 | 0.0060 [0.0052, 0.0069] |
| M2 DE-KELM | 10% | 0.037 | 0.460 | 0.970 | 0.190 | 0.992 | 0.0037 [0.0027, 0.0046] |
| M2 DE-KELM | 20% | 0.014 | 0.265 | 0.989 | 0.277 | 0.989 | 0.0014 [0.0007, 0.0020] |
| M4 Grid-tuned KELM | 5% | 0.067 | 0.577 | 0.941 | 0.131 | 0.993 | 0.0057 [0.0049, 0.0065] |
| M4 Grid-tuned KELM | 10% | 0.033 | 0.400 | 0.973 | 0.184 | 0.991 | 0.0031 [0.0022, 0.0039] |
| M4 Grid-tuned KELM | 20% | 0.011 | 0.175 | 0.991 | 0.235 | 0.987 | 0.0005 [-0.0000, 0.0010] |
| M5 Sparse GP (ARD) | 5% | 0.069 | 0.592 | 0.939 | 0.130 | 0.993 | 0.0058 [0.0050, 0.0066] |
| M5 Sparse GP (ARD) | 10% | 0.038 | 0.456 | 0.968 | 0.179 | 0.991 | 0.0034 [0.0024, 0.0043] |
| M5 Sparse GP (ARD) | 20% | 0.014 | 0.243 | 0.990 | 0.265 | 0.988 | 0.0011 [0.0005, 0.0018] |
| M6 XGBoost | 5% | 0.059 | 0.579 | 0.949 | 0.148 | 0.993 | 0.0061 [0.0055, 0.0067] |
| M6 XGBoost | 10% | 0.028 | 0.425 | 0.978 | 0.228 | 0.991 | 0.0040 [0.0033, 0.0047] |
| M6 XGBoost | 20% | 0.012 | 0.261 | 0.992 | 0.332 | 0.989 | 0.0020 [0.0014, 0.0025] |
| M8a Logistic regression | 5% | 0.045 | 0.466 | 0.962 | 0.157 | 0.992 | 0.0050 [0.0044, 0.0057] |
| M8a Logistic regression | 10% | 0.021 | 0.261 | 0.983 | 0.192 | 0.989 | 0.0021 [0.0016, 0.0026] |
| M8a Logistic regression | 20% | 0.009 | 0.125 | 0.993 | 0.219 | 0.987 | 0.0002 [-0.0002, 0.0006] |
| M8b Random forest | 5% | 0.070 | 0.594 | 0.938 | 0.128 | 0.993 | 0.0058 [0.0050, 0.0065] |
| M8b Random forest | 10% | 0.030 | 0.402 | 0.976 | 0.204 | 0.991 | 0.0034 [0.0027, 0.0042] |
| M8b Random forest | 20% | 0.009 | 0.171 | 0.994 | 0.288 | 0.987 | 0.0010 [0.0005, 0.0015] |
| LGBM LightGBM | 5% | 0.054 | 0.558 | 0.954 | 0.155 | 0.993 | 0.0060 [0.0053, 0.0067] |
| LGBM LightGBM | 10% | 0.026 | 0.400 | 0.980 | 0.232 | 0.991 | 0.0038 [0.0033, 0.0044] |
| LGBM LightGBM | 20% | 0.012 | 0.253 | 0.992 | 0.325 | 0.989 | 0.0018 [0.0013, 0.0024] |

Net benefit, M1 minus comparator (corrected t-test, two-sided):

| Comparator | 5% | 10% | 20% |
|---|---|---|---|
| M2 DE-KELM | +0.0001 (p=0.462) | +0.0001 (p=0.588) | -0.0002 (p=0.331) |
| M4 Grid-tuned KELM | +0.0004 (p=0.163) | +0.0007 (p=0.0777) | +0.0007 (p=0.0391) |
| M5 Sparse GP (ARD) | +0.0003 (p=0.241) | +0.0003 (p=0.174) | +0.0001 (p=0.822) |
| M6 XGBoost | +0.0000 (p=1) | -0.0003 (p=0.426) | -0.0007 (p=0.00776) |
| M8a Logistic regression | +0.0011 (p=0.00362) | +0.0016 (p=0.00381) | +0.0010 (p=0.031) |
| M8b Random forest | +0.0003 (p=0.271) | +0.0003 (p=0.405) | +0.0002 (p=0.488) |
| LGBM LightGBM | +0.0001 (p=0.741) | -0.0001 (p=0.639) | -0.0006 (p=0.069) |

## Fixed-sensitivity operating points (threshold from one half of the test fold, applied to the other half; mean of folds)

| Model | target | achieved sensitivity | specificity | PPV | NPV | flagged |
|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 80% | 0.799 | 0.847 | 0.077 | 0.996 | 0.163 |
| M1 DE-KELM-ARD | 90% | 0.898 | 0.715 | 0.048 | 0.998 | 0.294 |
| M2 DE-KELM | 80% | 0.795 | 0.842 | 0.075 | 0.996 | 0.168 |
| M2 DE-KELM | 90% | 0.893 | 0.706 | 0.047 | 0.998 | 0.303 |
| M4 Grid-tuned KELM | 80% | 0.795 | 0.842 | 0.076 | 0.996 | 0.168 |
| M4 Grid-tuned KELM | 90% | 0.890 | 0.718 | 0.047 | 0.998 | 0.291 |
| M5 Sparse GP (ARD) | 80% | 0.797 | 0.840 | 0.073 | 0.996 | 0.170 |
| M5 Sparse GP (ARD) | 90% | 0.897 | 0.709 | 0.047 | 0.998 | 0.300 |
| M6 XGBoost | 80% | 0.791 | 0.830 | 0.070 | 0.996 | 0.179 |
| M6 XGBoost | 90% | 0.899 | 0.690 | 0.044 | 0.998 | 0.319 |
| M8a Logistic regression | 80% | 0.794 | 0.825 | 0.068 | 0.996 | 0.185 |
| M8a Logistic regression | 90% | 0.895 | 0.688 | 0.044 | 0.998 | 0.321 |
| M8b Random forest | 80% | 0.798 | 0.830 | 0.069 | 0.996 | 0.180 |
| M8b Random forest | 90% | 0.907 | 0.655 | 0.041 | 0.998 | 0.353 |
| LGBM LightGBM | 80% | 0.792 | 0.831 | 0.070 | 0.996 | 0.178 |
| LGBM LightGBM | 90% | 0.894 | 0.683 | 0.043 | 0.998 | 0.326 |

## Decision curves

See cv_sec_st2_dca.png (pooled recalibrated predictions, repeat 1 solid, repeat 2 dashed).

## Subgroups (discrimination; mean of folds, pooled AUPRC per repeat in the csv)

| Subgroup | n | events | Model | AUROC [CI] | AUPRC [CI] | O/E recal. |
|---|---|---|---|---|---|---|
| Cardiac surgery with CPB | 2212 | 189 | M1 DE-KELM-ARD | 0.778 [0.737, 0.818] | 0.276 [0.204, 0.349] | 0.88 |
| Cardiac surgery with CPB | 2212 | 189 | M2 DE-KELM | 0.777 [0.743, 0.811] | 0.282 [0.198, 0.367] | 0.89 |
| Cardiac surgery with CPB | 2212 | 189 | M4 Grid-tuned KELM | 0.736 [0.684, 0.788] | 0.223 [0.178, 0.268] | 1.04 |
| Cardiac surgery with CPB | 2212 | 189 | M5 Sparse GP (ARD) | 0.774 [0.735, 0.814] | 0.255 [0.184, 0.327] | 1.02 |
| Cardiac surgery with CPB | 2212 | 189 | M6 XGBoost | 0.787 [0.754, 0.820] | 0.282 [0.233, 0.331] | 1.11 |
| Cardiac surgery with CPB | 2212 | 189 | M8a Logistic regression | 0.747 [0.713, 0.781] | 0.211 [0.175, 0.248] | 1.16 |
| Cardiac surgery with CPB | 2212 | 189 | M8b Random forest | 0.752 [0.720, 0.784] | 0.232 [0.195, 0.269] | 0.94 |
| Cardiac surgery with CPB | 2212 | 189 | LGBM LightGBM | 0.777 [0.744, 0.810] | 0.276 [0.242, 0.310] | 1.11 |
| Urology | 5083 | 135 | M1 DE-KELM-ARD | 0.761 [0.712, 0.810] | 0.193 [0.136, 0.250] | 1.10 |
| Urology | 5083 | 135 | M2 DE-KELM | 0.757 [0.711, 0.802] | 0.187 [0.129, 0.244] | 1.29 |
| Urology | 5083 | 135 | M4 Grid-tuned KELM | 0.761 [0.716, 0.807] | 0.176 [0.137, 0.216] | 0.89 |
| Urology | 5083 | 135 | M5 Sparse GP (ARD) | 0.764 [0.726, 0.803] | 0.201 [0.145, 0.258] | 1.14 |
| Urology | 5083 | 135 | M6 XGBoost | 0.753 [0.694, 0.813] | 0.189 [0.128, 0.250] | 1.01 |
| Urology | 5083 | 135 | M8a Logistic regression | 0.746 [0.693, 0.798] | 0.159 [0.123, 0.196] | 1.10 |
| Urology | 5083 | 135 | M8b Random forest | 0.738 [0.701, 0.775] | 0.183 [0.126, 0.239] | 1.29 |
| Urology | 5083 | 135 | LGBM LightGBM | 0.748 [0.700, 0.796] | 0.196 [0.141, 0.251] | 1.08 |
| Other surgery | 43189 | 438 | M1 DE-KELM-ARD | 0.906 [0.890, 0.922] | 0.192 [0.154, 0.229] | 1.04 |
| Other surgery | 43189 | 438 | M2 DE-KELM | 0.902 [0.884, 0.920] | 0.187 [0.155, 0.220] | 0.99 |
| Other surgery | 43189 | 438 | M4 Grid-tuned KELM | 0.901 [0.886, 0.916] | 0.149 [0.129, 0.168] | 1.02 |
| Other surgery | 43189 | 438 | M5 Sparse GP (ARD) | 0.903 [0.886, 0.921] | 0.210 [0.168, 0.251] | 0.96 |
| Other surgery | 43189 | 438 | M6 XGBoost | 0.896 [0.877, 0.915] | 0.239 [0.205, 0.273] | 0.96 |
| Other surgery | 43189 | 438 | M8a Logistic regression | 0.888 [0.867, 0.908] | 0.120 [0.104, 0.136] | 0.92 |
| Other surgery | 43189 | 438 | M8b Random forest | 0.898 [0.875, 0.920] | 0.186 [0.137, 0.235] | 0.96 |
| Other surgery | 43189 | 438 | LGBM LightGBM | 0.896 [0.877, 0.915] | 0.224 [0.188, 0.261] | 0.94 |
| Age >= 65 | 21896 | 421 | M1 DE-KELM-ARD | 0.879 [0.845, 0.913] | 0.217 [0.164, 0.270] | 0.96 |
| Age >= 65 | 21896 | 421 | M2 DE-KELM | 0.876 [0.843, 0.909] | 0.205 [0.156, 0.254] | 0.95 |
| Age >= 65 | 21896 | 421 | M4 Grid-tuned KELM | 0.872 [0.841, 0.902] | 0.165 [0.131, 0.199] | 0.99 |
| Age >= 65 | 21896 | 421 | M5 Sparse GP (ARD) | 0.876 [0.845, 0.908] | 0.220 [0.161, 0.278] | 1.02 |
| Age >= 65 | 21896 | 421 | M6 XGBoost | 0.871 [0.838, 0.905] | 0.234 [0.157, 0.312] | 0.99 |
| Age >= 65 | 21896 | 421 | M8a Logistic regression | 0.859 [0.827, 0.891] | 0.141 [0.110, 0.173] | 1.01 |
| Age >= 65 | 21896 | 421 | M8b Random forest | 0.865 [0.831, 0.900] | 0.194 [0.137, 0.251] | 0.98 |
| Age >= 65 | 21896 | 421 | LGBM LightGBM | 0.872 [0.840, 0.905] | 0.226 [0.158, 0.295] | 1.00 |
| Emergency surgery | 5067 | 147 | M1 DE-KELM-ARD | 0.889 [0.833, 0.944] | 0.229 [0.150, 0.308] | 1.03 |
| Emergency surgery | 5067 | 147 | M2 DE-KELM | 0.883 [0.828, 0.937] | 0.225 [0.143, 0.307] | 1.02 |
| Emergency surgery | 5067 | 147 | M4 Grid-tuned KELM | 0.864 [0.808, 0.920] | 0.170 [0.108, 0.233] | 1.04 |
| Emergency surgery | 5067 | 147 | M5 Sparse GP (ARD) | 0.888 [0.837, 0.940] | 0.203 [0.133, 0.272] | 0.96 |
| Emergency surgery | 5067 | 147 | M6 XGBoost | 0.879 [0.806, 0.952] | 0.233 [0.169, 0.297] | 1.09 |
| Emergency surgery | 5067 | 147 | M8a Logistic regression | 0.861 [0.805, 0.917] | 0.158 [0.106, 0.209] | 0.91 |
| Emergency surgery | 5067 | 147 | M8b Random forest | 0.885 [0.833, 0.938] | 0.218 [0.131, 0.306] | 1.12 |
| Emergency surgery | 5067 | 147 | LGBM LightGBM | 0.879 [0.818, 0.940] | 0.236 [0.167, 0.304] | 1.10 |
| CKD (eGFR < 60) | 3279 | 223 | M1 DE-KELM-ARD | 0.853 [0.815, 0.890] | 0.377 [0.294, 0.459] | 1.04 |
| CKD (eGFR < 60) | 3279 | 223 | M2 DE-KELM | 0.850 [0.812, 0.887] | 0.353 [0.269, 0.438] | 1.05 |
| CKD (eGFR < 60) | 3279 | 223 | M4 Grid-tuned KELM | 0.817 [0.777, 0.857] | 0.248 [0.186, 0.310] | 1.13 |
| CKD (eGFR < 60) | 3279 | 223 | M5 Sparse GP (ARD) | 0.844 [0.815, 0.873] | 0.393 [0.334, 0.453] | 1.00 |
| CKD (eGFR < 60) | 3279 | 223 | M6 XGBoost | 0.876 [0.849, 0.903] | 0.441 [0.370, 0.513] | 1.13 |
| CKD (eGFR < 60) | 3279 | 223 | M8a Logistic regression | 0.775 [0.737, 0.813] | 0.159 [0.128, 0.191] | 1.03 |
| CKD (eGFR < 60) | 3279 | 223 | M8b Random forest | 0.856 [0.816, 0.897] | 0.389 [0.296, 0.481] | 1.30 |
| CKD (eGFR < 60) | 3279 | 223 | LGBM LightGBM | 0.878 [0.855, 0.902] | 0.420 [0.353, 0.488] | 1.13 |

M1 minus comparator within subgroups (corrected t-test, two-sided; Holm within each subgroup and metric; descriptive, protocol Section 8):

| Subgroup | Comparator | metric | difference [CI] | p | Holm p |
|---|---|---|---|---|---|
| Cardiac surgery with CPB | M2 DE-KELM | AUROC | +0.0008 [-0.0173, +0.0189] | 0.92 | 1 |
| Cardiac surgery with CPB | M2 DE-KELM | AUPRC | -0.0059 [-0.0519, +0.0400] | 0.778 | 1 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUROC | +0.0414 [+0.0139, +0.0689] | 0.00781 | 0.0546 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUPRC | +0.0534 [-0.0193, +0.1261] | 0.131 | 0.655 |
| Cardiac surgery with CPB | M5 Sparse GP (ARD) | AUROC | +0.0031 [-0.0217, +0.0278] | 0.785 | 1 |
| Cardiac surgery with CPB | M5 Sparse GP (ARD) | AUPRC | +0.0212 [-0.0333, +0.0758] | 0.402 | 1 |
| Cardiac surgery with CPB | M6 XGBoost | AUROC | -0.0094 [-0.0528, +0.0341] | 0.638 | 1 |
| Cardiac surgery with CPB | M6 XGBoost | AUPRC | -0.0055 [-0.0691, +0.0580] | 0.848 | 1 |
| Cardiac surgery with CPB | M8a Logistic regression | AUROC | +0.0305 [-0.0071, +0.0682] | 0.1 | 0.6 |
| Cardiac surgery with CPB | M8a Logistic regression | AUPRC | +0.0652 [-0.0170, +0.1473] | 0.106 | 0.638 |
| Cardiac surgery with CPB | M8b Random forest | AUROC | +0.0255 [-0.0172, +0.0682] | 0.21 | 1 |
| Cardiac surgery with CPB | M8b Random forest | AUPRC | +0.0443 [-0.0056, +0.0943] | 0.0755 | 0.529 |
| Cardiac surgery with CPB | LGBM LightGBM | AUROC | +0.0004 [-0.0306, +0.0314] | 0.976 | 1 |
| Cardiac surgery with CPB | LGBM LightGBM | AUPRC | +0.0008 [-0.0713, +0.0728] | 0.982 | 1 |
| Urology | M2 DE-KELM | AUROC | +0.0043 [-0.0117, +0.0203] | 0.56 | 1 |
| Urology | M2 DE-KELM | AUPRC | +0.0065 [-0.0174, +0.0304] | 0.553 | 1 |
| Urology | M4 Grid-tuned KELM | AUROC | -0.0006 [-0.0171, +0.0159] | 0.94 | 1 |
| Urology | M4 Grid-tuned KELM | AUPRC | +0.0168 [-0.0256, +0.0591] | 0.394 | 1 |
| Urology | M5 Sparse GP (ARD) | AUROC | -0.0035 [-0.0230, +0.0161] | 0.698 | 1 |
| Urology | M5 Sparse GP (ARD) | AUPRC | -0.0081 [-0.0440, +0.0278] | 0.622 | 1 |
| Urology | M6 XGBoost | AUROC | +0.0075 [-0.0441, +0.0590] | 0.751 | 1 |
| Urology | M6 XGBoost | AUPRC | +0.0040 [-0.0667, +0.0746] | 0.902 | 1 |
| Urology | M8a Logistic regression | AUROC | +0.0152 [-0.0133, +0.0438] | 0.258 | 1 |
| Urology | M8a Logistic regression | AUPRC | +0.0338 [+0.0072, +0.0605] | 0.0184 | 0.129 |
| Urology | M8b Random forest | AUROC | +0.0230 [-0.0192, +0.0652] | 0.249 | 1 |
| Urology | M8b Random forest | AUPRC | +0.0104 [-0.0602, +0.0810] | 0.747 | 1 |
| Urology | LGBM LightGBM | AUROC | +0.0125 [-0.0283, +0.0533] | 0.505 | 1 |
| Urology | LGBM LightGBM | AUPRC | -0.0030 [-0.0608, +0.0548] | 0.909 | 1 |
| Other surgery | M2 DE-KELM | AUROC | +0.0042 [-0.0001, +0.0085] | 0.0557 | 0.331 |
| Other surgery | M2 DE-KELM | AUPRC | +0.0045 [-0.0175, +0.0264] | 0.657 | 1 |
| Other surgery | M4 Grid-tuned KELM | AUROC | +0.0052 [-0.0005, +0.0110] | 0.0699 | 0.331 |
| Other surgery | M4 Grid-tuned KELM | AUPRC | +0.0430 [+0.0218, +0.0642] | 0.00133 | 0.00796 |
| Other surgery | M5 Sparse GP (ARD) | AUROC | +0.0027 [-0.0045, +0.0100] | 0.415 | 0.619 |
| Other surgery | M5 Sparse GP (ARD) | AUPRC | -0.0181 [-0.0460, +0.0098] | 0.176 | 0.527 |
| Other surgery | M6 XGBoost | AUROC | +0.0100 [-0.0022, +0.0221] | 0.097 | 0.331 |
| Other surgery | M6 XGBoost | AUPRC | -0.0472 [-0.0843, -0.0101] | 0.0183 | 0.0913 |
| Other surgery | M8a Logistic regression | AUROC | +0.0186 [+0.0067, +0.0304] | 0.00623 | 0.0436 |
| Other surgery | M8a Logistic regression | AUPRC | +0.0716 [+0.0444, +0.0987] | 0.000212 | 0.00148 |
| Other surgery | M8b Random forest | AUROC | +0.0084 [-0.0093, +0.0261] | 0.309 | 0.619 |
| Other surgery | M8b Random forest | AUPRC | +0.0058 [-0.0296, +0.0411] | 0.721 | 1 |
| Other surgery | LGBM LightGBM | AUROC | +0.0105 [-0.0003, +0.0212] | 0.0552 | 0.331 |
| Other surgery | LGBM LightGBM | AUPRC | -0.0326 [-0.0717, +0.0064] | 0.0914 | 0.365 |
| Age >= 65 | M2 DE-KELM | AUROC | +0.0030 [-0.0015, +0.0075] | 0.165 | 0.659 |
| Age >= 65 | M2 DE-KELM | AUPRC | +0.0119 [-0.0150, +0.0388] | 0.342 | 1 |
| Age >= 65 | M4 Grid-tuned KELM | AUROC | +0.0072 [-0.0016, +0.0160] | 0.0979 | 0.49 |
| Age >= 65 | M4 Grid-tuned KELM | AUPRC | +0.0520 [+0.0197, +0.0842] | 0.00535 | 0.0321 |
| Age >= 65 | M5 Sparse GP (ARD) | AUROC | +0.0030 [-0.0030, +0.0089] | 0.29 | 0.659 |
| Age >= 65 | M5 Sparse GP (ARD) | AUPRC | -0.0026 [-0.0229, +0.0178] | 0.782 | 1 |
| Age >= 65 | M6 XGBoost | AUROC | +0.0076 [-0.0046, +0.0199] | 0.192 | 0.659 |
| Age >= 65 | M6 XGBoost | AUPRC | -0.0171 [-0.0667, +0.0325] | 0.455 | 1 |
| Age >= 65 | M8a Logistic regression | AUROC | +0.0201 [+0.0054, +0.0349] | 0.013 | 0.091 |
| Age >= 65 | M8a Logistic regression | AUPRC | +0.0759 [+0.0377, +0.1140] | 0.00149 | 0.0105 |
| Age >= 65 | M8b Random forest | AUROC | +0.0135 [-0.0011, +0.0281] | 0.0666 | 0.4 |
| Age >= 65 | M8b Random forest | AUPRC | +0.0232 [-0.0220, +0.0683] | 0.276 | 1 |
| Age >= 65 | LGBM LightGBM | AUROC | +0.0067 [-0.0048, +0.0182] | 0.221 | 0.659 |
| Age >= 65 | LGBM LightGBM | AUPRC | -0.0094 [-0.0383, +0.0195] | 0.482 | 1 |
| Emergency surgery | M2 DE-KELM | AUROC | +0.0061 [-0.0011, +0.0132] | 0.088 | 0.44 |
| Emergency surgery | M2 DE-KELM | AUPRC | +0.0034 [-0.0297, +0.0365] | 0.823 | 1 |
| Emergency surgery | M4 Grid-tuned KELM | AUROC | +0.0247 [+0.0081, +0.0413] | 0.00832 | 0.0499 |
| Emergency surgery | M4 Grid-tuned KELM | AUPRC | +0.0583 [+0.0165, +0.1001] | 0.0116 | 0.0814 |
| Emergency surgery | M5 Sparse GP (ARD) | AUROC | +0.0003 [-0.0173, +0.0179] | 0.968 | 1 |
| Emergency surgery | M5 Sparse GP (ARD) | AUPRC | +0.0262 [-0.0213, +0.0737] | 0.243 | 1 |
| Emergency surgery | M6 XGBoost | AUROC | +0.0098 [-0.0109, +0.0305] | 0.312 | 0.935 |
| Emergency surgery | M6 XGBoost | AUPRC | -0.0043 [-0.0541, +0.0454] | 0.848 | 1 |
| Emergency surgery | M8a Logistic regression | AUROC | +0.0279 [+0.0212, +0.0345] | 5.53e-06 | 3.87e-05 |
| Emergency surgery | M8a Logistic regression | AUPRC | +0.0711 [+0.0169, +0.1253] | 0.0158 | 0.0946 |
| Emergency surgery | M8b Random forest | AUROC | +0.0035 [-0.0166, +0.0236] | 0.703 | 1 |
| Emergency surgery | M8b Random forest | AUPRC | +0.0105 [-0.0483, +0.0692] | 0.696 | 1 |
| Emergency surgery | LGBM LightGBM | AUROC | +0.0099 [-0.0027, +0.0225] | 0.109 | 0.44 |
| Emergency surgery | LGBM LightGBM | AUPRC | -0.0070 [-0.0561, +0.0421] | 0.754 | 1 |
| CKD (eGFR < 60) | M2 DE-KELM | AUROC | +0.0026 [-0.0080, +0.0133] | 0.587 | 1 |
| CKD (eGFR < 60) | M2 DE-KELM | AUPRC | +0.0234 [-0.0189, +0.0657] | 0.242 | 0.727 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUROC | +0.0357 [+0.0167, +0.0548] | 0.00213 | 0.0128 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUPRC | +0.1282 [+0.0651, +0.1913] | 0.00129 | 0.00777 |
| CKD (eGFR < 60) | M5 Sparse GP (ARD) | AUROC | +0.0085 [-0.0122, +0.0291] | 0.379 | 1 |
| CKD (eGFR < 60) | M5 Sparse GP (ARD) | AUPRC | -0.0166 [-0.0565, +0.0233] | 0.372 | 0.744 |
| CKD (eGFR < 60) | M6 XGBoost | AUROC | -0.0236 [-0.0492, +0.0019] | 0.0661 | 0.331 |
| CKD (eGFR < 60) | M6 XGBoost | AUPRC | -0.0647 [-0.1266, -0.0029] | 0.042 | 0.21 |
| CKD (eGFR < 60) | M8a Logistic regression | AUROC | +0.0774 [+0.0418, +0.1130] | 0.000824 | 0.00577 |
| CKD (eGFR < 60) | M8a Logistic regression | AUPRC | +0.2173 [+0.1520, +0.2826] | 3.58e-05 | 0.000251 |
| CKD (eGFR < 60) | M8b Random forest | AUROC | -0.0039 [-0.0270, +0.0191] | 0.709 | 1 |
| CKD (eGFR < 60) | M8b Random forest | AUPRC | -0.0120 [-0.1249, +0.1010] | 0.816 | 0.816 |
| CKD (eGFR < 60) | LGBM LightGBM | AUROC | -0.0258 [-0.0551, +0.0035] | 0.0779 | 0.331 |
| CKD (eGFR < 60) | LGBM LightGBM | AUPRC | -0.0439 [-0.1064, +0.0187] | 0.147 | 0.588 |

