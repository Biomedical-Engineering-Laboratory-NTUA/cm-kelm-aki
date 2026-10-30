# Secondary analyses: outcome aki, baseline window 90 days

Outer CV, two repeats of five folds (protocol Sections 7-9). Recalibration: Platt scaling cross-fitted inside each test fold (5 internal splits). Aggregate results only.

## Availability (order check passed: predictions align with fold_ids)

| Model | folds | files per fold |
|---|---|---|
| M1 DE-KELM-ARD | 10/10 | 10-10 |
| M2 DE-KELM | 10/10 | 10-10 |
| M4 Grid-tuned KELM | 10/10 | 1-1 |
| M6 XGBoost | 10/10 | 1-1 |
| M8a Logistic regression | 10/10 | 1-1 |
| M8b Random forest | 10/10 | 1-1 |
| LGBM LightGBM | 10/10 | 1-1 |

## Subgroups (discrimination; mean of folds, pooled AUPRC per repeat in the csv)

| Subgroup | n | events | Model | AUROC [CI] | AUPRC [CI] | O/E recal. |
|---|---|---|---|---|---|---|
| Cardiac surgery with CPB | 2212 | 500 | M1 DE-KELM-ARD | 0.790 [0.767, 0.813] | 0.532 [0.485, 0.579] | 1.00 |
| Cardiac surgery with CPB | 2212 | 500 | M2 DE-KELM | 0.770 [0.741, 0.799] | 0.494 [0.438, 0.551] | 1.02 |
| Cardiac surgery with CPB | 2212 | 500 | M4 Grid-tuned KELM | 0.730 [0.696, 0.763] | 0.406 [0.354, 0.458] | 1.14 |
| Cardiac surgery with CPB | 2212 | 500 | M6 XGBoost | 0.781 [0.751, 0.810] | 0.516 [0.452, 0.581] | 1.02 |
| Cardiac surgery with CPB | 2212 | 500 | M8a Logistic regression | 0.731 [0.696, 0.766] | 0.417 [0.352, 0.481] | 1.08 |
| Cardiac surgery with CPB | 2212 | 500 | M8b Random forest | 0.751 [0.726, 0.776] | 0.456 [0.413, 0.500] | 0.92 |
| Cardiac surgery with CPB | 2212 | 500 | LGBM LightGBM | 0.783 [0.754, 0.812] | 0.513 [0.452, 0.574] | 1.02 |
| Urology | 5083 | 775 | M1 DE-KELM-ARD | 0.771 [0.745, 0.798] | 0.399 [0.374, 0.424] | 0.85 |
| Urology | 5083 | 775 | M2 DE-KELM | 0.768 [0.745, 0.792] | 0.404 [0.374, 0.434] | 0.85 |
| Urology | 5083 | 775 | M4 Grid-tuned KELM | 0.729 [0.703, 0.755] | 0.327 [0.289, 0.366] | 0.95 |
| Urology | 5083 | 775 | M6 XGBoost | 0.791 [0.761, 0.821] | 0.470 [0.428, 0.512] | 1.02 |
| Urology | 5083 | 775 | M8a Logistic regression | 0.619 [0.603, 0.635] | 0.227 [0.202, 0.252] | 1.09 |
| Urology | 5083 | 775 | M8b Random forest | 0.762 [0.730, 0.794] | 0.431 [0.385, 0.478] | 1.05 |
| Urology | 5083 | 775 | LGBM LightGBM | 0.787 [0.753, 0.822] | 0.462 [0.419, 0.505] | 1.02 |
| Other surgery (not pre-specified) | 43189 | 1187 | M1 DE-KELM-ARD | 0.877 [0.861, 0.893] | 0.269 [0.241, 0.297] | 1.13 |
| Other surgery (not pre-specified) | 43189 | 1187 | M2 DE-KELM | 0.870 [0.852, 0.888] | 0.255 [0.236, 0.274] | 1.13 |
| Other surgery (not pre-specified) | 43189 | 1187 | M4 Grid-tuned KELM | 0.849 [0.834, 0.864] | 0.204 [0.185, 0.222] | 0.99 |
| Other surgery (not pre-specified) | 43189 | 1187 | M6 XGBoost | 0.872 [0.858, 0.885] | 0.279 [0.258, 0.301] | 0.98 |
| Other surgery (not pre-specified) | 43189 | 1187 | M8a Logistic regression | 0.828 [0.810, 0.845] | 0.196 [0.175, 0.217] | 0.92 |
| Other surgery (not pre-specified) | 43189 | 1187 | M8b Random forest | 0.861 [0.846, 0.875] | 0.244 [0.208, 0.279] | 1.00 |
| Other surgery (not pre-specified) | 43189 | 1187 | LGBM LightGBM | 0.868 [0.854, 0.883] | 0.267 [0.243, 0.292] | 0.98 |
| Male | 26333 | 1517 | M1 DE-KELM-ARD | 0.879 [0.869, 0.888] | 0.352 [0.324, 0.380] | 0.98 |
| Male | 26333 | 1517 | M2 DE-KELM | 0.875 [0.864, 0.886] | 0.342 [0.309, 0.376] | 1.00 |
| Male | 26333 | 1517 | M4 Grid-tuned KELM | 0.860 [0.848, 0.873] | 0.275 [0.250, 0.300] | 0.98 |
| Male | 26333 | 1517 | M6 XGBoost | 0.881 [0.872, 0.890] | 0.390 [0.366, 0.414] | 1.01 |
| Male | 26333 | 1517 | M8a Logistic regression | 0.847 [0.836, 0.857] | 0.260 [0.231, 0.289] | 1.00 |
| Male | 26333 | 1517 | M8b Random forest | 0.870 [0.861, 0.879] | 0.345 [0.323, 0.367] | 1.01 |
| Male | 26333 | 1517 | LGBM LightGBM | 0.879 [0.871, 0.887] | 0.386 [0.359, 0.414] | 1.01 |
| Female | 24151 | 945 | M1 DE-KELM-ARD | 0.901 [0.884, 0.917] | 0.389 [0.349, 0.430] | 1.03 |
| Female | 24151 | 945 | M2 DE-KELM | 0.896 [0.882, 0.911] | 0.373 [0.333, 0.412] | 0.99 |
| Female | 24151 | 945 | M4 Grid-tuned KELM | 0.879 [0.865, 0.893] | 0.307 [0.266, 0.348] | 1.03 |
| Female | 24151 | 945 | M6 XGBoost | 0.899 [0.883, 0.914] | 0.392 [0.351, 0.433] | 0.98 |
| Female | 24151 | 945 | M8a Logistic regression | 0.858 [0.841, 0.875] | 0.267 [0.219, 0.314] | 1.00 |
| Female | 24151 | 945 | M8b Random forest | 0.890 [0.871, 0.910] | 0.353 [0.302, 0.403] | 0.98 |
| Female | 24151 | 945 | LGBM LightGBM | 0.896 [0.884, 0.909] | 0.379 [0.340, 0.418] | 0.98 |
| Age >= 65 | 21896 | 1184 | M1 DE-KELM-ARD | 0.865 [0.851, 0.879] | 0.344 [0.288, 0.400] | 0.97 |
| Age >= 65 | 21896 | 1184 | M2 DE-KELM | 0.861 [0.845, 0.876] | 0.323 [0.272, 0.374] | 0.97 |
| Age >= 65 | 21896 | 1184 | M4 Grid-tuned KELM | 0.844 [0.827, 0.861] | 0.257 [0.218, 0.297] | 0.96 |
| Age >= 65 | 21896 | 1184 | M6 XGBoost | 0.864 [0.851, 0.877] | 0.349 [0.294, 0.405] | 0.99 |
| Age >= 65 | 21896 | 1184 | M8a Logistic regression | 0.832 [0.816, 0.849] | 0.259 [0.207, 0.311] | 0.96 |
| Age >= 65 | 21896 | 1184 | M8b Random forest | 0.855 [0.842, 0.868] | 0.311 [0.272, 0.350] | 0.99 |
| Age >= 65 | 21896 | 1184 | LGBM LightGBM | 0.863 [0.848, 0.877] | 0.343 [0.291, 0.396] | 0.99 |
| Emergency surgery | 5067 | 357 | M1 DE-KELM-ARD | 0.868 [0.844, 0.892] | 0.385 [0.324, 0.446] | 1.07 |
| Emergency surgery | 5067 | 357 | M2 DE-KELM | 0.856 [0.832, 0.880] | 0.356 [0.309, 0.403] | 1.11 |
| Emergency surgery | 5067 | 357 | M4 Grid-tuned KELM | 0.826 [0.798, 0.854] | 0.288 [0.243, 0.333] | 1.04 |
| Emergency surgery | 5067 | 357 | M6 XGBoost | 0.866 [0.837, 0.895] | 0.404 [0.339, 0.469] | 1.02 |
| Emergency surgery | 5067 | 357 | M8a Logistic regression | 0.831 [0.798, 0.863] | 0.308 [0.249, 0.366] | 0.92 |
| Emergency surgery | 5067 | 357 | M8b Random forest | 0.856 [0.848, 0.865] | 0.358 [0.309, 0.407] | 1.05 |
| Emergency surgery | 5067 | 357 | LGBM LightGBM | 0.861 [0.836, 0.887] | 0.381 [0.335, 0.427] | 1.03 |
| CKD (eGFR < 60) | 3277 | 444 | M1 DE-KELM-ARD | 0.845 [0.818, 0.872] | 0.454 [0.401, 0.507] | 1.04 |
| CKD (eGFR < 60) | 3277 | 444 | M2 DE-KELM | 0.824 [0.800, 0.849] | 0.416 [0.354, 0.478] | 1.12 |
| CKD (eGFR < 60) | 3277 | 444 | M4 Grid-tuned KELM | 0.769 [0.744, 0.793] | 0.288 [0.251, 0.324] | 1.18 |
| CKD (eGFR < 60) | 3277 | 444 | M6 XGBoost | 0.835 [0.816, 0.854] | 0.430 [0.389, 0.471] | 1.05 |
| CKD (eGFR < 60) | 3277 | 444 | M8a Logistic regression | 0.755 [0.721, 0.790] | 0.285 [0.234, 0.337] | 1.02 |
| CKD (eGFR < 60) | 3277 | 444 | M8b Random forest | 0.818 [0.792, 0.844] | 0.392 [0.342, 0.443] | 1.16 |
| CKD (eGFR < 60) | 3277 | 444 | LGBM LightGBM | 0.833 [0.816, 0.850] | 0.419 [0.376, 0.463] | 1.06 |

M1 minus comparator within subgroups (corrected t-test, two-sided; Holm within each subgroup and metric; descriptive, protocol Section 8):

| Subgroup | Comparator | metric | difference [CI] | p | Holm p |
|---|---|---|---|---|---|
| Cardiac surgery with CPB | M2 DE-KELM | AUROC | +0.0195 [+0.0063, +0.0326] | 0.0086 | 0.0258 |
| Cardiac surgery with CPB | M2 DE-KELM | AUPRC | +0.0377 [-0.0018, +0.0772] | 0.0593 | 0.178 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUROC | +0.0601 [+0.0238, +0.0963] | 0.00457 | 0.0183 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUPRC | +0.1262 [+0.0718, +0.1806] | 0.00053 | 0.00318 |
| Cardiac surgery with CPB | M6 XGBoost | AUROC | +0.0091 [-0.0150, +0.0332] | 0.416 | 0.832 |
| Cardiac surgery with CPB | M6 XGBoost | AUPRC | +0.0157 [-0.0306, +0.0620] | 0.462 | 0.648 |
| Cardiac surgery with CPB | M8a Logistic regression | AUROC | +0.0587 [+0.0358, +0.0816] | 0.000261 | 0.00157 |
| Cardiac surgery with CPB | M8a Logistic regression | AUPRC | +0.1156 [+0.0581, +0.1730] | 0.00139 | 0.00695 |
| Cardiac surgery with CPB | M8b Random forest | AUROC | +0.0389 [+0.0172, +0.0605] | 0.00287 | 0.0144 |
| Cardiac surgery with CPB | M8b Random forest | AUPRC | +0.0760 [+0.0352, +0.1167] | 0.00224 | 0.00898 |
| Cardiac surgery with CPB | LGBM LightGBM | AUROC | +0.0067 [-0.0150, +0.0284] | 0.502 | 0.832 |
| Cardiac surgery with CPB | LGBM LightGBM | AUPRC | +0.0188 [-0.0220, +0.0596] | 0.324 | 0.648 |
| Urology | M2 DE-KELM | AUROC | +0.0033 [-0.0063, +0.0129] | 0.456 | 0.456 |
| Urology | M2 DE-KELM | AUPRC | -0.0045 [-0.0229, +0.0139] | 0.594 | 0.594 |
| Urology | M4 Grid-tuned KELM | AUROC | +0.0422 [+0.0285, +0.0559] | 6.63e-05 | 0.000331 |
| Urology | M4 Grid-tuned KELM | AUPRC | +0.0720 [+0.0420, +0.1019] | 0.000415 | 0.00208 |
| Urology | M6 XGBoost | AUROC | -0.0200 [-0.0288, -0.0112] | 0.000607 | 0.00243 |
| Urology | M6 XGBoost | AUPRC | -0.0705 [-0.1040, -0.0370] | 0.00103 | 0.00406 |
| Urology | M8a Logistic regression | AUROC | +0.1523 [+0.1229, +0.1816] | 9.28e-07 | 5.57e-06 |
| Urology | M8a Logistic regression | AUPRC | +0.1723 [+0.1362, +0.2085] | 1.9e-06 | 1.14e-05 |
| Urology | M8b Random forest | AUROC | +0.0094 [-0.0041, +0.0228] | 0.149 | 0.297 |
| Urology | M8b Random forest | AUPRC | -0.0320 [-0.0702, +0.0063] | 0.0913 | 0.183 |
| Urology | LGBM LightGBM | AUROC | -0.0160 [-0.0298, -0.0021] | 0.0284 | 0.0851 |
| Urology | LGBM LightGBM | AUPRC | -0.0626 [-0.0922, -0.0329] | 0.00102 | 0.00406 |
| Other surgery (not pre-specified) | M2 DE-KELM | AUROC | +0.0064 [-0.0002, +0.0130] | 0.0548 | 0.125 |
| Other surgery (not pre-specified) | M2 DE-KELM | AUPRC | +0.0143 [-0.0033, +0.0319] | 0.0998 | 0.3 |
| Other surgery (not pre-specified) | M4 Grid-tuned KELM | AUROC | +0.0277 [+0.0190, +0.0364] | 5.04e-05 | 0.000252 |
| Other surgery (not pre-specified) | M4 Grid-tuned KELM | AUPRC | +0.0655 [+0.0484, +0.0825] | 1.15e-05 | 5.77e-05 |
| Other surgery (not pre-specified) | M6 XGBoost | AUROC | +0.0050 [-0.0010, +0.0109] | 0.0915 | 0.125 |
| Other surgery (not pre-specified) | M6 XGBoost | AUPRC | -0.0102 [-0.0316, +0.0111] | 0.307 | 0.614 |
| Other surgery (not pre-specified) | M8a Logistic regression | AUROC | +0.0492 [+0.0384, +0.0600] | 2.79e-06 | 1.67e-05 |
| Other surgery (not pre-specified) | M8a Logistic regression | AUPRC | +0.0731 [+0.0610, +0.0851] | 2.48e-07 | 1.49e-06 |
| Other surgery (not pre-specified) | M8b Random forest | AUROC | +0.0162 [+0.0081, +0.0244] | 0.0015 | 0.00599 |
| Other surgery (not pre-specified) | M8b Random forest | AUPRC | +0.0253 [+0.0066, +0.0441] | 0.0137 | 0.0548 |
| Other surgery (not pre-specified) | LGBM LightGBM | AUROC | +0.0084 [+0.0004, +0.0164] | 0.0418 | 0.125 |
| Other surgery (not pre-specified) | LGBM LightGBM | AUPRC | +0.0017 [-0.0180, +0.0214] | 0.85 | 0.85 |
| Male | M2 DE-KELM | AUROC | +0.0038 [+0.0014, +0.0061] | 0.00551 | 0.0165 |
| Male | M2 DE-KELM | AUPRC | +0.0102 [-0.0039, +0.0244] | 0.135 | 0.27 |
| Male | M4 Grid-tuned KELM | AUROC | +0.0184 [+0.0139, +0.0229] | 7.17e-06 | 3.58e-05 |
| Male | M4 Grid-tuned KELM | AUPRC | +0.0773 [+0.0526, +0.1020] | 5.74e-05 | 0.000287 |
| Male | M6 XGBoost | AUROC | -0.0024 [-0.0066, +0.0019] | 0.239 | 0.478 |
| Male | M6 XGBoost | AUPRC | -0.0377 [-0.0525, -0.0228] | 0.000276 | 0.000841 |
| Male | M8a Logistic regression | AUROC | +0.0323 [+0.0268, +0.0378] | 3.38e-07 | 2.03e-06 |
| Male | M8a Logistic regression | AUPRC | +0.0923 [+0.0786, +0.1060] | 9.87e-08 | 5.92e-07 |
| Male | M8b Random forest | AUROC | +0.0091 [+0.0045, +0.0138] | 0.00165 | 0.0066 |
| Male | M8b Random forest | AUPRC | +0.0074 [-0.0034, +0.0183] | 0.156 | 0.27 |
| Male | LGBM LightGBM | AUROC | -0.0004 [-0.0051, +0.0043] | 0.852 | 0.852 |
| Male | LGBM LightGBM | AUPRC | -0.0339 [-0.0468, -0.0211] | 0.00021 | 0.000841 |
| Female | M2 DE-KELM | AUROC | +0.0048 [-0.0010, +0.0106] | 0.0962 | 0.289 |
| Female | M2 DE-KELM | AUPRC | +0.0166 [-0.0063, +0.0395] | 0.135 | 0.406 |
| Female | M4 Grid-tuned KELM | AUROC | +0.0218 [+0.0137, +0.0299] | 0.000181 | 0.000905 |
| Female | M4 Grid-tuned KELM | AUPRC | +0.0820 [+0.0619, +0.1021] | 6.87e-06 | 4.12e-05 |
| Female | M6 XGBoost | AUROC | +0.0021 [-0.0029, +0.0072] | 0.367 | 0.367 |
| Female | M6 XGBoost | AUPRC | -0.0028 [-0.0266, +0.0211] | 0.799 | 0.819 |
| Female | M8a Logistic regression | AUROC | +0.0429 [+0.0347, +0.0512] | 8.9e-07 | 5.34e-06 |
| Female | M8a Logistic regression | AUPRC | +0.1226 [+0.0917, +0.1536] | 8.89e-06 | 4.45e-05 |
| Female | M8b Random forest | AUROC | +0.0104 [+0.0018, +0.0190] | 0.0228 | 0.0914 |
| Female | M8b Random forest | AUPRC | +0.0366 [-0.0028, +0.0760] | 0.065 | 0.26 |
| Female | LGBM LightGBM | AUROC | +0.0043 [-0.0016, +0.0102] | 0.132 | 0.289 |
| Female | LGBM LightGBM | AUPRC | +0.0105 [-0.0169, +0.0379] | 0.409 | 0.819 |
| Age >= 65 | M2 DE-KELM | AUROC | +0.0048 [+0.0014, +0.0081] | 0.0103 | 0.0309 |
| Age >= 65 | M2 DE-KELM | AUPRC | +0.0208 [-0.0023, +0.0439] | 0.0721 | 0.216 |
| Age >= 65 | M4 Grid-tuned KELM | AUROC | +0.0211 [+0.0143, +0.0278] | 5.78e-05 | 0.000289 |
| Age >= 65 | M4 Grid-tuned KELM | AUPRC | +0.0864 [+0.0590, +0.1137] | 5.41e-05 | 0.000271 |
| Age >= 65 | M6 XGBoost | AUROC | +0.0017 [-0.0054, +0.0087] | 0.604 | 0.614 |
| Age >= 65 | M6 XGBoost | AUPRC | -0.0056 [-0.0290, +0.0178] | 0.599 | 1 |
| Age >= 65 | M8a Logistic regression | AUROC | +0.0331 [+0.0253, +0.0410] | 5.2e-06 | 3.12e-05 |
| Age >= 65 | M8a Logistic regression | AUPRC | +0.0845 [+0.0652, +0.1038] | 3.89e-06 | 2.33e-05 |
| Age >= 65 | M8b Random forest | AUROC | +0.0106 [+0.0064, +0.0149] | 0.000324 | 0.0013 |
| Age >= 65 | M8b Random forest | AUPRC | +0.0331 [+0.0017, +0.0644] | 0.0407 | 0.163 |
| Age >= 65 | LGBM LightGBM | AUROC | +0.0027 [-0.0029, +0.0082] | 0.307 | 0.614 |
| Age >= 65 | LGBM LightGBM | AUPRC | +0.0005 [-0.0261, +0.0270] | 0.97 | 1 |
| Emergency surgery | M2 DE-KELM | AUROC | +0.0115 [+0.0038, +0.0192] | 0.00802 | 0.0321 |
| Emergency surgery | M2 DE-KELM | AUPRC | +0.0286 [-0.0042, +0.0614] | 0.0801 | 0.32 |
| Emergency surgery | M4 Grid-tuned KELM | AUROC | +0.0416 [+0.0304, +0.0528] | 1.45e-05 | 8.69e-05 |
| Emergency surgery | M4 Grid-tuned KELM | AUPRC | +0.0968 [+0.0567, +0.1369] | 0.000401 | 0.0024 |
| Emergency surgery | M6 XGBoost | AUROC | +0.0016 [-0.0176, +0.0208] | 0.853 | 0.853 |
| Emergency surgery | M6 XGBoost | AUPRC | -0.0189 [-0.0693, +0.0315] | 0.418 | 0.836 |
| Emergency surgery | M8a Logistic regression | AUROC | +0.0373 [+0.0188, +0.0559] | 0.00139 | 0.00694 |
| Emergency surgery | M8a Logistic regression | AUPRC | +0.0772 [+0.0116, +0.1429] | 0.0259 | 0.13 |
| Emergency surgery | M8b Random forest | AUROC | +0.0117 [-0.0084, +0.0319] | 0.219 | 0.658 |
| Emergency surgery | M8b Random forest | AUPRC | +0.0268 [-0.0188, +0.0725] | 0.216 | 0.649 |
| Emergency surgery | LGBM LightGBM | AUROC | +0.0066 [-0.0093, +0.0224] | 0.373 | 0.747 |
| Emergency surgery | LGBM LightGBM | AUPRC | +0.0039 [-0.0337, +0.0414] | 0.822 | 0.836 |
| CKD (eGFR < 60) | M2 DE-KELM | AUROC | +0.0205 [+0.0082, +0.0328] | 0.00445 | 0.0151 |
| CKD (eGFR < 60) | M2 DE-KELM | AUPRC | +0.0385 [-0.0114, +0.0884] | 0.115 | 0.344 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUROC | +0.0761 [+0.0617, +0.0905] | 8.06e-07 | 4.84e-06 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUPRC | +0.1664 [+0.1270, +0.2059] | 5.27e-06 | 3.16e-05 |
| CKD (eGFR < 60) | M6 XGBoost | AUROC | +0.0100 [-0.0053, +0.0253] | 0.172 | 0.344 |
| CKD (eGFR < 60) | M6 XGBoost | AUPRC | +0.0242 [-0.0141, +0.0625] | 0.186 | 0.372 |
| CKD (eGFR < 60) | M8a Logistic regression | AUROC | +0.0896 [+0.0541, +0.1250] | 0.000288 | 0.00144 |
| CKD (eGFR < 60) | M8a Logistic regression | AUPRC | +0.1689 [+0.1239, +0.2139] | 1.37e-05 | 6.84e-05 |
| CKD (eGFR < 60) | M8b Random forest | AUROC | +0.0271 [+0.0112, +0.0429] | 0.00378 | 0.0151 |
| CKD (eGFR < 60) | M8b Random forest | AUPRC | +0.0621 [+0.0209, +0.1033] | 0.00776 | 0.0311 |
| CKD (eGFR < 60) | LGBM LightGBM | AUROC | +0.0122 [-0.0077, +0.0320] | 0.2 | 0.344 |
| CKD (eGFR < 60) | LGBM LightGBM | AUPRC | +0.0350 [-0.0209, +0.0909] | 0.19 | 0.372 |

