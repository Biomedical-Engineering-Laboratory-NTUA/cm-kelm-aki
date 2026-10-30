# Secondary analyses: outcome aki, baseline window 7 days

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
| Cardiac surgery with CPB | 2212 | 487 | M1 DE-KELM-ARD | 0.785 [0.729, 0.842] | 0.529 [0.433, 0.626] | 0.96 |
| Cardiac surgery with CPB | 2212 | 487 | M2 DE-KELM | 0.762 [0.704, 0.819] | 0.478 [0.390, 0.565] | 0.98 |
| Cardiac surgery with CPB | 2212 | 487 | M4 Grid-tuned KELM | 0.721 [0.661, 0.781] | 0.421 [0.338, 0.505] | 1.09 |
| Cardiac surgery with CPB | 2212 | 487 | M6 XGBoost | 0.773 [0.719, 0.827] | 0.506 [0.436, 0.575] | 1.03 |
| Cardiac surgery with CPB | 2212 | 487 | M8a Logistic regression | 0.735 [0.686, 0.785] | 0.421 [0.378, 0.464] | 1.11 |
| Cardiac surgery with CPB | 2212 | 487 | M8b Random forest | 0.751 [0.705, 0.796] | 0.459 [0.407, 0.511] | 0.93 |
| Cardiac surgery with CPB | 2212 | 487 | LGBM LightGBM | 0.776 [0.720, 0.833] | 0.514 [0.444, 0.583] | 1.03 |
| Urology | 5083 | 457 | M1 DE-KELM-ARD | 0.788 [0.749, 0.826] | 0.540 [0.474, 0.607] | 0.88 |
| Urology | 5083 | 457 | M2 DE-KELM | 0.772 [0.720, 0.824] | 0.527 [0.433, 0.620] | 0.89 |
| Urology | 5083 | 457 | M4 Grid-tuned KELM | 0.727 [0.676, 0.779] | 0.442 [0.355, 0.529] | 1.00 |
| Urology | 5083 | 457 | M6 XGBoost | 0.807 [0.772, 0.841] | 0.606 [0.553, 0.660] | 1.04 |
| Urology | 5083 | 457 | M8a Logistic regression | 0.552 [0.484, 0.620] | 0.276 [0.226, 0.326] | 1.09 |
| Urology | 5083 | 457 | M8b Random forest | 0.783 [0.748, 0.818] | 0.560 [0.510, 0.610] | 1.08 |
| Urology | 5083 | 457 | LGBM LightGBM | 0.805 [0.767, 0.842] | 0.596 [0.534, 0.657] | 1.05 |
| Other surgery (not pre-specified) | 43189 | 1031 | M1 DE-KELM-ARD | 0.876 [0.863, 0.888] | 0.288 [0.257, 0.319] | 1.09 |
| Other surgery (not pre-specified) | 43189 | 1031 | M2 DE-KELM | 0.867 [0.854, 0.880] | 0.261 [0.228, 0.294] | 1.07 |
| Other surgery (not pre-specified) | 43189 | 1031 | M4 Grid-tuned KELM | 0.846 [0.836, 0.856] | 0.215 [0.184, 0.247] | 0.96 |
| Other surgery (not pre-specified) | 43189 | 1031 | M6 XGBoost | 0.870 [0.860, 0.880] | 0.287 [0.254, 0.320] | 0.97 |
| Other surgery (not pre-specified) | 43189 | 1031 | M8a Logistic regression | 0.833 [0.820, 0.847] | 0.209 [0.170, 0.249] | 0.92 |
| Other surgery (not pre-specified) | 43189 | 1031 | M8b Random forest | 0.863 [0.850, 0.876] | 0.250 [0.217, 0.283] | 1.00 |
| Other surgery (not pre-specified) | 43189 | 1031 | LGBM LightGBM | 0.868 [0.854, 0.882] | 0.283 [0.247, 0.320] | 0.97 |
| Male | 26333 | 1178 | M1 DE-KELM-ARD | 0.889 [0.876, 0.902] | 0.393 [0.352, 0.433] | 0.99 |
| Male | 26333 | 1178 | M2 DE-KELM | 0.883 [0.871, 0.894] | 0.361 [0.318, 0.404] | 1.00 |
| Male | 26333 | 1178 | M4 Grid-tuned KELM | 0.869 [0.858, 0.880] | 0.311 [0.267, 0.356] | 0.99 |
| Male | 26333 | 1178 | M6 XGBoost | 0.888 [0.876, 0.900] | 0.425 [0.381, 0.468] | 1.02 |
| Male | 26333 | 1178 | M8a Logistic regression | 0.860 [0.849, 0.872] | 0.283 [0.241, 0.326] | 0.99 |
| Male | 26333 | 1178 | M8b Random forest | 0.878 [0.866, 0.890] | 0.384 [0.350, 0.419] | 1.01 |
| Male | 26333 | 1178 | LGBM LightGBM | 0.887 [0.872, 0.901] | 0.424 [0.373, 0.476] | 1.01 |
| Female | 24151 | 797 | M1 DE-KELM-ARD | 0.904 [0.891, 0.917] | 0.438 [0.395, 0.481] | 1.01 |
| Female | 24151 | 797 | M2 DE-KELM | 0.898 [0.880, 0.915] | 0.406 [0.347, 0.466] | 1.00 |
| Female | 24151 | 797 | M4 Grid-tuned KELM | 0.880 [0.862, 0.898] | 0.335 [0.292, 0.377] | 1.02 |
| Female | 24151 | 797 | M6 XGBoost | 0.902 [0.888, 0.917] | 0.421 [0.387, 0.455] | 0.98 |
| Female | 24151 | 797 | M8a Logistic regression | 0.867 [0.855, 0.880] | 0.299 [0.262, 0.336] | 1.01 |
| Female | 24151 | 797 | M8b Random forest | 0.899 [0.885, 0.914] | 0.382 [0.346, 0.418] | 0.98 |
| Female | 24151 | 797 | LGBM LightGBM | 0.901 [0.885, 0.917] | 0.416 [0.380, 0.453] | 0.98 |
| Age >= 65 | 21896 | 968 | M1 DE-KELM-ARD | 0.874 [0.862, 0.885] | 0.379 [0.342, 0.415] | 0.97 |
| Age >= 65 | 21896 | 968 | M2 DE-KELM | 0.867 [0.857, 0.877] | 0.340 [0.301, 0.379] | 0.96 |
| Age >= 65 | 21896 | 968 | M4 Grid-tuned KELM | 0.854 [0.845, 0.862] | 0.286 [0.246, 0.325] | 0.96 |
| Age >= 65 | 21896 | 968 | M6 XGBoost | 0.871 [0.859, 0.882] | 0.370 [0.331, 0.409] | 1.00 |
| Age >= 65 | 21896 | 968 | M8a Logistic regression | 0.845 [0.829, 0.860] | 0.276 [0.252, 0.301] | 0.96 |
| Age >= 65 | 21896 | 968 | M8b Random forest | 0.865 [0.852, 0.878] | 0.339 [0.309, 0.369] | 1.01 |
| Age >= 65 | 21896 | 968 | LGBM LightGBM | 0.869 [0.857, 0.881] | 0.369 [0.333, 0.406] | 1.00 |
| Emergency surgery | 5067 | 346 | M1 DE-KELM-ARD | 0.870 [0.841, 0.898] | 0.411 [0.335, 0.486] | 1.05 |
| Emergency surgery | 5067 | 346 | M2 DE-KELM | 0.859 [0.825, 0.893] | 0.373 [0.283, 0.463] | 1.07 |
| Emergency surgery | 5067 | 346 | M4 Grid-tuned KELM | 0.827 [0.784, 0.870] | 0.303 [0.222, 0.384] | 1.06 |
| Emergency surgery | 5067 | 346 | M6 XGBoost | 0.865 [0.832, 0.898] | 0.404 [0.312, 0.495] | 1.04 |
| Emergency surgery | 5067 | 346 | M8a Logistic regression | 0.834 [0.806, 0.863] | 0.308 [0.254, 0.362] | 0.94 |
| Emergency surgery | 5067 | 346 | M8b Random forest | 0.860 [0.830, 0.889] | 0.365 [0.283, 0.446] | 1.05 |
| Emergency surgery | 5067 | 346 | LGBM LightGBM | 0.863 [0.832, 0.894] | 0.405 [0.328, 0.483] | 1.04 |
| CKD (eGFR < 60) | 2701 | 400 | M1 DE-KELM-ARD | 0.843 [0.817, 0.868] | 0.480 [0.412, 0.548] | 1.01 |
| CKD (eGFR < 60) | 2701 | 400 | M2 DE-KELM | 0.808 [0.774, 0.842] | 0.393 [0.311, 0.475] | 1.09 |
| CKD (eGFR < 60) | 2701 | 400 | M4 Grid-tuned KELM | 0.767 [0.733, 0.801] | 0.310 [0.250, 0.371] | 1.14 |
| CKD (eGFR < 60) | 2701 | 400 | M6 XGBoost | 0.827 [0.789, 0.865] | 0.439 [0.368, 0.510] | 1.05 |
| CKD (eGFR < 60) | 2701 | 400 | M8a Logistic regression | 0.749 [0.716, 0.782] | 0.289 [0.240, 0.339] | 0.98 |
| CKD (eGFR < 60) | 2701 | 400 | M8b Random forest | 0.812 [0.784, 0.840] | 0.414 [0.334, 0.494] | 1.15 |
| CKD (eGFR < 60) | 2701 | 400 | LGBM LightGBM | 0.826 [0.787, 0.865] | 0.448 [0.380, 0.516] | 1.04 |

M1 minus comparator within subgroups (corrected t-test, two-sided; Holm within each subgroup and metric; descriptive, protocol Section 8):

| Subgroup | Comparator | metric | difference [CI] | p | Holm p |
|---|---|---|---|---|---|
| Cardiac surgery with CPB | M2 DE-KELM | AUROC | +0.0238 [+0.0118, +0.0359] | 0.00157 | 0.00626 |
| Cardiac surgery with CPB | M2 DE-KELM | AUPRC | +0.0514 [-0.0031, +0.1060] | 0.0618 | 0.243 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUROC | +0.0644 [+0.0383, +0.0904] | 0.000336 | 0.00168 |
| Cardiac surgery with CPB | M4 Grid-tuned KELM | AUPRC | +0.1081 [+0.0514, +0.1648] | 0.00196 | 0.0117 |
| Cardiac surgery with CPB | M6 XGBoost | AUROC | +0.0122 [-0.0074, +0.0318] | 0.192 | 0.385 |
| Cardiac surgery with CPB | M6 XGBoost | AUPRC | +0.0238 [-0.0206, +0.0681] | 0.256 | 0.512 |
| Cardiac surgery with CPB | M8a Logistic regression | AUROC | +0.0502 [+0.0319, +0.0685] | 0.000157 | 0.000945 |
| Cardiac surgery with CPB | M8a Logistic regression | AUPRC | +0.1086 [+0.0423, +0.1749] | 0.00487 | 0.0244 |
| Cardiac surgery with CPB | M8b Random forest | AUROC | +0.0346 [+0.0139, +0.0554] | 0.00433 | 0.013 |
| Cardiac surgery with CPB | M8b Random forest | AUPRC | +0.0706 [-0.0040, +0.1452] | 0.0609 | 0.243 |
| Cardiac surgery with CPB | LGBM LightGBM | AUROC | +0.0090 [-0.0215, +0.0395] | 0.522 | 0.522 |
| Cardiac surgery with CPB | LGBM LightGBM | AUPRC | +0.0157 [-0.0348, +0.0661] | 0.501 | 0.512 |
| Urology | M2 DE-KELM | AUROC | +0.0156 [-0.0096, +0.0408] | 0.195 | 0.39 |
| Urology | M2 DE-KELM | AUPRC | +0.0137 [-0.0349, +0.0622] | 0.54 | 0.922 |
| Urology | M4 Grid-tuned KELM | AUROC | +0.0601 [+0.0281, +0.0920] | 0.00213 | 0.0107 |
| Urology | M4 Grid-tuned KELM | AUPRC | +0.0987 [+0.0648, +0.1326] | 0.000101 | 0.000503 |
| Urology | M6 XGBoost | AUROC | -0.0191 [-0.0327, -0.0055] | 0.0111 | 0.0446 |
| Urology | M6 XGBoost | AUPRC | -0.0659 [-0.0999, -0.0320] | 0.00175 | 0.00699 |
| Urology | M8a Logistic regression | AUROC | +0.2355 [+0.1745, +0.2966] | 1.1e-05 | 6.57e-05 |
| Urology | M8a Logistic regression | AUPRC | +0.2644 [+0.2064, +0.3224] | 2.76e-06 | 1.66e-05 |
| Urology | M8b Random forest | AUROC | +0.0047 [-0.0282, +0.0377] | 0.752 | 0.752 |
| Urology | M8b Random forest | AUPRC | -0.0198 [-0.0781, +0.0384] | 0.461 | 0.922 |
| Urology | LGBM LightGBM | AUROC | -0.0170 [-0.0355, +0.0015] | 0.0669 | 0.201 |
| Urology | LGBM LightGBM | AUPRC | -0.0551 [-0.0878, -0.0224] | 0.00414 | 0.0124 |
| Other surgery (not pre-specified) | M2 DE-KELM | AUROC | +0.0085 [+0.0018, +0.0153] | 0.0187 | 0.0374 |
| Other surgery (not pre-specified) | M2 DE-KELM | AUPRC | +0.0272 [+0.0139, +0.0404] | 0.00123 | 0.00492 |
| Other surgery (not pre-specified) | M4 Grid-tuned KELM | AUROC | +0.0294 [+0.0199, +0.0389] | 6.34e-05 | 0.000381 |
| Other surgery (not pre-specified) | M4 Grid-tuned KELM | AUPRC | +0.0730 [+0.0584, +0.0876] | 1.27e-06 | 6.34e-06 |
| Other surgery (not pre-specified) | M6 XGBoost | AUROC | +0.0051 [-0.0024, +0.0127] | 0.157 | 0.157 |
| Other surgery (not pre-specified) | M6 XGBoost | AUPRC | +0.0015 [-0.0201, +0.0231] | 0.88 | 1 |
| Other surgery (not pre-specified) | M8a Logistic regression | AUROC | +0.0421 [+0.0256, +0.0585] | 0.000262 | 0.00131 |
| Other surgery (not pre-specified) | M8a Logistic regression | AUPRC | +0.0793 [+0.0654, +0.0933] | 4.29e-07 | 2.57e-06 |
| Other surgery (not pre-specified) | M8b Random forest | AUROC | +0.0124 [+0.0041, +0.0207] | 0.00796 | 0.0239 |
| Other surgery (not pre-specified) | M8b Random forest | AUPRC | +0.0385 [+0.0188, +0.0581] | 0.00164 | 0.00493 |
| Other surgery (not pre-specified) | LGBM LightGBM | AUROC | +0.0076 [+0.0031, +0.0122] | 0.004 | 0.016 |
| Other surgery (not pre-specified) | LGBM LightGBM | AUPRC | +0.0050 [-0.0167, +0.0267] | 0.616 | 1 |
| Male | M2 DE-KELM | AUROC | +0.0063 [+0.0034, +0.0092] | 0.000861 | 0.0028 |
| Male | M2 DE-KELM | AUPRC | +0.0313 [+0.0080, +0.0545] | 0.0139 | 0.0557 |
| Male | M4 Grid-tuned KELM | AUROC | +0.0198 [+0.0109, +0.0287] | 0.000699 | 0.0028 |
| Male | M4 Grid-tuned KELM | AUPRC | +0.0811 [+0.0510, +0.1111] | 0.000179 | 0.000893 |
| Male | M6 XGBoost | AUROC | +0.0012 [-0.0042, +0.0066] | 0.629 | 0.9 |
| Male | M6 XGBoost | AUPRC | -0.0320 [-0.0617, -0.0023] | 0.0375 | 0.112 |
| Male | M8a Logistic regression | AUROC | +0.0286 [+0.0218, +0.0353] | 5.09e-06 | 3.05e-05 |
| Male | M8a Logistic regression | AUPRC | +0.1093 [+0.0802, +0.1384] | 1.35e-05 | 8.09e-05 |
| Male | M8b Random forest | AUROC | +0.0107 [+0.0061, +0.0152] | 0.000498 | 0.00249 |
| Male | M8b Random forest | AUPRC | +0.0084 [-0.0257, +0.0424] | 0.592 | 0.592 |
| Male | LGBM LightGBM | AUROC | +0.0023 [-0.0042, +0.0087] | 0.45 | 0.9 |
| Male | LGBM LightGBM | AUPRC | -0.0318 [-0.0647, +0.0012] | 0.057 | 0.114 |
| Female | M2 DE-KELM | AUROC | +0.0066 [-0.0005, +0.0138] | 0.0662 | 0.265 |
| Female | M2 DE-KELM | AUPRC | +0.0313 [+0.0035, +0.0590] | 0.0312 | 0.0935 |
| Female | M4 Grid-tuned KELM | AUROC | +0.0241 [+0.0172, +0.0310] | 2.32e-05 | 0.000139 |
| Female | M4 Grid-tuned KELM | AUPRC | +0.1031 [+0.0756, +0.1305] | 1.37e-05 | 8.19e-05 |
| Female | M6 XGBoost | AUROC | +0.0019 [-0.0027, +0.0066] | 0.371 | 0.519 |
| Female | M6 XGBoost | AUPRC | +0.0163 [-0.0117, +0.0443] | 0.22 | 0.265 |
| Female | M8a Logistic regression | AUROC | +0.0367 [+0.0230, +0.0505] | 0.000193 | 0.000966 |
| Female | M8a Logistic regression | AUPRC | +0.1389 [+0.0967, +0.1811] | 3.92e-05 | 0.000196 |
| Female | M8b Random forest | AUROC | +0.0048 [-0.0031, +0.0127] | 0.204 | 0.519 |
| Female | M8b Random forest | AUPRC | +0.0557 [+0.0342, +0.0772] | 0.000243 | 0.000973 |
| Female | LGBM LightGBM | AUROC | +0.0031 [-0.0016, +0.0078] | 0.173 | 0.519 |
| Female | LGBM LightGBM | AUPRC | +0.0213 [-0.0079, +0.0505] | 0.133 | 0.265 |
| Age >= 65 | M2 DE-KELM | AUROC | +0.0063 [+0.0008, +0.0117] | 0.0286 | 0.0857 |
| Age >= 65 | M2 DE-KELM | AUPRC | +0.0391 [+0.0140, +0.0641] | 0.00643 | 0.0257 |
| Age >= 65 | M4 Grid-tuned KELM | AUROC | +0.0199 [+0.0096, +0.0301] | 0.00175 | 0.00877 |
| Age >= 65 | M4 Grid-tuned KELM | AUPRC | +0.0928 [+0.0698, +0.1159] | 7.74e-06 | 3.87e-05 |
| Age >= 65 | M6 XGBoost | AUROC | +0.0031 [-0.0049, +0.0111] | 0.4 | 0.548 |
| Age >= 65 | M6 XGBoost | AUPRC | +0.0082 [-0.0235, +0.0399] | 0.572 | 1 |
| Age >= 65 | M8a Logistic regression | AUROC | +0.0289 [+0.0156, +0.0422] | 0.000818 | 0.00491 |
| Age >= 65 | M8a Logistic regression | AUPRC | +0.1022 [+0.0786, +0.1259] | 4.3e-06 | 2.58e-05 |
| Age >= 65 | M8b Random forest | AUROC | +0.0086 [+0.0027, +0.0146] | 0.00953 | 0.0381 |
| Age >= 65 | M8b Random forest | AUPRC | +0.0397 [+0.0005, +0.0789] | 0.0476 | 0.143 |
| Age >= 65 | LGBM LightGBM | AUROC | +0.0047 [-0.0044, +0.0138] | 0.274 | 0.548 |
| Age >= 65 | LGBM LightGBM | AUPRC | +0.0093 [-0.0242, +0.0428] | 0.545 | 1 |
| Emergency surgery | M2 DE-KELM | AUROC | +0.0104 [+0.0022, +0.0185] | 0.018 | 0.0539 |
| Emergency surgery | M2 DE-KELM | AUPRC | +0.0377 [+0.0039, +0.0716] | 0.0327 | 0.131 |
| Emergency surgery | M4 Grid-tuned KELM | AUROC | +0.0428 [+0.0222, +0.0634] | 0.00112 | 0.00559 |
| Emergency surgery | M4 Grid-tuned KELM | AUPRC | +0.1077 [+0.0596, +0.1559] | 0.000677 | 0.00406 |
| Emergency surgery | M6 XGBoost | AUROC | +0.0044 [-0.0077, +0.0165] | 0.434 | 0.434 |
| Emergency surgery | M6 XGBoost | AUPRC | +0.0073 [-0.0264, +0.0410] | 0.635 | 1 |
| Emergency surgery | M8a Logistic regression | AUROC | +0.0354 [+0.0261, +0.0447] | 1.21e-05 | 7.27e-05 |
| Emergency surgery | M8a Logistic regression | AUPRC | +0.1031 [+0.0463, +0.1599] | 0.00266 | 0.0133 |
| Emergency surgery | M8b Random forest | AUROC | +0.0100 [+0.0033, +0.0167] | 0.00822 | 0.0329 |
| Emergency surgery | M8b Random forest | AUPRC | +0.0464 [-0.0077, +0.1005] | 0.0845 | 0.253 |
| Emergency surgery | LGBM LightGBM | AUROC | +0.0065 [-0.0043, +0.0172] | 0.206 | 0.413 |
| Emergency surgery | LGBM LightGBM | AUPRC | +0.0055 [-0.0450, +0.0560] | 0.81 | 1 |
| CKD (eGFR < 60) | M2 DE-KELM | AUROC | +0.0346 [+0.0189, +0.0502] | 0.00075 | 0.003 |
| CKD (eGFR < 60) | M2 DE-KELM | AUPRC | +0.0868 [+0.0291, +0.1444] | 0.00782 | 0.0313 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUROC | +0.0753 [+0.0571, +0.0935] | 6.27e-06 | 3.76e-05 |
| CKD (eGFR < 60) | M4 Grid-tuned KELM | AUPRC | +0.1696 [+0.1297, +0.2095] | 4.93e-06 | 2.46e-05 |
| CKD (eGFR < 60) | M6 XGBoost | AUROC | +0.0158 [-0.0053, +0.0368] | 0.125 | 0.226 |
| CKD (eGFR < 60) | M6 XGBoost | AUPRC | +0.0414 [-0.0105, +0.0934] | 0.105 | 0.21 |
| CKD (eGFR < 60) | M8a Logistic regression | AUROC | +0.0935 [+0.0698, +0.1171] | 8.96e-06 | 4.48e-05 |
| CKD (eGFR < 60) | M8a Logistic regression | AUPRC | +0.1906 [+0.1524, +0.2288] | 1.3e-06 | 7.81e-06 |
| CKD (eGFR < 60) | M8b Random forest | AUROC | +0.0308 [+0.0104, +0.0513] | 0.00774 | 0.0232 |
| CKD (eGFR < 60) | M8b Random forest | AUPRC | +0.0659 [-0.0039, +0.1356] | 0.0615 | 0.184 |
| CKD (eGFR < 60) | LGBM LightGBM | AUROC | +0.0166 [-0.0048, +0.0381] | 0.113 | 0.226 |
| CKD (eGFR < 60) | LGBM LightGBM | AUPRC | +0.0319 [-0.0184, +0.0823] | 0.185 | 0.21 |

