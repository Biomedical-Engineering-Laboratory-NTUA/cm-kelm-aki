# Secondary analyses: outcome st3, baseline window 365 days

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
| M1 DE-KELM-ARD | 0.1566 [0.1534, 0.1597] | 0.0060 [0.0054, 0.0066] | -4.592 | 4.187 | 0.382 | -0.000 | 0.993 | 0.002 |
| M2 DE-KELM | 0.1543 [0.1511, 0.1574] | 0.0061 [0.0054, 0.0067] | -4.582 | 4.080 | 0.377 | 0.000 | 0.992 | 0.002 |
| M4 Grid-tuned KELM | 0.1495 [0.1461, 0.1528] | 0.0063 [0.0056, 0.0070] | -4.557 | 3.775 | 0.367 | 0.000 | 0.992 | 0.002 |
| M5 Sparse GP (ARD) | 0.0059 [0.0054, 0.0064] | 0.0059 [0.0053, 0.0064] | -0.255 | 1.036 | 0.003 | -0.000 | 0.992 | 0.001 |
| M6 XGBoost | 0.0057 [0.0051, 0.0063] | 0.0057 [0.0051, 0.0063] | 0.872 | 0.810 | 0.003 | -0.003 | 0.991 | 0.002 |
| M8a Logistic regression | 0.1043 [0.0986, 0.1099] | 0.0067 [0.0060, 0.0074] | -5.926 | 0.532 | 0.202 | -0.001 | 0.989 | 0.003 |
| M8b Random forest | 0.0060 [0.0055, 0.0065] | 0.0059 [0.0053, 0.0065] | -0.748 | 1.109 | 0.016 | 0.003 | 1.008 | 0.003 |
| LGBM LightGBM | 0.0061 [0.0054, 0.0067] | 0.0058 [0.0053, 0.0064] | 2.863 | 0.576 | 0.005 | -0.003 | 0.988 | 0.002 |

Raw outputs of the kernel models are not probabilities (KELM regression scores, clipped to [0, 1] for the raw Brier); compare models after recalibration.

## Risk thresholds (recalibrated; mean of folds)

| Model | threshold | flagged | sensitivity | specificity | PPV | NPV | net benefit [corrected 95% CI] |
|---|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 5% | 0.027 | 0.458 | 0.976 | 0.115 | 0.996 | 0.0019 [0.0013, 0.0025] |
| M1 DE-KELM-ARD | 10% | 0.011 | 0.343 | 0.991 | 0.212 | 0.996 | 0.0014 [0.0008, 0.0019] |
| M1 DE-KELM-ARD | 20% | 0.004 | 0.221 | 0.997 | 0.335 | 0.995 | 0.0008 [0.0002, 0.0014] |
| M2 DE-KELM | 5% | 0.027 | 0.430 | 0.976 | 0.108 | 0.996 | 0.0017 [0.0011, 0.0022] |
| M2 DE-KELM | 10% | 0.011 | 0.325 | 0.992 | 0.208 | 0.995 | 0.0013 [0.0008, 0.0018] |
| M2 DE-KELM | 20% | 0.004 | 0.202 | 0.998 | 0.388 | 0.995 | 0.0008 [0.0004, 0.0012] |
| M4 Grid-tuned KELM | 5% | 0.020 | 0.313 | 0.982 | 0.108 | 0.995 | 0.0012 [0.0007, 0.0017] |
| M4 Grid-tuned KELM | 10% | 0.007 | 0.201 | 0.994 | 0.187 | 0.995 | 0.0007 [0.0003, 0.0011] |
| M4 Grid-tuned KELM | 20% | 0.003 | 0.116 | 0.998 | 0.297 | 0.994 | 0.0003 [-0.0001, 0.0007] |
| M5 Sparse GP (ARD) | 5% | 0.028 | 0.476 | 0.976 | 0.117 | 0.996 | 0.0019 [0.0013, 0.0026] |
| M5 Sparse GP (ARD) | 10% | 0.014 | 0.379 | 0.989 | 0.183 | 0.996 | 0.0013 [0.0009, 0.0018] |
| M5 Sparse GP (ARD) | 20% | 0.004 | 0.235 | 0.997 | 0.387 | 0.995 | 0.0010 [0.0004, 0.0015] |
| M6 XGBoost | 5% | 0.020 | 0.468 | 0.983 | 0.155 | 0.996 | 0.0023 [0.0016, 0.0029] |
| M6 XGBoost | 10% | 0.010 | 0.359 | 0.993 | 0.253 | 0.996 | 0.0016 [0.0012, 0.0021] |
| M6 XGBoost | 20% | 0.005 | 0.277 | 0.997 | 0.410 | 0.995 | 0.0012 [0.0009, 0.0015] |
| M8a Logistic regression | 5% | 0.016 | 0.288 | 0.986 | 0.121 | 0.995 | 0.0012 [0.0007, 0.0017] |
| M8a Logistic regression | 10% | 0.007 | 0.147 | 0.994 | 0.143 | 0.994 | 0.0003 [-0.0000, 0.0007] |
| M8a Logistic regression | 20% | 0.003 | 0.050 | 0.998 | 0.124 | 0.994 | -0.0002 [-0.0005, -0.0000] |
| M8b Random forest | 5% | 0.022 | 0.478 | 0.981 | 0.148 | 0.996 | 0.0023 [0.0014, 0.0032] |
| M8b Random forest | 10% | 0.009 | 0.341 | 0.994 | 0.271 | 0.996 | 0.0016 [0.0010, 0.0022] |
| M8b Random forest | 20% | 0.003 | 0.189 | 0.998 | 0.477 | 0.994 | 0.0009 [0.0004, 0.0013] |
| LGBM LightGBM | 5% | 0.017 | 0.412 | 0.986 | 0.164 | 0.996 | 0.0020 [0.0015, 0.0026] |
| LGBM LightGBM | 10% | 0.008 | 0.320 | 0.994 | 0.276 | 0.995 | 0.0015 [0.0010, 0.0021] |
| LGBM LightGBM | 20% | 0.004 | 0.241 | 0.997 | 0.395 | 0.995 | 0.0010 [0.0006, 0.0015] |

Net benefit, M1 minus comparator (corrected t-test, two-sided):

| Comparator | 5% | 10% | 20% |
|---|---|---|---|
| M2 DE-KELM | +0.0002 (p=0.071) | +0.0001 (p=0.312) | -0.0000 (p=0.873) |
| M4 Grid-tuned KELM | +0.0007 (p=0.0215) | +0.0007 (p=0.00205) | +0.0005 (p=0.164) |
| M5 Sparse GP (ARD) | -0.0001 (p=0.572) | +0.0001 (p=0.47) | -0.0002 (p=0.458) |
| M6 XGBoost | -0.0004 (p=0.0105) | -0.0003 (p=0.182) | -0.0004 (p=0.0568) |
| M8a Logistic regression | +0.0007 (p=0.0392) | +0.0010 (p=0.00582) | +0.0010 (p=0.00591) |
| M8b Random forest | -0.0004 (p=0.0208) | -0.0002 (p=0.162) | -0.0001 (p=0.703) |
| LGBM LightGBM | -0.0002 (p=0.144) | -0.0002 (p=0.291) | -0.0002 (p=0.168) |

## Fixed-sensitivity operating points (threshold from one half of the test fold, applied to the other half; mean of folds)

| Model | target | achieved sensitivity | specificity | PPV | NPV | flagged |
|---|---|---|---|---|---|---|
| M1 DE-KELM-ARD | 80% | 0.783 | 0.857 | 0.039 | 0.998 | 0.148 |
| M1 DE-KELM-ARD | 90% | 0.890 | 0.740 | 0.024 | 0.999 | 0.264 |
| M2 DE-KELM | 80% | 0.802 | 0.849 | 0.038 | 0.998 | 0.155 |
| M2 DE-KELM | 90% | 0.883 | 0.752 | 0.025 | 0.999 | 0.253 |
| M4 Grid-tuned KELM | 80% | 0.756 | 0.852 | 0.036 | 0.998 | 0.152 |
| M4 Grid-tuned KELM | 90% | 0.894 | 0.736 | 0.024 | 0.999 | 0.268 |
| M5 Sparse GP (ARD) | 80% | 0.806 | 0.867 | 0.044 | 0.999 | 0.138 |
| M5 Sparse GP (ARD) | 90% | 0.879 | 0.740 | 0.026 | 0.999 | 0.264 |
| M6 XGBoost | 80% | 0.787 | 0.855 | 0.042 | 0.998 | 0.149 |
| M6 XGBoost | 90% | 0.880 | 0.708 | 0.023 | 0.999 | 0.296 |
| M8a Logistic regression | 80% | 0.779 | 0.822 | 0.036 | 0.998 | 0.182 |
| M8a Logistic regression | 90% | 0.889 | 0.648 | 0.019 | 0.999 | 0.355 |
| M8b Random forest | 80% | 0.807 | 0.841 | 0.038 | 0.998 | 0.164 |
| M8b Random forest | 90% | 0.892 | 0.647 | 0.022 | 0.949 | 0.356 |
| LGBM LightGBM | 80% | 0.792 | 0.825 | 0.034 | 0.998 | 0.180 |
| LGBM LightGBM | 90% | 0.870 | 0.681 | 0.020 | 0.999 | 0.323 |

## Decision curves

See cv_sec_st3_dca.png (pooled recalibrated predictions, repeat 1 solid, repeat 2 dashed).

