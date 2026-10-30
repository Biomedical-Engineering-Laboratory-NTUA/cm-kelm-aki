# Interpretability (protocol Section 9): primary outcome, 10 outer folds

Models rebuilt from the saved hyperparameters and checked against the saved out-of-fold predictions:

| fold | M1 max abs dp | M1 AUPRC rebuilt / saved | XGB max abs dp | RF max abs dp |
|---|---|---|---|---|
| 11 | 1.9e-11 | 0.3586 / 0.3586 | 0 | 0 |
| 12 | 1e-11 | 0.3442 / 0.3442 | 0 | 0 |
| 13 | 1.3e-11 | 0.3624 / 0.3624 | 0 | 0 |
| 14 | 4.7e-11 | 0.3479 / 0.3479 | 0 | 0 |
| 15 | 9.9e-12 | 0.4001 / 0.4001 | 0 | 0 |
| 21 | 7.6e-12 | 0.3767 / 0.3767 | 0 | 0 |
| 22 | 4.9e-12 | 0.3529 / 0.3529 | 0 | 0 |
| 23 | 2.7e-11 | 0.3608 / 0.3608 | 0 | 0 |
| 24 | 7.1e-12 | 0.3578 / 0.3578 | 0 | 0 |
| 25 | 9e-12 | 0.3486 / 0.3486 | 0 | 0 |

## Stability across outer folds (Kendall's W over the folds)

| Method | folds | Kendall's W | p |
|---|---|---|---|
| ARD relevance (M1, gamma w^2) | 10 | 0.595 | 5.5e-66 |
| GP-ARD relevance (M5) | 10 | 0.931 | 2.4e-124 |
| Permutation, M1 | 10 | 0.556 | 1.8e-59 |
| Permutation, XGBoost | 10 | 0.548 | 3.8e-58 |
| Permutation, RF | 10 | 0.487 | 3.5e-48 |

## Agreement between methods (mean importance over folds; Spearman rho / top-10 overlap)

| | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) | (9) |
|---|---|---|---|---|---|---|---|---|---|
| (1) ARD relevance (M1, gamma w^2) | - | 0.66 / 7 | 0.41 / 7 | 0.24 / 5 | 0.45 / 5 | 0.21 / 4 | 0.09 / 3 | -0.01 / 3 | -0.00 / 4 |
| (2) GP-ARD relevance (M5) | 0.66 / 7 | - | 0.55 / 8 | 0.45 / 6 | 0.53 / 7 | 0.44 / 4 | 0.37 / 4 | 0.22 / 4 | 0.27 / 4 |
| (3) Permutation, M1 | 0.41 / 7 | 0.55 / 8 | - | 0.68 / 6 | 0.41 / 5 | 0.57 / 5 | 0.58 / 3 | 0.52 / 3 | 0.51 / 4 |
| (4) Permutation, XGBoost | 0.24 / 5 | 0.45 / 6 | 0.68 / 6 | - | 0.50 / 6 | 0.58 / 6 | 0.80 / 6 | 0.78 / 5 | 0.53 / 4 |
| (5) Permutation, RF | 0.45 / 5 | 0.53 / 7 | 0.41 / 5 | 0.50 / 6 | - | 0.31 / 3 | 0.46 / 5 | 0.31 / 5 | 0.10 / 3 |
| (6) KernelSHAP, M1 (fold 11) | 0.21 / 4 | 0.44 / 4 | 0.57 / 5 | 0.58 / 6 | 0.31 / 3 | - | 0.67 / 7 | 0.59 / 5 | 0.74 / 7 |
| (7) TreeSHAP, XGBoost (fold 11) | 0.09 / 3 | 0.37 / 4 | 0.58 / 3 | 0.80 / 6 | 0.46 / 5 | 0.67 / 7 | - | 0.92 / 7 | 0.55 / 5 |
| (8) TreeSHAP, RF (fold 11) | -0.01 / 3 | 0.22 / 4 | 0.52 / 3 | 0.78 / 5 | 0.31 / 5 | 0.59 / 5 | 0.92 / 7 | - | 0.54 / 4 |
| (9) KernelSHAP, GP (fold 11) | -0.00 / 4 | 0.27 / 4 | 0.51 / 4 | 0.53 / 4 | 0.10 / 3 | 0.74 / 7 | 0.55 / 5 | 0.54 / 4 | - |

Kendall's W across the 9 methods: 0.519 (p = 4.3e-45).

## Top 25 variables by mean rank across methods

| variable | mean rank | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) | (9) |
|---|---|---|---|---|---|---|---|---|---|---|
| department_UR | 1.2 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| preop_creatinine | 1.8 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| andur | 3.3 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | 4 | 3 |
| uo_ml_kg_h | 12.2 | 8 | 9 | 10 | 11 | 6 | 20 | 11 | 15 | 20 |
| department_NS | 13.6 | 20 | 7 | 15 | 7 | 9 | 21 | 4 | 6 | 33 |
| cvp_line | 14.1 | 12 | 15 | 6 | 6 | 31 | 8 | 12 | 20 | 17 |
| department_GS | 14.4 | 6 | 3 | 5 | 22 | 14 | 4 | 26 | 46 | 4 |
| preop_albumin | 14.6 | 34 | 20 | 24 | 21 | 12 | 5 | 6 | 3 | 6 |
| asa | 14.7 | 7 | 11 | 16 | 20 | 15 | 17 | 13 | 11 | 22 |
| art_line | 15.7 | 3 | 5 | 4 | 5 | 4 | 13 | 44 | 48 | 15 |
| sex_M | 16.1 | 33 | 12 | 11 | 8 | 17 | 6 | 7 | 44 | 7 |
| age | 16.8 | 23 | 6 | 7 | 13 | 13 | 19 | 29 | 29 | 12 |
| preop_platelet | 22.0 | 60 | 26 | 20 | 15 | 27 | 9 | 9 | 13 | 19 |
| department_OS | 22.0 | 14 | 23 | 21 | 14 | 10 | 35 | 8 | 12 | 61 |
| co_monitor | 23.7 | 29 | 8 | 13 | 12 | 8 | 27 | 30 | 54 | 32 |
| colloid_sum | 24.6 | 53 | 35 | 22 | 16 | 21 | 10 | 16 | 40 | 8 |
| uo_sum | 24.6 | 13 | 32 | 12 | 4 | 5 | 39 | 23 | 19 | 74 |
| preop_bun | 24.9 | 5 | 10 | 8 | 10 | 73 | 33 | 40 | 27 | 18 |
| preop_hb | 25.7 | 41 | 25 | 23 | 24 | 11 | 38 | 21 | 10 | 38 |
| map_mean | 33.1 | 37 | 27 | 18 | 18 | 87 | 47 | 19 | 9 | 36 |
| preop_sodium | 34.0 | 42 | 45 | 37 | 38 | 26 | 26 | 15 | 33 | 44 |
| map_auc_lt65 | 34.3 | 71 | 88 | 32 | 9 | 78 | 7 | 5 | 5 | 14 |
| crystalloid_sum | 34.3 | 40 | 33 | 29 | 26 | 16 | 52 | 25 | 17 | 71 |
| department_CTS | 34.3 | 21 | 21 | 9 | 48 | 48 | 12 | 75 | 59 | 16 |
| preop_potassium | 35.0 | 47 | 30 | 28 | 39 | 24 | 46 | 22 | 38 | 41 |

Cells give the rank of the variable under each method (1 = most important).

