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
| Permutation, M1 | 10 | 0.606 | 9.4e-68 |
| Permutation, XGBoost | 10 | 0.581 | 1.2e-63 |
| Permutation, RF | 10 | 0.541 | 6.4e-57 |

## Agreement between methods (mean importance over folds; Spearman rho / top-10 overlap)

| | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) | (9) |
|---|---|---|---|---|---|---|---|---|---|
| (1) ARD relevance (M1, gamma w^2) | - | 0.66 / 7 | 0.43 / 7 | 0.30 / 5 | 0.43 / 5 | 0.18 / 4 | 0.09 / 3 | -0.01 / 3 | -0.05 / 4 |
| (2) GP-ARD relevance (M5) | 0.66 / 7 | - | 0.55 / 8 | 0.47 / 6 | 0.52 / 8 | 0.45 / 4 | 0.37 / 4 | 0.22 / 4 | 0.23 / 4 |
| (3) Permutation, M1 | 0.43 / 7 | 0.55 / 8 | - | 0.65 / 6 | 0.43 / 6 | 0.59 / 5 | 0.57 / 3 | 0.49 / 3 | 0.49 / 4 |
| (4) Permutation, XGBoost | 0.30 / 5 | 0.47 / 6 | 0.65 / 6 | - | 0.52 / 6 | 0.61 / 6 | 0.78 / 6 | 0.73 / 5 | 0.49 / 4 |
| (5) Permutation, RF | 0.43 / 5 | 0.52 / 8 | 0.43 / 6 | 0.52 / 6 | - | 0.31 / 3 | 0.50 / 5 | 0.35 / 4 | 0.10 / 3 |
| (6) KernelSHAP, M1 (fold 11) | 0.18 / 4 | 0.45 / 4 | 0.59 / 5 | 0.61 / 6 | 0.31 / 3 | - | 0.69 / 6 | 0.60 / 5 | 0.79 / 7 |
| (7) TreeSHAP, XGBoost (fold 11) | 0.09 / 3 | 0.37 / 4 | 0.57 / 3 | 0.78 / 6 | 0.50 / 5 | 0.69 / 6 | - | 0.92 / 7 | 0.55 / 5 |
| (8) TreeSHAP, RF (fold 11) | -0.01 / 3 | 0.22 / 4 | 0.49 / 3 | 0.73 / 5 | 0.35 / 4 | 0.60 / 5 | 0.92 / 7 | - | 0.54 / 4 |
| (9) KernelSHAP, GP (fold 11) | -0.05 / 4 | 0.23 / 4 | 0.49 / 4 | 0.49 / 4 | 0.10 / 3 | 0.79 / 7 | 0.55 / 5 | 0.54 / 4 | - |

Kendall's W across the 9 methods: 0.520 (p = 3.8e-45).

## Top 25 variables by mean rank across methods

| variable | mean rank | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) | (9) |
|---|---|---|---|---|---|---|---|---|---|---|
| department_UR | 1.2 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| preop_creatinine | 1.8 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| andur | 3.4 | 4 | 4 | 3 | 3 | 4 | 3 | 3 | 4 | 3 |
| uo_ml_kg_h | 12.4 | 8 | 9 | 9 | 11 | 6 | 21 | 11 | 15 | 22 |
| asa | 13.4 | 7 | 11 | 14 | 19 | 15 | 18 | 13 | 11 | 13 |
| preop_albumin | 14.4 | 34 | 20 | 23 | 21 | 13 | 5 | 6 | 3 | 5 |
| department_NS | 14.6 | 20 | 7 | 16 | 7 | 8 | 20 | 4 | 6 | 43 |
| department_GS | 14.9 | 6 | 3 | 5 | 26 | 14 | 4 | 26 | 46 | 4 |
| cvp_line | 15.3 | 12 | 15 | 7 | 6 | 41 | 8 | 12 | 20 | 17 |
| art_line | 15.6 | 3 | 5 | 4 | 5 | 3 | 13 | 44 | 48 | 15 |
| sex_M | 16.3 | 33 | 12 | 11 | 8 | 18 | 7 | 7 | 44 | 7 |
| age | 16.3 | 23 | 6 | 6 | 13 | 10 | 19 | 29 | 29 | 12 |
| preop_platelet | 21.2 | 60 | 26 | 21 | 15 | 17 | 11 | 9 | 13 | 19 |
| department_OS | 21.6 | 14 | 23 | 18 | 14 | 9 | 38 | 8 | 12 | 58 |
| preop_bun | 23.0 | 5 | 10 | 8 | 10 | 52 | 35 | 40 | 27 | 20 |
| co_monitor | 23.6 | 29 | 8 | 13 | 12 | 7 | 29 | 30 | 54 | 30 |
| uo_sum | 24.2 | 13 | 32 | 12 | 4 | 5 | 37 | 23 | 19 | 73 |
| colloid_sum | 25.3 | 53 | 35 | 22 | 16 | 28 | 9 | 16 | 40 | 9 |
| preop_hb | 28.6 | 41 | 25 | 25 | 23 | 11 | 30 | 21 | 10 | 71 |
| map_mean | 34.7 | 37 | 27 | 19 | 18 | 87 | 58 | 19 | 9 | 38 |
| map_auc_lt65 | 34.8 | 71 | 88 | 30 | 9 | 81 | 6 | 5 | 5 | 18 |
| preop_sodium | 35.8 | 42 | 45 | 35 | 40 | 29 | 31 | 15 | 33 | 52 |
| hr_mean | 36.0 | 68 | 29 | 92 | 45 | 32 | 15 | 10 | 7 | 26 |
| preop_potassium | 36.1 | 47 | 30 | 26 | 38 | 26 | 52 | 22 | 38 | 46 |
| crystalloid_sum | 36.3 | 40 | 33 | 28 | 27 | 16 | 56 | 25 | 17 | 84 |

Cells give the rank of the variable under each method (1 = most important).

