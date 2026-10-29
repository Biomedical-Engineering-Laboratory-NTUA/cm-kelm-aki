# Confirmatory protocol: deviations and execution notes

This file records how the confirmatory analysis was carried out. It complements
`docs/PROTOCOL_confirmatory.md`, which was registered on 26 October 2026 (commit
`ddd8bd6`) and has not been changed since. Everything not listed below followed the
registered protocol.

## Execution

- All 1,240 runs of the 5 × 2 outer cross-validation were completed (fold seeds 101
  and 102). Each outer test fold had 10,096–10,097 patients, with an AKI prevalence of
  4.87–4.89 % (stage ≥ 2: 1.51–1.52 %).
- Scripts:
  - `src/outer_cv.py`: outcome stages, folds, cache;
  - `src/cv_tasks.py` and `slurm/cv_array.sh`: task lists and Slurm array;
  - `src/cv_eval.py`: hypotheses H1/H2 and the secondary comparisons;
  - `src/cv_secondary.py`: calibration, thresholds, decision curves, subgroups, relevance;
  - `src/cv_interp.py`, `src/gp_shap.py` and `slurm/interp_array.sh`: interpretability;
  - `src/fig_cv_paper.py`: Fig. 2 of the article.
- Aggregate results are in `results/confirmatory/`.

## Deviations

1. **Holm family of the secondary comparisons.** The protocol states seven comparisons
   of M1 with the other models. M8 comprises two models (logistic regression and random
   forest), so the adjustment was made over eight comparisons. This is slightly more
   conservative.
2. **Storage of the ARD weights.** The original M1 runs stored the hyperparameters of the
   ensemble members but not their ARD weights. `src/population_ensemble.py` was changed
   to store them (`w`). The 100 M1 searches of the primary outcome were then rerun with the
   same seeds and code. They reproduced the stored out-of-fold predictions exactly
   (maximum absolute difference 0). The interpretability analyses use the rerun files.
3. **Gaussian-process SHAP.** KernelSHAP for M5 needs the fitted models, which were not
   stored. `src/run_gp_ard.py --save-model` refitted the GP on the first fold (repeat 1,
   fold 1) with the same seeds (42–46). GPU training of the GP is not bit-reproducible:
   the ensemble AUPRC of the refit is 0.378, against 0.374 reported.
4. **LightGBM.** It was run together with the other tree models but is not among M1–M8,
   so it is reported as exploratory, with unadjusted p-values.

## Checks

- **Rebuilding M1.** M1 was rebuilt from the stored γ, λ and w of its 50 members per fold.
  It reproduced the stored predictions to within 1e-11 in all ten folds.
- **Rebuilding the tree models.** XGBoost and random forest were rebuilt with
  `run_baselines_preds.classical_models()` and reproduced the stored predictions.
- **Normality.** The fold differences of H1 and H2 passed Shapiro–Wilk
  (p = 0.63 and 0.79). The paired bootstrap of the pooled predictions agreed in both repeats.
