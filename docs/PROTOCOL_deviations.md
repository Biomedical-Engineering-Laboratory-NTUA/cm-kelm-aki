# Confirmatory protocol: execution notes and deviations

This file records how the confirmatory analysis was carried out and every point in which it
differs from `docs/PROTOCOL_confirmatory.md`.

The protocol was committed on 26 October 2026 (commit `ddd8bd6`), before any confirmatory run,
and has not been changed since. At that time the project repository was private. Its status line
calls the repository public; the repository was made public later, with its full history. As
the repository was private when the protocol was committed, the commit date is not
independently verifiable.

## Execution

- **Runs.** All 1,240 runs of the outer cross-validation were completed: two repeats of five
  folds, fold seeds 101 and 102. Each outer test fold had 10,096–10,097 patients, with an AKI
  prevalence of 4.87–4.89 % (stage ≥ 2: 1.51–1.52 %).
- **Sensitivity analyses.** The protocol's analyses with the stricter exclusion rules (`cr2_z49`,
  `cr4_z49_n18`, Section 7.8) and with the 47 preoperative variables (`preop2`, Section 2) were
  run afterwards as waves 5 and 6 of `src/cv_tasks.py`: 630 further runs. Wave 7 (100 runs) added
  the exploratory ablation below.
- **Scripts.**
  - `src/outer_cv.py`: outcome stages, folds and cache.
  - `src/cv_tasks.py` with `slurm/cv_array.sh`: the task lists.
  - `src/cv_eval.py`: hypotheses and secondary comparisons.
  - `src/cv_oof.py` with `slurm/oof_array.sh`: calibration sets outside the held-out fold.
  - `src/cv_secondary.py`: calibration, thresholds, decision curves, subgroups and relevance.
  - `src/cv_interp.py`, `src/gp_shap.py` and `slurm/interp10_array.sh`: interpretability.
  - `src/fig_cv_paper.py`, `src/fig_forest.py` and `src/fig_interp.py`: the figures.
- **Results.** The aggregate results are in `results/confirmatory/`.

## Deviations

1. **Holm family of the secondary comparisons.** The protocol states seven comparisons of M1 with
   M2–M8. M8 comprises two models (logistic regression and random forest), so the adjustment
   was made over eight comparisons. This is slightly more conservative; with seven comparisons
   (M8 taken as either model) no conclusion changes (values in the supplementary material, S2).
2. **Storage of the ARD weights.** The original M1 runs stored the hyperparameters of the
   ensemble members but not their ARD weights. `src/population_ensemble.py` was changed to
   store them (`w`), and the 100 M1 searches of the primary outcome were rerun with the same
   seeds and code. They reproduced the stored predictions exactly (maximum absolute
   difference 0).
3. **Gaussian-process SHAP.** The fitted GP models were not stored, so the GP was refitted on
   the first fold (repeat 1, fold 1) with the same seeds (42–46), using
   `src/run_gp_ard.py --save-model`. GPU training of the GP is not bit-reproducible: the ensemble
   AUPRC of the refit is 0.378, against 0.374 reported.
4. **KernelSHAP sample.** KernelSHAP for M1 and M5 used 100 test patients (50 with AKI), a
   background of 10 k-means centroids, 500 coalitions and the first fold only. The protocol states 500 patients
   and 100 centroids per fold, which would cost about 50 times more model evaluations.
5. **First analysis, corrected to the protocol.** The first analysis (28 October 2026):
   - fitted Platt scaling and the fixed-sensitivity thresholds within each test fold (cross-fitted);
   - used 2,000 bootstrap replicates;
   - used two permutations for permutation importance;
   - omitted the stricter exclusion rules and the 47 preoperative variables.

   On 29–30 October 2026, after the primary results were known, these were corrected to the
   protocol:
   - calibration sets outside the held-out fold (`cv_oof.py`: 3-fold out-of-fold predictions of
     the outer training set; validation predictions for the GP and the FT-Transformer);
   - 10,000 replicates;
   - 10 permutations;
   - waves 5 and 6.

   The within-fold calibration is reported as a sensitivity analysis. The files of the first
   analysis are kept in `results/confirmatory/first_analysis_2026-10-28/`.
6. **Analyses not prespecified.** LightGBM, random forest in the permutation importance, the
   sex subgroup, the subgroup of other surgery, the comparisons with the classical models in the
   stricter cohorts and with the 47 preoperative variables, and the 2 × 2 ablation of M1 (wave 7: the
   cultural-memetic search with ARD weights, ten seeds, and top-5 ensembles of the M3 runs;
   `src/cv_ablation.py`) were not prespecified and are reported as exploratory (unadjusted
   p-values).
7. **Baseline windows.** The protocol reports the baseline windows of 90 and 7 days
   descriptively, overall and in the CKD subgroup. The article does so; the tests of H1 and H2 in
   these windows are shown for information only.

8. **Random-search controls.** The protocol gives them the budget of one DE run (480
   individuals); `src/cv_tasks.py` ran `random_search.py` with its default of 500 individuals (the
   budget of one CM-KELM run), which favours random search. The first evaluation also averaged
   the island-champion ensembles of each run, whereas M1 and M2 use top-5 ensembles. The
   controls are now reported with top-5 ensembles (`src/cv_eval.py`: RS-ARD, RS-mask); the
   island ensembles are kept as RS-ARD-isl and RS-mask-isl (`results/confirmatory/cv_ablation.md`).
   Conclusions are unchanged (M1 − random ARD weights: +0.044 AUPRC with top-5 ensembles,
   +0.042 with island ensembles). The secondary files (`cv_sec_aki.*`) list the random-search
   controls with island ensembles.

## Checks

- **Rebuilding M1.** M1 was rebuilt from the stored γ, λ and w of its 50 members per fold. It
  reproduced the stored predictions to within 1e-11 in all ten folds.
- **Rebuilding the tree models.** XGBoost and random forest were rebuilt with
  `run_baselines_preds.classical_models()` and reproduced the stored predictions.
- **Normality.** The fold differences of H1 and H2 passed Shapiro–Wilk (p = 0.63 and 0.79).
