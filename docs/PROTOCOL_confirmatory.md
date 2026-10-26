# Confirmatory analysis protocol (pre-specified)

**Study:** Evolutionary Ensembles of Nyström Kernel Extreme Learning Machines for Postoperative Acute Kidney Injury Prediction: An Equal-Budget Study of Search Complexity

**Authors:** J. Digalakis, M. Cafaro, T. Chatzistamatiou, I. Vezakis, G. K. Matsopoulos, D. D. Koutsouris

**Status:** FINAL (26 Oct 2026, after comments by M.C. and T.C.), committed before any analysis of Sections 6–7 was run. This protocol is committed to the public repository before any of the analyses below is run; the commit hash and timestamp serve as its registration. Deviations are listed in Section 11.

## 1. Background and rationale

All analyses so far used one stratified 70/30 split of the cohort. On that test set:

- plain differential evolution with population ensembles (DE-KELM, binary variable mask) improved on the grid-tuned Nyström KELM;
- the cultural-memetic variant (CM-KELM, specified first) added nothing;
- random variable subsets ensembled as well as evolved ones;
- with continuous per-variable kernel weights (ARD), the evolutionary search outperformed random search over the same weights at the same budget, and matched a sparse Gaussian process with ARD kernel trained by the marginal likelihood.

The ARD variant and its controls were added after the main analysis, and all variants were compared on the same test set. This protocol fixes, before any new run, a confirmatory evaluation with three properties:

- it does not depend on a single split;
- it evaluates plain DE with ARD weights (DE-KELM-ARD), which has not been run;
- it adds a clinically more relevant outcome, subgroups, interpretability and clinically motivated thresholds.

The data have been analysed before. The protocol therefore pre-specifies a new evaluation design and new hypotheses, not a new dataset.

## 2. Data and cohort

- INSPIRE v1.4.2 (PhysioNet). The cohort is the same as in the manuscript:
  - first eligible operation per patient;
  - preoperative creatinine within one year;
  - at least one postoperative creatinine within 7 days.
- Exclusion of pre-existing kidney failure: creatinine at the top code (5.54 mg/dL) or ICD-10 Z49 before surgery. Rule `cr4_z49`; n = 50,484.
- Predictors: the 93 end-of-surgery variables (`eos2`). The 47 preoperative variables form a secondary analysis.
- Patient-level data stay on the analysis server under the PhysioNet data use agreement; only aggregate results leave it.

## 3. Outcomes

- **Primary outcome:** AKI by the KDIGO relative creatinine criterion. The peak postoperative creatinine within 7 days is ≥ 1.5 × baseline. This is unchanged from the manuscript.
- **Key secondary outcome:** KDIGO stage ≥ 2 AKI (stages 2 and 3 combined), from creatinine only:
  - stage 2: peak 2.0–2.9 × baseline;
  - stage 3: peak ≥ 3.0 × baseline, or peak ≥ 4.0 mg/dL in a patient who meets stage 1. Initiation of renal replacement therapy cannot be identified reliably (ICD-10 Z49 recorded after surgery and within 7 days in only one patient), and no patient of the cohort is younger than 18 years, so the pediatric eGFR criterion does not apply.

  Stage 3 alone is modelled only if it has at least 100 events. The counts computed before this commit (Section 11) give 342 stage-3 events, so stage 3 is modelled as a further secondary outcome, with the same models and design as stage ≥ 2.

  Postoperative urine output is not recorded, so the urine-output criterion cannot be applied. Initiation of renal replacement therapy is used only if it can be identified from ICD-10 procedure or diagnosis codes after surgery; otherwise this is stated as a limitation.
- **Baseline creatinine:** the most recent preoperative value within one year (as in the manuscript). KDIGO (2012, Chapter 2.4) considers a value obtained 6 months or even 1 year earlier a reasonable reflection of the premorbid baseline in patients without chronic kidney disease, but not in patients with CKD and a slowly rising creatinine. The distribution of the interval between the baseline value and surgery is reported. Pre-specified sensitivity analyses restrict the baseline to the last 90 days and to the last 7 days (the routine preoperative test), with the outcome redefined accordingly; they are run for M1, M2 and M4 and reported descriptively, overall and in the CKD subgroup, where the concern is largest.
- Percentile coding of laboratory values in INSPIRE is reported as a limitation of both outcome definitions.

## 4. Evaluation design

- **Repeated stratified outer cross-validation on the whole cohort:**
  - 5 folds × 2 repeats;
  - stratified jointly on the outcome and on the clinical groups, with nine strata: outcome (no AKI, stage 1, stage ≥ 2) × surgical group (cardiac surgery with cardiopulmonary bypass, urology, other). Stratification is random within strata, so every fold reproduces the prevalence of both outcomes and the size of the pre-specified surgical subgroups;
  - fold seeds 101 and 102, not used before;
  - the balance of the folds is reported: per fold, n, prevalence of both outcomes, and size of every subgroup of Section 7.6, including age ≥ 65 and CKD, which are not used for stratification.
- Within each outer training fold, everything is fitted on the training fold only:
  - imputation and standardization;
  - the inner 3-fold CV fitness of the searches;
  - grid search;
  - model training;
  - Platt calibration (on inner out-of-fold or validation predictions).
- Each model then predicts the held-out fold.
- For each repeat, the held-out predictions are pooled into one out-of-fold prediction per patient.
- The deep and GP models use 85/15 train/validation splits inside each outer training fold, as in the manuscript.
- **Secondary-outcome models:** separate models are trained for stage 2–3 AKI with the same design. The fitness of the searches uses the stage 2–3 label.

## 5. Models

All search budgets are those of the manuscript.

| ID | Model | Seeds per outer fold |
|---|---|---|
| M1 | **DE-KELM-ARD**: plain DE (population 20, 23 generations, 480 evaluations), ARD encoding w~j~ = exp(clip(z~j~, −6, 3)), penalty 0, top-5 ensemble per run, averaged over seeds | 10 |
| M2 | DE-KELM: plain DE with binary mask, penalty 0.001, top-5 ensemble | 10 |
| M3 | CM-KELM: island ensemble | 5 |
| M4 | Grid-tuned Nyström KELM: 8×6 grid, same CV criterion | landmark seed 42 |
| M5 | Sparse variational GP with ARD kernel: 1000 inducing points, ELBO | 5 |
| M6 | XGBoost (default configuration of the manuscript) | — |
| M7 | FT-Transformer: rtdl reference implementation, default configuration | 5 |
| M8 | Logistic regression, random forest (as in the manuscript) | — |

Additional controls:

- Random search over ARD weights at the budget of M1, 10 seeds, in the first repeat only.
- Random search over masks, same budget, in the first repeat only.

## 6. Hypotheses (confirmatory)

Primary metric: AUPRC of the pooled out-of-fold predictions. Each repeat is evaluated, and the paired differences over the 10 outer folds are tested.

- **H1:** M1 (DE-KELM-ARD) has higher AUPRC than M4 (grid-tuned KELM).
- **H2:** M1 has higher AUPRC than M2 (DE-KELM with binary mask).

Testing:

- H1 and H2 are tested one-sided at α = 0.05, Holm-adjusted over the two.
- Test: the corrected resampled t-test of Nadeau and Bengio over the 10 fold-level paired differences (5 folds × 2 repeats), which accounts for the overlap of training sets. The mean difference is reported with its corrected 95% confidence interval.
- The normality of the 10 fold-level differences, which the test assumes, is examined with a normal quantile plot and the Shapiro–Wilk test. With 10 differences these checks have little power, so they are descriptive and do not change the decision rule; the paired bootstrap below does not assume normality.
- A paired bootstrap of the pooled predictions (10,000 replicates) is reported as a sensitivity analysis.

H1 or H2 is confirmed only if its adjusted p < 0.05.

## 7. Secondary and descriptive analyses (no confirmatory claim)

1. AUROC for H1–H2. All pairwise differences of M1 with M2–M8, with Holm adjustment over the seven comparisons. For every model, AUPRC and AUROC are reported per fold and as mean with the corrected (Nadeau–Bengio) 95% confidence interval, so that the stability of each difference across folds can be judged.
2. **M1 vs M5 (evolved vs marginal-likelihood ARD):** the difference with 95% CI. Agreement is described, not claimed as equivalence.
3. Stage 2–3 outcome: discrimination of M1, M2, M4, M5 and M6.
4. **Calibration** (Platt fitted outside the held-out fold):
   - Brier score;
   - slope and intercept;
   - reliability diagram.
5. **Clinical thresholds 5%, 10%, 20%:**
   - share of patients flagged;
   - sensitivity and specificity;
   - PPV and NPV;
   - net benefit and decision curves.

   In addition, operating points at fixed sensitivity (80% and 90%), with the corresponding specificity, PPV, NPV and share flagged, chosen on the training folds and applied unchanged to the held-out folds.

   Each threshold is linked to an action of increasing intensity, following the KDIGO prevention measures: 5%, review of nephrotoxic drugs and contrast, and a creatinine check 24–48 h after surgery; 10%, the KDIGO prevention bundle (avoidance of nephrotoxins, optimization of volume status and perfusion, avoidance of hyperglycemia, close monitoring of creatinine and urine output); 20%, the bundle plus early nephrology review or confirmation by a kidney stress biomarker, as in the PrevAKI trial.
6. **Subgroups:**
   - cardiac surgery with cardiopulmonary bypass;
   - urology;
   - age ≥ 65 years;
   - emergency surgery;
   - CKD, defined as eGFR < 60 mL/min/1.73 m² (CKD-EPI 2021 without race, from baseline creatinine, age and sex). Because creatinine is percentile-coded, eGFR is approximate; this is reported as a limitation.

   For each subgroup: AUROC and AUPRC with 95% CI, and calibration within the subgroup.
7. **Interpretability:**
   - Normalized ARD weights of M1, averaged over seeds and folds.
   - GP relevances (1/lengthscale) of M5.
   - Permutation importance (drop in held-out AUPRC, 10 permutations) for M1 and M6.
   - SHAP values: KernelSHAP for M1 and M5 (500 held-out patients per fold, background of 100 k-means centroids of the training fold), TreeSHAP for M6 and the random forest; mean absolute SHAP value per variable.

   The first two are model-parameter (a priori) importances, the last two prediction-based (a posteriori) ones; agreement within and between the two families is reported. Reported for each:
   - Spearman correlation between the rankings;
   - overlap of the top-10 variables;
   - stability across folds (Kendall's W).

   The ranking is reviewed for clinical and laboratory plausibility by T.C.
8. Sensitivity analyses:
   - baseline creatinine (Section 3);
   - the stricter exclusion rules of the manuscript (`cr2_z49`, `cr4_z49_n18`), for M1, M2 and M4 only.

## 8. Reporting

- All analyses in Sections 6–7 are reported, whatever their result.
- Results of the single-split analysis of the manuscript remain reported and are labelled as exploratory where they concern ARD.
- The code of every analysis is in the repository. A list of all runs is kept in `docs/RUNS.md`.

## 9. Computational plan

- **Searches:**
  - M1 and M2: 10 seeds × 10 folds each;
  - M3: 5 seeds × 10 folds;
  - random-search controls: 10 seeds × 5 folds.

  This is about 300 searches on GPU (about 3 min each) or CPU (about 25 min each, 8 cores).
- **Deep and GP models:** about 5 min per seed and fold.

## 10. Roles

| Person | Role |
|---|---|
| J.D. | Code, runs, analysis |
| T.C. | Laboratory and clinical definitions (Sections 3, 4 stratification, 7.5, 7.6), plausibility review (7.7) |
| M.C. | Review of the design and of the code of the searches |
| I.V., G.K.M., D.D.K. | Review of the protocol and the manuscript |

## 11. Deviations

*(none yet; any deviation from this protocol will be listed here with its date and reason)*

Before the commit, the following aggregate counts were computed without fitting any model (`src/protocol_counts.py`, 26 Oct 2026):

- Cohort: n = 50,484; no patient younger than 18 years.
- Interval between baseline creatinine and anesthesia start: median 1.9 days (quartiles 1.0 and 7.8; 90th percentile 24.9). 74.1% of baselines were measured within 7 days, 92.8% within 30 days and 99.9% within 90 days. AKI prevalence: 5.3%, 5.0% and 4.9% in these groups, against 4.9% overall.
- KDIGO stage from creatinine: stage 0 in 48,019 patients, stage 1 in 1,703, stage 2 in 420 and stage 3 in 342 (133 of them by the 4.0 mg/dL criterion only). Stage ≥ 1: 2,465 (4.9%); stage ≥ 2: 762 (1.5%). Post-operative Z49 within 7 days: 1 patient.
- Strata, stage 0 / 1 / ≥ 2:
  - cardiac surgery with bypass: 1,712 / 311 / 189;
  - urology: 4,306 / 642 / 135;
  - other: 42,001 / 750 / 438.
- Subgroups, with prevalence of stage ≥ 1 and stage ≥ 2:

  | Subgroup | n | Stage ≥ 1 | Stage ≥ 2 |
  |---|---|---|---|
  | Age ≥ 65 | 21,896 | 5.4% | 1.9% |
  | Emergency surgery | 5,067 | 7.1% | 2.9% |
  | CKD | 3,279 | 13.6% | 6.8% |
  | Cardiac surgery with bypass | 2,212 | 22.6% | 8.5% |
  | Urology | 5,087 | 15.3% | 2.7% |
