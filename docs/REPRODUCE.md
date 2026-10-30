# Reproducing the results

All commands run from `src/` with `INSPIRE_DIR` pointing to the INSPIRE 1.4.2 files
(PhysioNet, credentialed access). Heavy steps are submitted through Slurm (`../slurm/`).
Patient-level files (`cv_cache/`, `preds*/`, `gp_models/`, `interp_cv/`) stay on the analysis
server and are never committed; only the aggregate outputs below are in `results/`.

Cohort rules: `cr4_z49` (main), `cr2_z49` and `cr4_z49_n18` (stricter exclusions of
pre-existing kidney failure). Feature sets: `eos2` (93 end-of-surgery variables, main) and
`preop2` (47 preoperative variables).

## Map of the article

| Item | Content | Produced by |
|---|---|---|
| Table I, Table S1, Fig. S1 | cohort, attrition | `table1.py` (attrition printed by the loader) |
| Table S2 | the 93 variables | `intraop_features.py`, `cohort_split.py` |
| Table II | confirmatory discrimination, H1/H2, M1 vs M2–M8, search controls | `cv_eval.py --outcome aki` → `cv_primary.md` |
| Table S3 | stage ≥2, stage 3, baseline windows, stricter exclusions, preoperative variables | `cv_eval.py` (`--outcome`, `--baseline-days`, `--cohort`), `cv_secondary.py --parts subgroups` (CKD) |
| Fig. 1, Table S4 | calibration, utility, decision curves | `cv_oof.py` → `cv_secondary.py --platt oof` → `fig_cv_paper.py` |
| Table S5 | subgroups | `cv_secondary.py --parts subgroups` |
| Fig. S2 | paired differences | `fig_forest.py cv_primary.csv` |
| Fig. 2, Table S6 | interpretability | `cv_interp.py`, `gp_shap.py`, `cv_interp.py --aggregate`, `fig_interp.py` |
| Table S7, Table S8, Fig. S3, Section S5 | development phase (single split), search loop | sections D1–D5 below; Fig. S3 is TikZ in the LaTeX source |
| Table II (last rows), Table S9 | exploratory 2 × 2 ablation of M1 (search × encoding), random-search controls | wave 7 → `cv_eval.py --outcome aki` → `cv_ablation.py cv_primary.csv cv_ablation` |
| `results/confirmatory/hyperparameters.csv` | γ, λ and kernel weights (or masks) of every ensemble member of M1–M4 and the ablation | `export_hyperparams.py` |

## Confirmatory phase

```bash
python outer_cv.py --prepare --balance                    # cache and fold balance (main cohort)
python outer_cv.py --prepare --esrd cr2_z49               # sensitivity cohorts
python outer_cv.py --prepare --esrd cr4_z49_n18
python outer_cv.py --prepare --features preop2
for w in 1 2 3 4 5 6 7; do
  python cv_tasks.py --wave $w --part search > tasks_w${w}s.txt     # one command per line
  python cv_tasks.py --wave $w --part other  > tasks_w${w}o.txt
done
sbatch --array=1-<n>%1 --gres=gpu:1 ../slurm/cv_array.sh tasks_w1s.txt     # likewise every file
```

Waves: 1, M1/M2 searches and the baselines; 2, M3, GP, FT-Transformer and the random-search
controls; 3, stage ≥2 and stage 3; 4, baseline windows of 90 and 7 days; 5, stricter exclusions;
6, the 47 preoperative variables; 7, the cultural-memetic search with ARD weights (exploratory
ablation). `docs/RUNS.md` lists the runs.

```bash
python cv_eval.py --outcome aki --out cv_primary                       # 10,000 bootstrap replicates
python cv_eval.py --outcome st2 --out cv_st2;  python cv_eval.py --outcome st3 --out cv_st3
python cv_eval.py --baseline-days 90 --out cv_b90; python cv_eval.py --baseline-days 7 --out cv_b7
python cv_eval.py --cohort esrd-cr2_z49_eos2 --out cv_cr2
python cv_eval.py --cohort esrd-cr4_z49_n18_eos2 --out cv_n18
python cv_eval.py --cohort esrd_preop2 --out cv_preop2
python cv_ablation.py cv_primary.csv cv_ablation                         # after wave 7
python export_hyperparams.py --out ../results/confirmatory/hyperparameters.csv
sbatch --array=0-9%2 ../slurm/oof_array.sh                            # calibration sets (cv_oof.py)
python cv_secondary.py --outcome aki --platt oof --out cv_sec_aki      # Platt and thresholds outside the held-out fold
python cv_secondary.py --outcome aki --platt testcv --parts calibration,thresholds --out cv_sec_aki_testcv
python cv_secondary.py --baseline-days 90 --platt testcv --parts subgroups --out cv_sec_b90   # CKD; also 7
python fig_cv_paper.py --out fig_cv_calib_dca
python fig_forest.py cv_primary.csv f2_cv_forest
sbatch --array=0-9%3 ../slurm/interp10_array.sh                        # permutation importance (10), SHAP fold 11
CMKELM_CV=1-1 python run_gp_ard.py --esrd cr4_z49 --features eos2 --outdir preds_gpshap \
  --val-outdir preds_gpshap_val --save-model gp_models               # GP refit for its SHAP values
python gp_shap.py --models gp_models --fold 11
python cv_interp.py --aggregate --out cv_interp_aki
python fig_interp.py cv_interp_aki.csv f4_interp
```

## Development phase (single 70/30 split)

### D1. End-of-surgery variables and cohort

```bash
python intraop_features.py --data-dir $INSPIRE_DIR --out features_intraop
python table1.py --esrd cr4_z49 --features eos2 --out table1_esrd_eos2
```

### D2. Discrimination (Table S7)

```bash
bash ../slurm/reproduce_main.sh     # baselines with the 8x6 KELM grid, controls, tuned trees, CM-KELM seeds, statistics
python -u run_dl_baselines.py --model all --esrd cr4_z49 --features eos2     # TabTransformer, Mamba, stacked
python -u run_tabdl.py --model mlp --esrd cr4_z49 --features eos2            # also --model ftt
```

The JIT classifier runs from an exported split (`export_split_csv.py`, kept on the
analysis server) with `run_jit_split.py` (K = 2–8 on seed 42, then K = 3 on seeds 43–46) and
`seed_ensemble.py`. AUROC differences use DeLong's test and AUPRC differences a paired
bootstrap with 10,000 replicates (`stats_eval.py`).

### D3. Search controls and ablations (Table S8)

```bash
# full CM-KELM: 20 + 15 x (20 + 12) = 500 evaluations; plain DE 20 + 23 x 20 = 480
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",GEN=23,EXTRA="--no-cultural --no-memetic" ../slurm/jobs_gpu.sh
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",GEN=15,EXTRA="--no-cultural" ../slurm/jobs_gpu.sh
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",GEN=23,EXTRA="--no-memetic" ../slurm/jobs_gpu.sh
for A in KAH R; do sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",ADAPT=$A ../slurm/jobs_gpu.sh; done
for C in fc fuzzy credit uniform hc tabu sa; do sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",MM=$C ../slurm/jobs_gpu.sh; done
for s in 1 2 3 4 5 6 7 8 9 10; do python -u random_search.py --backend gpu --seed $s --esrd cr4_z49 --features eos2; done
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",EXTRA="--ard --penalty 0" ../slurm/jobs_gpu.sh
python -u run_gp_ard.py --esrd cr4_z49 --features eos2
python compare_seeds.py cmkelm_m1000_gpu_p20_g15__esrd_eos2 <variant>__esrd_eos2   # paired Wilcoxon, 95% CI
python cv_fitness_ablation.py --logs "logs/*.log" --out cv_fitness_ablation        # CV fitness (no test data)
```

### D4. Sensitivity analyses (Section S5)

Preoperative variables (`FEAT=preop2`), stricter exclusions (`ESRD=cr2_z49`, `cr4_z49_n18`),
equal ensemble size (`grid_ensemble.py --k 40`), extended grid (`--xwide-grid`), complexity
penalty (`--penalty 0 / 0.0005 / 0.002`), missing-value indicators (`--miss-ind`), calibration
outside the test set (`calib_oof.py`, `calibration_dca.py --calib-dir`), selected variables and
their stability (`feature_selection_freq.py`, `mask_stability.py`).

### D5. Computation (Section S5)

```bash
python compare_gpu_cpu.py <cpu_run_name> <gpu_run_name>     # equivalence and speed-up
python collect_timings.py                                   # timings.csv
```

The C++ backends (`../cpp/`) print their training time; the article reports the median of
three runs on the training set of the final cohort with m = 1000 and 5000.

`figs_paper.py` draws the figures of the earlier, single-split versions of the article and is
kept for the development phase only.
