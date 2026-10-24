# Reproducing the results

All commands run from `src/` with `INSPIRE_DIR` set. Heavy steps should be submitted through Slurm
(`../slurm/`); CPU runs take minutes, a CM-KELM search about 3 min on one GPU.
Cohort tags: `cr4_z49` (primary), `cr2_z49` and `cr4_z49_n18` (sensitivity). Feature sets:
`preop2` (47 preoperative variables), `eos2` (93 end-of-surgery variables) and `preop` (an earlier
37-variable set with anesthesia duration and without medications, reported in Table S5).
`jobs_gpu.sh` defaults to `ESRD=cr4_z49`, `FEAT=eos2`. Statistics use 10,000 bootstrap replicates
(`--n-boot 10000`).

Tables and figures: main text Tables I-III and Figs. 1-3; supplement Tables S1-S9 and Figs. S1-S4.

## 0. End-of-surgery variables

```bash
python intraop_features.py --data-dir $INSPIRE_DIR --out features_intraop   # -> features_intraop.csv.gz
```

## 1. Cohort (Fig. S1, Table I, Table S2)

```bash
python table1.py --esrd cr4_z49 --features eos2 --out table1_esrd_eos2   # attrition is printed by the loader
```

## 2. Main comparison (Table II, Figs. 2 and S2)

```bash
bash ../slurm/reproduce_main.sh       # baselines with the 8x6 KELM grid, feature-selection controls,
                                      # tuned trees, 10 CM-KELM seeds, seed ensembles, statistics
```

The primary model is `preds_ens/cmkelm_m1000_gpu_p20_g15__esrd_eos2__popens_islands_ens.npz`
(mean probability of the island ensembles of seeds 1-10); the reference is
`preds/nystrom_kelm_grid_m1000_wide__esrd_eos2.npz` (grid-tuned KELM, 8x6 grid).

## 2b. Deep-learning baselines (Table II)

GPU jobs; the split CSV contains patient-level data and must stay in a private directory.

```bash
python -u run_dl_baselines.py --model all --esrd cr4_z49 --features eos2     # TabTransformer, Mamba, stacked_lr_h8
python export_split_csv.py --esrd cr4_z49 --features eos2 --out <private_dir>/inspire_esrd_eos2_split.csv
# feedback steps K = 2..8 with seed 42; K = 3 has the best validation AUROC
for K in 2 3 4 5 6 7 8; do
  python run_jit_split.py --split-csv <private_dir>/inspire_esrd_eos2_split.csv --seed 42 \
    --name dl_jit_fb${K}_s42__esrd_eos2 -- --feedback_steps $K
done
for s in 43 44 45 46; do
  python run_jit_split.py --split-csv <private_dir>/inspire_esrd_eos2_split.csv --seed $s \
    --name dl_jit_fb3_s${s}__esrd_eos2 -- --feedback_steps 3
done
python seed_ensemble.py --runs dl_jit_fb3_s42__esrd_eos2 dl_jit_fb3_s43__esrd_eos2 dl_jit_fb3_s44__esrd_eos2 \
  dl_jit_fb3_s45__esrd_eos2 dl_jit_fb3_s46__esrd_eos2 --name dl_jit_fb3_ens5__esrd_eos2
```

The article's files are `preds/dl_transformer_e100_s42__esrd_eos2.npz`, `preds/dl_mamba_e100_s42__esrd_eos2.npz`,
`preds/dl_stacked_lr_h8_e100_s42__esrd_eos2.npz` and `preds/dl_jit_fb3_ens5__esrd_eos2.npz`.

## 3. Preoperative versus end of surgery (Fig. S2, Table S5)

Repeat step 2 with `FEAT=preop2` (and `FEAT=preop` for the 37-variable set); baselines with `WIDE=1`,
as for the end-of-surgery model.

## 4. Calibration and decision curves (Fig. 3; see 6c for calibration outside the test set)

```bash
python calibration_dca.py preds/logistic_regression__esrd_eos2.npz preds/nystrom_kelm_grid_m1000_wide__esrd_eos2.npz \
  preds/random_forest__esrd_eos2.npz preds/lightgbm__esrd_eos2.npz preds/xgboost__esrd_eos2.npz \
  preds/dl_transformer_e100_s42__esrd_eos2.npz preds/dl_mamba_e100_s42__esrd_eos2.npz \
  preds/dl_stacked_lr_h8_e100_s42__esrd_eos2.npz preds/dl_jit_fb3_ens5__esrd_eos2.npz \
  preds_ens/cmkelm_m1000_gpu_p20_g15__esrd_eos2__popens_islands_ens.npz \
  --calib-models cmkelm,dl_jit,xgboost,nystrom_kelm_grid --out fig_calib_eos2
```

## 5. Selected variables and their stability (Fig. S3, Section S7)

```bash
python feature_selection_freq.py "preds/cmkelm_m1000_gpu_p20_g15_s*__esrd_eos2.npz" --out fsel_esrd_eos2
python mask_stability.py --esrd cr4_z49 --features eos2 --out mask_stability   # Jaccard, Nogueira et al. (2018)
```

## 6. Search ablations on the final cohort (Table III, Table S8)

All variants use seeds 1-10 (`SEEDS="1 2 3 4 5 6 7 8 9 10"`); replication and confirmation seeds
11-50 are written to `preds_rep/` with `EXTRA="--outdir preds_rep"`, so that the ten-seed ensembles
stay seeds 1-10.

```bash
# component ablation (equal number of fitness evaluations)
# full CM-KELM: 20 + 15 x (20 + 12) = 500 evaluations; without local search 20 + 23 x 20 = 480
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",GEN=23,EXTRA="--no-cultural --no-memetic" ../slurm/jobs_gpu.sh
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",GEN=15,EXTRA="--no-cultural" ../slurm/jobs_gpu.sh
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",GEN=23,EXTRA="--no-memetic" ../slurm/jobs_gpu.sh
# adaptive mechanisms (K+A+H) and noise-robust fitness (R)
for A in KAH R; do sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",ADAPT=$A ../slurm/jobs_gpu.sh; done
# multimeme controllers and single memes
for C in fc fuzzy credit uniform hc tabu sa; do
  sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",MM=$C ../slurm/jobs_gpu.sh
done
sbatch --export=ALL,SEEDS="$(seq -s ' ' 11 30)",MM=tabu,EXTRA="--outdir preds_rep" ../slurm/jobs_gpu.sh
sbatch --export=ALL,SEEDS="$(seq -s ' ' 31 50)",MM=tabu,EXTRA="--outdir preds_rep" ../slurm/jobs_gpu.sh
# (the static reference for seeds 11-50 is run the same way without MM)
# paired comparisons with the 95% CI of the mean difference
B=cmkelm_m1000_gpu_p20_g15__esrd_eos2
python compare_seeds.py $B cmkelm_m1000_gpu_nocult_nomem_p20_g23__esrd_eos2
python compare_seeds.py $B cmkelm_m1000_mmtabu_gpu_p20_g15__esrd_eos2 1-10
```

## 6b. Controls of the search (Table III, Table S9)

```bash
# random search over masks and (gamma, lambda) with the budget of one CM-KELM run, island ensembling
for s in 1 2 3 4 5 6 7 8 9 10; do python -u random_search.py --backend gpu --seed $s --esrd cr4_z49 --features eos2; done
# ARD: the same search evolving one continuous kernel weight per variable, no penalty
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",EXTRA="--ard --penalty 0" ../slurm/jobs_gpu.sh
python summarize_seeds.py
python compare_seeds.py cmkelm_m1000_gpu_p20_g15__esrd_eos2 cmkelm_m1000_gpu_randsearch_n500__esrd_eos2
python compare_seeds.py cmkelm_m1000_gpu_p20_g15__esrd_eos2 cmkelm_m1000_gpu_ard_pen0_p20_g15__esrd_eos2
# the same comparisons on the CV fitness of the best individual (no test data), from the Slurm logs
python cv_fitness_ablation.py --logs "logs/*.log" --out cv_fitness_ablation
# variables used by the ensembles, overlap with the development extraction
python masks_overlap.py --out masks_overlap
```

## 6c. MLP and FT-Transformer; calibration outside the test set (Table II, Fig. 3, Table S6)

```bash
pip install rtdl_revisiting_models
python -u run_tabdl.py --model mlp --esrd cr4_z49 --features eos2      # seeds 42-46 -> dl_mlp_ens5
python -u run_tabdl.py --model ftt --esrd cr4_z49 --features eos2      # -> dl_ftt_ens5
python -u run_dl_baselines.py --model all --esrd cr4_z49 --features eos2 --outdir preds_rerun --val-outdir preds_val
python -u calib_oof.py --esrd cr4_z49 --features eos2 --outdir preds_cal \
  --dl dl_transformer_e100_s42 dl_mamba_e100_s42 dl_stacked_lr_h8_e100_s42 dl_mlp_ens5 dl_ftt_ens5
python calibration_dca.py <test prediction files> --calib-dir preds_cal --calib-models cmkelm,dl_jit,xgboost,nystrom_kelm_grid
```

`calib_oof.py` writes calibration sets that never use the test set (training out-of-fold predictions for the
KELM, linear and tree models; validation predictions for the deep models); `calibration_dca.py --calib-dir`
fits Platt scaling there and applies it unchanged to the test predictions.

All GPU searches of the article are listed in `docs/RUNS.md`.

## 7. Sensitivity analyses and additional controls (Tables S1 and S9)

```bash
for E in cr2_z49 cr4_z49_n18; do
  sbatch --export=ALL,STAGE=baselines,ESRD=$E,FEAT=eos2,WIDE=1 ../slurm/jobs_cpu.sh
  sbatch --export=ALL,SEEDS="1 2 3 4 5",ESRD=$E,FEAT=eos2 ../slurm/jobs_gpu.sh
done
# grid-tuned KELM averaged over 40 landmark draws (equal ensemble size)
python -u grid_ensemble.py --esrd cr4_z49 --features eos2 --k 40
# extended grid (gamma >= 1e-5, lambda >= 1e-4)
python -u run_baselines_preds.py --skip-classical --grid-m 1000 --xwide-grid --esrd cr4_z49 --features eos2
# complexity penalty and missing-value indicators
for P in 0 0.0005 0.002; do
  sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",EXTRA="--penalty $P" ../slurm/jobs_gpu.sh
done
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",EXTRA="--miss-ind" ../slurm/jobs_gpu.sh
python -u run_baselines_preds.py --skip-classical --grid-m 1000 --wide-grid --miss-ind --esrd cr4_z49 --features eos2
# then seed_ensemble.py / summarize_seeds.py and stats_eval.py --n-boot 10000 against the reference
```

## 8. Computation (Fig. S4)

```bash
python compare_gpu_cpu.py <cpu_run_name> <gpu_run_name>     # equivalence and speed-up
python collect_timings.py                                   # timings.csv
```

The C++ backends (`../cpp/`) print their training time; the article reports the median of three
runs on the training set of the final cohort with m = 1000 and 5000.

## 9. Figures

`figs_paper.py stats_main.md` redraws Fig. 2 and Figs. S1-S4 from the aggregate numbers above
(`stats_main.md` is the `stats_eval.py` output of the main comparison).
