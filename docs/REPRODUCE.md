# Reproducing the results

All commands run from `src/` with `INSPIRE_DIR` set. Heavy steps should be submitted through Slurm
(`../slurm/`); CPU runs take minutes, a CM-KELM search about 3 min on one GPU.
Cohort tags: `cr4_z49` (primary), `cr2_z49` and `cr4_z49_n18` (sensitivity). Feature sets:
`preop` (37 variables), `eos2` (93 end-of-surgery variables).

## 0. End-of-surgery variables

```bash
python intraop_features.py --data-dir $INSPIRE_DIR --out features_intraop   # -> features_intraop.csv.gz
```

## 1. Cohort (Fig. S1, Table II, Table S2)

```bash
python table1.py --esrd cr4_z49 --features eos2 --out table1_esrd_eos2   # attrition is printed by the loader
```

## 2. Main comparison (Table III, Figs. 2 and S2)

```bash
bash ../slurm/reproduce_main.sh       # baselines, tuned trees, 10 CM-KELM seeds, seed ensembles, statistics
```

The primary model is `preds_ens/cmkelm_m1000_gpu_p20_g15__esrd_eos2__popens_islands_ens.npz`
(mean probability of the island ensembles of seeds 1-10).

## 3. Preoperative versus end of surgery (Fig. 3, Table S4)

Repeat step 2 with `FEAT=preop`.

## 4. Calibration and decision curves (Fig. 4)

```bash
python calibration_dca.py preds/logistic_regression__esrd_eos2.npz preds/nystrom_kelm_grid_m1000__esrd_eos2.npz \
  preds/random_forest__esrd_eos2.npz preds/lightgbm__esrd_eos2.npz preds/xgboost__esrd_eos2.npz \
  preds_ens/cmkelm_m1000_gpu_p20_g15__esrd_eos2__popens_islands_ens.npz \
  --calib-models cmkelm,xgboost,nystrom_kelm_grid --out fig_calib_eos2
```

(The paper's figure also includes the external deep-learning baselines.)

## 5. Selected variables (Fig. 5)

```bash
python feature_selection_freq.py "preds/cmkelm_m1000_gpu_p20_g15_s*__esrd_eos2.npz" --out fsel_esrd_eos2
```

## 6. Search ablations (Table IV, Table S5, Fig. S3)

```bash
# multimeme controllers at equal budget (seeds 1-10; replication 11-20; confirmation 21-50)
sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",ADAPT=none,MM=fc,FEAT=eos ../slurm/jobs_gpu.sh
python compare_seeds.py cmkelm_m1000_gpu_p20_g15__esrd_eos cmkelm_m1000_mmfc_gpu_p20_g15__esrd_eos
# component ablation (equal number of fitness evaluations)
sbatch --export=ALL,GEN=23,EXTRA="--no-cultural --no-memetic" ../slurm/jobs_gpu.sh
sbatch --export=ALL,GEN=15,EXTRA="--no-cultural" ../slurm/jobs_gpu.sh
sbatch --export=ALL,GEN=23,EXTRA="--no-memetic" ../slurm/jobs_gpu.sh
```

## 7. Sensitivity analyses (Table S1)

```bash
for E in cr2_z49 cr4_z49_n18; do
  sbatch --export=ALL,STAGE=baselines,ESRD=$E,FEAT=eos2 ../slurm/jobs_cpu.sh
  sbatch --export=ALL,SEEDS="1 2 3 4 5",ESRD=$E,FEAT=eos2 ../slurm/jobs_gpu.sh
done
# then seed_ensemble.py over the five *__popens_islands runs and stats_eval.py vs the grid KELM
```

## 8. Computation (Fig. 6)

```bash
python compare_gpu_cpu.py <cpu_run_name> <gpu_run_name>     # equivalence and speed-up
python collect_timings.py                                   # timings.csv
```

## 9. Figures

`figs_paper.py` redraws Figs. 2, 3, 5, 6 and S1-S3 from the aggregate numbers above.
