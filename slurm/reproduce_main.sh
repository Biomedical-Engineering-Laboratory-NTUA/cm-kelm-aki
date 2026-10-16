#!/bin/bash
# Main end-of-surgery results of the paper (Table 3, Figs. 2-5). Run from src/ after
# INSPIRE_DIR is set and features_intraop.csv.gz has been built (see docs/REPRODUCE.md).
set -e
mkdir -p logs preds preds_ens
b=$(sbatch --parsable --export=ALL,STAGE=baselines,ESRD=cr4_z49,FEAT=eos2 ../slurm/jobs_cpu.sh)
t=$(sbatch --parsable --export=ALL,STAGE=tune,ESRD=cr4_z49,FEAT=eos2 ../slurm/jobs_cpu.sh)
g=$(sbatch --parsable --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",ESRD=cr4_z49,FEAT=eos2 ../slurm/jobs_gpu.sh)
# seed ensembles (preds_ens/) and the comparison against the grid-tuned KELM
sbatch --dependency=afterok:$b:$g --job-name=paper-stats --output=logs/%x_%j.log --cpus-per-task=4 \
  --wrap="python -u summarize_seeds.py && python -u stats_eval.py \
          preds/logistic_regression__esrd_eos2.npz preds/random_forest__esrd_eos2.npz \
          preds/xgboost__esrd_eos2.npz preds/lightgbm__esrd_eos2.npz preds/nystrom_kelm_grid_m1000__esrd_eos2.npz \
          preds_ens/cmkelm_m1000_gpu_p20_g15__esrd_eos2__popens_islands_ens.npz \
          --reference nystrom_kelm_grid_m1000__esrd_eos2 --n-boot 2000 --out stats_esrd_eos2"
# Note: the paper's Holm family also contains the four deep-learning baselines (external code, see README),
# so adjusted p-values from this command differ slightly; unadjusted p-values and CIs are identical.
echo "submitted baselines $b, tuning $t, CM-KELM $g (+ stats)"
