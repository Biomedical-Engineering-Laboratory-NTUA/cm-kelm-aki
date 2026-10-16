#!/bin/bash
# CPU stages (run from src/):  sbatch --export=ALL,STAGE=<stage>,... ../slurm/jobs_cpu.sh
#   STAGE=baselines  ESRD=cr4_z49 FEAT=eos2 [WIDE=1]   LR, RF, MLP, XGBoost, LightGBM, grid-tuned Nystrom-KELM
#   STAGE=tune       ESRD=cr4_z49 FEAT=eos2            random-search tuning of the boosted trees
#   STAGE=fsgrid     ESRD=cr4_z49 FEAT=eos2            grid KELM on L1 / random-forest feature subsets
#   STAGE=stats      GLOB='...' REF=<name> OUT=<file>   DeLong, bootstrap CIs, paired bootstrap, Holm
#SBATCH --job-name=cmkelm-cpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p logs preds
E=${ESRD:-cr4_z49}; F=${FEAT:-eos2}
case "$STAGE" in
  baselines) python -u run_baselines_preds.py --grid-m 1000 --esrd $E --features $F ${WIDE:+--wide-grid} ;;
  tune)      python -u tune_trees.py --esrd $E --features $F ;;
  fsgrid)    python -u run_fs_grid_kelm.py --esrd $E --features $F --selector l1 rf --k 39 ;;
  stats)     python -u stats_eval.py $GLOB --reference "$REF" --n-boot 2000 --out "$OUT" ;;
  *) echo "Set STAGE=baselines|tune|fsgrid|stats"; exit 1 ;;
esac
