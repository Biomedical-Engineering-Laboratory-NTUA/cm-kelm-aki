#!/bin/bash
# Slurm array 0-9: permutation importance with 10 permutations (protocol Section 7.7), one outer
# fold per task; SHAP on fold 11.
#   sbatch --array=0-9%3 --cpus-per-task=4 --mem=24G ../slurm/interp10_array.sh   (from src/)
F=(11 12 13 14 15 21 22 23 24 25)
f=${F[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "${SLURM_SUBMIT_DIR:-.}"
[ -n "$VENV" ] && source "$VENV"
extra=""; [ "$f" = "11" ] && extra="--shap"
python -u cv_interp.py --fold "$f" --n-perm 10 $extra
