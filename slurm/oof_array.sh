#!/bin/bash
# Slurm array 0-9: calibration sets outside the held-out fold (cv_oof.py), one outer fold per task.
#   sbatch --array=0-9%2 --cpus-per-task=8 --mem=32G ../slurm/oof_array.sh      (from src/)
F=(11 12 13 14 15 21 22 23 24 25)
f=${F[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${SLURM_SUBMIT_DIR:-.}"
[ -n "$VENV" ] && source "$VENV"
python -u cv_oof.py --fold "$f"
