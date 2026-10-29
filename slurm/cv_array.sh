#!/bin/bash
# One line of a task file of cv_tasks.py per array task (run from src/):
#   python cv_tasks.py --wave 1 --part search > tasks_w1s.txt
#   sbatch --array=1-$(wc -l < tasks_w1s.txt)%2 --gres=gpu:1 ../slurm/cv_array.sh tasks_w1s.txt
#   python cv_tasks.py --wave 1 --part other > tasks_w1o.txt
#   sbatch --array=1-$(wc -l < tasks_w1o.txt)%2 ../slurm/cv_array.sh tasks_w1o.txt
#SBATCH --job-name=cv
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
export PYTHONUNBUFFERED=1
export CUPY_TF32=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
[ -n "$VENV" ] && source "$VENV"
mkdir -p logs preds_cv preds_cv_val
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
[ -z "$LINE" ] && { echo "no line ${SLURM_ARRAY_TASK_ID} in $1"; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID}: $LINE"
# a task that finished before is skipped (marker per command line), so arrays can be resubmitted
mkdir -p cv_done
MARK="cv_done/$(printf '%s' "$LINE" | md5sum | cut -c1-16)"
if [ -f "$MARK" ]; then echo "done before, skipped"; exit 0; fi
eval "$LINE" && echo "$LINE" > "$MARK"
