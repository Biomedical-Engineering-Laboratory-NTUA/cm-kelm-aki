#!/bin/bash
# CM-KELM searches on one GPU (run from src/):
#   sbatch --export=ALL,SEEDS="1 2 3 4 5 6 7 8 9 10",ESRD=cr4_z49,FEAT=eos2 ../slurm/jobs_gpu.sh
# Optional: GEN, POP, M, ADAPT (e.g. none), MM (multimeme controller: fc|fuzzy|credit|uniform|hc|tabu|sa),
#           EXTRA (e.g. "--no-cultural --no-memetic" for the component ablation)
#SBATCH --job-name=cmkelm-gpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=07:00:00
export PYTHONUNBUFFERED=1
export CUPY_TF32=0            # keep FP32 GEMMs in true FP32 (no TF32)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p logs preds
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for S in ${SEEDS:-1 2 3 4 5 6 7 8 9 10}; do
  python -u run_fullcohort_cmkelm_v4.py --backend gpu --precision fp64 \
    --m "${M:-1000}" --pop "${POP:-20}" --gen "${GEN:-15}" --seed "$S" \
    --esrd "${ESRD:-cr4_z49}" --features "${FEAT:-eos2}" \
    ${ADAPT:+--adaptive $ADAPT} ${MM:+--multimeme $MM} ${EXTRA} || exit 1
done
