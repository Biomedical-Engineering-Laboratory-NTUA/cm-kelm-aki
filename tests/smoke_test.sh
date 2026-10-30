#!/bin/bash
# Runs the whole pipeline on synthetic dummy data (no INSPIRE access needed), in a few minutes on a CPU.
set -e
cd "$(dirname "$0")/../src"
OUT=$(mktemp -d)
python run_baselines_preds.py --dummy --grid-m 200 --outdir $OUT
python run_fullcohort_cmkelm_v4.py --dummy --m 200 --pop 8 --gen 3 --seed 1 --outdir $OUT
python run_fullcohort_cmkelm_v4.py --dummy --m 200 --pop 8 --gen 3 --seed 1 --outdir $OUT --no-cultural --no-memetic
python run_fs_grid_kelm.py --dummy --features preop --m 200 --k 10 --outdir $OUT
python stats_eval.py $OUT/*.npz --reference nystrom_kelm_grid_m200 --n-boot 50 --out $OUT/stats
# confirmatory cross-validation (outer folds, one ARD search) and end-of-surgery variables on synthetic data
python outer_cv.py --dummy --balance
CMKELM_CV=1-1 python run_fullcohort_cmkelm_v4.py --dummy --m 50 --pop 6 --gen 2 --seed 1 --outdir $OUT --no-cultural --no-memetic --ard --penalty 0
python intraop_features.py --dummy --out $OUT/intraop
echo "smoke test passed ($OUT)"
