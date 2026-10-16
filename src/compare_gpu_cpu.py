"""
Compare a GPU run with the CPU run of the same configuration and seed.
Usage: python compare_gpu_cpu.py cmkelm_m1000_adnone_p20_g15_s1 cmkelm_m1000_adnone_gpu_p20_g15_s1 [more GPU runs]
Reports: identical test predictions?, AUROC/AUPRC, best-fitness trace difference, run time and speed-up.
"""
import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def load(name):
    d = np.load(f"preds/{name}.npz")
    j = json.load(open(f"preds/{name}.json"))
    return d["y_true"].astype(int), d["p"], j


def main(ref, others):
    y, p0, j0 = load(ref)
    t0 = j0["fit_time_min"]
    tr0 = [g["best"] for g in (j0.get("trace") or [])]
    print(f"{ref:48s} AUROC {roc_auc_score(y, p0):.6f}  AUPRC {average_precision_score(y, p0):.6f}  "
          f"time {t0:6.2f} min")
    for name in others:
        _, p1, j1 = load(name)
        tr1 = [g["best"] for g in (j1.get("trace") or [])]
        dtr = max((abs(a - b) for a, b in zip(tr0, tr1)), default=float("nan"))
        ev = j1.get("fitness_evals"); es = j1.get("fitness_eval_seconds")
        print(f"{name:48s} AUROC {roc_auc_score(y, p1):.6f}  AUPRC {average_precision_score(y, p1):.6f}  "
              f"time {j1['fit_time_min']:6.2f} min  (x{t0 / j1['fit_time_min']:.1f})")
        print(f"{'':48s} max |dp| {np.abs(p0 - p1).max():.2e}  identical={np.array_equal(p0, p1)}  "
              f"max |d best-fitness trace| {dtr:.2e}"
              + (f"  | {ev} evals, {es / ev:.3f} s each" if ev else ""))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
