"""
Aggregate CM-KELM runs across search seeds.
===========================================

1. Mean +/- SD of AUROC / AUPRC per configuration (the numbers the paper
   reports for a stochastic method, rather than a single lucky seed).
2. Seed ensemble per configuration: the average of the predicted
   probabilities of the seeds. Saved to preds_ens/<config>_ens.npz, kept in a
   separate folder so stats_eval.py never mixes ensembles and single seeds
   by accident.

Usage:
    python summarize_seeds.py                      # reads preds/cmkelm_*.npz
    python stats_eval.py preds_ens/*.npz preds/nystrom_kelm_grid_m1000.npz \
        preds/random_forest.npz --reference nystrom_kelm_grid_m1000 --out stats_ens_vs_grid
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from stats_eval import save_predictions


def main(pattern="preds/cmkelm_*.npz", outdir="preds_ens"):
    groups = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        d = np.load(path, allow_pickle=False)
        name = json.loads(str(d["meta"]))["name"]
        cfg = re.sub(r"_s\d+(?=(__|$))", "", name)
        groups[cfg].append((name, d["y_true"].astype(int), d["p"]))

    os.makedirs(outdir, exist_ok=True)
    print(f"{'configuration':32s} {'n':>2s}  {'AUROC mean±SD':>16s}  {'AUPRC mean±SD':>16s}  "
          f"{'ensemble AUROC':>14s}  {'ensemble AUPRC':>14s}")
    for cfg, runs in groups.items():
        y = runs[0][1]
        if any(not np.array_equal(r[1], y) for r in runs):
            raise SystemExit(f"{cfg}: seeds do not share the same test set")
        au = np.array([roc_auc_score(y, r[2]) for r in runs])
        ap = np.array([average_precision_score(y, r[2]) for r in runs])
        p_ens = np.mean([r[2] for r in runs], axis=0)
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
        print(f"{cfg:32s} {len(runs):2d}  {au.mean():.4f} ± {sd(au):.4f}  {ap.mean():.4f} ± {sd(ap):.4f}  "
              f"{roc_auc_score(y, p_ens):14.4f}  {average_precision_score(y, p_ens):14.4f}")
        save_predictions(f"{outdir}/{cfg}_ens.npz", f"{cfg}_ens", y, p_ens,
                         seeds=[r[0] for r in runs], kind="seed_ensemble")


if __name__ == "__main__":
    main()
