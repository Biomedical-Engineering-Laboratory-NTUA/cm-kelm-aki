"""Average the probabilities of several seed runs of one model (same split) into one prediction file."""
import argparse
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from stats_eval import load_predictions, save_predictions

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--dir", default="preds")
a = ap.parse_args()
R = load_predictions([os.path.join(a.dir, r + ".npz") for r in a.runs])
y = R[a.runs[0]]["y"]
assert all(np.array_equal(R[r]["y"], y) for r in a.runs)
p = np.mean([R[r]["p"] for r in a.runs], axis=0)
print(f"{a.name} [{a.dir}]: AUROC {roc_auc_score(y, p):.4f}  AUPRC {average_precision_score(y, p):.4f}  ({len(a.runs)} runs)")
save_predictions(os.path.join(a.dir, a.name + ".npz"), a.name, y, p, runs=a.runs, combine="mean probability")
