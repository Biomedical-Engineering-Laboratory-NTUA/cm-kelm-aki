"""
Control for the ensemble size of CM-KELM (review point A1).

The grid-tuned Nystrom-KELM is one model with one draw of landmarks, whereas the
reported CM-KELM averages 40 models (4 island champions x 10 runs). This script refits
the grid-tuned model with its selected (gamma, lambda) on all variables for 40 landmark
draws (random_state 42..81; draw 42 is the reported grid-tuned model) and averages the
probabilities, so the value of averaging alone can be separated from feature selection.

    python -u grid_ensemble.py --esrd cr4_z49 --features eos2 --k 40
Output: preds/nystrom_kelm_grid_m1000_wide_ens40__esrd_eos2.npz (+ per-draw AUROC/AUPRC printed)
"""
import argparse
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cohort_split import load_split
from nystrom_kelm import NystromKELM
from stats_eval import load_predictions, save_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2")
    ap.add_argument("--grid", default="nystrom_kelm_grid_m1000_wide")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--m", type=int, default=1000)
    a = ap.parse_args()

    Xtr, Xte, ytr, yte, cols, tag = load_split(esrd=a.esrd, features=a.features, verbose=False)
    sfx = "" if tag == "all" else f"__{tag}"
    ref = load_predictions([f"preds/{a.grid}{sfx}.npz"])[a.grid + sfx]
    g, lam = float(ref["meta"]["gamma"]), float(ref["meta"]["lam"])
    print(f"{a.grid}{sfx}: gamma={g}, lambda={lam}; {a.k} landmark draws", flush=True)
    ps, au, ap_ = [], [], []
    t0 = time.time()
    for k in range(a.k):
        p = NystromKELM(gamma=g, lam=lam, n_landmarks=a.m, random_state=42 + k).fit(Xtr, ytr).predict_proba(Xte)
        ps.append(p)
        au.append(roc_auc_score(yte, p)); ap_.append(average_precision_score(yte, p))
        if k == 0:
            print(f"  draw 42 reproduces the reported model: max |dp| = {np.abs(p - ref['p']).max():.2e}", flush=True)
    pe = np.mean(ps, axis=0)
    print(f"single draws: AUROC {np.mean(au):.4f} +/- {np.std(au, ddof=1):.4f}, "
          f"AUPRC {np.mean(ap_):.4f} +/- {np.std(ap_, ddof=1):.4f}", flush=True)
    print(f"RESULT {a.grid}_ens{a.k}{sfx}: AUROC {roc_auc_score(yte, pe):.4f} | "
          f"AUPRC {average_precision_score(yte, pe):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    save_predictions(f"preds/{a.grid}_ens{a.k}{sfx}.npz", f"{a.grid}_ens{a.k}{sfx}", yte, pe,
                     gamma=g, lam=lam, n_draws=a.k, single_auroc=au, single_auprc=ap_)


if __name__ == "__main__":
    main()
