"""
Control: grid-tuned Nystrom-KELM on a feature subset chosen by a STANDARD selector
(instead of the evolutionary mask). Answers the reviewer question "is the gain of
CM-KELM just feature selection, which any filter/embedded method would give?".

Selectors (fitted on the training set only):
  l1   L1-penalised logistic regression (balanced weights); C chosen by the same
       3-fold CV AUPRC on the training set; variables with non-zero coefficients.
  rf   random-forest impurity importance (300 trees, balanced); top-k variables,
       k = --k (default 39 = mean mask size of CM-KELM, eos2).
Age and creatinine are always kept, as in CM-KELM (domain knowledge).

Usage (server, CPU; always via srun/sbatch):
  python -u run_fs_grid_kelm.py --esrd cr4_z49 --features eos2 --selector l1 rf --k 39
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cohort_split import load_split
from nystrom_kelm import NystromKELM
from run_baselines_preds import RANDOM_STATE, grid_nystrom, GAMMAS_WIDE, LAMBDAS_WIDE
from stats_eval import save_predictions

PROTECTED = [0, 5]   # age, preoperative creatinine (same indices as CM-KELM)
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def select_l1(Xtr, ytr):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    best = (-np.inf, None)
    for C in C_GRID:
        s = []
        for tr, va in skf.split(Xtr, ytr):
            m = LogisticRegression(penalty="l1", C=C, solver="liblinear", class_weight="balanced",
                                   max_iter=2000).fit(Xtr[tr], ytr[tr])
            s.append(average_precision_score(ytr[va], m.predict_proba(Xtr[va])[:, 1]))
        print(f"    L1 C={C:<6} CV-AUPRC={np.mean(s):.4f}", flush=True)
        if np.mean(s) > best[0]:
            best = (float(np.mean(s)), C)
    m = LogisticRegression(penalty="l1", C=best[1], solver="liblinear", class_weight="balanced",
                           max_iter=2000).fit(Xtr, ytr)
    sel = set(np.flatnonzero(np.abs(m.coef_.ravel()) > 1e-8)) | set(PROTECTED)
    return np.array(sorted(sel)), {"C": best[1], "cv_auprc_lr": best[0]}


def select_rf(Xtr, ytr, k):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE,
                                n_jobs=-1).fit(Xtr, ytr)
    order = list(np.argsort(rf.feature_importances_)[::-1])
    sel = list(PROTECTED) + [j for j in order if j not in PROTECTED][:max(0, k - len(PROTECTED))]
    return np.array(sorted(sel)), {"k": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esrd", default="cr4_z49", choices=["cr4_z49", "cr2_z49", "cr4_z49_n18"])
    ap.add_argument("--features", default="eos2", choices=["preop", "eos", "eos2"])
    ap.add_argument("--selector", nargs="+", default=["l1", "rf"], choices=["l1", "rf"])
    ap.add_argument("--k", type=int, default=39)
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    a = ap.parse_args()

    Xtr, Xte, ytr, yte, names, tag = load_split(dummy=a.dummy, esrd=a.esrd, features=a.features)
    suffix = "" if tag == "all" else f"__{tag}"
    m = min(a.m, len(ytr) // 3)
    for sel_name in a.selector:
        t0 = time.time()
        if sel_name == "l1":
            sel, info = select_l1(Xtr, ytr)
            name = f"nystrom_kelm_grid_m{a.m}_l1sel_wide"
        else:
            sel, info = select_rf(Xtr, ytr, a.k)
            name = f"nystrom_kelm_grid_m{a.m}_rfsel{a.k}_wide"
        print(f"\n{name}: {len(sel)} of {Xtr.shape[1]} variables selected", flush=True)
        cv, g, lam = grid_nystrom(Xtr[:, sel], ytr, m, gammas=GAMMAS_WIDE, lambdas=LAMBDAS_WIDE)
        mdl = NystromKELM(gamma=g, lam=lam, n_landmarks=m, random_state=RANDOM_STATE).fit(Xtr[:, sel], ytr)
        p = mdl.predict_proba(Xte[:, sel])
        dt = time.time() - t0
        print(f"{name}: gamma={g}, lambda={lam} (CV-AUPRC {cv:.4f}) -> test AUROC "
              f"{roc_auc_score(yte, p):.4f}  AUPRC {average_precision_score(yte, p):.4f}  ({dt / 60:.1f} min)",
              flush=True)
        save_predictions(f"{a.outdir}/{name}{suffix}.npz", name + suffix, yte, p, gamma=g, lam=lam,
                         n_landmarks=a.m, cv_auprc=cv, n_selected=len(sel),
                         selected=[str(names[j]) for j in sel], fit_time_s=dt, cohort=tag, split_seed=42,
                         **info)


if __name__ == "__main__":
    main()
