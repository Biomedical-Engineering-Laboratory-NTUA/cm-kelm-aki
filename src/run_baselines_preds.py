"""
Baselines on the same split, saving test-set predictions for stats_eval.py.
==========================================================================

  logistic_regression, random_forest (300 trees)      class-balanced
  xgboost, lightgbm (400 trees, learning rate 0.05, depth 5 / 31 leaves, row and
                     column subsampling 0.8)          if installed
  mlp                                                  not reported in the paper
  nystrom_kelm_grid_m{m}[_wide]  the key control: Nystrom-KELM on all variables, with
                     (gamma, lambda) chosen by grid search on the criterion of the
                     CM-KELM fitness (3-fold CV AUPRC). --wide-grid = the 8 x 6 grid
                     of the paper (name suffix _wide).

Usage (always via srun/sbatch, unbuffered):
    python -u run_baselines_preds.py --grid-m 1000 --esrd cr4_z49 --features eos2 --wide-grid
    python -u run_baselines_preds.py --skip-grid           # classical baselines only
    python -u run_baselines_preds.py --dummy               # smoke test, no real data
"""

import argparse
import itertools
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from cohort_split import load_split
from nystrom_kelm import NystromKELM
from stats_eval import save_predictions

RANDOM_STATE = 42
GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3]
LAMBDAS = [0.01, 0.1, 1.0, 10.0]
# grid of the paper (the initial 5 x 4 grid above selected its smallest gamma)
GAMMAS_WIDE = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
LAMBDAS_WIDE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# extended grid (name suffix _xwide): three smaller gammas and one smaller lambda
GAMMAS_XWIDE = [0.00001, 0.00003, 0.0001] + GAMMAS_WIDE
LAMBDAS_XWIDE = [0.0001] + LAMBDAS_WIDE


def classical_models():
    models = [
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                                  class_weight="balanced", n_jobs=-1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)),
    ]
    try:
        from xgboost import XGBClassifier
        models.append(("xgboost", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                                subsample=0.8, colsample_bytree=0.8,
                                                eval_metric="logloss", random_state=RANDOM_STATE,
                                                n_jobs=-1)))
    except ImportError:
        print("xgboost not installed -- skipped (pip install xgboost)", flush=True)
    try:
        from lightgbm import LGBMClassifier
        models.append(("lightgbm", LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                  num_leaves=31, subsample=0.8, subsample_freq=1,
                                                  colsample_bytree=0.8, random_state=RANDOM_STATE,
                                                  verbose=-1, n_jobs=-1)))
    except ImportError:
        print("lightgbm not installed -- skipped (pip install lightgbm)", flush=True)
    return models


def grid_nystrom(Xtr, ytr, m, seed=RANDOM_STATE, gammas=None, lambdas=None):
    """Plain grid search, inner 3-fold CV AUPRC (same criterion as the EA fitness,
    without feature selection or complexity penalty)."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(skf.split(Xtr, ytr))
    best = (-np.inf, None, None)
    for g, lam in itertools.product(gammas or GAMMAS, lambdas or LAMBDAS):
        scores = []
        for tr, va in folds:
            mdl = NystromKELM(gamma=g, lam=lam, n_landmarks=m, random_state=seed).fit(Xtr[tr], ytr[tr])
            scores.append(average_precision_score(ytr[va], mdl.predict_proba(Xtr[va])))
        s = float(np.mean(scores))
        print(f"    m={m} gamma={g:<6} lambda={lam:<5} CV-AUPRC={s:.4f}", flush=True)
        if s > best[0]:
            best = (s, g, lam)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-m", type=int, nargs="+", default=[500, 1000])
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--wide-grid", action="store_true", help="8 gammas x 6 lambdas (name suffix _wide)")
    ap.add_argument("--xwide-grid", action="store_true", help="11 gammas x 7 lambdas (name suffix _xwide)")
    ap.add_argument("--miss-ind", action="store_true", help="missing-value indicators (cohort_split.py)")
    ap.add_argument("--max-baseline-creatinine", type=float, default=None)
    ap.add_argument("--esrd", default=None, choices=["cr4_z49", "cr2_z49", "cr4_z49_n18"],
                    help="exclude pre-existing kidney failure (cohort tag 'esrd')")
    ap.add_argument("--features", default="preop", choices=["preop", "preop2", "eos", "eos2"],
                    help="eos = + end-of-surgery features (features_intraop.csv.gz)")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    a = ap.parse_args()

    Xtr, Xte, ytr, yte, _, tag = load_split(dummy=a.dummy,
                                            max_baseline_creatinine=a.max_baseline_creatinine,
                                            esrd=a.esrd, features=a.features, missing_ind=a.miss_ind)
    suffix = "" if tag == "all" else f"__{tag}"

    for name, model in ([] if a.skip_classical else classical_models()):
        t0 = time.time()
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        dt = time.time() - t0
        print(f"{name:22s} AUROC {roc_auc_score(yte, p):.4f}  AUPRC "
              f"{average_precision_score(yte, p):.4f}  ({dt:.1f}s)", flush=True)
        save_predictions(f"{a.outdir}/{name}{suffix}.npz", name + suffix, yte, p,
                         fit_time_s=dt, cohort=tag, split_seed=42)

    if a.skip_grid:
        return
    for m in a.grid_m:
        m_eff = min(m, len(ytr) // 3)  # dummy-data guard
        gs, ls = ((GAMMAS_XWIDE, LAMBDAS_XWIDE) if a.xwide_grid else
                  (GAMMAS_WIDE, LAMBDAS_WIDE) if a.wide_grid else (GAMMAS, LAMBDAS))
        print(f"\nGrid Nystrom-KELM, m={m_eff} ({len(gs) * len(ls)} configs x 3 folds)", flush=True)
        t0 = time.time()
        cv, g, lam = grid_nystrom(Xtr, ytr, m_eff, gammas=gs, lambdas=ls)
        mdl = NystromKELM(gamma=g, lam=lam, n_landmarks=m_eff, random_state=RANDOM_STATE).fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)
        dt = time.time() - t0
        name = f"nystrom_kelm_grid_m{m}" + ("_xwide" if a.xwide_grid else "_wide" if a.wide_grid else "")
        print(f"{name}: best gamma={g}, lambda={lam} (CV-AUPRC {cv:.4f}) -> test AUROC "
              f"{roc_auc_score(yte, p):.4f}  AUPRC {average_precision_score(yte, p):.4f}  "
              f"({dt / 60:.1f} min)", flush=True)
        save_predictions(f"{a.outdir}/{name}{suffix}.npz", name + suffix, yte, p,
                         gamma=g, lam=lam, n_landmarks=m, cv_auprc=cv, fit_time_s=dt,
                         cohort=tag, split_seed=42)


if __name__ == "__main__":
    main()
