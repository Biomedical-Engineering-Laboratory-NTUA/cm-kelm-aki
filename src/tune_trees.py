"""
Hyperparameter tuning of the boosted-tree baselines (sensitivity analysis).
=======================================================================

Random search over XGBoost / LightGBM hyperparameters on the TRAINING set only,
with the same criterion as the CM-KELM fitness and the grid-tuned KELM:
mean AUPRC over a stratified 3-fold CV (the same folds as the CM-KELM fitness). The best configuration
is refitted on the whole training set and evaluated once on the test set.
Predictions are saved in the standard format, so stats_eval.py and
calibration_dca.py work on them unchanged.

Search space (per model, --n-iter random draws, plus the current default):
  XGBoost   n_estimators 200-1500, learning_rate 0.01-0.2 (log), max_depth 3-8,
            min_child_weight 1-20 (log), subsample 0.6-1, colsample_bytree 0.5-1,
            reg_lambda 0.1-20 (log), gamma 0-5, scale_pos_weight {1, sqrt(neg/pos), neg/pos}
  LightGBM  n_estimators 200-1500, learning_rate 0.01-0.2 (log), num_leaves 8-128 (log),
            min_child_samples 10-200 (log), subsample 0.6-1, colsample_bytree 0.5-1,
            reg_lambda 0.1-20 (log), scale_pos_weight {1, sqrt(neg/pos), neg/pos}

Usage (server, CPU job):
  python -u tune_trees.py --esrd cr4_z49 --features eos2 --models xgboost lightgbm --n-iter 40 --jobs 16
Outputs:
  preds/xgboost_tuned__esrd_eos2.npz/.json, preds/lightgbm_tuned__esrd_eos2.npz/.json
  tune_trees_<tag>.csv (every configuration with its CV-AUPRC)
"""

import argparse
import csv
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cohort_split import load_split
from stats_eval import save_predictions

RANDOM_STATE = 42


def loguni(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample(model, rng, ratio):
    spw = [1.0, float(np.sqrt(ratio)), float(ratio)][rng.integers(3)]
    if model == "xgboost":
        return dict(n_estimators=int(rng.integers(200, 1501)), learning_rate=loguni(rng, 0.01, 0.2),
                    max_depth=int(rng.integers(3, 9)), min_child_weight=loguni(rng, 1, 20),
                    subsample=float(rng.uniform(0.6, 1.0)), colsample_bytree=float(rng.uniform(0.5, 1.0)),
                    reg_lambda=loguni(rng, 0.1, 20), gamma=float(rng.uniform(0, 5)), scale_pos_weight=spw)
    return dict(n_estimators=int(rng.integers(200, 1501)), learning_rate=loguni(rng, 0.01, 0.2),
                num_leaves=int(round(loguni(rng, 8, 128))), min_child_samples=int(round(loguni(rng, 10, 200))),
                subsample=float(rng.uniform(0.6, 1.0)), colsample_bytree=float(rng.uniform(0.5, 1.0)),
                reg_lambda=loguni(rng, 0.1, 20), scale_pos_weight=spw)


DEFAULTS = {"xgboost": dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8),
            "lightgbm": dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                             colsample_bytree=0.8)}


def make(model, params, jobs):
    if model == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=jobs, **params)
    from lightgbm import LGBMClassifier
    p = dict(params)
    p.setdefault("subsample_freq", 1)
    return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=jobs, **p)


def cv_auprc(model, params, X, y, folds, jobs):
    s = []
    for tr, va in folds:
        m = make(model, params, jobs).fit(X[tr], y[tr])
        s.append(average_precision_score(y[va], m.predict_proba(X[va])[:, 1]))
    return float(np.mean(s)), float(np.std(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2", choices=["preop", "eos", "eos2"])
    ap.add_argument("--models", nargs="+", default=["xgboost", "lightgbm"])
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    a = ap.parse_args()

    Xtr, Xte, ytr, yte, _, tag = load_split(dummy=a.dummy, esrd=a.esrd, features=a.features)
    suffix = "" if tag == "all" else f"__{tag}"
    ratio = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xtr, ytr))  # same folds as the EA fitness
    log = []
    for model in a.models:
        rng = np.random.default_rng(a.seed)
        configs = [DEFAULTS[model]] + [sample(model, rng, ratio) for _ in range(a.n_iter)]
        best = (-np.inf, None)
        t0 = time.time()
        for i, params in enumerate(configs):
            mu, sd = cv_auprc(model, params, Xtr, ytr, folds, a.jobs)
            log.append(dict(model=model, i=i, cv_auprc=round(mu, 5), cv_sd=round(sd, 5), params=json.dumps(params)))
            flag = ""
            if mu > best[0]:
                best, flag = (mu, params), "  <- best"
            print(f"{model} {i:3d}/{len(configs) - 1} CV-AUPRC {mu:.4f} ± {sd:.4f}{flag}", flush=True)
        cv_best, params = best
        t1 = time.time()
        m = make(model, params, a.jobs).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        name = f"{model}_tuned"
        print(f"\n{name}: CV-AUPRC {cv_best:.4f} (default {log[[r['model'] for r in log].index(model)]['cv_auprc']:.4f})"
              f" -> test AUROC {roc_auc_score(yte, p):.4f}  AUPRC {average_precision_score(yte, p):.4f}"
              f"  ({(time.time() - t0) / 60:.1f} min)\n  params {json.dumps(params)}\n", flush=True)
        save_predictions(f"{a.outdir}/{name}{suffix}.npz", name + suffix, yte, p, cv_auprc=cv_best,
                         params=params, n_configs=len(configs), fit_time_s=time.time() - t1,
                         cohort=tag, split_seed=42)
    with open(f"tune_trees_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"written tune_trees_{tag}.csv")


if __name__ == "__main__":
    main()
