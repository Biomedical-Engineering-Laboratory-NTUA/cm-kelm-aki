"""
Calibration sets of the confirmatory protocol (Sections 4 and 7.4-7.5): predictions that never
touch the held-out fold.
================================================================================================

For each outer fold (repeat r, fold k) and model, writes preds_cv_oof/<model>__<tag>.npz with the
labels and predicted risks of the outer TRAINING patients, from models that did not see them:

  * KELM models (M1 DE-KELM-ARD, M2 DE-KELM, M3 CM-KELM, M4 grid-tuned KELM) and the classical
    models (M6 XGBoost, M8 logistic regression and random forest, LightGBM): out-of-fold
    predictions from a stratified 3-fold split of the outer training set (random_state 0, as the
    fitness folds). Every fold model uses the settings of the reported model: for the searches,
    gamma, lambda and the mask or ARD weights of every ensemble member of every seed, averaged
    per seed and over seeds as in the reported ensemble;
  * M5 sparse GP and M7 FT-Transformer: their validation predictions (15% of the outer training
    set, used only for early stopping), written by run_gp_ard.py / run_tabdl.py --val-outdir.

cv_secondary.py --platt oof then fits Platt scaling and the fixed-sensitivity thresholds on these
predictions and applies them unchanged to the held-out fold. These files are patient-level and
stay on the server (preds_cv_oof/ is in .gitignore).

    python -u cv_oof.py --fold 11            # one outer fold (Slurm array: slurm/oof_array.sh)
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import outer_cv
from cv_eval import MODELS, EXCLUDE
from nystrom_kelm import NystromKELM
from run_baselines_preds import classical_models

KELM_SEARCH = ("M1", "M2", "M3")
CLASSICAL = {"M6": "xgboost", "M8a": "logistic_regression", "M8b": "random_forest", "LGBM": "lightgbm"}
VALIDATION = {"M5": "gp_ard_ens5__{tag}.npz", "M7": "dl_ftt_ens5__{tag}.npz"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def files_of(mid, tag, preds):
    files = sorted(glob.glob(os.path.join(preds, MODELS[mid][1].format(tag=tag))))
    if mid in EXCLUDE:
        files = [f for f in files if not re.search(EXCLUDE[mid], os.path.basename(f))]
    return [f for f in files if f"__{tag}." in f or f"__{tag}__" in f]


def meta(path):
    return json.loads(str(np.load(path, allow_pickle=False)["meta"]))


def kelm_oof(X, y, folds, gamma, lam, mask, rs, m=1000):
    out = np.empty(len(y))
    for tr, va in folds:
        mdl = NystromKELM(gamma=gamma, lam=lam, n_landmarks=m, random_state=rs).fit(X[tr] * mask, y[tr])
        out[va] = mdl.predict_proba(X[va] * mask)
    return out


def member_mask(mb, d):
    if "w" in mb:
        return np.asarray(mb["w"], float)
    mask = np.zeros(d)
    mask[mb["features"]] = 1.0
    return mask


def save(path, y, p, src):
    np.savez_compressed(path, y_true=np.asarray(y).astype(np.int8), p=np.asarray(p, float),
                        meta=json.dumps({"source": src}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="repeat and fold, e.g. 11")
    ap.add_argument("--outcome", default="aki", choices=["aki", "st2", "st3"])
    ap.add_argument("--preds", default="preds_cv")
    ap.add_argument("--val", default="preds_cv_val")
    ap.add_argument("--outdir", default="preds_cv_oof")
    ap.add_argument("--models", default="M1,M2,M3,M4,M5,M6,M7,M8a,M8b,LGBM")
    a = ap.parse_args()
    r, k = int(a.fold[0]), int(a.fold[1])
    Xtr, _, ytr, _, names, tag = outer_cv.load_fold(r, k, a.outcome, verbose=False)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(Xtr, ytr))
    os.makedirs(a.outdir, exist_ok=True)
    log(f"fold {r}{k}: training set {Xtr.shape}, AKI {ytr.mean():.4f}; 3-fold out-of-fold predictions")
    todo = a.models.split(",")
    classical = dict(classical_models())

    for mid in todo:
        t0 = time.time()
        out = os.path.join(a.outdir, f"{mid}__{tag}.npz")
        if mid in KELM_SEARCH:
            per_seed = []
            for f in files_of(mid, tag, a.preds):
                mem = meta(f)["members"]
                if mid == "M1" and not all("w" in mb for mb in mem):
                    raise SystemExit(f"{f}: ARD members without weights")
                per_seed.append(np.mean([kelm_oof(Xtr, ytr, folds, mb["gamma"], mb["lam"],
                                                  member_mask(mb, Xtr.shape[1]), rs=0) for mb in mem], axis=0))
            if not per_seed:
                log(f"{mid}: no files"); continue
            p, y, src = np.mean(per_seed, axis=0), ytr, f"train-OOF ({len(per_seed)} seeds)"
        elif mid == "M4":
            fs = files_of(mid, tag, a.preds)
            if not fs:
                log("M4: no file"); continue
            mt = meta(fs[0])
            p, y, src = kelm_oof(Xtr, ytr, folds, mt["gamma"], mt["lam"], np.ones(Xtr.shape[1]), rs=42,
                                 m=mt.get("n_landmarks", 1000)), ytr, "train-OOF"
        elif mid in CLASSICAL:
            p = np.empty(len(ytr))
            for tr, va in folds:
                p[va] = clone(classical[CLASSICAL[mid]]).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
            y, src = ytr, "train-OOF"
        elif mid in VALIDATION:
            f = os.path.join(a.val, VALIDATION[mid].format(tag=tag))
            if not os.path.exists(f):
                log(f"{mid}: no validation file {f}"); continue
            z = np.load(f)
            y, p, src = z["y_true"].astype(int), z["p"].astype(float), "validation"
        else:
            continue
        save(out, y, p, src)
        log(f"{mid:5s} {src:22s} n={len(y)}  AUROC {roc_auc_score(y, p):.4f}  AUPRC "
            f"{average_precision_score(y, p):.4f}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
