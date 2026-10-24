"""
Calibration sets that never touch the test set (review point: test-set recalibration).
=====================================================================================

For each model of the calibration figure, writes <outdir>/<test file name>.npz with labels and
predicted risks of patients that the model did not see when it was fitted:

  * KELM models (CM-KELM ensemble, grid-tuned KELM) and the classical baselines: out-of-fold
    predictions on the TRAINING set, from a stratified 3-fold split (the folds of the CM-KELM
    fitness); each fold model uses the same settings as the final model (for CM-KELM: the
    gamma, lambda and variables of every island champion of every seed, averaged as in the
    reported ensemble).
  * deep-learning models: their validation predictions (15% of the training set, used only for
    early stopping), written by run_jit_split.py / run_dl_baselines.py --val-outdir.

calibration_dca.py --calib-dir <outdir> then fits Platt scaling on these predictions and applies
it unchanged to the test predictions.

    python -u calib_oof.py --esrd cr4_z49 --features eos2 --outdir preds_cal
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cohort_split import load_split
from nystrom_kelm import NystromKELM
from run_baselines_preds import classical_models
from stats_eval import load_predictions, save_predictions


def meta(path):
    return json.loads(str(np.load(path, allow_pickle=False)["meta"]))


def kelm_oof(X, y, folds, gamma, lam, m, mask, rs):
    out = np.empty(len(y))
    for tr, va in folds:
        mdl = NystromKELM(gamma=gamma, lam=lam, n_landmarks=m, random_state=rs).fit(X[tr] * mask, y[tr])
        out[va] = mdl.predict_proba(X[va] * mask)
    return out


def report(name, y, p, outdir, src):
    save_predictions(f"{outdir}/{name}.npz", name, y, p, calibration_source=src)
    print(f"  {name:60s} {src:10s} n={len(y)}  AUROC {roc_auc_score(y, p):.4f}  AUPRC "
          f"{average_precision_score(y, p):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2")
    ap.add_argument("--cmkelm", default="cmkelm_m1000_gpu_p20_g15", help="configuration (seeds 1-10)")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--grid", default="nystrom_kelm_grid_m1000_wide")
    ap.add_argument("--jit", default="dl_jit_fb3_s{s}", help="JIT runs in preds_val, {s} = seed")
    ap.add_argument("--jit-seeds", default="42-46")
    ap.add_argument("--jit-name", default="dl_jit_fb3_ens5")
    ap.add_argument("--dl", nargs="*", default=["dl_transformer_e100_s42", "dl_mamba_e100_s42",
                                                "dl_stacked_lr_h8_e100_s42"])
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--val-dir", default="preds_val")
    ap.add_argument("--outdir", default="preds_cal")
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()

    Xtr, _, ytr, _, cols, tag = load_split(dummy=a.dummy, esrd=None if a.dummy else a.esrd,
                                           features="preop" if a.dummy else a.features, verbose=False)
    sfx = "" if tag == "all" else f"__{tag}"
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(Xtr, ytr))
    os.makedirs(a.outdir, exist_ok=True)
    print(f"training set {Xtr.shape}, AKI {ytr.mean():.4f}; out-of-fold predictions (3 folds)", flush=True)

    # CM-KELM: island champions of every seed, averaged per seed and then over seeds
    lo, hi = (int(v) for v in a.seeds.split("-"))
    per_seed = []
    for s in range(lo, hi + 1):
        f = f"{a.preds}/{a.cmkelm}_s{s}{sfx}__popens_islands.npz"
        if not os.path.exists(f):
            print(f"  missing {f}", flush=True)
            continue
        mt = meta(f)
        m = mt.get("n_landmarks", 1000)
        m = min(m, len(ytr) // 3) if a.dummy else m
        ps = []
        for mb in mt["members"]:
            mask = np.zeros(Xtr.shape[1]); mask[mb["features"]] = 1.0
            ps.append(kelm_oof(Xtr, ytr, folds, mb["gamma"], mb["lam"], m, mask, rs=0))
        per_seed.append(np.mean(ps, axis=0))
    if per_seed:
        report(f"{a.cmkelm}{sfx}__popens_islands_ens", ytr, np.mean(per_seed, axis=0), a.outdir,
               f"train-OOF ({len(per_seed)} seeds)")

    # grid-tuned KELM (landmark seed 42, as the reported model)
    f = f"{a.preds}/{a.grid}{sfx}.npz"
    if os.path.exists(f):
        mt = meta(f)
        m = mt.get("n_landmarks", 1000)
        m = min(m, len(ytr) // 3) if a.dummy else m
        report(f"{a.grid}{sfx}", ytr, kelm_oof(Xtr, ytr, folds, mt["gamma"], mt["lam"], m,
                                               np.ones(Xtr.shape[1]), rs=42), a.outdir, "train-OOF")

    # classical baselines, same constructors as run_baselines_preds.py
    for name, model in classical_models():
        if name == "mlp":
            continue
        out = np.empty(len(ytr))
        for tr, va in folds:
            out[va] = clone(model).fit(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
        report(f"{name}{sfx}", ytr, out, a.outdir, "train-OOF")

    # deep-learning models: validation predictions
    lo, hi = (int(v) for v in a.jit_seeds.split("-"))
    jf = [f"{a.val_dir}/{a.jit.format(s=s)}{sfx}.npz" for s in range(lo, hi + 1)]
    if all(os.path.exists(f) for f in jf):
        R = list(load_predictions(jf).values())
        assert all(np.array_equal(r["y"], R[0]["y"]) for r in R), "JIT seeds with different validation sets"
        report(f"{a.jit_name}{sfx}", R[0]["y"], np.mean([r["p"] for r in R], axis=0), a.outdir, "validation")
    else:
        print(f"  JIT validation files missing: {[f for f in jf if not os.path.exists(f)]}", flush=True)
    for d in a.dl:
        f = f"{a.val_dir}/{d}{sfx}.npz"
        if os.path.exists(f):
            r = list(load_predictions([f]).values())[0]
            report(f"{d}{sfx}", r["y"], r["p"], a.outdir, "validation")
        else:
            print(f"  missing {f} (run_dl_baselines.py --val-outdir {a.val_dir})", flush=True)


if __name__ == "__main__":
    main()
