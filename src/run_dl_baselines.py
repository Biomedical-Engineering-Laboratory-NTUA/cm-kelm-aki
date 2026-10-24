"""
Deep-learning baselines (dl/, M. Cafaro) on the article's cohort and split.
==========================================================================

Trains the networks of ../dl/ on the same cohort, features and test set as every other model:
  transformer   TabTransformer                 (dl/tabtransformer/transformer.py)
  mamba         tabular Mamba / selective SSM  (dl/mamba/mamba.py)
  stacked       TabTransformer + Mamba base learners with three meta-learners on the
                validation predictions: stacked_lr (logistic regression; the stacked model of
                the article), stacked (calibrated SVC) and stacked_svcbal (balanced SVC)

The model classes are imported unchanged; only the data handling and bookkeeping differ from
the stand-alone scripts:

* Same cohort, features and 70/30 test split as every other model (cohort_split.load_split),
  so the test predictions go straight into stats_eval.py. The stand-alone scripts use a
  70/15/15 split of a CSV instead.
* The validation set (best epoch by validation AUROC; meta-learners of the stacked model) is a
  stratified 15% of the TRAINING part, so the test set is never used for training or selection.
* AdamW (learning rate 1e-3, weight decay 1e-4), batch 256, 100 epochs, float32; positive-class
  weight 20 for the stand-alone TabTransformer, n_neg/n_pos otherwise; ReduceLROnPlateau
  (factor 0.5, patience 3) only for the stand-alone TabTransformer; 8 attention heads.
* The best epoch is kept by deep-copying the state dict (--faithful keeps a reference to the
  live parameters instead, i.e. the last epoch, as in early versions of the stand-alone scripts).
* The stacked model re-uses the stand-alone Mamba and trains its own TabTransformer (no
  scheduler, positive-class weight n_neg/n_pos). The Mamba base predictions (validation + test)
  are cached in dl_cache/, so --model stacked --reuse-mamba re-runs only the stacked
  TabTransformer.

Outputs (aggregate, safe to share):
  preds/dl_<model>_e<epochs>_s<seed>[__<cohort tag>].npz   test predictions
  preds/dl_<model>_e<epochs>_s<seed>[__<cohort tag>].json  per-epoch validation
                                                            AUROC/AUPRC, times

Usage (GPU job):
  python -u run_dl_baselines.py --model all --esrd cr4_z49 --features eos2
  python -u run_dl_baselines.py --model transformer --epochs 1 --device cpu   # smoke test
"""

import argparse
import copy
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from cohort_split import load_split
from stats_eval import save_predictions

HERE = os.path.dirname(os.path.abspath(__file__))


DL_DIR = os.path.join(os.path.dirname(HERE), "dl")


def load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DL_DIR, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)       # the files only define classes and print the device
    return mod


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# data in the two input formats of the models
# ----------------------------------------------------------------------
def transformer_inputs(Xtr, Xva, Xte, threshold=10):
    """Same rule as prepare_data of transformer.py: <= 10 distinct values -> categorical."""
    nun = np.array([len(np.unique(Xtr[:, j])) for j in range(Xtr.shape[1])])
    cat = np.flatnonzero(nun <= threshold)
    cont = np.flatnonzero(nun > threshold)
    if len(cat):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        c_tr = enc.fit_transform(Xtr[:, cat]).astype(np.int64) + 1
        c_va = enc.transform(Xva[:, cat]).astype(np.int64) + 1
        c_te = enc.transform(Xte[:, cat]).astype(np.int64) + 1
        card = [int(nun[j]) + 2 for j in cat]
    else:
        c_tr, c_va, c_te = (np.zeros((len(X), 0), np.int64) for X in (Xtr, Xva, Xte))
        card = []
    sc = StandardScaler().fit(Xtr[:, cont])
    k_tr, k_va, k_te = (sc.transform(X[:, cont]).astype(np.float32) for X in (Xtr, Xva, Xte))
    return (c_tr, k_tr), (c_va, k_va), (c_te, k_te), card, len(cont)


def loader(tensors, bs, shuffle, drop_last=False):
    return DataLoader(TensorDataset(*[torch.as_tensor(t) for t in tensors]), batch_size=bs,
                      shuffle=shuffle, drop_last=drop_last)


# ----------------------------------------------------------------------
# training loop shared by both architectures (best state deep-copied)
# ----------------------------------------------------------------------
def train(model, fwd, tr_loader, va_loader, y_va, pos_weight, epochs, device, scheduler=False,
          faithful=False, tag=""):
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3) if scheduler else None
    best_auc, best_state, hist = 0.0, None, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for *xs, y in tr_loader:
            xs = [x.to(device) for x in xs]
            y = y.to(device).float().view(-1, 1)
            opt.zero_grad()
            loss = crit(fwd(model, xs).view(-1, 1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        p = predict(model, fwd, va_loader, device)
        auc, ap = roc_auc_score(y_va, p), average_precision_score(y_va, p)
        if sch is not None:
            sch.step(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict() if faithful else copy.deepcopy(model.state_dict())
        hist.append({"epoch": ep, "train_loss": tot / len(tr_loader.dataset), "val_auroc": auc,
                     "val_auprc": ap, "seconds": time.time() - t0})
        if ep % 10 == 0 or ep == epochs or ep == 1:
            print(f"  [{tag}] epoch {ep:3d}/{epochs} | loss {hist[-1]['train_loss']:.4f} | "
                  f"val AUROC {auc:.4f} AUPRC {ap:.4f} | {hist[-1]['seconds']:.1f}s", flush=True)
    model.load_state_dict(best_state)
    best_ep = int(np.argmax([h["val_auroc"] for h in hist])) + 1
    print(f"  [{tag}] best validation AUROC {best_auc:.4f} at epoch {best_ep}", flush=True)
    return model, hist, best_ep


@torch.no_grad()
def predict(model, fwd, dl, device):
    model.eval()
    out = []
    for *xs, _ in dl:
        out.append(torch.sigmoid(fwd(model, [x.to(device) for x in xs])).view(-1).cpu().numpy())
    return np.concatenate(out)


def fwd_tab(model, xs):
    return model(xs[0], xs[1])


def fwd_flat(model, xs):
    return model(xs[0])


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["transformer", "mamba", "stacked", "all"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--esrd", default=None, choices=["cr4_z49", "cr2_z49", "cr4_z49_n18"])
    ap.add_argument("--features", default="preop", choices=["preop", "preop2", "eos", "eos2"])
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--faithful", action="store_true",
                    help="keep the original state_dict() reference behaviour (last epoch, not best)")
    ap.add_argument("--tf-heads", type=int, default=8, help="stand-alone Transformer heads")
    ap.add_argument("--stack-heads", type=int, default=8, help="Transformer heads in the stacked model")
    ap.add_argument("--reuse-mamba", action="store_true",
                    help="stacked: load cached Mamba base predictions instead of retraining Mamba")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    ap.add_argument("--val-outdir", default=None,
                    help="also save the validation predictions of every model there (calibration sets for "
                         "calib_oof.py; the stacked meta-learners are fitted on the same validation set)")
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tf = load_module("dl_transformer", os.path.join("tabtransformer", "transformer.py"))
    mb = load_module("dl_mamba", os.path.join("mamba", "mamba.py"))

    Xtr_all, Xte, ytr_all, yte, cols, tag = load_split(dummy=a.dummy, verbose=False,
                                                       esrd=a.esrd, features=a.features)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=a.val_frac,
                                          stratify=ytr_all, random_state=a.seed)
    sfx = "" if tag == "all" else f"__{tag}"
    neg_pos = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"device {device} | train {Xtr.shape} val {Xva.shape} test {Xte.shape} | "
          f"pos {ytr.mean():.4f} | cohort {tag} | n_neg/n_pos {neg_pos:.1f}"
          + (" | FAITHFUL (last-epoch) mode" if a.faithful else ""), flush=True)

    # flat standardised inputs (Mamba) -- load_split already scales on the training part
    f32 = lambda X: np.ascontiguousarray(X, dtype=np.float32)
    flat = {k: loader((f32(X), y.astype(np.float32)), 256, k == "tr", drop_last=(k == "tr" and len(y) % 256 == 1))
            for k, X, y in (("tr", Xtr, ytr), ("va", Xva, yva), ("te", Xte, yte))}
    (ctr, ktr), (cva, kva), (cte, kte), card, ncont = transformer_inputs(Xtr, Xva, Xte)
    print(f"Transformer input: {len(card)} categorical, {ncont} continuous features", flush=True)
    tab = {k: loader((c, x, y.astype(np.float32)), 256 if k == "tr" else 512, k == "tr", drop_last=(k == "tr" and len(y) % 256 == 1))
           for k, c, x, y in (("tr", ctr, ktr, ytr), ("va", cva, kva, yva), ("te", cte, kte, yte))}

    results = {}

    def finish(name, p_te, hist, extra, p_va=None):
        full = f"dl_{name}_e{a.epochs}_s{a.seed}" + ("_faithful" if a.faithful else "") + sfx
        if a.val_outdir and p_va is not None:
            save_predictions(f"{a.val_outdir}/{full}.npz", full, yva, p_va, model=name, split="validation",
                             seed=a.seed, cohort=tag)
        au, ap_ = roc_auc_score(yte, p_te), average_precision_score(yte, p_te)
        print(f"RESULT {full}: test AUROC {au:.4f} | AUPRC {ap_:.4f}", flush=True)
        save_predictions(f"{a.outdir}/{full}.npz", full, yte, p_te, model=name, epochs=a.epochs,
                         seed=a.seed, cohort=tag, split_seed=42, **extra)
        with open(f"{a.outdir}/{full}.json", "w") as fh:
            json.dump({"auroc": au, "auprc": ap_, "history": hist, **extra}, fh, indent=1)
        results[name] = (au, ap_)

    want = {"transformer", "mamba", "stacked"} if a.model == "all" else {a.model}
    mamba_model = mamba_hist = None

    cache = f"dl_cache/mamba_e{a.epochs}_s{a.seed}{sfx}.npz"
    tf_name = "transformer" if a.tf_heads == 8 else f"transformer_h{a.tf_heads}"
    if "transformer" in want:
        seed_all(a.seed)
        t0 = time.time()
        m = tf.TabTransformer(categories=card, num_continuous=ncont, dim=32, depth=3, heads=a.tf_heads,
                              attn_dropout=0.1, ff_dropout=0.1, mlp_hidden_dims=(128, 64)).to(device)
        m, hist, be = train(m, fwd_tab, tab["tr"], tab["va"], yva, 20.0, a.epochs, device,
                            scheduler=True, faithful=a.faithful, tag="transformer")
        finish(tf_name, predict(m, fwd_tab, tab["te"], device), hist,
               {"best_epoch": be, "train_minutes": (time.time() - t0) / 60},
               p_va=predict(m, fwd_tab, tab["va"], device))

    reuse = a.reuse_mamba and "mamba" not in want and os.path.exists(cache)
    if a.reuse_mamba and not reuse:
        print(f"no Mamba cache at {cache} (or --model includes mamba): training Mamba", flush=True)
    if want & {"mamba", "stacked"} and not reuse:
        seed_all(a.seed)
        t0 = time.time()
        mamba_model = mb.TabularMambaClassifier(num_features=Xtr.shape[1], d_model=64, d_state=16, depth=2).to(device)
        mamba_model, mamba_hist, be = train(mamba_model, fwd_flat, flat["tr"], flat["va"], yva, neg_pos,
                                            a.epochs, device, faithful=False, tag="mamba")
        mamba_min = (time.time() - t0) / 60
        pm_va = predict(mamba_model, fwd_flat, flat["va"], device)
        pm_te = predict(mamba_model, fwd_flat, flat["te"], device)
        os.makedirs("dl_cache", exist_ok=True)
        np.savez_compressed(cache, p_va=pm_va, p_te=pm_te, minutes=mamba_min, best_epoch=be)
        if "mamba" in want:
            finish("mamba", pm_te, mamba_hist, {"best_epoch": be, "train_minutes": mamba_min}, p_va=pm_va)
    elif reuse:
        c = np.load(cache)
        pm_va, pm_te, mamba_min = c["p_va"], c["p_te"], float(c["minutes"])
        print(f"re-using cached Mamba predictions {cache}", flush=True)

    if "stacked" in want:
        seed_all(a.seed + 1)
        t0 = time.time()
        st = tf.TabTransformer(categories=card, num_continuous=ncont, dim=32, depth=3,
                               heads=a.stack_heads).to(device)
        st, hist_t, be_t = train(st, fwd_tab, tab["tr"], tab["va"], yva, neg_pos, a.epochs, device,
                                 scheduler=False, faithful=a.faithful, tag="stacked-transformer")
        Z_va = np.column_stack([predict(st, fwd_tab, tab["va"], device), pm_va])
        Z_te = np.column_stack([predict(st, fwd_tab, tab["te"], device), pm_te])
        extra = {"best_epoch_transformer": be_t, "train_minutes": (time.time() - t0) / 60 + mamba_min,
                 "base_test_auroc": [roc_auc_score(yte, Z_te[:, 0]), roc_auc_score(yte, Z_te[:, 1])],
                 "base_test_auprc": [average_precision_score(yte, Z_te[:, 0]),
                                     average_precision_score(yte, Z_te[:, 1])]}
        # meta-learners on the same base predictions:
        #   stacked         calibrated SVC (RBF, unweighted), as in stacked-model-svm.py
        #   stacked_svcbal  same SVC with class_weight='balanced'
        #   stacked_lr      logistic regression (the stacked model of the article)
        # (an unweighted SVC on 2 probability features with 5% positives can collapse
        #  to an almost constant decision function: test AUROC 0.58 in the 1-epoch smoke test)
        metas = {"stacked": CalibratedClassifierCV(SVC(), ensemble=False),
                 "stacked_svcbal": CalibratedClassifierCV(SVC(class_weight="balanced"), ensemble=False),
                 "stacked_lr": LogisticRegression()}
        for nm, meta in metas.items():
            meta.fit(Z_va, yva)
            finish(f"{nm}_h{a.stack_heads}", meta.predict_proba(Z_te)[:, 1], hist_t,
                   {**extra, "stack_heads": a.stack_heads}, p_va=meta.predict_proba(Z_va)[:, 1])

    print("\nSummary (test set):")
    for k, (au, ap_) in results.items():
        print(f"  {k:19s} AUROC {au:.4f}  AUPRC {ap_:.4f}")


if __name__ == "__main__":
    main()
