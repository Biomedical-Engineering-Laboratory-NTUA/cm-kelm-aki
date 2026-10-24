"""
Standard tabular deep-learning baselines: MLP and FT-Transformer (review point: baselines).
==========================================================================================

Gorishniy et al., "Revisiting deep learning models for tabular data" (NeurIPS 2021), with the
reference implementation of the authors (pip install rtdl_revisiting_models):
  mlp   3 blocks of 256 units, ReLU, dropout 0.1, AdamW (learning rate 1e-3, weight decay 1e-4)
  ftt   FT-Transformer with the default configuration of the paper (3 blocks, width 192, 8 heads,
        attention dropout 0.2, FFN dropout 0.1), AdamW (1e-4, 1e-5); variables with at most 10
        distinct values enter as categorical tokens, as for the TabTransformer
Same split as run_dl_baselines.py (85% of the training set for training, 15% for validation,
random_state 42; the common test set), class-weighted cross-entropy (positive weight =
negative-to-positive ratio), batches of 256, at most --epochs epochs with early stopping
(patience 10) on the validation AUROC, best epoch kept. Five seeds (42-46) vary only the
initialisation and the batch order; their probabilities are averaged (name *_ens5).

    python -u run_tabdl.py --model ftt --esrd cr4_z49 --features eos2
    -> preds/dl_ftt_s42__esrd_eos2.npz ... preds/dl_ftt_ens5__esrd_eos2.npz, preds_val/ (validation)
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from cohort_split import load_split
from run_dl_baselines import loader, seed_all, transformer_inputs
from stats_eval import save_predictions


def build(kind, n_features, card, n_cont):
    import rtdl_revisiting_models as rtdl
    if kind == "mlp":
        return rtdl.MLP(d_in=n_features, d_out=1, n_blocks=3, d_block=256, dropout=0.1)
    return rtdl.FTTransformer(n_cont_features=n_cont, cat_cardinalities=card, d_out=1,
                              **rtdl.FTTransformer.get_default_kwargs())


def make_opt(kind, model):
    if kind == "mlp":
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model.make_default_optimizer()   # AdamW 1e-4 / 1e-5, no decay on embeddings and norms


def forward(kind, model, xs):
    if kind == "mlp":
        return model(xs[0])
    x_cat = xs[0] if xs[0].shape[1] else None
    return model(xs[1], x_cat)


@torch.no_grad()
def predict(kind, model, dl, device):
    model.eval()
    out = [torch.sigmoid(forward(kind, model, [x.to(device) for x in xs])).view(-1).cpu().numpy()
           for *xs, _ in dl]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "ftt"], required=True)
    ap.add_argument("--seeds", default="42-46")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--esrd", default=None, choices=["cr4_z49", "cr2_z49", "cr4_z49_n18"])
    ap.add_argument("--features", default="eos2", choices=["preop", "preop2", "eos", "eos2"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    ap.add_argument("--val-outdir", default="preds_val")
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Xtr_all, Xte, ytr_all, yte, cols, tag = load_split(dummy=a.dummy, verbose=False, esrd=a.esrd,
                                                       features="preop" if a.dummy else a.features)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.15, stratify=ytr_all, random_state=42)
    sfx = "" if tag == "all" else f"__{tag}"
    neg_pos = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    f32 = lambda X: np.ascontiguousarray(X, dtype=np.float32)
    if a.model == "mlp":
        data = {k: (f32(X), y.astype(np.float32)) for k, X, y in (("tr", Xtr, ytr), ("va", Xva, yva), ("te", Xte, yte))}
        card, n_cont = [], Xtr.shape[1]
    else:
        (ctr, ktr), (cva, kva), (cte, kte), card, n_cont = transformer_inputs(Xtr, Xva, Xte)
        data = {"tr": (ctr, ktr, ytr.astype(np.float32)), "va": (cva, kva, yva.astype(np.float32)),
                "te": (cte, kte, yte.astype(np.float32))}
    print(f"{a.model}: device {device} | train {Xtr.shape} val {Xva.shape} test {Xte.shape} | "
          f"{len(card)} categorical, {n_cont} continuous | n_neg/n_pos {neg_pos:.1f}", flush=True)

    lo, hi = (int(v) for v in a.seeds.split("-"))
    names = []
    for seed in range(lo, hi + 1):
        seed_all(seed)
        t0 = time.time()
        tr = loader(data["tr"], 256, True, drop_last=len(ytr) % 256 == 1)
        va, te = loader(data["va"], 1024, False), loader(data["te"], 1024, False)
        model = build(a.model, Xtr.shape[1], card, n_cont).to(device)
        opt = make_opt(a.model, model)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_pos], device=device))
        best, best_state, best_ep, wait = -1.0, None, 0, 0
        for ep in range(1, a.epochs + 1):
            model.train()
            for *xs, y in tr:
                xs = [x.to(device) for x in xs]
                opt.zero_grad()
                loss = crit(forward(a.model, model, xs).view(-1), y.to(device))
                loss.backward()
                opt.step()
            auc = roc_auc_score(yva, predict(a.model, model, va, device))
            if auc > best:
                best, best_state, best_ep, wait = auc, copy.deepcopy(model.state_dict()), ep, 0
            else:
                wait += 1
            if ep == 1 or ep % 10 == 0:
                print(f"  [{a.model} s{seed}] epoch {ep:3d} val AUROC {auc:.4f} (best {best:.4f} @ {best_ep})", flush=True)
            if wait >= a.patience:
                break
        model.load_state_dict(best_state)
        p_te, p_va = predict(a.model, model, te, device), predict(a.model, model, va, device)
        name = f"dl_{a.model}_s{seed}{sfx}"
        print(f"RESULT {name}: test AUROC {roc_auc_score(yte, p_te):.4f} AUPRC {average_precision_score(yte, p_te):.4f} | "
              f"val AUROC {best:.4f} (epoch {best_ep}) | {(time.time() - t0) / 60:.1f} min", flush=True)
        save_predictions(f"{a.outdir}/{name}.npz", name, yte, p_te, model=a.model, seed=seed, best_epoch=best_ep,
                         cohort=tag, split_seed=42)
        save_predictions(f"{a.val_outdir}/{name}.npz", name, yva, p_va, model=a.model, seed=seed,
                         split="validation", cohort=tag)
        names.append(name)
    if len(names) > 1:
        from stats_eval import load_predictions
        for d in (a.outdir, a.val_outdir):
            R = load_predictions([f"{d}/{n}.npz" for n in names])
            y = R[names[0]]["y"]
            p = np.mean([R[n]["p"] for n in names], axis=0)
            ens = f"dl_{a.model}_ens{len(names)}{sfx}"
            save_predictions(f"{d}/{ens}.npz", ens, y, p, runs=names, combine="mean probability")
            if d == a.outdir:
                print(f"RESULT {ens}: test AUROC {roc_auc_score(y, p):.4f} AUPRC {average_precision_score(y, p):.4f}",
                      flush=True)


if __name__ == "__main__":
    main()
