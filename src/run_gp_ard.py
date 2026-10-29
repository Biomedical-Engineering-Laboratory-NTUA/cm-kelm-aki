"""
Sparse variational Gaussian-process classifier with an ARD kernel (control: gradient-learned relevance).
=====================================================================================================

The ARD variant of CM-KELM searches continuous per-variable kernel weights with differential
evolution. This control learns the same kind of weights by gradient ascent on the evidence lower
bound (ELBO) of the marginal likelihood instead (Hensman et al., AISTATS 2015; GPyTorch):
  kernel   s * exp(-0.5 * sum_j (x_j - x'_j)^2 / l_j^2)  (one lengthscale l_j per variable)
  model    1000 inducing points (initialised on random training patients), Cholesky variational
           distribution, constant mean, Bernoulli likelihood with probit link
  training Adam (learning rate 0.01), batches of 1024, at most --epochs epochs, early stopping
           (patience 10) on the validation AUROC, best epoch kept
Same split as the deep-learning baselines (85% of the training set for training, 15% for
validation, random_state 42; the common test set). Five seeds (42-46) vary the inducing-point
initialisation and the batch order; their probabilities are averaged (name gp_ard_ens5).
The relevance 1/l_j of every variable is written to <outdir>/gp_ard_relevance<sfx>.json.

    python -u run_gp_ard.py --esrd cr4_z49 --features eos2
    -> preds/gp_ard_s42__esrd_eos2.npz ... preds/gp_ard_ens5__esrd_eos2.npz, preds_val/ (validation)
    python -u run_gp_ard.py --dummy --epochs 3 --inducing 50     # CPU self-test
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from cohort_split import load_split
from stats_eval import save_predictions


def make_model(Z, d):
    import gpytorch

    class SVGP(gpytorch.models.ApproximateGP):
        def __init__(self, Z):
            q = gpytorch.variational.CholeskyVariationalDistribution(Z.size(0))
            s = gpytorch.variational.VariationalStrategy(self, Z, q, learn_inducing_locations=True)
            super().__init__(s)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
            # standardised inputs in d dimensions: start from lengthscales of order sqrt(d)
            self.covar_module.base_kernel.lengthscale = torch.full((1, d), float(np.sqrt(d)))

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    return SVGP(Z)


@torch.no_grad()
def predict(model, lik, X, device, bs=2048):
    model.eval(); lik.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], device=device)
        out.append(lik(model(x)).mean.cpu().numpy())   # P(y=1) = Phi(mu / sqrt(1 + var))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42-46")
    ap.add_argument("--inducing", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--save-model", default=None, help="directory for the fitted models")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--esrd", default=None, choices=["cr4_z49", "cr2_z49", "cr4_z49_n18"])
    ap.add_argument("--features", default="eos2", choices=["preop", "preop2", "eos", "eos2"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    ap.add_argument("--val-outdir", default="preds_val")
    a = ap.parse_args()
    import gpytorch

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Xtr_all, Xte, ytr_all, yte, cols, tag = load_split(dummy=a.dummy, verbose=False, esrd=a.esrd,
                                                       features="preop" if a.dummy else a.features)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.15, stratify=ytr_all, random_state=42)
    sfx = "" if tag == "all" else f"__{tag}"
    f32 = lambda X: np.ascontiguousarray(X, dtype=np.float32)
    Xtr, Xva, Xte = f32(Xtr), f32(Xva), f32(Xte)
    d = Xtr.shape[1]
    n_ind = min(a.inducing, len(ytr))
    print(f"gp_ard: device {device} | train {Xtr.shape} val {Xva.shape} test {Xte.shape} | {n_ind} inducing points",
          flush=True)
    Xt, yt = torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr.astype(np.float32), device=device)

    lo, hi = (int(v) for v in a.seeds.split("-"))
    names, relevance = [], {}
    for seed in range(lo, hi + 1):
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        Z = torch.as_tensor(Xtr[np.random.default_rng(seed).choice(len(Xtr), n_ind, replace=False)])
        model = make_model(Z, d).to(device)
        lik = gpytorch.likelihoods.BernoulliLikelihood().to(device)
        mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(ytr))
        opt = torch.optim.Adam(list(model.parameters()) + list(lik.parameters()), lr=a.lr)
        best, best_state, best_ep, wait = -1.0, None, 0, 0
        for ep in range(1, a.epochs + 1):
            model.train(); lik.train()
            perm = torch.randperm(len(yt), device=device)
            tot = 0.0
            for i in range(0, len(perm), a.batch):
                b = perm[i:i + a.batch]
                opt.zero_grad()
                loss = -mll(model(Xt[b]), yt[b])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            auc = roc_auc_score(yva, predict(model, lik, Xva, device))
            if auc > best:
                best, best_ep, wait = auc, ep, 0
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(lik.state_dict()))
            else:
                wait += 1
            if ep == 1 or ep % 10 == 0:
                print(f"  [gp s{seed}] epoch {ep:3d} -ELBO/n {tot / len(yt):.4f} val AUROC {auc:.4f} "
                      f"(best {best:.4f} @ {best_ep})", flush=True)
            if wait >= a.patience:
                break
        model.load_state_dict(best_state[0]); lik.load_state_dict(best_state[1])
        if a.save_model:
            os.makedirs(a.save_model, exist_ok=True)
            torch.save({"model": best_state[0], "lik": best_state[1]}, f"{a.save_model}/gp_ard_s{seed}{sfx}.pt")
        p_te, p_va = predict(model, lik, Xte, device), predict(model, lik, Xva, device)
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        relevance[seed] = (1.0 / ls).tolist()
        name = f"gp_ard_s{seed}{sfx}"
        print(f"RESULT {name}: test AUROC {roc_auc_score(yte, p_te):.4f} AUPRC {average_precision_score(yte, p_te):.4f} | "
              f"val AUROC {best:.4f} (epoch {best_ep}) | {(time.time() - t0) / 60:.1f} min", flush=True)
        save_predictions(f"{a.outdir}/{name}.npz", name, yte, p_te, model="svgp_ard", seed=seed, best_epoch=best_ep,
                         n_inducing=n_ind, cohort=tag, split_seed=42)
        save_predictions(f"{a.val_outdir}/{name}.npz", name, yva, p_va, model="svgp_ard", seed=seed,
                         split="validation", cohort=tag)
        names.append(name)
    if len(names) > 1:
        from stats_eval import load_predictions
        for dd in (a.outdir, a.val_outdir):
            R = load_predictions([f"{dd}/{n}.npz" for n in names])
            y = R[names[0]]["y"]
            p = np.mean([R[n]["p"] for n in names], axis=0)
            ens = f"gp_ard_ens{len(names)}{sfx}"
            save_predictions(f"{dd}/{ens}.npz", ens, y, p, runs=names, combine="mean probability")
            if dd == a.outdir:
                print(f"RESULT {ens}: test AUROC {roc_auc_score(y, p):.4f} AUPRC {average_precision_score(y, p):.4f}",
                      flush=True)
    # relevance 1/l_j, normalised to a maximum of 1 per seed, then averaged over seeds
    R = np.array([np.array(v) / max(v) for v in relevance.values()])
    mean = R.mean(axis=0)
    order = np.argsort(-mean)
    os.makedirs(a.outdir, exist_ok=True)
    with open(f"{a.outdir}/gp_ard_relevance{sfx}.json", "w") as fh:
        json.dump({"variables": cols, "relevance_per_seed": {str(k): v for k, v in relevance.items()},
                   "relevance_normalised_mean": mean.tolist(),
                   "ranking": [[cols[i], float(mean[i])] for i in order],
                   "n_relevance_ge_0.1": int((mean >= 0.1).sum())}, fh, indent=2)
    print("top 15 variables by relevance: " + ", ".join(f"{cols[i]} {mean[i]:.2f}" for i in order[:15]), flush=True)
    print(f"variables with normalised relevance >= 0.1: {(mean >= 0.1).sum()}/{d}", flush=True)


if __name__ == "__main__":
    main()
