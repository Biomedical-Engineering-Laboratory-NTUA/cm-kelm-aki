#!/usr/bin/env python
"""KernelSHAP for the sparse GP with ARD (M5) on outer fold 11 (protocol Section 9).

Uses the GP models saved by run_gp_ard.py --save-model (same seeds 42-46, same code),
checks them against the saved out-of-fold predictions, and explains the same 100 test
patients with the same background and number of samples as cv_interp.py for M1.
The result is added to interp_cv/interp_esrd_eos2_cv11.json under shap["M5"].

  python gp_shap.py --models gp_models --fold 11
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from sklearn.metrics import average_precision_score


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gp_models")
    ap.add_argument("--fold", default="11")
    ap.add_argument("--shap-n", type=int, default=100)
    ap.add_argument("--shap-bg", type=int, default=10)
    ap.add_argument("--shap-nsamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)      # same as cv_interp.py
    a = ap.parse_args()

    import gpytorch, shap, outer_cv
    from run_gp_ard import make_model, predict
    r, k = int(a.fold[0]), int(a.fold[1])
    Xtr, Xte, ytr, yte, names, tag = outer_cv.load_fold(r, k, "aki", verbose=False)
    f32 = lambda X: np.ascontiguousarray(X, dtype=np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Xte.shape[1]

    files = sorted(glob.glob(os.path.join(a.models, f"gp_ard_s*__{tag}.pt")))
    if len(files) != 5:
        raise SystemExit(f"expected 5 saved GP models for {tag}, found {len(files)}")
    models, checks = [], {}
    for f in files:
        st = torch.load(f, map_location=dev)
        n_ind = st["model"]["variational_strategy.inducing_points"].shape[0]
        m = make_model(torch.zeros(n_ind, d), d).to(dev)
        m.load_state_dict(st["model"])
        lik = gpytorch.likelihoods.BernoulliLikelihood().to(dev)
        lik.load_state_dict(st["lik"])
        models.append((m, lik))
        name = os.path.basename(f)[:-3]
        p = predict(m, lik, f32(Xte), dev)
        ref = os.path.join("preds_cv", name + ".npz")
        if os.path.exists(ref):
            z = np.load(ref)
            checks[name] = dict(max_abs_dp_vs_reported=float(np.abs(p - z["p"]).max()),
                                auprc_rebuilt=float(average_precision_score(yte, p)),
                                auprc_reported=float(average_precision_score(z["y_true"], z["p"])))
        log(name, checks.get(name))

    def f_ens(X):
        X = f32(X)
        return np.mean([predict(m, l, X, dev) for m, l in models], axis=0)

    p = f_ens(Xte)
    zr = np.load(os.path.join("preds_cv", f"gp_ard_ens5__{tag}.npz"))
    checks["ensemble"] = dict(max_abs_dp_vs_reported=float(np.abs(p - zr["p"]).max()),
                              auprc_rebuilt=float(average_precision_score(yte, p)),
                              auprc_reported=float(average_precision_score(zr["y_true"], zr["p"])))
    log("ensemble", checks["ensemble"])

    # the same explained patients and background as cv_interp.py
    rng = np.random.default_rng(a.seed)
    pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    n2 = a.shap_n // 2
    idx = np.concatenate([rng.choice(pos, min(n2, len(pos)), replace=False),
                          rng.choice(neg, n2, replace=False)])
    bg = shap.kmeans(Xtr, a.shap_bg)
    t = time.time()
    sv = np.asarray(shap.KernelExplainer(f_ens, bg).shap_values(Xte[idx], nsamples=a.shap_nsamples, silent=True))
    log(f"KernelSHAP M5: {time.time() - t:.0f} s")

    fn = os.path.join("interp_cv", f"interp_{tag}.json")
    J = json.load(open(fn))
    J["shap"]["M5"] = dict(method="KernelSHAP", n_explained=int(len(idx)), background=a.shap_bg,
                           nsamples=a.shap_nsamples, mean_abs=np.abs(sv).mean(0).tolist(),
                           seconds=round(time.time() - t, 1), checks=checks,
                           note="GP refitted with run_gp_ard.py (seeds 42-46) and saved; "
                                "see checks for agreement with the reported predictions")
    json.dump(J, open(fn, "w"))
    log("added shap M5 to", fn)


if __name__ == "__main__":
    main()
