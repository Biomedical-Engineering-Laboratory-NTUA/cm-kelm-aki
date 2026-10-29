#!/usr/bin/env python
"""Interpretability of the confirmatory protocol (Section 9): permutation importance
(M1 DE-KELM-ARD, XGBoost, random forest) on every outer fold and SHAP on one fold
(KernelSHAP for M1, TreeSHAP for the tree ensembles), plus the aggregation with the
ARD and GP relevance (Spearman, top-10 overlap, Kendall's W).

The models are rebuilt from what the CV runs saved: every M1 member from (gamma,
lambda, w) in the population-ensemble meta, refitted exactly as in
population_ensemble.py; the trees with run_baselines_preds.classical_models().
Before any importance is computed the rebuilt predictions are compared with the saved
out-of-fold predictions.

Per fold (one Slurm array task each):
  python cv_interp.py --fold 11 [--shap]
Aggregate (after all folds):
  python cv_interp.py --aggregate --out cv_interp_aki
Only aggregate importances are written (no patient-level values).
"""
import argparse, glob, json, os, sys, time
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

PRED = "preds_cv"
OUTDIR = "interp_cv"
M1_GLOB = "cmkelm_m1000*_nocult_nomem_ard_pen0_p20_g23_s*__{tag}__popens_topk5.npz"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- models
class M1Ensemble:
    """Seed ensemble of the DE-KELM-ARD population ensembles, rebuilt from the meta."""

    def __init__(self, tag):
        from nystrom_kelm import NystromKELM
        self.NK = NystromKELM
        self.files = sorted(glob.glob(os.path.join(PRED, M1_GLOB.format(tag=tag))))
        if not self.files:
            sys.exit(f"no M1 files for {tag}")
        self.seeds = []
        for f in self.files:
            z = np.load(f, allow_pickle=True)
            meta = json.loads(str(z["meta"]))
            mem = meta["members"]
            if not all("w" in m for m in mem):
                sys.exit(f"{f}: members without 'w' (rerun with the patched population_ensemble.py)")
            self.seeds.append(dict(members=mem, p_saved=z["p"].astype(float),
                                   y=z["y_true"].astype(int)))
        self.p_saved = np.mean([s["p_saved"] for s in self.seeds], axis=0)
        self.y_saved = self.seeds[0]["y"]

    def fit(self, Xtr, ytr):
        self.models = []
        for s in self.seeds:
            ms = []
            for m in s["members"]:
                w = np.asarray(m["w"], float)
                k = self.NK(gamma=m["gamma"], lam=m["lam"], n_landmarks=1000, random_state=0,
                            rank=None, eig_rtol=None)
                ms.append((k.fit(Xtr * w, ytr), w))
            self.models.append(ms)
        return self

    def predict(self, X):
        return np.mean([np.mean([k.predict_proba(X * w) for k, w in ms], axis=0)
                        for ms in self.models], axis=0)

    @property
    def weights(self):
        """Mean normalised relevance gamma * w_j^2 over members and seeds."""
        v = []
        for s in self.seeds:
            for m in s["members"]:
                r = m["gamma"] * np.asarray(m["w"], float) ** 2
                v.append(r / r.sum())
        return np.mean(v, axis=0)


def tree_model(name):
    from run_baselines_preds import classical_models
    d = dict(classical_models())
    if name not in d:
        sys.exit(f"{name} not in classical_models(): {list(d)}")
    return d[name]


def proba(model, X):
    return model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- importance
def perm_importance(predict, X, y, n_perm, seed):
    base_p = predict(X)
    base = (average_precision_score(y, base_p), roc_auc_score(y, base_p))
    d = X.shape[1]
    imp = np.zeros((d, 2))
    for j in range(d):
        acc = []
        for r in range(n_perm):
            rng = np.random.default_rng(seed + 1000 * r + j)
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]
            p = predict(Xp)
            acc.append((base[0] - average_precision_score(y, p), base[1] - roc_auc_score(y, p)))
        imp[j] = np.mean(acc, axis=0)
    return base, imp


# ----------------------------------------------------------------------------- per fold
def run_fold(a):
    import outer_cv
    r, k = int(a.fold[0]), int(a.fold[1])
    Xtr, Xte, ytr, yte, names, tag = outer_cv.load_fold(r, k, "aki", verbose=False)
    log(f"fold {r}{k}: train {Xtr.shape}, test {Xte.shape}, tag {tag}")
    out = dict(fold=f"{r}{k}", tag=tag, names=list(map(str, names)), n_test=int(len(yte)),
               events=int(yte.sum()), checks={}, perm={}, shap={})

    # rebuild the models and check them against the saved predictions
    models = {}
    m1 = M1Ensemble(tag)
    if not np.array_equal(m1.y_saved, yte):
        sys.exit("label order differs from load_fold; stop")
    t = time.time(); m1.fit(Xtr, ytr)
    p = m1.predict(Xte)
    out["checks"]["M1"] = dict(max_abs_dp=float(np.abs(p - m1.p_saved).max()),
                               auprc_rebuilt=float(average_precision_score(yte, p)),
                               auprc_saved=float(average_precision_score(yte, m1.p_saved)),
                               n_members=int(sum(len(s["members"]) for s in m1.seeds)),
                               fit_s=round(time.time() - t, 1))
    out["ard_relevance"] = m1.weights.tolist()
    models["M1"] = m1.predict
    log("M1 check", out["checks"]["M1"])
    for name in ("xgboost", "random_forest"):
        f = os.path.join(PRED, f"{name}__{tag}.npz")
        mdl = tree_model(name)
        t = time.time(); mdl.fit(Xtr, ytr)
        p = proba(mdl, Xte)
        ps = np.load(f)["p"] if os.path.exists(f) else np.full(len(p), np.nan)
        out["checks"][name] = dict(max_abs_dp=float(np.nanmax(np.abs(p - ps))),
                                   auprc_rebuilt=float(average_precision_score(yte, p)),
                                   auprc_saved=float(average_precision_score(yte, ps)) if np.isfinite(ps).all() else None,
                                   fit_s=round(time.time() - t, 1))
        models[name] = (lambda m: (lambda X: proba(m, X)))(mdl)
        models[name + "_obj"] = mdl
        log(name, "check", out["checks"][name])
    for key, c in out["checks"].items():
        if c["max_abs_dp"] > a.tol:
            log(f"WARNING: {key} rebuilt predictions differ from the saved ones "
                f"(max |dp| {c['max_abs_dp']:.3g}); importances describe the rebuilt model")

    # permutation importance
    if not a.no_perm:
        for name in ("M1", "xgboost", "random_forest"):
            t = time.time()
            base, imp = perm_importance(models[name], Xte, yte, a.n_perm, seed=a.seed)
            out["perm"][name] = dict(base_auprc=base[0], base_auroc=base[1],
                                     drop_auprc=imp[:, 0].tolist(), drop_auroc=imp[:, 1].tolist(),
                                     seconds=round(time.time() - t, 1))
            log(f"permutation {name}: {time.time() - t:.0f} s")

    # SHAP on the selected fold
    if a.shap:
        import shap
        rng = np.random.default_rng(a.seed)
        pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
        n2 = a.shap_n // 2
        idx = np.concatenate([rng.choice(pos, min(n2, len(pos)), replace=False),
                              rng.choice(neg, n2, replace=False)])
        Xs = Xte[idx]
        bg = shap.kmeans(Xtr, a.shap_bg)
        t = time.time()
        ex = shap.KernelExplainer(models["M1"], bg)
        sv = ex.shap_values(Xs, nsamples=a.shap_nsamples, silent=True)
        sv = np.asarray(sv)
        out["shap"]["M1"] = dict(method="KernelSHAP", n_explained=int(len(idx)),
                                 background=a.shap_bg, nsamples=a.shap_nsamples,
                                 mean_abs=np.abs(sv).mean(0).tolist(),
                                 seconds=round(time.time() - t, 1))
        log(f"KernelSHAP M1: {time.time() - t:.0f} s")
        for name in ("xgboost", "random_forest"):
            t = time.time()
            te = shap.TreeExplainer(models[name + "_obj"])
            def tree_sv(X):
                v = te.shap_values(X)
                v = np.asarray(v)
                if v.ndim == 3:          # (n, d, classes) or (classes, n, d)
                    v = v[..., 1] if v.shape[-1] == 2 else v[1]
                return v
            sub = Xte if name == "xgboost" else Xte[rng.choice(len(Xte), min(2000, len(Xte)), replace=False)]
            out["shap"][name] = dict(method="TreeSHAP", n_explained=int(len(sub)),
                                     mean_abs=np.abs(tree_sv(sub)).mean(0).tolist(),
                                     mean_abs_same_sample=np.abs(tree_sv(Xs)).mean(0).tolist(),
                                     seconds=round(time.time() - t, 1))
            log(f"TreeSHAP {name}: {time.time() - t:.0f} s")

    os.makedirs(OUTDIR, exist_ok=True)
    fn = os.path.join(OUTDIR, f"interp_{tag}.json")
    json.dump(out, open(fn, "w"))
    log("written", fn)


# ----------------------------------------------------------------------------- aggregate
def kendall_w(R):
    m, n = R.shape
    ranks = np.vstack([stats.rankdata(-r) for r in R])
    S = np.sum((ranks.sum(0) - m * (n + 1) / 2) ** 2)
    T = sum(np.sum(c ** 3 - c) for c in (np.unique(r, return_counts=True)[1] for r in ranks))
    W = 12 * S / (m ** 2 * (n ** 3 - n) - m * T)
    return W, stats.chi2.sf(m * (n - 1) * W, n - 1)


def top_overlap(a, b, k=10):
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k]))


def gp_relevance(tag, names):
    f = os.path.join(PRED, f"gp_ard_relevance__{tag}.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    rel = dict(zip(map(str, j["variables"]), np.asarray(j["relevance_normalised_mean"], float)))
    v = np.array([rel.get(n, 0.0) for n in names])
    return v / v.sum()


def aggregate(a):
    files = sorted(glob.glob(os.path.join(OUTDIR, "interp_esrd_eos2_cv[12][1-5].json")))
    if not files:
        sys.exit("no per-fold results")
    F = [json.load(open(f)) for f in files]
    names = F[0]["names"]; d = len(names)
    L = [f"# Interpretability (protocol Section 9): primary outcome, {len(F)} outer folds\n"]
    L.append("Models rebuilt from the saved hyperparameters and checked against the saved "
             "out-of-fold predictions:\n")
    L.append("| fold | M1 max abs dp | M1 AUPRC rebuilt / saved | XGB max abs dp | RF max abs dp |")
    L.append("|---|---|---|---|---|")
    for f in F:
        c = f["checks"]
        L.append(f"| {f['fold']} | {c['M1']['max_abs_dp']:.2g} | {c['M1']['auprc_rebuilt']:.4f} / "
                 f"{c['M1']['auprc_saved']:.4f} | {c['xgboost']['max_abs_dp']:.2g} | "
                 f"{c['random_forest']['max_abs_dp']:.2g} |")
    L.append("")

    # matrices folds x features per method
    M = {}
    M["ARD relevance (M1, gamma w^2)"] = np.array([f["ard_relevance"] for f in F])
    gp = [gp_relevance(f["tag"], names) for f in F]
    if all(g is not None for g in gp):
        M["GP-ARD relevance (M5)"] = np.array(gp)
    for name, lab in (("M1", "Permutation, M1"), ("xgboost", "Permutation, XGBoost"),
                      ("random_forest", "Permutation, RF")):
        if all(name in f["perm"] for f in F):
            M[lab] = np.array([f["perm"][name]["drop_auprc"] for f in F])

    L.append("## Stability across outer folds (Kendall's W over the folds)\n")
    L.append("| Method | folds | Kendall's W | p |\n|---|---|---|---|")
    for lab, R in M.items():
        W, p = kendall_w(R)
        L.append(f"| {lab} | {len(R)} | {W:.3f} | {p:.2g} |")
    L.append("")

    mean = {lab: R.mean(0) for lab, R in M.items()}
    shapf = [f for f in F if f["shap"]]
    if shapf:
        s = shapf[0]
        mean[f"KernelSHAP, M1 (fold {s['fold']})"] = np.array(s["shap"]["M1"]["mean_abs"])
        mean[f"TreeSHAP, XGBoost (fold {s['fold']})"] = np.array(s["shap"]["xgboost"]["mean_abs"])
        mean[f"TreeSHAP, RF (fold {s['fold']})"] = np.array(s["shap"]["random_forest"]["mean_abs"])
        if "M5" in s["shap"]:
            mean[f"KernelSHAP, GP (fold {s['fold']})"] = np.array(s["shap"]["M5"]["mean_abs"])

    labs = list(mean)
    L.append("## Agreement between methods (mean importance over folds; Spearman rho / "
             "top-10 overlap)\n")
    L.append("| | " + " | ".join(f"({i+1})" for i in range(len(labs))) + " |")
    L.append("|---|" + "---|" * len(labs))
    for i, a_ in enumerate(labs):
        cells = []
        for j, b_ in enumerate(labs):
            if i == j:
                cells.append("-")
            else:
                rho = stats.spearmanr(mean[a_], mean[b_]).correlation
                cells.append(f"{rho:.2f} / {top_overlap(mean[a_], mean[b_])}")
        L.append(f"| ({i+1}) {a_} | " + " | ".join(cells) + " |")
    Wm, pm = kendall_w(np.vstack([mean[l] for l in labs]))
    L.append(f"\nKendall's W across the {len(labs)} methods: {Wm:.3f} (p = {pm:.2g}).\n")

    # consensus ranking table
    ranks = np.vstack([stats.rankdata(-mean[l]) for l in labs])
    order = np.argsort(ranks.mean(0))[:25]
    L.append("## Top 25 variables by mean rank across methods\n")
    L.append("| variable | mean rank | " + " | ".join(f"({i+1})" for i in range(len(labs))) + " |")
    L.append("|---|---|" + "---|" * len(labs))
    for j in order:
        L.append(f"| {names[j]} | {ranks[:, j].mean():.1f} | " +
                 " | ".join(f"{int(ranks[i, j])}" for i in range(len(labs))) + " |")
    L.append("\nCells give the rank of the variable under each method (1 = most important).\n")

    if not a.out:
        a.out = "cv_interp_aki"
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    import csv
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variable"] + labs)
        for j in range(d):
            w.writerow([names[j]] + [f"{mean[l][j]:.6g}" for l in labs])
    _plot(mean, names, labs, a.out)
    log("written", a.out + ".md")


def _plot(mean, names, labs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranks = np.vstack([stats.rankdata(-mean[l]) for l in labs])
    order = np.argsort(ranks.mean(0))[:20]
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(labs), 6.5))
    im = ax.imshow(ranks[:, order].T, cmap="viridis_r", vmin=1, vmax=40, aspect="auto")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[j] for j in order], fontsize=8)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([f"({i+1})" for i in range(len(labs))], fontsize=8)
    for i in range(len(labs)):
        for jj, j in enumerate(order):
            ax.text(i, jj, int(ranks[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if ranks[i, j] < 20 else "black")
    fig.colorbar(im, ax=ax, label="rank (1 = most important)", shrink=0.6)
    ax.set_title("Variable importance rank by method", fontsize=9)
    fig.tight_layout()
    fig.savefig(out + "_ranks.png", dpi=200); fig.savefig(out + "_ranks.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", help="repeat and fold, e.g. 11")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--shap-n", type=int, default=100)
    ap.add_argument("--shap-bg", type=int, default=10)
    ap.add_argument("--shap-nsamples", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a)
    elif a.fold:
        run_fold(a)
    else:
        ap.error("--fold or --aggregate")


if __name__ == "__main__":
    main()
