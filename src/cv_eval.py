"""
Statistics of the confirmatory protocol (Sections 6 and 7.1-7.2): outer cross-validation.
=========================================================================================

Reads the held-out predictions of preds_cv/ and writes aggregate tables only.
  - per model: AUPRC and AUROC per fold, mean with the corrected (Nadeau-Bengio) 95% CI, and on
    the pooled out-of-fold predictions of each repeat;
  - H1 (M1 > M4) and H2 (M1 > M2) on AUPRC: corrected resampled t-test over the 10 fold-level
    differences, one-sided, Holm over the two; Shapiro-Wilk and a normal quantile plot of the
    differences (diagnostic); paired bootstrap of the pooled predictions (sensitivity);
  - secondary: AUROC for H1-H2; M1 against M2-M8 (two-sided, Holm over the eight models M2-M8, both metrics);
    M1 against M5 with its CI.
Search models are averaged over their seeds within each fold (top-5 ensembles for plain DE,
island ensembles for CM-KELM and the random-search controls).

    python cv_eval.py --outcome aki --out cv_primary           # -> cv_primary.md, .csv, _qq.png
    python cv_eval.py --outcome st2 --out cv_st2
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from stats_eval import load_predictions

D = "preds_cv"
REPEATS, FOLDS = (1, 2), range(1, 6)
# model id -> (description, glob of one fold with {tag}; seed files are averaged)
MODELS = {
    "M1": ("DE-KELM-ARD", "cmkelm_m1000*_nocult_nomem_ard_pen0_p20_g23_s*__{tag}__popens_topk5.npz"),
    "M2": ("DE-KELM", "cmkelm_m1000*_nocult_nomem_p20_g23_s*__{tag}__popens_topk5.npz"),
    "M3": ("CM-KELM", "cmkelm_m1000*_p20_g15_s*__{tag}__popens_islands.npz"),
    "M4": ("Grid-tuned KELM", "nystrom_kelm_grid_m1000_wide__{tag}.npz"),
    "M5": ("Sparse GP (ARD)", "gp_ard_ens5__{tag}.npz"),
    "M6": ("XGBoost", "xgboost__{tag}.npz"),
    "M7": ("FT-Transformer", "dl_ftt_ens5__{tag}.npz"),
    "M8a": ("Logistic regression", "logistic_regression__{tag}.npz"),
    "M8b": ("Random forest", "random_forest__{tag}.npz"),
    "LGBM": ("LightGBM", "lightgbm__{tag}.npz"),
    "RS-ARD": ("Random search, ARD weights", "cmkelm_m1000*_randsearch_ard_n500_s*__{tag}__popens_islands.npz"),
    "RS-mask": ("Random search, masks", "cmkelm_m1000*_randsearch_n500_s*__{tag}__popens_islands.npz"),
    # exploratory ablation of M1 (wave 7): cultural-memetic search, top-5 ensembles as M1 and M2
    "CM-ARD": ("CM-KELM-ARD (top-5)", "cmkelm_m1000*_ard_pen0_p20_g15_s*__{tag}__popens_topk5.npz"),
    "M3-top5": ("CM-KELM (top-5)", "cmkelm_m1000*_p20_g15_s*__{tag}__popens_topk5.npz"),
}
EXCLUDE = {"M3": r"(nocult|nomem|ard|randsearch|_ad|_mm)", "M2": r"_ard_",
           "M3-top5": r"(nocult|nomem|ard|randsearch|_ad|_mm)", "CM-ARD": r"(nocult|nomem|randsearch|_ad|_mm)"}


COHORT = "esrd_eos2"   # set by --cohort (e.g. esrd-cr2_z49_eos2, esrd-cr4_z49_n18_eos2, esrd_preop2)


def fold_tag(r, k, outcome, bdays):
    return f"{COHORT}_cv{r}{k}" + ("" if outcome == "aki" else f"_{outcome}") + ("" if bdays == 365 else f"_b{bdays}")


def load_model(mid, outcome, bdays):
    """{(r, k): (y, p, n_files)} for the folds where the model exists."""
    out = {}
    for r in REPEATS:
        for k in FOLDS:
            pat = os.path.join(D, MODELS[mid][1].format(tag=fold_tag(r, k, outcome, bdays)))
            files = sorted(glob.glob(pat))
            if mid in EXCLUDE:
                files = [f for f in files if not re.search(EXCLUDE[mid], os.path.basename(f))]
            # the tag must end exactly here (no further suffix such as _st2 or _b90)
            files = [f for f in files if f"__{fold_tag(r, k, outcome, bdays)}." in f or
                     f"__{fold_tag(r, k, outcome, bdays)}__" in f]
            if not files:
                continue
            R = list(load_predictions(files).values())
            y = R[0]["y"]
            assert all(np.array_equal(x["y"], y) for x in R), f"{mid} fold {r}{k}: different test sets"
            out[(r, k)] = (y, np.mean([x["p"] for x in R], axis=0), len(files))
    return out


def metrics(y, p):
    return average_precision_score(y, p), roc_auc_score(y, p)


def nb_test(d, n_train, n_test, alternative="two-sided"):
    """Corrected resampled t-test (Nadeau and Bengio, 2003) over the J fold-level differences d."""
    d = np.asarray(d, float)
    J = len(d)
    m, s2 = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + n_test / n_train) * s2)
    t = m / se if se > 0 else np.inf * np.sign(m)
    if alternative == "greater":
        p = stats.t.sf(t, J - 1)
    else:
        p = 2 * stats.t.sf(abs(t), J - 1)
    h = stats.t.ppf(0.975, J - 1) * se
    return m, (m - h, m + h), t, p


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(run, 1.0)
    return adj


def boot_pooled(y, pa, pb, n_boot, rng):
    """Paired stratified bootstrap of AUPRC(a) - AUPRC(b) on pooled predictions; one-sided p (a > b)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[b] = average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i])
    obs = average_precision_score(y, pa) - average_precision_score(y, pb)
    return obs, np.percentile(diffs, [2.5, 97.5]), (1 + (diffs <= 0).sum()) / (n_boot + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="aki", choices=["aki", "st2", "st3"])
    ap.add_argument("--baseline-days", type=int, default=365)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cohort", default="esrd_eos2", help="cohort tag of the prediction files")
    ap.add_argument("--preds", default="preds_cv")
    ap.add_argument("--out", default="cv_primary")
    a = ap.parse_args()
    global D, COHORT
    D, COHORT = a.preds, a.cohort
    rng = np.random.default_rng(0)
    data = {m: load_model(m, a.outcome, a.baseline_days) for m in MODELS}
    data = {m: v for m, v in data.items() if v}
    L = [f"# Confirmatory analysis: outcome {a.outcome}, baseline window {a.baseline_days} days\n",
         "Outer cross-validation 5 folds x 2 repeats (protocol, Section 4). Aggregate results only.\n",
         "## Availability\n", "| Model | folds | files per fold |", "|---|---|---|"]
    for m, v in data.items():
        L.append(f"| {m} {MODELS[m][0]} | {len(v)}/10 | {min(x[2] for x in v.values())}-{max(x[2] for x in v.values())} |")

    # per-model metrics
    full = [m for m, v in data.items() if len(v) == 10]
    L += ["\n## Discrimination (mean of 10 folds with corrected 95% CI; pooled per repeat)\n",
          "| Model | AUPRC mean [CI] | AUROC mean [CI] | pooled AUPRC r1 / r2 | pooled AUROC r1 / r2 |", "|---|---|---|---|---|"]
    rows = []
    for m in data:
        v = data[m]
        keys = sorted(v)
        ap_, au = np.array([metrics(*v[k][:2]) for k in keys]).T
        ntr = np.mean([sum(len(v[x][0]) for x in keys if x[0] == k[0] and x != k) for k in keys])
        nte = np.mean([len(v[k][0]) for k in keys])
        line = [f"{m} {MODELS[m][0]}"]
        for vals in (ap_, au):
            if len(vals) >= 2:
                mm, ci, _, _ = nb_test(vals, ntr, nte)
                line.append(f"{mm:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                line.append(f"{vals.mean():.4f}")
        pooled = []
        for idx in (0, 1):
            pr = []
            for r in REPEATS:
                ks = [x for x in keys if x[0] == r]
                if len(ks) == 5:
                    y = np.concatenate([v[x][0] for x in ks]); p = np.concatenate([v[x][1] for x in ks])
                    pr.append(f"{metrics(y, p)[idx]:.4f}")
                else:
                    pr.append("-")
            pooled.append(" / ".join(pr))
        L.append("| " + " | ".join(line + pooled) + " |")
        for kk, (x1, x2) in zip(keys, zip(ap_, au)):
            rows.append(f"{m},{kk[0]},{kk[1]},{x1:.6f},{x2:.6f},{len(v[kk][0])},{int(v[kk][0].sum())}")

    def diffs(ma, mb, idx):
        ka = sorted(set(data[ma]) & set(data[mb]))
        d = []
        for k in ka:
            ya, pa, _ = data[ma][k]; yb, pb, _ = data[mb][k]
            assert np.array_equal(ya, yb)
            d.append(metrics(ya, pa)[idx] - metrics(yb, pb)[idx])
        nte = np.mean([len(data[ma][k][0]) for k in ka])
        return np.array(d), ka, 4 * nte, nte    # training folds hold 4/5 of the cohort

    # confirmatory hypotheses
    L += ["\n## Confirmatory hypotheses (AUPRC; one-sided; Holm over H1-H2)\n"]
    hyp = [("H1", "M1", "M4"), ("H2", "M1", "M2")]
    res = []
    for h, ma, mb in hyp:
        if ma in data and mb in data:
            d, ka, ntr, nte = diffs(ma, mb, 0)
            if len(d) == 10:
                mm, ci, t, p = nb_test(d, ntr, nte, "greater")
                sw = stats.shapiro(d)
                res.append((h, ma, mb, mm, ci, t, p, sw, d))
    if res:
        adj = holm([x[6] for x in res])
        L += ["| Hypothesis | mean difference [corrected 95% CI] | t (9 df) | p one-sided | Holm p | confirmed | Shapiro-Wilk W (p) |",
              "|---|---|---|---|---|---|---|"]
        for (h, ma, mb, mm, ci, t, p, sw, d), pa in zip(res, adj):
            L.append(f"| {h}: {MODELS[ma][0]} > {MODELS[mb][0]} | {mm:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {t:.2f} | "
                     f"{p:.4g} | {pa:.4g} | {'yes' if pa < 0.05 else 'no'} | {sw.statistic:.3f} ({sw.pvalue:.3f}) |")
        L.append("\nFold-level differences: " + "; ".join(f"{x[0]}: " + ", ".join(f"{v:+.4f}" for v in x[8]) for x in res))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, len(res), figsize=(3.2 * len(res), 3))
            for ax, x in zip(np.atleast_1d(axs), res):
                stats.probplot(x[8], dist="norm", plot=ax)
                ax.set_title(f"{x[0]}: fold-level AUPRC differences", fontsize=8)
            fig.tight_layout(); fig.savefig(f"{a.out}_qq.png", dpi=200); fig.savefig(f"{a.out}_qq.pdf")
        except Exception as e:   # the figure is optional
            L.append(f"(Q-Q plot not drawn: {e})")
        # sensitivity: paired bootstrap of the pooled predictions, per repeat
        L += ["\nSensitivity, paired bootstrap of the pooled predictions (AUPRC difference, 95% CI, one-sided p):\n"]
        for h, ma, mb, *_ in res:
            for r in REPEATS:
                ks = [(r, k) for k in FOLDS]
                y = np.concatenate([data[ma][k][0] for k in ks])
                pa = np.concatenate([data[ma][k][1] for k in ks]); pb = np.concatenate([data[mb][k][1] for k in ks])
                obs, ci, p = boot_pooled(y, pa, pb, a.n_boot, rng)
                L.append(f"- {h}, repeat {r}: {obs:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], p = {p:.4g} ({a.n_boot} replicates)")
    else:
        L.append("(not all 10 folds available for H1-H2)")

    # secondary comparisons of M1
    L += ["\n## Secondary: M1 against the other models (two-sided corrected t-test; Holm over the models available)\n",
          "| Comparison | metric | mean difference [corrected 95% CI] | p | Holm p |", "|---|---|---|---|---|"]
    if "M1" in data:
        others = [m for m in ("M2", "M3", "M4", "M5", "M6", "M7", "M8a", "M8b") if m in data]
        for idx, name in ((0, "AUPRC"), (1, "AUROC")):
            out = []
            for mb in others:
                d, ka, ntr, nte = diffs("M1", mb, idx)
                if len(d) >= 2:
                    out.append((mb, *nb_test(d, ntr, nte), len(d)))
            adj = holm([x[4] for x in out]) if out else []
            for (mb, mm, ci, t, p, J), pa in zip(out, adj):
                L.append(f"| M1 - {MODELS[mb][0]} ({J} folds) | {name} | {mm:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4g} | {pa:.4g} |")

    open(f"{a.out}.md", "w").write("\n".join(L) + "\n")
    open(f"{a.out}.csv", "w").write("model,repeat,fold,auprc,auroc,n,events\n" + "\n".join(rows) + "\n")
    print("\n".join(L))
    print(f"written {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
