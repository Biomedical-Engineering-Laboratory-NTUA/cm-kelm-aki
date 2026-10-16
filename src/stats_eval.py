"""
Statistical evaluation of saved test-set predictions (DE-CM-KELM paper).
=======================================================================

Every run script now saves its held-out predictions as preds/<name>.npz
(keys: y_true, p, plus a JSON 'meta' string). This module loads any set of
those files and produces the comparison table for the Results section:

  - AUROC, AUPRC, Brier score, each with a stratified bootstrap 95% CI
  - vs a reference model (default: random_forest):
      * Delta AUROC with DeLong's test for correlated ROC curves
        (fast O(n log n) implementation, Sun & Xu 2014)
      * Delta AUPRC with a paired stratified bootstrap CI and p-value
        (no closed-form test exists for AUPRC)
  - Holm correction across all comparisons against the reference

All models MUST share the identical test set (same split seed). This is
checked: y_true arrays must match exactly, otherwise the script stops.

Usage (on the server, after runs have saved predictions):
    python stats_eval.py preds/*.npz --reference random_forest
    python stats_eval.py preds/*.npz --reference random_forest --n-boot 5000

Outputs: stats_table.csv and stats_table.md next to the working directory.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# ----------------------------------------------------------------------
# DeLong (fast version, Sun & Xu 2014, IEEE Signal Processing Letters)
# ----------------------------------------------------------------------
def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1  # average rank, 1-based
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_structural(y_true, preds):
    """preds: array (k_models, n). Returns AUCs (k,) and covariance (k, k)."""
    y_true = np.asarray(y_true).astype(int)
    pos, neg = preds[:, y_true == 1], preds[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y_true, p1, p2):
    """Two-sided DeLong test for two correlated AUROCs on the same cases.
    Returns (auc1, auc2, delta, se_delta, z, p_value)."""
    aucs, cov = _delong_structural(y_true, np.vstack([p1, p2]))
    delta = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var, 1e-300))
    z = delta / se
    return aucs[0], aucs[1], delta, se, z, 2 * stats.norm.sf(abs(z))


def delong_ci(y_true, p, alpha=0.05):
    aucs, cov = _delong_structural(y_true, np.atleast_2d(p))
    se = np.sqrt(cov[0, 0])
    q = stats.norm.ppf(1 - alpha / 2)
    return aucs[0], max(0.0, aucs[0] - q * se), min(1.0, aucs[0] + q * se)


# ----------------------------------------------------------------------
# Stratified bootstrap (positives and negatives resampled separately,
# so every replicate keeps the 4.83% AKI prevalence -- important for AUPRC)
# ----------------------------------------------------------------------
METRICS = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "brier": brier_score_loss,
}


def _boot_indices(y_true, n_boot, rng):
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])


def bootstrap_ci(y_true, p, metric="auroc", n_boot=2000, alpha=0.05, seed=0):
    f = METRICS[metric]
    rng = np.random.default_rng(seed)
    vals = np.array([f(y_true[i], p[i]) for i in _boot_indices(y_true, n_boot, rng)])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return f(y_true, p), lo, hi


def paired_bootstrap_diff(y_true, p1, p2, metric="auprc", n_boot=2000, alpha=0.05, seed=0):
    """Delta = metric(p1) - metric(p2) on identical resamples.
    p-value: two-sided, from the bootstrap distribution of delta
    (proportion of replicates on the other side of zero, doubled)."""
    f = METRICS[metric]
    rng = np.random.default_rng(seed)
    d = np.array([f(y_true[i], p1[i]) - f(y_true[i], p2[i])
                  for i in _boot_indices(y_true, n_boot, rng)])
    point = f(y_true, p1) - f(y_true, p2)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p_val = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    p_val = max(p_val, 1.0 / n_boot)  # resolution floor, never report 0
    return point, lo, hi, p_val


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    k = len(pvals)
    adj = np.empty(k)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (k - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def save_predictions(path, name, y_true, p, **meta):
    """Called by run scripts. Saves preds/<name>.npz."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = {"name": name, **meta}
    np.savez_compressed(path, y_true=np.asarray(y_true).astype(np.int8),
                        p=np.asarray(p, dtype=np.float64), meta=json.dumps(meta))
    print(f"Saved predictions -> {path}", flush=True)


def load_predictions(paths):
    runs = {}
    for path in paths:
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        runs[meta["name"]] = {"y": d["y_true"].astype(int), "p": d["p"], "meta": meta}
    return runs


def _fmt(v, lo, hi, nd=4):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def build_table(runs, reference, n_boot=2000, seed=0):
    names = list(runs)
    y0 = runs[names[0]]["y"]
    for nm in names[1:]:
        if not np.array_equal(runs[nm]["y"], y0):
            sys.exit(f"ERROR: '{nm}' has a different test set from '{names[0]}' "
                     f"-- all runs must use the same split seed.")
    if reference not in runs:
        sys.exit(f"ERROR: reference '{reference}' not found. Available: {names}")

    rows = []
    for nm in names:
        p = runs[nm]["p"]
        r = {"model": nm}
        for met in ("auroc", "auprc", "brier"):
            r[met] = bootstrap_ci(y0, p, met, n_boot=n_boot, seed=seed)
        rows.append(r)

    ref_p = runs[reference]["p"]
    comps = [nm for nm in names if nm != reference]
    dl = {nm: delong_test(y0, runs[nm]["p"], ref_p) for nm in comps}
    pb = {nm: paired_bootstrap_diff(y0, runs[nm]["p"], ref_p, "auprc", n_boot=n_boot, seed=seed)
          for nm in comps}
    adj_auc = dict(zip(comps, holm([dl[c][5] for c in comps]))) if comps else {}
    adj_pr = dict(zip(comps, holm([pb[c][3] for c in comps]))) if comps else {}

    for r in rows:
        nm = r["model"]
        if nm == reference:
            r.update(d_auroc="(reference)", p_auroc="", d_auprc="(reference)", p_auprc="")
            continue
        _, _, delta, se, _, p = dl[nm]
        q = stats.norm.ppf(0.975)
        r["d_auroc"] = _fmt(delta, delta - q * se, delta + q * se)
        r["p_auroc"] = f"{p:.3g} (Holm {adj_auc[nm]:.3g})"
        d, lo, hi, pp = pb[nm]
        r["d_auprc"] = _fmt(d, lo, hi)
        r["p_auprc"] = f"{pp:.3g} (Holm {adj_pr[nm]:.3g})"
    return rows, len(y0), int(y0.sum())


def write_outputs(rows, n, n_pos, reference, n_boot, stem="stats_table"):
    head = ["Model", "AUROC [95% CI]", "AUPRC [95% CI]", "Brier [95% CI]",
            f"ΔAUROC vs {reference} [95% CI]", "p (DeLong)",
            f"ΔAUPRC vs {reference} [95% CI]", "p (paired bootstrap)"]
    lines = []
    for r in rows:
        lines.append([r["model"], _fmt(*r["auroc"]), _fmt(*r["auprc"]), _fmt(*r["brier"]),
                      r["d_auroc"], r["p_auroc"], r["d_auprc"], r["p_auprc"]])
    with open(f"{stem}.md", "w", encoding="utf-8") as fh:
        fh.write(f"Test set: n={n}, AKI={n_pos} ({100*n_pos/n:.2f}%). "
                 f"Stratified bootstrap, {n_boot} replicates. "
                 f"ΔAUROC CI and p from DeLong; ΔAUPRC from paired bootstrap. "
                 f"Holm-adjusted p in parentheses.\n\n")
        fh.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for ln in lines:
            fh.write("| " + " | ".join(ln) + " |\n")
    with open(f"{stem}.csv", "w", encoding="utf-8") as fh:
        fh.write(";".join(head) + "\n")
        for ln in lines:
            fh.write(";".join(ln) + "\n")
    print(open(f"{stem}.md", encoding="utf-8").read())
    print(f"Written: {stem}.md, {stem}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--reference", default="random_forest")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="stats_table")
    a = ap.parse_args()
    runs = load_predictions(a.files)
    rows, n, n_pos = build_table(runs, a.reference, a.n_boot, a.seed)
    write_outputs(rows, n, n_pos, a.reference, a.n_boot, a.out)


if __name__ == "__main__":
    main()
