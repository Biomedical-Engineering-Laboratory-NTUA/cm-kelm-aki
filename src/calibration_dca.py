"""
Calibration, recalibration and decision-curve analysis on saved test predictions.
================================================================================

Reads prediction files (preds/*.npz or preds_ens/*.npz, all on the same test set)
and reports, per model:

  * Brier score, calibration intercept (calibration-in-the-large) and slope
    (logistic regression of the outcome on logit(p)), expected calibration error
  * the same after RECALIBRATION. The class-weighted KELM outputs are not
    probabilities (Brier ~0.165 vs ~0.036 for trees), so every model is
    recalibrated with Platt scaling on logit(p), cross-fitted on the test set
    (K stratified folds: fit on K-1 folds, apply to the held-out fold), so no
    case is calibrated with its own label. Platt scaling is monotone, so AUROC
    and AUPRC are unchanged; isotonic recalibration is reported as a sensitivity.
    (Methods sentence: "Predicted risks were recalibrated by Platt scaling,
    cross-fitted over 5 stratified folds of the test set.")
  * decision-curve analysis (Vickers & Elkin 2006): net benefit of each
    recalibrated model at risk thresholds 1-30 %, against treat-all and
    treat-none, plus a table at clinically relevant thresholds (5, 10, 15, 20 %)
  * sensitivity, specificity, PPV and NPV at those thresholds

Outputs (aggregate only, safe to share):
  <out>.md    tables
  <out>.csv   per-model metrics
  <out>.png / <out>.pdf  publication figure (JBHI double column, 7.16 in): (a) reliability
              diagram on log axes, 20 quantile groups, for the --calib-models only (default
              CM-KELM, XGBoost, grid KELM), 95% Wilson intervals for the highlighted model,
              zero-event groups as open triangles, histogram of its predicted risks;
              (b) decision curves of every model with the 5-20 % range shaded.
              Use --calib-models all --out <x>_supp for the supplementary all-model version.

Usage:
  python calibration_dca.py preds/xgboost__esrd_eos2.npz preds/random_forest__esrd_eos2.npz \\
      preds_ens/cmkelm_m1000_gpu_p20_g15__esrd_eos2__popens_islands_ens.npz --out calib_eos2
"""

import argparse
import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_intercept_slope(y, p):
    z = logit(p).reshape(-1, 1)
    slope = LogisticRegression(C=1e6).fit(z, y).coef_[0, 0]
    # calibration-in-the-large: intercept with logit(p) as offset (one Newton solve)
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(z[:, 0] + a)))
        g = np.sum(y - q)
        h = np.sum(q * (1 - q))
        a += g / max(h, 1e-12)
        if abs(g) < 1e-8:
            break
    return a, slope


def ece(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def crossfit(y, p, kind="platt", k=5, seed=0):
    out = np.empty_like(p, dtype=float)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(p, y):
        if kind == "platt":
            m = LogisticRegression(C=1e6).fit(logit(p[tr]).reshape(-1, 1), y[tr])
            out[te] = m.predict_proba(logit(p[te]).reshape(-1, 1))[:, 1]
        else:
            m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr])
            out[te] = m.predict(p[te])
    return out


def net_benefit(y, p, t):
    n = len(y)
    pos = p >= t
    tp = np.sum(pos & (y == 1))
    fp = np.sum(pos & (y == 0))
    return tp / n - fp / n * t / (1 - t)


def at_threshold(y, p, t):
    pos = p >= t
    tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
    fn, tn = np.sum(~pos & (y == 1)), np.sum(~pos & (y == 0))
    div = lambda a, b: a / b if b else float("nan")
    return dict(sens=div(tp, tp + fn), spec=div(tn, tn + fp), ppv=div(tp, tp + fp), npv=div(tn, tn + fn),
                flagged=pos.mean())


def load(path):
    d = np.load(path, allow_pickle=False)
    name = json.loads(str(d["meta"])).get("name", os.path.basename(path)[:-4])
    return name, d["y_true"].astype(int), d["p"].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="calibration_dca")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--calib-dir", default=None,
                    help="directory of calibration sets written by calib_oof.py (training out-of-fold or "
                         "validation predictions, same file names as the test files): Platt scaling is fitted "
                         "there and applied unchanged to the test set; test-set cross-fitting is then only "
                         "reported as a sensitivity analysis")
    ap.add_argument("--bins", type=int, default=20, help="quantile groups in the reliability plot")
    ap.add_argument("--highlight", default="cmkelm", help="substring of the model drawn in bold")
    ap.add_argument("--calib-models", default="cmkelm,xgboost,nystrom_kelm_grid",
                    help="comma-separated name prefixes/substrings shown in the reliability panel "
                         "('all' = every model); the decision curves always show every model")
    ap.add_argument("--width", type=float, default=7.16, help="figure width in inches (JBHI double column 7.16)")
    a = ap.parse_args()

    runs = [load(f) for f in a.files]
    y0 = runs[0][1]
    for name, y, _ in runs:
        if len(y) != len(y0) or np.any(y != y0):
            raise SystemExit(f"{name}: different test set from {runs[0][0]}")
    prev = y0.mean()

    rows, recal = [], {}
    for name, y, p in runs:
        pc_cv = crossfit(y, p, "platt", a.folds)
        source = "test cross-fit"
        pc = pc_cv
        cf = os.path.join(a.calib_dir, name + ".npz") if a.calib_dir else None
        if cf and os.path.exists(cf):
            _, yc, pcal = load(cf)
            cal = LogisticRegression(C=1e6).fit(logit(pcal).reshape(-1, 1), yc)
            pc = cal.predict_proba(logit(p).reshape(-1, 1))[:, 1]
            source = json.loads(str(np.load(cf, allow_pickle=False)["meta"])).get("calibration_source", "external")
        elif a.calib_dir:
            print(f"WARNING: no calibration set for {name} in {a.calib_dir}; test cross-fitting used", flush=True)
        pi = crossfit(y, p, "isotonic", a.folds)
        recal[name] = pc
        ci, cs = cal_intercept_slope(y, p)
        ci2, cs2 = cal_intercept_slope(y, pc)
        r = dict(model=name, auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                 auprc_platt=average_precision_score(y, pc),
                 brier_raw=brier_score_loss(y, p), brier_platt=brier_score_loss(y, pc),
                 calibration_source=source, brier_platt_testcv=brier_score_loss(y, pc_cv),
                 brier_iso=brier_score_loss(y, pi), ece_raw=ece(y, p), ece_platt=ece(y, pc),
                 cal_int_raw=ci, cal_slope_raw=cs, cal_int_platt=ci2, cal_slope_platt=cs2,
                 mean_p_raw=p.mean())
        for t in THRESHOLDS:
            r[f"nb_{int(t * 100)}"] = net_benefit(y, pc, t)
            for k_, v in at_threshold(y, pc, t).items():
                r[f"{k_}_{int(t * 100)}"] = v
        rows.append(r)

    # ----- report
    n, npos = len(y0), int(y0.sum())
    L = [f"# Calibration and decision-curve analysis\n",
         f"Test set: n={n}, AKI={npos} ({100 * prev:.2f}%). Recalibration: Platt scaling on logit(p), "
         + (f"fitted on the calibration sets of {a.calib_dir} (column 'source') and applied unchanged to the "
            f"test set; 'Brier Platt test-CV' = cross-fitted over {a.folds} folds of the test set (sensitivity). "
            if a.calib_dir else
            f"cross-fitted over {a.folds} stratified folds of the test set (isotonic as sensitivity). ")
         + "AUROC is unchanged by the monotone recalibration.\n",
         "## 1. Calibration before and after recalibration\n",
         "| Model | source | AUROC | AUPRC | mean predicted risk (raw) | Brier raw | Brier Platt | Brier Platt test-CV | "
         "Brier isotonic | ECE raw | ECE Platt | intercept raw | slope raw | intercept Platt | slope Platt |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['calibration_source']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['mean_p_raw']:.3f} | "
                 f"{r['brier_raw']:.4f} | {r['brier_platt']:.4f} | {r['brier_platt_testcv']:.4f} | {r['brier_iso']:.4f} | {r['ece_raw']:.4f} | "
                 f"{r['ece_platt']:.4f} | {r['cal_int_raw']:+.2f} | {r['cal_slope_raw']:.2f} | "
                 f"{r['cal_int_platt']:+.2f} | {r['cal_slope_platt']:.2f} |")
    L.append(f"\nReference: Brier of a constant prediction equal to the prevalence = {prev * (1 - prev):.4f}. "
             "Ideal intercept 0, slope 1.\n")

    L.append("## 2. Net benefit (recalibrated) at clinically relevant thresholds\n")
    L.append("| Model | " + " | ".join(f"NB @{int(t * 100)}%" for t in THRESHOLDS) + " |")
    L.append("|---|" + "---|" * len(THRESHOLDS))
    L.append("| treat all | " + " | ".join(f"{prev - (1 - prev) * t / (1 - t):.4f}" for t in THRESHOLDS) + " |")
    L.append("| treat none | " + " | ".join("0.0000" for _ in THRESHOLDS) + " |")
    for r in rows:
        L.append(f"| {r['model']} | " + " | ".join(f"{r[f'nb_{int(t * 100)}']:.4f}" for t in THRESHOLDS) + " |")
    L.append("\nNet benefit x 100 = true positives per 100 patients net of false positives weighted by "
             "the threshold odds.\n")

    L.append("## 3. Operating points (recalibrated risk)\n")
    L.append("| Model | threshold | flagged % | sensitivity | specificity | PPV | NPV |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        for t in THRESHOLDS:
            k = int(t * 100)
            L.append(f"| {r['model']} | {k}% | {100 * r[f'flagged_{k}']:.1f} | {r[f'sens_{k}']:.3f} | "
                     f"{r[f'spec_{k}']:.3f} | {r[f'ppv_{k}']:.3f} | {r[f'npv_{k}']:.3f} |")

    open(f"{a.out}.md", "w").write("\n".join(L) + "\n")
    import csv
    with open(f"{a.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(L))

    # ----- figure (publication style)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig = plt.figure(figsize=(a.width, a.width * 0.45))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1.15], hspace=0.08, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[:, 1])
    lo, hi = 1e-3, 1.0
    style = {}
    for i, (name, _, _) in enumerate(runs):
        hl = a.highlight and a.highlight.lower() in name.lower()
        style[name] = dict(color=("#c2185b" if hl else COLORS.get(short(name), PALETTE[i % len(PALETTE)])), lw=2.0 if hl else 1.0,
                           zorder=5 if hl else 2, alpha=1.0 if hl else 0.85)
    want = [w.strip().lower() for w in a.calib_models.split(",") if w.strip()]
    show = [r for r in runs if "all" in want or any(w in r[0].lower() for w in want)] or runs
    for name, y, _ in show:
        pc = recal[name]
        edges = np.quantile(pc, np.linspace(0, 1, a.bins + 1))
        idx = np.clip(np.searchsorted(edges, pc, side="right") - 1, 0, a.bins - 1)
        grp = [idx == b for b in range(a.bins) if (idx == b).any()]
        mp = np.array([pc[g].mean() for g in grp])
        k = np.array([y[g].sum() for g in grp])
        nn = np.array([g.sum() for g in grp])
        mo = k / nn
        st = style[name]
        hl = st["lw"] > 1.5
        ax1.plot(mp, np.where(k > 0, mo, np.nan), "-", label=short(name), **st)   # line broken at zero-event groups
        pos = k > 0
        ax1.plot(mp[pos], mo[pos], "o", ms=3 if hl else 2.2, color=st["color"], zorder=st["zorder"])
        # groups with no AKI event: open triangle on the lower axis (observed rate 0 cannot be drawn on a log axis)
        ax1.plot(np.clip(mp[~pos], lo * 1.15, None), np.full((~pos).sum(), lo * 1.15), "v", ms=3.5, mfc="none", color=st["color"],
                 zorder=st["zorder"])
        if hl:   # 95% Wilson interval of the observed rate for the highlighted model
            z = 1.96
            c = (mo + z * z / (2 * nn)) / (1 + z * z / nn)
            h = z * np.sqrt(mo * (1 - mo) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
            ax1.vlines(mp, np.clip(c - h, lo, None), c + h, color=st["color"], lw=0.8, alpha=0.6, zorder=4)
    ax1.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(lo, 0.8)
    ax1.set_ylim(lo, 0.8)
    ax1.set_ylabel("observed AKI rate")
    ax1.set_title(f"(a) Reliability after Platt recalibration ({a.bins} risk groups)" if not a.calib_dir else
                  f"(a) Reliability, Platt fitted outside the test set ({a.bins} groups)")
    ax1.tick_params(labelbottom=False)
    hl_name = next((n for n, _, _ in runs if a.highlight and a.highlight.lower() in n.lower()), runs[0][0])
    ph = np.clip(recal[hl_name], lo, None)
    axh.hist(ph, bins=np.logspace(np.log10(lo), np.log10(0.8), 40), color="#c2185b", alpha=0.6)
    axh.set_yticks([])
    axh.set_xlabel(f"predicted risk (log scale); histogram: {short(hl_name)}")
    for s_ in ("top", "right", "left"):
        axh.spines[s_].set_visible(False)

    ts = np.linspace(0.01, 0.30, 59)
    ax2.axvspan(0.05, 0.20, color="#9e9e9e", alpha=0.12, lw=0, label="clinically relevant range")
    ax2.plot(ts, [prev - (1 - prev) * t / (1 - t) for t in ts], color="grey", lw=0.8, ls=":", label="treat all")
    ax2.axhline(0, color="black", lw=0.8, label="treat none")
    for name, y, _ in runs:
        ax2.plot(ts, [net_benefit(y, recal[name], t) for t in ts], label=short(name), **style[name])
    ax2.set_ylim(-0.003, prev * 1.02)
    ax2.set_xlim(0.01, 0.30)
    ax2.set_xlabel("risk threshold")
    ax2.set_ylabel("net benefit")
    ax2.set_title("(b) Decision curves")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, which="both")
    ax1.legend(loc="upper left", frameon=False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=4, fontsize=5.8)
    fig.savefig(f"{a.out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{a.out}.pdf", bbox_inches="tight")
    print(f"\nwritten {a.out}.md, {a.out}.csv, {a.out}.png, {a.out}.pdf")


COLORS = {"Logistic regression": "#1f77b4", "Random forest": "#ff7f0e", "XGBoost": "#2ca02c", "LightGBM": "#17becf",
          "Grid-tuned KELM": "#9467bd", "Stacked DL": "#8c564b", "TabTransformer": "#7f7f7f",
          "Tabular Mamba": "#9e9d24", "MLP": "#e377c2", "JIT Transformer": "#3949ab", "FT-Transformer": "#6d4c41",
          "Sparse GP (ARD)": "#00897b"}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#7f7f7f", "#bcbd22"]
SHORT = [(r"^logistic_regression", "Logistic regression"), (r"^random_forest", "Random forest"), (r"^xgboost", "XGBoost"),
         (r"^lightgbm", "LightGBM"), (r"^nystrom_kelm_grid", "Grid-tuned KELM"), (r"^dl_stacked_lr", "Stacked DL"),
         (r"^dl_jit", "JIT Transformer"), (r"^dl_mlp", "MLP"), (r"^dl_ftt", "FT-Transformer"), (r"^gp_ard", "Sparse GP (ARD)"),
         (r"^dl_stacked_svcbal", "DL stacked (bal. SVC)"), (r"^dl_transformer", "TabTransformer"),
         (r"^dl_mamba", "Tabular Mamba"), (r"^mlp", "MLP"), (r"^cmkelm.*mmtabu.*islands", "CM-KELM + tabu"),
         (r"^cmkelm.*islands", "CM-KELM"), (r"^cmkelm", "CM-KELM (single)")]


def short(name):
    import re
    for pat, lab in SHORT:
        if re.search(pat, name):
            return lab
    return name


if __name__ == "__main__":
    main()
