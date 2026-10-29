#!/usr/bin/env python
"""Fig. 3 of the manuscript (confirmatory cross-validation): (a) reliability after Platt
recalibration cross-fitted inside each test fold, pooled over the 10 test folds of both
repeats (each patient appears twice, once per repeat), 20 risk groups on a log scale;
(b) decision curves of all models (mean net benefit of the two repeats).
Writes only the figure (aggregate curves).

  python fig_cv_paper.py --out fig_cv_calib_dca
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import outer_cv
import cv_secondary as cs

LABEL = {"M1": "DE-KELM-ARD", "M2": "DE-KELM", "M3": "CM-KELM", "M4": "Grid-tuned KELM",
         "M5": "Sparse GP (ARD)", "M6": "XGBoost", "M7": "FT-Transformer",
         "M8a": "Logistic regression", "M8b": "Random forest", "LGBM": "LightGBM"}
STYLE = {"M1": ("#c0392b", "-", 2.0), "M2": ("#e67e22", "--", 1.2), "M3": ("#d4a017", ":", 1.2),
         "M4": ("#7f8c8d", "-", 1.4), "M5": ("#8e44ad", "-", 1.4), "M6": ("#2471a3", "-", 1.4),
         "M7": ("#17a589", "--", 1.2), "M8a": ("#000000", ":", 1.2), "M8b": ("#27ae60", "-.", 1.2),
         "LGBM": ("#5dade2", "--", 1.2)}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_cv_calib_dca")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    C = np.load("cv_cache/cohort_esrd_eos2.npz", allow_pickle=True)
    folds = outer_cv.fold_ids(C["stage"], C["group"])
    yall = np.asarray(outer_cv.label(C["stage"], "aki")).astype(int)
    pats = cs.model_patterns()
    keys = [k for k in LABEL if k in pats]
    Q, Y = {k: {} for k in keys}, {}
    for r in (1, 2):
        for kk in range(1, 6):
            te = np.flatnonzero(folds[r] == kk)
            Y[(r, kk)] = yall[te]
            for k in keys:
                y, p, n = cs.load_preds(pats[k][1], cs.tag_of(r, kk, "aki", 365), "preds_cv")
                if p is None:
                    continue
                assert np.array_equal(y, Y[(r, kk)])
                Q[k][(r, kk)] = cs.platt_crossfit(p, y, seed=a.seed)
    keys = [k for k in keys if len(Q[k]) == 10]

    fig = plt.figure(figsize=(7.16, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], width_ratios=[1, 1.15], hspace=0.08, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0]); axh = fig.add_subplot(gs[1, 0], sharex=ax)
    axd = fig.add_subplot(gs[:, 1])
    lo, hi = 1e-3, 0.6
    for k in ("M4", "M6", "M5", "M1"):
        if k not in keys:
            continue
        q = np.concatenate([Q[k][f] for f in sorted(Q[k])])
        y = np.concatenate([Y[f] for f in sorted(Q[k])])
        b = np.quantile(q, np.linspace(0, 1, 21))
        idx = np.clip(np.searchsorted(b, q, side="right") - 1, 0, 19)
        px = np.array([q[idx == i].mean() for i in range(20)])
        n = np.array([(idx == i).sum() for i in range(20)])
        e = np.array([y[idx == i].sum() for i in range(20)])
        py = e / n
        c, ls, lw = STYLE[k]
        ok = e > 0
        ax.plot(px[ok], py[ok], marker="o", ms=2.5, color=c, ls=ls, lw=lw * 0.7, label=LABEL[k])
        ax.plot(px[~ok], np.full((~ok).sum(), lo * 1.2), "v", ms=3, color=c)
        if k == "M1":
            l, h = wilson(e, n)
            ax.vlines(px, np.maximum(l, lo), h, color=c, lw=0.6)
            axh.hist(q, bins=np.logspace(np.log10(lo), np.log10(hi), 40), color=c, alpha=0.6)
    ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_ylabel("observed AKI rate", fontsize=7)
    ax.set_title("(a) Reliability, Platt fitted within each test fold", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    plt.setp(ax.get_xticklabels(), visible=False)
    axh.set_yscale("log"); axh.yaxis.set_major_locator(matplotlib.ticker.NullLocator()); axh.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axh.set_xlabel("predicted risk (log scale); histogram: DE-KELM-ARD", fontsize=7)
    for x in (ax, axh, axd):
        x.tick_params(labelsize=6.5)

    ts = np.linspace(0.01, 0.30, 59)
    prev = np.mean([Y[f].mean() for f in Y])
    axd.axvspan(0.05, 0.20, color="0.92", zorder=0)
    axd.plot(ts, prev - (1 - prev) * ts / (1 - ts), color="0.5", lw=0.8, ls="--", label="treat all")
    axd.axhline(0, color="0.3", lw=0.8, label="treat none")
    for k in keys:
        nb = []
        for t in ts:
            v = []
            for r in (1, 2):
                q = np.concatenate([Q[k][(r, j)] for j in range(1, 6)])
                y = np.concatenate([Y[(r, j)] for j in range(1, 6)])
                v.append(cs.op_threshold(q, y, t)["nb"])
            nb.append(np.mean(v))
        c, ls, lw = STYLE[k]
        axd.plot(ts, nb, color=c, ls=ls, lw=lw, label=LABEL[k])
    axd.set_xlim(0.01, 0.30); axd.set_ylim(-0.002, 0.045)
    axd.set_xlabel("risk threshold", fontsize=7); axd.set_ylabel("net benefit", fontsize=7)
    axd.set_title("(b) Decision curves (mean of two repeats)", fontsize=7.5)
    axd.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper right")
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=250, bbox_inches="tight")
    print("written", a.out + ".pdf")


if __name__ == "__main__":
    main()
