"""
Exploratory 2 x 2 ablation of M1 in the confirmatory cross-validation (not prespecified):
search (plain DE / cultural-memetic) x encoding (mask / ARD), all as top-5 population ensembles
averaged over seeds, plus the equal-budget random-search controls (first repeat).
Reads the per-fold table of cv_eval.py and writes a Markdown table.

    python cv_ablation.py results_v18/cv_primary_w7.csv results_v18/cv_ablation
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

d = pd.read_csv(sys.argv[1])
out = sys.argv[2]
LAB = {"M1": "DE + ARD (M1)", "M2": "DE + mask (M2)", "CM-ARD": "CM + ARD", "M3-top5": "CM + mask",
       "RS-ARD": "random search + ARD", "RS-mask": "random search + mask"}


def nb(x, ratio=0.25):
    x = np.asarray(x, float)
    J = len(x)
    se = np.sqrt(x.var(ddof=1) * (1 / J + ratio))
    t = stats.t.ppf(0.975, J - 1)
    m = x.mean()
    return m, m - t * se, m + t * se, 2 * stats.t.sf(abs(m / se), J - 1), J


def diff(a, b, m):
    A = d[d.model == a].set_index(["repeat", "fold"])[m]
    B = d[d.model == b].set_index(["repeat", "fold"])[m]
    i = A.index.intersection(B.index)
    return (A[i] - B[i]).values


L = ["# Exploratory 2 x 2 ablation of DE-KELM-ARD (confirmatory folds; not prespecified)\n",
     "| Model | folds | AUROC | AUPRC |", "|---|---|---|---|"]
for k in LAB:
    s = d[d.model == k]
    if len(s):
        L.append(f"| {LAB[k]} | {len(s)} | {s.auroc.mean():.4f} | {s.auprc.mean():.4f} |")
L += ["", "| Comparison | metric | difference [corrected 95% CI] | p (unadjusted) | folds |", "|---|---|---|---|---|"]
for a, b, what in (("M1", "M2", "encoding, plain DE"), ("CM-ARD", "M3-top5", "encoding, cultural-memetic"),
                   ("M1", "CM-ARD", "search, ARD"), ("M2", "M3-top5", "search, mask"),
                   ("M1", "RS-ARD", "evolution vs random, ARD"), ("M2", "RS-mask", "evolution vs random, mask")):
    if not (len(d[d.model == a]) and len(d[d.model == b])):
        continue
    for m in ("auprc", "auroc"):
        x = diff(a, b, m)
        if len(x) < 2:
            continue
        v = nb(x)
        L.append(f"| {LAB[a]} - {LAB[b]} ({what}) | {m.upper()} | {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] | {v[3]:.3g} | {v[4]} |")
open(out + ".md", "w").write("\n".join(L) + "\n")
print("written", out + ".md")
