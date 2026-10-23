"""
Stability of the evolved variable masks (review point A8).

Reads the island-ensemble members (4 champions per run, 10 runs = 40 masks) and the best
individual of each run (10 masks) and reports: union and intersection sizes, how many
variables are kept by at least 50%/90% of the masks, the mean pairwise Jaccard index and the
stability estimator of Nogueira et al. (JMLR 2018), which allows masks of different sizes
(1 = identical masks, 0 = random selection of the same average size).

    python mask_stability.py --esrd cr4_z49 --features eos2 --out mask_stability_v3
"""
import argparse
import glob
import itertools
import json
import re

import numpy as np

from cohort_split import load_split


def nogueira(Z):
    """Z: masks x variables (0/1)."""
    M, d = Z.shape
    kbar = Z.sum(1).mean()
    var = Z.var(axis=0, ddof=1)
    return 1 - var.mean() / ((kbar / d) * (1 - kbar / d))


def jaccard(Z):
    v = [np.logical_and(a, b).sum() / np.logical_or(a, b).sum() for a, b in itertools.combinations(Z.astype(bool), 2)]
    return float(np.mean(v))


def summary(Z, cols, label):
    f = Z.mean(0)
    lines = [f"## {label} ({Z.shape[0]} masks, {Z.shape[1]} candidate variables)", "",
             f"- mask size: mean {Z.sum(1).mean():.1f}, range {Z.sum(1).min()}-{Z.sum(1).max()}",
             f"- union {int((f > 0).sum())}, intersection {int((f == 1).sum())}, "
             f">=90% of masks {int((f >= 0.9).sum())}, >=50% {int((f >= 0.5).sum())}",
             f"- mean pairwise Jaccard {jaccard(Z):.3f}; Nogueira stability {nogueira(Z):.3f}", "",
             "| variable | fraction of masks |", "|---|---|"]
    lines += [f"| {cols[j]} | {f[j]:.2f} |" for j in np.argsort(-f) if f[j] >= 0.5]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2")
    ap.add_argument("--prefix", default="cmkelm_m1000_gpu_p20_g15")
    ap.add_argument("--out", default="mask_stability_v3")
    a = ap.parse_args()
    _, _, _, _, cols, tag = load_split(esrd=a.esrd, features=a.features, verbose=False)
    sfx = "" if tag == "all" else f"__{tag}"
    champ, best = [], []
    for f in sorted(glob.glob(f"preds/{a.prefix}_s*{sfx}__popens_islands.npz")):
        if not re.search(rf"_s\d+{re.escape(sfx)}__popens_islands\.npz$", f):
            continue
        meta = json.loads(str(np.load(f)["meta"]))
        for m in meta["members"]:
            z = np.zeros(len(cols)); z[m["features"]] = 1; champ.append(z)
        b = json.loads(str(np.load(f.replace("__popens_islands", ""))["meta"]))
        best.append(np.isin(cols, b["active_features"]).astype(float))
    out = summary(np.array(champ), cols, "Island champions") + "\n" + summary(np.array(best), cols, "Best individual per run")
    open(f"{a.out}.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
