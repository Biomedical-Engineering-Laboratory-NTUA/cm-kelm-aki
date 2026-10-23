"""
Seed-level comparison of two CM-KELM variants (paired by search seed).
======================================================================

Each run writes preds/<name>.json with its single-best and population-
ensemble AUROC/AUPRC. For two variants, this script pairs runs by seed and
reports mean +/- SD, the mean paired difference and a two-sided Wilcoxon
signed-rank test, for the single best model and for each single-run
population ensemble (topk5, islands, diverse5).

Unit of analysis = one evolutionary run, which is what matters for a
stochastic method: "if I run it once, what do I get?"

Usage:
    python compare_seeds.py cmkelm_m1000_adnone_p20_g15 cmkelm_m1000_adKAH_p20_g15
    python compare_seeds.py cmkelm_m1000_gpu_p20_g15__esrd_eos cmkelm_m1000_mmfc_gpu_p20_g15__esrd_eos
    python compare_seeds.py A B 11-20        # only seeds 11..20 (confirmatory run)
"""
import glob
import json
import re
import sys

import numpy as np
from scipy.stats import t as t_dist, wilcoxon


def load(prefix):
    """prefix = run name without _s<seed>; a cohort tag may follow after '__'
    (e.g. cmkelm_m1000_gpu_p20_g15__esrd_eos)."""
    base, _, tag = prefix.partition("__")
    tail = f"__{tag}" if tag else ""
    runs = {}
    for f in glob.glob(f"preds/{base}_s*{tail}.json"):
        m = re.search(r"_s(\d+)" + re.escape(tail) + r"\.json$", f)
        if m:
            runs[int(m.group(1))] = json.load(open(f))
    return runs


def metric(run, which, key):
    return run[key] if which == "single" else run["population_ensembles"][which][key]


def main(a, b, seed_range=None):
    ra, rb = load(a), load(b)
    seeds = sorted(set(ra) & set(rb))
    if seed_range:                      # e.g. "11-20": confirmatory seeds only
        lo, hi = (int(v) for v in seed_range.split("-"))
        seeds = [s for s in seeds if lo <= s <= hi]
    print(f"A = {a}\nB = {b}\npaired seeds: {seeds} (n={len(seeds)})\n")
    print(f"{'model':10s} {'metric':6s} {'A mean±SD':>16s} {'B mean±SD':>16s} {'B-A':>8s} "
          f"{'95% CI (t)':>19s} {'Wilcoxon p':>11s}")
    for which in ("single", "topk5", "islands", "diverse5"):
        for key in ("auroc", "auprc"):
            try:
                xa = np.array([metric(ra[s], which, key) for s in seeds])
                xb = np.array([metric(rb[s], which, key) for s in seeds])
            except (KeyError, TypeError):
                continue
            p = wilcoxon(xb, xa).pvalue if len(seeds) >= 5 and np.any(xb != xa) else float("nan")
            d = xb - xa
            h = t_dist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"{which:10s} {key:6s} {xa.mean():.4f}±{xa.std(ddof=1):.4f} "
                  f"{xb.mean():.4f}±{xb.std(ddof=1):.4f} {d.mean():+8.4f} "
                  f"[{d.mean() - h:+.4f}, {d.mean() + h:+.4f}] {p:11.3g}")
    if len(seeds) < 5:
        print("\n(Wilcoxon needs >= 5 paired seeds; p-values omitted)")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:])
