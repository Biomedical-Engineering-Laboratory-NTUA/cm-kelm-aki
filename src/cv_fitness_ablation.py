"""
Search ablations on the search objective itself, without the test set (review point: test reuse).
================================================================================================

The endpoint is the fitness of the best individual of each run: the 3-fold CV AUPRC on the
training set (fixed folds and landmarks) minus 0.001 per selected variable, i.e. what every
search variant tries to maximise with the same budget. It is read from preds/<run>.json
('best_fitness', written by newer runs) or, for older runs, from the last
'Gen g/G: best ...' line of each run in the Slurm logs (the newest log wins when a run name
occurs more than once). Variants are paired by seed as in compare_seeds.py.

    python cv_fitness_ablation.py --logs "logs/*.log" --out cv_fitness_ablation
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.stats import t as t_dist, wilcoxon

B = "cmkelm_m1000_gpu_p20_g15__esrd_eos2"
PAIRS = [
    ("Plain DE, no belief space or local search", B, "cmkelm_m1000_gpu_nocult_nomem_p20_g23__esrd_eos2", "1-10"),
    ("No belief space", B, "cmkelm_m1000_gpu_nocult_p20_g15__esrd_eos2", "1-10"),
    ("No local search", B, "cmkelm_m1000_gpu_nomem_p20_g23__esrd_eos2", "1-10"),
    ("Adaptive sources, acceptance, islands", "cmkelm_m1000_adnone_gpu_p20_g15__esrd_eos2",
     "cmkelm_m1000_adKAH_gpu_p20_g15__esrd_eos2", "1-10"),
    ("Multimeme fuzzy + credit", B, "cmkelm_m1000_mmfc_gpu_p20_g15__esrd_eos2", "1-10"),
    ("Multimeme fuzzy", B, "cmkelm_m1000_mmfuzzy_gpu_p20_g15__esrd_eos2", "1-10"),
    ("Multimeme credit", B, "cmkelm_m1000_mmcredit_gpu_p20_g15__esrd_eos2", "1-10"),
    ("Multimeme uniform", B, "cmkelm_m1000_mmuniform_gpu_p20_g15__esrd_eos2", "1-10"),
    ("Tabu/GLS meme", B, "cmkelm_m1000_mmtabu_gpu_p20_g15__esrd_eos2", "1-10"),
    ("Tabu/GLS meme, seed replication", B, "cmkelm_m1000_mmtabu_gpu_p20_g15__esrd_eos2", "11-20"),
    ("Tabu/GLS meme, confirmation", B, "cmkelm_m1000_mmtabu_gpu_p20_g15__esrd_eos2", "21-50"),
    ("SA meme", B, "cmkelm_m1000_mmsa_gpu_p20_g15__esrd_eos2", "1-10"),
    ("HC meme", B, "cmkelm_m1000_mmhc_gpu_p20_g15__esrd_eos2", "1-10"),
    ("Random search, same budget", B, "cmkelm_m1000_gpu_randsearch_n500__esrd_eos2", "1-10"),
]
HEAD = re.compile(r"^(cmkelm_\S+): population=")
GEN = re.compile(r"Gen (\d+)/(\d+): best(?: fitness so far =)? (-?[0-9.]+)")


def from_logs(pattern):
    out = {}
    for f in sorted(glob.glob(pattern), key=os.path.getmtime):
        cur, last = None, None
        for line in open(f, errors="replace"):
            h = HEAD.match(line.strip())
            if h:
                if cur and last is not None:
                    out[cur] = last
                cur, last = h.group(1), None
                continue
            g = GEN.search(line)
            if g and cur:
                last = float(g.group(3))
        if cur and last is not None:
            out[cur] = last
    return out


def from_json(dirs):
    out = {}
    for d in dirs:
        for f in glob.glob(f"{d}/cmkelm_*.json"):
            v = json.load(open(f)).get("best_fitness")
            if v is not None:
                out[os.path.basename(f)[:-5]] = float(v)
    return out


def by_seed(fit, prefix):
    base, _, tag = prefix.partition("__")
    tail = f"__{tag}" if tag else ""
    res = {}
    for name, v in fit.items():
        m = re.fullmatch(re.escape(base) + r"_s(\d+)" + re.escape(tail), name)
        if m:
            res[int(m.group(1))] = v
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/*.log")
    ap.add_argument("--json-dirs", nargs="*", default=["preds", "preds_rep"])
    ap.add_argument("--out", default="cv_fitness_ablation")
    a = ap.parse_args()
    fit = from_logs(a.logs)
    fit.update(from_json(a.json_dirs))
    L = ["# Search ablations on the CV fitness of the best individual (training set only)\n",
         "Fitness = 3-fold CV AUPRC on the training set minus 0.001 per selected variable. "
         "Difference B - A, mean over paired seeds, 95% t-interval, Wilcoxon signed-rank test.\n",
         "| Variant (B) vs A | seeds | n | A mean | B mean | B - A | 95% CI | p |", "|---|---|---|---|---|---|---|---|"]
    for label, A, Bv, rng in PAIRS:
        fa, fb = by_seed(fit, A), by_seed(fit, Bv)
        lo, hi = (int(v) for v in rng.split("-"))
        seeds = [s for s in sorted(set(fa) & set(fb)) if lo <= s <= hi]
        if len(seeds) < 2:
            L.append(f"| {label} | {rng} | {len(seeds)} | | | | | (missing) |")
            continue
        xa, xb = np.array([fa[s] for s in seeds]), np.array([fb[s] for s in seeds])
        d = xb - xa
        h = t_dist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        p = wilcoxon(xb, xa).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
        L.append(f"| {label} | {rng} | {len(d)} | {xa.mean():.4f} | {xb.mean():.4f} | {d.mean():+.4f} | "
                 f"[{d.mean() - h:+.4f}, {d.mean() + h:+.4f}] | {p:.3g} |")
    L.append(f"\n{len(fit)} runs with a best fitness found.")
    open(f"{a.out}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
