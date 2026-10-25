"""
Random search over variable masks and kernel parameters at the CM-KELM budget (control).
========================================================================================

Answers whether the evolutionary search, rather than ensembling of random subsets, produces
the gain: 500 individuals (the evaluation budget of one CM-KELM run) are drawn from the same
initial distribution as the DE population (log-uniform gamma in [1e-3, 10] and lambda in
[1e-3, 100], selection genes z_j ~ N(0, 1), variable j selected when z_j > 0, age and creatinine
always selected) and scored with the same fitness (3-fold CV AUPRC with the same folds and
landmarks, minus 0.001 per selected variable). The individuals are split into four groups of
125, mirroring the four islands, and the champions of the groups form the 'islands' ensemble;
the five best form 'topk5'. Output files follow run_fullcohort_cmkelm_v4.py, so that
compare_seeds.py and summarize_seeds.py treat the runs like any other configuration:

    python -u random_search.py --backend gpu --seed 1 --esrd cr4_z49 --features eos2
    -> preds/cmkelm_m1000_gpu_randsearch_n500_s1__esrd_eos2{.npz,.json,__popens_islands.npz,...}

With --ard the variable genes are decoded into continuous kernel weights w_j = exp(z_j)
(kelm_variant.ARDEncoding, as in the ARD control of run_fullcohort_cmkelm_v4.py) and the
penalty is 0; the name gets "_ard" (cmkelm_m1000_gpu_randsearch_ard_n500_s1__esrd_eos2).
This separates the effect of the continuous weights from that of the evolutionary search.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cmeelm import CulturalBeliefSpace
from cohort_split import load_split
from kelm_variant import ARDEncoding, KELMEncoding
from nystrom_cmkelm import nystrom_kelm_fitness
from nystrom_kelm import NystromKELM
from stats_eval import save_predictions


def fit_predict(members, Xtr, ytr, Xte, m):
    ps = []
    for mbr in members:
        mask = mbr["mask"]
        kelm = NystromKELM(gamma=mbr["gamma"], lam=mbr["lam"], n_landmarks=m, random_state=0)
        ps.append(kelm.fit(Xtr * mask, ytr).predict_proba(Xte * mask))
    return np.mean(ps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--n", type=int, default=500, help="evaluations (CM-KELM: 500 per run)")
    ap.add_argument("--groups", type=int, default=4, help="groups mirroring the islands")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--esrd", default=None, choices=["cr4_z49", "cr2_z49", "cr4_z49_n18"])
    ap.add_argument("--features", default="eos2", choices=["preop", "preop2", "eos", "eos2"])
    ap.add_argument("--penalty", type=float, default=0.001)
    ap.add_argument("--ard", action="store_true", help="continuous per-variable weights (ARD), penalty 0")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    a = ap.parse_args()
    if a.ard:
        a.penalty = 0.0

    Xtr, Xte, ytr, yte, cols, tag = load_split(dummy=a.dummy, esrd=a.esrd, features=a.features)
    m = min(a.m, len(ytr) // 3) if a.dummy else a.m
    name = (f"cmkelm_m{a.m}" + ("_gpu" if a.backend == "gpu" else "") + f"_randsearch{'_ard' if a.ard else ''}_n{a.n}"
            + ("" if a.penalty == 0.001 or a.ard else f"_pen{a.penalty:g}") + f"_s{a.seed}"
            + ("" if tag == "all" else f"__{tag}"))
    enc = (ARDEncoding if a.ard else KELMEncoding)(len(cols))
    on = (lambda w: w >= 0.1) if a.ard else (lambda w: w > 0)   # ARD: weight >= 0.1 counts as selected
    protected = [cols.index("age"), cols.index("preop_creatinine")]
    belief = CulturalBeliefSpace(enc.total_size, protected_feature_indices=protected, mask_gene_offset=2)
    if a.backend == "gpu":
        from gpu_fitness import NystromFitness
        fit_fn = NystromFitness(Xtr, ytr, enc, m, a.penalty, backend="gpu")
    else:
        fit_fn = lambda g: nystrom_kelm_fitness(g, Xtr, ytr, enc, m, a.penalty)

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    pop = [belief.enforce_domain_knowledge(enc.random_individual(rng)) for _ in range(a.n)]
    fit = np.array([fit_fn(g) for g in pop])
    search_min = (time.time() - t0) / 60
    dec = []
    for g, f in zip(pop, fit):
        gamma, lam, mask = enc.decode(g)
        dec.append(dict(gamma=float(gamma), lam=float(lam), mask=mask, fitness=float(f)))
    order = np.argsort(-fit)
    best = dec[order[0]]
    groups = np.array_split(np.arange(a.n), a.groups)
    champs = [dec[g[np.argmax(fit[g])]] for g in groups]
    print(f"{name}: {a.n} evaluations in {search_min:.1f} min, best fitness {best['fitness']:.4f}, "
          f"{int(on(best['mask']).sum())} variables", flush=True)

    p = fit_predict([best], Xtr, ytr, Xte, m)
    res = {"auroc": roc_auc_score(yte, p), "auprc": average_precision_score(yte, p)}
    active = [c for c, k in zip(cols, best["mask"]) if on(k)]
    save_predictions(f"{a.outdir}/{name}.npz", name, yte, p, n_landmarks=a.m, seed=a.seed, n_evals=a.n,
                     gamma=best["gamma"], lam=best["lam"], active_features=active, cohort=tag, split_seed=42)
    pop_results = {}
    for tag_e, members in (("islands", champs), ("topk5", [dec[i] for i in order[:5]])):
        p_e = fit_predict(members, Xtr, ytr, Xte, m)
        pop_results[tag_e] = {"auroc": roc_auc_score(yte, p_e), "auprc": average_precision_score(yte, p_e),
                              "n_members": len(members)}
        info = [dict(gamma=mb["gamma"], lam=mb["lam"], n_features=int(on(mb["mask"]).sum()),
                     features=np.flatnonzero(on(mb["mask"])).tolist(), fitness=mb["fitness"],
                     **({"weights": np.round(mb["mask"], 4).tolist()} if a.ard else {})) for mb in members]
        save_predictions(f"{a.outdir}/{name}__popens_{tag_e}.npz", f"{name}__popens_{tag_e}", yte, p_e,
                         strategy=tag_e, members=info, n_landmarks=a.m, seed=a.seed, cohort=tag, split_seed=42)
    print(f"RESULT {name}: single {res['auroc']:.4f}/{res['auprc']:.4f} | "
          + " | ".join(f"{k} {v['auroc']:.4f}/{v['auprc']:.4f}" for k, v in pop_results.items()), flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    with open(f"{a.outdir}/{name}.json", "w") as fh:
        json.dump({**res, "best_fitness": best["fitness"], "fit_time_min": search_min, "active_features": active,
                   "population_ensembles": pop_results, "search_evals": a.n,
                   "fitness_quantiles": np.quantile(fit, [0, 0.5, 0.9, 1]).tolist()}, fh, indent=2)


if __name__ == "__main__":
    main()
