"""
Full-cohort CM-KELM (Nystrom) -- v4: parameterized, multi-seed, saves predictions.
================================================================================

Replaces the one-file-per-run pattern (v1/v2/v3) with command-line arguments,
and fixes two issues found on 25 Sep 2026:

  1. Output buffering: v3 was piped through `tee`, so Python block-buffered
     stdout and the log stayed at 0 bytes until the end. Every print here
     uses flush=True; run with `python -u` as well.
  2. No saved predictions: v1-v3 printed AUROC/AUPRC only, so no CI or
     paired test (DeLong) was possible afterwards. v4 saves
     preds/cmkelm_m{m}_p{pop}_g{gen}_s{seed}.npz for stats_eval.py.

The test set is ALWAYS the same (split seed 42, see cohort_split.py).
--seed only changes the evolutionary search (initial population, landmarks,
DE randomness), so runs with different --seed are directly comparable and
give the seed-to-seed spread the paper needs.

Examples (via srun/sbatch):
    python -u run_fullcohort_cmkelm_v4.py --m 1000 --pop 12 --gen 8 --seed 42   # = Run 3, with preds
    python -u run_fullcohort_cmkelm_v4.py --m 500  --pop 20 --gen 15 --seed 42  # = Run 2, with preds
    for s in 1 2 3; do python -u run_fullcohort_cmkelm_v4.py --m 1000 --pop 12 --gen 8 --seed $s; done
"""

import argparse
import json
import os
import time

from sklearn.metrics import average_precision_score, roc_auc_score

from cohort_split import load_split
from adaptive_cultural import AdaptiveCulturalCMKELM
from nystrom_cmkelm import FullCohortCMKELM
from population_ensemble import population_ensemble_predict
from stats_eval import save_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42, help="search seed (split is fixed at 42)")
    ap.add_argument("--adaptive", default=None, metavar="FLAGS",
                    help="use AdaptiveCulturalCMKELM with any of K (adaptive knowledge sources), "
                         "A (diversity-driven acceptance), H (heterogeneous islands), "
                         "R (re-drawn CV folds per generation), L (+ landmarks), Z (fuzzy acceptance), "
                         "e.g. KAH or R or RZ; "
                         "'none' = adaptive class with all flags off (static scheme)")
    ap.add_argument("--backend", choices=["cpu", "gpu", "cpu-xp"], default="cpu",
                    help="where the fitness evaluation runs (gpu needs CuPy and --gres=gpu:1); "
                         "cpu-xp = the GPU code path on NumPy, for validation only")
    ap.add_argument("--precision", choices=["fp64", "mixed"], default="fp64",
                    help="GPU only: 'mixed' = FP32 kernel/GEMMs, FP64 eigh/solve (validate first)")
    ap.add_argument("--rank", type=int, default=None, help="rank-k Nystrom (default: full m)")
    ap.add_argument("--eig-rtol", type=float, default=None, help="drop eigenvalues <= rtol*max")
    ap.add_argument("--max-baseline-creatinine", type=float, default=None)
    ap.add_argument("--esrd", default=None, choices=["cr4_z49", "cr2_z49", "cr4_z49_n18"],
                    help="exclude pre-existing kidney failure (cohort tag 'esrd')")
    ap.add_argument("--features", default="preop", choices=["preop", "eos", "eos2"],
                    help="eos = + end-of-surgery features (features_intraop.csv.gz)")
    ap.add_argument("--multimeme", default=None,
                    choices=["fuzzy", "credit", "fc", "uniform", "hc", "tabu", "sa"],
                    help="hybrid multimeme local search with per-island meme selection (hybrid_memes.py); "
                         "implies the adaptive class, flags from --adaptive (default none)")
    ap.add_argument("--meme-budget", type=int, default=3, help="evaluations per island per generation")
    ap.add_argument("--pop-ens", nargs="*", default=["topk:5", "islands", "diverse:5"],
                    help="population ensembles to build from this single run (empty to skip)")
    ap.add_argument("--no-cultural", action="store_true",
                    help="ablation: plain DE without belief space (no norms, exemplar pull, islands)")
    ap.add_argument("--no-memetic", action="store_true", help="ablation: no Gaussian hill-climb")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--outdir", default="preds")
    a = ap.parse_args()

    if a.multimeme and not a.adaptive:
        a.adaptive = "none"
    Xtr, Xte, ytr, yte, cols, tag = load_split(dummy=a.dummy,
                                               max_baseline_creatinine=a.max_baseline_creatinine,
                                            esrd=a.esrd, features=a.features)
    m = min(a.m, len(ytr) // 3) if a.dummy else a.m
    name = (f"cmkelm_m{a.m}" + (f"_k{a.rank}" if a.rank else "") + (f"_rt{a.eig_rtol:g}" if a.eig_rtol else "")
            + (f"_ad{a.adaptive}" if a.adaptive and not a.multimeme else "")
            + (f"_mm{a.multimeme}" + ("" if a.adaptive.lower() == "none" else a.adaptive.upper())
               + ("" if a.meme_budget == 3 else f"b{a.meme_budget}") if a.multimeme else "")
            + ("" if a.backend == "cpu" else ("_cpuxp" if a.backend == "cpu-xp" else
                                         "_gpu" if a.precision == "fp64" else "_gpumix"))
            + ("_nocult" if a.no_cultural else "") + ("_nomem" if a.no_memetic else "")
            + f"_p{a.pop}_g{a.gen}_s{a.seed}" + ("" if tag == "all" else f"__{tag}"))
    print(f"\n{name}: population={a.pop}, generations={a.gen}, n_landmarks={m}, seed={a.seed}", flush=True)
    print("(protecting feature indices 0=age, 5=preop_creatinine from exclusion)\n", flush=True)

    t0 = time.time()
    common = dict(population_size=a.pop, generations=a.gen, n_landmarks=m,
                  protected_feature_indices=[0, 5], n_cells=4, migration_every=5,
                  random_state=a.seed, rank=a.rank, eig_rtol=a.eig_rtol,
                  backend=a.backend, precision=a.precision,
                  use_cultural=not a.no_cultural, use_memetic=not a.no_memetic)
    if a.adaptive:
        fl = "" if a.adaptive.lower() == "none" else a.adaptive.upper()
        model = AdaptiveCulturalCMKELM(adaptive_ks="K" in fl, adaptive_accept="A" in fl,
                                       hetero_islands="H" in fl, robust_fitness="R" in fl,
                                       robust_landmarks="L" in fl, fuzzy="Z" in fl,
                                       multimeme=a.multimeme, meme_budget=a.meme_budget,
                                       **{k: v for k, v in common.items() if k != "use_cultural"})
    else:
        model = FullCohortCMKELM(**common)
    model.fit(Xtr, ytr)
    fit_time = time.time() - t0
    if hasattr(model, "fitness_"):
        print("fitness evaluations:", model.fitness_.summary(), flush=True)

    p = model.predict_proba(Xte)
    auroc, auprc = roc_auc_score(yte, p), average_precision_score(yte, p)
    active = [c for c, keep in zip(cols, model.best_mask_) if keep]

    print("\n" + "=" * 70, flush=True)
    print(f"RESULT {name}", flush=True)
    print(f"AUROC {auroc:.4f} | AUPRC {auprc:.4f} | fit {fit_time / 60:.1f} min", flush=True)
    print(f"gamma {model.best_gamma_:.4f}, lambda {model.best_lambda_:.4f}, "
          f"active features {len(active)}/{len(cols)}", flush=True)
    print("=" * 70, flush=True)

    save_predictions(f"{a.outdir}/{name}.npz", name, yte, p,
                     n_landmarks=a.m, rank=a.rank, eig_rtol=a.eig_rtol, population=a.pop, generations=a.gen, seed=a.seed,
                     gamma=float(model.best_gamma_), lam=float(model.best_lambda_),
                     active_features=active, fit_time_s=fit_time, cohort=tag, split_seed=42)
    # population ensembles from this same run (no extra search, k final fits)
    pop_results = {}
    for strat in a.pop_ens:
        t1 = time.time()
        p_e, members = population_ensemble_predict(model, Xtr, ytr, Xte, strat)
        tag_e = strat.replace(":", "")
        name_e = f"{name}__popens_{tag_e}"
        au_e, ap_e = roc_auc_score(yte, p_e), average_precision_score(yte, p_e)
        pop_results[tag_e] = {"auroc": au_e, "auprc": ap_e, "n_members": len(members)}
        print(f"POP-ENSEMBLE {strat:10s} ({len(members)} members): AUROC {au_e:.4f} | AUPRC {ap_e:.4f} "
              f"| +{time.time() - t1:.1f}s", flush=True)
        save_predictions(f"{a.outdir}/{name_e}.npz", name_e, yte, p_e, strategy=strat,
                         members=members, n_landmarks=a.m, population=a.pop, generations=a.gen,
                         seed=a.seed, cohort=tag, split_seed=42)

    os.makedirs(a.outdir, exist_ok=True)
    with open(f"{a.outdir}/{name}.json", "w") as fh:
        json.dump({"auroc": auroc, "auprc": auprc, "fit_time_min": fit_time / 60,
                   "active_features": active, "population_ensembles": pop_results,
                   "backend": a.backend, "precision": a.precision,
                   "fitness_eval_seconds": getattr(getattr(model, "fitness_", None), "seconds", None),
                   "fitness_evals": getattr(getattr(model, "fitness_", None), "n_evals", None),
                   "search_evals": getattr(model, "n_evals_", None),
                   "meme_usage": getattr(model, "meme_usage_", None),
                   "trace": getattr(model, "trace_", None)}, fh, indent=2)


if __name__ == "__main__":
    main()
