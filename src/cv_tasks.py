"""
Task lists of the confirmatory protocol (one command per line), run by slurm/cv_array.sh.
=========================================================================================

    python cv_tasks.py --wave 1 --backend gpu > tasks_w1.txt
    sbatch --array=1-$(wc -l < tasks_w1.txt)%2 --gres=gpu:1 ../slurm/cv_array.sh tasks_w1.txt

Waves (Section 9 of the protocol):
  1  primary hypotheses: M1 DE-KELM-ARD and M2 DE-KELM (10 seeds x 10 outer folds),
     M4 grid-tuned KELM with the classical baselines M6/M8 (one task per fold)
  2  M3 CM-KELM (5 seeds), M5 sparse GP (5 seeds), M7 FT-Transformer (5 seeds),
     random-search controls (repeat 1, 10 seeds)
  3  secondary outcomes (stage >= 2, stage 3): M1, M2 (10 seeds), M4/M6/M8, M5
  4  baseline sensitivity (90 and 7 days): M1, M2, M4
Every line sets CMKELM_CV (and CMKELM_OUTCOME / CMKELM_BASELINE_DAYS); outputs go to preds_cv/.
"""
import argparse

COMMON = "--m 1000 --esrd cr4_z49 --features eos2 --outdir preds_cv"
DE = "--pop 20 --gen 23 --no-cultural --no-memetic"          # plain DE, 480 evaluations


def env(r, k, outcome="aki", bdays=365):
    e = f"CMKELM_CV={r}-{k}"
    if outcome != "aki":
        e += f" CMKELM_OUTCOME={outcome}"
    if bdays != 365:
        e += f" CMKELM_BASELINE_DAYS={bdays}"
    return e


def folds():
    return [(r, k) for r in (1, 2) for k in range(1, 6)]


def m1(r, k, s, b, **kw):
    return f"{env(r, k, **kw)} python -u run_fullcohort_cmkelm_v4.py --backend {b} {DE} --ard --penalty 0 --seed {s} {COMMON}"


def m2(r, k, s, b, **kw):
    return f"{env(r, k, **kw)} python -u run_fullcohort_cmkelm_v4.py --backend {b} {DE} --seed {s} {COMMON}"


def m3(r, k, s, b):
    return f"{env(r, k)} python -u run_fullcohort_cmkelm_v4.py --backend {b} --pop 20 --gen 15 --seed {s} {COMMON}"


def base(r, k, **kw):
    return f"{env(r, k, **kw)} python -u run_baselines_preds.py --grid-m 1000 --wide-grid --esrd cr4_z49 --features eos2 --outdir preds_cv"


def gp(r, k, **kw):
    return f"{env(r, k, **kw)} python -u run_gp_ard.py --esrd cr4_z49 --features eos2 --outdir preds_cv --val-outdir preds_cv_val"


def ftt(r, k):
    return f"{env(r, k)} python -u run_tabdl.py --model ftt --esrd cr4_z49 --features eos2 --outdir preds_cv --val-outdir preds_cv_val"


def rand(r, k, s, b, ard):
    return (f"{env(r, k)} python -u random_search.py --backend {b} {'--ard ' if ard else ''}--seed {s} "
            f"--esrd cr4_z49 --features eos2 --outdir preds_cv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--part", choices=["search", "other", "all"], default="all",
                    help="search = evolutionary/random searches only; other = baselines, GP, FT-T")
    a = ap.parse_args()
    b, S, O = a.backend, [], []
    seeds10, seeds5 = range(1, 11), range(1, 6)
    if a.wave == 1:
        S += [m1(r, k, s, b) for r, k in folds() for s in seeds10]
        S += [m2(r, k, s, b) for r, k in folds() for s in seeds10]
        O += [base(r, k) for r, k in folds()]
    elif a.wave == 2:
        S += [m3(r, k, s, b) for r, k in folds() for s in seeds5]
        S += [rand(1, k, s, b, ard) for k in range(1, 6) for s in seeds10 for ard in (True, False)]
        O += [gp(r, k) for r, k in folds()] + [ftt(r, k) for r, k in folds()]
    elif a.wave == 3:
        for oc in ("st2", "st3"):
            S += [m1(r, k, s, b, outcome=oc) for r, k in folds() for s in seeds10]
            S += [m2(r, k, s, b, outcome=oc) for r, k in folds() for s in seeds10]
            O += [base(r, k, outcome=oc) for r, k in folds()] + [gp(r, k, outcome=oc) for r, k in folds()]
    elif a.wave == 4:
        for bd in (90, 7):
            S += [m1(r, k, s, b, bdays=bd) for r, k in folds() for s in seeds10]
            S += [m2(r, k, s, b, bdays=bd) for r, k in folds() for s in seeds10]
            O += [base(r, k, bdays=bd) for r, k in folds()]
    lines = (S if a.part in ("search", "all") else []) + (O if a.part in ("other", "all") else [])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
