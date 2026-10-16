"""
Subgroup discrimination on the test set (by sex, urgency, age and surgical
department). Output is AGGREGATE only (n, events, AUROC/AUPRC with
bootstrap 95% CI per subgroup and model) and may leave the server.

Subgroups are read from the test-set design matrix of cohort_split.load_split
(standardized; binary indicators are > 0 when present):
  sex_M (male / female), emop (emergency / elective), age above / below the median,
  department_* (each department with >= --min-events AKI events in the test set).

Usage (server, CPU):
  python -u subgroup_eval.py --esrd cr4_z49 --features eos2 \
      preds_ens/cmkelm_m1000_gpu_p20_g15__esrd_eos2__popens_islands_ens.npz \
      preds/xgboost__esrd_eos2.npz preds/random_forest__esrd_eos2.npz \
      preds/nystrom_kelm_grid_m1000__esrd_eos2.npz --out subgroups_esrd_eos2
"""
import argparse
import csv

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cohort_split import load_split
from stats_eval import load_predictions


def boot_ci(y, p, fn, n_boot, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(fn(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2")
    ap.add_argument("--min-events", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out", default="subgroups")
    a = ap.parse_args()

    _, Xte, _, yte, names, _ = load_split(dummy=a.dummy, esrd=a.esrd, features=a.features)
    col = {n: j for j, n in enumerate(names)}
    groups = {"all": np.ones(len(yte), bool)}
    if "sex_M" in col:
        groups["male"] = Xte[:, col["sex_M"]] > 0
        groups["female"] = ~groups["male"]
    if "emop" in col:
        groups["emergency"] = Xte[:, col["emop"]] > 0
        groups["elective"] = ~groups["emergency"]
    if "age" in col:
        med = np.median(Xte[:, col["age"]])
        groups["age > median"] = Xte[:, col["age"]] > med
        groups["age <= median"] = ~groups["age > median"]
    for n in names:
        if str(n).startswith("department_"):
            g = Xte[:, col[n]] > 0
            if yte[g].sum() >= a.min_events and (yte[g] == 0).sum() >= a.min_events:
                groups[str(n).replace("department_", "dept ")] = g

    runs = load_predictions(a.files)
    rng = np.random.default_rng(0)
    rows = []
    for gname, g in groups.items():
        y = yte[g]
        if y.sum() < a.min_events and gname != "all":
            continue
        for mname, r in runs.items():
            if not np.array_equal(r["y"], yte):
                raise SystemExit(f"{mname}: labels/order differ from the test split")
            p = r["p"][g]
            au, ap_ = roc_auc_score(y, p), average_precision_score(y, p)
            lo, hi = boot_ci(y, p, roc_auc_score, a.n_boot, rng)
            rows.append(dict(subgroup=gname, n=int(g.sum()), events=int(y.sum()), model=mname,
                             auroc=round(au, 4), auroc_lo=round(lo, 4), auroc_hi=round(hi, 4),
                             auprc=round(ap_, 4), prevalence=round(float(y.mean()), 4)))
            print(f"{gname:16s} n={g.sum():6d} ev={y.sum():4d}  {mname[:45]:45s} "
                  f"AUROC {au:.3f} [{lo:.3f}, {hi:.3f}]  AUPRC {ap_:.3f}", flush=True)
    with open(f"{a.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written {a.out}.csv")


if __name__ == "__main__":
    main()
