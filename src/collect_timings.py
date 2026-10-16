"""
Collect run times of all CM-KELM / grid runs from preds/*.json into timings.csv
(aggregate run metadata only: name, backend, cohort tag, minutes, evaluations).

Usage:  python collect_timings.py            -> timings.csv + summary on screen
"""
import csv
import glob
import json
import os
import re


def main():
    rows = []
    for f in sorted(glob.glob("preds/*.json")):
        name = os.path.basename(f)[:-5]
        if not (name.startswith("cmkelm") or name.startswith("nystrom_kelm_grid")):
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        t = j.get("fit_time_min")
        if t is None and j.get("fit_time_s") is not None:
            t = j["fit_time_s"] / 60
        if t is None:
            continue
        ev, es = j.get("fitness_evals"), j.get("fitness_eval_seconds")
        m = re.search(r"_p(\d+)_g(\d+)", name)
        tag = name.split("__", 1)[1] if "__" in name else "all"
        rows.append(dict(
            name=name, backend="gpu" if "_gpu" in name else "cpu",
            multimeme=(re.search(r"_mm([a-z]+)", name).group(1) if "_mm" in name else ""),
            pop=m.group(1) if m else "", gen=m.group(2) if m else "", cohort=tag,
            n_features=j.get("n_features", ""), minutes=round(float(t), 3),
            evals=ev or "", sec_per_eval=round(es / ev, 4) if ev and es else ""))
    keys = ["name", "backend", "multimeme", "pop", "gen", "cohort", "n_features", "minutes", "evals", "sec_per_eval"]
    with open("timings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> timings.csv")
    groups = {}
    for r in rows:
        k = (r["backend"], r["pop"], r["gen"], r["cohort"], r["multimeme"])
        groups.setdefault(k, []).append(r["minutes"])
    print(f"{'backend':7s} {'pop':>4s} {'gen':>4s} {'cohort':22s} {'mm':8s} {'n':>3s} {'median min':>10s}")
    for k, v in sorted(groups.items()):
        v = sorted(v)
        print(f"{k[0]:7s} {k[1]:>4s} {k[2]:>4s} {k[3]:22s} {k[4]:8s} {len(v):3d} {v[len(v) // 2]:10.2f}")


if __name__ == "__main__":
    main()
