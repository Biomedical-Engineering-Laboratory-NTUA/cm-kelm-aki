"""
Hyperparameters and kernel weights of every ensemble member of the confirmatory KELM models.
=============================================================================================

Writes one aggregate table (no patient data): for each model, outer fold, seed and ensemble
member, gamma, lambda and the 93 per-variable values (ARD weight w_j, or the 0/1 mask).
Together with the code, these values let a credentialed INSPIRE user rebuild every reported model
(the Nystrom landmarks are patient records and are therefore not released; they are drawn with
the fixed random_state of the code).

    python export_hyperparams.py --out ../results/confirmatory/hyperparameters.csv
"""
import argparse
import csv
import re

import numpy as np

import outer_cv
from cv_oof import files_of, meta

MODELS = ("M1", "M2", "M3", "CM-ARD", "M4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds_cv")
    ap.add_argument("--out", default="hyperparameters.csv")
    a = ap.parse_args()
    names = None
    n = 0
    with open(a.out, "w", newline="") as fh:
        w = None
        for r in (1, 2):
            for k in range(1, 6):
                Xtr, _, _, _, nm, tag = outer_cv.load_fold(r, k, "aki", verbose=False)
                if names is None:
                    names = list(nm)
                    w = csv.writer(fh)
                    w.writerow(["model", "repeat", "fold", "seed", "member", "encoding", "gamma", "lambda"] + names)
                for mid in MODELS:
                    for f in files_of(mid, tag, a.preds):
                        mt = meta(f)
                        s = re.search(r"_s(\d+)__", f)
                        seed = int(s.group(1)) if s else ""
                        if mid == "M4":
                            w.writerow([mid, r, k, seed, 1, "all", mt["gamma"], mt["lam"]] + [1] * len(names))
                            n += 1
                            continue
                        for i, mb in enumerate(mt["members"], 1):
                            if "w" in mb:
                                vals, enc = np.round(np.asarray(mb["w"], float), 6).tolist(), "ARD"
                            else:
                                v = np.zeros(len(names), int)
                                v[mb["features"]] = 1
                                vals, enc = v.tolist(), "mask"
                            w.writerow([mid, r, k, seed, i, enc, mb["gamma"], mb["lam"]] + vals)
                            n += 1
    print(f"written {a.out}: {n} rows")


if __name__ == "__main__":
    main()
