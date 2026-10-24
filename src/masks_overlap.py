"""
Two descriptive checks for the review (aggregate output only).

1. Size of the variable set an ensemble needs: union of the masks of the island champions per run
   and over all runs of a configuration (the ensemble needs every variable in the union).
2. Overlap between the test set of the final cohort and the cohort of the development runs
   (same split seed on the cohort before the exclusion of pre-existing kidney failure).

    python masks_overlap.py --config cmkelm_m1000_gpu_p20_g15 --esrd cr4_z49 --features eos2
"""
import argparse
import glob
import json

import numpy as np

from cohort_split import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cmkelm_m1000_gpu_p20_g15")
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2")
    ap.add_argument("--out", default="masks_overlap")
    a = ap.parse_args()
    _, _, _, _, cols, tag, id_tr, id_te = load_split(esrd=a.esrd, features=a.features, verbose=False, return_ids=True)
    sfx = f"__{tag}"
    L = [f"# Mask unions and development overlap ({a.config}{sfx})\n"]
    union_all, per_run, best_sizes = set(), [], []
    for f in sorted(glob.glob(f"preds/{a.config}_s*{sfx}__popens_islands.npz")):
        mt = json.loads(str(np.load(f)["meta"]))
        u = set()
        for mb in mt["members"]:
            u |= set(mb["features"])
        per_run.append(len(u))
        union_all |= u
    L.append(f"- runs: {len(per_run)}; variables needed by one run's island ensemble: mean {np.mean(per_run):.1f}, "
             f"range {min(per_run)}-{max(per_run)} of {len(cols)}")
    L.append(f"- variables needed by the ten-run ensemble (union of all champions): {len(union_all)} of {len(cols)}")

    _, _, _, _, _, tag0, d_tr, d_te = load_split(esrd=None, features="preop", verbose=False, return_ids=True)
    te, tr = set(id_te), set(id_tr)
    dev_tr, dev_te = set(d_tr), set(d_te)
    L.append(f"- development cohort ({tag0}, current extraction): {len(dev_tr) + len(dev_te)} operations, "
             f"test {len(dev_te)}")
    L.append(f"- final test patients that were in the development TEST set: {len(te & dev_te)} of {len(te)} "
             f"({100 * len(te & dev_te) / len(te):.1f}%); in the development TRAINING set: {len(te & dev_tr)}")
    L.append(f"- final training patients in the development test set: {len(tr & dev_te)} of {len(tr)}")
    open(f"{a.out}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
