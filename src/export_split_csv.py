"""
Export the paper's split as ONE CSV for external training pipelines (e.g. collaborators'
models) so that they are trained and evaluated on exactly the same patients.

Columns: the model features (already imputed and standardised exactly as in
cohort_split.load_split), `target` (0/1 AKI), `split` (train / val / test) and `row_id`
(the position of each row inside its split, so predictions can be mapped back; for the
test split it is the position in the paper's 15,146-case test set).

The validation part is the same stratified 15% of the training set used for the DL
baselines (run_dl_baselines.py, random_state = seed), so early stopping never sees
the test set.

The file contains patient-level data: it must stay on the lab server (PhysioNet DUA).
Write it inside a private directory (chmod 700).

Usage (server, CPU):
  python export_split_csv.py --esrd cr4_z49 --features eos2 \
      --out <private_dir>/inspire_esrd_eos2_split.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cohort_split import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2", choices=["preop", "eos", "eos2"])
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Xtr_all, Xte, ytr_all, yte, names, tag = load_split(dummy=a.dummy, esrd=a.esrd, features=a.features)
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=a.val_frac,
                                          stratify=ytr_all, random_state=a.seed)
    parts = []
    for split, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        df = pd.DataFrame(X.astype(np.float32), columns=[str(n) for n in names])
        df["target"] = y.astype(int)
        df["split"] = split
        df["row_id"] = np.arange(len(y))
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    os.chmod(a.out, 0o600)
    # labels of the test split in paper order, for checking the import later
    ytest_path = os.path.splitext(a.out)[0] + "_ytest.npy"
    np.save(ytest_path, yte.astype(np.int8))
    os.chmod(ytest_path, 0o600)
    c = df.groupby("split")["target"].agg(["size", "sum"])
    print(f"cohort {tag}: {len(names)} features\n{c}\nwritten {a.out}")
    if not a.dummy and len(yte) != 15146:
        print(f"WARNING: test set has {len(yte)} cases, expected 15146")


if __name__ == "__main__":
    main()
