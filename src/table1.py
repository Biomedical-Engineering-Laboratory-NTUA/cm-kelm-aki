"""
Table 1 (cohort characteristics) and attrition for the paper.
=============================================================

Uses exactly the cohort, label and 70/30 split of cohort_split.load_split
(same criteria, same SPLIT_SEED, stratified on the label), but describes the
RAW feature values (before imputation and scaling).

Per variable:
  binary (values 0/1 only)  n (%) of 1s
  continuous                median [IQR]
  plus % missing, the standardized mean difference (SMD) AKI vs no AKI, and a p-value
  (Mann-Whitney U for continuous, chi-square for binary). SMD is mean-based and can be
  misleading for skewed variables (e.g. anaesthesia duration); the p-value is rank-based.

Columns: Overall | Train | Test | AKI | No AKI | SMD | missing %.
Counts below 5 are shown as "<5" (small-cell suppression; output is aggregate only).

Usage (on the server, via sbatch):
  python table1.py --esrd cr4_z49 --features eos2 --out table1_esrd_eos2
  python table1.py --dummy --features preop --out /tmp/t1      # self-test on synthetic tables
"""

import argparse
import contextlib
import io
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cohort_split import DATA_DIR, SPLIT_SEED, _load_tables
from inspire_preprocessing import apply_cohort_criteria, build_feature_matrix, compute_kdigo_aki_label

SMALL = 5


def fmt_n(k, n):
    if 0 < k < SMALL:
        return f"<{SMALL}"
    return f"{k} ({100 * k / n:.1f})" if n else "-"


def describe(s, binary):
    s = s.dropna()
    if len(s) == 0:
        return "-"
    if binary:
        return fmt_n(int(s.sum()), len(s))
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    return f"{q2:.3g} [{q1:.3g}, {q3:.3g}]"


def smd(a, b, binary):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if binary:
        p1, p2 = a.mean(), b.mean()
        d = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    else:
        p1, p2 = a.mean(), b.mean()
        d = np.sqrt((a.var() + b.var()) / 2)
    return abs(p1 - p2) / d if d > 0 else 0.0


def pval(a, b, binary):
    from scipy.stats import chi2_contingency, mannwhitneyu
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if binary:
        t = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        if (t.sum(axis=0) == 0).any():
            return np.nan
        return chi2_contingency(t)[1]
    return mannwhitneyu(a, b).pvalue


def fmt_p(p):
    if not np.isfinite(p):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esrd", default="cr4_z49")
    ap.add_argument("--features", default="eos2", choices=["preop", "eos", "eos2"])
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--intraop", default=None)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out", default="table1")
    a = ap.parse_args()

    operations, labs, diagnosis = _load_tables(a.data_dir, a.dummy)
    n_ops_raw = len(operations)
    n_pat_raw = operations["subject_id"].nunique() if "subject_id" in operations else None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ops, labs = apply_cohort_criteria(operations, labs, first_procedure_only=True,
                                          esrd=(None if a.esrd in ("", "none") else a.esrd),
                                          diagnosis=diagnosis, verbose=True)
    attrition = buf.getvalue()
    ops = ops.reset_index(drop=True)
    X, _ = build_feature_matrix(ops, labs)
    X = X.reset_index(drop=True)
    if a.features in ("eos", "eos2"):
        path = a.intraop or os.path.join(os.path.dirname(os.path.abspath(__file__)), "features_intraop.csv.gz")
        intra = pd.read_csv(path).set_index("op_id")
        X = pd.concat([X, intra.loc[ops["op_id"]].reset_index(drop=True)], axis=1)
        X = X.drop(columns=[c for c in intra.columns if X[c].isna().all()])
        if a.features == "eos2":
            drop = [c for c in intra.columns if c in X and X[c].nunique(dropna=True) <= 1]
            drop += [c for c in ("ebl_max", "uo_max") if c in X]
            X = X.drop(columns=sorted(set(drop)))
    X = X.replace([np.inf, -np.inf], np.nan)
    y = compute_kdigo_aki_label(ops, labs).reindex(ops["op_id"]).values.astype(int)

    # identical split to cohort_split.load_split (stratified permutation depends only on n, y, seed)
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.30, random_state=SPLIT_SEED, stratify=y)
    is_test = np.zeros(len(y), bool)
    is_test[i_te] = True

    groups = {"Overall": np.ones(len(y), bool), "Train": ~is_test, "Test": is_test,
              "AKI": y == 1, "No AKI": y == 0}
    rows = []
    rows.append(["Patients (operations)"] + [f"{g.sum()}" for g in groups.values()] + ["", "", ""])
    rows.append(["AKI (KDIGO), n (%)"] + [fmt_n(int(y[g].sum()), int(g.sum())) for g in groups.values()]
                + ["", "", ""])
    for c in X.columns:
        s = X[c]
        vals = set(pd.unique(s.dropna()))
        binary = len(vals) > 0 and vals <= {0, 1, 0.0, 1.0}
        label = f"{c}, n (%)" if binary else f"{c}, median [IQR]"
        rows.append([label] + [describe(s[g], binary) for g in groups.values()]
                    + [f"{smd(s[y == 1], s[y == 0], binary):.2f}", fmt_p(pval(s[y == 1], s[y == 0], binary)),
                       f"{100 * s.isna().mean():.1f}"])
    cols = ["Variable"] + list(groups) + ["SMD (AKI vs no AKI)", "p (AKI vs no AKI)", "missing %"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(f"{a.out}.csv", index=False)

    L = [f"# Table 1. Cohort characteristics ({a.features}, ESRD rule {a.esrd})\n",
         f"Source: {n_ops_raw} operations" + (f", {n_pat_raw} patients" if n_pat_raw else "")
         + " before criteria. Split: stratified 70/30, seed "
         f"{SPLIT_SEED} (same as all models). Raw values before imputation. "
         f"Counts below {SMALL} suppressed. Lab and vital values are INSPIRE 5th-percentile codes.\n",
         "## Attrition\n", "```", attrition.strip(), "```\n", "## Characteristics\n",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    open(f"{a.out}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"... {len(rows)} rows\nwritten {a.out}.md, {a.out}.csv")


if __name__ == "__main__":
    main()
