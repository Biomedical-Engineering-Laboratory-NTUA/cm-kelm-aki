"""
Cohort, features and the fixed train/test split (single source of truth).
=========================================================================

Cohort criteria (inspire_preprocessing.apply_cohort_criteria), feature matrix,
KDIGO label, stratified 70/30 split with random_state=42, and median imputation
and standard scaling fitted on the training set only. Every model therefore sees
the same test patients, as the paired tests of stats_eval.py require.

esrd (exclusion of pre-existing kidney failure; cohort tag "esrd"):
  cr4_z49      baseline creatinine >= 4 mg/dL or pre-operative ICD-10 Z49 (primary)
  cr2_z49      baseline creatinine >= 2 mg/dL or Z49 (sensitivity analysis)
  cr4_z49_n18  cr4_z49 plus pre-operative N18 (sensitivity analysis)
features:
  preop  37 preoperative variables
  eos    + end-of-surgery variables of intraop_features.py (features_intraop.csv.gz, joined on op_id)
  eos2   eos without constant columns and the duplicated cumulative maxima (93 variables, final set)
  preop2 strictly preoperative set: preop without anesthesia duration, plus the 11 medication
         indicators of the 30 days before surgery (47 variables)
missing_ind: append a missing-value indicator for every variable with >10% missing values in the
  training set (computed before imputation; columns appended at the end, named miss_<variable>)
The split depends only on the cohort and the label, so the preoperative and
end-of-surgery models of one cohort share the same test patients. End-of-surgery
columns follow the preoperative ones, so the protected indices (0 = age,
5 = preop_creatinine) are unchanged. max_baseline_creatinine is a legacy option,
not used in the paper.
"""

import os

import pandas as pd

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from inspire_preprocessing import (
    apply_cohort_criteria, build_feature_matrix, compute_kdigo_aki_label, load_raw_tables,
)

# INSPIRE v1.4.2 folder (operations.csv.gz, labs.csv.gz, vitals.csv.gz, ...); set INSPIRE_DIR
DATA_DIR = os.environ.get("INSPIRE_DIR", "data/inspire/1.4.2")
SPLIT_SEED = 42  # fixed: defines the test set for the whole paper


def _load_tables(data_dir, dummy):
    if dummy:
        t = load_raw_tables(dummy=True)
        return t["operations"], t["labs"], t["diagnosis"]
    # only what the cohort and the preoperative features need (vitals.csv is large)
    return (pd.read_csv(f"{data_dir}/operations.csv.gz"), pd.read_csv(f"{data_dir}/labs.csv.gz"),
            pd.read_csv(f"{data_dir}/diagnosis.csv.gz"))


def load_split(data_dir=DATA_DIR, dummy=False, max_baseline_creatinine=None, verbose=True,
               esrd=None, features="preop", intraop_path=None, return_ids=False, missing_ind=False):
    operations, labs, diagnosis = _load_tables(data_dir, dummy)
    ops_final, labs = apply_cohort_criteria(
        operations, labs, first_procedure_only=True,
        max_baseline_creatinine=max_baseline_creatinine, esrd=esrd, diagnosis=diagnosis,
        verbose=verbose,
    )
    ops_final = ops_final.reset_index(drop=True)
    X_df, _ = build_feature_matrix(ops_final, labs)
    X_df = X_df.reset_index(drop=True)
    if features in ("eos", "eos2", "preop2"):
        path = intraop_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "features_intraop.csv.gz")
        intra = pd.read_csv(path).set_index("op_id")
        missing = set(ops_final["op_id"]) - set(intra.index)
        if missing:
            raise ValueError(f"{len(missing)} cohort operations missing from {path}: "
                             "re-run intraop_features.py with the same cohort options")
        X_df = pd.concat([X_df, intra.loc[ops_final["op_id"]].reset_index(drop=True)], axis=1)
        empty = [c for c in intra.columns if X_df[c].isna().all()]
        if empty:   # the imputer would drop them silently and misalign the names
            print(f"dropping all-missing end-of-surgery columns: {empty}", flush=True)
            X_df = X_df.drop(columns=empty)
        if features == "preop2":
            # strictly preoperative: drop anesthesia duration, keep only the preoperative
            # medication indicators of the end-of-surgery file (pre_contrast is constant)
            pre_meds = [c for c in intra.columns if c.startswith("pre_") and c != "pre_contrast"]
            X_df = X_df.drop(columns=[c for c in intra.columns if c not in pre_meds] + ["andur"],
                             errors="ignore")
        if features == "eos2":
            # final feature set: drop end-of-surgery columns with a single value
            # in the cohort (e.g. pre_contrast, no V08 codes) and the *_max duplicates of
            # incrementally charted volumes (ebl_max, uo_max; *_sum are the totals)
            const = [c for c in intra.columns if c in X_df and X_df[c].nunique(dropna=True) <= 1]
            dup = [c for c in ("ebl_max", "uo_max") if c in X_df]
            drop = sorted(set(const + dup))
            print(f"eos2: dropping {drop}", flush=True)
            X_df = X_df.drop(columns=drop)
    elif features != "preop":
        raise ValueError("features must be 'preop', 'preop2', 'eos' or 'eos2'")
    y = compute_kdigo_aki_label(ops_final, labs).reindex(ops_final["op_id"]).values

    X = X_df.values.astype(float)
    X[np.isinf(X)] = np.nan

    ids = ops_final["op_id"].values
    Xtr, Xte, ytr, yte, id_tr, id_te = train_test_split(
        X, y, ids, test_size=0.30, random_state=SPLIT_SEED, stratify=y)

    names = list(X_df.columns)
    if missing_ind:
        frac = np.isnan(Xtr).mean(axis=0)
        cols_mi = np.flatnonzero(frac > 0.10)
        Xtr = np.hstack([Xtr, np.isnan(Xtr[:, cols_mi]).astype(float)])
        Xte = np.hstack([Xte, np.isnan(Xte[:, cols_mi]).astype(float)])
        names += [f"miss_{names[j]}" for j in cols_mi]
        if verbose:
            print(f"missing indicators for {len(cols_mi)} variables: {[names[j] for j in cols_mi]}", flush=True)

    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    cohort_tag = "all" if max_baseline_creatinine is None else f"crmax{max_baseline_creatinine:g}"
    if esrd is not None:
        cohort_tag = "esrd" if esrd == "cr4_z49" else f"esrd-{esrd}"
        if max_baseline_creatinine is not None:
            cohort_tag += f"-crmax{max_baseline_creatinine:g}"
    if features in ("eos", "eos2", "preop2"):
        cohort_tag += f"_{features}"
    if missing_ind:
        cohort_tag += "_mi"
    if verbose:
        print(f"Train {Xtr_s.shape}, pos {ytr.mean():.4f} | Test {Xte_s.shape}, "
              f"pos {yte.mean():.4f} | cohort={cohort_tag}", flush=True)
    if return_ids:
        return Xtr_s, Xte_s, ytr.astype(int), yte.astype(int), names, cohort_tag, id_tr, id_te
    return Xtr_s, Xte_s, ytr.astype(int), yte.astype(int), names, cohort_tag
