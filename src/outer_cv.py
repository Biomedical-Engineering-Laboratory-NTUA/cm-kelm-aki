"""
Outer cross-validation of the confirmatory protocol (docs/PROTOCOL_confirmatory.md, Section 4).
===============================================================================================

Whole cohort (rule cr4_z49, 93 end-of-surgery variables), stratified 5-fold cross-validation
repeated twice (fold seeds 101 and 102). Strata: KDIGO stage (0, 1, >= 2) x surgical group
(cardiac surgery with cardiopulmonary bypass, urology, other). Imputation and scaling are fitted
on the training folds only.

Every existing runner uses it through cohort_split.load_split when the environment variable
CMKELM_CV="r-k" is set (repeat r = 1, 2; fold k = 1..5); CMKELM_OUTCOME selects the outcome:
  aki  KDIGO stage >= 1 (primary)      st2  stage >= 2      st3  stage 3
and CMKELM_BASELINE_DAYS (default 365) restricts the baseline creatinine (sensitivity analyses;
patients without a baseline in the window are left out of training and test folds).
The cohort tag of the output files becomes e.g. esrd_eos2_cv13 or esrd_eos2_cv13_st2.

    CMKELM_CV=1-3 python -u run_fullcohort_cmkelm_v4.py ... --outdir preds_cv
    python outer_cv.py --prepare          # build the cache (patient-level, server only: cv_cache/)
    python outer_cv.py --balance          # aggregate balance of every fold (Section 4)
    python outer_cv.py --dummy --balance  # self-test
"""
import argparse
import os

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5
FOLD_SEEDS = (101, 102)
CACHE_DIR = "cv_cache"   # patient-level arrays: never leave the server, never committed


def kdigo_stage(ops, labs, baseline_days=365, window_days=7):
    """KDIGO stage from creatinine (Section 3): 1 >= 1.5x, 2 >= 2.0x, 3 >= 3.0x or peak >= 4.0 mg/dL
    in a patient who meets stage 1. Baseline: most recent value before anesthesia start within
    baseline_days. Returns stage (NaN where no baseline in the window) and the baseline value,
    both aligned with ops."""
    from inspire_preprocessing import CREATININE_ITEM_NAME, extract_preop_lab
    base = extract_preop_lab(labs, ops, CREATININE_ITEM_NAME, lookback_days=baseline_days)
    cr = labs[labs["item_name"] == CREATININE_ITEM_NAME][["subject_id", "chart_time", "value"]]
    m = cr.merge(ops[["op_id", "subject_id", "orout_time"]], on="subject_id")
    dt = (m["chart_time"] - m["orout_time"]) / (24 * 60)
    peak = m[(dt >= 0) & (dt <= window_days)].groupby("op_id")["value"].max()
    b = base.reindex(ops["op_id"]).values.astype(float)
    p = peak.reindex(ops["op_id"]).values.astype(float)
    ratio = p / b
    stage = np.select([ratio >= 3.0, ratio >= 2.0, ratio >= 1.5], [3, 2, 1], 0).astype(float)
    stage = np.where((stage >= 1) & (p >= 4.0), 3.0, stage)
    stage[~np.isfinite(ratio) | (b <= 0)] = np.nan
    return stage, b


def _cache_path(tag, baseline_days):
    return os.path.join(CACHE_DIR, f"cohort_{tag}" + ("" if baseline_days == 365 else f"_b{baseline_days}") + ".npz")


def prepare(data_dir=None, dummy=False, esrd="cr4_z49", features="eos2", intraop_path=None,
            baseline_days=365, verbose=True, use_cache=True):
    """Raw matrix, stages and the variables of the strata and subgroups; cached on disk."""
    import cohort_split as cs
    if dummy:
        esrd, features = None, "preop"
    tag = cs.cohort_tag_of(esrd, features)
    path = _cache_path(tag, baseline_days)
    if use_cache and not dummy and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files} | {"names": list(z["names"]), "tag": tag}
    X, y, names, ops, labs = cs.cohort_matrix(data_dir or cs.DATA_DIR, dummy, None, verbose, esrd,
                                             features, intraop_path)
    stage, base = kdigo_stage(ops, labs, baseline_days)
    if baseline_days == 365:   # same label as the manuscript
        assert np.array_equal((stage >= 1), (y == 1)), "stage >= 1 differs from compute_kdigo_aki_label"
    dep = ops["department"].astype(str).values
    cpb = (ops["cpbon_time"].notna() & ops["cpboff_time"].notna()
           & ((ops["cpboff_time"] - ops["cpbon_time"]) > 0)).values
    group = np.where(cpb, 0, np.where(dep == "UR", 1, 2))
    age = ops["age"].astype(float).values
    female = ops["sex"].astype(str).str.upper().str.startswith("F").values
    kk, aa = np.where(female, 0.7, 0.9), np.where(female, -0.241, -0.302)
    r = base / kk
    egfr = 142 * np.minimum(r, 1) ** aa * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1.0)
    d = dict(X=X, stage=stage, group=group, age=age, emop=ops["emop"].fillna(0).astype(int).values,
             egfr=egfr, op_id=ops["op_id"].values, names=np.array(names))
    if not dummy:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, **d)
    return d | {"names": names, "tag": tag}


def fold_ids(stage, group):
    """Fold number (1..5) of every patient for each repeat; patients without a stage get 0."""
    ok = np.isfinite(stage)
    strata = np.minimum(np.nan_to_num(stage), 2).astype(int) * 3 + group
    out = {}
    for r, seed in enumerate(FOLD_SEEDS, start=1):
        f = np.zeros(len(stage), int)
        idx = np.flatnonzero(ok)
        for k, (_, te) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
                                    .split(idx, strata[idx]), start=1):
            f[idx[te]] = k
        out[r] = f
    return out


def label(stage, outcome):
    t = {"aki": 1, "st2": 2, "st3": 3}[outcome]
    return (stage >= t).astype(int)


def load_fold(r, k, outcome="aki", data_dir=None, dummy=False, esrd="cr4_z49", features="eos2",
              intraop_path=None, verbose=True, return_ids=False):
    baseline_days = int(os.environ.get("CMKELM_BASELINE_DAYS", "365"))
    d = prepare(data_dir, dummy, esrd, features, intraop_path, baseline_days, verbose=False)
    folds = fold_ids(d["stage"], d["group"])[r]
    tr, te = (folds > 0) & (folds != k), folds == k
    y = label(np.nan_to_num(d["stage"]), outcome)
    imp = SimpleImputer(strategy="median").fit(d["X"][tr])
    Xtr, Xte = imp.transform(d["X"][tr]), imp.transform(d["X"][te])
    sc = StandardScaler().fit(Xtr)
    tag = d["tag"] + f"_cv{r}{k}" + ("" if outcome == "aki" else f"_{outcome}") \
        + ("" if baseline_days == 365 else f"_b{baseline_days}")
    if verbose:
        print(f"outer CV repeat {r} fold {k} ({outcome}): train {Xtr.shape}, pos {y[tr].mean():.4f} | "
              f"test {Xte.shape}, pos {y[te].mean():.4f} | cohort={tag}", flush=True)
    out = (sc.transform(Xtr), sc.transform(Xte), y[tr], y[te], list(d["names"]), tag)
    if return_ids:
        out += (d["op_id"][tr], d["op_id"][te])
    return out


def balance(d):
    """Aggregate balance of the folds (Section 4): n, prevalence, subgroup sizes."""
    st = np.nan_to_num(d["stage"])
    sub = {"age>=65": d["age"] >= 65, "emergency": d["emop"] == 1, "CKD": d["egfr"] < 60,
           "cardiac CPB": d["group"] == 0, "urology": d["group"] == 1}
    print("repeat fold      n  stage>=1%  stage>=2%  stage3%  " + "  ".join(f"{s:>11}" for s in sub))
    for r, f in fold_ids(d["stage"], d["group"]).items():
        for k in range(1, N_SPLITS + 1):
            m = f == k
            print(f"{r:6d} {k:4d} {m.sum():6d} {100 * (st[m] >= 1).mean():9.2f} {100 * (st[m] >= 2).mean():9.2f} "
                  f"{100 * (st[m] >= 3).mean():8.2f}  " + "  ".join(f"{int((s & m).sum()):11d}" for s in sub.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--baseline-days", type=int, default=365)
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    d = prepare(dummy=a.dummy, baseline_days=a.baseline_days, use_cache=not a.prepare)
    print(f"cohort {d['tag']}: n = {len(d['stage'])}, with stage {int(np.isfinite(d['stage']).sum())}, "
          f"{d['X'].shape[1]} variables")
    if a.balance:
        balance(d)
