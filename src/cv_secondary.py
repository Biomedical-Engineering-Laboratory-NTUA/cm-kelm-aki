#!/usr/bin/env python
"""Secondary analyses of the confirmatory protocol (Sections 7-9), computed from the
out-of-fold predictions of the 5x2 outer cross-validation.

Parts
  calibration  : raw and recalibrated (Platt, cross-fitted inside each test fold)
                 Brier, calibration intercept and slope, ICI; calibration plot
  thresholds   : 5/10/20% risk thresholds (flagged, sens, spec, PPV, NPV, net benefit)
                 and fixed-sensitivity operating points (80/90%, threshold chosen on one
                 half of the test fold and applied to the other half)
  dca          : decision curves (pooled per repeat, recalibrated)
  subgroups    : CPB cardiac surgery, urology, other, age >= 65, emergency, CKD
                 (eGFR < 60); discrimination per model and M1 differences
  ard          : ARD relevance of M1 (gamma * w_j^2, normalised), GP relevance,
                 Kendall's W across folds, Spearman / top-10 overlap M1 vs GP

Only aggregate results are written. Usage:
  python cv_secondary.py --outcome aki --out cv_sec_aki
"""
import argparse, glob, json, os, sys, warnings
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
import outer_cv

RATIO = 1.0 / 4.0          # n_test / n_train for 5-fold CV (Nadeau-Bengio)
THRESH = [0.05, 0.10, 0.20]
SENS = [0.80, 0.90]
EPS = 1e-6


# ----------------------------------------------------------------------------- models
def model_patterns():
    """Model globs from cv_eval.py, so both scripts read exactly the same files."""
    import cv_eval
    M = cv_eval.MODELS
    out = {}
    for key, v in M.items():
        label, pat = key, None
        if isinstance(v, str):
            pat = v
        elif isinstance(v, (tuple, list)):
            strs = [s for s in v if isinstance(s, str)]
            pat = next((s for s in strs if "{tag}" in s or "*" in s), None)
            other = [s for s in strs if s is not pat]
            if other:
                label = f"{key} {other[0]}"
        elif isinstance(v, dict):
            pat = v.get("glob") or v.get("pattern") or v.get("pat")
            label = f"{key} {v.get('label', v.get('name', ''))}".strip()
        if pat is None:
            print("WARNING: cannot parse MODELS entry", key, v)
            continue
        out[key] = (label, pat)
    return out


COHORT = "esrd_eos2"   # set by --cohort


def tag_of(r, k, outcome, bdays):
    t = f"{COHORT}_cv{r}{k}"
    if outcome != "aki":
        t += f"_{outcome}"
    if bdays != 365:
        t += f"_b{bdays}"
    return t


def fmt_pat(pat, tag):
    if "{tag}" in pat:
        return pat.format(tag=tag)
    return pat.replace("TAG", tag)


def load_preds(pat, tag, preds_dir):
    p = fmt_pat(pat, tag)
    if not p.startswith(preds_dir):
        p = os.path.join(preds_dir, os.path.basename(p))
    files = sorted(glob.glob(p))
    if not files:
        return None, None, 0
    ys, ps = None, []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if ys is None:
            ys = z["y_true"].astype(int)
        ps.append(z["p"].astype(float))
    return ys, np.mean(ps, axis=0), len(files)


# ----------------------------------------------------------------------------- stats
def nb_ci(x):
    x = np.asarray(x, float)
    J = len(x)
    if J < 2:
        return x.mean(), np.nan, np.nan, np.nan
    se = np.sqrt(x.var(ddof=1) * (1.0 / J + RATIO))
    t = stats.t.ppf(0.975, J - 1)
    m = x.mean()
    p = 2 * stats.t.sf(abs(m / se), J - 1) if se > 0 else np.nan
    return m, m - t * se, m + t * se, p


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def logit(q):
    q = np.clip(q, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def score_feature(p):
    """KELM outputs may fall outside [0, 1]; use them linearly then, else the logit."""
    if p.min() < 0 or p.max() > 1:
        return (p - p.mean()) / (p.std() + 1e-12)
    return logit(p)


def logistic_fit(X, y, offset=None, iters=50):
    X = np.column_stack([np.ones(len(y))] + ([X] if X is not None else []))
    off = np.zeros(len(y)) if offset is None else offset
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b + off, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-12
        g = X.T @ (y - mu)
        H = (X * W[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def platt_crossfit(p, y, seed=0, n_splits=5):
    s = score_feature(p)
    q = np.empty(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(s, y):
        a, b = logistic_fit(s[tr], y[tr])
        q[te] = 1 / (1 + np.exp(-(a + b * s[te])))
    return q


def platt_fit(p_cal, y_cal):
    """Platt scaling fitted on a calibration set (outer-training out-of-fold or validation
    predictions); returns a function applied unchanged to the held-out predictions."""
    lin = p_cal.min() < 0 or p_cal.max() > 1
    f = (lambda p: p) if lin else logit
    a, b = logistic_fit(f(p_cal), y_cal)
    return lambda p: 1 / (1 + np.exp(-(a + b * f(p))))


def sens_threshold(q_cal, y_cal, target):
    """Risk threshold that reaches the target sensitivity on the calibration set."""
    pos = np.sort(q_cal[y_cal == 1])
    return pos[int(np.floor((1 - target) * len(pos)))]


def at_threshold(q, y, thr):
    f = q >= thr
    tp = np.sum(f & (y == 1)); fp = np.sum(f & (y == 0))
    fn = np.sum(~f & (y == 1)); tn = np.sum(~f & (y == 0))
    return dict(sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), ppv=tp / max(tp + fp, 1),
                flag=f.mean(), npv=tn / max(tn + fn, 1))


def calib_metrics(q, y):
    qc = np.clip(q, 0, 1)
    brier = np.mean((qc - y) ** 2)
    lq = logit(qc)
    citl = logistic_fit(None, y, offset=lq)[0]
    _, slope = logistic_fit(lq, y)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(y, qc, frac=0.3, it=0, return_sorted=False)
        ici = np.mean(np.abs(sm - qc))
    except Exception:
        bins = np.quantile(qc, np.linspace(0, 1, 21))
        idx = np.clip(np.searchsorted(bins, qc, side="right") - 1, 0, 19)
        ici = np.mean([abs(y[idx == b].mean() - qc[idx == b].mean()) * (idx == b).mean() * 20
                       for b in range(20) if (idx == b).any()]) if len(qc) else np.nan
    oe = y.sum() / qc.sum()
    return dict(brier=brier, citl=citl, slope=slope, ici=ici, oe=oe)


def op_threshold(q, y, t):
    f = q >= t
    tp = np.sum(f & (y == 1)); fp = np.sum(f & (y == 0))
    fn = np.sum(~f & (y == 1)); tn = np.sum(~f & (y == 0))
    n = len(y)
    return dict(flag=f.mean(), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1) if tp + fp else np.nan,
                npv=tn / max(tn + fn, 1) if tn + fn else np.nan,
                nb=tp / n - fp / n * t / (1 - t))


def fixed_sens(p, y, target, seed=0):
    """Threshold chosen on one half of the test fold, applied to the other half."""
    res = []
    for a, b in StratifiedKFold(2, shuffle=True, random_state=seed).split(p, y):
        pos = np.sort(p[a][y[a] == 1])
        thr = pos[int(np.floor((1 - target) * len(pos)))]
        f = p[b] >= thr
        yb = y[b]
        tp = np.sum(f & (yb == 1)); fp = np.sum(f & (yb == 0))
        fn = np.sum(~f & (yb == 1)); tn = np.sum(~f & (yb == 0))
        res.append([tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1), f.mean(),
                    tn / max(tn + fn, 1)])
    return dict(zip(["sens", "spec", "ppv", "flag", "npv"], np.mean(res, axis=0)))


def kendall_w(R):
    """R: raters x items matrix of scores (higher = more relevant)."""
    m, n = R.shape
    ranks = np.vstack([stats.rankdata(-r) for r in R])
    S = np.sum((ranks.sum(0) - m * (n + 1) / 2) ** 2)
    T = 0.0
    for r in ranks:
        _, c = np.unique(r, return_counts=True)
        T += np.sum(c ** 3 - c)
    W = 12 * S / (m ** 2 * (n ** 3 - n) - m * T)
    chi2 = m * (n - 1) * W
    return W, stats.chi2.sf(chi2, n - 1)


# ----------------------------------------------------------------------------- relevance
def ard_vector(meta_str, d):
    """Mean normalised relevance gamma * w_j^2 over the ensemble members (0 if masked out)."""
    meta = json.loads(meta_str)
    mem = meta.get("members", [])
    vecs, keys_seen = [], None
    for m in mem:
        keys_seen = list(m.keys())
        feats = m.get("features", list(range(d)))
        w = None
        for k in ("w", "weights", "ard_w", "ard_weights", "ard", "scales", "z", "log_w"):
            if k in m and isinstance(m[k], list):
                w = np.asarray(m[k], float)
                if k in ("z", "log_w"):
                    w = np.exp(np.clip(w, -6, 3))
                break
        if w is None:
            continue
        full = np.zeros(d)
        if len(w) == d:
            full = w.copy()
            mask = np.zeros(d, bool); mask[feats] = True
            full[~mask] = 0
        elif len(w) == len(feats):
            full[feats] = w
        else:
            continue
        # inputs are multiplied by w, so the kernel uses gamma * w_j^2 (kelm_variant.ARDEncoding)
        full = float(m.get("gamma", 1.0)) * full ** 2
        if full.sum() > 0:
            vecs.append(full / full.sum())
    return (np.mean(vecs, axis=0) if vecs else None), keys_seen


def gp_vector(path, names):
    j = json.load(open(path))
    d = len(names)
    if isinstance(j, dict) and "variables" in j and "relevance_normalised_mean" in j:
        rel = dict(zip([str(x) for x in j["variables"]],
                       np.asarray(j["relevance_normalised_mean"], float)))
        missing = [n for n in names if n not in rel]
        if len(missing) == len(names):
            print("WARNING: GP variables do not match feature names")
            return None
        v = np.array([rel.get(n, 0.0) for n in names])
    elif isinstance(j, dict) and set(names) <= set(j.keys()):
        v = np.array([float(j[n]) for n in names])
    else:
        v = None
        src = j if isinstance(j, dict) else {"v": j}
        for k in ("relevance", "mean_relevance", "inv_lengthscale2", "inv_lengthscale",
                  "lengthscale", "lengthscales", "v"):
            if k in src:
                x = src[k]
                if isinstance(x, dict) and set(names) <= set(x.keys()):
                    x = [x[n] for n in names]
                x = np.asarray(x, float)
                if x.ndim == 2:
                    x = x.mean(0)
                if len(x) == d:
                    v = 1.0 / x ** 2 if k.startswith("lengthscale") else x
                    break
        if v is None:
            print("WARNING: GP relevance format not recognised; keys:",
                  list(j.keys())[:20] if isinstance(j, dict) else type(j))
            return None
    v = np.abs(v)
    return v / v.sum()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="aki", choices=["aki", "st2", "st3"])
    ap.add_argument("--baseline-days", type=int, default=365)
    ap.add_argument("--preds", default="preds_cv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--parts", default="calibration,thresholds,dca,subgroups,ard")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cohort", default="esrd_eos2", help="cohort tag of the prediction files")
    ap.add_argument("--platt", default="oof", choices=["oof", "testcv"],
                    help="oof: Platt and thresholds fitted outside the held-out fold (protocol), "
                         "testcv: cross-fitted within it (sensitivity analysis)")
    ap.add_argument("--oof", default="preds_cv_oof", help="calibration sets of cv_oof.py")
    a = ap.parse_args()
    global COHORT
    COHORT = a.cohort
    out = a.out or f"cv_sec_{a.outcome}"
    parts = set(a.parts.split(","))

    cache = f"cv_cache/cohort_{a.cohort}" + ("" if a.baseline_days == 365 else f"_b{a.baseline_days}") + ".npz"
    C = np.load(cache, allow_pickle=True)
    stage, group = C["stage"], C["group"]
    age, emop, egfr, names = C["age"], C["emop"], C["egfr"], [str(n) for n in C["names"]]
    folds = outer_cv.fold_ids(stage, group)
    yall = np.asarray(outer_cv.label(stage, a.outcome)).astype(int)

    # identify the stratification groups by their size (CPB < urology < other)
    gvals, gn = np.unique(group, return_counts=True)
    order = gvals[np.argsort(gn)]
    gname = {order[0]: "Cardiac surgery with CPB", order[1]: "Urology", order[2]: "Other surgery"}

    pats = model_patterns()
    L = []  # markdown lines
    L.append(f"# Secondary analyses: outcome {a.outcome}, baseline window {a.baseline_days} days\n")
    L.append("Outer CV, two repeats of five folds (protocol Sections 7-9). Recalibration: "
             + ("Platt scaling and fixed-sensitivity thresholds fitted outside the held-out fold, on "
                "out-of-fold predictions of the outer training set (validation predictions for the GP "
                "and the FT-Transformer), as in the protocol; Brier score with Platt cross-fitted within "
                "the test fold as sensitivity analysis." if a.platt == "oof" else
                "Platt scaling cross-fitted inside each test fold (5 internal splits).")
             + " Aggregate results only.\n")

    # ---- collect predictions aligned with patient indices
    P = {}      # key -> {(r,k): p}
    Y = {}      # (r,k) -> y
    IDX = {}
    for r in (1, 2):
        for k in range(1, 6):
            te = np.flatnonzero(folds[r] == k)
            IDX[(r, k)] = te
            Y[(r, k)] = yall[te]
    avail = []
    for key, (label, pat) in pats.items():
        P[key] = {}
        nf = []
        for (r, k), te in IDX.items():
            y, p, n = load_preds(pat, tag_of(r, k, a.outcome, a.baseline_days), a.preds)
            if p is None:
                continue
            if len(y) != len(te) or not np.array_equal(y, Y[(r, k)]):
                sys.exit(f"ORDER CHECK FAILED for {key} fold {r}{k}: predictions are not "
                         f"in fold_ids order (n {len(y)} vs {len(te)}). Stop.")
            P[key][(r, k)] = p
            nf.append(n)
        if P[key]:
            avail.append(f"| {label} | {len(P[key])}/10 | {min(nf)}-{max(nf)} |")
        else:
            del P[key]
    L.append("## Availability (order check passed: predictions align with fold_ids)\n")
    L.append("| Model | folds | files per fold |\n|---|---|---|")
    L += avail + [""]
    labels = {k: pats[k][0] for k in P}
    full = [k for k in P if len(P[k]) == 10]
    m1 = next((k for k in P if k.upper().startswith("M1")), None)

    # recalibrated predictions: protocol (outside the held-out fold) and test-fold cross-fitting
    Qcv = {k: {f: platt_crossfit(p, Y[f], seed=a.seed) for f, p in P[k].items()} for k in P}
    CAL, src = {}, {}
    if a.platt == "oof":
        for k in list(P):
            CAL[k] = {}
            for f in P[k]:
                fn = os.path.join(a.oof, f"{k}__{tag_of(f[0], f[1], a.outcome, a.baseline_days)}.npz")
                if os.path.exists(fn):
                    z = np.load(fn, allow_pickle=True)
                    CAL[k][f] = (z["y_true"].astype(int), z["p"].astype(float))
            src[k] = "outside" if len(CAL[k]) == len(P[k]) else "test-CV"
            if src[k] == "test-CV":
                print(f"WARNING: no calibration set for {k} in some folds; test-fold cross-fitting used")
        Q = {}
        for k in P:
            if src[k] == "outside":
                Q[k] = {f: platt_fit(*CAL[k][f][::-1])(p) for f, p in P[k].items()}
            else:
                Q[k] = Qcv[k]
        labels = {k: labels[k] + ("" if src[k] == "outside" else " (test-CV)") for k in labels}
    else:
        Q = Qcv

    rows_csv = []

    # ---- calibration
    if "calibration" in parts:
        L.append("## Calibration (mean of folds [corrected 95% CI])\n")
        L.append("| Model | Brier raw | Brier recal. | intercept raw | slope raw | ICI raw | "
                 "intercept recal. | slope recal. | ICI recal. | Brier, test-CV Platt |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k in P:
            raw = [calib_metrics(np.clip(p, 0, 1), Y[f]) for f, p in P[k].items()]
            rec = [calib_metrics(Q[k][f], Y[f]) for f in P[k]]
            def c(lst, m):
                v = nb_ci([d[m] for d in lst]); return f"{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]"
            def s(lst, m):
                return f"{np.mean([d[m] for d in lst]):.3f}"
            L.append(f"| {labels[k]} | {c(raw,'brier')} | {c(rec,'brier')} | {s(raw,'citl')} | "
                     f"{s(raw,'slope')} | {s(raw,'ici')} | {s(rec,'citl')} | {s(rec,'slope')} | "
                     f"{s(rec,'ici')} | {np.mean([calib_metrics(Qcv[k][f], Y[f])['brier'] for f in P[k]]):.4f} |")
            for f, d1, d2 in zip(P[k], raw, rec):
                rows_csv.append(dict(part="calibration", model=k, fold=f"{f[0]}{f[1]}",
                                     **{f"raw_{x}": d1[x] for x in d1},
                                     **{f"recal_{x}": d2[x] for x in d2}))
        L.append("\nRaw outputs of the kernel models are not probabilities (logistic function of the "
                 "class-weighted KELM output); compare models after recalibration.\n")
        _calib_plot(Q, Y, labels, [m for m in P if m in full][:6], out)

    # ---- thresholds and operating points
    if "thresholds" in parts:
        L.append("## Risk thresholds (recalibrated; mean of folds)\n")
        L.append("| Model | threshold | flagged | sensitivity | specificity | PPV | NPV | "
                 "net benefit [corrected 95% CI] |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k in P:
            for t in THRESH:
                d = [op_threshold(Q[k][f], Y[f], t) for f in P[k]]
                nb = nb_ci([x["nb"] for x in d])
                mm = lambda m: np.nanmean([x[m] for x in d])
                L.append(f"| {labels[k]} | {t:.0%} | {mm('flag'):.3f} | {mm('sens'):.3f} | "
                         f"{mm('spec'):.3f} | {mm('ppv'):.3f} | {mm('npv'):.3f} | "
                         f"{nb[0]:.4f} [{nb[1]:.4f}, {nb[2]:.4f}] |")
                for f, x in zip(P[k], d):
                    rows_csv.append(dict(part="threshold", model=k, fold=f"{f[0]}{f[1]}",
                                         threshold=t, **x))
        if m1:
            L.append("\nNet benefit, M1 minus comparator (corrected t-test, two-sided):\n")
            L.append("| Comparator | 5% | 10% | 20% |\n|---|---|---|---|")
            for k in P:
                if k == m1 or len(P[k]) < 10:
                    continue
                cells = []
                for t in THRESH:
                    dd = [op_threshold(Q[m1][f], Y[f], t)["nb"] - op_threshold(Q[k][f], Y[f], t)["nb"]
                          for f in P[k]]
                    v = nb_ci(dd)
                    cells.append(f"{v[0]:+.4f} (p={v[3]:.3g})")
                L.append(f"| {labels[k]} | " + " | ".join(cells) + " |")
        L.append("\n## Fixed-sensitivity operating points (" + ("threshold chosen on the calibration set "
                 "outside the held-out fold and applied unchanged" if a.platt == "oof" else
                 "threshold from one half of the test fold, applied to the other half") + "; mean of folds)\n")
        L.append("| Model | target | achieved sensitivity | specificity | PPV | NPV | flagged |")
        L.append("|---|---|---|---|---|---|---|")
        for k in P:
            for s_ in SENS:
                if a.platt == "oof" and src.get(k) == "outside":
                    d = []
                    for f in P[k]:
                        yc, pc = CAL[k][f]
                        g = platt_fit(pc, yc)
                        d.append(at_threshold(Q[k][f], Y[f], sens_threshold(g(pc), yc, s_)))
                else:
                    d = [fixed_sens(P[k][f], Y[f], s_, seed=a.seed) for f in P[k]]
                mm = lambda m: np.mean([x[m] for x in d])
                L.append(f"| {labels[k]} | {s_:.0%} | {mm('sens'):.3f} | {mm('spec'):.3f} | "
                         f"{mm('ppv'):.3f} | {mm('npv'):.3f} | {mm('flag'):.3f} |")
                for f, x in zip(P[k], d):
                    rows_csv.append(dict(part="fixed_sens", model=k, fold=f"{f[0]}{f[1]}",
                                         target=s_, **x))
        L.append("")

    # ---- decision curves
    if "dca" in parts:
        _dca(Q, Y, labels, [m for m in P if m in full], out)
        L.append(f"## Decision curves\n\nSee {os.path.basename(out)}_dca.png (pooled recalibrated predictions, "
                 "repeat 1 solid, repeat 2 dashed).\n")

    # ---- subgroups
    if "subgroups" in parts:
        subs = {gname[g] + (" (not pre-specified)" if gname[g] == "Other surgery" else ""): group == g
                for g in order}
        if "sex_M" in names:
            male = C["X"][:, names.index("sex_M")] == 1
            subs["Male"], subs["Female"] = male, ~male
        subs["Age >= 65"] = age >= 65
        subs["Emergency surgery"] = emop == 1
        subs["CKD (eGFR < 60)"] = egfr < 60
        L.append("## Subgroups (discrimination; mean of folds, pooled AUPRC per repeat in "
                 "the csv)\n")
        L.append("| Subgroup | n | events | Model | AUROC [CI] | AUPRC [CI] | O/E recal. |")
        L.append("|---|---|---|---|---|---|---|")
        diffs = []
        for sname, mask in subs.items():
            n_tot, ev_tot = int(mask.sum()), int(yall[mask].sum())
            per = {}
            for k in P:
                au, ap_, oe = [], [], []
                for f, p in P[k].items():
                    mloc = mask[IDX[f]]
                    y = Y[f][mloc]
                    if y.sum() < 5 or y.sum() == len(y):
                        continue
                    au.append(roc_auc_score(y, p[mloc]))
                    ap_.append(average_precision_score(y, p[mloc]))
                    oe.append(y.sum() / Q[k][f][mloc].sum())
                if not au:
                    continue
                per[k] = (au, ap_)
                A, B = nb_ci(au), nb_ci(ap_)
                L.append(f"| {sname} | {n_tot} | {ev_tot} | {labels[k]} | {A[0]:.3f} "
                         f"[{A[1]:.3f}, {A[2]:.3f}] | {B[0]:.3f} [{B[1]:.3f}, {B[2]:.3f}] | "
                         f"{np.mean(oe):.2f} |")
                for f in P[k]:
                    pass
            if m1 in per:
                for k in per:
                    if k == m1 or len(per[k][0]) != len(per[m1][0]):
                        continue
                    for mi, met in enumerate(["AUROC", "AUPRC"]):
                        dd = np.array(per[m1][mi]) - np.array(per[k][mi])
                        v = nb_ci(dd)
                        diffs.append((sname, labels[k], met, v))
        if diffs:
            L.append("\nM1 minus comparator within subgroups (corrected t-test, two-sided; "
                     "Holm within each subgroup and metric; descriptive, protocol Section 8):\n")
            L.append("| Subgroup | Comparator | metric | difference [CI] | p | Holm p |")
            L.append("|---|---|---|---|---|---|")
            from collections import defaultdict
            grp = defaultdict(list)
            for i, (s_, c_, m_, v) in enumerate(diffs):
                grp[(s_, m_)].append(i)
            hp = np.empty(len(diffs))
            for idx in grp.values():
                hp[idx] = holm([diffs[i][3][3] for i in idx])
            for i, (s_, c_, m_, v) in enumerate(diffs):
                L.append(f"| {s_} | {c_} | {m_} | {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] | "
                         f"{v[3]:.3g} | {hp[i]:.3g} |")
                rows_csv.append(dict(part="subgroup_diff", subgroup=s_, model=c_, metric=m_,
                                     diff=v[0], lo=v[1], hi=v[2], p=v[3], holm=hp[i]))
        L.append("")

    # ---- ARD / GP relevance
    if "ard" in parts and m1:
        d = len(names)
        R1, RG, fl = [], [], []
        keys_seen = None
        for (r, k) in IDX:
            tag = tag_of(r, k, a.outcome, a.baseline_days)
            pat = fmt_pat(pats[m1][1], tag)
            pat = os.path.join(a.preds, os.path.basename(pat))
            vs = []
            for f in sorted(glob.glob(pat)):
                v, keys_seen = ard_vector(str(np.load(f, allow_pickle=True)["meta"]), d)
                if v is not None:
                    vs.append(v)
            gpf = os.path.join(a.preds, f"gp_ard_relevance__{tag}.json")
            g = gp_vector(gpf, names) if os.path.exists(gpf) else None
            if vs:
                R1.append(np.mean(vs, axis=0)); fl.append(f"{r}{k}")
                RG.append(g)
        L.append("## Interpretability: ARD relevance\n")
        if not R1:
            L.append(f"No ARD weights found in the M1 meta (member keys: {keys_seen}). "
                     "Weights must be added to the meta or read from the json.\n")
        else:
            R1 = np.array(R1)
            W1, pW1 = kendall_w(R1)
            mean1 = R1.mean(0)
            top = np.argsort(-mean1)[:20]
            sel = (R1 > 1e-4).mean(0)
            hasg = all(g is not None for g in RG)
            if hasg:
                RG_ = np.array(RG); WG, pWG = kendall_w(RG_); meanG = RG_.mean(0)
                rho = [stats.spearmanr(x, y_).correlation for x, y_ in zip(R1, RG_)]
                ov = [len(set(np.argsort(-x)[:10]) & set(np.argsort(-y_)[:10])) for x, y_ in zip(R1, RG_)]
            L.append(f"Stability across the {len(R1)} outer folds: Kendall's W (M1) = {W1:.3f} "
                     f"(p = {pW1:.2g})" + (f"; Kendall's W (GP) = {WG:.3f} (p = {pWG:.2g})" if hasg else "") + ".\n")
            if hasg:
                L.append(f"Agreement M1 vs GP per fold: Spearman rho mean {np.mean(rho):.3f} "
                         f"(range {np.min(rho):.3f}-{np.max(rho):.3f}); top-10 overlap mean "
                         f"{np.mean(ov):.1f}/10 (range {min(ov)}-{max(ov)}). Overall rankings: "
                         f"Spearman {stats.spearmanr(mean1, meanG).correlation:.3f}, top-10 overlap "
                         f"{len(set(np.argsort(-mean1)[:10]) & set(np.argsort(-meanG)[:10]))}/10.\n")
            L.append("| rank | feature | M1 relevance (mean of folds) | SD | folds active | "
                     "rank in top-10 (folds) |" + (" GP relevance | GP rank |" if hasg else ""))
            L.append("|---|---|---|---|---|---|" + ("---|---|" if hasg else ""))
            rk1 = np.vstack([stats.rankdata(-x) for x in R1])
            if hasg:
                gr = stats.rankdata(-meanG)
            for i, j in enumerate(top, 1):
                row = (f"| {i} | {names[j]} | {mean1[j]:.4f} | {R1[:, j].std():.4f} | "
                       f"{sel[j]*len(R1):.0f}/{len(R1)} | {(rk1[:, j] <= 10).sum()}/{len(R1)} |")
                if hasg:
                    row += f" {meanG[j]:.4f} | {gr[j]:.0f} |"
                L.append(row)
            for j in range(d):
                rows_csv.append(dict(part="relevance", feature=names[j], m1_mean=mean1[j],
                                     m1_sd=R1[:, j].std(),
                                     gp_mean=(meanG[j] if hasg else np.nan)))
            L.append("")

    open(f"{out}.md", "w").write("\n".join(L) + "\n")
    try:
        import pandas as pd
        pd.DataFrame(rows_csv).to_csv(f"{out}.csv", index=False)
    except Exception as e:
        print("csv not written:", e)
    print("written", f"{out}.md")


def _calib_plot(Q, Y, labels, keys, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for k in keys:
        q = np.concatenate([Q[k][(1, j)] for j in range(1, 6)])
        y = np.concatenate([Y[(1, j)] for j in range(1, 6)])
        b = np.quantile(q, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(b, q, side="right") - 1, 0, 9)
        px = [q[idx == i].mean() for i in range(10)]
        py = [y[idx == i].mean() for i in range(10)]
        ax[0].plot(px, py, "o-", ms=3, label=labels[k])
        ax[1].hist(q, bins=50, histtype="step", label=labels[k])
    m = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])
    ax[0].plot([0, m], [0, m], "k:", lw=1)
    ax[0].set_xlabel("Predicted risk (recalibrated)"); ax[0].set_ylabel("Observed proportion")
    ax[0].set_title("Calibration, deciles (repeat 1)"); ax[0].legend(fontsize=7)
    ax[1].set_yscale("log"); ax[1].set_xlabel("Predicted risk"); ax[1].set_title("Distribution")
    fig.tight_layout()
    fig.savefig(f"{out}_calib.png", dpi=200); fig.savefig(f"{out}_calib.pdf")


def _dca(Q, Y, labels, keys, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.linspace(0.01, 0.30, 59)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r, ls in ((1, "-"), (2, "--")):
        y = np.concatenate([Y[(r, j)] for j in range(1, 6)])
        prev = y.mean()
        if r == 1:
            ax.plot(ts, prev - (1 - prev) * ts / (1 - ts), "k:", lw=1, label="Treat all")
            ax.axhline(0, color="grey", lw=0.8)
        for i, k in enumerate(keys):
            q = np.concatenate([Q[k][(r, j)] for j in range(1, 6)])
            nb = [op_threshold(q, y, t)["nb"] for t in ts]
            ax.plot(ts, nb, ls, color=f"C{i}", lw=1.2, label=labels[k] if r == 1 else None)
    ax.set_ylim(-0.005, None); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{out}_dca.png", dpi=200); fig.savefig(f"{out}_dca.pdf")


if __name__ == "__main__":
    main()
