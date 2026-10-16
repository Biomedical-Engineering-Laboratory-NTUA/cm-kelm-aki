"""
End-of-surgery feature extraction (intraoperative vitals, fluids, blood
products, vasoactive drugs, medications, CPB) for the INSPIRE cohort.
==========================================================================

Adds a second prediction time point to the paper:
  * preoperative model  : the current 37 features (prediction before surgery)
  * end-of-surgery model: + the features below, all computed from data
                          recorded no later than anaesthesia end (no leakage
                          of postoperative information)

Sources and windows
-------------------
vitals.csv       rows with anstart_time <= chart_time <= anend_time
medications.csv  intraoperative: same window;
                 preoperative:   [anstart - 30 days, anstart)
operations.csv   cardiopulmonary bypass duration (cpboff - cpbon), weight

Monitoring signals are time-weighted: each value holds until the next
measurement, capped at CAP_MIN minutes (so a gap in monitoring is not treated
as a long hypotensive or normotensive period). Physiologically impossible
values are removed first (e.g. arterial MAP p1 = 6 mmHg in the exploration).

MAP source: invasive arterial MAP when the case has an arterial line inside
the window, otherwise non-invasive MAP. `art_line` records which one.

Event items (fluids, blood products, boluses, EBL, urine output) are summed;
a case with no row for an item gets 0 (nothing charted). The script also
reports, per event item, the fraction of cases whose values never decrease
over time: a high fraction would suggest cumulative charting, in which case
the *_max columns (also saved) are the right totals instead of *_sum.

Output
------
features_intraop.csv.gz   one row per cohort operation (op_id + features)
features_intraop.md       coverage, missingness, univariable AKI association,
                          cumulative-charting check

Usage (CPU job, a few minutes):
    sbatch --cpus-per-task=4 --mem=64G --job-name=intraop --output=logs/%x_%j.log \
           --wrap="python -u intraop_features.py"
"""

import os
import argparse
import tempfile
import time

import numpy as np
import pandas as pd

from inspire_preprocessing import apply_cohort_criteria, compute_kdigo_aki_label

# INSPIRE v1.4.2 folder (operations.csv.gz, labs.csv.gz, vitals.csv.gz, ...); set INSPIRE_DIR
DATA_DIR = os.environ.get("INSPIRE_DIR", "data/inspire/1.4.2")
CAP_MIN = 15.0

VALID = {  # plausible physiological ranges; values outside are dropped
    "art_mbp": (20, 200), "nibp_mbp": (20, 200), "hr": (20, 250), "spo2": (50, 100),
    "bt": (30, 42), "cvp": (-10, 40),
}
MONITOR = list(VALID)
CRYSTALLOID = ["hs", "ns", "psa", "hns", "d5w", "d10w", "d50w", "ds"]
COLLOID = ["hes", "alb5", "alb20"]
BLOOD = {"rbc": "rbc", "ffp": "ffp", "pc": "plt", "pheresis": "plt", "cryo": "cryo"}
BOLUS = ["eph", "phe", "epi", "vaso"]
INFUSION = ["nepi", "epii", "dopai", "dobui", "mlni"]
EVENT = ["ebl", "uo"] + CRYSTALLOID + COLLOID + list(BLOOD) + BOLUS
NEEDED = set(MONITOR) | set(EVENT) | set(INFUSION) | {"ci"}

# ATC prefixes (WHO) for medication flags; drug-name fallbacks for rows without ATC
PREOP_MEDS = {
    "pre_loop_diuretic": (["C03C"], ["furosemide", "torsemide", "bumetanide"]),
    "pre_any_diuretic": (["C03"], []),
    "pre_acei_arb": (["C09"], []),
    "pre_nsaid": (["M01A"], ["ketorolac", "ibuprofen", "celecoxib", "naproxen", "diclofenac"]),
    "pre_vancomycin": (["J01XA01", "A07AA09"], ["vancomycin"]),
    "pre_aminoglycoside": (["J01G"], ["gentamicin", "amikacin", "tobramycin"]),
    "pre_insulin": (["A10A"], ["insulin"]),
    "pre_oral_antidiabetic": (["A10B"], ["metformin"]),
    "pre_statin": (["C10AA"], []),
    "pre_antiplatelet": (["B01AC"], ["aspirin", "clopidogrel"]),
    "pre_anticoagulant": (["B01AA", "B01AB", "B01AE", "B01AF"], ["heparin", "warfarin"]),
    "pre_contrast": (["V08"], ["iohexol", "iopromide", "iodixanol", "iopamidol"]),
}
INTRA_MEDS = {
    "intra_vancomycin": (["J01XA01", "A07AA09"], ["vancomycin"]),
    "intra_aminoglycoside": (["J01G"], ["gentamicin", "amikacin", "tobramycin"]),
    "intra_nsaid": (["M01A"], ["ketorolac"]),
    "intra_diuretic": (["C03"], ["furosemide", "mannitol"]),
    "intra_heparin": (["B01AB"], ["heparin"]),
    "intra_cardioplegia": (["B05XA16"], ["htk", "cardioplegi"]),
}


def time_weighted(g, t_end):
    """g: DataFrame(chart_time, value) of one op/item, sorted. Returns weights."""
    t = g["chart_time"].to_numpy(float)
    nxt = np.append(t[1:], t_end)
    return np.clip(nxt - t, 0, CAP_MIN)


def summarize_monitor(df, ends, item, thresholds_low=(), thresholds_high=()):
    """Per-op stats of one monitoring signal. df has op_id, chart_time, value."""
    out = {}
    for op, g in df.groupby("op_id", sort=False):
        g = g.sort_values("chart_time")
        v = g["value"].to_numpy(float)
        w = time_weighted(g, ends[op])
        tot = w.sum()
        r = {f"{item}_min": v.min(), f"{item}_max": v.max(),
             f"{item}_mean": (v * w).sum() / tot if tot > 0 else v.mean()}
        for thr in thresholds_low:
            below = v < thr
            r[f"{item}_t_lt{thr}"] = w[below].sum()
            r[f"{item}_auc_lt{thr}"] = ((thr - v[below]) * w[below]).sum()
            r[f"{item}_frac_lt{thr}"] = w[below].sum() / tot if tot > 0 else np.nan
        for thr in thresholds_high:
            r[f"{item}_t_gt{thr}"] = w[v > thr].sum()
        out[op] = r
    return pd.DataFrame.from_dict(out, orient="index")


def med_flags(meds, spec):
    atc = meds["atc_code"].astype(str).str.upper()
    name = meds["drug_name"].astype(str).str.lower()
    cols = {}
    for flag, (prefixes, names) in spec.items():
        hit = np.zeros(len(meds), bool)
        for pfx in prefixes:
            hit |= atc.str.startswith(pfx).to_numpy()
        for nm in names:
            hit |= name.str.contains(nm, regex=False).to_numpy()
        cols[flag] = set(meds.loc[hit, "op_id"])
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default="features_intraop")
    ap.add_argument("--chunksize", type=int, default=5_000_000)
    ap.add_argument("--dummy", action="store_true", help="synthetic tables (smoke test)")
    a = ap.parse_args()
    t0 = time.time()

    if a.dummy:
        from explore_vitals_esrd import make_dummy
        data_dir = tempfile.mkdtemp()
        operations, labs = make_dummy(data_dir)
    else:
        data_dir = a.data_dir
        operations = pd.read_csv(f"{data_dir}/operations.csv.gz")
        labs = pd.read_csv(f"{data_dir}/labs.csv.gz")
    ops, labs = apply_cohort_criteria(operations, labs, first_procedure_only=True, verbose=False)
    ops = ops.set_index("op_id")
    aki = compute_kdigo_aki_label(ops.reset_index(), labs).reindex(ops.index).astype(int)
    win = ops[["anstart_time", "anend_time"]]
    ends = win["anend_time"].to_dict()
    print(f"cohort {len(ops)} operations", flush=True)

    # ---------------- vitals: filter once, keep the needed items in memory
    parts, n_read = [], 0
    for ch in pd.read_csv(f"{data_dir}/vitals.csv.gz", usecols=["op_id", "chart_time", "item_name", "value"],
                          chunksize=a.chunksize):
        n_read += len(ch)
        ch = ch[ch["op_id"].isin(win.index) & ch["item_name"].isin(NEEDED)]
        w = win.loc[ch["op_id"].values]
        ch = ch[(ch["chart_time"].values >= w["anstart_time"].values) &
                (ch["chart_time"].values <= w["anend_time"].values)]
        parts.append(ch)
        print(f"  vitals rows read {n_read:,}", flush=True)
    v = pd.concat(parts, ignore_index=True)
    v["value"] = pd.to_numeric(v["value"], errors="coerce")
    v = v.dropna(subset=["value"])
    for item, (lo, hi) in VALID.items():
        bad = (v["item_name"] == item) & ((v["value"] < lo) | (v["value"] > hi))
        v = v[~bad]
    print(f"kept {len(v):,} vitals rows", flush=True)

    feats = pd.DataFrame(index=ops.index)
    # MAP: arterial if present, else non-invasive
    art_ops = set(v.loc[v["item_name"] == "art_mbp", "op_id"])
    mp = v[((v["item_name"] == "art_mbp") & v["op_id"].isin(art_ops)) |
           ((v["item_name"] == "nibp_mbp") & ~v["op_id"].isin(art_ops))]
    feats = feats.join(summarize_monitor(mp, ends, "map", thresholds_low=(65, 55)))
    feats["art_line"] = feats.index.isin(art_ops).astype(int)
    for item, lows, highs in (("hr", (), (100,)), ("spo2", (90,), ()), ("bt", (36,), ()), ("cvp", (), ())):
        s = summarize_monitor(v[v["item_name"] == item], ends, item, lows, highs)
        keep = {"hr": ["hr_mean", "hr_max", "hr_t_gt100"], "spo2": ["spo2_min", "spo2_t_lt90"],
                "bt": ["bt_min", "bt_mean", "bt_t_lt36"], "cvp": ["cvp_mean"]}[item]
        feats = feats.join(s[[c for c in keep if c in s.columns]])
    for flag, items in (("cvp_line", ["cvp"]), ("co_monitor", ["ci"])):
        feats[flag] = feats.index.isin(set(v.loc[v["item_name"].isin(items), "op_id"])).astype(int)
    print("monitor features done", flush=True)

    # events: sums (and max for the cumulative-charting check)
    ev = v[v["item_name"].isin(EVENT)]
    s_sum = ev.pivot_table(index="op_id", columns="item_name", values="value", aggfunc="sum")
    s_max = ev.pivot_table(index="op_id", columns="item_name", values="value", aggfunc="max")
    s_sum = s_sum.reindex(ops.index).fillna(0.0)
    s_max = s_max.reindex(ops.index).fillna(0.0)
    get = lambda df, c: df[c] if c in df.columns else pd.Series(0.0, index=df.index)
    feats["ebl_sum"], feats["ebl_max"] = get(s_sum, "ebl"), get(s_max, "ebl")
    feats["uo_sum"], feats["uo_max"] = get(s_sum, "uo"), get(s_max, "uo")
    feats["crystalloid_sum"] = sum(get(s_sum, c) for c in CRYSTALLOID)
    feats["colloid_sum"] = sum(get(s_sum, c) for c in COLLOID)
    for src, dst in BLOOD.items():
        feats[f"{dst}_sum"] = feats.get(f"{dst}_sum", 0) + get(s_sum, src)
    feats["transfusion_any"] = ((feats["rbc_sum"] + feats["ffp_sum"] + feats["plt_sum"] + feats["cryo_sum"]) > 0).astype(int)
    for b in BOLUS:
        feats[f"{b}_sum"] = get(s_sum, b)
    inf_ops = set(v.loc[v["item_name"].isin(INFUSION) & (v["value"] > 0), "op_id"])
    feats["vasoactive_infusion"] = feats.index.isin(inf_ops).astype(int)
    feats["vasopressor_any"] = ((feats[[f"{b}_sum" for b in BOLUS]].sum(axis=1) > 0) |
                                (feats["vasoactive_infusion"] == 1)).astype(int)
    dur_h = ((ops["anend_time"] - ops["anstart_time"]) / 60.0).clip(lower=0.25)
    wt = pd.to_numeric(ops.get("weight"), errors="coerce")
    feats["uo_ml_kg_h"] = feats["uo_sum"] / (wt * dur_h)
    feats.loc[feats["uo_sum"] == 0, "uo_ml_kg_h"] = np.nan      # not charted != anuric
    if {"cpbon_time", "cpboff_time"} <= set(ops.columns):
        cpb = (ops["cpboff_time"] - ops["cpbon_time"]).clip(lower=0)
        feats["cpb_min"] = cpb.fillna(0.0)
        feats["cpb_any"] = (feats["cpb_min"] > 0).astype(int)
    print("event features done", flush=True)

    # cumulative-charting check
    mono = {}
    for item, g in ev.groupby("item_name"):
        gg = g.sort_values(["op_id", "chart_time"])
        dec = (gg.groupby("op_id")["value"].diff() < 0).groupby(gg["op_id"]).any()
        cnt = gg.groupby("op_id").size()
        multi = cnt.index[cnt > 1]
        mono[item] = (len(multi), float((~dec.loc[multi]).mean()) if len(multi) else np.nan)

    # ---------------- medications
    meds = pd.read_csv(f"{data_dir}/medications.csv.gz")
    mm = meds.merge(ops.reset_index()[["op_id", "subject_id", "anstart_time", "anend_time"]],
                    on="subject_id", how="inner")
    pre = mm[(mm["chart_time"] < mm["anstart_time"]) & (mm["chart_time"] >= mm["anstart_time"] - 30 * 24 * 60)]
    intra = mm[(mm["chart_time"] >= mm["anstart_time"]) & (mm["chart_time"] <= mm["anend_time"])]
    for spec, df in ((PREOP_MEDS, pre), (INTRA_MEDS, intra)):
        for flag, ids in med_flags(df, spec).items():
            feats[flag] = feats.index.isin(ids).astype(int)
    print("medication features done", flush=True)

    feats.index.name = "op_id"
    feats.to_csv(f"{a.out}.csv.gz")

    # ---------------- report
    n = len(feats)
    lines = [f"# End-of-surgery features: {feats.shape[1]} columns, {n} operations\n",
             "| feature | non-missing | % | median | AKI rate: low half / high half (or 0 / 1) |",
             "|---|---|---|---|---|"]
    for c in feats.columns:
        x = feats[c]
        nm = int(x.notna().sum())
        if set(x.dropna().unique()) <= {0, 1}:
            r0 = aki[x == 0].mean() if (x == 0).any() else np.nan
            r1 = aki[x == 1].mean() if (x == 1).any() else np.nan
            assoc, med = f"{100 * r0:.1f}% / {100 * r1:.1f}% (n1={int((x == 1).sum())})", "-"
        else:
            m_ = x.median()
            lo, hi = aki[x <= m_].mean(), aki[x > m_].mean()
            assoc, med = f"{100 * lo:.1f}% / {100 * hi:.1f}%", f"{m_:.3g}"
        lines.append(f"| {c} | {nm} | {100 * nm / n:.1f}% | {med} | {assoc} |")
    lines += ["\n## Coded values of the raw items (INSPIRE 5th-percentile coding)\n",
              "Values after the plausibility filter; thresholds such as MAP < 65 act on these categories.\n",
              "| item | distinct values | values (sorted, first 25) |", "|---|---|---|"]
    for item in sorted(set(v["item_name"])):
        u = np.sort(v.loc[v["item_name"] == item, "value"].unique())
        lines.append(f"| {item} | {len(u)} | {', '.join(f'{x:.3g}' for x in u[:25])} |")
    lines += ["\n## Cumulative-charting check (event items)\n",
              "| item | cases with >1 row | fraction never decreasing |", "|---|---|---|"]
    for item, (k, f) in sorted(mono.items()):
        lines.append(f"| {item} | {k} | {f:.2f} |")
    lines.append("\nA fraction close to 1.00 suggests cumulative charting (use *_max as total); "
                 "values well below 1 indicate incremental entries (use *_sum).")
    lines.append(f"\n_Generated in {(time.time() - t0) / 60:.1f} min._")
    open(f"{a.out}.md", "w").write("\n".join(lines))
    print(f"written {a.out}.csv.gz and {a.out}.md ({feats.shape[1]} features)", flush=True)


if __name__ == "__main__":
    main()
