"""
Aggregate counts needed before the confirmatory protocol is committed (no model is fitted).
==========================================================================================

Same cohort as the manuscript (rule cr4_z49). Prints only aggregate numbers:
  1. timing of the baseline creatinine (days before anesthesia start) and AKI prevalence when the
     baseline is restricted to the last 7 / 30 / 90 days;
  2. KDIGO stage from creatinine (stage 1: >=1.5x; 2: >=2.0x; 3: >=3.0x or peak >=4.0 mg/dL), plus
     a renal-replacement proxy (ICD-10 Z49 recorded after surgery, within 7 days);
  3. counts per stratum: stage (0, 1, >=2) x surgical group (cardiac surgery with bypass, urology, other);
  4. subgroup sizes: age >=65, children, emergency, CKD (eGFR < 60; CKD-EPI 2021 for adults,
     bedside Schwartz 0.413*height/SCr for children), with AKI prevalence in each.
    python protocol_counts.py            # on the server, from the run directory
    python protocol_counts.py --dummy    # self-test on synthetic tables
"""
import contextlib
import io
import sys

import numpy as np
import pandas as pd

from cohort_split import DATA_DIR, _load_tables
from inspire_preprocessing import CREATININE_ITEM_NAME, apply_cohort_criteria

DUMMY = "--dummy" in sys.argv
operations, labs, diagnosis = _load_tables(DATA_DIR, DUMMY)
with contextlib.redirect_stdout(io.StringIO()):
    ops, labs = apply_cohort_criteria(operations, labs, first_procedure_only=True,
                                      esrd=None if DUMMY else "cr4_z49", diagnosis=diagnosis, verbose=True)
ops = ops.reset_index(drop=True)
cr = labs[labs["item_name"] == CREATININE_ITEM_NAME][["subject_id", "chart_time", "value"]]

# baseline: most recent creatinine before anesthesia start, within 365 days (as in the manuscript)
m = cr.merge(ops[["op_id", "subject_id", "anstart_time", "orout_time"]], on="subject_id")
pre = m[(m["chart_time"] < m["anstart_time"]) & (m["chart_time"] >= m["anstart_time"] - 365 * 1440)]
pre = pre.sort_values("chart_time").groupby("op_id").tail(1).set_index("op_id")
base = pre["value"]
base_days = (pre["anstart_time"] - pre["chart_time"]) / 1440
post = m[((m["chart_time"] - m["orout_time"]) >= 0) & ((m["chart_time"] - m["orout_time"]) <= 7 * 1440)]
peak = post.groupby("op_id")["value"].max()

d = ops.set_index("op_id")
d["base"], d["base_days"], d["peak"] = base, base_days, peak
d["ratio"] = d["peak"] / d["base"]
stage = np.select([d["ratio"] >= 3.0, d["ratio"] >= 2.0, d["ratio"] >= 1.5], [3, 2, 1], 0)
stage = np.where((stage >= 1) & (d["peak"] >= 4.0), 3, stage)
# children: stage 3 also when the eGFR from the peak falls below 35 (bedside Schwartz, needs height)
egfr_peak_child = 0.413 * d["height"].where(d["height"] > 50) / d["peak"]
stage = np.where((stage >= 1) & (d["age"].astype(float) < 18) & (egfr_peak_child < 35), 3, stage)
d["stage"] = stage

# renal-replacement proxy: Z49 (dialysis care) recorded after surgery, within 7 days of leaving the OR
if len(diagnosis):
    z = diagnosis[diagnosis["icd10_cm"].astype(str).str[:3] == "Z49"][["subject_id", "chart_time"]]
    zz = z.merge(d.reset_index()[["op_id", "subject_id", "orout_time"]], on="subject_id")
    zz = zz[((zz["chart_time"] - zz["orout_time"]) >= 0) & ((zz["chart_time"] - zz["orout_time"]) <= 7 * 1440)]
    rrt = d.index.isin(zz["op_id"].unique())
else:
    rrt = np.zeros(len(d), bool)
d["rrt"] = rrt

n = len(d)
pct = lambda k: f"{k} ({100 * k / n:.1f}%)"
print(f"cohort n = {n}")
print("\n1. baseline creatinine timing (days before anesthesia start)")
q = d["base_days"].quantile([0.25, 0.5, 0.75, 0.9]).round(1).tolist()
print(f"   quartiles/90th: {q}")
for lim in (1, 7, 30, 90, 180, 365):
    s = d["base_days"] <= lim
    print(f"   <= {lim:3d} d: {pct(int(s.sum()))}; AKI (stage>=1) among them {100 * (d.loc[s, 'stage'] >= 1).mean():.2f}%")

print("\n2. KDIGO stage (creatinine only)")
for k in range(4):
    print(f"   stage {k}: {pct(int((d['stage'] == k).sum()))}")
print(f"   stage >=1: {pct(int((d['stage'] >= 1).sum()))}; stage >=2: {pct(int((d['stage'] >= 2).sum()))}")
print(f"   stage 3 by peak >=4.0 mg/dL only: {int(((d['stage'] == 3) & (d['ratio'] < 3)).sum())}")
print(f"   renal-replacement proxy (post-op Z49 within 7 d): {int(d['rrt'].sum())} "
      f"(of which stage <3 by creatinine: {int((d['rrt'] & (d['stage'] < 3)).sum())})")

dep = d["department"].astype(str)
cpb = d["cpbon_time"].notna() & d["cpboff_time"].notna() & ((d["cpboff_time"] - d["cpbon_time"]) > 0)
grp = np.where(cpb, "cardiac CPB", np.where(dep == "UR", "urology", "other"))
st3 = np.where(d["stage"] >= 2, ">=2", d["stage"].astype(str))
print("\n3. strata: stage x surgical group (counts)")
print(pd.crosstab(pd.Series(grp, name="group"), pd.Series(st3, name="stage")).to_string())
print(f"   cardiac surgery with bypass outside department CTS: {int((cpb & (dep != 'CTS')).sum())}")

age = d["age"].astype(float)
female = d["sex"].astype(str).str.upper().str.startswith("F")
k = np.where(female, 0.7, 0.9)
a = np.where(female, -0.241, -0.302)
r = d["base"] / k
egfr_adult = 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1.0)
height = d["height"].where(d["height"] > 50)
egfr_child = 0.413 * height / d["base"]
egfr = np.where(age >= 18, egfr_adult, egfr_child)
ckd = egfr < 60
print("\n4. subgroups (n, AKI stage>=1 %, stage>=2 %)")
for name, s in [("age >= 65", age >= 65), ("age < 18", age < 18), ("emergency", d["emop"] == 1),
                ("CKD eGFR<60", ckd), ("CKD eGFR<60, adults", ckd & (age >= 18)),
                ("eGFR missing (children without height)", np.isnan(egfr)), ("cardiac CPB", cpb),
                ("urology", dep == "UR")]:
    s = np.asarray(s, bool)
    print(f"   {name}: {pct(int(s.sum()))}; stage>=1 {100 * (d['stage'][s] >= 1).mean():.2f}%, "
          f"stage>=2 {100 * (d['stage'][s] >= 2).mean():.2f}%")
