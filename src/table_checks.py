"""Two checks of Table S2 (aggregate output only):
1. urea nitrogen: cumulative distribution around the medians (percentile codes), overall / AKI / no AKI;
2. urine output: share of operations without charted urine output (total 0) and missing mL/kg/h.
    python table_checks.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

from cohort_split import DATA_DIR, _load_tables
from inspire_preprocessing import apply_cohort_criteria, build_feature_matrix, compute_kdigo_aki_label

DUMMY = "--dummy" in sys.argv
operations, labs, diagnosis = _load_tables(DATA_DIR, DUMMY)
with contextlib.redirect_stdout(io.StringIO()):
    ops, labs = apply_cohort_criteria(operations, labs, first_procedure_only=True, esrd="cr4_z49",
                                      diagnosis=diagnosis, verbose=True)
ops = ops.reset_index(drop=True)
X, _ = build_feature_matrix(ops, labs)
X = X.reset_index(drop=True)
if not DUMMY:
    intra = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "features_intraop.csv.gz")).set_index("op_id")
    X = pd.concat([X, intra.loc[ops["op_id"]].reset_index(drop=True)], axis=1)
y = compute_kdigo_aki_label(ops, labs).reindex(ops["op_id"]).values.astype(int)
groups = {"overall": np.ones(len(y), bool), "AKI": y == 1, "no AKI": y == 0}
s = X["preop_bun"]
print("urea nitrogen (mg/dL): cumulative share of patients at or below each code")
codes = sorted(pd.unique(s.dropna()))
for c in codes:
    if 9 <= c <= 21:
        print(f"  <= {c:g}: " + ", ".join(f"{g} {100 * (s[m] <= c).sum() / s[m].notna().sum():.1f}%" for g, m in groups.items()))
for g, m in groups.items():
    print(f"  median {g}: {np.percentile(s[m].dropna(), 50):g} (numpy), {s[m].median():g} (pandas)")
if "uo_sum" in X:
    uo, uok = X["uo_sum"], X["uo_ml_kg_h"]
    print(f"urine output total = 0 (not charted): {100 * (uo == 0).mean():.1f}%; total missing {100 * uo.isna().mean():.1f}%; "
          f"mL/kg/h missing {100 * uok.isna().mean():.1f}%")
