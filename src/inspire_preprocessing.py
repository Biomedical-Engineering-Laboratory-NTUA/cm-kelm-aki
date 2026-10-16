"""
INSPIRE feature-engineering pipeline -- REBUILT on the official schema.
======================================================================

Source of truth for every column name below: the dataset's own
`schema.csv` (retrieved and confirmed from the real downloaded files on
the lab server, 7 Sep 2026) plus the real CSV headers of each table
(confirmed via `zcat *.csv.gz | head -1` on the server). This SUPERSEDES
the earlier version of this module, which had inferred several
`operations.csv` columns (the `preop_*` lab fields) from a third-party
sample script (vitaldb/inspire gbm_mortality.py) that turned out not to
match the real, current table structure.

KEY CORRECTION from the previous version:
    operations.csv does NOT contain precomputed preop_creatinine,
    preop_hb, etc. columns. All laboratory values -- pre- and
    post-operative alike -- live ONLY in labs.csv, in long format
    (subject_id, chart_time, item_name, value). Preoperative lab
    features must be extracted by filtering item_name and taking the
    most recent value before surgery, exactly like the postoperative
    values used for the KDIGO label.

Both corrections are now fully resolved: this version uses the real,
confirmed operations.csv/labs.csv schema (from schema.csv) AND the real,
confirmed labs.csv item_name labels (from parameters.csv), both verified
against the actual files on the lab server on 7 Sep 2026. No placeholder
column names or item_name guesses remain in this module.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# CONFIRMED schema (from schema.csv + real file headers)
# ----------------------------------------------------------------------
OPERATIONS_COLUMNS = [
    "op_id", "subject_id", "hadm_id", "case_id", "opdate",
    "age", "sex", "weight", "height", "race", "asa", "emop",
    "department", "antype", "icd10_pcs",
    "orin_time", "orout_time", "opstart_time", "opend_time",
    "admission_time", "discharge_time",
    "anstart_time", "anend_time", "cpbon_time", "cpboff_time",
    "icuin_time", "icuout_time",
    "inhosp_death_time", "allcause_death_time",
]

DIAGNOSIS_COLUMNS = ["subject_id", "chart_time", "icd10_cm"]
VITALS_COLUMNS = ["op_id", "subject_id", "chart_time", "item_name", "value"]
WARD_VITALS_COLUMNS = ["subject_id", "chart_time", "item_name", "value"]
LABS_COLUMNS = ["subject_id", "chart_time", "item_name", "value"]

# NOTE: schema.csv describes drug_name2/drug_name3/atc_code2/atc_code3 as
# part of medications.csv, but the REAL header on disk (zcat | head -1)
# only shows 5 columns. Trusting the real header as ground truth; flagged
# for a quick sanity check once we look at a few actual rows.
MEDICATIONS_COLUMNS = ["subject_id", "chart_time", "route", "drug_name", "atc_code"]

# Department abbreviation -> full name (from department.csv)
DEPARTMENT_MAP = {
    "AN": "Anesthesiology", "CTS": "Cardio-Thoracic Surgery", "DM": "Dermatology",
    "EM": "Emergency Medicine", "GS": "General Surgery", "IM": "Internal Medicine",
    "NS": "Neurosurgery", "OG": "Obstetrics & Gynecology", "OL": "Oto-laryngology",
    "OS": "Orthopedic Surgery", "OT": "Ophthalmology", "PED": "Pediatrics",
    "PS": "Plastic Surgery", "RAD": "Radiology", "RO": "Radiation Oncology",
    "UR": "Urology",
}

# ----------------------------------------------------------------------
# CONFIRMED item_name strings for labs.csv, verified against the real
# parameters.csv (7 Sep 2026). All labs table entries use these exact
# lowercase labels.
# ----------------------------------------------------------------------
CREATININE_ITEM_NAME = "creatinine"
HEMOGLOBIN_ITEM_NAME = "hb"           # NOTE: "hb", not "hemoglobin" -- corrected
PLATELET_ITEM_NAME = "platelet"
WBC_ITEM_NAME = "wbc"
ALBUMIN_ITEM_NAME = "albumin"
GLUCOSE_ITEM_NAME = "glucose"
SODIUM_ITEM_NAME = "sodium"
POTASSIUM_ITEM_NAME = "potassium"
BUN_ITEM_NAME = "bun"                 # restored -- directly relevant to renal function
AST_ITEM_NAME = "ast"                 # restored, per Methods section's stated lab panel
ALT_ITEM_NAME = "alt"                 # restored
APTT_ITEM_NAME = "aptt"               # restored
PTINR_ITEM_NAME = "ptinr"             # restored


# ----------------------------------------------------------------------
# Step 1 -- load raw tables
# ----------------------------------------------------------------------
def load_raw_tables(data_dir=None, dummy=False):
    """
    Returns a dict of DataFrames matching the real INSPIRE table layout:
    operations, diagnosis, vitals, ward_vitals, labs, medications,
    department.

    If dummy=True, fabricates tiny synthetic tables with the confirmed
    real column names, for pipeline testing before/without touching the
    real files.
    """
    if not dummy:
        operations = pd.read_csv(f"{data_dir}/operations.csv.gz")
        diagnosis = pd.read_csv(f"{data_dir}/diagnosis.csv.gz")
        vitals = pd.read_csv(f"{data_dir}/vitals.csv.gz")
        ward_vitals = pd.read_csv(f"{data_dir}/ward_vitals.csv.gz")
        labs = pd.read_csv(f"{data_dir}/labs.csv.gz")
        medications = pd.read_csv(f"{data_dir}/medications.csv.gz")
        department = pd.read_csv(f"{data_dir}/department.csv")
        return {"operations": operations, "diagnosis": diagnosis, "vitals": vitals,
                "ward_vitals": ward_vitals, "labs": labs, "medications": medications,
                "department": department}

    rng = np.random.default_rng(0)
    n_cases = 200

    height = rng.normal(165, 9, n_cases).clip(140, 195)
    weight = rng.normal(68, 14, n_cases).clip(35, 130)
    anstart = rng.integers(0, 500, n_cases)
    andur = rng.normal(150, 60, n_cases).clip(20, 600).astype(int)
    orout = anstart + andur + rng.integers(5, 20, n_cases)

    # deliberately give ~15% of cases a REPEATED subject_id (simulating
    # patients with more than one operation), to stress-test the op_id
    # vs. subject_id distinction that caused a real bug on first contact
    # with the actual dataset (some subject_ids legitimately repeat).
    subject_ids = np.arange(1_000_000, 1_000_000 + n_cases)
    n_repeat = int(n_cases * 0.15)
    repeat_positions = rng.choice(n_cases, n_repeat, replace=False)
    donor_positions = rng.choice(n_cases, n_repeat, replace=False)
    subject_ids[repeat_positions] = subject_ids[donor_positions]

    operations = pd.DataFrame({
        "op_id": np.arange(4_000_000, 4_000_000 + n_cases),
        "subject_id": subject_ids,
        "hadm_id": np.arange(2_000_000, 2_000_000 + n_cases),
        "case_id": np.arange(n_cases),
        "opdate": rng.integers(0, 3650, n_cases),
        "age": rng.integers(18, 90, n_cases),
        "sex": rng.choice(["M", "F"], n_cases),
        "weight": weight,
        "height": height,
        "race": "Asian",
        "asa": rng.integers(1, 5, n_cases),
        "emop": rng.choice([0, 1], n_cases, p=[0.90, 0.10]),
        "department": rng.choice(list(DEPARTMENT_MAP.keys()), n_cases),
        "antype": rng.choice(["General", "Neuraxial", "Regional", "MAC"], n_cases),
        "icd10_pcs": "0DTJ0ZZ",
        "orin_time": anstart - rng.integers(5, 20, n_cases),
        "orout_time": orout,
        "opstart_time": anstart + 10,
        "opend_time": anstart + andur - 10,
        "admission_time": anstart - rng.integers(1000, 5000, n_cases),
        "discharge_time": orout + rng.integers(1000, 20000, n_cases),
        "anstart_time": anstart,
        "anend_time": anstart + andur,
        "cpbon_time": np.nan,
        "cpboff_time": np.nan,
        "icuin_time": np.where(rng.random(n_cases) < 0.3, orout + rng.integers(0, 500, n_cases), np.nan),
        "icuout_time": np.nan,
        "inhosp_death_time": np.where(rng.random(n_cases) < 0.012,
                                       orout + rng.integers(1000, 40000, n_cases), np.nan),
        "allcause_death_time": np.nan,
    })

    # long-format labs -- both PRE- and POST-operative creatinine draws
    # (needed for the KDIGO label), plus a plausible preop draw for every
    # other lab item used in build_feature_matrix, so self-tests exercise
    # the full feature set realistically (not just creatinine).
    OTHER_LAB_RANGES = {
        HEMOGLOBIN_ITEM_NAME: (13.0, 2.0), PLATELET_ITEM_NAME: (250, 70),
        WBC_ITEM_NAME: (7.0, 2.0), APTT_ITEM_NAME: (30, 5),
        PTINR_ITEM_NAME: (1.0, 0.15), GLUCOSE_ITEM_NAME: (100, 25),
        BUN_ITEM_NAME: (15, 6), ALBUMIN_ITEM_NAME: (4.0, 0.5),
        AST_ITEM_NAME: (25, 10), ALT_ITEM_NAME: (25, 12),
        SODIUM_ITEM_NAME: (140, 3), POTASSIUM_ITEM_NAME: (4.2, 0.4),
    }
    rows = []
    for sid, aidx in zip(operations["subject_id"], range(n_cases)):
        baseline_cr = rng.normal(0.9, 0.3)
        an_start = operations.loc[aidx, "anstart_time"]
        or_out = operations.loc[aidx, "orout_time"]
        for days_before in rng.choice(range(1, 30), 2, replace=False):
            rows.append((sid, an_start - int(days_before) * 24 * 60,
                         CREATININE_ITEM_NAME, max(baseline_cr + rng.normal(0, 0.05), 0.2)))
        trend = rng.choice([0, 1], p=[0.85, 0.15])
        n_draws = rng.integers(2, 6)
        for day in sorted(rng.choice(range(1, 8), n_draws, replace=False)):
            value = baseline_cr + (trend * rng.uniform(0.3, 1.5)) + rng.normal(0, 0.1)
            rows.append((sid, or_out + int(day) * 24 * 60, CREATININE_ITEM_NAME, max(value, 0.2)))
        # one preop draw for every other lab item (occasionally missing,
        # like the real data, via a 5% random skip)
        for item, (mean, std) in OTHER_LAB_RANGES.items():
            if rng.random() < 0.05:
                continue
            days_before = rng.integers(1, 30)
            rows.append((sid, an_start - int(days_before) * 24 * 60,
                         item, max(rng.normal(mean, std), 0.01)))
    labs = pd.DataFrame(rows, columns=LABS_COLUMNS)

    diagnosis = pd.DataFrame(columns=DIAGNOSIS_COLUMNS)
    vitals = pd.DataFrame(columns=VITALS_COLUMNS)
    ward_vitals = pd.DataFrame(columns=WARD_VITALS_COLUMNS)
    medications = pd.DataFrame(columns=MEDICATIONS_COLUMNS)
    department = pd.DataFrame(list(DEPARTMENT_MAP.items()), columns=["Abbreviations", "Full name"])

    return {"operations": operations, "diagnosis": diagnosis, "vitals": vitals,
            "ward_vitals": ward_vitals, "labs": labs, "medications": medications,
            "department": department}


# ----------------------------------------------------------------------
# Step 2 -- extract a preoperative lab value (generic helper)
# ----------------------------------------------------------------------
def extract_preop_lab(labs, operations, item_name, lookback_days=365,
                       cutoff_col="anstart_time"):
    """
    For each OPERATION (op_id), returns the most recent labs.csv value
    for the given item_name, drawn from any time strictly before that
    operation's `cutoff_col` (default: anesthesia start), within
    `lookback_days`.

    IMPORTANT: labs.csv is keyed by subject_id only (no op_id), and a
    single subject_id can correspond to MULTIPLE operations (repeat
    surgeries). We therefore join labs to operations on subject_id
    (deliberately many-to-many -- every lab draw is a candidate for
    every operation of that same patient), filter each candidate against
    THAT OPERATION's own cutoff time, and finally group by op_id (not
    subject_id) so that a patient with several operations gets a
    separate, correctly time-aligned preop value for each one.
    """
    sub = labs[labs["item_name"] == item_name][["subject_id", "chart_time", "value"]]
    cutoffs = operations[["op_id", "subject_id", cutoff_col]].rename(columns={cutoff_col: "cutoff"})
    merged = sub.merge(cutoffs, on="subject_id", how="inner")
    merged = merged[(merged["chart_time"] < merged["cutoff"]) &
                     (merged["chart_time"] >= merged["cutoff"] - lookback_days * 24 * 60)]
    # most recent (closest to cutoff) value per OPERATION
    merged = merged.sort_values("chart_time").groupby("op_id").tail(1)
    return merged.set_index("op_id")["value"]


# ----------------------------------------------------------------------
# Step 3 -- KDIGO AKI label (postoperative creatinine, from labs.csv)
# ----------------------------------------------------------------------
def compute_kdigo_aki_label(operations, labs, window_days=7,
                             item_name=CREATININE_ITEM_NAME):
    """KDIGO stage >=1: postop creatinine >= 1.5x preop baseline within window_days.
    Indexed and computed per OPERATION (op_id), not per patient, since a
    patient with multiple surgeries needs a separate label -- and a
    separate postop creatinine window relative to THAT operation's own
    orout_time -- for each one."""
    baseline = extract_preop_lab(labs, operations, item_name, cutoff_col="anstart_time")  # indexed by op_id

    cr_labs = labs[labs["item_name"] == item_name][["subject_id", "chart_time", "value"]]
    op_cutoffs = operations[["op_id", "subject_id", "orout_time"]]
    merged = cr_labs.merge(op_cutoffs, on="subject_id", how="inner")  # many-to-many by design
    merged["days_since_orout"] = (merged["chart_time"] - merged["orout_time"]) / (24 * 60)
    merged = merged[(merged["days_since_orout"] >= 0) & (merged["days_since_orout"] <= window_days)]
    peak = merged.groupby("op_id")["value"].max()

    label = pd.Series(0, index=operations["op_id"], name="aki_label")
    for op_id in label.index:
        if op_id in peak.index and op_id in baseline.index and baseline[op_id] > 0:
            if peak[op_id] / baseline[op_id] >= 1.5:
                label[op_id] = 1
    return label


# ----------------------------------------------------------------------
# Step 4 -- assemble the feature matrix (pre-outcome information only)
# ----------------------------------------------------------------------
def build_feature_matrix(operations, labs):
    """
    Combines operations.csv fields (available by end of surgery) with
    preoperative lab values extracted from labs.csv. No postoperative
    lab value, and no death/ICU/discharge column, may enter this matrix
    -- enforced by an explicit leakage guard below.
    """
    df = operations.copy()
    df["andur"] = df["anend_time"] - df["anstart_time"]
    # guard against height==0 (or implausibly small, e.g. a data artifact)
    # producing a division-by-zero / infinite BMI -- treat as missing
    # (to be median-imputed downstream) rather than propagating inf
    safe_height = df["height"].where(df["height"] > 50, np.nan)
    df["bmi"] = df["weight"] / (safe_height / 100) ** 2

    df = pd.get_dummies(df, columns=["sex", "department", "antype"], drop_first=True)

    # preoperative labs, extracted from the long-format labs table
    preop_features = {}
    for label, item in [
        ("preop_creatinine", CREATININE_ITEM_NAME),
        ("preop_hb", HEMOGLOBIN_ITEM_NAME),
        ("preop_platelet", PLATELET_ITEM_NAME),
        ("preop_wbc", WBC_ITEM_NAME),
        ("preop_aptt", APTT_ITEM_NAME),
        ("preop_ptinr", PTINR_ITEM_NAME),
        ("preop_glucose", GLUCOSE_ITEM_NAME),
        ("preop_bun", BUN_ITEM_NAME),
        ("preop_albumin", ALBUMIN_ITEM_NAME),
        ("preop_ast", AST_ITEM_NAME),
        ("preop_alt", ALT_ITEM_NAME),
        ("preop_sodium", SODIUM_ITEM_NAME),
        ("preop_potassium", POTASSIUM_ITEM_NAME),
    ]:
        preop_features[label] = extract_preop_lab(labs, operations, item, cutoff_col="anstart_time")

    for label, series in preop_features.items():
        df[label] = df["op_id"].map(series)

    numeric_feats = [
        "age", "bmi", "asa", "emop", "andur",
        "preop_creatinine", "preop_hb", "preop_platelet", "preop_wbc",
        "preop_aptt", "preop_ptinr", "preop_glucose", "preop_bun",
        "preop_albumin", "preop_ast", "preop_alt", "preop_sodium", "preop_potassium",
    ]
    dummy_feats = [c for c in df.columns if c.startswith(("sex_", "department_", "antype_"))]
    feature_cols = numeric_feats + dummy_feats

    X = df[feature_cols].copy()

    # EXPLICIT leakage guard
    forbidden = {"inhosp_death_time", "allcause_death_time", "icuin_time", "icuout_time",
                 "discharge_time", "icd10_pcs"}
    leaked = forbidden & set(X.columns)
    if leaked:
        raise ValueError(f"Leakage guard triggered -- forbidden columns in features: {leaked}")

    subject_ids = df["subject_id"].values
    return X, subject_ids


# ----------------------------------------------------------------------
# Step 3b -- cohort inclusion/exclusion criteria (per Materials section)
# ----------------------------------------------------------------------
def apply_cohort_criteria(operations, labs, window_days=7,
                           item_name=CREATININE_ITEM_NAME,
                           first_procedure_only=True,
                           max_baseline_creatinine=None,
                           esrd=None, diagnosis=None,
                           verbose=True):
    """
    Applies the inclusion/exclusion criteria stated in the manuscript's
    Materials section (4.2):
      - require a preoperative baseline creatinine (needed to determine
        AKI status at all -- cases without one CANNOT be silently
        labeled "no AKI", they must be excluded)
      - require at least one postoperative creatinine draw within
        `window_days` (same reasoning: no follow-up means AKI status is
        genuinely unknown, not negative)
      - optionally retain only the first eligible procedure per patient
        (subject_id), consistent with the "first eligible procedure per
        patient retained" criterion already stated in Materials 4.2 --
        avoids repeated-measures non-independence across operations of
        the same patient
      - optionally exclude cases with an implausibly high baseline
        creatinine suggestive of pre-existing ESRD/dialysis dependence
        (disabled by default: this is a coarse heuristic, not a real
        ESRD flag -- a proper implementation would check diagnosis.csv
        for relevant ICD-10-CM codes, left as a future refinement)

      - esrd="cr4_z49" (recommended, 27 Sep 2026): after the first-procedure
        step, exclude patients with pre-existing kidney failure, defined as
        baseline creatinine >= 4.0 mg/dL OR an ICD-10 Z49 (dialysis care) code
        recorded BEFORE anaesthesia start. esrd="cr4_z49_n18" also excludes
        pre-operative N18 (CKD, any stage: a sensitivity analysis only).
        Only pre-operative codes are used: codes recorded later include
        dialysis started for the postoperative AKI itself. Needs `diagnosis`.

    Returns (operations_filtered, labs) and prints a cohort attrition
    report if verbose=True.
    """
    n_start = len(operations)
    ops = operations.copy()
    steps = [("Starting cohort (all operations)", len(ops))]

    baseline = extract_preop_lab(labs, ops, item_name, cutoff_col="anstart_time")
    ops = ops[ops["op_id"].isin(baseline.index)]
    steps.append(("+ require preoperative baseline creatinine", len(ops)))

    cr_labs = labs[labs["item_name"] == item_name][["subject_id", "chart_time", "value"]]
    op_cutoffs = ops[["op_id", "subject_id", "orout_time"]]
    merged = cr_labs.merge(op_cutoffs, on="subject_id", how="inner")
    merged["days_since_orout"] = (merged["chart_time"] - merged["orout_time"]) / (24 * 60)
    has_followup = set(merged[(merged["days_since_orout"] >= 0) &
                               (merged["days_since_orout"] <= window_days)]["op_id"])
    ops = ops[ops["op_id"].isin(has_followup)]
    steps.append((f"+ require postoperative creatinine within {window_days} days", len(ops)))

    if max_baseline_creatinine is not None:
        eligible_baseline = baseline[baseline <= max_baseline_creatinine]
        ops = ops[ops["op_id"].isin(eligible_baseline.index)]
        steps.append((f"+ exclude baseline creatinine > {max_baseline_creatinine} mg/dL "
                      f"(coarse pre-existing-ESRD heuristic)", len(ops)))

    if first_procedure_only:
        ops = ops.sort_values("anstart_time").groupby("subject_id", as_index=False).first()
        steps.append(("+ first eligible procedure per patient only", len(ops)))

    if esrd is not None:
        excl = esrd_op_ids(ops, labs, diagnosis, esrd, item_name=item_name)
        ops = ops[~ops["op_id"].isin(excl)]
        steps.append((f"+ exclude pre-existing kidney failure ({esrd})", len(ops)))

    if verbose:
        print("Cohort attrition report:")
        prev = n_start
        for description, n in steps:
            delta = n - prev
            print(f"  {description}: n={n}" + (f"  ({delta:+d})" if description != steps[0][0] else ""))
            prev = n
        print(f"Final cohort: {len(ops)} / {n_start} operations retained "
              f"({100 * len(ops) / n_start:.1f}%)")

    return ops, labs


# INSPIRE >= v1.1 codes every lab value to its nearest 5th-percentile value
# (bottom/top coded at the 2.5th/97.5th percentile). Creatinine therefore takes
# 21 values: ..., 1.33, 1.68 (P90), 3.03 (P95, true ~2.0-5.5), 5.54 (P97.5, top
# code = every true value >= 5.54). Any threshold in (3.03, 5.54] selects exactly
# the top-coded category, in which AKI cannot be ascertained at all (no higher
# value exists): these patients must be excluded regardless of ESRD status.
#   cr4_z49     primary      : top-coded baseline (>= 5.54) OR pre-op Z49
#   cr2_z49     sensitivity  : also the P95 category (true ~2.0-5.5), where a
#                              1.5x rise is detectable only as a jump to the top code
#   cr4_z49_n18 sensitivity  : + pre-op N18 (CKD, any stage)
ESRD_RULES = {"cr4_z49": (4.0, ("Z49",)), "cr2_z49": (2.0, ("Z49",)),
              "cr4_z49_n18": (4.0, ("Z49", "N18"))}


def esrd_op_ids(ops, labs, diagnosis, rule="cr4_z49", item_name=CREATININE_ITEM_NAME):
    """op_ids with pre-existing kidney failure under `rule` (see ESRD_RULES):
    baseline creatinine >= threshold OR a listed 3-character ICD-10 code
    charted before anaesthesia start."""
    if rule not in ESRD_RULES:
        raise ValueError(f"unknown ESRD rule '{rule}', choose from {list(ESRD_RULES)}")
    if diagnosis is None:
        raise ValueError("ESRD exclusion needs the diagnosis table")
    thr, codes = ESRD_RULES[rule]
    base = extract_preop_lab(labs, ops, item_name, cutoff_col="anstart_time")
    excl = set(base.index[base >= thr])
    d = diagnosis[diagnosis["icd10_cm"].astype(str).str[:3].isin(codes)]
    dm = d.merge(ops[["op_id", "subject_id", "anstart_time"]], on="subject_id", how="inner")
    excl |= set(dm.loc[dm["chart_time"] < dm["anstart_time"], "op_id"])
    return excl & set(ops["op_id"])


# ----------------------------------------------------------------------
# Step 3c -- characterize selection bias (included vs excluded cohort)
# ----------------------------------------------------------------------
def compare_included_vs_excluded(operations_full, included_op_ids,
                                  numeric_cols=("age", "asa"),
                                  categorical_cols=("sex", "emop", "department", "antype")):
    """
    Compares the final analytic cohort against everyone excluded by
    apply_cohort_criteria(), across key variables -- intended to
    characterize (not hide) the selection bias introduced mainly by the
    "postoperative creatinine within window" requirement, for reporting
    in the manuscript's Discussion/Limitations section (e.g. as a
    Table 1-style included-vs-excluded comparison).

    numeric_cols: compared via group mean/median/std + Mann-Whitney U test
                  (non-parametric, robust to the skewed distributions
                  typical of clinical variables like age/ASA).
    categorical_cols: compared via group proportions + chi-square test
                       of independence.

    Returns a dict with two DataFrames ("numeric_summary",
    "categorical_summary") and prints a concise report.
    """
    from scipy.stats import mannwhitneyu, chi2_contingency

    df = operations_full.copy()
    df["included"] = df["op_id"].isin(included_op_ids)
    included = df[df["included"]]
    excluded = df[~df["included"]]

    print(f"Included: n={len(included)}  |  Excluded: n={len(excluded)}\n")

    numeric_rows = []
    for col in numeric_cols:
        inc_vals = pd.to_numeric(included[col], errors="coerce").dropna()
        exc_vals = pd.to_numeric(excluded[col], errors="coerce").dropna()
        if len(inc_vals) == 0 or len(exc_vals) == 0:
            continue
        stat, p = mannwhitneyu(inc_vals, exc_vals, alternative="two-sided")
        numeric_rows.append({
            "variable": col,
            "included_mean": round(inc_vals.mean(), 2), "included_median": round(inc_vals.median(), 2),
            "excluded_mean": round(exc_vals.mean(), 2), "excluded_median": round(exc_vals.median(), 2),
            "mannwhitney_p": p,
        })
    numeric_summary = pd.DataFrame(numeric_rows)

    categorical_rows = []
    for col in categorical_cols:
        inc_counts = included[col].value_counts()
        exc_counts = excluded[col].value_counts()
        contingency = pd.DataFrame({"included": inc_counts, "excluded": exc_counts}).fillna(0)
        if contingency.shape[0] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(contingency.T.values)
        top_level = inc_counts.idxmax()
        categorical_rows.append({
            "variable": col,
            "most_common_included": top_level,
            "included_pct_of_top_level": round(100 * inc_counts.get(top_level, 0) / len(included), 1),
            "excluded_pct_of_top_level": round(100 * exc_counts.get(top_level, 0) / max(len(excluded), 1), 1),
            "chi2_p": p,
        })
    categorical_summary = pd.DataFrame(categorical_rows)

    print("Numeric variables (included vs excluded):")
    print(numeric_summary.to_string(index=False) if len(numeric_summary) else "  (none)")
    print("\nCategorical variables (included vs excluded, top level shown):")
    print(categorical_summary.to_string(index=False) if len(categorical_summary) else "  (none)")

    return {"numeric_summary": numeric_summary, "categorical_summary": categorical_summary}


# ----------------------------------------------------------------------
# Self-test with dummy data
# ----------------------------------------------------------------------
def _self_test():
    tables = load_raw_tables(dummy=True)
    operations, labs = tables["operations"], tables["labs"]

    labels = compute_kdigo_aki_label(operations, labs)
    X, subject_ids = build_feature_matrix(operations, labs)

    assert len(X) == len(operations), "row count mismatch after feature assembly"
    assert X[["age", "asa"]].isnull().sum().sum() == 0, "unexpected NaNs in core demographic features"

    n_missing_preop_cr = X["preop_creatinine"].isnull().sum()
    print(f"Self-test OK: {X.shape[0]} cases, {X.shape[1]} features, "
          f"AKI positive rate = {labels.mean():.3f}")
    print(f"  (missing preop_creatinine: {n_missing_preop_cr}/{len(X)} -- "
          f"expected to be 0 in this synthetic test, would be nonzero on real "
          f"data for patients without a qualifying preop draw)")
    print(f"Feature columns: {list(X.columns)}")
    print("\nAll item_name values confirmed against the real parameters.csv -- no placeholders remain.")

    print("\n--- apply_cohort_criteria self-test ---")
    ops_filtered, _ = apply_cohort_criteria(operations, labs, first_procedure_only=True)
    assert len(ops_filtered) <= len(operations), "filtering should never increase cohort size"
    assert ops_filtered["subject_id"].is_unique, "first_procedure_only should leave one row per patient"

    print("\n--- compare_included_vs_excluded self-test ---")
    result = compare_included_vs_excluded(operations, set(ops_filtered["op_id"]))
    assert "numeric_summary" in result and "categorical_summary" in result


if __name__ == "__main__":
    _self_test()
