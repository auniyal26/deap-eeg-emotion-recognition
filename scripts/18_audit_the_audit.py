import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None):
        if iterable is None:
            return range(total or 0)
        print(desc or "Progress")
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TABLE_DIR = REPO_ROOT / "results" / "tables"
METRIC_DIR = REPO_ROOT / "results" / "metrics"

OUT_CHECKS = TABLE_DIR / "deap_audit_of_audit_checks.csv"
OUT_CLAIMS = TABLE_DIR / "deap_audit_of_audit_claim_verification.csv"
OUT_LEAKAGE = TABLE_DIR / "deap_audit_of_audit_fold_leakage.csv"
OUT_FEATURES = TABLE_DIR / "deap_audit_of_audit_feature_integrity.csv"
OUT_METADATA = METRIC_DIR / "deap_audit_of_audit_metadata.json"

LABEL_COLS = ["valence", "arousal", "dominance", "liking"]
VA_COLS = ["valence", "arousal"]
META_COLS = ["subject", "trial"] + LABEL_COLS

FEATURE_FILES = {
    "absolute": "bandpower_features_absolute.csv",
    "log": "bandpower_features_log.csv",
    "relative_channel": "bandpower_features_relative_channel.csv",
    "trial_proportion": "bandpower_features_trial_proportion.csv",
    "asymmetry_only": "bandpower_features_asymmetry_only.csv",
    "absolute_plus_asymmetry": "bandpower_features_absolute_plus_asymmetry.csv",
    "log_plus_asymmetry": "bandpower_features_log_plus_asymmetry.csv",
    "relative_channel_plus_asymmetry": "bandpower_features_relative_channel_plus_asymmetry.csv",
    "trial_proportion_plus_asymmetry": "bandpower_features_trial_proportion_plus_asymmetry.csv",
}

EXPECTED_FEATURE_COUNTS = {
    "absolute": 160,
    "log": 160,
    "relative_channel": 160,
    "trial_proportion": 160,
    "asymmetry_only": 70,
    "absolute_plus_asymmetry": 230,
    "log_plus_asymmetry": 230,
    "relative_channel_plus_asymmetry": 230,
    "trial_proportion_plus_asymmetry": 230,
}

FOLD_FILES = {
    "regression_baseline": "regression_baseline_fold_metrics_absolute.csv",
    "affect_score": "affect_score_baseline_fold_metrics_absolute.csv",
    "pls_axis": "pls_axis_fold_metrics_absolute.csv",
    "master_axis": "master_axis_audit_fold_metrics.csv",
    "asymmetry_axis": "asymmetry_axis_audit_fold_metrics.csv",
    "split_optimism": "deap_split_optimism_fold_metrics.csv",
    "label_space_4d": "deap_label_space_4d_fold_metrics.csv",
    "loso": "deap_loso_axis_audit_fold_metrics.csv",
}

MANIFEST_FILES = {
    "regression_baseline": "regression_baseline_groupkfold_manifest.csv",
    "affect_score": "affect_score_groupkfold_manifest.csv",
    "pls_axis": "pls_axis_groupkfold_manifest.csv",
    "master_axis": "master_axis_audit_groupkfold_manifest.csv",
    "asymmetry_axis": "asymmetry_axis_audit_groupkfold_manifest.csv",
}

CLAIM_FILES = {
    "master_table": "deap_master_audit_table.csv",
    "master_by_target": "deap_master_audit_by_target.csv",
    "master_by_representation": "deap_master_audit_by_representation.csv",
    "split_optimism_summary": "deap_split_optimism_summary.csv",
    "split_optimism_table": "deap_split_optimism_table.csv",
    "label_space_4d_summary": "deap_label_space_4d_summary.csv",
    "loso_summary": "deap_loso_axis_audit_summary.csv",
    "statistical_tests": "deap_statistical_tests_vs_dummy.csv",
}


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def status_from_bool(value):
    return "PASS" if value else "FAIL"


def add_check(rows, group, check_name, status, details, source_file="", expected_value="", observed_value=""):
    rows.append(
        {
            "group": group,
            "check_name": check_name,
            "status": status,
            "details": details,
            "source_file": source_file,
            "expected_value": expected_value,
            "observed_value": observed_value,
        }
    )


def read_csv_if_exists(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def finite_numeric_frame(df, cols):
    if len(cols) == 0:
        return True
    values = df[cols].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def parse_subject_set(value):
    if pd.isna(value):
        return set()
    text = str(value)
    for token in [";", "|", ","]:
        if token in text:
            return set(x.strip() for x in text.split(token) if x.strip())
    return set(x.strip() for x in text.split() if x.strip())


def infer_score_column(df):
    for col in ["score", "rmse_or_error", "model_score", "best_model_score", "primary_score", "rmse", "mean_2d_error", "valence_rmse", "arousal_rmse"]:
        if col in df.columns:
            return col
    return None


def audit_raw_labels(check_rows):
    path = TABLE_DIR / "raw_trial_labels.csv"
    df = read_csv_if_exists(path)

    if df is None:
        add_check(check_rows, "raw_labels", "raw_trial_labels_exists", "FAIL", "Missing raw trial labels.", str(path.relative_to(REPO_ROOT)))
        return None

    add_check(check_rows, "raw_labels", "raw_trial_labels_exists", "PASS", "Raw trial labels file exists.", str(path.relative_to(REPO_ROOT)))

    required = ["subject", "trial"] + LABEL_COLS
    missing = [c for c in required if c not in df.columns]
    add_check(check_rows, "raw_labels", "required_columns", status_from_bool(len(missing) == 0), f"Missing columns: {missing}", str(path.relative_to(REPO_ROOT)), str(required), str(list(df.columns)))

    if missing:
        return df

    df["subject"] = df["subject"].astype(str)
    df["trial"] = df["trial"].astype(int)

    add_check(check_rows, "raw_labels", "n_rows_1280", status_from_bool(len(df) == 1280), "DEAP should have 32 subjects × 40 trials.", str(path.relative_to(REPO_ROOT)), "1280", str(len(df)))
    add_check(check_rows, "raw_labels", "n_subjects_32", status_from_bool(df["subject"].nunique() == 32), "DEAP should have 32 subjects.", str(path.relative_to(REPO_ROOT)), "32", str(df["subject"].nunique()))

    trial_counts = df.groupby("subject")["trial"].nunique()
    trial_count_ok = bool((trial_counts == 40).all() and len(trial_counts) == 32)
    add_check(check_rows, "raw_labels", "forty_trials_per_subject", status_from_bool(trial_count_ok), "Each subject should have 40 trials.", str(path.relative_to(REPO_ROOT)), "all subjects = 40", trial_counts.describe().to_json())

    duplicate_count = int(df.duplicated(["subject", "trial"]).sum())
    add_check(check_rows, "raw_labels", "unique_subject_trial", status_from_bool(duplicate_count == 0), "subject/trial pairs should be unique.", str(path.relative_to(REPO_ROOT)), "0 duplicates", str(duplicate_count))

    label_min = df[LABEL_COLS].min().to_dict()
    label_max = df[LABEL_COLS].max().to_dict()
    range_ok = bool((df[LABEL_COLS].min().min() >= 1.0) and (df[LABEL_COLS].max().max() <= 9.0))
    add_check(check_rows, "raw_labels", "labels_range_1_to_9", status_from_bool(range_ok), "All DEAP labels should be in 1–9 range.", str(path.relative_to(REPO_ROOT)), "1 <= labels <= 9", f"min={label_min}, max={label_max}")

    expected_s01 = np.array(
        [
            [7.71, 7.60, 6.90, 7.83],
            [8.10, 7.31, 7.28, 8.47],
            [8.58, 7.54, 9.00, 7.08],
        ]
    )

    s01 = df[df["subject"] == "s01"].sort_values("trial").head(3)[LABEL_COLS].to_numpy(dtype=float)
    s01_ok = bool(s01.shape == expected_s01.shape and np.allclose(s01, expected_s01, atol=1e-6))
    add_check(check_rows, "raw_labels", "s01_first_three_official_match", status_from_bool(s01_ok), "First three s01 labels should match the verified clean DEAP values.", str(path.relative_to(REPO_ROOT)), str(expected_s01.tolist()), str(s01.tolist()))

    means = df[LABEL_COLS].mean().to_dict()
    expected_means = {
        "valence": 5.2543125,
        "arousal": 5.1567109375,
        "dominance": 5.38275,
        "liking": 5.5181328125,
    }
    means_ok = bool(all(abs(means[k] - expected_means[k]) < 1e-9 for k in LABEL_COLS))
    add_check(check_rows, "raw_labels", "global_label_means_match_clean_run", status_from_bool(means_ok), "Global means should match the verified clean DEAP integrity run.", str(path.relative_to(REPO_ROOT)), str(expected_means), str(means))

    hh = int(((df["valence"] > 5) & (df["arousal"] > 5)).sum())
    add_check(check_rows, "raw_labels", "high_high_quadrant_nonzero_clean", status_from_bool(hh == 439), "Clean labels should have 439 high-valence/high-arousal trials.", str(path.relative_to(REPO_ROOT)), "439", str(hh))

    return df


def audit_feature_files(check_rows, feature_rows, labels):
    for representation, filename in tqdm(FEATURE_FILES.items(), desc="Auditing feature files"):
        path = TABLE_DIR / filename
        df = read_csv_if_exists(path)

        if df is None:
            add_check(check_rows, "features", f"{representation}_exists", "FAIL", "Missing feature file.", str(path.relative_to(REPO_ROOT)))
            feature_rows.append({"representation": representation, "source_file": filename, "status": "FAIL", "details": "missing"})
            continue

        required = META_COLS
        missing = [c for c in required if c not in df.columns]
        feature_cols = [c for c in df.columns if c not in META_COLS]
        numeric_cols = feature_cols

        row = {
            "representation": representation,
            "source_file": filename,
            "status": "PASS",
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "n_features": int(len(feature_cols)),
            "expected_features": int(EXPECTED_FEATURE_COUNTS[representation]),
            "n_subjects": int(df["subject"].astype(str).nunique()) if "subject" in df.columns else np.nan,
            "duplicate_subject_trial": int(df.duplicated(["subject", "trial"]).sum()) if "subject" in df.columns and "trial" in df.columns else np.nan,
            "missing_required_columns": str(missing),
            "finite_features": False,
            "labels_match_raw": False,
            "details": "",
        }

        checks = []

        checks.append(len(missing) == 0)
        checks.append(len(df) == 1280)
        checks.append(len(feature_cols) == EXPECTED_FEATURE_COUNTS[representation])

        if "subject" in df.columns and "trial" in df.columns:
            df["subject"] = df["subject"].astype(str)
            df["trial"] = df["trial"].astype(int)
            checks.append(df["subject"].nunique() == 32)
            checks.append(df.duplicated(["subject", "trial"]).sum() == 0)
            trial_counts = df.groupby("subject")["trial"].nunique()
            checks.append(bool((trial_counts == 40).all() and len(trial_counts) == 32))
        else:
            checks.extend([False, False, False])

        finite_ok = finite_numeric_frame(df, numeric_cols)
        row["finite_features"] = finite_ok
        checks.append(finite_ok)

        labels_match = False
        if labels is not None and len(missing) == 0:
            left = df[["subject", "trial"] + LABEL_COLS].copy()
            left["subject"] = left["subject"].astype(str)
            left["trial"] = left["trial"].astype(int)
            right = labels[["subject", "trial"] + LABEL_COLS].copy()
            right["subject"] = right["subject"].astype(str)
            right["trial"] = right["trial"].astype(int)
            merged = left.merge(right, on=["subject", "trial"], suffixes=("_feature", "_raw"), how="inner")
            if len(merged) == 1280:
                label_checks = []
                for col in LABEL_COLS:
                    label_checks.append(np.allclose(merged[f"{col}_feature"], merged[f"{col}_raw"], atol=1e-12))
                labels_match = bool(all(label_checks))
        row["labels_match_raw"] = labels_match
        checks.append(labels_match)

        status = "PASS" if all(checks) else "FAIL"
        row["status"] = status
        row["details"] = "all checks passed" if status == "PASS" else "one or more feature integrity checks failed"
        feature_rows.append(row)

        add_check(check_rows, "features", f"{representation}_row_count", status_from_bool(len(df) == 1280), "Feature file should have 1280 rows.", str(path.relative_to(REPO_ROOT)), "1280", str(len(df)))
        add_check(check_rows, "features", f"{representation}_feature_count", status_from_bool(len(feature_cols) == EXPECTED_FEATURE_COUNTS[representation]), "Feature count should match expected representation size.", str(path.relative_to(REPO_ROOT)), str(EXPECTED_FEATURE_COUNTS[representation]), str(len(feature_cols)))
        add_check(check_rows, "features", f"{representation}_finite_features", status_from_bool(finite_ok), "Feature values should be finite.", str(path.relative_to(REPO_ROOT)), "finite", str(finite_ok))
        add_check(check_rows, "features", f"{representation}_labels_match_raw", status_from_bool(labels_match), "Feature-table labels should match raw_trial_labels.csv.", str(path.relative_to(REPO_ROOT)), "True", str(labels_match))


def audit_manifest_leakage(check_rows, leakage_rows):
    for name, filename in tqdm(MANIFEST_FILES.items(), desc="Auditing fold manifests"):
        path = TABLE_DIR / filename
        df = read_csv_if_exists(path)

        if df is None:
            add_check(check_rows, "leakage", f"{name}_manifest_exists", "WARN", "Manifest file missing; fold leakage could not be checked from manifest.", str(path.relative_to(REPO_ROOT)))
            leakage_rows.append({"source": name, "source_file": filename, "status": "WARN", "details": "missing manifest"})
            continue

        train_col = None
        test_col = None

        for col in df.columns:
            low = col.lower()
            if "train" in low and "subject" in low:
                train_col = col
            if "test" in low and "subject" in low:
                test_col = col

        if train_col is None or test_col is None:
            add_check(check_rows, "leakage", f"{name}_manifest_subject_columns", "WARN", "Could not identify train/test subject columns.", str(path.relative_to(REPO_ROOT)), "train/test subject columns", str(list(df.columns)))
            leakage_rows.append({"source": name, "source_file": filename, "status": "WARN", "details": "subject columns not found"})
            continue

        overlaps = []
        fold_rows = []

        for _, row in df.iterrows():
            train_subjects = parse_subject_set(row[train_col])
            test_subjects = parse_subject_set(row[test_col])
            overlap = sorted(train_subjects.intersection(test_subjects))
            overlaps.append(overlap)
            fold_rows.append(
                {
                    "source": name,
                    "source_file": filename,
                    "fold": row["fold"] if "fold" in df.columns else "",
                    "status": "PASS" if len(overlap) == 0 else "FAIL",
                    "train_subject_count": len(train_subjects),
                    "test_subject_count": len(test_subjects),
                    "overlap_count": len(overlap),
                    "overlap_subjects": ",".join(overlap),
                    "details": "no subject overlap" if len(overlap) == 0 else "subject leakage detected",
                }
            )

        leakage_rows.extend(fold_rows)
        ok = all(len(x) == 0 for x in overlaps)
        add_check(check_rows, "leakage", f"{name}_no_train_test_subject_overlap", status_from_bool(ok), "Train and test subjects should be disjoint in grouped validation.", str(path.relative_to(REPO_ROOT)), "0 overlaps", str([x for x in overlaps if len(x) > 0]))


def audit_loso_leakage(check_rows, leakage_rows):
    path = TABLE_DIR / "deap_loso_axis_audit_fold_metrics.csv"
    df = read_csv_if_exists(path)

    if df is None:
        add_check(check_rows, "leakage", "loso_fold_metrics_exists", "WARN", "LOSO fold metrics missing.", str(path.relative_to(REPO_ROOT)))
        return

    if "heldout_subject" not in df.columns or "fold" not in df.columns:
        add_check(check_rows, "leakage", "loso_has_fold_and_subject", "FAIL", "LOSO file should have fold and heldout_subject columns.", str(path.relative_to(REPO_ROOT)), "fold, heldout_subject", str(list(df.columns)))
        return

    per_fold = df.groupby("fold")["heldout_subject"].nunique()
    unique_subjects = df["heldout_subject"].astype(str).nunique()
    ok_one_subject = bool((per_fold == 1).all())
    ok_32 = bool(unique_subjects == 32)

    add_check(check_rows, "leakage", "loso_one_heldout_subject_per_fold", status_from_bool(ok_one_subject), "Each LOSO fold should contain exactly one held-out subject.", str(path.relative_to(REPO_ROOT)), "1 per fold", per_fold.describe().to_json())
    add_check(check_rows, "leakage", "loso_32_unique_heldout_subjects", status_from_bool(ok_32), "LOSO should cover all 32 subjects.", str(path.relative_to(REPO_ROOT)), "32", str(unique_subjects))

    for fold, sub in df.groupby("fold"):
        subjects = sorted(sub["heldout_subject"].astype(str).unique())
        leakage_rows.append(
            {
                "source": "loso",
                "source_file": "deap_loso_axis_audit_fold_metrics.csv",
                "fold": fold,
                "status": "PASS" if len(subjects) == 1 else "FAIL",
                "train_subject_count": np.nan,
                "test_subject_count": len(subjects),
                "overlap_count": np.nan,
                "overlap_subjects": "",
                "details": f"heldout_subjects={subjects}",
            }
        )


def audit_fold_metric_files(check_rows):
    for name, filename in tqdm(FOLD_FILES.items(), desc="Auditing fold metric schemas"):
        path = TABLE_DIR / filename
        df = read_csv_if_exists(path)

        if df is None:
            add_check(check_rows, "fold_metrics", f"{name}_exists", "WARN", "Fold metric file missing.", str(path.relative_to(REPO_ROOT)))
            continue

        score_col = infer_score_column(df)
        has_model = "model" in df.columns or "model_name" in df.columns or "best_model" in df.columns
        has_fold = "fold" in df.columns or "fold_id" in df.columns
        has_target = "target_name" in df.columns or "target" in df.columns or "axis_name" in df.columns or name in ["regression_baseline", "affect_score"]
        score_ok = score_col is not None

        add_check(check_rows, "fold_metrics", f"{name}_has_score_column", status_from_bool(score_ok), "Fold metric file should contain a usable score column.", str(path.relative_to(REPO_ROOT)), "score-like column", str(score_col))
        add_check(check_rows, "fold_metrics", f"{name}_has_model_column", status_from_bool(has_model), "Fold metric file should contain a model column.", str(path.relative_to(REPO_ROOT)), "model-like column", str(has_model))
        add_check(check_rows, "fold_metrics", f"{name}_has_fold_column", status_from_bool(has_fold), "Fold metric file should contain a fold column.", str(path.relative_to(REPO_ROOT)), "fold-like column", str(has_fold))
        add_check(check_rows, "fold_metrics", f"{name}_has_target_column", status_from_bool(has_target), "Fold metric file should contain a target column or be a known baseline file.", str(path.relative_to(REPO_ROOT)), "target-like column or known baseline", str(has_target))

        if score_col is not None:
            score_values = pd.to_numeric(df[score_col], errors="coerce")
            finite_fraction = float(np.isfinite(score_values).mean())
            add_check(check_rows, "fold_metrics", f"{name}_score_finite_fraction", "PASS" if finite_fraction > 0 else "FAIL", "At least some scores should be finite.", str(path.relative_to(REPO_ROOT)), ">0", str(finite_fraction))


def col_first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def audit_master_claims(claim_rows, check_rows):
    path = TABLE_DIR / "deap_master_audit_table.csv"
    df = read_csv_if_exists(path)

    if df is None:
        add_check(check_rows, "claims", "master_audit_table_exists", "FAIL", "Master audit table missing.", str(path.relative_to(REPO_ROOT)))
        return

    improvement_col = col_first(df, ["improvement_over_dummy_pct", "improvement_pct"])
    representation_col = col_first(df, ["representation"])
    target_col = col_first(df, ["target_name", "target"])
    model_col = col_first(df, ["best_model", "model"])

    if improvement_col is None:
        add_check(check_rows, "claims", "master_has_improvement_column", "FAIL", "Master table has no improvement column.", str(path.relative_to(REPO_ROOT)), "improvement column", str(list(df.columns)))
        return

    vals = pd.to_numeric(df[improvement_col], errors="coerce")
    n_rows = int(vals.notna().sum())
    n_positive = int((vals > 0).sum())
    n_non_positive = int((vals <= 0).sum())
    max_val = float(vals.max())
    min_val = float(vals.min())

    claim_rows.append(
        {
            "claim_name": "master_all_best_rows_non_positive",
            "status": "PASS" if n_positive == 0 and n_rows == 63 else "WARN" if n_positive == 0 else "FAIL",
            "source_file": "deap_master_audit_table.csv",
            "expected_value": "63 rows, 0 positive improvements",
            "observed_value": f"rows={n_rows}, positives={n_positive}, non_positive={n_non_positive}, max={max_val}, min={min_val}",
            "details": "Master table supports no positive best-model improvements." if n_positive == 0 else "Positive improvements found in master table.",
        }
    )

    best_idx = vals.idxmax()
    best_row = df.loc[best_idx].to_dict()
    claim_rows.append(
        {
            "claim_name": "master_best_result_value",
            "status": "PASS" if max_val <= 0 else "FAIL",
            "source_file": "deap_master_audit_table.csv",
            "expected_value": "<= 0",
            "observed_value": str(max_val),
            "details": json.dumps({k: str(best_row.get(k, "")) for k in [representation_col, target_col, model_col, improvement_col] if k is not None}),
        }
    )

    add_check(check_rows, "claims", "master_all_best_rows_non_positive", "PASS" if n_positive == 0 else "FAIL", "No master best-model row should be positive.", str(path.relative_to(REPO_ROOT)), "0 positive rows", str(n_positive))


def audit_label_space_claims(claim_rows, check_rows):
    path = TABLE_DIR / "deap_label_space_4d_summary.csv"
    df = read_csv_if_exists(path)

    if df is None:
        add_check(check_rows, "claims", "label_space_4d_summary_exists", "FAIL", "4D summary missing.", str(path.relative_to(REPO_ROOT)))
        return

    improvement_col = col_first(df, ["improvement_over_dummy_pct"])
    target_col = col_first(df, ["target_name"])

    if improvement_col is None:
        add_check(check_rows, "claims", "label_space_4d_has_improvement", "FAIL", "4D summary has no improvement column.", str(path.relative_to(REPO_ROOT)))
        return

    vals = pd.to_numeric(df[improvement_col], errors="coerce")
    n_positive = int((vals > 0).sum())
    max_val = float(vals.max())
    best = df.loc[vals.idxmax()].to_dict()

    claim_rows.append(
        {
            "claim_name": "four_d_no_target_beats_dummy",
            "status": "PASS" if n_positive == 0 else "FAIL",
            "source_file": "deap_label_space_4d_summary.csv",
            "expected_value": "0 positive improvements",
            "observed_value": f"positive={n_positive}, max={max_val}",
            "details": json.dumps({target_col: str(best.get(target_col, "")), improvement_col: str(best.get(improvement_col, ""))}),
        }
    )

    add_check(check_rows, "claims", "four_d_no_target_beats_dummy", "PASS" if n_positive == 0 else "FAIL", "4D label-space audit should have no positive RF improvements.", str(path.relative_to(REPO_ROOT)), "0 positives", str(n_positive))


def audit_loso_claims(claim_rows, check_rows):
    path = TABLE_DIR / "deap_loso_axis_audit_summary.csv"
    df = read_csv_if_exists(path)

    if df is None:
        add_check(check_rows, "claims", "loso_summary_exists", "FAIL", "LOSO summary missing.", str(path.relative_to(REPO_ROOT)))
        return

    improvement_col = col_first(df, ["improvement_over_dummy_pct"])
    target_col = col_first(df, ["target_name"])
    representation_col = col_first(df, ["representation"])

    if improvement_col is None:
        add_check(check_rows, "claims", "loso_has_improvement", "FAIL", "LOSO summary has no improvement column.", str(path.relative_to(REPO_ROOT)))
        return

    vals = pd.to_numeric(df[improvement_col], errors="coerce")
    n_positive = int((vals > 0).sum())
    max_val = float(vals.max())
    best = df.loc[vals.idxmax()].to_dict()

    claim_rows.append(
        {
            "claim_name": "loso_no_target_beats_dummy",
            "status": "PASS" if n_positive == 0 else "FAIL",
            "source_file": "deap_loso_axis_audit_summary.csv",
            "expected_value": "0 positive improvements",
            "observed_value": f"positive={n_positive}, max={max_val}",
            "details": json.dumps({representation_col: str(best.get(representation_col, "")), target_col: str(best.get(target_col, "")), improvement_col: str(best.get(improvement_col, ""))}),
        }
    )

    add_check(check_rows, "claims", "loso_no_target_beats_dummy", "PASS" if n_positive == 0 else "FAIL", "LOSO audit should have no positive RF improvements.", str(path.relative_to(REPO_ROOT)), "0 positives", str(n_positive))


def audit_split_optimism_claims(claim_rows, check_rows):
    path = TABLE_DIR / "deap_split_optimism_summary.csv"
    df = read_csv_if_exists(path)

    if df is None:
        add_check(check_rows, "claims", "split_optimism_summary_exists", "WARN", "Split optimism summary missing.", str(path.relative_to(REPO_ROOT)))
        return

    required = ["representation", "protocol", "target_name", "model", "improvement_over_dummy_pct"]

    if all(c in df.columns for c in required):
        key_cols = ["representation", "target_name", "model"]
        pivot = df.pivot_table(index=key_cols, columns="protocol", values="improvement_over_dummy_pct", aggfunc="mean").reset_index()
        protocol_cols = [c for c in pivot.columns if c not in key_cols]

        random_col = None
        grouped_col = None

        for c in protocol_cols:
            low = str(c).lower()
            if "random" in low:
                random_col = c
            if "group" in low or "subject" in low:
                grouped_col = c

        if random_col is not None and grouped_col is not None:
            random_vals = pd.to_numeric(pivot[random_col], errors="coerce")
            grouped_vals = pd.to_numeric(pivot[grouped_col], errors="coerce")
            mask = np.isfinite(random_vals) & np.isfinite(grouped_vals)
            usable = pivot[mask].copy()
            random_vals = random_vals[mask]
            grouped_vals = grouped_vals[mask]
            has_random_positive_group_negative = bool(((random_vals > 0) & (grouped_vals <= 0)).any())
            mean_gap = float((random_vals - grouped_vals).mean()) if len(random_vals) else np.nan
            max_gap = float((random_vals - grouped_vals).max()) if len(random_vals) else np.nan

            claim_rows.append(
                {
                    "claim_name": "random_split_optimism_present",
                    "status": "PASS" if has_random_positive_group_negative else "WARN",
                    "source_file": "deap_split_optimism_summary.csv",
                    "expected_value": "random-positive grouped-nonpositive rows exist",
                    "observed_value": f"usable_rows={len(usable)}, mean_gap={mean_gap}, max_gap={max_gap}",
                    "details": f"random_col={random_col}, grouped_col={grouped_col}",
                }
            )

            add_check(check_rows, "claims", "random_split_optimism_present", "PASS" if has_random_positive_group_negative else "WARN", "Random split should show optimism relative to grouped validation.", str(path.relative_to(REPO_ROOT)), "True", str(has_random_positive_group_negative))
            return

    cols = list(df.columns)
    add_check(check_rows, "claims", "split_optimism_columns_identified", "WARN", "Could not identify split optimism protocol structure.", str(path.relative_to(REPO_ROOT)), "long-format protocol comparison", str(cols))


def audit_statistical_claims(claim_rows, check_rows):
    path = TABLE_DIR / "deap_statistical_tests_vs_dummy.csv"
    df = read_csv_if_exists(path)

    if df is None:
        add_check(check_rows, "claims", "statistical_tests_exists", "FAIL", "Statistical tests table missing.", str(path.relative_to(REPO_ROOT)))
        return

    improvement_col = col_first(df, ["improvement_over_dummy_pct"])
    model_col = col_first(df, ["model"])

    if improvement_col is None:
        add_check(check_rows, "claims", "statistical_tests_has_improvement", "FAIL", "Statistical table has no improvement column.", str(path.relative_to(REPO_ROOT)))
        return

    vals = pd.to_numeric(df[improvement_col], errors="coerce")
    finite = vals[np.isfinite(vals)]
    positive_count = int((finite > 0).sum())
    max_val = float(finite.max()) if len(finite) else np.nan

    claim_rows.append(
        {
            "claim_name": "statistical_tests_no_positive_mean_improvement",
            "status": "PASS" if positive_count == 0 else "WARN",
            "source_file": "deap_statistical_tests_vs_dummy.csv",
            "expected_value": "0 positive mean improvements",
            "observed_value": f"positive={positive_count}, max={max_val}",
            "details": f"model_col={model_col}",
        }
    )

    add_check(check_rows, "claims", "statistical_tests_no_positive_mean_improvement", "PASS" if positive_count == 0 else "WARN", "Statistical tests should not reveal positive mean model improvements.", str(path.relative_to(REPO_ROOT)), "0 positives", str(positive_count))


def audit_claims(check_rows, claim_rows):
    for fn in tqdm([audit_master_claims, audit_label_space_claims, audit_loso_claims, audit_split_optimism_claims, audit_statistical_claims], desc="Auditing claims"):
        fn(claim_rows, check_rows)


def main():
    ensure_dirs()

    check_rows = []
    feature_rows = []
    leakage_rows = []
    claim_rows = []

    print("Starting audit of audit")

    labels = audit_raw_labels(check_rows)
    audit_feature_files(check_rows, feature_rows, labels)
    audit_manifest_leakage(check_rows, leakage_rows)
    audit_loso_leakage(check_rows, leakage_rows)
    audit_fold_metric_files(check_rows)
    audit_claims(check_rows, claim_rows)

    checks = pd.DataFrame(check_rows)
    features = pd.DataFrame(feature_rows)
    leakage = pd.DataFrame(leakage_rows)
    claims = pd.DataFrame(claim_rows)

    checks.to_csv(OUT_CHECKS, index=False)
    features.to_csv(OUT_FEATURES, index=False)
    leakage.to_csv(OUT_LEAKAGE, index=False)
    claims.to_csv(OUT_CLAIMS, index=False)

    status_counts = checks["status"].value_counts(dropna=False).to_dict() if len(checks) else {}
    claim_status_counts = claims["status"].value_counts(dropna=False).to_dict() if len(claims) else {}
    leakage_status_counts = leakage["status"].value_counts(dropna=False).to_dict() if len(leakage) else {}
    feature_status_counts = features["status"].value_counts(dropna=False).to_dict() if len(features) else {}

    metadata = {
        "script": "18_audit_the_audit.py",
        "n_checks": int(len(checks)),
        "n_feature_rows": int(len(features)),
        "n_leakage_rows": int(len(leakage)),
        "n_claim_rows": int(len(claims)),
        "status_counts": status_counts,
        "claim_status_counts": claim_status_counts,
        "leakage_status_counts": leakage_status_counts,
        "feature_status_counts": feature_status_counts,
        "outputs": {
            "checks": str(OUT_CHECKS.relative_to(REPO_ROOT)),
            "claims": str(OUT_CLAIMS.relative_to(REPO_ROOT)),
            "leakage": str(OUT_LEAKAGE.relative_to(REPO_ROOT)),
            "features": str(OUT_FEATURES.relative_to(REPO_ROOT)),
            "metadata": str(OUT_METADATA.relative_to(REPO_ROOT)),
        },
    }

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved outputs:")
    print(f"  {OUT_CHECKS.relative_to(REPO_ROOT)}")
    print(f"  {OUT_CLAIMS.relative_to(REPO_ROOT)}")
    print(f"  {OUT_LEAKAGE.relative_to(REPO_ROOT)}")
    print(f"  {OUT_FEATURES.relative_to(REPO_ROOT)}")
    print(f"  {OUT_METADATA.relative_to(REPO_ROOT)}")

    print("\nCheck status counts:")
    print(status_counts)

    print("\nClaim status counts:")
    print(claim_status_counts)

    print("\nFeature status counts:")
    print(feature_status_counts)

    print("\nLeakage status counts:")
    print(leakage_status_counts)

    if len(checks):
        problem_checks = checks[checks["status"].isin(["FAIL", "WARN"])]
        print("\nWARN/FAIL checks:")
        if len(problem_checks):
            print(problem_checks[["group", "check_name", "status", "details", "source_file", "expected_value", "observed_value"]].to_string(index=False))
        else:
            print("None")

    if len(claims):
        print("\nClaim verification:")
        print(claims.to_string(index=False))


if __name__ == "__main__":
    main()