import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None):
        if iterable is None:
            return range(total or 0)
        print(desc or "Progress")
        return iterable


RANDOM_STATE = 42
N_BOOTSTRAP = 10000

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "results" / "tables"
METRIC_DIR = REPO_ROOT / "results" / "metrics"

INPUT_FILES = {
    "master_groupkfold": TABLE_DIR / "master_axis_audit_fold_metrics.csv",
    "asymmetry_groupkfold": TABLE_DIR / "asymmetry_axis_audit_fold_metrics.csv",
    "label_space_4d_groupkfold": TABLE_DIR / "deap_label_space_4d_fold_metrics.csv",
    "loso": TABLE_DIR / "deap_loso_axis_audit_fold_metrics.csv",
}

OUT_TABLE = TABLE_DIR / "deap_statistical_tests_vs_dummy.csv"
OUT_METADATA = METRIC_DIR / "deap_statistical_tests_metadata.json"


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def first_existing(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def normalize_base_columns(df, source_name):
    df = df.copy()

    rename_map = {}

    candidates = {
        "representation": ["representation", "feature_representation", "feature_set"],
        "target_name": ["target_name", "target", "axis_name"],
        "model": ["model", "model_name", "best_model"],
        "fold": ["fold", "fold_id"],
        "score": ["score", "rmse_or_error", "primary_score", "test_score", "metric_value", "rmse"],
        "primary_metric": ["primary_metric", "metric"],
    }

    for canonical, options in candidates.items():
        if canonical not in df.columns:
            found = first_existing(df.columns, options)
            if found is not None:
                rename_map[found] = canonical

    df = df.rename(columns=rename_map)

    if "representation" not in df.columns:
        df["representation"] = "unknown"

    if "target_name" not in df.columns:
        raise ValueError(f"{source_name} has no target_name-compatible column.")

    if "fold" not in df.columns:
        raise ValueError(f"{source_name} has no fold-compatible column.")

    if "primary_metric" not in df.columns:
        df["primary_metric"] = np.nan

    df["primary_metric"] = df["primary_metric"].replace("unknown", np.nan)

    if df["primary_metric"].isna().all():
        if "score" in df.columns:
            df["primary_metric"] = "rmse_or_error"
        elif "rmse" in df.columns:
            df["primary_metric"] = "rmse"
        else:
            df["primary_metric"] = "unknown"

    df["primary_metric"] = df["primary_metric"].fillna("unknown")

    if "target_type" not in df.columns:
        df["target_type"] = "unknown"

    if "heldout_subject" not in df.columns:
        df["heldout_subject"] = ""

    return df


def to_long_if_wide(df, source_name):
    df = normalize_base_columns(df, source_name)

    if "score" in df.columns and "model" in df.columns:
        out = df.copy()
        out["score"] = out["score"].astype(float)
        return out

    dummy_cols = [
        "dummy_score",
        "dummy_mean_score",
        "dummy_primary_score",
        "dummy_metric",
        "dummy_rmse",
        "dummy_error",
    ]

    model_cols = [
        "model_score",
        "best_model_score",
        "random_forest_score",
        "rf_score",
        "ridge_score",
        "svr_rbf_score",
        "svr_score",
        "forest_score",
    ]

    dummy_col = first_existing(df.columns, dummy_cols)

    if dummy_col is None:
        possible = [c for c in df.columns if "dummy" in c.lower() and any(x in c.lower() for x in ["score", "rmse", "error", "metric"])]
        dummy_col = possible[0] if possible else None

    if dummy_col is None:
        raise ValueError(f"{source_name} has no dummy score column. Columns: {list(df.columns)}")

    found_model_cols = [c for c in model_cols if c in df.columns]

    extra_score_cols = [
        c for c in df.columns
        if c.endswith("_score")
        and c != dummy_col
        and c not in ["score"]
        and not c.lower().startswith("dummy")
    ]

    for col in extra_score_cols:
        if col not in found_model_cols:
            found_model_cols.append(col)

    if not found_model_cols:
        raise ValueError(f"{source_name} has no model score column. Columns: {list(df.columns)}")

    id_cols = [
        c for c in [
            "source",
            "representation",
            "target_name",
            "target_type",
            "primary_metric",
            "fold",
            "heldout_subject",
            "model",
        ]
        if c in df.columns
    ]

    rows = []

    for _, row in df.iterrows():
        base = {c: row[c] for c in id_cols if c in row.index}

        dummy_row = base.copy()
        dummy_row["model"] = "dummy_mean"
        dummy_row["score"] = float(row[dummy_col])
        rows.append(dummy_row)

        for model_col in found_model_cols:
            model_row = base.copy()

            if "model" in df.columns and pd.notna(row.get("model", np.nan)) and model_col in ["model_score", "best_model_score"]:
                model_name = str(row["model"])
            else:
                model_name = model_col.replace("_score", "")

            if model_name == "rf":
                model_name = "random_forest"

            model_row["model"] = model_name
            model_row["score"] = float(row[model_col])
            rows.append(model_row)

    return pd.DataFrame(rows)


def normalize_columns(df, source_name):
    df = to_long_if_wide(df, source_name)

    df["source"] = source_name

    if "score" in df.columns:
        for backup_col in ["rmse", "rmse_or_error", "model_score", "best_model_score"]:
            if backup_col in df.columns:
                df["score"] = df["score"].fillna(df[backup_col])

    required = ["source", "representation", "target_name", "model", "fold", "score", "primary_metric", "target_type", "heldout_subject"]

    for col in required:
        if col not in df.columns:
            if col == "representation":
                df[col] = "unknown"
            elif col == "primary_metric":
                df[col] = "unknown"
            elif col == "target_type":
                df[col] = "unknown"
            elif col == "heldout_subject":
                df[col] = ""
            else:
                raise ValueError(f"{source_name} missing required normalized column: {col}")

    df["source"] = df["source"].astype(str)
    df["representation"] = df["representation"].astype(str)
    df["target_name"] = df["target_name"].astype(str)
    df["model"] = df["model"].astype(str)
    df["fold"] = df["fold"].astype(str)
    df["score"] = df["score"].astype(float)
    df["primary_metric"] = df["primary_metric"].astype(str)
    df["target_type"] = df["target_type"].astype(str)
    df["heldout_subject"] = df["heldout_subject"].astype(str)

    return df[required].copy()


def load_inputs():
    frames = []
    loaded = {}
    missing = {}

    for source_name, path in INPUT_FILES.items():
        if not path.exists():
            missing[source_name] = str(path)
            continue

        df = pd.read_csv(path)
        normalized = normalize_columns(df, source_name)
        frames.append(normalized)

        loaded[source_name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "rows_raw": int(len(df)),
            "rows_normalized": int(len(normalized)),
            "columns_raw": list(df.columns),
            "columns_normalized": list(normalized.columns),
        }

    if not frames:
        raise FileNotFoundError("No statistical-test input files were found.")

    combined = pd.concat(frames, axis=0, ignore_index=True)

    return combined, loaded, missing


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, rng=None):
    values = np.asarray(values, dtype=float)

    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    if len(values) == 0:
        return np.nan, np.nan

    boot = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = np.mean(sample)

    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_test_p_value(values):
    values = np.asarray(values, dtype=float)
    values = values[values != 0]

    n = len(values)
    if n == 0:
        return np.nan

    positives = int(np.sum(values > 0))
    k = min(positives, n - positives)

    prob = 0.0

    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)

    return float(min(1.0, 2.0 * prob))


def safe_wilcoxon(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan

    if np.allclose(values, 0):
        return np.nan

    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def cohens_d_paired(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    sd = np.std(values, ddof=1)

    if sd == 0:
        return np.nan

    return float(np.mean(values) / sd)


def rank_biserial_from_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    values = values[values != 0]

    if len(values) == 0:
        return np.nan

    abs_values = np.abs(values)
    order = np.argsort(abs_values)
    ranks = np.empty(len(values), dtype=float)

    sorted_abs = abs_values[order]
    i = 0

    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    positive_rank_sum = float(np.sum(ranks[values > 0]))
    negative_rank_sum = float(np.sum(ranks[values < 0]))
    total_rank_sum = positive_rank_sum + negative_rank_sum

    if total_rank_sum == 0:
        return np.nan

    return float((positive_rank_sum - negative_rank_sum) / total_rank_sum)


def summarize_pair(group, keys, rng):
    dummy = group[group["model"].str.lower().isin(["dummy_mean", "dummy"])]
    models = group[~group["model"].str.lower().isin(["dummy_mean", "dummy"])]

    rows = []

    if dummy.empty or models.empty:
        return rows

    for model_name, model_df in models.groupby("model"):
        key_cols = ["fold"]

        if group["heldout_subject"].astype(str).str.len().sum() > 0:
            key_cols = ["fold", "heldout_subject"]

        dummy_small = dummy[key_cols + ["score"]].rename(columns={"score": "dummy_score"})
        model_small = model_df[key_cols + ["score"]].rename(columns={"score": "model_score"})

        paired = dummy_small.merge(model_small, on=key_cols, how="inner")

        if paired.empty:
            continue

        dummy_scores = paired["dummy_score"].to_numpy(dtype=float)
        model_scores = paired["model_score"].to_numpy(dtype=float)
        deltas = dummy_scores - model_scores

        ci_low, ci_high = bootstrap_ci(deltas, rng=rng)

        row = {
            **keys,
            "model": model_name,
            "n_pairs": int(len(deltas)),
            "dummy_score_mean": float(np.mean(dummy_scores)),
            "model_score_mean": float(np.mean(model_scores)),
            "dummy_minus_model_mean": float(np.mean(deltas)),
            "dummy_minus_model_median": float(np.median(deltas)),
            "dummy_minus_model_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else np.nan,
            "model_minus_dummy_mean": float(np.mean(model_scores - dummy_scores)),
            "improvement_over_dummy_pct": float(np.mean(deltas) / np.mean(dummy_scores)) if np.mean(dummy_scores) != 0 else np.nan,
            "ci95_low_dummy_minus_model": ci_low,
            "ci95_high_dummy_minus_model": ci_high,
            "model_beats_dummy_n": int(np.sum(model_scores < dummy_scores)),
            "dummy_beats_model_n": int(np.sum(dummy_scores < model_scores)),
            "ties_n": int(np.sum(model_scores == dummy_scores)),
            "model_beats_dummy_fraction": float(np.mean(model_scores < dummy_scores)),
            "wilcoxon_p": safe_wilcoxon(deltas),
            "sign_test_p": sign_test_p_value(deltas),
            "cohens_d_paired": cohens_d_paired(deltas),
            "rank_biserial": rank_biserial_from_wilcoxon(deltas),
            "lower_is_better": True,
            "positive_delta_means_model_better": True,
        }

        rows.append(row)

    return rows


def run_tests(df):
    rng = np.random.default_rng(RANDOM_STATE)
    df = df[np.isfinite(df["score"].to_numpy(dtype=float))].copy()
    group_cols = [
        "source",
        "representation",
        "target_name",
        "target_type",
        "primary_metric",
    ]

    rows = []
    grouped = list(df.groupby(group_cols, dropna=False))

    for group_values, group in tqdm(grouped, desc="Running statistical tests"):
        keys = dict(zip(group_cols, group_values))
        rows.extend(summarize_pair(group, keys, rng))

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No paired model-vs-dummy statistical tests could be computed.")

    result = result.sort_values(
        by=["source", "improvement_over_dummy_pct"],
        ascending=[True, False],
    )

    return result


def main():
    ensure_dirs()

    print("Loading fold metrics")
    df, loaded, missing = load_inputs()

    print(f"Loaded rows: {len(df)}")
    print(f"Sources loaded: {list(loaded.keys())}")

    if missing:
        print(f"Missing sources: {list(missing.keys())}")

    results = run_tests(df)
    results.to_csv(OUT_TABLE, index=False)

    metadata = {
        "script": "17_statistical_tests_vs_dummy.py",
        "random_state": RANDOM_STATE,
        "n_bootstrap": N_BOOTSTRAP,
        "loaded_sources": loaded,
        "missing_sources": missing,
        "n_input_rows": int(len(df)),
        "n_output_rows": int(len(results)),
        "delta_definition": "dummy_score_minus_model_score",
        "positive_delta_meaning": "model_better_than_dummy",
        "lower_metric_is_better": True,
        "outputs": {
            "statistical_tests": str(OUT_TABLE.relative_to(REPO_ROOT)),
            "metadata": str(OUT_METADATA.relative_to(REPO_ROOT)),
        },
    }

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved outputs:")
    print(f"  {OUT_TABLE.relative_to(REPO_ROOT)}")
    print(f"  {OUT_METADATA.relative_to(REPO_ROOT)}")

    print("\nTop results:")
    display_cols = [
        "source",
        "representation",
        "target_name",
        "model",
        "n_pairs",
        "dummy_score_mean",
        "model_score_mean",
        "improvement_over_dummy_pct",
        "model_beats_dummy_n",
        "wilcoxon_p",
        "sign_test_p",
        "ci95_low_dummy_minus_model",
        "ci95_high_dummy_minus_model",
    ]

    print(results[display_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()