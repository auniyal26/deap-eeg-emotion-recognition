import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, METRICS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

FEATURE_TABLE = TABLES_DIR / "bandpower_features_absolute.csv"

RANDOM_STATE = 42
N_SPLITS_GROUP = 5

TARGET_COLS = ["valence", "arousal"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    if method == "pearson":
        return float(pearsonr(y_true, y_pred)[0])

    if method == "spearman":
        return float(spearmanr(y_true, y_pred)[0])

    raise ValueError(f"Unknown correlation method: {method}")


def clip_predictions(y_pred: np.ndarray, y_min: np.ndarray, y_max: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, y_min, y_max)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}

    for idx, target in enumerate(TARGET_COLS):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        metrics[f"{target}_mae"] = float(mean_absolute_error(yt, yp))
        metrics[f"{target}_rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
        metrics[f"{target}_r2"] = float(r2_score(yt, yp))
        metrics[f"{target}_pearson"] = safe_corr(yt, yp, method="pearson")
        metrics[f"{target}_spearman"] = safe_corr(yt, yp, method="spearman")

    errors_2d = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

    metrics["mean_2d_euclidean_error"] = float(np.mean(errors_2d))
    metrics["median_2d_euclidean_error"] = float(np.median(errors_2d))
    metrics["std_2d_euclidean_error"] = float(np.std(errors_2d))

    return metrics


def get_models() -> dict:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "svr_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.1))),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_random_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y,
        groups,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    rows = []
    pred_frames = []

    models = get_models()

    for model_name, model in models.items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)

        y_pred_raw = fitted.predict(X_test)
        y_pred_clipped = clip_predictions(y_pred_raw, y_min, y_max)

        for version, y_pred in [
            ("raw", y_pred_raw),
            ("clipped", y_pred_clipped),
        ]:
            row = {
                "protocol": "random_split",
                "fold": -1,
                "model": model_name,
                "prediction_version": version,
            }
            row.update(regression_metrics(y_test, y_pred))
            rows.append(row)

        pred_frames.append(
            pd.DataFrame(
                {
                    "protocol": "random_split",
                    "fold": -1,
                    "model": model_name,
                    "subject": groups_test,
                    "true_valence": y_test[:, 0],
                    "true_arousal": y_test[:, 1],
                    "pred_valence_raw": y_pred_raw[:, 0],
                    "pred_arousal_raw": y_pred_raw[:, 1],
                    "pred_valence_clipped": y_pred_clipped[:, 0],
                    "pred_arousal_clipped": y_pred_clipped[:, 1],
                }
            )
        )

    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True)


def evaluate_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv = GroupKFold(n_splits=N_SPLITS_GROUP)

    rows = []
    pred_frames = []
    fold_manifest_rows = []

    models = get_models()

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        train_subjects = sorted(pd.Series(groups[train_idx]).astype(str).unique().tolist())
        test_subjects = sorted(pd.Series(groups[test_idx]).astype(str).unique().tolist())

        fold_manifest_rows.append(
            {
                "protocol": "groupkfold",
                "fold": fold_idx,
                "train_subjects": ",".join(train_subjects),
                "test_subjects": ",".join(test_subjects),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_valence_mean": float(np.mean(y_train[:, 0])),
                "test_valence_mean": float(np.mean(y_test[:, 0])),
                "train_arousal_mean": float(np.mean(y_train[:, 1])),
                "test_arousal_mean": float(np.mean(y_test[:, 1])),
            }
        )

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)

            y_pred_raw = fitted.predict(X_test)
            y_pred_clipped = clip_predictions(y_pred_raw, y_min, y_max)

            for version, y_pred in [
                ("raw", y_pred_raw),
                ("clipped", y_pred_clipped),
            ]:
                row = {
                    "protocol": "groupkfold",
                    "fold": fold_idx,
                    "model": model_name,
                    "prediction_version": version,
                }
                row.update(regression_metrics(y_test, y_pred))
                rows.append(row)

            pred_frames.append(
                pd.DataFrame(
                    {
                        "protocol": "groupkfold",
                        "fold": fold_idx,
                        "model": model_name,
                        "subject": groups[test_idx],
                        "true_valence": y_test[:, 0],
                        "true_arousal": y_test[:, 1],
                        "pred_valence_raw": y_pred_raw[:, 0],
                        "pred_arousal_raw": y_pred_raw[:, 1],
                        "pred_valence_clipped": y_pred_clipped[:, 0],
                        "pred_arousal_clipped": y_pred_clipped[:, 1],
                    }
                )
            )

    return (
        pd.DataFrame(rows),
        pd.concat(pred_frames, ignore_index=True),
        pd.DataFrame(fold_manifest_rows),
    )


def aggregate_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "valence_mae",
        "valence_rmse",
        "valence_r2",
        "valence_pearson",
        "valence_spearman",
        "arousal_mae",
        "arousal_rmse",
        "arousal_r2",
        "arousal_pearson",
        "arousal_spearman",
        "mean_2d_euclidean_error",
        "median_2d_euclidean_error",
        "std_2d_euclidean_error",
    ]

    random_rows = fold_df[fold_df["protocol"] == "random_split"].copy()
    group_rows = fold_df[fold_df["protocol"] == "groupkfold"].copy()

    random_summary = random_rows.copy()
    random_summary["aggregation"] = "single_split"

    group_summary = (
        group_rows
        .groupby(["protocol", "model", "prediction_version"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )

    group_summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in group_summary.columns
    ]

    group_summary = group_summary.reset_index()
    group_summary["aggregation"] = "mean_across_folds"

    # Normalize column names so both random and grouped summaries can be stacked.
    random_out_rows = []

    for _, row in random_summary.iterrows():
        out = {
            "protocol": row["protocol"],
            "model": row["model"],
            "prediction_version": row["prediction_version"],
            "aggregation": "single_split",
        }

        for col in metric_cols:
            out[f"{col}_mean"] = row[col]
            out[f"{col}_std"] = np.nan

        random_out_rows.append(out)

    random_out = pd.DataFrame(random_out_rows)

    return pd.concat([random_out, group_summary], ignore_index=True, sort=False)


def add_dummy_improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (protocol, version), sub in summary_df.groupby(["protocol", "prediction_version"]):
        dummy = sub[sub["model"] == "dummy_mean"]

        if dummy.empty:
            continue

        dummy_error = float(dummy.iloc[0]["mean_2d_euclidean_error_mean"])

        for _, row in sub.iterrows():
            model_error = float(row["mean_2d_euclidean_error_mean"])
            improvement = dummy_error - model_error
            improvement_pct = improvement / dummy_error if dummy_error != 0 else np.nan

            out = row.to_dict()
            out["dummy_mean_2d_error"] = dummy_error
            out["improvement_over_dummy_2d_error"] = improvement
            out["improvement_over_dummy_2d_error_pct"] = improvement_pct
            rows.append(out)

    return pd.DataFrame(rows)


def best_model_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    clipped = summary_df[summary_df["prediction_version"] == "clipped"].copy()
    non_dummy = clipped[clipped["model"] != "dummy_mean"].copy()

    rows = []

    for protocol, sub in non_dummy.groupby("protocol"):
        best = sub.sort_values("mean_2d_euclidean_error_mean", ascending=True).iloc[0]

        rows.append(
            {
                "protocol": protocol,
                "best_model": best["model"],
                "best_model_2d_error": best["mean_2d_euclidean_error_mean"],
                "best_model_2d_error_std": best.get("mean_2d_euclidean_error_std", np.nan),
                "dummy_mean_2d_error": best["dummy_mean_2d_error"],
                "improvement_over_dummy_2d_error": best["improvement_over_dummy_2d_error"],
                "improvement_over_dummy_2d_error_pct": best["improvement_over_dummy_2d_error_pct"],
                "valence_mae": best["valence_mae_mean"],
                "arousal_mae": best["arousal_mae_mean"],
                "valence_pearson": best["valence_pearson_mean"],
                "arousal_pearson": best["arousal_pearson_mean"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(f"Missing feature table: {FEATURE_TABLE}")

    df = pd.read_csv(FEATURE_TABLE)

    feature_cols = get_feature_columns(df)

    if len(feature_cols) != 160:
        raise RuntimeError(f"Expected 160 feature columns, found {len(feature_cols)}")

    X = df[feature_cols].values
    y = df[TARGET_COLS].values
    groups = df["subject"].values

    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)

    print("Running random split baseline...")
    random_folds, random_preds = evaluate_random_split(X, y, groups, y_min, y_max)

    print("Running GroupKFold baseline...")
    group_folds, group_preds, fold_manifest = evaluate_groupkfold(X, y, groups, y_min, y_max)

    fold_df = pd.concat([random_folds, group_folds], ignore_index=True)
    preds_df = pd.concat([random_preds, group_preds], ignore_index=True)

    summary = aggregate_metrics(fold_df)
    summary_with_dummy = add_dummy_improvement(summary)
    best_summary = best_model_summary(summary_with_dummy)

    fold_path = TABLES_DIR / "regression_baseline_fold_metrics_absolute.csv"
    summary_path = TABLES_DIR / "regression_baseline_summary_absolute.csv"
    best_path = TABLES_DIR / "regression_baseline_best_models_absolute.csv"
    pred_path = TABLES_DIR / "regression_baseline_predictions_absolute.csv"
    manifest_path = TABLES_DIR / "regression_baseline_groupkfold_manifest.csv"
    metrics_path = METRICS_DIR / "regression_baseline_absolute_summary.json"

    fold_df.to_csv(fold_path, index=False)
    summary_with_dummy.to_csv(summary_path, index=False)
    best_summary.to_csv(best_path, index=False)
    preds_df.to_csv(pred_path, index=False)
    fold_manifest.to_csv(manifest_path, index=False)

    metrics = {
        "feature_table": str(FEATURE_TABLE),
        "target_cols": TARGET_COLS,
        "n_trials": int(df.shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "n_features": int(len(feature_cols)),
        "models": list(get_models().keys()),
        "protocols": ["random_split", "groupkfold"],
        "best_models": best_summary.to_dict(orient="records"),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved:")
    print(f"- {fold_path}")
    print(f"- {summary_path}")
    print(f"- {best_path}")
    print(f"- {pred_path}")
    print(f"- {manifest_path}")
    print(f"- {metrics_path}")

    print("\nBest model summary:")
    print(best_summary.to_string(index=False))

    print("\nGroupKFold manifest:")
    print(fold_manifest.to_string(index=False))


if __name__ == "__main__":
    main()