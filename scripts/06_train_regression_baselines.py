import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

from src.config import RESULTS_DIR, METRICS_DIR, FIGURES_DIR

TABLES_DIR = RESULTS_DIR / "tables"
FEATURE_TABLE = TABLES_DIR / "deap_bandpower_features.csv"

RANDOM_STATE = 42
N_SPLITS_GROUP = 5

# Based on current loaded dataset. Some files include 0.0 values,
# so we audit actual observed range rather than assuming 1-9.
CLIP_TO_OBSERVED_TARGET_RANGE = True


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str = "pearson") -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    if method == "pearson":
        return float(pearsonr(y_true, y_pred)[0])

    if method == "spearman":
        return float(spearmanr(y_true, y_pred)[0])

    raise ValueError(f"Unknown correlation method: {method}")


def clip_predictions(y_pred: np.ndarray, target_min: np.ndarray, target_max: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, target_min, target_max)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    target_names = ["valence", "arousal"]
    metrics = {}

    for idx, target in enumerate(target_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        mse = mean_squared_error(yt, yp)
        rmse = float(np.sqrt(mse))

        metrics[target] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": rmse,
            "r2": float(r2_score(yt, yp)),
            "pearson": safe_corr(yt, yp, method="pearson"),
            "spearman": safe_corr(yt, yp, method="spearman"),
        }

    euclidean_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

    metrics["affect_space"] = {
        "mean_2d_euclidean_error": float(np.mean(euclidean_errors)),
        "median_2d_euclidean_error": float(np.median(euclidean_errors)),
        "std_2d_euclidean_error": float(np.std(euclidean_errors)),
    }

    metrics["prediction_range"] = {
        "pred_valence_min": float(np.min(y_pred[:, 0])),
        "pred_valence_max": float(np.max(y_pred[:, 0])),
        "pred_arousal_min": float(np.min(y_pred[:, 1])),
        "pred_arousal_max": float(np.max(y_pred[:, 1])),
    }

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
    models: dict,
    target_min: np.ndarray,
    target_max: np.ndarray,
) -> tuple[dict, dict]:
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y,
        groups,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    results = {}
    predictions = {}

    for model_name, model in models.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        y_pred_raw = fitted_model.predict(X_test)
        y_pred_clipped = clip_predictions(y_pred_raw, target_min, target_max)

        results[model_name] = {
            "raw": regression_metrics(y_test, y_pred_raw),
            "clipped": regression_metrics(y_test, y_pred_clipped),
        }

        predictions[model_name] = pd.DataFrame(
            {
                "protocol": "random_split",
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

    return results, predictions


def evaluate_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict,
    target_min: np.ndarray,
    target_max: np.ndarray,
) -> tuple[dict, dict]:
    groupkfold = GroupKFold(n_splits=N_SPLITS_GROUP)

    fold_results = {model_name: [] for model_name in models.keys()}
    prediction_frames = {model_name: [] for model_name in models.keys()}

    for fold_idx, (train_idx, test_idx) in enumerate(groupkfold.split(X, y, groups)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_test = groups[test_idx]

        for model_name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)

            y_pred_raw = fitted_model.predict(X_test)
            y_pred_clipped = clip_predictions(y_pred_raw, target_min, target_max)

            metrics = {
                "fold": int(fold_idx),
                "raw": regression_metrics(y_test, y_pred_raw),
                "clipped": regression_metrics(y_test, y_pred_clipped),
            }

            fold_results[model_name].append(metrics)

            prediction_frames[model_name].append(
                pd.DataFrame(
                    {
                        "protocol": "groupkfold",
                        "fold": fold_idx,
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

    aggregated = {}

    scalar_paths = [
        ("valence", "mae"),
        ("valence", "rmse"),
        ("valence", "r2"),
        ("valence", "pearson"),
        ("valence", "spearman"),
        ("arousal", "mae"),
        ("arousal", "rmse"),
        ("arousal", "r2"),
        ("arousal", "pearson"),
        ("arousal", "spearman"),
        ("affect_space", "mean_2d_euclidean_error"),
        ("affect_space", "median_2d_euclidean_error"),
    ]

    for model_name, metrics_list in fold_results.items():
        aggregated[model_name] = {
            "folds": metrics_list,
            "mean": {"raw": {}, "clipped": {}},
            "std": {"raw": {}, "clipped": {}},
        }

        for version in ["raw", "clipped"]:
            for section, metric_name in scalar_paths:
                values = [
                    fold_metric[version][section][metric_name]
                    for fold_metric in metrics_list
                ]
                key = f"{section}_{metric_name}"
                aggregated[model_name]["mean"][version][key] = float(np.nanmean(values))
                aggregated[model_name]["std"][version][key] = float(np.nanstd(values))

    predictions = {
        model_name: pd.concat(frames, ignore_index=True)
        for model_name, frames in prediction_frames.items()
    }

    return aggregated, predictions


def flatten_summary(random_results: dict, group_results: dict) -> pd.DataFrame:
    rows = []

    for protocol, results in [
        ("random_split", random_results),
        ("groupkfold_mean", group_results),
    ]:
        for model_name, content in results.items():
            for version in ["raw", "clipped"]:
                if protocol == "random_split":
                    metrics = content[version]
                    row = {
                        "protocol": protocol,
                        "prediction_version": version,
                        "model": model_name,
                        "valence_mae": metrics["valence"]["mae"],
                        "valence_rmse": metrics["valence"]["rmse"],
                        "valence_r2": metrics["valence"]["r2"],
                        "valence_pearson": metrics["valence"]["pearson"],
                        "valence_spearman": metrics["valence"]["spearman"],
                        "arousal_mae": metrics["arousal"]["mae"],
                        "arousal_rmse": metrics["arousal"]["rmse"],
                        "arousal_r2": metrics["arousal"]["r2"],
                        "arousal_pearson": metrics["arousal"]["pearson"],
                        "arousal_spearman": metrics["arousal"]["spearman"],
                        "mean_2d_euclidean_error": metrics["affect_space"]["mean_2d_euclidean_error"],
                        "median_2d_euclidean_error": metrics["affect_space"]["median_2d_euclidean_error"],
                        "pred_valence_min": metrics["prediction_range"]["pred_valence_min"],
                        "pred_valence_max": metrics["prediction_range"]["pred_valence_max"],
                        "pred_arousal_min": metrics["prediction_range"]["pred_arousal_min"],
                        "pred_arousal_max": metrics["prediction_range"]["pred_arousal_max"],
                    }
                else:
                    mean = content["mean"][version]
                    std = content["std"][version]
                    row = {
                        "protocol": protocol,
                        "prediction_version": version,
                        "model": model_name,
                        "valence_mae": mean["valence_mae"],
                        "valence_rmse": mean["valence_rmse"],
                        "valence_r2": mean["valence_r2"],
                        "valence_pearson": mean["valence_pearson"],
                        "valence_spearman": mean["valence_spearman"],
                        "arousal_mae": mean["arousal_mae"],
                        "arousal_rmse": mean["arousal_rmse"],
                        "arousal_r2": mean["arousal_r2"],
                        "arousal_pearson": mean["arousal_pearson"],
                        "arousal_spearman": mean["arousal_spearman"],
                        "mean_2d_euclidean_error": mean["affect_space_mean_2d_euclidean_error"],
                        "median_2d_euclidean_error": mean["affect_space_median_2d_euclidean_error"],
                        "mean_2d_euclidean_error_std": std["affect_space_mean_2d_euclidean_error"],
                        "valence_mae_std": std["valence_mae"],
                        "arousal_mae_std": std["arousal_mae"],
                    }

                rows.append(row)

    return pd.DataFrame(rows)


def add_dummy_improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    rows = []

    for (protocol, version), sub in df.groupby(["protocol", "prediction_version"]):
        dummy = sub[sub["model"] == "dummy_mean"]

        if dummy.empty:
            continue

        dummy_error = float(dummy.iloc[0]["mean_2d_euclidean_error"])

        for _, row in sub.iterrows():
            improvement = dummy_error - float(row["mean_2d_euclidean_error"])
            improvement_pct = improvement / dummy_error if dummy_error != 0 else np.nan

            new_row = row.to_dict()
            new_row["dummy_mean_2d_error"] = dummy_error
            new_row["improvement_over_dummy_2d_error"] = improvement
            new_row["improvement_over_dummy_2d_error_pct"] = improvement_pct
            rows.append(new_row)

    return pd.DataFrame(rows)


def plot_best_group_model_predictions(summary_df: pd.DataFrame, prediction_frames: dict, output_path: Path) -> None:
    group_summary = summary_df[
        (summary_df["protocol"] == "groupkfold_mean")
        & (summary_df["prediction_version"] == "clipped")
        & (summary_df["model"] != "dummy_mean")
    ].copy()

    best_row = group_summary.sort_values("mean_2d_euclidean_error", ascending=True).iloc[0]
    best_model = best_row["model"]

    pred_df = prediction_frames[best_model].copy()

    plt.figure(figsize=(8, 7))

    plt.scatter(
        pred_df["true_valence"],
        pred_df["true_arousal"],
        alpha=0.55,
        s=28,
        label="True ratings",
    )

    plt.scatter(
        pred_df["pred_valence_clipped"],
        pred_df["pred_arousal_clipped"],
        alpha=0.55,
        s=28,
        label=f"Predicted ratings ({best_model}, clipped)",
    )

    plt.axvline(5, linestyle="--", linewidth=1)
    plt.axhline(5, linestyle="--", linewidth=1)

    plt.xlabel("Valence rating")
    plt.ylabel("Arousal rating")
    plt.title(f"True vs Predicted Affect Space — Best GroupKFold Model: {best_model}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE}\n"
            "Run scripts/03_extract_bandpower_features.py first."
        )

    df = pd.read_csv(FEATURE_TABLE)

    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df[["valence_rating", "arousal_rating"]].values
    groups = df["subject"].values

    target_min = np.min(y, axis=0)
    target_max = np.max(y, axis=0)

    print("Loaded feature table.")
    print(f"Shape: {df.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print("\nRegression targets:")
    print("- valence_rating")
    print("- arousal_rating")
    print("\nNo thresholded labels are used in this script.")
    print("\nObserved target ranges:")
    print(f"Valence: {target_min[0]:.3f} to {target_max[0]:.3f}")
    print(f"Arousal: {target_min[1]:.3f} to {target_max[1]:.3f}")

    models = get_models()

    random_results, random_predictions = evaluate_random_split(
        X, y, groups, models, target_min, target_max
    )
    group_results, group_predictions = evaluate_groupkfold(
        X, y, groups, models, target_min, target_max
    )

    random_json = METRICS_DIR / "regression_random_split_metrics.json"
    group_json = METRICS_DIR / "regression_groupkfold_metrics.json"
    summary_csv = TABLES_DIR / "regression_model_summary.csv"
    improvement_csv = TABLES_DIR / "regression_model_summary_with_dummy_improvement.csv"

    random_pred_csv = TABLES_DIR / "regression_random_split_predictions.csv"
    group_pred_csv = TABLES_DIR / "regression_groupkfold_predictions.csv"

    with open(random_json, "w", encoding="utf-8") as f:
        json.dump(random_results, f, indent=4)

    with open(group_json, "w", encoding="utf-8") as f:
        json.dump(group_results, f, indent=4)

    summary_df = flatten_summary(random_results, group_results)
    improvement_df = add_dummy_improvement(summary_df)

    summary_df.to_csv(summary_csv, index=False)
    improvement_df.to_csv(improvement_csv, index=False)

    pd.concat(random_predictions.values(), ignore_index=True).to_csv(random_pred_csv, index=False)
    pd.concat(group_predictions.values(), ignore_index=True).to_csv(group_pred_csv, index=False)

    prediction_fig = FIGURES_DIR / "groupkfold_true_vs_predicted_affect_space.png"
    plot_best_group_model_predictions(improvement_df, group_predictions, prediction_fig)

    print("\nSaved:")
    print(f"- {random_json}")
    print(f"- {group_json}")
    print(f"- {summary_csv}")
    print(f"- {improvement_csv}")
    print(f"- {random_pred_csv}")
    print(f"- {group_pred_csv}")
    print(f"- {prediction_fig}")

    print("\nRegression model summary with dummy improvement:")
    print(improvement_df.to_string(index=False))


if __name__ == "__main__":
    main()