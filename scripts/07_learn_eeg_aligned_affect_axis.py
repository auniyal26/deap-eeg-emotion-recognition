import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, FIGURES_DIR, METRICS_DIR

TABLES_DIR = RESULTS_DIR / "tables"
FEATURE_TABLE = TABLES_DIR / "deap_bandpower_features.csv"

RANDOM_STATE = 42
N_SPLITS_GROUP = 5


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return axis
    return axis / norm


def axis_angle_degrees(axis: np.ndarray) -> float:
    axis = normalize_axis(axis)
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str = "pearson") -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    if method == "pearson":
        return float(pearsonr(y_true, y_pred)[0])

    if method == "spearman":
        return float(spearmanr(y_true, y_pred)[0])

    raise ValueError(f"Unknown method: {method}")


def scalar_metrics(z_true: np.ndarray, z_pred: np.ndarray, z_dummy: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(z_true, z_pred)))
    dummy_rmse = float(np.sqrt(mean_squared_error(z_true, z_dummy)))

    mae = float(mean_absolute_error(z_true, z_pred))
    dummy_mae = float(mean_absolute_error(z_true, z_dummy))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(z_true, z_pred)),
        "pearson": safe_corr(z_true, z_pred, method="pearson"),
        "spearman": safe_corr(z_true, z_pred, method="spearman"),
        "dummy_mae": dummy_mae,
        "dummy_rmse": dummy_rmse,
        "rmse_improvement_over_dummy": float(dummy_rmse - rmse),
        "rmse_improvement_over_dummy_pct": float((dummy_rmse - rmse) / dummy_rmse) if dummy_rmse != 0 else np.nan,
        "mae_improvement_over_dummy": float(dummy_mae - mae),
        "mae_improvement_over_dummy_pct": float((dummy_mae - mae) / dummy_mae) if dummy_mae != 0 else np.nan,
    }


def get_fixed_axes() -> dict:
    return {
        "valence_axis": normalize_axis(np.array([1.0, 0.0])),
        "arousal_axis": normalize_axis(np.array([0.0, 1.0])),
        "positive_affect_axis": normalize_axis(np.array([1.0, 1.0])),
        "valence_minus_arousal_axis": normalize_axis(np.array([1.0, -1.0])),
    }


def evaluate_fixed_axis(
    X_train,
    X_test,
    Y_train_scaled,
    Y_test_scaled,
    axis_name: str,
    axis: np.ndarray,
) -> dict:
    axis = normalize_axis(axis)

    z_train = Y_train_scaled @ axis
    z_test = Y_test_scaled @ axis

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    dummy = DummyRegressor(strategy="mean")

    model.fit(X_train, z_train)
    dummy.fit(X_train, z_train)

    z_pred = model.predict(X_test)
    z_dummy = dummy.predict(X_test)

    metrics = scalar_metrics(z_test, z_pred, z_dummy)

    return {
        "axis_name": axis_name,
        "axis_v_weight": float(axis[0]),
        "axis_a_weight": float(axis[1]),
        "axis_angle_deg": axis_angle_degrees(axis),
        **metrics,
    }


def evaluate_label_pca_axis(X_train, X_test, Y_train_scaled, Y_test_scaled) -> dict:
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    pca.fit(Y_train_scaled)

    axis = normalize_axis(pca.components_[0])

    # Sign alignment for readability: keep valence weight positive.
    if axis[0] < 0:
        axis = -axis

    return evaluate_fixed_axis(
        X_train=X_train,
        X_test=X_test,
        Y_train_scaled=Y_train_scaled,
        Y_test_scaled=Y_test_scaled,
        axis_name="label_pca_axis",
        axis=axis,
    )


def evaluate_pls_eeg_axis(X_train, X_test, Y_train_scaled, Y_test_scaled) -> dict:
    """
    Proposed fold-wise EEG-aligned axis.

    PLS is fit only on training data.
    The label-side first component gives a direction in valence-arousal space.
    """
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    pls = PLSRegression(n_components=1)
    pls.fit(X_train_scaled, Y_train_scaled)

    axis = normalize_axis(pls.y_weights_[:, 0])

    # Sign alignment for readability: keep valence weight positive.
    # Sign does not change predictive content, only plotting/interpretation.
    if axis[0] < 0:
        axis = -axis

    z_train = Y_train_scaled @ axis
    z_test = Y_test_scaled @ axis

    Y_pred_scaled = pls.predict(X_test_scaled)
    z_pred = Y_pred_scaled @ axis

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_scaled, z_train)
    z_dummy = dummy.predict(X_test_scaled)

    metrics = scalar_metrics(z_test, z_pred, z_dummy)

    return {
        "axis_name": "pls_eeg_aligned_axis",
        "axis_v_weight": float(axis[0]),
        "axis_a_weight": float(axis[1]),
        "axis_angle_deg": axis_angle_degrees(axis),
        **metrics,
    }


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE}\n"
            "Run scripts/03_extract_bandpower_features.py first."
        )

    df = pd.read_csv(FEATURE_TABLE)

    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    Y = df[["valence_rating", "arousal_rating"]].values
    groups = df["subject"].values

    print("Loaded feature table.")
    print(f"Shape: {df.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print("Targets: continuous valence_rating and arousal_rating")
    print("No thresholded labels used.")

    groupkfold = GroupKFold(n_splits=N_SPLITS_GROUP)

    fold_rows = []

    fixed_axes = get_fixed_axes()

    for fold_idx, (train_idx, test_idx) in enumerate(groupkfold.split(X, Y, groups)):
        print(f"\nRunning fold {fold_idx + 1}/{N_SPLITS_GROUP}")

        X_train = X[train_idx]
        X_test = X[test_idx]

        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        y_scaler = StandardScaler()
        Y_train_scaled = y_scaler.fit_transform(Y_train)
        Y_test_scaled = y_scaler.transform(Y_test)

        # Fixed axes
        for axis_name, axis in fixed_axes.items():
            result = evaluate_fixed_axis(
                X_train=X_train,
                X_test=X_test,
                Y_train_scaled=Y_train_scaled,
                Y_test_scaled=Y_test_scaled,
                axis_name=axis_name,
                axis=axis,
            )
            result["fold"] = fold_idx
            fold_rows.append(result)

        # Label PCA axis, learned from training labels only
        result = evaluate_label_pca_axis(
            X_train=X_train,
            X_test=X_test,
            Y_train_scaled=Y_train_scaled,
            Y_test_scaled=Y_test_scaled,
        )
        result["fold"] = fold_idx
        fold_rows.append(result)

        # PLS EEG-aligned axis, learned from training X/Y only
        result = evaluate_pls_eeg_axis(
            X_train=X_train,
            X_test=X_test,
            Y_train_scaled=Y_train_scaled,
            Y_test_scaled=Y_test_scaled,
        )
        result["fold"] = fold_idx
        fold_rows.append(result)

    fold_df = pd.DataFrame(fold_rows)

    metric_cols = [
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
        "dummy_mae",
        "dummy_rmse",
        "rmse_improvement_over_dummy",
        "rmse_improvement_over_dummy_pct",
        "mae_improvement_over_dummy",
        "mae_improvement_over_dummy_pct",
        "axis_v_weight",
        "axis_a_weight",
        "axis_angle_deg",
    ]

    summary_df = (
        fold_df
        .groupby("axis_name")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns
    ]

    fold_csv = TABLES_DIR / "eeg_aligned_axis_fold_details.csv"
    summary_csv = TABLES_DIR / "eeg_aligned_axis_summary.csv"
    metrics_json = METRICS_DIR / "eeg_aligned_axis_summary.json"

    fold_df.to_csv(fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_folds": N_SPLITS_GROUP,
                "feature_columns": len(feature_cols),
                "target_space": "standardized continuous valence-arousal",
                "axis_methods": sorted(fold_df["axis_name"].unique().tolist()),
                "summary_csv": str(summary_csv),
                "fold_details_csv": str(fold_csv),
            },
            f,
            indent=4,
        )

    # Plot axis angles
    plt.figure(figsize=(10, 5))

    for axis_name in sorted(fold_df["axis_name"].unique()):
        sub = fold_df[fold_df["axis_name"] == axis_name]
        plt.scatter(
            [axis_name] * len(sub),
            sub["axis_angle_deg"],
            alpha=0.75,
            s=55,
            label=axis_name,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axhline(45, linestyle="--", linewidth=1)
    plt.axhline(90, linestyle="--", linewidth=1)

    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Axis angle in valence-arousal plane (degrees)")
    plt.title("Fold-wise Affect Axis Angles")
    plt.tight_layout()

    angle_fig = FIGURES_DIR / "eeg_aligned_axis_angles.png"
    plt.savefig(angle_fig, dpi=300)
    plt.close()

    print("\nSaved:")
    print(f"- {fold_csv}")
    print(f"- {summary_csv}")
    print(f"- {metrics_json}")
    print(f"- {angle_fig}")

    print("\nAxis summary:")
    display_cols = [
        "axis_name",
        "rmse_mean",
        "rmse_std",
        "dummy_rmse_mean",
        "rmse_improvement_over_dummy_mean",
        "rmse_improvement_over_dummy_pct_mean",
        "pearson_mean",
        "spearman_mean",
        "axis_v_weight_mean",
        "axis_a_weight_mean",
        "axis_angle_deg_mean",
        "axis_angle_deg_std",
    ]

    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()