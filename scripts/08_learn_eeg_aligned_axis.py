import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    if method == "pearson":
        return float(pearsonr(y_true, y_pred)[0])

    if method == "spearman":
        return float(spearmanr(y_true, y_pred)[0])

    raise ValueError(f"Unknown method: {method}")


def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": safe_corr(y_true, y_pred, method="pearson"),
        "spearman": safe_corr(y_true, y_pred, method="spearman"),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "true_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
    }


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
                ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def learn_pls_axis(
    X_train: np.ndarray,
    y_train_va: np.ndarray,
    X_test: np.ndarray,
    y_test_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Learns a supervised EEG-aligned 1D affect axis.

    Important:
    - Feature scaler fit on X_train only.
    - Label scaler fit on y_train only.
    - PLS fit on training fold only.
    - Test labels are projected using train-fold label scaler and train-fold axis.
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_z = x_scaler.fit_transform(X_train)
    X_test_z = x_scaler.transform(X_test)

    y_train_z2 = y_scaler.fit_transform(y_train_va)
    y_test_z2 = y_scaler.transform(y_test_va)

    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X_train_z, y_train_z2)

    # PLS y_weights_ gives the label-space direction associated with the latent component.
    axis = pls.y_weights_[:, 0].astype(float)

    if np.linalg.norm(axis) == 0:
        raise RuntimeError("PLS returned zero label axis.")

    axis = axis / np.linalg.norm(axis)

    # Sign convention: positive valence direction.
    if axis[0] < 0:
        axis = -axis

    z_train = y_train_z2 @ axis
    z_test = y_test_z2 @ axis

    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))

    meta = {
        "axis_type": "supervised_pls_rank1",
        "axis_valence_weight": float(axis[0]),
        "axis_arousal_weight": float(axis[1]),
        "axis_angle_deg": angle,
        "pls_x_score_train_std": float(np.std(pls.x_scores_[:, 0])),
        "label_scaler_mean_valence": float(y_scaler.mean_[0]),
        "label_scaler_mean_arousal": float(y_scaler.mean_[1]),
        "label_scaler_std_valence": float(np.sqrt(y_scaler.var_[0])),
        "label_scaler_std_arousal": float(np.sqrt(y_scaler.var_[1])),
    }

    return z_train, z_test, meta


def evaluate_one_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_va: np.ndarray,
    y_test_va: np.ndarray,
    groups_test: np.ndarray,
    protocol: str,
    fold: int,
) -> tuple[list[dict], list[pd.DataFrame], dict]:
    z_train, z_test, axis_meta = learn_pls_axis(X_train, y_train_va, X_test, y_test_va)

    rows = []
    pred_frames = []

    for model_name, model in get_models().items():
        fitted = clone(model)
        fitted.fit(X_train, z_train)

        z_pred = fitted.predict(X_test)

        row = {
            "protocol": protocol,
            "fold": fold,
            "axis_name": "pls_eeg_aligned_axis",
            "model": model_name,
            **axis_meta,
        }
        row.update(scalar_metrics(z_test, z_pred))
        rows.append(row)

        pred_frames.append(
            pd.DataFrame(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "axis_name": "pls_eeg_aligned_axis",
                    "model": model_name,
                    "subject": groups_test,
                    "z_true": z_test,
                    "z_pred": z_pred,
                    "axis_valence_weight": axis_meta["axis_valence_weight"],
                    "axis_arousal_weight": axis_meta["axis_arousal_weight"],
                    "axis_angle_deg": axis_meta["axis_angle_deg"],
                }
            )
        )

    axis_row = {
        "protocol": protocol,
        "fold": fold,
        "axis_name": "pls_eeg_aligned_axis",
        **axis_meta,
    }

    return rows, pred_frames, axis_row


def evaluate_random_split(X: np.ndarray, y_va: np.ndarray, groups: np.ndarray):
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y_va,
        groups,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    return evaluate_one_split(
        X_train,
        X_test,
        y_train,
        y_test,
        groups_test,
        protocol="random_split",
        fold=-1,
    )


def evaluate_groupkfold(X: np.ndarray, y_va: np.ndarray, groups: np.ndarray):
    cv = GroupKFold(n_splits=N_SPLITS_GROUP)

    all_rows = []
    all_pred_frames = []
    axis_rows = []
    manifest_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_va, groups)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y_va[train_idx]
        y_test = y_va[test_idx]
        groups_test = groups[test_idx]

        train_subjects = sorted(pd.Series(groups[train_idx]).astype(str).unique().tolist())
        test_subjects = sorted(pd.Series(groups[test_idx]).astype(str).unique().tolist())

        manifest_rows.append(
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

        rows, pred_frames, axis_row = evaluate_one_split(
            X_train,
            X_test,
            y_train,
            y_test,
            groups_test,
            protocol="groupkfold",
            fold=fold_idx,
        )

        all_rows.extend(rows)
        all_pred_frames.extend(pred_frames)
        axis_rows.append(axis_row)

    return all_rows, all_pred_frames, axis_rows, pd.DataFrame(manifest_rows)


def aggregate_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
        "true_mean",
        "pred_mean",
        "true_std",
        "pred_std",
        "axis_valence_weight",
        "axis_arousal_weight",
        "axis_angle_deg",
    ]

    random_rows = fold_df[fold_df["protocol"] == "random_split"].copy()
    group_rows = fold_df[fold_df["protocol"] == "groupkfold"].copy()

    random_out = random_rows.copy()
    random_out["aggregation"] = "single_split"

    for col in metric_cols:
        random_out[f"{col}_mean"] = random_out[col]
        random_out[f"{col}_std"] = np.nan

    random_out = random_out[
        ["protocol", "axis_name", "model", "aggregation"]
        + [f"{col}_mean" for col in metric_cols]
        + [f"{col}_std" for col in metric_cols]
    ]

    group_out = (
        group_rows
        .groupby(["protocol", "axis_name", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )

    group_out.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in group_out.columns
    ]

    group_out = group_out.reset_index()
    group_out["aggregation"] = "mean_across_folds"

    return pd.concat([random_out, group_out], ignore_index=True, sort=False)


def add_dummy_improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (protocol, axis_name), sub in summary_df.groupby(["protocol", "axis_name"]):
        dummy = sub[sub["model"] == "dummy_mean"]

        if dummy.empty:
            continue

        dummy_rmse = float(dummy.iloc[0]["rmse_mean"])

        for _, row in sub.iterrows():
            model_rmse = float(row["rmse_mean"])
            improvement = dummy_rmse - model_rmse
            improvement_pct = improvement / dummy_rmse if dummy_rmse != 0 else np.nan

            out = row.to_dict()
            out["dummy_rmse"] = dummy_rmse
            out["improvement_over_dummy_rmse"] = improvement
            out["improvement_over_dummy_rmse_pct"] = improvement_pct
            rows.append(out)

    return pd.DataFrame(rows)


def best_model_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    non_dummy = summary_df[summary_df["model"] != "dummy_mean"].copy()

    rows = []

    for protocol, sub in non_dummy.groupby("protocol"):
        best = sub.sort_values("rmse_mean", ascending=True).iloc[0]

        rows.append(
            {
                "protocol": protocol,
                "axis_name": best["axis_name"],
                "best_model": best["model"],
                "best_rmse": best["rmse_mean"],
                "best_rmse_std": best.get("rmse_std", np.nan),
                "dummy_rmse": best["dummy_rmse"],
                "improvement_over_dummy_rmse": best["improvement_over_dummy_rmse"],
                "improvement_over_dummy_rmse_pct": best["improvement_over_dummy_rmse_pct"],
                "pearson": best["pearson_mean"],
                "spearman": best["spearman_mean"],
                "r2": best["r2_mean"],
                "axis_valence_weight": best["axis_valence_weight_mean"],
                "axis_arousal_weight": best["axis_arousal_weight_mean"],
                "axis_angle_deg": best["axis_angle_deg_mean"],
                "axis_angle_std": best.get("axis_angle_deg_std", np.nan),
            }
        )

    return pd.DataFrame(rows)


def axis_stability(axis_df: pd.DataFrame) -> pd.DataFrame:
    group = axis_df[axis_df["protocol"] == "groupkfold"].copy()

    return pd.DataFrame(
        [
            {
                "axis_name": "pls_eeg_aligned_axis",
                "axis_type": "supervised_pls_rank1",
                "mean_axis_valence_weight": float(group["axis_valence_weight"].mean()),
                "std_axis_valence_weight": float(group["axis_valence_weight"].std(ddof=0)),
                "mean_axis_arousal_weight": float(group["axis_arousal_weight"].mean()),
                "std_axis_arousal_weight": float(group["axis_arousal_weight"].std(ddof=0)),
                "mean_axis_angle_deg": float(group["axis_angle_deg"].mean()),
                "std_axis_angle_deg": float(group["axis_angle_deg"].std(ddof=0)),
                "min_axis_angle_deg": float(group["axis_angle_deg"].min()),
                "max_axis_angle_deg": float(group["axis_angle_deg"].max()),
                "n_folds": int(group.shape[0]),
            }
        ]
    )


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
    y_va = df[["valence", "arousal"]].values
    groups = df["subject"].values

    print("Running random split PLS EEG-aligned axis...")
    random_rows, random_pred_frames, random_axis_row = evaluate_random_split(X, y_va, groups)

    print("Running GroupKFold PLS EEG-aligned axis...")
    group_rows, group_pred_frames, group_axis_rows, fold_manifest = evaluate_groupkfold(X, y_va, groups)

    fold_df = pd.DataFrame(random_rows + group_rows)
    predictions_df = pd.concat(random_pred_frames + group_pred_frames, ignore_index=True)
    axis_df = pd.DataFrame([random_axis_row] + group_axis_rows)

    summary = aggregate_results(fold_df)
    summary_with_dummy = add_dummy_improvement(summary)
    best_df = best_model_summary(summary_with_dummy)
    stability_df = axis_stability(axis_df)

    fold_path = TABLES_DIR / "pls_axis_fold_metrics_absolute.csv"
    summary_path = TABLES_DIR / "pls_axis_summary_absolute.csv"
    best_path = TABLES_DIR / "pls_axis_best_models_absolute.csv"
    pred_path = TABLES_DIR / "pls_axis_predictions_absolute.csv"
    axis_path = TABLES_DIR / "pls_axis_definitions_absolute.csv"
    stability_path = TABLES_DIR / "pls_axis_stability_absolute.csv"
    manifest_path = TABLES_DIR / "pls_axis_groupkfold_manifest.csv"
    metrics_path = METRICS_DIR / "pls_axis_absolute_summary.json"

    fold_df.to_csv(fold_path, index=False)
    summary_with_dummy.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    axis_df.to_csv(axis_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    fold_manifest.to_csv(manifest_path, index=False)

    metrics = {
        "feature_table": str(FEATURE_TABLE),
        "n_trials": int(df.shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "n_features": int(len(feature_cols)),
        "axis": "PLSRegression(n_components=1), fold-wise train-only",
        "models": list(get_models().keys()),
        "protocols": ["random_split", "groupkfold"],
        "best_models": best_df.to_dict(orient="records"),
        "axis_stability": stability_df.to_dict(orient="records"),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved:")
    print(f"- {fold_path}")
    print(f"- {summary_path}")
    print(f"- {best_path}")
    print(f"- {pred_path}")
    print(f"- {axis_path}")
    print(f"- {stability_path}")
    print(f"- {manifest_path}")
    print(f"- {metrics_path}")

    print("\nBest model summary:")
    print(best_df.to_string(index=False))

    print("\nPLS axis stability:")
    print(stability_df.to_string(index=False))

    print("\nFold-wise PLS axes:")
    print(axis_df.to_string(index=False))


if __name__ == "__main__":
    main()