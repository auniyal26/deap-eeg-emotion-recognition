import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, METRICS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

FEATURE_TABLES = {
    "absolute": TABLES_DIR / "bandpower_features_absolute.csv",
    "log": TABLES_DIR / "bandpower_features_log.csv",
    "relative_channel": TABLES_DIR / "bandpower_features_relative_channel.csv",
    "trial_proportion": TABLES_DIR / "bandpower_features_trial_proportion.csv",
}

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
    raise ValueError(method)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": safe_corr(y_true, y_pred, "pearson"),
        "spearman": safe_corr(y_true, y_pred, "spearman"),
        "true_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
    }


def va_2d_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors_2d = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    return {
        "rmse_or_error": float(np.mean(errors_2d)),
        "valence_rmse": rmse(y_true[:, 0], y_pred[:, 0]),
        "arousal_rmse": rmse(y_true[:, 1], y_pred[:, 1]),
        "valence_r2": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "arousal_r2": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "valence_pearson": safe_corr(y_true[:, 0], y_pred[:, 0], "pearson"),
        "arousal_pearson": safe_corr(y_true[:, 1], y_pred[:, 1], "pearson"),
        "valence_spearman": safe_corr(y_true[:, 0], y_pred[:, 0], "spearman"),
        "arousal_spearman": safe_corr(y_true[:, 1], y_pred[:, 1], "spearman"),
    }


def scalar_models() -> dict:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "svr_rbf": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1))]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def multioutput_models() -> dict:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "svr_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.1))),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def zscore_train_apply(train: np.ndarray, test: np.ndarray):
    mean = float(np.mean(train))
    std = float(np.std(train))
    if std == 0:
        raise ValueError("Zero std in training target")
    return (train - mean) / std, (test - mean) / std


def fixed_scalar_scores(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    v_train, v_test = zscore_train_apply(y_train[:, 0], y_test[:, 0])
    a_train, a_test = zscore_train_apply(y_train[:, 1], y_test[:, 1])

    return {
        "z_valence": {
            "z_train": v_train,
            "z_test": v_test,
            "axis_type": "fixed",
            "axis_valence_weight": 1.0,
            "axis_arousal_weight": 0.0,
            "axis_angle_deg": 0.0,
        },
        "z_arousal": {
            "z_train": a_train,
            "z_test": a_test,
            "axis_type": "fixed",
            "axis_valence_weight": 0.0,
            "axis_arousal_weight": 1.0,
            "axis_angle_deg": 90.0,
        },
        "positive_diagonal_zv_plus_za": {
            "z_train": (v_train + a_train) / np.sqrt(2),
            "z_test": (v_test + a_test) / np.sqrt(2),
            "axis_type": "fixed",
            "axis_valence_weight": 1.0 / np.sqrt(2),
            "axis_arousal_weight": 1.0 / np.sqrt(2),
            "axis_angle_deg": 45.0,
        },
        "anti_diagonal_zv_minus_za": {
            "z_train": (v_train - a_train) / np.sqrt(2),
            "z_test": (v_test - a_test) / np.sqrt(2),
            "axis_type": "fixed",
            "axis_valence_weight": 1.0 / np.sqrt(2),
            "axis_arousal_weight": -1.0 / np.sqrt(2),
            "axis_angle_deg": -45.0,
        },
    }


def label_pca_score(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    scaler = StandardScaler()
    y_train_z = scaler.fit_transform(y_train)
    y_test_z = scaler.transform(y_test)

    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    z_train = pca.fit_transform(y_train_z).reshape(-1)
    z_test = pca.transform(y_test_z).reshape(-1)

    axis = pca.components_[0].astype(float)
    if axis[0] < 0:
        axis = -axis
        z_train = -z_train
        z_test = -z_test

    axis = axis / np.linalg.norm(axis)
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))

    return {
        "z_train": z_train,
        "z_test": z_test,
        "axis_type": "label_pca",
        "axis_valence_weight": float(axis[0]),
        "axis_arousal_weight": float(axis[1]),
        "axis_angle_deg": angle,
    }


def pls_eeg_axis_score(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_z = x_scaler.fit_transform(X_train)
    X_test_z = x_scaler.transform(X_test)

    y_train_z = y_scaler.fit_transform(y_train)
    y_test_z = y_scaler.transform(y_test)

    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X_train_z, y_train_z)

    axis = pls.y_weights_[:, 0].astype(float)

    if np.linalg.norm(axis) == 0:
        raise RuntimeError("PLS returned zero axis")

    axis = axis / np.linalg.norm(axis)

    if axis[0] < 0:
        axis = -axis

    z_train = y_train_z @ axis
    z_test = y_test_z @ axis
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))

    return {
        "z_train": z_train,
        "z_test": z_test,
        "axis_type": "supervised_pls_rank1",
        "axis_valence_weight": float(axis[0]),
        "axis_arousal_weight": float(axis[1]),
        "axis_angle_deg": angle,
    }


def evaluate_scalar_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    representation: str,
    target_name: str,
    axis_type: str,
    axis_valence_weight: float,
    axis_arousal_weight: float,
    axis_angle_deg: float,
    fold: int,
) -> list[dict]:
    rows = []

    for model_name, model in scalar_models().items():
        fitted = clone(model)
        fitted.fit(X_train, z_train)
        z_pred = fitted.predict(X_test)

        row = {
            "representation": representation,
            "fold": fold,
            "target_name": target_name,
            "target_type": "scalar",
            "axis_type": axis_type,
            "model": model_name,
            "axis_valence_weight": axis_valence_weight,
            "axis_arousal_weight": axis_arousal_weight,
            "axis_angle_deg": axis_angle_deg,
        }
        row.update(scalar_metrics(z_test, z_pred))
        rows.append(row)

    return rows


def evaluate_va_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    representation: str,
    fold: int,
) -> list[dict]:
    rows = []

    for model_name, model in multioutput_models().items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)

        row = {
            "representation": representation,
            "fold": fold,
            "target_name": "direct_valence_arousal",
            "target_type": "2d_va",
            "axis_type": "none",
            "model": model_name,
            "axis_valence_weight": np.nan,
            "axis_arousal_weight": np.nan,
            "axis_angle_deg": np.nan,
        }
        row.update(va_2d_metrics(y_test, y_pred))
        rows.append(row)

    return rows


def run_representation_audit(representation: str, table_path: Path):
    df = pd.read_csv(table_path)
    feature_cols = get_feature_columns(df)

    if len(feature_cols) != 160:
        raise RuntimeError(f"{representation}: expected 160 features, got {len(feature_cols)}")

    X = df[feature_cols].values
    y = df[["valence", "arousal"]].values
    groups = df["subject"].values

    cv = GroupKFold(n_splits=N_SPLITS_GROUP)

    rows = []
    manifest_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        manifest_rows.append(
            {
                "representation": representation,
                "fold": fold,
                "train_subjects": ",".join(sorted(pd.Series(groups[train_idx]).astype(str).unique())),
                "test_subjects": ",".join(sorted(pd.Series(groups[test_idx]).astype(str).unique())),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_valence_mean": float(np.mean(y_train[:, 0])),
                "test_valence_mean": float(np.mean(y_test[:, 0])),
                "train_arousal_mean": float(np.mean(y_train[:, 1])),
                "test_arousal_mean": float(np.mean(y_test[:, 1])),
            }
        )

        rows.extend(evaluate_va_target(X_train, X_test, y_train, y_test, representation, fold))

        scalar_targets = fixed_scalar_scores(y_train, y_test)
        scalar_targets["label_pca_axis"] = label_pca_score(y_train, y_test)
        scalar_targets["pls_eeg_aligned_axis"] = pls_eeg_axis_score(X_train, X_test, y_train, y_test)

        for target_name, info in scalar_targets.items():
            rows.extend(
                evaluate_scalar_target(
                    X_train=X_train,
                    X_test=X_test,
                    z_train=info["z_train"],
                    z_test=info["z_test"],
                    representation=representation,
                    target_name=target_name,
                    axis_type=info["axis_type"],
                    axis_valence_weight=info["axis_valence_weight"],
                    axis_arousal_weight=info["axis_arousal_weight"],
                    axis_angle_deg=info["axis_angle_deg"],
                    fold=fold,
                )
            )

    return pd.DataFrame(rows), pd.DataFrame(manifest_rows)


def aggregate_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "rmse",
        "rmse_or_error",
        "r2",
        "pearson",
        "spearman",
        "valence_rmse",
        "arousal_rmse",
        "valence_r2",
        "arousal_r2",
        "valence_pearson",
        "arousal_pearson",
        "valence_spearman",
        "arousal_spearman",
        "axis_valence_weight",
        "axis_arousal_weight",
        "axis_angle_deg",
    ]

    available = [c for c in metric_cols if c in fold_df.columns]

    out = (
        fold_df
        .groupby(
            ["representation", "target_name", "target_type", "axis_type", "model"],
            as_index=False,
        )[available]
        .agg(["mean", "std"])
    )

    out.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in out.columns
    ]

    return out.reset_index()


def add_dummy_improvement(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (representation, target_name), sub in summary_df.groupby(["representation", "target_name"]):
        dummy = sub[sub["model"] == "dummy_mean"]

        if dummy.empty:
            continue

        metric_col = "rmse_or_error_mean" if target_name == "direct_valence_arousal" else "rmse_mean"

        dummy_score = float(dummy.iloc[0][metric_col])

        for _, row in sub.iterrows():
            model_score = float(row[metric_col])
            improvement = dummy_score - model_score
            improvement_pct = improvement / dummy_score if dummy_score != 0 else np.nan

            out = row.to_dict()
            out["primary_metric"] = metric_col.replace("_mean", "")
            out["dummy_score"] = dummy_score
            out["model_score"] = model_score
            out["improvement_over_dummy"] = improvement
            out["improvement_over_dummy_pct"] = improvement_pct
            rows.append(out)

    return pd.DataFrame(rows)


def best_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    non_dummy = summary_df[summary_df["model"] != "dummy_mean"].copy()

    rows = []

    for (representation, target_name), sub in non_dummy.groupby(["representation", "target_name"]):
        best = sub.sort_values("model_score", ascending=True).iloc[0]
        rows.append(
            {
                "representation": representation,
                "target_name": target_name,
                "target_type": best["target_type"],
                "axis_type": best["axis_type"],
                "best_model": best["model"],
                "primary_metric": best["primary_metric"],
                "model_score": best["model_score"],
                "dummy_score": best["dummy_score"],
                "improvement_over_dummy": best["improvement_over_dummy"],
                "improvement_over_dummy_pct": best["improvement_over_dummy_pct"],
                "axis_valence_weight_mean": best.get("axis_valence_weight_mean", np.nan),
                "axis_arousal_weight_mean": best.get("axis_arousal_weight_mean", np.nan),
                "axis_angle_deg_mean": best.get("axis_angle_deg_mean", np.nan),
                "axis_angle_deg_std": best.get("axis_angle_deg_std", np.nan),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("improvement_over_dummy_pct", ascending=False)
        .reset_index(drop=True)
    )


def axis_stability(fold_df: pd.DataFrame) -> pd.DataFrame:
    axis_df = fold_df[
        (fold_df["model"] == "dummy_mean")
        & (fold_df["target_type"] == "scalar")
    ].copy()

    rows = []

    for (representation, target_name, axis_type), sub in axis_df.groupby(
        ["representation", "target_name", "axis_type"]
    ):
        rows.append(
            {
                "representation": representation,
                "target_name": target_name,
                "axis_type": axis_type,
                "mean_axis_valence_weight": float(sub["axis_valence_weight"].mean()),
                "std_axis_valence_weight": float(sub["axis_valence_weight"].std(ddof=0)),
                "mean_axis_arousal_weight": float(sub["axis_arousal_weight"].mean()),
                "std_axis_arousal_weight": float(sub["axis_arousal_weight"].std(ddof=0)),
                "mean_axis_angle_deg": float(sub["axis_angle_deg"].mean()),
                "std_axis_angle_deg": float(sub["axis_angle_deg"].std(ddof=0)),
                "min_axis_angle_deg": float(sub["axis_angle_deg"].min()),
                "max_axis_angle_deg": float(sub["axis_angle_deg"].max()),
                "n_folds": int(sub.shape[0]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_results = []
    all_manifests = []

    for representation, path in FEATURE_TABLES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing table for {representation}: {path}")

        print(f"\nRunning audit for representation: {representation}")
        fold_df, manifest_df = run_representation_audit(representation, path)

        all_fold_results.append(fold_df)
        all_manifests.append(manifest_df)

    fold_df = pd.concat(all_fold_results, ignore_index=True)
    manifest_df = pd.concat(all_manifests, ignore_index=True)

    summary = aggregate_fold_metrics(fold_df)
    summary = add_dummy_improvement(summary)
    best_df = best_models(summary)
    stability_df = axis_stability(fold_df)

    fold_path = TABLES_DIR / "master_axis_audit_fold_metrics.csv"
    summary_path = TABLES_DIR / "master_axis_audit_summary.csv"
    best_path = TABLES_DIR / "master_axis_audit_best_models.csv"
    stability_path = TABLES_DIR / "master_axis_stability_across_representations.csv"
    manifest_path = TABLES_DIR / "master_axis_audit_groupkfold_manifest.csv"
    metrics_path = METRICS_DIR / "master_axis_audit_across_representations.json"

    fold_df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    metrics = {
        "feature_tables": {k: str(v) for k, v in FEATURE_TABLES.items()},
        "n_representations": len(FEATURE_TABLES),
        "representations": list(FEATURE_TABLES.keys()),
        "n_groupkfold_splits": N_SPLITS_GROUP,
        "targets_tested": sorted(fold_df["target_name"].unique().tolist()),
        "models_tested": sorted(fold_df["model"].unique().tolist()),
        "best_models": best_df.to_dict(orient="records"),
        "axis_stability": stability_df.to_dict(orient="records"),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved:")
    print(f"- {fold_path}")
    print(f"- {summary_path}")
    print(f"- {best_path}")
    print(f"- {stability_path}")
    print(f"- {manifest_path}")
    print(f"- {metrics_path}")

    print("\nBest models across representations:")
    print(best_df.to_string(index=False))

    print("\nAxis stability across representations:")
    print(stability_df.to_string(index=False))


if __name__ == "__main__":
    main()