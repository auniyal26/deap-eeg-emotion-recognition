import json
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

LABEL_COLS = ["valence", "arousal", "dominance", "liking"]
VA_COLS = ["valence", "arousal"]
META_COLS = ["subject", "trial"] + LABEL_COLS

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "results" / "tables"
METRIC_DIR = REPO_ROOT / "results" / "metrics"

REPRESENTATION_FILES = {
    "relative_channel": TABLE_DIR / "bandpower_features_relative_channel.csv",
    "relative_channel_plus_asymmetry": TABLE_DIR / "bandpower_features_relative_channel_plus_asymmetry.csv",
}

OUT_FOLD_METRICS = TABLE_DIR / "deap_loso_axis_audit_fold_metrics.csv"
OUT_SUMMARY = TABLE_DIR / "deap_loso_axis_audit_summary.csv"
OUT_AXIS_STABILITY = TABLE_DIR / "deap_loso_axis_stability.csv"
OUT_METADATA = METRIC_DIR / "deap_loso_axis_audit_metadata.json"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def safe_std(x: np.ndarray) -> np.ndarray:
    std = np.std(x, axis=0, ddof=0)
    return np.where(std == 0, 1.0, std)


def zscore_with_train(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(y_train, axis=0)
    sd = safe_std(y_train)
    return (y_train - mu) / sd, (y_test - mu) / sd, mu, sd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_euclidean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def canonicalize_axis(axis: np.ndarray, positive_index: int = 0) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return axis
    axis = axis / norm
    if axis[positive_index] < 0:
        axis = -axis
    return axis


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def angle_deg_from_cos(cos_value: float) -> float:
    if np.isnan(cos_value):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def load_representation(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing representation file: {path}")

    df = pd.read_csv(path)

    missing = set(META_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    df["subject"] = df["subject"].astype(str)
    df["trial"] = df["trial"].astype(int)

    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[LABEL_COLS].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(f"{path.name} contains NaN or Inf in features.")
    if not np.isfinite(y).all():
        raise ValueError(f"{path.name} contains NaN or Inf in labels.")

    return df, X, y, groups, feature_cols


def evaluate_scalar_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_scalar: np.ndarray,
    y_test_scalar: np.ndarray,
    representation: str,
    target_name: str,
    fold: int,
    heldout_subject: str,
    extra: dict | None = None,
) -> list[dict]:
    rows = []

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    for model_name, model in models.items():
        fitted = clone(model)
        fitted.fit(X_train, y_train_scalar)
        pred = fitted.predict(X_test)

        score = rmse(y_test_scalar, pred)

        row = {
            "fold": fold,
            "heldout_subject": heldout_subject,
            "representation": representation,
            "target_name": target_name,
            "target_type": "scalar",
            "model": model_name,
            "primary_metric": "rmse",
            "score": score,
            "n_test_trials": int(len(y_test_scalar)),
        }

        if extra:
            row.update(extra)

        rows.append(row)

    return rows


def evaluate_multioutput_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_multi: np.ndarray,
    y_test_multi: np.ndarray,
    representation: str,
    target_name: str,
    fold: int,
    heldout_subject: str,
    label_names: list[str],
    extra: dict | None = None,
) -> list[dict]:
    rows = []

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    for model_name, model in models.items():
        fitted = clone(model)
        fitted.fit(X_train, y_train_multi)
        pred = fitted.predict(X_test)

        score = mean_euclidean_error(y_test_multi, pred)

        row = {
            "fold": fold,
            "heldout_subject": heldout_subject,
            "representation": representation,
            "target_name": target_name,
            "target_type": "multioutput",
            "model": model_name,
            "primary_metric": "mean_euclidean_error",
            "score": score,
            "n_test_trials": int(len(y_test_multi)),
        }

        for idx, label in enumerate(label_names):
            row[f"{label}_rmse"] = rmse(y_test_multi[:, idx], pred[:, idx])

        if extra:
            row.update(extra)

        rows.append(row)

    return rows


def run_loso_for_representation(
    representation: str,
    X: np.ndarray,
    y_raw: np.ndarray,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, list[dict]]:
    logo = LeaveOneGroupOut()

    rows = []
    axis_rows = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_raw, groups)):
        heldout_subject = str(np.unique(groups[test_idx])[0])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

        y_train_z, y_test_z, _, _ = zscore_with_train(y_train_raw, y_test_raw)

        y_train_va = y_train_z[:, :2]
        y_test_va = y_test_z[:, :2]

        # 1. Direct V/A.
        rows.extend(
            evaluate_multioutput_target(
                X_train=X_train,
                X_test=X_test,
                y_train_multi=y_train_va,
                y_test_multi=y_test_va,
                representation=representation,
                target_name="direct_valence_arousal_z",
                fold=fold,
                heldout_subject=heldout_subject,
                label_names=VA_COLS,
                extra={"axis_family": "direct_va"},
            )
        )

        # 2. z_valence.
        rows.extend(
            evaluate_scalar_target(
                X_train=X_train,
                X_test=X_test,
                y_train_scalar=y_train_z[:, 0],
                y_test_scalar=y_test_z[:, 0],
                representation=representation,
                target_name="z_valence",
                fold=fold,
                heldout_subject=heldout_subject,
                extra={"axis_family": "individual_label"},
            )
        )

        # 3. V/A label PCA axis.
        pca_va = PCA(n_components=1, random_state=RANDOM_STATE)
        pca_va.fit(y_train_va)

        va_pca_axis = canonicalize_axis(pca_va.components_[0], positive_index=0)
        va_pca_angle = float(np.degrees(np.arctan2(va_pca_axis[1], va_pca_axis[0])))

        axis_rows.append(
            {
                "fold": fold,
                "heldout_subject": heldout_subject,
                "representation": representation,
                "axis_name": "label_pca_va_axis",
                "loading_valence": float(va_pca_axis[0]),
                "loading_arousal": float(va_pca_axis[1]),
                "loading_dominance": np.nan,
                "loading_liking": np.nan,
                "angle_deg_va": va_pca_angle,
                "explained_variance_ratio": float(pca_va.explained_variance_ratio_[0]),
            }
        )

        y_train_va_pca = y_train_va @ va_pca_axis
        y_test_va_pca = y_test_va @ va_pca_axis

        rows.extend(
            evaluate_scalar_target(
                X_train=X_train,
                X_test=X_test,
                y_train_scalar=y_train_va_pca,
                y_test_scalar=y_test_va_pca,
                representation=representation,
                target_name="label_pca_va_axis",
                fold=fold,
                heldout_subject=heldout_subject,
                extra={
                    "axis_family": "label_pca_va",
                    "axis_loading_valence": float(va_pca_axis[0]),
                    "axis_loading_arousal": float(va_pca_axis[1]),
                    "axis_angle_deg_va": va_pca_angle,
                    "explained_variance_ratio": float(pca_va.explained_variance_ratio_[0]),
                },
            )
        )

        # 4. V/A PLS EEG-aligned axis.
        x_scaler_va = StandardScaler()
        X_train_scaled_va = x_scaler_va.fit_transform(X_train)
        X_test_scaled_va = x_scaler_va.transform(X_test)

        pls_va = PLSRegression(n_components=1)
        pls_va.fit(X_train_scaled_va, y_train_va)

        pls_va_axis = canonicalize_axis(pls_va.y_weights_[:, 0], positive_index=0)
        pls_va_angle = float(np.degrees(np.arctan2(pls_va_axis[1], pls_va_axis[0])))

        axis_rows.append(
            {
                "fold": fold,
                "heldout_subject": heldout_subject,
                "representation": representation,
                "axis_name": "pls_va_eeg_aligned_axis",
                "loading_valence": float(pls_va_axis[0]),
                "loading_arousal": float(pls_va_axis[1]),
                "loading_dominance": np.nan,
                "loading_liking": np.nan,
                "angle_deg_va": pls_va_angle,
                "explained_variance_ratio": np.nan,
            }
        )

        y_train_va_pls = y_train_va @ pls_va_axis
        y_test_va_pls = y_test_va @ pls_va_axis

        rows.extend(
            evaluate_scalar_target(
                X_train=X_train,
                X_test=X_test,
                y_train_scalar=y_train_va_pls,
                y_test_scalar=y_test_va_pls,
                representation=representation,
                target_name="pls_va_eeg_aligned_axis",
                fold=fold,
                heldout_subject=heldout_subject,
                extra={
                    "axis_family": "pls_va_eeg_aligned",
                    "axis_loading_valence": float(pls_va_axis[0]),
                    "axis_loading_arousal": float(pls_va_axis[1]),
                    "axis_angle_deg_va": pls_va_angle,
                },
            )
        )

        # 5. Direct 4D.
        rows.extend(
            evaluate_multioutput_target(
                X_train=X_train,
                X_test=X_test,
                y_train_multi=y_train_z,
                y_test_multi=y_test_z,
                representation=representation,
                target_name="direct_4d_vadl_z",
                fold=fold,
                heldout_subject=heldout_subject,
                label_names=LABEL_COLS,
                extra={"axis_family": "direct_4d"},
            )
        )

        # 6. 4D label PCA axis 1.
        pca_4d = PCA(n_components=1, random_state=RANDOM_STATE)
        pca_4d.fit(y_train_z)

        pca_4d_axis = canonicalize_axis(pca_4d.components_[0], positive_index=0)

        axis_rows.append(
            {
                "fold": fold,
                "heldout_subject": heldout_subject,
                "representation": representation,
                "axis_name": "label_pca_4d_axis1",
                "loading_valence": float(pca_4d_axis[0]),
                "loading_arousal": float(pca_4d_axis[1]),
                "loading_dominance": float(pca_4d_axis[2]),
                "loading_liking": float(pca_4d_axis[3]),
                "angle_deg_va": np.nan,
                "explained_variance_ratio": float(pca_4d.explained_variance_ratio_[0]),
            }
        )

        y_train_4d_pca = y_train_z @ pca_4d_axis
        y_test_4d_pca = y_test_z @ pca_4d_axis

        rows.extend(
            evaluate_scalar_target(
                X_train=X_train,
                X_test=X_test,
                y_train_scalar=y_train_4d_pca,
                y_test_scalar=y_test_4d_pca,
                representation=representation,
                target_name="label_pca_4d_axis1",
                fold=fold,
                heldout_subject=heldout_subject,
                extra={
                    "axis_family": "label_pca_4d",
                    "axis_loading_valence": float(pca_4d_axis[0]),
                    "axis_loading_arousal": float(pca_4d_axis[1]),
                    "axis_loading_dominance": float(pca_4d_axis[2]),
                    "axis_loading_liking": float(pca_4d_axis[3]),
                    "explained_variance_ratio": float(pca_4d.explained_variance_ratio_[0]),
                },
            )
        )

        # 7. 4D PLS EEG-aligned axis.
        x_scaler_4d = StandardScaler()
        X_train_scaled_4d = x_scaler_4d.fit_transform(X_train)
        X_test_scaled_4d = x_scaler_4d.transform(X_test)

        pls_4d = PLSRegression(n_components=1)
        pls_4d.fit(X_train_scaled_4d, y_train_z)

        pls_4d_axis = canonicalize_axis(pls_4d.y_weights_[:, 0], positive_index=0)

        axis_rows.append(
            {
                "fold": fold,
                "heldout_subject": heldout_subject,
                "representation": representation,
                "axis_name": "pls_4d_eeg_aligned_axis1",
                "loading_valence": float(pls_4d_axis[0]),
                "loading_arousal": float(pls_4d_axis[1]),
                "loading_dominance": float(pls_4d_axis[2]),
                "loading_liking": float(pls_4d_axis[3]),
                "angle_deg_va": np.nan,
                "explained_variance_ratio": np.nan,
            }
        )

        y_train_4d_pls = y_train_z @ pls_4d_axis
        y_test_4d_pls = y_test_z @ pls_4d_axis

        rows.extend(
            evaluate_scalar_target(
                X_train=X_train,
                X_test=X_test,
                y_train_scalar=y_train_4d_pls,
                y_test_scalar=y_test_4d_pls,
                representation=representation,
                target_name="pls_4d_eeg_aligned_axis1",
                fold=fold,
                heldout_subject=heldout_subject,
                extra={
                    "axis_family": "pls_4d_eeg_aligned",
                    "axis_loading_valence": float(pls_4d_axis[0]),
                    "axis_loading_arousal": float(pls_4d_axis[1]),
                    "axis_loading_dominance": float(pls_4d_axis[2]),
                    "axis_loading_liking": float(pls_4d_axis[3]),
                },
            )
        )

    return pd.DataFrame(rows), axis_rows


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["representation", "target_name", "target_type", "primary_metric"]

    for keys, sub in fold_df.groupby(group_cols, dropna=False):
        representation, target_name, target_type, metric = keys

        dummy = (
            sub[sub["model"] == "dummy_mean"]
            .sort_values("fold")["score"]
            .to_numpy(dtype=float)
        )

        rf = (
            sub[sub["model"] == "random_forest"]
            .sort_values("fold")["score"]
            .to_numpy(dtype=float)
        )

        if len(dummy) == 0 or len(rf) == 0:
            continue

        dummy_mean = float(np.mean(dummy))
        rf_mean = float(np.mean(rf))
        improvement = dummy_mean - rf_mean
        improvement_pct = improvement / dummy_mean if dummy_mean != 0 else float("nan")

        per_subject_delta = dummy - rf

        rows.append(
            {
                "representation": representation,
                "target_name": target_name,
                "target_type": target_type,
                "primary_metric": metric,
                "dummy_score_mean": dummy_mean,
                "random_forest_score_mean": rf_mean,
                "dummy_minus_random_forest": improvement,
                "random_forest_minus_dummy": rf_mean - dummy_mean,
                "improvement_over_dummy_pct": improvement_pct,
                "n_subjects": int(len(dummy)),
                "rf_beats_dummy_n_subjects": int(np.sum(rf < dummy)),
                "mean_subject_delta": float(np.mean(per_subject_delta)),
                "median_subject_delta": float(np.median(per_subject_delta)),
                "std_subject_delta": float(np.std(per_subject_delta, ddof=0)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="improvement_over_dummy_pct",
        ascending=False,
    )


def axis_stability_table(axis_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    loading_cols = ["loading_valence", "loading_arousal", "loading_dominance", "loading_liking"]

    for (representation, axis_name), sub in axis_df.groupby(["representation", "axis_name"], dropna=False):
        available_cols = [
            col for col in loading_cols
            if col in sub.columns and not sub[col].isna().all()
        ]

        axes = sub[available_cols].to_numpy(dtype=float)

        if axes.ndim != 2 or axes.shape[0] == 0:
            continue

        ref = axes[0]
        aligned = []

        for axis in axes:
            if np.dot(axis, ref) < 0:
                axis = -axis
            norm = np.linalg.norm(axis)
            if norm != 0:
                axis = axis / norm
            aligned.append(axis)

        aligned = np.asarray(aligned)

        mean_axis = np.mean(aligned, axis=0)
        norm = np.linalg.norm(mean_axis)
        if norm != 0:
            mean_axis = mean_axis / norm

        cos_to_mean = [cosine_similarity(axis, mean_axis) for axis in aligned]
        angles = [angle_deg_from_cos(c) for c in cos_to_mean]

        pairwise_cos = []
        for i, j in combinations(range(len(aligned)), 2):
            pairwise_cos.append(cosine_similarity(aligned[i], aligned[j]))

        row = {
            "representation": representation,
            "axis_name": axis_name,
            "n_axes": int(len(aligned)),
            "mean_cosine_to_mean_axis": float(np.nanmean(cos_to_mean)),
            "min_cosine_to_mean_axis": float(np.nanmin(cos_to_mean)),
            "mean_angle_to_mean_axis_deg": float(np.nanmean(angles)),
            "max_angle_to_mean_axis_deg": float(np.nanmax(angles)),
            "pairwise_cosine_mean": float(np.nanmean(pairwise_cos)) if pairwise_cos else np.nan,
            "pairwise_cosine_min": float(np.nanmin(pairwise_cos)) if pairwise_cos else np.nan,
        }

        for col_idx, col in enumerate(available_cols):
            clean_label = col.replace("loading_", "")
            row[f"mean_loading_{clean_label}"] = float(mean_axis[col_idx])
            row[f"std_loading_{clean_label}"] = float(np.std(aligned[:, col_idx], ddof=0))

        if "angle_deg_va" in sub.columns and not sub["angle_deg_va"].isna().all():
            row["mean_va_angle_deg"] = float(sub["angle_deg_va"].mean())
            row["std_va_angle_deg"] = float(sub["angle_deg_va"].std(ddof=0))
            row["min_va_angle_deg"] = float(sub["angle_deg_va"].min())
            row["max_va_angle_deg"] = float(sub["angle_deg_va"].max())

        if "explained_variance_ratio" in sub.columns and not sub["explained_variance_ratio"].isna().all():
            row["mean_explained_variance_ratio"] = float(sub["explained_variance_ratio"].mean())
            row["std_explained_variance_ratio"] = float(sub["explained_variance_ratio"].std(ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()

    all_fold_dfs = []
    all_axis_rows = []

    metadata = {
        "script": "16_loso_axis_audit.py",
        "random_state": RANDOM_STATE,
        "validation": "LeaveOneGroupOut by subject",
        "representations": list(REPRESENTATION_FILES.keys()),
        "targets": [
            "direct_valence_arousal_z",
            "z_valence",
            "label_pca_va_axis",
            "pls_va_eeg_aligned_axis",
            "direct_4d_vadl_z",
            "label_pca_4d_axis1",
            "pls_4d_eeg_aligned_axis1",
        ],
        "models": ["dummy_mean", "random_forest"],
        "feature_files": {
            name: str(path.relative_to(REPO_ROOT))
            for name, path in REPRESENTATION_FILES.items()
        },
    }

    for representation, path in REPRESENTATION_FILES.items():
        print(f"\nLoading representation: {representation}")
        df, X, y, groups, feature_cols = load_representation(path)

        print(f"  shape: {df.shape}")
        print(f"  features: {len(feature_cols)}")
        print(f"  subjects: {len(np.unique(groups))}")

        print(f"Running LOSO audit: {representation}")
        fold_df, axis_rows = run_loso_for_representation(
            representation=representation,
            X=X,
            y_raw=y,
            groups=groups,
        )

        all_fold_dfs.append(fold_df)
        all_axis_rows.extend(axis_rows)

    fold_metrics = pd.concat(all_fold_dfs, axis=0, ignore_index=True)
    axis_df = pd.DataFrame(all_axis_rows)

    summary = summarize_metrics(fold_metrics)
    stability = axis_stability_table(axis_df)

    fold_metrics.to_csv(OUT_FOLD_METRICS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    stability.to_csv(OUT_AXIS_STABILITY, index=False)

    metadata["n_fold_metric_rows"] = int(len(fold_metrics))
    metadata["n_summary_rows"] = int(len(summary))
    metadata["n_axis_rows"] = int(len(axis_df))
    metadata["outputs"] = {
        "fold_metrics": str(OUT_FOLD_METRICS.relative_to(REPO_ROOT)),
        "summary": str(OUT_SUMMARY.relative_to(REPO_ROOT)),
        "axis_stability": str(OUT_AXIS_STABILITY.relative_to(REPO_ROOT)),
    }

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved outputs:")
    for path in [OUT_FOLD_METRICS, OUT_SUMMARY, OUT_AXIS_STABILITY, OUT_METADATA]:
        print(f"  {path.relative_to(REPO_ROOT)}")

    print("\nLOSO summary:")
    print(
        summary[
            [
                "representation",
                "target_name",
                "primary_metric",
                "dummy_score_mean",
                "random_forest_score_mean",
                "improvement_over_dummy_pct",
                "rf_beats_dummy_n_subjects",
            ]
        ].to_string(index=False)
    )

    print("\nAxis stability:")
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()