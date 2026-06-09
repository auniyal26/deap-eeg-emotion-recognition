import json
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5

LABEL_COLS = ["valence", "arousal", "dominance", "liking"]
VA_COLS = ["valence", "arousal"]
META_COLS = ["subject", "trial"] + LABEL_COLS

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "results" / "tables"
METRIC_DIR = REPO_ROOT / "results" / "metrics"

RAW_LABELS_PATH = TABLE_DIR / "raw_trial_labels.csv"
FEATURES_PATH = TABLE_DIR / "bandpower_features_relative_channel_plus_asymmetry.csv"

OUT_VA_GEOMETRY = TABLE_DIR / "deap_label_space_sensitivity_va_geometry.csv"
OUT_4D_CORR = TABLE_DIR / "deap_label_space_4d_correlation_matrix.csv"
OUT_4D_PCA_LOADINGS = TABLE_DIR / "deap_label_space_4d_pca_loadings.csv"
OUT_4D_PCA_VARIANCE = TABLE_DIR / "deap_label_space_4d_pca_variance.csv"
OUT_4D_FOLD_METRICS = TABLE_DIR / "deap_label_space_4d_fold_metrics.csv"
OUT_4D_SUMMARY = TABLE_DIR / "deap_label_space_4d_summary.csv"
OUT_METADATA = METRIC_DIR / "deap_label_space_sensitivity_4d_metadata.json"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def safe_std(x: np.ndarray) -> np.ndarray:
    std = np.std(x, axis=0, ddof=0)
    return np.where(std == 0, 1.0, std)


def zscore_with_train(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels = pd.read_csv(RAW_LABELS_PATH)
    features = pd.read_csv(FEATURES_PATH)

    required = set(META_COLS)
    missing_labels = required - set(labels.columns)
    missing_features = required - set(features.columns)

    if missing_labels:
        raise ValueError(f"Missing columns in raw labels: {sorted(missing_labels)}")
    if missing_features:
        raise ValueError(f"Missing columns in features: {sorted(missing_features)}")

    labels["subject"] = labels["subject"].astype(str)
    features["subject"] = features["subject"].astype(str)

    labels["trial"] = labels["trial"].astype(int)
    features["trial"] = features["trial"].astype(int)

    merged = features.merge(
        labels[META_COLS],
        on=["subject", "trial"],
        suffixes=("_feature", ""),
        validate="one_to_one",
    )

    # Keep label columns from raw labels after merge.
    for col in LABEL_COLS:
        feature_col = f"{col}_feature"
        if feature_col in merged.columns:
            merged.drop(columns=[feature_col], inplace=True)

    feature_cols = [c for c in merged.columns if c not in META_COLS]

    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged[LABEL_COLS].to_numpy(dtype=float)
    groups = merged["subject"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or Inf.")
    if not np.isfinite(y).all():
        raise ValueError("Label matrix contains NaN or Inf.")

    return labels, merged, X, y, groups, feature_cols


def make_va_conditions(labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)

    base = labels[["subject", "trial"] + VA_COLS].copy()

    conditions: dict[str, pd.DataFrame] = {}

    clean = base.copy()
    clean["condition"] = "clean"
    conditions["clean"] = clean

    bad = base.copy()
    bad["valence"] = 9.0 - bad["valence"]
    bad["arousal"] = 9.0 - bad["arousal"]
    bad["condition"] = "bad_mirror_9_minus"
    conditions["bad_mirror_9_minus"] = bad

    reverse = base.copy()
    reverse["valence"] = 10.0 - reverse["valence"]
    reverse["arousal"] = 10.0 - reverse["arousal"]
    reverse["condition"] = "proper_reverse_10_minus"
    conditions["proper_reverse_10_minus"] = reverse

    within_pair = base.copy()
    shuffled_parts = []
    for subject, sub_df in within_pair.groupby("subject", sort=False):
        sub_df = sub_df.copy()
        shuffled_va = sub_df[VA_COLS].to_numpy().copy()
        rng.shuffle(shuffled_va, axis=0)
        sub_df.loc[:, VA_COLS] = shuffled_va
        shuffled_parts.append(sub_df)
    within_pair = pd.concat(shuffled_parts, axis=0).sort_index()
    within_pair["condition"] = "within_subject_trial_pair_shuffle"
    conditions["within_subject_trial_pair_shuffle"] = within_pair

    across_pair = base.copy()
    shuffled_va = across_pair[VA_COLS].to_numpy().copy()
    rng.shuffle(shuffled_va, axis=0)
    across_pair[VA_COLS] = shuffled_va
    across_pair["condition"] = "across_subject_trial_pair_shuffle"
    conditions["across_subject_trial_pair_shuffle"] = across_pair

    within_independent = base.copy()
    shuffled_parts = []
    for subject, sub_df in within_independent.groupby("subject", sort=False):
        sub_df = sub_df.copy()
        for col in VA_COLS:
            values = sub_df[col].to_numpy().copy()
            rng.shuffle(values)
            sub_df.loc[:, col] = values
        shuffled_parts.append(sub_df)
    within_independent = pd.concat(shuffled_parts, axis=0).sort_index()
    within_independent["condition"] = "within_subject_independent_va_shuffle"
    conditions["within_subject_independent_va_shuffle"] = within_independent

    across_independent = base.copy()
    for col in VA_COLS:
        values = across_independent[col].to_numpy().copy()
        rng.shuffle(values)
        across_independent[col] = values
    across_independent["condition"] = "across_subject_independent_va_shuffle"
    conditions["across_subject_independent_va_shuffle"] = across_independent

    return conditions


def quadrant_counts(y_va: np.ndarray) -> dict[str, int]:
    v = y_va[:, 0]
    a = y_va[:, 1]

    return {
        "low_v_low_a": int(np.sum((v <= 5.0) & (a <= 5.0))),
        "low_v_high_a": int(np.sum((v <= 5.0) & (a > 5.0))),
        "high_v_low_a": int(np.sum((v > 5.0) & (a <= 5.0))),
        "high_v_high_a": int(np.sum((v > 5.0) & (a > 5.0))),
    }


def va_groupkfold_sanity(
    condition_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    merged = features_df[["subject", "trial"] + feature_cols].merge(
        condition_df[["subject", "trial"] + VA_COLS],
        on=["subject", "trial"],
        validate="one_to_one",
    )

    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged[VA_COLS].to_numpy(dtype=float)
    groups = merged["subject"].to_numpy()

    gkf = GroupKFold(n_splits=N_SPLITS)

    dummy_scores = []
    rf_scores = []

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    dummy = DummyRegressor(strategy="mean")

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        dummy_model = clone(dummy)
        rf_model = clone(rf)

        dummy_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        pred_dummy = dummy_model.predict(X_test)
        pred_rf = rf_model.predict(X_test)

        dummy_scores.append(mean_euclidean_error(y_test, pred_dummy))
        rf_scores.append(mean_euclidean_error(y_test, pred_rf))

    dummy_score = float(np.mean(dummy_scores))
    rf_score = float(np.mean(rf_scores))
    improvement = dummy_score - rf_score
    improvement_pct = improvement / dummy_score if dummy_score != 0 else float("nan")

    return {
        "direct_va_dummy_2d_error": dummy_score,
        "direct_va_rf_2d_error": rf_score,
        "direct_va_rf_minus_dummy": rf_score - dummy_score,
        "direct_va_dummy_minus_rf": improvement,
        "direct_va_improvement_pct": improvement_pct,
    }


def run_va_sensitivity(labels: pd.DataFrame, features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    conditions = make_va_conditions(labels)

    for name, cond in conditions.items():
        y_va = cond[VA_COLS].to_numpy(dtype=float)

        pca_input = StandardScaler().fit_transform(y_va)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca.fit(pca_input)

        pc1 = canonicalize_axis(pca.components_[0], positive_index=0)
        pc1_angle = float(np.degrees(np.arctan2(pc1[1], pc1[0])))

        q = quadrant_counts(y_va)

        pearson = pearsonr(y_va[:, 0], y_va[:, 1]).statistic
        spearman = spearmanr(y_va[:, 0], y_va[:, 1]).statistic

        sanity = va_groupkfold_sanity(cond, features_df, feature_cols)

        row = {
            "condition": name,
            "n_trials": int(len(cond)),
            "valence_min": float(np.min(y_va[:, 0])),
            "valence_max": float(np.max(y_va[:, 0])),
            "valence_mean": float(np.mean(y_va[:, 0])),
            "valence_std": float(np.std(y_va[:, 0], ddof=0)),
            "arousal_min": float(np.min(y_va[:, 1])),
            "arousal_max": float(np.max(y_va[:, 1])),
            "arousal_mean": float(np.mean(y_va[:, 1])),
            "arousal_std": float(np.std(y_va[:, 1], ddof=0)),
            "va_pearson": float(pearson),
            "va_spearman": float(spearman),
            "va_pca_pc1_loading_valence": float(pc1[0]),
            "va_pca_pc1_loading_arousal": float(pc1[1]),
            "va_pca_pc1_angle_deg": pc1_angle,
            "va_pca_pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "va_pca_pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
            **q,
            **sanity,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def run_4d_correlation(labels_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x, y in combinations(LABEL_COLS, 2):
        x_values = labels_df[x].to_numpy(dtype=float)
        y_values = labels_df[y].to_numpy(dtype=float)

        rows.append(
            {
                "label_x": x,
                "label_y": y,
                "pearson": float(pearsonr(x_values, y_values).statistic),
                "spearman": float(spearmanr(x_values, y_values).statistic),
            }
        )

    return pd.DataFrame(rows)


def run_global_4d_pca(labels_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = labels_df[LABEL_COLS].to_numpy(dtype=float)
    y_z = StandardScaler().fit_transform(y)

    pca = PCA(n_components=4, random_state=RANDOM_STATE)
    pca.fit(y_z)

    loading_rows = []
    variance_rows = []

    for pc_idx in range(4):
        axis = canonicalize_axis(pca.components_[pc_idx], positive_index=0)

        variance_rows.append(
            {
                "component": f"PC{pc_idx + 1}",
                "explained_variance_ratio": float(pca.explained_variance_ratio_[pc_idx]),
                "cumulative_explained_variance_ratio": float(
                    np.sum(pca.explained_variance_ratio_[: pc_idx + 1])
                ),
            }
        )

        for label, loading in zip(LABEL_COLS, axis):
            loading_rows.append(
                {
                    "component": f"PC{pc_idx + 1}",
                    "label": label,
                    "loading": float(loading),
                }
            )

    return pd.DataFrame(loading_rows), pd.DataFrame(variance_rows)


def evaluate_scalar_target(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_scalar: np.ndarray,
    y_test_scalar: np.ndarray,
    target_name: str,
    fold: int,
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
            "target_name": target_name,
            "target_type": "scalar",
            "model": model_name,
            "primary_metric": "rmse",
            "score": score,
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
    target_name: str,
    fold: int,
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
            "target_name": target_name,
            "target_type": "multioutput",
            "model": model_name,
            "primary_metric": "mean_euclidean_error",
            "score": score,
        }

        for idx, label in enumerate(label_names):
            row[f"{label}_rmse"] = rmse(y_test_multi[:, idx], pred[:, idx])

        if extra:
            row.update(extra)

        rows.append(row)

    return rows


def run_4d_groupkfold_audit(
    X: np.ndarray,
    y_raw: np.ndarray,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []

    pca_axes = []
    pls_axes = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_raw, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

        y_train_z, y_test_z, y_mu, y_sd = zscore_with_train(y_train_raw, y_test_raw)

        # Direct 4D standardized target.
        fold_rows.extend(
            evaluate_multioutput_target(
                X_train=X_train,
                X_test=X_test,
                y_train_multi=y_train_z,
                y_test_multi=y_test_z,
                target_name="direct_4d_vadl_z",
                fold=fold,
                label_names=LABEL_COLS,
                extra={
                    "axis_family": "direct_4d",
                    "representation": "relative_channel_plus_asymmetry",
                },
            )
        )

        # Individual standardized labels.
        for label_idx, label in enumerate(LABEL_COLS):
            fold_rows.extend(
                evaluate_scalar_target(
                    X_train=X_train,
                    X_test=X_test,
                    y_train_scalar=y_train_z[:, label_idx],
                    y_test_scalar=y_test_z[:, label_idx],
                    target_name=f"z_{label}",
                    fold=fold,
                    extra={
                        "axis_family": "individual_label",
                        "representation": "relative_channel_plus_asymmetry",
                    },
                )
            )

        # Fixed interpretable 4D scalar functions.
        radius_train = np.sqrt(np.sum(y_train_z ** 2, axis=1))
        radius_test = np.sqrt(np.sum(y_test_z ** 2, axis=1))

        positive_train = (y_train_z[:, 0] + y_train_z[:, 3]) / math.sqrt(2.0)
        positive_test = (y_test_z[:, 0] + y_test_z[:, 3]) / math.sqrt(2.0)

        activation_train = (y_train_z[:, 1] + y_train_z[:, 2]) / math.sqrt(2.0)
        activation_test = (y_test_z[:, 1] + y_test_z[:, 2]) / math.sqrt(2.0)

        fixed_targets = [
            ("radius_4d", radius_train, radius_test),
            ("positive_self_report_zv_plus_zl", positive_train, positive_test),
            ("activation_control_za_plus_zd", activation_train, activation_test),
        ]

        for target_name, train_target, test_target in fixed_targets:
            fold_rows.extend(
                evaluate_scalar_target(
                    X_train=X_train,
                    X_test=X_test,
                    y_train_scalar=train_target,
                    y_test_scalar=test_target,
                    target_name=target_name,
                    fold=fold,
                    extra={
                        "axis_family": "fixed_4d_scalar",
                        "representation": "relative_channel_plus_asymmetry",
                    },
                )
            )

        # Fold-wise 4D label PCA axes.
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca.fit(y_train_z)

        for pc_idx in range(2):
            axis = canonicalize_axis(pca.components_[pc_idx], positive_index=0)

            # For PC2, sign canonicalization by valence can be unstable if valence near zero.
            # Fall back to the largest absolute loading if needed.
            if abs(axis[0]) < 1e-8:
                strongest = int(np.argmax(np.abs(axis)))
                axis = canonicalize_axis(axis, positive_index=strongest)

            if pc_idx == 0:
                pca_axes.append(axis)

            y_train_pc = y_train_z @ axis
            y_test_pc = y_test_z @ axis

            extra = {
                "axis_family": "label_pca_4d",
                "representation": "relative_channel_plus_asymmetry",
                "explained_variance_ratio": float(pca.explained_variance_ratio_[pc_idx]),
            }

            for label, loading in zip(LABEL_COLS, axis):
                extra[f"axis_loading_{label}"] = float(loading)

            fold_rows.extend(
                evaluate_scalar_target(
                    X_train=X_train,
                    X_test=X_test,
                    y_train_scalar=y_train_pc,
                    y_test_scalar=y_test_pc,
                    target_name=f"label_pca_4d_axis{pc_idx + 1}",
                    fold=fold,
                    extra=extra,
                )
            )

        # Fold-wise supervised EEG-aligned PLS 4D axis.
        x_scaler = StandardScaler()
        X_train_z = x_scaler.fit_transform(X_train)
        X_test_z = x_scaler.transform(X_test)

        pls = PLSRegression(n_components=1)
        pls.fit(X_train_z, y_train_z)

        pls_axis = canonicalize_axis(pls.y_weights_[:, 0], positive_index=0)
        pls_axes.append(pls_axis)

        y_train_pls = y_train_z @ pls_axis
        y_test_pls = y_test_z @ pls_axis

        extra = {
            "axis_family": "pls_4d_eeg_aligned",
            "representation": "relative_channel_plus_asymmetry",
        }

        for label, loading in zip(LABEL_COLS, pls_axis):
            extra[f"axis_loading_{label}"] = float(loading)

        fold_rows.extend(
            evaluate_scalar_target(
                X_train=X_train,
                X_test=X_test,
                y_train_scalar=y_train_pls,
                y_test_scalar=y_test_pls,
                target_name="pls_4d_eeg_aligned_axis1",
                fold=fold,
                extra=extra,
            )
        )

    fold_df = pd.DataFrame(fold_rows)

    stability = {
        "pca_4d_axis1": axis_stability_summary(np.asarray(pca_axes), LABEL_COLS),
        "pls_4d_eeg_aligned_axis1": axis_stability_summary(np.asarray(pls_axes), LABEL_COLS),
    }

    return fold_df, stability


def axis_stability_summary(axes: np.ndarray, label_names: list[str]) -> dict:
    if axes.size == 0:
        return {}

    # Re-orient all axes to the first axis to prevent sign-flip artifacts.
    ref = axes[0]
    aligned = []

    for axis in axes:
        axis = np.asarray(axis, dtype=float)
        if np.dot(axis, ref) < 0:
            axis = -axis
        aligned.append(axis)

    aligned = np.asarray(aligned)

    mean_axis = np.mean(aligned, axis=0)
    mean_axis = canonicalize_axis(mean_axis, positive_index=0)

    cos_to_mean = [cosine_similarity(axis, mean_axis) for axis in aligned]
    angles_to_mean = [angle_deg_from_cos(c) for c in cos_to_mean]

    pairwise_cos = []
    for i, j in combinations(range(len(aligned)), 2):
        pairwise_cos.append(cosine_similarity(aligned[i], aligned[j]))

    summary = {
        "n_axes": int(len(aligned)),
        "mean_cosine_to_mean_axis": float(np.nanmean(cos_to_mean)),
        "min_cosine_to_mean_axis": float(np.nanmin(cos_to_mean)),
        "max_angle_to_mean_axis_deg": float(np.nanmax(angles_to_mean)),
        "mean_angle_to_mean_axis_deg": float(np.nanmean(angles_to_mean)),
        "pairwise_cosine_mean": float(np.nanmean(pairwise_cos)) if pairwise_cos else float("nan"),
        "pairwise_cosine_min": float(np.nanmin(pairwise_cos)) if pairwise_cos else float("nan"),
    }

    for idx, label in enumerate(label_names):
        summary[f"mean_loading_{label}"] = float(mean_axis[idx])
        summary[f"std_loading_{label}"] = float(np.std(aligned[:, idx], ddof=0))

    return summary


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = fold_df.groupby(["target_name", "target_type", "primary_metric"], dropna=False)

    for (target_name, target_type, metric), sub in grouped:
        dummy_scores = (
            sub[sub["model"] == "dummy_mean"]
            .sort_values("fold")["score"]
            .to_numpy(dtype=float)
        )
        rf_scores = (
            sub[sub["model"] == "random_forest"]
            .sort_values("fold")["score"]
            .to_numpy(dtype=float)
        )

        if len(dummy_scores) == 0 or len(rf_scores) == 0:
            continue

        dummy_mean = float(np.mean(dummy_scores))
        rf_mean = float(np.mean(rf_scores))

        improvement = dummy_mean - rf_mean
        improvement_pct = improvement / dummy_mean if dummy_mean != 0 else float("nan")

        row = {
            "target_name": target_name,
            "target_type": target_type,
            "primary_metric": metric,
            "dummy_score_mean": dummy_mean,
            "random_forest_score_mean": rf_mean,
            "dummy_minus_random_forest": improvement,
            "random_forest_minus_dummy": rf_mean - dummy_mean,
            "improvement_over_dummy_pct": improvement_pct,
            "n_folds": int(len(dummy_scores)),
            "rf_beats_dummy_n_folds": int(np.sum(rf_scores < dummy_scores)),
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        by="improvement_over_dummy_pct",
        ascending=False,
    )


def main() -> None:
    ensure_dirs()

    labels, merged, X, y, groups, feature_cols = load_data()

    print("Loaded data")
    print(f"Raw labels: {labels.shape}")
    print(f"Merged features: {merged.shape}")
    print(f"Feature matrix: {X.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Subjects: {len(np.unique(groups))}")

    # Part A: V/A sensitivity controls.
    print("\nRunning V/A sensitivity controls...")
    va_geometry = run_va_sensitivity(labels, merged, feature_cols)
    va_geometry.to_csv(OUT_VA_GEOMETRY, index=False)

    # Part B: 4D descriptive geometry.
    print("Running 4D correlation and global PCA...")
    corr_4d = run_4d_correlation(labels)
    corr_4d.to_csv(OUT_4D_CORR, index=False)

    pca_loadings_4d, pca_variance_4d = run_global_4d_pca(labels)
    pca_loadings_4d.to_csv(OUT_4D_PCA_LOADINGS, index=False)
    pca_variance_4d.to_csv(OUT_4D_PCA_VARIANCE, index=False)

    # Part C: 4D GroupKFold predictive audit.
    print("Running 4D GroupKFold predictive audit...")
    fold_df, stability = run_4d_groupkfold_audit(X, y, groups)
    fold_df.to_csv(OUT_4D_FOLD_METRICS, index=False)

    summary_df = summarize_fold_metrics(fold_df)
    summary_df.to_csv(OUT_4D_SUMMARY, index=False)

    metadata = {
        "script": "15_label_space_sensitivity_and_4d_audit.py",
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "validation": "GroupKFold by subject",
        "feature_file": str(FEATURES_PATH.relative_to(REPO_ROOT)),
        "raw_labels_file": str(RAW_LABELS_PATH.relative_to(REPO_ROOT)),
        "n_trials": int(len(merged)),
        "n_subjects": int(len(np.unique(groups))),
        "n_features": int(len(feature_cols)),
        "labels": LABEL_COLS,
        "va_conditions": [
        "clean",
        "bad_mirror_9_minus",
        "proper_reverse_10_minus",
        "within_subject_trial_pair_shuffle",
        "across_subject_trial_pair_shuffle",
        "within_subject_independent_va_shuffle",
        "across_subject_independent_va_shuffle",
        ],
        "four_d_targets": sorted(summary_df["target_name"].unique().tolist()),
        "axis_stability": stability,
        "outputs": {
            "va_geometry": str(OUT_VA_GEOMETRY.relative_to(REPO_ROOT)),
            "correlation_4d": str(OUT_4D_CORR.relative_to(REPO_ROOT)),
            "pca_loadings_4d": str(OUT_4D_PCA_LOADINGS.relative_to(REPO_ROOT)),
            "pca_variance_4d": str(OUT_4D_PCA_VARIANCE.relative_to(REPO_ROOT)),
            "fold_metrics_4d": str(OUT_4D_FOLD_METRICS.relative_to(REPO_ROOT)),
            "summary_4d": str(OUT_4D_SUMMARY.relative_to(REPO_ROOT)),
        },
    }

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved outputs:")
    for path in [
        OUT_VA_GEOMETRY,
        OUT_4D_CORR,
        OUT_4D_PCA_LOADINGS,
        OUT_4D_PCA_VARIANCE,
        OUT_4D_FOLD_METRICS,
        OUT_4D_SUMMARY,
        OUT_METADATA,
    ]:
        print(f"  {path.relative_to(REPO_ROOT)}")

    print("\nV/A sensitivity summary:")
    print(
        va_geometry[
            [
                "condition",
                "va_pearson",
                "va_spearman",
                "high_v_high_a",
                "va_pca_pc1_angle_deg",
                "direct_va_improvement_pct",
            ]
        ].to_string(index=False)
    )

    print("\n4D predictive audit summary:")
    print(
        summary_df[
            [
                "target_name",
                "primary_metric",
                "dummy_score_mean",
                "random_forest_score_mean",
                "improvement_over_dummy_pct",
                "rf_beats_dummy_n_folds",
            ]
        ].to_string(index=False)
    )

    print("\n4D axis stability:")
    print(json.dumps(stability, indent=2))


if __name__ == "__main__":
    main()