from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"

TABLES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)


FEATURE_FILES = {
    "absolute": TABLES / "bandpower_features_absolute.csv",
    "relative_channel": TABLES / "bandpower_features_relative_channel.csv",
    "relative_channel_plus_asymmetry": TABLES / "bandpower_features_relative_channel_plus_asymmetry.csv",
}

TARGETS = [
    "direct_valence_arousal",
    "z_valence",
    "positive_diagonal_zv_plus_za",
    "label_pca_axis",
    "pls_eeg_aligned_axis",
]

MODELS = {
    "dummy_mean": lambda: DummyRegressor(strategy="mean"),
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    ),
}

RANDOM_STATE = 42
N_GROUP_SPLITS = 5


def feature_columns(df):
    exclude = {"subject", "trial", "valence", "arousal", "dominance", "liking"}
    return [c for c in df.columns if c not in exclude]


def zscore_train_test(y_train, y_test):
    mu = y_train.mean(axis=0)
    sd = y_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (y_train - mu) / sd, (y_test - mu) / sd, mu, sd


def safe_corr(y_true, y_pred, kind="pearson"):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    if kind == "pearson":
        return float(pearsonr(y_true, y_pred)[0])
    if kind == "spearman":
        return float(spearmanr(y_true, y_pred)[0])
    raise ValueError(kind)


def target_from_fold(target_name, y_train_va_raw, y_test_va_raw, X_train=None):
    """
    Returns y_train, y_test, target_type, axis_type, axis weights.
    All scalar targets are created using train-fold standardization only.
    """
    y_train_z, y_test_z, _, _ = zscore_train_test(y_train_va_raw, y_test_va_raw)

    if target_name == "direct_valence_arousal":
        return y_train_va_raw, y_test_va_raw, "2d_va", "none", np.nan, np.nan, np.nan

    if target_name == "z_valence":
        return (
            y_train_z[:, 0],
            y_test_z[:, 0],
            "scalar",
            "fixed",
            1.0,
            0.0,
            0.0,
        )

    if target_name == "positive_diagonal_zv_plus_za":
        w = np.array([1.0, 1.0]) / np.sqrt(2)
        angle = float(np.degrees(np.arctan2(w[1], w[0])))
        return (
            y_train_z @ w,
            y_test_z @ w,
            "scalar",
            "fixed",
            float(w[0]),
            float(w[1]),
            angle,
        )

    if target_name == "label_pca_axis":
        pca = PCA(n_components=1)
        y_train_score = pca.fit_transform(y_train_z).ravel()
        y_test_score = pca.transform(y_test_z).ravel()
        w = pca.components_[0]

        # Canonicalize sign so valence weight is positive.
        if w[0] < 0:
            w = -w
            y_train_score = -y_train_score
            y_test_score = -y_test_score

        angle = float(np.degrees(np.arctan2(w[1], w[0])))
        return (
            y_train_score,
            y_test_score,
            "scalar",
            "label_pca",
            float(w[0]),
            float(w[1]),
            angle,
        )

    if target_name == "pls_eeg_aligned_axis":
        if X_train is None:
            raise ValueError("X_train required for PLS target.")

        x_scaler = StandardScaler()
        X_train_z = x_scaler.fit_transform(X_train)

        pls = PLSRegression(n_components=1)
        pls.fit(X_train_z, y_train_z)

        w = np.asarray(pls.y_weights_[:, 0], dtype=float)
        norm = np.linalg.norm(w)
        if norm == 0:
            w = np.array([1.0, 0.0])
        else:
            w = w / norm

        # Canonicalize sign so valence weight is positive.
        if w[0] < 0:
            w = -w

        y_train_score = y_train_z @ w
        y_test_score = y_test_z @ w
        angle = float(np.degrees(np.arctan2(w[1], w[0])))

        return (
            y_train_score,
            y_test_score,
            "scalar",
            "supervised_pls_rank1",
            float(w[0]),
            float(w[1]),
            angle,
        )

    raise ValueError(f"Unknown target: {target_name}")


def evaluate_predictions(y_true, y_pred, target_type):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if target_type == "2d_va":
        val_rmse = float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])))
        aro_rmse = float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])))
        error = float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))

        return {
            "primary_metric": "mean_2d_error",
            "primary_score": error,
            "valence_rmse": val_rmse,
            "arousal_rmse": aro_rmse,
            "valence_r2": float(r2_score(y_true[:, 0], y_pred[:, 0])),
            "arousal_r2": float(r2_score(y_true[:, 1], y_pred[:, 1])),
            "pearson": np.nan,
            "spearman": np.nan,
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "primary_metric": "rmse",
        "primary_score": rmse,
        "valence_rmse": np.nan,
        "arousal_rmse": np.nan,
        "valence_r2": np.nan,
        "arousal_r2": np.nan,
        "pearson": safe_corr(y_true, y_pred, "pearson"),
        "spearman": safe_corr(y_true, y_pred, "spearman"),
    }


def fit_predict(model_name, X_train, y_train, X_test, target_type):
    model = MODELS[model_name]()

    if target_type == "2d_va" and model_name == "ridge":
        # Pipeline Ridge supports multi-output directly, but this keeps behavior explicit.
        model = MultiOutputRegressor(make_pipeline(StandardScaler(), Ridge(alpha=10.0)))

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pred


def run_protocol(df, representation, protocol):
    X = df[feature_columns(df)].values
    y_raw = df[["valence", "arousal"]].values.astype(float)
    groups = df["subject"].values

    fold_rows = []

    if protocol == "random_split":
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        splits = [(0, train_idx, test_idx)]

    elif protocol == "groupkfold_subject":
        gkf = GroupKFold(n_splits=N_GROUP_SPLITS)
        splits = [
            (fold, train_idx, test_idx)
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_raw, groups))
        ]

    else:
        raise ValueError(protocol)

    for fold, train_idx, test_idx in splits:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train_va = y_raw[train_idx]
        y_test_va = y_raw[test_idx]

        for target_name in TARGETS:
            (
                y_train,
                y_test,
                target_type,
                axis_type,
                axis_v,
                axis_a,
                axis_angle,
            ) = target_from_fold(
                target_name,
                y_train_va,
                y_test_va,
                X_train=X_train,
            )

            fold_model_scores = {}

            for model_name in MODELS:
                y_pred = fit_predict(model_name, X_train, y_train, X_test, target_type)
                metrics = evaluate_predictions(y_test, y_pred, target_type)

                row = {
                    "representation": representation,
                    "protocol": protocol,
                    "fold": fold,
                    "target_name": target_name,
                    "target_type": target_type,
                    "axis_type": axis_type,
                    "model": model_name,
                    "axis_valence_weight": axis_v,
                    "axis_arousal_weight": axis_a,
                    "axis_angle_deg": axis_angle,
                    **metrics,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_train_subjects": int(len(np.unique(groups[train_idx]))),
                    "n_test_subjects": int(len(np.unique(groups[test_idx]))),
                }

                fold_rows.append(row)
                fold_model_scores[model_name] = metrics["primary_score"]

    return fold_rows


def summarize_fold_metrics(fold_df):
    group_cols = [
        "representation",
        "protocol",
        "target_name",
        "target_type",
        "axis_type",
        "model",
        "primary_metric",
    ]

    summary = (
        fold_df.groupby(group_cols, dropna=False)
        .agg(
            mean_score=("primary_score", "mean"),
            std_score=("primary_score", "std"),
            mean_pearson=("pearson", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_axis_valence_weight=("axis_valence_weight", "mean"),
            mean_axis_arousal_weight=("axis_arousal_weight", "mean"),
            mean_axis_angle_deg=("axis_angle_deg", "mean"),
            std_axis_angle_deg=("axis_angle_deg", "std"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )

    dummy = summary[summary["model"] == "dummy_mean"][
        [
            "representation",
            "protocol",
            "target_name",
            "dummy_score",
        ]
    ] if False else None

    dummy_scores = summary[summary["model"] == "dummy_mean"][
        ["representation", "protocol", "target_name", "mean_score"]
    ].rename(columns={"mean_score": "dummy_score"})

    summary = summary.merge(
        dummy_scores,
        on=["representation", "protocol", "target_name"],
        how="left",
    )

    # Lower score is better for RMSE/error.
    summary["improvement_over_dummy"] = summary["dummy_score"] - summary["mean_score"]
    summary["improvement_over_dummy_pct"] = (
        summary["improvement_over_dummy"] / summary["dummy_score"]
    )

    return summary


def make_optimism_table(summary):
    non_dummy = summary[summary["model"] != "dummy_mean"].copy()

    key_cols = ["representation", "target_name", "target_type", "axis_type", "model"]

    wide = non_dummy.pivot_table(
        index=key_cols,
        columns="protocol",
        values=[
            "mean_score",
            "dummy_score",
            "improvement_over_dummy",
            "improvement_over_dummy_pct",
            "mean_axis_angle_deg",
            "std_axis_angle_deg",
        ],
        aggfunc="first",
    )

    wide.columns = [f"{metric}_{protocol}" for metric, protocol in wide.columns]
    wide = wide.reset_index()

    if (
        "improvement_over_dummy_pct_random_split" in wide.columns
        and "improvement_over_dummy_pct_groupkfold_subject" in wide.columns
    ):
        wide["optimism_gap_pct_points"] = (
            wide["improvement_over_dummy_pct_random_split"]
            - wide["improvement_over_dummy_pct_groupkfold_subject"]
        )

    if "mean_score_random_split" in wide.columns and "mean_score_groupkfold_subject" in wide.columns:
        wide["score_gap_group_minus_random"] = (
            wide["mean_score_groupkfold_subject"] - wide["mean_score_random_split"]
        )

    if "optimism_gap_pct_points" in wide.columns:
        wide = wide.sort_values("optimism_gap_pct_points", ascending=False)

    return wide


def main():
    all_rows = []

    for representation, path in FEATURE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"\nLoading {representation}: {path.name}")
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}")

        for protocol in ["random_split", "groupkfold_subject"]:
            print(f"Running {representation} / {protocol}")
            rows = run_protocol(df, representation, protocol)
            all_rows.extend(rows)

    fold_df = pd.DataFrame(all_rows)
    fold_path = TABLES / "deap_split_optimism_fold_metrics.csv"
    fold_df.to_csv(fold_path, index=False)

    summary = summarize_fold_metrics(fold_df)
    summary_path = TABLES / "deap_split_optimism_summary.csv"
    summary.to_csv(summary_path, index=False)

    optimism = make_optimism_table(summary)
    optimism_path = TABLES / "deap_split_optimism_table.csv"
    optimism.to_csv(optimism_path, index=False)

    metadata = {
        "feature_files": {k: str(v) for k, v in FEATURE_FILES.items()},
        "targets": TARGETS,
        "models": list(MODELS.keys()),
        "protocols": ["random_split", "groupkfold_subject"],
        "random_state": RANDOM_STATE,
        "n_group_splits": N_GROUP_SPLITS,
        "n_fold_rows": int(len(fold_df)),
        "n_summary_rows": int(len(summary)),
        "n_optimism_rows": int(len(optimism)),
        "outputs": {
            "fold_metrics": str(fold_path),
            "summary": str(summary_path),
            "optimism_table": str(optimism_path),
        },
    }

    metadata_path = METRICS / "deap_split_optimism_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSplit optimism audit complete.")
    print(f"Saved: {fold_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {optimism_path}")
    print(f"Saved: {metadata_path}")

    print("\nTop optimism gaps:")
    cols = [
        c for c in [
            "representation",
            "target_name",
            "model",
            "mean_score_random_split",
            "mean_score_groupkfold_subject",
            "improvement_over_dummy_pct_random_split",
            "improvement_over_dummy_pct_groupkfold_subject",
            "optimism_gap_pct_points",
        ]
        if c in optimism.columns
    ]

    print(optimism[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()