import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, METRICS_DIR, FIGURES_DIR

TABLES_DIR = RESULTS_DIR / "tables"

FEATURE_TABLE = TABLES_DIR / "bandpower_features_absolute.csv"
LABEL_COLS = ["valence", "arousal", "dominance", "liking"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def feature_label_correlations(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

    for feature in feature_cols:
        x = df[feature].values

        for label in LABEL_COLS:
            y = df[label].values

            if np.std(x) == 0 or np.std(y) == 0:
                corr = np.nan
            else:
                corr = float(spearmanr(x, y)[0])

            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "spearman": corr,
                    "abs_spearman": abs(corr) if not np.isnan(corr) else np.nan,
                }
            )

    out = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    out.to_csv(TABLES_DIR / "feature_label_correlations_absolute.csv", index=False)
    return out


def band_mean_correlations(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    bands = ["delta", "theta", "alpha", "beta", "gamma"]

    band_df = df[["subject", "trial"] + LABEL_COLS].copy()

    for band in bands:
        cols = [col for col in feature_cols if col.endswith(f"_{band}")]
        band_df[f"{band}_mean_power"] = df[cols].mean(axis=1)

    rows = []

    for band in bands:
        x = band_df[f"{band}_mean_power"].values

        for label in LABEL_COLS:
            y = band_df[label].values

            if np.std(x) == 0 or np.std(y) == 0:
                corr = np.nan
            else:
                corr = float(spearmanr(x, y)[0])

            rows.append(
                {
                    "band": band,
                    "label": label,
                    "spearman": corr,
                    "abs_spearman": abs(corr) if not np.isnan(corr) else np.nan,
                }
            )

    out = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    out.to_csv(TABLES_DIR / "band_mean_label_correlations_absolute.csv", index=False)
    return out


def plot_band_mean_correlation_heatmap(corr_df: pd.DataFrame) -> None:
    pivot = corr_df.pivot(index="band", columns="label", values="spearman")
    pivot = pivot.loc[["delta", "theta", "alpha", "beta", "gamma"], LABEL_COLS]

    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.values, aspect="auto", vmin=-0.25, vmax=0.25)

    plt.xticks(np.arange(len(LABEL_COLS)), LABEL_COLS)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(im, label="Spearman correlation")
    plt.title("Mean bandpower correlation with continuous labels")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "band_mean_label_correlation_heatmap_absolute.png", dpi=300)
    plt.close()


def pca_feature_space(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = df[feature_cols].values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "subject": df["subject"].values,
            "trial": df["trial"].values,
            "pc1": z[:, 0],
            "pc2": z[:, 1],
            "valence": df["valence"].values,
            "arousal": df["arousal"].values,
            "dominance": df["dominance"].values,
            "liking": df["liking"].values,
        }
    )

    pca_df.to_csv(TABLES_DIR / "feature_pca_projection_absolute.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_df["pc1"],
        pca_df["pc2"],
        c=pd.factorize(pca_df["subject"])[0],
        s=18,
        alpha=0.7,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Bandpower PCA colored by subject")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_pca_by_subject_absolute.png", dpi=300)
    plt.close()

    for label in LABEL_COLS:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            pca_df["pc1"],
            pca_df["pc2"],
            c=pca_df[label],
            s=18,
            alpha=0.7,
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Bandpower PCA colored by {label}")
        plt.colorbar(label=label)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"feature_pca_by_{label}_absolute.png", dpi=300)
        plt.close()

    return {
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
        "pc1_pc2_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def subject_feature_scale_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    tmp = df[["subject"]].copy()
    tmp["feature_mean"] = df[feature_cols].mean(axis=1)
    tmp["feature_std"] = df[feature_cols].std(axis=1)
    tmp["feature_total_power"] = df[feature_cols].sum(axis=1)

    out = (
        tmp.groupby("subject")
        .agg(
            feature_mean_avg=("feature_mean", "mean"),
            feature_std_avg=("feature_std", "mean"),
            feature_total_power_avg=("feature_total_power", "mean"),
            feature_total_power_std=("feature_total_power", "std"),
        )
        .reset_index()
        .sort_values("feature_total_power_avg", ascending=False)
    )

    out.to_csv(TABLES_DIR / "subject_feature_scale_summary_absolute.csv", index=False)

    plt.figure(figsize=(14, 6))
    x = np.arange(len(out))
    plt.bar(x, out["feature_total_power_avg"])
    plt.xticks(x, out["subject"], rotation=90)
    plt.ylabel("Average total bandpower")
    plt.title("Subject-wise average total bandpower")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "subject_total_bandpower_absolute.png", dpi=300)
    plt.close()

    return out


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(f"Missing feature table: {FEATURE_TABLE}")

    df = pd.read_csv(FEATURE_TABLE)
    feature_cols = get_feature_columns(df)

    if len(feature_cols) != 160:
        raise RuntimeError(f"Expected 160 feature columns, found {len(feature_cols)}")

    corr_df = feature_label_correlations(df, feature_cols)
    band_corr_df = band_mean_correlations(df, feature_cols)
    plot_band_mean_correlation_heatmap(band_corr_df)

    pca_summary = pca_feature_space(df, feature_cols)
    subject_scale = subject_feature_scale_summary(df, feature_cols)

    summary = {
        "feature_table": str(FEATURE_TABLE),
        "n_trials": int(df.shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "n_features": int(len(feature_cols)),
        "top_feature_label_correlations": corr_df.head(30).to_dict(orient="records"),
        "top_band_mean_label_correlations": band_corr_df.head(20).to_dict(orient="records"),
        "pca_summary": pca_summary,
        "top_subjects_by_total_bandpower": subject_scale.head(10).to_dict(orient="records"),
    }

    summary_path = METRICS_DIR / "feature_label_audit_absolute_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Saved:")
    print(f"- {TABLES_DIR / 'feature_label_correlations_absolute.csv'}")
    print(f"- {TABLES_DIR / 'band_mean_label_correlations_absolute.csv'}")
    print(f"- {TABLES_DIR / 'feature_pca_projection_absolute.csv'}")
    print(f"- {TABLES_DIR / 'subject_feature_scale_summary_absolute.csv'}")
    print(f"- {summary_path}")

    print("\nTop feature-label correlations:")
    print(corr_df.head(20).to_string(index=False))

    print("\nTop band-mean label correlations:")
    print(band_corr_df.head(20).to_string(index=False))

    print("\nPCA summary:")
    print(json.dumps(pca_summary, indent=4))

    print("\nTop subjects by total bandpower:")
    print(subject_scale.head(10).to_string(index=False))


if __name__ == "__main__":
    main()