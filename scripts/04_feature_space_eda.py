import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, FIGURES_DIR


TABLES_DIR = RESULTS_DIR / "tables"
FEATURE_TABLE = TABLES_DIR / "deap_bandpower_features.csv"

TOP_N = 20


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def add_quadrant_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = [
        (df["valence_label"] == 0) & (df["arousal_label"] == 0),
        (df["valence_label"] == 0) & (df["arousal_label"] == 1),
        (df["valence_label"] == 1) & (df["arousal_label"] == 0),
        (df["valence_label"] == 1) & (df["arousal_label"] == 1),
    ]

    labels = [
        "low_valence_low_arousal",
        "low_valence_high_arousal",
        "high_valence_low_arousal",
        "high_valence_high_arousal",
    ]

    df["quadrant"] = np.select(conditions, labels, default="unknown")

    return df


def plot_valence_arousal_space(df: pd.DataFrame, output_path: Path) -> None:
    df = add_quadrant_labels(df)

    plt.figure(figsize=(8, 7))

    for quadrant in sorted(df["quadrant"].unique()):
        subset = df[df["quadrant"] == quadrant]
        plt.scatter(
            subset["valence_rating"],
            subset["arousal_rating"],
            alpha=0.7,
            s=28,
            label=quadrant,
        )

    plt.axvline(5, linestyle="--", linewidth=1)
    plt.axhline(5, linestyle="--", linewidth=1)

    plt.xlabel("Valence rating")
    plt.ylabel("Arousal rating")
    plt.title("DEAP Valence-Arousal Rating Space")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pca_feature_space(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    output_path: Path,
) -> None:
    X = df[feature_cols].values
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))

    for label_value in sorted(np.unique(y)):
        subset = X_pca[y == label_value]
        label_name = "low_0" if label_value == 0 else "high_1"

        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            alpha=0.65,
            s=24,
            label=label_name,
        )

    explained = pca.explained_variance_ratio_

    plt.xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    plt.title(f"PCA of EEG Bandpower Features Colored by {label_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_feature_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for feature in feature_cols:
        valence_corr = df[feature].corr(df["valence_rating"], method="spearman")
        arousal_corr = df[feature].corr(df["arousal_rating"], method="spearman")

        rows.append(
            {
                "feature": feature,
                "spearman_valence": valence_corr,
                "spearman_arousal": arousal_corr,
                "abs_spearman_valence": abs(valence_corr),
                "abs_spearman_arousal": abs(arousal_corr),
            }
        )

    corr_df = pd.DataFrame(rows)

    corr_df["max_abs_spearman"] = corr_df[
        ["abs_spearman_valence", "abs_spearman_arousal"]
    ].max(axis=1)

    corr_df = corr_df.sort_values("max_abs_spearman", ascending=False)

    return corr_df


def add_band_mean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    bands = ["delta", "theta", "alpha", "beta", "gamma"]

    for band in bands:
        band_cols = [col for col in feature_cols if col.endswith(f"_{band}")]
        df[f"mean_{band}"] = df[band_cols].mean(axis=1)

    return df


def plot_band_mean_correlations(df: pd.DataFrame, output_path: Path) -> None:
    bands = ["delta", "theta", "alpha", "beta", "gamma"]

    valence_corrs = []
    arousal_corrs = []

    for band in bands:
        valence_corrs.append(df[f"mean_{band}"].corr(df["valence_rating"], method="spearman"))
        arousal_corrs.append(df[f"mean_{band}"].corr(df["arousal_rating"], method="spearman"))

    x = np.arange(len(bands))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, valence_corrs, width, label="Valence")
    plt.bar(x + width / 2, arousal_corrs, width, label="Arousal")

    plt.axhline(0, linewidth=1)
    plt.xticks(x, bands)
    plt.ylabel("Spearman correlation")
    plt.title("Mean Bandpower Correlation with Valence/Arousal Ratings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE}\n"
            "Run scripts/03_extract_bandpower_features.py first."
        )

    df = pd.read_csv(FEATURE_TABLE)
    feature_cols = get_feature_columns(df)

    print(f"Loaded feature table: {FEATURE_TABLE}")
    print(f"Shape: {df.shape}")
    print(f"Feature columns: {len(feature_cols)}")

    # 1. Target-space plot
    va_space_path = FIGURES_DIR / "valence_arousal_rating_space.png"
    plot_valence_arousal_space(df, va_space_path)

    # 2. PCA plots
    pca_valence_path = FIGURES_DIR / "pca_bandpower_valence_labels.png"
    pca_arousal_path = FIGURES_DIR / "pca_bandpower_arousal_labels.png"

    plot_pca_feature_space(
        df=df,
        feature_cols=feature_cols,
        label_col="valence_label",
        output_path=pca_valence_path,
    )

    plot_pca_feature_space(
        df=df,
        feature_cols=feature_cols,
        label_col="arousal_label",
        output_path=pca_arousal_path,
    )

    # 3. Correlation table
    corr_df = compute_feature_correlations(df, feature_cols)
    corr_path = TABLES_DIR / "top_feature_correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    # 4. Mean band correlation plot
    df_with_band_means = add_band_mean_features(df, feature_cols)
    band_corr_path = FIGURES_DIR / "band_mean_correlation.png"
    plot_band_mean_correlations(df_with_band_means, band_corr_path)

    print("\nSaved figures:")
    print(f"- {va_space_path}")
    print(f"- {pca_valence_path}")
    print(f"- {pca_arousal_path}")
    print(f"- {band_corr_path}")

    print("\nSaved table:")
    print(f"- {corr_path}")

    print("\nTop 10 feature correlations:")
    print(corr_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()