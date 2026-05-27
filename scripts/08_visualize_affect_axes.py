import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, FIGURES_DIR

TABLES_DIR = RESULTS_DIR / "tables"
FEATURE_TABLE = TABLES_DIR / "deap_bandpower_features.csv"
AXIS_DETAILS = TABLES_DIR / "eeg_aligned_axis_fold_details.csv"


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        return axis
    return axis / norm


def plot_pls_axis_original_space(df, axis_df, output_path):
    """
    Plot actual valence/arousal points, dummy mean point, and PLS learned axes.

    Important:
    PLS axis weights were learned in standardized valence-arousal space.
    To plot them in original rating space, we map the direction using target std.
    """
    valence = df["valence_rating"].values
    arousal = df["arousal_rating"].values

    y_mean = np.array([np.mean(valence), np.mean(arousal)])
    y_std = np.array([np.std(valence), np.std(arousal)])

    pls_df = axis_df[axis_df["axis_name"] == "pls_eeg_aligned_axis"].copy()

    if pls_df.empty:
        raise ValueError("No pls_eeg_aligned_axis rows found in axis details file.")

    plt.figure(figsize=(8, 7))

    # Actual points
    plt.scatter(
        valence,
        arousal,
        alpha=0.45,
        s=24,
        label="Actual trials",
    )

    # Dummy mean predictor
    plt.scatter(
        y_mean[0],
        y_mean[1],
        s=140,
        marker="X",
        label="Dummy mean prediction",
    )

    # Neutral threshold reference only for visual orientation, not modelling
    plt.axvline(5, linestyle="--", linewidth=1, label="Valence = 5 reference")
    plt.axhline(5, linestyle="--", linewidth=1, label="Arousal = 5 reference")

    # Fold-wise PLS axes
    t = np.linspace(-5, 5, 200)

    mapped_axes = []

    for _, row in pls_df.iterrows():
        axis_std = normalize_axis(
            np.array([row["axis_v_weight"], row["axis_a_weight"]], dtype=float)
        )

        # Convert standardized-space direction to original rating-space direction
        axis_original = normalize_axis(axis_std * y_std)
        mapped_axes.append(axis_original)

        line_points = y_mean.reshape(1, 2) + t.reshape(-1, 1) * axis_original.reshape(1, 2)

        plt.plot(
            line_points[:, 0],
            line_points[:, 1],
            alpha=0.35,
            linewidth=1.5,
            label=f"PLS fold {int(row['fold'])}",
        )

    # Mean PLS axis
    mean_axis = normalize_axis(np.mean(np.vstack(mapped_axes), axis=0))
    mean_line = y_mean.reshape(1, 2) + t.reshape(-1, 1) * mean_axis.reshape(1, 2)

    plt.plot(
        mean_line[:, 0],
        mean_line[:, 1],
        linewidth=3,
        label="Mean PLS EEG-aligned axis",
    )

    plt.xlim(-0.5, 9.5)
    plt.ylim(-0.5, 9.5)
    plt.xlabel("Valence rating")
    plt.ylabel("Arousal rating")
    plt.title("DEAP Affect Space: Actual Points, Dummy Mean, and EEG-Aligned PLS Axis")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pls_axis_standardized_space(df, axis_df, output_path):
    """
    Plot the same geometry in standardized valence-arousal space.
    This is closest to the space where the axis was learned.
    """
    valence = df["valence_rating"].values
    arousal = df["arousal_rating"].values

    Y = np.column_stack([valence, arousal])
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)
    Y_scaled = (Y - Y_mean) / Y_std

    pls_df = axis_df[axis_df["axis_name"] == "pls_eeg_aligned_axis"].copy()

    plt.figure(figsize=(8, 7))

    plt.scatter(
        Y_scaled[:, 0],
        Y_scaled[:, 1],
        alpha=0.45,
        s=24,
        label="Actual trials, standardized",
    )

    # Dummy mean in standardized space is exactly (0, 0)
    plt.scatter(
        0,
        0,
        s=140,
        marker="X",
        label="Dummy mean prediction",
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.axhline(0, linestyle="--", linewidth=1)

    t = np.linspace(-3, 3, 200)

    axes = []

    for _, row in pls_df.iterrows():
        axis = normalize_axis(
            np.array([row["axis_v_weight"], row["axis_a_weight"]], dtype=float)
        )
        axes.append(axis)

        line_points = t.reshape(-1, 1) * axis.reshape(1, 2)

        plt.plot(
            line_points[:, 0],
            line_points[:, 1],
            alpha=0.35,
            linewidth=1.5,
            label=f"PLS fold {int(row['fold'])}",
        )

    mean_axis = normalize_axis(np.mean(np.vstack(axes), axis=0))
    mean_line = t.reshape(-1, 1) * mean_axis.reshape(1, 2)

    plt.plot(
        mean_line[:, 0],
        mean_line[:, 1],
        linewidth=3,
        label="Mean PLS EEG-aligned axis",
    )

    plt.xlabel("Standardized valence")
    plt.ylabel("Standardized arousal")
    plt.title("Standardized Affect Space: Fold-wise EEG-Aligned PLS Axes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_TABLE}")

    if not AXIS_DETAILS.exists():
        raise FileNotFoundError(
            f"Axis details file not found: {AXIS_DETAILS}\n"
            "Run scripts/07_learn_eeg_aligned_affect_axis.py first."
        )

    df = pd.read_csv(FEATURE_TABLE)
    axis_df = pd.read_csv(AXIS_DETAILS)

    original_output = FIGURES_DIR / "affect_space_pls_axis_vs_dummy_original.png"
    standardized_output = FIGURES_DIR / "affect_space_pls_axis_vs_dummy_standardized.png"

    plot_pls_axis_original_space(df, axis_df, original_output)
    plot_pls_axis_standardized_space(df, axis_df, standardized_output)

    pls_df = axis_df[axis_df["axis_name"] == "pls_eeg_aligned_axis"]

    print("Saved:")
    print(f"- {original_output}")
    print(f"- {standardized_output}")

    print("\nPLS EEG-aligned axis fold details:")
    print(
        pls_df[
            [
                "fold",
                "axis_v_weight",
                "axis_a_weight",
                "axis_angle_deg",
                "rmse",
                "dummy_rmse",
                "rmse_improvement_over_dummy",
            ]
        ].to_string(index=False)
    )

    print("\nMean axis:")
    print(f"valence weight mean: {pls_df['axis_v_weight'].mean():.4f}")
    print(f"arousal weight mean: {pls_df['axis_a_weight'].mean():.4f}")
    print(f"angle mean: {pls_df['axis_angle_deg'].mean():.2f} degrees")
    print(f"angle std: {pls_df['axis_angle_deg'].std():.2f} degrees")


if __name__ == "__main__":
    main()