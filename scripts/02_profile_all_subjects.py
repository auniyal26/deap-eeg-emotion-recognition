import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DEAP_PYTHON_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    BINARY_THRESHOLD,
)
from src.data import load_deap_subject, make_binary_labels


def get_subject_id(file_path: Path) -> str:
    return file_path.stem


def profile_subject(file_path: Path) -> dict:
    subject = load_deap_subject(file_path)
    labels = subject["labels"]
    binary_labels = make_binary_labels(labels, threshold=BINARY_THRESHOLD)

    valence = binary_labels["valence"]
    arousal = binary_labels["arousal"]

    return {
        "subject": get_subject_id(file_path),
        "n_trials": int(labels.shape[0]),
        "valence_low": int(np.sum(valence == 0)),
        "valence_high": int(np.sum(valence == 1)),
        "arousal_low": int(np.sum(arousal == 0)),
        "arousal_high": int(np.sum(arousal == 1)),
        "valence_high_ratio": float(np.mean(valence == 1)),
        "arousal_high_ratio": float(np.mean(arousal == 1)),
        "valence_min": float(np.min(labels[:, 0])),
        "valence_max": float(np.max(labels[:, 0])),
        "valence_mean": float(np.mean(labels[:, 0])),
        "arousal_min": float(np.min(labels[:, 1])),
        "arousal_max": float(np.max(labels[:, 1])),
        "arousal_mean": float(np.mean(labels[:, 1])),
    }


def plot_global_distribution(df: pd.DataFrame, output_path: Path) -> None:
    valence_low = int(df["valence_low"].sum())
    valence_high = int(df["valence_high"].sum())
    arousal_low = int(df["arousal_low"].sum())
    arousal_high = int(df["arousal_high"].sum())

    x = np.arange(2)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, [valence_low, valence_high], width, label="Valence")
    plt.bar(x + width / 2, [arousal_low, arousal_high], width, label="Arousal")

    plt.xticks(x, ["Low (<=5)", "High (>5)"])
    plt.ylabel("Number of trials")
    plt.title("DEAP All Subjects — Binary Label Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_subjectwise_balance(df: pd.DataFrame, output_path: Path) -> None:
    subjects = df["subject"].tolist()
    x = np.arange(len(subjects))

    plt.figure(figsize=(14, 6))
    plt.plot(x, df["valence_high_ratio"], marker="o", linewidth=1.2, label="Valence high ratio")
    plt.plot(x, df["arousal_high_ratio"], marker="o", linewidth=1.2, label="Arousal high ratio")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Balanced reference")

    plt.xticks(x, subjects, rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("High-class ratio")
    plt.xlabel("Subject")
    plt.title("DEAP Subject-wise High-Class Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subject_files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if not subject_files:
        raise FileNotFoundError(
            f"No subject .dat files found in: {DEAP_PYTHON_DIR}\n"
            "Expected files: s01.dat to s32.dat"
        )

    rows = [profile_subject(file_path) for file_path in subject_files]
    df = pd.DataFrame(rows)

    csv_path = METRICS_DIR / "all_subject_label_profile.csv"
    summary_path = METRICS_DIR / "all_subject_label_summary.json"

    df.to_csv(csv_path, index=False)

    summary = {
        "n_subjects": int(df.shape[0]),
        "total_trials": int(df["n_trials"].sum()),
        "threshold": BINARY_THRESHOLD,
        "global_counts": {
            "valence_low": int(df["valence_low"].sum()),
            "valence_high": int(df["valence_high"].sum()),
            "arousal_low": int(df["arousal_low"].sum()),
            "arousal_high": int(df["arousal_high"].sum()),
        },
        "global_high_ratios": {
            "valence_high_ratio": float(df["valence_high"].sum() / df["n_trials"].sum()),
            "arousal_high_ratio": float(df["arousal_high"].sum() / df["n_trials"].sum()),
        },
        "subject_high_ratio_summary": {
            "valence_high_ratio_min": float(df["valence_high_ratio"].min()),
            "valence_high_ratio_max": float(df["valence_high_ratio"].max()),
            "valence_high_ratio_mean": float(df["valence_high_ratio"].mean()),
            "arousal_high_ratio_min": float(df["arousal_high_ratio"].min()),
            "arousal_high_ratio_max": float(df["arousal_high_ratio"].max()),
            "arousal_high_ratio_mean": float(df["arousal_high_ratio"].mean()),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    global_plot_path = FIGURES_DIR / "all_subject_valence_arousal_distribution.png"
    subject_plot_path = FIGURES_DIR / "subjectwise_label_balance.png"

    plot_global_distribution(df, global_plot_path)
    plot_subjectwise_balance(df, subject_plot_path)

    print(df)
    print("\nSummary:")
    print(json.dumps(summary, indent=4))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {global_plot_path}")
    print(f"Saved figure: {subject_plot_path}")


if __name__ == "__main__":
    main()