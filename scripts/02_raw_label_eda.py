import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR

TABLES_DIR = RESULTS_DIR / "tables"

LABEL_NAMES = ["valence", "arousal", "dominance", "liking"]
THRESHOLD = 5.0


def load_dat(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_all_labels() -> pd.DataFrame:
    rows = []

    for path in sorted(DEAP_PYTHON_DIR.glob("s*.dat")):
        d = load_dat(path)
        labels = d["labels"]

        for trial_idx in range(labels.shape[0]):
            rows.append(
                {
                    "subject": path.stem,
                    "trial": trial_idx,
                    "valence": float(labels[trial_idx, 0]),
                    "arousal": float(labels[trial_idx, 1]),
                    "dominance": float(labels[trial_idx, 2]),
                    "liking": float(labels[trial_idx, 3]),
                }
            )

    return pd.DataFrame(rows)


def plot_label_histograms(df: pd.DataFrame) -> None:
    for label in LABEL_NAMES:
        plt.figure(figsize=(7, 4))
        plt.hist(df[label], bins=30)
        plt.xlabel(label.capitalize())
        plt.ylabel("Trial count")
        plt.title(f"DEAP raw continuous {label} distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"raw_label_histogram_{label}.png", dpi=300)
        plt.close()


def plot_va_space(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(df["valence"], df["arousal"], alpha=0.65, s=18)
    plt.axvline(THRESHOLD, linestyle="--", linewidth=1)
    plt.axhline(THRESHOLD, linestyle="--", linewidth=1)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("DEAP raw continuous valence-arousal space")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_valence_arousal_space.png", dpi=300)
    plt.close()


def subjectwise_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("subject")
        .agg(
            valence_mean=("valence", "mean"),
            valence_std=("valence", "std"),
            arousal_mean=("arousal", "mean"),
            arousal_std=("arousal", "std"),
            dominance_mean=("dominance", "mean"),
            liking_mean=("liking", "mean"),
        )
        .reset_index()
    )

    out.to_csv(TABLES_DIR / "raw_subjectwise_label_summary.csv", index=False)

    x = np.arange(len(out))

    plt.figure(figsize=(14, 6))
    plt.plot(x, out["valence_mean"], marker="o", label="Valence mean")
    plt.plot(x, out["arousal_mean"], marker="o", label="Arousal mean")
    plt.axhline(THRESHOLD, linestyle="--", linewidth=1, label="Threshold = 5")
    plt.xticks(x, out["subject"], rotation=90)
    plt.ylabel("Mean rating")
    plt.title("Subject-wise valence/arousal means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_subjectwise_valence_arousal_means.png", dpi=300)
    plt.close()

    return out


def quadrant_counts(df: pd.DataFrame) -> pd.DataFrame:
    q = df.copy()

    q["valence_bin"] = np.where(q["valence"] > THRESHOLD, "high_valence", "low_valence")
    q["arousal_bin"] = np.where(q["arousal"] > THRESHOLD, "high_arousal", "low_arousal")
    q["quadrant"] = q["valence_bin"] + "__" + q["arousal_bin"]

    expected = pd.DataFrame(
        {
            "quadrant": [
                "low_valence__low_arousal",
                "low_valence__high_arousal",
                "high_valence__low_arousal",
                "high_valence__high_arousal",
            ]
        }
    )

    counts = (
        q["quadrant"]
        .value_counts()
        .rename_axis("quadrant")
        .reset_index(name="count")
    )

    counts = expected.merge(counts, on="quadrant", how="left").fillna({"count": 0})
    counts["count"] = counts["count"].astype(int)
    counts["percentage"] = counts["count"] / counts["count"].sum()

    counts.to_csv(TABLES_DIR / "raw_threshold_quadrant_counts.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(counts["quadrant"], counts["count"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Trial count")
    plt.title("Valence-arousal quadrant counts using >5 threshold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_threshold_quadrant_counts.png", dpi=300)
    plt.close()

    return counts


def label_correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[LABEL_NAMES].corr(method="spearman")
    corr.to_csv(TABLES_DIR / "raw_label_spearman_correlation.csv")
    return corr


def extreme_trials(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label in LABEL_NAMES:
        for name, idx in [
            (f"lowest_{label}", df[label].idxmin()),
            (f"highest_{label}", df[label].idxmax()),
        ]:
            row = df.loc[idx].to_dict()
            row["selection"] = name
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "raw_extreme_label_trials.csv", index=False)
    return out


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_labels()

    label_table = TABLES_DIR / "raw_trial_labels.csv"
    df.to_csv(label_table, index=False)

    plot_label_histograms(df)
    plot_va_space(df)

    subject_summary = subjectwise_label_summary(df)
    quadrants = quadrant_counts(df)
    corr = label_correlation(df)
    extremes = extreme_trials(df)

    summary = {
        "n_trials": int(df.shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "label_summary": {},
        "quadrant_counts": quadrants.to_dict(orient="records"),
        "spearman_label_correlation": corr.to_dict(),
        "extreme_trials": extremes.to_dict(orient="records"),
    }

    for label in LABEL_NAMES:
        x = df[label].values
        summary["label_summary"][label] = {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "count_above_5": int(np.sum(x > THRESHOLD)),
            "count_below_or_equal_5": int(np.sum(x <= THRESHOLD)),
        }

    summary_path = METRICS_DIR / "raw_label_eda_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Saved:")
    print(f"- {label_table}")
    print(f"- {summary_path}")
    print(f"- {TABLES_DIR / 'raw_subjectwise_label_summary.csv'}")
    print(f"- {TABLES_DIR / 'raw_threshold_quadrant_counts.csv'}")
    print(f"- {TABLES_DIR / 'raw_label_spearman_correlation.csv'}")
    print(f"- {TABLES_DIR / 'raw_extreme_label_trials.csv'}")

    print("\nRaw label summary:")
    print(json.dumps(summary["label_summary"], indent=4))

    print("\nQuadrant counts:")
    print(quadrants.to_string(index=False))

    print("\nSpearman label correlation:")
    print(corr.to_string())

    print("\nExtreme trials:")
    print(extremes.to_string(index=False))


if __name__ == "__main__":
    main()