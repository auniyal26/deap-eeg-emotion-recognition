import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR, RESULTS_DIR, BINARY_THRESHOLD

TABLES_DIR = RESULTS_DIR / "tables"


def load_raw_deap_labels(file_path: Path):
    with open(file_path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")
    return subject["labels"]


def assign_quadrant(valence_rating: float, arousal_rating: float, threshold: float = 5.0):
    valence_label = int(valence_rating > threshold)
    arousal_label = int(arousal_rating > threshold)

    if valence_label == 0 and arousal_label == 0:
        return "low_valence_low_arousal"
    if valence_label == 0 and arousal_label == 1:
        return "low_valence_high_arousal"
    if valence_label == 1 and arousal_label == 0:
        return "high_valence_low_arousal"
    if valence_label == 1 and arousal_label == 1:
        return "high_valence_high_arousal"

    return "unknown"


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    subject_files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if not subject_files:
        raise FileNotFoundError(f"No .dat files found in {DEAP_PYTHON_DIR}")

    for subject_file in subject_files:
        labels = load_raw_deap_labels(subject_file)

        for trial_idx in range(labels.shape[0]):
            valence_rating = float(labels[trial_idx, 0])
            arousal_rating = float(labels[trial_idx, 1])

            valence_label = int(valence_rating > BINARY_THRESHOLD)
            arousal_label = int(arousal_rating > BINARY_THRESHOLD)

            rows.append(
                {
                    "subject": subject_file.stem,
                    "trial": trial_idx,
                    "valence_rating_raw": valence_rating,
                    "arousal_rating_raw": arousal_rating,
                    "valence_label": valence_label,
                    "arousal_label": arousal_label,
                    "quadrant": assign_quadrant(
                        valence_rating,
                        arousal_rating,
                        threshold=BINARY_THRESHOLD,
                    ),
                }
            )

    df = pd.DataFrame(rows)

    quadrant_counts = (
        df["quadrant"]
        .value_counts()
        .rename_axis("quadrant")
        .reset_index(name="count")
    )
    quadrant_counts["ratio"] = quadrant_counts["count"] / len(df)

    high_high = df[
        (df["valence_rating_raw"] > BINARY_THRESHOLD)
        & (df["arousal_rating_raw"] > BINARY_THRESHOLD)
    ]

    output_trials = TABLES_DIR / "raw_label_quadrant_audit_trials.csv"
    output_counts = TABLES_DIR / "raw_label_quadrant_counts.csv"
    output_high_high = TABLES_DIR / "raw_high_valence_high_arousal_trials.csv"

    df.to_csv(output_trials, index=False)
    quadrant_counts.to_csv(output_counts, index=False)
    high_high.to_csv(output_high_high, index=False)

    print("\nRaw-label quadrant counts:")
    print(quadrant_counts.to_string(index=False))

    print("\nHigh-valence/high-arousal raw trials:")
    print(high_high.to_string(index=False))

    print(f"\nNumber of high/high trials: {len(high_high)}")
    print(f"\nSaved trial audit: {output_trials}")
    print(f"Saved quadrant counts: {output_counts}")
    print(f"Saved high/high trials: {output_high_high}")


if __name__ == "__main__":
    main()