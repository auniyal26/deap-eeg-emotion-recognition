import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, FIGURES_DIR

TABLES_DIR = RESULTS_DIR / "tables"
FEATURE_TABLE = TABLES_DIR / "deap_bandpower_features.csv"


def assign_quadrant(row):
    valence = row["valence_label"]
    arousal = row["arousal_label"]

    if valence == 0 and arousal == 0:
        return "low_valence_low_arousal"
    if valence == 0 and arousal == 1:
        return "low_valence_high_arousal"
    if valence == 1 and arousal == 0:
        return "high_valence_low_arousal"
    if valence == 1 and arousal == 1:
        return "high_valence_high_arousal"

    return "unknown"


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE}\n"
            "Run scripts/03_extract_bandpower_features.py first."
        )

    df = pd.read_csv(FEATURE_TABLE)
    df["quadrant"] = df.apply(assign_quadrant, axis=1)

    quadrant_counts = (
        df["quadrant"]
        .value_counts()
        .rename_axis("quadrant")
        .reset_index(name="count")
    )

    quadrant_counts["ratio"] = quadrant_counts["count"] / len(df)

    output_csv = TABLES_DIR / "quadrant_counts.csv"
    quadrant_counts.to_csv(output_csv, index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(quadrant_counts["quadrant"], quadrant_counts["count"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of trials")
    plt.title("DEAP Valence-Arousal Quadrant Counts")
    plt.tight_layout()

    output_fig = FIGURES_DIR / "quadrant_counts.png"
    plt.savefig(output_fig, dpi=300)
    plt.close()

    print("\nQuadrant counts:")
    print(quadrant_counts.to_string(index=False))
    print(f"\nSaved table: {output_csv}")
    print(f"Saved figure: {output_fig}")


if __name__ == "__main__":
    main()