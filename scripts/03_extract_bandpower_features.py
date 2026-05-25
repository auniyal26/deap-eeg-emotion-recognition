import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DEAP_PYTHON_DIR,
    EEG_CHANNELS,
    BINARY_THRESHOLD,
    RESULTS_DIR,
    METRICS_DIR,
)
from src.data import load_deap_subject, extract_eeg, make_binary_labels
from src.features import FREQUENCY_BANDS, extract_subject_bandpower_features


TABLES_DIR = RESULTS_DIR / "tables"
FS = 128


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    subject_files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if not subject_files:
        raise FileNotFoundError(
            f"No DEAP subject files found in {DEAP_PYTHON_DIR}. "
            "Expected files: s01.dat to s32.dat."
        )

    all_rows = []

    for subject_file in tqdm(subject_files, desc="Extracting bandpower features"):
        subject_id = subject_file.stem

        subject = load_deap_subject(subject_file)
        data = subject["data"]
        labels = subject["labels"]

        eeg = extract_eeg(data, n_eeg_channels=EEG_CHANNELS)
        binary_labels = make_binary_labels(labels, threshold=BINARY_THRESHOLD)

        subject_feature_rows = extract_subject_bandpower_features(
            eeg=eeg,
            fs=FS,
            frequency_bands=FREQUENCY_BANDS,
        )

        for trial_idx, feature_row in enumerate(subject_feature_rows):
            row = {
                "subject": subject_id,
                "trial": trial_idx,
                "valence_rating": float(labels[trial_idx, 0]),
                "arousal_rating": float(labels[trial_idx, 1]),
                "valence_label": int(binary_labels["valence"][trial_idx]),
                "arousal_label": int(binary_labels["arousal"][trial_idx]),
            }

            row.update(feature_row)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    feature_cols = [
        col for col in df.columns
        if col.startswith("ch") and "_" in col
    ]

    output_csv = TABLES_DIR / "deap_bandpower_features.csv"
    metadata_json = METRICS_DIR / "bandpower_feature_metadata.json"

    df.to_csv(output_csv, index=False)

    metadata = {
        "n_subjects": int(len(subject_files)),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_feature_columns": int(len(feature_cols)),
        "sampling_rate_hz": FS,
        "eeg_channels": EEG_CHANNELS,
        "frequency_bands": FREQUENCY_BANDS,
        "binary_threshold": BINARY_THRESHOLD,
        "expected_rows": int(len(subject_files) * 40),
        "expected_feature_columns": int(EEG_CHANNELS * len(FREQUENCY_BANDS)),
        "label_counts": {
            "valence_low_0": int(np.sum(df["valence_label"] == 0)),
            "valence_high_1": int(np.sum(df["valence_label"] == 1)),
            "arousal_low_0": int(np.sum(df["arousal_label"] == 0)),
            "arousal_high_1": int(np.sum(df["arousal_label"] == 1)),
        },
        "output_csv": str(output_csv),
    }

    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nBandpower feature extraction complete.")
    print(f"Saved feature table: {output_csv}")
    print(f"Saved metadata: {metadata_json}")
    print("\nFeature table shape:")
    print(df.shape)
    print("\nLabel counts:")
    print(metadata["label_counts"])
    print("\nFeature columns:")
    print(len(feature_cols))


if __name__ == "__main__":
    main()