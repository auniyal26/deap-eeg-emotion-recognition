import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import SUBJECT_FILE, METRICS_DIR, EEG_CHANNELS, BINARY_THRESHOLD
from src.data import load_deap_subject, extract_eeg, make_binary_labels


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    subject = load_deap_subject(SUBJECT_FILE)

    data = subject["data"]
    labels = subject["labels"]

    eeg = extract_eeg(data, n_eeg_channels=EEG_CHANNELS)
    binary_labels = make_binary_labels(labels, threshold=BINARY_THRESHOLD)

    profile = {
        "subject_file": str(SUBJECT_FILE),
        "data_shape": list(data.shape),
        "labels_shape": list(labels.shape),
        "eeg_shape": list(eeg.shape),
        "valence_rating_min": float(np.min(labels[:, 0])),
        "valence_rating_max": float(np.max(labels[:, 0])),
        "arousal_rating_min": float(np.min(labels[:, 1])),
        "arousal_rating_max": float(np.max(labels[:, 1])),
        "valence_binary_counts": {
            "low_0": int(np.sum(binary_labels["valence"] == 0)),
            "high_1": int(np.sum(binary_labels["valence"] == 1)),
        },
        "arousal_binary_counts": {
            "low_0": int(np.sum(binary_labels["arousal"] == 0)),
            "high_1": int(np.sum(binary_labels["arousal"] == 1)),
        },
        "threshold": BINARY_THRESHOLD,
    }

    output_path = METRICS_DIR / "s01_profile.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4)

    print(json.dumps(profile, indent=4))
    print(f"\nSaved profile to: {output_path}")


if __name__ == "__main__":
    main()