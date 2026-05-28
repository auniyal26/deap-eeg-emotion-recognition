import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR, RESULTS_DIR, METRICS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

LABEL_NAMES = ["valence", "arousal", "dominance", "liking"]


def load_dat(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if len(files) != 32:
        raise RuntimeError(f"Expected 32 subject files, found {len(files)}")

    subject_rows = []
    all_labels = []

    for path in files:
        d = load_dat(path)
        data = d["data"]
        labels = d["labels"]

        all_labels.append(labels)

        row = {
            "subject": path.stem,
            "data_shape": str(tuple(data.shape)),
            "labels_shape": str(tuple(labels.shape)),
            "n_trials": int(labels.shape[0]),
            "n_channels": int(data.shape[1]),
            "n_samples": int(data.shape[2]),
        }

        for i, name in enumerate(LABEL_NAMES):
            x = labels[:, i]
            row[f"{name}_min"] = float(np.min(x))
            row[f"{name}_max"] = float(np.max(x))
            row[f"{name}_mean"] = float(np.mean(x))
            row[f"{name}_std"] = float(np.std(x))
            row[f"{name}_median"] = float(np.median(x))

        subject_rows.append(row)

    subject_df = pd.DataFrame(subject_rows)
    labels_all = np.concatenate(all_labels, axis=0)

    global_summary = {
        "n_subjects": int(len(files)),
        "n_trials_total": int(labels_all.shape[0]),
        "expected_data_shape_per_subject": [40, 40, 8064],
        "expected_label_shape_per_subject": [40, 4],
        "global_label_summary": {},
    }

    for i, name in enumerate(LABEL_NAMES):
        x = labels_all[:, i]
        global_summary["global_label_summary"][name] = {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
        }

    subject_csv = TABLES_DIR / "raw_subject_profile.csv"
    summary_json = METRICS_DIR / "raw_dataset_profile_summary.json"

    subject_df.to_csv(subject_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=4)

    print("Saved:")
    print(f"- {subject_csv}")
    print(f"- {summary_json}")

    print("\nGlobal label summary:")
    print(json.dumps(global_summary, indent=4))


if __name__ == "__main__":
    main()