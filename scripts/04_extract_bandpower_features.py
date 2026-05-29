import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR, RESULTS_DIR, METRICS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

FS = 128
N_EEG_CHANNELS = 32

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

LABEL_NAMES = ["valence", "arousal", "dominance", "liking"]


def load_dat(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def compute_bandpower(signal: np.ndarray, fs: int = FS) -> dict:
    freqs, psd = welch(signal, fs=fs, nperseg=fs * 2)

    out = {}

    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)

        if not np.any(mask):
            out[band_name] = np.nan
        else:
            out[band_name] = float(np.trapezoid(psd[mask], freqs[mask]))

    return out


def extract_subject_features(path: Path) -> list[dict]:
    d = load_dat(path)

    data = d["data"]
    labels = d["labels"]

    eeg = data[:, :N_EEG_CHANNELS, :]

    rows = []

    for trial_idx in range(eeg.shape[0]):
        row = {
            "subject": path.stem,
            "trial": int(trial_idx),
            "valence": float(labels[trial_idx, 0]),
            "arousal": float(labels[trial_idx, 1]),
            "dominance": float(labels[trial_idx, 2]),
            "liking": float(labels[trial_idx, 3]),
        }

        for ch_idx in range(N_EEG_CHANNELS):
            ch_name = f"ch{ch_idx + 1:02d}"
            bp = compute_bandpower(eeg[trial_idx, ch_idx, :])

            for band_name, value in bp.items():
                row[f"{ch_name}_{band_name}"] = value

        rows.append(row)

    return rows


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if len(files) != 32:
        raise RuntimeError(f"Expected 32 subject files, found {len(files)}")

    all_rows = []

    for path in files:
        print(f"Extracting: {path.name}")
        all_rows.extend(extract_subject_features(path))

    df = pd.DataFrame(all_rows)

    feature_cols = [
        col for col in df.columns
        if col.startswith("ch") and "_" in col
    ]

    output_path = TABLES_DIR / "bandpower_features_absolute.csv"
    metadata_path = METRICS_DIR / "bandpower_feature_metadata.json"

    df.to_csv(output_path, index=False)

    metadata = {
        "source": "DEAP preprocessed Python .dat files",
        "sampling_rate_hz": FS,
        "n_subjects": int(df["subject"].nunique()),
        "n_trials": int(df.shape[0]),
        "n_eeg_channels": N_EEG_CHANNELS,
        "bands": BANDS,
        "n_feature_columns": len(feature_cols),
        "expected_feature_columns": N_EEG_CHANNELS * len(BANDS),
        "label_columns": LABEL_NAMES,
        "output_table": str(output_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved:")
    print(f"- {output_path}")
    print(f"- {metadata_path}")

    print("\nFeature table shape:")
    print(df.shape)

    print("\nFeature metadata:")
    print(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    main()