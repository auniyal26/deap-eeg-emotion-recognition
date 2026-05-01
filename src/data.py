import pickle
from pathlib import Path
import numpy as np


def load_deap_subject(file_path: str | Path) -> dict:
    """
    Load one DEAP preprocessed Python .dat file.

    Expected keys:
    - data: trials x channels x samples
    - labels: trials x label_dimensions
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"DEAP subject file not found: {file_path}\n"
            "Place s01.dat inside data/raw/data_preprocessed_python/"
        )

    with open(file_path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    return subject


def extract_eeg(data: np.ndarray, n_eeg_channels: int = 32) -> np.ndarray:
    """
    Extract EEG channels only.

    DEAP preprocessed format:
    trials x channels x samples
    """
    return data[:, :n_eeg_channels, :]


def make_binary_labels(labels: np.ndarray, threshold: float = 5.0) -> dict:
    """
    DEAP label order:
    labels[:, 0] = valence
    labels[:, 1] = arousal
    labels[:, 2] = dominance
    labels[:, 3] = liking

    Binary split:
    low  = rating <= threshold
    high = rating > threshold
    """
    valence = (labels[:, 0] > threshold).astype(int)
    arousal = (labels[:, 1] > threshold).astype(int)

    return {
        "valence": valence,
        "arousal": arousal,
    }