import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import SUBJECT_FILE, FIGURES_DIR, EEG_CHANNELS, BINARY_THRESHOLD
from src.data import load_deap_subject, extract_eeg, make_binary_labels


def plot_eeg_trial(eeg, output_path):
    """
    Plot first 8 EEG channels from the first trial.
    """
    trial_idx = 0
    channels_to_plot = 8

    trial = eeg[trial_idx, :channels_to_plot, :]

    # DEAP preprocessed data has 8064 samples per trial.
    # Sampling rate is 128 Hz, so this is about 63 seconds.
    fs = 128
    time = np.arange(trial.shape[1]) / fs

    plt.figure(figsize=(14, 8))

    offset = 0
    for ch in range(channels_to_plot):
        signal = trial[ch]
        signal = signal - np.mean(signal)
        plt.plot(time, signal + offset, linewidth=0.8, label=f"EEG Ch {ch + 1}")
        offset += np.std(signal) * 6

    plt.title("DEAP Subject 01 — Trial 01 EEG Signal Snapshot")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude + channel offset")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_label_distribution(binary_labels, output_path):
    """
    Plot binary valence/arousal label counts.
    """
    valence_counts = [
        int(np.sum(binary_labels["valence"] == 0)),
        int(np.sum(binary_labels["valence"] == 1)),
    ]

    arousal_counts = [
        int(np.sum(binary_labels["arousal"] == 0)),
        int(np.sum(binary_labels["arousal"] == 1)),
    ]

    x = np.arange(2)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, valence_counts, width, label="Valence")
    plt.bar(x + width / 2, arousal_counts, width, label="Arousal")

    plt.xticks(x, ["Low (<=5)", "High (>5)"])
    plt.ylabel("Number of trials")
    plt.title("DEAP Subject 01 — Binary Label Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subject = load_deap_subject(SUBJECT_FILE)

    data = subject["data"]
    labels = subject["labels"]

    eeg = extract_eeg(data, n_eeg_channels=EEG_CHANNELS)
    binary_labels = make_binary_labels(labels, threshold=BINARY_THRESHOLD)

    eeg_plot_path = FIGURES_DIR / "s01_eeg_trial0_channels.png"
    label_plot_path = FIGURES_DIR / "s01_label_distribution.png"

    plot_eeg_trial(eeg, eeg_plot_path)
    plot_label_distribution(binary_labels, label_plot_path)

    print(f"Saved EEG plot: {eeg_plot_path}")
    print(f"Saved label distribution plot: {label_plot_path}")


if __name__ == "__main__":
    main()