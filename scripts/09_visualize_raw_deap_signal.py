import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR, FIGURES_DIR


SUBJECT_ID = "s01"
TRIAL_IDX = 0
FS = 128


def load_subject_raw(subject_id: str):
    file_path = DEAP_PYTHON_DIR / f"{subject_id}.dat"

    if not file_path.exists():
        raise FileNotFoundError(f"Subject file not found: {file_path}")

    with open(file_path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    return subject


def plot_raw_eeg_channels(data, labels, output_path):
    """
    Plot raw loaded EEG signals directly from the DEAP .dat file.

    No bandpower.
    No thresholding.
    No filtering added by our project.
    """
    trial = data[TRIAL_IDX, :32, :]
    valence = labels[TRIAL_IDX, 0]
    arousal = labels[TRIAL_IDX, 1]

    channels_to_plot = 16
    time = np.arange(trial.shape[1]) / FS

    plt.figure(figsize=(15, 10))

    offset = 0.0
    for ch in range(channels_to_plot):
        signal = trial[ch]

        # Only mean-centering for visual stacking.
        # This does not change the underlying data file.
        centered = signal - np.mean(signal)

        scale = np.std(centered)
        if scale == 0:
            scale = 1

        plt.plot(
            time,
            centered + offset,
            linewidth=0.7,
            label=f"EEG ch{ch + 1:02d}",
        )

        offset += scale * 6

    plt.title(
        f"Raw loaded DEAP EEG signal — {SUBJECT_ID}, trial {TRIAL_IDX}\n"
        f"Continuous labels: valence={valence:.3f}, arousal={arousal:.3f}"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude + visual offset")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_all_channel_overview(data, labels, output_path):
    """
    Plot all 40 loaded channels for one trial.

    First 32 are EEG.
    Remaining channels are peripheral/other physiological channels in DEAP.
    """
    trial = data[TRIAL_IDX, :, :]
    valence = labels[TRIAL_IDX, 0]
    arousal = labels[TRIAL_IDX, 1]

    time = np.arange(trial.shape[1]) / FS

    plt.figure(figsize=(15, 13))

    offset = 0.0
    for ch in range(trial.shape[0]):
        signal = trial[ch]
        centered = signal - np.mean(signal)

        scale = np.std(centered)
        if scale == 0:
            scale = 1

        plt.plot(
            time,
            centered + offset,
            linewidth=0.55,
            label=f"ch{ch + 1:02d}",
        )

        offset += scale * 5

    plt.axhline(0, linewidth=0.5)

    plt.title(
        f"Raw loaded DEAP all-channel overview — {SUBJECT_ID}, trial {TRIAL_IDX}\n"
        f"Continuous labels: valence={valence:.3f}, arousal={arousal:.3f}"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude + visual offset")
    plt.legend(fontsize=6, ncol=4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_raw_psd_example(data, labels, output_path):
    """
    PSD plot for visual sanity only.
    This does not affect modelling.
    """
    trial = data[TRIAL_IDX, :32, :]
    valence = labels[TRIAL_IDX, 0]
    arousal = labels[TRIAL_IDX, 1]

    channels_to_plot = [0, 1, 2, 3]

    plt.figure(figsize=(10, 6))

    for ch in channels_to_plot:
        signal = trial[ch]
        freqs, psd = welch(signal, fs=FS, nperseg=512)

        mask = freqs <= 50
        plt.plot(freqs[mask], psd[mask], linewidth=1.0, label=f"EEG ch{ch + 1:02d}")

    plt.title(
        f"PSD from raw loaded DEAP EEG — {SUBJECT_ID}, trial {TRIAL_IDX}\n"
        f"Continuous labels: valence={valence:.3f}, arousal={arousal:.3f}"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subject = load_subject_raw(SUBJECT_ID)

    data = subject["data"]
    labels = subject["labels"]

    print("Loaded raw DEAP subject file directly.")
    print(f"Subject: {SUBJECT_ID}")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Trial index: {TRIAL_IDX}")
    print(f"Continuous valence rating: {labels[TRIAL_IDX, 0]}")
    print(f"Continuous arousal rating: {labels[TRIAL_IDX, 1]}")
    print("\nNo thresholded labels used.")
    print("No feature table used.")
    print("No bandpower table used.")

    eeg_output = FIGURES_DIR / f"{SUBJECT_ID}_trial{TRIAL_IDX}_raw_eeg_channels.png"
    all_output = FIGURES_DIR / f"{SUBJECT_ID}_trial{TRIAL_IDX}_raw_all_channels.png"
    psd_output = FIGURES_DIR / f"{SUBJECT_ID}_trial{TRIAL_IDX}_raw_psd_example.png"

    plot_raw_eeg_channels(data, labels, eeg_output)
    plot_all_channel_overview(data, labels, all_output)
    plot_raw_psd_example(data, labels, psd_output)

    print("\nSaved figures:")
    print(f"- {eeg_output}")
    print(f"- {all_output}")
    print(f"- {psd_output}")


if __name__ == "__main__":
    main()