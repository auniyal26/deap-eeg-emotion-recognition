import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR, RESULTS_DIR, FIGURES_DIR

TABLES_DIR = RESULTS_DIR / "tables"

FS = 128
N_EEG_CHANNELS = 32
CHANNELS_TO_PLOT = [0, 1, 2, 3]
SECONDS_TO_PLOT = 20


def load_subject_file(subject_path: Path):
    with open(subject_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def build_trial_index() -> pd.DataFrame:
    rows = []

    subject_files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if not subject_files:
        raise FileNotFoundError(f"No .dat files found in {DEAP_PYTHON_DIR}")

    for subject_file in subject_files:
        subject = load_subject_file(subject_file)
        labels = subject["labels"]

        for trial_idx in range(labels.shape[0]):
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            distance_from_center = float(np.sqrt((valence - 5) ** 2 + (arousal - 5) ** 2))

            rows.append(
                {
                    "subject": subject_file.stem,
                    "trial": trial_idx,
                    "valence_rating": valence,
                    "arousal_rating": arousal,
                    "distance_from_center": distance_from_center,
                }
            )

    return pd.DataFrame(rows)


def select_extreme_trials(index_df: pd.DataFrame) -> pd.DataFrame:
    selected = []

    def add_case(case_name, row):
        item = row.copy()
        item["case"] = case_name
        selected.append(item)

    add_case("lowest_valence", index_df.loc[index_df["valence_rating"].idxmin()])
    add_case("highest_valence", index_df.loc[index_df["valence_rating"].idxmax()])
    add_case("lowest_arousal", index_df.loc[index_df["arousal_rating"].idxmin()])
    add_case("highest_arousal", index_df.loc[index_df["arousal_rating"].idxmax()])
    add_case("closest_to_center", index_df.loc[index_df["distance_from_center"].idxmin()])
    add_case("farthest_from_center", index_df.loc[index_df["distance_from_center"].idxmax()])

    selected_df = pd.DataFrame(selected)

    # If the same trial is selected for multiple cases, keep the cases but mark duplicates.
    selected_df["trial_key"] = selected_df["subject"] + "_trial" + selected_df["trial"].astype(str)

    return selected_df


def load_trial(subject_id: str, trial_idx: int) -> tuple[np.ndarray, np.ndarray]:
    subject_path = DEAP_PYTHON_DIR / f"{subject_id}.dat"
    subject = load_subject_file(subject_path)

    data = subject["data"]
    labels = subject["labels"]

    trial_eeg = data[trial_idx, :N_EEG_CHANNELS, :]
    trial_labels = labels[trial_idx]

    return trial_eeg, trial_labels


def plot_extreme_eeg_snapshots(selected_df: pd.DataFrame, output_path: Path) -> None:
    n_cases = selected_df.shape[0]
    max_samples = SECONDS_TO_PLOT * FS

    fig, axes = plt.subplots(n_cases, 1, figsize=(15, 3.2 * n_cases), sharex=True)

    if n_cases == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, selected_df.iterrows()):
        subject_id = row["subject"]
        trial_idx = int(row["trial"])

        trial_eeg, trial_labels = load_trial(subject_id, trial_idx)

        trial_eeg = trial_eeg[:, :max_samples]
        time = np.arange(trial_eeg.shape[1]) / FS

        offset = 0.0

        for ch in CHANNELS_TO_PLOT:
            signal = trial_eeg[ch]
            centered = signal - np.mean(signal)

            scale = np.std(centered)
            if scale == 0:
                scale = 1.0

            ax.plot(
                time,
                centered + offset,
                linewidth=0.75,
                label=f"ch{ch + 1:02d}",
            )

            offset += scale * 6

        ax.set_title(
            f"{row['case']} | {subject_id}, trial {trial_idx} | "
            f"V={row['valence_rating']:.2f}, A={row['arousal_rating']:.2f}"
        )
        ax.set_ylabel("Amplitude + offset")
        ax.legend(fontsize=7, ncol=len(CHANNELS_TO_PLOT), loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Raw-loaded DEAP EEG snapshots for extreme continuous ratings", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_extreme_psd(selected_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 7))

    for _, row in selected_df.iterrows():
        subject_id = row["subject"]
        trial_idx = int(row["trial"])

        trial_eeg, _ = load_trial(subject_id, trial_idx)

        # Average PSD across first 32 EEG channels for a compact trial-level view.
        psds = []
        for ch in range(N_EEG_CHANNELS):
            freqs, psd = welch(trial_eeg[ch], fs=FS, nperseg=512)
            psds.append(psd)

        mean_psd = np.mean(np.vstack(psds), axis=0)
        mask = freqs <= 50

        plt.plot(
            freqs[mask],
            mean_psd[mask],
            linewidth=1.3,
            label=f"{row['case']} | V={row['valence_rating']:.2f}, A={row['arousal_rating']:.2f}",
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean PSD across EEG channels")
    plt.title("Mean EEG PSD for extreme continuous-rating trials")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    index_df = build_trial_index()
    selected_df = select_extreme_trials(index_df)

    output_csv = TABLES_DIR / "raw_extreme_trial_selection.csv"
    selected_df.to_csv(output_csv, index=False)

    eeg_fig = FIGURES_DIR / "raw_extreme_trials_eeg_snapshots.png"
    psd_fig = FIGURES_DIR / "raw_extreme_trials_psd.png"

    plot_extreme_eeg_snapshots(selected_df, eeg_fig)
    plot_extreme_psd(selected_df, psd_fig)

    print("Selected extreme continuous-rating trials:")
    print(
        selected_df[
            [
                "case",
                "subject",
                "trial",
                "valence_rating",
                "arousal_rating",
                "distance_from_center",
                "trial_key",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"- {output_csv}")
    print(f"- {eeg_fig}")
    print(f"- {psd_fig}")

    print("\nNo thresholded labels used.")
    print("No feature table used.")
    print("Signals loaded directly from DEAP .dat files.")


if __name__ == "__main__":
    main()