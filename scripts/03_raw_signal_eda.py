import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR

TABLES_DIR = RESULTS_DIR / "tables"

FS = 128
N_EEG_CHANNELS = 32

LABEL_NAMES = ["valence", "arousal", "dominance", "liking"]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def load_dat(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def bandpower(x: np.ndarray, fs: int = FS) -> dict:
    freqs, psd = welch(x, fs=fs, nperseg=fs * 2)

    out = {}

    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[band_name] = float(np.trapezoid(psd[mask], freqs[mask]))

    return out


def collect_raw_signal_summary() -> pd.DataFrame:
    rows = []

    files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if len(files) != 32:
        raise RuntimeError(f"Expected 32 subject files, found {len(files)}")

    for path in files:
        subject = path.stem
        d = load_dat(path)

        data = d["data"]
        labels = d["labels"]

        eeg = data[:, :N_EEG_CHANNELS, :]

        for trial_idx in range(eeg.shape[0]):
            trial = eeg[trial_idx]
            flat = trial.reshape(-1)

            row = {
                "subject": subject,
                "trial": int(trial_idx),
                "valence": float(labels[trial_idx, 0]),
                "arousal": float(labels[trial_idx, 1]),
                "dominance": float(labels[trial_idx, 2]),
                "liking": float(labels[trial_idx, 3]),
                "eeg_mean": float(np.mean(flat)),
                "eeg_std": float(np.std(flat)),
                "eeg_rms": float(np.sqrt(np.mean(flat ** 2))),
                "eeg_abs_mean": float(np.mean(np.abs(flat))),
                "eeg_min": float(np.min(flat)),
                "eeg_max": float(np.max(flat)),
                "eeg_peak_to_peak": float(np.ptp(flat)),
                "eeg_skew": float(skew(flat)),
                "eeg_kurtosis": float(kurtosis(flat)),
            }

            band_values = {band: [] for band in BANDS}

            for ch_idx in range(N_EEG_CHANNELS):
                bp = bandpower(trial[ch_idx])

                for band_name, value in bp.items():
                    band_values[band_name].append(value)

            for band_name, values in band_values.items():
                values = np.array(values)
                row[f"{band_name}_mean_power"] = float(np.mean(values))
                row[f"{band_name}_std_power"] = float(np.std(values))

            rows.append(row)

    return pd.DataFrame(rows)


def subject_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("subject")
        .agg(
            eeg_rms_mean=("eeg_rms", "mean"),
            eeg_rms_std=("eeg_rms", "std"),
            eeg_peak_to_peak_mean=("eeg_peak_to_peak", "mean"),
            eeg_kurtosis_mean=("eeg_kurtosis", "mean"),
            valence_mean=("valence", "mean"),
            arousal_mean=("arousal", "mean"),
            dominance_mean=("dominance", "mean"),
            liking_mean=("liking", "mean"),
        )
        .reset_index()
    )

    out.to_csv(TABLES_DIR / "raw_signal_subject_summary.csv", index=False)

    x = np.arange(len(out))

    plt.figure(figsize=(14, 6))
    plt.plot(x, out["eeg_rms_mean"], marker="o", label="EEG RMS mean")
    plt.xticks(x, out["subject"], rotation=90)
    plt.ylabel("Raw-loaded EEG RMS")
    plt.title("Subject-wise raw EEG RMS")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_signal_subjectwise_eeg_rms.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(x, out["eeg_peak_to_peak_mean"], marker="o", label="Peak-to-peak mean")
    plt.xticks(x, out["subject"], rotation=90)
    plt.ylabel("Peak-to-peak amplitude")
    plt.title("Subject-wise raw EEG peak-to-peak amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_signal_subjectwise_peak_to_peak.png", dpi=300)
    plt.close()

    return out


def raw_signal_label_correlations(df: pd.DataFrame) -> pd.DataFrame:
    signal_cols = [
        "eeg_mean",
        "eeg_std",
        "eeg_rms",
        "eeg_abs_mean",
        "eeg_min",
        "eeg_max",
        "eeg_peak_to_peak",
        "eeg_skew",
        "eeg_kurtosis",
    ]

    signal_cols += [
        col for col in df.columns
        if col.endswith("_mean_power") or col.endswith("_std_power")
    ]

    rows = []

    for feature in signal_cols:
        x = df[feature].values

        for label in LABEL_NAMES:
            y = df[label].values

            if np.std(x) == 0 or np.std(y) == 0:
                corr = np.nan
            else:
                corr = float(spearmanr(x, y)[0])

            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "spearman": corr,
                    "abs_spearman": abs(corr) if not np.isnan(corr) else np.nan,
                }
            )

    out = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)
    out.to_csv(TABLES_DIR / "raw_signal_label_correlations.csv", index=False)

    return out


def extreme_trials(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label in LABEL_NAMES:
        for selection_name, idx in [
            (f"lowest_{label}", df[label].idxmin()),
            (f"highest_{label}", df[label].idxmax()),
        ]:
            row = df.loc[
                idx,
                [
                    "subject",
                    "trial",
                    "valence",
                    "arousal",
                    "dominance",
                    "liking",
                    "eeg_rms",
                    "eeg_peak_to_peak",
                    "eeg_kurtosis",
                ],
            ].to_dict()
            row["selection"] = selection_name
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "raw_signal_extreme_label_trials.csv", index=False)

    return out


def pca_raw_signal_summary(df: pd.DataFrame) -> dict:
    signal_cols = [
        "eeg_std",
        "eeg_rms",
        "eeg_abs_mean",
        "eeg_peak_to_peak",
        "eeg_skew",
        "eeg_kurtosis",
    ]

    signal_cols += [
        col for col in df.columns
        if col.endswith("_mean_power")
    ]

    X = df[signal_cols].values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X)

    out = pd.DataFrame(
        {
            "subject": df["subject"].values,
            "trial": df["trial"].values,
            "pc1": z[:, 0],
            "pc2": z[:, 1],
            "valence": df["valence"].values,
            "arousal": df["arousal"].values,
            "eeg_rms": df["eeg_rms"].values,
        }
    )

    out.to_csv(TABLES_DIR / "raw_signal_pca_projection.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(out["pc1"], out["pc2"], c=pd.factorize(out["subject"])[0], s=18, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Raw signal-summary PCA colored by subject")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_signal_pca_by_subject.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(out["pc1"], out["pc2"], c=out["valence"], s=18, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Raw signal-summary PCA colored by valence")
    plt.colorbar(label="Valence")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_signal_pca_by_valence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(out["pc1"], out["pc2"], c=out["arousal"], s=18, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Raw signal-summary PCA colored by arousal")
    plt.colorbar(label="Arousal")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "raw_signal_pca_by_arousal.png", dpi=300)
    plt.close()

    return {
        "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
        "pc1_pc2_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw DEAP .dat files and computing signal summaries...")
    df = collect_raw_signal_summary()

    trial_path = TABLES_DIR / "raw_signal_trial_summary.csv"
    df.to_csv(trial_path, index=False)

    subject_summary = subject_signal_summary(df)
    corr_df = raw_signal_label_correlations(df)
    extreme_df = extreme_trials(df)
    pca_summary = pca_raw_signal_summary(df)

    summary = {
        "n_trials": int(df.shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "pca_summary": pca_summary,
        "top_signal_label_correlations": corr_df.head(30).to_dict(orient="records"),
        "extreme_trials": extreme_df.to_dict(orient="records"),
    }

    summary_path = METRICS_DIR / "raw_signal_eda_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved:")
    print(f"- {trial_path}")
    print(f"- {TABLES_DIR / 'raw_signal_subject_summary.csv'}")
    print(f"- {TABLES_DIR / 'raw_signal_label_correlations.csv'}")
    print(f"- {TABLES_DIR / 'raw_signal_extreme_label_trials.csv'}")
    print(f"- {TABLES_DIR / 'raw_signal_pca_projection.csv'}")
    print(f"- {summary_path}")

    print("\nTop raw signal-label correlations:")
    print(corr_df.head(20).to_string(index=False))

    print("\nExtreme label trials:")
    print(extreme_df.to_string(index=False))

    print("\nRaw signal PCA summary:")
    print(json.dumps(pca_summary, indent=4))

    print("\nTop subject RMS summary:")
    print(subject_summary.sort_values("eeg_rms_mean", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()