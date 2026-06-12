import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import json


def load_dreamer(mat_path):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return mat["DREAMER"]


def safe_array(x):
    return np.array(x).astype(float)


def subject_stats(subject):
    V = safe_array(subject.ScoreValence)
    A = safe_array(subject.ScoreArousal)
    D = safe_array(subject.ScoreDominance)

    return {
        "valence_mean": float(np.mean(V)),
        "arousal_mean": float(np.mean(A)),
        "dominance_mean": float(np.mean(D)),
        "valence_std": float(np.std(V)),
        "arousal_std": float(np.std(A)),
        "dominance_std": float(np.std(D)),
        "va_corr": float(np.corrcoef(V, A)[0, 1]),
        "vd_corr": float(np.corrcoef(V, D)[0, 1]),
        "ad_corr": float(np.corrcoef(A, D)[0, 1]),
    }


def main():
    mat_path = Path("data/raw/dreamer/DREAMER.mat")

    dreamer = load_dreamer(mat_path)
    subjects = dreamer.Data

    rows = []

    for i, subject in enumerate(subjects):
        stats = subject_stats(subject)
        stats["subject"] = i
        rows.append(stats)

    df = pd.DataFrame(rows)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    df.to_csv("results/tables/dreamer_subject_stats.csv", index=False)

    corr = df[["valence_mean", "arousal_mean", "dominance_mean"]].corr()
    corr.to_csv("results/tables/dreamer_label_correlations.csv")

    summary = {
        "n_subjects": len(df),
        "valence_mean_global": float(df.valence_mean.mean()),
        "arousal_mean_global": float(df.arousal_mean.mean()),
        "dominance_mean_global": float(df.dominance_mean.mean()),
        "mean_va_corr": float(df.va_corr.mean()),
        "mean_vd_corr": float(df.vd_corr.mean()),
        "mean_ad_corr": float(df.ad_corr.mean()),
    }

    with open("results/metrics/dreamer_dataset_profile.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()