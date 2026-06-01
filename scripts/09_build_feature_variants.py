import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, METRICS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

INPUT_TABLE = TABLES_DIR / "bandpower_features_absolute.csv"

OUTPUT_TABLES = {
    "absolute": TABLES_DIR / "bandpower_features_absolute.csv",
    "log": TABLES_DIR / "bandpower_features_log.csv",
    "relative_channel": TABLES_DIR / "bandpower_features_relative_channel.csv",
    "trial_proportion": TABLES_DIR / "bandpower_features_trial_proportion.csv",
}

LABEL_COLS = ["valence", "arousal", "dominance", "liking"]
META_COLS = ["subject", "trial"] + LABEL_COLS

EPS = 1e-12


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def get_channel_name(feature_col: str) -> str:
    return feature_col.split("_")[0]


def make_log_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    # Bandpower should be non-negative, but clip defensively.
    out[feature_cols] = np.log(out[feature_cols].clip(lower=0.0) + EPS)

    return out


def make_relative_channel_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    For each trial and channel:
        relative_channel_bandpower = bandpower / sum(channel bandpowers)

    This removes per-channel absolute scale and keeps spectral composition.
    """
    out = df.copy()

    channels = sorted({get_channel_name(col) for col in feature_cols})

    for ch in channels:
        ch_cols = [col for col in feature_cols if col.startswith(f"{ch}_")]

        if len(ch_cols) == 0:
            continue

        denom = out[ch_cols].sum(axis=1) + EPS
        out[ch_cols] = out[ch_cols].div(denom, axis=0)

    return out


def make_trial_proportion_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    For each trial:
        trial_proportion = each feature / total bandpower across all channels and bands

    This removes global trial power scale.
    """
    out = df.copy()

    denom = out[feature_cols].sum(axis=1) + EPS
    out[feature_cols] = out[feature_cols].div(denom, axis=0)

    return out


def sanity_check(df: pd.DataFrame, feature_cols: list[str], feature_set: str) -> dict:
    values = df[feature_cols].values

    return {
        "feature_set": feature_set,
        "shape": list(df.shape),
        "n_feature_columns": int(len(feature_cols)),
        "feature_min": float(np.nanmin(values)),
        "feature_max": float(np.nanmax(values)),
        "feature_mean": float(np.nanmean(values)),
        "feature_std": float(np.nanstd(values)),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_TABLE.exists():
        raise FileNotFoundError(f"Missing input table: {INPUT_TABLE}")

    df = pd.read_csv(INPUT_TABLE)

    required_cols = META_COLS
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns in input table: {missing}")

    feature_cols = get_feature_columns(df)

    if len(feature_cols) != 160:
        raise RuntimeError(f"Expected 160 feature columns, found {len(feature_cols)}")

    outputs = {
        "absolute": df.copy(),
        "log": make_log_features(df, feature_cols),
        "relative_channel": make_relative_channel_features(df, feature_cols),
        "trial_proportion": make_trial_proportion_features(df, feature_cols),
    }

    summaries = []

    for feature_set, out_df in outputs.items():
        output_path = OUTPUT_TABLES[feature_set]
        out_df.to_csv(output_path, index=False)

        summary = sanity_check(out_df, feature_cols, feature_set)
        summary["output_path"] = str(output_path)
        summaries.append(summary)

        print(f"Saved {feature_set}: {output_path}")
        print(f"  Shape: {out_df.shape}")
        print(f"  Feature min/max: {summary['feature_min']:.6g} / {summary['feature_max']:.6g}")
        print(f"  NaN count: {summary['nan_count']} | Inf count: {summary['inf_count']}")

    metadata = {
        "input_table": str(INPUT_TABLE),
        "feature_variants": summaries,
        "notes": {
            "absolute": "Original Welch bandpower features.",
            "log": "log(power + epsilon), epsilon=1e-12.",
            "relative_channel": "Each channel-band feature divided by total bandpower of that channel within the trial.",
            "trial_proportion": "Each channel-band feature divided by total bandpower across all EEG channels and bands within the trial.",
        },
    }

    metadata_path = METRICS_DIR / "feature_variant_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved metadata:")
    print(f"- {metadata_path}")

    print("\nFeature variant summary:")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()