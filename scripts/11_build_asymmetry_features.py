import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RESULTS_DIR, METRICS_DIR

TABLES_DIR = RESULTS_DIR / "tables"

INPUT_TABLES = {
    "absolute": TABLES_DIR / "bandpower_features_absolute.csv",
    "log": TABLES_DIR / "bandpower_features_log.csv",
    "relative_channel": TABLES_DIR / "bandpower_features_relative_channel.csv",
    "trial_proportion": TABLES_DIR / "bandpower_features_trial_proportion.csv",
}

OUTPUT_TABLES = {
    "asymmetry_only": TABLES_DIR / "bandpower_features_asymmetry_only.csv",
    "absolute_plus_asymmetry": TABLES_DIR / "bandpower_features_absolute_plus_asymmetry.csv",
    "log_plus_asymmetry": TABLES_DIR / "bandpower_features_log_plus_asymmetry.csv",
    "relative_channel_plus_asymmetry": TABLES_DIR / "bandpower_features_relative_channel_plus_asymmetry.csv",
    "trial_proportion_plus_asymmetry": TABLES_DIR / "bandpower_features_trial_proportion_plus_asymmetry.csv",
}

LABEL_COLS = ["valence", "arousal", "dominance", "liking"]
META_COLS = ["subject", "trial"] + LABEL_COLS

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# Standard DEAP preprocessed Python EEG channel order:
# ch01 Fp1, ch02 AF3, ch03 F3, ch04 F7, ch05 FC5, ch06 FC1, ch07 C3, ch08 T7,
# ch09 CP5, ch10 CP1, ch11 P3, ch12 P7, ch13 PO3, ch14 O1, ch15 Oz, ch16 Pz,
# ch17 Fp2, ch18 AF4, ch19 Fz, ch20 F4, ch21 F8, ch22 FC6, ch23 FC2, ch24 Cz,
# ch25 C4, ch26 T8, ch27 CP6, ch28 CP2, ch29 P4, ch30 P8, ch31 PO4, ch32 O2
#
# Midline channels are intentionally excluded from left-right asymmetry:
# Fz, Cz, Pz, Oz.

CHANNEL_PAIRS = [
    ("Fp1", "ch01", "Fp2", "ch17"),
    ("AF3", "ch02", "AF4", "ch18"),
    ("F3", "ch03", "F4", "ch20"),
    ("F7", "ch04", "F8", "ch21"),
    ("FC5", "ch05", "FC6", "ch22"),
    ("FC1", "ch06", "FC2", "ch23"),
    ("C3", "ch07", "C4", "ch25"),
    ("T7", "ch08", "T8", "ch26"),
    ("CP5", "ch09", "CP6", "ch27"),
    ("CP1", "ch10", "CP2", "ch28"),
    ("P3", "ch11", "P4", "ch29"),
    ("P7", "ch12", "P8", "ch30"),
    ("PO3", "ch13", "PO4", "ch31"),
    ("O1", "ch14", "O2", "ch32"),
]

EPS = 1e-12


def get_bandpower_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("ch") and "_" in col]


def get_asymmetry_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("asym_")]


def add_asymmetry_features(df: pd.DataFrame, input_name: str) -> pd.DataFrame:
    out = df.copy()

    for left_name, left_ch, right_name, right_ch in CHANNEL_PAIRS:
        for band in BANDS:
            left_col = f"{left_ch}_{band}"
            right_col = f"{right_ch}_{band}"

            if left_col not in out.columns or right_col not in out.columns:
                raise RuntimeError(
                    f"{input_name}: missing columns for asymmetry pair: {left_col}, {right_col}"
                )

            new_col = f"asym_{right_name}_{left_name}_{band}"

            if input_name == "log":
                # log table is already log(power + eps), so right-left is a log ratio.
                out[new_col] = out[right_col] - out[left_col]
            else:
                # For non-log positive-valued features, use log-ratio.
                # right-left convention:
                # positive = right hemisphere stronger than left.
                right = np.abs(out[right_col].values) + EPS
                left = np.abs(out[left_col].values) + EPS
                out[new_col] = np.log(right) - np.log(left)

    return out


def make_asymmetry_only(df_with_asymmetry: pd.DataFrame) -> pd.DataFrame:
    asym_cols = get_asymmetry_columns(df_with_asymmetry)

    if len(asym_cols) != len(CHANNEL_PAIRS) * len(BANDS):
        raise RuntimeError(
            f"Expected {len(CHANNEL_PAIRS) * len(BANDS)} asymmetry columns, found {len(asym_cols)}"
        )

    return df_with_asymmetry[META_COLS + asym_cols].copy()


def sanity_summary(df: pd.DataFrame, feature_set: str) -> dict:
    feature_cols = get_bandpower_feature_columns(df) + get_asymmetry_columns(df)

    values = df[feature_cols].values

    return {
        "feature_set": feature_set,
        "shape": list(df.shape),
        "n_total_feature_columns": int(len(feature_cols)),
        "n_bandpower_columns": int(len(get_bandpower_feature_columns(df))),
        "n_asymmetry_columns": int(len(get_asymmetry_columns(df))),
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

    summaries = []

    asymmetry_only_written = False

    for input_name, input_path in INPUT_TABLES.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input table: {input_path}")

        df = pd.read_csv(input_path)

        missing_meta = [col for col in META_COLS if col not in df.columns]
        if missing_meta:
            raise RuntimeError(f"{input_name}: missing metadata columns: {missing_meta}")

        band_cols = get_bandpower_feature_columns(df)

        if len(band_cols) != 160:
            raise RuntimeError(f"{input_name}: expected 160 bandpower columns, found {len(band_cols)}")

        df_with_asym = add_asymmetry_features(df, input_name=input_name)

        output_key = f"{input_name}_plus_asymmetry"
        output_path = OUTPUT_TABLES[output_key]
        df_with_asym.to_csv(output_path, index=False)

        summary = sanity_summary(df_with_asym, output_key)
        summary["output_path"] = str(output_path)
        summaries.append(summary)

        print(f"Saved {output_key}: {output_path}")
        print(f"  Shape: {df_with_asym.shape}")
        print(f"  Bandpower cols: {summary['n_bandpower_columns']}")
        print(f"  Asymmetry cols: {summary['n_asymmetry_columns']}")
        print(f"  NaN count: {summary['nan_count']} | Inf count: {summary['inf_count']}")

        if input_name == "absolute" and not asymmetry_only_written:
            asym_only = make_asymmetry_only(df_with_asym)
            asym_only_path = OUTPUT_TABLES["asymmetry_only"]
            asym_only.to_csv(asym_only_path, index=False)

            summary = sanity_summary(asym_only, "asymmetry_only")
            summary["output_path"] = str(asym_only_path)
            summaries.append(summary)

            print(f"Saved asymmetry_only: {asym_only_path}")
            print(f"  Shape: {asym_only.shape}")
            print(f"  Asymmetry cols: {summary['n_asymmetry_columns']}")
            print(f"  NaN count: {summary['nan_count']} | Inf count: {summary['inf_count']}")

            asymmetry_only_written = True

    metadata = {
        "input_tables": {k: str(v) for k, v in INPUT_TABLES.items()},
        "output_tables": {k: str(v) for k, v in OUTPUT_TABLES.items()},
        "channel_pairs": CHANNEL_PAIRS,
        "bands": BANDS,
        "n_channel_pairs": len(CHANNEL_PAIRS),
        "n_bands": len(BANDS),
        "expected_asymmetry_columns": len(CHANNEL_PAIRS) * len(BANDS),
        "asymmetry_definition": {
            "log": "right_log_power - left_log_power",
            "non_log": "log(abs(right_feature)+eps) - log(abs(left_feature)+eps)",
            "sign": "positive means right channel greater than left channel",
        },
        "summaries": summaries,
    }

    metadata_path = METRICS_DIR / "asymmetry_feature_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved metadata:")
    print(f"- {metadata_path}")

    print("\nAsymmetry feature summary:")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()