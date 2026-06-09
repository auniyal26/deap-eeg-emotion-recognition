from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"

METRICS.mkdir(parents=True, exist_ok=True)


INPUTS = [
    {
        "path": TABLES / "master_axis_audit_best_models.csv",
        "audit_family": "bandpower_representations",
    },
    {
        "path": TABLES / "asymmetry_axis_audit_best_models.csv",
        "audit_family": "asymmetry_representations",
    },
]


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_table(path, audit_family):
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    df = pd.read_csv(path)
    df["audit_family"] = audit_family
    df["source_file"] = path.name
    return df


def main():
    frames = []

    for spec in INPUTS:
        df = load_table(spec["path"], spec["audit_family"])
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)

    improvement_col = find_col(
        master,
        [
            "improvement_over_dummy_pct",
            "dummy_improvement_pct",
            "improvement_pct",
            "pct_improvement_over_dummy",
        ],
    )

    delta_col = find_col(
        master,
        [
            "improvement_over_dummy",
            "dummy_improvement",
            "delta_vs_dummy",
            "score_delta_vs_dummy",
        ],
    )

    metric_col = find_col(
        master,
        [
            "metric",
            "primary_metric",
            "score_metric",
        ],
    )

    target_col = find_col(
        master,
        [
            "target_name",
            "target",
            "target_type",
        ],
    )

    representation_col = find_col(
        master,
        [
            "representation",
            "feature_set",
            "feature_variant",
        ],
    )

    model_col = find_col(
        master,
        [
            "model",
            "model_name",
        ],
    )

    # Normalize sorting. For our metrics, higher improvement over dummy is better.
    if improvement_col:
        master = master.sort_values(improvement_col, ascending=False)
    elif delta_col:
        master = master.sort_values(delta_col, ascending=False)

    output_path = TABLES / "deap_master_audit_table.csv"
    master.to_csv(output_path, index=False)

    summary = {
        "n_rows": int(len(master)),
        "input_files": [spec["path"].name for spec in INPUTS],
        "columns": list(master.columns),
        "detected_columns": {
            "improvement_pct": improvement_col,
            "improvement_delta": delta_col,
            "metric": metric_col,
            "target": target_col,
            "representation": representation_col,
            "model": model_col,
        },
    }

    if improvement_col:
        summary["improvement_pct_summary"] = {
            "max": float(master[improvement_col].max()),
            "min": float(master[improvement_col].min()),
            "mean": float(master[improvement_col].mean()),
            "median": float(master[improvement_col].median()),
            "n_positive": int((master[improvement_col] > 0).sum()),
            "n_zero_or_negative": int((master[improvement_col] <= 0).sum()),
        }

        top_cols = [
            c for c in [
                "audit_family",
                representation_col,
                target_col,
                model_col,
                metric_col,
                improvement_col,
                delta_col,
                "dummy_score",
                "model_score",
                "mean_angle_deg",
                "std_angle_deg",
            ]
            if c is not None and c in master.columns
        ]

        summary["top_10_by_improvement_pct"] = (
            master[top_cols]
            .head(10)
            .to_dict(orient="records")
        )

    if target_col and improvement_col:
        by_target = (
            master.groupby(target_col)[improvement_col]
            .agg(["count", "max", "mean", "median"])
            .reset_index()
            .sort_values("max", ascending=False)
        )
        by_target_path = TABLES / "deap_master_audit_by_target.csv"
        by_target.to_csv(by_target_path, index=False)
        summary["by_target_path"] = str(by_target_path)

    if representation_col and improvement_col:
        by_representation = (
            master.groupby(representation_col)[improvement_col]
            .agg(["count", "max", "mean", "median"])
            .reset_index()
            .sort_values("max", ascending=False)
        )
        by_representation_path = TABLES / "deap_master_audit_by_representation.csv"
        by_representation.to_csv(by_representation_path, index=False)
        summary["by_representation_path"] = str(by_representation_path)

    summary_path = METRICS / "deap_master_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDEAP master audit table created.")
    print(f"Saved: {output_path}")
    print(f"Saved: {summary_path}")

    if target_col and improvement_col:
        print(f"Saved: {TABLES / 'deap_master_audit_by_target.csv'}")

    if representation_col and improvement_col:
        print(f"Saved: {TABLES / 'deap_master_audit_by_representation.csv'}")

    print("\nDetected columns:")
    print(json.dumps(summary["detected_columns"], indent=2))

    if improvement_col:
        print("\nImprovement summary:")
        print(json.dumps(summary["improvement_pct_summary"], indent=2))

        print("\nTop 10 rows:")
        print(master.head(10).to_string(index=False))


if __name__ == "__main__":
    main()