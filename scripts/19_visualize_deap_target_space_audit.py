import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None):
        if iterable is None:
            return range(total or 0)
        print(desc or "Progress")
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = REPO_ROOT / "results" / "tables"
FIG_DIR = REPO_ROOT / "results" / "final_figures"
METRIC_DIR = REPO_ROOT / "results" / "metrics"

OUT_MANIFEST = FIG_DIR / "deap_figure_manifest.csv"
OUT_METADATA = METRIC_DIR / "deap_final_figure_metadata.json"

INK = "#1f1f1f"
MUTED = "#5f5f5f"
GRID = "#e8e8e8"
FRAME = "#bdbdbd"
WHITE = "#ffffff"

CUBE = sns.cubehelix_palette(
    14,
    start=2,
    rot=0,
    dark=0.16,
    light=0.94,
    reverse=True,
)

CUBE_HEX = CUBE.as_hex()
ACCENT_DARK = CUBE[1]
ACCENT = CUBE[3]
ACCENT_MID = CUBE[6]
ACCENT_LIGHT = CUBE[9]
ACCENT_PALE = CUBE[11]

HEATMAP = sns.cubehelix_palette(
    256,
    start=2,
    rot=0,
    dark=0.16,
    light=0.94,
    reverse=True,
    as_cmap=True,
)


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.edgecolor": FRAME,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def read_required_csv(filename):
    path = TABLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path), str(path.relative_to(REPO_ROOT))


def save_figure(fig, name, manifest_rows, title, source_files):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)

    manifest_rows.append(
        {
            "figure_name": name,
            "title": title,
            "png_path": str(path.relative_to(REPO_ROOT)),
            "source_files": ";".join(source_files),
        }
    )


def style_axis(ax, grid_axis=None):
    ax.tick_params(axis="both", which="major", length=3, width=0.6, color=FRAME)
    ax.spines["left"].set_color(FRAME)
    ax.spines["bottom"].set_color(FRAME)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def gradient_colors(n, start=1, stop=11):
    if n <= 0:
        return []
    if n == 1:
        return [CUBE[start]]
    idx = np.linspace(start, stop, n)
    idx = np.clip(np.round(idx).astype(int), 0, len(CUBE) - 1)
    return [CUBE[int(i)] for i in idx]


def percent_formatter(x, pos):
    return f"{x:.0f}%"


def clean_label(text):
    text = str(text)
    replacements = {
        "direct_valence_arousal": "direct V/A",
        "direct_valence_arousal_z": "direct V/A",
        "direct_4d_vadl_z": "direct V/A/D/L",
        "label_pca_axis": "V/A PCA",
        "label_pca_va_axis": "V/A PCA",
        "label_pca_4d_axis1": "VADL PCA-1",
        "label_pca_4d_axis2": "VADL PCA-2",
        "pls_eeg_aligned_axis": "V/A PLS",
        "pls_va_eeg_aligned_axis": "V/A PLS",
        "pls_4d_eeg_aligned_axis1": "VADL PLS-1",
        "positive_diagonal_zv_plus_za": "zV + zA",
        "anti_diagonal_zv_minus_za": "zV − zA",
        "positive_self_report_zv_plus_zl": "zV + zL",
        "activation_control_za_plus_zd": "zA + zD",
        "radius_4d": "VADL radius",
        "z_valence": "z-valence",
        "z_arousal": "z-arousal",
        "z_dominance": "z-dominance",
        "z_liking": "z-liking",
        "relative_channel_plus_asymmetry": "relative + asymmetry",
        "trial_proportion_plus_asymmetry": "trial proportion + asymmetry",
        "absolute_plus_asymmetry": "absolute + asymmetry",
        "log_plus_asymmetry": "log + asymmetry",
        "relative_channel": "relative",
        "trial_proportion": "trial proportion",
        "asymmetry_only": "asymmetry only",
        "absolute": "absolute",
        "log": "log",
    }
    return replacements.get(text, text.replace("_", " "))


def normalize_axis_key(text):
    text = str(text)
    aliases = {
        "label_pca_axis": "label_pca_va_axis",
        "pls_eeg_aligned_axis": "pls_va_eeg_aligned_axis",
        "pls_4d_axis1": "pls_4d_eeg_aligned_axis1",
        "label_pca_4d_axis1": "label_pca_4d_axis1",
        "label_pca_va_axis": "label_pca_va_axis",
        "pls_va_eeg_aligned_axis": "pls_va_eeg_aligned_axis",
        "pls_4d_eeg_aligned_axis1": "pls_4d_eeg_aligned_axis1",
    }
    return aliases.get(text, text)


def compact_target_label(text):
    text = clean_label(text)
    replacements = {
        "relative + asymmetry": "rel+asym",
        "trial proportion + asymmetry": "trial+asym",
        "asymmetry only": "asym",
    }
    return replacements.get(text, text)


def plot_va_sensitivity(manifest_rows):
    df, source = read_required_csv("deap_label_space_sensitivity_va_geometry.csv")

    condition_order = [
        "clean",
        "bad_mirror_9_minus",
        "proper_reverse_10_minus",
        "within_subject_trial_pair_shuffle",
        "across_subject_trial_pair_shuffle",
        "within_subject_independent_va_shuffle",
        "across_subject_independent_va_shuffle",
    ]

    labels = {
        "clean": "Clean",
        "bad_mirror_9_minus": "9 − V/A",
        "proper_reverse_10_minus": "10 − V/A",
        "within_subject_trial_pair_shuffle": "Within\npair shuffle",
        "across_subject_trial_pair_shuffle": "Global\npair shuffle",
        "within_subject_independent_va_shuffle": "Within\nindependent shuffle",
        "across_subject_independent_va_shuffle": "Global\nindependent shuffle",
    }

    quad_cols = ["low_v_low_a", "low_v_high_a", "high_v_low_a", "high_v_high_a"]
    quad_names = ["Low V / low A", "Low V / high A", "High V / low A", "High V / high A"]

    df = df[df["condition"].isin(condition_order)].copy()
    df["condition"] = pd.Categorical(df["condition"], categories=condition_order, ordered=True)
    df = df.sort_values("condition")

    x = np.arange(len(df))
    counts = df[quad_cols].to_numpy(dtype=float)
    bottoms = np.zeros(len(df))
    colors = gradient_colors(len(quad_cols), 2, 10)

    fig, ax = plt.subplots(figsize=(7.4, 4.1))

    for idx, col in enumerate(quad_cols):
        ax.bar(
            x,
            counts[:, idx],
            bottom=bottoms,
            width=0.72,
            color=colors[idx],
            edgecolor=WHITE,
            linewidth=0.7,
            label=quad_names[idx],
        )
        bottoms += counts[:, idx]

    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in df["condition"].astype(str)])
    ax.set_ylabel("Trials")
    ax.set_title("DEAP valence-arousal quadrant structure under label perturbations", loc="left", pad=10)
    ax.set_ylim(0, max(bottoms) * 1.08)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), handlelength=1.4, columnspacing=1.4)
    style_axis(ax, "y")

    save_figure(
        fig,
        "deap_va_sensitivity_quadrants",
        manifest_rows,
        "DEAP V/A target-space sensitivity",
        [source],
    )


def plot_4d_pca_loadings(manifest_rows):
    loadings, source_loadings = read_required_csv("deap_label_space_4d_pca_loadings.csv")
    variance, source_variance = read_required_csv("deap_label_space_4d_pca_variance.csv")

    order = ["valence", "arousal", "dominance", "liking"]
    label_map = {"valence": "Valence", "arousal": "Arousal", "dominance": "Dominance", "liking": "Liking"}

    pc1 = loadings[loadings["component"] == "PC1"].copy()
    pc1["label"] = pd.Categorical(pc1["label"], categories=order, ordered=True)
    pc1 = pc1.sort_values("label")
    pc1["display"] = pc1["label"].astype(str).map(label_map)

    ev = float(variance[variance["component"] == "PC1"]["explained_variance_ratio"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    ax.bar(
        pc1["display"],
        pc1["loading"],
        color=gradient_colors(len(pc1), 2, 9),
        edgecolor=WHITE,
        linewidth=0.7,
        width=0.62,
    )

    ax.axhline(0, color=FRAME, linewidth=0.8)
    ax.set_ylabel("Loading")
    ax.set_title(f"Dominant DEAP VADL self-report axis\nPC1 explained variance = {ev:.3f}", loc="left", pad=9)
    ax.set_ylim(min(0, pc1["loading"].min() - 0.08), pc1["loading"].max() + 0.1)
    style_axis(ax, "y")

    save_figure(
        fig,
        "deap_4d_pca_loadings",
        manifest_rows,
        "DEAP 4D PCA loadings",
        [source_loadings, source_variance],
    )


def plot_master_heatmap(manifest_rows):
    df, source = read_required_csv("deap_master_audit_table.csv")

    improvement_col = "improvement_over_dummy_pct"
    if improvement_col not in df.columns:
        improvement_col = "improvement_pct"

    df = df.copy()
    df["target_label"] = df["target_name"].map(clean_label)
    df["representation_label"] = df["representation"].map(clean_label)
    df["improvement_percent"] = pd.to_numeric(df[improvement_col], errors="coerce") * 100.0

    pivot = df.pivot_table(
        index="target_label",
        columns="representation_label",
        values="improvement_percent",
        aggfunc="max",
    )

    target_order = [
        "z-valence",
        "z-arousal",
        "direct V/A",
        "zV + zA",
        "zV − zA",
        "V/A PCA",
        "V/A PLS",
        "z-dominance",
        "z-liking",
        "zV + zL",
        "zA + zD",
        "VADL PCA-1",
        "VADL PLS-1",
        "VADL radius",
        "direct V/A/D/L",
    ]

    col_order = [
        "absolute",
        "log",
        "relative",
        "trial proportion",
        "asymmetry only",
        "absolute + asymmetry",
        "log + asymmetry",
        "relative + asymmetry",
        "trial proportion + asymmetry",
    ]

    existing_targets = [x for x in target_order if x in pivot.index]
    existing_cols = [x for x in col_order if x in pivot.columns]
    remaining_targets = [x for x in pivot.index if x not in existing_targets]
    remaining_cols = [x for x in pivot.columns if x not in existing_cols]

    pivot = pivot.loc[existing_targets + remaining_targets, existing_cols + remaining_cols]

    matrix = pivot.to_numpy(dtype=float)
    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.nanmin(finite)) if len(finite) else -1.0
    vmax = 0.0

    fig_width = max(7.8, 0.86 * len(pivot.columns))
    fig_height = max(5.4, 0.36 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(matrix, aspect="auto", cmap=HEATMAP, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("EEG representation")
    ax.set_ylabel("Target formulation")
    ax.set_title("Best observed DEAP improvement over fold-matched dummy baseline", loc="left", pad=10)

    best_value = np.nanmax(finite) if len(finite) else np.nan

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value) and (abs(value) >= 1.0 or np.isclose(value, best_value)):
                color = WHITE if value < (vmin * 0.55) else INK
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.4, color=color)

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color=WHITE, linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    style_axis(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Improvement over dummy (%)")
    cbar.outline.set_edgecolor(FRAME)
    cbar.outline.set_linewidth(0.6)

    save_figure(
        fig,
        "deap_master_improvement_heatmap",
        manifest_rows,
        "DEAP master improvement-over-dummy heatmap",
        [source],
    )


def plot_split_optimism(manifest_rows):
    df, source = read_required_csv("deap_split_optimism_summary.csv")

    required = ["representation", "protocol", "target_name", "model", "improvement_over_dummy_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in split optimism summary: {missing}")

    df = df[df["model"].isin(["random_forest"])].copy()
    df = df[
        df["target_name"].isin(
            [
                "direct_valence_arousal",
                "z_valence",
                "positive_diagonal_zv_plus_za",
                "label_pca_axis",
                "pls_eeg_aligned_axis",
            ]
        )
    ].copy()

    pivot = df.pivot_table(
        index=["representation", "target_name", "model"],
        columns="protocol",
        values="improvement_over_dummy_pct",
        aggfunc="mean",
    ).reset_index()

    if "random_split" not in pivot.columns or "groupkfold_subject" not in pivot.columns:
        raise ValueError(f"Expected protocol columns not found. Found: {list(pivot.columns)}")

    pivot["random_percent"] = pivot["random_split"] * 100.0
    pivot["group_percent"] = pivot["groupkfold_subject"] * 100.0
    pivot["optimism_gap"] = pivot["random_percent"] - pivot["group_percent"]
    pivot["label"] = pivot["representation"].map(clean_label) + "\n" + pivot["target_name"].map(clean_label)
    pivot = pivot.sort_values("optimism_gap", ascending=True).tail(10)

    y = np.arange(len(pivot))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    ax.hlines(y, pivot["group_percent"], pivot["random_percent"], color=ACCENT_PALE, linewidth=2.1)
    ax.scatter(pivot["group_percent"], y, s=32, color=ACCENT_MID, label="Subject GroupKFold", zorder=3)
    ax.scatter(pivot["random_percent"], y, s=32, color=ACCENT_DARK, label="Random split", zorder=3)
    ax.axvline(0, color=FRAME, linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(pivot["label"])
    ax.set_xlabel("Improvement over dummy (%)")
    ax.set_title("Apparent DEAP performance gain under random trial splitting", loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    style_axis(ax, "x")

    save_figure(
        fig,
        "deap_split_optimism",
        manifest_rows,
        "DEAP random split optimism",
        [source],
    )


def plot_loso_summary(manifest_rows):
    df, source = read_required_csv("deap_loso_axis_audit_summary.csv")

    df = df.copy()
    df["label"] = df["representation"].map(clean_label) + "\n" + df["target_name"].map(clean_label)
    df["improvement_percent"] = pd.to_numeric(df["improvement_over_dummy_pct"], errors="coerce") * 100.0
    df = df.sort_values("improvement_percent", ascending=True)

    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.4, 5.7))

    ax.barh(
        y,
        df["improvement_percent"],
        color=gradient_colors(len(df), 2, 10),
        edgecolor=WHITE,
        linewidth=0.6,
        height=0.64,
    )

    ax.axvline(0, color=FRAME, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Improvement over dummy (%)")
    ax.set_title("DEAP LOSO validation does not recover positive target predictability", loc="left", pad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    style_axis(ax, "x")

    save_figure(
        fig,
        "deap_loso_summary",
        manifest_rows,
        "DEAP LOSO improvement-over-dummy summary",
        [source],
    )


def plot_axis_stability(manifest_rows):
    df, source = read_required_csv("deap_loso_axis_stability.csv")
    gkf_meta_path = METRIC_DIR / "deap_label_space_sensitivity_4d_metadata.json"

    metric_col = "mean_angle_to_mean_axis_deg"

    df = df.copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df[np.isfinite(df[metric_col])].copy()
    df["label"] = df["representation"].map(clean_label) + "\n" + df["axis_name"].map(clean_label)
    df = df.sort_values(metric_col, ascending=True)

    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.barh(
        y,
        df[metric_col],
        color=gradient_colors(len(df), 2, 10),
        edgecolor=WHITE,
        linewidth=0.6,
        height=0.64,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Mean angle to mean axis (degrees)")
    ax.set_title("DEAP target-axis stability under LOSO", loc="left", pad=10)
    style_axis(ax, "x")

    source_files = [source]
    if gkf_meta_path.exists():
        source_files.append(str(gkf_meta_path.relative_to(REPO_ROOT)))

    save_figure(
        fig,
        "deap_axis_stability",
        manifest_rows,
        "DEAP axis stability summary",
        source_files,
    )


def plot_stability_predictability_map(manifest_rows):
    stability, source_stability = read_required_csv("deap_loso_axis_stability.csv")
    summary, source_summary = read_required_csv("deap_loso_axis_audit_summary.csv")

    metric_col = "mean_angle_to_mean_axis_deg"

    stability = stability.copy()
    summary = summary.copy()

    stability[metric_col] = pd.to_numeric(stability[metric_col], errors="coerce")
    stability["axis_key"] = stability["axis_name"].map(normalize_axis_key)
    summary["axis_key"] = summary["target_name"].map(normalize_axis_key)
    summary["improvement_percent"] = pd.to_numeric(summary["improvement_over_dummy_pct"], errors="coerce") * 100.0

    keep = [
        "label_pca_va_axis",
        "pls_va_eeg_aligned_axis",
        "label_pca_4d_axis1",
        "pls_4d_eeg_aligned_axis1",
    ]

    stability = stability[stability["axis_key"].isin(keep)].copy()
    summary = summary[summary["axis_key"].isin(keep)].copy()

    merged = stability.merge(
        summary,
        on=["representation", "axis_key"],
        how="inner",
        suffixes=("_stability", "_summary"),
    )

    merged = merged[np.isfinite(merged[metric_col]) & np.isfinite(merged["improvement_percent"])].copy()
    merged["label"] = merged["representation"].map(compact_target_label) + " | " + merged["axis_key"].map(clean_label)

    if merged.empty:
        raise ValueError("No matching rows found for stability-predictability map")

    merged = merged.sort_values(metric_col)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    colors = gradient_colors(len(merged), 2, 10)
    sizes = np.interp(
        merged[metric_col].to_numpy(dtype=float),
        (merged[metric_col].min(), merged[metric_col].max()),
        (72, 160),
    )

    ax.scatter(
        merged[metric_col],
        merged["improvement_percent"],
        s=sizes,
        color=colors,
        edgecolor=WHITE,
        linewidth=0.7,
        alpha=0.95,
        zorder=3,
    )

    ax.axhline(0, color=FRAME, linewidth=0.9)
    ax.axvline(5, color=GRID, linewidth=0.8)
    ax.set_xlabel("Axis instability under LOSO\nmean angle to mean axis, degrees")
    ax.set_ylabel("Improvement over dummy (%)")
    ax.set_title("Stable target geometry does not imply EEG predictability", loc="left", pad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    style_axis(ax, "both")

    priority = [
        "relative + asymmetry | VADL PCA-1",
        "relative + asymmetry | VADL PLS-1",
        "relative | V/A PCA",
        "relative | V/A PLS",
    ]

    labeled = set()

    for _, row in merged.iterrows():
        if row["label"] in priority:
            ax.text(
                row[metric_col] + 0.7,
                row["improvement_percent"],
                row["label"],
                fontsize=7.2,
                va="center",
                color=INK,
            )
            labeled.add(row["label"])

    best = merged.loc[merged["improvement_percent"].idxmax()]
    if best["label"] not in labeled:
        ax.text(
            best[metric_col] + 0.7,
            best["improvement_percent"],
            "best observed",
            fontsize=7.2,
            va="center",
            color=INK,
        )

    save_figure(
        fig,
        "deap_stability_predictability_map",
        manifest_rows,
        "DEAP stability versus EEG predictability map",
        [source_stability, source_summary],
    )


def plot_4d_pca_compass(manifest_rows):
    loadings, source_loadings = read_required_csv("deap_label_space_4d_pca_loadings.csv")
    variance, source_variance = read_required_csv("deap_label_space_4d_pca_variance.csv")

    order = ["valence", "arousal", "dominance", "liking"]
    display = {
        "valence": "Valence",
        "arousal": "Arousal",
        "dominance": "Dominance",
        "liking": "Liking",
    }

    components = ["PC1", "PC2", "PC3"]

    wide = loadings[loadings["component"].isin(components)].copy()
    wide = wide.pivot(index="label", columns="component", values="loading").reset_index()
    wide["label"] = pd.Categorical(wide["label"], categories=order, ordered=True)
    wide = wide.sort_values("label")

    ev = {
        row["component"]: float(row["explained_variance_ratio"])
        for _, row in variance[variance["component"].isin(components)].iterrows()
    }

    points = wide[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    labels = wide["label"].astype(str).to_list()
    colors = gradient_colors(len(points), 2, 10)

    views = [
        ("View A", "PC1-forward", 62, 34),
        ("View B", "PC2/PC3 separation", 112, 30),
        ("View C", "PC1/PC3 oblique", 28, 42),
        ("View D", "reverse view", 154, 36),
    ]

    lim = 0.82

    def rotation_matrix(yaw_deg, pitch_deg):
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)

        ry = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ]
        )

        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )

        return rx @ ry

    def make_projector(rot):
        def project(p):
            q = rot @ np.asarray(p, dtype=float)
            scale = 1.0 / (1.34 - 0.22 * q[2])
            return np.array([q[0] * scale, q[1] * scale])
        return project

    def draw_plane(ax, project, vertices, color, alpha):
        pts = np.array([project(v) for v in vertices])
        ax.fill(
            pts[:, 0],
            pts[:, 1],
            color=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=0.65,
            zorder=0,
        )

    def draw_panel(ax, view_name, view_description, yaw_deg, pitch_deg):
        rot = rotation_matrix(yaw_deg, pitch_deg)
        project = make_projector(rot)

        origin = project([0, 0, 0])
        projected = np.array([project(p) for p in points])

        plane_xy = [
            [-lim, -lim, 0],
            [lim, -lim, 0],
            [lim, lim, 0],
            [-lim, lim, 0],
        ]

        plane_xz = [
            [-lim, 0, -lim],
            [lim, 0, -lim],
            [lim, 0, lim],
            [-lim, 0, lim],
        ]

        plane_yz = [
            [0, -lim, -lim],
            [0, lim, -lim],
            [0, lim, lim],
            [0, -lim, lim],
        ]

        draw_plane(ax, project, plane_xy, ACCENT_PALE, 0.060)
        draw_plane(ax, project, plane_xz, ACCENT_LIGHT, 0.045)
        draw_plane(ax, project, plane_yz, ACCENT_MID, 0.026)

        axes_positive = {
            "PC1": np.array([lim, 0, 0]),
            "PC2": np.array([0, lim, 0]),
            "PC3": np.array([0, 0, lim]),
        }

        axes_negative = {
            "PC1": np.array([-lim, 0, 0]),
            "PC2": np.array([0, -lim, 0]),
            "PC3": np.array([0, 0, -lim]),
        }

        axis_labels = {
            "PC1": f"PC1 {ev.get('PC1', np.nan):.2f}",
            "PC2": f"PC2 {ev.get('PC2', np.nan):.2f}",
            "PC3": f"PC3 {ev.get('PC3', np.nan):.2f}",
        }

        for name in ["PC1", "PC2", "PC3"]:
            a = project(axes_negative[name])
            b = project(axes_positive[name])
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="#cfcfcf",
                linewidth=0.85,
                zorder=1,
            )

            direction = b - origin
            norm = np.linalg.norm(direction)
            if norm > 0:
                label_pos = b + direction / norm * 0.045
                ax.text(
                    label_pos[0],
                    label_pos[1],
                    axis_labels[name],
                    fontsize=6.0,
                    color=MUTED,
                    ha="center",
                    va="center",
                    zorder=4,
                )

        depth = (rot @ points.T).T[:, 2]
        draw_order = np.argsort(depth)

        for idx in draw_order:
            p2 = projected[idx]
            label = labels[idx]
            color = colors[idx]

            ax.plot(
                [origin[0], p2[0]],
                [origin[1], p2[1]],
                color=color,
                linewidth=2.6,
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )

            direction = p2 - origin
            norm = np.linalg.norm(direction)

            if norm > 0:
                unit = direction / norm
                perp = np.array([-unit[1], unit[0]])
                tip = p2
                left = p2 - unit * 0.044 + perp * 0.021
                right = p2 - unit * 0.044 - perp * 0.021

                ax.fill(
                    [tip[0], left[0], right[0]],
                    [tip[1], left[1], right[1]],
                    color=color,
                    alpha=0.98,
                    zorder=4,
                )

                label_pos = p2 + unit * 0.060

                ax.text(
                    label_pos[0],
                    label_pos[1],
                    display[label],
                    fontsize=7.0,
                    color=INK,
                    ha="center",
                    va="center",
                    zorder=6,
                )

            ax.scatter(
                [p2[0]],
                [p2[1]],
                s=60,
                color=color,
                edgecolor=WHITE,
                linewidth=0.9,
                zorder=5,
            )

        ax.scatter(
            [origin[0]],
            [origin[1]],
            s=46,
            color=ACCENT_DARK,
            edgecolor=WHITE,
            linewidth=0.8,
            zorder=6,
        )

        all_axis_points = np.array(
            [project(v) for v in list(axes_positive.values()) + list(axes_negative.values())]
        )
        all_xy = np.vstack([projected, all_axis_points, origin.reshape(1, 2)])

        xmin, ymin = all_xy.min(axis=0)
        xmax, ymax = all_xy.max(axis=0)

        pad_x = (xmax - xmin) * 0.20
        pad_y = (ymax - ymin) * 0.24

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ax.text(
            0.02,
            0.98,
            view_name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            color=INK,
        )

        ax.text(
            0.02,
            0.925,
            view_description,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=MUTED,
        )

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.7))
    fig.patch.set_facecolor(WHITE)

    for ax, (view_name, view_description, yaw_deg, pitch_deg) in zip(axes.ravel(), views):
        ax.set_facecolor(WHITE)
        draw_panel(ax, view_name, view_description, yaw_deg, pitch_deg)

    fig.suptitle(
        "DEAP VADL self-report vectors in PCA loading space",
        x=0.03,
        y=0.985,
        ha="left",
        fontsize=10.8,
        color=INK,
    )

    fig.text(
        0.03,
        0.947,
        "Each arrow is one rating dimension projected into PC1-PC3 loading space; shaded planes provide orientation.",
        ha="left",
        va="top",
        fontsize=7.3,
        color=MUTED,
    )

    table_rows = []
    for label, point in zip(labels, points):
        table_rows.append(
            f"{display[label]} ({point[0]:+.2f}, {point[1]:+.2f}, {point[2]:+.2f})"
        )

    fig.text(
        0.03,
        0.032,
        "Coordinates as PC1, PC2, PC3 loadings:",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=INK,
    )

    fig.text(
        0.03,
        0.014,
        "   ·   ".join(table_rows),
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=MUTED,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        top=0.895,
        bottom=0.090,
        wspace=0.08,
        hspace=0.12,
    )

    save_figure(
        fig,
        "deap_4d_pca_compass",
        manifest_rows,
        "DEAP VADL PCA multi-view vector geometry",
        [source_loadings, source_variance],
    )


def plot_audit_landscape(manifest_rows):
    master, source_master = read_required_csv("deap_master_audit_table.csv")
    loso, source_loso = read_required_csv("deap_loso_axis_audit_summary.csv")
    fourd, source_4d = read_required_csv("deap_label_space_4d_summary.csv")

    rows = []

    for _, row in master.iterrows():
        rows.append(
            {
                "source": "GroupKFold V/A audit",
                "target": clean_label(row["target_name"]),
                "representation": clean_label(row["representation"]),
                "improvement": float(row["improvement_over_dummy_pct"]) * 100.0,
            }
        )

    for _, row in fourd.iterrows():
        rows.append(
            {
                "source": "GroupKFold VADL audit",
                "target": clean_label(row["target_name"]),
                "representation": "relative + asymmetry",
                "improvement": float(row["improvement_over_dummy_pct"]) * 100.0,
            }
        )

    for _, row in loso.iterrows():
        rows.append(
            {
                "source": "LOSO selected audit",
                "target": clean_label(row["target_name"]),
                "representation": clean_label(row["representation"]),
                "improvement": float(row["improvement_over_dummy_pct"]) * 100.0,
            }
        )

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["improvement"])].copy()
    df = df.sort_values("improvement", ascending=True).reset_index(drop=True)

    x = np.arange(len(df))
    colors = gradient_colors(len(df), 2, 11)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    ax.scatter(
        x,
        df["improvement"],
        s=34,
        color=colors,
        edgecolor=WHITE,
        linewidth=0.45,
        alpha=0.95,
        zorder=3,
    )

    ax.fill_between(
        x,
        df["improvement"],
        0,
        where=df["improvement"] <= 0,
        color=ACCENT_PALE,
        alpha=0.24,
        linewidth=0,
    )

    ax.axhline(0, color=FRAME, linewidth=0.9)
    ax.set_xlabel("Audited target-feature-model summaries sorted by improvement")
    ax.set_ylabel("Improvement over dummy (%)")
    ax.set_title("DEAP audit landscape: all audited summaries remain below dummy", loc="left", pad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.set_xticks([])

    best = df.iloc[df["improvement"].idxmax()]
    worst = df.iloc[df["improvement"].idxmin()]

    ax.text(
        len(df) - 1,
        best["improvement"] + 0.35,
        f"best: {best['target']}",
        fontsize=7.2,
        ha="right",
        va="bottom",
        color=INK,
    )

    ax.text(
        0,
        worst["improvement"] - 0.35,
        f"worst: {worst['target']}",
        fontsize=7.2,
        ha="left",
        va="top",
        color=INK,
    )

    style_axis(ax, "y")

    save_figure(
        fig,
        "deap_audit_landscape",
        manifest_rows,
        "DEAP audit landscape",
        [source_master, source_4d, source_loso],
    )


def main():
    apply_style()
    ensure_dirs()

    manifest_rows = []

    tasks = [
        plot_va_sensitivity,
        plot_4d_pca_loadings,
        plot_4d_pca_compass,
        plot_master_heatmap,
        plot_split_optimism,
        plot_loso_summary,
        plot_axis_stability,
        plot_stability_predictability_map,
        plot_audit_landscape,
    ]

    for task in tqdm(tasks, desc="Generating DEAP publication figures"):
        task(manifest_rows)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_MANIFEST, index=False)

    metadata = {
        "script": "19_visualize_deap_target_space_audit.py",
        "figure_dir": str(FIG_DIR.relative_to(REPO_ROOT)),
        "n_figures": int(len(manifest)),
        "manifest": str(OUT_MANIFEST.relative_to(REPO_ROOT)),
        "style": {
            "palette": CUBE_HEX,
            "outputs": ["png"],
            "theme": "cubehelix_green_gradient",
        },
        "figures": manifest.to_dict(orient="records"),
    }

    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved DEAP publication figures:")
    print(manifest[["figure_name", "png_path"]].to_string(index=False))
    print(f"\nManifest: {OUT_MANIFEST.relative_to(REPO_ROOT)}")
    print(f"Metadata: {OUT_METADATA.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()