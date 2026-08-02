#!/usr/bin/env python3
"""
analyze.py — Professor's analysis: raw distributions, trend plots, and PCA plots
comparing raw data, PyGifi-transformed, and R-Gifi-transformed datasets.

Usage (from project root):
    python3 plot.py
  or directly:
    python3 analysis/analyze.py

Outputs (per dataset):
    analysis/plots/<dataset_name>/
        01_raw_distributions.png   — category frequency bar charts for every variable
        02_trend_plots.png         — overlapping quantification trends (Line plots)
        03_pca_plots.png           — PCA scatter on all 3 forms of the data (Subplots)
        04_overlapping_pca.png     — All 3 PCA forms on a single plot for direct comparison
        05_global_transform.png    — Overlay of all variable transformations
        06_distribution_comp.png   — Density comparison of transformed datasets

Requirements:
    pip install matplotlib seaborn scikit-learn
    R with Gifi package: install.packages("Gifi")
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from pygifi import Princals

DATA_DIR        = os.path.join(PROJECT_ROOT, "validation", "datasets", "processed")
R_SCRIPT        = os.path.join(HERE, "get_r_transforms.R")
R_TRANSFORM_DIR = os.path.join(HERE, "r_transforms")
PLOT_DIR        = os.path.join(HERE, "plots")

os.makedirs(R_TRANSFORM_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor":   "#2a2a3e",
    "axes.edgecolor":   "#555577",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "text.color":       "#cdd6f4",
    "grid.color":       "#44475a",
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
})

COLORS = {
    "raw":    "#89b4fa",
    "pygifi": "#a6e3a1",
    "r":      "#f38ba8",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate R Gifi transforms
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  [Step 1] Generating R Gifi transforms …")
print("=" * 62)

r_result = subprocess.run(
    ["Rscript", R_SCRIPT, DATA_DIR, R_TRANSFORM_DIR],
    capture_output=True, text=True
)
if r_result.returncode != 0:
    print(f"  [WARNING] R script failed:\n{r_result.stderr[:300]}")
    print("  R series will be omitted from plots.")
    R_AVAILABLE = False
else:
    print(r_result.stdout.strip())
    R_AVAILABLE = True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Process each dataset
# ══════════════════════════════════════════════════════════════════════════════
csv_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".csv") and "transformed" not in f
])

if not csv_files:
    print("No datasets found in", DATA_DIR)
    sys.exit(0)

print(f"\n[Step 2] Found {len(csv_files)} dataset(s): {csv_files}")

for ds_file in csv_files:
    ds_name = ds_file.replace(".csv", "")
    print(f"\n{'─' * 62}")
    print(f"  DATASET: {ds_name}")
    print(f"{'─' * 62}")

    out_dir = os.path.join(PLOT_DIR, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, ds_file))
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
    for col in df.columns:
        df[col] = df[col].astype("category")

    n_vars = len(df.columns)
    ncols  = min(4, n_vars)
    nrows  = (n_vars + ncols - 1) // ncols

    # ── A: Raw integer codes ───────────────────────────────────────────────────
    df_raw = df.apply(lambda col: col.cat.codes.astype(float))

    # ── B: PyGifi transform ───────────────────────────────────────────────────
    print("  Running PyGifi Princals …")
    model = Princals(ndim=2)
    model.fit(df)
    quantifications = model.result_["quantifications"]
    df_pygifi = pd.DataFrame(
        model.result_["transform"],
        index=df.index, columns=df.columns
    )

    # ── C: R Gifi transform ───────────────────────────────────────────────────
    r_path = os.path.join(R_TRANSFORM_DIR, f"r_transform_{ds_file}")
    df_r = None
    if R_AVAILABLE and os.path.exists(r_path):
        df_r = pd.read_csv(r_path)
        df_r.columns = df.columns

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1: Raw Category Distributions
    # ══════════════════════════════════════════════════════════════════════════
    print("  →  01_raw_distributions.png")
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * 3.8),
                             squeeze=False)
    fig.suptitle(f"Raw Category Distributions — {ds_name}",
                 fontsize=14, fontweight="bold", color="#cdd6f4", y=1.01)

    for idx, col in enumerate(df.columns):
        ax = axes[idx // ncols][idx % ncols]
        counts = df[col].value_counts().sort_index()
        ax.bar(range(len(counts)), counts.values,
               color=COLORS["raw"], edgecolor="#6699cc", alpha=0.85, linewidth=0.5)
        ax.set_title(col, fontsize=9, fontweight="bold", color="#cdd6f4")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(axis="y")

    for j in range(n_vars, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_raw_distributions.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2: Trend — Category Quantification Across 3 Series
    # ══════════════════════════════════════════════════════════════════════════
    print("  →  02_trend_plots.png")
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.2),
                             squeeze=False)
    fig.suptitle(f"Category Quantification Trends — {ds_name}\n"
                 f"(Raw scaled codes vs PyGifi vs R Gifi per variable)",
                 fontsize=12, fontweight="bold", color="#cdd6f4", y=1.02)

    legend_drawn = False
    for idx, (col, q) in enumerate(zip(df.columns, quantifications)):
        ax = axes[idx // ncols][idx % ncols]
        cats = df[col].cat.categories
        n_cats = len(cats)
        x = np.arange(n_cats)

        # Scale raw codes for comparison
        raw_vals = np.arange(n_cats, dtype=float)
        py_quants = q[:, 0]
        py_range  = py_quants.max() - py_quants.min() if n_cats > 1 else 1.0
        raw_range = float(n_cats - 1) if n_cats > 1 else 1.0
        raw_scaled = (raw_vals - raw_vals.mean()) / raw_range * py_range

        ax.plot(x, raw_scaled, marker="o", linestyle="--", linewidth=1.2,
                color=COLORS["raw"], label="Raw (scaled)", alpha=0.6)
        ax.plot(x, py_quants, marker="s", linestyle="-", linewidth=1.8,
                color=COLORS["pygifi"], label="PyGifi", alpha=0.9)

        if df_r is not None and col in df_r.columns:
            r_col_vals = df_r[col].values
            r_cat_means = np.array([
                r_col_vals[df[col].values == cat].mean()
                if (df[col].values == cat).any() else 0.0
                for cat in cats
            ])
            ax.plot(x, r_cat_means, marker="^", linestyle="-.", linewidth=1.2,
                    color=COLORS["r"], label="R Gifi", alpha=0.6)

        ax.set_title(col, fontsize=9, fontweight="bold", color="#cdd6f4")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="#888899", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.15)

        if not legend_drawn:
            ax.legend(fontsize=7, loc="best", facecolor="#1e1e2e", edgecolor="#555577")
            legend_drawn = True

    for j in range(n_vars, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_trend_plots.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 3: PCA on all 3 forms of the data
    # ══════════════════════════════════════════════════════════════════════════
    print("  →  03_pca_plots.png")
    series = [
        ("Raw\n(integer codes)", df_raw,   COLORS["raw"]),
        ("PyGifi\nTransformed",  df_pygifi, COLORS["pygifi"]),
    ]
    if df_r is not None:
        series.append(("R Gifi\nTransformed", df_r, COLORS["r"]))

    n_series = len(series)
    fig, axes = plt.subplots(1, n_series,
                             figsize=(n_series * 5.5, 5.2),
                             squeeze=False)
    fig.suptitle(f"PCA Comparison — {ds_name}",
                 fontsize=14, fontweight="bold", color="#cdd6f4")

    for j, (title, data_df, color) in enumerate(series):
        ax = axes[0][j]
        X = np.nan_to_num(data_df.values.astype(float), nan=0.0)
        X_scaled = StandardScaler().fit_transform(X)
        pca    = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        ev     = pca.explained_variance_ratio_

        ax.scatter(coords[:, 0], coords[:, 1],
                   color=color, alpha=0.45, s=16, edgecolors="none")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#cdd6f4")
        ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% var.)", fontsize=9)
        ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}% var.)", fontsize=9)
        ax.axhline(0, color="#444466", linewidth=0.5)
        ax.axvline(0, color="#444466", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_pca_plots.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 4: Overlapping PCA Scatter
    # ══════════════════════════════════════════════════════════════════════════
    print("  →  04_overlapping_pca.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f"Overlapping PCA Comparison — {ds_name}",
                 fontsize=14, fontweight="bold", color="#cdd6f4")

    for title, data_df, color in series:
        X = np.nan_to_num(data_df.values.astype(float), nan=0.0)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        
        lbl = title.replace("\n", " ")
        ax.scatter(coords[:, 0], coords[:, 1],
                   color=color, alpha=0.5, s=25, label=lbl, edgecolors="white", linewidth=0.3)

    ax.set_xlabel("Principal Component 1", color="#cdd6f4")
    ax.set_ylabel("Principal Component 2", color="#cdd6f4")
    ax.axhline(0, color="#444466", linewidth=0.8)
    ax.axvline(0, color="#444466", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(facecolor="#1e1e2e", edgecolor="#555577", loc="best")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_overlapping_pca.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 5: Global Transformation Comparison
    # ══════════════════════════════════════════════════════════════════════════
    print("  →  05_global_transform.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Global Transformation Overview — {ds_name}", fontsize=14, fontweight="bold", color="#cdd6f4")
    
    for idx, (col, q) in enumerate(zip(df.columns, quantifications)):
        cats = df[col].cat.categories
        n_cats = len(cats)
        # normalize x to 0-1 for overlapping multiple variables
        x_norm = np.linspace(0, 1, n_cats)
        py_quants = q[:, 0]
        ax.plot(x_norm, py_quants, label=col, alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel("Category Index (Normalized 0-1)", color="#cdd6f4")
    ax.set_ylabel("Quantification Value", color="#cdd6f4")
    ax.grid(True, alpha=0.2)
    # Legend can get very large, so we move it outside or limit it
    if len(df.columns) <= 15:
        ax.legend(fontsize=7, ncol=2, facecolor="#1e1e2e", edgecolor="#555577", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_global_transform.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 6: Overall Distribution Comparison
    # ══════════════════════════════════════════════════════════════════════════
    print("  →  06_distribution_comp.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Distribution Comparison (All Variables) — {ds_name}", fontsize=14, fontweight="bold", color="#cdd6f4")
    
    import seaborn as sns
    combined_data = []
    
    # Raw
    combined_data.append(pd.DataFrame({"Value": df_raw.values.flatten(), "Source": "Raw (integer codes)"}))
    # PyGifi
    combined_data.append(pd.DataFrame({"Value": df_pygifi.values.flatten(), "Source": "PyGifi Transformed"}))
    # R Gifi
    if df_r is not None:
        combined_data.append(pd.DataFrame({"Value": df_r.values.flatten(), "Source": "R Gifi Transformed"}))
    
    plot_df = pd.concat(combined_data)
    sns.kdeplot(data=plot_df, x="Value", hue="Source", fill=True, ax=ax, palette=[COLORS["raw"], COLORS["pygifi"], COLORS["r"]][:len(combined_data)])
    
    ax.set_xlabel("Transformed Value", color="#cdd6f4")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_distribution_comp.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  ✓  Done — analysis/plots/{ds_name}/")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 62}")
print(f"  All plots saved to: analysis/plots/")
print(f"  Files per dataset:")
print(f"    01_raw_distributions.png  — category frequency bar charts")
print(f"    02_trend_plots.png        — raw vs PyGifi vs R Gifi quantifications")
print(f"    03_pca_plots.png          — PCA on all 3 data forms")
print(f"{'=' * 62}\n")
