"""
tests/parity/plots.py

Renders the results/comparison/*.json output (produced by compare.py) as
plots: one grouped-bar figure per method (homals / princals / morals) showing
the max % difference between R and Python for every dataset, colored by
verdict (PASS / WARN / FAIL) using the project's fixed status palette so
color never has to be read alone (each bar is also direct-labeled).

Usage (from repo root):
    python tests/parity/plots.py

Writes results/plots/<method>_pct_diff.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
COMPARISON_DIR = os.path.join(ROOT, "results", "comparison")
PLOT_DIR = os.path.join(ROOT, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Fixed status palette (dataviz skill reference) -- never reused for series identity.
STATUS_COLOR = {
    "PASS": "#0ca30c",
    "WARN": "#fab219",
    "FAIL": "#d03b3b",
    "R_ERROR": "#d03b3b",
    "PY_ERROR": "#d03b3b",
    "NO_R_OUTPUT": "#ec835a",
    "NO_PY_OUTPUT": "#ec835a",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d0"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "text.color": TEXT_PRIMARY,
    "grid.color": GRID,
    "font.family": "DejaVu Sans",
})


def load_comparisons():
    records = []
    for fname in sorted(os.listdir(COMPARISON_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(COMPARISON_DIR, fname)) as f:
            records.append(json.load(f))
    return records


def plot_method(method, records):
    records = sorted(records, key=lambda r: r["dataset"])
    datasets = [r["dataset"] for r in records]
    verdicts = [r["verdict"] for r in records]
    max_pct = []
    for r in records:
        vals = [v.get("max_pct_diff") for v in r.get("fields", {}).values() if "max_pct_diff" in v]
        max_pct.append(max(vals) if vals else np.nan)

    # log scale needs a positive floor; treat exact 0 (bit-identical) as the floor value
    floor = 1e-10
    plot_vals = [max(v, floor) if not np.isnan(v) else floor for v in max_pct]

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 0.9), 5))
    colors = [STATUS_COLOR.get(v, "#9a9990") for v in verdicts]
    bars = ax.bar(datasets, plot_vals, color=colors, width=0.6,
                   edgecolor=SURFACE, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("Max % difference (R vs Python, log scale)")
    ax.set_title(f"{method.capitalize()} — R vs Python parity", fontweight="bold")
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    # Direct labels -- verdict must never be read from color alone.
    for bar, verdict, raw in zip(bars, verdicts, max_pct):
        label = verdict if np.isnan(raw) else f"{verdict}\n{raw:.2g}%"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=TEXT_SECONDARY)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               [STATUS_COLOR["PASS"], STATUS_COLOR["WARN"], STATUS_COLOR["FAIL"]]]
    ax.legend(handles, ["PASS (<1%)", "WARN (<10%)", "FAIL (≥10% or error)"],
              loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"{method}_pct_diff.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def main():
    records = load_comparisons()
    by_method = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r)

    for method, recs in sorted(by_method.items()):
        plot_method(method, recs)

    print(f"\n{len(records)} comparisons plotted across {len(by_method)} method(s).")


if __name__ == "__main__":
    main()
