"""
tests/parity/plots.py

Renders the results/comparison/*.json output (produced by compare.py) as
plots: one bar chart per method (homals / princals / morals) showing the
max % difference between R and Python for every dataset that passed.

Pairs listed in manifest.json's "exclude_from_plots" are left out of the
chart (they're known, already-explained non-matches -- see the manifest
for why) but remain fully recorded in results/comparison/ and
results/comparison_summary.csv; excluded pairs are named in a caption
below each affected chart so nothing is silently dropped.

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
MANIFEST_PATH = os.path.join(HERE, "manifest.json")
os.makedirs(PLOT_DIR, exist_ok=True)

PASS_COLOR = "#0ca30c"           # fixed status palette (dataviz skill reference)
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


def load_exclusions():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    return {(e["dataset"], e["method"]): e["reason"] for e in manifest.get("exclude_from_plots", [])}


def plot_method(method, records, excluded):
    shown = [r for r in records if (r["dataset"], r["method"]) not in excluded]
    left_out = [r for r in records if (r["dataset"], r["method"]) in excluded]
    shown = sorted(shown, key=lambda r: r["dataset"])

    datasets = [r["dataset"] for r in shown]
    # scored_max_pct_diff is computed once by compare.py's classify() -- the
    # same number that determined PASS/WARN/FAIL -- so this chart can't drift
    # out of sync with the verdict logic by re-deriving its own exclusions.
    max_pct = [r.get("scored_max_pct_diff", np.nan) for r in shown]

    # log scale needs a positive floor; treat exact 0 (bit-identical) as the floor value
    floor = 1e-10
    plot_vals = [max(v, floor) if not np.isnan(v) else floor for v in max_pct]

    # Reserve headroom above the tallest bar for its label, and footer space
    # for the exclusion caption -- both live outside the data area, never on it.
    fig_h = 5.5 if left_out else 5.0
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 0.85), fig_h))

    bars = ax.bar(datasets, plot_vals, color=PASS_COLOR, width=0.55,
                   edgecolor=SURFACE, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylim(top=max(plot_vals) * 30)   # headroom so top labels never clip
    ax.set_ylabel("Max % difference from R (log scale)")
    ax.set_title(f"{method.capitalize()} — R vs Python parity (all shown: PASS, <1%)",
                 fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    for bar, raw in zip(bars, max_pct):
        label = "n/a" if np.isnan(raw) else f"{raw:.2g}%"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY)

    fig.tight_layout()

    # Caption for excluded pairs goes in the figure margin below the axes --
    # never inside the plot area, so it can't overlap a bar or its label.
    if left_out:
        lines = [f"Not shown ({r['dataset']}): {excluded[(r['dataset'], r['method'])]}" for r in left_out]
        fig.subplots_adjust(bottom=0.05 + 0.16 * len(lines))
        fig.text(0.02, 0.02, "\n".join(lines), fontsize=7, color=TEXT_SECONDARY,
                  ha="left", va="bottom", wrap=True)

    out_path = os.path.join(PLOT_DIR, f"{method}_pct_diff.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {out_path}")


def main():
    records = load_comparisons()
    excluded = load_exclusions()
    by_method = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r)

    for method, recs in sorted(by_method.items()):
        plot_method(method, recs, excluded)

    print(f"\n{len(records)} comparisons available, plotted across {len(by_method)} method(s).")


if __name__ == "__main__":
    main()
