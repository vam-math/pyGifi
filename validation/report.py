"""
report.py — Run the full PyGifi versus R Gifi validation pipeline.

Steps:
  1. Preprocess datasets
  2. Generate PyGifi result CSVs   (python_scripts/run_pygifi.py)
  3. Generate R Gifi result CSVs   (r_scripts/run_gifi.R)
  4. Run parameter comparison      (compare/compare_results.py)

Usage:
    cd validation/
    python3 report.py
"""

import subprocess
import os
import sys

def run(cmd, cwd=None, env=None):
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"\n  [ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

HERE = os.path.dirname(os.path.abspath(__file__))
ENV  = {**os.environ, "PYTHONPATH": os.path.dirname(HERE)}

print("=" * 60)
print("  PyGifi Validation Pipeline")
print("=" * 60)

print("\n[Step 1] Preprocessing datasets ...")
run(["python3", os.path.join(HERE, "preprocess_datasets.py")], env=ENV)

print("\n[Step 2] Generating PyGifi result CSVs ...")
run(["python3", os.path.join(HERE, "python_scripts", "run_pygifi.py")], env=ENV)

print("\n[Step 3] Generating R Gifi result CSVs ...")
r_script = os.path.join(HERE, "r_scripts", "run_gifi.R")
run(["Rscript", r_script], env=ENV)

print("\n[Step 4] Running parameter comparison ...")
compare_script = os.path.join(HERE, "compare", "compare_results.py")
run(["python3", compare_script], env=ENV)

print("\n[Step 4.5] Phase 2 — Distribution Comparison ...")
dist_script = os.path.join(HERE, "compare", "visualize_distributions.py")
run(["python3", dist_script], env=ENV)

print("\n[Step 4.6] Phase 3 — Structural PCA Comparison ...")
pca_script = os.path.join(HERE, "compare", "pca_comparison.py")
run(["python3", pca_script], env=ENV)

# ── Step 5: Clean up intermediate files ──────────────────────────────────────
print("\n[Step 5] Cleaning up intermediate files ...")

RESULTS_DIR  = os.path.join(HERE, "results")
DATASETS_DIR = os.path.join(HERE, "datasets", "processed")

keep_in_results = {"comparison_report.txt", "comparison_summary.csv"}
removed = 0

for fname in os.listdir(RESULTS_DIR):
    if fname not in keep_in_results and fname.endswith(".csv"):
        os.remove(os.path.join(RESULTS_DIR, fname))
        removed += 1

for fname in os.listdir(DATASETS_DIR):
    if "transformed" in fname and fname.endswith(".csv"):
        os.remove(os.path.join(DATASETS_DIR, fname))
        removed += 1

print(f"  Removed {removed} intermediate file(s).")

print("\n" + "=" * 60)
print("  Done. Final outputs:")
print(f"    {os.path.join(RESULTS_DIR, 'comparison_report.txt')}")
print(f"    {os.path.join(RESULTS_DIR, 'comparison_summary.csv')}")
print("=" * 60)
