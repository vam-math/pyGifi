"""
tests/parity/run_all.py — single entry point for the full R-vs-Python parity check.

Steps:
  1. tests/parity/generate_r_ground_truth.R  -- real CRAN Gifi package output
  2. tests/parity/run_python.py              -- matching pygifi output
  3. tests/parity/compare.py                 -- % difference per (dataset, method)
  4. tests/parity/plots.py                   -- one chart per method

Usage (from repo root):
    python tests/parity/run_all.py
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, **kwargs)
    if result.returncode != 0:
        print(f"[ERROR] command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def find_rscript():
    for candidate in ("Rscript", "Rscript.exe"):
        path = shutil.which(candidate)
        if path:
            return path
    # Fall back to the winget-installed default location.
    default = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
    if os.path.exists(default):
        return default
    raise FileNotFoundError("Rscript not found on PATH and no default install found.")


print("=" * 60)
print("  PyGifi <-> R Gifi parity check")
print("=" * 60)

print("\n[1/4] Generating R ground truth ...")
run([find_rscript(), os.path.join(HERE, "generate_r_ground_truth.R")])

print("\n[2/4] Generating Python output ...")
run([sys.executable, os.path.join(HERE, "run_python.py")])

print("\n[3/4] Comparing R vs Python ...")
run([sys.executable, os.path.join(HERE, "compare.py")])

print("\n[4/4] Rendering plots ...")
run([sys.executable, os.path.join(HERE, "plots.py")])

print("\nDone. See results/comparison_summary.csv and results/plots/.")
