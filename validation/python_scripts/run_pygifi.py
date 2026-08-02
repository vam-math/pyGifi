"""
run_pygifi.py — Generate per-parameter result CSVs from PyGifi for Princals only.

Exports:
  1. Category Quantifications per variable
  2. The full PyGifi Transformed Dataset
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure pygifi can be imported when run from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # PyGifi2/PyGifi2/
sys.path.insert(0, _ROOT)
# Also add pygifi/rng so pygifi_rng C extension can be found
_RNG_DIR = os.path.join(_ROOT, "pygifi", "rng")
if _RNG_DIR not in sys.path:
    sys.path.insert(0, _RNG_DIR)

from pygifi import Princals


def check_rng_available():
    """Check if the pygifi_rng C extension is compiled and ready."""
    try:
        import pygifi_rng  # noqa: F401
        print("  [RNG] pygifi_rng extension loaded — exact R parity mode")
        return True
    except ImportError:
        print("  [RNG] WARNING: pygifi_rng not found — using SVD fallback")
        print("  [RNG] Build it: cd pygifi/rng && python3 setup_rng.py build_ext --inplace")
        return False


DATASETS_DIR = os.path.join(_HERE, "..", "datasets", "processed")
RESULTS_DIR  = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Check extension before processing any datasets
_rng_available = check_rng_available()
_r_seed = 123 if _rng_available else None

datasets = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv") and "transformed" not in f])

def _save_df(df, path):
    """Save a pandas DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"    -> {os.path.basename(path)}  {df.shape}")

for ds in datasets:
    prefix = ds.replace(".csv", "")
    data = pd.read_csv(os.path.join(DATASETS_DIR, ds))

    # Drop unnamed index columns
    data.drop(columns=[c for c in data.columns if "Unnamed" in c], inplace=True)

    # All columns will be converted to categorical
    for col in data.columns:
        data[col] = data[col].astype("category")

    print(f"\n[{prefix}]")

    # ─────────────────────────────────────────────────────────────────
    # PRINCALS  — use r_seed=1 for exact R parity when extension available
    # ─────────────────────────────────────────────────────────────────
    print("  Fitting PyGifi PRINCALS ...")
    p = Princals(ndim=2, r_seed=_r_seed, itmax=1000, levels="ordinal")
    p.fit(data)

    r = p.result_
    base = f"py_princals_{prefix}"

    quantifications = r["quantifications"]

    # 1. Export Category Quantifications
    for col, q in zip(data.columns, quantifications):
        # Match R's default replacement of spaces to periods
        safe = col.replace(" ", ".").replace("/", ".")
        categories = data[col].cat.categories
        
        # We only care about dimension 1 mapping as standard for report, but export all.
        q_df = pd.DataFrame(q, index=categories, columns=[f"dim{i+1}" for i in range(q.shape[1])])
        q_df.index.name = "Category"
        # R's write.csv(row.names=FALSE) with a Category column
        q_df_export = q_df.reset_index()
        _save_df(q_df_export, os.path.join(RESULTS_DIR, f"{base}_quant_{safe}.csv"))

    # 2. Build and export the Transformed Dataset
    df_transformed = pd.DataFrame(p.result_["transform"], index=data.index, columns=data.columns)

    transformed_path = os.path.join(DATASETS_DIR, f"pygifi_transformed_{prefix}.csv")
    df_transformed.to_csv(transformed_path, index=False)
    print(f"    -> {os.path.basename(transformed_path)}  {df_transformed.shape} [Transformed Dataset]")

    # 3. Export Eigenvalues
    ev_df = pd.DataFrame(p.result_["evals"], columns=["eigenvalue"])
    # Eigenvalues in R: write.csv(pr$evals, row.names=FALSE) -> one column named "eigenvalue"
    _save_df(ev_df, os.path.join(RESULTS_DIR, f"{base}_evals.csv"))

print("\nDone — all PyGifi Princals results exported.")
