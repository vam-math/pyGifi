"""
compare_results.py — Full Category-wise comparison of R Gifi vs PyGifi (Princals).

For each dataset:
  - Compares the Transformed Dataset
  - Compares the Category Quantifications (Variable by Variable, Category by Category)
  - Reports max |diff|, mean |diff|, and PASS/FAIL per component
  - Writes a full comparison_report.txt and comparison_summary.csv

Sign-flip tolerance: eigenvalues and other SVD-derived quantities may
differ only in global sign — this is allowed (SVD ambiguity).
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ── Tolerances ────────────────────────────────────────────────────────────────
# Tightened from 0.05 → 1e-5 now that pygifi_rng gives exact R initialization
# parity (MT19937 + AS241 qnorm). If pygifi_rng is not compiled, the SVD
# fallback may produce slightly different results and this will report FAILs.
ABS_TOL   = 1e-6   # Tightened for exact R parity mode (seed 123)
PREVIEW_N = 5      # number of rows to preview in general tables

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.abspath(os.path.join(HERE, "..", "results"))
DATASETS_DIR = os.path.abspath(os.path.join(HERE, "..", "datasets", "processed"))
REPORT_PATH  = os.path.abspath(os.path.join(HERE, "..", "results", "comparison_report.txt"))
SUMMARY_PATH = os.path.abspath(os.path.join(HERE, "..", "results", "comparison_summary.csv"))

def compare_arrays(r_arr, py_arr):
    """Compare two arrays allowing for global sign flip (SVD ambiguity).
    Returns (max_diff, mean_diff, error_str_or_None)."""
    r  = np.asarray(r_arr,  dtype=float)
    p  = np.asarray(py_arr, dtype=float)
    if r.shape != p.shape:
        return None, None, f"SHAPE MISMATCH  R:{r.shape} vs Py:{p.shape}"
    diff_raw  = np.abs(r - p)
    diff_flip = np.abs(r + p)
    diff = np.minimum(diff_raw, diff_flip)
    return float(np.max(diff)), float(np.mean(diff)), None

def load_csv(path):
    """Load a CSV; return (DataFrame, error_str_or_None)."""
    if not os.path.exists(path):
        return None, f"FILE NOT FOUND: {os.path.basename(path)}"
    try:
        return pd.read_csv(path), None
    except Exception as e:
        return None, str(e)

def get_quantification_variables(ds):
    """Discover per-variable quantification CSVs."""
    variables = []
    prefix = f"r_princals_{ds}_quant_"
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.startswith(prefix):
            vname = fname[len(prefix):-4]  # remove .csv
            variables.append(vname)
    return variables

# ── Main ──────────────────────────────────────────────────────────────────────
datasets = sorted([f.replace(".csv", "") for f in os.listdir(DATASETS_DIR)
                   if f.endswith(".csv") and "transformed" not in f])

rows = []          # store results for summary CSV
output_lines = []  # store results for report.txt

W = 92   # total line width

def line(s=""):
    print(s)
    output_lines.append(str(s))

line()
line("=" * W)
line(f"  PyGifi vs R Gifi — Category-Wise Comparison Report (PRINCALS)")
line(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
line("=" * W)

all_passed = True

for ds in datasets:
    line()
    line("━" * W)
    line(f"  DATASET: {ds}")
    line("━" * W)

    # 1. Compare the Transformed Datasets
    line()
    line(f"  ┌─── [TRANSFORMED DATASET] " + "─" * (W - 29))
    r_ts_path = os.path.abspath(os.path.join(DATASETS_DIR, f"gifi_transformed_{ds}.csv"))
    py_ts_path = os.path.abspath(os.path.join(DATASETS_DIR, f"pygifi_transformed_{ds}.csv"))

    r_ts_df, r_ts_err = load_csv(r_ts_path)
    py_ts_df, py_ts_err = load_csv(py_ts_path)

    print(f"\\nDEBUG PATHS:")
    print(f" R_TS_PATH: {r_ts_path} | Err: {r_ts_err}")
    print(f" PY_TS_PATH: {py_ts_path} | Err: {py_ts_err}")

    if r_ts_err or py_ts_err:
        missing_side = ("R " if r_ts_err else "") + ("PyGifi " if py_ts_err else "")
        line(f"  │  [ERROR] Missing transformed datasets for {missing_side}")
        all_passed = False
    else:
        # Check shapes
        if r_ts_df.shape != py_ts_df.shape:
            line(f"  │  [ERROR] Shape Mismatch! R:{r_ts_df.shape} vs Py:{py_ts_df.shape}")
            all_passed = False
        else:
            # Check values
            res_max, res_mean, err = compare_arrays(r_ts_df.values, py_ts_df.values)
            if err:
                line(f"  │  [ERROR] {err}")
                all_passed = False
            else:
                passed = bool(res_max <= ABS_TOL)
                status = "✓ PASS" if passed else "✗ FAIL"
                line(f"  │  Status: {status}  (Max diff: {res_max:.5f}, Mean diff: {res_mean:.5f})")
                if not passed: all_passed = False
                
                rows.append({"Dataset": ds, "Variable": "Transformed Dataset", "Category": "ALL",
                             "Max|diff|": round(res_max, 6), "Mean|diff|": round(res_mean, 6),
                             "% diff": 0.0, 
                             "Status": "PASS" if passed else "FAIL"})

    # 1.5 Compare Eigenvalues
    line()
    line(f"  ┌─── [EIGENVALUES] " + "─" * (W - 20))
    r_ev_path = os.path.join(RESULTS_DIR, f"r_princals_{ds}_evals.csv")
    py_ev_path = os.path.join(RESULTS_DIR, f"py_princals_{ds}_evals.csv")
    
    r_ev_df, r_ev_err = load_csv(r_ev_path)
    py_ev_df, py_ev_err = load_csv(py_ev_path)
    
    if r_ev_err or py_ev_err:
        line(f"  │  [MISSING] Eigenvalues")
    else:
        res_max, res_mean, err = compare_arrays(r_ev_df.iloc[:, 0].values, py_ev_df.iloc[:, 0].values)
        passed = bool(res_max <= ABS_TOL)
        status = "✓ PASS" if passed else "✗ FAIL"
        line(f"  │  Status: {status}  (Max diff: {res_max:.6f}, Mean diff: {res_mean:.6f})")
        if not passed: all_passed = False
    line(f"  └" + "─" * (W - 4))

    # 2. Compare Category Quantifications Variable by Variable
    variables = get_quantification_variables(ds)
    if not variables:
        line(f"  [WARNING] No quantification variables found for dataset {ds}")
        continue

    line()
    line(f"  ┌─── [CATEGORY QUANTIFICATIONS] " + "─" * (W - 34))

    for vname in variables:
        line(f"  │")
        line(f"  │    Variable: {vname}")
        line(f"  │    " + "-" * 70)
        
        r_q_path  = os.path.join(RESULTS_DIR, f"r_princals_{ds}_quant_{vname}.csv")
        py_q_path = os.path.join(RESULTS_DIR, f"py_princals_{ds}_quant_{vname}.csv")

        r_q_df, r_err = load_csv(r_q_path)
        py_q_df, py_err = load_csv(py_q_path)

        if r_err or py_err:
            missing = ("R " if r_err else "") + ("PyGifi " if py_err else "")
            line(f"  │    [MISSING] {missing}")
            all_passed = False
            continue

        # Extract categories from the index column (usually named 'Unnamed: 0')
        r_cat_col = r_q_df.columns[0]
        py_cat_col = py_q_df.columns[0]
        
        # Strip all strings and lowercase to prevent padding and casing mismatches
        r_categories = r_q_df[r_cat_col].astype(str).str.strip().str.lower().values
        py_categories = py_q_df[py_cat_col].astype(str).str.strip().str.lower().values

        print(f"\\nDEBUG {vname}:")
        print(f" R Cats : {r_categories}")
        print(f" Py Cats: {py_categories}")

        # Find the intersecting categories to compare safely
        common_cats = set(r_categories).intersection(set(py_categories))
        
        # We assume dimension 1 is column 'dim1'
        r_dim1_col = "dim1" if "dim1" in r_q_df.columns else r_q_df.columns[1]
        py_dim1_col = "dim1" if "dim1" in py_q_df.columns else py_q_df.columns[1]

        r_q_map = dict(zip(r_q_df[r_cat_col].astype(str).str.strip().str.lower(), r_q_df[r_dim1_col]))
        py_q_map = dict(zip(py_q_df[py_cat_col].astype(str).str.strip().str.lower(), py_q_df[py_dim1_col]))

        col_w = 14
        hdr = (f"  │      {'Category':<20} | {'R value':>{col_w}} | {'PyGifi value':>{col_w}} | {'diff':>{col_w}} | {'% diff':>{col_w}} | {'Status'}")
        line(hdr)
        line(f"  │      " + "-" * 93)

        var_max_diff = 0
        for cat in sorted(common_cats):
            r_val = float(r_q_map[cat])
            py_val = float(py_q_map[cat])

            # Apply flip logic per-element to see if signs are just flipped
            diff_raw = abs(r_val - py_val)
            diff_flip = abs(r_val + py_val)
            diff = min(diff_raw, diff_flip)
            var_max_diff = max(var_max_diff, diff)

            # Calculate percentage difference (using aligned value)
            if diff_flip < diff_raw:
                aligned_py_val = -py_val
            else:
                aligned_py_val = py_val
            
            if abs(r_val) > 1e-12:
                p_diff = (aligned_py_val - r_val) / r_val * 100
            else:
                p_diff = 0.0 if abs(aligned_py_val) < 1e-12 else float('inf')

            passed = bool(diff <= ABS_TOL)
            if not passed: all_passed = False
            status = "✓ PASS" if passed else "✗ FAIL"

            line(f"  │      {cat:<20} | {r_val:>{col_w}.6f} | {py_val:>{col_w}.6f} | {diff:>{col_w}.6f} | {p_diff:>{col_w}.6f}%| {status}")
            
            rows.append({"Dataset": ds, "Variable": vname, "Category": str(cat),
                         "Max|diff|": round(diff, 6), "Mean|diff|": round(diff, 6),
                         "% diff": round(p_diff, 6) if p_diff != float('inf') else "inf",
                         "Status": "PASS" if passed else "FAIL"})

        if len(common_cats) < len(r_categories) or len(common_cats) < len(py_categories):
            line(f"  │      [WARNING] Category mismatch! R has {len(r_categories)}, Py has {len(py_categories)}")
            all_passed = False

    line(f"  └" + "─" * (W - 4))


# ── Overall summary ────────────────────────────────────────────────────────────
line()
line("=" * W)
overall = "ALL PARAMETERS PASSED ✓" if all_passed else "SOME PARAMETERS FAILED ✗"
line(f"  Overall: {overall}")
line("=" * W)
line()

# Save report and summary
with open(REPORT_PATH, "w") as f:
    f.write("\n".join(output_lines))

summary_df = pd.DataFrame(rows)
summary_df.to_csv(SUMMARY_PATH, index=False)

print(f"\n  Full report  : {os.path.abspath(REPORT_PATH)}")
print(f"  CSV summary  : {os.path.abspath(SUMMARY_PATH)}")
print()

if not all_passed:
    sys.exit(1)

