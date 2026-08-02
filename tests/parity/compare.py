"""
tests/parity/compare.py

Pairs up results/r_ground_truth/<dataset>_<method>.json with
results/python_output/<dataset>_<method>.json, computes a percentage
difference per field, and writes:

  results/comparison/<dataset>_<method>.json   -- per-field % diff detail
  results/comparison_summary.csv               -- one row per (dataset, method)

Sign-flip tolerance: SVD/eigen-derived quantities (eigenvectors, loadings,
quantifications) are only identifiable up to a sign flip per column, so for
each field we compare both as-is and negated, keeping whichever is closer
(same convention the project's earlier validation scripts used).

Usage (from repo root):
    python tests/parity/compare.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R_DIR = os.path.join(ROOT, "results", "r_ground_truth")
PY_DIR = os.path.join(ROOT, "results", "python_output")
OUT_DIR = os.path.join(ROOT, "results", "comparison")
SUMMARY_PATH = os.path.join(ROOT, "results", "comparison_summary.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# Fields present in every result payload that aren't comparable model outputs.
META_FIELDS = {"dataset", "method", "error"}
EPS = 1e-8


def flatten(value):
    """Flatten arbitrarily nested lists/numbers (dict values taken in order) into a 1D float array."""
    out = []

    def walk(v):
        if v is None:
            out.append(np.nan)
        elif isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, dict):
            for item in v.values():
                walk(item)
        elif isinstance(v, list):
            for item in v:
                walk(item)
        else:
            raise TypeError(f"Cannot flatten value of type {type(v)}: {v!r}")

    walk(value)
    return np.asarray(out, dtype=float)


def compare_field(r_val, py_val):
    """Return dict with max/mean % diff, or an error string on shape mismatch."""
    r_flat = flatten(r_val)
    py_flat = flatten(py_val)
    if r_flat.shape != py_flat.shape:
        return {"error": f"shape mismatch: R={r_flat.shape} vs Python={py_flat.shape}"}

    diff_raw = np.abs(r_flat - py_flat)
    diff_flip = np.abs(r_flat + py_flat)
    diff = np.minimum(diff_raw, diff_flip)

    denom = np.abs(r_flat) + EPS
    pct = 100.0 * diff / denom

    return {
        "n": int(r_flat.size),
        "max_abs_diff": float(np.nanmax(diff)),
        "mean_abs_diff": float(np.nanmean(diff)),
        "max_pct_diff": float(np.nanmax(pct)),
        "mean_pct_diff": float(np.nanmean(pct)),
    }


def classify(field_reports):
    """Overall verdict for one (dataset, method) pair from its field comparisons."""
    errors = [f for f, rep in field_reports.items() if "error" in rep]
    if errors:
        return "FAIL", f"shape mismatch in: {', '.join(errors)}"
    max_pct = max((rep["max_pct_diff"] for rep in field_reports.values()), default=0.0)
    if max_pct < 1.0:
        return "PASS", f"max % diff = {max_pct:.4g}%"
    if max_pct < 10.0:
        return "WARN", f"max % diff = {max_pct:.4g}%"
    return "FAIL", f"max % diff = {max_pct:.4g}%"


def main():
    r_files = {f[:-5] for f in os.listdir(R_DIR) if f.endswith(".json")}
    py_files = {f[:-5] for f in os.listdir(PY_DIR) if f.endswith(".json")}
    keys = sorted(r_files | py_files)

    summary_rows = []

    for key in keys:
        dataset, _, method = key.rpartition("_")
        r_path = os.path.join(R_DIR, f"{key}.json")
        py_path = os.path.join(PY_DIR, f"{key}.json")

        if not os.path.exists(r_path):
            verdict, detail = "NO_R_OUTPUT", "R ground truth missing"
            field_reports = {}
        elif not os.path.exists(py_path):
            verdict, detail = "NO_PY_OUTPUT", "Python output missing"
            field_reports = {}
        else:
            with open(r_path) as f:
                r_data = json.load(f)
            with open(py_path) as f:
                py_data = json.load(f)

            if "error" in r_data:
                verdict, detail, field_reports = "R_ERROR", r_data["error"], {}
            elif "error" in py_data:
                verdict, detail, field_reports = "PY_ERROR", py_data["error"], {}
            else:
                fields = (set(r_data) | set(py_data)) - META_FIELDS
                field_reports = {}
                for field in sorted(fields):
                    if field not in r_data or field not in py_data:
                        field_reports[field] = {"error": "field missing on one side"}
                        continue
                    if r_data[field] is None or py_data[field] is None:
                        continue
                    field_reports[field] = compare_field(r_data[field], py_data[field])
                verdict, detail = classify(field_reports)

        out_path = os.path.join(OUT_DIR, f"{key}.json")
        with open(out_path, "w") as f:
            json.dump({
                "dataset": dataset, "method": method,
                "verdict": verdict, "detail": detail,
                "fields": field_reports,
            }, f, indent=2)

        max_pct = max((rep.get("max_pct_diff", float("nan")) for rep in field_reports.values()),
                      default=float("nan"))
        summary_rows.append([dataset, method, verdict, detail, max_pct])
        print(f"  [{verdict:12s}] {dataset:15s} {method:10s} — {detail}")

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "verdict", "detail", "max_pct_diff"])
        writer.writerows(summary_rows)

    print(f"\nWrote {len(summary_rows)} comparison(s) to {OUT_DIR}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
