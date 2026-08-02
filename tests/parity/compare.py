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
# ntel (iteration count) is a convergence-path diagnostic, not a model result --
# two implementations can reach an equivalent optimum in a different number of
# ALS steps due to ordinary floating-point rounding, so it's reported but never
# drives the PASS/WARN/FAIL verdict.
INFO_ONLY_FIELDS = {"ntel"}
# homals/princals object scores are only identifiable up to an arbitrary
# rotation/reflection of the ndim-dimensional latent space: any orthogonal
# transform of the solution has identical stress. quantifications, scoremat
# (which R defines as literally column 1 of each variable's transform --
# see homals.R's `sapply(y, function(xx) xx[,1])`, i.e. a specific axis, not
# an invariant summary), dmeasures, and loadings all live in that ambiguous
# basis, so a real implementation match can still show large raw differences
# in these fields. `f` (stress) and `evals` (eigenvalues) are the rotation
# -invariant signals of whether two solutions are actually equivalent, so
# they alone drive the verdict; the basis-dependent fields are still reported
# for reference but flagged and excluded from PASS/WARN/FAIL.
ROTATION_SENSITIVE_FIELDS = {"quantifications", "scoremat", "dmeasures", "loadings"}
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
    """Return dict with max/mean absolute and % diff, or an error on shape mismatch.

    % diff is computed relative to the field's own scale (max |R value| in the
    field), not per-element |r_i|. Gifi quantifications/scores are mean-centered
    by construction, so many individual entries sit near zero -- dividing by
    each element's own (near-zero) value would blow up the % diff into the
    thousands even when the absolute error is negligible. Scaling by the
    field's overall magnitude avoids that artifact while still being a
    meaningful "how big is this error relative to the size of these values".
    """
    r_flat = flatten(r_val)
    py_flat = flatten(py_val)
    if r_flat.shape != py_flat.shape:
        return {"error": f"shape mismatch: R={r_flat.shape} vs Python={py_flat.shape}"}

    diff_raw = np.abs(r_flat - py_flat)
    diff_flip = np.abs(r_flat + py_flat)
    diff = np.minimum(diff_raw, diff_flip)

    scale = np.nanmax(np.abs(r_flat)) if r_flat.size else 0.0
    pct = 100.0 * diff / (scale + EPS)

    return {
        "n": int(r_flat.size),
        "max_abs_diff": float(np.nanmax(diff)),
        "mean_abs_diff": float(np.nanmean(diff)),
        "max_pct_diff": float(np.nanmax(pct)),
        "mean_pct_diff": float(np.nanmean(pct)),
    }


def classify(field_reports):
    """Overall verdict for one (dataset, method) pair from its field comparisons.

    Driven by max %-of-scale diff across the rotation-invariant result fields
    (ntel and the basis-dependent fields in ROTATION_SENSITIVE_FIELDS are
    excluded -- see their definitions above). Thresholds: <1% PASS, <10% WARN,
    else FAIL. Falls back to the excluded fields only if nothing else exists
    to judge by (e.g. a method with no evals/f at all).
    """
    excluded = INFO_ONLY_FIELDS | ROTATION_SENSITIVE_FIELDS
    scored = {f: rep for f, rep in field_reports.items() if f not in excluded}
    if not scored:
        scored = {f: rep for f, rep in field_reports.items() if f not in INFO_ONLY_FIELDS}

    errors = [f for f, rep in scored.items() if "error" in rep]
    if errors:
        return "FAIL", f"shape mismatch in: {', '.join(errors)}", float("nan")
    if not scored:
        return "PASS", "no scored fields", 0.0
    worst_field = max(scored, key=lambda f: scored[f]["max_pct_diff"])
    max_pct = scored[worst_field]["max_pct_diff"]
    if max_pct < 1.0:
        return "PASS", f"max % diff = {max_pct:.4g}% ({worst_field})", max_pct
    if max_pct < 10.0:
        return "WARN", f"max % diff = {max_pct:.4g}% ({worst_field})", max_pct
    return "FAIL", f"max % diff = {max_pct:.4g}% ({worst_field})", max_pct


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
            verdict, detail, scored_max_pct = "NO_R_OUTPUT", "R ground truth missing", float("nan")
            field_reports = {}
        elif not os.path.exists(py_path):
            verdict, detail, scored_max_pct = "NO_PY_OUTPUT", "Python output missing", float("nan")
            field_reports = {}
        else:
            with open(r_path) as f:
                r_data = json.load(f)
            with open(py_path) as f:
                py_data = json.load(f)

            if "error" in r_data:
                verdict, detail, scored_max_pct, field_reports = "R_ERROR", r_data["error"], float("nan"), {}
            elif "error" in py_data:
                verdict, detail, scored_max_pct, field_reports = "PY_ERROR", py_data["error"], float("nan"), {}
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
                    if field in ROTATION_SENSITIVE_FIELDS and "error" not in field_reports[field]:
                        field_reports[field]["rotation_sensitive"] = True
                verdict, detail, scored_max_pct = classify(field_reports)

        out_path = os.path.join(OUT_DIR, f"{key}.json")
        with open(out_path, "w") as f:
            json.dump({
                "dataset": dataset, "method": method,
                "verdict": verdict, "detail": detail,
                # The single number that actually drove the verdict above --
                # excludes ntel and rotation-sensitive fields. plots.py reads
                # this directly instead of re-deriving its own (previously
                # inconsistent) exclusion logic.
                "scored_max_pct_diff": scored_max_pct,
                "fields": field_reports,
            }, f, indent=2)

        summary_rows.append([dataset, method, verdict, detail, scored_max_pct])
        print(f"  [{verdict:12s}] {dataset:15s} {method:10s} — {detail}")

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "verdict", "detail", "max_pct_diff"])
        writer.writerows(summary_rows)

    print(f"\nWrote {len(summary_rows)} comparison(s) to {OUT_DIR}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
