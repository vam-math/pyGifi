"""
tests/parity/run_python.py

Mirrors generate_r_ground_truth.R: runs pygifi's Homals / Princals / Morals
on every dataset + method listed in tests/parity/manifest.json, using the
*same* r_seed as R (pygifi_rng gives Python the exact ported R RNG, so both
sides draw an identical starting matrix independently — no manual hand-off
of R's random numbers is needed).

Column roles (numeric vs categorical) are resolved with
pygifi.utils.type_inference.prepare_dataframe_with_inference, the same
heuristic used elsewhere in the project, run non-interactively here.

Usage (from repo root):
    python tests/parity/run_python.py

Writes one JSON result file per (dataset, method) pair to
results/python_output/.
"""
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from pygifi import Homals, Princals, Morals  # noqa: E402
from pygifi.utils.type_inference import prepare_dataframe_with_inference  # noqa: E402

MANIFEST_PATH = os.path.join(HERE, "manifest.json")
OUT_DIR = os.path.join(ROOT, "results", "python_output")
DATA_DIR = os.path.join(ROOT, "pygifi", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def write_result(name, method, payload):
    payload = dict(payload)
    payload["dataset"] = name
    payload["method"] = method
    path = os.path.join(OUT_DIR, f"{name}_{method}.json")
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    print(f"  [OK] {os.path.basename(path)}")


def write_failure(name, method, exc):
    path = os.path.join(OUT_DIR, f"{name}_{method}.json")
    with open(path, "w") as f:
        json.dump({"dataset": name, "method": method, "error": str(exc)}, f, indent=2)
    print(f"  [FAIL] {os.path.basename(path)}: {exc}")


def load_typed_csv(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    prepared, resolved_kinds, _ = prepare_dataframe_with_inference(df, interactive=False)
    for col, kind in resolved_kinds.items():
        if kind == "categorical":
            prepared[col] = prepared[col].astype("category")
    return prepared


def run_homals(name, df, r_seed):
    try:
        model = Homals(ndim=2, eps=1e-8, itmax=1000, r_seed=r_seed).fit(df)
        r = model.result_
        write_result(name, "homals", {
            "evals": model.eigenvalues_, "f": r["f"], "ntel": r["ntel"],
            "quantifications": r["quantifications"],
            "scoremat": r["scoremat"], "dmeasures": r["dmeasures"],
        })
    except Exception as e:  # noqa: BLE001
        write_failure(name, "homals", e)
        traceback.print_exc()


def run_princals(name, df, r_seed):
    try:
        model = Princals(ndim=2, eps=1e-8, itmax=1000, r_seed=r_seed).fit(df)
        r = model.result_
        write_result(name, "princals", {
            "evals": model.eigenvalues_, "f": r["f"], "ntel": r["ntel"],
            "loadings": r["loadings"], "lambda": r["lambda_"],
            "quantifications": r["quantifications"],
            "scoremat": r["scoremat"], "dmeasures": r["dmeasures"],
        })
    except Exception as e:  # noqa: BLE001
        write_failure(name, "princals", e)
        traceback.print_exc()


def run_morals(name, X, y, r_seed):
    try:
        model = Morals(eps=1e-8, itmax=1000, r_seed=r_seed).fit(X, y)
        r = model.result_
        write_result(name, "morals", {
            "beta": r["beta"], "smc": r["smc"], "evals": r["evals"],
            "yhat": r["yhat"], "xhat": r["xhat"],
            "ntel": r["ntel"], "f": r["f"],
        })
    except Exception as e:  # noqa: BLE001
        write_failure(name, "morals", e)
        traceback.print_exc()


def read_any(path):
    full = os.path.join(ROOT, path)
    if path.endswith(".xlsx"):
        return pd.read_excel(full)
    return pd.read_csv(full)


def main():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    r_seed = manifest["r_seed"]

    print("== Classic datasets (homals, princals) ==")
    for name in manifest["classic_datasets"]:
        print(f"-- {name} --")
        df = load_typed_csv(name)
        if "homals" in manifest["classic_methods"]:
            run_homals(name, df, r_seed)
        if "princals" in manifest["classic_methods"]:
            run_princals(name, df, r_seed)

    print("\n== Morals runs ==")
    for run in manifest["morals_runs"]:
        print(f"-- {run['name']} --")
        df = read_any(run["file"])
        X = df.iloc[:, run["x_cols"]]
        y = df.iloc[:, run["y_col"]]
        run_morals(run["name"], X, y, r_seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
