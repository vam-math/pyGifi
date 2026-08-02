# ==============================================================
# PYTHON PYGIFI TRANSFORMATION SCRIPT
# ==============================================================

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pygifi import Princals
from pygifi.utils.type_inference import prepare_dataframe_with_inference


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


VALIDATION_DIR = os.path.join(ROOT, "validation")
RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")
DATA_DIR = os.path.join(VALIDATION_DIR, "datasets", "processed")
os.makedirs(RESULTS_DIR, exist_ok=True)
tee = Tee(os.path.join(RESULTS_DIR, "python_master_report.txt"), "w")

print("\n============================================")
print("Finding Datasets")
print("============================================")

csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "transformed" not in f])

if not csv_files:
    print("No datasets found in", DATA_DIR)
    sys.exit(0)

print(f"Found {len(csv_files)} datasets: {csv_files}")

for ds_file in csv_files:
    print("\n" + "=" * 60)
    print(f"PROCESSING DATASET: {ds_file}")
    print("=" * 60)

    data_path = os.path.join(DATA_DIR, ds_file)
    df = pd.read_csv(data_path)
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

    interactive = bool(sys.stdin.isatty())
    prepared_df, resolved_kinds, inferred = prepare_dataframe_with_inference(
        df,
        interactive=interactive,
    )

    print("\nColumn type decisions:")
    for col in prepared_df.columns:
        info = inferred[str(col)]
        print(
            f"  {col}: {resolved_kinds[str(col)]} "
            f"(inferred={info.kind}; unique={info.unique_count})"
        )

    print("\nRows:", prepared_df.shape[0])
    print("Columns:", prepared_df.shape[1])

    print("\nFirst 5 rows:")
    print(prepared_df.head())

    print("\n============================================")
    print("Running PRINCALS")
    print("============================================")

    model = Princals(ndim=2, r_seed=123)
    model.fit(prepared_df)

    result = model.result_

    print("\nEigenvalues:")
    print(result["evals"])

    print("\nLoadings:")
    print(result["loadings"])

    print("\n============================================")
    print("Category Quantifications (Dimension 1)")
    print("============================================")

    quantifications = result["quantifications"]

    for col, q in zip(prepared_df.columns, quantifications):
        print("\n----------------------------------")
        print("Variable:", col)
        print("----------------------------------")

        if str(prepared_df[col].dtype) == "category":
            categories = prepared_df[col].cat.categories
        else:
            categories = sorted(prepared_df[col].dropna().unique())
        values = q[:, 0]

        for cat, val in zip(categories, values):
            print(f"{str(cat):20s} -> {val:.9f}")

    print("\n============================================")
    print("Building Transformed Dataset")
    print("============================================")

    df_transformed = pd.DataFrame(result["transform"], index=prepared_df.index, columns=prepared_df.columns)

    print("\nFirst 10 rows of transformed dataset:")
    print(df_transformed.head(10))

    out_file = os.path.join(DATA_DIR, f"pygifi_transformed_master_{ds_file}")
    df_transformed.to_csv(out_file, index=False)
    print(f"\nSaved file: {out_file}")
