# ==============================================================
# PYTHON PYGIFI TRANSFORMATION SCRIPT
# ==============================================================

import pandas as pd
import numpy as np
import pygifi
from pygifi import Princals
import sys
import os

# --------------------------------------------------------------
# SETUP LOGGING
# --------------------------------------------------------------
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

os.makedirs("validation/results", exist_ok=True)
tee = Tee("validation/results/python_master_report.txt", "w")

print("\n============================================")
print("Finding Datasets")
print("============================================")

DATA_DIR = "validation/datasets/processed/"
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and "transformed" not in f])

if not csv_files:
    print("No datasets found in", DATA_DIR)
    sys.exit(0)

print(f"Found {len(csv_files)} datasets: {csv_files}")

for ds_file in csv_files:
    print("\n" + "="*60)
    print(f"PROCESSING DATASET: {ds_file}")
    print("="*60)

    DATA_PATH = os.path.join(DATA_DIR, ds_file)
    df = pd.read_csv(DATA_PATH)

    # Clean indices
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

    # convert to categorical
    for col in df.columns:
        df[col] = df[col].astype("category")

    print("\nRows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nFirst 5 rows:")
    print(df.head())

    # --------------------------------------------------------------
    # RUN PRINCALS
    # --------------------------------------------------------------
    print("\n============================================")
    print("Running PRINCALS")
    print("============================================")

    model = Princals(ndim=2, r_seed=123)
    model.fit(df)

    result = model.result_

    print("\nEigenvalues:")
    print(result["evals"])

    print("\nLoadings:")
    print(result["loadings"])

    # --------------------------------------------------------------
    # CATEGORY QUANTIFICATIONS
    # --------------------------------------------------------------
    print("\n============================================")
    print("Category Quantifications (Dimension 1)")
    print("============================================")

    quantifications = result["quantifications"]

    for col, q in zip(df.columns, quantifications):
        print("\n----------------------------------")
        print("Variable:", col)
        print("----------------------------------")

        categories = df[col].cat.categories
        values = q[:,0]

        for cat, val in zip(categories, values):
            print(f"{cat:20s} -> {val:.9f}")

    # --------------------------------------------------------------
    # BUILD TRANSFORMED DATASET
    # --------------------------------------------------------------
    print("\n============================================")
    print("Building Transformed Dataset")
    print("============================================")

    df_transformed = pd.DataFrame(result["transform"], index=df.index, columns=df.columns)

    print("\nFirst 10 rows of transformed dataset:")
    print(df_transformed.head(10))

    out_file = os.path.join(DATA_DIR, f"pygifi_transformed_master_{ds_file}")
    df_transformed.to_csv(out_file, index=False)
    print(f"\nSaved file: {out_file}")
