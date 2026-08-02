
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygifi
from pygifi import Homals, Princals, Morals, plot
from datetime import datetime

print(f"pygifi version : {pygifi.__version__}")
print(f"numpy          : {np.__version__}")
print(f"pandas         : {pd.__version__}")

# ==============================================================
# CONFIGURATION
# ==============================================================

# Search paths for datasets
DATASET_DIRS = [
    "validation/datasets/processed",
    "datasets/processed",
    "validation/datasets",
    "datasets"
]

# Mapping of dataset filename to target variable for Morals
# (If not in mapping, Morals will be skipped for that dataset)
MORALS_TARGETS = {
    "bike_dataset.csv": "Purchased.Bike",
    "car_dataset.csv": "price"
}

OUTPUT_DIR = "results/master_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==============================================================
# HELPERS
# ==============================================================

def separator(title):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")

def save_plot(fig, dataset_name, model_name, plot_type):
    filename = f"{dataset_name.replace('.', '_')}_{model_name}_{plot_type}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    if isinstance(fig, plt.Figure):
        fig.savefig(filepath, dpi=150)
    else:
        # If it's an Axes, get the figure
        fig.figure.savefig(filepath, dpi=150)
    print(f"  [PLOT] Saved: {filename}")
    plt.close(fig if isinstance(fig, plt.Figure) else fig.figure)

# ==============================================================
# MODEL RUNNERS
# ==============================================================

def run_homals(df, dataset_name):
    separator(f"[{dataset_name}] RUNNING HOMALS")
    try:
        model = Homals(ndim=2, itmax=50).fit(df)
        res = model.result_
        print(f"  Loss (f): {res.get('f', 'N/A')}")
        evals = res.get('evals', [])
        print(f"  Eigenvalues: {evals}")
        
        # Plotting
        ax = plot(model, plot_type='biplot', title=f"HOMALS Biplot - {dataset_name}")
        save_plot(ax, dataset_name, "homals", "biplot")
        
        fig = plot(model, plot_type='transplot', title=f"HOMALS Transformations - {dataset_name}")
        save_plot(fig, dataset_name, "homals", "transplot")
        
        ax = plot(model, plot_type='screeplot', title=f"HOMALS Scree - {dataset_name}")
        save_plot(ax, dataset_name, "homals", "screeplot")
        
        return model
    except Exception as e:
        print(f"  [ERROR] Homals failed: {e}")
        return None

def run_princals(df, dataset_name, levels="nominal"):
    separator(f"[{dataset_name}] RUNNING PRINCALS ({levels})")
    try:
        model = Princals(ndim=2, levels=levels, itmax=50).fit(df)
        res = model.result_
        print(f"  Loss (f): {res.get('f', 'N/A')}")
        evals = res.get('evals', [])
        print(f"  Eigenvalues: {evals}")
        
        # Plotting
        ax = plot(model, plot_type='biplot', title=f"PRINCALS ({levels}) Biplot - {dataset_name}")
        save_plot(ax, dataset_name, f"princals_{levels}", "biplot")
        
        fig = plot(model, plot_type='transplot', title=f"PRINCALS ({levels}) Transformations - {dataset_name}")
        save_plot(fig, dataset_name, f"princals_{levels}", "transplot")
        
        return model
    except Exception as e:
        print(f"  [ERROR] Princals ({levels}) failed: {e}")
        return None

def run_morals(df, dataset_name, target_col):
    separator(f"[{dataset_name}] RUNNING MORALS (Target: {target_col})")
    if target_col not in df.columns:
        print(f"  [SKIP] Target column '{target_col}' not found.")
        return None
        
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # If target is string/category, convert to binary if possible or keep as coded
        if y.dtype == 'category' or y.dtype == object:
            if set(y.unique()) == {'Yes', 'No'}:
                y = (y == 'Yes').astype(float)
            else:
                y = pd.factorize(y)[0].astype(float)
        else:
            y = y.astype(float)
            
        model = Morals(itmax=50).fit(X, y)
        res = model.result_
        print(f"  SMC (R²): {res.get('smc', 'N/A')}")
        print(f"  Loss (f): {res.get('f', 'N/A')}")
        
        # Plotting (Morals has its own plot utility or use transformed values)
        from pygifi.visualization.plot import plot_morals
        fig = plot_morals(model)
        save_plot(fig, dataset_name, "morals", "transplot")
        
        return model
    except Exception as e:
        print(f"  [ERROR] Morals failed: {e}")
        return None

# ==============================================================
# MAIN LOOP
# ==============================================================

def main():
    separator("MASTER ANALYSIS START")
    
    found_any = False
    for ddir in DATASET_DIRS:
        if not os.path.exists(ddir):
            continue
            
        print(f"\nScanning directory: {ddir}")
        files = [f for f in os.listdir(ddir) if f.endswith('.csv')]
        for fname in files:
            found_any = True
            fpath = os.path.join(ddir, fname)
            print(f"\nProcessing Dataset: {fname}")
            
            try:
                df = pd.read_csv(fpath)
                # Cleaning column names for consistency
                df.columns = df.columns.str.replace(' ', '.')
                
                # Pre-processing for Homals/Princals: convert to categorical
                df_cat = df.copy()
                for col in df_cat.columns:
                    df_cat[col] = df_cat[col].astype('category')
                
                # 1. HOMALS
                run_homals(df_cat, fname)
                
                # 2. PRINCALS (Nominal)
                run_princals(df_cat, fname, levels="nominal")
                
                # 3. PRINCALS (Ordinal)
                run_princals(df_cat, fname, levels="ordinal")
                
                # 4. MORALS
                target = MORALS_TARGETS.get(fname)
                if target:
                    # Target might be in cleaned names
                    target_cleaned = target.replace(' ', '.')
                    run_morals(df_cat, fname, target_cleaned)
                else:
                    print(f"\n[{fname}] No Morals target defined. Skipping Morals.")
                    
            except Exception as e:
                print(f"  [CRITICAL ERROR] Failed to process {fname}: {e}")

    if not found_any:
        print("\n[WARNING] No datasets found in search paths.")
    
    separator("MASTER ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
