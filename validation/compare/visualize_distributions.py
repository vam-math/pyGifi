import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from datetime import datetime

# Path Configuration
HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(HERE, "..", "datasets", "processed"))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def compute_stats(data, label):
    return {
        "Version": label,
        "Mean": np.mean(data),
        "Std": np.std(data),
        "Skewness": skew(data),
        "Kurtosis": kurtosis(data)
    }

def visualize_distributions():
    # Identify datasets
    datasets = sorted([f.replace(".csv", "") for f in os.listdir(DATASETS_DIR)
                       if f.endswith(".csv") and "transformed" not in f])
    
    if not datasets:
        print("No datasets found for visualization.")
        return

    summary_stats = []

    for ds in datasets:
        print(f"\nProcessing distributions for dataset: {ds}")
        raw_path = os.path.join(DATASETS_DIR, f"{ds}.csv")
        py_path = os.path.join(DATASETS_DIR, f"pygifi_transformed_{ds}.csv")
        r_path = os.path.join(DATASETS_DIR, f"gifi_transformed_{ds}.csv")

        if not (os.path.exists(raw_path) and os.path.exists(py_path) and os.path.exists(r_path)):
            print(f"  [SKIP] Missing one or more versions for {ds}")
            continue

        raw_df = pd.read_csv(raw_path)
        py_df = pd.read_csv(py_path)
        r_df = pd.read_csv(r_path)

        # Drop index columns if they exist
        for df in [raw_df, py_df, r_df]:
            cols_to_drop = [c for c in df.columns if "Unnamed" in c or c == "X"]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)

        # Intersection of columns
        common_cols = [c for c in raw_df.columns if c in py_df.columns and c in r_df.columns]
        
        ds_plot_dir = os.path.join(PLOTS_DIR, ds, "distributions")
        ensure_dir(ds_plot_dir)

        for col in common_cols:
            print(f"  -> Plotting {col} ...")
            
            # Extract values, dropping NaNs for stats/plotting
            v_raw = raw_df[col].dropna()
            v_py = py_df[col].dropna()
            v_r = r_df[col].dropna()

            # Ensure data is numeric
            try:
                v_raw = pd.to_numeric(v_raw)
                v_py = pd.to_numeric(v_py)
                v_r = pd.to_numeric(v_r)
            except:
                # If categorical string, we can't do histograms/stats easily this way
                # PyGifi transformed should be numeric anyway.
                # If raw is string, map to codes for visualization purposes
                if v_raw.dtype == object or v_raw.dtype.name == 'category':
                    v_raw = pd.Series(v_raw.astype('category').cat.codes)
                else:
                    continue

            # Compute stats
            s_raw = compute_stats(v_raw, "Raw")
            s_py = compute_stats(v_py, "PyGifi")
            s_r = compute_stats(v_r, "R Gifi")
            
            for s in [s_raw, s_py, s_r]:
                s["Dataset"] = ds
                s["Variable"] = col
                summary_stats.append(s)

            # Plotting
            plt.figure(figsize=(12, 6))
            
            # Use three subplots or overlaid? Professor said "3 overlaid histograms"
            # Overlaid can get messy, let's use transparency
            sns.histplot(v_raw, color="gray", label="Raw", kde=True, stat="density", alpha=0.3, element="step")
            sns.histplot(v_py, color="blue", label="PyGifi", kde=True, stat="density", alpha=0.3, element="step")
            sns.histplot(v_r, color="red", label="R Gifi", kde=True, stat="density", alpha=0.3, element="step")

            plt.title(f"Distribution Comparison: {ds} - {col}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            
            # Add stats text box
            stats_text = (
                f"Raw: skew={s_raw['Skewness']:.2f}, kurt={s_raw['Kurtosis']:.2f}\n"
                f"Py:  skew={s_py['Skewness']:.2f}, kurt={s_py['Kurtosis']:.2f}\n"
                f"R:   skew={s_r['Skewness']:.2f}, kurt={s_r['Kurtosis']:.2f}"
            )
            plt.gca().text(0.95, 0.5, stats_text, transform=plt.gca().transAxes, 
                           verticalalignment='center', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            safe_col = col.replace("/", "_").replace("\\", "_")
            plt.savefig(os.path.join(ds_plot_dir, f"{safe_col}.png"), bbox_inches='tight')
            plt.close()

    # Save summary stats CSV
    if summary_stats:
        stats_df = pd.DataFrame(summary_stats)
        # Reorder columns
        cols = ["Dataset", "Variable", "Version", "Mean", "Std", "Skewness", "Kurtosis"]
        stats_df = stats_df[cols]
        stats_df.to_csv(os.path.join(RESULTS_DIR, "distribution_stats.csv"), index=False)
        print(f"\nSaved distribution stats to {os.path.join(RESULTS_DIR, 'distribution_stats.csv')}")

if __name__ == "__main__":
    visualize_distributions()
