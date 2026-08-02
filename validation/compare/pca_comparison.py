import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from datetime import datetime

# Path Configuration
HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(os.path.join(HERE, "..", "datasets", "processed"))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def tucker_congruence(A, B):
    """Compute Tucker's Congruence Coefficient between two loaders matrices."""
    # A, B shape (n_features, n_components)
    congruence = []
    for i in range(A.shape[1]):
        a = A[:, i]
        b = B[:, i]
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom == 0:
            congruence.append(0.0)
        else:
            # Handle sign ambiguity: congruence is absolute or we check flip
            val = np.sum(a * b) / denom
            congruence.append(abs(val))
    return np.array(congruence)

def run_pca_on_df(df, n_components=2):
    # Standardize first
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return {
        "evals": pca.explained_variance_,
        "var_ratio": pca.explained_variance_ratio_,
        "cum_var": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "scores": scores
    }

def pca_comparison():
    datasets = sorted([f.replace(".csv", "") for f in os.listdir(DATASETS_DIR)
                       if f.endswith(".csv") and "transformed" not in f])
    
    if not datasets:
        print("No datasets found for PCA comparison.")
        return

    results_summary = []

    for ds in datasets:
        print(f"\nProcessing PCA Comparison for dataset: {ds}")
        raw_path = os.path.join(DATASETS_DIR, f"{ds}.csv")
        py_path = os.path.join(DATASETS_DIR, f"pygifi_transformed_{ds}.csv")
        r_path = os.path.join(DATASETS_DIR, f"gifi_transformed_{ds}.csv")

        if not (os.path.exists(raw_path) and os.path.exists(py_path) and os.path.exists(r_path)):
            print(f"  [SKIP] Missing one or more versions for {ds}")
            continue

        raw_df = pd.read_csv(raw_path)
        py_df = pd.read_csv(py_path)
        r_df = pd.read_csv(r_path)

        # Basic cleaning - ensure only numeric columns are used for PCA
        for df in [raw_df, py_df, r_df]:
            cols_to_drop = [c for c in df.columns if "Unnamed" in c or c == "X"]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
            
        # For Raw, we must handle categorical strings by encoding for the sake of "baseline PCA"
        for col in raw_df.columns:
            if raw_df[col].dtype == object or raw_df[col].dtype.name == 'category':
                raw_df[col] = raw_df[col].astype('category').cat.codes

        # Ensure all columns are numeric
        raw_df = raw_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        py_df = py_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        r_df = r_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Align columns
        common_cols = sorted(list(set(raw_df.columns).intersection(set(py_df.columns)).intersection(set(r_df.columns))))
        raw_df = raw_df[common_cols]
        py_df = py_df[common_cols]
        r_df = r_df[common_cols]

        n_comp = min(len(common_cols), 5) # compare up to 5 components
        
        pca_raw = run_pca_on_df(raw_df, n_components=n_comp)
        pca_py = run_pca_on_df(py_df, n_components=n_comp)
        pca_r = run_pca_on_df(r_df, n_components=n_comp)

        # 1. Eigenvalues Comparison
        ev_diff = np.abs(pca_py["evals"] - pca_r["evals"])
        ev_max_diff = np.max(ev_diff)
        
        # 2. Loadings (Tucker Congruence)
        tucker = tucker_congruence(pca_py["loadings"], pca_r["loadings"])
        tucker_min = np.min(tucker)

        # 3. Procrustes (Scores)
        # Use first 2 components for Procrustes
        mtx1, mtx2, disparity = procrustes(pca_py["scores"][:, :2], pca_r["scores"][:, :2])
        pro_r2 = 1 - disparity

        # 4. Save statistics
        results_summary.append({
            "Dataset": ds,
            "Max_EV_Diff": ev_max_diff,
            "Min_Tucker": tucker_min,
            "Procrustes_R2": pro_r2,
            "Py_Var_Expl": pca_py["cum_var"][-1]
        })

        # --- Plotting ---
        ds_plot_dir = os.path.join(PLOTS_DIR, ds, "pca")
        ensure_dir(ds_plot_dir)

        # A. Scree Plots
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, n_comp+1), pca_raw["evals"], 'go--', label='Raw')
        plt.plot(range(1, n_comp+1), pca_py["evals"], 'bo-', label='PyGifi')
        plt.plot(range(1, n_comp+1), pca_r["evals"], 'ro-', label='R Gifi')
        plt.title(f"Scree Plot: {ds}")
        plt.xlabel("Component")
        plt.ylabel("Eigenvalue")
        plt.legend()
        plt.savefig(os.path.join(ds_plot_dir, "scree_plot.png"))
        plt.close()

        # B. Biplots (Scores PC1 vs PC2)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        titles = ["Raw PCA", "PyGifi PCA", "R Gifi PCA"]
        datas = [pca_raw["scores"], pca_py["scores"], pca_r["scores"]]
        colors = ["green", "blue", "red"]

        for ax, title, data, color in zip(axes, titles, datas, colors):
            ax.scatter(data[:, 0], data[:, 1], alpha=0.5, c=color)
            ax.set_title(title)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        plt.tight_layout()
        plt.savefig(os.path.join(ds_plot_dir, "biplots.png"))
        plt.close()

        # C. Cumulative Variance
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, n_comp+1), pca_raw["cum_var"], 'go--', label='Raw')
        plt.plot(range(1, n_comp+1), pca_py["cum_var"], 'bo-', label='PyGifi')
        plt.plot(range(1, n_comp+1), pca_r["cum_var"], 'ro-', label='R Gifi')
        plt.title(f"Cumulative Variance Explained: {ds}")
        plt.xlabel("Component")
        plt.ylabel("Cumulative Variance")
        plt.legend()
        plt.savefig(os.path.join(ds_plot_dir, "variance_explained.png"))
        plt.close()

    # Save summary report
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(os.path.join(RESULTS_DIR, "pca_comparison_summary.csv"), index=False)
        print(f"\nSaved PCA comparison summary to {os.path.join(RESULTS_DIR, 'pca_comparison_summary.csv')}")

        # Console report
        print("\n" + "="*60)
        print("  PCA Structural Accuracy Report")
        print("="*60)
        for res in results_summary:
            print(f"\n[{res['Dataset']}]")
            print(f"  Eigenvalue Max Diff : {res['Max_EV_Diff']:.6f} (Target < 1e-4)")
            print(f"  Min Tucker Congruence: {res['Min_Tucker']:.4f} (Target > 0.95)")
            print(f"  Procrustes R2       : {res['Procrustes_R2']:.4f} (Target > 0.99)")
        print("\n" + "="*60)

if __name__ == "__main__":
    pca_comparison()
