import os
import pandas as pd

def preprocess_datasets():
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    processed_dir = os.path.join(raw_dir, "processed")

    os.makedirs(processed_dir, exist_ok=True)

    dataset_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    if not dataset_files:
        print("No .csv files found in validation/datasets/.")
        return

    print("Preprocessing datasets for validation suite...")

    for file_name in dataset_files:
        print(f"  -> Cleaning {file_name}...")
        file_path = os.path.join(raw_dir, file_name)
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"     Error reading {file_name}: {e}")
            continue

        # 1. Drop common artifact columns
        cols_to_drop = [c for c in df.columns if c.startswith("Unnamed:") or c == "X"]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # 2. Impute missing values with Mode for categorical handling fairness
        if df.isna().any().any():
            for col in df.columns:
                if df[col].isna().any():
                    # Get mode (most frequent value)
                    mode_series = df[col].mode()
                    if not mode_series.empty:
                        df[col].fillna(mode_series[0], inplace=True)
                    else:
                        # Fallback if column is entirely NaNs
                        df[col].fillna("Missing", inplace=True)

        # 2.5 Cap at 2,000 rows to ensure validation completes in reasonable time/memory
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=123).sort_index()

        # 3. Save to processed directory
        processed_path = os.path.join(processed_dir, file_name)
        df.to_csv(processed_path, index=False)
        
    print(f"Preprocessing complete. {len(dataset_files)} files saved to {processed_dir}\\n")

if __name__ == "__main__":
    preprocess_datasets()
