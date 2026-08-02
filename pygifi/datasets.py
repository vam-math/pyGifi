"""
pygifi.datasets — Interfaces to load built-in example datasets from the original R Gifi package.
"""
import os
import pandas as pd

# Mapping of dataset names to their corresponding CSV files in pygifi/data/
DATASET_MAP = {
    'abc': 'ABC.csv',
    'galo': 'galo.csv',
    'hartigan': 'hartigan.csv',
    'gubell': 'gubell.csv',
    'roskam': 'roskam.csv',
    'neumann': 'neumann.csv',
    'mammals': 'mammals.csv',
    'sleeping': 'sleeping.csv',
    'house': 'house.csv',
    'senate07': 'senate07.csv',
    'small': 'small.csv',
    'wilpat2': 'WilPat2.csv'
}


def get_dataset(name: str) -> pd.DataFrame:
    """
    Load a built-in pyGifi dataset as a pandas DataFrame.

    Available datasets:
    - 'abc': ABC Customer Satisfaction (nominal)
    - 'galo': GALO school data (ordinal)
    - 'hartigan': Hartigan's Hardware (mixed)
    - 'gubell': Guttman-Bell (nominal)
    - 'roskam': Roskam personality scales (ordinal)
    - 'neumann': Neumann dataset (metric/ordinal)
    - 'mammals': Mammals sleep data (mixed)
    - 'sleeping': Sleeping bags (ordinal)
    - 'house': House data (nominal/ordinal)
    - 'senate07': 2007 Senate votes (nominal)
    - 'small': Small toy dataset for testing
    - 'wilpat2': Wilson-Patterson attitude data

    Parameters
    ----------
    name : str
        Name of the dataset (case-insensitive).
        
    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If the dataset name is unknown or the file cannot be located.
    """
    # Force lowercase for matching
    name_check = name.lower()
    
    if name_check not in DATASET_MAP:
        valid_names = ", ".join(sorted(DATASET_MAP.keys()))
        raise ValueError(
            f"Unknown dataset '{name}'. Available built-in datasets are: {valid_names}"
        )
        
    filename = DATASET_MAP[name_check]
    
    # Resolve absolute path to the data directory alongside this file
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise ValueError(
            f"Dataset file '{filename}' was not found at expected path: {filepath}"
        )
        
    # Read the dataset assuming standard comma-delimited export from R `write.csv()`
    # We drop the row index (column 0) as it is usually the R row names
    return pd.read_csv(filepath, index_col=0)
