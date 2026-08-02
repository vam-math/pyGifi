# mypy: ignore-errors
"""
pygifi._coding — Category encoding and decoding (label mapping).

Python port of Gifi/R/coding.R + src/coding.c (Mair, De Leeuw, Groenen. GPL-3.0).
Provides utilities for mapping arbitrary labels (strings, factors) to
ordered integer codes, and back again.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple


def categorical_encode(
        data: Union[pd.Series, List, np.ndarray]) -> Tuple[np.ndarray, Dict[int, Any]]:
    """
    Encode category labels into 1-indexed numeric codes.
    """
    series = pd.Series(data)
    cat = pd.Categorical(series)
    codes = (cat.codes + 1.0).astype(float)
    mapping = {i + 1: label for i, label in enumerate(cat.categories)}
    return codes, mapping


def categorical_decode(
        codes: np.ndarray, mapping: Dict[int, Any]) -> np.ndarray:
    """
    Decode numeric codes back into original category labels.
    """
    codes_int = np.round(np.asarray(codes)).astype(int)
    labels = np.empty(len(codes_int), dtype=object)
    for i, c in enumerate(codes_int):
        labels[i] = mapping.get(c, np.nan)
    return labels


def decode(cell: Union[np.ndarray, List[int]],
           dims: Union[np.ndarray, List[int]]) -> int:
    """
    Python port of R's decode(cell, dims) / src/coding.c:DECODE.
    Converts multi-way cell coordinates to a single 1-indexed integer index.
    """
    cell = np.asarray(cell, dtype=int)
    dims = np.asarray(dims, dtype=int)
    if len(cell) != len(dims):
        raise ValueError(
            "Dimension error: cell and dims must have same length")
    if np.any(cell > dims) or np.any(cell < 1):
        raise ValueError("No such cell")

    aux = 1
    ind = 1
    for i in range(len(dims)):
        ind += aux * (cell[i] - 1)
        aux *= dims[i]
    return int(ind)


def encode(ind: int, dims: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Python port of R's encode(ind, dims) / src/coding.c:ENCODE.
    Converts a single 1-indexed integer index to multi-way cell coordinates.
    """
    dims = np.asarray(dims, dtype=int)
    n = len(dims)
    if ind < 1 or ind > np.prod(dims):
        raise ValueError("No such cell")

    cell = np.zeros(n, dtype=int)
    aux = int(ind)
    pdim = 1
    for i in range(n - 1):
        pdim *= dims[i]

    for i in range(n - 1, 0, -1):
        cell[i] = (aux - 1) // pdim
        aux -= pdim * cell[i]
        pdim //= dims[i - 1]
        cell[i] += 1

    cell[0] = aux
    return cell


def make_numeric(data: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
    """
    Convert a DataFrame with factor/character columns to a numeric NumPy array.

    R: makeNumeric(data)

    For factor columns, converts to 1-indexed integer codes matching R's behavior:
    - Numeric-valued factor levels: convert via float conversion first
    - Character/object columns: use Categorical codes (0-indexed) + 1

    Parameters
    ----------
    data : pd.DataFrame or similar
        Input data, may contain mixed types.

    Returns
    -------
    np.ndarray of shape (n_obs, n_vars), dtype float64
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    nobs, nvars = data.shape
    result = np.empty((nobs, nvars), dtype=float)

    for i in range(nvars):
        series = data.iloc[:, i]
        # Check if column is categorical or string-like (needs encoding)
        is_discrete = (
            isinstance(series.dtype, pd.CategoricalDtype) or
            pd.api.types.is_string_dtype(series) or
            series.dtype == object
        )

        if is_discrete:
            # Try converting factor levels to numeric (R:
            # as.numeric(levels(x))[x])
            try:
                # For string-like, ensure it's categorical for easier mapping
                cat = pd.Categorical(series)
                # Attempt numeric level conversion (e.g., "1.0", "2.0")
                numeric_levels = pd.to_numeric(cat.categories, errors='raise')
                # Map each observation to its numeric level
                level_map = {
                    c: float(v) for c, v in zip(
                        cat.categories, numeric_levels)}
                result[:, i] = series.map(level_map).astype(float)
            except (ValueError, TypeError):
                # Fall back to ordinal codes + 1 (R: as.numeric(x))
                cat = pd.Categorical(series)
                codes = cat.codes.astype(float)
                codes[codes == -1] = np.nan
                result[:, i] = codes + 1.0
        else:
            # Already numeric, just cast to float
            result[:, i] = series.astype(float).values

    return result
