# mypy: ignore-errors
"""
pygifi._utilities: Core utility functions.

Python port of Gifi/R/gifiUtilities.R (Mair, De Leeuw, Groenen. GPL-3.0).

Functions
---------
center             : R center      : column-center a matrix
normalize          : R normalize   : column unit-normalize
make_indicator     : R makeIndicator: one-hot indicator matrix
make_missing       : R makeMissing : extend basis for missing values
reshape            : R reshape     : scalar-to-vector broadcasting
direct_sum         : R directSum   : block-diagonal matrix assembly
cor_list           : R corList     : correlation over horizontally stacked matrices
"""

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from pygifi.utils.type_inference import prepare_dataframe_with_inference

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize pandas DataFrame for Gifi modeling:
    1. Removes any index columns usually named ^Unnamed
    2. Removes any constant columns (nunique <= 1)
    3. Uses heuristics to keep metric columns numeric and coerce discrete columns
       to pandas 'category' dtype.
    """
    df_clean = df.copy()
    
    # 1. Drop ^Unnamed columns
    unnamed_cols = df_clean.columns[df_clean.columns.astype(str).str.contains("^Unnamed", regex=True)]
    if len(unnamed_cols) > 0:
        df_clean = df_clean.drop(columns=unnamed_cols)
        
    # 2. Drop constant columns
    nunique = df_clean.nunique()
    constant_cols = nunique[nunique <= 1].index
    if len(constant_cols) > 0:
        df_clean = df_clean.drop(columns=constant_cols)
        
    # 3. Infer and coerce column roles non-interactively for library use
    coerced, _, _ = prepare_dataframe_with_inference(df_clean, interactive=False)
    return coerced


def center(x):
    """
    Subtract column means (or global mean for 1D) from x.

    R: center(x): applies `z - mean(z)` column-wise.

    Parameters
    ----------
    x : np.ndarray, 1D or 2D

    Returns
    -------
    np.ndarray of same shape with zero column means.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x - np.mean(x)
    return x - x.mean(axis=0)


def normalize(x):
    """
    Divide each column (or the array if 1D) by its L2 norm.

    R: normalize(x): applies `z / sqrt(sum(z^2))` column-wise.

    Parameters
    ----------
    x : np.ndarray, 1D or 2D

    Returns
    -------
    np.ndarray of same shape with unit-norm columns.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        norm = np.sqrt(np.dot(x, x))
        return x / norm if norm > 0 else x
    norms = np.sqrt((x ** 2).sum(axis=0))
    norms[norms == 0] = 1.0  # avoid divide-by-zero
    return x / norms


def make_indicator(x):
    """
    Create a one-hot indicator matrix for the unique sorted values of x.

    R: makeIndicator(x): `ifelse(outer(x, sort(unique(x)), "=="), 1, 0)`

    Parameters
    ----------
    x : array-like of shape (n,)
        Integer or categorical values (NaN-free).

    Returns
    -------
    np.ndarray of shape (n, n_categories) with 1.0 / 0.0 values.
    """
    x = np.asarray(x)
    categories = np.sort(np.unique(x))
    return (x[:, None] == categories[None, :]).astype(float)


def make_sparse_indicator(data):
    """
    Convert categorical columns into sparse indicator matrices.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame or pd.Series
        The input data containing categorical variables.

    Returns
    -------
    dict
        A dictionary with keys 'matrices' and 'mappings'.
        'matrices' is a list of scipy.sparse.csr_matrix objects for each column.
        'mappings' stores the category-to-index mapping for each column.
    """
    matrices = []
    mappings = []

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            series = data[col]
            categories = np.sort(series.dropna().unique())
            mapping = {cat: i for i, cat in enumerate(categories)}

            # Map values to indices (missing values become -1 and are omitted from sparse matrix)
            indices = series.map(mapping).fillna(-1).astype(int)

            valid_mask = indices != -1
            row_ind = np.where(valid_mask)[0]
            col_ind = indices[valid_mask].values

            n_samples = len(series)
            n_cats = len(categories)

            mat = csr_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(n_samples, n_cats))

            matrices.append(mat)
            mappings.append(mapping)

    else:  # np.ndarray
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape

        for j in range(n_features):
            col_data = data[:, j]
            # Handle float NaNs cleanly if strings/objects aren't used
            if np.issubdtype(col_data.dtype, np.number):
                valid_mask = ~np.isnan(col_data)
            else:
                valid_mask = col_data != np.array(None)

            valid_data = col_data[valid_mask]
            categories = np.sort(np.unique(valid_data))
            mapping = {cat: i for i, cat in enumerate(categories)}

            row_ind = np.where(valid_mask)[0]
            col_ind = np.array([mapping[val] for val in valid_data])

            n_cats = len(categories)

            mat = csr_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(n_samples, n_cats))

            matrices.append(mat)
            mappings.append(mapping)

    return {'matrices': matrices, 'mappings': mappings}


def make_missing(data, basis, missing):
    """
    Extend the basis matrix to handle missing (NaN) values.

    R: makeMissing(data, basis, missing)

    Parameters
    ----------
    data : array-like of shape (n,)
        Original variable values; NaN indicates missing.
    basis : np.ndarray of shape (n_nonmissing, k)
        Basis matrix for non-missing rows.
    missing : str
        One of:
        'm': each missing obs gets its own diagonal column (basis grows to k+nmis)
        'a': missing rows get 1/k uniform imputation (basis stays k)
        's': missing rows share one extra column (basis grows to k+1)

    Returns
    -------
    np.ndarray of shape (n, k'), where k' depends on mode.
    """
    data = np.asarray(data, dtype=float)
    basis = np.asarray(basis, dtype=float)
    there = np.where(~np.isnan(data))[0]
    notthere = np.where(np.isnan(data))[0]
    nmis = len(notthere)
    nobs = len(data)
    k = basis.shape[1]

    if missing == 'm':
        abasis = np.zeros((nobs, k + nmis))
        abasis[there, :k] = basis
        abasis[np.ix_(notthere, np.arange(k, k + nmis))] = np.eye(nmis)
        return abasis

    elif missing == 'a':
        abasis = np.zeros((nobs, k))
        abasis[there, :] = basis
        abasis[notthere, :] = 1.0 / k
        return abasis

    elif missing == 's':
        abasis = np.zeros((nobs, k + 1))
        abasis[there, :k] = basis
        abasis[notthere, k] = 1.0
        return abasis

    else:
        raise ValueError(f"missing must be 'm', 'a', or 's', got '{missing}'")


def reshape(x, n):
    """
    Broadcast a scalar x to a list of length n, or return x as-is if already length n.

    R: reshape(x, n): `if (length(x) == 1) rep(x, n) else x`

    Parameters
    ----------
    x : scalar or list/array-like
    n : int

    Returns
    -------
    list of length n
    """
    if np.isscalar(x):
        return [x] * n
    x = list(x)
    if len(x) == 1:
        return x * n
    if len(x) != n:
        raise ValueError(f"reshape: x has length {len(x)}, expected 1 or {n}")
    return x


def direct_sum(matrices):
    """
    Construct a block-diagonal matrix from a list of matrices.

    R: directSum(x)

    Parameters
    ----------
    matrices : list of np.ndarray

    Returns
    -------
    np.ndarray: block-diagonal concatenation of all matrices.
    """
    return block_diag(*matrices)


def cor_list(matrices):
    """
    Stack matrices horizontally and compute the correlation matrix.

    R: corList(x): `cor(cbind(x[[1]], x[[2]], ...))`

    Parameters
    ----------
    matrices : list of np.ndarray, each of shape (n, k_i)

    Returns
    -------
    np.ndarray: correlation matrix of shape (sum(k_i), sum(k_i))
    """
    h = np.concatenate(matrices, axis=1)
    # Centering + unit-normalizing each column first makes the inner product
    # h.T @ h exactly equal cor(cbind(...)), regardless of whether R
    # the caller's columns were already centered/normalized (Gifi transforms
    # already are, so this is a no-op for them up to floating point).
    h = normalize(center(h))
    return h.T @ h


def svd_orthogonalize(X):
    """
    Orthogonalize the object scores matrix X using SVD to enforce X^T X = I.

    Given the SVD of the centered X = U S V^T, the orthogonalized matrix is U V^T.
    The output X satisfies X^T X = I.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_dimensions)
        Object scores matrix to orthogonalize.

    Returns
    -------
    np.ndarray
        Orthogonalized object scores matrix.
    """
    X = np.asarray(X)

    # Center the matrix
    X_centered = X - np.mean(X, axis=0)

    # Compute SVD: X = U S V^T
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Reconstruct orthogonal matrix: U V^T satisfies (U V^T)^T (U V^T) = I
    X_ortho = U @ Vt

    return X_ortho
