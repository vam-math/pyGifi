# mypy: ignore-errors
"""
pygifi._splines: B-spline basis functions and knot generation.

Python port of Gifi/R/splineBasis.R + Gifi/R/knotsGifi.R + src/splinebasis.c
(Mair, De Leeuw, Groenen. GPL-3.0).

The C splinebasis.c implements the Cox-de Boor recursion for B-spline evaluation.
This module uses scipy.interpolate.BSpline.design_matrix (same algorithm) with
a special-case fix for the right endpoint (x == knots[-1]).

Functions
---------
bspline_basis    : R bsplineBasis: B-spline design matrix
knots_gifi       : R knotsGifi   : Knot sequence generation (Q/R/E/D types)
"""

import numpy as np
from scipy.interpolate import BSpline
from pygifi.utils.coding import make_numeric
import pandas as pd


def bspline_basis(x, degree, innerknots, lowknot=None, highknot=None, keep_cols=None):
    """
    Compute B-spline basis design matrix.

    Python port of R's bsplineBasis() + C splinebasis().
    Uses the same Cox-de Boor recursion as the C implementation.

    Parameters
    ----------
    x : array-like of shape (n,)
        Evaluation points.
    degree : int
        B-spline degree (e.g., 1=linear, 2=quadratic, 3=cubic).
    innerknots : array-like
        Interior knot locations (sorted; may be empty).
    lowknot : float, optional
        Left boundary knot. Defaults to min(x, innerknots).
    highknot : float, optional
        Right boundary knot. Defaults to max(x, innerknots).
    keep_cols : array-like of int, optional
        Indices of columns (basis functions) to retain. Used for out-of-sample
        projection where we must keep exactly the same basis functions as
        the training set, even if they are zero-sum in the test set.

    Returns
    -------
    np.ndarray of shape (n, nf) where nf = len(innerknots) + degree + 1.
    Zero-sum columns are removed (using the same rule as R's `which(colSums(basis) > 0)`),
    UNLESS `keep_cols` is specified, in which case those columns are returned.
    """
    x = np.asarray(x, dtype=float)
    innerknots = np.sort(np.unique(np.asarray(innerknots, dtype=float)))

    if lowknot is None:
        lowknot = min(x.min(), innerknots.min()) if len(
            innerknots) > 0 else x.min()
    if highknot is None:
        highknot = max(x.max(), innerknots.max()) if len(
            innerknots) > 0 else x.max()

    # R: knots = c(rep(lowknot, degree+1), innerknots, rep(highknot, degree+1))
    knots = np.concatenate([
        np.repeat(lowknot, degree + 1),
        innerknots,
        np.repeat(highknot, degree + 1),
    ])

    len(x)
    nf = len(innerknots) + degree + 1  # number of basis functions

    # Evaluate using scipy BSpline design matrix
    # scipy requires knots sorted and x within [knots[k], knots[m-k]]
    try:
        basis = BSpline.design_matrix(x, knots, degree).toarray()
    except Exception:
        # Fallback: manual de Boor evaluation matching C splinebasis.c
        basis = _deboor_basis(x, knots, degree, nf)

    # Special case: right endpoint x == highknot → last basis = 1, others = 0
    right_mask = x == highknot
    if np.any(right_mask):
        basis[right_mask, :] = 0.0
        basis[right_mask, nf - 1] = 1.0

    if keep_cols is not None:
        basis = basis[:, keep_cols]
    else:
        # Remove zero-sum columns (R: `basis[, which(colSums(basis) > 0)]`)
        nonzero_cols = np.where(basis.sum(axis=0) > 0)[0]
        basis = basis[:, nonzero_cols]

    return basis


def _deboor_basis(x, knots, degree, nf):
    """
    Fallback: Direct de Boor recursion, matching C src/splinebasis.c.

    Used if scipy.BSpline.design_matrix fails (e.g., boundary issues).
    """
    n = len(x)
    basis = np.zeros((n, nf))
    for i in range(n):
        if x[i] == knots[-1]:
            basis[i, nf - 1] = 1.0
        else:
            for j in range(nf):
                basis[i, j] = _bs(len(knots), j + 1, degree + 1, x[i], knots)
    return basis


def _bs(nknots, j, updegree, x, knots):
    """Recursive de Boor basis function evaluation (mirrors C `bs` function)."""
    if updegree == 1:
        return 1.0 if knots[j - 1] <= x < knots[j] else 0.0
    else:
        temp1 = 0.0
        denom1 = knots[j + updegree - 2] - knots[j - 1]
        if denom1 > 0:
            temp1 = (x - knots[j - 1]) / denom1
        temp2 = 0.0
        denom2 = knots[j + updegree - 1] - knots[j]
        if denom2 > 0:
            temp2 = (knots[j + updegree - 1] - x) / denom2
        y1 = _bs(nknots, j, updegree - 1, x, knots)
        y2 = _bs(nknots, j + 1, updegree - 1, x, knots)
        return temp1 * y1 + temp2 * y2


def knots_gifi(x, type='Q', nknots=None, n=None,
               xdegrees=None, ydegrees=None, ordinal=None):
    """
    Generate knot sequences for B-spline basis.

    Python port of R's knotsGifi(x, type, n).

    Matches R signature exactly:
        knotsGifi(x, type="Q", n=NULL, xdegrees=NULL, ydegrees=NULL)

    Parameters
    ----------
    x : array-like or pd.DataFrame
        Input data. If DataFrame, processes each column.
    type : str, one of:
        'Q': Quantile knots (n+2 quantile pts, drop endpoints)
        'R': Regular equally-spaced knots over data range
        'E': No interior knots (empty list per column)
        'D': Knots at each unique data value (drop min/max)
    nknots : int or None, default=None
        Canonical alias (using R's `n` parameter name in knotsGifi).
        Number of interior knots. If None, dynamically set per column:
        max(1, len(unique_values) // 3).
    n : int or None, default=None
        Legacy alias for nknots (kept for backward compatibility).
    xdegrees : ignored: degree is set upstream in gifi_engine.
    ydegrees : ignored
    ordinal   : ignored

    Returns
    -------
    list of np.ndarray: one interior-knot array per column.
    """
    # Resolve nknots / n alias
    n_eff_global = nknots if nknots is not None else n

    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    xnum = make_numeric(x)
    ncols = xnum.shape[1]

    if type == 'Q':
        out = []
        for i in range(ncols):
            col = xnum[:, i]
            n_col = n_eff_global if n_eff_global is not None else max(
                1, len(np.unique(col[~np.isnan(col)])) // 3)
            n_pts = n_col + 2
            probs = np.linspace(0, 1, max(2, n_pts))
            y = np.nanquantile(col, probs)
            out.append(y[1:-1])

    elif type == 'R':
        out = []
        for i in range(ncols):
            col = xnum[:, i]
            n_col = n_eff_global if n_eff_global is not None else max(
                1, len(np.unique(col[~np.isnan(col)])) // 3)
            n_pts = n_col + 2
            y = np.linspace(np.nanmin(col), np.nanmax(col), max(2, n_pts))
            out.append(y[1:-1])

    elif type == 'E':
        out = [np.array([]) for _ in range(max(1, ncols))]

    elif type == 'D':
        def do_d(col):
            y = np.sort(np.unique(col[~np.isnan(col)]))
            return y[1:-1]  # drop min and max endpoints
        out = [do_d(xnum[:, i]) for i in range(ncols)]

    else:
        raise ValueError(f"type must be 'Q', 'R', 'E', or 'D', got '{type}'")

    return out
