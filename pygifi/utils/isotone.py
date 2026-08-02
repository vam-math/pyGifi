# mypy: ignore-errors
"""
pygifi._isotone — Isotonic regression and optimal transformation.

Python port of Gifi/R/coneRegression.R + src/pava.f (Mair, De Leeuw, Groenen. GPL-3.0).

PAVA is a direct translation of the Fortran AMALGM subroutine (Algorithm AS 149,
Applied Statistics 1980, Vol.29, No.2 — Kruskal's up-and-down blocks algorithm).

Functions
---------
pava             : R amalgm1       — Pool Adjacent Violators Algorithm (AMALGM AS149)
isotone          : R isotone       — Monotone regression with 3 tie-handling modes
cone_regression  : R coneRegression — Routes to subspace / isotone / Dykstra
dykstra          : R dykstra       — Dykstra's alternating projection algorithm
"""

import numpy as np
from pygifi.core.linalg import ls_rc


try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(cache=True)
    def _pava_core(x: np.ndarray, w: np.ndarray, c: np.ndarray, m: int, k: int) -> np.ndarray:
        i = 0
        while i < m:
            if i < m - 1 and x[i] > x[i + 1]:
                ww = w[i] + w[i + 1]
                if ww > 0:
                    x[i] = (w[i] * x[i] + w[i + 1] * x[i + 1]) / ww
                else:
                    x[i] = (x[i] + x[i + 1]) / 2.0
                w[i] = ww
                c[i] = c[i] + c[i + 1]

                for loop_idx in range(i + 1, m - 1):
                    x[loop_idx] = x[loop_idx + 1]
                    w[loop_idx] = w[loop_idx + 1]
                    c[loop_idx] = c[loop_idx + 1]
                m -= 1
                if m == 1:
                    break
                if i > 0 and x[i - 1] > x[i]:
                    i -= 1
            elif i > 0 and x[i - 1] > x[i]:
                ww = w[i - 1] + w[i]
                if ww > 0:
                    x[i - 1] = (w[i - 1] * x[i - 1] + w[i] * x[i]) / ww
                else:
                    x[i - 1] = (x[i - 1] + x[i]) / 2.0
                w[i - 1] = ww
                c[i - 1] = c[i - 1] + c[i]

                for loop_idx in range(i, m - 1):
                    x[loop_idx] = x[loop_idx + 1]
                    w[loop_idx] = w[loop_idx + 1]
                    c[loop_idx] = c[loop_idx + 1]
                m -= 1
                if m == 1:
                    break
                i = max(0, i - 1)
            else:
                i += 1
        return x[:m], w[:m]
else:
    def _pava_core(x, w, c, m, k):
        i = 0
        while i < m:
            if i < m - 1 and x[i] > x[i + 1]:
                ww = w[i] + w[i + 1]
                if ww > 0:
                    x[i] = (w[i] * x[i] + w[i + 1] * x[i + 1]) / ww
                else:
                    x[i] = (x[i] + x[i + 1]) / 2.0
                w[i] = ww
                c[i] = c[i] + c[i + 1]

                x[i + 1:m - 1] = x[i + 2:m]
                w[i + 1:m - 1] = w[i + 2:m]
                c[i + 1:m - 1] = c[i + 2:m]
                m -= 1
                if m == 1:
                    break
                if i > 0 and x[i - 1] > x[i]:
                    i -= 1
            elif i > 0 and x[i - 1] > x[i]:
                ww = w[i - 1] + w[i]
                if ww > 0:
                    x[i - 1] = (w[i - 1] * x[i - 1] + w[i] * x[i]) / ww
                else:
                    x[i - 1] = (x[i - 1] + x[i]) / 2.0
                w[i - 1] = ww
                c[i - 1] = c[i - 1] + c[i]

                x[i:m - 1] = x[i + 1:m]
                w[i:m - 1] = w[i + 1:m]
                c[i:m - 1] = c[i + 1:m]
                m -= 1
                if m == 1:
                    break
                i = max(0, i - 1)
            else:
                i += 1
        return x[:m], w[:m]


def pava(xo, wo=None):
    """
    Pool Adjacent Violators Algorithm (weighted).

    Direct Python translation of Fortran AMALGM (Algorithm AS 149).
    Kruskal's up-and-down blocks algorithm. Returns isotonically
    non-decreasing sequence of same length as xo.

    Parameters
    ----------
    xo : array-like of shape (k,)
        Values to make isotonically non-decreasing.
    wo : array-like of shape (k,), optional
        Positive weights. Defaults to ones.

    Returns
    -------
    np.ndarray of shape (k,) — isotonically non-decreasing values.
    """
    xo = np.asarray(xo, dtype=float)
    k = len(xo)
    if k < 2:
        return xo.copy()
    wo = np.ones(k) if wo is None else np.asarray(wo, dtype=float)

    # Track block values, weights, and counts (sizes)
    x = xo.copy()
    w = wo.copy()
    c = np.ones(k, dtype=int)  # count of items in each block

    m = k         # active block count

    # Process blocks in JIT compiled/or normal core (returns shortened arrays)
    x_pooled, w_pooled = _pava_core(x, w, c, m, k)
    m = len(x_pooled)

    # Reconstruct xa(k) from working arrays — exact Fortran AMALGM logic (labels 14-17).
    # Accumulate original weights WO(J) until sum matches pooled block weight
    # W(I).
    tol = 1e-15
    xa = np.zeros(k)
    i1 = 0                           # start index into original observations
    for block_i in range(m):
        s = 0.0
        for j in range(i1, k):
            s += wo[j]
            xa[j] = x_pooled[block_i]
            if abs(s - w_pooled[block_i]) < tol:
                i1 = j + 1
                break

    return xa


def isotone(x, y, w=None, ties='s'):
    """
    Monotone regression of y on the order of x, with tie-handling.

    Python port of R's isotone(x, y, ties).

    Parameters
    ----------
    x : array-like of shape (n,) — predictor (may contain NaN)
    y : array-like of shape (n,) — target values
    w : array-like, optional — (unused; PAVA uses group weights internally)
    ties : str, one of 's', 'p', 't'
        's' — secondary (pool groups, PAVA on group means with group weights)
        'p' — primary (order within ties, single PAVA on all observations)
        't' — tertiary (pool groups, translate to tie-corrected values)

    Returns
    -------
    np.ndarray of shape (n,) with isotonic values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    s = np.zeros(n)

    there = np.where(~np.isnan(x))[0]
    notthere = np.where(np.isnan(x))[0]
    xthere = x[there]

    # Build groups: one group per unique value of x (sorted)
    f = np.sort(np.unique(xthere))   # sorted unique non-NaN values
    g = [np.where(x == fv)[0] for fv in f]  # indices per group
    k = len(f)

    if ties == 's':
        # Secondary: pool tied obs to group means, PAVA on group means
        gw = np.array([len(gi) for gi in g], dtype=float)
        gh = [y[gi] for gi in g]
        gm = np.array([np.sum(h) for h in gh]) / gw   # group means
        r = pava(gm, gw)
        for i in range(k):
            s[g[i]] = r[i]
        s[notthere] = y[notthere]

    elif ties == 'p':
        # Primary: order within ties by y, then single PAVA call
        ordered_vals = []
        ordered_idx = []
        for gi in g:
            yi = y[gi]
            order = np.argsort(yi)
            ordered_idx.extend(gi[order].tolist())
            ordered_vals.extend(yi[order].tolist())
        m_arr = np.array(ordered_vals, dtype=float)
        r = pava(m_arr)
        # Map back directly onto the global sequence indices
        s[ordered_idx] = r
        s[notthere] = y[notthere]

    elif ties == 't':
        # Tertiary: pool by mean, PAVA, translate: add (isotone_mean -
        # raw_mean) to each obs
        gw = np.array([len(gi) for gi in g], dtype=float)
        gh = [y[gi] for gi in g]
        gm = np.array([np.sum(h) for h in gh]) / gw
        r = pava(gm, gw)
        for i in range(k):
            s[g[i]] = y[g[i]] + (r[i] - gm[i])
        s[notthere] = y[notthere]

    else:
        raise ValueError(f"ties must be 's', 'p', or 't', got '{ties}'")

    return s


def cone_regression(data, target, basis=None, type='i', ties='s', missing='s',
                    itmax=1000, eps=1e-6):
    """
    Project target onto the transformation cone for a variable.

    Python port of R's coneRegression(data, target, basis, type, ties, missing).

    Parameters
    ----------
    data   : array-like (n,) — original variable values (may have NaN)
    target : array-like (n,) — current target (scores)
    basis  : np.ndarray (n, k), optional — basis matrix for the variable
    type   : str, one of:
        's' — subspace projection (ordinary least squares via basis)
        'c' — isotone categorical (monotone regression, no basis)
        'i' — Dykstra's iterative subspace+isotone projection
    ties   : str — tie mode for isotone ('s', 'p', 't')
    missing: str — missing value mode ('m', 'a', 's')
    itmax  : int — max iterations for Dykstra
    eps    : float — convergence for Dykstra

    Returns
    -------
    np.ndarray of shape (n,) — optimal transformed values.
    """
    data = np.asarray(data, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(data)

    if basis is None:
        basis = data[:, None].copy()

    if type == 's':
        # Subspace projection: OLS via ls_rc
        sol = ls_rc(basis, target)['solution']
        return (basis @ sol).flatten()

    elif type == 'c' and missing != 'a':
        # Categorical isotone on non-NaN subset
        there = np.where(~np.isnan(data))[0]
        notthere = np.where(np.isnan(data))[0]
        solution = np.zeros(n)
        solution[there] = isotone(x=data[there], y=target[there], ties=ties)
        if len(notthere) > 0:
            if missing == 'm':
                solution[notthere] = target[notthere]
            elif missing == 's':
                solution[notthere] = np.mean(target[notthere])
        return solution

    else:
        # Dykstra: type='i' or (type='c' and missing='a')
        return dykstra(target=target, basis=basis, data=data,
                       ties=ties, itmax=itmax, eps=eps)


def dykstra(target, basis, data, ties, itmax=1000, eps=1e-6):
    """
    Dykstra's alternating projection onto subspace ∩ isotone cone.

    Python port of R's dykstra(target, basis, data, ties, itmax, eps).

    Projects target onto the intersection of:
      - The column space of basis (subspace projection via ls_rc)
      - The isotone cone (monotone ordering on data)

    Parameters
    ----------
    target : np.ndarray (n,)
    basis  : np.ndarray (n, k)
    data   : np.ndarray (n,) — ordering variable (may have NaN)
    ties   : str — tie mode
    itmax  : int
    eps    : float

    Returns
    -------
    np.ndarray (n,) — projection onto intersection.
    """
    x0 = target.copy().astype(float)
    a = np.zeros_like(x0)
    b = np.zeros_like(x0)

    for itel in range(itmax):
        # Project onto column space of basis
        x1 = (basis @ ls_rc(basis, x0 - a)['solution']).flatten()
        a = a + x1 - x0

        # Project onto isotone cone
        x2 = isotone(data, x1 - b, ties=ties)
        b = b + x2 - x1

        # Convergence check
        xdif = np.max(np.abs(x1 - x2))
        if xdif < eps:
            break
        x0 = x2

    return (x1 + x2) / 2


def monotone_regression(Z, x_ord, weights=None):
    """
    Enforce monotone (isotone) ordering on category quantifications Z.

    Applies the Pool Adjacent Violators Algorithm (PAVA) independently
    to each column (copy) of Z, constraining category quantifications to
    be non-decreasing with respect to the ordinal rank order x_ord.

    Parameters
    ----------
    Z : np.ndarray (n_categories, n_copies)
        Category quantifications for one ordinal variable.
    x_ord : array-like of shape (n_categories,)
        Ordinal rank values for each category (e.g., [1, 2, 3]).
        Quantifications will be forced non-decreasing along this ordering.
    weights : array-like of shape (n_categories,), optional
        Per-category weights (typically observation counts n_c). Z here is
        already a per-category *mean* (H^T H)^-1 H^T X A^T, so it is the
        exact minimizer of sum_c n_c * (z_c - Z[c])^2 subject to monotonicity
        only when PAVA pools adjacent violators using these same n_c weights.
        Omitting them (equivalent to weights=1) is only correct when every
        category has the same number of observations; otherwise it silently
        returns a suboptimal (non-minimizing) monotone solution. Defaults to
        uniform weights for backward compatibility.

    Returns
    -------
    np.ndarray (n_categories, n_copies)
        Z with each column made monotone non-decreasing w.r.t. x_ord.
    """
    Z = np.asarray(Z, dtype=float)
    x_ord = np.asarray(x_ord, dtype=float)

    if Z.ndim == 1:
        Z = Z[:, None]

    order = np.argsort(x_ord, kind='stable')
    w = np.ones(Z.shape[0]) if weights is None else np.asarray(weights, dtype=float)[order]

    Z_mono = Z.copy()
    for j in range(Z.shape[1]):
        pooled = pava(Z[order, j], w)
        Z_mono[order, j] = pooled

    return Z_mono
