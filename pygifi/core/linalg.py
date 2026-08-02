# mypy: ignore-errors
"""
pygifi._linalg — Linear algebra primitives.

Python port of Gifi/R/GramSchmidt.R + src/gsC.c (Mair, De Leeuw, Groenen. GPL-3.0).

CRITICAL: The C implementation uses MODIFIED GRAM-SCHMIDT (MGS), NOT Householder QR.
DO NOT replace with scipy.linalg.qr — it uses Householder reflections and produces
different Q columns and a different pivot ordering strategy.

This module is a direct NumPy translation of gsC.c.

Functions
---------
gs_rc    : R gsRC  — column-pivoted Modified Gram-Schmidt orthogonalization
ls_rc    : R lsRC  — least squares via gs_rc
null_rc  : R nullRC — null space via gs_rc
ginv_rc  : R ginvRC — Moore-Penrose pseudo-inverse via gs_rc
"""

import numpy as np
from scipy.linalg import solve_triangular


def gs_rc(x, eps=1e-10):
    """
    Column-pivoted Modified Gram-Schmidt orthogonalization.

    Direct NumPy translation of src/gsC.c (gsC function).

    The C algorithm:
    - Loops over columns (jwork), orthogonalizing each against previous
    - If the remaining norm is <= eps: column is rank-deficient; swap it
      with the last active column (rk-1), decrement rk (active rank).
      Swapped columns accumulate at the end of x and r.
    - Normalization is by dividing by the positive norm → Q columns always
      have positive diagonal elements (no sign ambiguity unlike Householder).

    Parameters
    ----------
    x : np.ndarray of shape (n, m)
        Input matrix. Will NOT be modified (copy made internally).
    eps : float, default=1e-10
        Rank threshold (columns with norm <= eps are rank-deficient).

    Returns
    -------
    dict with keys:
        'q'     : np.ndarray (n, rank) orthonormal columns
        'r'     : np.ndarray (rank, m) upper triangular factor
        'rank'  : int — numerical rank
        'pivot' : list[int] — column permutation (0-indexed)
    """
    x = np.array(x, dtype=float, copy=True)
    n, m = x.shape
    r = np.zeros((m, m))
    pivot = list(range(m))  # 0-indexed, matches C's 1-indexed - 1
    rk = m
    jwork = 0

    while jwork < rk:
        # Orthogonalize column jwork against all previous (MGS inner loop vectorized)
        if jwork > 0:
            # x[:, :jwork] is shape (n, jwork)
            # x[:, jwork] is shape (n,)
            # s_vec is shape (jwork,)
            s_vec = x[:, :jwork].T @ x[:, jwork]
            r[:jwork, jwork] = s_vec
            x[:, jwork] -= x[:, :jwork] @ s_vec

        # Squared norm of remaining column
        s = float(x[:, jwork] @ x[:, jwork])

        if s > eps:
            # Full-rank column: normalize and advance
            s = np.sqrt(s)
            r[jwork, jwork] = s
            x[:, jwork] /= s
            jwork += 1
        else:
            # Rank-deficient: swap column jwork with last active column (rk-1)
            if jwork != rk - 1:
                x[:, [jwork, rk - 1]] = x[:, [rk - 1, jwork]]
                r[:, [jwork, rk - 1]] = r[:, [rk - 1, jwork]]
                pivot[jwork], pivot[rk - 1] = pivot[rk - 1], pivot[jwork]
            rk -= 1

    return {
        'q': x[:, :rk],
        'r': r[:rk, :],
        'rank': rk,
        'pivot': pivot,
    }


def ls_rc(x, y, eps=1e-10):
    """
    Least squares via column-pivoted Modified Gram-Schmidt.

    Python port of R's lsRC(x, y).

    Parameters
    ----------
    x : np.ndarray of shape (n, m)
    y : np.ndarray of shape (n,) or (n, k)
    eps : float, default=1e-10

    Returns
    -------
    dict with keys:
        'solution'  : np.ndarray (m, k) — least squares solution
        'residuals' : np.ndarray (n, k) — residuals y - x @ solution
        'minssq'    : float — sum of squared residuals
        'nullspace' : np.ndarray (m, ?) — null space basis
        'rank'      : int
        'pivot'     : list[int]
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] == 1 and y.shape[1] > 1:
        y = y.T  # handle 1D input transposed
    orig_y = np.asarray(y, dtype=float)
    if orig_y.ndim == 1:
        y_2d = orig_y[:, None]
    else:
        y_2d = orig_y

    h = gs_rc(x, eps)
    rank = h['rank']
    q = h['q']
    r_full = h['r']
    pivot = h['pivot']
    m = x.shape[1]

    # p = reorder back to original column order
    p = np.argsort(pivot)

    # leading rank×rank upper triangular submatrix
    a = r_full[:rank, :rank]
    v = r_full[:rank, rank:]        # right part

    u = q.T @ y_2d            # (l, k)
    b = solve_triangular(a, u, lower=False)  # (l, k)

    residuals = y_2d - q @ u  # (n, k)
    minssq = float(np.sum(residuals ** 2))

    # Pad b with zeros for rank-deficient rows, then reorder
    b_full = np.vstack([b, np.zeros((m - rank, b.shape[1]))])
    solution = b_full[p, :]   # reorder by inverse pivot

    # Null space
    if rank == m:
        nullspace = np.zeros((m, 1))
    else:
        ns_top = -solve_triangular(a, v, lower=False)
        ns_bottom = np.eye(m - rank)
        nullspace_raw = np.vstack([ns_top, ns_bottom])
        nullspace = nullspace_raw[p, :]

    return {
        'solution': solution,
        'residuals': residuals,
        'minssq': minssq,
        'nullspace': nullspace,
        'rank': rank,
        'pivot': pivot,
    }


def null_rc(x, eps=1e-10):
    """
    Null space of x via Modified Gram-Schmidt.

    Python port of R's nullRC(x).

    Parameters
    ----------
    x : np.ndarray of shape (n, m)
    eps : float

    Returns
    -------
    np.ndarray (m, ?) — null space basis, or zeros(m,1) if full rank.
    """
    h = gs_rc(x, eps=eps)
    rank = h['rank']
    m = x.shape[1]
    if rank == m:
        return np.zeros((m, 1))
    r = h['r']
    p = np.argsort(h['pivot'])
    t = r[:rank, :rank]
    s = r[:rank, rank:]
    ns_top = -solve_triangular(t, s, lower=False)
    ns_bottom = np.eye(m - rank)
    nullspace_raw = np.vstack([ns_top, ns_bottom])
    nullspace = nullspace_raw[p, :]
    return gs_rc(nullspace, eps=eps)['q']


def ginv_rc(x, eps=1e-10):
    """
    Moore-Penrose pseudo-inverse via Modified Gram-Schmidt.

    Python port of R's ginvRC(x).

    Parameters
    ----------
    x : np.ndarray of shape (n, m)
    eps : float

    Returns
    -------
    np.ndarray (m, n) — pseudo-inverse of x.
    """
    h = gs_rc(x, eps)
    p = np.argsort(h['pivot'])
    q = h['q']
    s = h['r']
    # z = S.T @ inv(S @ S.T) @ Q.T
    sst = s @ s.T
    z = s.T @ np.linalg.solve(sst, q.T)
    return z[p, :]
