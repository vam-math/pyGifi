# mypy: ignore-errors
"""
pygifi.utils._cone — Cone projection module.

Python port of Gifi/R/coneRegression.R (Mair, De Leeuw, Groenen. GPL-3.0).

This module is the single authoritative implementation of all cone projections
used in the H-update step of the Gifi ALS engine.  Each column of H_j (one
optimally-scaled copy of variable j) is projected onto its admissible cone
K_js defined by the variable's measurement level and spline/degree constraints.

Cone types
----------
's'  — Subspace  : OLS projection of target onto column space of G_j (basis).
                   Used for nominal multi-copy variables and polynomial/spline
                   non-ordinal transforms. Enforces no ordering constraint.
'c'  — Categorical isotone : apply PAVA directly on the raw category means of
                   target, respecting the order of the original data values.
                   No basis — categorical indicator coding assumed upstream.
'i'  — Dykstra   : alternating projection onto (col-space of G_j) ∩ (isotone
                   cone). Used for ordinal spline / ordinal polynomial.
'm'  — Monotone spline: project onto spline subspace then enforce PAVA on the
                   fitted spline coefficients rather than the node values.
                   This is stricter than Dykstra when the spline basis has many
                   internal knots.
'l'  — Linear (metric): project onto the 1-D column space of a single linear
                   predictor; identical to 's' with a [x | 1] basis.
'n'  — NNLS cone : general inequality-constrained cone via non-negative least
                   squares (ported from R coneRegression.R general case).
                   Constraint matrix C is computed from the ordering of data
                   values: C @ h >= 0 represents h_{i} <= h_{i+1}.

Public API
----------
project_cone(target, data, basis, cone_type, ties, missing) -> np.ndarray
    Dispatch function — mirrors R's coneRegression() signature.

All functions accept 1-D targets (one copy at a time) and return 1-D arrays.
"""

import warnings
import numpy as np
from typing import Optional
from scipy.optimize import nnls
from scipy.linalg import lstsq

from pygifi.core.linalg import ls_rc
from pygifi.utils.isotone import isotone, pava, dykstra


__all__ = ['project_cone']


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subspace(target: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    OLS projection of target onto column space of basis.

    Equivalent to R: basis %*% solve(t(basis) %*% basis, t(basis) %*% target)
    Uses our ls_rc (Modified Gram-Schmidt) which is numerically robust.
    """
    sol = ls_rc(basis, target)['solution']
    return (basis @ sol).flatten()


def _categorical_isotone(data: np.ndarray, target: np.ndarray,
                          ties: str, missing: str) -> np.ndarray:
    """
    Categorical monotone projection (type='c'):
    Apply PAVA on the group means of target, grouped by the unique sorted
    values of data.  Matches R gifiTransform → coneRegression(type='c').
    """
    n = len(data)
    there = np.where(~np.isnan(data))[0]
    notthere = np.where(np.isnan(data))[0]
    solution = np.zeros(n)

    solution[there] = isotone(x=data[there], y=target[there], ties=ties)

    if len(notthere) > 0:
        if missing == 'm':
            solution[notthere] = target[notthere]
        elif missing == 's':
            solution[notthere] = np.mean(target[notthere])
        else:
            solution[notthere] = np.mean(target[notthere])

    return solution


def _dykstra_isotone_spline(target: np.ndarray, basis: np.ndarray,
                             data: np.ndarray, ties: str,
                             itmax: int, eps: float) -> np.ndarray:
    """
    Dykstra's alternating projection: col-space(basis) ∩ isotone cone.
    type='i' — used for ordinal polynomial/spline.
    Already implemented in isotone.py; called here for a unified interface.
    """
    return dykstra(target=target, basis=basis, data=data,
                   ties=ties, itmax=itmax, eps=eps)


def _monotone_spline(target: np.ndarray, basis: np.ndarray,
                     data: np.ndarray, ties: str) -> np.ndarray:
    """
    Monotone spline cone (type='m') — R coneRegression.R monotone branch.

    Algorithm:
    1. Fit target onto spline basis via lstsq → coefficients c.
    2. Apply PAVA to c to enforce c[0] <= c[1] <= ... (monotone coefficients).
    3. Reconstruct: h = basis @ c_monotone.

    This enforces the spline *coefficients* to be non-decreasing, which is a
    stronger constraint than the Dykstra approach and matches R's
    monospline branch for ordinal B-spline transformations.
    """
    # Step 1: lstsq fit
    c, _, _, _ = lstsq(basis, target, cond=None)

    # Step 2: PAVA on coefficients
    c_mono = pava(c)

    # Step 3: reconstruct
    return (basis @ c_mono).flatten()


def _linear_metric(target: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Linear/metric cone (type='l') — project target onto 1-D linear subspace.

    Builds basis = [data_centered | 1] and projects.  Matches R metric branch:
    the transformation is constrained to be a linear function of the original
    variable values.
    """
    n = len(data)
    there = ~np.isnan(data)
    if not np.any(there):
        return target.copy()

    d = data.copy()
    d[~there] = np.nanmean(d)  # impute NaN before centering
    d_c = d - np.mean(d)

    # basis: linear + intercept
    basis = np.column_stack([d_c, np.ones(n)])
    sol = ls_rc(basis, target)['solution']
    return (basis @ sol).flatten()


def _nnls_cone(target: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    General inequality-constrained cone via NNLS (type='n').

    This ports R's coneRegression.R general cone branch which uses NNLS
    with a difference-constraint matrix to enforce h_{σ(i)} <= h_{σ(i+1)}
    across sorted unique values of data.

    Algorithm (De Leeuw 2017, §4):
    ─ Let σ = argsort(data), σ-values are the sorting permutation.
    ─ Build first-difference constraint:  C h >= 0  where
        C[i, σ(i+1)] = +1,  C[i, σ(i)] = -1   for i = 0..k-2.
    ─ Equivalent dual problem has the form:
        min ||target - z - C^T u||^2  s.t. u >= 0
      solved by alternating (inner) iterations; here we use the standard
      NNLS re-parameterisation:  z = target - C^T u, u >= 0.
    ─ scipy.optimize.nnls solves min_{u>=0} ||C^T u - target||_2^2 directly
      after transposing.
    """
    n = len(data)
    there = ~np.isnan(data)

    if not np.any(there):
        return target.copy()

    # Build sort permutation over non-NaN values
    idx_present = np.where(there)[0]
    d_present = data[idx_present]
    sort_order = np.argsort(d_present, kind='stable')
    sigma = idx_present[sort_order]   # global indices, sorted by data value
    k = len(sigma)

    if k < 2:
        return target.copy()

    # Build first-difference constraint matrix C (k-1, n)
    C = np.zeros((k - 1, n))
    for i in range(k - 1):
        C[i, sigma[i + 1]] = 1.0    # +h_{σ(i+1)}
        C[i, sigma[i]] = -1.0       # -h_{σ(i)} >= 0 ↔ h_{σ(i)} <= h_{σ(i+1)}

    # Solve: min_{u >= 0} ||C^T u - target||  via NNLS
    # This gives u; reconstruct h = target - C^T u
    try:
        u, _ = nnls(C.T, target)
        h = target - C.T @ u
    except Exception:
        warnings.warn(
            "NNLS cone solver failed; falling back to categorical isotone.",
            RuntimeWarning, stacklevel=3)
        h = target.copy()

    # After NNLS, verify monotonicity over the sort order (numerical guard)
    h_sorted = h[sigma]
    if not np.all(np.diff(h_sorted) >= -1e-10):
        # If still non-monotone (numerical noise), apply final PAVA clamping
        w = np.ones(k)
        h_sorted = pava(h_sorted, w)
        h[sigma] = h_sorted

    return h


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def project_cone(
        target: np.ndarray,
        data: np.ndarray,
        basis: Optional[np.ndarray] = None,
        cone_type: str = 'i',
        ties: str = 's',
        missing: str = 's',
        itmax: int = 1000,
        eps: float = 1e-6,
) -> np.ndarray:
    """
    Project target onto the admissible cone for variable j, copy s.

    Single dispatch entry-point — mirrors R's coneRegression() signature.
    Called from gifi_engine for every (variable j, copy s) at each ALS iter.

    Parameters
    ----------
    target    : np.ndarray (n,)
        Current majorization target for this copy (H̃_js + gradient correction).
    data      : np.ndarray (n,)
        Original variable values (may contain NaN for missing).
    basis     : np.ndarray (n, k) or None
        Pre-computed spline / polynomial / indicator basis matrix. Required for
        types 's', 'i', 'm'. Ignored for types 'c', 'l', 'n'.
    cone_type : str
        's' — subspace (OLS onto basis column space)
        'c' — categorical isotone (PAVA on group means, no basis)
        'i' — Dykstra: intersection of subspace and isotone cone
        'm' — monotone spline: PAVA on spline coefficients
        'l' — linear/metric: project onto 1-D [data | 1] subspace
        'n' — NNLS general cone: difference-constrained via NNLS
    ties      : str  — tie-handling for isotone ('s', 'p', 't')
    missing   : str  — missing value strategy ('m', 's', 'a')
    itmax     : int  — max iterations for Dykstra
    eps       : float — Dykstra convergence threshold

    Returns
    -------
    np.ndarray (n,) — projected (optimally scaled) values.

    Notes
    -----
    All outputs are 1-D.  The engine normalises them afterward via
    gs_rc(normalize(center(h))) before storing in the state matrix H_j.
    """
    target = np.asarray(target, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()

    if cone_type == 's':
        if basis is None:
            basis = data[:, None].copy()
        return _subspace(target, basis)

    elif cone_type == 'c':
        return _categorical_isotone(data, target, ties=ties, missing=missing)

    elif cone_type == 'i':
        if basis is None:
            basis = data[:, None].copy()
        return _dykstra_isotone_spline(
            target, basis, data, ties=ties, itmax=itmax, eps=eps)

    elif cone_type == 'm':
        if basis is None:
            basis = data[:, None].copy()
        return _monotone_spline(target, basis, data, ties=ties)

    elif cone_type == 'l':
        return _linear_metric(target, data)

    elif cone_type == 'n':
        return _nnls_cone(target, data)

    else:
        raise ValueError(
            f"project_cone: unknown cone_type '{cone_type}'. "
            "Must be one of: 's', 'c', 'i', 'm', 'l', 'n'.")
