# mypy: ignore-errors
"""
pygifi._structures: Gifi data structure factory functions.

Python port of Gifi/R/gifiStructures.R (Mair, De Leeuw, Groenen. GPL-3.0).

All structures are plain dicts (not dataclasses) for simplicity and consistency.

Functions
---------
make_gifi_variable    : R makeGifiVariable : static variable info + basis
make_gifi_set         : R makeGifiSet      : list of gifi_variables
make_gifi             : R makeGifi         : list of gifi_sets
make_x_gifi_variable  : R xGifiVariable    : mutable ALS state per variable
make_x_gifi_set       : R xGifiSet         : list of x_gifi_variables
make_x_gifi           : R xGifi            : list of x_gifi_sets
"""

import numpy as np
from pygifi.utils.utilities import make_indicator, make_missing, center, normalize
from pygifi.core.linalg import gs_rc, ls_rc
from pygifi.utils.splines import bspline_basis


def make_gifi_variable(
        data,
        knots,
        degree,
        ordinal,
        ties,
        copies,
        missing,
        active,
        name):
    """
    Create a static GifiVariable dict: holds input data and preprocessed basis.

    Python port of R's makeGifiVariable().

    Parameters
    ----------
    data    : array-like (n,): raw column values (may contain NaN)
    knots   : array-like: interior B-spline knots (empty for categorical/polynomial)
    degree  : int: -2=orthoblock, -1=categorical, >=0=spline/polynomial
    ordinal : bool: use isotonic projection (True) or subspace (False)
    ties    : str: tie mode ('s','p','t') for isotone
    copies  : int: number of copies for multi-dimensional treatment
    missing : str: missing value mode ('m','a','s')
    active  : bool: whether variable participates in ALS
    name    : str: variable name

    Returns
    -------
    dict with keys: data, basis, qr, copies, degree, ties, missing,
                    ordinal, active, name, type
    """
    data = np.asarray(data, dtype=float)
    len(data)
    there = np.where(~np.isnan(data))[0]
    notthere = np.where(np.isnan(data))[0]
    nmis = len(notthere)

    if len(there) == 0:
        raise ValueError(
            f"gifi_variable '{name}' cannot be completely missing")

    work = data[there]  # non-missing values only

    # --- Build basis matrix (for non-missing rows) ---
    if degree == -2:
        gifi_type = 'orthoblock'
        basis = None

    elif degree == -1:
        basis = make_indicator(work)
        if basis.shape[1] < 2:
            raise ValueError(
                f"gifi_variable '{name}' must have more than one unique category/level to be scaled, but only {basis.shape[1]} was found.")
        gifi_type = 'binary' if basis.shape[1] == 2 else 'categorical'

    else:  # degree >= 0
        knots_arr = np.asarray(
            knots,
            dtype=float) if len(knots) > 0 else np.array(
            [])
        basis = bspline_basis(work, degree=degree, innerknots=knots_arr)
        gifi_type = 'polynomial' if len(knots_arr) == 0 else 'splinical'

    # --- Extend basis for missing rows ---
    if nmis > 0 and basis is not None:
        basis = make_missing(data, basis, missing)

    # --- Correct copies to not exceed basis columns - 1 ---
    if basis is not None:
        copies = min(copies, basis.shape[1] - 1)
        if copies < 1:
            copies = 1

    # --- QR decomposition of centered basis ---
    if basis is not None:
        qr = gs_rc(center(basis))
        if qr['rank'] == 0:
            raise ValueError(
                f"gifi_variable '{name}' has zero-rank basis after centering")
    else:
        qr = None

    return {
        'data': data,
        'basis': basis,
        'qr': qr,
        'copies': copies,
        'degree': degree,
        'ties': ties,
        'missing': missing,
        'ordinal': ordinal,
        'active': active,
        'name': name,
        'type': gifi_type,
    }


def make_gifi_set(
        data_cols,
        knots,
        degrees,
        ordinal,
        ties,
        copies,
        missing,
        active,
        names):
    """
    Create a GifiSet: a list of GifiVariable dicts.

    Python port of R's makeGifiSet().

    Parameters
    ----------
    data_cols : np.ndarray (n, nvars)
    All other params : list of length nvars (one entry per variable)

    Returns
    -------
    list of dicts (one GifiVariable per column)
    """
    nvars = data_cols.shape[1]
    return [
        make_gifi_variable(
            data=data_cols[:, i],
            knots=knots[i] if knots[i] is not None else [],
            degree=degrees[i],
            ordinal=ordinal[i],
            ties=ties[i],
            copies=copies[i],
            missing=missing[i],
            active=active[i],
            name=names[i],
        )
        for i in range(nvars)
    ]


def make_gifi(
        data,
        knots,
        degrees,
        ordinal,
        ties,
        copies,
        missing,
        active,
        names,
        sets):
    """
    Create a Gifi: a list of GifiSets, one per set.

    Python port of R's makeGifi().

    Parameters
    ----------
    data   : np.ndarray (n, nvars)
    sets   : list of int: set membership for each column (0-indexed)
    All other params : list of length nvars

    Returns
    -------
    list of lists: gifi[i][j] is the j-th GifiVariable in set i
    """
    sets = list(sets)
    nsets = max(sets) + 1
    result = []

    # Ensure data is a numpy array for consistent slicing
    data_arr = np.asarray(data)

    for s in range(nsets):
        k = [i for i, sv in enumerate(sets) if sv == s]
        set_data = data_arr[:, k]
        result.append(make_gifi_set(
            data_cols=set_data,
            knots=[knots[i] for i in k],
            degrees=[degrees[i] for i in k],
            ordinal=[ordinal[i] for i in k],
            ties=[ties[i] for i in k],
            copies=[copies[i] for i in k],
            missing=[missing[i] for i in k],
            active=[active[i] for i in k],
            names=[names[i] for i in k],
        ))
    return result


def make_x_gifi_variable(gifi_variable, x):
    """
    Create a mutable XGifiVariable dict: holds ALS iteration state.

    Python port of R's xGifiVariable(gifiVariable, x).
    Called while np.random.seed(123) is active (set by gifi_engine).

    Parameters
    ----------
    gifi_variable : dict: from make_gifi_variable
    x             : np.ndarray (nobs, ndim): current object scores

    Returns
    -------
    dict with keys: transform, weights, scores, quantifications
    """
    x.shape[1]
    basis = gifi_variable['basis']
    copies = gifi_variable['copies']
    nobs = len(gifi_variable['data'])
    nbas = basis.shape[1]

    # Initialize transform: column 0 = ramp, others = random
    transform = np.zeros((nobs, copies))
    transform[:, 0] = basis @ np.arange(1, nbas + 1, dtype=float)
    for i in range(1, copies):
        transform[:, i] = basis @ np.random.randn(nbas)

    # Normalize and orthogonalize
    transform = gs_rc(normalize(center(transform)))['q']
    transform.shape[1]

    # Regression onto current x
    weights = ls_rc(transform, x)['solution']       # (copies_actual, ndim)
    scores = transform @ weights                    # (nobs, ndim)
    quantifications = ls_rc(basis, scores)['solution']  # (nbas, ndim)

    return {
        'transform': transform,
        'weights': weights,
        'scores': scores,
        'quantifications': quantifications,
    }


def make_x_gifi_set(gifi_set, x):
    """
    Create an XGifiSet: a list of XGifiVariable dicts.
    Python port of R's xGifiSet().
    """
    return [make_x_gifi_variable(gv, x) for gv in gifi_set]


def make_x_gifi(gifi, x):
    """
    Create an XGifi: a list of XGifiSet lists.
    Python port of R's xGifi().
    """
    return [make_x_gifi_set(gifi_set, x) for gifi_set in gifi]
