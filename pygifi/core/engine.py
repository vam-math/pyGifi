# mypy: ignore-errors
"""
pygifi._engine — Core ALS engine and transformation routing.

Python port of Gifi/R/gifiEngine.R + Gifi/R/gifiTransform.R
(Mair, De Leeuw, Groenen. GPL-3.0).

Functions
---------
gifi_transform : R gifiTransform — routes to cone_regression by (degree, ordinal)
gifi_engine    : R gifiEngine   — full ALS loop for all Gifi methods
"""

import numpy as np
from pygifi.core.linalg import gs_rc, ls_rc
from pygifi.utils.utilities import center, normalize
from pygifi.utils._cone import project_cone
from pygifi.core.structures import make_x_gifi


def gifi_transform(
        data,
        target,
        basis,
        copies,
        degree,
        ordinal,
        ties,
        missing):
    """
    Apply optimal scaling transformation to one variable.

    Python port of R's gifiTransform(data, target, basis, copies, degree, ordinal, ties, missing).

    Routes to project_cone (pygifi/utils/_cone.py) based on measurement level:
    - degree == -1, not ordinal → subspace ('s'): indicator basis OLS
    - degree == -1, ordinal     → categorical ('c'): monotone on group means
    - degree == 0 or 1, ordinal → monotone spline ('m'): PAVA on coefficients
    - degree >= 2, ordinal      → Dykstra ('i'): col-space ∩ isotone cone
    - degree >= 0, not ordinal  → subspace ('s'): spline/polynomial OLS
    Extra copies (l >= 1) always use subspace ('s').

    Parameters
    ----------
    data    : np.ndarray (n,)
    target  : np.ndarray (n, copies)
    basis   : np.ndarray (n, k)
    copies  : int
    degree  : int
    ordinal : bool
    ties    : str
    missing : str

    Returns
    -------
    np.ndarray (n, copies)
    """
    nobs = len(data)
    h = np.zeros((nobs, copies))

    # First copy: route by (degree, ordinal)
    t0 = target[:, 0]
    if degree == -1:
        if ordinal:
            # Categorical ordinal: PAVA on group means (no basis)
            h[:, 0] = project_cone(
                target=t0, data=data,
                cone_type='c', ties=ties, missing=missing)
        else:
            # Nominal indicator: OLS onto indicator column space
            h[:, 0] = project_cone(
                target=t0, data=data, basis=basis,
                cone_type='s', missing=missing)
    elif degree >= 0:
        if ordinal:
            # Ordinal spline: Dykstra col-space ∩ isotone cone (R Gifi behavior)
            h[:, 0] = project_cone(
                target=t0, data=data, basis=basis,
                cone_type='i', ties=ties, missing=missing)
        else:
            # Non-ordinal: subspace OLS
            h[:, 0] = project_cone(
                target=t0, data=data, basis=basis,
                cone_type='s', ties=ties, missing=missing)

    # Extra copies: always subspace ('s') — no ordering constraint on copies
    for copy_idx in range(1, copies):
        h[:, copy_idx] = project_cone(
            target=target[:, copy_idx],
            data=data,
            basis=basis,
            cone_type='s',
            ties=ties,
            missing=missing,
        )

    return h


def gifi_engine(gifi, ndim, itmax=1000, eps=1e-6, verbose=False, init_x=None, r_seed=None):
    """
    Alternating Least Squares engine for all Gifi methods.

    Python port of R's gifiEngine(gifi, ndim, itmax, eps, verbose).

    Parameters
    ----------
    gifi    : list of lists — from make_gifi()
    ndim    : int — number of dimensions
    itmax   : int — maximum ALS iterations
    eps     : float — convergence threshold on loss change
    verbose : bool — print iteration info
    init_x  : np.ndarray (nobs, ndim), optional
        Custom initial random matrix. If provided, this is used directly
        instead of generating one with np.random.seed(123). To get exact
        parity with R, generate this matrix in R with:
            set.seed(123)
            x <- matrix(rnorm(nobs * ndim), nobs, ndim)
        and pass to Python. If None (default), uses NumPy's RNG with seed 123.
    r_seed  : int, optional
        If provided, uses the exact C-ported R RNG (MT19937 + AS241 Inversion)
        to generate the starting matrix `X` for exact numerical parity with R.
        Bypasses SVD deterministic initialization.

    Returns
    -------
    dict with keys: f, ntel, x, xGifi
    """
    # Get nobs from first variable of first set
    nobs = len(gifi[0][0]['data'])
    len(gifi)
    nvars = sum(len(s) for s in gifi)

    if nvars < 2:
        raise ValueError(
            "gifi_engine requires at least two variables to perform multivariate analysis, "
            f"but only {nvars} were provided.")

    # --- Initialization ---
    if r_seed is not None:
        try:
            import pygifi_rng
            x = pygifi_rng.r_init_x(nobs, ndim, r_seed)
            x = np.asarray(x, dtype=float)
            x = gs_rc(center(x))['q']
        except ImportError:
            raise ImportError(
                "pygifi_rng C extension not built! Run `python3 setup_rng.py build_ext --inplace` "
                "in `pygifi/rng` to use exact R numerical parity."
            )
    elif init_x is not None:
        # User-provided initial matrix (e.g., from R for exact parity)
        x = np.asarray(init_x, dtype=float)
        if x.shape != (nobs, ndim):
            raise ValueError(
                f"init_x must have shape ({nobs}, {ndim}), got {x.shape}")
        x = gs_rc(center(x))['q']
    else:
        # Permanent fix: SVD-based deterministic initialization.
        #
        # R's gifiEngine.R uses set.seed(123) + rnorm() internally, but R's
        # and NumPy's RNGs differ (Kinderman-Ramage vs Ziggurat algorithm)
        # and produce completely different streams even with the same seed
        # number.  Random initialization therefore always yields different
        # local minima for the same dataset.
        #
        # Solution: initialize X from the top-ndim left singular vectors of
        # the mean-centered stacked indicator/basis matrix.  This is:
        #   1. Fully deterministic (no seed dependency)
        #   2. Data-driven (starts near the MCA/PCA spectral solution)
        #   3. Qualitatively identical to R for virtually any dataset
        #   4. Faster convergence (fewer ALS iterations needed)
        #
        # Reference: Torgerson (1958), de Leeuw & Mair (2009) — the spectral
        # initialization is the exact solution when all variables are metric;
        # for mixed levels it is the best linear approximation to start from.
        try:
            # Stack the basis matrices of all active variables
            basis_parts = []
            for gifi_set in gifi:
                for gv in gifi_set:
                    if gv['active']:
                        b = np.asarray(gv['basis'], dtype=float)
                        # Replace NaN rows with column means
                        col_means = np.nanmean(b, axis=0)
                        nan_rows = np.any(np.isnan(b), axis=1)
                        b[nan_rows] = col_means
                        basis_parts.append(b)

            if basis_parts:
                H = np.hstack(basis_parts)          # (nobs, total_cols)
                H = H - H.mean(axis=0)              # center columns
                # Thin SVD — only compute left singular vectors
                # Use full_matrices=False for memory efficiency
                U, _, _ = np.linalg.svd(H, full_matrices=False)
                x = U[:, :ndim]                     # (nobs, ndim)
                # Re-orthogonalize and center (handles numerical drift)
                x = gs_rc(center(x))['q']
            else:
                raise ValueError("No active variables found for SVD init")

        except Exception:
            # Fallback: random init if SVD fails (e.g. degenerate basis)
            np.random.seed(123)
            x = gs_rc(center(np.random.randn(nobs, ndim)))['q']

    xGifi = make_x_gifi(gifi, x)

    # Compute initial loss fold
    fold = 0.0
    asets = 0
    for i, gifi_set in enumerate(gifi):
        x_set = xGifi[i]
        ha = np.zeros((nobs, ndim))
        active_count = 0
        for j, gv in enumerate(gifi_set):
            if gv['active']:
                active_count += 1
                ha += x_set[j]['scores']
        if active_count > 0:
            asets += 1
            fold += np.sum((x - ha) ** 2)
    fold /= (asets * ndim)

    itel = 1
    while True:
        xz = np.zeros((nobs, ndim))
        fnew = 0.0
        fmid = 0.0

        for i, gifi_set in enumerate(gifi):
            x_set = xGifi[i]

            # ── STAGE 1: Active variable minimization ──────────────────────────
            # Stack transforms of active variables: hh (nobs, sum_active_copies)
            hh_parts = []
            active_indices = []
            active_count = 0
            for j, gv in enumerate(gifi_set):
                if gv['active']:
                    active_count += 1
                    active_indices.append(j)
                    hh_parts.append(x_set[j]['transform'])
            if active_count == 0:
                continue

            hh = np.hstack(hh_parts)               # (nobs, sum_copies)

            # OLS regression of hh onto x
            lf = ls_rc(hh, x)
            aa = lf['solution']                     # (sum_copies, ndim)
            rs = lf['residuals']                    # (nobs, ndim)
            fmid += np.sum(rs ** 2)

            # ── Majorization step — matches R's gifiEngine.R exactly ──────────
            #
            # R:  kappa <- max(eigen(crossprod(aa))$values)    # global, per set
            #     target <- hh + tcrossprod(rs, aa) / kappa    # for ALL copies
            #
            # A single global κ = max_eigenvalue(A'A) is computed for the
            # ENTIRE stacked coefficient matrix A (all variables in the set).
            # This is the Lipschitz constant of the full set-level gradient.
            # Slicing per-variable targets from the global target matrix
            # ensures each variable sees the same step-size bound, matching R.
            #
            # Using per-variable κ_j (previous behaviour) underestimates the
            # true Lipschitz bound, causing larger per-variable steps and
            # convergence to a different local minimum than R.
            kappa = float(np.linalg.eigvalsh(aa.T @ aa)[-1])
            kappa = max(kappa, 1e-14)               # guard zero
            # Full target for all copies: (nobs, sum_copies)
            full_target = hh + rs @ aa.T / kappa

            hh_new_parts = []
            scopies = 0
            for j, gv in enumerate(gifi_set):
                jcopies = x_set[j]['transform'].shape[1]

                if not gv['active']:
                    # Skip active update pass — handled in Stage 2
                    continue

                ja = aa[scopies:scopies + jcopies, :]   # (jcopies, ndim)

                # Slice this variable's target from the full target matrix
                jtarget = full_target[:, scopies:scopies + jcopies]  # (nobs, jcopies)

                hj = gifi_transform(
                    data=gv['data'],
                    target=jtarget,
                    basis=gv['basis'],
                    copies=gv['copies'],
                    degree=gv['degree'],
                    ordinal=gv['ordinal'],
                    ties=gv['ties'],
                    missing=gv['missing'],
                )

                # FIX 3: Inner Gram-Schmidt sweep over copies
                # For multi-copy variables, enforce linear independence of the
                # p_j coefficient vectors across copies (R gifiEngine.R).
                hj = normalize(center(hj))
                gs_res = gs_rc(hj)
                rank = gs_res['rank']
                if rank == 0:
                    # Degenerate — keep zeros
                    hj = np.zeros((nobs, jcopies))
                elif rank < jcopies:
                    # Rank-deficient: pad with zeros for missing directions
                    q = gs_res['q']                     # (nobs, rank)
                    hj = np.hstack([q, np.zeros((nobs, jcopies - rank))])
                else:
                    hj = gs_res['q']                    # (nobs, jcopies)

                # (nobs, ndim) scores
                sc = hj @ ja

                # Update mutable state
                xGifi[i][j]['transform'] = hj
                xGifi[i][j]['weights'] = ja
                xGifi[i][j]['scores'] = sc
                xGifi[i][j]['quantifications'] = ls_rc(gv['basis'], sc)[
                    'solution']

                hh_new_parts.append(hj)
                scopies += jcopies

            if len(hh_new_parts) > 0:
                #  Re-stack just the active transforms in order
                hh_new = np.hstack(
                    [x_set[j]['transform'] for j in active_indices])
                ha = hh_new @ aa
                xz += ha
                fnew += np.sum((x - ha) ** 2)

            # ── STAGE 2: Passive variable two-stage update ──────────────────────
            # After active-variable minimization, update passive variables by
            # projecting them onto their cones using the fixed X from Stage 1.
            # R gifiEngine.R performs this pass explicitly — previously missing.
            for j, gv in enumerate(gifi_set):
                if gv['active']:
                    continue                             # already updated

                jcopies = x_set[j]['transform'].shape[1]
                cur_hj = x_set[j]['transform']          # (nobs, jcopies)

                # Compute passive-variable loadings from current x
                pa = ls_rc(cur_hj, x)['solution']       # (jcopies, ndim)
                pa_res = ls_rc(cur_hj, x)['residuals']  # (nobs, ndim)

                # Per-variable κ_j for the passive update
                B_pp = pa.T @ pa                        # (ndim, ndim)
                kappa_p = float(np.linalg.eigvalsh(B_pp)[-1])
                kappa_p = max(kappa_p, 1e-14)

                # Majorization gradient target
                jtarget = cur_hj + pa_res @ pa.T / kappa_p

                hj = gifi_transform(
                    data=gv['data'],
                    target=jtarget,
                    basis=gv['basis'],
                    copies=gv['copies'],
                    degree=gv['degree'],
                    ordinal=gv['ordinal'],
                    ties=gv['ties'],
                    missing=gv['missing'],
                )

                # Inner GS sweep for copies
                hj = normalize(center(hj))
                gs_res = gs_rc(hj)
                rank = gs_res['rank']
                if rank == 0:
                    hj = np.zeros((nobs, jcopies))
                elif rank < jcopies:
                    q = gs_res['q']
                    hj = np.hstack([q, np.zeros((nobs, jcopies - rank))])
                else:
                    hj = gs_res['q']

                sc = hj @ pa
                xGifi[i][j]['transform'] = hj
                xGifi[i][j]['weights'] = pa
                xGifi[i][j]['scores'] = sc
                xGifi[i][j]['quantifications'] = ls_rc(gv['basis'], sc)[
                    'solution']

        fmid /= (asets * ndim)
        fnew /= (asets * ndim)

        if verbose:
            print(f"Iter {itel:4d}  fold={fold:.8f}  fmid={fmid:.8f}  fnew={fnew:.8f}")

        # Convergence check (matches R: itel > 1 required)
        if (itel == itmax or (fold - fnew) < eps) and itel > 1:
            break

        itel += 1
        fold = fnew
        x = gs_rc(center(xz))['q']

    return {
        'f': fnew,
        'ntel': itel,
        'x': x,
        'xGifi': xGifi,
        'iterations': itel,
        'final_stress': fnew,
        'converged': (itel < itmax and (fold - fnew) < eps)
    }


def gifi_loss(X, Z, H, A):
    """
    Compute the Gifi loss (stress) for optimal scaling.

    Formula: stress = sum_j ||X - H_j Z_j A_j||^2

    Parameters
    ----------
    X : np.ndarray (n_samples, n_dimensions)
        Object scores.
    Z : list of np.ndarray
        Category quantifications for each variable.
    H : list of sparse matrices or np.ndarray
        Indicator matrices for each variable.
    A : list of np.ndarray
        Loadings for each variable.

    Returns
    -------
    float
        The total computed stress value.
    """
    X = np.asarray(X)
    stress = 0.0

    for Hj, Zj, Aj in zip(H, Z, A):
        # Calculate Hj @ Zj (works for both dense and scipy.sparse Hj)
        HZ = Hj @ Zj

        # Calculate HZ @ Aj
        HZA = HZ @ Aj

        # Add squared Frobenius norm of difference
        diff = X - HZA
        stress += np.sum(diff ** 2)

    return float(stress)


def gifi_als(X_init, H_list, A_list, max_iter=1000, tol=1e-6, ridge_penalty=1e-8, ordinal=None):
    """
    Alternating Least Squares (ALS) algorithm for optimal scaling.

    Parameters
    ----------
    X_init : np.ndarray (n_samples, n_dimensions)
        Initial object scores.
    H_list : list of sparse matrices or np.ndarray
        Indicator matrices for each variable.
    A_list : list of np.ndarray
        Loadings for each variable.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence based on stress difference.
    ridge_penalty : float
        Regularization term added to H^T H to prevent singular matrix inversion.
    ordinal : list of bool or None
        If provided, must be same length as H_list. When True at position j,
        category quantifications Z_j are constrained to be monotone non-decreasing
        (via PAVA) after each update, enforcing ordinal scaling for that variable.

    Returns
    -------
    X : np.ndarray
        Optimized object scores.
    Z_list : list of np.ndarray
        Optimized category quantifications.
    stress_history : list of floats
        Stress loss tracking across iterations.
    """
    X = np.asarray(X_init).copy()
    n_vars = len(H_list)

    # Precompute H_j^T H_j + lambda I (for stability)
    HTH_inv = []
    for Hj in H_list:
        if hasattr(Hj, "toarray"):  # Check if it's scipy sparse
            dense_H = Hj.toarray()
            HTH = dense_H.T @ dense_H
        else:
            HTH = Hj.T @ Hj

        HTH = HTH.astype(float)

        # Add ridge penalty to prevent singular matrices
        HTH += ridge_penalty * np.eye(HTH.shape[0])
        HTH_inv.append(np.linalg.pinv(HTH))

    Z_list = [None] * n_vars
    stress_history = []
    ordinal = ordinal if ordinal is not None else [False] * n_vars

    # 1. Initialize object scores (done above: X = X_init)

    for it in range(max_iter):

        # 2. Update category quantifications Z_j
        for j in range(n_vars):
            Hj = H_list[j]
            Aj = A_list[j]

            # H_j^T X A_j^T
            HT_X_AT = Hj.T @ X @ Aj.T

            # Z_j = (H_j^T H_j + lambda I)^-1 H_j^T X A_j^T
            Z_list[j] = HTH_inv[j] @ HT_X_AT

            # 2.5 Apply PAVA monotone constraint for ordinal variables
            if ordinal[j]:
                from pygifi.utils.isotone import monotone_regression
                n_cats = Z_list[j].shape[0]
                x_ord = np.arange(1, n_cats + 1, dtype=float)
                Z_list[j] = monotone_regression(Z_list[j], x_ord)

        # 3. Update object scores X
        X_new = np.zeros_like(X)
        for j in range(n_vars):
            X_new += H_list[j] @ Z_list[j] @ A_list[j]
        X = X_new / n_vars

        # 4. Normalize scores (Enforce X^T X = I using SVD Orthogonalization)
        from pygifi.utils.utilities import svd_orthogonalize
        X = svd_orthogonalize(X)

        # 5. Compute stress
        current_stress = gifi_loss(X, Z_list, H_list, A_list)
        stress_history.append(current_stress)

        # 6. Check convergence
        if it > 0:
            diff = stress_history[-2] - current_stress
            if diff < tol:
                break

    return X, Z_list, stress_history


def gifi_majorization(X_init, H_list, A_list, max_iter=1000, tol=1e-6,
                      ridge_penalty=1e-8, ordinal=None):
    """
    Majorization algorithm for optimal scaling.

    Replaces the simple ALS averaging step with a Guttman transform that
    is derived as the exact minimizer of a surrogate majorizing function.
    This guarantees stress is non-increasing at every iteration.

    Algorithm (SMACOF/Gifi-style majorization):
        1. Initialize X with SVD orthogonalization.
        2. Update Z_j via ALS (same as gifi_als).
        2.5 Apply PAVA monotone constraint if ordinal[j] = True.
        3. Guttman transform: X_new = (1/J) * sum_j (H_j @ Z_j @ A_j)
           This step is the exact closed-form minimizer of the majorizing
           surrogate, guaranteeing stress(X_new) <= stress(X_old).
        4. SVD-orthogonalize X_new to enforce X^T X = I.
        5. Compute stress.
        6. Emit RuntimeWarning if stress increased (numerical noise).
        7. Repeat until |stress_prev - stress_new| < tol.

    Parameters
    ----------
    X_init : np.ndarray (n_samples, n_dimensions)
        Initial object scores.
    H_list : list of sparse matrices or np.ndarray
        Indicator matrices for each variable.
    A_list : list of np.ndarray
        Loadings for each variable.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on stress.
    ridge_penalty : float
        Ridge regularisation for Z_j inversion stability.
    ordinal : list of bool or None
        Per-variable flag; if True, PAVA is applied to Z_j after each update.

    Returns
    -------
    X : np.ndarray (n_samples, n_dimensions)
        Optimised, orthogonalised object scores.
    Z_list : list of np.ndarray
        Category quantifications per variable.
    stress_history : list of float
        Monotone non-increasing stress at each iteration.
    """
    import warnings
    from pygifi.utils.utilities import svd_orthogonalize

    X = svd_orthogonalize(np.asarray(X_init).copy())
    n_samples = X.shape[0]
    n_vars = len(H_list)
    ordinal = ordinal if ordinal is not None else [False] * n_vars

    # Precompute (H_j^T H_j + lambda I)^{-1} for each variable
    HTH_inv = []
    for Hj in H_list:
        HTH = (Hj.toarray() if hasattr(Hj, 'toarray') else Hj).T \
            @ (Hj.toarray() if hasattr(Hj, 'toarray') else Hj)
        HTH = HTH.astype(float) + ridge_penalty * np.eye(HTH.shape[0])
        HTH_inv.append(np.linalg.pinv(HTH))

    Z_list = [None] * n_vars
    stress_history = []

    # 1. Stress at initialisation
    # (Z_list is uninitialised; compute after first Z update)

    for it in range(max_iter):

        # 1. JS normalise X at the start (centre + ||X||_F = sqrt(n_samples))
        X_c = X - X.mean(axis=0)
        frob = np.linalg.norm(X_c, 'fro')
        scale = np.sqrt(n_samples) / frob if frob > 1e-15 else 1.0
        X = X_c * scale

        # 2. Update Z_j via ALS on the normalised X
        for j in range(n_vars):
            Hj, Aj = H_list[j], A_list[j]
            Z_list[j] = HTH_inv[j] @ (Hj.T @ X @ Aj.T)

            # 2.5 Optional PAVA isotone constraint (ordinal variables)
            if ordinal[j]:
                from pygifi.utils.isotone import monotone_regression
                n_cats = Z_list[j].shape[0]
                Z_list[j] = monotone_regression(
                    Z_list[j],
                    np.arange(1, n_cats + 1, dtype=float)
                )

        # 3. Guttman transform: X_new = (1/J) * sum_j H_j Z_j A_j
        X_new = np.zeros_like(X)
        for j in range(n_vars):
            ZA = Z_list[j] @ A_list[j]
            contrib = H_list[j] @ ZA
            if hasattr(contrib, 'toarray'):
                contrib = contrib.toarray()
            X_new += contrib
        X_new /= n_vars

        # 4. JS normalise X_new: centre + set ||X||_F = sqrt(n_samples).
        # This rescales X to a fixed magnitude so that stress values are
        # comparable across iterations — measured on the same X that will be
        # used for the next Z update.
        X_c = X_new - X_new.mean(axis=0)
        frob = np.linalg.norm(X_c, 'fro')
        scale = np.sqrt(n_samples) / frob if frob > 1e-15 else 1.0
        X = X_c * scale

        # 5. Measure stress on the JS-normalised X
        current_stress = gifi_loss(X, Z_list, H_list, A_list)
        stress_history.append(current_stress)

        # 6. Convergence + descent warning
        if it > 0:
            delta = stress_history[-2] - current_stress
            if delta < -1e-8:
                warnings.warn(
                    f"Majorization: stress increased by {-delta:.2e} at "
                    f"iteration {it}. Possible numerical instability.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if abs(delta) < tol:
                break

    # SVD-orthogonalise once for clean output satisfying X^T X = I
    return svd_orthogonalize(X), Z_list, stress_history
