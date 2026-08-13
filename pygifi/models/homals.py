"""
pygifi.homals: Multiple Correspondence Analysis (Homals).

Python port of Gifi/R/homals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Homals finds a low-dimensional representation of categorical (or mixed) data
by finding object scores and category quantifications that best fit each other
in a least-squares sense, using Alternating Least Squares with optimal scaling.

Compatible with scikit-learn: BaseEstimator, TransformerMixin, fit/transform API.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pygifi.utils.coding import make_numeric
from pygifi.utils.utilities import reshape, cor_list
from pygifi.utils.prepspline import level_to_spline
from pygifi.core.structures import make_gifi
from pygifi.core.engine import gifi_engine


class Homals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Multiple Correspondence Analysis via ALS optimal scaling.

    Python port of R's homals() function (Gifi package).

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions (components).
    levels : str or list of str, default='nominal'
        Measurement level per variable. One of:
        'nominal' : categorical, no order (indicator basis, subspace transform)
        'ordinal' : categorical with order (indicator basis, isotone transform)
        'metric'  : continuous (polynomial basis, isotone transform)
        A single string is broadcast to all variables.
    ties : str, default='s'
        Tie-handling for isotone regression: 's' (secondary), 'p' (primary),
        't' (tertiary).
    missing : str, default='s'
        Missing value treatment: 'm' (individual), 'a' (average), 's' (shared).
    normobj_z : bool, default=True
        If True, scale objectscores by sqrt(nobs), the same default used in R.
    active : bool or list of bool, default=True
        Which variables participate actively in ALS.
    sets : list of int or None, default=None
        Set membership for each variable (0-indexed integers). Variables in
        the same set share the same object scores. Enables regression,
        canonical analysis, and discriminant analysis layouts: following R's
        homals() ``sets`` argument semantics.
        If None, each variable forms its own set: ``list(range(nvars))``.
    rank : int or list of int or None, default=None
        Per-variable rank constraint: maximum number of active columns in H_j.
        If None, defaults to ``ndim`` (no extra constraint). Setting
        ``rank=1`` reduces each variable to a single quantification vector.
    itmax : int, default=1000
        Maximum ALS iterations.
    eps : float, default=1e-6
        Convergence threshold on loss change.
    verbose : bool, default=False
        Print iteration details.

    Attributes
    ----------
    result_ : dict
        Dictionary with all fitted quantities:
        - objectscores    : (nobs, ndim): object coordinates
        - quantifications : list of (nbasis, ndim) arrays: category coordinates
        - transform       : list of (nobs, copies) arrays: optimal transforms
        - weights         : list of (copies, ndim) arrays: regression weights
        - loadings        : list of (ndim, copies) arrays: x.T @ transform
        - rhat            : (sum_copies, sum_copies) correlation matrix
        - evals           : eigenvalues of rhat
        - lambda_         : (ndim, ndim) average discriminant matrix
        - f               : final loss value
        - ntel            : number of iterations
    n_obs_ : int
    n_vars_ : int
    n_iter_ : int
    converged_ : bool
    """

    def __init__(
        self,
        ndim: int = 2,
        levels: Union[str, List[str]] = 'nominal',
        ordinal: Optional[Union[bool, List[bool]]] = None,
        knots: Optional[List[Any]] = None,
        ties: Union[str, List[str]] = 's',
        degrees: Union[int, List[int]] = -1,
        missing: Union[str, List[str]] = 's',
        normobj_z: bool = True,
        active: Union[bool, List[bool]] = True,
        sets: Optional[List[int]] = None,
        rank: Optional[Union[int, List[int]]] = None,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[Any] = None,
        r_seed: Optional[int] = None,
        optimizer: str = 'als',
    ) -> None:
        self.ndim = ndim
        self.levels = levels
        self.ordinal = ordinal
        self.knots = knots
        self.ties = ties
        self.degrees = degrees
        self.missing = missing
        self.normobj_z = normobj_z
        self.active = active
        self.sets = sets
        self.rank = rank
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x
        self.r_seed = r_seed
        self.optimizer = optimizer

    def fit(self, X: Union[pd.DataFrame, Any],
            y: Any = None) -> 'Homals':
        """
        Fit Homals on X.

        Parameters
        ----------
        X : pd.DataFrame or array-like (nobs, nvars)
        y : ignored (scikit-learn API compatibility)

        Returns
        -------
        self
        """
        # --- Coerce input ---
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        from pygifi.utils.utilities import sanitize_dataframe
        X = sanitize_dataframe(X)
        if X.empty:
            raise ValueError("Input data X cannot be empty (or was entirely dropped).")
        names = list(X.columns)
        list(X.index)
        data_orig = X.copy()

        data = make_numeric(X)           # (nobs, nvars) numeric array
        nobs, nvars = data.shape

        # --- Validate + broadcast scalar params ---
        valid_ties = ('s', 'p', 't')
        valid_missing = ('m', 's', 'a')
        if self.ties not in valid_ties:
            raise ValueError(
                f"ties must be one of {valid_ties}, got '{self.ties}'")
        if self.missing not in valid_missing:
            raise ValueError(
                f"missing must be one of {valid_missing}, got '{self.missing}'")

        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore
        active_v = reshape(self.active, nvars)  # type: ignore
        levels_v = reshape(self.levels, nvars)  # type: ignore

        # --- Prepare spline/knot params from levels ---
        if self.knots is None:
            levelprep = level_to_spline(levels_v, data)  # type: ignore
            ordinal_v = levelprep['ordvec']
            knots_v = levelprep['knotList']
            degrees_v = reshape(self.degrees, nvars)  # type: ignore
        else:
            ordinal_v = reshape(
                self.ordinal if self.ordinal is not None else True,
                nvars)  # type: ignore
            knots_v = self.knots
            degrees_v = reshape(self.degrees, nvars)  # type: ignore

        copies_v = [self.ndim] * nvars   # each variable gets ndim copies

        # --- Resolve sets (user-supplied or default: each var = own set) ---
        if self.sets is not None:
            if len(self.sets) != nvars:
                raise ValueError(
                    f"sets must have length nvars={nvars}, "
                    f"got len(sets)={len(self.sets)}")
            sets_v = list(self.sets)
        else:
            sets_v = list(range(nvars))  # R default: each var in own set

        # --- Resolve per-variable rank constraint ---
        if self.rank is not None:
            from pygifi.utils.utilities import reshape as _reshape
            rank_v = _reshape(self.rank, nvars)  # type: ignore
            # Cap rank at ndim
            rank_v = [min(int(r), self.ndim) for r in rank_v]
        else:
            rank_v = [self.ndim] * nvars

        # Apply rank to copies: copies = min(copies, rank)
        copies_v = [min(c, r) for c, r in zip(copies_v, rank_v)]

        # --- Build Gifi structure ---
        gifi = make_gifi(  # type: ignore
            data=data,
            knots=knots_v,
            degrees=degrees_v,
            ordinal=ordinal_v,
            ties=ties_v,
            copies=copies_v,
            missing=missing_v,
            active=active_v,
            names=names,
            sets=sets_v,
            # each variable is its own set (homals convention)
        )

        # --- Run ALS engine ---
        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x, r_seed=self.r_seed)

        # --- Optional majorization refinement (Homals: nominal, no ordinal) ---
        if self.optimizer == 'majorization':
            from pygifi.core.engine import gifi_majorization
            from pygifi.utils.utilities import make_indicator
            X_als = h['x']
            H_list = [make_indicator(data[:, j]) for j in range(nvars)]  # type: ignore[no-untyped-call]
            A_list = [np.ones((1, self.ndim)) for _ in range(nvars)]
            X_ref, _, _ = gifi_majorization(  # type: ignore[no-untyped-call]
                X_als, H_list, A_list,
                max_iter=self.itmax, tol=self.eps,
                ordinal=[False] * nvars,   # Homals: nominal, no monotone constraint
            )
            h['x'] = X_ref

        # --- Assemble result dict (mirrors homals.R output) ---
        xGifi = h['xGifi']

        transform = []
        weights = []
        scores = []
        quantifications = []
        dsum = np.zeros((self.ndim, self.ndim))
        nact = 0

        dmeasures = []
        for j in range(nvars):
            jgifi = xGifi[j][0]   # set j, variable 0 (each var in own set)
            tr_j = jgifi['transform']
            w_j = jgifi['weights']
            sc_j = jgifi['scores']
            q_j = jgifi['quantifications']

            transform.append(tr_j)
            weights.append(w_j)
            scores.append(sc_j)
            quantifications.append(q_j)

            cy = sc_j.T @ sc_j                           # (ndim, ndim)
            dmeasures.append(cy)
            if gifi[j][0]['active']:
                dsum += cy
                nact += 1

        # Correlation matrix across all transforms
        rhat = cor_list(transform)  # type: ignore
        evals_raw = np.linalg.eigh(rhat)[0]
        evals = evals_raw[::-1]

        objectscores = h['x'].copy()

        # Scaling parameter logic: scale objectscores by sqrt(nobs) if enabled
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        lambda_ = dsum / nvars                           # average discriminant

        # Score matrix: first dim of each variable's scores (R
        # scoremat equivalent)
        scoremat = np.column_stack([sc[:, 0] for sc in scores])

        # Bug 2: Proper per-variable dictionary for quantifications and
        # loadings
        self.quantifications = {
            name: q for name, q in zip(
                names, quantifications)}
        self.loadings = {name: q.T for name, q in zip(names, quantifications)}
        # Also expose via trailing underscore for scikit-learn convention
        self.quantifications_ = self.quantifications
        self.loadings_ = self.loadings

        # --- R homals() additional fields ---
        # cat.centroids: category centroids in object score space = G_j^+ @ X
        # where G_j^+ is the pseudo-inverse (lstsq) of indicator G_j
        from pygifi.utils.utilities import make_indicator
        from scipy.linalg import lstsq as _lstsq
        cat_centroids = []
        ind_mat = []
        for j in range(nvars):
            G_j = make_indicator(data[:, j])  # type: ignore[no-untyped-call]      # (nobs, n_cats)
            ind_mat.append(G_j)
            # G_j^+ @ X = least-squares centroid for each category
            cx, _, _, _ = _lstsq(G_j, objectscores, cond=None)
            cat_centroids.append(cx)              # (n_cats, ndim)

        call_str = (
            f"Homals(ndim={self.ndim}, levels={self.levels!r}, "
            f"ties={self.ties!r}, missing={self.missing!r}, "
            f"normobj_z={self.normobj_z}, active={self.active!r})")

        self.result_ = {
            'transform': transform,
            'rhat': rhat,
            'evals': evals,
            'objectscores': objectscores,
            'scoremat': scoremat,
            'quantifications': self.quantifications,
            'dmeasures': dmeasures,
            'weights': weights,
            'loadings': self.loadings,
            'lambda_': lambda_,
            'f': h['f'],
            'ntel': h['ntel'],
            'data': data_orig,
            'datanum': data,
            'ndim': self.ndim,
            'knots': knots_v,
            'degrees': degrees_v,
            'ordinal': ordinal_v,
            'cat.centroids': cat_centroids,    # R homals() field
            'ind.mat': ind_mat,                # R homals() field
            'rank': rank_v,                    # R homals() field
            'call_': call_str,                 # R match.call() equivalent
        }

        self.n_obs_ = nobs
        self.n_vars_ = nvars
        self.n_iter_ = h['ntel']
        self.converged_ = (h['ntel'] < self.itmax)
        self.is_fitted_ = True

        return self

    def transform(self, X: Any, y: Any = None) -> Any:
        """
        Return object scores for the fitted data.

        Note: Homals does not support out-of-sample projection, as in R.
        The input X is accepted for sklearn API compatibility but must be
        the same data used in fit().

        Parameters
        ----------
        X : ignored (out-of-sample not supported)

        Returns
        -------
        np.ndarray of shape (nobs, ndim): object scores
        """
        check_is_fitted(self, 'is_fitted_')
        return self.result_['objectscores']

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            call_str = self.result_.get('call_', f"Homals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Homals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary in the style of R's summary.homals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nCategory Quantifications:")
        for name, q in self.result_['quantifications'].items():
            print(f"\n-- {name} --")
            print(pd.DataFrame(q, columns=[f"D{d + 1}" for d in range(self.ndim)]))
        print()

    @property
    def eigenvalues_(self) -> Optional[Any]:
        """Array of eigenvalues from the optimal scaling correlation matrix."""
        return self.result_.get('evals') if hasattr(self, 'result_') else None

    @property
    def variance_explained_(self) -> Optional[Any]:
        """Percentage of variance explained by each dimension."""
        evals = self.eigenvalues_
        if evals is not None:
            return (evals / np.sum(evals)) * 100
        return None

    @property
    def component_loadings_(self) -> Optional[Dict[str, Any]]:
        """Component loadings for each variable."""
        return self.result_.get('loadings') if hasattr(self, 'result_') else None

    @property
    def category_quantifications_(self) -> Optional[Dict[str, Any]]:
        """Category quantifications (optimal scaling maps) per variable."""
        return self.result_.get('quantifications') if hasattr(self, 'result_') else None

    @property
    def object_scores_(self) -> Optional[Any]:
        """The optimized object scores (coordinates)."""
        return self.result_.get('objectscores') if hasattr(self, 'result_') else None
