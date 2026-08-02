"""
pygifi.princals — Categorical Principal Component Analysis (Princals).

Python port of Gifi/R/princals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Princals finds low-dimensional latent structure in mixed-level data
by optimally scaling each variable (polynomial, spline, monotone, or
nominal) and fitting the resulting components to object scores via ALS.

Differences from Homals:
- Default levels='ordinal' (not 'nominal')
- Default degrees=1 — linear polynomial (or higher spline)
- Default copies=1 — single component per variable
- transform output: (nobs, nvars) matrix when copies=1 (not a list)
- weights output: (nvars, ndim) matrix
- loadings output: (nvars, ndim) matrix — transposed vs R's internal

Compatible with scikit-learn: BaseEstimator, TransformerMixin, fit/transform API.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pygifi.utils.coding import make_numeric
from pygifi.utils.utilities import reshape, cor_list
from pygifi.utils.prepspline import level_to_spline
from pygifi.core.structures import make_gifi
from pygifi.core.engine import gifi_engine


class Princals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Categorical Principal Component Analysis via ALS optimal scaling.

    Python port of R's princals() function (Gifi package).

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions (principal components).
    levels : str or list of str, default='ordinal'
        Measurement level per variable: 'nominal', 'ordinal', or 'metric'.
        A single string is broadcast to all variables.
    degrees : int or list of int, default=1
        B-spline degree per variable:
        -1 = categorical indicator basis
         0 = piecewise constant
         1 = piecewise linear (default)
         k = k-th degree polynomial/spline
    copies : int or list of int, default=1
        Number of copies (components) per variable.
        When copies=1 (default), transform output is a matrix (nobs, nvars).
    ties : str, default='s'
        Tie-handling for isotone: 's', 'p', or 't'.
    missing : str, default='s'
        Missing value mode: 'm', 'a', or 's'.
    normobj_z : bool, default=True
        Scale objectscores by sqrt(nobs) to match R's output.
    active : bool or list of bool, default=True
        Which variables participate actively in ALS.
    itmax : int, default=1000
    eps : float, default=1e-6
    verbose : bool, default=False

    Attributes
    ----------
    result_ : dict
        - objectscores   : (nobs, ndim)
        - transform      : (nobs, nvars) matrix [copies=1] or list
        - weights        : (nvars, ndim) matrix
        - loadings       : (nvars, ndim) matrix
        - quantifications: list of (nbasis, ndim) arrays
        - rhat           : correlation matrix
        - evals          : eigenvalues of rhat
        - lambda_        : (ndim, ndim) average discriminant
        - f, ntel        : final loss and iteration count
    n_obs_, n_vars_, n_iter_, converged_
    """

    def __init__(
        self,
        ndim: int = 2,
        levels: Union[str, List[str]] = 'ordinal',
        ordinal: Optional[Union[bool, List[bool]]] = None,
        knots: Optional[List[Any]] = None,
        degrees: Union[int, List[int]] = 1,
        copies: Union[int, List[int]] = 1,
        ties: Union[str, List[str]] = 's',
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
        self.degrees = degrees
        self.copies = copies
        self.ties = ties
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

    def _get_ordinal_flags(self,
                           data: Union[pd.DataFrame,
                                       Any],
                           ordinal: Union[Any,
                                         List[Any]]) -> List[bool]:
        from pygifi.utils.utilities import reshape
        return reshape(ordinal, data.shape[1])  # type: ignore[no-untyped-call,no-any-return]

    def fit(self, X: Union[pd.DataFrame, Any],
            y: Any = None) -> 'Princals':
        """
        Fit Princals on X.

        Parameters
        ----------
        X : pd.DataFrame or array-like (nobs, nvars)
        y : ignored

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
        data_orig = X.copy()
        data = make_numeric(X)
        nobs, nvars = data.shape

        valid_ties = ('s', 'p', 't')
        valid_missing = ('m', 's', 'a')
        if self.ties not in valid_ties:
            raise ValueError(f"ties must be one of {valid_ties}")
        if self.missing not in valid_missing:
            raise ValueError(f"missing must be one of {valid_missing}")

        # --- Broadcast per-variable params ---
        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore
        active_v = reshape(self.active, nvars)  # type: ignore
        levels_v = reshape(self.levels, nvars)  # type: ignore
        # --- Prepare knots + ordinal flags ---
        if self.knots is None:
            levelprep = level_to_spline(levels_v, data)  # type: ignore
            knots_v = levelprep['knotList']
            ordinal_v = levelprep['ordvec']
            degrees_v = reshape(self.degrees, nvars)  # type: ignore
        else:
            knots_v = self.knots
            if self.ordinal is not None:
                ordinal_v = reshape(self.ordinal, nvars)  # type: ignore
            else:
                ordinal_v = self._get_ordinal_flags(data, self.levels)
            degrees_v = reshape(self.degrees, nvars)  # type: ignore

        copies_v = reshape(self.copies, nvars)  # type: ignore

        # --- Resolve sets (user-supplied or default: each var = own set) ---
        if self.sets is not None:
            if len(self.sets) != nvars:
                raise ValueError(
                    f"sets must have length nvars={nvars}, "
                    f"got len(sets)={len(self.sets)}")
            sets_v = list(self.sets)
        else:
            sets_v = list(range(nvars))

        # --- Resolve per-variable rank constraint ---
        if self.rank is not None:
            rank_v = reshape(self.rank, nvars)  # type: ignore
            rank_v = [min(int(r), self.ndim) for r in rank_v]
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
            sets=sets_v,  # each variable = own set
        )

        # --- Run ALS engine ---
        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x, r_seed=self.r_seed)

        # --- Optional majorization refinement ---
        if self.optimizer == 'majorization':
            from pygifi.core.engine import gifi_majorization
            from pygifi.utils.utilities import make_indicator
            X_als = h['x']                        # (nobs, ndim)
            # Build H_list from indicator matrices for each variable
            H_list = [make_indicator(data[:, j]) for j in range(nvars)]  # type: ignore[no-untyped-call]
            A_list = [np.ones((1, self.ndim)) for _ in range(nvars)]
            X_ref, _, _ = gifi_majorization(  # type: ignore[no-untyped-call]
                X_als, H_list, A_list,
                max_iter=self.itmax, tol=self.eps,
                ordinal=ordinal_v,
            )
            h['x'] = X_ref

        # --- Assemble result (mirrors princals.R lines 33-110) ---
        xGifi = h['xGifi']
        transforms_list = []
        weights_list = []
        scores_list = []
        quants = []
        loadings_list = []
        dsum = np.zeros((self.ndim, self.ndim))

        dmeasures = []
        for j in range(nvars):
            jg = xGifi[j][0]
            tr_j = jg['transform']          # (nobs, copies_j)
            w_j = jg['weights']             # (copies_j, ndim)
            sc_j = jg['scores']             # (nobs, ndim)

            transforms_list.append(tr_j)
            weights_list.append(w_j)
            scores_list.append(sc_j)
            quants.append(jg['quantifications'])
            loadings_list.append(h['x'].T @ tr_j)    # (ndim, copies_j)

            cy = sc_j.T @ sc_j
            dmeasures.append(cy)
            dsum += cy

        # --- R: if length(copies)==1 → transform = cbind(v) ---
        scalar_copies = np.isscalar(self.copies) or len(set(copies_v)) == 1
        if scalar_copies and copies_v[0] == 1:
            # All variables have exactly 1 copy → stack into matrix
            transform_out = np.hstack(transforms_list)   # (nobs, nvars)
        else:
            transform_out = transforms_list              # type: ignore  # type: ignore

        # --- R: weights = do.call(rbind, a) → (nvars, ndim) ---
        weights_out = np.vstack(weights_list)            # (nvars, ndim)

        # --- R: loadings = t(do.call(cbind, o)) → (nvars, ndim) ---
        loadings_out = np.hstack(loadings_list).T        # (nvars, ndim)

        # --- rhat, evals ---
        # R: rhat <- corList(v); evals <- eigen(rhat)$values
        # cor_list uses Inner Product of unit-length vectors, which is the correlation.
        # This matches R's corList behavior for unit-normalized variables.
        # The eigenvalues of the correlation matrix sum to nvars.
        rhat = cor_list(transforms_list)
        evals = np.linalg.eigvalsh(rhat)[::-1]

        # --- objectscores with optional sqrt(nobs) scaling ---
        objectscores = h['x'].copy()
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        # --- scoremat: first dim of each var's scores ---
        scoremat = np.column_stack([sc[:, 0] for sc in scores_list])

        # --- lambda_ (Discrimination Measures) ---
        # R: lambda <- dsum / nvars. dsum is sum(crossprod(y[[j]]))
        # Here lambda_ should be a matrix of shape (ndim, ndim).
        # We'll use the dsum accumulated in the engine.
        lambda_ = dsum / nvars

        # call_ mirrors R's match.call() — stores constructor args as string
        call_str = (
            f"Princals(ndim={self.ndim}, levels={self.levels!r}, "
            f"degrees={self.degrees!r}, copies={self.copies!r}, "
            f"ties={self.ties!r}, missing={self.missing!r}, "
            f"normobj_z={self.normobj_z}, active={self.active!r})")

        self.result_ = {
            'transform': transform_out,
            'rhat': rhat,
            'evals': evals,
            'objectscores': objectscores,
            'scoremat': scoremat,
            'quantifications': quants,
            'dmeasures': dmeasures,
            'weights': weights_out,
            'loadings': loadings_out,
            'lambda_': lambda_,
            'f': h['f'],
            'ntel': h['ntel'],
            'data': data_orig,
            'datanum': data,
            'ndim': self.ndim,
            'knots': knots_v,
            'degrees': degrees_v,
            'ordinal': ordinal_v,
            'call_': call_str,
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

        Out-of-sample projection not supported (matches R).

        Returns
        -------
        np.ndarray (nobs, ndim) — objectscores
        """
        check_is_fitted(self, 'is_fitted_')
        return self.result_['objectscores']

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            call_str = self.result_.get('call_', f"Princals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Princals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.princals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nComponent Loadings:")
        df_loadings = pd.DataFrame(self.result_['loadings'], columns=[f"D{d + 1}" for d in range(self.ndim)])
        if hasattr(self.result_['data'], 'columns'):
            df_loadings.index = self.result_['data'].columns
        print(df_loadings)
        print("\nProportion of Variance Accounted For:")
        evals = self.result_['evals'][:self.ndim]
        total_var = self.n_vars_  # total variance is number of active variables in princals
        vaf = evals / total_var
        df_vaf = pd.DataFrame({'VAF': vaf, 'Cumulative VAF': np.cumsum(vaf)},
                              index=[f"D{d + 1}" for d in range(self.ndim)])
        print(df_vaf)

    @property
    def eigenvalues_(self) -> Optional[Any]:
        """Array of eigenvalues from the optimal scaling correlation matrix."""
        return self.result_.get('evals') if hasattr(self, 'result_') else None

    @property
    def variance_explained_(self) -> Optional[Any]:
        """Percentage of variance explained by each dimension."""
        evals = self.eigenvalues_
        if evals is not None:
            # sum(evals) = number of active variables for a correlation matrix
            return (evals / np.sum(evals)) * 100
        return None

    @property
    def component_loadings_(self) -> Optional[Any]:
        """Component loadings for each variable."""
        return self.result_.get('loadings') if hasattr(self, 'result_') else None

    @property
    def category_quantifications_(self) -> Optional[Any]:
        """Category quantifications (optimal scaling maps) per variable."""
        return self.result_.get('quantifications') if hasattr(self, 'result_') else None

    @property
    def object_scores_(self) -> Optional[Any]:
        """The optimized object scores (coordinates)."""
        return self.result_.get('objectscores') if hasattr(self, 'result_') else None
