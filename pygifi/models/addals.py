# mypy: ignore-errors
"""
pygifi.addals — Conjoint / Additive Analysis (Addals).

Python port of Gifi/R/addals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Addals is an additive conjoint analysis where each variable is constrained
to a rank-1 solution — i.e., only a single component per variable is used
(copies=1) and the joint solution is the sum of the individual part-worths.
It is a Princals wrapper with strict rank=1 per variable and ordinal transforms.
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


class Addals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Additive Conjoint Analysis via Gifi ALS (Addals).

    Python port of R's addals() — Princals with rank=1 per variable and
    ordinal/metric transforms used as additive part-worth functions.

    Parameters
    ----------
    ndim : int, default=1
        Number of additive dimensions (almost always 1 for conjoint).
    levels : str or list, default='ordinal'
    degrees : int or list, default=1
    ties : str, default='s'
    missing : str, default='s'
    normobj_z : bool, default=True
    active : bool or list, default=True
    itmax : int, default=1000
    eps : float, default=1e-6
    verbose : bool, default=False
    """

    def __init__(
        self,
        ndim: int = 1,
        levels: Union[str, List[str]] = 'ordinal',
        degrees: Union[int, List[int]] = 1,
        ties: str = 's',
        missing: str = 's',
        normobj_z: bool = True,
        active: Union[bool, List[bool]] = True,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[np.ndarray] = None,
    ) -> None:
        self.ndim = ndim
        self.levels = levels
        self.degrees = degrees
        self.ties = ties
        self.missing = missing
        self.normobj_z = normobj_z
        self.active = active
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Any = None) -> 'Addals':
        """Fit Addals on X (rank-1 conjoint part-worth functions)."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        names = list(X.columns)
        data_orig = X.copy()
        data = make_numeric(X)
        nobs, nvars = data.shape

        levels_v = reshape(self.levels, nvars)  # type: ignore
        levelprep = level_to_spline(levels_v, data)  # type: ignore
        knots_v = levelprep['knotList']
        ordinal_v = levelprep['ordvec']
        degrees_v = reshape(self.degrees, nvars)  # type: ignore
        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore
        active_v = reshape(self.active, nvars)  # type: ignore

        # Addals: copies=1 enforces rank-1 per variable (additive constraint)
        copies_v = [1] * nvars

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
            sets=list(range(nvars)),  # each var = own set
        )

        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x)

        objectscores = h['x'].copy()
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        xGifi = h['xGifi']
        # Part-worth functions: the 1-D transform for each variable
        partworths = [xGifi[j][0]['transform'][:, 0] for j in range(nvars)]
        weights = [xGifi[j][0]['weights'] for j in range(nvars)]
        quants = [xGifi[j][0]['quantifications'] for j in range(nvars)]
        transforms_list = [xGifi[j][0]['transform'] for j in range(nvars)]
        rhat = cor_list(transforms_list)  # type: ignore
        evals = np.linalg.eigh(rhat)[0][::-1]

        call_str = (
            f"Addals(ndim={self.ndim}, levels={self.levels!r}, "
            f"degrees={self.degrees!r})")

        self.result_ = {
            'objectscores': objectscores,
            'partworths': partworths,       # R addals()$partworths
            'quantifications': quants,
            'weights': weights,
            'rhat': rhat,
            'evals': evals,
            'f': h['f'],
            'ntel': h['ntel'],
            'data': data_orig,
            'datanum': data,
            'call_': call_str,
        }
        self.n_obs_ = nobs
        self.n_vars_ = nvars
        self.n_iter_ = h['ntel']
        self.converged_ = h['ntel'] < self.itmax
        self.is_fitted_ = True
        return self

    def transform(self, X: Any, y: Any = None) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.result_['objectscores']  # type: ignore

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            call_str = self.result_.get('call_', f"Addals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Addals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.addals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nPart-worths (Transformations):")
        # Ensure partworths are arrays to stack them properly
        pws = [np.asarray(pw).flatten() for pw in self.result_['partworths']]
        df_pw = pd.DataFrame(pws).T
        if hasattr(self.result_['data'], 'columns'):
            df_pw.columns = self.result_['data'].columns
        else:
            df_pw.columns = [f"X{i}" for i in range(len(pws))]
        print(df_pw)
