# mypy: ignore-errors
"""
pygifi.criminals — Nonlinear Discriminant Analysis (Criminals).

Python port of Gifi/R/criminals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Criminals is a special case of Homals where one variable is the nominal
group indicator with rank constrained to 1 (a single discriminant vector).
The predictor variables use ordinal or metric transforms.
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


class Criminals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Nonlinear Discriminant Analysis via Gifi ALS (Criminals).

    Python port of R's criminals() — Homals where the group variable has
    rank=1 (nominal, single discriminant) and predictor variables get
    ordinal/metric transforms.

    Parameters
    ----------
    ndim : int, default=2
        Number of discriminant dimensions.
    group_col : int, default=-1
        Column index (0-based) of the nominal group variable.
        Default: last column (−1).
    pred_levels : str or list, default='ordinal'
        Measurement levels for predictor variables.
    ties : str, default='s'
    missing : str, default='s'
    normobj_z : bool, default=True
    itmax : int, default=1000
    eps : float, default=1e-6
    verbose : bool, default=False
    """

    def __init__(
        self,
        ndim: int = 2,
        group_col: int = -1,
        pred_levels: Union[str, List[str]] = 'ordinal',
        ties: str = 's',
        missing: str = 's',
        normobj_z: bool = True,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[np.ndarray] = None,
    ) -> None:
        self.ndim = ndim
        self.group_col = group_col
        self.pred_levels = pred_levels
        self.ties = ties
        self.missing = missing
        self.normobj_z = normobj_z
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Any = None) -> 'Criminals':
        """
        Fit Criminals on X.

        Parameters
        ----------
        X : DataFrame or array (nobs, nvars)
            One column must be the nominal group indicator.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        names = list(X.columns)
        data_orig = X.copy()
        data = make_numeric(X)
        nobs, nvars = data.shape

        # Resolve group column index
        grp_idx = self.group_col % nvars
        pred_idx = [j for j in range(nvars) if j != grp_idx]

        # Build levels: predictors use pred_levels; group is 'nominal'
        pred_lv = reshape(self.pred_levels, len(pred_idx))  # type: ignore
        levels_v = []
        pi = 0
        for j in range(nvars):
            if j == grp_idx:
                levels_v.append('nominal')
            else:
                levels_v.append(pred_lv[pi])
                pi += 1

        levelprep = level_to_spline(levels_v, data)  # type: ignore
        knots_v = levelprep['knotList']
        ordinal_v = levelprep['ordvec']
        degrees_v = levelprep['degvec']
        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore

        # Copies: group var gets rank 1 (single discriminant); predictors ndim
        copies_v = [self.ndim] * nvars
        copies_v[grp_idx] = 1   # rank=1 for group variable

        gifi = make_gifi(  # type: ignore
            data=data,
            knots=knots_v,
            degrees=degrees_v,
            ordinal=ordinal_v,
            ties=ties_v,
            copies=copies_v,
            missing=missing_v,
            active=[True] * nvars,
            names=names,
            sets=list(range(nvars)),  # each var in own set
        )

        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x)

        objectscores = h['x'].copy()
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        xGifi = h['xGifi']
        group_quants = xGifi[grp_idx][0]['quantifications']  # group centroids
        pred_transforms = [xGifi[j][0]['transform'] for j in pred_idx]
        rhat = cor_list([xGifi[j][0]['transform'] for j in range(nvars)])  # type: ignore
        evals = np.linalg.eigh(rhat)[0][::-1]

        call_str = (
            f"Criminals(ndim={self.ndim}, group_col={self.group_col}, "
            f"pred_levels={self.pred_levels!r})")

        self.result_ = {
            'objectscores': objectscores,
            'group_quantifications': group_quants,   # R: class.scores
            'pred_transforms': pred_transforms,
            'rhat': rhat,
            'evals': evals,
            'f': h['f'],
            'ntel': h['ntel'],
            'data': data_orig,
            'datanum': data,
            'group_col': grp_idx,
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
            call_str = self.result_.get('call_', f"Criminals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Criminals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.criminals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nGroup Quantifications (Class Coords):")
        df_group = pd.DataFrame(self.result_['group_quantifications'], 
                                columns=[f"D{d + 1}" for d in range(self.ndim)])
        print(df_group)
