# mypy: ignore-errors
"""
pygifi.corals — Correspondence Analysis (Corals).

Python port of Gifi/R/corals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Corals is a special case of Homals for 2-variable correspondence analysis.
Row variable is set 0; column variable is set 1.  The resulting object scores
give the principal coordinates of both row and column categories.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pygifi.utils.coding import make_numeric
from pygifi.utils.utilities import cor_list
from pygifi.utils.prepspline import level_to_spline
from pygifi.core.structures import make_gifi
from pygifi.core.engine import gifi_engine


class Corals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Correspondence Analysis via Gifi ALS (Corals).

    Python port of R's corals() — a 2-variable special case of homals()
    where rows form set 0 and columns form set 1.

    Parameters
    ----------
    ndim : int, default=2
        Number of dimensions.
    ties : str, default='s'
    missing : str, default='s'
    normobj_z : bool, default=True
    itmax : int, default=1000
    eps : float, default=1e-6
    verbose : bool, default=False

    Usage
    -----
    Pass a two-column DataFrame (or 2-D array) where each column is one
    categorical variable.  The first column is the *row* variable; the
    second is the *column* variable.
    """

    def __init__(
        self,
        ndim: int = 2,
        ties: str = 's',
        missing: str = 's',
        normobj_z: bool = True,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[np.ndarray] = None,
    ) -> None:
        self.ndim = ndim
        self.ties = ties
        self.missing = missing
        self.normobj_z = normobj_z
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Any = None) -> 'Corals':
        """
        Fit Corals on a two-column data matrix.

        Parameters
        ----------
        X : DataFrame or array (nobs, 2)
            Column 0 = row variable (set 0); Column 1 = column variable (set 1).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] != 2:
            raise ValueError(
                f"Corals expects exactly 2 columns (row var, col var), "
                f"got {X.shape[1]}.")

        names = list(X.columns)
        data_orig = X.copy()
        data = make_numeric(X)
        nobs, nvars = data.shape

        levelprep = level_to_spline(['nominal', 'nominal'], data)
        knots_v = levelprep['knotList']
        ordinal_v = levelprep['ordvec']
        degrees_v = levelprep['degvec']
        copies_v = [self.ndim, self.ndim]

        # Corals: row var = set 0, col var = set 1
        gifi = make_gifi(  # type: ignore
            data=data,
            knots=knots_v,
            degrees=degrees_v,
            ordinal=ordinal_v,
            ties=[self.ties, self.ties],
            copies=copies_v,
            missing=[self.missing, self.missing],
            active=[True, True],
            names=names,
            sets=[0, 1],
        )

        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x)

        xGifi = h['xGifi']
        objectscores = h['x'].copy()
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        row_quantifications = xGifi[0][0]['quantifications']  # row categories
        col_quantifications = xGifi[1][0]['quantifications']  # col categories
        row_transform = xGifi[0][0]['transform']
        col_transform = xGifi[1][0]['transform']
        rhat = cor_list([row_transform, col_transform])  # type: ignore
        evals = np.linalg.eigh(rhat)[0][::-1]

        call_str = (
            f"Corals(ndim={self.ndim}, ties={self.ties!r}, "
            f"missing={self.missing!r}, normobj_z={self.normobj_z})")

        self.result_ = {
            'objectscores': objectscores,
            'row_quantifications': row_quantifications,  # R: row.scores
            'col_quantifications': col_quantifications,  # R: col.scores
            'row_transform': row_transform,
            'col_transform': col_transform,
            'rhat': rhat,
            'evals': evals,
            'f': h['f'],
            'ntel': h['ntel'],
            'data': data_orig,
            'datanum': data,
            'call_': call_str,
        }
        self.n_obs_ = nobs
        self.n_iter_ = h['ntel']
        self.converged_ = h['ntel'] < self.itmax
        self.is_fitted_ = True
        return self

    def transform(self, X: Any, y: Any = None) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.result_['objectscores']  # type: ignore

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            call_str = self.result_.get('call_', f"Corals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Corals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.corals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nRow Quantifications (Scores):")
        df_row = pd.DataFrame(self.result_['row_quantifications'], 
                              columns=[f"D{d + 1}" for d in range(self.ndim)])
        print(df_row)
        print("\nColumn Quantifications (Scores):")
        df_col = pd.DataFrame(self.result_['col_quantifications'], 
                              columns=[f"D{d + 1}" for d in range(self.ndim)])
        print(df_col)
