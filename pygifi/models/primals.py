# mypy: ignore-errors
"""
pygifi.primals — Optimal Scaling PCA (Primals).

Python port of Gifi/R/primals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Primals is a Princals wrapper where measurement levels are user-specified
metric/ordinal transforms — the standard "optimal scaling PCA" workflow.
"""
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pygifi.models.princals import Princals


class Primals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Optimal Scaling Principal Component Analysis (Primals).

    Python port of R's primals() — Princals with metric/ordinal transforms
    specified explicitly per variable.

    Parameters
    ----------
    ndim : int, default=2
    levels : str or list of str, default='metric'
        'nominal', 'ordinal', or 'metric' per variable.
    degrees : int or list, default=2
        Degree of polynomial/spline transformation.
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
        ndim: int = 2,
        levels: Union[str, List[str]] = 'metric',
        degrees: Union[int, List[int]] = 2,
        ties: str = 's',
        missing: str = 's',
        normobj_z: bool = True,
        active: Union[bool, List[bool]] = True,
        copies: Union[int, List[int]] = 1,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[np.ndarray] = None,
        optimizer: str = 'als',
    ) -> None:
        self.ndim = ndim
        self.levels = levels
        self.degrees = degrees
        self.ties = ties
        self.missing = missing
        self.normobj_z = normobj_z
        self.active = active
        self.copies = copies
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x
        self.optimizer = optimizer

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Any = None) -> 'Primals':
        """Fit Primals on X."""
        self._princals = Princals(
            ndim=self.ndim,
            levels=self.levels,
            degrees=self.degrees,
            copies=self.copies,
            ties=self.ties,
            missing=self.missing,
            normobj_z=self.normobj_z,
            active=self.active,
            itmax=self.itmax,
            eps=self.eps,
            verbose=self.verbose,
            init_x=self.init_x,
            optimizer=self.optimizer,
        ).fit(X)

        self.result_ = self._princals.result_.copy()
        # Override call_ to reflect Primals
        self.result_['call_'] = (
            f"Primals(ndim={self.ndim}, levels={self.levels!r}, "
            f"degrees={self.degrees!r})")

        self.n_obs_ = self._princals.n_obs_
        self.n_vars_ = self._princals.n_vars_
        self.n_iter_ = self._princals.n_iter_
        self.converged_ = self._princals.converged_
        self.is_fitted_ = True
        return self

    def transform(self, X: Any, y: Any = None) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        return self.result_['objectscores']  # type: ignore

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            call_str = self.result_.get('call_', f"Primals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Primals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.primals()."""
        check_is_fitted(self, 'is_fitted_')
        self._princals.summary()
