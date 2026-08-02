# mypy: ignore-errors
"""
pygifi.canals — Canonical Correlation Analysis (Canals).

Python port of Gifi/R/canals.R (Mair, De Leeuw, Groenen. GPL-3.0).

Canals finds canonical variates between two (or more) blocks of variables.
Each block forms a set; variables within a block share the same object scores.
normobj_z=False matches R's canals default (unit-norm object scores).
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


class Canals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Canonical Correlation Analysis via Gifi ALS (Canals).

    Python port of R's canals() — Homals with user-defined variable blocks.

    Parameters
    ----------
    ndim : int, default=2
    sets : list of int
        Block membership per variable (0-indexed). E.g. [0,0,1,1] means
        vars 0–1 in block A, vars 2–3 in block B. **Required.**
    levels : str or list of str, default='ordinal'
    ties : str, default='s'
    missing : str, default='s'
    normobj_z : bool, default=False
        R canals() uses unit-norm object scores (False matches R default).
    itmax : int, default=1000
    eps : float, default=1e-6
    verbose : bool, default=False
    """

    def __init__(
        self,
        ndim: int = 2,
        sets: Optional[List[int]] = None,
        levels: Union[str, List[str]] = 'ordinal',
        copies: Union[int, List[int]] = 1,
        ties: str = 's',
        missing: str = 's',
        normobj_z: bool = False,
        active: Union[bool, List[bool]] = True,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[np.ndarray] = None,
    ) -> None:
        self.ndim = ndim
        self.sets = sets
        self.levels = levels
        self.copies = copies
        self.ties = ties
        self.missing = missing
        self.normobj_z = normobj_z
        self.active = active
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Any = None) -> 'Canals':
        """Fit Canals on X (nvars columns; block membership given by sets)."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        names = list(X.columns)
        data_orig = X.copy()
        data = make_numeric(X)
        nobs, nvars = data.shape

        if self.sets is None:
            raise ValueError(
                "Canals requires a 'sets' parameter specifying block "
                "membership for each variable. E.g. sets=[0,0,1,1].")
        if len(self.sets) != nvars:
            raise ValueError(
                f"sets must have length nvars={nvars}, "
                f"got {len(self.sets)}.")

        levels_v = reshape(self.levels, nvars)  # type: ignore
        levelprep = level_to_spline(levels_v, data)  # type: ignore
        knots_v = levelprep['knotList']
        ordinal_v = levelprep['ordvec']
        degrees_v = levelprep['degvec']
        copies_v = reshape(self.copies, nvars)  # type: ignore
        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore
        active_v = reshape(self.active, nvars)  # type: ignore

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
            sets=list(self.sets),
        )

        h = gifi_engine(gifi, ndim=self.ndim, itmax=self.itmax,  # type: ignore
                        eps=self.eps, verbose=self.verbose,
                        init_x=self.init_x)

        objectscores = h['x'].copy()
        if self.normobj_z:
            objectscores = objectscores * np.sqrt(nobs)

        xGifi = h['xGifi']
        # xGifi is indexed by set, then by variable-within-set:
        # xGifi[set_idx][var_within_set_idx]
        # Build per-set variable lists for correct indexing
        nblocks = max(self.sets) + 1
        set_var_counts = [0] * nblocks          # track position within each set
        block_quants: list = [[] for _ in range(nblocks)]
        transforms_list = []
        for j in range(nvars):
            sv = self.sets[j]
            vi = set_var_counts[sv]             # position of var j within set sv
            jg = xGifi[sv][vi]
            set_var_counts[sv] += 1
            transforms_list.append(jg['transform'])
            block_quants[sv].append(jg['quantifications'])



        rhat = cor_list(transforms_list)  # type: ignore
        evals = np.linalg.eigh(rhat)[0][::-1]

        call_str = (
            f"Canals(ndim={self.ndim}, sets={self.sets!r}, "
            f"levels={self.levels!r})")

        self.result_ = {
            'objectscores': objectscores,
            'block_quantifications': block_quants,
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
            call_str = self.result_.get('call_', f"Canals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Canals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.canals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        for sv, block_q in enumerate(self.result_['block_quantifications']):
            print(f"\nBlock {sv} Quantifications:")
            for var_idx, q in enumerate(block_q):
                print(f"  -- Variable {var_idx} --")
                df_q = pd.DataFrame(q, columns=[f"D{d + 1}" for d in range(self.ndim)])
                print(df_q)
