# mypy: ignore-errors
"""
pygifi.overals — Multiblock Canonical Correlation (OVERALS).

Python port of Gifi/R/overals.R (Mair, De Leeuw, Groenen. GPL-3.0).

OVERALS generalises canonical correlation to multiple blocks of variables
with full optimal scaling.  Each block has its own set membership, level,
copies (rank), and ordinal constraints.
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


class Overals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Multiblock Canonical Correlation Analysis via Gifi ALS (OVERALS).

    Python port of R's overals() — full per-block sets + copies + rank sweep.

    Parameters
    ----------
    ndim : int, default=2
        Number of canonical variates.
    sets : list of int
        Block membership per variable (0-indexed). **Required.**
    levels : str or list, default='ordinal'
    copies : int or list, default=1
        Copies per variable (rank within each variable's basis).
    rank : int or list or None, default=None
        Per-block rank constraint.
    ties : str, default='s'
    missing : str, default='s'
    normobj_z : bool, default=False
        OVERALS uses unit-norm object scores (False = R default).
    active : bool or list, default=True
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
        rank: Optional[Union[int, List[int]]] = None,
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
        self.rank = rank
        self.ties = ties
        self.missing = missing
        self.normobj_z = normobj_z
        self.active = active
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Any = None) -> 'Overals':
        """Fit OVERALS on X with block membership given by sets."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        names = list(X.columns)
        data_orig = X.copy()
        data = make_numeric(X)
        nobs, nvars = data.shape

        if self.sets is None:
            raise ValueError(
                "Overals requires a 'sets' parameter specifying block "
                "membership for each variable.")
        if len(self.sets) != nvars:
            raise ValueError(
                f"sets must have length nvars={nvars}, got {len(self.sets)}.")

        levels_v = reshape(self.levels, nvars)  # type: ignore
        levelprep = level_to_spline(levels_v, data)  # type: ignore
        knots_v = levelprep['knotList']
        ordinal_v = levelprep['ordvec']
        degrees_v = levelprep['degvec']
        copies_v = reshape(self.copies, nvars)  # type: ignore
        ties_v = reshape(self.ties, nvars)  # type: ignore
        missing_v = reshape(self.missing, nvars)  # type: ignore
        active_v = reshape(self.active, nvars)  # type: ignore

        # Apply per-variable rank constraint
        if self.rank is not None:
            rank_v = reshape(self.rank, nvars)  # type: ignore
            rank_v = [min(int(r), self.ndim) for r in rank_v]
            copies_v = [min(c, r) for c, r in zip(copies_v, rank_v)]

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
        # xGifi is indexed by set, then var-within-set
        nblocks = max(self.sets) + 1
        set_var_counts = [0] * nblocks
        all_transforms = []
        block_transforms: list = [[] for _ in range(nblocks)]
        for j in range(nvars):
            sv = self.sets[j]
            vi = set_var_counts[sv]
            jg = xGifi[sv][vi]
            set_var_counts[sv] += 1
            tr_j = jg['transform']
            all_transforms.append(tr_j)
            block_transforms[sv].append(tr_j)

        block_evals = []
        for b in range(nblocks):
            if block_transforms[b]:
                brhat = cor_list(block_transforms[b])  # type: ignore
                block_evals.append(np.linalg.eigh(brhat)[0][::-1])

        rhat = cor_list(all_transforms)  # type: ignore
        evals = np.linalg.eigh(rhat)[0][::-1]

        # Fit quality (R overals() reports sum of eigenvalues / nblocks)
        fit = float(np.sum(evals[:self.ndim])) / max(nblocks, 1)

        call_str = (
            f"Overals(ndim={self.ndim}, sets={self.sets!r}, "
            f"levels={self.levels!r}, copies={self.copies!r})")

        self.result_ = {
            'objectscores': objectscores,
            'block_evals': block_evals,  # R overals()$eigenvalues per block
            'rhat': rhat,
            'evals': evals,
            'fit': fit,               # R overals()$fit
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
            call_str = self.result_.get('call_', f"Overals(ndim={self.ndim})")
            evals = self.result_['evals'][:self.ndim]
            evals_str = "  ".join([f"{e:.6f}" for e in evals])
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Fit value: {self.result_['fit']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Eigenvalues: {evals_str}")
        return f"Overals(ndim={self.ndim})"

    def summary(self) -> None:
        """Print extended summary matching R's summary.overals()."""
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nBlock Eigenvalues:")
        for idx, block_ev in enumerate(self.result_['block_evals']):
            print(f"Block {idx}: {list(block_ev[:self.ndim])}")
