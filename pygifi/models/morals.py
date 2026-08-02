"""
pygifi.morals — Monotone Regression Analysis (Morals).

Python port of Gifi/R/morals.R.
R original: morals() by Mair, De Leeuw, Groenen. GPL-3.0.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin

from pygifi.utils.coding import make_numeric
from pygifi.utils.utilities import reshape
from pygifi.utils.splines import knots_gifi
from pygifi.core.structures import make_gifi
from pygifi.core.engine import gifi_engine
from pygifi.core.linalg import ls_rc


class Morals(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Monotone Regression Analysis via Optimal Scaling.

    Python port of R's morals(). Regresses y on X with optimal monotone
    transformation of both predictor and response variables.

    Parameters
    ----------
    xknots : list, default=None
    yknots : list, default=None
    xdegrees : int or list, default=2
    ydegrees : int, default=2
    xordinal : bool or list, default=True
    yordinal : bool, default=True
    xties : str or list, default="s"
    yties : str, default="s"
    xmissing : str or list, default="m"
    ymissing : str, default="m"
    xactive : bool or list, default=True
    xcopies : int or list, default=1
    itmax : int, default=1000
        Maximum number of ALS iterations.
    eps : float, default=1e-6
        Convergence criterion.
    verbose : bool, default=False
        Print iteration info.
    """

    def __init__(
        self,
        xknots: Optional[List[Any]] = None,
        yknots: Optional[List[Any]] = None,
        xdegrees: Union[int, List[int]] = 2,
        ydegrees: int = 2,
        xordinal: Union[bool, List[bool]] = True,
        yordinal: bool = True,
        xties: Union[str, List[str]] = "s",
        yties: str = "s",
        xmissing: Union[str, List[str]] = "m",
        ymissing: str = "m",
        xactive: Union[bool, List[bool]] = True,
        xcopies: Union[int, List[int]] = 1,
        itmax: int = 1000,
        eps: float = 1e-6,
        verbose: bool = False,
        init_x: Optional[Any] = None,
        r_seed: Optional[int] = None,
        optimizer: str = 'als',
    ) -> None:
        self.xknots = xknots
        self.yknots = yknots
        self.xdegrees = xdegrees
        self.ydegrees = ydegrees
        self.xordinal = xordinal
        self.yordinal = yordinal
        self.xties = xties
        self.yties = yties
        self.xmissing = xmissing
        self.ymissing = ymissing
        self.xactive = xactive
        self.xcopies = xcopies
        self.itmax = itmax
        self.eps = eps
        self.verbose = verbose
        self.init_x = init_x
        self.r_seed = r_seed
        self.optimizer = optimizer

    def fit(self,
            X: Union[pd.DataFrame,
                     Any],
            y: Union[pd.Series,
                     pd.DataFrame,
                     Any],
            sample_weight: Any = None) -> 'Morals':
        """Fit Morals model to predictors X and response y."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        from pygifi.utils.utilities import sanitize_dataframe
        X = sanitize_dataframe(X)
        if X.empty:
            raise ValueError("Input data X cannot be empty (or was entirely dropped).")
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)

        npred, nobs = X.shape[1], X.shape[0]

        # Coerce to numeric (handles categories/factors like R)
        X_num = make_numeric(X)
        y_num = make_numeric(pd.DataFrame(y))
        y_num = y_num.ravel()

        xnames = list(X.columns)

        # Broadcast per-predictor params
        xdegrees_v = reshape(self.xdegrees, npred)  # type: ignore
        xordinal_v = reshape(self.xordinal, npred)  # type: ignore
        xties_v = reshape(self.xties, npred)  # type: ignore
        xmissing_v = reshape(self.xmissing, npred)  # type: ignore
        xactive_v = reshape(self.xactive, npred)  # type: ignore
        xcopies_v = reshape(self.xcopies, npred)  # type: ignore

        # Default knots
        if self.xknots is None:
            xknots = [
                knots_gifi(X.iloc[:, [i]], type="E", n=None)[0]  # type: ignore
                for i in range(npred)
            ]
        else:
            xknots = self.xknots

        if self.yknots is None:
            yknots = [
                knots_gifi(
                    pd.DataFrame(y),
                    type="E",
                    n=None)[0]]  # type: ignore
        else:
            yknots = self.yknots

        # Combine X+y into one data matrix
        data = np.column_stack([X_num, y_num])

        # All X share set 0, Y is set 1
        sets = [0] * npred + [1]

        gifi = make_gifi(  # type: ignore
            data=pd.DataFrame(data),
            knots=xknots + yknots,
            degrees=xdegrees_v + [self.ydegrees],
            ordinal=xordinal_v + [self.yordinal],
            ties=xties_v + [self.yties],
            copies=xcopies_v + [1],
            missing=xmissing_v + [self.ymissing],
            active=xactive_v + [True],
            names=xnames + ["Y"],
            sets=sets,
        )

        h = gifi_engine(  # type: ignore
            gifi, ndim=1, itmax=self.itmax, eps=self.eps,
            verbose=self.verbose, init_x=self.init_x, r_seed=self.r_seed
        )

        # --- Optional majorization refinement (Morals: all monotone, ndim=1) ---
        if self.optimizer == 'majorization':
            from pygifi.core.engine import gifi_majorization
            from pygifi.utils.utilities import make_indicator
            n_total = npred + 1          # predictors + response
            X_als = h['x']
            H_list = [make_indicator(data[:, j]) for j in range(n_total)]  # type: ignore[no-untyped-call]
            A_list = [np.ones((1, 1)) for _ in range(n_total)]
            ordinals = xordinal_v + [self.yordinal]
            X_ref, _, _ = gifi_majorization(  # type: ignore[no-untyped-call]
                X_als, H_list, A_list,
                max_iter=self.itmax, tol=self.eps,
                ordinal=ordinals,        # Morals: monotone constraint per var
            )
            h['x'] = X_ref

        # Collect transforms
        xhat = np.hstack([h["xGifi"][0][j]["transform"] for j in range(npred)])
        yhat = h["xGifi"][1][0]["transform"][:, 0]

        # OLS: xhat -> yhat
        beta = ls_rc(xhat, yhat)["solution"].flatten()  # type: ignore
        ypred = xhat @ beta
        yres = yhat - ypred
        smc = np.sum(yhat * ypred)

        rhat = np.corrcoef(np.column_stack([xhat, yhat]).T)
        evals = np.linalg.eigh(rhat)[0][::-1]

        # --- F-statistics per predictor (R morals() fstats field) ---
        # F_j = (smc_j / 1) / ((1 - smc) / (nobs - npred - 1))
        # where smc_j is the partial correlation of predictor j with yhat
        df_res = max(nobs - npred - 1, 1)
        fstats = np.zeros(npred)
        for j in range(npred):
            rho_j = float(np.corrcoef(xhat[:, j], yhat)[0, 1])
            smc_j = rho_j ** 2
            denom = (1.0 - smc_j) / df_res
            fstats[j] = (smc_j / 1.0) / denom if denom > 1e-15 else np.inf

        # Standardised betas (R: standardise by std of each xhat column)
        betas_std = beta.copy()
        for j in range(npred):
            std_j = np.std(xhat[:, j])
            if std_j > 1e-12:
                betas_std[j] = beta[j] * std_j

        call_str = (
            f"Morals(xdegrees={self.xdegrees!r}, ydegrees={self.ydegrees!r}, "
            f"xordinal={self.xordinal!r}, yordinal={self.yordinal!r})")

        self.result_ = {
            "rhat": rhat,
            "evals": evals,
            "objectscores": h["x"],
            "xGifi": h["xGifi"],
            "xhat": xhat,
            "yhat": yhat,
            "beta": beta,
            "betas": betas_std,           # R morals() field (standardised)
            "ypred": ypred,
            "yres": yres,
            "residuals": yres,            # R morals() alias
            "smc": smc,
            "fstats": fstats,            # R morals() field
            "f": h["f"],
            "ntel": h["ntel"],
            "xknots": xknots,
            "yknots": yknots,
            "xordinal": xordinal_v,
            "yordinal": self.yordinal,
            "call_": call_str,           # R match.call() equivalent
        }
        # --- Extract basis parameters for out-of-sample projection ---
        self.basis_params_ = []
        for j in range(npred):
            xj = X_num[:, j]
            valid_xj = xj[~np.isnan(xj)]
            lowknot = float(np.min(valid_xj)) if len(valid_xj) > 0 else 0.0
            highknot = float(np.max(valid_xj)) if len(valid_xj) > 0 else 0.0
            
            degree = xdegrees_v[j]
            innerknots = np.asarray(xknots[j], dtype=float)
            knots = np.concatenate([np.repeat(lowknot, degree + 1), innerknots, np.repeat(highknot, degree + 1)])
            nf = len(innerknots) + degree + 1
            
            try:
                from scipy.interpolate import BSpline
                b_full = BSpline.design_matrix(valid_xj, knots, degree).toarray()
            except Exception:
                from pygifi.utils.splines import _deboor_basis
                b_full = _deboor_basis(valid_xj, knots, degree, nf)
                
            right_mask = valid_xj == highknot
            if np.any(right_mask):
                b_full[right_mask, :] = 0.0
                b_full[right_mask, nf - 1] = 1.0
                
            nonzero_cols = np.where(b_full.sum(axis=0) > 0)[0]
            self.basis_params_.append({
                'lowknot': lowknot,
                'highknot': highknot,
                'keep_cols': nonzero_cols,
                'degree': degree,
                'innerknots': innerknots
            })

        self.n_obs_ = nobs
        self.n_pred_ = npred
        self.n_iter_ = h["ntel"]
        self.X_ = X
        self.y_ = y
        self.converged_ = h["ntel"] < self.itmax
        self.is_fitted_ = True
        return self

    def _transform_x(self, X: Any) -> np.ndarray:
        """Apply learned transformations to new data X."""
        from pygifi.utils.splines import bspline_basis
        from pygifi.utils.utilities import sanitize_dataframe
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "is_fitted_")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = sanitize_dataframe(X)
        X_num = make_numeric(X)
        
        nobs = X_num.shape[0]
        npred = self.n_pred_

        if X_num.shape[1] != npred:
            raise ValueError(f"Expected {npred} predictors, got {X_num.shape[1]}")

        # Reconstruct transformed X (xhat)
        h_x_list = []
        for j in range(npred):
            xj = X_num[:, j]
            params = self.basis_params_[j]
            
            # 1. B-spline basis
            basis = bspline_basis(
                xj, 
                degree=params['degree'], 
                innerknots=params['innerknots'], 
                lowknot=params['lowknot'], 
                highknot=params['highknot'], 
                keep_cols=params['keep_cols']
            )
            
            # 2. Multiply by optimal quantifications
            qj = self.result_["xGifi"][0][j]["quantifications"]
            scores_j = basis @ qj
            
            # 3. Multiply by weights
            wj = self.result_["xGifi"][0][j]["weights"]
            hj = scores_j @ wj
            h_x_list.append(hj)

        return np.hstack(h_x_list)

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict response values for new data X.
        
        Applies out-of-sample optimal scaling transformations to X and
        computes predictions using the fitted regression coefficients (beta).
        """
        xhat_new = self._transform_x(X)
        beta = self.result_["beta"]
        return xhat_new @ beta

    def transform(self, X: Any) -> Any:
        """
        Returns the transformed variables for the fitted data.
        """
        import warnings
        warnings.warn("transform() on new data returns the optimally scaled predictor matrix.", UserWarning)
        return self._transform_x(X)

    def __repr__(self) -> str:
        if hasattr(self, 'result_'):
            call_str = self.result_.get('call_', "Morals()")
            return (f"Call: {call_str}\n\n"
                    f"Loss value: {self.result_['f']:.6f}\n"
                    f"Number of iterations: {self.result_['ntel']}\n"
                    f"Squared Multiple Correlation (SMC): {self.result_['smc']:.6f}")
        return "Morals()"

    def summary(self) -> None:
        """Print extended summary matching R's summary.morals()."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, 'is_fitted_')
        print(repr(self))
        print("\nRegression Coefficients (Beta):")
        beta = self.result_['beta']
        columns = getattr(self.X_, 'columns', [f"X{i}" for i in range(len(beta))])
        df_beta = pd.DataFrame({'Beta': beta}, index=columns)
        print(df_beta)

    @property
    def eigenvalues_(self) -> Optional[Any]:
        """Array of eigenvalues from the correlation matrix of transformations."""
        return self.result_.get('evals') if hasattr(self, 'result_') else None

    @property
    def variance_explained_(self) -> Optional[Any]:
        """Percentage of variance explained by each dimension."""
        evals = self.eigenvalues_
        if evals is not None:
            return (evals / np.sum(evals)) * 100
        return None

    @property
    def component_loadings_(self) -> Optional[Any]:
        """Regression coefficients (beta) serving as component loadings."""
        return self.result_.get('beta') if hasattr(self, 'result_') else None

    @property
    def category_quantifications_(self) -> None:
        """Morals (monotone regression) extracts transformations directly instead of standard maps."""
        return None

    @property
    def object_scores_(self) -> Optional[Any]:
        """The optimized object scores (coordinates)."""
        return self.result_.get('objectscores') if hasattr(self, 'result_') else None
