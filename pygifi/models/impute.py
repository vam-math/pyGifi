"""
pygifi.models.impute — Iterative Missing Value Imputation for Gifi Models.

This module provides a scikit-learn compatible wrapper that iteratively imputes
missing values within categorical datasets by leveraging the optimal scaling
transformations of an underlying Gifi model (Princals, Homals, Morals).
"""

import numpy as np
import pandas as pd
from typing import Union, Any
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from pygifi.utils.coding import make_numeric


class GifiIterativeImputer(BaseEstimator, TransformerMixin):  # type: ignore
    """
    Iterative Missing Value Imputer using Optimal Scaling.

    Wraps a Gifi model (e.g., Princals, Homals) to iteratively impute NaNs.

    Algorithm:
    1. Initialize missing entries using the specified `init_strategy` (e.g., mode).
    2. Fit the base `estimator` on the completed data to obtain object scores and
       category quantifications.
    3. Update missing entries: assign them to the category whose quantification is
       closest (Euclidean distance) to the observation's object score.
    4. Repeat until convergence (imputed values stop changing) or `max_iter`.

    Parameters
    ----------
    estimator : estimator object
        A Gifi model instance (e.g., Princals(), Homals()) to use for scaling.
        Note: The estimator's `missing` parameter should ideally be 's' (single
        omission) or standard handling since there will be no actual NaNs during
        its fit.
    max_iter : int, default=10
        Maximum number of imputation iterations.
    init_strategy : str, default='mode'
        Initialization strategy for NaNs:
        - 'mode': Most frequent observed category per variable.
        - 'random': Uniform random draw from observed categories per variable.
    verbose : bool, default=False
        Print iteration progress and number of changed entries.

    Attributes
    ----------
    estimator_ : estimator object
        The final fitted base estimator.
    imputed_data_ : np.ndarray
        The final completed numeric dataset.
    n_iter_ : int
        Number of iterations executed.
    converged_ : bool
        Whether the imputation converged before max_iter.
    """

    def __init__(
        self,
        estimator: Any,
        max_iter: int = 10,
        init_strategy: str = 'mode',
        verbose: bool = False
    ) -> None:
        self.estimator = estimator
        self.max_iter = max_iter
        self.init_strategy = init_strategy
        self.verbose = verbose

    def fit(self, X: Union[pd.DataFrame, Any], y: Any = None) -> 'GifiIterativeImputer':
        """Fit the imputer and the base estimator on X."""
        # --- 1. Preparation and Initialization ---
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.empty:
            raise ValueError("Input data X cannot be empty.")

        # Extract numeric representation (preserves NaNs as np.nan)
        data = make_numeric(X)
        self.nobs_, self.nvars_ = data.shape

        # Identify missing mask
        self.missing_mask_ = np.isnan(data)
        if not np.any(self.missing_mask_):
            if self.verbose:
                print("No missing values found. Fitting standard estimator directly.")
            self.estimator_ = clone(self.estimator).fit(data)
            self.imputed_data_ = data.copy()
            self.n_iter_ = 0
            self.converged_ = True
            self.is_fitted_ = True
            return self

        # Initialize missing values
        X_imputed = data.copy()
        for j in range(self.nvars_):
            col_missing = self.missing_mask_[:, j]
            if not np.any(col_missing):
                continue

            valid_vals = data[~col_missing, j]
            if len(valid_vals) == 0:
                raise ValueError(f"Variable {j} has no valid observations to impute from.")

            unique_vals, counts = np.unique(valid_vals, return_counts=True)

            if self.init_strategy == 'mode':
                mode_val = unique_vals[np.argmax(counts)]
                fill_vals = mode_val
            elif self.init_strategy == 'random':
                fill_vals = np.random.choice(unique_vals, size=np.sum(col_missing))
            else:
                raise ValueError(f"Unknown init_strategy: {self.init_strategy}. Use 'mode' or 'random'.")

            X_imputed[col_missing, j] = fill_vals

        if self.verbose:
            total_missing = np.sum(self.missing_mask_)
            print(f"Initialized {total_missing} missing values using '{self.init_strategy}' strategy.")

        # --- 2. Iterative Imputation Loop ---
        self.estimator_ = clone(self.estimator)

        for it in range(1, self.max_iter + 1):
            if self.verbose:
                print(f"Iteration {it}/{self.max_iter}...", end=" ")

            # Fit estimator on currently imputed data
            self.estimator_.fit(X_imputed)

            # Retrieve object scores (nobs, ndim)
            object_scores = self.estimator_.result_['objectscores']

            # Retrieve category quantifications
            # Note: Homals stores them in a dict {col_name: ...}, Princals in a list
            if hasattr(self.estimator_, 'quantifications'):
                q_source = self.estimator_.quantifications
                if isinstance(q_source, dict):
                    quants = list(q_source.values())
                else:
                    quants = q_source
            elif 'quantifications' in self.estimator_.result_:
                quants = self.estimator_.result_['quantifications']
            else:
                raise AttributeError("Estimator result does not contain 'quantifications'.")

            changes = 0

            # Update missing entries by minimal distance
            for j in range(self.nvars_):
                col_missing = self.missing_mask_[:, j]
                if not np.any(col_missing):
                    continue

                # Get observed categories map: the Gifi structure extracts levels
                # The basis indices correspond to sorted unique valid values.
                unique_vals = np.sort(np.unique(data[~col_missing, j]))
                Q_j = quants[j]  # shape: (n_categories, ndim)

                # For every missing observation in variable j
                for i in np.where(col_missing)[0]:
                    x_i = object_scores[i, :]

                    # Compute squared Euclidean distance from object score x_i to all categories q_jk
                    # Since quantifications matrix Q_j row k corresponds to unique_vals[k]
                    diffs = Q_j - x_i
                    dists = np.sum(diffs ** 2, axis=1)

                    # Find closest category index
                    best_idx = np.argmin(dists)
                    new_val = unique_vals[best_idx]

                    if X_imputed[i, j] != new_val:
                        X_imputed[i, j] = new_val
                        changes += 1

            if self.verbose:
                print(f"{changes} entries reassigned.")

            # Check convergence
            if changes == 0:
                if self.verbose:
                    print(f"Converged at iteration {it}.")
                self.converged_ = True
                self.n_iter_ = it
                break
        else:
            if self.verbose:
                print(f"Reached max_iter ({self.max_iter}) without full convergence.")
            self.converged_ = False
            self.n_iter_ = self.max_iter

        self.imputed_data_ = X_imputed
        self.is_fitted_ = True
        return self

    def transform(self, X: Any, y: Any = None) -> Any:
        """
        Return the object scores for the fitted data.
        """
        check_is_fitted(self, 'is_fitted_')
        # We delegate to the underlying estimator's transform which expects
        # the same data it was trained on.
        return self.estimator_.transform(X)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Fit the imputer and return the object scores."""
        return self.fit(X).transform(X)
