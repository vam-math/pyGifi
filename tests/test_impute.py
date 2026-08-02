import pytest
import numpy as np
import pandas as pd
from pygifi.models.impute import GifiIterativeImputer
from pygifi.models.princals import Princals


@pytest.fixture
def data_with_nans():
    # Make a simple dataset with missing values
    np.random.seed(42)
    # A clear 1D structure: var A matches B perfectly but has NaNs
    df = pd.DataFrame({
        'A': [1, 2, 3, 1, 2, 3, np.nan, 2, 3, 1],
        'B': [1, 2, 3, 1, 2, 3, 1, np.nan, 3, 1],
        'C': [2, 3, 1, 2, 3, 1, 2, 3, np.nan, 2]
    })
    return df


def test_imputer_initialization(data_with_nans):
    """Test that it can initialize and fit without crashing."""
    base_model = Princals(ndim=1)
    imputer = GifiIterativeImputer(estimator=base_model, max_iter=2, verbose=True)

    # Fit the imputer
    imputer = imputer.fit(data_with_nans)

    assert imputer.is_fitted_
    assert hasattr(imputer, 'imputed_data_')
    assert hasattr(imputer, 'estimator_')

    # Check that original base model is unchanged
    assert not hasattr(base_model, 'result_')


def test_imputer_no_nans():
    """Test that passing data with no NaNs skips imputation loop."""
    df = pd.DataFrame({'A': [1, 2], 'B': [2, 1]})
    base_model = Princals(ndim=1)
    imputer = GifiIterativeImputer(estimator=base_model)

    imputer.fit(df)
    assert imputer.n_iter_ == 0
    assert imputer.converged_
    assert np.array_equal(imputer.imputed_data_, df.values)


def test_imputer_fills_nans(data_with_nans):
    """Test that the resulting imputed data has no NaNs."""
    base_model = Princals(ndim=1)
    # Give it enough iterations to converge
    imputer = GifiIterativeImputer(estimator=base_model, max_iter=10)

    imputer.fit(data_with_nans)

    assert not np.isnan(imputer.imputed_data_).any()
    # It should have run at least 1 iteration
    assert imputer.n_iter_ > 0


def test_imputer_transform(data_with_nans):
    """Test standard transform method delegation."""
    base_model = Princals(ndim=2)
    imputer = GifiIterativeImputer(estimator=base_model, max_iter=5)

    imputer.fit(data_with_nans)
    scores = imputer.transform(data_with_nans)

    assert scores.shape == (10, 2)
    assert not np.isnan(scores).any()
