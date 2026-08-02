"""Tests for pygifi.princals — Categorical Principal Component Analysis."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pygifi import Princals


@pytest.fixture
def numeric_df():
    """10 obs, 3 continuous vars with 5+ unique values for polynomial basis."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'a': np.round(rng.uniform(1, 5, 10), 1),
        'b': np.round(rng.uniform(1, 5, 10), 1),
        'c': np.round(rng.uniform(1, 5, 10), 1),
    })


@pytest.fixture
def cat_df():
    """10 obs, 3 categorical vars (3+ categories each)."""
    return pd.DataFrame({
        'a': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
        'b': [2, 1, 3, 2, 1, 3, 1, 2, 3, 1],
        'c': [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
    })


# ---------- Basic fit / shape ----------

def test_princals_fit_runs(numeric_df):
    """fit() should not raise on numeric ordinal data."""
    Princals(ndim=2, itmax=200).fit(numeric_df)


def test_princals_objectscores_shape(numeric_df):
    """objectscores shape must be (nobs, ndim)."""
    model = Princals(ndim=2, itmax=200).fit(numeric_df)
    assert model.result_['objectscores'].shape == (10, 2)


def test_princals_ndim_3(numeric_df):
    """ndim=3 → (nobs, 3)."""
    model = Princals(ndim=3, itmax=200).fit(numeric_df)
    assert model.result_['objectscores'].shape == (10, 3)


def test_princals_transform_returns_scores(numeric_df):
    """transform(X) returns result_['objectscores']."""
    model = Princals(ndim=2, itmax=200).fit(numeric_df)
    assert np.allclose(
        model.transform(numeric_df),
        model.result_['objectscores'])


# ---------- Output matrix shapes ----------

def test_princals_transform_matrix_shape_copies1(numeric_df):
    """copies=1 (default) → transform is (nobs, nvars) matrix."""
    model = Princals(ndim=2, copies=1, itmax=200).fit(numeric_df)
    tr = model.result_['transform']
    assert hasattr(tr, 'shape'), "transform should be ndarray when copies=1"
    assert tr.shape == (10, 3), f"Got {tr.shape}"


def test_princals_transform_list_when_copies_multi(cat_df):
    """copies=2 → transform should be list."""
    model = Princals(
        ndim=2,
        copies=2,
        levels='nominal',
        degrees=-1,
        itmax=200).fit(cat_df)
    tr = model.result_['transform']
    assert isinstance(tr, list), f"Expected list for copies>1, got {type(tr)}"


def test_princals_weights_matrix_shape(numeric_df):
    """weights shape must be (nvars, ndim)."""
    model = Princals(ndim=2, itmax=200).fit(numeric_df)
    assert model.result_['weights'].shape == (
        3, 2), f"Got {model.result_['weights'].shape}"


def test_princals_loadings_matrix_shape(numeric_df):
    """loadings shape must be (nvars, ndim)."""
    model = Princals(ndim=2, itmax=200).fit(numeric_df)
    assert model.result_['loadings'].shape == (
        3, 2), f"Got {model.result_['loadings'].shape}"


def test_princals_quantifications_shape(numeric_df):
    """quantifications[j] must have ndim columns."""
    model = Princals(ndim=2, itmax=200).fit(numeric_df)
    for q in model.result_['quantifications']:
        assert q.shape[1] == 2


# ---------- Convergence ----------

def test_princals_convergence(numeric_df):
    model = Princals(ndim=2, itmax=1000, eps=1e-6).fit(numeric_df)
    assert model.n_iter_ < 1000, f"Did not converge: {model.n_iter_}"


def test_princals_loss_positive(numeric_df):
    model = Princals(ndim=2, itmax=200).fit(numeric_df)
    assert model.result_['f'] > 0


# ---------- Level handling ----------

def test_princals_nominal_level(cat_df):
    """levels='nominal', degrees=-1 → categorical indicator treatment."""
    model = Princals(
        ndim=2,
        levels='nominal',
        degrees=-1,
        itmax=200).fit(cat_df)
    assert model.result_['objectscores'].shape == (10, 2)


def test_princals_metric_level(numeric_df):
    """levels='metric' → empty knots, polynomial isotone."""
    model = Princals(ndim=2, levels='metric', itmax=200).fit(numeric_df)
    assert model.result_['objectscores'].shape == (10, 2)


# ---------- sklearn compatibility ----------

def test_princals_get_params():
    model = Princals(ndim=3, degrees=2, copies=1)
    params = model.get_params()
    assert params['ndim'] == 3
    assert params['degrees'] == 2
    assert params['copies'] == 1
    assert 'levels' in params


def test_princals_set_params():
    model = Princals(ndim=2)
    model.set_params(ndim=4)
    assert model.ndim == 4


def test_princals_sklearn_clone():
    model = Princals(ndim=2, degrees=1)
    cloned = clone(model)
    assert cloned.ndim == 2
    assert cloned.degrees == 1
    assert not hasattr(cloned, 'is_fitted_')


def test_princals_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        Princals().transform(pd.DataFrame({'a': [1., 2., 3.]}))


# ---------- optimizer='majorization' ----------

def test_princals_majorization_runs(numeric_df):
    """optimizer='majorization' should not raise and produce result_."""
    model = Princals(ndim=2, itmax=100, optimizer='majorization').fit(numeric_df)
    assert hasattr(model, 'result_')


def test_princals_majorization_objectscores_shape(numeric_df):
    """majorization objectscores must be (nobs, ndim)."""
    model = Princals(ndim=2, itmax=100, optimizer='majorization').fit(numeric_df)
    assert model.result_['objectscores'].shape == (10, 2)


def test_princals_majorization_orthogonality(numeric_df):
    """majorization objectscores (before sqrt-scaling) must be orthonormal."""
    model = Princals(ndim=2, itmax=100,
                     optimizer='majorization', normobj_z=False).fit(numeric_df)
    X = model.result_['objectscores']
    XTX = X.T @ X
    assert np.allclose(XTX, np.eye(2), atol=1e-5), f"X^T X not I:\n{XTX}"


def test_princals_als_default_unchanged(numeric_df):
    """Default optimizer='als' must give same result as before (backward compat)."""
    m1 = Princals(ndim=2, itmax=200).fit(numeric_df)
    m2 = Princals(ndim=2, itmax=200, optimizer='als').fit(numeric_df)
    assert np.allclose(m1.result_['objectscores'],
                       m2.result_['objectscores'], atol=1e-10)
