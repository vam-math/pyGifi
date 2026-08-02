"""Tests for pygifi.homals — Homals (Multiple Correspondence Analysis)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pygifi import Homals


# --- Sample data fixture ---

@pytest.fixture
def sample_df():
    """Small 3-variable, 10-obs categorical DataFrame."""
    return pd.DataFrame({
        'a': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
        'b': [2, 1, 3, 2, 1, 3, 1, 2, 3, 1],
        'c': [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
    })


# ---------- Basic fit / shape ----------

def test_homals_fit_runs(sample_df):
    """fit() should not raise an error."""
    model = Homals(ndim=2, itmax=200)
    model.fit(sample_df)


def test_homals_objectscores_shape(sample_df):
    """objectscores must be (nobs, ndim)."""
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    scores = model.result_['objectscores']
    assert scores.shape == (10, 2), f"Got {scores.shape}"


def test_homals_transform_equals_objectscores(sample_df):
    """transform(X) must return the same thing as result_['objectscores']."""
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    assert np.allclose(
        model.transform(sample_df),
        model.result_['objectscores'])


def test_homals_ndim_3(sample_df):
    """ndim=3 → objectscores shape (10, 3)."""
    model = Homals(ndim=3, itmax=200).fit(sample_df)
    assert model.result_['objectscores'].shape == (10, 3)


# ---------- Convergence ----------

def test_homals_convergence(sample_df):
    """n_iter_ should be < itmax for simple data."""
    model = Homals(ndim=2, itmax=1000, eps=1e-6).fit(sample_df)
    assert model.n_iter_ < 1000, f"Did not converge: n_iter_={model.n_iter_}"


def test_homals_loss_positive(sample_df):
    """Final loss f must be positive."""
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    assert model.result_['f'] > 0


# ---------- Result structure ----------

def test_homals_result_keys(sample_df):
    """result_ must contain all expected keys."""
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    required = {'objectscores', 'quantifications', 'transform', 'weights',
                'loadings', 'rhat', 'evals', 'lambda_', 'f', 'ntel'}
    assert required.issubset(model.result_.keys())


def test_homals_quantifications_shape(sample_df):
    """quantifications[j] must have shape (nbas, ndim)."""
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    for q in model.result_['quantifications'].values():
        assert q.shape[1] == 2


def test_homals_rhat_symmetric(sample_df):
    """rhat must be symmetric and have ones on diagonal."""
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    rhat = model.result_['rhat']
    assert np.allclose(rhat, rhat.T, atol=1e-8)
    assert np.allclose(np.diag(rhat), 1.0, atol=1e-8)


def test_homals_normobj_z_scales(sample_df):
    """normobj_z=True should scale objectscores by sqrt(nobs)."""
    nobs = len(sample_df)
    m_scaled = Homals(ndim=2, itmax=200, normobj_z=True).fit(sample_df)
    m_raw = Homals(ndim=2, itmax=200, normobj_z=False).fit(sample_df)
    s1 = m_scaled.result_['objectscores']
    s0 = m_raw.result_['objectscores']
    # Reset seed: both use same seed, so raw objectscores should be
    # proportional
    ratio = np.linalg.norm(s1) / np.linalg.norm(s0)
    assert np.isclose(
        ratio, np.sqrt(nobs), rtol=0.05
    ), f"ratio={ratio}, sqrt(nobs)={np.sqrt(nobs):.3f}"


# ---------- sklearn compatibility ----------

def test_homals_get_params():
    """get_params() must return a dict with all constructor params."""
    model = Homals(ndim=3, itmax=500)
    params = model.get_params()
    assert params['ndim'] == 3
    assert params['itmax'] == 500
    assert 'levels' in params
    assert 'eps' in params


def test_homals_set_params():
    """set_params() must update the instance."""
    model = Homals(ndim=2)
    model.set_params(ndim=4)
    assert model.ndim == 4


def test_homals_sklearn_clone():
    """sklearn.clone(Homals()) must return a fresh unfitted instance."""
    model = Homals(ndim=2, itmax=100)
    cloned = clone(model)
    assert cloned.ndim == 2
    assert not hasattr(cloned, 'is_fitted_')


def test_homals_transform_before_fit_raises():
    """transform() before fit() must raise NotFittedError."""
    from sklearn.exceptions import NotFittedError
    model = Homals()
    with pytest.raises(NotFittedError):
        model.transform(pd.DataFrame({'a': [1, 2]}))


# ---------- Attribute checks ----------

def test_homals_n_obs_set(sample_df):
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    assert model.n_obs_ == 10


def test_homals_n_vars_set(sample_df):
    model = Homals(ndim=2, itmax=200).fit(sample_df)
    assert model.n_vars_ == 3


def test_homals_numpy_input():
    """Should accept plain numpy array (not just DataFrame)."""
    # Use 3+ categories per variable so ndim=2 copies aren't capped
    data = np.array([[1, 3], [2, 1], [1, 2], [3, 1], [2, 3],
                     [1, 2], [3, 1], [2, 3], [1, 2], [3, 1]])
    model = Homals(ndim=2, itmax=100).fit(data)
    assert model.result_['objectscores'].shape == (10, 2)


# ---------- optimizer='majorization' ----------

def test_homals_majorization_runs(sample_df):
    """optimizer='majorization' should not raise."""
    model = Homals(ndim=2, itmax=100, optimizer='majorization').fit(sample_df)
    assert hasattr(model, 'result_')


def test_homals_majorization_objectscores_shape(sample_df):
    """majorization objectscores must be (nobs, ndim)."""
    model = Homals(ndim=2, itmax=100, optimizer='majorization').fit(sample_df)
    assert model.result_['objectscores'].shape == (10, 2)


def test_homals_majorization_orthogonality(sample_df):
    """majorization objectscores define orthogonal directions."""
    model = Homals(ndim=2, itmax=100,
                   optimizer='majorization', normobj_z=False).fit(sample_df)
    X = model.result_['objectscores']
    # Normalize to length 1 to check orthogonality of directions
    X_norm = X / np.linalg.norm(X, axis=0)

    XTX = X_norm.T @ X_norm
    assert np.allclose(XTX, np.eye(2), atol=1e-5), f"X^T X not I:\n{XTX}"


def test_homals_als_default_unchanged(sample_df):
    """Default optimizer='als' must give same result as before (backward compat)."""
    m1 = Homals(ndim=2, itmax=200).fit(sample_df)
    m2 = Homals(ndim=2, itmax=200, optimizer='als').fit(sample_df)
    assert np.allclose(m1.result_['objectscores'],
                       m2.result_['objectscores'], atol=1e-10)
