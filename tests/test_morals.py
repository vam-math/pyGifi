import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import clone

from pygifi import Morals


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    # 20 observations, 2 predictors
    X = rng.uniform(1, 5, (20, 2))
    # y is a function of X[:, 0] plus noise
    y = X[:, 0] + rng.normal(0, 0.5, 20)
    return X, y


def test_morals_fit_runs(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    assert model.is_fitted_


def test_morals_xhat_shape(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    assert model.result_['xhat'].shape == (20, 2)


def test_morals_yhat_shape(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    assert model.result_['yhat'].shape == (20,)


def test_morals_beta_shape(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    assert model.result_['beta'].shape == (2,)


def test_morals_ypred_shape(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    assert model.result_['ypred'].shape == (20,)


def test_morals_yres_shape(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    assert model.result_['yres'].shape == (20,)


def test_morals_smc_positive(sample_data):
    X, y = sample_data
    model = Morals(itmax=100).fit(X, y)
    assert model.result_['smc'] > 0


def test_morals_rhat_symmetric(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    rhat = model.result_['rhat']
    assert np.allclose(rhat, rhat.T)
    assert np.allclose(np.diag(rhat), 1.0)


def test_morals_objectscores_shape(sample_data):
    X, y = sample_data
    model = Morals(itmax=10).fit(X, y)
    assert model.result_['objectscores'].shape == (20, 1)


def test_morals_get_params():
    model = Morals()
    params = model.get_params()
    expected_keys = {
        'xknots', 'yknots', 'xdegrees', 'ydegrees',
        'xordinal', 'yordinal', 'xties', 'yties',
        'xmissing', 'ymissing', 'xactive', 'xcopies',
        'itmax', 'eps', 'verbose', 'init_x', 'r_seed', 'optimizer'
    }
    assert set(params.keys()) == expected_keys


def test_morals_transform_before_fit_raises(sample_data):
    X, _ = sample_data
    model = Morals()
    with pytest.raises(NotFittedError):
        model.transform(X)


def test_morals_sklearn_clone(sample_data):
    X, y = sample_data
    model = Morals(itmax=5).fit(X, y)
    cloned = clone(model)
    assert not hasattr(cloned, "is_fitted_")
    assert cloned.itmax == 5


# ---------- optimizer='majorization' ----------

def test_morals_majorization_runs(sample_data):
    """optimizer='majorization' should not raise."""
    X, y = sample_data
    model = Morals(itmax=50, optimizer='majorization').fit(X, y)
    assert model.is_fitted_


def test_morals_majorization_objectscores_shape(sample_data):
    """majorization objectscores must be (nobs, 1) for ndim=1."""
    X, y = sample_data
    model = Morals(itmax=50, optimizer='majorization').fit(X, y)
    assert model.result_['objectscores'].shape == (20, 1)


def test_morals_majorization_orthogonality(sample_data):
    """majorization output X must satisfy X^T X = I (1×1 = [[1]])."""
    X, y = sample_data
    model = Morals(itmax=50, optimizer='majorization').fit(X, y)
    X_out = model.result_['objectscores']
    XTX = X_out.T @ X_out
    assert np.allclose(XTX, np.eye(1), atol=1e-5), f"X^T X not I:\n{XTX}"


def test_morals_als_default_unchanged(sample_data):
    """Default optimizer='als' must produce same result as before (backward compat)."""
    X, y = sample_data
    m1 = Morals(itmax=50).fit(X, y)
    m2 = Morals(itmax=50, optimizer='als').fit(X, y)
    assert np.allclose(m1.result_['objectscores'],
                       m2.result_['objectscores'], atol=1e-10)
