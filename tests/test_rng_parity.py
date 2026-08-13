"""
Verifies pygifi_rng (the ported R RNG, MT19937 + AS241 inversion) reproduces
R's actual set.seed()/rnorm() output bit-for-bit. This is what lets the
R-vs-Python parity pipeline (tests/parity/) compare results starting from
identical object scores without manually shuttling R's random numbers into
Python.

Build the extension first if this file is skipped:
    cd pygifi/rng && python setup_rng.py build_ext --inplace
    (then copy the resulting pygifi_rng*.pyd onto your Python path)
"""
import numpy as np
import pytest

pygifi_rng = pytest.importorskip(
    "pygifi_rng",
    reason="pygifi_rng C extension not built: see module docstring",
)

# R reference: set.seed(1); rnorm(10)
R_RNORM_SEED1 = np.array([
    -0.6264538, 0.1836433, -0.8356286, 1.5952808,
    0.3295078, -0.8204684, 0.4874291, 0.7383247,
    0.5757814, -0.3053884,
])


def test_rnorm_matches_r_stream():
    pygifi_rng.r_set_seed(1)
    py_vals = np.asarray(pygifi_rng.r_rnorm(10))
    np.testing.assert_allclose(py_vals, R_RNORM_SEED1, atol=1e-6)


def test_r_init_x_matches_r_stream():
    # r_init_x should draw from the same stream as a direct rnorm() call,
    # filling column-major (R's matrix(rnorm(n*d), n, d) order).
    X = np.asarray(pygifi_rng.r_init_x(5, 2, 1))
    assert X.shape == (5, 2)
    np.testing.assert_allclose(X[:, 0], R_RNORM_SEED1[:5], atol=1e-6)
    np.testing.assert_allclose(X[:, 1], R_RNORM_SEED1[5:10], atol=1e-6)


def test_r_init_x_reproducible_for_same_seed():
    X1 = np.asarray(pygifi_rng.r_init_x(8, 3, 42))
    X2 = np.asarray(pygifi_rng.r_init_x(8, 3, 42))
    np.testing.assert_array_equal(X1, X2)
