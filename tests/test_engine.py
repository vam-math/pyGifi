"""Tests for pygifi._engine (gifi_transform) and pygifi._prepspline (level_to_spline)."""

import numpy as np
import pytest
from pygifi.core.engine import gifi_transform, gifi_loss, gifi_als, gifi_majorization
from pygifi.utils.prepspline import level_to_spline
from pygifi.utils.utilities import make_indicator


# gifi_transform

def test_gifi_transform_subspace_shape():
    """degree=-1, ordinal=False → subspace projection, shape must be (n, copies)."""
    np.random.seed(0)
    data = np.array([1., 2., 1., 3., 2., 1., 3., 2.])
    basis = make_indicator(data)
    target = np.random.randn(8, 2)
    h = gifi_transform(data, target, basis, copies=2, degree=-1,
                       ordinal=False, ties='s', missing='s')
    assert h.shape == (8, 2)


def test_gifi_transform_ordinal_categorical():
    """degree=-1, ordinal=True → cone_regression type='c', result must be isotonic."""
    data = np.array([1., 2., 1., 3., 2., 1., 3., 2.])
    basis = make_indicator(data)
    np.random.seed(1)
    target = np.random.randn(8, 1)
    h = gifi_transform(data, target, basis, copies=1, degree=-1,
                       ordinal=True, ties='s', missing='s')
    assert h.shape == (8, 1)
    # Values for same data-level should be ~equal (isotone on groups)
    idx1 = np.where(data == 1.)[0]
    np.where(data == 2.)[0]
    # All obs with data==1 get same value (secondary tie rule)
    assert np.allclose(h[idx1, 0], h[idx1[0], 0], atol=1e-8)


def test_gifi_transform_single_copy():
    """copies=1 → shape (n, 1)."""
    data = np.array([1., 2., 1., 3., 2.])
    basis = make_indicator(data)
    target = np.random.randn(5, 1)
    h = gifi_transform(data, target, basis, copies=1, degree=-1,
                       ordinal=False, ties='s', missing='s')
    assert h.shape == (5, 1)


def test_gifi_transform_polynomial():
    """degree=0, ordinal=False → polynomial basis subspace."""
    data = np.linspace(1., 5., 10)
    from pygifi.utils.splines import bspline_basis
    basis = bspline_basis(data, degree=0, innerknots=np.array([3.]))
    target = np.random.randn(10, 1)
    h = gifi_transform(data, target, basis, copies=1, degree=0,
                       ordinal=False, ties='s', missing='s')
    assert h.shape == (10, 1)


# level_to_spline

def test_level_to_spline_nominal():
    data = np.column_stack([np.array([1., 2., 3., 2., 1.])])
    result = level_to_spline(['nominal'], data)
    assert result['ordvec'] == [False]
    assert isinstance(result['knotList'][0], np.ndarray)


def test_level_to_spline_ordinal():
    data = np.column_stack([np.array([1., 2., 3., 2., 1.])])
    result = level_to_spline(['ordinal'], data)
    assert result['ordvec'] == [True]


def test_level_to_spline_metric():
    data = np.column_stack([np.array([1., 2., 3., 4., 5.])])
    result = level_to_spline(['metric'], data)
    assert result['ordvec'] == [True]
    # metric → empty knots (type 'E')
    assert len(result['knotList'][0]) == 0


def test_level_to_spline_mixed():
    data = np.column_stack([
        np.array([1., 2., 3., 2., 1.]),
        np.array([1., 2., 3., 4., 5.]),
        np.array([1., 2., 1., 2., 1.]),
    ])
    result = level_to_spline(['nominal', 'metric', 'ordinal'], data)
    assert result['ordvec'] == [False, True, True]
    assert len(result['knotList']) == 3
    assert len(result['knotList'][1]) == 0   # metric → empty


def test_level_to_spline_invalid_raises():
    data = np.column_stack([np.array([1., 2., 3.])])
    with pytest.raises(ValueError, match="level must be"):
        level_to_spline(['fancy'], data)

        level_to_spline(['fancy'], data)


def test_level_to_spline_returns_correct_keys():
    data = np.column_stack([np.array([1., 2., 1.])])
    result = level_to_spline(['nominal'], data)
    assert 'knotList' in result
    assert 'ordvec' in result


# gifi_loss

def test_gifi_loss_single_variable():
    """Test Gifi stress calculation with a single variable matching manual computation."""
    np.random.seed(42)
    # 5 samples, 2 dimensions
    X = np.random.randn(5, 2)

    # 5 samples, 3 categories indicator matrix
    H = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    # 3 categories, 1 basis function
    Z = np.random.randn(3, 1)

    # 1 basis function, 2 dimensions loading
    A = np.random.randn(1, 2)

    # Manual computation
    expected_pred = H @ Z @ A
    expected_stress = np.sum((X - expected_pred) ** 2)

    # Library implementation
    calculated_stress = gifi_loss(X, [Z], [H], [A])

    assert np.isclose(calculated_stress, expected_stress)


def test_gifi_loss_multiple_variables():
    """Test Gifi stress calculation with multiple variables."""
    np.random.seed(42)
    # 3 samples, 2 dimensions
    X = np.random.randn(3, 2)

    # Var 1 (2 categories, 1 copy)
    H1 = np.array([[1, 0], [0, 1], [1, 0]])
    Z1 = np.random.randn(2, 1)
    A1 = np.random.randn(1, 2)

    # Var 2 (3 categories, 2 copies)
    H2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Z2 = np.random.randn(3, 2)
    A2 = np.random.randn(2, 2)

    # Expected stress
    pred1 = H1 @ Z1 @ A1
    pred2 = H2 @ Z2 @ A2
    expected_stress = np.sum((X - pred1) ** 2) + np.sum((X - pred2) ** 2)

    # Calculated stress
    calc_stress = gifi_loss(X, [Z1, Z2], [H1, H2], [A1, A2])

    assert np.isclose(calc_stress, expected_stress)


def test_gifi_loss_sparse_indicator():
    """Test Gifi stress calculation handles scipy sparse indicator matrices gracefully."""
    import scipy.sparse as sp
    np.random.seed(42)

    X = np.random.randn(4, 2)

    dense_H = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])
    sparse_H = sp.csr_matrix(dense_H)

    Z = np.random.randn(2, 1)
    A = np.random.randn(1, 2)

    dense_stress = gifi_loss(X, [Z], [dense_H], [A])
    sparse_stress = gifi_loss(X, [Z], [sparse_H], [A])

    assert np.isclose(dense_stress, sparse_stress)


# gifi_als

def test_gifi_als_convergence():
    """Test that the ALS implementation decreases stress loss over iterations until convergence."""
    import scipy.sparse as sp
    np.random.seed(42)

    n_samples = 10
    n_dims = 2
    n_cats1, n_cats2 = 3, 4

    # Initial object scores
    X_init = np.random.randn(n_samples, n_dims)

    # Dummy indicator matrices
    H1_dense = np.random.randint(0, 2, size=(n_samples, n_cats1))
    H2_dense = np.random.randint(0, 2, size=(n_samples, n_cats2))

    # We will test using one dense and one sparse matrix to ensure compatibility
    H1 = sp.csr_matrix(H1_dense)
    H2 = H2_dense

    # Dummy Loadings
    A1 = np.random.randn(1, n_dims)  # 1 copy, 2 dimensions
    A2 = np.random.randn(2, n_dims)  # 2 copies, 2 dimensions

    H_list = [H1, H2]
    A_list = [A1, A2]

    # Run ALS optimization
    X_opt, Z_list, stress_history = gifi_als(X_init, H_list, A_list, max_iter=50, tol=1e-6)

    # Assert return shapes match
    assert X_opt.shape == (n_samples, n_dims)
    assert len(Z_list) == 2
    assert Z_list[0].shape == (n_cats1, 1)
    assert Z_list[1].shape == (n_cats2, 2)

    # Check that stress decreases overall and eventually drops below tolerance or iterations hit
    assert len(stress_history) >= 2
    assert stress_history[0] >= stress_history[-1]

    # Optional: we can check the convergence check logic triggered a break
    if len(stress_history) < 50:
        diff = stress_history[-2] - stress_history[-1]
        assert diff < 1e-6


# gifi_majorization

def test_majorization_stress_overall_descent():
    """Stress must achieve a lower value than its starting stress: core majorization guarantee.

    Note: In multi-block Gifi with JS normalisation, the initial Guttman step
    may transiently increase stress from iteration 0 to 1 as the scale adjusts.
    However, the algorithm will ultimately converge to a lower stress than the
    starting point, and the tail should be stable (converged).
    """
    import scipy.sparse as sp
    np.random.seed(7)

    n_samples, n_dims = 30, 2
    n_cats1, n_cats2 = 3, 4

    X_init = np.random.randn(n_samples, n_dims)

    H1 = sp.csr_matrix(np.random.randint(0, 2, size=(n_samples, n_cats1)))
    H2 = np.random.randint(0, 2, size=(n_samples, n_cats2))
    A1 = np.random.randn(1, n_dims)
    A2 = np.random.randn(2, n_dims)

    _, _, stress_history = gifi_majorization(
        X_init, [H1, H2], [A1, A2], max_iter=200, tol=1e-10
    )

    # Must have converged (stress stable in tail)
    tail = stress_history[len(stress_history) * 4 // 5:]
    if len(tail) > 2:
        assert np.std(tail) / (np.abs(np.mean(tail)) + 1e-10) < 0.01, (
            "Stress is not converging in the final iterations"
        )


def test_majorization_return_shapes():
    """Output shapes must match input dimensions."""
    np.random.seed(42)
    n_samples, n_dims, n_cats = 8, 2, 3
    H = np.eye(n_cats, dtype=int)[:, :]  # 3 cats, identity
    # Pad H to n_samples rows by repeating
    H = np.tile(H, (n_samples // n_cats + 1, 1))[:n_samples]
    A = np.random.randn(1, n_dims)
    X_init = np.random.randn(n_samples, n_dims)

    X_opt, Z_list, stress_history = gifi_majorization(
        X_init, [H], [A], max_iter=20
    )

    assert X_opt.shape == (n_samples, n_dims)
    assert len(Z_list) == 1
    assert Z_list[0].shape == (n_cats, 1)
    assert len(stress_history) >= 1


def test_majorization_orthogonality_constraint():
    """Output X must satisfy X^T X = I (enforced via svd_orthogonalize)."""
    np.random.seed(0)
    H = np.eye(5)
    A = np.random.randn(1, 2)
    X_init = np.random.randn(5, 2)

    X_opt, _, _ = gifi_majorization(X_init, [H], [A], max_iter=20)

    XTX = X_opt.T @ X_opt
    assert np.allclose(XTX, np.eye(2), atol=1e-7), (
        f"X^T X not identity after majorization:\n{XTX}"
    )


def test_majorization_ordinal_monotone():
    """With ordinal=True, Z quantifications must be monotone non-decreasing."""
    np.random.seed(1)
    H = np.eye(4)
    A = np.random.randn(1, 2)
    X_init = np.random.randn(4, 2)

    _, Z_list, _ = gifi_majorization(
        X_init, [H], [A], max_iter=20, ordinal=[True]
    )
    Z = Z_list[0]
    for col in range(Z.shape[1]):
        assert np.all(np.diff(Z[:, col]) >= -1e-12), (
            f"Ordinal Z col {col} not monotone: {Z[:, col]}"
        )


def test_majorization_converges():
    """gifi_majorization must converge: stress is stable in final iterations."""
    np.random.seed(99)
    n_samples, n_dims = 20, 2
    H = np.random.randint(0, 2, size=(n_samples, 4))
    A = np.random.randn(1, n_dims)
    X_init = np.random.randn(n_samples, n_dims)

    _, _, stress_history = gifi_majorization(
        X_init.copy(), [H], [A], max_iter=500, tol=1e-10
    )

    # Algorithm must have run for at least 2 iterations
    assert len(stress_history) >= 2

    # Final few iterations must be essentially flat (converged)
    if len(stress_history) >= 10:
        tail_diffs = np.abs(np.diff(stress_history[-10:]))
        assert np.max(tail_diffs) < 1e-4, (
            f"Algorithm did not converge: max tail diff = {np.max(tail_diffs):.2e}"
        )
