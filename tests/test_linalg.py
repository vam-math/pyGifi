"""Tests for pygifi._linalg — Modified Gram-Schmidt orthogonalization."""

import numpy as np
from pygifi.core.linalg import gs_rc, ls_rc, null_rc, ginv_rc
from pygifi.utils.utilities import center


# ---------- gs_rc ----------

def test_gs_rc_orthonormal_full_rank():
    """Q.T @ Q should be identity for a full-rank matrix."""
    np.random.seed(42)
    x = np.random.randn(10, 4)
    h = gs_rc(center(x))
    assert h['rank'] == 4
    identity = h['q'].T @ h['q']
    assert np.allclose(identity, np.eye(h['rank']), atol=1e-10)


def test_gs_rc_rank_deficient():
    """Rank-deficient matrix should return rank < n_cols."""
    x = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],   # col 3 = col 1 + col 2 + something → rank 2
        [1., 0., 1.],
        [0., 1., 1.],
    ])
    # Make the third column a linear combination of the first two
    x[:, 2] = x[:, 0] + x[:, 1]
    h = gs_rc(x)
    assert h['rank'] == 2, f"Expected rank 2, got {h['rank']}"


def test_gs_rc_returns_expected_keys():
    x = np.random.randn(5, 3)
    h = gs_rc(x)
    assert set(h.keys()) == {'q', 'r', 'rank', 'pivot'}


def test_gs_rc_q_shape():
    x = np.random.randn(8, 3)
    h = gs_rc(x)
    assert h['q'].shape[0] == 8
    assert h['q'].shape[1] == h['rank']


def test_gs_rc_single_column():
    """Single column should return rank 1."""
    x = np.array([[1.], [2.], [3.]])
    h = gs_rc(center(x))
    assert h['rank'] == 1
    assert np.isclose(np.linalg.norm(h['q'][:, 0]), 1.0)


# ---------- ls_rc ----------

def test_ls_rc_matches_lstsq_full_rank():
    """ls_rc solution should match np.linalg.lstsq on full-rank system."""
    np.random.seed(7)
    x = np.random.randn(20, 4)
    y = np.random.randn(20)
    sol_ours = ls_rc(x, y)['solution'].flatten()
    sol_ref, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    assert np.allclose(sol_ours, sol_ref, atol=1e-8), \
        f"ls_rc solution differs from lstsq:\nours:  {sol_ours}\nlstsq: {sol_ref}"


def test_ls_rc_residuals_shape_1d():
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    result = ls_rc(x, y)
    assert result['residuals'].shape[0] == 10


def test_ls_rc_residuals_shape_2d():
    x = np.random.randn(10, 3)
    y = np.random.randn(10, 2)
    result = ls_rc(x, y)
    assert result['solution'].shape == (3, 2)


def test_ls_rc_minssq_nonnegative():
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    assert ls_rc(x, y)['minssq'] >= 0.0


def test_ls_rc_rank_deficient():
    """ls_rc should gracefully handle rank deficient matrices without crashing."""
    np.random.seed(8)
    x = np.random.randn(10, 3)
    x[:, 2] = x[:, 0] * 2  # make it explicitly rank 2
    y = np.random.randn(10)
    result = ls_rc(x, y)
    assert result['solution'].shape == (
        3, 1) or result['solution'].shape == (3,)
    assert np.isfinite(result['solution']).all()

# ---------- null_rc ----------


def test_null_rc_perpendicular_to_x():
    """Null space vectors should be approximately orthogonal to x.T."""
    np.random.seed(99)
    x = np.random.randn(10, 5)
    x[:, 4] = x[:, 0] + x[:, 1]  # make rank-deficient
    ns = null_rc(x)
    # x @ ns should be ~0 at each column
    residual = np.abs(x @ ns)
    assert np.allclose(
        residual, 0, atol=1e-8), f"Null space not perpendicular: {residual}"


def test_null_rc_full_rank_returns_zeros():
    x = np.random.randn(10, 3)
    ns = null_rc(x)
    assert ns.shape == (3, 1)
    assert np.allclose(ns, 0)


# ---------- ginv_rc ----------

def test_ginv_rc_left_inverse():
    """For full-column-rank matrix: ginv(x) @ x ≈ I."""
    np.random.seed(5)
    x = np.random.randn(8, 3)  # overdetermined, full column rank
    g = ginv_rc(x)
    product = g @ x
    assert np.allclose(product, np.eye(3), atol=1e-8)
