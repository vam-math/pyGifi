"""Tests for pygifi._utilities: core utility primitives."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from pygifi.utils.coding import make_numeric
from pygifi.utils.utilities import (
    center, normalize, make_indicator,
    make_missing, reshape, direct_sum, cor_list
)


# center

def test_center_zeros_mean():
    x = np.array([[1., 2.], [3., 4.], [5., 6.]])
    cx = center(x)
    assert np.allclose(
        cx.mean(
            axis=0), 0.0), "Column means should be zero after centering"


def test_center_1d():
    x = np.array([1., 2., 3., 4., 5.])
    cx = center(x)
    assert np.isclose(cx.mean(), 0.0)


def test_center_preserves_shape():
    x = np.random.randn(10, 4)
    assert center(x).shape == x.shape


# normalize

def test_normalize_unit_norm_2d():
    x = np.array([[1., 2.], [3., 4.], [5., 6.]])
    nx = normalize(x)
    norms = np.linalg.norm(nx, axis=0)
    assert np.allclose(norms, 1.0), f"Column norms should be 1.0, got {norms}"


def test_normalize_1d():
    x = np.array([3., 4.])
    nx = normalize(x)
    assert np.isclose(np.linalg.norm(nx), 1.0)


# make_indicator

def test_make_indicator_shape():
    x = np.array([1, 2, 1, 3])
    ind = make_indicator(x)
    assert ind.shape == (4, 3), f"Expected (4,3), got {ind.shape}"


def test_make_indicator_values():
    x = np.array([1, 2, 1, 3])
    ind = make_indicator(x)
    # Row 0 (value=1) → column 0 is 1
    assert ind[0, 0] == 1.0 and ind[0, 1] == 0.0 and ind[0, 2] == 0.0
    # Row 1 (value=2) → column 1 is 1
    assert ind[1, 1] == 1.0
    # Row 3 (value=3) → column 2 is 1
    assert ind[3, 2] == 1.0


def test_make_indicator_binary():
    x = np.array([0, 1, 0, 1])
    ind = make_indicator(x)
    assert ind.shape == (4, 2)
    assert np.all((ind == 0) | (ind == 1))  # only 0/1 values


# make_missing

def test_make_missing_mode_m():
    data = np.array([1., 2., np.nan, 4., np.nan])
    basis = np.eye(3)  # 3 non-missing obs
    result = make_missing(data, basis, 'm')
    # shape should be (5, 3+2) = (5, 5)
    assert result.shape == (5, 5), f"Expected (5,5), got {result.shape}"
    # NaN rows should have unit vectors in the extra columns
    assert result[2, 3] == 1.0
    assert result[4, 4] == 1.0


def test_make_missing_mode_a():
    data = np.array([1., np.nan, 3.])
    basis = np.array([[1., 0.], [0., 1.]])  # 2 non-missing
    result = make_missing(data, basis, 'a')
    assert result.shape == (3, 2)
    # Missing row should be 1/2
    assert np.allclose(result[1, :], [0.5, 0.5])


def test_make_missing_mode_s():
    data = np.array([1., np.nan, 3.])
    basis = np.array([[1., 0.], [0., 1.]])  # 2 non-missing
    result = make_missing(data, basis, 's')
    assert result.shape == (3, 3)  # k+1 columns
    # Missing row: extra column = 1
    assert result[1, 2] == 1.0 and result[1, 0] == 0.0


# reshape

def test_reshape_scalar():
    result = reshape(5, 4)
    assert result == [5, 5, 5, 5]


def test_reshape_list_length_n():
    result = reshape([1, 2, 3], 3)
    assert result == [1, 2, 3]


def test_reshape_single_element_list():
    result = reshape([7], 3)
    assert result == [7, 7, 7]


def test_reshape_wrong_length_raises():
    with pytest.raises(ValueError):
        reshape([1, 2], 3)


# direct_sum

def test_direct_sum_shape():
    A = np.eye(2)
    B = np.eye(3)
    result = direct_sum([A, B])
    assert result.shape == (5, 5)


def test_direct_sum_block_structure():
    A = np.ones((2, 2))
    B = np.full((3, 3), 2.0)
    result = direct_sum([A, B])
    expected = block_diag(A, B)
    assert np.allclose(result, expected)


# cor_list

def test_cor_list_shape():
    matrices = [np.random.randn(10, 2), np.random.randn(10, 3)]
    result = cor_list(matrices)
    assert result.shape == (5, 5)


def test_cor_list_diagonal_ones():
    matrices = [np.random.randn(20, 2), np.random.randn(20, 2)]
    result = cor_list(matrices)
    # Diagonal must be 1.0 (self-correlation)
    assert np.allclose(np.diag(result), 1.0)


# make_numeric

def test_make_numeric_passthrough():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    result = make_numeric(df)
    assert result.shape == (3, 2)
    assert np.allclose(result[:, 0], [1, 2, 3])


def test_make_numeric_categorical():
    df = pd.DataFrame({'x': pd.Categorical(['a', 'b', 'a', 'c'])})
    result = make_numeric(df)
    # Should be integer codes (1-indexed)
    assert result.shape == (4, 1)
    assert result[0, 0] == result[2, 0]  # same category → same code
    assert result[0, 0] != result[1, 0]  # different category → different code


def test_make_numeric_numeric_factors():
    df = pd.DataFrame({'x': pd.Categorical(['1.0', '2.0', '3.0', '1.0'])})
    result = make_numeric(df)
    # Numeric-valued string factors should convert to floats
    assert result[0, 0] == result[3, 0]


# svd_orthogonalize

def test_svd_orthogonalize_identity_constraint():
    """Verify that the SVD orthogonalization returns an X satisfying X^T X = I."""
    from pygifi.utils.utilities import svd_orthogonalize
    np.random.seed(42)

    # 50 samples, 4 dimensions
    X_init = np.random.randn(50, 4)

    # Apply orthogonalization
    X_ortho = svd_orthogonalize(X_init)

    # Check shape is preserved
    assert X_ortho.shape == (50, 4)

    # Compute X^T X
    xtx = X_ortho.T @ X_ortho

    # Assert X^T X = I (Identity matrix 4x4)
    expected_identity = np.eye(4)
    assert np.allclose(xtx, expected_identity, atol=1e-7)
