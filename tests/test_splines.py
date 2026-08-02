"""Tests for pygifi._splines — B-spline basis and knot generation."""

import numpy as np
import pytest
from pygifi.utils.splines import bspline_basis, knots_gifi
import pandas as pd


# ---------- bspline_basis ----------

def test_bspline_degree1_one_innerknot_shape():
    """degree=1, 1 innerknot → nf = 1+1+1 = 3 basis functions."""
    x = np.linspace(0., 1., 10)
    b = bspline_basis(x, degree=1, innerknots=np.array([0.5]))
    assert b.shape[1] == 3, f"Expected 3 columns, got {b.shape[1]}"
    assert b.shape[0] == 10


def test_bspline_partition_of_unity():
    """B-spline rows should sum to 1 (partition of unity property)."""
    x = np.linspace(0., 1., 20)
    b = bspline_basis(x, degree=2, innerknots=np.array([0.33, 0.67]))
    row_sums = b.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-8), f"Rows don't sum to 1: {row_sums}"


def test_bspline_nonnegative():
    """B-spline values must be non-negative."""
    x = np.linspace(0., 1., 30)
    b = bspline_basis(x, degree=3, innerknots=np.array([0.25, 0.5, 0.75]))
    assert np.all(b >= -1e-12), "Negative B-spline values"


def test_bspline_no_innerknots():
    """No inner knots: degree+1 basis functions."""
    x = np.linspace(0., 1., 10)
    b = bspline_basis(x, degree=2, innerknots=np.array([]))
    # Should have degree+1 = 3 basis functions
    assert b.shape[1] == 3, f"Expected 3, got {b.shape[1]}"


def test_bspline_right_endpoint():
    """Right endpoint x == highknot: last basis = 1, others = 0."""
    x = np.array([0., 0.5, 1.0])
    b = bspline_basis(x, degree=1, innerknots=np.array([0.5]))
    # Row at x=1.0 (right endpoint): last col = 1, others = 0
    assert b[-1, -
             1] == 1.0, f"Right endpoint last col should be 1, got {b[-1, :]}"
    assert np.allclose(b[-1, :-1], 0.0)


def test_bspline_degree0():
    """Degree 0: piecewise constant (indicator functions)."""
    x = np.array([0.1, 0.6])
    b = bspline_basis(x, degree=0, innerknots=np.array([0.5]))
    assert b.shape[1] == 2
    # Each row should be a unit vector
    assert np.allclose(b.sum(axis=1), 1.0)


# ---------- knots_gifi ----------

def test_knots_gifi_quantile_length():
    """'Q' type: should return n internal knots per column."""
    x = pd.DataFrame({'a': np.arange(1., 11.)})
    knots = knots_gifi(x, type='Q', n=3)
    assert len(knots) == 1
    # n+2 quantile points minus 2 endpoints = n internal knots
    assert len(knots[0]) == 3, f"Expected 3 knots, got {len(knots[0])}"


def test_knots_gifi_empty():
    """'E' type: should return empty list per column."""
    x = pd.DataFrame({'a': [1., 2., 3.], 'b': [4., 5., 6.]})
    knots = knots_gifi(x, type='E')
    assert len(knots) == 2
    for k in knots:
        assert len(k) == 0


def test_knots_gifi_regular():
    """'R' type: knots equally spaced (n interior)."""
    x = pd.DataFrame({'a': np.linspace(0., 10., 50)})
    knots = knots_gifi(x, type='R', n=4)
    assert len(knots[0]) == 4


def test_knots_gifi_invalid_type():
    """Invalid knot type should raise ValueError."""
    x = pd.DataFrame({'a': [1., 2., 3.]})
    with pytest.raises(ValueError, match="type must be 'Q', 'R', 'E', or 'D', got 'INVALID'"):
        knots_gifi(x, type='INVALID')


def test_knots_gifi_data_few_unique():
    """'D' type with too few unique values for knots."""
    x = pd.DataFrame({'a': [1., 1., 2.]})
    knots = knots_gifi(x, type='D')
    assert len(knots[0]) == 0  # No interior knots possible


def test_deboor_fallback(monkeypatch):
    """Force BSpline.design_matrix to fail so _deboor_basis executes."""
    x = np.linspace(0., 1., 10)
    degree = 2
    innerknots = np.array([0.5])

    # Run once normally to get the expected result
    expected_b = bspline_basis(x, degree=degree, innerknots=innerknots)

    # Patch BSpline.design_matrix in the pygifi._splines namespace
    def mock_design(*args, **kwargs):
        raise Exception('Forced SciPy failure')

    import pygifi.utils.splines
    monkeypatch.setattr(
        pygifi.utils.splines.BSpline,
        'design_matrix',
        mock_design)

    fallback_b = bspline_basis(x, degree=degree, innerknots=innerknots)

    # The fallback should compute the mathematically identical basis matrix
    assert np.allclose(fallback_b, expected_b, atol=1e-8)
