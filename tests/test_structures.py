"""Tests for pygifi._structures — GifiVariable and XGifiVariable factory functions."""

import numpy as np
import pytest
from pygifi.core.structures import (
    make_gifi_variable, make_gifi,
    make_x_gifi_variable, make_x_gifi
)
from pygifi.utils.utilities import center
from pygifi.core.linalg import gs_rc


# ---------- make_gifi_variable ----------

def test_gifi_variable_categorical():
    data = np.array([1., 2., 1., 3., 2.])
    gv = make_gifi_variable(
        data,
        knots=[],
        degree=-1,
        ordinal=False,
        ties='s',
        copies=2,
        missing='s',
        active=True,
        name='x')
    assert gv['type'] == 'categorical'
    assert gv['basis'].shape[0] == 5
    assert gv['basis'].shape[1] == 3   # 3 unique values
    assert gv['qr']['rank'] > 0


def test_gifi_variable_binary():
    data = np.array([1., 2., 1., 2., 1.])
    gv = make_gifi_variable(
        data,
        knots=[],
        degree=-1,
        ordinal=False,
        ties='s',
        copies=2,
        missing='s',
        active=True,
        name='y')
    assert gv['type'] == 'binary'
    assert gv['basis'].shape[1] == 2


def test_gifi_variable_copies_capped():
    """copies should be capped at basis.ncol - 1."""
    data = np.array([1., 2., 1., 2., 1.])   # binary → 2 basis cols
    gv = make_gifi_variable(
        data,
        knots=[],
        degree=-1,
        ordinal=False,
        ties='s',
        copies=10,
        missing='s',
        active=True,
        name='y')
    # 2 basis cols → copies capped at 1
    assert gv['copies'] == 1


def test_gifi_variable_polynomial():
    data = np.linspace(1., 5., 10)
    gv = make_gifi_variable(
        data,
        knots=[],
        degree=2,
        ordinal=False,
        ties='s',
        copies=2,
        missing='s',
        active=True,
        name='z')
    assert gv['type'] == 'polynomial'
    assert gv['basis'].shape[0] == 10


def test_gifi_variable_splinical():
    data = np.linspace(1., 5., 10)
    gv = make_gifi_variable(
        data,
        knots=[3.0],
        degree=2,
        ordinal=False,
        ties='s',
        copies=2,
        missing='s',
        active=True,
        name='z')
    assert gv['type'] == 'splinical'


def test_gifi_variable_qr_orthonormal():
    """QR from gs_rc should give Q.T @ Q = I."""
    data = np.array([1., 2., 1., 3., 2., 1., 3., 2.])
    gv = make_gifi_variable(
        data,
        knots=[],
        degree=-1,
        ordinal=False,
        ties='s',
        copies=2,
        missing='s',
        active=True,
        name='x')
    q = gv['qr']['q']
    assert np.allclose(q.T @ q, np.eye(gv['qr']['rank']), atol=1e-10)


def test_gifi_variable_completely_missing_raises():
    data = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="completely missing"):
        make_gifi_variable(
            data,
            knots=[],
            degree=-1,
            ordinal=False,
            ties='s',
            copies=2,
            missing='s',
            active=True,
            name='bad')


def test_gifi_variable_single_category_raises():
    # all same → 1 unique → indicator has 1 col
    data = np.array([1., 1., 1., 1.])
    with pytest.raises(ValueError, match="more than one unique category/level"):
        make_gifi_variable(
            data,
            knots=[],
            degree=-1,
            ordinal=False,
            ties='s',
            copies=2,
            missing='s',
            active=True,
            name='bad')


# ---------- make_gifi ----------

def test_make_gifi_structure():
    data = np.column_stack([
        np.array([1., 2., 1., 3., 2.]),
        np.array([2., 1., 2., 1., 2.]),
        np.array([3., 3., 1., 2., 1.]),
    ])
    knots = [[], [], []]
    degrees = [-1, -1, -1]
    ordinal = [False, False, False]
    ties = ['s', 's', 's']
    copies = [2, 2, 2]
    missing = ['s', 's', 's']
    active = [True, True, True]
    names = ['a', 'b', 'c']
    sets = [0, 1, 2]

    gifi = make_gifi(
        data,
        knots,
        degrees,
        ordinal,
        ties,
        copies,
        missing,
        active,
        names,
        sets)
    assert len(gifi) == 3   # 3 sets
    assert len(gifi[0]) == 1  # 1 variable per set


def test_make_gifi_single_set():
    """All variables in the same set."""
    data = np.column_stack([
        np.array([1., 2., 1., 2.]),
        np.array([2., 1., 2., 1.]),
    ])
    gifi = make_gifi(data,
                     knots=[[], []],
                     degrees=[-1, -1],
                     ordinal=[False, False],
                     ties=['s', 's'],
                     copies=[2, 2],
                     missing=['s', 's'],
                     active=[True, True],
                     names=['a', 'b'],
                     sets=[0, 0])   # both in set 0
    assert len(gifi) == 1
    assert len(gifi[0]) == 2


# ---------- make_x_gifi_variable ----------

def test_x_gifi_variable_transform_shape():
    np.random.seed(123)
    data = np.array([1., 2., 1., 3., 2., 1., 3., 2.])
    gv = make_gifi_variable(
        data,
        knots=[],
        degree=-1,
        ordinal=False,
        ties='s',
        copies=2,
        missing='s',
        active=True,
        name='x')
    nobs, ndim = 8, 2
    x = np.random.randn(nobs, ndim)
    x = gs_rc(center(x))['q']
    xgv = make_x_gifi_variable(gv, x)

    assert xgv['transform'].ndim == 2
    assert xgv['transform'].shape[0] == nobs
    assert xgv['weights'].shape[1] == ndim
    assert xgv['scores'].shape == (nobs, ndim)


def test_x_gifi_variable_transform_orthonormal():
    np.random.seed(123)
    data = np.array([1., 2., 1., 3., 2., 1., 3., 2.])
    gv = make_gifi_variable(
        data,
        knots=[],
        degree=-1,
        ordinal=False,
        ties='s',
        copies=2,
        missing='s',
        active=True,
        name='x')
    x = gs_rc(center(np.random.randn(8, 2)))['q']
    xgv = make_x_gifi_variable(gv, x)
    T = xgv['transform']
    assert np.allclose(T.T @ T, np.eye(T.shape[1]), atol=1e-10)


# ---------- make_x_gifi ----------

def test_make_x_gifi_structure():
    np.random.seed(123)
    data = np.column_stack([
        np.array([1., 2., 1., 3., 2.]),
        np.array([2., 1., 2., 1., 2.]),
    ])
    gifi = make_gifi(data, [[], []], [-1, -1], [False, False],
                     ['s', 's'], [2, 2], ['s', 's'], [True, True],
                     ['a', 'b'], [0, 1])
    x = gs_rc(center(np.random.randn(5, 2)))['q']
    xgifi = make_x_gifi(gifi, x)

    assert len(xgifi) == 2       # 2 sets
    assert len(xgifi[0]) == 1    # 1 variable per set
    assert 'transform' in xgifi[0][0]
    assert 'quantifications' in xgifi[0][0]
