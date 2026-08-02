
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from pygifi.utils.utilities import make_sparse_indicator


def test_make_sparse_indicator_numpy_1d():
    data = np.array(['A', 'B', 'A', 'C'])
    res = make_sparse_indicator(data)

    assert len(res['matrices']) == 1
    assert len(res['mappings']) == 1

    mat = res['matrices'][0]
    mapping = res['mappings'][0]

    assert issparse(mat)
    assert mat.shape == (4, 3)
    assert mapping == {'A': 0, 'B': 1, 'C': 2}

    dense = mat.toarray()
    assert np.array_equal(dense, np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.]
    ]))


def test_make_sparse_indicator_numpy_2d():
    data = np.array([
        ['A', 1],
        ['B', 2],
        ['A', 2],
        ['C', 1]
    ])
    res = make_sparse_indicator(data)

    assert len(res['matrices']) == 2

    # First column ('A', 'B', 'C')
    mat0 = res['matrices'][0]
    map0 = res['mappings'][0]
    assert issparse(mat0)
    assert mat0.shape == (4, 3)
    assert map0 == {'A': 0, 'B': 1, 'C': 2}

    # Second column (1, 2) parsed as strings due to string dtype array
    mat1 = res['matrices'][1]
    map1 = res['mappings'][1]
    assert issparse(mat1)
    assert mat1.shape == (4, 2)
    assert map1 == {'1': 0, '2': 1}
    assert np.array_equal(mat1.toarray(), np.array([
        [1., 0.],
        [0., 1.],
        [0., 1.],
        [1., 0.]
    ]))


def test_make_sparse_indicator_pandas_dataframe():
    df = pd.DataFrame({
        'Color': ['Red', 'Blue', 'Red', 'Green'],
        'Size': ['S', 'M', 'L', 'S']
    })

    res = make_sparse_indicator(df)
    assert len(res['matrices']) == 2

    mat_c = res['matrices'][0]
    map_c = res['mappings'][0]
    assert issparse(mat_c)
    assert mat_c.shape == (4, 3)
    assert map_c == {'Blue': 0, 'Green': 1, 'Red': 2}

    assert np.array_equal(mat_c.toarray(), np.array([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.]
    ]))


def test_make_sparse_indicator_with_missing_pandas():
    df = pd.DataFrame({
        'Cat': ['A', np.nan, 'B', 'A']
    })

    res = make_sparse_indicator(df)
    mat = res['matrices'][0]
    mapping = res['mappings'][0]

    assert mapping == {'A': 0, 'B': 1}
    assert mat.shape == (4, 2)

    assert np.array_equal(mat.toarray(), np.array([
        [1., 0.],
        [0., 0.],
        [0., 1.],
        [1., 0.]
    ]))


def test_make_sparse_indicator_with_missing_numpy():
    data = np.array([1.0, np.nan, 2.0, 1.0])
    res = make_sparse_indicator(data)

    mat = res['matrices'][0]
    mapping = res['mappings'][0]

    assert mapping == {1.0: 0, 2.0: 1}
    assert mat.shape == (4, 2)
    assert np.array_equal(mat.toarray(), np.array([
        [1., 0.],
        [0., 0.],
        [0., 1.],
        [1., 0.]
    ]))
