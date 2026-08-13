import pytest
import numpy as np
import pandas as pd
from pygifi import categorical_encode, categorical_decode, encode, decode, make_numeric


def test_categorical_mapping():
    labels = ["a", "b", "c", "a", "b"]
    codes, mapping = categorical_encode(labels)
    assert np.allclose(codes, [1, 2, 3, 1, 2])
    assert mapping[1] == "a"
    assert mapping[2] == "b"
    assert mapping[3] == "c"

    decoded = categorical_decode(codes, mapping)
    assert list(decoded) == labels


def test_r_style_coding():
    # R: decode(c(1,1), c(2,2)) -> 1
    # R: decode(c(2,2), c(2,2)) -> 4
    assert decode([1, 1], [2, 2]) == 1
    assert decode([2, 1], [2, 2]) == 2
    assert decode([1, 2], [2, 2]) == 3
    assert decode([2, 2], [2, 2]) == 4

    # R: encode(1, c(2,2)) -> (1,1)
    # R: encode(4, c(2,2)) -> (2,2)
    assert np.array_equal(encode(1, [2, 2]), [1, 1])
    assert np.array_equal(encode(2, [2, 2]), [2, 1])
    assert np.array_equal(encode(3, [2, 2]), [1, 2])
    assert np.array_equal(encode(4, [2, 2]), [2, 2])


def test_make_numeric_robustness():
    df = pd.DataFrame({
        'A': [1.0, 2.0, 1.0],
        'B': pd.Categorical(['x', 'y', 'x']),
        'C': ['1', '2', '2']  # Numeric-valued strings
    })
    res = make_numeric(df)
    assert res.shape == (3, 3)
    # Col A: unchanged
    assert np.allclose(res[:, 0], [1, 2, 1])
    # Col B: categorical (x=1, y=2)
    assert np.allclose(res[:, 1], [1, 2, 1])
    # Col C: numeric-valued strings (1=1.0, 2=2.0)
    assert np.allclose(res[:, 2], [1, 2, 2])


def test_coding_errors():
    with pytest.raises(ValueError, match="No such cell"):
        decode([3, 1], [2, 2])
    with pytest.raises(ValueError, match="No such cell"):
        encode(5, [2, 2])
    with pytest.raises(ValueError, match="Dimension error"):
        decode([1], [2, 2])
