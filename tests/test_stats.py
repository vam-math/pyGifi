import pytest
import numpy as np
import pandas as pd
from pygifi import Princals, Homals, Morals


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        'A': np.random.randint(1, 4, 20),
        'B': np.random.randint(1, 5, 20),
        'C': np.random.randint(1, 3, 20)
    })


def test_princals_statistical_properties(sample_df):
    model = Princals(ndim=2).fit(sample_df)

    assert model.eigenvalues_ is not None
    assert model.eigenvalues_.shape[0] >= 2

    assert model.variance_explained_ is not None
    assert model.variance_explained_.shape[0] >= 2
    assert np.all(model.variance_explained_ >= 0)

    assert model.component_loadings_ is not None
    assert model.component_loadings_.shape == (3, 2)

    assert model.category_quantifications_ is not None
    assert len(model.category_quantifications_) == 3  # one for each var

    assert model.object_scores_ is not None
    assert model.object_scores_.shape == (20, 2)


def test_homals_statistical_properties(sample_df):
    model = Homals(ndim=2).fit(sample_df)

    assert model.eigenvalues_ is not None

    assert model.variance_explained_ is not None
    assert np.all(model.variance_explained_ >= 0)

    assert model.component_loadings_ is not None
    assert len(model.component_loadings_) == 3  # Homals uses a dict

    assert model.category_quantifications_ is not None
    assert len(model.category_quantifications_) == 3

    assert model.object_scores_ is not None
    assert model.object_scores_.shape == (20, 2)


def test_morals_statistical_properties(sample_df):
    X = sample_df[['A', 'B']]
    y = sample_df['C']

    model = Morals().fit(X, y)

    assert model.eigenvalues_ is not None
    assert model.variance_explained_ is not None
    assert model.component_loadings_ is not None
    assert model.object_scores_ is not None
    assert model.category_quantifications_ is None  # explicitly None for Morals
