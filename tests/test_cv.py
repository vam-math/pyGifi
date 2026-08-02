import pytest
import numpy as np
import pandas as pd
from pygifi import Morals
from pygifi.core.cv import cv_morals


def test_cv_morals_basic():
    # Create simple dataset
    # Create simple dataset using integers/categories (Gifi breaks on 100
    # purely distinct continuous floats)
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randint(
            1, 5, size=(
                100, 3)), columns=[
            'X1', 'X2', 'X3'])
    y = pd.Series(np.random.randint(1, 4, size=100))

    # Fit base Morals model
    print("Initializing Morals model...")
    model = Morals(itmax=2, xdegrees=1, ydegrees=1)
    print("Fitting Morals base model...")
    model.fit(X, y)
    print("Base model fitted.")

    # Run CV
    res = cv_morals(model, k=3, random_state=42)

    assert hasattr(res, 'cv_error')
    assert isinstance(res.cv_error, float)
    assert res.cv_error > 0
    assert len(res.fold_errors) == 3


def test_cv_morals_unfitted():
    model = Morals()
    with pytest.raises(ValueError, match="must be fitted prior"):
        cv_morals(model, k=3)
