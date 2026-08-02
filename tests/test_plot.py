"""Tests for pygifi.plot — Smoke tests to ensure plotting functions do not crash."""

from pygifi.visualization.plot import plot_homals, plot_princals, plot_morals
from pygifi import Morals
from pygifi import Princals
from pygifi import Homals
import matplotlib.pyplot as plt
import pytest
import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so tests don't open windows
matplotlib.use('Agg')


@pytest.fixture
def dummy_data():
    """Create a tiny dataset for fitting dummy models."""
    rng = np.random.default_rng(42)
    # Just enough data to fit the models without perfect collinearity
    X = pd.DataFrame({
        'A': rng.choice([1, 2, 3], size=20),
        'B': rng.uniform(0, 5, size=20),
        'C': rng.normal(0, 1, size=20)
    })
    return X


def test_plot_homals_objectscores(dummy_data):
    """Ensure plot_homals executes without errors."""
    model = Homals().fit(dummy_data)

    fig, ax = plt.subplots()
    plot_homals(model.result_, ax=ax, which='objectscores')

    assert plt.gcf() is not None
    plt.close('all')


def test_plot_princals_biplot(dummy_data):
    """Ensure plot_princals(type='biplot') executes without errors."""
    model = Princals().fit(dummy_data)

    fig, ax = plt.subplots()
    plot_princals(model.result_, ax=ax, type='biplot')

    assert plt.gcf() is not None
    plt.close('all')


def test_plot_princals_loadings(dummy_data):
    """Ensure plot_princals(type='loadings') executes without errors."""
    model = Princals().fit(dummy_data)

    fig, ax = plt.subplots()
    plot_princals(model.result_, ax=ax, type='loadings')

    assert plt.gcf() is not None
    plt.close('all')


def test_plot_morals_transformation(dummy_data):
    """Ensure plot_morals executes without errors."""
    X = dummy_data[['A', 'B']]
    y = dummy_data['C']
    model = Morals().fit(X, y)

    # plot_morals returns a Figure
    fig = plot_morals(model.result_)

    assert len(fig.axes) > 0   # confirm axes are generated
    assert plt.gcf() is not None
    plt.close('all')


def test_generic_plot_object_scores(dummy_data):
    """Ensure generic plot_object_scores executes for Princals without errors."""
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    from pygifi import plot_object_scores
    plot_object_scores(model, ax=ax)
    assert plt.gcf() is not None
    plt.close('all')


def test_generic_plot_quantifications(dummy_data):
    """Ensure generic plot_quantifications executes for Homals without errors."""
    model = Homals().fit(dummy_data)
    fig, ax = plt.subplots()
    from pygifi import plot_quantifications
    plot_quantifications(model, ax=ax)
    assert plt.gcf() is not None
    plt.close('all')


def test_generic_plot_biplot(dummy_data):
    """Ensure generic plot_biplot executes for Princals without errors."""
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    from pygifi import plot_biplot
    plot_biplot(model, ax=ax)
    assert plt.gcf() is not None
    plt.close('all')
