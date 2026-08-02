"""Tests for pygifi.visualization.plot — per-model plot functions and the unified plot() dispatcher."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pygifi import Homals, Princals, Morals
from pygifi import plot_object_scores, plot_quantifications, plot_biplot
from pygifi.visualization.plot import plot_homals, plot_princals, plot_morals, plot

# Use a non-interactive backend so tests don't open windows
matplotlib.use('Agg')


@pytest.fixture
def dummy_data():
    """Small mixed dataset for fitting dummy Homals/Princals/Morals models."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'A': rng.choice([1, 2, 3], size=20),
        'B': rng.uniform(0, 5, size=20),
        'C': rng.normal(0, 1, size=20),
    })


@pytest.fixture
def princals_model(dummy_data):
    return Princals(ndim=2).fit(dummy_data)


@pytest.fixture
def homals_model(dummy_data):
    return Homals(ndim=2).fit(dummy_data)


# ---------- plot_homals / plot_princals / plot_morals ----------

def test_plot_homals_objectscores(dummy_data):
    model = Homals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_homals(model.result_, ax=ax, which='objectscores')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_homals_screeplot(dummy_data):
    model = Homals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_homals(model.result_, ax=ax, type='screeplot')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_homals_transplot(dummy_data):
    model = Homals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_homals(model.result_, ax=ax, type='transplot')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_homals_objplot(dummy_data):
    model = Homals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_homals(model.result_, ax=ax, type='objplot')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_princals_biplot(dummy_data):
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_princals(model.result_, ax=ax, type='biplot')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_princals_loadings(dummy_data):
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_princals(model.result_, ax=ax, type='loadings')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_princals_screeplot(dummy_data):
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_princals(model.result_, ax=ax, type='screeplot')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_princals_transplot(dummy_data):
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_princals(model.result_, ax=ax, type='transplot')
    assert plt.gcf() is not None
    plt.close('all')


def test_plot_morals_transformation(dummy_data):
    X = dummy_data[['A', 'B']]
    y = dummy_data['C']
    model = Morals().fit(X, y)
    fig = plot_morals(model.result_)
    assert len(fig.axes) > 0
    assert plt.gcf() is not None
    plt.close('all')


# ---------- Generic dispatcher: plot_object_scores / plot_quantifications / plot_biplot ----------

def test_generic_plot_object_scores(dummy_data):
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_object_scores(model, ax=ax)
    assert plt.gcf() is not None
    plt.close('all')


def test_generic_plot_quantifications(dummy_data):
    model = Homals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_quantifications(model, ax=ax)
    assert plt.gcf() is not None
    plt.close('all')


def test_generic_plot_biplot(dummy_data):
    model = Princals().fit(dummy_data)
    fig, ax = plt.subplots()
    plot_biplot(model, ax=ax)
    assert plt.gcf() is not None
    plt.close('all')


# ---------- Unified plot() dispatcher ----------

class TestUnifiedPlotPrincals:
    def test_loadplot(self, princals_model):
        fig, ax = plt.subplots()
        ax_ret = plot(princals_model, plot_type='loadplot', ax=ax)
        assert ax_ret is ax
        assert len(ax.texts) > 0  # Should have annotations
        plt.close(fig)

    def test_biplot(self, princals_model):
        fig, ax = plt.subplots()
        ax_ret = plot(princals_model, plot_type='biplot', ax=ax)
        assert ax_ret is ax
        assert len(ax.collections) > 0  # Should have scatter points
        plt.close(fig)

    def test_transplot(self, princals_model):
        fig = plot(princals_model, plot_type='transplot')
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_screeplot(self, princals_model):
        fig = plot(princals_model, plot_type='screeplot')
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)


class TestUnifiedPlotHomals:
    def test_objplot_no_group(self, homals_model):
        fig, ax = plt.subplots()
        ax_ret = plot(homals_model, plot_type='objplot', ax=ax)
        assert ax_ret is ax
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_objplot_with_group(self, homals_model, dummy_data):
        fig, ax = plt.subplots()
        group = dummy_data['A'].values
        ax_ret = plot(homals_model, plot_type='objplot', group=group, ax=ax)
        assert ax_ret is ax
        assert len(ax.collections) > 0
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_prjplot(self, homals_model):
        fig, ax = plt.subplots()
        ax_ret = plot(homals_model, plot_type='prjplot', ax=ax)
        assert ax_ret is ax
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_vecplot(self, homals_model):
        fig, ax = plt.subplots()
        ax_ret = plot(homals_model, plot_type='vecplot', ax=ax)
        assert ax_ret is ax
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_transplot(self, homals_model):
        fig = plot(homals_model, plot_type='transplot')
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_screeplot(self, homals_model):
        fig = plot(homals_model, plot_type='screeplot')
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)


def test_plot_invalid_type(princals_model):
    with pytest.raises(ValueError, match="Unknown plot_type 'invalid'"):
        plot(princals_model, plot_type='invalid')


def test_plot_with_dict(princals_model):
    # Should work directly with the result_ dict, not just the fitted model.
    fig, ax = plt.subplots()
    plot(princals_model.result_, plot_type='biplot', ax=ax)
    assert len(ax.collections) > 0
    plt.close(fig)
