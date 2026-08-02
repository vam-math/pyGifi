import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pygifi import Princals, Homals
from pygifi.visualization.plot import plot

# Use non-interactive backend for tests
matplotlib.use('Agg')


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'A': rng.choice([1, 2, 3], size=20),
        'B': rng.choice([0, 1], size=20),
        'C': rng.normal(0, 1, size=20)
    })


@pytest.fixture
def princals_model(dummy_data):
    return Princals(ndim=2).fit(dummy_data)


@pytest.fixture
def homals_model(dummy_data):
    return Homals(ndim=2).fit(dummy_data)


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
    # Should work directly with the result_ dict
    fig, ax = plt.subplots()
    plot(princals_model.result_, plot_type='biplot', ax=ax)
    assert len(ax.collections) > 0
    plt.close(fig)
