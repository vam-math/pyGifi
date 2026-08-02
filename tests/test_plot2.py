from pygifi.visualization.plot import plot_homals, plot_princals
from pygifi import Princals
from pygifi import Homals
import matplotlib.pyplot as plt
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'A': rng.choice([1, 2, 3], size=20),
        'B': rng.uniform(0, 5, size=20)
    })


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
