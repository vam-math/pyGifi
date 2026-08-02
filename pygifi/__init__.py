# mypy: ignore-errors
"""
pygifi — Python port of the R Gifi library.

Multivariate Analysis with Optimal Scaling.

Original R package: Mair, De Leeuw, Groenen (GPL-3.0)
Python port: GPL-3.0-or-later
"""

from pygifi.models.homals import Homals
from pygifi.models.princals import Princals
from pygifi.models.morals import Morals
from pygifi.models.impute import GifiIterativeImputer
from pygifi.models.corals import Corals
from pygifi.models.canals import Canals
from pygifi.models.criminals import Criminals
from pygifi.models.overals import Overals
from pygifi.models.primals import Primals
from pygifi.models.addals import Addals
from pygifi.utils.coding import make_numeric, encode, decode, categorical_encode, categorical_decode
from pygifi.utils.splines import knots_gifi
from pygifi.core.engine import gifi_transform
from pygifi.datasets import get_dataset
from pygifi.core.cv import cv_morals
from pygifi.utils.isotone import cone_regression
from pygifi.utils._cone import project_cone
from pygifi.visualization.plot import plot_object_scores, plot_quantifications, plot_biplot, plot

__version__ = "0.1.0"
from typing import List

__all__: List[str] = [
    "Homals", "Princals", "Morals", "GifiIterativeImputer",
    "Corals", "Canals", "Criminals", "Overals", "Primals", "Addals",
    "make_numeric", "knots_gifi", "gifi_transform", "project_cone",
    "encode", "decode", "categorical_encode", "categorical_decode",
    "get_dataset", "cv_morals", "cone_regression",
    "plot_object_scores", "plot_quantifications", "plot_biplot", "plot",
    "__version__"]

