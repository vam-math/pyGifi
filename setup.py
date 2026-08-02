"""
Minimal setup.py shim for legacy pip/setuptools compatibility.

All real configuration lives in pyproject.toml (PEP 517/518).
This file is only needed for tools that do not yet support PEP 517.
"""
from setuptools import setup

setup()
