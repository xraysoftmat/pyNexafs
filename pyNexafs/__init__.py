"""
Pynexafs: Near edge X-ray absorption fine structure (NEXAFS) analysis in Python.

===============================================================================

This package provides tools for analyzing NEXAFS data, including parsers for
various file formats, a graphical user interface (GUI) for interactive analysis,
and utilities for data normalisation, manipulation, fitting and visualization.
"""

from importlib.metadata import version as get_version

__version__ = get_version(__package__)

from pyNexafs import gui, nexafs, parsers, types, utils
from pyNexafs.types import dtype

__all__ = [
    # Modules
    "parsers",
    "nexafs",
    "gui",
    "utils",
    "types",
    # Classes
    "dtype",
    # Objects
    "__version__",
]
