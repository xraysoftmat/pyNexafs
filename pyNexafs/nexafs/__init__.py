"""
This module contains the classes and functions to handle and process 1D NEXAFS scan data.

This includes treating the data, such as normalising it, and performing background subtraction.
"""

from pyNexafs.nexafs.normalisation.norm_settings import (
    configChannel,
    configEdges,
    configExternalChannel,
    configSeries,
    normMethod,
)
from pyNexafs.nexafs.normalisation.scan_normalised import (
    scanDoubleNorm,
    scanEnergyNorm,
    scanNorm,
    scanNormEdges,
    scanNormExt,
)
from pyNexafs.nexafs.scan import parsedScanAbstract, scanAbstract, scanBase, scanSimple

__all__ = [
    # Configuration classes
    "configChannel",
    "configExternalChannel",
    "configEdges",
    "configSeries",
    # Normalisation classes
    "normMethod",
    "scanNorm",
    "scanNormExt",
    "scanDoubleNorm",
    "scanNormEdges",
    # Scan data classes
    "scanBase",
    "scanAbstract",
    "parsedScanAbstract",
    "scanSimple",
    "scanEnergyNorm",
]
