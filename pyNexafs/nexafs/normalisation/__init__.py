"""
A module for all normalisation and background subtraction operations on the data.

Normalisation operations are designed to be repeatable and tracable, such that
the order of operations is clear and the normalisation can be reproduced between
different datasets.
"""

# Config Classes
# Config Enumerates
from pyNexafs.nexafs.normalisation.norm_settings import (
    configBase,
    configChannel,
    configEdges,
    configExternalChannel,
    configSeries,
    configX,
    extSelection,
    normMethod,
)
from pyNexafs.nexafs.normalisation.scan_normalised import (
    scanDoubleNorm,
    scanEnergyNorm,
    scanNorm,
    scanNormEdges,
    scanNormExt,
)

# from pyNexafs.nexafs.normalisation.scan_normalised import (
#     scan_norm,
#     scan_normalised_edges,
#     # scan_background_subtraction,
#     scan_normalised_background_channel,
# )

__all__ = [
    # Config Classes
    "configBase",
    "configSeries",
    "configX",
    "configChannel",
    "configExternalChannel",
    "configEdges",
    # Enumerates
    "normMethod",
    "extSelection",
    # Normalisation Classes
    "scanNorm",
    "scanNormExt",
    "scanDoubleNorm",
    "scanNormEdges",
    "scanEnergyNorm",
]
