"""
Parser classes for the Australian Synchrotron.

Current beamlines supported include are:
- SXR_NEXAFS : Australian Synchrotron Soft X-ray NEXAFS parser.
- MEX2_NEXAFS : Australian Synchrotron Medium Energy X-ray NEXAFS parser.
"""

from pyNexafs.parsers.au.aus_sync.MEX1 import MEX1_NEXAFS, MEX1_to_QANT_AUMainAsc
from pyNexafs.parsers.au.aus_sync.MEX2 import MEX2_NEXAFS, MEX2_to_QANT_AUMainAsc
from pyNexafs.parsers.au.aus_sync.SXR import SXR_NEXAFS

all = [
    SXR_NEXAFS,
    MEX1_NEXAFS,
    MEX1_to_QANT_AUMainAsc,
    MEX2_NEXAFS,
    MEX2_to_QANT_AUMainAsc,
]
