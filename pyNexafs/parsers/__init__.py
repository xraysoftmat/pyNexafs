"""
Parser classes for NEXAFS data, including support for synchrotron specific beamlines.

Parsers are registered in the parser_loaders dictionary, which is used to register parsers
in GUI applications.
"""

# Base objects
# Specific Parsers
from pyNexafs.parsers import au
from pyNexafs.parsers._base import parserBase, parserMeta
from pyNexafs.parsers.au import MEX1_NEXAFS, MEX2_NEXAFS, SXR_NEXAFS

# Define loaders with string representation.
parser_loaders = {
    "au SXR:NEXAFS": au.SXR_NEXAFS,
    "au MEX1:NEXAFS": au.MEX1_NEXAFS,
    "au MEX2:NEXAFS": au.MEX2_NEXAFS,
}

# Check that all parsers are subclasses of the base parser.
for parser in parser_loaders.values():
    assert issubclass(parser, parserBase), f"{parser} is not a subclass of parserBase."

# Check that no parser names overlap. Important for GUIs.
parser_names = [parser_name for parser_name in parser_loaders]
assert len(parser_names) == len(set(parser_names)), "Parser names overlap."

__all__ = [
    "MEX1_NEXAFS",
    "MEX2_NEXAFS",
    "SXR_NEXAFS",
    "au",
    "parserBase",
    "parserMeta",
    "parser_loaders",
]
