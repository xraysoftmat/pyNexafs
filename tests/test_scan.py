"""Tests the scan classes and their methods"""

import tempfile

import numpy as np
import pytest

from pyNexafs.types import dtype

from .test_parser_base import TestParserBase

# Test the creation of a new parser class.

##############################################################################
###################  ####################
##############################################################################


class TestScanAbstract:
    """Tests the scan abstract class."""


##############################################################################
###################  ####################
##############################################################################


class TestScanBase:
    """Tests the scan base class."""

    @pytest.mark.parametrize(
        "key,err",
        [
            ("a", None),
            ("b", None),
            ("g", KeyError),
            ("c", None),
            (dtype.E, None),
            (dtype.PFY, KeyError),
        ],
    )
    def test_getitem_loadall(self, key, err):
        """Tests the `__getitem__` method of the scan class,
        which should return a 1D channel, using relabels if necessary."""

        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Fake data for testing.")
            parser = TestParserBase.TestGetItem.ParserGetitem(f)
            scan = parser.to_scan(load_all_columns=True)

            # Test the getitem method
            if err:
                with pytest.raises(err):
                    _ = scan[key]
            else:
                channel = scan[key]
                assert isinstance(channel, np.ndarray), (
                    "Returned channel is not a numpy array."
                )
                assert channel.ndim == 1, "Returned channel is not 1D."

    @pytest.mark.parametrize(
        "key,err",
        [
            ("a", None),
            ("b", None),
            ("g", KeyError),
            ("c", KeyError),
            (dtype.E, KeyError),
            (dtype.PFY, KeyError),
        ],
    )
    def test_getitem_not_loadall(self, key, err):
        """Tests the `__getitem__` method of the scan class,
        which should return a 1D channel, using relabels if necessary."""

        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Fake data for testing.")
            parser = TestParserBase.TestGetItem.ParserGetitem(f)
            scan = parser.to_scan(load_all_columns=False)

            # Test the getitem method
            if err:
                with pytest.raises(err):
                    _ = scan[key]
            else:
                channel = scan[key]
                assert isinstance(channel, np.ndarray), (
                    "Returned channel is not a numpy array."
                )
                assert channel.ndim == 1, "Returned channel is not 1D."

    @pytest.mark.parametrize(
        "dt",
        [
            *TestParserBase.TestGetItem.ParserGetitem._get_dtype_relabel_channels(),
            # Try another method not in the relabels to test that it raises an error.
            dtype.PFY,
        ],
    )
    def test_getattr_dtypes(self, dt: dtype):
        """Tests the `__getattr__` method of the parserBase class for specific datatypes defined,
        which should return a 1D channel, using relabels if necessary."""

        # Create a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Fake data for testing.")
            obj = TestParserBase.TestGetItem.ParserGetitem(f)
            scan = obj.to_scan(load_all_columns=True)

            # Test the getitem method
            if dt == dtype.PFY:
                with pytest.raises(AttributeError):
                    _ = getattr(scan, dt.name)
            else:
                data = getattr(scan, dt.name)  # Gets the property object
                # Perform the fget to get the data
                data = data.fget(scan)
                assert isinstance(data, np.ndarray), (
                    f"Returned data for dtype {dt} is not a numpy array but `{type(data).__name__}`."
                )
                assert data.ndim == 1, (
                    f"Returned data for dtype {dt} is not 1D, but {data.ndim}."
                )
