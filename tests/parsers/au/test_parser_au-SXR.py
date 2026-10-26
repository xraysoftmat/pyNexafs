"""Tests for the MEX2 parser - check that example files load correctly for each parser function version"""

import os

import numpy as np
import pytest

from pyNexafs.parsers.au.aus_sync.SXR import SXR_NEXAFS

# Relative directories for the test data.
LOCAL_DIR = "tests/test_data/au/SXR/"
PATH_202403 = os.path.join(
    LOCAL_DIR, "2024-03"
)  # Not sure why I can't put this as a class attr.

# 128577 is the Photodiode scan (double normalisation)
# 129578 is the 90 deg (normal) scan
# 129582 is the 20 deg (grazing) scan


class TestSXR:
    """Tests the loading of example files"""

    class Test_SXR_2024_03:
        @pytest.mark.parametrize("header_only", [True, False])
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    # if "5641" not in file
                    # else pytest.param(
                    #     os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    # )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".asc")
            ],
        )
        def test_parser_sxr_asc_2024_03(self, filepath, header_only):
            # Test the class constructor success
            parser = SXR_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )
            # Test the parser method
            with open(filepath, "r") as f:
                parser_vals = SXR_NEXAFS.parse_asc_202403(f, header_only=header_only)

            # Check equivalence (i.e. the correct parser method was called)
            data, labels, units, params = parser_vals
            assert np.all(parser.data == data)
            assert np.all(parser.labels == labels)
            assert np.all(parser.units == units)
            for key in params:
                assert np.all(parser.params[key] == params[key])

        @pytest.mark.parametrize("header_only", [True, False])
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    # if "5641" not in file
                    # else pytest.param(
                    #     os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    # )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".mda")
            ],
        )
        def test_parser_sxr_mda_2024_03(self, filepath, header_only):
            # Test the class constructor success
            parser = SXR_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )
            # Test the parser method
            with open(filepath, "r") as f:
                parser_vals = SXR_NEXAFS.parse_mda_2024_03(f, header_only=header_only)

            # Check equivalence (i.e. the correct parser method was called)
            data, labels, units, params = parser_vals
            assert np.all(parser.data == data)
            assert np.all(parser.labels == labels)
            assert np.all(parser.units == units)
            for key in params:
                assert np.all(parser.params[key] == params[key])

        @pytest.mark.parametrize("header_only", [True, False])
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    # if "5641" not in file
                    # else pytest.param(
                    #     os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    # )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".txt")
            ],
        )
        def test_parser_sxr_txt_2024_03(self, filepath, header_only):
            # Test the class constructor success
            parser = SXR_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )
            # Test the parser method
            with open(filepath, "r") as f:
                parser_vals = SXR_NEXAFS.parse_txt_qant_2025_12(
                    f, header_only=header_only
                )

            # Check equivalence (i.e. the correct parser method was called)
            data, labels, units, params = parser_vals
            if data is None and not header_only:
                raise ValueError(
                    f"Data is None but header_only is FalsE. Also, {labels}"
                )

            pdata = parser.data
            plabels = parser.labels
            punits = parser.units
            pparams = parser.params
            assert np.all(pdata == data)
            assert np.all(plabels == labels)
            assert np.all(punits == units)
            for key in params:
                assert np.all(pparams[key] == params[key])

            assert parser.units is None  # Because units are not parsed in txt files.

            # Check not none data here
            assert parser.labels is not None
            assert parser.params is not None
            if not header_only:
                assert data is not None
                assert pdata is not None
