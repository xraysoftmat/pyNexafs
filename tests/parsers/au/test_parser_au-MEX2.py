"""Tests for the MEX2 parser - check that example files load correctly for each parser function version"""

import pytest
import os
import numpy as np
from pyNexafs.parsers.au.aus_sync.MEX2 import MEX2_NEXAFS

# Relative directories for the test data.
LOCAL_DIR = "tests/test_data/au/MEX2/"
PATH_202503 = os.path.join(
    LOCAL_DIR, "2025-03"
)  # Not sure why I can't put this as a class attr.
PATH_202403 = os.path.join(LOCAL_DIR, "2024-03")


class TestMEX2:
    """Tests the loading of example files"""

    class Test_MDA_2024_11:
        @pytest.mark.parametrize(
            "header_only, energy_bin_domain", [(True, None), (False, (2200, 2600))]
        )
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    if "5641" not in file
                    else pytest.param(
                        os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".mda")
            ],
        )
        def test_parser_mda_2024_11(self, filepath, header_only, energy_bin_domain):
            # Test the class constructor success
            parser = MEX2_NEXAFS(
                filepath,
                header_only=header_only,
                energy_bin_domain=energy_bin_domain,
                relabel=False,
            )
            # Test the parser method
            with open(filepath, "r") as f:
                parser_vals = MEX2_NEXAFS.parse_mda_2024_11(
                    f, header_only=header_only, energy_bin_domain=energy_bin_domain
                )

            # Check equivalence (i.e. the correct parser method was called)
            data, labels, units, params = parser_vals
            assert np.all(parser.data == data)
            assert np.all(parser.labels == labels)
            assert np.all(parser.units == units)
            for key in params:
                assert np.all(parser.params[key] == params[key])

    class Test_MDA_2025_02:
        @pytest.mark.parametrize(
            "header_only, energy_bin_domain", [(True, None), (False, (2200, 2600))]
        )
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202503, file)
                    if "3366" not in file
                    else pytest.param(
                        os.path.join(PATH_202503, file), marks=pytest.mark.xfail
                    )
                )  # 3366 is a partial scan. It should fail.
                for file in os.listdir(PATH_202503)
                if file.endswith(".mda")
            ],
        )
        def test_parser_mda_2025_02(self, filepath, header_only, energy_bin_domain):
            # Test the class constructor success
            parser = MEX2_NEXAFS(
                filepath,
                header_only=header_only,
                energy_bin_domain=energy_bin_domain,
                relabel=False,
            )
            # Test the parser method
            with open(filepath, "r") as f:
                parser_vals = MEX2_NEXAFS.parse_mda_2025_02(
                    f, header_only=header_only, energy_bin_domain=energy_bin_domain
                )

            # Check equivalence (i.e. the correct parser method was called)
            data, labels, units, params = parser_vals
            assert np.all(parser.data == data)
            assert np.all(parser.labels == labels)
            assert np.all(parser.units == units)
            for key in params:
                assert np.all(parser.params[key] == params[key])
