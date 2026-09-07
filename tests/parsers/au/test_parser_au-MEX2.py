"""Tests for the MEX2 parser - check that example files load correctly for each parser function version"""

import os

import numpy as np
import pytest

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
        @pytest.mark.parametrize("header_only", [True, False])
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
        def test_parser_mda_2024_11(self, filepath, header_only):
            # Test the class constructor success
            parser = MEX2_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )
            # Test the parser method
            with open(filepath, "r") as f:
                parser_vals = MEX2_NEXAFS.parse_mda_2024_11(f, header_only=header_only)

            # Check equivalence (i.e. the correct parser method was called)
            data, labels, units, params = parser_vals
            pdata, plabels, punits, pparams = (
                parser.data,
                parser.labels,
                parser.units,
                parser.params,
            )
            # Check consistency
            if isinstance(data, tuple) and isinstance(pdata, tuple):
                for dat, pdat in zip(data, pdata):
                    assert (
                        isinstance(dat, np.ndarray) and isinstance(pdat, np.ndarray)
                    ) or (dat is None and pdat is None)
                    if dat is not None:
                        assert np.all(pdat == dat)
            else:
                assert (
                    isinstance(data, np.ndarray) and isinstance(pdata, np.ndarray)
                ) or (data is None and pdata is None)
                if data is not None:
                    assert np.all(pdata == data)
            # Check labels, units and params
            if isinstance(labels, tuple) and isinstance(plabels, tuple):
                for lab, plab in zip(labels, plabels):
                    if lab is not None:
                        assert np.all(plab == lab)
                    else:
                        assert plab is None
            else:
                if labels is not None and plabels is not None:
                    assert np.all(plabels == labels)
                else:
                    assert labels is None and plabels is None
            # Check units
            if isinstance(units, tuple) and isinstance(punits, tuple):
                for unit, punit in zip(units, punits):
                    if unit is not None:
                        assert np.all(punit == unit)
                    else:
                        assert punit is None
            else:
                if units is not None and punits is not None:
                    assert np.all(punits == units)
                else:
                    assert units is None and punits is None

            # Check params:
            if isinstance(params, dict) and isinstance(pparams, dict):
                for key in params:
                    assert np.all(parser.params[key] == params[key])

        # Test scan conversion for reduction of multi-dimensional flourescence data
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
        def test_scan_mda_2024_11(self, filepath, header_only, energy_bin_domain):
            # Test the class constructor success
            parser = MEX2_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )

            data = parser.data
            if header_only:
                with pytest.raises(
                    ValueError, match=r"(No data loaded into the parser object)(.*+)"
                ):
                    scan = parser.to_scan(energy_bin_domain=energy_bin_domain)
                return

            assert data is not None and isinstance(data, tuple) and len(data) == 2
            data1D, data2D = data
            assert data1D is not None and data2D is not None

            scan = parser.to_scan(
                energy_bin_domain=energy_bin_domain, load_all_columns=True
            )
            # Add flourescence channels
            assert scan.y is not None
            assert (
                scan.y.shape[-1] == data1D.shape[-1] + data2D.shape[-1] + 1 - 1
            )  # +1 for sum fluor, -1 for energy (x).

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
                relabel=False,
            )

            # Depending on `to_scan`, the scan data might or might not include the 2D variables.
            # Check instead that there has been "some" additional processing.
            data = parser.data
            if header_only:
                with pytest.raises(
                    ValueError, match=r"(No data loaded into the parser object)(.*+)"
                ):
                    scan = parser.to_scan(energy_bin_domain=energy_bin_domain)
                return

            assert data is not None and isinstance(data, tuple) and len(data) == 2
            data1D, data2D = data
            assert data1D is not None and data2D is not None

            scan = parser.to_scan(
                energy_bin_domain=energy_bin_domain, load_all_columns=True
            )
            # Add flourescence channels
            assert scan.y is not None
            assert (
                scan.y.shape[-1] == data1D.shape[-1] + data2D.shape[-1] + 1 - 1
            )  # +1 for sum fluor, -1 for energy (x).


if __name__ == "__main__":
    # Test each of the parser functions with the test data.
    # MEX2 2024-03
    folder = "2025-03"
    # fname = "MEX2_13366.mda"
    fname = "MEX2_13385.mda"
    # fname = "MEX2_13366_processed.xdi"
    # MEX2 2025-03
    folder = "2024-03"
    fname = "MEX2_5641.mda"
    # fname = "MEX2_5641_processed.xdi"

    import io
    import pkgutil

    bdata = pkgutil.get_data("pyNexafs", f"../tests/test_data/au/MEX2/{folder}/{fname}")
    assert bdata is not None
    reader = io.BytesIO(bdata)
    reader.name = fname
    if fname.endswith(".xdi"):
        reader = io.TextIOWrapper(reader, encoding="utf-8")

    if "2024-03" in folder:
        if fname.endswith(".mda"):
            result = MEX2_NEXAFS.parse_mda_2024_11(reader, header_only=False)
        elif fname.endswith(".xdi"):
            result = MEX2_NEXAFS.parse_xdi(reader, header_only=False)
        else:
            raise ValueError("File extension not recognised.")
    elif "2025-03" in folder:
        if fname.endswith(".mda"):
            result = MEX2_NEXAFS.parse_mda_2025_02(reader, header_only=False)
        elif fname.endswith(".xdi"):
            result = MEX2_NEXAFS.parse_xdi(reader, header_only=False)
        else:
            raise ValueError("File extension not recognised.")
    else:
        raise ValueError("Folder not recognised.")

    print(result)
