"""
Parser classes for the Australian Synchrotron.

Current beamlines supported include:
- MEX2_NEXAFS : Australian Synchrotron Medium Energy X-ray NEXAFS parser.
- SXR_NEXAFS : Australian Synchrotron Soft X-ray NEXAFS parser.
"""

from pyNexafs.parsers import parser_base
from pyNexafs.nexafs.scan import scan_base
from io import TextIOWrapper
from typing import Any
from numpy.typing import NDArray
import numpy as np
import ast
import overrides


class MEX2_NEXAFS(parser_base):
    """
    Australian Synchrotron Soft X-ray (SXR) NEXAFS parser.

    Parses data formats including '.asc' and '.mda' formats from the SXR
    Near Edge X-ray Absorption Fine Structure (NEXAFS) tool.

    Attributes
    ----------
    ALLOWED_EXTENSIONS
    SUMMARY_PARAM_RAW_NAMES
    COLUMN_ASSIGNMENTS
    RELABELS

    Notes
    -----
    Implemented for data as of 2024-Mar.
    """

    ALLOWED_EXTENSIONS = [".xdi", ".mda"]
    SUMMARY_PARAM_RAW_NAMES = [
        "Sample",
        "ROI.start_bin",
        "ROI.end_bin",
        "Element.symbol",
        "Element.edge",
    ]
    COLUMN_ASSIGNMENTS = {
        "x": "energy",
        "y": [
            "bragg",
            "ifluor|ifluor_sum",
            "count_time",
            "i0",
            "SampleDrain",
            "ICR_AVG",
            "OCR_AVG",
        ],
        "y_errs": None,
        "x_errs": None,
    }
    RELABELS = {
        "ROI.start_bin": r"$E_1$",
        "ROI.end_bin": r"$E_2$",
        "Element.symbol": "Element",
        "Element.edge": "Edge",
        "OCR_AVG": "Output Count Rate",
        "ICR_AVG": "Input Count Rate",
        "ifluor": "Fluorescence",
    }

    # @classmethod
    # @overrides.overrides
    # def file_parser(
    #     cls, file: TextIOWrapper, header_only: bool = False
    # ) -> tuple[NDArray | None, list[str], list[str], dict[str, Any]]:
    #     """Reads Australian Synchrotron Medium Energy Xray2 (MEX2) Spectroscopy files.

    #     Parameters
    #     ----------
    #     file : TextIOWrapper
    #         TextIOWrapper of the datafile (i.e. open('file.asc', 'r'))
    #     header_only : bool, optional
    #         If True, then only the header of the file is read and NDArray is returned as None, by default False

    #     Returns
    #     -------
    #     tuple[NDArray | None, list[str], dict[str, Any]]
    #         Returns a set of data as a numpy array,
    #         labels as a list of strings,
    #         units as a list of strings,
    #         and parameters as a dictionary.

    #     Raises
    #     ------
    #     ValueError
    #         If the file is not a valid filetype.
    #     """
    #     # Init vars, check file type using super method.
    #     data, labels, units, params = super().file_parser(file)

    #     # Use specific parser based on file extension.
    #     if file.name.endswith(".xdi"):
    #         data, labels, units, params = cls.parse_xdi(file, header_only=header_only)
    #     elif file.name.endswith(".mda"):
    #         data, labels, units, params = cls.parse_mda(file, header_only=header_only)
    #     else:
    #         raise NotImplementedError(
    #             f"File {file.name} is not yet supported by the {cls.__name__} parser."
    #         )

    #     # Add filename to params at the end, to avoid incorrect filename from copy files internal params.
    #     params["filename"] = file.name

    #     return data, labels, units, params

    @classmethod
    def parse_xdi(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Reads Australian Synchrotron .xdi files.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.xdi', 'r'))
        header_only : bool, optional
            If True, then only the header of the file is read and NDArray is returned as None, by default False

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array,
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .xdi file.
        """
        # Initialise structures
        params = {}
        labels = []
        units = None

        # Check valid format.
        if not file.name.endswith(".xdi"):
            raise ValueError(f"File {file.name} is not a valid .xdi file.")

        ### Read file
        # Check header is correct
        assert file.readline() == "# XDI/1.0\n"

        ## 1 Initial Parameters
        # Read first param line.
        line = file.readline()
        while "# " == line[0:2] and line != "# ///\n":
            line = line[2:].strip().split(": ", 1)  # split first colon
            param, value = tuple(line)

            # Categorise information
            if "Column." in param:
                if "Describe." in param:
                    # Label description. No dedicated structure for this, add to params.
                    params[param] = value
                else:
                    # Label name
                    labels.append(value)
            else:
                if param == "Samples":
                    # Parse the datapoint length list
                    samples = ast.literal_eval(value)
                    samples = [x.strip() if isinstance(x, str) else x for x in samples]
                    # Add to params
                    for i in range(len(samples)):
                        params[f"Datapoints.Column.{i}"] = samples[i]
                else:
                    # General parameter, add to params.
                    params[param] = value
            # Load new param line
            line = file.readline()

        assert line == "# ///\n"  # Check end of initial parameters

        # Get samplename
        line = file.readline()
        sample_name = line[2:].strip()
        params["Sample"] = sample_name

        # Skip header lines before data
        assert file.readline() == "# \n"
        line = file.readline()
        # Some xdi conversion have a "default mda2xdi" line.
        try:
            assert line == "# xdi from default mda2xdi preset for mex2.\n"
            assert file.readline() == "#--------\n"
        except AssertionError:
            assert line == "#--------\n"

        # Read data columns
        header_line = file.readline()
        assert header_line[0:2] == "# "
        labels = (
            header_line[2:].strip().split()
        )  # split on whitespace, even though formatting seems to use "   " (i.e. three spaces).
        labels = [label.strip() if type(label) == str else label for label in labels]

        if header_only:
            # Do not process remaining lines
            return None, labels, units, params

        # Read data
        lines = file.readlines()  # read remaining lines efficiently

        # Convert data to numpy array.
        data = np.loadtxt(lines)
        data = np.array(data)

        return data, labels, units, params

    @classmethod
    def parse_mda(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Reads Australian Synchrotron .mda files.

        CURRENTLY NOT IMPLEMENTED.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.mda', 'r'))

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array,
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .mda file.
        """
        # Initialise parameter list
        params = {}

        # Check valid format.
        if not file.name.endswith(".mda"):
            raise ValueError(f"File {file.name} is not a valid .mda file.")

        # Add temporary error for
        raise NotImplementedError(
            "The .mda file format is not yet supported by the SXR_NEXAFS parser."
        )

        # return np.array([]), [], [], params


class SXR_NEXAFS(parser_base):
    """
    Australian Synchrotron Soft X-ray (SXR) NEXAFS parser.

    Parses data formats including '.asc' and '.mda' formats from the SXR
    Near Edge X-ray Absorption Fine Structure (NEXAFS) tool.

    Attributes
    ----------
    ALLOWED_EXTENSIONS
    COLUMN_ASSIGNMENTS
    SUMMARY_PARAM_RAW_NAMES
    RELABELS

    Notes
    -----
    Implemented for data as of 2024-Mar.
    """

    ALLOWED_EXTENSIONS = [
        ".asc",
        ".mda",
    ]  # ascii files exported from the binary .mda files generated.

    COLUMN_ASSIGNMENTS = {
        "x": "SR14ID01PGM_CALC_ENERGY_MONITOR.P",
        "y": [
            "SR14ID01PGM:LOCAL_SP",
            "SR14ID01IOC68:scaler1.S20",
        ],
        "y_errs": None,
        "x_errs": None,
    }

    SUMMARY_PARAM_RAW_NAMES = [
        "SR14ID01NEXSCAN:saveData_comment1",
        "SR14ID01NEXSCAN:saveData_comment2",
        "SR14ID01NEX01:R_MTR.RBV",
        "SR14ID01NEX01:RULER_ID",
        "SR14ID01:BRANCH_MODE",
        "SR14ID01NEX01:C_MTR.RBV",
        "SR14ID01NEX01:Z_MTR.RBV",
        "SR14ID01NEX01:X_MTR.RBV",
        "SR14ID01NEX01:Y_MTR.RBV",
    ]

    RELABELS = {
        "SR14ID01PGM:REMOTE_SP": "Photon Energy",
        "SR14ID01PGM:LOCAL_SP": "Energy Setpoint",
        "SR14ID01PGM_ENERGY_SP": "Energy Setpoint",
        "SR14ID01PGM_CALC_ENERGY_MONITOR.P": "Encoder Photon Energy",
        "SR14ID01IOC68:scaler1.TP": "Exp Time",
        "SR14ID01IOC68:scaler1.S2": "Drain Current VF",
        "SR14ID01IOC68:scaler1.S3": "I0 VF",
        "SR14ID01IOC68:scaler1.S4": "Ref Foil VF",
        "SR14ID01IOC68:scaler1.S6": "MCP",
        "SR14ID01IOC68:scaler1.S8": "Direct PHD VF",
        "SR14ID01IOC68:scaler1.S9": "BL PHD VF",
        "SR14ID01IOC68:scaler1.S10": "Channeltron",
        "SR14ID01IOC68:scaler1.S11": "TFY PHD VF",
        "SR14ID01IOC68:scaler1.S18": "I0 VF",
        "SR14ID01IOC68:scaler1.S19": "Ref Foil VF",
        "SR14ID01IOC68:scaler1.S20": "Drain Current VF",
        "SR14ID01IOC68:scaler1.S21": "Channeltron Front (PEY)",
        "SR14ID01IOC68:scaler1.S22": "MCP (TFY)",
        "SR14ID01IOC68:scaler1.S23": "Hemispherical Analyser (AEY)",
        "SR14ID01IOC68:scaler1.S17": "Direct PHD VF",
        "SR14ID01AMP01:CURR_MONITOR": "Drain Current - Keithley1",
        "SR14ID01AMP02:CURR_MONITOR": "BL PHD - Keithley2",
        "SR14ID01AMP03:CURR_MONITOR": "I0 - Keithley3",
        "SR14ID01AMP04:CURR_MONITOR": "Ref Foil -Keithley4",
        "SR14ID01AMP05:CURR_MONITOR": "Direct PHD - Keithley5",
        "SR14ID01AMP06:CURR_MONITOR": "Keithley6",
        "SR14ID01AMP07:CURR_MONITOR": "Ref Foil - Keithley7",
        "SR14ID01AMP08:CURR_MONITOR": "Drain Current - Keithley8",
        "SR14ID01AMP09:CURR_MONITOR": "I0 - Keithley9",
        "SR14ID01:BL_GAP_REQUEST": "Undulator Gap Request",
        "SR14ID01:GAP_MONITOR": "Undulator Gap Readback",
        "SR11BCM01:CURRENT_MONITOR": "Ring Current",
        "SR14ID01NEXSCAN:saveData_comment1": "Note 1",
        "SR14ID01NEXSCAN:saveData_comment2": "Note 2",
        "SR14ID01:BRANCH_MODE": "Branch",
        "SR14ID01NEX01:RULER_ID": "Ruler ID",
        "SR14ID01NEX01:R_MTR.RBV": "R",
        "SR14ID01NEX01:C_MTR.RBV": "C",
        "SR14ID01NEX01:Z_MTR.RBV": "Z",
        "SR14ID01NEX01:X_MTR.RBV": "X",
        "SR14ID01NEX01:Y_MTR.RBV": "Y",
    }

    @classmethod
    def parse_asc(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Reads Australian Synchrotron .asc files.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.asc', 'r'))
        header_only : bool, optional
            If True, then only the header of the file is read and NDArray is returned as None, by default False

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array,
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .asc file.
        """

        # Initialise structures
        params = {"filename": file.name}
        labels = []
        units = []

        # Check valid format.
        if not file.name.endswith(".asc"):
            raise ValueError(f"File {file.name} is not a valid .asc file.")

        ### Read file
        # Check header is correct
        assert file.readline() == "## mda2ascii 0.3.2 generated output\n"

        # skip 2 lines
        [file.readline() for i in range(2)]

        ## 1 Initial Parameters including MDA File v, Scan Num, Ovrl scn dim, scn size
        for i in range(4):
            line = file.readline()[2:].strip().split(" = ", 1)
            if len(line) == 2:
                params[line[0]] = line[1]

        # skip 2 lines
        [file.readline() for i in range(2)]
        # Check PV header is correct
        assert file.readline() == "#  Extra PV: name, descr, values (, unit)\n"

        # skip 1 line
        file.readline()

        ## 2 Extra Parameter/Values
        line = file.readline()
        while line != "\n":
            # Discard first 11 characters and linenumber '# Extra PV 1:'.
            line = line.strip().split(":", 1)[1]
            # Separate the name, description, values and units.
            line = line.split(",")
            # Trim values and append to params in a tuple.
            vals = [line[i].strip().replace('"', "") for i in range(1, len(line))]
            # Convert string values to float / int if possible:
            if len(vals) == 4:  # has units column
                try:
                    vals[3] = float(vals[3]) if "." in vals[3] else int(vals[3])
                except ValueError:
                    pass
            # Add vals to params
            params[line[0].strip()] = vals

            # read the newline
            line = file.readline()

        # skip 1 extra line
        file.readline()

        ## 3 Scan Header:
        assert file.readline() == "# 1-D Scan\n"
        for i in range(3):
            line = file.readline()[2:].strip().split(" = ", 1)
            if len(line) == 2:
                params[line[0]] = line[1]

        # skip 1 line
        file.readline()

        ## 4 Scan Column properties:

        # Labels for columns describing data columns.
        column_types = {}
        line = file.readline()
        while line != "\n":
            line = line.strip().split(": ", 1)
            col_name = line[0][3:]  # ignore "#  " before name
            col_attrs = line[1].split(", ")
            column_types[col_name] = col_attrs
            # read the newline
            line = file.readline()

        # skip 1 line
        file.readline()

        # Column descriptions
        column_descriptions = {}
        line = file.readline()
        while line != "\n":
            # Take index info
            index = int(line[1:6].strip())
            index_line = (
                index == 1
            )  # Boolean to determine if on index description line.
            # Take coltype info
            desc_type = line[8:26]
            assert (
                desc_type[0] == "[" and desc_type[-1] == "]"
            )  # Check parenthesis of the line parameters
            desc_type = desc_type[1:-1].strip()
            # Check if valid col type
            valid = False if not index_line else True
            for coltype in column_types:
                if coltype in desc_type:
                    valid = True
                    break
            if not valid:
                raise ValueError(f"Invalid column type {desc_type} in line {line}")

            # Take info
            desc_info = line[28:].split(", ")
            column_descriptions[index] = desc_info
            # Check that the initial parameter begins with the Instrument descriptor.
            if not index_line:
                assert desc_info[0].startswith("SR14ID01")  # code for initial
            # Add to labels and units to lists
            labels += [desc_info[0].strip()] if not index_line else ["Index"]
            if "Positioner" in desc_type:
                pUnit = desc_info[3].strip()  # hardcoded 'unit' position
                units += [pUnit]
            elif "Detector" in desc_type:
                dUnit = desc_info[2].strip()  # hardcoded 'unit' position
                units += [dUnit if dUnit != "" else None]
            elif index_line:
                units += [None]

            # read next line
            line = file.readline()

        # add column data to params
        params["column_types"] = column_types
        params["column_descriptions"] = column_descriptions

        if header_only:
            # Do not process remaining lines
            return None, labels, units, params

        ## 5 Scan Values
        assert file.readline() == "# 1-D Scan Values\n"
        lines = file.readlines()  # read remaining lines efficiently

        # Convert data to numpy array.
        data = np.loadtxt(lines, delimiter="\t")
        # [np.loadtxt(line, delimiter="\t")
        #     for line in lines]
        data = np.array(data)

        return data, labels, units, params

    @classmethod
    def parse_mda(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Reads Australian Synchrotron .mda files.

        CURRENTLY NOT IMPLEMENTED.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.mda', 'r'))

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array,
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .mda file.
        """

        # Initialise parameter list
        params = {}

        # Check valid format.
        if not file.name.endswith(".mda"):
            raise ValueError(f"File {file.name} is not a valid .mda file.")

        # Add temporary error for
        raise NotImplementedError(
            "The .mda file format is not yet supported by the SXR_NEXAFS parser."
        )

        # return np.array([]), [], [], params

    @property
    @overrides.overrides
    def summary_param_values(self) -> list[Any]:
        """
        Returns a list of important parameter values of the data file.

        Uses the list element corresponding to 'value' for each file.
        Overrides base method to use the 'value' element of the SXR parameter list.

        Returns
        -------
        list
            List of important parameter values.
        """
        # Get second element which is the parameter number.
        return [self.params[key][1] for key in self.SUMMARY_PARAM_RAW_NAMES]

    @property
    @overrides.overrides
    def summary_param_names_with_units(self) -> list[str]:
        """
        Returns a list of important parameter names with units.

        Requires a loaded dataset to return the units of the parameters.
        Not a pre-defined class method.

        Returns
        -------
        list
            List of important parameter names with units.
        """
        pNames = self.SUMMARY_PARAM_RAW_NAMES
        pUnits = [
            self.params[pName][2]
            if (
                self.params is not None  # Params loaded
                and pName in self.params  # Parameter listed
                and hasattr(
                    self.params[pName], "__len__"
                )  # Parameter has a list of values
                and len(self.params[pName])
                == 3  # 3 params for value, description, unit
                and self.params[pName][2] != ""  # Unit value is not empty.
            )
            else None
            for pName in pNames
        ]
        if not self.relabel:
            return [
                f"{pName} ({unit})" if unit is not None else pName
                for pName, unit in zip(pNames, pUnits)
            ]
        else:
            relabel_names = []
            for pName, unit in zip(pNames, pUnits):
                if unit is not None:
                    relabel_names.append(
                        f"{self.RELABELS[pName]} ({unit})"
                        if pName in self.RELABELS
                        else f"{pName} ({unit})"
                    )
                else:
                    relabel_names.append(
                        self.RELABELS[pName] if pName in self.RELABELS else pName
                    )
            return relabel_names
