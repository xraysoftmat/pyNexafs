"""
Parser classes for the Medium Energy X-ray 2 (MEX2) beamline at the Australian Synchrotron.
"""

import PyQt6
from PyQt6 import QtWidgets
from pyNexafs.parsers import parser_base, parser_meta
from pyNexafs.nexafs.scan import scan_base
from pyNexafs.utils.mda import MDAFileReader
from pyNexafs.gui.widgets.reducer import EnergyBinReducerDialog
from io import TextIOWrapper
from typing import Any, Self, Hashable
from numpy.typing import NDArray
import numpy as np
import ast
import warnings
import datetime as dt
import os
import json, io
from pyNexafs.utils.reduction import reducer
import traceback


# Additional data provided by the MEX2 beamline for the data reduction
BIN_ENERGY_DELTA = 11.935
BIN_96_ENERGY = 1146.7
TOTAL_BINS = 4096
TOTAL_BIN_ENERGIES = np.linspace(
    start=BIN_96_ENERGY - 95 * BIN_ENERGY_DELTA,
    stop=BIN_96_ENERGY + (TOTAL_BINS - 96) * BIN_ENERGY_DELTA,
    num=TOTAL_BINS,
)
INTERESTING_BINS_IDX = [80, 900]
INTERESTING_BINS_ENERGIES = TOTAL_BIN_ENERGIES[
    INTERESTING_BINS_IDX[0] : INTERESTING_BINS_IDX[1]
]


class MEX2_NEXAFS_META(parser_meta):
    def __init__(
        cls: type,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwds: Any,
    ) -> "MEX2_NEXAFS":

        # Add extra class property for MEX2 mda data, to track binning settings
        cls.reduction_bin_domain: list[tuple[int, int]] | None = None
        """Tracker for the binning settings used in the most recent data reduction."""
        return super().__init__(name=name, bases=bases, namespace=namespace, **kwds)


class MEX2_NEXAFS(parser_base, metaclass=MEX2_NEXAFS_META):
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

    Parameters
    ----------
    filepath : str | None
        The file path to the data file.
    header_only : bool, optional
        If True, only the header of the file is read, by default False
    relabel : bool | None, optional
        If True, then the parser will relabel the data columns, by default None
    use_recent_binning : bool, optional
        If True, then the '.mda' parsers will use the most recent class
        reduction binning settings. By default True, as UIs will use this,
        and assume the parser will use the same binning settings.
        TODO: Perhaps make this a property of the base class, that way UIs can
        reset such settings if needed upon directory change etc.

    Notes
    -----
    Implemented for data as of 2024-Mar.
    """

    ALLOWED_EXTENSIONS = [".xdi", ".mda"]
    SUMMARY_PARAM_RAW_NAMES = [
        "Sample|Comment 1",
        "Angle" "Element.symbol",
        "Element.edge",
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    COLUMN_ASSIGNMENTS = {
        "x": "Energy|Energy Setpoint|energy",
        "y": [
            "Bragg|bragg",
            "Fluorescence|iflour|Fluorescence Sum|Florescence Sum (Reduced)",
            "Count Time|count_time",
            "I0|i0",
            "Sample Drain|SampleDrain",
            "ICR_AVG",
            "OCR_AVG",
            "Fluorescence Detector 1",
            "Fluorescence Detector 2",
            "Fluorescence Detector 3",
            "Fluorescence Detector 4",
        ],
        "y_errs": None,
        "x_errs": None,
    }

    RELABELS = {
        # MDA File
        "MEX2DCM01:ENERGY": "Energy Setpoint",
        "MEX2ES01ZEB01:CALC_ENERGY_EV": "Energy",
        "MEX2ES01ZEB01:GATE_TIME_SET": "Gate Time Setpoint",
        "MEX2SSCAN01:saveData_comment1": "Comment 1",
        "MEX2SSCAN01:saveData_comment2": "Comment 2",
        "MEX2ES01ZEB01:BRAGG_WITH_OFFSET": "Bragg",
        "SR11BCM01:CURRENT_MONITOR": "Current Monitor",  # What is this? I0?
        "MEX2ES01DAQ01:ch1:S:MeanValue_RBV": "Beam Intensity Monitor",  # Beam Intensity Monitor
        "MEX2ES01DAQ01:ch2:S:MeanValue_RBV": "I0",
        "MEX2ES01DAQ01:ch3:S:MeanValue_RBV": "SampleDrain",
        # 'MEX2ES01DAQ01:ch4:S:MeanValue_RBV',
        "MEX2ES01DPP01:dppAVG:InputCountRate": "ICR_AVG",
        "MEX2ES01DPP01:dppAVG:OutputCountRate": "OCR_AVG",
        # 'MEX2ES01DPP01:R:AVG:kCPS',
        "MEX2ES01DPP01:dppAVG:ElapsedLiveTime": "Count Time",
        # 'MEX2ES01ZEB01:PC_GATE_WID:RBV',
        # 'MEX2ES01DAQ01:ArrayCounter_RBV',
        # 'MEX2ES01DPP01:R:1:Total_RBV',
        # 'MEX2ES01DPP01:R:2:Total_RBV',
        # 'MEX2ES01DPP01:R:3:Total_RBV',
        # 'MEX2ES01DPP01:R:4:Total_RBV',
        # 'MEX2ES01DPP01:dpp1:InputCountRate',
        # 'MEX2ES01DPP01:dpp2:InputCountRate',
        # 'MEX2ES01DPP01:dpp3:InputCountRate',
        # 'MEX2ES01DPP01:dpp4:InputCountRate',
        # 'MEX2ES01DPP01:dpp1:OutputCountRate',
        # 'MEX2ES01DPP01:dpp2:OutputCountRate',
        # 'MEX2ES01DPP01:dpp3:OutputCountRate',
        # 'MEX2ES01DPP01:dpp4:OutputCountRate',
        # 'MEX2ES01DPP01:dpp1:ElapsedRealTime',
        # 'MEX2ES01DPP01:dpp2:ElapsedRealTime',
        # 'MEX2ES01DPP01:dpp3:ElapsedRealTime',
        # 'MEX2ES01DPP01:dpp4:ElapsedRealTime',
        # 'MEX2ES01DPP01:dpp1:ElapsedLiveTime',
        # 'MEX2ES01DPP01:dpp2:ElapsedLiveTime',
        # 'MEX2ES01DPP01:dpp3:ElapsedLiveTime',
        # 'MEX2ES01DPP01:dpp4:ElapsedLiveTime',
        # 'MEX2ES01DPP01:dpp1:DeadTime',
        # 'MEX2ES01DPP01:dpp2:DeadTime',
        # 'MEX2ES01DPP01:dpp3:DeadTime',
        # 'MEX2ES01DPP01:dpp4:DeadTime',
        # 'MEX2ES01DPP01:dpp1:PileUp',
        # 'MEX2ES01DPP01:dpp2:PileUp',
        # 'MEX2ES01DPP01:dpp3:PileUp',
        # 'MEX2ES01DPP01:dpp4:PileUp',
        # 'MEX2ES01DPP01:dpp1:F1PileUp',
        # 'MEX2ES01DPP01:dpp2:F1PileUp',
        # 'MEX2ES01DPP01:dpp3:F1PileUp',
        # 'MEX2ES01DPP01:dpp4:F1PileUp',
        # 'MEX2ES01DPP01:dpp1:Triggers',
        # 'MEX2ES01DPP01:dpp2:Triggers',
        # 'MEX2ES01DPP01:dpp3:Triggers',
        # 'MEX2ES01DPP01:dpp4:Triggers',
        # 'MEX2ES01DPP01:dpp1:Events',
        # 'MEX2ES01DPP01:dpp2:Events',
        # 'MEX2ES01DPP01:dpp3:Events',
        # 'MEX2ES01DPP01:dpp4:Events',
        # 'MEX2ES01DPP01:dpp1:F1DeadTime',
        # 'MEX2ES01DPP01:dpp2:F1DeadTime',
        # 'MEX2ES01DPP01:dpp3:F1DeadTime',
        # 'MEX2ES01DPP01:dpp4:F1DeadTime',
        # 'MEX2ES01DPP01:dpp1:FastDeadTime',
        # 'MEX2ES01DPP01:dpp2:FastDeadTime',
        # 'MEX2ES01DPP01:dpp3:FastDeadTime',
        # 'MEX2ES01DPP01:dpp4:FastDeadTime',
        # 'MEX2ES01DPP01:dpp1:InUse',
        # 'MEX2ES01DPP01:dpp2:InUse',
        # 'MEX2ES01DPP01:dpp3:InUse',
        # 'MEX2ES01DPP01:dpp4:InUse',
        # 'MEX2ES01DPP01:dpp:ArrayCounter_RBV'
        "MEX2SSCAN01:saveData_comment1": "Sample",
        "MEX2SSCAN01:saveData_comment2": "Comment 2",
        # MEX2SSCAN01:saveData_realTime1D
        # MEX2SSCAN01:saveData_fileSystem
        # MEX2SSCAN01:saveData_subDir
        # MEX2SSCAN01:saveData_fileName
        # MEX2SSCAN01:scan1.P1SM
        # MEX2SSCAN01:scan1.P2SM
        # MEX2SSCAN01:scan1.P3SM
        # MEX2SSCAN01:scan1.P4SM
        # MEX2SSCAN01:scanTypeSpec
        # MEX2SSCAN01:scan1.BSPV
        # MEX2SSCAN01:scan1.BSCD
        # MEX2SSCAN01:scan1.BSWAIT
        # MEX2SSCAN01:scan1.ASPV
        # MEX2SSCAN01:scan1.ASCD
        # MEX2SSCAN01:scan1.ASWAIT
        # MEX2SSCAN01:scan1.PDLY
        # MEX2SSCAN01:scan1.DDLY
        # MEX1ES01GLU01:MEX_TIME
        # MEX2MIR01MOT01.RBV
        # MEX2MIR01MOT02.RBV
        # MEX2MIR01MOT03.RBV
        # MEX2MIR01MOT04.RBV
        # MEX2MIR01MOT09.RBV
        # MEX2MIR01MOT10.RBV
        # MEX2MIR01MOT11.RBV
        # MEX2MIR01MOT12.RBV
        # MEX2FE01MIR01:X.RBV
        # MEX2FE01MIR01:Y.RBV
        "MEX2FE01MIR01:PITCH.RBV": "PITCH",
        "MEX2FE01MIR01:YAW.RBV": "YAW",
        # MEX2FE01MIR01:ENC_X.RBV
        # MEX2FE01MIR01:ENC_Y.RBV
        # MEX2FE01MIR01:ENC_PITCH.RBV
        # MEX2FE01MIR01:ENC_YAW.RBV
        # MEX2SLT01MOT01.RBV
        # MEX2SLT01MOT02.RBV
        # MEX2SLT01MOT03.RBV
        # MEX2SLT01MOT04.RBV
        # MEX2SLT01:VSIZE.RBV
        # MEX2SLT01:VCENTRE.RBV
        # MEX2SLT01:HSIZE.RBV
        # MEX2SLT01:HCENTRE.RBV
        # MEX2SLT01:VSIZE.OFF
        # MEX2SLT01:VCENTRE.OFF
        # MEX2SLT01:HSIZE.OFF
        # MEX2SLT01:HCENTRE.OFF
        # MEX2MIR02MOT01.RBV
        # MEX2MIR02MOT02.RBV
        # MEX2MIR02MOT03.RBV
        # MEX2MIR02MOT04.RBV
        # MEX2MIR02MOT05.RBV
        # MEX2MIR02:TRANS.RBV
        "MEX2MIR02:PITCH.RBV": "PITCH2",
        # MEX2MIR02:HEIGHT.RBV
        # MEX2MIR02:ROLL.RBV
        "MEX2MIR02:YAW.RBV": "YAW2",
        # MEX2MIR02TES04:TEMPERATURE_MONITOR
        # MEX2SLT02MOT01.RBV
        # MEX2SLT02MOT02.RBV
        # MEX2SLT02MOT03.RBV
        # MEX2SLT02MOT04.RBV
        # MEX2SLT02:VSIZE.RBV
        # MEX2SLT02:VCENTRE.RBV
        # MEX2SLT02:HSIZE.RBV
        # MEX2SLT02:HCENTRE.RBV
        # MEX2SLT02:VSIZE.OFF
        # MEX2SLT02:VCENTRE.OFF
        # MEX2SLT02:HSIZE.OFF
        # MEX2SLT02:HCENTRE.OFF
        # MEX2DCM01:ENERGY_RBV
        # MEX2DCM01:ENERGY_EV_RBV
        # MEX2DCM01:OFFSET_RBV
        # MEX2DCM01:XTAL_INBEAM.RVAL
        # MEX2DCM01:FINE_PITCH_MRAD_RBV
        # MEX2DCM01:FINE_ROLL_MRAD_RBV
        # MEX2DCM01MOT01.RBV
        # MEX2DCM01MOT02.RBV
        # MEX2DCM01MOT05.RBV
        # MEX2DCM01MOT03.RBV
        # MEX2DCM01MOT04.RBV
        # MEX2DCM01MOT01.OFF
        # MEX2DCM01MOT02.OFF
        # MEX2DCM01:y2_track
        # MEX2DCM01:y2_mvmin
        # MEX2DCM01:th_mvmin
        # MEX2DCM01:Dspace
        # MEX2DCM01:Mono111DSpace
        # MEX2DCM01:Mono111ThetaOffset
        # MEX2DCM01:Mono111HeightOffset
        # MEX2DCM01:Mono111Pitch
        # MEX2DCM01:Mono111Roll
        # MEX2DCM01:Mono111Centre
        # MEX2DCM01:MonoInSbDSpace
        # MEX2DCM01:MonoInSbThetaOffset
        # MEX2DCM01:MonoInSbHeightOffset
        # MEX2DCM01:MonoInSbPitch
        # MEX2DCM01:MonoInSbRoll
        # MEX2DCM01:MonoInSbCentre
        # MEX2AUTOROCK:PITCH_SCAN.P1WD
        # MEX2AUTOROCK:PITCH_SCAN.P1SI
        # MEX2AUTOROCK:PITCH_SCAN.NPTS
        # MEX2AUTOROCK:DETECTOR_SELECT
        # MEX2AUTOROCK:COUNTER
        # MEX2BIM01MOT01.RBV
        # MEX2BIM01:FOIL:select
        # MEX2BIM01AMP01:sens_put
        # MEX2BIM01AMP01:offset_put
        # MEX2BIM01AMP01:offset_on
        # MEX2BIM01AMP01:invert_on
        # MEX2BIM01AMP01:filter_type.RVAL
        # MEX2BIM01AMP01:low_freq.RVAL
        # MEX2REF01MOT01.RBV
        # MEX2REF01:REF:select
        # MEX2REF01AMP01:sens_put
        # MEX2REF01AMP01:offset_put
        # MEX2REF01AMP01:offset_on
        # MEX2REF01AMP01:invert_on
        # MEX2REF01AMP01:filter_type.RVAL
        # MEX2REF01AMP01:low_freq.RVAL
        # MEX2SLT03MOT01.RBV
        # MEX2SLT03MOT02.RBV
        # MEX2SLT03MOT03.RBV
        # MEX2SLT03MOT04.RBV
        # MEX2SLT03:VSIZE.RBV
        # MEX2SLT03:VCENTRE.RBV
        # MEX2SLT03:HSIZE.RBV
        # MEX2SLT03:HCENTRE.RBV
        # MEX2SLT03:VSIZE.OFF
        # MEX2SLT03:VCENTRE.OFF
        # MEX2SLT03:HSIZE.OFF
        # MEX2SLT03:HCENTRE.OFF
        "MEX2STG01MOT01.RBV": "Sample X",
        "MEX2STG01MOT02.RBV": "Sample Y",
        "MEX2STG01MOT03.RBV": "Sample Z",
        # MEX2STG01:XHAT.RBV
        # MEX2STG01:ZHAT.RBV
        # MEX2ES01MOT01.RBV
        # MEX2ES01AMP01:sens_put
        # MEX2ES01AMP01:offset_put
        # MEX2ES01AMP01:offset_on
        # MEX2ES01AMP01:invert_on
        # MEX2ES01AMP01:filter_type.RVAL
        # MEX2ES01AMP01:low_freq.RVAL
        # MEX2ES01AMP02:sens_put
        # MEX2ES01AMP02:offset_put
        # MEX2ES01AMP02:offset_on
        # MEX2ES01AMP02:invert_on
        # MEX2ES01AMP02:filter_type.RVAL
        # MEX2ES01AMP02:low_freq.RVAL
        # MEX2BLSH01SHT01:OPEN_CLOSE_STATUS
        # MEX2SZ03KSW01:KEY_CACHE_STATUS
        # MEX2SZ03KSW02:KEY_CACHE_STATUS
        # MEX2ES01DPP01:mca1.R0LO
        # MEX2ES01DPP01:mca1.R0HI
        # MEX2ES01DPP01:mca2.R0LO
        # MEX2ES01DPP01:mca2.R0HI
        # MEX2ES01DPP01:mca3.R0LO
        # MEX2ES01DPP01:mca3.R0HI
        # MEX2ES01DPP01:mca4.R0LO
        # MEX2ES01DPP01:mca4.R0HI
        # MEX2ES01DPP01:dpp1:InUse
        # MEX2ES01DPP01:dpp2:InUse
        # MEX2ES01DPP01:dpp3:InUse
        # MEX2ES01DPP01:dpp4:InUse
        # MEX2SSCAN01:PHASE_BOUNDARY_VALUE
        # MEX2SSCAN01:PHASE_STEP_VALUE
        # MEX2SSCAN01:PHASE_DURATION_VALUE
        # MEX2SSCAN01:PHASE_NUMBER_OF_POINTS
        # MEX2SSCAN01:PHASE_IN_USE
        # MEX2SSCAN01:PHASE_USES_KSPACE
        # MEX2SSCAN01:PHASE_USES_SQTIME
        # MEX2SSCAN01:TOTAL_NUMBER_OF_POINTS
        # MEX2SSCAN01:MODE
        # MEX2SSCAN01:EDGE_ENERGY
        "MEX2SSCAN01:SIMPLE_START_1_VALUE": "E1",
        # MEX2SSCAN01:SIMPLE_STEP_1_VALUE
        "MEX2SSCAN01:SIMPLE_END_1_VALUE": "E2",
        "MEX2SSCAN01:SIMPLE_START_2_VALUE": "E3",
        # MEX2SSCAN01:SIMPLE_STEP_2_VALUE
        "MEX2SSCAN01:SIMPLE_END_2_VALUE": "E4",
        # MEX2SSCAN01:SIMPLE_DURATION_VALUE
        # MEX2SSCAN01:SIMPLE_NUMBER_OF_POINTS
        "MEX2ES01DPP01:ch1:W:ArrayData": "Fluorescence Detector 1",
        "MEX2ES01DPP01:ch2:W:ArrayData": "Fluorescence Detector 2",
        "MEX2ES01DPP01:ch3:W:ArrayData": "Fluorescence Detector 3",
        "MEX2ES01DPP01:ch4:W:ArrayData": "Fluorescence Detector 4",
        #### ASC Files:
        "mda_version": "MDA File Version",
        "mda_scan_number": "Scan Number",
        "mda_rank": "Overall scan dimension",
        "mda_dimensions": "Total requested scan size",
        #### XDI File:
        # "energy": "Energy",
        # "bragg": "Bragg",
        # "count_time": "Count Time",
        # "BIM": "BIM",
        # "i0": "IO",
        # "SampleDrain": "Sample Drain",
        #### XDI Names ####
        "OCR_AVG": "Output Average Count Rate",
        "ICR_AVG": "Input Average Count Rate",
        "ROI_AD_AVG": "ROI Average",
        "ifluor": "Fluorescence",
        "ROI.start_bin": "E1",
        "ROI.end_bin": "E2",
        "Element.symbol": "Element",
        "Element.edge": "Absorption Edge",
    }

    def __init__(
        self,
        filepath: str | None,
        header_only: bool = False,
        relabel: bool | None = None,
        use_recent_binning: bool = True,
        **kwargs,
    ) -> None:
        # Manually add kwargs
        common_kwargs = {}
        if use_recent_binning is not None:
            common_kwargs.update(use_recent_binning=use_recent_binning)
        # User kwargs override common kwargs
        kwargs.update(common_kwargs)
        # Init
        super().__init__(filepath, header_only, relabel, **kwargs)

    @classmethod
    def parse_xdi(
        cls,
        file: TextIOWrapper,
        header_only: bool = False,
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
        assert file.readline() == "# XDI/1.0\n", "Invalid XDI file header."

        ## 1 Initial Parameters
        column_descriptions = {}
        column_type_assignments = {}
        # Read first param line.
        line = file.readline()
        i = 0
        while "# " == line[0:2] and line != "# ///\n":
            line = line[2:].strip().split(": ", 1)  # split first colon
            param, value = tuple(line)

            # Categorise information
            if param.startswith("Column."):
                # Label name
                labels.append(value)
                params[param] = value
            elif param.startswith("Describe.Column."):
                # Label description. No dedicated structure for this, add to params.
                col_name = params[param.replace("Describe.", "")]
                if col_name + " - " in value:
                    value = value.replace(col_name + " - ", "")
                if " -- " in value:
                    value, descr = value.split(" -- ", 1)
                else:
                    descr = None
                params[param] = value
                column_descriptions[i] = (
                    value,
                    col_name + " -- " + descr if descr is not None else col_name,
                    "",  # No units
                )
                column_type_assignments[i] = f"1-D Detector{i:4}"
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
            i += 1

        assert (
            line == "# ///\n"
        ), "End of initial parameter values"  # Check end of initial parameters

        # Get samplename from first comment line
        line = file.readline()
        sample_name = line[2:].strip()
        params["Sample"] = sample_name
        params["Comment 1"] = sample_name

        # Read the second comment line:
        line = file.readline()
        if line != "# \n":
            comment2 = line[2:].strip()
            params["Comment 2"] = comment2

        # Some xdi conversion have a "default mda2xdi" line.
        line = file.readline()
        try:
            assert (
                line == "# xdi from default mda2xdi preset for mex2.\n"
            ), "Conversion line"
            assert file.readline() == "#--------\n", "Conversion line"
        except AssertionError:
            assert line == "#--------\n", "Conversion line"

        # Read data columns
        header_line = file.readline()
        assert header_line[0:2] == "# ", "Start of data columns"
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
    def parse_mda_2024_04(
        cls,
        file: TextIOWrapper,
        header_only: bool = False,
        use_recent_binning: bool = False,
        energy_bin_domain: tuple[float, float] | None = None,
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """
        Reads Australian Synchrotron .mda files for MEX2 Data

        Created for data as of 2024-Apr.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.mda', 'r'))
        header_only : bool, optional
            If True, then only the header of the file is read and
            NDArray is returned as None, by default False
        energy_bin_domain : tuple[float, float] | None, optional
            The energy domain to bin the data, by default None
        use_recent_binning : bool, optional
            If True, then the most recent binning settings are used
            Ignored if `energy_bin_range` is specified.
            for data reduction, by default False

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
        labels = []
        units = []

        # Check valid format.
        if not file.name.endswith(".mda"):
            raise ValueError(f"File {file.name} is not a valid .mda file.")

        # Need to reopen the file in byte mode.
        file.close()
        mda = MDAFileReader(file.name)
        mda_header = mda.read_header_as_dict()
        ## Previously threw error for higher dimensional data, now just a warning.
        mda_params = mda.read_parameters()
        mda_arrays, mda_scans = mda.read_scans(header_only=header_only)

        # Add values to params dict
        params.update(mda_header)
        params.update(mda_params)

        # Initialise to None for header only reading.
        mda_1d = None
        # Add column types and descriptions to params.
        if not header_only:
            # 1D array essential
            mda_1d = mda_arrays[0]
            # 2D array optional
            if len(mda_arrays) > 1:
                mda_2d = mda_arrays[1]
                mda_2d_scan = mda_scans[1]
                # Check 'multi-channel-analyser-spectra of fluorescence-detector' names are as expected
                florescence_labels = [
                    "MEX2ES01DPP01:ch1:W:ArrayData",
                    "MEX2ES01DPP01:ch2:W:ArrayData",
                    "MEX2ES01DPP01:ch3:W:ArrayData",
                    "MEX2ES01DPP01:ch4:W:ArrayData",
                ]
                assert mda_2d_scan.labels() == florescence_labels

                # Take properties from 1D and 2D arrays:
                energies = mda_1d[:, 0] * 1000  # Convert keV to eV for MEX beamline
                dataset = mda_2d[
                    :, INTERESTING_BINS_IDX[0] : INTERESTING_BINS_IDX[1], :
                ]
                bin_e = INTERESTING_BINS_ENERGIES  # pre-calibrated.

                ## Perform binning on 2D array:
                # Is an existing binning range available?
                if energy_bin_domain is not None:
                    # Update the class variable with the new binning settings.
                    cls.reduction_bin_domain = energy_bin_domain
                    red = reducer(energies, dataset, bin_e)
                elif use_recent_binning and cls.reduction_bin_domain is not None:
                    # Uses the most recent binning settings.
                    red = reducer(energies, dataset, bin_e)
                else:
                    # Create a QT application to run the dialog.
                    if QtWidgets.QApplication.instance() is None:
                        app = QtWidgets.QApplication([])
                    # Run the Bin Selector dialog
                    window = EnergyBinReducerDialog(
                        energies=energies, dataset=dataset, bin_energies=bin_e
                    )
                    window.show()
                    if window.exec():
                        # If successful, store the binning settings and data reducer
                        cls.reduction_bin_domain = window.domain
                        red = window.reducer
                    else:
                        raise ValueError("No binning settings selected.")

                # Use the binning settings to reduce the data.
                _, reduced_summed_data = red.reduce_by_sum(
                    bin_domain=cls.reduction_bin_domain, axis=None  # all axes
                )
                _, reduced_single_detector_data = red.reduce_by_sum(
                    bin_domain=cls.reduction_bin_domain,
                    axis="bin_energies",  # just the bin energies
                )

            # 3D array unhandled.
            if len(mda_arrays) > 2:
                warnings.warn(
                    "MDA file(s) contain more than two dimension, handling of higher dimensions is not yet implemented."
                )
        # Collect units and labels:
        scan_1d = mda_scans[0]
        positioners = scan_1d.positioners
        detectors = scan_1d.detectors
        if len(mda_scans) > 1:
            scan_2d = mda_scans[1]
            positioners += scan_2d.positioners
            detectors += scan_2d.detectors

        column_types = {
            "Positioner": (
                "name",
                "descr",
                "step mode",
                "unit",
                "rdbk name",
                "rdbk descr",
                "rdbk unit",
            ),
            "Detector": (
                "name",
                "descr",
                "unit",
            ),
        }
        column_descriptions = {
            i: [
                p.name,
                p.desc,
                p.step_mode,
                p.unit,
                p.readback_name,
                p.readback_desc,
                p.readback_unit,
            ]
            for i, p in enumerate(positioners)
        }
        column_descriptions.update(
            {
                i + len(positioners): [d.name, d.desc, d.unit]
                for i, d in enumerate(detectors)
            }
        )
        params["column_types"] = column_types
        params["column_descriptions"] = column_descriptions

        # Collect units and labels:
        for i, p in enumerate(positioners):
            labels.append(p.name)
            units.append(p.unit)
        for i, d in enumerate(detectors):
            labels.append(d.name)
            units.append(d.unit)
        # If 2D data is present, add reduced data to 1D data.
        if not header_only and len(mda_arrays) > 1:
            # Check rows (energies) match length
            assert reduced_summed_data.shape[0] == mda_1d.shape[0]
            assert reduced_single_detector_data.shape[0] == mda_1d.shape[0]
            # Add reduced data to 1D data as extra columns
            mda_1d = np.c_[mda_1d, reduced_single_detector_data, reduced_summed_data]
            # Add labels and units for reduced data
            # (Detector data labels/units already added via positioners and detectors above.)
            labels += ["Florescence Sum (Reduced)"]
            units += ["a.u."]
        # Use scan time if available, otherwise let system time be used.
        if "MEX1ES01GLU01:MEX_TIME" in params:
            params["created"] = params["MEX1ES01GLU01:MEX_TIME"]

        return mda_1d, labels, units, params


def MEX2_to_QANT_AUMainAsc(
    parser: parser_base,
    extrainfo_mapping: dict[str, None | str] = {
        "SR14ID01MCS02FAM:X.RBV": None,
        "SR14ID01MCS02FAM:Y.RBV": None,
        "SR14ID01MCS02FAM:Z.RBV": None,
        "SR14ID01MCS02FAM:R1.RBV": None,
        "SR14ID01MCS02FAM:R2.RBV": None,
        "SR14ID01NEXSCAN:saveData_comment1": "Sample",
        # "SR14ID01NEXSCAN:saveData_comment1": "MEX2SSCAN01:saveData_comment1",
        "SR14ID01NEXSCAN:saveData_comment2": None,
    },
) -> list[str]:
    """
    Converts a parser mapping to to QANT format.

    Parameters
    ----------
    parser : parser_base
        The parser object (with data, labels, units, and params loaded) to convert.
    extrainfo_mapping : dict[str, str | None], optional
        Optional mapping for known read-values for the QANT AUMainAsc format to
        parser parameter names. By default the dictionary key values (readable by QANT) are:
            {"SR14ID01MCS02FAM:X.RBV": None,
            "SR14ID01MCS02FAM:Y.RBV": None,
            "SR14ID01MCS02FAM:Z.RBV": None,
            "SR14ID01MCS02FAM:R1.RBV": None,
            "SR14ID01MCS02FAM:R2.RBV": None,
            "SR14ID01NEXSCAN:saveData_comment1": None,
            "SR14ID01NEXSCAN:saveData_comment2": None,}

    Returns
    -------
    list[str]
        A list of lines for the QANT AUMainAsc format, with newline terminations included.
    """
    possible_read_values = [
        "SR14ID01MCS02FAM:X.RBV",
        "SR14ID01MCS02FAM:Y.RBV",
        "SR14ID01MCS02FAM:Z.RBV",
        "SR14ID01MCS02FAM:R1.RBV",
        "SR14ID01MCS02FAM:R2.RBV",
        "SR14ID01NEXSCAN:saveData_comment1",
        "SR14ID01NEXSCAN:saveData_comment2",
    ]
    # Check validity of the extrainfo_mapping
    for key, value in extrainfo_mapping.items():
        if value is not None:
            if value not in parser.params:
                raise ValueError(f"Parameter {value} not found in parser params.")
            elif key not in possible_read_values:
                raise ValueError(f"Parameter {key} not found in possible read values.")

    # Create reverse dict
    extrainfo_remapping = {}
    for k, v in extrainfo_mapping.items():
        if v is None:
            continue
        if v in extrainfo_remapping:
            raise ValueError(
                f"Value {v} already in remapping - conflicting mapping for `{extrainfo_remapping[v]}` and `{k}`."
            )
        extrainfo_remapping[v] = k

    # Check vailidty of parser object:
    if parser.data is None:
        raise ValueError("Parser object does not have data loaded.")

    # Check validity of the dimensionality of the object:
    if len(parser.data.shape) != 2:
        raise ValueError("Parser object data is not 2D.")

    # Define a container for the output strings, line by line.
    ostrs = []

    # Define fake asc version number:
    ostrs.append("## mda2ascii 0.3.2 generated output\n")
    ostrs.append("\n")
    ostrs.append("\n")

    # Rename for consistency between mda and asc formats.
    mda_param_names = ["mda_version", "mda_scan_number", "mda_rank", "mda_dimensions"]
    asc_param_names = [
        "MDA File Version",
        "Scan Number",
        "Overall scan dimension",
        "Total requested scan size",
    ]
    for mda, asc in zip(mda_param_names, asc_param_names):
        if mda in parser.params:
            parser.params[asc] = parser.params[mda]
            del parser.params[mda]

    # Define MDA versioning from parameters, or create fictitious version.
    ostrs.append(
        "# MDA File Version = 1.3\n"
        if "MDA File Version" not in parser.params
        else f"# MDA File Version = {parser.params['MDA File Version']}\n"
    )
    ostrs.append(
        "# Scan Number = 1\n"
        if "Scan Number" not in parser.params
        else f"# Scan Number = {parser.params['Scan Number']}\n"
    )
    ostrs.append(
        "# Overall scan dimension = 1-D\n"
        if "Overall scan dimension" not in parser.params
        else f"# Overall scan dimension = {parser.params['Overall scan dimension']}-D\n"
    )
    ostrs.append(
        f"# Total requested scan size = {len(parser.data)}\n"
        if "Total requested scan size" not in parser.params
        else f"# Total requested scan size = {parser.params['Total requested scan size']}\n"
    )
    ostrs.append("\n")
    ostrs.append("\n")

    # Define the extra PVs
    ostrs.append("#  Extra PV: name, descr, values (, unit)\n")
    ostrs.append("\n")
    param_idx = 1
    for param in parser.params:
        if param not in asc_param_names:
            wparam = (
                param
                if param not in extrainfo_remapping
                or extrainfo_remapping[param] is None
                else extrainfo_remapping[param]
            )
            line = f"# Extra PV {param_idx}: {wparam}"
            if (
                hasattr(parser.params[param], "__len__")
                and not type(parser.params[param]) is str
            ):
                for val in parser.params[param]:
                    wval = (
                        val
                        if not isinstance(val, Hashable)
                        or val not in extrainfo_remapping
                        or extrainfo_remapping[val] is None
                        else extrainfo_remapping[val]
                    )
                    line += f", {wval}"
                line += "\n"
            else:
                line += f", {parser.params[param]}, \n"
            ostrs.append(line)
            param_idx += 1
    ostrs.append("\n")
    ostrs.append("\n")

    # Define the scan header:
    ostrs.append("# 1-D Scan\n")
    ostrs.append(
        f"# Points completed = {parser.data.shape[0]} of {parser.data.shape[0]}\n"
        if "Points completed" not in parser.params
        else f"# Points completed = {parser.params['Points completed']}\n"
    )
    ostrs.append(
        f"# Scanner = SR14ID01NEXSCAN:scan1\n"
        if "Scanner" not in parser.params
        else f"# Scanner = {parser.params['Scanner']}\n"
    )
    if "Scan time" in parser.params:
        ostrs.append(f"# Scan time = {parser.params['Scan time']}\n")
    elif "created" in parser.params and isinstance(
        parser.params["created"], dt.datetime
    ):
        ostrs.append(
            f"# Scan time = {parser.params['created'].strftime(r"%b %d, %Y %H:%M:%S.%f")}\n"
        )
    elif "modified" in parser.params and isinstance(
        parser.params["modified"], dt.datetime
    ):
        ostrs.append(
            f"# Scan time = {parser.params['modified'].strftime(r"%b %d, %Y %H:%M:%S.%f")}\n"
        )
    else:
        # Use current time
        ostrs.append(
            f"# Scan time = {dt.datetime.now().strftime(r"%b %d, %Y %H:%M:%S.%f")}\n"
        )
    ostrs.append("\n")

    if "column_types" in parser.params and isinstance(
        parser.params["column_types"], dict
    ):

        for coltype in parser.params["column_types"].keys():
            line = f"#  {coltype}:"
            for val in parser.params["column_types"][coltype]:
                line += f" {val}"
                if val != parser.params["column_types"][coltype][-1]:
                    line += ","
                else:
                    line += "\n"
            ostrs.append(line)
    else:
        # Use default
        ostrs.append(
            "#  Positioner: name, descr, step mode, unit, rdbk name, rdbk descr, rdbk unit\n"
        )
        ostrs.append("#  Detector: name, descr, unit\n")
        ostrs.append("\n")

    # Define the Column Descriptions
    ostrs.append("# Column Descriptions:\n")
    if "column_descriptions" in parser.params:
        # Column descriptions have been saved. Use these.
        init_idx = 1  # default, add 1 to column description indexes
        if parser.params["column_descriptions"][0][0] != "Index":
            # Create index column if not present.
            ostrs.append(f"#{1:5}  [     Index      ]\n")
            init_idx = 2  # Because of added extra index.
        for i, col in parser.params["column_descriptions"].items():
            if "column_type_assignments" in parser.params:
                col_type = parser.params["column_type_assignments"][i]
            else:
                col_type = f"1-D Detector{i+1:4}"  # Nth detector
            line = f"#{i+init_idx+1:5}  [" + col_type + "]  "
            for val in col:
                line += f"{val}" if val is not None else ""
                if val != col[-1]:
                    line += ", "
                else:
                    line += "\n"
            ostrs.append(line)
    else:
        # Create column descriptions from units and labels.
        ostrs.append(f"#{1:5}  [     Index      ]\n")
        for i, label in enumerate(parser.labels):
            line = f"#{i+2:5}  "
            if i == 0:
                # Assume energy labels...
                line += f"[1-D Positioner 1]  {label}, "
                if "units" in parser.params and parser.params["units"][i] is not None:
                    unit = parser.params["units"][i]
                    line += f"Mono setpoint, TABLE, {unit}, {label}, Mono setpoint, {unit}\n"
                else:
                    line += f"Mono setpoint, TABLE, eV, {label}, Mono setpoint, eV\n"
            else:
                # Assume detector labels, no description.
                line += f"[1-D Detector{i:4}]  {label}, "
                if "units" in parser.params and parser.params["units"][i] is not None:
                    unit = parser.params["units"][i]
                    line += f"{label}, , {unit}\n"
                else:  # no unit.
                    line += f"{label}, , \n"
            ostrs.append(line)
    ostrs.append("\n")

    # Define the Scan Values
    ostrs.append("# 1-D Scan Values\n")
    for i, row in enumerate(parser.data):
        line = f"{i+1}"
        for val in row:
            line += f"\t{val}"
        line = line[:-1] + "\n"
        ostrs.append(line)

    # End of file
    return ostrs


if __name__ == "__main__":
    # Example usage
    path = os.path.dirname(__file__)
    package_path = os.path.normpath(os.path.join(path, "../../../../"))
    mda_paths = [
        os.path.normpath(
            os.path.join(package_path, f"tests/test_data/au/MEX2/MEX2_564{i}.mda")
        )
        for i in range(4)
        if i != 1
    ]
    mda_path1, mda_path2, mda_path3 = mda_paths
    # mda_path1, mda_path2, mda_path3, mda_path4, mda_path5 = mda_paths
    print(mda_path1)
    print(mda_path2)
    # HEADER
    test1 = MEX2_NEXAFS(mda_path1, header_only=True)
    # BODY
    test2 = MEX2_NEXAFS(mda_path1, header_only=False)
    # Check if previous binning is applied to new data.
    tests = [
        MEX2_NEXAFS(mda_path, header_only=False, use_recent_binning=True)
        for mda_path in mda_paths[1:]
    ]

    # Check that the domain can be manually applied.
    test3 = MEX2_NEXAFS(
        mda_path2,
        header_only=False,
        use_recent_binning=False,
        energy_bin_domain=(3.1e3, 3.8e3),
    )

    import matplotlib.pyplot as plt

    plt.close("all")
    subplts = plt.subplots(1, 1)
    fig: plt.Figure = subplts[0]
    ax: plt.Axes = subplts[1]
    idx = -1
    ax.plot(test2.data[:, 0], test2.data[:, idx], label="Test2" + test2.labels[idx])
    [
        ax.plot(
            test.data[:, 0], test.data[:, idx], label=f"Tests[{i}]" + test.labels[idx]
        )
        for i, test in enumerate(tests)
    ]
    ax.plot(test3.data[:, 0], test3.data[:, idx], label="Test3" + test3.labels[idx])
    ax.legend()
    # plt.ioff()

    plt.ion()
    # plt.show(block=False)
    plt.show(block=True)
