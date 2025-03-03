import os, io
from xdrlib3 import Unpacker, Packer
import pandas as pd
from enum import Enum
from typing import Any, Self
import numpy as np
import numpy.typing as npt


class DBR(Enum):
    """Enumerate for the type of the parameter in the MDA file.

    Seems to be based off EPICS data storage.
    https://epics.anl.gov/docs/CAproto.html"""

    STRING = 0
    SHORT = 1
    INT = 2
    FLOAT = 3
    ENUM = 4
    LONG = 5
    DOUBLE = 6
    STS_STRING = 7
    STS_SHORT = 8
    STS_FLOAT = 9
    STS_ENUM = 10
    STS_CHAR = 11
    STS_LONG = 12
    STS_DOUBLE = 13
    TIME_STRING = 14
    TIME_SHORT = 15
    TIME_FLOAT = 16
    TIME_ENUM = 17
    TIME_CHAR = 18
    TIME_DOUBLE = 19
    GR_STRING = 20
    GR_SHORT = 21
    GR_INT = 22
    GR_FLOAT = 23
    GR_ENUM = 24
    GR_CHAR = 25
    GR_LONG = 26
    GR_DOUBLE = 27
    CTRL_STRING = 28
    CTRL_SHORT = 29  # INT16
    CTRL_FLOAT = 30
    CTRL_ENUM = 31
    CTRL_CHAR = 32  # UINT8
    CTRL_LONG = 33  # INT32
    CTRL_DOUBLE = 34


class MDAHeader:
    """
    Container for header data in an MDA file.
    """

    def __init__(
        self,
        version: float,
        scan_number: int,
        rank: int,
        dimensions: list[int],
        isRegular: int,
        pExtra: int,
    ):
        self.version = version
        self.scan_number = scan_number
        self.rank = rank
        self.dimensions = dimensions
        self.isRegular = isRegular
        self.pExtra = pExtra

    @property
    def values(self) -> tuple[float, int, int, list[int], int, int]:
        """
        Returns the header information of the MDA file as a tuple.

        Returns
        -------
        tuple[float, int, int, list[int], int, int]
            A tuple of the header information containing:
                version : float
                    The version of the MDA file.
                scan_number : int
                    The scan number of the MDA file.
                rank : int
                    The rank (number of dimensions) of the MDA file.
                dimensions,
                    The dimensions (size of each dimension) of the MDA file.
                isRegular,
                    The regularity of the MDA file.
                pExtra,
                    The byte location of the extra parameters in the MDA file.
        """
        return (
            self.version,
            self.scan_number,
            self.rank,
            self.dimensions,
            self.isRegular,
            self.pExtra,
        )


class MDAPositioner:
    """
    Container for positioner header data in an MDA file.
    """

    def __init__(
        self,
        number: int,
        fieldName: str,
        name: str,
        desc: str,
        step_mode: str,
        unit: str,
        readback_name: str,
        readback_desc: str,
        readback_unit: str,
        data: npt.NDArray | int | None,
    ):
        self.number = number
        self.fieldName = fieldName
        self.name = name
        self.desc = desc
        self.step_mode = step_mode
        self.unit = unit
        self.readback_name = readback_name
        self.readback_desc = readback_desc
        self.readback_unit = readback_unit
        self.data = data

    @property
    def values(self) -> tuple[int, str, str, str, str, str, str, str, str]:
        """
        Returns the positioner information of the MDA file as a tuple.

        Returns
        -------
        tuple[int, str, str, str, str, str, str, str, str]
            A tuple of the positioner information containing:
                number : int
                    The number of the positioner.
                fieldName : str
                    The field name of the positioner, usually number + 1.
                name : str
                    The name of the positioner.
                desc : str
                    The description of the positioner.
                step_mode : str
                    The step mode of the positioner.
                unit : str
                    The unit of the positioner.
                readback_name : str
                    A humanised name of the positioner.
                readback_desc : str
                    A humanised description of the positioner.
                readback_unit : str
                    A humanised unit of the positioner.
        """
        return (
            self.number,
            self.fieldName,
            self.name,
            self.desc,
            self.step_mode,
            self.unit,
            self.readback_name,
            self.readback_desc,
            self.readback_unit,
        )


class MDADetector:
    """
    Container for detector header data in an MDA file.
    """

    def __init__(
        self,
        number: int,
        fieldName: str,
        name: str,
        desc: str,
        unit: str,
        data: npt.NDArray | int | None,
    ):
        self.number = number
        self.fieldName = fieldName
        self.name = name
        self.desc = desc
        self.unit = unit
        self.data = data

    @property
    def values(self) -> tuple[int, str, str, str, str]:
        """
        Returns the detector information of the MDA file as a tuple.

        Returns
        -------
        tuple[int, str, str, str, str]
            A tuple of the detector information containing:
                number : int
                    The number of the detector.
                fieldName : str
                    The field name of the detector, usually number + 1.
                name : str
                    The name of the detector.
                desc : str
                    The description of the detector.
                unit : str
                    The unit of the detector.
        """
        return self.number, self.fieldName, self.name, self.desc, self.unit


class MDATrigger:
    """
    Container for trigger header data in an MDA file.
    """

    def __init__(self, number: int, name: str, command: float):
        self.number = number
        self.name = name
        self.command = command

    @property
    def values(self) -> tuple[int, str, float]:
        return self.number, self.name, self.command


class MDAScanHeader:
    """
    Container for scan header data in an MDA file.
    """

    def __init__(
        self,
        rank: int,
        points: int,
        curr_point: int,
        lower_scan_positions: list[int] | None,
        name: str | None,
        time: str | None,
        num_positioners: int,
        num_detectors: int,
        num_triggers: int,
        positioners: list[MDAPositioner],
        detectors: list[MDADetector],
        triggers: list[MDATrigger],
    ):
        self.rank = rank
        self.points = points
        self.curr_point = curr_point
        self.lower_scan_positions = lower_scan_positions
        self.name = name
        self.time = time
        self.num_positioners = num_positioners
        self.num_detectors = num_detectors
        self.num_triggers = num_triggers
        self.positioners = positioners
        self.detectors = detectors
        self.triggers = triggers


class MDAScan:
    """
    A class to store a scan header and data from an MDA file.

    Attributes
    ----------
    rank : int
        The rank (number of dimensions) of the scan data.
    points : int
        The number of points in the scan data.
    curr_point : int
        The current point in the scan data. If the scan is incomplete, this will be less than points.
    lower_scans : list[Self] | None
        A list of lower rank scans. If the scan is a 1D scan, this will be None.
    name : str
        The name of the scan.
    time : str
        The time of the scan.
    positioners : list[MDAPositioner]
        A list of positioners in the scan.
    detectors : list[MDADetector]
        A list of detectors in the scan.
    triggers : list[MDATrigger]
        A list of triggers in the scan.
    position_data : npt.NDArray | None
        The position data of the scan. None if the scan is header only.
    detector_data : npt.NDArray | None
        The detector data of the scan. None if the scan is header only.
    """

    def __init__(
        self,
        rank: int,
        points: int,
        curr_point: int,
        lower_scans: list[Self] | None,
        name: str,
        time: str,
        positioners: list[MDAPositioner],
        detectors: list[MDADetector],
        triggers: list[MDATrigger],
        position_data: npt.NDArray | None,
        detector_data: npt.NDArray | None,
    ):
        self.rank = rank
        self.points = points
        self.curr_point = curr_point
        self.lower_scans = lower_scans
        self.name = name
        self.time = time
        self.positioners = positioners
        self.detectors = detectors
        self.triggers = triggers
        self.positioner_data = position_data
        self.detector_data = detector_data

    def units(self, readback: bool = False) -> list[str]:
        """
        Returns the units of the positioners and detectors in the scan.

        Returns
        -------
        list[str]
            A list of the units of the positioners and detectors in the scan.
        """
        if not readback:
            return [p.unit for p in self.positioners] + [d.unit for d in self.detectors]
        else:
            return [p.readback_unit for p in self.positioners] + [
                d.unit for d in self.detectors
            ]

    def labels(self, readback: bool = False) -> list[str]:
        """
        Returns the labels of the positioners and detectors in the scan.

        Returns
        -------
        list[str]
            A list of the labels of the positioners and detectors in the scan.
        """
        if not readback:
            return [p.name for p in self.positioners] + [d.name for d in self.detectors]
        else:
            return [p.readback_name for p in self.positioners] + [
                d.name for d in self.detectors
            ]


class MDAFileReader:
    """
    A class to read MDA files.

    Parameters
    ----------
    path : str
        The path to the MDA file.

    Methods
    -------
    read_header(byte_default=100)
        Reads the header information of an MDA file.
    read_header_as_dict()
        Finds the header information of the MDA file and returns it as a dictionary.
    read_parameters()
        Reads the pExtra values in the MDA file.
    read_scans(header_only=False)
        Reads the scan data from the MDA file.
    """

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        self.path = path
        # Stored buffer to open and close on public functions.
        self._buffered_reader = None

        # Stored for reading various parts of the MDA file fast.
        self.pointer_header = 0
        self.pointer_pExtra = None
        self.pointer_main_scan = None

        # Store rank and dimension information.
        self.rank = None
        self.dimensions = None

    @property
    def buffered_reader(self) -> io.BufferedReader:
        if self._buffered_reader is None:
            self._buffered_reader = open(self.path, "rb")
        return self._buffered_reader

    @buffered_reader.deleter
    def buffered_reader(self):
        if self._buffered_reader is not None:
            self._buffered_reader.close()
            self._buffered_reader = None

    def read_header(
        self, byte_default=100
    ) -> tuple[float, int, int, list[int], int, int, int]:
        """Reads the header information of an MDA file.

        Records the following MDA file variables:
            version,
            scan_number,
            rank,
            dimensions,
            isRegular,
            pExtra,
            pmain_scan

        Parameters
        ----------
        byte_default : int, optional
            The number of bytes to read from the file. Default is 100.

        Returns
        -------
        tuple[float, int, int, list[int], int, int, int]
            A tuple of the header information containing:
                version : float
                    The version of the MDA file.
                scan_number : int
                    The scan number of the MDA file.
                rank : int
                    The rank (number of dimensions) of the MDA file.
                dimensions,
                    The dimensions (size of each dimension) of the MDA file.
                isRegular,
                    The regularity of the MDA file. TODO: Find out what this means.
                pExtra,
                    The byte location of the extra parameters in the MDA file.
                pmain_scan
                    The byte location of the main scan in the MDA file.
        """
        br = self.buffered_reader
        br.seek(0)
        data = br.read(byte_default)
        # Unpack the header
        u = Unpacker(data)
        header = MDAFileReader._read_header(u)
        # Record the position of the scan in the file.
        self.pointer_main_scan = br.tell() - (byte_default - u.get_position())
        del br
        # Record the position of the pExtra in the file.
        self.pointer_pExtra = header.pExtra
        self.dimensions = header.dimensions
        # Add the main scan position to the header
        return (*header.values, self.pointer_main_scan)

    def read_header_as_dict(self) -> dict[str, Any]:
        """
        Finds the header information of the MDA file and returns it as a dictionary.

        Returns
        -------
        dict[str, Any]
            The name and values of the header information of the MDA file.
        """
        header = self.read_header()
        return {
            "mda_version": header[0],
            "mda_scan_number": header[1],
            "mda_rank": header[2],
            "mda_dimensions": header[3],
            "mda_isRegular": header[4],
            "mda_pExtra": header[5],
            "mda_pmain_scan": header[6],
        }

    @staticmethod
    def _read_header(u: Unpacker, min_rank: int = 1) -> MDAHeader:
        """
        Unpacking script for header information of the MDA file.

        Parameters
        ----------
        u : Unpacker
            Unpacking object of the initial bytes of the MDA file.
        min_rank : int, optional
            Used to calculate the minimum buffer size required to read the MDA header. Default is 1.
            Raises a ValueError if the buffer is too short to read the header.

        Returns
        -------
        MDAHeader
            A MDAHeader object containing the header information of the MDA file.
        """
        # Check if the unpacker is sufficiently long enough to contain all the header information.
        if len(u.get_buffer()) < 4 * 5 + 4 * min_rank:
            raise ValueError(
                "Unpacker obj insufficiently long to decode all header information."
            )
        # Decode
        version = u.unpack_float()  # 4 bytes
        scan_number = u.unpack_int()  # 4 bytes
        rank = u.unpack_int()  # 4 bytes
        dimensions = u.unpack_farray(rank, u.unpack_int)  # 4 bytes * rank
        isRegular = u.unpack_int()  # 4 bytes
        pExtra = u.unpack_int()  # 4 bytes
        return MDAHeader(
            version=version,
            scan_number=scan_number,
            rank=rank,
            dimensions=dimensions,
            isRegular=isRegular,
            pExtra=pExtra,
        )

    @staticmethod
    def header_to_dataFrame(
        header: tuple[float, int, int, list[int], int, int, int],
    ) -> pd.DataFrame:
        (
            version,
            scan_number,
            rank,
            dimensions,
            is_regular,
            p_extra,
            p_main_scan,
        ) = header
        return pd.DataFrame(
            {
                "version": [version],
                "scan_number": [scan_number],
                "rank": [rank],
                "dimensions": [dimensions],
                "isRegular": [is_regular],
                "pExtra": [p_extra],
                "pmain_scan": [p_main_scan],
            }
        )

    def read_parameters(self) -> dict[str, tuple[str, str, str]]:
        """
        Reads the pExtra values in the MDA file.

        Returns
        -------
        dict[str, tuple[str, str, str]]
            A dictionary of pExtra data, ordered by name : tuple.
            The tuple contains the following data:
                description : str
                    The description of the parameter.
                unit : str
                    The unit of the parameter.
                value : str
                    The value of the parameter.
        """
        if self.pointer_pExtra is None:
            self.read_header()
            if self.pointer_pExtra is None:
                # Check if the pExtra position was found in the MDA file.
                raise ValueError("pExtra position not found in MDA file.")
        br = self.buffered_reader
        br.seek(self.pointer_pExtra)
        buff = br.read()
        u = Unpacker(buff)
        params = MDAFileReader._read_pExtra(u)
        del self.buffered_reader
        return params

    @staticmethod
    def _read_pExtra(
        u: Unpacker,
    ) -> dict[str | None, tuple[str | None, str | None, Any]]:
        """
        Reads the parameter information from a pExtra buffer of the MDA file.

        Parameters
        ----------
        u : Unpacker
            Unpacker object of the pExtra section of the MDA file.

        Returns
        -------
        dict[str | None, tuple[str | None, str | None , Any]]
            A dictionary of pExtra data, ordered by name : tuple.
            The tuple contains 3 items, with the following data:
                description : str | None
                    The description of the parameter.
                unit : str | None
                    The unit of the parameter.
                value : Any
                    The value of the parameter.

        Raises
        ------
        ImportError
            _description_
        """
        # Read the number of extra parameters
        numExtra = u.unpack_int()
        params = {}
        for i in range(numExtra):
            ## Read the name of the parameter
            name = MDAFileReader._read_string(u)
            ## Read the description of the parameter:
            desc = MDAFileReader._read_string(u)
            ## Does the parameter have a unit?
            count = None
            param_type = u.unpack_int()
            if param_type != 0:  # Then is not simple string value, and has unit.
                count = u.unpack_int()  # Number of values in the parameter
                unit = MDAFileReader._read_string(u)
            else:
                unit = None
            ## Read the value of the parameter:
            param_type = DBR(param_type)
            match param_type:
                case DBR.STRING:
                    val = MDAFileReader._read_string(u)
                case DBR.CTRL_CHAR:
                    arr = u.unpack_farray(count, u.unpack_int)
                    val = ""
                    # Need to treat byte array as null-terminated string
                    for i in range(count):
                        if arr[i] == 0:
                            break
                        val += chr(arr[i])
                case DBR.CTRL_SHORT:
                    val = u.unpack_farray(count, u.unpack_int)
                case DBR.CTRL_LONG:
                    val = u.unpack_farray(count, u.unpack_int)
                case DBR.CTRL_FLOAT:
                    val = u.unpack_farray(count, u.unpack_float)
                case DBR.CTRL_DOUBLE:
                    val = u.unpack_farray(count, u.unpack_double)
                case _:
                    raise ImportError(
                        f"Unspecified unpacking for DBR type: {param_type}"
                    )
            if count is not None:
                if len(val) == 1:
                    val = val[0]
            params[name] = (desc, unit, val)
        return params

    def read_scans(
        self, header_only: bool = False
    ) -> tuple[list[npt.NDArray] | None, list[MDAScan]]:
        """
        Reads the scan data from the MDA file.

        Parameters
        ----------
        header_only : bool, optional
            If True, only the header of the main scan data is read. Default is False.

        Returns
        -------
        list[npt.NDArray] | None
            A list of arrays of the scan data if header_only is False.
            Each array corresponds to data collected at each rank of the scan.
            None if header_only is True.
        list[MDAScan]
            A list of the 0th data-point scan objects on each dimension.
            list[MDAScan][0] will be the main scan.
            Provides information of the data columns via the positioners and
            detectors, and triggers attributes, which are different for each rank.
        """

        if self.pointer_main_scan is None:
            self.read_header()
            if self.pointer_main_scan is None:
                raise ValueError("Main scan position not found in MDA file.")
        br = self.buffered_reader
        br.seek(self.pointer_main_scan)
        # Initialise default data value:
        data = None
        # Read the main scan data
        if header_only:
            scans = [MDAFileReader._read_scan(br, header_only)]
            s = scans[0]
            while s.rank > 1:
                br.seek(s.lower_scans[0])
                s = MDAFileReader._read_scan(br, header_only)
                scans.append(s)
        else:
            data, scans = MDAFileReader._read_ND_scans(br, self.dimensions)
        # Close file after reading
        del self.buffered_reader
        return data, scans

    @staticmethod
    def _read_ND_scans(
        br: io.BufferedReader,
        dimensions: list[int],
        accum_data: list[npt.NDArray] | None = None,
    ) -> tuple[list[npt.NDArray], list[MDAScan]]:
        """
        Collects the scan data across various dimensions.

        As the MDA file format is a nested structure, this function is recursive
        to collect each level of scan data.

        Parameters
        ----------
        br : io.BufferedReader
            Buffered reader of the MDA file.
        dimensions : list[int]
            The dimensions of each rank of the scan data.
            Read from the head of the MDA file.
        accum_data : list[npt.NDArray] | None, optional
            Data array to enter values into, from parent recursive calls.
            Default is None.

        Returns
        -------
        tuple[list[npt.NDArray], list[MDAScan]]
            A tuple containing:
                list[npt.NDArray]
                    An array of the scan data.
                list[MDAScan]
                    A list of the 0th data-point scan objects on each dimension.
                    These scans will contain the information of the corresponding
                    data columns.
        """
        scan = MDAFileReader._read_scan(br)
        if accum_data is None:
            # Initialise scan data collection across all dimensions by
            # collecting the first scan in each dimension and storing it with dimensions.
            arrays = []
            scans = []
            diving_scan = scan
            # Use for loop to iterate over ranks.
            for i in range(diving_scan.rank):
                # Setup array to store scan ND data
                plen = len(diving_scan.positioners)
                dlen = len(diving_scan.detectors)
                tlen = len(diving_scan.triggers)
                # don't add tlen, as triggers are not stored in the data array.
                datastreams = plen + dlen  # + tlen
                # Setup shape of each rank to match datastreams present in the scan,
                # along with the dimensions previously traversed.
                shape = (*dimensions[: i + 1], datastreams)
                array = np.zeros(shape)
                # Store objects
                arrays.append(array)
                scans.append(diving_scan)
                # Dive into the next dimension scan
                if diving_scan.rank > 1:
                    diving_scan_pointer = diving_scan.lower_scans[0]
                    br.seek(diving_scan_pointer)
                    diving_scan = MDAFileReader._read_scan(br)
                else:
                    diving_scan = None
                    break
        else:
            # Unpack accumulated data:
            arrays = accum_data
            scans = None
        # Check first array element matches size of scan data
        plen = len(scan.positioners)
        dlen = len(scan.detectors)
        tlen = len(scan.triggers)
        datastreams = (
            plen + dlen
        )  # Don't add tlen, as triggers are not stored in the data array.
        assert arrays[0].shape == (dimensions[0], datastreams)
        # Store data from the current scan.
        arrays[0][:, 0:plen] = scan.positioner_data  # [datapoints, positioner_num]
        arrays[0][:, plen : plen + dlen] = scan.detector_data
        # Store data from the lower scans
        if scan.rank > 1:
            for i in range(scan.curr_point):
                pointer = scan.lower_scans[i]
                br.seek(pointer)
                arrays_subset = [array[i] for array in arrays[1:]]
                # This function will write to the arrays_subset, no explicit override is needed.
                MDAFileReader._read_ND_scans(
                    br=br, dimensions=dimensions[1:], accum_data=arrays_subset
                )
        elif scan.rank < 1:
            raise ImportError(f"Rank of scan ({scan.rank}) less than 1 is invalid.")
        # Set to NaN if the scan is incomplete.
        if scan.curr_point < scan.points:
            for array in arrays:
                array[scan.curr_point + 1 :] = np.nan
        return arrays, scans

    @staticmethod
    def _read_scan(br: io.BufferedReader, header_only: bool = False) -> MDAScan:
        # Scan header
        buff = br.read(10000)  # default used by Aus Sync Sakura.
        u = Unpacker(buff)
        scan_header = MDAFileReader._read_scan_header(u)

        if header_only:
            return MDAScan(
                rank=scan_header.rank,
                points=scan_header.points,
                curr_point=scan_header.curr_point,
                lower_scans=scan_header.lower_scan_positions,
                name=scan_header.name,
                time=scan_header.time,
                positioners=scan_header.positioners,
                detectors=scan_header.detectors,
                triggers=scan_header.triggers,
                position_data=None,
                detector_data=None,
            )

        ### Data
        ## Positioners
        # Get the new data position to read from the buffer
        extra_unpacker_bytes = len(buff) - u.get_position()
        data_pointer = br.tell() - extra_unpacker_bytes
        br.seek(data_pointer)
        buff = br.read(scan_header.num_positioners * scan_header.points * 8)
        u = Unpacker(buff)
        # Read positioner data
        position_data = np.zeros((scan_header.points, scan_header.num_positioners))
        for i in range(scan_header.num_positioners):
            position_data[:, i] = u.unpack_farray(scan_header.points, u.unpack_double)
        # Get the new detector data position to read from the buffer
        extra_unpacker_bytes = len(buff) - u.get_position()
        detector_data_pointer = br.tell() - extra_unpacker_bytes
        br.seek(detector_data_pointer)
        buff = br.read(scan_header.num_detectors * scan_header.points * 4)
        u = Unpacker(buff)
        # Read detector data
        detector_data = np.zeros((scan_header.points, scan_header.num_detectors))
        for i in range(scan_header.num_detectors):
            detector_data[:, i] = u.unpack_farray(scan_header.points, u.unpack_float)

        # Set data to NaNs if correct data if current point is less than points
        if scan_header.curr_point < scan_header.points:
            position_data[scan_header.curr_point + 1 :, :] = np.nan
            detector_data[scan_header.curr_point + 1 :, :] = np.nan

        return MDAScan(
            rank=scan_header.rank,
            points=scan_header.points,
            curr_point=scan_header.curr_point,
            lower_scans=scan_header.lower_scan_positions,
            name=scan_header.name,
            time=scan_header.time,
            positioners=scan_header.positioners,
            detectors=scan_header.detectors,
            triggers=scan_header.triggers,
            position_data=position_data,
            detector_data=detector_data,
        )

    @staticmethod
    def _read_scan_header(u: Unpacker) -> MDAScanHeader:
        rank = u.unpack_int()
        points = u.unpack_int()
        curr_point = u.unpack_int()
        if rank > 1:
            lower_scan_positions = u.unpack_farray(points, u.unpack_int)
        else:
            lower_scan_positions = None
        namelen = u.unpack_int()
        if namelen:
            assert namelen == u.unpack_int()
            name = u.unpack_fstring(namelen).decode("utf-8")
        else:
            name = None
        timelen = u.unpack_int()
        if timelen:
            assert timelen == u.unpack_int()
            time = u.unpack_fstring(timelen).decode("utf-8")
        else:
            time = None
        num_positioners = u.unpack_int()
        num_detectors = u.unpack_int()
        num_triggers = u.unpack_int()
        positioners = [
            MDAFileReader._read_positioner(u) for i in range(num_positioners)
        ]
        detectors = [MDAFileReader._read_detector(u) for i in range(num_detectors)]
        triggers = [MDAFileReader._read_trigger(u) for i in range(num_triggers)]
        return MDAScanHeader(
            rank=rank,
            points=points,
            curr_point=curr_point,
            lower_scan_positions=lower_scan_positions,
            name=name,
            time=time,
            num_positioners=num_positioners,
            num_detectors=num_detectors,
            num_triggers=num_triggers,
            positioners=positioners,
            detectors=detectors,
            triggers=triggers,
        )

    @staticmethod
    def _read_positioner(u: Unpacker) -> MDAPositioner:
        number = u.unpack_int()
        fieldName = str(number + 1)  # TODO: Check this is correct..
        name = MDAFileReader._read_string(u)
        desc = MDAFileReader._read_string(u)
        step_mode = MDAFileReader._read_string(u)
        unit = MDAFileReader._read_string(u)
        readback_name = MDAFileReader._read_string(u)
        readback_desc = MDAFileReader._read_string(u)
        readback_unit = MDAFileReader._read_string(u)
        return MDAPositioner(
            number=number,
            fieldName=fieldName,
            name=name,
            desc=desc,
            step_mode=step_mode,
            unit=unit,
            readback_name=readback_name,
            readback_desc=readback_desc,
            readback_unit=readback_unit,
            data=None,
        )

    @staticmethod
    def _read_detector(u: Unpacker) -> MDADetector:
        number = u.unpack_int()
        fieldName = str(number + 1)
        name = MDAFileReader._read_string(u)
        desc = MDAFileReader._read_string(u)
        unit = MDAFileReader._read_string(u)
        return MDADetector(
            number=number,
            fieldName=fieldName,
            name=name,
            desc=desc,
            unit=unit,
            data=None,
        )

    @staticmethod
    def _read_trigger(u: Unpacker) -> MDATrigger:
        number = u.unpack_int()
        name = MDAFileReader._read_string(u)
        command = u.unpack_float()
        return MDATrigger(number=number, name=name, command=command)

    @staticmethod
    def _read_string(u: Unpacker) -> str:
        strlen = u.unpack_int()
        if strlen:
            assert strlen == u.unpack_int()
            return u.unpack_fstring(strlen).decode("utf-8")
        else:
            return None


# TODO: Implmement reverse parser: filewriter!
# class MDAFileWriter:
#     @staticmethod
#     def write_mda_file(
#         path: str,
#         scan_arrays: list[npt.NDArray],
#         labels: list[list[str]],
#         units: list[list[str]],
#         pExtra: dict[str, tuple[str, str, Any]],
#     ):
#         # If units not supplied, set to empty lists matching labels
#         if units is None:
#             units = [["" for label in label_set] for label_set in labels]
#         # Check dimensions of inputs match appropriately
#         if np.any(len(scan_arrays) != len(labels),
#                   len(scan_arrays) != len(units)):
#             raise ValueError("Length of lists of scan_arrays, labels arrays and unit arrays must match.")
#         for i in range(len(scan_arrays)):
#             arr = scan_arrays[i]
#             label_set = labels[i]
#             unit_set = units[i] if units is not None else [""]*len(label_set)
#             if np.any(arr.shape[-1] != len(label_set),
#                       arr.shape[-1] != len(unit_set)):
#                 raise ValueError("Last dimension of scan arrays must match labels and unit array length.")


#             if np.any(len(scan_arrays[i]) != len(labels[i]),
#                       len(scan_arrays[i]) != len(units[i])):
#                 raise ValueError("Length of scan arrays, labels arrays and unit arrays must match.")

#         # Check if the path exists, if not create it.
#         os.path.isdir(os.path.dirname(path)) or os.makedirs(os.path.dirname(path))


#         return
