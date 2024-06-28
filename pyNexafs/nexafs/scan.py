"""
Pynexafs.nexafs.scan: A set of classes for handling synchrotron scan data.

This module defines validated scan objects that can perform intelligent operations
on synchrotron data. These objects are constructed from a `parser` object that contains
the raw data and metadata.

# See Also
# --------
pynexafs.parsers.parser_base : Base class for synchrotron data parsers.
"""

from __future__ import annotations  # For type hinting within class definitions
import numpy.typing as npt
import numpy as np
import matplotlib as mpl
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import abc
import warnings
import overrides
import datetime
from scipy import optimize as sopt
from enum import Enum
from types import NoneType
from typing import Type, TypeVar, TYPE_CHECKING, Self

if TYPE_CHECKING:
    from ..parsers import parser_base


class scan_abstract(metaclass=abc.ABCMeta):
    """
    Abstract class for defining properties of a scan object.

    Base abstract class includes x,y attributes for data, errors, labels and units.
    """

    def __init__(self) -> None:
        # Internal variables
        self._x = None  # NDArray - single variable
        self._y = None  # NDArray - multiple variables
        self._x_errs = None  # NDArray - single variable
        self._y_errs = None  # NDArray - multiple variables
        self._x_label = None  # str
        self._x_unit = None  # str
        self._y_labels = []  # List[str]
        self._y_units = []  # List[str]
        return

    @property
    def ctime(self) -> datetime.datetime:
        """
        Returns the creation time of the file as a datetime object.
        """
        raise NotImplementedError("ctime property not implemented.")

    @property
    def mtime(self) -> datetime.datetime:
        """
        Returns the modification time of the file as a datetime object.
        """
        raise NotImplementedError("mtime property not implemented.")

    @property
    def filename(self) -> str:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str
            The filename of the scan object.
        """
        return ""

    @property
    def x(self) -> npt.NDArray:
        """
        Property for the 1D X data (beam energy) of the scan.

        Parameters
        ----------
        val: npt.NDArray | list[int|float]
            A new 1D array to define beam energies. Removes x_errs if new dimensions don't match.

        Returns
        -------
        npt.NDArray
            The beam energies of the scan.
        """
        return self._x

    @x.setter
    def x(self, val: npt.NDArray | list[int | float]) -> None:
        if isinstance(val, np.ndarray):
            self._x = val.copy()
        elif isinstance(val, list):
            self._x = np.array(val)
        else:
            raise ValueError("Is not a list or a np.ndarray.")
        # Remove errors if dimensions have changed.
        if self._x_errs is not None and self._x_errs.shape != self._x.shape:
            self._x_errs = None

    @property
    def y(self) -> npt.NDArray:
        """
        Property for the Y data of the scan.

        Parameters
        ----------
        val: npt.NDArray | list[int|float]
            A numpy array of data or a list of datapoints.

        Returns
        -------
        npt.NDArray
            The beam energies of the scan.
        """
        return self._y

    @y.setter
    def y(
        self, vals: npt.NDArray | list[list[int | float]] | list[int | float]
    ) -> None:
        if isinstance(vals, np.ndarray):
            if len(vals.shape) == 2:
                self._y = vals.copy()
            else:
                # Add additional axis to make y 2D instead of 1D
                self._y = vals.copy()[:, np.newaxis]
        elif isinstance(vals, list):
            if isinstance(vals[0], list):
                self._y = np.array(vals)
            else:
                # Assuming int/floats, add additional axis as well.
                self._y = np.array(vals)[:, np.newaxis]
        else:
            raise ValueError("Is not a list or a np.ndarray.")

        # Remove attributes if dimensions have changed.
        if self.y_errs is not None and self.y_errs.shape != self._y.shape:
            self.y_errs = None
        if len(self.y_labels) != len(self._y.shape[1]):
            self.y_labels = None
        if len(self.y_units) != len(self._y.shape[1]):
            self.y_units = None

    @property
    def y_errs(self) -> npt.NDArray | None:
        """
        Property for error data corresponding to the Y values.

        Parameters
        ----------
        errs : np.NDArray | None
            Error values in a numpy array matching y shape, or None to remove errors.

        Returns
        -------
        np.NDArray | None
            Errors corresponding to y values or None if not defined.
        """
        return self._y_errs

    @y_errs.setter
    def y_errs(self, errs: npt.NDArray | None) -> None:
        if errs is None:
            self._y_errs = None
        elif isinstance(errs, np.ndarray) and self.y.shape == errs.shape:
            self._y_errs = errs.copy()
        else:
            raise ValueError(
                f"Shape of errors {errs.shape} doesn't match existing y shape {self.y.shape}."
            )
        return

    @property
    def x_errs(self) -> npt.NDArray:
        """
        Property for error data corresponding to the X values.

        Parameters
        ----------
        errs : np.NDArray | None
            Error values in a numpy array matching x shape, or None to remove errors.

        Returns
        -------
        np.NDArray | None
            Errors corresponding to x values or None if not defined.
        """
        return self._x_errs

    @x_errs.setter
    def x_errs(self, errs: npt.NDArray | None) -> None:
        if errs is None:
            self._x_errs = None
        elif isinstance(errs, npt.NDArray) and errs.shape == self.x.shape:
            self._x_errs = errs.copy()
        else:
            raise ValueError(
                f"Shape of errors {errs.shape} doesn't match existing x shape {self.x.shape}."
            )

    @property
    def x_label(self) -> str:
        """
        Property for the label of the x axis.

        Parameters
        ----------
        label : str
            String to replace existing x label.

        Returns
        -------
        str
            The x label.
        """
        return self._x_label

    @x_label.setter
    def x_label(self, label: str):
        if isinstance(label, str):
            self._x_label = label
        else:
            raise ValueError(f"New label `{label}` is not string.")

    @property
    def y_labels(self) -> list[str]:
        """
        Property for the labels of the y axis.

        In the event new y data is supplied and doesn't match number of existing y_labels,
        generic y_labels are created in the form 'Data Col. N' where N is the column index.

        Parameters
        ----------
        label : list[str]
            String to replace existing y labels. Must match number of columns in the y attribute.

        Returns
        -------
        list[str]
            A direct reference to the y labels list. If y data has been redefined
            without redefining y_labels, y labels are generated in the form 'Data Col. N'.
        """
        if self._y_labels is not None:
            return self._y_labels
        else:
            chars_col_len = np.ceil(
                np.log10(self.y.shape[1] + 1)
            )  # Gets number of characters for a given number.
            ylabel_fmt_str = "Data Col. {:" + str(chars_col_len) + "d}"
            self._y_labels = [
                ylabel_fmt_str.format(i + 1) for i in range(self.y.shape[1])
            ]
            return self._y_labels

    @y_labels.setter
    def y_labels(self, labels: list[str]) -> None:
        if isinstance(labels, list) and np.all(
            [isinstance(label, str) for label in labels]
        ):
            if len(labels) == self.y.shape[1]:
                self._y_labels = labels.copy()
            else:
                raise ValueError(
                    f"Number of labels ({len(labels)}) does not match y data columns ({self.y.shape[1]})."
                )
        else:
            raise ValueError(
                f"Provided `labels` {labels} is not a list of only strings."
            )
        return

    @property
    def x_unit(self) -> str:
        """
        Property for the unit of the x data.

        Parameters
        ----------
        unit : str
            New string for the x data unit.

        Returns
        -------
        str
            The label of the x data.
        """
        return self._x_unit

    @x_unit.setter
    def x_unit(self, unit: str) -> None:
        if isinstance(unit, str):
            self._x_unit = unit
        else:
            raise ValueError(f"Provided `unit` {unit} is not a string.")
        return

    @property
    def y_units(self) -> list[str]:
        return self._y_units

    @y_units.setter
    def y_units(self, units: list[str] | None):
        if units is None:
            self._y_units = None
        elif isinstance(units, list) and np.all(
            [isinstance(unit, str) for unit in units]
        ):
            if len(units) == self.y.shape[1]:
                self._y_units = units.copy()
            else:
                raise ValueError(
                    f"Shape of 'units' ({len(units)}) does not match shape of y data ({self.y.shape[1]})."
                )
        else:
            raise ValueError(f"Provided 'units' {units} is not a list of strings.")
        return

    def copy(self, *args, **kwargs) -> Type[Self]:
        """
        Creates a copy of the scan object.
        Does reload parser object data, but does link to the same parser object.

        Returns
        -------
        Type[scan_base]
            A copy of the scan object with unique data.
        """
        newobj = type(self)(parser=None, *args, **kwargs)
        newobj.parser = self.parser

        # Copy Data
        newobj._x = self._x.copy()
        newobj._y = self._y.copy()
        newobj._y_errs = self._y_errs.copy() if self._y_errs is not None else None
        newobj._x_errs = self._x_errs.copy() if self._x_errs is not None else None

        # Copy Labels and Units
        newobj._x_label = self._x_label
        newobj._x_unit = self._x_unit
        if self._y_labels is not None:
            if isinstance(self._y_labels, list):
                newobj._y_labels = [label for label in self._y_labels]
            else:  # string
                newobj._y_labels = self._y_labels
        else:
            newobj._y_labels = None
        if self._y_units is not None:
            if isinstance(self._y_units, list):
                newobj._y_units = [unit for unit in self._y_units]
            else:  # string
                newobj._y_units = self._y_units
        else:
            newobj._y_units = None
        return newobj

    def snapshot(self, columns: int = None) -> matplotlib.figure.Figure:
        """
        Generates a grid of plots, showing all scan data.

        Parameters
        ----------
        columns : int, optional
            Number of columns used to grid data, by default None
            will use the square root of the number of Y channels.
        """
        if columns is None:
            columns = np.ceil(np.sqrt(len(self._y)))
        rows = np.ceil(len(self._y) / columns)

        rcParams = {"dpi": 300, "figure.figsize": (10, 10)}
        plt.rcParams.update(rcParams)
        fig, ax = plt.subplots(rows, columns)

        for i, ydata in enumerate(self._y):
            ax[i].plot(self._x, ydata)
            ax[i].set_title(self._y_labels[i])
        return fig

    @abc.abstractmethod
    def reload_labels_from_parser(self) -> None:
        """
        Re-loads the labels and units from the parser object.

        Useful when the user wants to switch from the raw parameter names to useful names.
        Alternatively scan labels can be manually set.
        """
        pass


class scan_base(scan_abstract):
    """
    Base class for synchrotron measurements that scans across photon beam energies (eV).

    Links to a `parser` instance that contains `parser.data` and `parser.params`.
    The `parser.COLUMN_ASSIGNMENTS` property must reflect the correct column assignments to x/y data.
    This allows for multiple Y channels, reflecting various collectors that can be used in the beamline for absorption data.

    Parameters
    ----------
    parser : Type[parser_base] | parser_base | None
        A parser object that contains the raw data and metadata.
        Parser data is loaded, and can be re-loaded.
        If None, the scan object is empty and must be loaded from a parser object.
    load_all_columns : bool, optional
        By default False, only loads the columns defined in `parser.COLUMN_ASSIGNMENTS`.
        If True, load all columns from the parser object, including those not defined in `parser.COLUMN_ASSIGNMENTS`.

    See Also
    --------
    parser_base : Base class for synchrotron data parsers.
    """

    @overrides.overrides
    def __init__(
        self,
        parser: Type[parser_base] | parser_base | None,
        load_all_columns: bool = False,
    ) -> None:
        # Initialise data arrays
        super().__init__()

        # Store parser reference.
        self._parser = parser
        self._all_columns_loaded = (
            False  # internal tracking on whether all columns have been loaded or not.
        )

        # Load data from parser
        if self.parser is not None:
            self._load_from_parser(load_all_columns=load_all_columns)
            self._all_columns_loaded = load_all_columns

    @property
    def ctime(self) -> datetime.datetime:
        """
        Returns the creation time of the file as a datetime object.
        """
        return self.parser.ctime

    @property
    def mtime(self) -> datetime.datetime:
        """
        Returns the modification time of the file as a datetime object.
        """
        return self.parser.mtime

    @scan_abstract.filename.getter
    def filename(self) -> str:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str
            The filename of the scan object.
        """
        return self.parser.filename

    @property
    def parser(self) -> Type[parser_base] | parser_base:
        """
        Property for the parser object linked to the scan object.

        Parameters
        ----------
        parser : Type[parser_base] | parser_base
            A parser object that contains the raw data and metadata.
            Parser data is loaded, and can be re-loaded.
            If None, the scan object is empty and must be loaded from a parser object.

        Returns
        -------
        Type[parser_base] | parser_base
            The parser object linked to the scan object.
        """
        return self._parser

    def reload(self, load_all_columns: bool = None) -> None:
        """
        Reloads all data from the parser object.
        """
        if load_all_columns is not None:
            self._all_columns_loaded = load_all_columns
            self._load_from_parser(load_all_columns)
        else:
            self._load_from_parser(self._all_columns_loaded)
        return

    def reload_labels_from_parser(self) -> None:
        """
        Re-loads the labels and units from the parser object.

        Useful when the user wants to switch from the raw parameter names to useful names.
        Alternatively scan labels can be manually set.
        """
        self._load_from_parser(self._all_columns_loaded, only_labels=True)
        return

    def _load_from_parser(self, load_all_columns=False, only_labels=False) -> None:
        """
        Load data from parser object.

        By default, only loads the y data specified by the class COLUMN_ASSIGNMENTS property.
        If `load_all_columns` is True, all columns are loaded.
        Data is indexed by the COLUMN_ASSIGNMENTS property.

        Parameters
        ----------
        load_all_columns : bool, optional
            Load all columns from the parser object, by default False.
        only_labels : bool, optional
            Only load labels and units from the parser object, by default False.
        """
        # Get assignments from parser object.
        assignments = self.parser.COLUMN_ASSIGNMENTS  # Validated column assignments.

        ### Load data from parser
        # X data - locate column
        x_index = self.parser.search_label_index(assignments["x"])
        # Y data - locate columns
        y_labels = assignments["y"]
        if isinstance(y_labels, list):
            y_indices = []
            for label in y_labels:
                try:
                    # label could be multiple labels, or a tuple of labels.
                    index = self.parser.search_label_index(label)
                    y_indices.append(index)
                except (AttributeError, ValueError) as e:
                    warnings.warn(
                        f"Label {label} not found in parser object {self.parser}."
                    )
        else:  # Singular index.
            y_indices = [self.parser.search_label_index(assignments["y"])]
        # Y errors - locate columns
        y_errs_labels = assignments["y_errs"]
        if isinstance(y_errs_labels, list):
            y_errs_indices = [
                (
                    self.parser.search_label_index(label)
                    if label in y_errs_labels and y_errs_labels[label] is not None
                    else None
                )
                for label in y_errs_labels
            ]
        elif isinstance(y_errs_labels, str):
            y_errs_indices = [self.parser.search_label_index(y_errs_labels)]
        else:  # y_errs_labels is None:
            y_errs_indices = None
        # X errors - locate column
        x_errs_label = assignments["x_errs"]
        x_errs_index = (
            self.parser.search_label_index(x_errs_label)
            if x_errs_label is not None
            else None
        )
        ### add conditional for load_all_columns.
        if load_all_columns:
            # iterate over columns, and add to y_indices if not existing indices.
            for i in range(self.parser.data.shape[1]):
                if (
                    i not in y_indices
                    and (y_errs_indices is None or i not in y_errs_indices)
                    and i != x_index
                    and (x_errs_index is None or i != x_errs_index)
                ):
                    y_indices.append(i)

        ### Generate data clones.
        # Data-points
        if not only_labels:
            self._x = self.parser.data[:, x_index].copy()
            self._y = self.parser.data[:, y_indices].copy()
            self._y_errs = (
                self.parser.data[:, y_errs_indices].copy()
                if y_errs_indices is not None
                else None
            )
            self._x_errs = (
                self.parser.data[:, x_errs_index].copy()
                if x_errs_index is not None
                else None
            )
        # Labels and Units
        self._x_label = (
            self.parser.labels[x_index] if self.parser.labels is not None else None
        )
        self._x_unit = (
            self.parser.units[x_index] if self.parser.units is not None else None
        )
        self._y_labels = (
            [self.parser.labels[i] for i in y_indices]
            if self.parser.labels is not None
            else None
        )
        self._y_units = (
            [self.parser.units[i] for i in y_indices]
            if self.parser.units is not None
            else None
        )
        return
