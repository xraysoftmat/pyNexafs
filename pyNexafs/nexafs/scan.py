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
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import abc
import warnings
import overrides
import datetime
from typing import Type, TYPE_CHECKING, Self

if TYPE_CHECKING:
    from ..parsers import parser_base


class scanAbstract(metaclass=abc.ABCMeta):
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

    @abc.abstractmethod
    def copy(self, *args, **kwargs) -> Self:
        """
        Create a copy of the scan object.

        Implemented method should call this method to copy the data attributes, and implement
        copying of any subclass specific attributes.

        Parameters
        ----------
        *args, **kwargs
            Additional arguments and keyword arguments to pass to the constructor of the subclass.

        Returns
        -------
        Self
            A copy of the scan object with unique data.
        """
        newobj = self.__class__(*args, **kwargs)

        # Copy Data
        newobj._x = self._x.copy() if self._x is not None else None
        newobj._y = self._y.copy() if self._y is not None else None
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

    @property
    def ctime(self) -> datetime.datetime:
        """
        The creation time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The creation time of the file.

        Raises
        ------
        NotImplementedError
            If the ctime property is not implemented in the subclass.
        """
        raise NotImplementedError("ctime property not implemented.")

    @property
    def mtime(self) -> datetime.datetime:
        """
        The modification time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The modification time of the file.

        Raises
        ------
        NotImplementedError
            If the mtime property is not implemented in the subclass.
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
        val : npt.NDArray | list[int|float]
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

    # @y_property
    @property
    def y(self) -> npt.NDArray:
        """
        Property for the Y data of the scan.

        Parameters
        ----------
        val : npt.NDArray | list[int|float]
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
        # Set Y values
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

        # Remove errs if number of yvals have changed.
        if self.y_errs is not None and self.y_errs.shape != self._y.shape:
            self.y_errs = None

        # Remove unit/labels if number of yvals have changed
        ylabels = self.y_labels
        if ylabels is not None and len(ylabels) != self._y.shape[1]:
            self.y_labels = None
        if self.y_units is not None and len(self.y_units) != self._y.shape[1]:
            self.y_units = None

    # @y.getter_item
    # def y(self, key: int | str) -> npt.NDArray:
    #     """
    #     Returns a single Y channel from the Y data.

    #     Parameters
    #     ----------
    #     index : int | str
    #         Index of the Y channel to return, or the label of the Y channel if defined in y_labels.

    #     Returns
    #     -------
    #     npt.NDArray
    #         The Y channel data.
    #     """
    #     print("Getting Y data.")
    #     if isinstance(key, int):
    #         return self._y[:, key]
    #     elif isinstance(key, str):
    #         if self._y_labels is not None:
    #             try:
    #                 index = self._y_labels.index(key)
    #                 return self._y[:, index]
    #             except ValueError:
    #                 raise ValueError(f"Label {key} not found in y_labels.")
    #         else:
    #             raise ValueError("No y_labels defined.")

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
        label : str | None
            String to replace existing x label.

        Returns
        -------
        str
            The x label.
            Defaults to "Energy" if no label is defined.
        """
        if self._x_label is None:
            return "Energy"
        return self._x_label

    @x_label.setter
    def x_label(self, label: str | None) -> None:
        if isinstance(label, str):
            self._x_label = label
        elif label is None:
            self._x_label = None
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
            chars_col_len = int(
                np.ceil(np.log10(self.y.shape[1] + 1))
            )  # Gets number of characters for a given number.
            self._y_labels = [
                "Data Col. " + str(i + 1).zfill(chars_col_len)
                for i in range(self.y.shape[1])
            ]
            return self._y_labels

    @y_labels.setter
    def y_labels(self, labels: list[str] | None) -> None:
        if isinstance(labels, list) and np.all(
            [isinstance(label, str) for label in labels]
        ):
            if len(labels) == self.y.shape[1]:
                self._y_labels = labels.copy()
            else:
                raise ValueError(
                    f"Number of labels ({len(labels)}) does not match y data columns ({self.y.shape[1]})."
                )
        elif labels is None:
            self._y_labels = None
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

    def snapshot(self, columns: int = None) -> matplotlib.figure.Figure:
        """
        Generate a grid of plots, showing all scan data.

        Parameters
        ----------
        columns : int, optional
            Number of columns used to grid data, by default None
            will use the square root of the number of Y channels.

        Returns
        -------
        matplotlib.figure.Figure
            A figure object containing the grid of plots.
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
        Re-load the labels and units from the parser object.

        This method is abstract and must be implemented in subclasses.
        Useful when the user wants to switch from the raw parameter names to useful names.
        Alternatively scan labels can be manually set.
        """
        pass


class scanSimple(scanAbstract):
    """
    Basic interface class for raw data that is not bundled in a parser object.

    This class is used for simple scans where data is provided directly as x and y arrays.

    Parameters
    ----------
    x : npt.NDArray
        1D array of x data (e.g., beam energy).
    y : npt.NDArray
        2D array of y data (e.g., multiple Y channels). First index is data points, second index is channels.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the parent class.
    """

    def __init__(self, x: npt.NDArray, y: npt.NDArray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        x = np.array(x)  # make unique
        y = np.array(y)
        # Validate x and y data
        if not isinstance(x, np.ndarray) or len(x.shape) != 1:
            raise ValueError("x must be a 1D numpy array.")
        if not isinstance(y, np.ndarray) or len(y.shape) > 2:
            raise ValueError("y must be a 1D or 2D numpy array.")
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        self._x = x
        self._y = y
        return

    def reload_labels_from_parser(self) -> None:
        """
        Implement the abstract method from scanAbstract.

        This method is not applicable for scanSimple as it does not use a parser object.
        It is included to satisfy the abstract base class interface.
        """
        return None

    @overrides.overrides
    def copy(self) -> scanSimple:
        """
        Create a copy of the scan object.

        Data is unique for a `scan_simple` object, so no need to reload parser data unlike scan_abstract.

        Returns
        -------
        scanSimple
            A new instance of scanSimple with copied data.
        """
        new_obj = scanSimple(x=self.x.copy(), y=self.y.copy())
        return new_obj


class scanBase(scanAbstract):
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
        # Initialise data arrays, including x, y, etc.
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

    @overrides.overrides
    def copy(self, *args, **kwargs) -> Self:
        """
        Create a copy of the scan object.

        Parameters
        ----------
        *args, **kwargs
            Additional arguments and keyword arguments to pass to the constructor of the subclass.

        Returns
        -------
        Self
            A new instance of scanBase with copied data.
        """
        # Create a new instance that copies the data
        newobj = super().copy(parser=None)
        # Copy the base scan attributes:
        newobj._parser = self._parser  # Keep the same parser reference.
        newobj._all_columns_loaded = self._all_columns_loaded
        # Return the copy
        return newobj

    @property
    def ctime(self) -> datetime.datetime:
        """
        Return the creation time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The creation time of the file.
        """
        return self.parser.ctime

    @property
    def mtime(self) -> datetime.datetime:
        """
        The modification time of the file as a datetime object.

        Returns
        -------
        datetime.datetime
            The modification time of the file.
        """
        return self.parser.mtime

    @scanAbstract.filename.getter
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

    def reload(self, load_all_columns: bool | None = None) -> None:
        """
        Reload all data from the parser object.

        If `load_all_columns` is provided, it will override the current attribute setting `_all_columns_loaded`
        otherwise uses the `_all_columns_loaded` attribute to determine to load all columns or only those
        defined in `COLUMN_ASSIGNMENTS`.
        Also reloads the x, y, y_errs, x_errs, labels and units from the parser object.

        Parameters
        ----------
        load_all_columns : bool, optional
            If True/False, updates the `_all_columns_loaded` attribute.
            If True, load all columns from the parser object.
            If False, only loads the columns defined in `parser.COLUMN_ASSIGNMENTS`.
            If None, uses the current value of `_all_columns_loaded`.
        """
        if load_all_columns is not None:
            self._all_columns_loaded = load_all_columns
            self._load_from_parser(load_all_columns)
        else:
            self._load_from_parser(self._all_columns_loaded)
        return

    def reload_labels_from_parser(self) -> None:
        """
        Re-load the labels and units from the parser object.

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
        x_index = self.parser.label_index(assignments["x"])
        # Y data - locate columns
        y_labels = assignments["y"]
        if isinstance(y_labels, list):
            y_indices = []
            for label in y_labels:
                try:
                    # label could be multiple labels, or a tuple of labels.
                    index = self.parser.label_index(label)
                    y_indices.append(index)
                except (AttributeError, ValueError):
                    warnings.warn(
                        f"Label {label} not found in parser object {self.parser}."
                    )
        else:  # Singular index.
            y_indices = [self.parser.label_index(assignments["y"])]
        # Y errors - locate columns
        y_errs_labels = assignments["y_errs"]
        if isinstance(y_errs_labels, list):
            y_errs_indices = [
                (
                    self.parser.label_index(label)
                    if label in y_errs_labels and y_errs_labels[label] is not None
                    else None
                )
                for label in y_errs_labels
            ]
        elif isinstance(y_errs_labels, str):
            y_errs_indices = [self.parser.label_index(y_errs_labels)]
        else:  # y_errs_labels is None:
            y_errs_indices = None
        # X errors - locate column
        x_errs_label = assignments["x_errs"]
        x_errs_index = (
            self.parser.label_index(x_errs_label) if x_errs_label is not None else None
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
