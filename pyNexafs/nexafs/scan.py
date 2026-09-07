"""
Pynexafs.nexafs.scan: A set of classes for handling synchrotron scan data.

This module defines validated scan objects that can perform intelligent operations
on synchrotron data. These objects are constructed from a `parser` object that contains
the raw data and metadata.

See Also
--------
pynexafs.parsers.parserBase : Base class for synchrotron data parsers.
"""

# Stdlib imports
from __future__ import annotations  # For type hinting within class definitions

import abc
import datetime
import io
import os
from typing import TYPE_CHECKING, Self, override

import numpy as np

# External imports
import numpy.typing as npt

try:
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Internal imports
from pyNexafs.types import dtype

if TYPE_CHECKING:
    from ..parsers import parserBase


class scanAbstract(metaclass=abc.ABCMeta):
    """
    Abstract class for defining properties of a scan object.

    Base abstract class includes x,y attributes for data, errors, labels and units.
    """

    def __init__(self) -> None:
        # Internal variables
        self._x: npt.NDArray | None = None
        self._y: npt.NDArray | None = None
        self._x_errs: npt.NDArray | None = None
        self._y_errs: npt.NDArray | None = None
        self._x_label: str | None = None
        self._x_unit: str | None = None
        self._y_labels: list[str | None] | None = None
        self._y_units: list[str | None] | None = None

    @abc.abstractmethod
    def copy(self, newobj: scanAbstract | None = None) -> Self:
        """
        Create a copy of the scan object.

        Implemented method should call this method to copy the data attributes, and implement
        copying of any subclass specific attributes.

        Parameters
        ----------
        newobj : scanAbstract | None
            An optional existing scan object to copy data into.
            If None, a new scan object is created.

        Returns
        -------
        Self
            A copy of the scan object with unique data.
        """
        if newobj is None:
            newobj = self.__class__()

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
    def ctime(self) -> datetime.datetime | None:
        """
        The creation time of the file as a datetime object.

        Returns
        -------
        datetime.datetime | None
            The creation time of the file.
            None if not defined.

        Raises
        ------
        NotImplementedError
            If the ctime property is not implemented in the subclass.
        """
        raise NotImplementedError("ctime property not implemented.")

    @property
    def mtime(self) -> datetime.datetime | None:
        """
        The modification time of the file as a datetime object.

        Returns
        -------
        datetime.datetime | None
            The modification time of the file.
            None if not defined.

        Raises
        ------
        NotImplementedError
            If the mtime property is not implemented in the subclass.
        """
        raise NotImplementedError("mtime property not implemented.")

    @property
    def x(self) -> npt.NDArray | None:
        """
        Property for the 1D X data (beam energy) of the scan.

        Parameters
        ----------
        val : npt.NDArray | list[int|float]
            A new 1D array to define beam energies. Removes x_errs if new dimensions don't match.

        Returns
        -------
        npt.NDArray | None
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
    def y(self) -> npt.NDArray | None:
        """
        Property for the Y data of the scan.

        Parameters
        ----------
        val : npt.NDArray | list[int|float] | None
            A numpy array of data or a list of datapoints.

        Returns
        -------
        npt.NDArray
            The beam energies of the scan.
        """
        return self._y

    @y.setter
    def y(
        self, vals: npt.NDArray | list[list[int | float]] | list[int | float] | None
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
        elif vals is None:
            self._y = None
            self._y_errs = None
            self._y_labels = None
            return
        else:
            raise ValueError("Is not a list or a np.ndarray.")

        # Remove errs if number of yvals have changed.
        if self.y_errs is not None and self.y_errs.shape != self._y.shape:
            self.y_errs = None

        # Remove unit/labels if number of yvals have changed
        ylabels = self.y_labels
        if ylabels is not None and len(ylabels) != self._y.shape[1]:
            self.y_labels = None
        yunits = self.y_units
        if yunits is not None and len(yunits) != self._y.shape[1]:
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
    def y_errs(
        self, errs: npt.NDArray | list[list[int | float]] | list[int | float] | None
    ) -> None:
        if errs is None:
            self._y_errs = None
            return
        y = self.y

        if isinstance(errs, np.ndarray):
            if len(errs.shape) == 2:
                errs = errs.copy()
            else:
                # Add additional axis to make y 2D instead of 1D
                errs = errs.copy()[:, np.newaxis]
        elif isinstance(errs, list):
            if isinstance(errs[0], list):
                errs = np.array(errs)
            else:
                # Assuming int/floats, add additional axis as well.
                errs = np.array(errs)[:, np.newaxis]
        else:
            raise ValueError("Is not a list of numbers or a np.ndarray.")

        if y is not None:
            if y.shape == errs.shape:
                self._y_errs = errs
            else:
                raise ValueError(
                    f"Shape of errors {errs.shape} doesn't match existing y shape {y.shape}."
                )
        else:
            self._y_errs = errs  # Allow setting errs if y is None.
        return

    @property
    def x_errs(self) -> npt.NDArray | None:
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
        x = self.x
        if isinstance(errs, np.ndarray):
            if x is not None:
                if errs.shape == x.shape:
                    self._x_errs = errs.copy()
                else:
                    raise ValueError(
                        f"Shape of errors {errs.shape} doesn't match existing x shape {x.shape}."
                    )
            else:
                self._x_errs = errs.copy()  # Allow setting errs if x is None.
        else:
            raise ValueError("Is not a np.ndarray.")

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
    def y_labels(self) -> list[str | None] | None:
        """
        Property for the labels of the y axis.

        In the event new y data is supplied and doesn't match number of existing y_labels,
        generic y_labels are created in the form 'Data Col. N' where N is the column index.

        Parameters
        ----------
        labels : list[str | None] | None
            String to replace existing y labels. Must match number of columns in the y attribute.

        Returns
        -------
        list[str | None] | None
            A direct reference to the y labels list. If y data has been redefined
            without redefining y_labels, y labels are generated in the form 'Data Col. N'.
        """
        if self._y_labels is not None:
            return self._y_labels
        else:
            y = self.y
            if y is None:
                return None

            chars_col_len = int(
                np.ceil(np.log10(y.shape[1] + 1))
            )  # Gets number of characters for a given number.
            self._y_labels = [
                "Data Col. " + str(i + 1).zfill(chars_col_len)
                for i in range(y.shape[1])
            ]
            return self._y_labels

    @y_labels.setter
    def y_labels(self, labels: list[str | None] | None) -> None:
        if isinstance(labels, list) and np.all(
            [isinstance(label, str) or label is None for label in labels]
        ):
            y = self.y
            if y is not None:
                if len(labels) == y.shape[1]:
                    self._y_labels = labels.copy()
                else:
                    raise ValueError(
                        f"Number of labels ({len(labels)}) does not match y data columns ({y.shape[1]})."
                    )
            else:
                self._y_labels = labels.copy()  # Allow setting labels if y is None.
        elif labels is None:
            self._y_labels = None
        else:
            raise ValueError(
                f"Provided `labels` {labels} is not a list of only strings."
            )

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
            The label of the x data. Defaults to "eV" if no unit is defined.
        """
        if self._x_unit is None:
            return "eV"
        return self._x_unit

    @x_unit.setter
    def x_unit(self, unit: str | None) -> None:
        if isinstance(unit, str):
            self._x_unit = unit
        elif unit is None:
            self._x_unit = None
        else:
            raise ValueError(f"Provided `unit` {unit} is not a string.")

    @property
    def y_units(self) -> list[str | None] | None:
        """
        Property for the units of the y data.

        Parameters
        ----------
        units : list[str | None] | None
            A list of strings to replace existing y units. Must match number of columns in the y attribute.

        Returns
        -------
        list[str | None] | None
            A direct reference to the y units list. If y data has been redefined
            without redefining y_units, y units are generated in the form 'Data Col. N'.
        """
        return self._y_units

    @y_units.setter
    def y_units(self, units: list[str | None] | None):
        if units is None:
            self._y_units = None
        elif isinstance(units, list) and np.all(
            [isinstance(unit, str | None) for unit in units]
        ):
            y = self.y
            if y is not None:
                if len(units) == y.shape[1]:
                    self._y_units = units.copy()
                else:
                    raise ValueError(
                        f"Shape of 'units' ({len(units)}) does not match shape of y data ({y.shape[1]})."
                    )
            else:
                self._y_units = units.copy()  # Allow setting units if y is None.
        else:
            raise ValueError(f"Provided 'units' {units} is not a list of strings.")

    def snapshot(self, columns: int | None = None) -> matplotlib.figure.Figure:
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

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        ValueError
            If Y data is not defined.
        """
        if not HAS_MPL:
            raise ImportError(
                "Matplotlib is not installed, cannot generate a snapshot graphic."
            )

        y = self.y
        if y is None:
            raise ValueError("Y data is not defined.")
        if columns is None:
            columns = int(np.ceil(np.sqrt(y.shape[1])))
        rows = int(np.ceil(y.shape[1] / columns))
        fig, ax = plt.subplots(rows, columns, sharex=True, figsize=(10, 10), dpi=300)  # type: ignore - Covered by HAS_MPL check.

        # Ensure axis array is 2D.
        if not isinstance(ax, np.ndarray):
            ax = np.array([[ax]])
        elif ax.ndim == 1:
            ax = np.array([ax])

        # Plot the data.
        ylabels = self.y_labels
        yunits = self.y_units
        for i, ydata in enumerate(y.T):
            r, c = divmod(i, columns)
            ax[r][c].plot(self.x, ydata)
            if ylabels is not None:
                ax[r][c].set_title(
                    ylabels[i] if yunits is None else f"{ylabels[i]} ({yunits[i]})"
                )
        for i in range(columns):
            ax[rows - 1][i].set_xlabel(self.x_label)
        return fig

    def to_csv(
        self,
        filename: str | None = None,
        delim: str = ",",
        combine_label_unit: bool = True,
    ) -> None | str:
        r"""
        Save the processed data as a CSV (comma-separated values) file.

        TODO: Currently doesn't save errors.

        Parameters
        ----------
        filename : str | None
            The name of the file to save the data to.
            If None, returns the CSV buffer as a string.
        delim : str, optional
            The delimiter to use in the csv file, by default ",".
            Another common option is to use "\t" for tab delimited.
        combine_label_unit : bool, optional
            If True, combines the label and unit in the header, by default True.
            Otherwise label and unit are separate rows.

        Returns
        -------
        None | str
            If filename is `None`, returns the CSV buffer as a string.

        Raises
        ------
        IOError
            If the specified file already exists.
        """
        x = self.x
        if x is None:
            raise ValueError("X data is not defined.")
        y = self.y
        if y is None:
            raise ValueError("Y data is not defined.")

        if filename is None:
            csv_file = io.StringIO()
        else:
            if os.path.exists(filename):
                raise OSError(f"The following filepath already exists:\n{filename}")
            csv_file = open(filename, "w")
        # Write the header
        y_labels = self.y_labels
        y_units = self.y_units
        if combine_label_unit:
            if y_labels is not None and y_units is not None:
                combo = []
                for label, unit in zip(y_labels, y_units):
                    if label is not None and unit is not None:
                        combo.append(f"{label} ({unit})")
                    elif label is not None:
                        combo.append(label)
                    elif unit is not None:
                        combo.append(f"({unit})")
                    else:
                        combo.append("")
                csv_file.write(
                    f"{self.x_label} ({self.x_unit}){delim}{delim.join(combo)}\n"
                )
            elif y_labels is not None:
                combo = [label if label is not None else "" for label in y_labels]
                csv_file.write(
                    f"{self.x_label} ({self.x_unit}){delim}{delim.join(combo)}\n"
                )
            else:
                csv_file.write(f"{self.x_label}{delim}Y Data\n")
        else:
            # Write labels, then units.
            if y_labels is not None:
                combo = [label if label is not None else "" for label in y_labels]
                csv_file.write(f"{self.x_label}{delim}{delim.join(combo)}\n")
            if y_units is not None:
                combo = [unit if unit is not None else "" for unit in y_units]
                csv_file.write(f"{self.x_unit}{delim}{delim.join(combo)}\n")
        # Write the data
        csv_file.writelines(
            f"{x[i]}{delim}{delim.join(str(y) for y in y[i])}\n" for i in range(len(x))
        )

        if filename is None:
            csv_file.seek(0)  # Reset the buffer to the beginning
            return csv_file.read()
        else:
            csv_file.close()

    @abc.abstractmethod
    def reload_labels_from_parser(self) -> None:
        """
        Re-load the labels and units from the parser object.

        This method is abstract and must be implemented in subclasses.
        Useful when the user wants to switch from the raw parameter names to useful names.
        Alternatively scan labels can be manually set.
        """

    def __getitem__(self, key: str | dtype) -> npt.NDArray:
        """
        Get a specific channel of data by its dtype name.

        Parameters
        ----------
        key : str | dtype
            The name or dtype of the channel to retrieve.

        Returns
        -------
        npt.NDArray
            The data corresponding to the requested channel.

        Raises
        ------
        ValueError
            If Y data is not defined, cannot get channel data.
        ValueError
            If Y labels are not defined, cannot get channel index.
        """
        y = self.y
        if y is None:
            raise ValueError(f"Y data is not defined, cannot get `{key}` data.")
        y_labels = self.y_labels
        if y_labels is None:
            raise ValueError(f"Y labels are not defined, cannot get `{key}` index.")
        # Attempt to get the channel from raw/relabelled names.
        idx = y_labels.index(key)
        return y[:, idx]


class parsedScanAbstract(scanAbstract):
    """
    Abstract class for defining properties of a scan object that is linked to a parser object.

    This class inherits from `scanAbstract` and adds properties and methods related to the parser object.

    Parameters
    ----------
    parser : parserBase | None
        A parser object that contains the raw data and metadata.
        Parser data is loaded, and can be re-loaded.
        If None, the scan object is empty and must be loaded from a parser object.
    """

    def __init__(
        self,
        parser: parserBase | None = None,
    ) -> None:
        super().__init__()
        self._parser = parser
        self._parser_class: type[parserBase] | None = (
            parser.__class__ if parser is not None else None
        )

        self._filepath = parser.filepath if parser is not None else None
        self._mtime = parser.mtime if parser is not None else None
        self._ctime = parser.ctime if parser is not None else None

        self._available_channels: list[dtype] = []
        """A list of available dtype channels that have been loaded from the parser.
        Populated when loading data from the parser. See `_base.parserBase.to_scan()` for more details."""

    @override
    def __getitem__(self, key: str | dtype) -> npt.NDArray:
        """
        Get a specific channel of data by its dtype name.

        Parameters
        ----------
        key : str | dtype
            The name or dtype of the channel to retrieve.

        Returns
        -------
        npt.NDArray
            The data corresponding to the requested channel.

        Raises
        ------
        ValueError
            If the parser object is not defined, cannot get channel data.
        ValueError
            If the requested channel is not found in the labels.
        """
        y = self.y
        if y is None:
            raise ValueError("Y data is not defined, cannot get channel data.")
        y_labels = self.y_labels
        if y_labels is None:
            raise ValueError("Y labels are not defined, cannot get channel data.")
        # Attempt to get the channel from raw/relabelled names.
        idx = None
        try:
            idx = y_labels.index(key)
        except ValueError:
            # Check x_label
            x_label = self.x_label
            if x_label is not None and x_label == key:
                x = self.x
                if x is None:
                    raise ValueError("X data is not defined, cannot get x_label data.")
                return x
            # Key not found.
            parser_cls = self.parser_class
            if parser_cls is None:
                raise ValueError(
                    "Parser class is not defined, cannot get RELABEL data."
                )
            # Check relabelled version of the x-label.
            if x_label is not None and (
                x_label in parser_cls.RELABELS
                and key in parser_cls.RELABELS
                and parser_cls.RELABELS[x_label] == parser_cls.RELABELS[key]
            ):
                x = self.x
                if x is None:
                    raise ValueError("X data is not defined, cannot get x_label data.")
                return x
            try:
                idx = parser_cls._label_index(
                    y_labels, key, True
                )  # Check relabelled names as well.
            except ValueError as e:
                if "not found in labels" in str(e):
                    raise KeyError(str(e)) from e
        if idx is None:
            raise ValueError(f"Requested channel '{key}' not found in labels.")
        # Use the idx to get the corresponding data column.
        return y[:, idx]

    @property
    def parser(self) -> parserBase | None:
        """
        Property for the parser object linked to the scan object.

        Parameters
        ----------
        parser : parserBase | None
            A parser object that contains the raw data and metadata.
            Parser data is loaded, and can be re-loaded.
            If None, the scan object is empty and must be loaded from a parser object.

        Returns
        -------
         | parserBase
            The parser object linked to the scan object.
        """
        return self._parser

    @parser.setter
    def parser(self, parser: parserBase | None) -> None:
        self._parser = parser
        if parser is not None:
            self._filename = parser.filename
            self._filepath = parser.filepath
            self._mtime = parser.mtime
            self._ctime = parser.ctime
            self._parser_class = parser.__class__

    @parser.deleter
    def parser(self) -> None:
        self._parser = None

    @property
    def parser_class(self) -> type[parserBase]:
        """
        Property for the class of the parser object linked to the scan object.

        Returns
        -------
        type[parserBase]
            The class of the parser object linked to the scan object.

        Raises
        ------
        ValueError
            If no parser class has been linked to the scan object.
        """
        if self._parser_class is None:
            raise ValueError("No parser class has been linked to this scan object.")
        return self._parser_class

    def detach_parser(self) -> None:
        """
        Remove the reference to the parser, to reduce memory usage.

        Allows for garbage collector to remove the parser object.
        Especially useful if parsed a large file that has been reduced.
        """
        del self.parser

    @property
    def channels(self) -> list[dtype]:
        """
        A list of the available `dtype` channels available on the object.

        Returns
        -------
        list[dtype]
            A list of the enumerate datatypes.

        Examples
        --------
        These names can be used to access data on the object directly, e.g. :
        >>> scan.channels  #
        ['I0', 'TEY', 'PFY']
        >>> scan['I0']  # returns the data corresponding to the 'I0' channel.
        >>> scan.I0 # returns the data corresponding to the 'I0' channel.
        """
        return self._available_channels.copy()

    def label_index(self, label: str | dtype) -> int:
        """
        Get the index of a y-channel based on its label or dtype name.

        Parameters
        ----------
        label : str | dtype
            The label or dtype name of the channel to find.

        Returns
        -------
        int
            The index of the channel corresponding to the provided label or dtype name.

        Raises
        ------
        ValueError
            If the provided label is not found in the y_labels or available channels.
        """
        y_labels = self.y_labels
        if y_labels is None:
            raise ValueError("Y labels are not defined, cannot get channel index.")
        # Attempt to get the channel from raw/relabelled names.
        idx = None
        try:
            idx = y_labels.index(label)
        except ValueError:
            # Key not found.
            parser_cls = self.parser_class
            if parser_cls is None:
                raise ValueError(
                    "Parser class is not defined, cannot get RELABEL data."
                )
            # Check if the label is the x_label.
            x_label = self.x_label
            if (
                x_label is not None
                and x_label == label
                or x_label is not None
                and (
                    x_label in parser_cls.RELABELS
                    and label in parser_cls.RELABELS
                    and parser_cls.RELABELS[x_label] == parser_cls.RELABELS[label]
                )
            ):
                raise ValueError(
                    f"Requested label '{label}' corresponds to x_label, not y_labels."
                )
            try:
                idx = parser_cls._label_index(
                    y_labels, label, True
                )  # Check relabelled names as well.
            except ValueError as e:
                if "not found in labels" in str(e):
                    raise ValueError(str(e)) from e
        if idx is None:
            raise ValueError(f"Requested channel '{label}' not found in labels.")
        return idx

    @property
    def filename(self) -> str | None:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str | None
            The filename of the scan object.
            None if not defined.
        """
        return os.path.basename(self._filepath) if self._filepath is not None else None

    @property
    def filepath(self) -> str | None:
        """
        Property for the full filepath of the scan object.

        Returns
        -------
        str | None
            The full filepath of the scan object.
            None if not defined.

        Raises
        ------
        ValueError
            If the filepath is not defined.
        """
        return self._filepath

    @property
    def ctime(self) -> datetime.datetime | None:
        """
        Return the creation time of the file as a datetime object.

        Returns
        -------
        datetime.datetime | None
            The creation time of the file.
            None if not defined.
        """
        return self._ctime

    @property
    def mtime(self) -> datetime.datetime | None:
        """
        The modification time of the file as a datetime object.

        Returns
        -------
        datetime.datetime | None
            The modification time of the file.
            None if not defined.
        """
        return self._mtime


class scanSimple(scanAbstract):
    """
    Basic interface class for raw data that is not bundled in a parser object.

    This class is used for simple scans where data is provided directly as x and y arrays.

    Parameters
    ----------
    x : npt.NDArray | None
        1D array of x data (e.g., beam energy).
    y : npt.NDArray | None
        2D array of y data (e.g., multiple Y channels). First index is data points, second index is channels.
    x_errs : npt.NDArray | None, optional
        1D array of errors corresponding to x data, by default None.
    y_errs : npt.NDArray | None, optional
        2D array of errors corresponding to y data, by default None. Must match shape of y.
    x_label : str | dtype | None, optional
        Label for the x data, by default None.
    x_unit : str | None, optional
        Unit for the x data, by default None.
    y_labels : list[str | dtype | None] | str | dtype | None, optional
        List of labels for each y channel, by default None. Length must match number of y channels.
    y_units : list[str | None] | None, optional
        List of units for each y channel, by default None. Length must match number of y channels.
    *args, **kwargs
        Additional arguments and keyword arguments to pass to the parent class.
    """

    def __init__(
        self,
        x: npt.NDArray | None,
        y: npt.NDArray | None,
        x_errs: npt.NDArray | None = None,
        y_errs: npt.NDArray | None = None,
        x_label: str | dtype | None = None,
        x_unit: str | None = None,
        y_labels: list[str | dtype | None] | str | dtype | None = None,
        y_units: list[str | None] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        x = np.array(x) if x is not None else None  # make unique
        y = np.array(y) if y is not None else None
        # Validate x and y data, and errors
        if x is not None:
            if not isinstance(x, np.ndarray) or len(x.shape) != 1:
                raise ValueError("x must be a 1D numpy array.")
        if y is not None:
            if not isinstance(y, np.ndarray) or len(y.shape) > 2:
                raise ValueError("y must be a 1D or 2D numpy array.")
            if len(y.shape) == 1:
                y = y[:, np.newaxis]
        if x_errs is not None:
            x_errs = np.array(x_errs)
            if not isinstance(x_errs, np.ndarray) or x_errs.shape != x.shape:
                raise ValueError(
                    "x_errs must be a numpy array with the same shape as x."
                )
        if y_errs is not None:
            y_errs = np.array(y_errs)
            if len(y_errs.shape) == 1:
                y_errs = y_errs[:, np.newaxis]
            if not isinstance(y_errs, np.ndarray) or y_errs.shape != y.shape:
                raise ValueError(
                    "y_errs must be a numpy array with the same shape as y."
                )
        # Assign data
        self._x = x
        self._y = y
        self._x_errs = x_errs
        self._y_errs = y_errs
        self._x_label = x_label
        self._x_unit = x_unit
        if y_labels is not None:
            # Ensure length matches number of y channels.
            if y is not None and len(y_labels) != y.shape[1]:
                raise ValueError("Length of y_labels must match number of y channels.")
            self._y_labels = y_labels.copy()
        else:
            self._y_labels = None
        if y_units is not None:
            # Ensure length matches number of y channels.
            if y is not None and len(y_units) != y.shape[1]:
                raise ValueError("Length of y_units must match number of y channels.")
            self._y_units = y_units.copy()
        else:
            self._y_units = None

    def reload_labels_from_parser(self) -> None:
        """
        Implement the abstract method from scanAbstract.

        This method is not applicable for scanSimple as it does not use a parser object.
        It is included to satisfy the abstract base class interface.
        """
        return


class scanBase(parsedScanAbstract):
    """
    Base class for synchrotron measurements that scans across photon beam energies (eV).

    Links to a `parser` instance that contains `parser.data` and `parser.params`.
    The `parser.COLUMN_ASSIGNMENTS` property must reflect the correct column assignments to x/y data.
    This allows for multiple Y channels, reflecting various collectors that can be used in the beamline for absorption data.

    Parameters
    ----------
    parser : Type[parserBase] | parserBase | None
        A parser object that contains the raw data and metadata.
        Parser data is loaded, and can be re-loaded.
        If None, the scan object is empty and must be loaded from a parser object.
    load_all_columns : bool, optional
        If True, load all columns from the parser object.
        If False, only loads the columns defined in `parser.COLUMN_ASSIGNMENTS`.
        Default is False, as this is more memory efficient and typically only the
        x/y columns are needed for analysis.

    See Also
    --------
    parserBase : Base class for synchrotron data parsers.
    """

    @override
    def __init__(
        self,
        parser: parserBase | None,
        load_all_columns: bool = False,
    ) -> None:
        # Initialise data arrays, including x, y, etc.
        super().__init__(parser)

        # Store parser reference.
        self._all_columns_loaded = (
            False  # internal tracking on whether all columns have been loaded or not.
        )

        # Load data from parser
        if parser is not None:
            parser.to_scan(load_all_columns=load_all_columns, scan_obj=self)
            self._all_columns_loaded = load_all_columns

    @override
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
        newobj = self.__class__(parser=None, *args, **kwargs)
        # Copy the base scan attributes:
        newobj._parser = self._parser  # Keep the same parser reference.
        newobj._parser_class = self._parser_class
        newobj._all_columns_loaded = self._all_columns_loaded
        # Invoke the parent class copy method to copy the data attributes.
        super().copy(newobj)
        # Return the copy
        return newobj

    def reload(self, load_all_columns: bool | None = None) -> None:
        """
        Reload all data from the parser object.

        If `load_all_columns` is provided, it will override the current attribute setting `_all_columns_loaded`
        otherwise uses the `_all_columns_loaded` attribute to determine to load all columns or only those
        defined in `COLUMN_ASSIGNMENTS`, which is defaulted to `false`.
        Also reloads the x, y, y_errs, x_errs, labels and units from the parser object.

        Parameters
        ----------
        load_all_columns : bool, optional
            If True/False, updates the `_all_columns_loaded` attribute.
            If True, load all columns from the parser object.
            If False, only loads the columns defined in `parser.COLUMN_ASSIGNMENTS`.
            If None, uses the current value of `_all_columns_loaded`.

        Raises
        ------
        ValueError
            If the parser object is not defined, cannot reload data.
        """
        parser = self.parser
        if parser is None:
            raise ValueError("Parser object is not defined, cannot reload data.")
        # Run the parser-scan conversion method, applying to self.
        parser.to_scan(
            load_all_columns=self._all_columns_loaded
            if load_all_columns is None
            else load_all_columns,
            scan_obj=self,
        )

    def reload_labels_from_parser(self) -> None:
        """
        Re-load the labels and units from the parser object.

        Useful when the user wants to switch from the raw parameter names to useful names.
        Alternatively scan labels can be manually set.
        """
        parser = self.parser
        if parser is None:
            raise ValueError("Parser object is not defined, cannot reload data.")
        parser.to_scan(
            load_all_columns=self._all_columns_loaded, scan_obj=None, only_labels=True
        )
