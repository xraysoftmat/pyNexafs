"""
Table model and widget for the file display.
"""

import re
import traceback
import overrides
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QWidget,
    QTableView,
    QAbstractItemView,
    QProgressBar,
    QApplication,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
    QSortFilterProxyModel,
    QRegularExpression,
    QAbstractTableModel,
    QThreadPool,
    QThread,
)
import os
from pyNexafs.parsers import parser_base
from typing import Any
import warnings
import numpy as np
from datetime import datetime as dt
from pyNexafs.utils.sizes import btyes_to_human_readable
from enum import Enum


# Construct the Table Model
class tableModel(QAbstractTableModel):
    """
    Implements the QAbstractTableModel for a fileviewer

    Implements headersData to display provided headers,
    and the Adds two icons for load status (error, success)
    """

    class loadStatus(Enum):
        """
        The enumerate class for the load status of the file.
        """

        ERROR = 0
        HEADER_ONLY = 1
        FULL_DATA = 2

    _status_index = 0  # index for loaded status column.
    _status_header = ""  # string header for the loaded status column.

    def __init__(self, data, header=None):
        super(tableModel, self).__init__()
        self._data = data
        """A list of lists containing the data to be displayed in the table."""
        self._header = header

        # Initalise graphics for loaded / unloaded files.
        self._icon_error = QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogCancelButton
        )
        """Icon for file load error status."""

        self._icon_header_only = QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_FileIcon
        )
        """Icon for file load header status."""

        self._icon_full_data = QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogApplyButton
        )
        """Icon for file load success status."""

    @overrides.overrides
    def headerData(self, section, orientation, role):
        if self._header is None:
            return None
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return self._header[section] if len(self._header) > section else None
        else:
            return super().headerData(section, orientation, role)

    @overrides.overrides
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        # Check that second column is being indexed for "loaded" status, to display icons.
        if (
            self._header is not None
            and len(self._header) > self._status_index
            and self._header[self._status_index] == self._status_header
            and index.column() == self._status_index  # Is the status column.
        ):
            if role == Qt.ItemDataRole.DecorationRole:
                # Icons
                dat = self._data[index.row()][index.column()]
                if isinstance(dat, int):
                    return dat

                dat = self.loadStatus(dat)  # Convert to enum.
                match dat:
                    case self.loadStatus.ERROR:
                        val = self._icon_error
                    case self.loadStatus.HEADER_ONLY:
                        val = self._icon_header_only
                    case self.loadStatus.FULL_DATA:
                        val = self._icon_full_data
                    case _:
                        raise ValueError(f"Incorrect data match for {dat}")
                # val = (
                #     self._icon_full_data
                #     if dat
                #     else self._icon_error
                # )
                return val
            else:
                # val = "T" if self._data[index.row()][index.column()] else "F")
                return None

        # General data accessing
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]
        return None

    @overrides.overrides
    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    @overrides.overrides
    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        if (
            hasattr(self._data, "__len__")
            and len(self._data) > 0
            and hasattr(self._data[0], "__len__")
        ):
            return len(self._data[0])
        else:
            return 0


class directoryViewerTable(QTableView):
    """
    A table widget that displays pyNexafs related parser files.

    Attributes:
    selection_loaded : pyqtSignal
        Emits after a new selection is made.
    __default_headers_list : list
        A
    """

    selection_loaded = pyqtSignal(bool)
    """A signal for emitting after selection change is loaded."""

    __default_headers_list: list[str] = [
        "#",
        "Filename",
    ]  # index and filename #TODO: Add "Created", "Modified" when implemented in parser_base.

    def __init__(
        self,
        init_dir: str | None = None,
        progress_bar: QProgressBar | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        # Instance properties
        self.files: list[str] = []
        """The list of parser filenames"""
        self._header_names: list[str] | None = None
        """Header names populated by the parser class `parser.SUMMARY_PARAM_NAMES`"""
        self._header_names_custom: list[str] | None = None
        """A custom set of parameters that can be set by setting the `headers` property.
        Overrides use of `directoryViewerTable._header_names`."""

        ## WE KEEP A DICT OF PARSERS, AND THEN USE THE PARSERS TO GENERATE THE TABLE.
        self._parser_headers: dict[str, parser_base] = {}
        self._parser_files: dict[str, parser_base] = {}

        self._viewing_directory: str | None = None
        self._parser: parser_base | None = None
        self._filetype_filters: list[str] | None = None
        self._str_filter: str | None = None

        # Instance Widgets
        self.files_model = None
        self.proxy_model = QSortFilterProxyModel()

        # Instance Attributes
        self.verticalHeader().hide()  # remove default row numbers
        self.setModel(self.proxy_model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSortingEnabled(True)  # enabled by proxymodel.
        self.proxy_model.setSourceModel(None)
        self.proxy_model.setFilterRegularExpression(None)
        self.proxy_model.setFilterKeyColumn(
            -1
        )  # use all columns for filtering instead of just first.
        self.proxy_model.sort(0, Qt.SortOrder.AscendingOrder)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.resizeColumnToContents(0)
        self.setMinimumWidth(1)
        self.horizontalHeader().setMinimumSectionSize(10)

        # Event connections
        self.selectionModel().selectionChanged.connect(self.load_selection)

        # Progress bar
        self._progress_bar = progress_bar

        # Initialise viewing directory
        if init_dir is not None:
            self.directory = init_dir

        # Setup default header columns if parser is defined. Use header/index definition in table_model.
        self.__default_parsed_headers_list = self.__default_headers_list.copy()
        self.__default_parsed_headers_list.insert(
            self._status_index(),
            self._status_header(),  # At 1st index, insert ""
        )

    @classmethod
    def _status_index(cls) -> int:
        """
        Returns the index of the loaded status column, an attribute of the abstract table model class.
        """
        return tableModel._status_index

    @classmethod
    def _status_header(cls) -> str:
        """
        Returns the string header of the loaded status column, an attribute of the abstract table model class.
        """
        return tableModel._status_header

    @property
    def _default_headers(self) -> list[str]:
        """
        Returns a copy of the default headers depending on if a parser is defined (adds a loading column).

        Returns
        -------
        list[str]
            List of the default headers
        """
        obj = (
            self.__default_headers_list
            if self.parser is None
            else self.__default_parsed_headers_list
        )
        return obj.copy()

    @property
    def header(self) -> list[str] | None:
        """
        Returns the current header list for selected parsed.

        Parameters
        ----------
        val : list[str] | None
            Sets a custom list of headers to use for the table summary.

        Returns
        -------
        list[str]
            A list of column names currently observed by the parser.
        """
        return (
            self._header_names_custom
            if self._header_names_custom is not None
            else self._header_names
        )

    @header.setter
    def header(self, val: list[str] | None):
        if val != self._header_names_custom:
            if val is None:
                self._header_names_custom = None
            else:
                self._header_names_custom = self._default_headers + val

            # Update the table with the new headers.
            self.update_header()

    @header.deleter
    def header(self):
        self._header_names_custom = None

    @property
    def parser(self) -> type[parser_base] | None:
        """
        Returns the currently selected parser.

        Returns
        -------
        type[parser_base] | None
            The currently selected parser, or None if no parser is selected.
        """
        return self._parser

    @parser.setter
    def parser(self, parser: type[parser_base] | None):
        """
        Set the parser, used for tabling header information.

        Resets the directory_viewer_table._file_params and directory_viewer_table.filedata dictionaries.

        Parameters
        ----------
        parser : type[parser_base] | None
            The currently selected parser, or None if no parser is selected.
        """
        self._parser = parser
        # Reset existing file data to use new parser.
        self._parser_files = {}
        self._parser_headers = {}
        self._filetype_filters = (
            parser.ALLOWED_EXTENSIONS if parser is not None else None
        )

        self.update_table()

    @parser.deleter
    def parser(self):
        """
        Remove the parser from the directory viewer.
        """
        self._parser = None
        self._parser_files = {}
        self._parser_headers = {}
        self.update_table()

    @property
    def directory(self):
        """
        Returns the currently viewed directory.

        Returns
        -------
        str
            The currently viewed directory.
        """
        return self._viewing_directory

    @directory.setter
    def directory(self, target: str):
        """
        Set the directory to view.

        Only updates the files list if directory is changed.
        """

        # Check if same directory.
        if self._viewing_directory == target:
            return

        # Get new file list
        self.files = [
            file
            for file in os.listdir(target)
            if os.path.isfile(os.path.join(target, file))
        ]
        # Clear existing data
        self._parser_headers = {}
        self._parser_files = {}

        # Update directory
        self._viewing_directory = target

        # Push update to table.
        self.update_table()

    @property
    def filters(self) -> tuple[str | None, list[str] | None]:
        """
        Returns the currently set filters.

        Returns
        -------
        tuple[str | None, list[str] | None]
            A tuple containing the filename filter and the filetype filter.
        """
        return (self._str_filter, self._filetype_filters)

    @filters.setter
    def filters(self, filters: tuple[str | None, list[str] | None]):
        """
        Set the filters for the directory viewer.

        Parameters
        ----------
        filters : tuple[str | None, list[str] | None]
            A tuple containing the filename filter and the filetype list filter.
        """
        str_filter, filetype_filters = filters

        # Correct type for empty strings.
        if str_filter == "":
            str_filter = None

        # Check if str filter is changed:
        if str_filter != self._str_filter:
            self._str_filter = str_filter
            # Update proxy model
            QRegExp = QRegularExpression(
                str_filter, QRegularExpression.PatternOption.CaseInsensitiveOption
            )
            self.proxy_model.setFilterRegularExpression(
                None if str_filter is None else QRegExp
            )
            # self.proxy_model.setFilterFixedString(str_filter)

        # Check if filetype filter is changed:
        if filetype_filters is not None and isinstance(filetype_filters, list):
            # Check if filetype filter is changed:
            if (
                self._filetype_filters is not None
                and len(self._filetype_filters) == len(filetype_filters)
                and np.all(
                    [
                        (
                            filetype_filters[i] == self._filetype_filters[i]
                            or filetype_filters[i] in self._filetype_filters
                        )
                        for i in range(len(filetype_filters))
                    ]
                )
            ):
                # Everything the same...
                pass
            else:
                self._filetype_filters = filetype_filters
                self.update_table()
        elif filetype_filters is None:
            self._filetype_filters = (
                self._parser.ALLOWED_EXTENSIONS if self._parser is not None else None
            )
        else:
            raise AttributeError(
                f"filters[1] '{filetype_filters}' is not a list of filetype strings or None."
            )

    @filters.deleter
    def filters(self):
        """
        Remove the filters for the directory viewer.
        """
        self._str_filter, self._filetype_filters = None, None
        self.update_table()

    @property
    def progress_bar(self) -> QProgressBar:
        """
        Returns an associated progress bar widget if provided, else None.

        Parameters
        ----------
        progress_bar: QProgressBar
            The progress bar widget.

        Returns
        -------
        QProgressBar
            The progress bar widget.
        """
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar: QProgressBar):
        self._progress_bar = progress_bar

    def update_header(self) -> None:
        """
        Generates the column headers using default parser important parameters.

        First attempts to call `summary_param_names_with_units` on the first non-None
        header-loaded parser object, to load the column labels and their appropriate units.
        Alternatively calls `summary_param_names` on the object, which doesn't require unit labels.
        These two methods take into account the parser `relabel` property.
        If no headers are loaded then calls the parser `summary_param_names`.
        A change in header will also trigger a change in the table model.
        """
        # Setup default header columns
        header = self._default_headers  # Get defaults, i.e. (loaded), number and name.

        # Add list values if parser is defined, and header objects allow unit calls.
        if self.parser is not None and len(self._parser_headers) > 0:
            val = None
            # Collect the first non-None parser object.
            for val in self._parser_headers.values():
                # Collect val (parser objects), where val is not None.
                if val is not None:
                    assert isinstance(val, self.parser)
                    break  # end for loop prematurely.
            # Establish parameter header names:
            if val is not None:
                try:
                    # Attempt to use 'summary_param_names_with_units' if available.
                    obj_head = val.summary_param_names_with_units
                except NotImplementedError:
                    warnings.warn(
                        f"{self.parser.__name__} has not implemented 'summary_param_names_with_units'. Defaulting to 'summary_param_names'."
                    )
                    obj_head = val.summary_param_names  # without units.
            else:
                # If all parser headers are None, then use parser default values.
                obj_head = self.parser.summary_param_names
            # Add obj_head to header
            header += [s[0] if isinstance(s, tuple) else s for s in obj_head]
        # If _parser_headers are not defined but the parser is.
        elif self.parser is not None:
            header += [
                s[0] if isinstance(s, tuple) else s
                for s in self.parser.summary_param_names
            ]

        # Update internal header names
        self._header_names = header

        # Ideally just updates header names, but occasionally crashes when attempting to layoutChange.emit()
        if self.files_model is not None:
            old_selection = self.selectionModel().selectedRows()
            new_model = tableModel(
                data=self.files_model._data,
                header=(
                    self._header_names
                    if self._header_names_custom is None
                    else self._header_names_custom
                ),
            )
            self.proxy_model.setSourceModel(new_model)
            self.files_model = new_model
            # Setup and restore selection: requires multiselection to select each row before using regular extended selection.
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            [self.selectRow(i.row()) for i in old_selection]
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Update column widths with header update.
        for i in range(0, len(self._header_names)):
            # If the current column width is larger than the size hint, resize to fit for all columns.
            if self.columnWidth(i) > self.sizeHintForColumn(i):
                self.resizeColumnToContents(i)
        return

    def update_table(self):
        """
        Update the directory view.

        Applies text filter, parser and permissionable extensions to filter the files list.
        Loads the filtered header data for the table, appling updates to the class progress bar if defined.

        """
        # Do nothing if an empty directory.
        if self.directory is None:
            return

        ### COLLECT FILES AND APPLY FILETYPE FILTER
        # Collect available files
        files = self.files
        # 1: Filetypes
        if self._filetype_filters is not None and len(self._filetype_filters) > 0:
            files = [
                file
                for file in self.files
                if file.endswith(tuple(self._filetype_filters))
            ]
        # Set the progress bar to zero.
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
            # Store number of files to be processed for progress bar
            self.progress_bar.setRange(0, len(files) if len(files) > 0 else 1)
        # 2: Load new fileheader data.
        if self.parser is not None:
            # Get header data
            with warnings.catch_warnings(record=True) as recorded_warnings:
                for i, file in enumerate(files):
                    if file not in self._parser_headers:
                        try:
                            self._parser_headers[file] = self.parser(
                                os.path.join(self.directory, file), header_only=True
                            )
                        # Catch unimplemented and import errors.
                        except (NotImplementedError, ImportError, PermissionError):
                            self._parser_headers[file] = None
                    if self.progress_bar is not None:
                        # self.progress_bar.setValue(int((i + 1) / files_len * 100))
                        self.progress_bar.setValue(i + 1)
            # Process the caught warnings
            warning_categories = {}
            warn_re = re.compile(
                r"Attempted method '(.*)' failed to load '(.*)' from '(.*)' with (.*)."
            )
            for w in recorded_warnings:
                matches = warn_re.search(str(w.message))
                if matches and len(matches.groups()) == 4:
                    method, file, parser, error = matches.groups()
                    # Collect warning
                    parser_method = f"{parser}.{method}"
                    filetype = file.split(".")[-1]
                    filetype_error = f"{filetype}:{error}"

                    if parser_method not in warning_categories:
                        # Add parser_method to categories
                        warning_categories[parser_method] = {filetype_error: 1}
                    elif filetype_error not in warning_categories[parser_method]:
                        # Add filetype_error to parser_method
                        warning_categories[parser_method][filetype_error] = 1
                    else:
                        # Increment filetype_error count
                        warning_categories[parser_method][filetype_error] += 1
                else:
                    # re-issue warning
                    warnings.warn_explicit(
                        message=w.message,
                        category=w.category,
                        filename=w.filename,
                        lineno=w.lineno,
                        source=w.source,
                    )
            # Print warning categories
            for parser_method, file_errors in warning_categories.items():
                for filetype_error, count in file_errors.items():
                    filetype, error = filetype_error.split(":")
                    print(
                        f"Parser method '{parser_method}' failed {count} times on filetype '{filetype}' with error '{error}'."
                    )
        else:
            # If parser is None, empty the existing data.
            self._parser_headers = {}

        # Update headers
        self.update_header()

        # Get header data for table:
        data = []  # list of file header lists.
        excess = [
            "" for _ in range(len(self._header_names) - len(self._default_headers))
        ]  # empty columns for invalid file headers.
        for i, file in enumerate(files):
            filedata = [i + 1, file]  # add index (1, ...) and filename to data.
            # If parser specified (for loading) add field for successful loading.
            if self.parser:
                status = (
                    True
                    if file in self._parser_headers
                    and self._parser_headers[file] is not None
                    else False
                )
                filedata.insert(self._status_index(), status)

            # Raise an error if mismatch between info and header variables.
            if len(filedata) != len(self._default_headers):
                raise AttributeError("Default headers do not match filedata.")

            # Add parameter values
            if file in self._parser_headers and isinstance(
                self._parser_headers[file], parser_base
            ):
                # Match indexing of parser.SUMMARY_PARAM_RAW_NAMES
                filedata.extend(self._parser_headers[file].summary_param_values)
            else:
                filedata.extend(excess)
            # Add header data to data list.
            data.append(filedata)

        # New table - Ideally would be a model update, but would crash when attempting to layoutChange.emit()
        self.files_model = tableModel(data=data, header=self._header_names)
        # Update proxy (filtering / sorting) model
        self.proxy_model.setSourceModel(self.files_model)

        self.resizeColumnToContents(0)
        for i in range(0, len(self._header_names)):
            # If the current column width is larger than the size hint, resize to fit for all columns.
            if self.columnWidth(i) > self.sizeHintForColumn(i):
                self.resizeColumnToContents(i)
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
        return

    def load_selection(self) -> None:
        """
        Loads full data from files of the current selection.

        Instead of just header information, loads all data lines
        from the selected files. Loaded datafiles are cached in
        self._parser_files, until the parser object is changed.
        """
        sm = self.selectionModel()
        # Require selection and parser to trigger load data.
        if sm.hasSelection() and self.parser is not None:
            rows = sm.selectedRows(
                column=1 if self._status_index() > 1 else 2
            )  # get filename, not index.

            # Set the progress bar to zero.
            if self.progress_bar is not None:
                self.progress_bar.setValue(0)
                # Store number of files to be processed for progress bar
                self.progress_bar.setRange(0, len(rows))

            # Setup warnings to show once.
            warnings.simplefilter("once", UserWarning)
            for i, row in enumerate(rows):
                filename = self.proxy_model.data(row, Qt.ItemDataRole.DisplayRole)
                # Check if the file already loaded:
                if (
                    filename not in self._parser_files
                    or self._parser_files[filename] is None
                ):
                    # Check for a header object (should always exist).
                    if (
                        filename in self._parser_headers
                        and self._parser_headers[filename] is not None
                    ):
                        # Use existing parser instance, add to files.
                        parser = self._parser_headers[filename]
                        assert isinstance(parser, parser_base)
                        try:
                            parser.load()  # use internal filepath to load
                            if parser.is_loaded:
                                self._parser_files[filename] = parser
                        except Exception as e:
                            traceback.print_exception(e)
                    # else: ignore unloaded headers (None) in the selection.
                if self.progress_bar is not None:
                    self.progress_bar.setValue(i + 1)
            # Restore warnings functionality.
            warnings.resetwarnings()
            # Trigger a loading completed signal.
            self.progress_bar.setValue(0)
            self.selection_loaded.emit(True)
        return

    @property
    def selected_filenames(self) -> list[str]:
        """
        Returns the selected filenames in the directory viewer.

        Returns
        -------
        list[str]
            A list of filenames selected in the directory viewer.
        """
        sm = self.selectionModel()
        if sm.hasSelection():
            rows = sm.selectedRows(
                column=self._index_filename_column
            )  # Filename in second column.
            # return self.files_model.data(rows, Qt.ItemDataRole.DisplayRole)
            return [
                self.proxy_model.data(row, Qt.ItemDataRole.DisplayRole) for row in rows
            ]
        return []

    @property
    def _index_filename_column(self) -> int:
        """
        Returns the index of the filename column in the table model.
        """
        if self.parser is None:
            return 1
        else:
            return 2 if self._status_index() < 2 else 1


# TODO: Implement threading for file header/data loading
# class fileLoaderWorker(QtCore.QRunnable):
#     """
#     Class to load files in a separate thread.

#     Parameters
#     ----------
#     n : int
#         The thread number.
#     files : list[str]
#         The list of files to load.
#     parser : type[parser_base]
#         The parser class to use for loading the files.
#     parsers: dict[str, parser_base] | None
#         The dict of existing parsers to use for loading the files.
#         I.e. If a header has been loaded but not the full data, use
#         the same parser to load the data.
#     QProgressBar : QProgressBar
#         The progress bar to update during loading.
#     """
#     def __init__(self, n: int,
#                  files: list[str],
#                  parser: type[parser_base],
#                  parsers: dict[str, parser_base] | None = None,
#                  progress_bar: QProgressBar | None = None,
#                  load_kwargs: dict[str, Any] | None = None):
#         super().__init__()
#         self.n = n
#         self.files = files
#         self.parser = parser
#         self.load_kwargs = load_kwargs
#         self.progress_bar = progress_bar

#         # Store the parsers if provided.
#         self.parsers = parsers
#         """
#         The dict of provided parsers in the thread.
#         """

#         # Create storage for loaded parsers.
#         self.loaded: dict[str, parser_base] = {}
#         """
#         The dict of parsers loaded in the thread.
#         """

#     def run(self):
#         """
#         Load the files in a separate thread.

#         This method is called when the thread is started.
#         """
#         # Load the files using the parser class.
#         for i, file in enumerate(self.files):
#             if file not in self.loaded or not isinstance(self.loaded[file], self.parser):
#                 try:
#                     # Check if the file already has a parser (i.e. header).
#                     if self.parsers is not None and file in self.parsers:
#                         # Use the provided parser if it exists.
#                         parser = self.parsers[file]
#                         parser.load(**(self.load_kwargs
#                                        if self.load_kwargs is not None
#                                        else {}))
#                     else:
#                         parser = self.parser(file,
#                                             **(self.load_kwargs
#                                                if self.load_kwargs is not None
#                                                else {}))
#                     if self.progress_bar is not None:
#                         self.progress_bar.setValue(i + 1)
#                     # Add the loaded parser to the dictionary.
#                     self.loaded[file] = parser
#                 except Exception as e:
#                     traceback.print_exception(e)
#                     if self.progress_bar is not None:
#                         self.progress_bar.setValue(self.progress_bar.value() + 1)
#             else:
#                 # If the file is already loaded, just update the progress bar.
#                 if self.progress_bar is not None:
#                     self.progress_bar.setValue(self.progress_bar.value() + 1)

# class fileLoaderCounter(QObject):
#     """
#     Counts the completion of
#     """


class directoryViewerTableNew(QTableView):
    """
    A table widget that displays pyNexafs related parser files.

    This TableView operates on the following properties.
    - `parser`: This is the parser class that is used to (try to) load the files.
    - `directory`: This is the directory that is being viewed.
    - `filters`: This is a tuple of the filename filter and the filetype filter.
    - `progress_bar`: This is a progress bar that is used to show the progress of loading files.
    - `header` : The parser default param names (default), but can be a selected list of parameter names.
            Uses the `_header_param_names_default` and `_header_param_names_custom` attributes to determine
            the header names (which are set by ).

    Attributes:
    selection_loaded : pyqtSignal
        Emits after a new selection is made.
    __default_headers_list : list
        A

    See Also
    --------
    SummaryParamSelector
        A widget that allows the user to select the summary parameters to be displayed in the table.
    """

    selection_loaded = pyqtSignal(bool)
    """A signal for emitting after selection change is loaded."""

    __REQUIRED_HEADER_LIST: list[str] = ["#", "Filename"]
    """Headers always created for the table: file number and filename."""

    __DEFAULT_FILE_HEADERS: list[str] = ["created", "modified", "memory_size"]
    """When no parser selected, default file info headers."""

    __DEFAULT_PARSER_HEADER_KEYS: list[str] = ["created", "modified", "memory_size"]
    """Default parser parameter values selected upon parser selection, in addition to SUMMARY_PARAM_NAMES."""

    def __init__(
        self,
        init_dir: str | None = None,
        progress_bar: QProgressBar | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        # Instance properties
        self._files: list[str] = []
        """The list of directory filenames"""

        self._header_param_names_default: list[str] | None = None
        """Header names populated by `__DEFAULT_HEADER_KEYS` the parser class `parser.SUMMARY_PARAM_NAMES`"""
        self._header_param_names_custom: list[str] | None = None
        """A custom set of parameters that can be set by setting the `headers` property.
        Overrides use of `directoryViewerTable._header_names`."""

        ## WE KEEP A DICT OF PARSERS, AND THEN USE THE PARSERS TO GENERATE THE TABLE.
        self._parser_headers: dict[str, parser_base | None] = {}
        """The parsers who have been loaded with headers only."""
        self._parser_files: dict[str, parser_base | None] = {}
        """The parsers who have been loaded with full data."""

        self._viewing_directory: str | None = None
        """The internal directory that is being viewed."""
        self._parser: type[parser_base] | None = None
        """The selected parser class."""
        self._filetype_filters: list[str] | None = None
        """The list of filetypes that are allowed to be loaded."""
        self._str_filter: str | None = None
        """The string filter that is used to filter the filenames. Can also be QRegex"""

        # Instance Widgets
        self.files_model = None
        """The model for the table, before filtering."""
        self.proxy_model = QSortFilterProxyModel()
        """The filtered model for the table, using `self.files_model` as the source model."""

        # Threading

        # Instance Attributes
        self.verticalHeader().hide()  # remove default row numbers
        self.setModel(self.proxy_model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSortingEnabled(True)  # enabled by proxymodel.
        self.proxy_model.setSourceModel(None)
        self.proxy_model.setFilterRegularExpression(None)
        self.proxy_model.setFilterKeyColumn(
            -1
        )  # use all columns for filtering instead of just first.
        self.proxy_model.sort(0, Qt.SortOrder.AscendingOrder)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.resizeColumnToContents(0)
        self.setMinimumWidth(1)
        self.horizontalHeader().setMinimumSectionSize(10)

        # Event connections
        self.selectionModel().selectionChanged.connect(self.load_selection)

        # Progress bar
        self._progress_bar = progress_bar

        # Initialise viewing directory
        if init_dir is not None:
            self.directory = init_dir

        self.update_header()
        self.update_table()

    @property
    def files(self) -> list[str]:
        """
        Returns the list of parser files in the directory.

        Subsets the filelist to only include files that
        are within the `parser.ALLOWED_EXTENSIONS` and `self._filetype_filters`.

        Returns
        -------
        list[str]
            A list of filenames in the directory.
        """
        # Is a parser defined? If not, return all files.
        if self.parser is None:
            return self._files
        # Check if a parser filetype filter has been selected
        elif self._filetype_filters is not None and len(self._filetype_filters) > 0:
            return [
                file
                for file in self._files
                if file.endswith(tuple(self._filetype_filters))
            ]
        # Check if a parser is selected but no filetype filter is defined.
        # though usually this is not the case.
        elif len(self.parser.ALLOWED_EXTENSIONS) > 0:
            return [
                file
                for file in self._files
                if file.endswith(tuple(self.parser.ALLOWED_EXTENSIONS))
            ]
        else:
            return self._files

    @classmethod
    def _status_index(cls) -> int:
        """
        Returns the index of the loaded status column, an attribute of the abstract table model class.
        """
        return tableModel._status_index

    @classmethod
    def _status_header(cls) -> str:
        """
        Returns the string header of the loaded status column, an attribute of the abstract table model class.
        """
        return tableModel._status_header

    @property
    def header(self) -> list[str] | None:
        """
        Returns the current header list for selected parsed.

        Includes values from `__REQUIRED_HEADER_LIST`, if a parser is defined or not.

        Parameters
        ----------
        val : list[str] | None
            Sets a custom list of headers to use for the table summary.

        Returns
        -------
        list[str]
            A list of column names currently observed by the parser.
        """
        return (
            self._header_param_names_custom
            if self._header_param_names_custom is not None
            else self._header_param_names_default
        )

    @header.setter
    def header(self, val: list[str] | None):
        if val != self._header_param_names_custom:
            if val is None:
                self._header_param_names_custom = None
            else:
                self._header_param_names_custom = self.__REQUIRED_HEADER_LIST + [
                    v for v in val if v not in self.__REQUIRED_HEADER_LIST
                ]

            # Update the table with the new headers.
            self.update_header()

    @header.deleter
    def header(self):
        self._header_param_names_custom = None

    # @property
    # def header_optional(self) -> list[str] | None:
    #     """
    #     The optional selection for the current header list for selected parser.

    #     Excludes values from `__REQUIRED_HEADER_LIST`.

    #     Returns
    #     -------
    #     list[str]
    #         A list of column names currently observed by the parser.
    #     """
    #     h = (self._header_param_names_custom
    #             if self._header_param_names_custom is not None
    #             else self._header_param_names_default
    #     )
    #     return

    @property
    def parser(self) -> type[parser_base] | None:
        """
        Property for the selected parser.

        Parameters
        ----------
        parser : type[parser_base] | None
            A new parser class to use for loading files.
            The parser is used to set default extensions, and also
            resets the file data and header data and updates the table.

        Returns
        -------
        type[parser_base] | None
            The currently selected parser, or None if no parser is selected.
        """
        return self._parser

    @parser.setter
    def parser(self, parser: type[parser_base] | None):
        if self._parser != parser:
            self._parser = parser
            # Reset existing file data to use new parser.
            self._parser_files = {}
            self._parser_headers = {}
            self._filetype_filters = (
                parser.ALLOWED_EXTENSIONS if parser is not None else None
            )

            # Load the parser headers for the current directory.
            self._collect_parser_headers()

            # Generate a new header from the parser class.
            self._header_param_names_default = self.generate_header()
            print(self._parser, "New header: ", self._header_param_names_default)

            # Reset the custom header names.
            self._header_param_names_custom = None

            # Generate a new header from the loaded parsers and update headers.
            # self.header = self.generate_header()
            self.update_header()

            # Update the table with the new parser.
            self.update_table()

    @parser.deleter
    def parser(self):
        self._parser = None
        self._parser_files = {}
        self._parser_headers = {}
        self._header_param_names_default = None
        self._header_param_names_custom = None
        self.update_header()
        self.update_table()

    @property
    def directory(self):
        """
        Returns the currently viewed directory.

        Returns
        -------
        str
            The currently viewed directory.
        """
        return self._viewing_directory

    @directory.setter
    def directory(self, target: str):
        """
        Set the directory to view.

        Only updates the files list if directory is changed.
        """

        # Check if same directory.
        if self._viewing_directory == target:
            return

        # Get new file list
        self._files = [
            file
            for file in os.listdir(target)
            if os.path.isfile(os.path.join(target, file))
        ]
        # Clear existing data
        self._parser_headers = {}
        self._parser_files = {}

        # Update directory
        self._viewing_directory = target

        # Load the new parsers if a parser is defined.
        if self.parser is not None:
            self._collect_parser_headers()

        # Push update to table.
        self.update_table()

    def _collect_parser_headers(self) -> None:
        """
        Load the headers for the current directory if the
        parsers are not loaded already (allows refreshing).

        Updates `self._parser_headers` with the loaded parsers.
        Also catches warnings.

        Raises
        -------
        AttributeError
            If no parser is defined.
        """
        if self.parser is None:
            raise AttributeError("No parser defined. Cannot collect parser headers.")
        if self.directory is None:
            raise AttributeError("No directory defined. Cannot collect parser headers.")
        # Set the progress bar to zero.
        files = self.files
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
            # Store number of files to be processed for progress bar
            self.progress_bar.setRange(0, len(files) if len(files) > 0 else 1)
        with warnings.catch_warnings(record=True) as recorded_warnings:
            for i, file in enumerate(files):
                # Check file not loaded already, or parser is not current type.
                if file not in self._parser_headers or (
                    self.parser is not None
                    and not isinstance(self._parser_headers[file], self.parser)
                ):
                    # Attempt to load
                    try:
                        self._parser_headers[file] = self.parser(
                            os.path.join(self.directory, file), header_only=True
                        )
                    except (NotImplementedError, ImportError, PermissionError):
                        self._parser_headers[file] = None
                if self.progress_bar is not None:
                    # self.progress_bar.setValue(int((i + 1) / files_len * 100))
                    self.progress_bar.setValue(i + 1)
            # When finished
            if self.progress_bar is not None:
                self.progress_bar.setValue(0)
        # Process the caught warnings
        warning_categories = {}
        warn_re = re.compile(
            r"Attempted method '(.*)' failed to load '(.*)' from '(.*)' with (.*)."
        )
        for w in recorded_warnings:
            matches = warn_re.search(str(w.message))
            if matches and len(matches.groups()) == 4:
                method, file, parser, error = matches.groups()
                # Collect warning
                parser_method = f"{parser}.{method}"
                filetype = file.split(".")[-1]
                filetype_error = f"{filetype}:{error}"

                if parser_method not in warning_categories:
                    # Add parser_method to categories
                    warning_categories[parser_method] = {filetype_error: 1}
                elif filetype_error not in warning_categories[parser_method]:
                    # Add filetype_error to parser_method
                    warning_categories[parser_method][filetype_error] = 1
                else:
                    # Increment filetype_error count
                    warning_categories[parser_method][filetype_error] += 1
            else:
                # re-issue warning
                warnings.warn_explicit(
                    message=w.message,
                    category=w.category,
                    filename=w.filename,
                    lineno=w.lineno,
                    source=w.source,
                )
        # Print warning categories
        for parser_method, file_errors in warning_categories.items():
            for filetype_error, count in file_errors.items():
                filetype, error = filetype_error.split(":")
                print(
                    f"Parser method '{parser_method}' failed {count} times on filetype '{filetype}' with error '{error}'."
                )

    @property
    def filters(self) -> tuple[str | None, list[str] | None]:
        """
        The current filters (filename and filetype) for the directory viewer.

        The filename filter is a string that is used to filter the filenames in the directory.
        Regular expressions
        The

        Parameters
        ----------
        filters : tuple[str | None, list[str] | None]
            A tuple containing the filename filter and the filetype list filter.

        Returns
        -------
        tuple[str | None, list[str] | None]
            A tuple containing the filename filter and the filetype filter.
        """
        return (self._str_filter, self._filetype_filters)

    @filters.setter
    def filters(self, filters: tuple[str | None, list[str] | None]):
        str_filter, filetype_filters = filters

        # Correct type for empty strings.
        if str_filter == "":
            str_filter = None

        # Check if str filter is changed:
        if str_filter != self._str_filter:
            self._str_filter = str_filter
            # Update proxy model
            QRegExp = QRegularExpression(
                str_filter, QRegularExpression.PatternOption.CaseInsensitiveOption
            )
            self.proxy_model.setFilterRegularExpression(
                None if str_filter is None else QRegExp
            )
            # self.proxy_model.setFilterFixedString(str_filter)

        # Check if filetype filter is changed:
        if filetype_filters is not None and isinstance(filetype_filters, list):
            # Check if filetype filter is changed:
            if (
                self._filetype_filters is not None
                and len(self._filetype_filters) == len(filetype_filters)
                and np.all(
                    [
                        (
                            filetype_filters[i] == self._filetype_filters[i]
                            or filetype_filters[i] in self._filetype_filters
                        )
                        for i in range(len(filetype_filters))
                    ]
                )
            ):
                # Everything the same...
                pass
            else:
                self._filetype_filters = filetype_filters
                self.update_table()
        elif filetype_filters is None:
            self._filetype_filters = (
                self.parser.ALLOWED_EXTENSIONS if self.parser is not None else None
            )
        else:
            raise AttributeError(
                f"filters[1] '{filetype_filters}' is not a list of filetype strings or None."
            )

    @filters.deleter
    def filters(self):
        """
        Remove the filters for the directory viewer.
        """
        self._str_filter, self._filetype_filters = None, None
        self.update_table()

    @property
    def filter_filename(self) -> str | None:
        """
        Returns the current filename filter.

        Returns
        -------
        str | None
            The current filename filter.
        """
        return self._str_filter

    @property
    def progress_bar(self) -> QProgressBar | None:
        """
        Returns an associated progress bar widget if provided, else None.

        Parameters
        ----------
        progress_bar: QProgressBar
            The progress bar widget.

        Returns
        -------
        QProgressBar | None
            The progress bar widget.
        """
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar: QProgressBar):
        self._progress_bar = progress_bar

    ### Functions to collect information to build and update the table model from the selected directory, parser and filters.
    def generate_header(self) -> list[str]:
        """
        Generate the column headers using the default parser important parameters.

        First attempts to call `summary_param_names_with_units` on the first non-None
        header-loaded parser object, to load the column labels and their appropriate units.
        Alternatively calls `summary_param_names` on the object, which doesn't require unit labels.
        These two methods take into account the parser `relabel` property.
        """
        # Setup default header columns
        header = (
            self.__REQUIRED_HEADER_LIST.copy()
        )  # Get the default headers, but don't modify the original list.
        # Add list values if parser is defined, and header objects allow unit calls.
        if not (self.parser is not None and len(self._parser_headers) > 0):
            # No parser or no headers (of files), so use default file headers.
            header += (
                self.__DEFAULT_FILE_HEADERS
            )  # Get defaults, i.e. (created, modified, size).
        else:
            header += (
                self.__DEFAULT_PARSER_HEADER_KEYS
            )  # Get defaults, i.e. (loaded), number and name.
            # Add the status header i.e. "Load Success"
            header.insert(self._status_index(), self._status_header())

            # Collect the first non-None parser object.
            val = None
            # Collect the first non-None parser object.
            for val in self._parser_headers.values():
                # Collect val (parser objects), where val is not None.
                if val is not None:
                    assert isinstance(val, self.parser)
                    break
            # Establish parameter header names:
            if val is not None:
                try:
                    # Attempt to use 'summary_param_names_with_units' if available.
                    obj_head = val.summary_param_names_with_units
                except NotImplementedError:
                    warnings.warn(
                        f"{self.parser.__name__} has not implemented 'summary_param_names_with_units'. Defaulting to 'summary_param_names'."
                    )
                    obj_head = val.summary_param_names  # without units.
            else:
                # If all parser headers are None, then use parser default values.
                obj_head = self.parser.summary_param_names

            # Add obj_head to header
            header += obj_head

        return header

    def generate_header_data(self) -> list[list[Any]]:
        """
        Collect the data corresponding to the header names.

        Uses the parser objects to generate the header data.
        """
        # Do nothing if an empty directory.
        if self.directory is None:
            return [[]]

        TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

        path = self.directory
        files = self.files
        parsers = self._parser_headers

        data = [[] for _ in range(len(files))]  # list of file header lists.
        if self.parser is None:
            # Get defaults, matching `self.__REQUIRED_HEADER_LIST` and `self.__DEFAULT_FILE_HEADERS`
            # - "#", "Filename", "Created", "Modified", "Size"
            for i, file in enumerate(files):
                data[i] = [i + 1, file]  # add index (1, ...) and filename to data.
                # Use OS to get file info
                data[i].extend(
                    [
                        dt.fromtimestamp(
                            os.path.getctime(os.path.join(path, file))
                        ).strftime(TIME_FORMAT),
                        dt.fromtimestamp(
                            os.path.getmtime(os.path.join(path, file))
                        ).strftime(TIME_FORMAT),
                        (
                            btyes_to_human_readable(
                                os.path.getsize(os.path.join(path, file))
                            )
                        ),
                    ]
                )
        else:
            # Get defaults or custom selections...
            all_headers = self.header
            assert (
                all_headers is not None
            ), f"`self._header_param_names_default`: Header names are None with parser {self.parser}."
            opt_headers = all_headers[
                len(self.__REQUIRED_HEADER_LIST) :
            ]  # remove 'required' headers.

            print(
                "HERE - adding data with headers: ",
                opt_headers,
                len(self.__REQUIRED_HEADER_LIST),
            )
            # - "Load Success", "#", "Filename", "Created", "Modified", "Size"
            for i, file in enumerate(files):
                # Get the loading index
                load_index = self._status_index()
                data[i] = [i + 1, file]  # add index (1, ...) and filename to data.

                if file in parsers and parsers[file] is not None:
                    # Use existing parser instance, add to files.
                    parser: parser_base = parsers[file]  # Checked above: # type: ignore
                    # Iterate over the optional headers and add them to the data list.
                    for param in opt_headers:
                        if i == 0:
                            print(f"HERE - adding {param}")
                        # Check if the parameter is in the parser.params list
                        if parser is not None and param in parser.params:
                            if param.lower() in [
                                "created",
                                "modified",
                                "ctime",
                                "mtime",
                            ]:
                                # Convert to datetime object
                                value = parser.params[param]
                                if isinstance(value, dt):
                                    data[i].append(value.strftime(TIME_FORMAT))
                                else:
                                    data[i].append(
                                        dt.fromtimestamp(float(value)).strftime(
                                            TIME_FORMAT
                                        )
                                    )

                            elif param.lower() in ["size", "memory_size"]:
                                # Convert to human readable size
                                data[i].append(
                                    btyes_to_human_readable(int(parser.params[param]))
                                )
                            else:
                                # Get the parameter value
                                data[i].append(parser.params[param])
                        # Otherwise check if the parameter is a valid attribute of the parser object.
                        elif parser is not None and hasattr(parser, param):
                            # Get the parameter value
                            data[i].append(getattr(parser, param))
                        else:
                            print(parser, param)
                            # If not, add an empty string to the data list.
                            data[i].append("")

                    status: tableModel.loadStatus
                    if parser.is_loaded:
                        status = tableModel.loadStatus.FULL_DATA
                    else:
                        status = tableModel.loadStatus.HEADER_ONLY
                else:
                    # Not in _parser_headers
                    status = tableModel.loadStatus.ERROR

                data[i].insert(
                    load_index,
                    status,
                    #    (True
                    #    if file in self._parser_headers
                    #    and self._parser_headers[file] is not None
                    #    else False)
                )
                if i == 0:
                    print("The data is ", data[i])
        return data

    def update_header(self) -> None:
        """
        Update the column headers to display.
        """

        header = self.header
        print("The header is ", header)
        if header is None:
            return

        # Format the header
        formatted_header = [label.replace("_", " ").capitalize() for label in header]

        # Ideally just updates header names, but occasionally crashes when attempting to layoutChange.emit()
        if self.files_model is not None:
            selection_model = self.selectionModel()
            if selection_model is not None:
                old_selection = selection_model.selectedRows()
                new_model = tableModel(
                    data=self.files_model._data, header=(formatted_header)
                )
                self.proxy_model.setSourceModel(new_model)
                self.files_model = new_model
                # Setup and restore selection: requires multiselection to select each row before using regular extended selection.
                self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
                [self.selectRow(i.row()) for i in old_selection]
                self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            else:
                new_model = tableModel(
                    data=self.files_model._data, header=(formatted_header)
                )
                self.proxy_model.setSourceModel(new_model)
                self.files_model = new_model
                self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Update column widths with header update.
        for i in range(0, len(formatted_header)):
            # If the current column width is larger than the size hint, resize to fit for all columns.
            if self.columnWidth(i) > self.sizeHintForColumn(i):
                self.resizeColumnToContents(i)
        return

    def update_table(self):
        """
        Update the directory view.

        Applies text filter, parser and permissionable extensions to filter the files list.
        Loads the filtered header data for the table, appling updates to the class progress bar if defined.
        """
        # Collect the data corresponding to the headers.
        data = self.generate_header_data()

        if len(data) > 1:
            print("The data is ", data[0], data[1], "...")

        # Create a table model
        header = self.header
        self.files_model = tableModel(data=data, header=header)

        # Set the model to the proxy model
        self.proxy_model.setSourceModel(self.files_model)

        # Resize the columns to fit the contents
        self.resizeColumnToContents(0)
        if header is not None:
            for i in range(0, len(header)):
                # If the current column width is larger than the size hint, resize to fit for all columns.
                if self.columnWidth(i) > self.sizeHintForColumn(i):
                    self.resizeColumnToContents(i)
            if self.progress_bar is not None:
                self.progress_bar.setValue(0)

    def load_selection(self) -> None:
        """
        Loads full data from files of the current selection.

        Instead of just header information, loads all data lines
        from the selected files. Loaded datafiles are cached in
        self._parser_files, until the parser object is changed.
        """
        sm = self.selectionModel()
        # Require selection and parser to trigger load data.
        if sm.hasSelection() and self.parser is not None:
            rows = sm.selectedRows(
                column=1 if self._status_index() > 1 else 2
            )  # get filename, not index.

            # Generate a thread pool to load files in parallel.
            pool = QThreadPool.globalInstance()
            """The global thread pool."""
            if pool is None:
                pool = QThreadPool()
            thread_count: int = pool.maxThreadCount()
            THREAD_MIN_DIV: int = 5
            """Minimum number of files to load in a thread."""
            main_thread_core = QThread.currentThread()
            """The main thread - leave this core for the GUI."""

            # Set the progress bar to zero.
            if self.progress_bar is not None:
                self.progress_bar.setValue(0)
                # Store number of files to be processed for progress bar
                self.progress_bar.setRange(0, len(rows))

            # Setup warnings to show once.
            warnings.simplefilter("once", UserWarning)

            # Collect files to load in a thread.
            # files_to_load: dict[str, parser_base] = {} # TODO: threading
            for i, row in enumerate(rows):
                filename: str = self.proxy_model.data(row, Qt.ItemDataRole.DisplayRole)
                # Check if the file already loaded:
                if (
                    filename not in self._parser_files
                    or self._parser_files[filename]
                    is None  # re-attempt if parser is None.
                ):
                    # Check for a header object (should always exist).
                    if (
                        filename in self._parser_headers
                        and self._parser_headers[filename] is not None
                    ):
                        # Use existing parser instance, add to files.
                        parser = self._parser_headers[filename]
                        assert isinstance(parser, parser_base)
                        # files_to_load[filename] = parser TODO: threading

                        # TODO: Implement threading generation and signal emit when complete.
                        # if len(files_to_load) > 0:
                        #     if thread_count > 0 and len(files_to_load)//THREAD_MIN_DIV > 0:
                        #         # Setup threads to load files in parallel.
                        #         files_per_thread = ((len(files_to_load) // thread_count)
                        #                             + ((len(files_to_load) % thread_count) > 0)) #Add one to cover remainder if not perfect division
                        #         # Generate threads to load files in parallel.
                        #         threads = [
                        #             fileLoaderWorker(
                        #                 n=i,
                        #                 files=list(files_to_load.keys())[i * files_per_thread : (i + 1) * files_per_thread],
                        #                 parser=self.parser,
                        #                 parsers=files_to_load,
                        #                 progress_bar=self.progress_bar,
                        #                 load_kwargs=dict(header_only = False)
                        #             )
                        #             for i in range(thread_count)
                        #         ]
                        #         # Run the threads
                        #         for thread in threads:
                        #             pool.start(thread)
                        # Wait for the threads to finish
                        #         pool.waitForDone()

                        try:
                            parser.load()  # use internal filepath to load
                            if parser.is_loaded:
                                self._parser_files[filename] = parser
                                ## Update the table icon
                                # Get index from proxy model
                                index = self.proxy_model.index(i, self._status_index())
                                print(
                                    self.proxy_model.data(
                                        index, Qt.ItemDataRole.DecorationRole
                                    ).name()
                                )
                                print(
                                    f"updating to {self.files_model._icon_full_data.name()}, SETDATA"
                                )
                                # self.proxy_model.setData(index,
                                #                          self.files_model._icon_full_data, #type ignore.
                                #                          Qt.ItemDataRole.DecorationRole
                                #                          )
                                assert self.files_model is not None
                                source_index = self.proxy_model.mapToSource(index)
                                self.files_model.setData(
                                    source_index,
                                    self.files_model._icon_full_data,  # type ignore.
                                    Qt.ItemDataRole.DecorationRole,
                                )
                                # Update the model
                                # print(f"Updated index {index}", index.row())
                                print(
                                    self.proxy_model.data(
                                        index, Qt.ItemDataRole.DecorationRole
                                    ).name()
                                )
                                print(
                                    self.files_model.data(
                                        source_index, Qt.ItemDataRole.DecorationRole
                                    ).name()
                                )
                        except Exception as e:
                            traceback.print_exception(e)
                    # else: ignore unloaded headers (None) in the selection.
                if self.progress_bar is not None:
                    self.progress_bar.setValue(i + 1)

            # Restore warnings functionality.
            warnings.resetwarnings()
            # Trigger a loading completed signal.
            self.progress_bar.setValue(0)
            self.selection_loaded.emit(True)
        return

    @property
    def selected_filenames(self) -> list[str]:
        """
        Returns the selected filenames in the directory viewer.

        Returns
        -------
        list[str]
            A list of filenames selected in the directory viewer.
        """
        sm = self.selectionModel()
        if sm.hasSelection():
            rows = sm.selectedRows(
                column=self._index_filename_column
            )  # Filename in second column.
            # return self.files_model.data(rows, Qt.ItemDataRole.DisplayRole)
            return [
                self.proxy_model.data(row, Qt.ItemDataRole.DisplayRole) for row in rows
            ]
        return []

    @property
    def _index_filename_column(self) -> int:
        """
        Returns the index of the filename column in the table model.
        """
        if self.parser is None:
            return 1
        else:
            return 2 if self._status_index() < 2 else 1


class SummaryParamSelector(QTableWidget):
    """
    A togglable interface for selecting parameters of parser objects.
    """

    def __init__(
        self,
        parsers: list[parser_base] | None = None,
        parent: QWidget | None = None,
        existing_selection: list[str] | None = None,
    ):
        # Init the table
        super().__init__(parent)

        # Copy the list of parser_objects
        self._parsers: list[parser_base] = parsers.copy() if parsers is not None else []
        """A list of parser objects to select parameters from."""

        # Create a set of unique parameters
        self._params: set[str] = self._collect_unique_parameters()
        self._filtered_params: set[str] | None = None
        """An optional set of filtered parameters."""
        self.filter_text: str | None = None
        """The current filter text."""

        # Collect the set of parser classes
        self._parser_cls: set[type[parser_base]] = self._collect_unique_parser_cls()
        """A list of the parser subclasses which default `param` values can be taken."""

        # Create a memory set of items (selected or not)
        self._items: dict[str, QTableWidgetItem]
        """A dictionary map of the parameter items that could be in the table, saving their selected state."""
        self._initialise_items()

        # Set a default value for max cols
        self._max_columns: int = 3

        # Disable multiselection
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Initialise the table view
        self._populate(existing_selection)

    def filter_parameters(self, text: str | None) -> None:
        """
        Filters the parameters by the text string.

        Parameters
        ----------
        text : str | None
            The text to filter the parameters by.
        """
        self.filter_text = text
        if text is None or text == "":
            self._filtered_params = None
            self._populate()
            return
        # Otherwise filter the parameters
        params = self.params
        self._filtered_params = set([p for p in params if text in p])
        self._populate()
        return

    @property
    def max_cols(self):
        """
        The number of columns to spread the parameter display over.
        """
        return self._max_columns

    @max_cols.setter
    def max_cols(self, val: int):
        self._max_columns = val

    @property
    def parsers(self) -> list[parser_base]:
        return self._parsers

    @parsers.setter
    def parsers(self, parsers: list[parser_base] | None) -> None:
        self._parsers = parsers.copy() if parsers is not None else []
        # Update the collection
        self._params = self._collect_unique_parameters()
        # Update the class list
        self._parser_cls = self._collect_unique_parser_cls()
        # Update the table
        self._populate()
        return

    @property
    def params(self) -> set[str]:
        """
        The unique set of parameter names belonging to the list of parsers.
        """
        return self._params

    def _collect_unique_parameters(self) -> set[str]:
        """
        Collects the common parser parameters for each item

        Returns
        -------
        list[str]
            A list of parameter string keys.
        """
        params = set()
        for parser in self.parsers:
            for p in parser.params.keys():
                if p not in params:
                    params.add(p)
        return params

    def _collect_unique_parser_cls(self) -> set[type[parser_base]]:
        """
        Collects the class types in the parser list.

        Returns
        -------
        list[type[parser_base]]
            A list of the unique parser classes in the parser
        """
        cls_list: set[type[parser_base]] = set()
        for p in self._parsers:
            if p.__class__ not in cls_list:
                cls_list.add(p.__class__)
        return cls_list

    def _initialise_items(self) -> None:
        """Initialises the table items with the parameter set and checks the state for summary parameters."""
        self._items = {p: QTableWidgetItem(p) for p in self.params}
        for x in self.params:
            # Make the item uneditable.
            item = self._items[x]
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            # Check the item if it is a default parameter of any parser classes.
            if any(
                [
                    x in cls.summary_param_names
                    or (
                        x in cls.RELABELS and cls.RELABELS[x] in cls.summary_param_names
                    )
                    for cls in self._parser_cls
                ]
            ):
                # Set the item to checked if it is a default parameter.
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
        return

    def _populate(self, existing_selection: list[str] | None = None) -> None:
        """
        Populates the table with the current set of parameter value checkboxes.
        """

        ### TODO: Disconnect the filtering from the active selection returned - currently filtering creates
        # a selection subset of elements you want to tick (and keep track of their tick).

        # Update the existing maps
        if existing_selection:
            for p in existing_selection:
                if p in self._items:
                    self._items[p].setCheckState(Qt.CheckState.Checked)
                else:
                    # Create a new item
                    self._items[p] = QTableWidgetItem(p, Qt.CheckState.Checked.value)

        # Clear the contents, and set the appropriate number of rows/columns
        self.clearContents()
        cols = self.max_cols
        self.setColumnCount(cols)

        # Convert params to an alphabetically ordered list
        if self._filtered_params is not None:
            params = sorted(list(self._filtered_params))
        else:
            params = sorted(list(self.params))

        nparams = len(params)
        # If no parameters, return
        if nparams == 0:
            return
        # Calculate the number of items per column
        items_per_column = nparams // cols
        if items_per_column == 0:
            items_per_column = 1
        # Add extra column for a modulus remainder
        if nparams % items_per_column > 0:
            items_per_column += 1
        self.setRowCount(items_per_column + 1)

        # Additionally, order the default selected parameters first.
        # These need to be checked for existing within the parser param objects, which handle relabelling.
        default_params = []
        for x in params:
            if any(
                [
                    x in cls.summary_param_names
                    or (
                        x in cls.RELABELS and cls.RELABELS[x] in cls.summary_param_names
                    )
                    for cls in self._parser_cls
                ]
            ):
                default_params.append(x)

        params = sorted(params, key=lambda x: x in default_params, reverse=True)

        # Set the items and checked values
        for i, p in enumerate(params):
            # Create the item if it doesn't exist
            if p not in self._items:
                self._items[p] = item = QTableWidgetItem(p)
                # Make the item uneditable.
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                # Set the check state
                if p in default_params:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
            else:
                # Get the item
                item = self._items[p]
            # Assign the item, use a copy so it isn't deleted...
            self.setItem(i % items_per_column, i // items_per_column, item.clone())

        # Adjust the column widths to match min width
        widths = []
        for i in range(self.columnCount()):
            self.resizeColumnToContents(i)
            widths.append(self.columnWidth(i))
        self.setMinimumWidth((sum(widths) * 19) // 18)
        return

    def refresh(self) -> None:
        """
        Updates the table with the current set of parameters.
        """
        self._params = self._collect_unique_parameters()
        self._parser_cls = self._collect_unique_parser_cls()
        self._initialise_items()
        self._populate()
        return

    def resizeEvent(self, e: QtGui.QResizeEvent | None) -> None:
        """
        Resize the columns to fit the new width.
        """
        if e is not None:
            widths = [self.columnWidth(i) for i in range(self.columnCount())]
            scale = self.width() / sum(widths)

            for i in range(self.columnCount()):
                self.setColumnWidth(i, int(widths[i] * scale))
        return super().resizeEvent(e)

    def selection(self) -> set[str]:
        """
        Finds the current item list that has checked values.

        Returns
        -------
        set[str]
            A set of toggled parameter names.
        """
        params = set()
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                item = self.item(i, j)
                if item is not None and item.checkState() == Qt.CheckState.Checked:
                    params.add(item.text())
        return params


class SummaryParamSelectorDialog(QtWidgets.QDialog):
    """Dialog for selecting parameters of parser objects."""

    def __init__(
        self,
        parsers: list[parser_base] | None = None,
        parent: QWidget | None = None,
        existing_selection: list[str] | None = None,
    ):
        super().__init__(parent=parent)

        # Setup layout.
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # Setup margins if a parent is provided to keep interface tight.
        if parent is not None:
            self._layout.setContentsMargins(0, 0, 0, 0)

        # Setup layout and init elements.
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel("Select parameters to display:"), stretch=3)
        relabels_checkbox = QtWidgets.QCheckBox("Relabel")
        hlayout.addWidget(relabels_checkbox, stretch=1)
        self._layout.addLayout(hlayout)
        param_filter = QtWidgets.QLineEdit()
        param_filter.setPlaceholderText("Filter parameters...")
        self._layout.addWidget(param_filter)

        # Check if parsers are provided and are relabelled or not. Default all parsers to relabel if not set.
        if parsers is not None and len(parsers) > 0:
            # Check if any parsers are relabelled.
            relabels = [p.relabel for p in parsers]
            # Majority check
            if sum(relabels) > len(relabels) // 2:
                relabels_checkbox.setCheckState(Qt.CheckState.Checked)
                for p in parsers:
                    p.__class__.relabel = True
            else:
                relabels_checkbox.setCheckState(Qt.CheckState.Unchecked)
                for p in parsers:
                    p.__class__.relabel = False

        self._summary_param_selector = SummaryParamSelector(
            parsers=parsers, parent=self, existing_selection=existing_selection
        )
        self._layout.addWidget(self._summary_param_selector)
        self.setWindowTitle("Select Parser Parameters")

        # Add buttons
        QBtn = (
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self._buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self._buttonBox.accepted.connect(self.accept)
        self._buttonBox.rejected.connect(self.reject)
        self._layout.addWidget(self._buttonBox)

        # Bind the relabel checkbox to the parser objects.
        relabels_checkbox.stateChanged.connect(self._relabel_parser_objects)

        # Bind the filter line edit to the parser objects.
        param_filter.textChanged.connect(self._filter_parameters)

    def _filter_parameters(self, text: str) -> None:
        """
        Filters the parameters by the text string.

        Parameters
        ----------
        text : str | None
            The text to filter the parameters by.
        """
        self.selector.filter_parameters(text)
        return

    def _relabel_parser_objects(self, state: int) -> None:
        """
        Relabels the parser objects if the checkbox is toggled.

        Parameters
        ----------
        state : int
            The state of the checkbox
        """
        # Convert the int to the enumerate.
        state = Qt.CheckState(state)
        # Check the state of the checkbox
        for cls in self.selector._parser_cls:
            cls.relabel = state is Qt.CheckState.Checked
        # Repopulate the table with (re)labelled parameters.
        self.selector.refresh()
        return

    @property
    def selector(self) -> SummaryParamSelector:
        """
        The SummaryParamSelector object.

        Returns
        -------
        SummaryParamSelector
            The SummaryParamSelector object.
        """
        return self._summary_param_selector

    @property
    def selection(self) -> set[str]:
        """
        The SummaryParamSelector toggled selection.

        Returns
        -------
        set[str]
            A set of selected parameter names.
        """
        return self.selector.selection()

    @overrides.override
    def keyPressEvent(self, a0: QtGui.QKeyEvent | None) -> None:
        if a0 is None:
            return
        if a0.key() == QtCore.Qt.Key.Key_Escape:
            self.reject()
        elif (
            a0.key() == QtCore.Qt.Key.Key_Return or a0.key() == QtCore.Qt.Key.Key_Enter
        ):
            return a0.accept()
        else:
            super().keyPressEvent(a0)

    @overrides.overrides
    def accept(self):
        print("OK")
        return super().accept()

    @overrides.overrides
    def reject(self):
        return super().reject()


# Example
import sys
from pyNexafs.parsers.au.aus_sync.MEX2 import MEX2_NEXAFS

if __name__ == "__main__":
    # Load some example parsers
    base_dir = os.getcwd()
    f1 = r"tests\\test_data\\au\\MEX2\\2024-03\\MEX2_5640.mda"
    f2 = r"tests\\test_data\\au\\MEX2\\2024-03\\MEX2_5642.mda"
    files = [os.path.normpath(os.path.join(base_dir, f)) for f in [f1, f2]]

    # Relabelling
    MEX2_NEXAFS.relabel = True
    parsers = [MEX2_NEXAFS(f, header_only=True) for f in files]

    app = QApplication(sys.argv)
    window = SummaryParamSelectorDialog(parsers=parsers)
    # window = directoryViewerTableNew(os.path.dirname(files[0]))
    window.show()
    window.setWindowTitle("Summary Param Selector")
    val = app.exec()
    print(val)
    if val:
        print(window.selection)
        sys.exit(True)
    else:
        sys.exit(False)
