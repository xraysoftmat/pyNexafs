from PyQt6 import QtWidgets, QtCore, QtGui
from pyNexafs.parsers._base import parser_base
from pyNexafs.nexafs.scan import scanBase
from typing import Type
import sys
from pyNexafs.gui.widgets.io.dir_selection import directorySelector
import pyNexafs.parsers.au as AU_PARSERS
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas, NEXAFS_NavQT
import warnings
from enum import Enum
import numpy as np
import numpy.typing as npt
import os
import matplotlib.pyplot as plt


class save_directory_selector(directorySelector):
    edit_description = "Path to save QANT-compatible data to."
    dialog_caption = "Select Save Directory"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout: QtWidgets.QHBoxLayout = self.layout()
        layout.insertWidget(0, QtWidgets.QLabel("Save Directory:"))


class NaN_OPTION(Enum):
    """
    Enumerate to define methods of processing NaN (Not a Number) values.

    Attributes
    ----------
    NONE : int
        No special treatment. Data stored with NaN labels. Incompatible with QANT?
    REMOVE : int
        Removes NaN rows. Affects multiple data columns that might not have NaNs.
    INTERPOLATE_2 : int
        Interpolates the nearest 2 (non-NaN) point neighbours to
    """

    NONE = 0
    REMOVE_ROWS = 1
    INTERPOLATE_2_POINTS = 2
    INTERP_EXTRAP_2_POINTS = 3
    # INTERPOLATE_4_POINTS = 3
    # INTERPOLATE_N_POINTS = 4


class horLine(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.setLineWidth(1)
        self.setMinimumHeight(1)
        self.setMinimumWidth(1)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )

    def event(self, event: QtCore.QEvent) -> bool:
        """
        Event handler for the widget.

        Adds palette change control for light/dark mode to QWidget event handler.

        Parameters
        ----------
        event : QtCore.QEvent
            The event to handle.

        Returns
        -------
        bool
            Whether the event was handled.
        """
        if (
            event.type() == QtCore.QEvent.Type.PaletteChange
            or event.type() == QtCore.QEvent.Type.ApplicationPaletteChange
        ):
            self.on_recolour()
        return super().event(event)

    def on_recolour(self):
        """
        Recolour the division lines based on the theme.
        """
        # Get theme
        toolbar_palette = self.palette()
        light_theme_bool = toolbar_palette.window().color().lightnessF() > 0.5
        self.setStyleSheet(
            "background-color: " + ("black;" if light_theme_bool else "white;")
        )


class nexafsParserConverter(QtWidgets.QWidget):
    def __init__(self, parsers: list[Type[parser_base | scanBase]] = [], parent=None):
        super().__init__(parent)
        self._parsers = {parser.filename: parser for parser in parsers}
        self._conversions = []
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)

        ## Setup UI elements
        # Title
        self._title_label = title_label = QtWidgets.QLabel("Converstion:")
        # Folder selector
        self.dir_sel = dir_sel = save_directory_selector()

        # File statistics
        file_widget = QtWidgets.QWidget()
        file_layout = QtWidgets.QGridLayout()
        file_label = QtWidgets.QLabel("File Statistics")
        file_layout.addWidget(file_label, 0, 0, 1, 4)
        file_widget.setLayout(file_layout)
        file_widget.setContentsMargins(0, 0, 0, 0)
        file_layout.setContentsMargins(0, 0, 0, 0)
        # Number of files
        files_num_label = QtWidgets.QLabel("Number of Files:")
        self.files_num_indicator = QtWidgets.QLabel("")
        file_layout.addWidget(files_num_label, 1, 0)
        file_layout.addWidget(self.files_num_indicator, 1, 1)
        # Total filesize
        files_size_label = QtWidgets.QLabel("Total Filesize:")
        self.files_size_indicator = QtWidgets.QLabel("")
        file_layout.addWidget(files_size_label, 1, 2)
        file_layout.addWidget(self.files_size_indicator, 1, 3)

        # # NaN inspector
        nan_widget = QtWidgets.QWidget()
        nan_layout = QtWidgets.QGridLayout()
        nan_label = QtWidgets.QLabel("NaN Inspector:")
        nan_layout.addWidget(nan_label, 0, 0, 1, 2)
        nan_widget.setLayout(nan_layout)
        nan_widget.setContentsMargins(0, 0, 0, 0)
        nan_layout.setContentsMargins(0, 0, 0, 0)
        # Number of files with NaNs
        total_nan_files_label = QtWidgets.QLabel("Files with NaNs:")
        self.total_nan_files_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(total_nan_files_label, 0, 2)
        nan_layout.addWidget(self.total_nan_files_indicator, 0, 3)
        # Total nan indicator:
        total_nan_instances_label = QtWidgets.QLabel("Total NaN Instances:")
        self.total_nan_instances_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(total_nan_instances_label, 1, 0)
        nan_layout.addWidget(self.total_nan_instances_indicator, 1, 1)
        # Per file nan indicator:
        per_file_nan_label = QtWidgets.QLabel("NaN Instances per Nan File:")
        self.per_file_nan_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(per_file_nan_label, 1, 2)
        nan_layout.addWidget(self.per_file_nan_indicator, 1, 3)
        # Method of dealing with Nan Values.
        nan_method_label = QtWidgets.QLabel("NaN Method:")
        self.nan_method = QtWidgets.QComboBox()
        self.nan_method.addItems(
            [
                key.replace("_", " ").lower().capitalize()
                for key in NaN_OPTION.__members__.keys()
            ]
        )
        nan_layout.addWidget(nan_method_label, 2, 0, 1, 1)
        nan_layout.addWidget(self.nan_method, 2, 1, 1, 3)

        # Number of files with Zeros:
        total_zero_files_label = QtWidgets.QLabel("Files with Zero Blocks:")
        self.total_zero_files_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(total_zero_files_label, 3, 2)
        nan_layout.addWidget(self.total_zero_files_indicator, 3, 3)
        # Total zeros indicator:
        total_zero_instances_label = QtWidgets.QLabel("Total Zero Instances:")
        self.total_zero_instances_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(total_zero_instances_label, 4, 0)
        nan_layout.addWidget(self.total_zero_instances_indicator, 4, 1)
        # Per file zeros indicator:
        per_file_zero_label = QtWidgets.QLabel("Zero Instances per Zero Block File:")
        self.per_file_zero_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(per_file_zero_label, 4, 2)
        nan_layout.addWidget(self.per_file_zero_indicator, 4, 3)
        # Treat block zeros as NaNs
        self.nan_zero_checkbox = QtWidgets.QCheckBox("Block Zeros as NaN?")
        self.nan_zero_checkbox.setToolTip(
            "Treat blocks of zeros in data as NaN values."
        )
        nan_layout.addWidget(self.nan_zero_checkbox, 3, 0, 1, 2)

        ## Converter Buttons / Options
        exec_widget = QtWidgets.QWidget()
        exec_layout = QtWidgets.QGridLayout()
        exec_widget.setLayout(exec_layout)
        exec_widget.setContentsMargins(0, 0, 0, 0)
        exec_layout.setContentsMargins(0, 0, 0, 0)
        conversion_text = QtWidgets.QLabel("Conversion Execution:")
        self.progress = progress = QtWidgets.QProgressBar()
        progress.setMinimumHeight(5)
        progress.setMinimumWidth(200)
        progress.setTextVisible(False)
        progress.setOrientation(QtCore.Qt.Orientation.Horizontal)
        progress.setValue(0)
        copy_button = QtWidgets.QPushButton("Copy (single) to Clipboard")
        save_button = QtWidgets.QPushButton("Save (selection) to File")
        self._override_checkbox = override_checkbox = QtWidgets.QCheckBox(
            "Override Existing Files?"
        )
        self._rename_checkbox = rename_checkbox = QtWidgets.QCheckBox(
            "Rename? (`<sample>_<scan_id>.asc`)"
        )
        exec_layout.addWidget(conversion_text, 0, 0, 1, 2)
        exec_layout.addWidget(copy_button, 0, 2, 1, 2)
        exec_layout.addWidget(override_checkbox, 1, 0, 1, 2)
        exec_layout.addWidget(rename_checkbox, 1, 2, 1, 2)
        exec_layout.addWidget(progress, 2, 0, 1, 4)
        exec_layout.addWidget(save_button, 3, 0, 1, 4)

        ## Difference viewer
        diff_widget = QtWidgets.QWidget()
        diff_layout = QtWidgets.QGridLayout()
        diff_widget.setLayout(diff_layout)
        diff_widget.setContentsMargins(0, 0, 0, 0)
        diff_layout.setContentsMargins(0, 0, 0, 0)
        diff_label = QtWidgets.QLabel("Differences:")
        diff_layout.addWidget(diff_label, 0, 1, 1, 3)
        self._selected_parser = None
        file_label = QtWidgets.QLabel("File:")
        self.diff_parser_selector = QtWidgets.QComboBox()
        series_label = QtWidgets.QLabel("Series:")
        self.diff_series_selector = QtWidgets.QComboBox()
        diff_layout.addWidget(file_label, 1, 0, 1, 3)
        diff_layout.addWidget(self.diff_parser_selector, 1, 1, 1, 3)
        diff_layout.addWidget(series_label, 2, 0, 1, 3)
        diff_layout.addWidget(self.diff_series_selector, 2, 1, 1, 3)
        self.fig = fig = plt.figure(figsize=(2, 2))
        self.diff_canvas = FigureCanvas(fig)
        self.diff_tb = NEXAFS_NavQT(self.diff_canvas, self)
        diff_layout.addWidget(self.diff_canvas, 3, 0, 1, 4)
        diff_layout.addWidget(self.diff_tb, 4, 0, 1, 4)

        # Track nan_parsers
        self._nan_parsers = {}

        # Add to layout
        self._layout.addWidget(title_label)
        self._layout.addWidget(dir_sel)
        self._layout.addWidget(horLine())
        self._layout.addWidget(file_widget)
        self._layout.addWidget(horLine())
        self._layout.addWidget(nan_widget)
        self._layout.addWidget(horLine())
        self._layout.addWidget(exec_widget)
        self._layout.addWidget(horLine())
        self._layout.addWidget(diff_widget)
        self._layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Connect signals
        copy_button.clicked.connect(self.on_copy_to_clipboard)
        save_button.clicked.connect(self.on_save_to_file)
        self.nan_method.currentIndexChanged.connect(self.on_conversion_change)
        self.nan_zero_checkbox.stateChanged.connect(self.on_conversion_change)
        self.diff_parser_selector.currentIndexChanged.connect(
            self.on_diff_parser_change
        )
        self.diff_series_selector.currentIndexChanged.connect(
            self.on_diff_series_change
        )

    def on_conversion_change(self):
        # Recalculate the conversions.
        self.calculate_nan()
        self.graph_diff()

    def on_parsers_selected(self):
        """Updates the parser selector based on the provided parsers objects."""
        self.diff_parser_selector.clear()
        self._nan_parsers.clear()
        current_idx = 0  # Used to keep track of ComboBox index.
        for i, parser in enumerate(self.parsers):
            if np.any(np.isnan(parser.data)):
                self.diff_parser_selector.addItem(parser.filename)
                # Add a mapping to
                self._nan_parsers[current_idx] = i
                current_idx += 1
        # Call the diff parser change to update the series selector.
        self.on_diff_parser_change()

    def on_diff_parser_change(self):
        """Updates the series selector based on the selected parser."""
        self.diff_series_selector.blockSignals(True)
        self.diff_series_selector.clear()
        current_idx = self.diff_parser_selector.currentIndex()
        parsers = self.parsers
        if current_idx == None:
            self.diff_series_selector.clear()
            return
        elif (
            isinstance(current_idx, int)
            and current_idx > -1
            and len(parsers) > current_idx
        ):
            parser = parsers[self._nan_parsers[current_idx]]
            for i in range(parser.data.shape[1]):
                if np.any(np.isnan(parser.data[:, i])):
                    self.diff_series_selector.addItem(parser.labels[i])
            self.on_diff_series_change()
        self.diff_series_selector.blockSignals(False)
        # Call the series change.
        self.on_diff_series_change()

    def on_diff_series_change(self):
        """Actions to perform when a parser/series combination is selected."""
        self.graph_diff()

    def graph_diff(self):
        """Graphs the raw and nan-processed data of the selected parser/series."""
        selector_index = self.diff_parser_selector.currentIndex()
        self.fig.clear()
        # 0 is the empty string
        if selector_index is not None and selector_index > -1:
            parser_idx = self._nan_parsers[selector_index]
            # Collect parser
            current_parser = self.parsers[parser_idx]
            current_converted = self._conversions[parser_idx]
            # Collect the index of the nan data series
            series_idx = current_parser.labels.index(
                self.diff_series_selector.currentText()
            )

            # Graph both converted and original data
            fig = self.fig
            ax = self.fig.add_subplot(111)
            x_idx = current_parser.label_index(current_parser.COLUMN_ASSIGNMENTS["x"])

            # Plot the converted data depending on the NaN method.
            nan_method = NaN_OPTION(self.nan_method.currentIndex())
            match nan_method:
                # # Locate the nan domain
                # parser_nans = np.isnan(current_parser.data[:, series_idx])
                # # Let nearest neighbours also be displayed in the nan data:
                # roll_pos = np.roll(parser_nans, 1)
                # roll_neg = np.roll(parser_nans, -1)
                # parser_nans = roll_neg | roll_pos | parser_nans
                case NaN_OPTION.REMOVE_ROWS:
                    # Plot raw data after, so gaps can be seen underneath.
                    ax.plot(
                        current_converted.data[:, x_idx],
                        current_converted.data[:, series_idx],
                        label="NaN Fixed",
                        c="r",
                    )
                    ax.plot(
                        current_parser.data[:, x_idx],
                        current_parser.data[:, series_idx],
                        label="Original",
                        c="b",
                    )
                case NaN_OPTION.INTERPOLATE_2_POINTS:
                    # Plot raw data after, so gaps can be seen underneath.
                    ax.plot(
                        current_converted.data[:, x_idx],
                        current_converted.data[:, series_idx],
                        label="NaN Fixed",
                        c="r",
                    )
                    ax.plot(
                        current_parser.data[:, x_idx],
                        current_parser.data[:, series_idx],
                        label="Original",
                        c="b",
                    )
                case NaN_OPTION.INTERP_EXTRAP_2_POINTS:
                    # Plot raw data after, so gaps can be seen underneath.
                    ax.plot(
                        current_converted.data[:, x_idx],
                        current_converted.data[:, series_idx],
                        label="NaN Fixed",
                        c="r",
                    )
                    ax.plot(
                        current_parser.data[:, x_idx],
                        current_parser.data[:, series_idx],
                        label="Original",
                        c="b",
                    )
                    pass
                case _:
                    ax.plot(
                        current_converted.data[:, x_idx],
                        current_converted.data[:, series_idx],
                        label="Data",
                        c="b",
                    )

            ax.set_xlabel(current_parser.labels[x_idx])
            ax.set_ylabel(current_parser.labels[series_idx])
            ax.legend()
            fig.tight_layout()
            self.diff_canvas.draw()

    def on_copy_to_clipboard(self):
        if len(self._conversions) > 0:
            if len(self._conversions) != 1:
                warnings.warn(
                    "More than one conversion performed. Copying first conversion.",
                    stacklevel=2,
                )
            # Run the conversion on the first file.
            parser_type = type(self.parsers[0])
            match parser_type:
                case AU_PARSERS.MEX1_NEXAFS:
                    lines = AU_PARSERS.MEX1_to_QANT_AUMainAsc(self._conversions[0])
                case AU_PARSERS.MEX2_NEXAFS:
                    lines = AU_PARSERS.MEX2_to_QANT_AUMainAsc(self._conversions[0])
                case _:
                    raise NotImplementedError(
                        f"Conversion for this parser type {type(self.parsers[0])} has not been implemented."
                    )
            cb = QtGui.QGuiApplication.clipboard()
            cb.setText("".join(lines), mode=QtGui.QClipboard.Mode.Clipboard)
        else:
            # Do nothing
            pass

    def on_save_to_file(self):
        base_dir = self.dir_sel._folder_path
        init_override = False
        if len(self._conversions) > 0:
            self.progress.setRange(0, len(self._conversions))
            for parser in self._conversions:
                ptype = type(parser)
                converter: function
                match ptype:
                    case AU_PARSERS.MEX1_NEXAFS:
                        converter = AU_PARSERS.MEX1_to_QANT_AUMainAsc
                    case AU_PARSERS.MEX2_NEXAFS:
                        converter = AU_PARSERS.MEX2_to_QANT_AUMainAsc
                self.progress.setValue(self.progress.value() + 1)
                # Convert to ascii format.
                if self._rename_checkbox.isChecked() and (
                    "SR14ID01NEXSCAN:saveData_comment1" in parser.params
                    or "Comment 1" in parser.params
                    or "Sample" in parser.params
                ):
                    # Get the sample name
                    if "SR14ID01NEXSCAN:saveData_comment1" in parser.params:
                        sample = parser.params["SR14ID01NEXSCAN:saveData_comment1"]
                    elif "Comment 1" in parser.params:
                        sample = parser.params["Comment 1"]
                    elif "Sample" in parser.params:
                        sample = parser.params["Sample"]
                    else:
                        print(f"Name error, skipping {parser}")
                        continue

                    # Get the scan ID / number
                    if "mda_scan_number" in parser.params:
                        scan_id = parser.params["mda_scan_number"]
                    elif "Scan Number" in parser.params:
                        scan_id = parser.params["Scan Number"]
                    else:
                        # use the filename?
                        scan_id = "".join(parser.filename.split(".")[:-1])
                    # Create the new filename
                    file = f"{sample}_{scan_id}.asc"
                else:
                    file = "".join(parser.filename.split(".")[:-1]) + ".asc"
                path = os.path.join(base_dir, file)
                exists = os.path.exists(path)
                if self._override_checkbox.isChecked() or not exists:
                    if init_override is False and exists:
                        diag = QtWidgets.QMessageBox(self)
                        diag.setWindowTitle("Files Exist")
                        diag.setText(
                            f"File {file} already exists. Override?\n\nPath: {path}"
                        )
                        diag.setStandardButtons(
                            QtWidgets.QMessageBox.StandardButton.YesToAll
                            | QtWidgets.QMessageBox.StandardButton.Yes
                            | QtWidgets.QMessageBox.StandardButton.No
                        )
                        response = diag.exec()
                        match response:
                            case QtWidgets.QMessageBox.StandardButton.YesToAll:
                                init_override = True
                            case QtWidgets.QMessageBox.StandardButton.Yes:
                                with open(path, "w") as f:
                                    f.write("".join(converter(parser)))
                            case _:
                                pass
                    else:
                        with open(path, "w") as f:
                            f.write("".join(converter(parser)))
                elif os.path.exists(path):
                    diag = QtWidgets.QMessageBox(self)
                    diag.setWindowTitle("Files Exist")
                    diag.setText(
                        f"File {file} already exists.\nSelect checkbox to overwrite."
                    )
            self.progress.setValue(0)

    def calculate_nan(self, single_file=False) -> None:
        """
        Calculate the NaN conversions of the data.
        """
        parsers = self.parsers
        method = NaN_OPTION(self.nan_method.currentIndex())
        zero_as_nan = self.nan_zero_checkbox.isChecked()
        if single_file:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self._conversions = [
                nexafsParserConverter.calculation_single(
                    parsers[0], method, zero_as_nan
                )
            ]
        elif len(parsers) > 0:
            self._conversions = []
            self.progress.setRange(0, len(parsers))
            self.progress.setValue(0)
            for i, parser in enumerate(parsers):
                # Update progress bar
                self.progress.setValue(i + 1)
                copy_parser = nexafsParserConverter.calculation_single(
                    parser, method, zero_as_nan
                )
                self._conversions.append(copy_parser)
        # Finished calculating.
        self.progress.setValue(0)

    @staticmethod
    def calculation_single(
        parser: parser_base | scanBase,
        method: NaN_OPTION,
        treat_zero_as_nan: bool = False,
        zero_block_min_size=3,
    ) -> parser_base | scanBase:
        # Copy parser
        copy_parser = parser.copy()
        # Treat zeros as NaN values
        if treat_zero_as_nan:
            blocks = nexafsParserConverter._get_block_zeros(
                data=copy_parser.data, zero_block_min_size=zero_block_min_size
            )
            # Set blocks to NaN
            for i, j, width, height in blocks:
                copy_parser.data[i : i + width, j : j + height] = np.nan

        # Trim nan values from beginning and end of data - leave values in between.
        nan_vals = np.isnan(copy_parser.data)
        row_nan_vals = np.all(nan_vals, axis=1)
        # Find the first non-nan value
        start_idx = np.argmax(~row_nan_vals)
        # Find the last non-nan value
        end_idx = len(row_nan_vals) - np.argmax(~row_nan_vals[::-1])
        # Trim the data
        copy_parser.data = copy_parser.data[start_idx:end_idx]

        # Act on NaN values
        x_idx = parser.label_index(parser.COLUMN_ASSIGNMENTS["x"])
        match method:
            case NaN_OPTION.REMOVE_ROWS:
                # Remove data rows that contain nans:
                nan_idx = np.isnan(copy_parser.data).any(axis=1)
                # For row, delete if any nans are present.
                copy_parser.data = np.delete(copy_parser.data, nan_idx, axis=0)

            case NaN_OPTION.INTERPOLATE_2_POINTS:
                # Interpolate the nearest 2 points to fill in NaN values.
                for j, column in enumerate(copy_parser.data.T):
                    nan_values = np.isnan(column)
                    if nan_values.any():
                        if not (True in nan_values and False in nan_values):
                            # All NaN values, skip.
                            continue
                        # Uses linear interpolation to fill in NaN values.
                        if j == x_idx:
                            # Assume equal separation between x values
                            copy_parser.data[:, j][nan_values] = np.interp(
                                x=np.arange(len(copy_parser.data))[nan_values],
                                xp=np.arange(len(copy_parser.data))[~nan_values],
                                fp=copy_parser.data[:, j][~nan_values],
                            )
                            continue
                        else:
                            copy_parser.data[:, j][nan_values] = np.interp(
                                x=copy_parser.data[:, x_idx][nan_values],
                                xp=copy_parser.data[:, x_idx][~nan_values],
                                fp=copy_parser.data[:, j][~nan_values],
                            )
            case NaN_OPTION.INTERP_EXTRAP_2_POINTS:
                ### Extrapolate and interpolate the nearest 2 points to fill in NaN values.
                start = []
                end = []
                # Interpolation
                for j, column in enumerate(copy_parser.data.T):
                    nan_values = np.isnan(column)
                    ## Find column endpoints for extrapolation
                    # Find the first non-nan value
                    start_idx = np.argmax(~nan_values)
                    start.append(start_idx)
                    # Find the last non-nan value
                    end_idx = len(nan_values) - np.argmax(~nan_values[::-1])
                    end.append(end_idx)
                    # Uses linear interpolation to fill in NaN values.
                    if nan_values.any():
                        if j == x_idx:
                            # Assume equal separation between x values
                            copy_parser.data[:, j][nan_values] = np.interp(
                                x=np.arange(len(copy_parser.data))[nan_values],
                                xp=np.arange(len(copy_parser.data))[~nan_values],
                                fp=copy_parser.data[:, j][~nan_values],
                            )
                            continue
                        else:
                            # Calculate the average delta between x values
                            xdiff = np.abs(
                                np.diff(copy_parser.data[:, x_idx][~nan_values])
                            )
                            delta = np.mean(xdiff)
                            # Interpolate x data
                            copy_parser.data[:, j][nan_values] = np.interp(
                                x=copy_parser.data[:, x_idx][nan_values],
                                xp=copy_parser.data[:, x_idx][~nan_values],
                                fp=copy_parser.data[:, j][~nan_values],
                            )

                ### Extrapolation
                # Do x first...
                if j == x_idx and xdiff.std() < delta / 30:
                    # Assume data density consistent and delta can be used for x extrapolation
                    if start[x_idx] > 0:
                        points = start[x_idx]
                        first_val = copy_parser.data[start[x_idx], x_idx]
                        # increasing or decreasing?
                        sign = (
                            -1
                            if first_val > copy_parser.data[start[x_idx] + 1, x_idx]
                            else 1
                        )
                        copy_parser.data[: start[x_idx], x_idx] = np.linspace(
                            start=first_val - delta * points * sign,
                            stop=first_val - delta * sign,
                            num=points,
                        )
                    if end[x_idx] < len(copy_parser.data):
                        last_val = copy_parser.data[end[x_idx], x_idx]
                        copy_parser.data[end[x_idx] :, x_idx] = np.linspace(
                            start=last_val + delta,
                            stop=last_val
                            + delta * (len(copy_parser.data) - end[x_idx]),
                            num=len(copy_parser.data) - end[x_idx],
                        )
                else:
                    if len(copy_parser.data[start[x_idx] : end[x_idx]]) > 1:
                        # Start
                        y1 = copy_parser.data[start[x_idx] + 1, x_idx]
                        y2 = copy_parser.data[start[x_idx], x_idx]
                        sgrad = (y2 - y1) / 2
                        n = start[x_idx]
                        copy_parser.data[: start[x_idx], x_idx] = np.linspace(
                            start=copy_parser.data[start[x_idx], x_idx] + sgrad,
                            stop=copy_parser.data[start[x_idx], x_idx] + sgrad * n,
                            num=n,
                        )
                        # End
                        y1 = copy_parser.data[end[x_idx] - 2, x_idx]
                        y2 = copy_parser.data[end[x_idx] - 1, x_idx]
                        sgrad = (y2 - y1) / 2
                        n = len(copy_parser.data) - end[x_idx]
                        copy_parser.data[end[x_idx] :, x_idx] = np.linspace(
                            start=copy_parser.data[end[x_idx] - 1, x_idx] + sgrad,
                            stop=copy_parser.data[end[x_idx] - 1, x_idx] + sgrad * n,
                            num=n,
                        )

                # Now do other columns
                for j in range(len(start)):
                    if j == x_idx:
                        continue
                    else:
                        start_x = copy_parser.data[: start[j], x_idx]
                        end_x = copy_parser.data[end[j] :, x_idx]
                        # Extrapolate everything closest last two points.
                        if len(copy_parser.data[start[j] : end[j]]) > 1:
                            ## Valid range to extrapolate.
                            # Start
                            y1 = copy_parser.data[start[j] + 1, j]
                            y2 = copy_parser.data[start[j], j]
                            sgrad = (y2 - y1) / 2
                            n = start[j]
                            copy_parser.data[: start[j], j] = np.interp(
                                x=start_x,
                                xp=copy_parser.data[start[j] : start[j] + 2, x_idx],
                                fp=copy_parser.data[start[j] : start[j] + 2, j],
                            )
                            # End
                            # Valid range to extrapolate.
                            y1 = copy_parser.data[end[j] - 1, j]
                            y2 = copy_parser.data[end[j] - 2, j]
                            sgrad = (y2 - y1) / 2
                            n = len(copy_parser.data) - end[j]
                            copy_parser.data[end[j] :, j] = np.interp(
                                x=end_x,
                                xp=copy_parser.data[
                                    end[j] - 1 : end[j] - 3 : -1, x_idx
                                ],
                                fp=copy_parser.data[end[j] - 1 : end[j] - 3 : -1, j],
                            )
                        else:
                            # Do nothing, not enough values to extrapolate.
                            pass

            # case NaN_OPTION.INTERPOLATE_4:
            # Uses polynomial interpolation to fill in NaN values.
            # pass
            # case NaN_OPTION.INTERPOLATE_N:
            # pass
            case _:
                # Do nothing for NaN_OPTION.NONE
                pass
        return copy_parser

    def analyse_nan(self):
        """
        Calculate the number of NaN values in the data.
        """
        parsers = self.parsers
        if len(parsers) > 0:
            total_nan_values = 0
            nanfiles = 0
            for parser in parsers:
                nans = np.isnan(parser.data).sum()
                nanfiles += nans > 0
                total_nan_values += nans
            self.total_nan_files_indicator.setText(str(nanfiles))
            self.total_nan_instances_indicator.setText(str(total_nan_values))
            self.per_file_nan_indicator.setText(
                f"{total_nan_values / nanfiles:.2f}" if nanfiles > 0 else "0"
            )
        else:
            self.total_nan_files_indicator.setText("")
            self.total_nan_instances_indicator.setText("")
            self.per_file_nan_indicator.setText("")

    @staticmethod
    def _get_block_zeros(
        data: npt.NDArray, zero_block_min_size: int = 3
    ) -> list[tuple[int, int, int, int]]:
        """
        Collects the blocks of zeros in a 2D Numpy Array.

        While each block found may not be the largest possible block,
        the list will provide full coverage of the data.

        Parameters
        ----------
        data : npt.NDArray
            Data to find blocks of zeros in.
        zero_block_min_size : int, optional
            Minimum 1D block size, by default 3

        Returns
        -------
        list[tuple[int, int, int, int]]
            _description_
        """
        # Get zero locations
        zero_row, zero_col = np.where(data == 0)
        # Collect blocks
        blocks = []  # Filled with i,j,length,width tuples.
        for idx in range(len(zero_row)):
            i, j = zero_row[idx], zero_col[idx]
            width, height = 1, 1
            # Check for vertical blocks
            for r in range(idx + 1, len(zero_row)):
                # Check next row, plus any existing column
                if zero_row[r] == i + width and (
                    zero_col[r] >= j or zero_col[r] < j + height
                ):
                    width += 1
                # Check next column, plus any existing row
                if zero_col[r] == j + height and (
                    zero_row[r] >= i or zero_row[r] < i + width
                ):
                    height += 1
            # If the block is larger than 1x1, add to block list.
            if width > zero_block_min_size or height > zero_block_min_size:
                blocks.append((i, j, width, height))
        return blocks

    def analyse_block_zeros(self):
        parsers = self.parsers
        self.progress.setRange(0, len(parsers))
        zero_files = 0
        total_zeros = 0
        for i, parser in enumerate(parsers):
            self.progress.setValue(i)
            blocks = nexafsParserConverter._get_block_zeros(parser.data)
            # Update the zero files count
            if len(blocks) > 0:
                zero_files += 1
                # Update the zero instances count
                mask = np.zeros_like(parser.data, dtype=bool)
                for i, j, width, height in blocks:
                    mask[i : i + width, j : j + height] = True
                zeros = np.sum(mask)
            else:
                zeros = 0
            total_zeros += zeros
        # Update the zero indicators
        self.total_zero_files_indicator.setText(str(zero_files))
        self.total_zero_instances_indicator.setText(str(total_zeros))
        self.per_file_zero_indicator.setText(
            f"{total_zeros / zero_files:.2f}" if zero_files > 0 else "0"
        )

    def analyse_filesize(self) -> None:
        """
        Analyses the total filesize of the data.
        """
        self.files_num_indicator.setText(str(self.num_files))
        total_filesize = sum([parser.filesize for parser in self.parsers])
        i = 0
        while total_filesize > 1024:
            total_filesize /= 1024
            i += 1
        total_filesize = (
            f"{total_filesize:.2f} {['B', 'KB', 'MB', 'GB', 'TB', 'PB'][i]}"
        )
        self.files_size_indicator.setText(total_filesize)

    def analyse_parsers(self):
        """
        Runs analysis on the provided parsers, updating GUI elements.
        """
        self.analyse_nan()
        self.analyse_block_zeros()
        self.analyse_filesize()

    @property
    def parsers(self) -> list[Type[parser_base] | parser_base]:
        """
        The current set of parsers to use for conversion.

        Parameters
        ----------
        parsers : list[Type[parser_base]]
            A list of parser classes to use for conversion.
            None values are disregarded when setting.

        Returns
        -------
        list[Type[parser_base]]
            A list of parser classes to use for conversion.
        """
        return list(self._parsers.values())

    @parsers.setter
    def parsers(self, parsers: list[Type[parser_base] | parser_base]):
        self._parsers = {
            parser.filename: parser for parser in parsers if parser is not None
        }
        # Analysis the files
        self.analyse_parsers()
        # Calculates the NaN properties
        self.calculate_nan()
        # Updates UI
        self.diff_parser_selector.blockSignals(True)
        self.on_parsers_selected()
        self.diff_parser_selector.blockSignals(False)

    @property
    def conversions(self) -> list[Type[parser_base]]:
        return self._conversions

    @property
    def num_files(self):
        """
        Calculate the number of files in the data.
        """
        return len(self.parsers)

    # def on_recolour(self):
    #     """
    #     Recolour the division lines based on the theme.
    #     """
    #     # Get theme
    #     toolbar_palette = self.palette()
    #     light_theme_bool = toolbar_palette.window().color().lightnessF() > 0.5
    #     self.setStyleSheet(
    #         "background-color: " + ("black;" if light_theme_bool else "white;")
    #     )

    def event(self, event: QtCore.QEvent) -> bool:
        """
        Event handler for the widget.

        Adds palette change control for light/dark mode to QWidget event handler.

        Parameters
        ----------
        event : QtCore.QEvent
            The event to handle.

        Returns
        -------
        bool
            Whether the event was handled.
        """
        if (
            event.type() == QtCore.QEvent.Type.PaletteChange
            or event.type() == QtCore.QEvent.Type.ApplicationPaletteChange
        ):
            self.on_recolour()
        return super().event(event)


class nexafsConverterQANT(nexafsParserConverter):
    def __init__(self, parsers: list[Type[parser_base]] = [], parent=None):
        super().__init__(parsers, parent)
        self._title_label.setText("QANT Conversion:")

    pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = nexafsParserConverter()
    main.setWindowTitle("Converter")
    main.show()
    app.exec()
