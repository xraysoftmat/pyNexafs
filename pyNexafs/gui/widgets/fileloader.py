from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QWidget,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QApplication,
    QLabel,
    QTextEdit,
    QCheckBox,
    QSplitter,
    QStyle,
    QProgressBar,
    QPushButton,
)

# from PyQt6.QtWidgets import QScrollBar, QHeaderView, QMainWindow, QTableWidget, QFrame, QGridLayout, QSizeGrip
from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
)
import sys
from pyNexafs.parsers import parser_loaders, parser_base
from typing import Type
from pyNexafs.gui.widgets.io.dir_selection import directorySelector
from pyNexafs.gui.widgets.io.fileviewer import (
    directoryViewerTableNew,
    SummaryParamSelectorDialog,
)


class nexafsFileLoader(QWidget):
    """
    Header-viewer and file loader for NEXAFS data.

    Includes a variety of elements for selecting a directory, parser,
    filetype filters and header-contents filters. Filtering and sorting
    is implemented through QSortFilterProxyModel.


    Attributes
    ----------
    selectionLoaded : pyqtSignal
        Signal for selection change and completion of consequent file loading.
        Does not trigger if selection changes while no parser is selected.
    """

    selectionLoaded = pyqtSignal(bool)
    """A signal to determine when the """

    relabelling = pyqtSignal(bool)
    """A signal for changes if smart `relabelling` is to be applied to field names.
    This can be specified by the parsing classes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        ## Instance attributes
        self._log_text = ""

        ## Initialise elements
        self.directory_selector = directorySelector(parent=self)
        self._progress_bar = QProgressBar()
        self.directory_viewer = directoryViewerTableNew(
            parent=self, progress_bar=self._progress_bar
        )
        self.nexafs_parser_selector = nexafsParserSelector()
        self.filter_relabelling = QCheckBox()
        self.filter = directoryFilterWidget(parent=self)
        self.log = QTextEdit()

        # Initalise graphics for loaded / unloaded files.
        self._icon_gear = QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_FileDialogDetailedView
        )
        self._select_params_btn = QPushButton(icon=self._icon_gear, text=None)
        self._select_params_btn.setToolTip("Select parser displayed parameters")

        ## Subplayouts
        dir_parser_layout = QHBoxLayout()
        dir_parser_layout.addWidget(self.directory_selector)
        dir_parser_layout.addWidget(QLabel("Parser:"))
        dir_parser_layout.addWidget(self.nexafs_parser_selector)
        dir_parser_layout.addWidget(QLabel("Relabel:"))
        dir_parser_layout.addWidget(self.filter_relabelling)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.filter)
        filter_layout.addWidget(self._select_params_btn)

        ## Qsplitter for expandable log
        draggable = QSplitter(Qt.Orientation.Vertical)
        draggable.addWidget(self.directory_viewer)
        draggable_log_widget = QWidget()
        draggable_log = QVBoxLayout()
        draggable_log.addWidget(QLabel("Log:"))
        draggable_log.addWidget(self.log)
        draggable_log_widget.setLayout(draggable_log)
        draggable_log_widget.setContentsMargins(0, 0, 0, 0)
        draggable.addWidget(draggable_log_widget)

        ## Dir loading progress bar
        self._progress_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._progress_bar.setFixedHeight(15)
        self._progress_bar.setTextVisible(False)

        ## Directory Layout
        dir_layout = QVBoxLayout()
        dir_layout.addLayout(dir_parser_layout)
        dir_layout.addLayout(filter_layout)
        dir_layout.addWidget(self._progress_bar)
        dir_layout.addWidget(draggable)

        # Assign viewer layout to widget.
        self.setLayout(dir_layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            dir_layout.setContentsMargins(0, 0, 0, 0)

        ## Element attributes
        self.log.setReadOnly(True)
        self.directory_viewer.directory = (
            self.directory_selector._folder_path
        )  # Match initial value.
        draggable.setStretchFactor(0, 5)
        draggable.setStretchFactor(1, 1)

        ## Element Connections
        # Reload directory upon path change.
        self.directory_selector.new_path.connect(self.update_dir)
        # Reload directory upon parser change.
        self.nexafs_parser_selector.currentIndexChanged.connect(self.update_parser)
        # Reload directory upon filter change.
        self.filter.filterChanged.connect(self.update_filters)
        # Update relabelling
        self.filter_relabelling.stateChanged.connect(self.update_relabelling)
        # Signal for selection change.
        self.directory_viewer.selection_loaded.connect(self.on_selection_loaded)
        # Signal for selecting parser parameters.
        self._select_params_btn.clicked.connect(self.on_select_params)

        # Initialize directory.
        self._log_entry()

    def on_select_params(self):
        # Create a SummaryParamsSelectionDialog
        dialog = SummaryParamSelectorDialog(
            parsers=list(self.loaded_parser_headers.values()),
            parent=None,
            existing_selection=self.directory_viewer.header,
        )
        result = dialog.exec()
        if result:
            self.directory_viewer.header = sorted(list(dialog.selection))

    def on_selection_loaded(self):
        """
        Signal for selection change.
        """
        self.selectionLoaded.emit(True)
        return

    def update_relabelling(self) -> None:
        """
        Updates the fieldnames when relabelling is checked.
        """
        parser = self.directory_viewer.parser
        if parser is not None:
            old_relabel = parser.relabel
            new_relabel = self.filter_relabelling.isChecked()
            if old_relabel != new_relabel:
                parser.relabel = new_relabel
                self.directory_viewer.update_header()
                self.relabelling.emit(parser.relabel)

    def update_dir(self) -> None:
        """
        Update the dir selection
        """
        new_dir = self.directory_selector._folder_path
        # Update directory viewer
        self.directory_viewer.directory = new_dir
        # Log file updates.
        self._log_entry()
        return

    def update_parser(self) -> None:
        """
        Update the parser selection.
        """
        new_parser = self.nexafs_parser_selector.current_parser
        # Disable filter signalling while parser is changed.
        self.filter._changing_parser = True
        # Update directory viewer and filter parser selection
        self.filter.parser = new_parser
        self.directory_viewer.parser = new_parser
        self.filter._changing_parser = False
        self._log_entry()
        return

    def update_filters(self):
        """
        Update the directory viewer.
        """
        # Push an update to the directory viewer.
        self.directory_viewer.filters = (
            self.filter.filter_text,
            self.filter.filetypes_selection,
        )
        self._log_entry()
        return

    def _log_entry(self):
        # Log the current selection
        self._log_selection()
        # Log the number of files.
        self._log_files()

    def _log_selection(self):
        """
        Logs the path, parser and filter selection.
        """
        # Log path selection
        path = self.directory_selector._folder_path
        self.log_text += (
            f"Path '{path}' loaded"
            if len(path) < 30
            else f"Path '{path[:13]}...{path[-13:]}' loaded"
        )

        # Log parser selection
        parser = self.nexafs_parser_selector.current_parser  # current parser selection
        if parser is not None:
            self.log_text += f" using parser '{parser.__name__}'"

        # Log Filetype Selection?
        filetypes = self.filter.filetypes_selection
        if len(filetypes) == 1:
            self.log_text += f" with filetype '{filetypes[0]}'"

        # Log Filter Selection
        filter_text = self.filter.filter_text
        if filter_text != "" and filter_text is not None:
            self.log_text += f" with filter '{filter_text}'"

        # End line
        self.log_text += ". "  # no whitespace, assuming log files will be called next.
        return

    def _log_files(self):
        """
        Logs the number of files located after updating parameters.
        """
        file_count = self.directory_viewer.verticalHeader().count()  # filtered viewport
        dv_f = self.directory_viewer.files
        dv_ph = self.directory_viewer._parser_headers
        dv_p = self.directory_viewer.parser
        # Loading method is phrased differently depending on if a parser is selected.
        method = "parsed" if dv_p is not None and len(dv_ph) >= 0 else "located"
        # Determine endpoint loading.
        if file_count != 0 and len(dv_f) > 0:  # Filtered files loaded
            self.log_text += f"{file_count} {method} files.\n"
        elif file_count != 0 and file_count > 0:  # Filtered files
            self.log_text += f"Located {file_count} files.\n"
        else:
            self.log_text += "No files located.\n"

    @property
    def log_text(self):
        return self._log_text

    @log_text.setter
    def log_text(self, text: str):
        # Check if at max height
        sb = self.log.verticalScrollBar()
        atMaxHeight = True if sb.maximum() - sb.value() < 30 else False
        prevHeight = sb.value()
        # Update log
        self._log_text = text
        self.log.setText(text)
        # If was at max height, return to new max height.
        sb.setValue(sb.maximum() if atMaxHeight else prevHeight)

    @log_text.deleter
    def log_text(self):
        self._log_text = ""
        self.log.setText("")

    @property
    def loaded_parser_headers(self) -> dict[str, Type[parser_base] | None]:
        """
        Returns the parser headers loaded in the directory viewer.

        Returns
        -------
        dict[str, Type[parser_base]]
            A dictionary of filenames and their corresponding parser headers.
        """
        return self.directory_viewer._parser_headers

    @property
    def loaded_parsers_files(self) -> dict[str, Type[parser_base] | None]:
        """
        Returns the full-file parsers loaded in the directory viewer.

        Returns
        -------
        dict[str, Type[parser_base]]
            A dictionary of filenames and their corresponding parser objects.
        """
        return self.directory_viewer._parser_files

    @property
    def loaded_parser_files_selection(self) -> dict[str, Type[parser_base] | None]:
        """
        Returns the selection subset of full-file parsers from the directory viewer.

        Returns
        -------
        dict[str, Type[parser_base]]
            A dictionary of filenames and their corresponding scan objects.
        """
        selected = self.directory_viewer.selected_filenames
        loaded = self.loaded_parsers_files
        return {
            name: loaded[name]
            for name in selected
            if name in loaded and loaded[name] is not None
        }

    @property
    def selected_filenames(self) -> list[str]:
        """
        Returns the selected filenames in the directory viewer.

        Returns
        -------
        list[str]
            A list of filenames selected in the directory viewer.
        """
        return self.directory_viewer.selected_filenames


class nexafsParserSelector(QComboBox):
    """
    Selector for the type of NEXAFS data to load.
    """

    def __init__(self):
        super().__init__()

        if "" in parser_loaders:
            raise ValueError("Empty string is a reserved parser name.")

        # Define the relationships
        self.parsers = {"": None}
        self.parsers.update(parser_loaders)
        # Add items to the combobox.
        self.addItems([key for key in self.parsers.keys()])

    @property
    def current_parser(self) -> type[parser_base] | None:
        """
        Current selected parser.

        Returns
        -------
        type[parser_base] | None
            Returns the current parser which inherits from parser_base, or None if no parser is selected.
        """
        return self.parsers[self.currentText()]

    def update_combo_list(self):
        """
        Update the combo box parser list from the parser attribute.
        """
        self.clear()
        self.addItems([key for key in self.parsers.keys()])


class directoryFilterWidget(QWidget):
    """
    A widget for filtering the directory based on text and/or filetype.

    Generates a QTextEdit & QComboBox for filtering by text and/or filetype
    respectively. Default filetype options for a selected parser are populated
    by the parser.ALLOWED_EXTENSIONS constant.

    Attributes
    ----------
    filterChanged : pyqtSignal
        Emits upon a change in filter, useful for triggering table update.
    """

    filterChanged = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        layout = QHBoxLayout()
        self.setLayout(layout)

        # Update margins to be compact if a parent is provided.
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            layout.setContentsMargins(0, 0, 0, 0)

        # Instance attributes
        self.filter_text = ""
        self.filter_filetype = None
        self._parser_selection = None
        self._changing_parser = False  # Flag to prevent filterChanged signal emitting when parser is changed.

        # Instance widgets
        self.filter_text_edit = QLineEdit()
        self.filter_filetype_select = QComboBox()

        # Setup layout
        layout.addWidget(self.filter_text_edit)
        layout.addWidget(QLabel("Filetype: "))
        layout.addWidget(self.filter_filetype_select)

        # Widget Attributes
        self.filter_text_edit.setPlaceholderText("Filter filename|header by text")

        # Connections
        self.filter_text_edit.editingFinished.connect(self.on_filter_text_edit)
        self.filter_filetype_select.currentIndexChanged.connect(self.on_filter_change)

    def on_filter_text_edit(self):
        """
        Update the filter text.
        """
        self.filter_text = self.filter_text_edit.text()
        self.on_filter_change()

    def on_filter_change(self):
        """
        Signal generator for filter changes if parser not changing.
        """
        if not self._changing_parser:
            self.filterChanged.emit(True)

    @property
    def filetypes_selection(self) -> list[str]:
        """
        Returns the currently selected filetype.

        Returns
        -------
        str
            The currently selected filetype, or a list of all possible filetypes, belonging to the parser.
            If parser is None, returns an empty list.
        """
        if self.parser is None:
            return []
        selection = self.filter_filetype_select.currentText()
        return [selection] if selection != "" else self.parser.ALLOWED_EXTENSIONS

    @property
    def parser(self) -> type[parser_base] | None:
        """
        Returns the currently selected parser.

        Returns
        -------
        type[parser_base] | None
            The currently used parser, or None if no parser is selected.
        """
        return self._parser_selection

    @parser.setter
    def parser(self, parser: type[parser_base] | None):
        """
        Sets the (external) parser selection.

        Additionally triggers the filetype selection to be updated and repopulates the filetype selection entries.

        Parameters
        ----------
        parser : type[parser_base] | None
            Currently used parser, or None if no parser is selected.
        """
        self._changing_parser = True
        self._parser_selection = parser
        self.filter_filetype_select.clear()  # clear existing file extensions.
        if parser is not None:
            self.filter_filetype_select.addItem("")
            self.filter_filetype_select.addItems(parser.ALLOWED_EXTENSIONS)
        self._changing_parser = False

    @parser.deleter
    def parser(self):
        """
        Removes the (external) parser selection.

        Additionally clears the filetype selections.
        """
        self._parser_selection = None
        self.filter_filetype_select.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = nexafsFileLoader()
    window.show()
    window.setWindowTitle("pyNexafs File Loader")
    sys.exit(app.exec())
