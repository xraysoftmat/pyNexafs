"""
This module contains the `directory_selector` class.

This allows the selection of a filepath.
"""

import os
import sys

from PyQt6 import QtCore, QtGui, QtWidgets

# Extras for __main__ testing
from PyQt6.QtWidgets import QApplication


class directorySelector(QtWidgets.QWidget):
    """
    Module for selecting a directory path.

    Highlights the text

    Attributes
    ----------
    newPath : QtCore.pyqtSignal
        Generates a signal when a new path entry is created.
    folder_path:
    """

    new_path = QtCore.pyqtSignal(bool)
    """A signal to indicate the selection of a new (valid) path."""

    # String constants for widget elements
    edit_description = "Path to load NEXAFS data from."
    dialog_caption = "Select NEXAFS Directory"

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        # Update margins to be compact if a parent is provided.
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            layout.setContentsMargins(0, 0, 0, 0)

        # Instance attributes
        self._folder_path = os.path.expanduser("~")  # default use home path.
        """The last valid directory selection"""

        self._folder_path = directorySelector.format_path(
            self._folder_path
        )  # adds final slash if not present.

        ### Instance widgets
        # Refresh button / toggle
        refresh_icon = QtGui.QIcon.fromTheme("view-refresh")
        self.refresh_button = QtWidgets.QPushButton(refresh_icon, "")
        self.refresh_button.clicked.connect(self.refresh_dir)
        # Folder path
        self.folder_path_edit = QtWidgets.QLineEdit()
        self.folder_path_edit.setText(self._folder_path)
        self.folder_path_edit.accessibleName = "Directory"
        self.folder_path_edit.accessibleDescription = self.edit_description
        self.folder_path_edit.editingFinished.connect(self.validate_edit_path)
        self.folder_path_edit_default_stylesheet = self.folder_path_edit.styleSheet()
        # Folder select button
        self.folder_select_button = QtWidgets.QPushButton("Browse")
        self.folder_select_button.clicked.connect(self.select_path)

        # Setup layout
        layout.addWidget(self.folder_path_edit)
        layout.addWidget(self.folder_select_button)
        layout.addWidget(self.refresh_button)

    def refresh_dir(self):
        """
        Re-emits a change signal for the current directory.
        """
        self.new_path.emit(True)

    @property
    def folder_path(self):
        """
        The direction selection of the selector widget.

        Parameters
        ----------
        path : str
            A directory string.

        Returns
        -------
        path : str
            The most recent valid directory.

        Raises
        ------
        ValueError
            If the
        """
        return self._folder_path

    @folder_path.setter
    def folder_path(self, path: str):
        # Hold old path
        old_path = self._folder_path
        # Validate the path
        v = self.validate_path(path)
        if v:
            formatted_path = self.format_path(path)
            # Check if the path is an update
            if formatted_path != old_path:
                # Update path
                self._folder_path = formatted_path
                # Update the UI
                self.folder_path_edit.setText(formatted_path)
                # Emit a path change
                self.new_path.emit(True)
            # Otherwise do nothing.
        else:
            raise ValueError(f"The path `{path}` does not exist.")

    def select_path(self) -> None:
        """
        Generates a file dialog to select a directory, then updates the path.
        """
        path = QtWidgets.QFileDialog.getExistingDirectory(
            parent=None,
            caption=self.dialog_caption,
            directory=self._folder_path,
            options=QtWidgets.QFileDialog.Option.ShowDirsOnly,
        )
        path = None if path == "" else path
        if path is None:
            # invalid path
            self.folder_path_edit.setText(
                self._folder_path
            )  # reset the text to the original path.
        else:
            # set editable path and validate.
            self.folder_path_edit.setText(path)
            self.validate_edit_path()

    @staticmethod
    def validate_path(path: str) -> bool:
        """
        Checks the validity of the selected path.

        Requires the path to exist.

        Parameters
        ----------
        path : str
            The path seeking to be used
        """
        return os.path.isdir(path)

    def validate_edit_path(self) -> None:
        """
        Validates the widget text edit path.

        Only updates internal path if the path is valid, otherwise sets background color to red.
        """
        # Get path text
        editable_path = self.format_path(self.folder_path_edit.text())

        # Check validity
        if editable_path is None:
            # invalid path
            self.folder_path_edit.setText(
                self.folder_path
            )  # reset the text to the original path.
            return

        if not self.validate_path(editable_path):
            # invalid path
            self.folder_path_edit.setStyleSheet("background-color: red;")
        else:
            # Valid path!
            # Reset the stylesheets.
            self.folder_path_edit.setStyleSheet(
                self.folder_path_edit_default_stylesheet
            )
            # Use the property to update the path and UI
            self.folder_path = editable_path

    @staticmethod
    def format_path(path: str) -> str:
        """
        Formats a path string to be consistent.

        Parameters
        ----------
        path : str
            The incoming path string. Must be a valid path, tested by os.path.isdir().

        Returns
        -------
        str
            A formatted path string that always contains a trailing slash, with slashes matching OS type.
        """

        # Strip whitespace, ensure tailing slash, and convert mixed slashes to forward slashes.
        formatted_path = os.path.join("", path.strip(), "").replace("\\", "/")
        # Remove redundant slash duplicates
        while "//" in formatted_path:
            formatted_path = formatted_path.replace("//", "/")
        # Convert slashes to match OS
        slashes = os.sep
        formatted_path = formatted_path.replace("/", slashes)
        # Add final slash if not present.
        if not formatted_path.endswith(slashes):
            formatted_path += slashes
        return formatted_path


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = directorySelector()
    window.show()
    window.setWindowTitle("Directory Selector")
    sys.exit(app.exec())
