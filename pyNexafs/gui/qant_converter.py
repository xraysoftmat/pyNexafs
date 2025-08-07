import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QSplitter,
)
from PyQt6 import QtWidgets
from pyNexafs.gui.widgets.fileloader import nexafsFileLoader
from pyNexafs.gui.widgets.converter import nexafsConverterQANT


class converterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._main_layout = QHBoxLayout()
        self.setLayout(self._main_layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setContentsMargins(0, 0, 0, 0)

        ## Copied from browserWidget
        # Initialise elements
        self._draggable = QSplitter(Qt.Orientation.Horizontal)
        self.loader = nexafsFileLoader(parent=self)
        # Add to layout
        self._draggable.addWidget(self.loader)
        self._main_layout.addWidget(self._draggable)
        ### Connections
        # Load scans
        self.loader.selectionLoaded.connect(self._on_load_selection)
        ##

        # Remove parsers other than MEX2 from the loader.
        self.loader.nexafs_parser_selector.blockSignals(True)
        for key in list(self.loader.nexafs_parser_selector.parsers.keys()):
            if key not in [
                "au MEX1:NEXAFS",
                "au MEX2:NEXAFS",
                "",  # Empty key is used for GUI init.
            ]:
                self.loader.nexafs_parser_selector.parsers.pop(key)
        self.loader.nexafs_parser_selector.update_combo_list()
        self.loader.nexafs_parser_selector.blockSignals(False)

        # Add converter widget to the draggable.
        self.converter = nexafsConverterQANT(parent=self)
        self.converter.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )

        # Add to layout
        self._draggable.addWidget(self.converter)

    def _on_load_selection(self):
        # Load in the parser data
        self.converter.parsers = self.loader.loaded_parser_files_selection.values()


def gui():
    app = QApplication(sys.argv)
    window = converterWidget()
    window.show()
    window.setWindowTitle("pyNexafs MEX2 - QANT Converter")
    app.exec()


if __name__ == "__main__":
    gui()
