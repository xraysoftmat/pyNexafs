import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QWidget,
    QSplitter,
)


from pyNexafs.gui.widgets.fileloader import nexafsFileLoader
from pyNexafs.gui.widgets.viewer import nexafsViewer


class browserWidget(QWidget):
    """A widget for browsing and loading NEXAFS files."""

    def __init__(self, parent=None, init_dir: str | None = None):
        super().__init__(parent=parent)
        # Initialise elements
        self._draggable = QSplitter(Qt.Orientation.Horizontal)
        self._main_layout = QHBoxLayout()
        self.loader = nexafsFileLoader(parent=self)
        self.viewer = nexafsViewer(parent=self)
        # self.converter = converterWidget()

        # Add to layout
        self._draggable.addWidget(self.loader)
        self._draggable.addWidget(self.viewer)
        self._main_layout.addWidget(self._draggable)
        # self.main_layout.addWidget(self.loader)
        # self.main_layout.addWidget(self.viewer)
        # self.sub_layout.addWidget(self.converter)
        self.setLayout(self._main_layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setContentsMargins(0, 0, 0, 0)

        ### Init UI
        # # TODO: Find another way to define a black line on handles with some margin two widgets.
        # grad = QLinearGradient(0, 0, 1, 0)
        # grad.setCoordinateMode(grad.CoordinateMode.ObjectMode)
        # for handle in self.draggable.findChildren(QSplitterHandle):
        #     if handle.parent().orientation() == Qt.Orientation.Horizontal:
        #         # handle.setStyleSheet("background-color: black;")
        #         # handle.setStyleSheet("qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 255, 255, 0), stop:0.407273 rgba(200, 200, 200, 255), stop:0.4825 rgba(101, 104, 113, 235), stop:0.6 rgba(255, 255, 255, 0));")
        #         # handle.setStyleSheet("qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);")

        #         handle.set

        #         handle.setMinimumWidth(1)
        #         handle.setMinimumHeight(1)
        #         handle.setFixedWidth(1)

        ### Connections
        # Load scans
        self.loader.selectionLoaded.connect(self._on_load_selection)
        # Remove scans with parser / directory change.
        self.loader.directory_selector.new_path.connect(self._on_dir_parser_change)
        self.loader.nexafs_parser_selector.currentIndexChanged.connect(
            self._on_dir_parser_change
        )
        self.loader.relabelling.connect(self.viewer.on_relabel)

        ### Init directory
        if init_dir is not None:
            self.loader.directory_selector.folder_path_edit.setText(init_dir)
            self.loader.directory_selector.folder_path_edit.editingFinished.emit()

    def _on_dir_parser_change(self):
        # Delete the current scan objects
        del self.viewer.scans

    def _on_load_selection(self):
        # Load in the scan objects
        selection_parse_objs = self.loader.loaded_parser_files_selection
        self.viewer.add_parsers_to_scans(selection_parse_objs)
        # Change the file selection in the viewer.
        self.viewer.selected_filenames = self.loader.selected_filenames


def gui(directory: str | None = None):
    app = QApplication(sys.argv)
    window = browserWidget(init_dir=directory)
    window.show()
    window.setWindowTitle("pyNexafs File Browser")
    app.exec()


if __name__ == "__main__":
    gui()
