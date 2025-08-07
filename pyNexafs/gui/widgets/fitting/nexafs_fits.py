"""
Module to perform graphical NEXAFS fitting
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas, NEXAFS_NavQT
from pyNexafs.nexafs import scanBase
from matplotlib.figure import Figure


class globalFitter(QWidget):
    """
    Global fitting class for a set of NEXAFS spectrums.
    """

    def __init__(self, parent: QWidget | None = None, scans: list[scanBase] = []):
        # Initialise widget
        super().__init__(parent=parent)

        # Setup layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Setup margins if has a parent.
        if parent is not None:
            layout.setContentsMargins(0, 0, 0, 0)

        # Save the scan list
        self._scans = scans.copy()

        # Setup the UI elements
        fig = Figure()
        fig_canvas = FigureCanvas(fig)
        fig_canvas_nav = NEXAFS_NavQT(canvas=fig_canvas)

        # Add elements to the UI
        layout.addWidget(QLabel("GLobal Fitting"))
        layout.addWidget(fig_canvas)
        layout.addWidget(fig_canvas_nav)


# Example
import sys
import os
from PyQt6.QtWidgets import QApplication
from pyNexafs.parsers.au.aus_sync.MEX2 import MEX2_NEXAFS

if __name__ == "__main__":
    # Load some example parsers
    base_dir = os.getcwd()
    f1 = r"tests\\test_data\\au\\MEX2\\MEX2_5640.mda"
    f2 = r"tests\\test_data\\au\\MEX2\\MEX2_5642.mda"
    files = [os.path.normpath(os.path.join(base_dir, f)) for f in [f1, f2]]
    energy_bins = (2135, 2560)  # eV
    parsers = [
        MEX2_NEXAFS(f, header_only=False, energy_bin_domain=energy_bins) for f in files
    ]
    scans = [p.to_scan() for p in parsers]

    app = QApplication(sys.argv)
    window = globalFitter(parent=None, scans=scans)
    window.show()
    window.setWindowTitle("Summary Param Selector")
    sys.exit(app.exec())
