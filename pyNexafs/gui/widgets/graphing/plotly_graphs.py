from PyQt6 import QtCore, QtWebEngineWidgets, QtWidgets
from PyQt6.QtWidgets import QApplication
import sys

import plotly.express as px


class PlotlyGraph(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.button = QtWidgets.QPushButton("Plot", self)
        self.browser = QtWebEngineWidgets.QWebEngineView(self)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addWidget(self.button, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        vlayout.addWidget(self.browser)

        self.button.clicked.connect(self.show_graph)
        self.resize(1000, 800)

    def show_graph(self):
        df = px.data.tips()
        fig = px.box(df, x="day", y="total_bill", color="smoker")
        fig.update_traces(
            quartilemethod="exclusive"
        )  # or "inclusive", or "linear" by default
        self.browser.setHtml(fig.to_html(include_plotlyjs="cdn"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotlyGraph()
    window.show()
    window.setWindowTitle("PlotlyGraph")
    sys.exit(app.exec())
