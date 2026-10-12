"""
Main entry point for the pyNexafs package.

Runs the GUI for the package if the dependencies are met.
Can be run from the command line with the command `python -m pyNexafs`.
Allowed arguments:
- `--help`: Show help message and exit.
- `[directory]`: Specify a directory to load NEXAFS files from.
- If no arguments are provided, the GUI will open pointed at the user folder '~'.
"""

import os
import sys
import traceback
from importlib.metadata import version

# Internal Imports
hasQT: bool
try:
    import PyQt6  # noqa: F401
    from PyQt6 import QtWidgets

    hasQT = True
except ImportError:
    hasQT = False


def main(dir_arg: str | None = None):
    dir_exists = os.path.isdir(dir_arg) if dir_arg else False
    # Check if the dependencies are met
    try:
        import PyQt6

        assert PyQt6 is not None
        print(f"PyQt6 ({version('PyQt6')}) is installed. Starting the GUI...")
    except ImportError as e:
        print(
            "PyQt6 is not installed. Please install PyQt6 to run the GUI."
            + f"\nTraceback: {e}"
        )
        exit()

    from pyNexafs.gui.data_browser import gui

    # Create and run the app
    if dir_arg and dir_exists:
        gui(directory=os.path.normpath(dir_arg))
    else:
        gui()


def main_with_traceback(dir_arg: str | None = None):
    try:
        main(dir_arg)
    except Exception as e:
        # Create a QT window to display the error
        app = QtWidgets.QApplication([])
        app.setApplicationName("kkcalc: Kramers-Kronig Calculator (Error)")
        error_dialog = QtWidgets.QErrorMessage()
        # Prepare the message: the error and the traceback
        msg = f"An error occurred, causing kkcalc to crash.:\
               \n{e!s}\
               \nPlease report this issue at https://github.com/xraysoftmat/kkcalc/issues"
        error_dialog.showMessage(msg)
        app.exec()

        # Add the detail
        tb = f"{traceback.format_exc()}"
        layout = error_dialog.layout()
        if layout is not None:
            layout.addWidget(QtWidgets.QLabel("Detailed traceback:"))
            layout.addWidget(QtWidgets.QLabel(tb))
        error_dialog.exec()


if __name__ == "__main__":
    args = sys.argv[1:]  # Get command line arguments, excluding the script name
    # Check if the script is run with the `--help` argument
    if "--help" in args:
        print(
            "pyNexafs: A Python package for analyzing NEXAFS data.\n"
            "Usage:\n"
            "\tpython -m pyNexafs [--help]\n"
            "\tpython -m pyNexafs [directory]\n"
            "Options:\n"
            "\t--help\tShow this help message and exit."
        )
        exit()

    if len(args) > 1:
        print("Error: Too many arguments. Use --help for usage information.")
        exit()

    dir_arg = args[0] if len(args) == 1 else None
