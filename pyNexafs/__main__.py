"""
Main entry point for the pyNexafs package.

Runs the GUI for the package if the dependencies are met.
Can be run from the command line with the command `python -m pyNexafs`.
Allowed arguments:
- `--help`: Show help message and exit.
- `[directory]`: Specify a directory to load NEXAFS files from.
- If no arguments are provided, the GUI will open pointed at the user folder '~'.
"""

import sys
import os

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
    dir_exists = os.path.isdir(dir_arg) if dir_arg else False

    # Check if the dependencies are met
    try:
        import PyQt6
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
