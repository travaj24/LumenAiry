#!/usr/bin/env python
"""
Launch the LumenAiry Designer application.

Preferred launcher (3.5.9+).  ``run_optical_designer.py`` remains
available as a backward-compatibility alias for users with
existing shortcuts.

Usage
-----
::

    python run_lumenairy_designer.py
    python run_lumenairy_designer.py --demo            # load a demo lens
    python run_lumenairy_designer.py path/to/file.zmx  # open a .zmx file
    python run_lumenairy_designer.py path/to/file.txt  # open a prescription text file

Author: Andrew Traverso
"""

from run_optical_designer import main


if __name__ == '__main__':
    main()
