"""
LumenAiry Designer — PySide6 optical design application.

This subpackage provides the graphical user interface for the
lumenairy library, built on Qt 6 (PySide6).

Architecture
------------
- ``model.py``    — SystemModel: shared state (surfaces, wavelengths)
- ``element_table.py`` — Prescription spreadsheet editor
- ``layout_2d.py``     — Interactive 2-D system layout
- ``analysis.py``      — Dockable analysis windows (spot, ray fan, MTF)
- ``main_window.py``   — Application shell with menus, toolbars, docks

This
list named two files that no longer back the application -- a 370-line
prescription-spreadsheet editor superseded by ``element_table.py`` (zero
references repo-wide; DELETED) and ``workers.py`` (never present in this
tree -- each dock owns its own ``QThread`` worker).

Author: Andrew Traverso
"""
