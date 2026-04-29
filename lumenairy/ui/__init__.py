"""
Optical Designer — PySide6 optical design application.

This subpackage provides the graphical user interface for the
lumenairy library, built on Qt 6 (PySide6).

Architecture
------------
- ``model.py``    — SystemModel: shared state (surfaces, wavelengths)
- ``surface_table.py`` — Prescription spreadsheet editor
- ``layout_2d.py``     — Interactive 2-D system layout
- ``analysis.py``      — Dockable analysis windows (spot, ray fan, MTF)
- ``main_window.py``   — Application shell with menus, toolbars, docks
- ``workers.py``       — Background computation with progress signals

Author: Andrew Traverso
"""
