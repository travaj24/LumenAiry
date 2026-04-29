"""
Materials dock — glass map + library in one place.

Replaces the split "Glass Map" and "Library" docks with a tabbed
container so the Abbe diagram, glass catalog, and user-saved
materials live under a single entry point.  The original dock
widgets are reused unchanged.

Author: Andrew Traverso
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget


class MaterialsDock(QWidget):
    """Glass map + user library in one tabbed container."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Re-use the existing GlassMapDock / LibraryDock widgets.  If
        # either fails to import (dependencies missing, e.g. matplotlib)
        # we silently drop that tab.
        try:
            from .glass_map_dock import GlassMapDock
            self.glassmap_widget = GlassMapDock(self.sm)
            self.tabs.addTab(self.glassmap_widget, 'Glass Map (Abbe)')
        except Exception:
            self.glassmap_widget = None

        try:
            from .library_dock import LibraryDock
            self.library_widget = LibraryDock(self.sm)
            self.tabs.addTab(self.library_widget, 'User Library')
        except Exception:
            self.library_widget = None
