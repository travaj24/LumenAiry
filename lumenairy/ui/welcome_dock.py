"""
WelcomeDock — empty-state guidance shown on the Design tab.

Provides quick-start buttons and a list of recent files so a fresh
launch isn't a wall of empty docks.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

import os


class WelcomeDock(QWidget):
    """Friendly landing panel: Recent Files + Quick Start buttons."""

    open_path_requested = Signal(str)     # emit a file path to open
    insert_singlet_requested = Signal()
    insert_achromat_requested = Signal()
    open_demo_requested = Signal()
    browse_library_requested = Signal()
    show_shortcuts_requested = Signal()
    show_repl_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('welcome_widget')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(14)

        title = QLabel('Welcome to LumenAiry Designer')
        f = title.font()
        f.setPointSize(f.pointSize() + 4)
        f.setBold(True)
        title.setFont(f)
        outer.addWidget(title)

        subtitle = QLabel(
            'Build, analyze, optimize, and validate optical systems '
            'interactively, with the full lumenairy library on tap.\n'
            'Drop any .zmx / .seq / .txt / .json file onto this window '
            'to load.  Have a Python prescription?  Open the Python '
            'Console (REPL) dock and call '
            '<code>model.load_prescription(rx)</code>.')
        subtitle.setStyleSheet('color: #97a8c2;')
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        # ── Hero: open the demo lens (3.6 redesign) ──
        # The single most valuable first action for new users.
        # Made larger and more prominent than the secondary row.
        hero = QPushButton('▶  Open Demo (AC254-100-C)')
        hero.setObjectName('run_button')
        hero_font = hero.font()
        hero_font.setPointSize(hero_font.pointSize() + 2)
        hero_font.setBold(True)
        hero.setFont(hero_font)
        hero.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hero.setMinimumHeight(40)
        hero.clicked.connect(self.open_demo_requested.emit)
        outer.addWidget(hero)

        # ── Secondary row ──
        qs_label = QLabel('Or jump straight in')
        qs_font = qs_label.font()
        qs_font.setBold(True)
        qs_label.setFont(qs_font)
        outer.addWidget(qs_label)

        qs_row = QHBoxLayout()
        for text, sig in [
            ('Insert Singlet',           self.insert_singlet_requested),
            ('Insert Achromat',          self.insert_achromat_requested),
            ('Browse Library',           self.browse_library_requested),
            ('Open Python REPL',         self.show_repl_requested),
            ('Keyboard Shortcuts',       self.show_shortcuts_requested),
        ]:
            b = QPushButton(text)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(sig.emit)
            qs_row.addWidget(b)
        outer.addLayout(qs_row)

        # ── Recent files ──
        rec_label = QLabel('Recent files')
        rec_label.setFont(qs_font)
        outer.addWidget(rec_label)

        self.recent_list = QListWidget()
        self.recent_list.setStyleSheet('QListWidget { font-family: Consolas; }')
        self.recent_list.itemActivated.connect(self._on_recent_activated)
        outer.addWidget(self.recent_list, stretch=1)

        # Subtle horizontal rule + tip footer.
        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setFrameShadow(QFrame.Sunken)
        outer.addWidget(rule)

        tip = QLabel(
            'Tip — Ctrl+1..6 jump between workspace tabs.  '
            'Right-click any tab to manage its docks.')
        tip.setStyleSheet('color: #6a7e98;')
        outer.addWidget(tip)

    # ------------------------------------------------------------------
    #  Recent-file list
    # ------------------------------------------------------------------

    def set_recent_files(self, paths):
        """Replace the recent-files list (most recent first)."""
        self.recent_list.clear()
        if not paths:
            it = QListWidgetItem('(no recent files yet)')
            it.setFlags(Qt.NoItemFlags)
            self.recent_list.addItem(it)
            return
        for p in paths:
            display = f'{os.path.basename(p)}    —    {p}'
            it = QListWidgetItem(display)
            it.setData(Qt.UserRole, p)
            it.setToolTip(p)
            self.recent_list.addItem(it)

    def _on_recent_activated(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.open_path_requested.emit(path)
