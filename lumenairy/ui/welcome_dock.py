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
    insert_template_requested = Signal(str)  # template kind name

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

        # ── 3.7.9: example designs row ──
        # One-click loaders for common multi-element templates --
        # same builders the Insert > From Template menu uses.
        # Lets new users see a non-trivial design in 1 click without
        # having to know the Insert menu structure.
        ex_label = QLabel('Example designs')
        ex_label.setFont(qs_font)
        outer.addWidget(ex_label)
        ex_row = QHBoxLayout()
        for text, kind in [
            ('Cemented doublet',  'cemented_doublet'),
            ('Plossl eyepiece',   'plossl'),
            ('Petzval objective', 'petzval'),
            ('Kepler telescope',  'kepler_telescope'),
            ('4-f relay',         '4f_relay'),
        ]:
            b = QPushButton(text)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(
                lambda checked=False, k=kind:
                self.insert_template_requested.emit(k))
            ex_row.addWidget(b)
        outer.addLayout(ex_row)

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

    def set_recent_files(self, entries):
        """Replace the recent-files list (most recent first).

        3.7.9: ``entries`` may now be either a list of strings (the
        legacy format, paths only) or a list of ``(path, timestamp)``
        tuples (timestamp is an ISO-8601 string or a float).  When a
        timestamp is present it's rendered as a relative age tag
        ("2h ago", "3d ago") next to the file name.
        """
        import datetime as _dt
        import time as _time
        self.recent_list.clear()
        if not entries:
            it = QListWidgetItem('(no recent files yet)')
            it.setFlags(Qt.NoItemFlags)
            self.recent_list.addItem(it)
            return
        now = _time.time()
        for entry in entries:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                p, ts = entry[0], entry[1]
            else:
                p, ts = entry, None
            try:
                if isinstance(ts, str):
                    dt = _dt.datetime.fromisoformat(ts)
                    ts_epoch = dt.timestamp()
                else:
                    ts_epoch = float(ts) if ts is not None else None
            except Exception:
                ts_epoch = None
            age = ''
            if ts_epoch is not None:
                dt_sec = max(0.0, now - ts_epoch)
                if dt_sec < 60:
                    age = ' (just now)'
                elif dt_sec < 3600:
                    age = f' ({int(dt_sec / 60)}m ago)'
                elif dt_sec < 86400:
                    age = f' ({int(dt_sec / 3600)}h ago)'
                elif dt_sec < 86400 * 30:
                    age = f' ({int(dt_sec / 86400)}d ago)'
                else:
                    age = f' ({int(dt_sec / (86400 * 30))}mo ago)'
            display = f'{os.path.basename(p)}{age}    —    {p}'
            it = QListWidgetItem(display)
            it.setData(Qt.UserRole, p)
            it.setToolTip(p + (f'\nLast opened: {ts}'
                                if ts is not None else ''))
            self.recent_list.addItem(it)

    def _on_recent_activated(self, item):
        path = item.data(Qt.UserRole)
        if path:
            self.open_path_requested.emit(path)

    def minimumSizeHint(self):
        """v5.4.4 (audit GUI-resize round 2): report a tiny minimum so
        the QDockWidget will let the user drag this dock pane down to
        almost nothing.  Inherited Qt implementation walks layout
        children (matplotlib canvas, tables, toolbars) and adds up
        their hints, producing a floor that locks the bottom dock
        area on non-Design tabs.  Matches the v3.6.1 fix in
        layout_2d.py / layout_3d.py.
        """
        from PySide6.QtCore import QSize
        return QSize(40, 40)

    def sizeHint(self):
        """v5.4.4: companion to minimumSizeHint() above.  Provides a
        reasonable initial size when the dock is first shown."""
        from PySide6.QtCore import QSize
        return QSize(400, 200)
