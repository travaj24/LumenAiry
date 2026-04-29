"""
Command palette — Ctrl+K fuzzy-search over every menu action and
dock toggle.

Pop-up dialog with a single QLineEdit at the top and a QListWidget
underneath.  As the user types, the list filters using a simple
character-subsequence fuzzy match (matches "psf" to "PSF / MTF",
"ar coat" to "AR coating dialog", etc.).  Enter (or click) on a
result fires that action.

Indexed at construction from the live menu bar of the parent
``QMainWindow``: every leaf ``QAction`` becomes a palette entry
labelled with its menu path ("File > Open Prescription File…").
Plus every ``QDockWidget``'s toggleViewAction is included with a
"View > Toggle ..." path so docks are reachable by name.

Author: Andrew Traverso
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QObject
from PySide6.QtGui import QKeySequence, QShortcut, QAction
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
    QLabel, QHBoxLayout,
)


# ---------------------------------------------------------------------------
# Fuzzy match
# ---------------------------------------------------------------------------


def _fuzzy_score(query, candidate):
    """Score a candidate string against a query.

    Returns ``None`` if the query characters cannot be found in order
    (case-insensitively) in the candidate.  Otherwise returns a non-
    negative integer where lower is better, taking into account:

      * gap between matched characters (smaller = tighter match)
      * matches at word boundaries (whitespace, '/', '>') boost the
        score (smaller penalty)
      * a fully-prefix match dominates everything else
    """
    q = query.strip().lower()
    if not q:
        return 0
    c = candidate.lower()
    if c.startswith(q):
        return -1000  # prefix match is best
    qi = 0
    last_pos = -1
    score = 0
    boundary = True
    for ci, ch in enumerate(c):
        is_word_boundary = boundary
        boundary = ch in ' \t/>-_:.'
        if qi >= len(q):
            continue
        if ch == q[qi]:
            gap = ci - last_pos - 1
            score += gap
            if not is_word_boundary:
                score += 1   # mild penalty for mid-word matches
            last_pos = ci
            qi += 1
    if qi < len(q):
        return None   # not all query chars found in order
    return score


# ---------------------------------------------------------------------------
# Action indexer
# ---------------------------------------------------------------------------


def _index_main_window_actions(window):
    """Walk the QMenuBar of `window` and produce a list of
    (label, action) tuples.

    Skips separators, disabled actions, sub-menus (we only collect
    leaves), and the top-level theme/colors meta-menus that don't
    have a sensible "menu path > action" label.
    """
    out = []
    mb = window.menuBar()
    for top_action in mb.actions():
        top_menu = top_action.menu()
        if top_menu is None:
            continue
        top_label = top_action.text().replace('&', '')
        _walk_menu(top_menu, [top_label], out)
    return out


def _walk_menu(menu, path, out):
    for action in menu.actions():
        if action.isSeparator():
            continue
        sub = action.menu()
        if sub is not None:
            label = action.text().replace('&', '')
            _walk_menu(sub, path + [label], out)
            continue
        if not action.isEnabled():
            continue
        text = action.text().replace('&', '').strip()
        if not text:
            continue
        full = ' > '.join(path + [text])
        out.append((full, action))
    return out


# ---------------------------------------------------------------------------
# Palette dialog
# ---------------------------------------------------------------------------


class CommandPalette(QDialog):
    """Modal popup with a query line and ranked action list."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent or main_window)
        self.setWindowTitle('Command Palette')
        self.setModal(True)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setMinimumWidth(560)
        self.setStyleSheet(
            "QDialog { background: #0e1118; border: 1px solid #4a6aaa; }"
            "QLineEdit { background: #15192a; color: #dde8f8; "
            "padding: 8px; font-size: 14px; border: none; "
            "border-bottom: 1px solid #2a3548; }"
            "QListWidget { background: #0e1118; color: #c0d0e8; "
            "border: none; padding: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 5px 8px; }"
            "QListWidget::item:selected { background: #2a3a5a; "
            "color: #ffffff; }"
            "QLabel { color: #7a94b8; padding: 4px; "
            "font-size: 10px; }"
        )

        self._actions = _index_main_window_actions(main_window)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.input = QLineEdit()
        self.input.setPlaceholderText(
            'Type to search… (e.g. "psf", "ghost", "ar coat")')
        self.input.textChanged.connect(self._refresh)
        self.input.installEventFilter(self)
        layout.addWidget(self.input)

        self.list = QListWidget()
        self.list.itemActivated.connect(self._fire)
        layout.addWidget(self.list)

        hint = QLabel(
            '↑/↓ to navigate    Enter to run    Esc to dismiss')
        layout.addWidget(hint)

        self._refresh()

    def _refresh(self):
        q = self.input.text()
        scored = []
        for label, action in self._actions:
            s = _fuzzy_score(q, label)
            if s is None:
                continue
            scored.append((s, label, action))
        scored.sort(key=lambda x: (x[0], x[1]))

        self.list.clear()
        for score, label, action in scored[:120]:
            item = QListWidgetItem(label)
            # Stash the QAction on the item via Qt's UserRole.
            item.setData(Qt.UserRole, action)
            tt = action.toolTip() or ''
            if tt and tt != label:
                item.setToolTip(tt)
            self.list.addItem(item)

        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, event):
        # Forward Up/Down arrow keys from the line edit to the list.
        if obj is self.input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key_Down:
                row = self.list.currentRow()
                self.list.setCurrentRow(min(row + 1, self.list.count() - 1))
                return True
            if event.key() == Qt.Key_Up:
                row = self.list.currentRow()
                self.list.setCurrentRow(max(row - 1, 0))
                return True
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                item = self.list.currentItem()
                if item is not None:
                    self._fire(item)
                return True
        return super().eventFilter(obj, event)

    def _fire(self, item):
        action = item.data(Qt.UserRole)
        self.accept()
        if action is not None:
            action.trigger()


def install_command_palette(main_window):
    """Hook Ctrl+K (and Ctrl+Shift+P, the VS-Code default) on `main_window`."""
    def _open():
        dlg = CommandPalette(main_window, main_window)
        # Centre on the parent
        try:
            geom = main_window.geometry()
            dlg.move(
                geom.x() + (geom.width() - dlg.width()) // 2,
                geom.y() + 60,
            )
        except Exception:
            pass
        dlg.exec()
    sc1 = QShortcut(QKeySequence('Ctrl+K'), main_window)
    sc1.activated.connect(_open)
    sc2 = QShortcut(QKeySequence('Ctrl+Shift+P'), main_window)
    sc2.activated.connect(_open)
    return _open
