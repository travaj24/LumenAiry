"""
Central diagnostics sink — replaces the scatter of silent ``except: pass``
blocks with a single place that collects, displays, and remembers
non-fatal UI/model errors.

Usage
-----
Wherever a handler previously swallowed an exception silently::

    try:
        ...
    except Exception:
        pass

replace with::

    from .diagnostics import diag
    try:
        ...
    except Exception as e:
        diag.report('subsystem-name', e)

``diag`` is a global singleton; its :meth:`emit` Qt signal fires on
every new entry so widgets (status-bar badge, dock) update live.

Author: Andrew Traverso
"""

from __future__ import annotations

import datetime
import traceback
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit,
    QLabel, QCheckBox,
)


# ---------------------------------------------------------------------------
# Singleton sink
# ---------------------------------------------------------------------------

class _DiagnosticsSink(QObject):
    """Collects tagged messages from anywhere in the UI.

    Emits :data:`emit_signal` every time :meth:`report` or :meth:`info`
    is called; widgets listen and update.
    """

    emit_signal = Signal(str, str, str)   # level, tag, message

    def __init__(self) -> None:
        super().__init__()
        # (timestamp, level, tag, message)
        self._entries: List[Tuple[str, str, str, str]] = []
        self._unseen_error_count = 0
        # Routing policy (editable via main_window -> Preferences).
        # Errors also pop a modal by default; warns flash the status bar.
        self.modal_on_error: bool = False
        self.status_on_warn: bool = True

    # -- API used by callers --------------------------------------------

    def report(self, tag: str, exc: BaseException,
               context: Optional[str] = None) -> None:
        """Log an exception that was caught and handled non-fatally."""
        msg = f'{type(exc).__name__}: {exc}'
        if context:
            msg = f'{context}: {msg}'
        # Short traceback tail (last frame) helps more than nothing.
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            last = tb[-1]
            msg += f'  [at {last.filename.split(chr(92))[-1]}:{last.lineno} in {last.name}]'
        self._push('error', tag, msg)

    def warn(self, tag: str, msg: str) -> None:
        self._push('warn', tag, msg)

    def info(self, tag: str, msg: str) -> None:
        self._push('info', tag, msg)

    # -- Internal -------------------------------------------------------

    def _push(self, level: str, tag: str, msg: str) -> None:
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self._entries.append((ts, level, tag, msg))
        # Cap at 500 so a runaway error loop doesn't balloon memory.
        if len(self._entries) > 500:
            self._entries = self._entries[-500:]
        if level == 'error':
            self._unseen_error_count += 1
        self.emit_signal.emit(level, tag, msg)

    # -- Accessors for views --------------------------------------------

    def entries(self):
        return list(self._entries)

    def unseen_error_count(self) -> int:
        return self._unseen_error_count

    def mark_seen(self) -> None:
        self._unseen_error_count = 0
        self.emit_signal.emit('clear', '', '')

    def clear(self) -> None:
        self._entries.clear()
        self._unseen_error_count = 0
        self.emit_signal.emit('clear', '', '')


# Single global instance -- import from anywhere in the UI.
diag = _DiagnosticsSink()


# ---------------------------------------------------------------------------
# Dock widget: full log viewer
# ---------------------------------------------------------------------------

class DiagnosticsDock(QWidget):
    """Viewer for the diagnostics log.

    Stays hidden by default; the status-bar badge flashes to draw
    attention when a new error arrives.
    """

    LEVEL_COLORS = {
        'info':  '#7a94b8',
        'warn':  '#ffd166',
        'error': '#ff6b6b',
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar
        bar = QHBoxLayout()
        bar.addWidget(QLabel('Diagnostics log'))
        bar.addStretch()
        self.chk_auto = QCheckBox('Auto-scroll')
        self.chk_auto.setChecked(True)
        bar.addWidget(self.chk_auto)
        btn_clear = QPushButton('Clear')
        btn_clear.clicked.connect(diag.clear)
        bar.addWidget(btn_clear)
        layout.addLayout(bar)

        # Text area
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(QFont('Consolas', 10))
        self.view.setStyleSheet(
            'QTextEdit { background:#0a0c10; color:#dde8f8; border:none; }')
        layout.addWidget(self.view, stretch=1)

        diag.emit_signal.connect(self._on_emit)
        # Replay any entries that existed before this widget was built.
        for ts, level, tag, msg in diag.entries():
            self._append(ts, level, tag, msg)

    def _on_emit(self, level, tag, msg):
        if level == 'clear':
            self.view.clear()
            return
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self._append(ts, level, tag, msg)

    def _append(self, ts, level, tag, msg):
        color = self.LEVEL_COLORS.get(level, '#dde8f8')
        html = (f'<span style="color:#557;">[{ts}]</span> '
                f'<span style="color:{color};font-weight:bold;">'
                f'{level.upper()}</span> '
                f'<span style="color:#5cb8ff;">{tag}</span> '
                f'<span style="color:#dde8f8;">{msg}</span>')
        self.view.append(html)
        if self.chk_auto.isChecked():
            sb = self.view.verticalScrollBar()
            sb.setValue(sb.maximum())


# ---------------------------------------------------------------------------
# Status-bar badge
# ---------------------------------------------------------------------------

class DiagnosticsBadge(QLabel):
    """Small clickable label that flashes when new errors arrive."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip('Click to open the Diagnostics log '
                        '(red = new errors since last view)')
        self._update_text()
        diag.emit_signal.connect(self._on_emit)

    def mousePressEvent(self, event):
        self.clicked.emit()
        diag.mark_seen()
        self._update_text()

    def _on_emit(self, level, tag, msg):
        self._update_text()

    def _update_text(self):
        n = diag.unseen_error_count()
        if n == 0:
            self.setText('diag: ok')
            self.setStyleSheet('color:#3ddc84; padding:0 6px;')
        else:
            self.setText(f'diag: {n} new \u25CF')
            self.setStyleSheet('color:#ff6b6b; font-weight:bold; '
                               'padding:0 6px;')
