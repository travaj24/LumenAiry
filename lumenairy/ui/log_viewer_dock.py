"""
LogViewerDock -- live viewer for the v5.3.2 library logging telemetry.

Implements P2-B from docs/audits/AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24.md.

The v5.3.2 library exposes per-iteration INFO-level telemetry via
``lumenairy._logging.get_logger(...)`` on the three long-running paths
(``apply_real_lens_traced``, ``design_optimize``, ``monte_carlo_tolerancing``).
By default the library root logger has a NullHandler -- nothing surfaces.
This dock attaches a Qt-signal-bridged ``logging.Handler`` to the
``lumenairy`` root logger and displays each record live.

Pair with the Part-5 worker-handler wiring: workers don't need to do
anything; library calls into ``logger.info(...)`` land here directly.

Author: Andrew Traverso
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QFont, QTextCharFormat, QTextCursor, QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Qt-signal-bridged logging handler
# ---------------------------------------------------------------------------

# Canonical telemetry modules from v5.3.2 (audit Part 6.2 P2-B).
# (display label, prefix passed to record.name.startswith)
_FILTER_OPTIONS = [
    ('(all)',                              ''),
    ('lumenairy.elements._lens_traced',    'lumenairy.elements._lens_traced'),
    ('lumenairy.optimize.driver',          'lumenairy.optimize.driver'),
    ('lumenairy.analysis.through_focus',   'lumenairy.analysis.through_focus'),
    ('lumenairy',                          'lumenairy'),
]

_LEVELS = [
    ('DEBUG',    logging.DEBUG),
    ('INFO',     logging.INFO),
    ('WARNING',  logging.WARNING),
    ('ERROR',    logging.ERROR),
    ('CRITICAL', logging.CRITICAL),
]

_LEVEL_COLORS = {
    'DEBUG':    '#7a94b8',
    'INFO':     '#dde8f8',
    'WARNING':  '#ffd166',
    'ERROR':    '#ff6b6b',
    'CRITICAL': '#ff3b3b',
}

_MAX_LINES = 5000  # cap to keep memory bounded for long runs


class _QSignalLogHandler(logging.Handler, QObject):
    """``logging.Handler`` that re-emits records as a Qt signal.

    Filters by level (via ``self.level``) and by logger-name prefix
    (via ``self._name_prefix``) BEFORE emitting so the Qt event loop
    stays light when the user has scoped the view.

    Multiple inheritance is required because ``logging.Handler`` and
    ``QObject`` are both base classes; ``QObject`` MUST come first in
    PySide6 for the metaclass to resolve.  We initialise both
    explicitly.
    """

    # (level_name, logger_name, formatted_message)
    record = Signal(str, str, str)

    def __init__(self, level: int = logging.INFO,
                 parent: Optional[QObject] = None) -> None:
        QObject.__init__(self, parent)
        logging.Handler.__init__(self, level=level)
        self._name_prefix: str = ''      # '' == accept all
        # Formatter -- simple, no timestamp duplication (we add our own
        # in the dock's UI layer if we want one later).
        self.setFormatter(logging.Formatter('%(message)s'))

    def set_name_prefix(self, prefix: str) -> None:
        """Restrict records to those whose ``name`` starts with ``prefix``."""
        self._name_prefix = prefix or ''

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Level filter is handled by Handler.__init__; double-check
            # name filter before re-emitting onto the GUI thread.
            if self._name_prefix and not record.name.startswith(
                    self._name_prefix):
                return
            msg = self.format(record)
            self.record.emit(record.levelname, record.name, msg)
        except Exception:
            # Never let a logging-handler raise -- swallow.
            self.handleError(record)


# ---------------------------------------------------------------------------
# Dock widget
# ---------------------------------------------------------------------------

class LogViewerDock(QWidget):
    """Live viewer of ``lumenairy``-root logger telemetry.

    See module docstring for context.  This widget is "passive": the
    library's existing ``logger.info(...)`` calls land here via the
    attached handler; workers do not need to be modified.
    """

    def __init__(self, model=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName('log_viewer_widget')
        self._model = model

        # Record-count tally for the status label.
        self._counts = {'INFO': 0, 'WARNING': 0, 'ERROR': 0}
        self._lines_in_view = 0
        self._dropped = 0
        self._paused = False

        self._build_ui()

        # Attach the handler to the library root logger.  The library's
        # default effective level is WARNING (root.level == 0/NOTSET,
        # which inherits from Python's root WARNING) -- meaning INFO
        # records are silently dropped at the logger BEFORE any handler
        # sees them.  We lower the library logger to match the dock's
        # selected level so the telemetry actually reaches us; the
        # original level is restored in closeEvent.
        self._handler = _QSignalLogHandler(level=logging.INFO)
        self._handler.record.connect(self._on_log_record)
        self._lib_logger = logging.getLogger('lumenairy')
        self._lib_logger.addHandler(self._handler)
        # Remember the level the logger had before we touched it so
        # closeEvent / the verbosity toggle can restore it cleanly.
        self._saved_lib_level = self._lib_logger.level
        self._lib_logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ---- Control bar -------------------------------------------------
        bar = QHBoxLayout()
        bar.setSpacing(6)

        bar.addWidget(QLabel('Level:'))
        self.cmb_level = QComboBox()
        for name, _ in _LEVELS:
            self.cmb_level.addItem(name)
        self.cmb_level.setCurrentText('INFO')
        self.cmb_level.currentIndexChanged.connect(self._on_level_changed)
        bar.addWidget(self.cmb_level)

        bar.addWidget(QLabel('Module:'))
        self.cmb_filter = QComboBox()
        for label, _ in _FILTER_OPTIONS:
            self.cmb_filter.addItem(label)
        self.cmb_filter.setCurrentIndex(0)
        self.cmb_filter.currentIndexChanged.connect(self._on_filter_changed)
        bar.addWidget(self.cmb_filter)

        # Verbosity bonus -- when checked, drop the library logger to
        # DEBUG so even DEBUG records propagate to our handler.
        self.chk_verbose = QCheckBox('Verbose (DEBUG)')
        self.chk_verbose.setToolTip(
            "Set logging.getLogger('lumenairy').setLevel(DEBUG); "
            'restores WARNING (library default) on uncheck.')
        self.chk_verbose.toggled.connect(self._on_verbose_toggled)
        bar.addWidget(self.chk_verbose)

        bar.addStretch()

        bar.addWidget(QLabel('Find:'))
        self.edt_search = QLineEdit()
        self.edt_search.setPlaceholderText('substring')
        self.edt_search.setMaximumWidth(180)
        self.edt_search.returnPressed.connect(self._on_find_next)
        bar.addWidget(self.edt_search)
        self.btn_find = QPushButton('Find next')
        self.btn_find.clicked.connect(self._on_find_next)
        bar.addWidget(self.btn_find)

        self.btn_pause = QPushButton('Pause')
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._on_pause_toggled)
        bar.addWidget(self.btn_pause)

        self.btn_clear = QPushButton('Clear')
        self.btn_clear.clicked.connect(self._on_clear)
        bar.addWidget(self.btn_clear)

        self.btn_save = QPushButton('Save to file...')
        self.btn_save.clicked.connect(self._on_save)
        bar.addWidget(self.btn_save)

        outer.addLayout(bar)

        # ---- Main text area ---------------------------------------------
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(_MAX_LINES)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        f = QFont('Consolas', 10)
        self.view.setFont(f)
        self.view.setStyleSheet(
            'QPlainTextEdit { background:#0a0c10; color:#dde8f8; '
            'border:1px solid #2a2f3a; }')
        outer.addWidget(self.view, stretch=1)

        # ---- Status bar --------------------------------------------------
        self.lbl_status = QLabel(self._status_text())
        self.lbl_status.setStyleSheet('color:#7a94b8; padding:2px 4px;')
        outer.addWidget(self.lbl_status)

    # ------------------------------------------------------------------
    # Handler bridge
    # ------------------------------------------------------------------

    def _on_log_record(self, level: str, name: str, message: str) -> None:
        """Slot for ``_QSignalLogHandler.record``.

        Updates the running tally regardless of pause state; only the
        append-to-widget step is skipped when paused.
        """
        # Tally
        if level in ('ERROR', 'CRITICAL'):
            self._counts['ERROR'] = self._counts.get('ERROR', 0) + 1
        elif level == 'WARNING':
            self._counts['WARNING'] = self._counts.get('WARNING', 0) + 1
        elif level == 'INFO':
            self._counts['INFO'] = self._counts.get('INFO', 0) + 1
        # (DEBUG is shown but not tallied -- volume can be enormous.)

        if not self._paused:
            self._append(level, name, message)

        self._refresh_status()

    def _append(self, level: str, name: str, message: str) -> None:
        # If we're at the cap, Qt's setMaximumBlockCount silently drops
        # the oldest block; track the drop count for the status label.
        if self._lines_in_view >= _MAX_LINES:
            self._dropped += 1
        else:
            self._lines_in_view += 1

        color = _LEVEL_COLORS.get(level, '#dde8f8')
        # Short logger-name -- strip the 'lumenairy.' prefix for compactness.
        short_name = name
        if short_name.startswith('lumenairy.'):
            short_name = short_name[len('lumenairy.'):]
        elif short_name == 'lumenairy':
            short_name = '(root)'

        # We use appendHtml so per-level colour is visible.  Escape
        # angle brackets etc. in the message body so they render as text.
        body = (message
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))
        html = (f'<span style="color:{color};font-weight:bold;">'
                f'{level[:4]:4s}</span> '
                f'<span style="color:#5cb8ff;">{short_name}</span>  '
                f'<span style="color:#dde8f8;">{body}</span>')
        self.view.appendHtml(html)
        # Auto-scroll to bottom (the dominant use is live tail).
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_level_changed(self, idx: int) -> None:
        if 0 <= idx < len(_LEVELS):
            _, lvl = _LEVELS[idx]
            self._handler.setLevel(lvl)
            # If the user is asking for DEBUG/INFO but the library
            # logger is still pinned at WARNING (its default), records
            # are filtered out before reaching the handler.  Bring the
            # logger down to match -- unless the verbose checkbox has
            # already lowered it to DEBUG, in which case leave it.
            if not self.chk_verbose.isChecked():
                self._lib_logger.setLevel(lvl)

    def _on_filter_changed(self, idx: int) -> None:
        if 0 <= idx < len(_FILTER_OPTIONS):
            _, prefix = _FILTER_OPTIONS[idx]
            self._handler.set_name_prefix(prefix)

    def _on_pause_toggled(self, checked: bool) -> None:
        self._paused = bool(checked)
        self.btn_pause.setText('Resume' if checked else 'Pause')
        self._refresh_status()

    def _on_clear(self) -> None:
        self.view.clear()
        self._lines_in_view = 0
        self._dropped = 0
        self._counts = {'INFO': 0, 'WARNING': 0, 'ERROR': 0}
        self._refresh_status()

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save log to file', 'lumenairy.log',
            'Log files (*.log);;Text files (*.txt);;All files (*)')
        if not path:
            return
        try:
            text = self.view.toPlainText()
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(text)
        except Exception as exc:
            # Avoid a hard dialog -- surface the failure into our own
            # status line.  Diagnostics-sink wiring is optional here.
            self.lbl_status.setText(f'Save failed: {exc}')

    def _on_find_next(self) -> None:
        needle = self.edt_search.text()
        if not needle:
            return
        # QPlainTextEdit.find() searches forward from the current cursor
        # and selects the match (which gives us the highlight for free).
        found = self.view.find(needle)
        if not found:
            # Wrap around: reset cursor to top and try once more.
            cursor = self.view.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.view.setTextCursor(cursor)
            self.view.find(needle)

    def _on_verbose_toggled(self, checked: bool) -> None:
        # Library default is WARNING (the docstring of _logging.py is
        # explicit that the root has a NullHandler and users opt in).
        if checked:
            self._lib_logger.setLevel(logging.DEBUG)
        else:
            self._lib_logger.setLevel(
                self._saved_lib_level or logging.WARNING)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _status_text(self) -> str:
        info = self._counts.get('INFO', 0)
        warn = self._counts.get('WARNING', 0)
        err = self._counts.get('ERROR', 0)
        suffix = '  [PAUSED]' if self._paused else ''
        return (f'{info} INFO / {warn} WARN / {err} ERROR; '
                f'{self._dropped} dropped to cap{suffix}')

    def _refresh_status(self) -> None:
        self.lbl_status.setText(self._status_text())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Detach the handler from the library root logger.

        Closing the dock should not leak a handler that keeps a Qt
        signal alive after the widget is gone.
        """
        try:
            if self._handler is not None:
                self._lib_logger.removeHandler(self._handler)
                self._handler = None
            # Restore the library logger's pre-dock level if we
            # bumped it via the verbosity toggle.
            self._lib_logger.setLevel(
                self._saved_lib_level or logging.WARNING)
        except Exception:
            pass
        super().closeEvent(event)

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
