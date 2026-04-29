"""
ReplDock — embedded Python REPL with the model + numpy pre-bound.

Useful for ad-hoc inspection during a design session: peek at trace
results, plot custom analyses, drive the optimizer, etc., without
leaving the GUI.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget,
)

import io
import sys
import traceback


class _CmdLine(QLineEdit):
    """QLineEdit with Up/Down history navigation."""

    history_prev = Signal()
    history_next = Signal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Up:
            self.history_prev.emit()
            return
        if event.key() == Qt.Key_Down:
            self.history_next.emit()
            return
        super().keyPressEvent(event)


class ReplDock(QWidget):
    """Embedded Python REPL.

    Pre-bound globals:
      model    -- the SystemModel instance
      np       -- numpy
      plt      -- matplotlib.pyplot (if available)
      result   -- the most recent geometric trace (refreshed on Enter)
      wave     -- the most recent wave-optics result dict (refreshed)
    """

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setObjectName('repl_widget')
        self._model = model
        self._history = []
        self._hist_idx = 0  # one past the end while not navigating

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        f = QFont('Consolas', 10)
        self.output.setFont(f)
        self.output.appendPlainText(
            '# Python REPL.  Bound: model, np, plt, result, wave.\n'
            '# Enter to execute, Up/Down for history.\n')
        outer.addWidget(self.output, stretch=1)

        row = QHBoxLayout()
        row.addWidget(QLabel('>>>'))
        self.input = _CmdLine()
        self.input.setFont(f)
        self.input.returnPressed.connect(self._exec_current)
        self.input.history_prev.connect(self._history_prev)
        self.input.history_next.connect(self._history_next)
        row.addWidget(self.input, stretch=1)
        outer.addLayout(row)

        # Build the persistent globals dict for exec().
        self._globals = {'__name__': '__repl__'}
        self._refresh_bindings()

    # ------------------------------------------------------------------

    def _refresh_bindings(self):
        """Sync standard names into the REPL globals."""
        try:
            import numpy as np
            self._globals['np'] = np
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            self._globals['plt'] = plt
        except Exception:
            pass
        self._globals['model'] = self._model
        self._globals['result'] = getattr(self._model, '_trace_result', None)
        # `wave` is set externally via set_wave_result.

    def set_wave_result(self, wave_result):
        """Hook from main_window to expose the latest wave-optics run."""
        self._globals['wave'] = wave_result

    # ------------------------------------------------------------------

    def _history_prev(self):
        if not self._history:
            return
        if self._hist_idx > 0:
            self._hist_idx -= 1
        self.input.setText(self._history[self._hist_idx])

    def _history_next(self):
        if not self._history:
            return
        if self._hist_idx < len(self._history) - 1:
            self._hist_idx += 1
            self.input.setText(self._history[self._hist_idx])
        else:
            self._hist_idx = len(self._history)
            self.input.setText('')

    def _exec_current(self):
        cmd = self.input.text()
        if not cmd.strip():
            return
        self.input.setText('')
        self._history.append(cmd)
        self._hist_idx = len(self._history)
        self._refresh_bindings()

        self.output.appendPlainText(f'>>> {cmd}')
        # Capture stdout/stderr for the duration of exec.
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            # Try to compile as an expression first so we can echo its
            # value (Python REPL semantics).  Fall back to statement.
            try:
                code = compile(cmd, '<repl>', 'eval')
                value = eval(code, self._globals)
                if value is not None:
                    print(repr(value))
            except SyntaxError:
                code = compile(cmd, '<repl>', 'exec')
                exec(code, self._globals)
        except Exception:
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr = old_out, old_err

        out_text = buf_out.getvalue()
        err_text = buf_err.getvalue()
        if out_text:
            self.output.appendPlainText(out_text.rstrip())
        if err_text:
            self.output.appendPlainText(err_text.rstrip())
        # Scroll to bottom.
        sb = self.output.verticalScrollBar()
        sb.setValue(sb.maximum())
