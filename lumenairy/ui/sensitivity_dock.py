"""
Sensitivity dock — rank variables by their effect on the merit.

Finite-difference d(merit)/d(var) plus a per-tolerance sensitivity
pass (same FD idea but across the user's tolerance dock values).
Shows both as horizontal bars so the eye picks out the dominant
variables immediately.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QDoubleSpinBox, QComboBox,
)
from PySide6.QtGui import QFont

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class SensitivityDock(QWidget):
    """Per-variable sensitivity ranking panel."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Controls
        ctl = QGroupBox('Finite-difference step')
        ctl_row = QHBoxLayout(ctl)
        ctl_row.addWidget(QLabel('relative step:'))
        self.spin_eps = QDoubleSpinBox()
        self.spin_eps.setRange(1e-6, 1e-1)
        self.spin_eps.setValue(1e-3)
        self.spin_eps.setDecimals(6)
        self.spin_eps.setToolTip(
            'Fractional perturbation applied to each variable to estimate '
            'd(merit)/d(var) via central finite differences.')
        ctl_row.addWidget(self.spin_eps)

        ctl_row.addWidget(QLabel('metric:'))
        self.combo_metric = QComboBox()
        self.combo_metric.addItems([
            'Merit (current optimizer objective)',
            'RMS spot radius',
            'EFL',
            'BFL',
        ])
        ctl_row.addWidget(self.combo_metric)

        self.btn_run = QPushButton('Rank variables')
        self.btn_run.clicked.connect(self._run_ranking)
        ctl_row.addWidget(self.btn_run)

        ctl_row.addStretch()
        layout.addWidget(ctl)

        # Plot
        if HAS_MPL:
            self.fig = Figure(figsize=(6, 3.5), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, stretch=1)
        else:
            layout.addWidget(QLabel('(matplotlib not available)'))

        # Text
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(180)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

    def _evaluate(self, values, metric):
        """Evaluate the selected metric at parameter vector `values`."""
        # Keep the model state intact: snapshot, perturb, restore.
        original = self.sm.get_variable_values().copy()
        try:
            self.sm.set_variable_values(values)
            self.sm._invalidate()
            if metric == 0:
                return float(self.sm.merit_function(values))
            # geometric metrics: re-trace & read off the result
            self.sm.retrace()
            if metric == 1:  # RMS spot
                r = self.sm.trace_result
                if r is None or r.image_rays is None:
                    return np.nan
                alive = r.image_rays.alive
                x = r.image_rays.x[alive]; y = r.image_rays.y[alive]
                if x.size == 0:
                    return np.nan
                return float(np.sqrt(np.mean(x * x + y * y)))
            if metric in (2, 3):
                _, efl, bfl = self.sm.get_abcd()
                return float(efl if metric == 2 else bfl)
        finally:
            self.sm.set_variable_values(original)
            self.sm._invalidate()
        return np.nan

    def _run_ranking(self):
        if not self.sm.opt_variables:
            self.summary.append(
                'No variables defined -- pick some in the Optimizer '
                'dock first.')
            return
        metric = self.combo_metric.currentIndex()
        eps_rel = self.spin_eps.value()
        x0 = self.sm.get_variable_values().copy()
        f0 = self._evaluate(x0, metric)
        self.summary.clear()
        self.summary.append(
            f'Base {self.combo_metric.currentText()} = {f0:.6g}')
        self.summary.append(
            f'{"#":>3s}  {"variable":<22s}  '
            f'{"d(merit)/d(var)":>18s}  {"|rel|":>10s}')

        sens = []
        labels = []
        for i, var in enumerate(self.sm.opt_variables):
            label = f'E{var[0]}.S{var[1]}.{var[2]}' if len(var) == 3 else str(var)
            v = x0[i]
            step = abs(v) * eps_rel if v != 0 else eps_rel
            if step < 1e-12:
                step = 1e-9
            x_plus = x0.copy(); x_plus[i] = v + step
            x_minus = x0.copy(); x_minus[i] = v - step
            f_plus = self._evaluate(x_plus, metric)
            f_minus = self._evaluate(x_minus, metric)
            if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
                sens.append(np.nan)
                labels.append(label)
                continue
            d = (f_plus - f_minus) / (2 * step)
            sens.append(d)
            labels.append(label)
            rel = abs(d * v / f0) if abs(f0) > 1e-30 else abs(d * v)
            self.summary.append(
                f'{i:>3d}  {label:<22s}  {d:>18.6e}  {rel:>10.4f}')

        self._plot_ranking(labels, sens, f0)

    def _plot_ranking(self, labels, sens, f0):
        if not HAS_MPL:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        # Rank by absolute sensitivity
        idx = np.argsort(np.abs(np.asarray(sens, dtype=float)))[::-1]
        ys = np.arange(len(idx))[::-1]
        vals = [sens[i] for i in idx]
        labs = [labels[i] for i in idx]
        colors = ['#5cb8ff' if v >= 0 else '#ff7a5c' for v in vals]
        ax.barh(ys, np.abs(vals), color=colors)
        ax.set_yticks(ys)
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_xlabel('|d(merit)/d(var)|')
        ax.set_xscale('log')
        ax.grid(alpha=0.2, axis='x')
        self.canvas.draw()
