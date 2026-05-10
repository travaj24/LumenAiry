"""
Caustic-diagnostic dock — locate caustic crossings along the chief ray.

Wraps :func:`lumenairy.caustic_diagnostic` (3.5.4): a small fan of
rays parallel to the chief ray is traced through the prescription
and the local Jacobian determinant is sampled at densely-spaced
planes between successive surfaces.  Sign flips of det(J) mark
caustic crossings, which are ranked and plotted alongside surface
positions and chief-ray height for context.

Pairs with :func:`lumenairy.plot_caustic_diagnostic` for the
matplotlib visualisation.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QTextEdit, QSizePolicy,
)
from PySide6.QtGui import QFont

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


class CausticWorker(QThread):
    """Run :func:`caustic_diagnostic` in a background thread."""
    finished_result = Signal(object)  # CausticDiagnostic or {'error': str}

    def __init__(self, prescription, wavelength, fan_radius_m,
                 n_z_per_gap, z_after_last_m):
        super().__init__()
        self.prescription = prescription
        self.wavelength = wavelength
        self.fan_radius_m = fan_radius_m
        self.n_z_per_gap = n_z_per_gap
        self.z_after_last_m = z_after_last_m

    def run(self):
        try:
            import lumenairy as la
            diag = la.caustic_diagnostic(
                self.prescription, self.wavelength,
                fan_radius=self.fan_radius_m,
                n_z_per_gap=self.n_z_per_gap,
                z_after_last_surface=self.z_after_last_m)
            self.finished_result.emit(diag)
        except Exception as exc:
            self.finished_result.emit(
                {'error': f'{type(exc).__name__}: {exc}'})


class CausticDock(QWidget):
    """Caustic-localisation analysis panel.

    Parameters mirror :func:`caustic_diagnostic` directly:
    ``fan_radius`` (the half-spacing of the FD-Jacobian fan around
    the chief ray) and ``n_z_per_gap`` (sample density between
    surfaces).  The result is drawn on a matplotlib canvas via
    :func:`plot_caustic_diagnostic` so the visualisation stays
    consistent with the library's reference rendering.
    """

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None
        self._last_result = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ── Diagnostic parameters ──
        param_group = QGroupBox('Diagnostic parameters')
        param_form = QFormLayout(param_group)

        self.spin_fan_um = QDoubleSpinBox()
        self.spin_fan_um.setRange(0.1, 10000.0)
        self.spin_fan_um.setValue(100.0)
        self.spin_fan_um.setDecimals(1)
        self.spin_fan_um.setSuffix(' µm')
        self.spin_fan_um.setToolTip(
            'Half-spacing of the four marginal rays around the '
            'chief ray.  Smaller than the smallest aperture, large '
            'enough for the FD-Jacobian to be numerically stable.  '
            'Default 100 µm.')
        param_form.addRow('Fan radius:', self.spin_fan_um)

        self.spin_n_z = QSpinBox()
        self.spin_n_z.setRange(4, 4096)
        self.spin_n_z.setValue(64)
        self.spin_n_z.setToolTip(
            'Sample planes per gap.  Increase for finer caustic '
            'localisation; cost is linear in this parameter.')
        param_form.addRow('Samples / gap:', self.spin_n_z)

        self.spin_z_after = QDoubleSpinBox()
        self.spin_z_after.setRange(0.0, 1000.0)
        self.spin_z_after.setValue(0.0)
        self.spin_z_after.setDecimals(2)
        self.spin_z_after.setSuffix(' mm')
        self.spin_z_after.setToolTip(
            'Optional sampling region after the last surface.  '
            'Set to non-zero to look for the system\'s focal-region '
            'caustic; 0 disables the post-surface scan.')
        param_form.addRow('Sample length after last surface:',
                          self.spin_z_after)

        outer.addWidget(param_group)

        # ── Run / clear ──
        run_row = QHBoxLayout()
        self.btn_run = QPushButton('▶ Run caustic diagnostic  (F10)')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        self.btn_clear = QPushButton('Clear')
        self.btn_clear.clicked.connect(self._clear)
        run_row.addWidget(self.btn_clear)
        run_row.addStretch()
        outer.addLayout(run_row)

        # ── Plot ──
        self.fig = Figure(figsize=(6, 3.4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas, stretch=1)

        # ── Summary text ──
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        outer.addWidget(self.summary)

        self._draw_empty()

    # ── Drawing helpers ─────────────────────────────────────────────

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Run the diagnostic to locate caustic crossings\n'
                'along the chief ray.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace',
                fontsize=10)
        ax.tick_params(colors='#7a94b8')
        for spine in ax.spines.values():
            spine.set_color('#334054')
        self.canvas.draw_idle()

    def _draw_result(self, diag):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        try:
            import lumenairy as la
            la.plot_caustic_diagnostic(diag, ax=ax,
                                        title='Caustic diagnostic')
        except Exception as exc:
            # plot_caustic_diagnostic may raise on degenerate
            # systems; fall back to a minimal det-J trace so the
            # user still sees the data.
            ax.plot(diag.z_samples * 1e3, diag.det_J,
                    color='#5cb8ff', linewidth=1.2)
            ax.axhline(0.0, color='#ff6b35', linewidth=0.8,
                       linestyle='--')
            ax.set_xlabel('z [mm]', color='#dde8f8',
                          fontfamily='monospace', fontsize=9)
            ax.set_ylabel('det J', color='#dde8f8',
                          fontfamily='monospace', fontsize=9)
            ax.set_title(f'(plot_caustic_diagnostic failed: '
                         f'{type(exc).__name__})',
                         color='#ff6b35', fontsize=9,
                         fontfamily='monospace')
        ax.tick_params(colors='#7a94b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#334054')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _format_summary(self, diag):
        caustics = list(getattr(diag, 'caustic_z', []) or [])
        surfaces = list(getattr(diag, 'surface_z', []) or [])
        lines = []
        lines.append(f'Sampled {len(diag.z_samples)} planes; '
                     f'{len(caustics)} caustic crossing(s) '
                     f'detected.')
        if caustics:
            lines.append('Caustic z (mm), Maslov index:')
            for i, z in enumerate(caustics):
                m = (diag.maslov_index[i]
                     if i < len(getattr(diag, 'maslov_index', []) or [])
                     else 'n/a')
                lines.append(f'  z = {z * 1e3:9.4f} mm   '
                             f'µ = {m}')
        else:
            lines.append('No caustic crossings detected — '
                         'fan-traced det(J) does not change sign in '
                         'the sampled region.  Increase the post-'
                         'surface sample length to reach the focus, '
                         'or enlarge the fan radius if the FD-Jacobian '
                         'is in the noise floor.')
        if surfaces:
            lines.append('')
            lines.append(f'Surface positions: '
                         f'{[round(z * 1e3, 3) for z in surfaces]} mm')
        return '\n'.join(lines)

    # ── Run path ────────────────────────────────────────────────────

    def _run(self):
        try:
            pres = self.sm.to_prescription()
        except Exception as exc:
            self.summary.setPlainText(
                f'Could not export prescription: {type(exc).__name__}: '
                f'{exc}.  The system needs at least one optical '
                f'surface before a caustic diagnostic can run.')
            return

        wavelength = self.sm.wavelength_m
        fan_radius_m = float(self.spin_fan_um.value()) * 1e-6
        n_z = int(self.spin_n_z.value())
        z_after_mm = float(self.spin_z_after.value())
        z_after_m = z_after_mm * 1e-3 if z_after_mm > 0 else None

        self.btn_run.setEnabled(False)
        self.summary.setPlainText('Running caustic diagnostic…')
        self._worker = CausticWorker(
            pres, wavelength, fan_radius_m, n_z, z_after_m)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, result):
        self.btn_run.setEnabled(True)
        self._worker = None
        if isinstance(result, dict) and 'error' in result:
            self.summary.setPlainText(
                f'Caustic diagnostic failed:\n  {result["error"]}')
            return
        self._last_result = result
        self._draw_result(result)
        self.summary.setPlainText(self._format_summary(result))

    def _clear(self):
        self._last_result = None
        self.summary.clear()
        self._draw_empty()
