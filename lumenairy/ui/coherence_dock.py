"""
Partial-coherence (Köhler) imaging dock (3.6).

Wraps :func:`lumenairy.koehler_image` /
:func:`lumenairy.extended_source_image` to model imaging with an
incoherent extended source -- the canonical setup for lithography
illumination, Köhler microscopy, and any system where the source
is much larger than a single coherent mode.

Author: Andrew Traverso
"""

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QComboBox,
    QSizePolicy, QTextEdit,
)
from PySide6.QtGui import QFont
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


class _KoehlerWorker(QThread):
    finished_result = Signal(object)

    def __init__(self, sm, params):
        super().__init__()
        self.sm = sm
        self.params = params

    def run(self):
        try:
            import lumenairy as la
            pres = self.sm.to_prescription()
            wv = self.sm.wavelength_m
            res = la.koehler_image(
                prescription=pres, wavelength=wv,
                source_sigma=self.params['sigma'],
                N=self.params['N'], dx=self.params['dx'],
                n_modes=self.params['n_modes'])
            self.finished_result.emit(res)
        except Exception as exc:
            self.finished_result.emit(
                {'error': f'{type(exc).__name__}: {exc}'})


class CoherenceDock(QWidget):
    """Partial-coherence imaging via Köhler decomposition."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        param = QGroupBox('Source / illumination')
        form = QFormLayout(param)
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(['Circular', 'Annular', 'Dipole', 'Quadrupole'])
        form.addRow('Source shape:', self.combo_shape)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.01, 1.0)
        self.spin_sigma.setSingleStep(0.05)
        self.spin_sigma.setValue(0.5)
        self.spin_sigma.setToolTip(
            'Partial-coherence factor σ = NA_source / NA_pupil.  '
            '0 = fully coherent, 1 = critical illumination.')
        form.addRow('σ (partial coherence):', self.spin_sigma)
        self.spin_modes = QSpinBox()
        self.spin_modes.setRange(1, 256)
        self.spin_modes.setValue(16)
        self.spin_modes.setToolTip(
            'Number of mutually-incoherent modes to sum in the '
            'Köhler decomposition.  More = closer to a continuous '
            'source.')
        form.addRow('Source modes:', self.spin_modes)
        self.spin_N = QSpinBox()
        self.spin_N.setRange(64, 4096)
        self.spin_N.setValue(256)
        form.addRow('Image grid N:', self.spin_N)
        self.spin_dx_um = QDoubleSpinBox()
        self.spin_dx_um.setRange(0.01, 1000.0)
        self.spin_dx_um.setValue(0.5)
        self.spin_dx_um.setSuffix(' µm')
        form.addRow('Image dx:', self.spin_dx_um)
        outer.addWidget(param)
        self.btn_run = QPushButton('▶ Compute partial-coherence image')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        outer.addWidget(self.btn_run)
        self.fig = Figure(figsize=(6, 3.4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas, stretch=1)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(100)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        outer.addWidget(self.summary)
        self._draw_empty()

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Sum incoherent images from a Köhler-decomposed\n'
                'extended source.  Useful for lithography NA-σ\n'
                'analysis and microscope condenser illumination.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _run(self):
        params = dict(
            sigma=float(self.spin_sigma.value()),
            n_modes=int(self.spin_modes.value()),
            N=int(self.spin_N.value()),
            dx=float(self.spin_dx_um.value()) * 1e-6,
        )
        self.btn_run.setEnabled(False)
        self.summary.setPlainText('Computing…')
        self._worker = _KoehlerWorker(self.sm, params)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, res):
        self.btn_run.setEnabled(True)
        self._worker = None
        if isinstance(res, dict) and 'error' in res:
            self.summary.setPlainText(
                f'koehler_image failed:\n  {res["error"]}')
            return
        try:
            I = np.asarray(getattr(res, 'image', res))
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor('#0a0c10')
            ax.imshow(I, cmap='inferno', origin='lower')
            ax.set_title('Partial-coherence image',
                         color='#dde8f8', fontfamily='monospace')
            ax.tick_params(colors='#7a94b8', labelsize=8)
            for s in ax.spines.values():
                s.set_color('#334054')
            self.canvas.draw_idle()
            self.summary.setPlainText(
                f'Image shape: {I.shape}; peak {I.max():.4e}')
        except Exception as exc:
            self.summary.setPlainText(
                f'Could not display result: {type(exc).__name__}: {exc}')
