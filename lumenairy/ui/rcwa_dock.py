"""
RCWA / grating-efficiency dock (3.6).

Wraps :func:`lumenairy.thin_grating_efficiency_1d` and
:func:`lumenairy.grating_efficiency_vs_wavelength` so users can
characterise diffraction-grating designs (groove profile, period,
duty cycle, polarization) without dropping to a script.

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


class RCWADock(QWidget):
    """Rigorous-coupled-wave grating analyser."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        param = QGroupBox('Grating geometry')
        form = QFormLayout(param)
        self.combo_profile = QComboBox()
        self.combo_profile.addItems(
            ['Binary', 'Sinusoidal', 'Triangular', 'Sawtooth (blazed)'])
        form.addRow('Groove profile:', self.combo_profile)
        self.spin_period = QDoubleSpinBox()
        self.spin_period.setRange(0.05, 100.0)
        self.spin_period.setValue(2.0)
        self.spin_period.setSuffix(' µm')
        form.addRow('Period:', self.spin_period)
        self.spin_depth = QDoubleSpinBox()
        self.spin_depth.setRange(0.001, 100.0)
        self.spin_depth.setValue(0.5)
        self.spin_depth.setDecimals(3)
        self.spin_depth.setSuffix(' µm')
        form.addRow('Groove depth:', self.spin_depth)
        self.spin_duty = QDoubleSpinBox()
        self.spin_duty.setRange(0.05, 0.95)
        self.spin_duty.setValue(0.5)
        self.spin_duty.setSingleStep(0.05)
        self.spin_duty.setToolTip('Fraction of the period in the high state '
                                   '(binary profile only).')
        form.addRow('Duty cycle:', self.spin_duty)
        self.spin_n_groove = QDoubleSpinBox()
        self.spin_n_groove.setRange(1.0, 5.0)
        self.spin_n_groove.setValue(1.5)
        form.addRow('Groove material n:', self.spin_n_groove)
        self.spin_n_substrate = QDoubleSpinBox()
        self.spin_n_substrate.setRange(1.0, 5.0)
        self.spin_n_substrate.setValue(1.0)
        form.addRow('Substrate n:', self.spin_n_substrate)
        self.combo_pol = QComboBox()
        self.combo_pol.addItems(['TE', 'TM'])
        form.addRow('Polarization:', self.combo_pol)
        self.spin_aoi = QDoubleSpinBox()
        self.spin_aoi.setRange(-89.9, 89.9)
        self.spin_aoi.setValue(0.0)
        self.spin_aoi.setSuffix(' °')
        form.addRow('Angle of incidence:', self.spin_aoi)
        self.spin_orders = QSpinBox()
        self.spin_orders.setRange(3, 51)
        self.spin_orders.setValue(11)
        self.spin_orders.setSingleStep(2)
        self.spin_orders.setToolTip(
            'Diffraction-order truncation (N=11 keeps orders -5..+5).')
        form.addRow('# orders kept:', self.spin_orders)
        outer.addWidget(param)

        sweep_group = QGroupBox('Wavelength sweep')
        sform = QFormLayout(sweep_group)
        self.spin_wl_min = QDoubleSpinBox()
        self.spin_wl_min.setRange(50.0, 50000.0)
        self.spin_wl_min.setValue(400.0)
        self.spin_wl_min.setSuffix(' nm')
        sform.addRow('λ min:', self.spin_wl_min)
        self.spin_wl_max = QDoubleSpinBox()
        self.spin_wl_max.setRange(50.0, 50000.0)
        self.spin_wl_max.setValue(800.0)
        self.spin_wl_max.setSuffix(' nm')
        sform.addRow('λ max:', self.spin_wl_max)
        self.spin_wl_n = QSpinBox()
        self.spin_wl_n.setRange(2, 1024)
        self.spin_wl_n.setValue(101)
        sform.addRow('# samples:', self.spin_wl_n)
        outer.addWidget(sweep_group)

        self.btn_run = QPushButton('▶ Compute RCWA efficiency')
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
        self.summary.setMaximumHeight(80)
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
                'Rigorous-coupled-wave analysis of a 1-D grating.\n'
                'Sweeps wavelength to plot per-order efficiency.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _run(self):
        try:
            import lumenairy as la
            wavelengths = np.linspace(
                self.spin_wl_min.value(),
                self.spin_wl_max.value(),
                int(self.spin_wl_n.value())) * 1e-9
            res = la.grating_efficiency_vs_wavelength(
                period=float(self.spin_period.value()) * 1e-6,
                depth=float(self.spin_depth.value()) * 1e-6,
                duty_cycle=float(self.spin_duty.value()),
                groove_index=float(self.spin_n_groove.value()),
                substrate_index=float(self.spin_n_substrate.value()),
                wavelengths=wavelengths,
                profile=self.combo_profile.currentText().lower().split()[0],
                polarization=self.combo_pol.currentText(),
                angle=float(self.spin_aoi.value()),
                n_orders=int(self.spin_orders.value()))
        except Exception as exc:
            self.summary.setPlainText(
                f'grating_efficiency_vs_wavelength failed: '
                f'{type(exc).__name__}: {exc}')
            return
        self._draw_result(wavelengths, res)

    def _draw_result(self, wavelengths, res):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.tick_params(colors='#7a94b8', labelsize=8)
        for s in ax.spines.values():
            s.set_color('#334054')
        try:
            efficiencies = (res.efficiencies if hasattr(res, 'efficiencies')
                            else np.asarray(res))
            efficiencies = np.atleast_2d(efficiencies)
            if efficiencies.shape[0] == len(wavelengths):
                efficiencies = efficiencies.T
            for i, eff in enumerate(efficiencies):
                ax.plot(wavelengths * 1e9, eff, label=f'order {i}')
            ax.set_xlabel('Wavelength [nm]', color='#dde8f8',
                          fontfamily='monospace')
            ax.set_ylabel('Diffraction efficiency',
                          color='#dde8f8', fontfamily='monospace')
            ax.legend(fontsize=8)
            self.summary.setPlainText(
                f'Computed {efficiencies.shape[0]} order(s) over '
                f'{len(wavelengths)} wavelengths.')
        except Exception as exc:
            self.summary.setPlainText(
                f'Could not unpack RCWA result: {type(exc).__name__}: {exc}')
        self.fig.tight_layout()
        self.canvas.draw_idle()
