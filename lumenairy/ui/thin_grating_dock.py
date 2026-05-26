"""
Thin-grating / grating-efficiency dock (3.6).

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


class ThinGratingDock(QWidget):
    """Thin-grating diffraction-efficiency analyser (analytical
    scalar thin-phase model)."""

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
        self.spin_n_ridge = QDoubleSpinBox()
        self.spin_n_ridge.setRange(1.0, 5.0)
        self.spin_n_ridge.setValue(1.5)
        self.spin_n_ridge.setDecimals(3)
        self.spin_n_ridge.setToolTip('Refractive index of the ridge '
                                      '(high-index) region.')
        form.addRow('Ridge material n:', self.spin_n_ridge)
        self.spin_n_groove = QDoubleSpinBox()
        self.spin_n_groove.setRange(1.0, 5.0)
        self.spin_n_groove.setValue(1.0)
        self.spin_n_groove.setDecimals(3)
        self.spin_n_groove.setToolTip('Refractive index of the groove '
                                       '(low-index / air) region.')
        form.addRow('Groove material n:', self.spin_n_groove)
        self.spin_n_substrate = QDoubleSpinBox()
        self.spin_n_substrate.setRange(1.0, 5.0)
        self.spin_n_substrate.setValue(1.5)
        self.spin_n_substrate.setDecimals(3)
        form.addRow('Substrate n:', self.spin_n_substrate)
        self.spin_n_superstrate = QDoubleSpinBox()
        self.spin_n_superstrate.setRange(1.0, 5.0)
        self.spin_n_superstrate.setValue(1.0)
        self.spin_n_superstrate.setDecimals(3)
        self.spin_n_superstrate.setToolTip('Refractive index of the '
                                            'superstrate (incident medium).')
        form.addRow('Superstrate n:', self.spin_n_superstrate)
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

        self.btn_run = QPushButton('▶ Compute grating efficiency')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        outer.addWidget(self.btn_run)

        self.fig = Figure(figsize=(6, 3.4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas.setMinimumSize(0, 0)
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
                'Analytical scalar thin-phase grating efficiencies.\n'
                'Sweeps wavelength to plot per-order efficiency.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _collect_inputs(self) -> dict:
        """Snapshot every UI widget's current value into a plain dict.

        Decoupled from the Qt run handler so the compute path can be
        unit-tested without round-tripping through a live ``QApplication``.
        """
        return {
            'period_m': float(self.spin_period.value()) * 1e-6,
            'depth_m': float(self.spin_depth.value()) * 1e-6,
            'duty_cycle': float(self.spin_duty.value()),
            'n_ridge': float(self.spin_n_ridge.value()),
            'n_groove': float(self.spin_n_groove.value()),
            'n_substrate': float(self.spin_n_substrate.value()),
            'n_superstrate': float(self.spin_n_superstrate.value()),
            'polarization': self.combo_pol.currentText().lower(),
            'aoi_rad': float(self.spin_aoi.value()) * np.pi / 180.0,
            'n_orders': int(self.spin_orders.value()),
            'wl_min_m': float(self.spin_wl_min.value()) * 1e-9,
            'wl_max_m': float(self.spin_wl_max.value()) * 1e-9,
            'wl_n': int(self.spin_wl_n.value()),
        }

    @staticmethod
    def _compute_efficiency_data(inputs: dict) -> dict:
        """Pure compute helper -- no Qt, no instance state.

        Takes the dict produced by :meth:`_collect_inputs` and returns a
        result dict::

            {
                'wavelengths': ndarray (n_wl,),
                'orders':      ndarray (n_orders,),
                'efficiencies': ndarray (n_orders, n_wl),
                'summary':     str  -- human-readable summary line.
            }

        Uses :func:`lumenairy.thin_grating_efficiency_1d` directly so we
        get the full per-order efficiency matrix in one pass (the
        single-order helper ``grating_efficiency_vs_wavelength`` would
        require N_orders independent sweeps).
        """
        import lumenairy as la
        wavelengths = np.linspace(
            inputs['wl_min_m'], inputs['wl_max_m'], inputs['wl_n'])
        n_orders = int(inputs['n_orders'])
        # thin_grating_efficiency_1d returns 2*n_orders+1 entries (-n..+n).
        # The dock's "# orders kept" spinbox is the *total* count, so
        # convert to the half-width parameter the analytical function
        # actually expects.
        half_w = max(1, n_orders // 2)
        n_total = 2 * half_w + 1
        efficiencies = np.empty((n_total, wavelengths.size))
        orders_idx = None
        for j, wl in enumerate(wavelengths):
            orders, _R, T = la.thin_grating_efficiency_1d(
                period=inputs['period_m'],
                n_ridge=inputs['n_ridge'],
                n_groove=inputs['n_groove'],
                n_substrate=inputs['n_substrate'],
                n_superstrate=inputs['n_superstrate'],
                depth=inputs['depth_m'],
                duty_cycle=inputs['duty_cycle'],
                wavelength=float(wl),
                angle=inputs['aoi_rad'],
                polarization=inputs['polarization'],
                n_orders=half_w)
            if orders_idx is None:
                orders_idx = orders
            efficiencies[:, j] = T
        total_T = efficiencies.sum(axis=0)
        summary = (
            f'Computed {n_total} order(s) over {wavelengths.size} '
            f'wavelengths.  '
            f'Sum(T) range: [{total_T.min():.4f}, {total_T.max():.4f}].')
        return {
            'wavelengths': wavelengths,
            'orders': orders_idx,
            'efficiencies': efficiencies,
            'summary': summary,
        }

    def _run(self):
        try:
            inputs = self._collect_inputs()
            data = self._compute_efficiency_data(inputs)
        except (TypeError, ValueError, RuntimeError,
                AttributeError, ImportError) as exc:
            self.summary.setPlainText(
                f'thin_grating_efficiency_1d failed: '
                f'{type(exc).__name__}: {exc}')
            return
        self._draw_result(data['wavelengths'], data)

    def _draw_result(self, wavelengths, res):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.tick_params(colors='#7a94b8', labelsize=8)
        for s in ax.spines.values():
            s.set_color('#334054')
        try:
            if isinstance(res, dict):
                efficiencies = np.asarray(res['efficiencies'])
                orders = res.get('orders')
                summary = res.get('summary', '')
            else:
                # Fallback for legacy callers passing a raw ndarray.
                efficiencies = (res.efficiencies
                                if hasattr(res, 'efficiencies')
                                else np.asarray(res))
                orders = None
                summary = ''
            efficiencies = np.atleast_2d(efficiencies)
            if efficiencies.shape[0] == len(wavelengths):
                efficiencies = efficiencies.T
            for i, eff in enumerate(efficiencies):
                label = (f'm={int(orders[i])}'
                         if orders is not None and i < len(orders)
                         else f'order {i}')
                ax.plot(wavelengths * 1e9, eff, label=label)
            ax.set_xlabel('Wavelength [nm]', color='#dde8f8',
                          fontfamily='monospace')
            ax.set_ylabel('Diffraction efficiency',
                          color='#dde8f8', fontfamily='monospace')
            ax.legend(fontsize=8)
            if not summary:
                summary = (f'Computed {efficiencies.shape[0]} order(s) '
                           f'over {len(wavelengths)} wavelengths.')
            self.summary.setPlainText(summary)
        except (TypeError, ValueError, KeyError,
                AttributeError, IndexError) as exc:
            self.summary.setPlainText(
                f'Could not unpack thin-grating result: '
                f'{type(exc).__name__}: {exc}')
        self.fig.tight_layout()
        self.canvas.draw_idle()

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
