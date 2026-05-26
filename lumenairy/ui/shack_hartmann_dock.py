"""
Shack-Hartmann sensing dock (3.6).

Wraps :func:`lumenairy.shack_hartmann` for wavefront-sensor-style
diagnostics: lay a virtual microlens array on the focal-plane field
and recover per-lenslet centroids, slopes, and reconstructed Zernike
coefficients.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QTextEdit,
    QSizePolicy,
)
from PySide6.QtGui import QFont
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


class ShackHartmannDock(QWidget):
    """Run a Shack-Hartmann sensor on the most recent wave-optics
    focal field (or on a unit plane wave if no run has happened yet).
    Useful for cross-validating the wave-optics-derived OPD against
    the ray-trace-derived one.
    """

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._last_field = None     # populated by external set_field
        self._last_dx = None
        self._last_result = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        param = QGroupBox('Sensor parameters')
        form = QFormLayout(param)
        self.spin_pitch_um = QDoubleSpinBox()
        self.spin_pitch_um.setRange(1.0, 1e6)
        self.spin_pitch_um.setValue(150.0)
        self.spin_pitch_um.setDecimals(2)
        self.spin_pitch_um.setSuffix(' µm')
        self.spin_pitch_um.setToolTip('Lenslet pitch (centre-to-centre).')
        form.addRow('Lenslet pitch:', self.spin_pitch_um)
        self.spin_fl_mm = QDoubleSpinBox()
        self.spin_fl_mm.setRange(0.1, 1e6)
        self.spin_fl_mm.setValue(5.0)
        self.spin_fl_mm.setDecimals(3)
        self.spin_fl_mm.setSuffix(' mm')
        self.spin_fl_mm.setToolTip('Lenslet focal length.')
        form.addRow('Lenslet focal length:', self.spin_fl_mm)
        self.spin_n_zern = QSpinBox()
        self.spin_n_zern.setRange(0, 78)
        self.spin_n_zern.setValue(15)
        self.spin_n_zern.setToolTip(
            'Number of Zernike modes to fit (OSA index).  0 reports '
            'centroid slopes only.')
        form.addRow('Zernike modes:', self.spin_n_zern)
        outer.addWidget(param)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton('▶ Run Shack-Hartmann')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        run_row.addStretch()
        outer.addLayout(run_row)

        self.fig = Figure(figsize=(6, 3.4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas, stretch=1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        outer.addWidget(self.summary)
        self._draw_empty()

    def set_field(self, E, dx):
        """Receive a wave-optics field from the Wave Optics dock."""
        self._last_field = np.asarray(E)
        self._last_dx = float(dx)

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Run the Wave Optics dock first, then run a\n'
                'Shack-Hartmann pass to recover per-lenslet slopes\n'
                'and a reconstructed Zernike spectrum.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _run(self):
        if self._last_field is None or self._last_dx is None:
            self.summary.setPlainText(
                'No field available. Run the Wave Optics dock first '
                '(F5); the focal-plane field is then routed here '
                'automatically.')
            return
        try:
            import lumenairy as la
            pitch = float(self.spin_pitch_um.value()) * 1e-6
            fl = float(self.spin_fl_mm.value()) * 1e-3
            n_zern = int(self.spin_n_zern.value())
            wv = self.sm.wavelength_m
            res = la.shack_hartmann(
                self._last_field, self._last_dx,
                lenslet_pitch=pitch, lenslet_focal_length=fl,
                wavelength=wv, n_zernike=n_zern)
        except Exception as exc:
            self.summary.setPlainText(
                f'shack_hartmann failed: {type(exc).__name__}: {exc}')
            return
        self._last_result = res
        self._draw_result(res)
        self._summarise(res)

    def _draw_result(self, res):
        self.fig.clear()
        # Two-axis layout: slopes (left) + reconstructed Zernike bar
        # chart (right) when both are available.
        try:
            sx, sy = res.slopes_x, res.slopes_y
            ax1 = self.fig.add_subplot(121)
            ax1.set_facecolor('#0a0c10')
            mag = np.hypot(sx, sy)
            im = ax1.imshow(mag, cmap='viridis', origin='lower')
            ax1.set_title('Slope magnitude', color='#dde8f8',
                          fontfamily='monospace', fontsize=9)
            ax1.tick_params(colors='#7a94b8', labelsize=8)
        except Exception:
            ax1 = self.fig.add_subplot(121)
            ax1.text(0.5, 0.5, '(no slopes)', color='#7a94b8',
                     transform=ax1.transAxes, ha='center')
        try:
            zern = getattr(res, 'zernike_coeffs', None)
            if zern is not None and len(zern):
                ax2 = self.fig.add_subplot(122)
                ax2.set_facecolor('#0a0c10')
                ax2.bar(range(len(zern)), zern * 1e9,
                        color='#5cb8ff', edgecolor='#2a3548')
                ax2.set_xlabel('OSA index', color='#dde8f8',
                               fontfamily='monospace', fontsize=8)
                ax2.set_ylabel('coeff [nm]', color='#dde8f8',
                               fontfamily='monospace', fontsize=8)
                ax2.tick_params(colors='#7a94b8', labelsize=7)
                for s in ax2.spines.values():
                    s.set_color('#334054')
        except Exception:
            pass
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _summarise(self, res):
        lines = [
            f'Lenslets: {getattr(res, "n_lenslets", "?")}',
        ]
        try:
            lines.append(
                f'Slope RMS: x={np.std(res.slopes_x)*1e6:.3f} µrad, '
                f'y={np.std(res.slopes_y)*1e6:.3f} µrad')
        except Exception:
            pass
        try:
            zern = res.zernike_coeffs
            lines.append('First Zernike coefficients (nm):')
            for i, z in enumerate(zern[:9]):
                lines.append(f'  Z{i:2d}: {z*1e9:+8.3f}')
        except Exception:
            pass
        self.summary.setPlainText('\n'.join(lines))

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
