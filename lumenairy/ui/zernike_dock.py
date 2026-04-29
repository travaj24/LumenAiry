"""
Zernike decomposition dock — interactive Zernike coefficient viewer.

Shows bar chart of Zernike coefficients from the latest wave-optics
OPD map, with mode names and RMS contributions.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QGroupBox, QTextEdit,
)
from PySide6.QtGui import QFont

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class ZernikeDock(QWidget):
    """Zernike coefficient display and decomposition."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._opd_map = None
        self._coeffs = None
        self._names = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('N modes:'))
        self.spin_modes = QSpinBox()
        self.spin_modes.setRange(3, 45)
        self.spin_modes.setValue(21)
        ctrl.addWidget(self.spin_modes)

        self.btn_decompose = QPushButton('Decompose (wave OPD)')
        self.btn_decompose.clicked.connect(self.decompose)
        self.btn_decompose.setToolTip(
            'Fit Zernike modes to the latest wave-optics OPD map.\n'
            'Requires a prior wave-optics run (auto-populated on '
            'completion).  Use "Decompose from ray trace" for a fast '
            'geometric pre-check.')
        ctrl.addWidget(self.btn_decompose)

        self.btn_decompose_trace = QPushButton('From ray trace')
        self.btn_decompose_trace.clicked.connect(self.decompose_from_trace)
        self.btn_decompose_trace.setToolTip(
            'Decompose the ray-traced OPD (fast, purely geometric).\n'
            'No wave-optics run required; works straight off the current '
            'ray bundle.')
        ctrl.addWidget(self.btn_decompose_trace)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Plot
        if HAS_MPL:
            self.fig = Figure(figsize=(6, 3), dpi=100)
            self.canvas = FigureCanvasQTAgg(self.fig)
            layout.addWidget(self.canvas, stretch=1)
        else:
            layout.addWidget(QLabel('(matplotlib not available)'))

        # Text summary
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

    def set_opd_map(self, opd_map, dx, aperture):
        """Store a pre-extracted OPD map for later decomposition.

        ``opd_map`` must already be **unwrapped** and (ideally) have a
        reference sphere subtracted -- raw wrapped phase will produce
        meaningless coefficients.  Use :meth:`set_field` to skip the
        manual extraction step.
        """
        self._opd_map = opd_map
        self._dx = dx
        self._aperture = aperture
        self.summary.append(
            f'OPD map received: {opd_map.shape}, '
            f'dx={dx*1e6:.2f} um, ap={aperture*1e3:.2f} mm')

    def set_field(self, field, dx, wavelength, aperture,
                  focal_length=None):
        """Extract an OPD map from a complex field and store it.

        Calls the core ``wave_opd_2d`` so unwrap + reference-sphere
        subtraction (when ``focal_length`` is supplied) happen the same
        way as in any other consumer.  This is the recommended entry
        point from the wave-optics dock.
        """
        try:
            from lumenairy.analysis import wave_opd_2d
            extra = {}
            if focal_length is not None and np.isfinite(focal_length):
                extra['focal_length'] = focal_length
                extra['f_ref'] = focal_length
            _, _, opd = wave_opd_2d(
                field, dx, wavelength,
                aperture=aperture, **extra)
            self.set_opd_map(opd, dx, aperture)
        except Exception as e:
            self.summary.append(
                f'wave_opd_2d failed: {type(e).__name__}: {e}')

    def decompose_from_trace(self):
        """Fast geometric decomposition off the current ray bundle.

        Works without a prior wave-optics run -- we rasterise the
        ray-traced OPDs onto a small grid and fit Zernike modes
        directly.  Lower fidelity than the wave-OPD path but enough
        to confirm the dominant aberrations while iterating.
        """
        try:
            result = self.sm.last_trace_result
            if result is None or result.image_rays is None:
                self.summary.append(
                    'No trace available -- hit Ctrl+T (retrace) first.')
                return
            rays = result.image_rays
            alive = rays.alive
            if np.sum(alive) < 16:
                self.summary.append(
                    'Too few alive rays for a meaningful decomposition.')
                return
            ap = float(self.sm.epd_m)
            if not np.isfinite(ap) or ap <= 0:
                ap = 2.0 * float(np.max(np.sqrt(
                    rays.x[alive] ** 2 + rays.y[alive] ** 2)))
            opl = rays.opl[alive] - np.mean(rays.opl[alive])
            x = rays.x[alive]
            y = rays.y[alive]
            N = 128
            dx = ap / N
            grid = np.full((N, N), np.nan)
            ix = np.clip(((x / dx) + N / 2).astype(int), 0, N - 1)
            iy = np.clip(((y / dx) + N / 2).astype(int), 0, N - 1)
            grid[iy, ix] = opl
            from lumenairy.analysis import zernike_decompose
            n_modes = self.spin_modes.value()
            coeffs, names = zernike_decompose(
                grid, dx, ap, n_modes=n_modes)
            self._coeffs = coeffs
            self._names = names
            self._opd_map = grid
            self._dx = dx
            self._aperture = ap
            self.summary.clear()
            self.summary.append(
                f'Ray-trace decomposition: {int(np.sum(alive))} rays, '
                f'ap={ap*1e3:.2f} mm')
            self._plot_coefficients()
            self._print_summary()
        except Exception as e:
            self.summary.append(
                f'Ray-trace decomposition failed: {type(e).__name__}: {e}')

    def decompose(self):
        """Run Zernike decomposition on the stored OPD map."""
        if self._opd_map is None:
            self.summary.clear()
            self.summary.append(
                'No OPD map available yet.\n'
                '  - Run a wave-optics simulation (Wave Optics dock -> Run) '
                'to populate the OPD.\n'
                '  - Or click "From ray trace" for a fast geometric '
                'decomposition from the current ray bundle.')
            return
        # Defensive wrap check: a properly unwrapped converging-wavefront
        # OPD is many waves PV; values bounded by ~+/-pi at the edges
        # almost certainly mean the caller passed raw wrapped phase.
        try:
            finite = self._opd_map[np.isfinite(self._opd_map)]
            if finite.size:
                pv = float(np.ptp(finite))
                if 0.0 < pv <= 2 * np.pi + 1e-6:
                    self.summary.append(
                        f'WARNING: OPD PV is {pv:.3f} rad ~ <=2*pi. '
                        f'Looks like wrapped phase, not unwrapped OPD. '
                        f'Use set_field() to extract via wave_opd_2d.')
        except Exception:
            pass
        try:
            from lumenairy.analysis import zernike_decompose
            n_modes = self.spin_modes.value()
            coeffs, names = zernike_decompose(
                self._opd_map, self._dx, self._aperture, n_modes=n_modes)
            self._coeffs = coeffs
            self._names = names
            self._plot_coefficients()
            self._print_summary()
        except Exception as e:
            self.summary.append(f'Decomposition failed: {type(e).__name__}: {e}')

    def _plot_coefficients(self):
        if not HAS_MPL or self._coeffs is None:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        n = len(self._coeffs)
        colors = ['#4488cc' if abs(c) > 0 else '#333' for c in self._coeffs]
        ax.bar(range(n), self._coeffs * 1e9, color=colors, edgecolor='none')
        ax.set_xlabel('Zernike mode (OSA index)')
        ax.set_ylabel('Coefficient [nm]')
        ax.set_title('Zernike Decomposition')
        ax.axhline(0, color='#555', lw=0.5)
        ax.set_xlim(-0.5, n - 0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    def _print_summary(self):
        if self._coeffs is None:
            return
        self.summary.clear()
        self.summary.append('Zernike coefficients (nm RMS):')
        total_sq = 0.0
        for j, (c, name) in enumerate(zip(self._coeffs, self._names)):
            if abs(c) > 1e-12:
                self.summary.append(
                    f'  Z{j:2d} {name:30s} {c*1e9:+8.3f} nm')
                if j >= 3:  # exclude piston/tilts for RMS
                    total_sq += c ** 2
        rms = np.sqrt(total_sq)
        self.summary.append(
            f'\nRMS wavefront error (modes 3+): {rms*1e9:.3f} nm '
            f'= {rms/(self.sm.wavelength_nm*1e-9):.4f} waves')
