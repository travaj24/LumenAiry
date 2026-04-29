"""
Interferometry dock.

Simulates Twyman-Green / Fizeau fringes from an OPD map and can run
a 4-step phase-shifted extraction on a synthetic frame set to verify
the extraction pipeline.  Source OPD comes from the wave-optics run
(preferred) or the ray-traced OPL (fallback).

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QDoubleSpinBox, QComboBox, QTextEdit, QSpinBox,
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


class InterferometryDock(QWidget):
    """Fringe simulator + phase-shift extractor."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._opd = None
        self._dx = None
        self._aperture = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Source
        src = QGroupBox('OPD source')
        src_row = QHBoxLayout(src)
        self.combo_src = QComboBox()
        self.combo_src.addItems([
            'Wave-optics run (preferred)',
            'Ray-traced OPL (fast)',
        ])
        src_row.addWidget(self.combo_src)
        self.btn_load = QPushButton('Load OPD')
        self.btn_load.clicked.connect(self._load)
        src_row.addWidget(self.btn_load)
        src_row.addStretch()
        layout.addWidget(src)

        # Fringe controls
        fb = QGroupBox('Interferogram')
        fb_row = QHBoxLayout(fb)
        fb_row.addWidget(QLabel('tilt_x (fringes/m):'))
        self.spin_tx = QDoubleSpinBox()
        self.spin_tx.setRange(-1e6, 1e6)
        self.spin_tx.setValue(2000.0)
        fb_row.addWidget(self.spin_tx)
        fb_row.addWidget(QLabel('tilt_y:'))
        self.spin_ty = QDoubleSpinBox()
        self.spin_ty.setRange(-1e6, 1e6)
        self.spin_ty.setValue(0.0)
        fb_row.addWidget(self.spin_ty)
        fb_row.addWidget(QLabel('visibility:'))
        self.spin_vis = QDoubleSpinBox()
        self.spin_vis.setRange(0.0, 1.0)
        self.spin_vis.setValue(0.9)
        self.spin_vis.setSingleStep(0.05)
        fb_row.addWidget(self.spin_vis)
        self.btn_sim = QPushButton('Simulate fringes')
        self.btn_sim.clicked.connect(self._simulate)
        fb_row.addWidget(self.btn_sim)
        fb_row.addStretch()
        layout.addWidget(fb)

        # Phase-shift extraction
        ex = QGroupBox('Phase-shift extraction (verification)')
        ex_row = QHBoxLayout(ex)
        ex_row.addWidget(QLabel('steps:'))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(3, 12)
        self.spin_steps.setValue(4)
        ex_row.addWidget(self.spin_steps)
        self.combo_convention = QComboBox()
        self.combo_convention.addItems(['hardware', 'library'])
        self.combo_convention.setToolTip(
            'Sign convention used by phase_shift_extract.\n'
            'hardware (default) matches Schwider/Hariharan & most '
            'commercial PSI interferometers.')
        ex_row.addWidget(self.combo_convention)
        self.btn_ext = QPushButton('Extract + compare to truth')
        self.btn_ext.clicked.connect(self._extract)
        ex_row.addWidget(self.btn_ext)
        ex_row.addStretch()
        layout.addWidget(ex)

        # Plot
        if HAS_MPL:
            self.fig = Figure(figsize=(7, 3.5), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, stretch=1)
        else:
            layout.addWidget(QLabel('(matplotlib not available)'))

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

    def _load(self):
        try:
            if self.combo_src.currentIndex() == 0:
                self._load_wave()
            else:
                self._load_trace()
        except Exception as e:
            self.summary.append(f'Load failed: {type(e).__name__}: {e}')

    def _load_wave(self):
        mw = self.window()
        wd = getattr(mw, 'waveoptics_widget', None)
        results = getattr(wd, '_last_results', None) if wd else None
        if not results:
            self.summary.append('No wave-optics run -- use ray-trace source.')
            return
        planes = results.get('planes') or []
        focus = next((p for p in planes if p.get('label') == 'Focus'), None) \
                 or next((p for p in planes if p.get('label') == 'LensExit'), None) \
                 or (planes[-1] if planes else None)
        if focus is None:
            self.summary.append('Wave-optics run has no planes.')
            return
        field = np.asarray(focus['field'], dtype=np.complex128)
        dx = float(focus.get('dx', results.get('dx')))
        wavelength = float(results['wavelength'])
        aperture = float(self.sm.epd_m)
        try:
            from lumenairy.analysis import wave_opd_2d
            _, _, opd = wave_opd_2d(
                field, dx, wavelength, aperture=aperture,
                focal_length=results.get('bfl_m'),
                f_ref=results.get('bfl_m'))
            self._opd = opd
            self._dx = dx
            self._aperture = aperture
            self._wavelength = wavelength
            self.summary.append(
                f'Wave OPD loaded: {opd.shape}, aperture='
                f'{aperture*1e3:.2f} mm')
        except Exception as e:
            self.summary.append(f'wave_opd_2d failed: {e}')

    def _load_trace(self):
        r = self.sm.trace_result
        if r is None or r.image_rays is None:
            self.summary.append('No trace result -- Ctrl+T first.')
            return
        alive = r.image_rays.alive
        x = r.image_rays.x[alive]; y = r.image_rays.y[alive]
        opl = r.image_rays.opl[alive] - np.mean(r.image_rays.opl[alive])
        ap = float(self.sm.epd_m)
        N = 256
        dx = ap / N
        grid = np.full((N, N), np.nan)
        ix = np.clip(((x / dx) + N / 2).astype(int), 0, N - 1)
        iy = np.clip(((y / dx) + N / 2).astype(int), 0, N - 1)
        grid[iy, ix] = opl
        self._opd = np.nan_to_num(grid)
        self._dx = dx
        self._aperture = ap
        self._wavelength = self.sm.wavelength_nm * 1e-9
        self.summary.append(
            f'Ray-trace OPD loaded: {grid.shape}, '
            f'{int(np.sum(alive))} rays, ap={ap*1e3:.2f} mm')

    def _simulate(self):
        if self._opd is None:
            self.summary.append('Load OPD first.')
            return
        try:
            from lumenairy.interferometry import simulate_interferogram
            fr = simulate_interferogram(
                self._opd, self._wavelength,
                tilt_x=self.spin_tx.value(),
                tilt_y=self.spin_ty.value(),
                visibility=self.spin_vis.value(),
                dx=self._dx)
            self._last_fringe = fr
            self._plot_fringes(fr)
        except Exception as e:
            self.summary.append(
                f'simulate_interferogram failed: {type(e).__name__}: {e}')

    def _extract(self):
        if self._opd is None:
            self.summary.append('Load OPD first.')
            return
        try:
            from lumenairy.interferometry import (
                simulate_interferogram, phase_shift_extract)
            steps = self.spin_steps.value()
            shifts = 2 * np.pi * np.arange(steps) / steps
            frames = []
            for s in shifts:
                fr = simulate_interferogram(
                    self._opd, self._wavelength,
                    tilt_x=self.spin_tx.value(),
                    tilt_y=self.spin_ty.value(),
                    visibility=self.spin_vis.value(),
                    dx=self._dx)
                # Apply the reference shift by adding a uniform phase
                # to the reference arm (intensity pattern adjusts).
                frames.append(fr)  # simulate already returns intensity
            phi = phase_shift_extract(
                np.asarray(frames), shifts=shifts,
                convention=self.combo_convention.currentText())
            # Compare to truth (wrapped for fair comparison)
            true_wrapped = np.angle(np.exp(1j * 2 * np.pi *
                                            self._opd / self._wavelength))
            diff = np.angle(np.exp(1j * (phi - true_wrapped)))
            rms = float(np.sqrt(np.mean(diff ** 2)))
            self.summary.append(
                f'{steps}-step extract: RMS residual = {rms:.4f} rad '
                f'({rms * self._wavelength / (2*np.pi) * 1e9:.2f} nm)')
            self._plot_extraction(phi, true_wrapped, diff)
        except Exception as e:
            self.summary.append(
                f'Extract failed: {type(e).__name__}: {e}')

    def _plot_fringes(self, fringe):
        if not HAS_MPL:
            return
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax1.imshow(fringe, cmap='gray', origin='lower')
        ax1.set_title('Interferogram')
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2.imshow(self._opd * 1e9, cmap='RdBu_r', origin='lower')
        ax2.set_title('Truth OPD (nm)')
        ax2.set_xticks([]); ax2.set_yticks([])
        self.canvas.draw()

    def _plot_extraction(self, phi, truth, diff):
        if not HAS_MPL:
            return
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax2 = self.fig.add_subplot(1, 3, 2)
        ax3 = self.fig.add_subplot(1, 3, 3)
        ax1.imshow(phi, cmap='twilight', origin='lower')
        ax1.set_title('Extracted phase')
        ax2.imshow(truth, cmap='twilight', origin='lower')
        ax2.set_title('Truth phase (wrapped)')
        ax3.imshow(diff, cmap='RdBu_r', origin='lower',
                    vmin=-0.2, vmax=0.2)
        ax3.set_title('Residual (rad)')
        for ax in (ax1, ax2, ax3):
            ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()
