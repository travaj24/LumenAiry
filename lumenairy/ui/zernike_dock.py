# v5.4 (audit P3-B): add normalization + weighting controls
"""
Zernike decomposition dock -- interactive Zernike coefficient viewer.

v5.4 (audit P3-B) adds two polish controls:
  * Normalization selector (library hard-codes OSA/ANSI -- non-OSA
    selections emit a summary warning until the library exposes a
    ``normalization=`` kwarg).
  * Weighting mask -- applied post-hoc (we multiply the OPD by the
    mask before ``zernike_decompose``); default ``'none'`` preserves
    v5.3.2 numerical behavior bit-for-bit.

Author: Andrew Traverso
"""

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTextEdit, QComboBox,
    QFileDialog,
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

        self.btn_decompose = QPushButton('▶ Decompose (wave OPD)  (F8)')
        self.btn_decompose.setObjectName('run_button')
        self.btn_decompose.clicked.connect(self.decompose)
        # F-key dispatcher alias.
        self.btn_run = self.btn_decompose
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

        # v5.4 (P3-B): normalization selector (library locked to OSA).
        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel('Normalization:'))
        self.combo_normalization = QComboBox()
        self.combo_normalization.addItems(
            ['OSA', 'Noll', 'Fringe', 'Standard'])
        self.combo_normalization.setToolTip(
            'Library hard-codes OSA/ANSI; non-OSA choices warn until '
            'a normalization= kwarg lands.')
        norm_row.addWidget(self.combo_normalization)
        norm_row.addStretch()
        layout.addLayout(norm_row)

        # v5.4 (P3-B): weighting mask group (post-hoc OPD multiply;
        # default 'none' reproduces v5.3.2 output bit-for-bit).
        self.group_weighting = QGroupBox('Weighting')
        w_layout = QHBoxLayout(self.group_weighting)
        w_layout.addWidget(QLabel('Source:'))
        self.combo_weighting_source = QComboBox()
        self.combo_weighting_source.addItems(
            ['none', 'circular_aperture', 'from_file'])
        self.combo_weighting_source.currentTextChanged.connect(
            self._on_weighting_source_changed)
        w_layout.addWidget(self.combo_weighting_source)
        self.lbl_weighting_radius = QLabel('Radius (pupil frac):')
        self.spin_weighting_radius = QDoubleSpinBox()
        self.spin_weighting_radius.setRange(0.0, 1.0)
        self.spin_weighting_radius.setSingleStep(0.05)
        self.spin_weighting_radius.setDecimals(3)
        self.spin_weighting_radius.setValue(1.0)
        w_layout.addWidget(self.lbl_weighting_radius)
        w_layout.addWidget(self.spin_weighting_radius)
        self.btn_weighting_file = QPushButton('Browse...')
        self.btn_weighting_file.clicked.connect(self._pick_weighting_file)
        w_layout.addWidget(self.btn_weighting_file)
        self.lbl_weighting_file = QLabel('(no file)')
        self.lbl_weighting_file.setStyleSheet('color:#7a94b8')
        w_layout.addWidget(self.lbl_weighting_file, stretch=1)
        self._weighting_file_path = None
        self._on_weighting_source_changed('none')
        layout.addWidget(self.group_weighting)

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
            self._warn_if_nonstandard_normalization()
            opd_in = self._apply_weighting(grid, dx, ap)
            coeffs, names = zernike_decompose(
                opd_in, dx, ap, n_modes=n_modes)
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
            self._warn_if_nonstandard_normalization()
            opd_in = self._apply_weighting(
                self._opd_map, self._dx, self._aperture)
            coeffs, names = zernike_decompose(
                opd_in, self._dx, self._aperture, n_modes=n_modes)
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

    # v5.4 (P3-B) helpers: normalization warning + weighting mask

    def _warn_if_nonstandard_normalization(self):
        """Warn when the user selects a non-OSA convention (no-op for OSA)."""
        norm = self.combo_normalization.currentText()
        if norm != 'OSA':
            self.summary.append(
                f"NOTE: normalization='{norm}' not yet wired through; "
                f"library hard-codes OSA/ANSI.  Fit used OSA.")

    def _on_weighting_source_changed(self, src):
        """Show/hide weighting sub-widgets based on the chosen source."""
        is_circ = (src == 'circular_aperture')
        is_file = (src == 'from_file')
        self.lbl_weighting_radius.setVisible(is_circ)
        self.spin_weighting_radius.setVisible(is_circ)
        self.btn_weighting_file.setVisible(is_file)
        self.lbl_weighting_file.setVisible(is_file)

    def _pick_weighting_file(self):
        """Open a file dialog to pick a .npy / .h5 mask."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select weighting mask',
            '', 'Mask files (*.npy *.h5 *.hdf5);;All files (*)')
        if path:
            self._weighting_file_path = path
            self.lbl_weighting_file.setText(os.path.basename(path))

    def _apply_weighting(self, opd_map, dx, aperture):
        """Multiply ``opd_map`` by the chosen mask (no-op for 'none')."""
        src = self.combo_weighting_source.currentText()
        if src == 'none' or opd_map is None:
            return opd_map
        try:
            Ny, Nx = opd_map.shape
            if src == 'circular_aperture':
                frac = float(self.spin_weighting_radius.value())
                r_cut = max(0.0, min(1.0, frac)) * 0.5 * aperture
                x = (np.arange(Nx) - Nx / 2) * dx
                y = (np.arange(Ny) - Ny / 2) * dx
                X, Y = np.meshgrid(x, y)
                mask = (X ** 2 + Y ** 2) <= (r_cut ** 2)
                self.summary.append(
                    f"Weighting: circular_aperture, frac={frac:.3f} "
                    f"({int(mask.sum())} active pixels)")
                return np.where(mask, opd_map, np.nan)
            if src == 'from_file':
                if not self._weighting_file_path:
                    self.summary.append(
                        "Weighting 'from_file' but no file chosen; "
                        "running unweighted.")
                    return opd_map
                path = self._weighting_file_path
                ext = os.path.splitext(path)[1].lower()
                if ext == '.npy':
                    mask = np.load(path)
                elif ext in ('.h5', '.hdf5'):
                    import h5py
                    with h5py.File(path, 'r') as f:
                        mask = np.asarray(f[next(iter(f.keys()))][...])
                else:
                    self.summary.append(
                        f"Weighting: unsupported ext '{ext}'; unweighted.")
                    return opd_map
                if mask.shape != opd_map.shape:
                    self.summary.append(
                        f"Weighting: shape mismatch {mask.shape} vs "
                        f"{opd_map.shape}; unweighted.")
                    return opd_map
                self.summary.append(
                    f"Weighting: from_file '{os.path.basename(path)}'")
                return opd_map * mask
        except Exception as e:
            self.summary.append(
                f'Weighting failed ({type(e).__name__}: {e}); unweighted.')
        return opd_map
