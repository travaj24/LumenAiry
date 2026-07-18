"""
PSF / MTF / polychromatic-Strehl dock.

Shows the intensity PSF (log-scaled), radial MTF, and the
polychromatic Strehl ratio across the user's wavelength list.

Sources: uses the wave-optics dock's latest exit-pupil field when
available, otherwise builds a synthetic pupil with ray-traced OPD
so the dock is usable without a wave run.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QGroupBox, QTextEdit, QComboBox,
    QProgressBar,
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


class _PolyStrehlWorker(QThread):
    finished_result = Signal(object)

    def __init__(self, prescription, wavelengths, weights, N, dx):
        super().__init__()
        self.prescription = prescription
        self.wavelengths = wavelengths
        self.weights = weights
        self.N = N
        self.dx = dx

    def run(self):
        try:
            from lumenairy.analysis import polychromatic_strehl
            s_poly, s_each, z_each = polychromatic_strehl(
                self.prescription, self.wavelengths, self.weights,
                N=self.N, dx=self.dx)
            self.finished_result.emit({
                'success': True,
                'strehl_poly': float(s_poly),
                'strehl_each': np.asarray(s_each),
                'z_each': np.asarray(z_each),
            })
        except Exception as e:
            self.finished_result.emit({
                'success': False,
                'error': f'{type(e).__name__}: {e}',
            })


class PSFMTFDock(QWidget):
    """PSF + MTF + polychromatic-Strehl analysis."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._pupil = None
        self._dx = None
        self._wavelength = None
        self._focal_length = None
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Source selector
        src_box = QGroupBox('Pupil source')
        src_layout = QHBoxLayout(src_box)
        self.combo_source = QComboBox()
        self.combo_source.addItems([
            'Latest wave-optics run (recommended)',
            'Ray-traced OPD + geometric pupil (fast)',
        ])
        src_layout.addWidget(self.combo_source)
        self.btn_load = QPushButton('Load pupil')
        self.btn_load.clicked.connect(self._load_pupil)
        src_layout.addWidget(self.btn_load)
        src_layout.addStretch()
        layout.addWidget(src_box)

        # PSF / MTF controls
        psf_box = QGroupBox('PSF / MTF')
        psf_layout = QHBoxLayout(psf_box)
        psf_layout.addWidget(QLabel('oversample:'))
        self.spin_over = QSpinBox()
        self.spin_over.setRange(1, 16)
        self.spin_over.setValue(2)
        self.spin_over.setToolTip(
            'FFT zero-pad factor.  Higher = finer focal-plane '
            'sampling (at memory cost).')
        psf_layout.addWidget(self.spin_over)
        self.btn_psf = QPushButton('▶ Compute PSF + MTF  (F9)')
        self.btn_psf.setObjectName('run_button')
        self.btn_psf.clicked.connect(self._compute_psf_mtf)
        # Alias for the F-key dispatcher (which looks for btn_run /
        # btn_compute / btn_recompute on the dock widget).
        self.btn_compute = self.btn_psf
        psf_layout.addWidget(self.btn_psf)
        psf_layout.addStretch()
        layout.addWidget(psf_box)

        # Polychromatic Strehl
        poly_box = QGroupBox('Polychromatic Strehl (uses optimizer wavelength list)')
        poly_layout = QHBoxLayout(poly_box)
        poly_layout.addWidget(QLabel('grid N:'))
        self.spin_N = QSpinBox()
        self.spin_N.setRange(64, 2048)
        self.spin_N.setValue(256)
        poly_layout.addWidget(self.spin_N)
        poly_layout.addWidget(QLabel('dx (um):'))
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(0.01, 100)
        self.spin_dx.setValue(16.0)
        self.spin_dx.setDecimals(2)
        poly_layout.addWidget(self.spin_dx)
        self.btn_poly = QPushButton('▶ Compute poly-Strehl')
        self.btn_poly.setObjectName('run_button')
        self.btn_poly.clicked.connect(self._compute_poly)
        poly_layout.addWidget(self.btn_poly)
        self.progress_poly = QProgressBar()
        self.progress_poly.setRange(0, 0)
        self.progress_poly.setVisible(False)
        poly_layout.addWidget(self.progress_poly)
        poly_layout.addStretch()
        layout.addWidget(poly_box)

        # Plot
        if HAS_MPL:
            self.fig = Figure(figsize=(7, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
            self.canvas.setMinimumSize(0, 0)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, stretch=1)
        else:
            layout.addWidget(QLabel('(matplotlib not available)'))

        # Summary
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

    # ------------------------------------------------------------------
    # Pupil acquisition
    # ------------------------------------------------------------------

    def _load_pupil(self):
        idx = self.combo_source.currentIndex()
        try:
            if idx == 0:
                self._load_from_waveoptics()
            else:
                self._load_from_raytrace()
        except Exception as e:
            self.summary.append(
                f'Pupil load failed: {type(e).__name__}: {e}')

    def _load_from_waveoptics(self):
        mw = self.window()
        wd = getattr(mw, 'waveoptics_widget', None)
        if wd is None:
            self.summary.append('No wave-optics dock present.')
            return
        results = getattr(wd, '_last_results', None)
        if not results or not results.get('planes'):
            self.summary.append(
                'No wave-optics run yet.  Switch to "Ray-traced OPD" '
                'for a fast path, or run wave optics first.')
            return
        planes = results['planes']
        exit_plane = next(
            (p for p in planes if p.get('label') == 'LensExit'), None) or planes[-1]
        self._pupil = np.asarray(exit_plane['field'], dtype=np.complex128)
        self._dx = float(exit_plane.get('dx', results.get('dx')))
        self._wavelength = float(results['wavelength'])
        self._focal_length = float(results.get('bfl_m') or 0.0)
        self.summary.append(
            f'Wave-optics pupil loaded: {self._pupil.shape}, '
            f'dx={self._dx*1e6:.2f} um, f={self._focal_length*1e3:.2f} mm')

    def _load_from_raytrace(self):
        """Build a geometric pupil from the ray-traced OPD: phase from
        OPL, amplitude = aperture mask."""
        from lumenairy.raytrace import (
            surfaces_from_prescription, system_abcd)
        pres = self.sm.to_prescription()
        surfs = surfaces_from_prescription(pres)
        _, efl, bfl, _ = system_abcd(surfs, self.sm.wavelength_nm * 1e-9)
        result = self.sm.trace_result
        if result is None or result.image_rays is None:
            self.summary.append(
                'No trace available -- hit Ctrl+T first.')
            return
        rays = result.image_rays
        alive = rays.alive
        x = rays.x[alive]
        y = rays.y[alive]
        opl = rays.opl[alive] - np.mean(rays.opl[alive])
        ap = float(self.sm.epd_m)
        N = 256
        dx = ap / N
        # v4.15 (P1-UI-7): bounds-mask the rays BEFORE indexing so any
        # ray landing outside the pupil grid is dropped rather than
        # snapped to the boundary via ``np.clip``.  Pre-4.15 the clip
        # silently piled out-of-aperture rays onto the {0, N-1} edges
        # and polluted the pupil edge with garbage OPDs unrelated to
        # the actual aperture clipping; geometric residue from a wide-
        # angle trace would corrupt the apparent rim of the pupil.
        ix_raw = ((x / dx) + N / 2).astype(int)
        iy_raw = ((y / dx) + N / 2).astype(int)
        in_bounds = ((ix_raw >= 0) & (ix_raw < N)
                     & (iy_raw >= 0) & (iy_raw < N))
        ix = ix_raw[in_bounds]
        iy = iy_raw[in_bounds]
        opl_in = opl[in_bounds]
        # v4.15 (P1-UI-6): accumulate mean OPD per pixel rather than
        # last-write-wins.  The pre-4.15 ``opd_grid[iy, ix] = opl``
        # depended on numpy's right-to-left iteration order, so for
        # any pixel hit by multiple rays the stored OPD was the LAST
        # ray to map there -- physically meaningless (a function of
        # the ray-loop order, not the optics).  Following the
        # ``analysis/core.py`` accumulator pattern (lines 986-988,
        # ``np.bincount`` with weights divided by count), we
        # accumulate sum-of-OPL and ray-count per pixel via
        # ``np.add.at`` and divide at the end.  Median was the other
        # option mentioned in the audit; mean matches the analysis
        # module's azimuthal-average idiom and behaves identically
        # for the 1-ray-per-pixel case.
        sum_grid = np.zeros((N, N), dtype=np.float64)
        cnt_grid = np.zeros((N, N), dtype=np.int64)
        np.add.at(sum_grid, (iy, ix), opl_in)
        np.add.at(cnt_grid, (iy, ix), 1)
        valid = cnt_grid > 0
        opd_grid = np.zeros((N, N), dtype=np.float64)
        opd_grid[valid] = sum_grid[valid] / cnt_grid[valid]
        k0 = 2 * np.pi / (self.sm.wavelength_nm * 1e-9)
        phase = k0 * opd_grid
        # v5.24.x (audit S4-20): use the CONJUGATE phase convention
        # ``exp(-i k0 OPD)`` to match the wave-optics lens phase-screens
        # (``elements/lenses.py``: ``E * exp(-1j*k0*opd)``).  Pre-fix the
        # ray-traced pupil used ``exp(+i k0 OPD)`` -- the opposite sign
        # -- so the PSF computed from a ray-traced pupil came out
        # MIRROR-FLIPPED relative to the PSF from a wave-optics field,
        # for the same physical system.  Aligning the sign makes the two
        # pupil sources interchangeable in the downstream FFT-based PSF.
        # v4.15: dtype-aware sentinel migration.  Pre-4.15 the
        # ``0.0 + 0.0j`` literal forced a complex128 upcast via numpy's
        # mixed-dtype rules (P3-rated in AUDIT_V4_14_1_2026_05_17.md
        # P1-NEW-4); explicit ``.astype(complex_t)`` recovery makes the
        # pupil dtype deterministic regardless of which numpy version
        # the user is running.
        _pupil_full = np.exp(-1j * phase)
        self._pupil = np.where(valid, _pupil_full,
                                np.zeros((), dtype=_pupil_full.dtype))
        self._dx = dx
        self._wavelength = self.sm.wavelength_nm * 1e-9
        self._focal_length = float(bfl) if np.isfinite(bfl) else float(efl)
        n_dropped = int(np.sum(~in_bounds))
        drop_msg = f', {n_dropped} out-of-bounds' if n_dropped else ''
        self.summary.append(
            f'Ray-trace pupil built: {self._pupil.shape}, '
            f'dx={dx*1e6:.2f} um, f={self._focal_length*1e3:.2f} mm, '
            f'{int(np.sum(valid))} pixels filled{drop_msg}')

    # ------------------------------------------------------------------
    # PSF + MTF
    # ------------------------------------------------------------------

    def _compute_psf_mtf(self):
        if self._pupil is None:
            self.summary.append('Load a pupil first.')
            return
        try:
            from lumenairy.analysis import compute_psf, compute_mtf
            over = self.spin_over.value()
            psf = compute_psf(
                self._pupil, self._wavelength,
                self._focal_length or 0.01, self._dx,
                oversample=over, normalize='peak')
            mtf = compute_mtf(psf)
            self._plot_psf_mtf(psf, mtf)
            self.summary.append(
                f'PSF: {psf.shape}, peak={psf.max():.3e}; '
                f'MTF DC={mtf[0, 0]:.3f}')
        except Exception as e:
            self.summary.append(
                f'PSF/MTF failed: {type(e).__name__}: {e}')

    def _plot_psf_mtf(self, psf, mtf):
        if not HAS_MPL:
            return
        self.fig.clear()
        ax_psf = self.fig.add_subplot(1, 2, 1)
        ax_mtf = self.fig.add_subplot(1, 2, 2)
        log_psf = np.log10(np.clip(psf / psf.max(), 1e-6, 1.0))
        ax_psf.imshow(log_psf, origin='lower', cmap='magma',
                       vmin=-6, vmax=0)
        ax_psf.set_title('PSF (log, norm.)')
        ax_psf.set_xticks([]); ax_psf.set_yticks([])

        # radial MTF
        N = mtf.shape[0]
        yy, xx = np.indices(mtf.shape) - N / 2
        rr = np.sqrt(xx * xx + yy * yy).astype(int)
        rbin = np.bincount(rr.ravel(), weights=np.fft.fftshift(mtf).ravel())
        rnorm = np.bincount(rr.ravel())
        rad = rbin / np.maximum(rnorm, 1)
        r = np.arange(len(rad)) / (N / 2)
        ax_mtf.plot(r, rad, color='#5cb8ff', lw=1.2)
        ax_mtf.set_xlabel('spatial freq (norm.)')
        ax_mtf.set_ylabel('MTF')
        ax_mtf.set_xlim(0, 1)
        ax_mtf.set_ylim(0, 1.05)
        ax_mtf.grid(alpha=0.2)
        ax_mtf.set_title('Radial MTF')
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Polychromatic Strehl
    # ------------------------------------------------------------------

    def _compute_poly(self):
        wls = sorted(set(
            float(w) * 1e-9 for w in self.sm.wavelengths_nm))
        if len(wls) < 2:
            self.summary.append(
                'Polychromatic Strehl needs >=2 wavelengths.  Add '
                'more via the Optimizer dock (+lambda).')
            return
        weights = np.ones(len(wls)) / len(wls)
        pres = self.sm.to_prescription()
        N = self.spin_N.value()
        dx = self.spin_dx.value() * 1e-6
        self.btn_poly.setEnabled(False)
        self.progress_poly.setVisible(True)
        self._worker = _PolyStrehlWorker(pres, wls, weights, N, dx)
        self._worker.finished_result.connect(self._on_poly_finished)
        self._worker.start()

    def _on_poly_finished(self, res):
        self.btn_poly.setEnabled(True)
        self.progress_poly.setVisible(False)
        self._worker = None
        if not res.get('success'):
            self.summary.append(f'poly-Strehl failed: {res.get("error")}')
            return
        s_poly = res['strehl_poly']
        s_each = res['strehl_each']
        z_each = res['z_each']
        self.summary.append(
            f'\nPolychromatic Strehl = {s_poly:.4f}')
        self.summary.append(f'{"wv (nm)":>10s}{"Strehl":>10s}{"z_best (mm)":>14s}')
        for wv, s, z in zip(self.sm.wavelengths_nm, s_each, z_each):
            self.summary.append(
                f'{wv:>10.1f}{s:>10.4f}{z*1e3:>14.5f}')

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
