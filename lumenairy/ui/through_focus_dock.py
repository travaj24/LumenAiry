"""
Through-focus scan dock.

Sweeps the axial position of the image plane and reports Strehl,
peak intensity, RMS radius, and d4-sigma widths as a function of z.
Feeds off the wave-optics dock's last exit-pupil field (no manual
wiring required) and runs the scan in a background thread with the
core ``through_focus_scan`` progress hook piped into a determinate
progress bar.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QProgressBar, QGroupBox, QTextEdit,
    QCheckBox, QFileDialog,
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


class ThroughFocusWorker(QThread):
    """Background thread for the through-focus scan."""
    fine_progress = Signal(float, str)
    finished_result = Signal(object)   # ThroughFocusResult or None

    def __init__(self, E_exit, dx, wavelength, z_values,
                 bucket_radius=None, ideal_peak=None):
        super().__init__()
        self.E_exit = E_exit
        self.dx = dx
        self.wavelength = wavelength
        self.z_values = z_values
        self.bucket_radius = bucket_radius
        self.ideal_peak = ideal_peak

    def _on_progress(self, stage, fraction, message=''):
        self.fine_progress.emit(float(fraction), str(message))

    def run(self):
        try:
            from lumenairy.through_focus import through_focus_scan
            res = through_focus_scan(
                self.E_exit, self.dx, self.wavelength, self.z_values,
                bucket_radius=self.bucket_radius,
                ideal_peak=self.ideal_peak,
                verbose=False, progress=self._on_progress)
            self.finished_result.emit(res)
        except Exception as e:
            self.finished_result.emit(
                {'error': f'{type(e).__name__}: {e}'})


class ThroughFocusDock(QWidget):
    """Axial through-focus sweep with Strehl / peak / RMS / d4sigma plots."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None
        self._last_result = None
        # Wave-optics context -- populated by set_source_field().
        self._E_exit = None
        self._dx = None
        self._wavelength = None
        self._z_center = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Source-field status row ──
        status_box = QGroupBox('Source')
        status_layout = QVBoxLayout(status_box)
        self.lbl_source = QLabel(
            'No exit-pupil field loaded -- run a wave-optics simulation '
            'first; the latest run is used automatically.')
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setStyleSheet('color:#7a94b8;')
        status_layout.addWidget(self.lbl_source)

        status_row = QHBoxLayout()
        self.btn_refresh = QPushButton('Use latest wave-optics run')
        self.btn_refresh.setToolTip(
            'Pull the most recent exit-pupil (LensExit) field from the '
            'Wave Optics dock.  Called automatically on every run '
            'finish, but you can re-sync manually if you tweaked the '
            'dock settings without re-running.')
        self.btn_refresh.clicked.connect(self._pull_from_waveoptics)
        status_row.addWidget(self.btn_refresh)
        status_row.addStretch()
        status_layout.addLayout(status_row)

        layout.addWidget(status_box)

        # ── Scan-range controls ──
        scan_box = QGroupBox('Scan range')
        scan_layout = QHBoxLayout(scan_box)
        scan_layout.addWidget(QLabel('z center (mm):'))
        self.spin_zcenter = QDoubleSpinBox()
        self.spin_zcenter.setRange(-1e4, 1e4)
        self.spin_zcenter.setDecimals(3)
        self.spin_zcenter.setValue(0.0)
        self.spin_zcenter.setToolTip(
            'Axial position [mm] about which the scan is centred.\n'
            'Default = the lens BFL from the current ray trace.')
        scan_layout.addWidget(self.spin_zcenter)

        scan_layout.addWidget(QLabel('half-range (mm):'))
        self.spin_zhalf = QDoubleSpinBox()
        self.spin_zhalf.setRange(0.001, 100.0)
        self.spin_zhalf.setDecimals(3)
        self.spin_zhalf.setValue(1.0)
        scan_layout.addWidget(self.spin_zhalf)

        scan_layout.addWidget(QLabel('points:'))
        self.spin_npts = QSpinBox()
        self.spin_npts.setRange(5, 501)
        self.spin_npts.setValue(31)
        scan_layout.addWidget(self.spin_npts)

        self.chk_normalized = QCheckBox('Normalize Strehl')
        self.chk_normalized.setChecked(True)
        self.chk_normalized.setToolTip(
            'Divide peak by the diffraction-limited peak of a '
            'same-aperture ideal wavefront so the number sits in [0, 1].')
        scan_layout.addWidget(self.chk_normalized)

        scan_layout.addStretch()
        layout.addWidget(scan_box)

        # ── Run controls + progress ──
        run_row = QHBoxLayout()
        self.btn_run = QPushButton('Run through-focus scan')
        self.btn_run.clicked.connect(self._start_scan)
        run_row.addWidget(self.btn_run)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_scan)
        run_row.addWidget(self.btn_stop)
        self.btn_export = QPushButton('Export CSV...')
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_export.setEnabled(False)
        run_row.addWidget(self.btn_export)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Plot ──
        if HAS_MPL:
            self.fig = Figure(figsize=(7, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, stretch=1)
        else:
            layout.addWidget(QLabel(
                '(matplotlib not available; numerical results only)'))

        # ── Summary ──
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

    # ------------------------------------------------------------------
    # Source-field wiring
    # ------------------------------------------------------------------

    def set_source_field(self, E_exit, dx, wavelength, z_center=None):
        """Accept an exit-pupil field for the scan.

        ``z_center`` is an initial guess (m) for the image plane --
        usually the BFL.  Used only to pre-populate the spinbox.
        """
        self._E_exit = np.asarray(E_exit, dtype=np.complex128)
        self._dx = float(dx)
        self._wavelength = float(wavelength)
        self._z_center = float(z_center) if z_center is not None else None
        self.lbl_source.setText(
            f'Exit-pupil field loaded: {self._E_exit.shape}, '
            f'dx = {self._dx*1e6:.2f} um,  lambda = '
            f'{self._wavelength*1e9:.1f} nm')
        if self._z_center is not None and np.isfinite(self._z_center):
            self.spin_zcenter.setValue(self._z_center * 1e3)
            # Default half-range = f/20 rule of thumb.
            self.spin_zhalf.setValue(
                max(abs(self._z_center) * 0.05 * 1e3, 0.1))

    def _pull_from_waveoptics(self):
        """Try to re-sync from the wave-optics dock's last run without
        requiring the main window to do the plumbing."""
        mw = self.window()
        wd = getattr(mw, 'waveoptics_widget', None)
        if wd is None:
            self.summary.append('No wave-optics dock found.')
            return
        results = getattr(wd, '_last_results', None)
        if not results:
            self.summary.append(
                'Wave-optics dock has no results yet -- run it first.')
            return
        planes = results.get('planes') or []
        exit_plane = next(
            (p for p in planes if p.get('label') == 'LensExit'), None)
        if exit_plane is None:
            # fall back to the last saved plane
            if not planes:
                self.summary.append('No saved planes in wave-optics run.')
                return
            exit_plane = planes[-1]
        try:
            self.set_source_field(
                exit_plane['field'], exit_plane.get('dx', results.get('dx')),
                results['wavelength'],
                z_center=results.get('bfl_m'))
        except Exception as e:
            self.summary.append(
                f'set_source_field failed: {type(e).__name__}: {e}')

    # ------------------------------------------------------------------
    # Scan control
    # ------------------------------------------------------------------

    def _start_scan(self):
        if self._E_exit is None:
            self.summary.append(
                'No exit-pupil field -- run wave-optics first or click '
                '"Use latest wave-optics run".')
            return
        z0 = self.spin_zcenter.value() * 1e-3
        half = self.spin_zhalf.value() * 1e-3
        n = self.spin_npts.value()
        z_values = np.linspace(z0 - half, z0 + half, n)

        ideal_peak = None
        if self.chk_normalized.isChecked():
            try:
                from lumenairy.through_focus import (
                    diffraction_limited_peak)
                # use z0 as the focal length reference
                ideal_peak = diffraction_limited_peak(
                    self._E_exit, self._wavelength, z0, self._dx)
            except Exception:
                ideal_peak = None

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.summary.clear()
        self.summary.append(
            f'Scanning {n} planes in [{z_values[0]*1e3:.3f}, '
            f'{z_values[-1]*1e3:.3f}] mm...')

        self._worker = ThroughFocusWorker(
            self._E_exit, self._dx, self._wavelength, z_values,
            ideal_peak=ideal_peak)
        self._worker.fine_progress.connect(self._on_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _stop_scan(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._on_finished(None)

    def _on_progress(self, fraction, message):
        self.progress_bar.setValue(int(1000 * max(0.0, min(1.0, fraction))))
        if message:
            self.progress_bar.setToolTip(message)

    def _on_finished(self, result):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._worker = None
        if result is None:
            self.summary.append('Stopped.')
            return
        if isinstance(result, dict) and result.get('error'):
            self.summary.append(f'Failed: {result["error"]}')
            return
        self._last_result = result
        self.btn_export.setEnabled(True)
        self._plot_result(result)
        self._summarize(result)

    # ------------------------------------------------------------------
    # Plot + summary
    # ------------------------------------------------------------------

    def _plot_result(self, r):
        if not HAS_MPL:
            return
        self.fig.clear()
        z_mm = r.z * 1e3
        ax_top = self.fig.add_subplot(2, 1, 1)
        ax_bot = self.fig.add_subplot(2, 1, 2, sharex=ax_top)

        if np.any(np.isfinite(r.strehl)):
            ax_top.plot(z_mm, r.strehl, '-o', color='#5cb8ff',
                        markersize=3, label='Strehl')
            ax_top.set_ylabel('Strehl')
            ax_top.set_ylim(0, max(1.05, float(np.nanmax(r.strehl)) * 1.1))
        else:
            peak = r.peak_I / max(float(np.nanmax(r.peak_I)), 1e-30)
            ax_top.plot(z_mm, peak, '-o', color='#5cb8ff',
                        markersize=3, label='peak (norm.)')
            ax_top.set_ylabel('peak intensity (norm.)')

        best_i = int(np.nanargmax(
            r.strehl if np.any(np.isfinite(r.strehl)) else r.peak_I))
        ax_top.axvline(z_mm[best_i], color='#ffd166', linestyle='--',
                       label=f'best @ {z_mm[best_i]:.4f} mm')
        ax_top.legend(loc='best', fontsize=8)
        ax_top.grid(alpha=0.2)

        ax_bot.plot(z_mm, r.rms_radius * 1e6, '-o',
                    color='#ff7a5c', markersize=3, label='RMS radius')
        ax_bot.plot(z_mm, r.d4sigma_x * 1e6, '-',
                    color='#ffd166', label='D4s x')
        ax_bot.plot(z_mm, r.d4sigma_y * 1e6, '-',
                    color='#7affa7', label='D4s y')
        ax_bot.set_xlabel('z (mm)')
        ax_bot.set_ylabel('spot size (um)')
        ax_bot.legend(loc='best', fontsize=8)
        ax_bot.grid(alpha=0.2)
        self.canvas.draw()

    def _summarize(self, r):
        z_mm = r.z * 1e3
        if np.any(np.isfinite(r.strehl)):
            i_best = int(np.nanargmax(r.strehl))
            self.summary.append(
                f'Best focus: z = {z_mm[i_best]:+.5f} mm, '
                f'Strehl = {r.strehl[i_best]:.4f}, '
                f'RMS radius = {r.rms_radius[i_best]*1e6:.2f} um')
        else:
            i_best = int(np.nanargmax(r.peak_I))
            self.summary.append(
                f'Best focus (peak): z = {z_mm[i_best]:+.5f} mm, '
                f'peak = {r.peak_I[i_best]:.3e}, '
                f'RMS = {r.rms_radius[i_best]*1e6:.2f} um')

    def _export_csv(self):
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export through-focus CSV', filter='CSV (*.csv)')
        if not path:
            return
        r = self._last_result
        try:
            cols = np.column_stack([
                r.z, r.peak_I, r.strehl,
                r.d4sigma_x, r.d4sigma_y, r.rms_radius, r.power_in_bucket])
            hdr = 'z_m,peak_I,strehl,d4sigma_x_m,d4sigma_y_m,rms_radius_m,power_in_bucket'
            np.savetxt(path, cols, delimiter=',', header=hdr, comments='')
            self.summary.append(f'Wrote {path}')
        except Exception as e:
            self.summary.append(f'CSV export failed: {type(e).__name__}: {e}')
