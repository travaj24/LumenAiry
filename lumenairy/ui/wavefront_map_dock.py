"""
Wavefront map dock -- Zemax-style OPD heatmap with rich controls.

# v5.4 (audit P1-B): surface ``plot_wavefront()`` (v4.14.0) in the
# Designer GUI.  Pre-v5.4 there was no dock for this rich, divergent-
# colormap OPD visualisation -- users exported raw OPD arrays and
# plotted them externally.  This dock wires the library function
# directly to (a) a ray-traced OPD computed from the current system
# at the user-selected field angle + wavelength, or (b) a complex
# field loaded from an HDF5 file (run through ``wave_opd_2d`` to get
# the absolute OPD).  Aperture overlay, units, colormap, and RMS / PV
# annotation all flow through to ``plot_wavefront()`` as kwargs.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QCheckBox, QFileDialog, QTextEdit,
    QSpinBox, QRadioButton, QButtonGroup, QStackedWidget,
    QLineEdit, QProgressBar,
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


# Built-in colormap list.  ``RdBu_r`` is the Zemax-style default
# (divergent, zero = neutral mid-tone); the rest cover the common
# requests from wavefront diagnostics: viridis / turbo for perceptual
# uniformity, gray for high-contrast prints, seismic as an alternate
# divergent (sharper red/blue than RdBu_r).
_COLORMAPS = ('RdBu_r', 'viridis', 'turbo', 'gray', 'seismic')

# Display-unit choices.  Matches ``plot_wavefront``'s supported set
# (``'waves'`` needs a wavelength, others are pure scale factors).
_UNIT_CHOICES = ('waves', 'um', 'mm', 'nm')


class _OPDWorker(QThread):
    """Compute a 2-D OPD map for the chosen field / wavelength in a
    background thread.

    Mirrors the ``_GSWorker`` / ``_PolyStrehlWorker`` pattern from the
    other v5.x analysis docks: a single ``finished_result`` signal
    carrying a result dict so the UI thread doesn't block during the
    trace + rasterise + reference-sphere subtraction.
    """

    finished_result = Signal(object)

    def __init__(self, system_model, fa_x_deg, fa_y_deg, wavelength_m,
                 grid_N=256):
        super().__init__()
        self.sm = system_model
        self.fa_x_deg = float(fa_x_deg)
        self.fa_y_deg = float(fa_y_deg)
        self.wavelength_m = float(wavelength_m)
        self.grid_N = int(grid_N)

    def run(self):
        try:
            opd, dx, ap_m = self._compute_ray_traced_opd()
            self.finished_result.emit({
                'success': True,
                'opd': opd,
                'dx': dx,
                'aperture_m': ap_m,
                'wavelength_m': self.wavelength_m,
                'fa_x_deg': self.fa_x_deg,
                'fa_y_deg': self.fa_y_deg,
            })
        except Exception as e:
            self.finished_result.emit({
                'success': False,
                'error': f'{type(e).__name__}: {e}',
            })

    def _compute_ray_traced_opd(self):
        """Trace the current prescription at the chosen field /
        wavelength and rasterise alive-ray OPL onto an N x N pupil
        grid.

        Returns (opd_map [m], dx [m], aperture [m]).  ``opd_map`` has
        ``NaN`` outside the aperture so ``plot_wavefront`` shows the
        background colour outside the pupil.
        """
        from ..raytrace import make_rings
        from ..raytrace.core import trace_world

        surfaces = self.sm.build_run_trace_world_surfaces()
        if not surfaces:
            raise RuntimeError(
                'No trace surfaces -- add elements to the prescription '
                'first.')
        ap = float(self.sm.epd_m)
        if not np.isfinite(ap) or ap <= 0:
            raise RuntimeError(
                f'Invalid EPD: {ap}; set a positive aperture in the '
                'source / system controls.')

        # Field angle: hypotenuse of (fa_x, fa_y) projected onto the
        # tangential axis -- matches ``run_trace``'s on-axis trace
        # convention.  Off-axis xy field grids go through the
        # optimizer dock's multi-field path.
        fa_rad = np.radians(np.hypot(self.fa_x_deg, self.fa_y_deg))

        # Higher-density bundle than ``run_trace``'s default so the
        # rasterised pupil has reasonable fill.  16 rings * 64 spokes
        # = 1024 rays; cheap for any UI-scale design.
        rays = make_rings(ap / 2.0, 16, 64, fa_rad, self.wavelength_m)
        result = trace_world(rays, surfaces, self.wavelength_m)
        if result is None or result.image_rays is None:
            raise RuntimeError(
                'Ray trace failed -- prescription may be ill-defined.')

        # Rasterise OPL.  Pattern (and rationale for the bounds-mask /
        # mean-accumulator) is lifted from
        # ``psf_mtf_dock._load_from_raytrace`` so the OPD shown here
        # matches the OPD the PSF/MTF dock builds.
        ir = result.image_rays
        alive = ir.alive
        if int(np.sum(alive)) < 16:
            raise RuntimeError(
                'Fewer than 16 alive rays at the image plane -- the '
                'aperture may be over-stopped or the trace is failing '
                'at a surface.')
        x = ir.x[alive]
        y = ir.y[alive]
        opl = ir.opl[alive] - np.mean(ir.opl[alive])

        N = self.grid_N
        dx = ap / N
        ix_raw = ((x / dx) + N / 2).astype(int)
        iy_raw = ((y / dx) + N / 2).astype(int)
        in_bounds = ((ix_raw >= 0) & (ix_raw < N) &
                     (iy_raw >= 0) & (iy_raw < N))
        ix = ix_raw[in_bounds]
        iy = iy_raw[in_bounds]
        opl_in = opl[in_bounds]

        sum_grid = np.zeros((N, N), dtype=np.float64)
        cnt_grid = np.zeros((N, N), dtype=np.int64)
        np.add.at(sum_grid, (iy, ix), opl_in)
        np.add.at(cnt_grid, (iy, ix), 1)
        valid = cnt_grid > 0
        opd_grid = np.full((N, N), np.nan, dtype=np.float64)
        opd_grid[valid] = sum_grid[valid] / cnt_grid[valid]
        return opd_grid, dx, ap


class WavefrontMapDock(QWidget):
    """OPD heatmap dock -- v4.14.0 ``plot_wavefront`` made first-class.

    Sources: ``Current system`` runs a ray-trace + pupil rasterise at
    the chosen (field, wavelength); ``Loaded HDF5 file`` reads a
    complex field and converts via ``wave_opd_2d``; ``Live optimiser
    run`` subscribes to ``OptimizerDock.OptimizeWorker.progress`` and
    redraws on each step (see ``_subscribe_optimizer``).
    """

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model

        # Most recently rendered OPD (kept so toggling unit / cmap /
        # show-stats / aperture re-uses the cached array instead of
        # re-tracing).
        self._opd = None
        self._dx = None
        self._wavelength_m = None
        self._aperture_m = None
        self._opd_label = ''
        self._worker = None
        self._loaded_path = None

        self._build_ui()
        # Initial population of the field / wavelength dropdowns.
        self._refresh_field_wavelength_lists()
        # Stay in sync with prescription edits.
        try:
            self.sm.system_changed.connect(self._refresh_field_wavelength_lists)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Source selector --------------------------------------
        src_box = QGroupBox('OPD source')
        src_layout = QVBoxLayout(src_box)
        radio_row = QHBoxLayout()
        self.radio_system = QRadioButton('Current system (ray trace)')
        self.radio_system.setChecked(True)
        self.radio_file = QRadioButton('Loaded HDF5 file')
        self.radio_live = QRadioButton('Live optimiser run')
        self._src_group = QButtonGroup(self)
        for i, b in enumerate((self.radio_system, self.radio_file,
                               self.radio_live)):
            self._src_group.addButton(b, i)
            radio_row.addWidget(b)
        radio_row.addStretch()
        src_layout.addLayout(radio_row)

        # Stacked sub-panels per source.  ``_StackedWidget`` is the
        # canonical Qt way to keep source-specific controls visible
        # only while their radio is selected.
        self.stacked = QStackedWidget()

        # Page 0: current-system trace controls -- field + wavelength
        # dropdowns.
        page_sys = QWidget()
        sys_layout = QHBoxLayout(page_sys)
        sys_layout.setContentsMargins(0, 0, 0, 0)
        sys_layout.addWidget(QLabel('Field:'))
        self.combo_field = QComboBox()
        self.combo_field.setToolTip(
            'Field angle from the system model.  The aperture-filling '
            'ray bundle is launched at this angle.')
        sys_layout.addWidget(self.combo_field)
        sys_layout.addWidget(QLabel('Wavelength:'))
        self.combo_wavelength = QComboBox()
        self.combo_wavelength.setToolTip(
            'Wavelength from the optimizer wavelength list.')
        sys_layout.addWidget(self.combo_wavelength)
        sys_layout.addWidget(QLabel('Grid N:'))
        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(64, 1024)
        self.spin_grid.setValue(256)
        self.spin_grid.setSingleStep(64)
        self.spin_grid.setToolTip(
            'Pupil-rasterisation grid size.  Larger N gives a sharper '
            'pupil edge but slower trace + accumulation.')
        sys_layout.addWidget(self.spin_grid)
        sys_layout.addStretch()
        self.stacked.addWidget(page_sys)

        # Page 1: HDF5 file picker.
        page_file = QWidget()
        file_layout = QHBoxLayout(page_file)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_path = QLineEdit()
        self.edit_path.setReadOnly(True)
        self.edit_path.setPlaceholderText(
            '(no file loaded -- click Browse...)')
        file_layout.addWidget(self.edit_path, stretch=1)
        self.btn_browse = QPushButton('Browse...')
        self.btn_browse.setToolTip(
            'Pick an .h5 / .hdf5 file containing a single /field '
            'dataset with dx, wavelength attributes (matches '
            'lumenairy.io.save_field_h5).')
        self.btn_browse.clicked.connect(self._on_browse)
        file_layout.addWidget(self.btn_browse)
        self.stacked.addWidget(page_file)

        # Page 2: live optimiser controls.  Status label only; the
        # actual subscribe-on-iteration is wired in
        # ``_subscribe_optimizer`` below.
        page_live = QWidget()
        live_layout = QHBoxLayout(page_live)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_live_status = QLabel('(not subscribed)')
        self.lbl_live_status.setStyleSheet('color: #7a94b8')
        live_layout.addWidget(self.lbl_live_status)
        self.btn_subscribe = QPushButton('Subscribe to optimizer')
        self.btn_subscribe.setToolTip(
            'Hook the optimizer dock\'s per-iteration progress signal '
            'so this dock auto-redraws after each step.  Falls back to '
            'a no-op message if the optimizer dock is not present.')
        self.btn_subscribe.clicked.connect(self._subscribe_optimizer)
        live_layout.addWidget(self.btn_subscribe)
        live_layout.addStretch()
        self.stacked.addWidget(page_live)

        src_layout.addWidget(self.stacked)
        self._src_group.idClicked.connect(self.stacked.setCurrentIndex)
        layout.addWidget(src_box)

        # --- Display options --------------------------------------
        disp_box = QGroupBox('Display options')
        disp_layout = QHBoxLayout(disp_box)
        disp_layout.addWidget(QLabel('Units:'))
        self.combo_units = QComboBox()
        self.combo_units.addItems(_UNIT_CHOICES)
        self.combo_units.setCurrentText('waves')
        self.combo_units.setToolTip(
            "Display units for the colour scale and the PV / RMS "
            "annotation.  'waves' requires the wavelength to be set.")
        self.combo_units.currentTextChanged.connect(self._on_display_changed)
        disp_layout.addWidget(self.combo_units)

        disp_layout.addWidget(QLabel('Cmap:'))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(_COLORMAPS)
        self.combo_cmap.setCurrentText('RdBu_r')
        self.combo_cmap.setToolTip(
            'Matplotlib colormap.  RdBu_r is the Zemax convention '
            '(divergent, zero = neutral).')
        self.combo_cmap.currentTextChanged.connect(self._on_display_changed)
        disp_layout.addWidget(self.combo_cmap)

        self.chk_aperture = QCheckBox('Aperture overlay')
        self.chk_aperture.setChecked(True)
        self.chk_aperture.setToolTip(
            'When on, pixels outside the system\'s pupil (circle of '
            'radius EPD/2) are rendered as NaN by plot_wavefront, '
            'matching Zemax / Code V wavefront-map conventions.')
        self.chk_aperture.toggled.connect(self._on_display_changed)
        disp_layout.addWidget(self.chk_aperture)

        self.chk_stats = QCheckBox('Show PV / RMS')
        self.chk_stats.setChecked(True)
        self.chk_stats.setToolTip(
            'Overlay a PV / RMS callout in the upper-left of the '
            'figure (units match the Units dropdown).')
        self.chk_stats.toggled.connect(self._on_display_changed)
        disp_layout.addWidget(self.chk_stats)
        disp_layout.addStretch()
        layout.addWidget(disp_box)

        # --- Run + progress ---------------------------------------
        run_row = QHBoxLayout()
        self.btn_run = QPushButton('Compute + plot OPD  (F9)')
        self.btn_run.setObjectName('run_button')
        self.btn_run.setToolTip(
            'Compute the OPD from the selected source and redraw the '
            'wavefront map.  Re-uses the cached OPD when only display '
            'options change.')
        self.btn_run.clicked.connect(self._on_run)
        run_row.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        run_row.addWidget(self.progress, stretch=1)
        layout.addLayout(run_row)

        # --- Plot -------------------------------------------------
        if HAS_MPL:
            self.fig = Figure(figsize=(6, 5), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.mpl_toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.mpl_toolbar)
            layout.addWidget(self.canvas, stretch=1)
        else:
            layout.addWidget(QLabel('(matplotlib not available)'))

        # --- Summary panel ----------------------------------------
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

    # ------------------------------------------------------------------
    # Model-driven dropdown population
    # ------------------------------------------------------------------
    def _refresh_field_wavelength_lists(self):
        """Repopulate the field / wavelength dropdowns from the model.

        Triggered on dock construction and on every ``system_changed``
        emit so the dropdowns track edits to the source's field-angle
        list or the optimizer wavelength list.
        """
        # Field angles.  Use ``field_angles_deg`` directly -- a single
        # tilt angle per field point, applied as the rms across X/Y
        # to match the run_trace convention.
        prev_field = self.combo_field.currentText()
        self.combo_field.blockSignals(True)
        self.combo_field.clear()
        try:
            fields = list(self.sm.field_angles_deg or [0.0])
        except Exception:
            fields = [0.0]
        for fa in fields:
            self.combo_field.addItem(f'{float(fa):.3f} deg')
        idx = self.combo_field.findText(prev_field)
        if idx >= 0:
            self.combo_field.setCurrentIndex(idx)
        self.combo_field.blockSignals(False)

        prev_wv = self.combo_wavelength.currentText()
        self.combo_wavelength.blockSignals(True)
        self.combo_wavelength.clear()
        try:
            wvs_nm = list(self.sm.wavelengths_nm or [1310.0])
        except Exception:
            wvs_nm = [1310.0]
        for wv in wvs_nm:
            self.combo_wavelength.addItem(f'{float(wv):.2f} nm')
        idx = self.combo_wavelength.findText(prev_wv)
        if idx >= 0:
            self.combo_wavelength.setCurrentIndex(idx)
        self.combo_wavelength.blockSignals(False)

    # ------------------------------------------------------------------
    # Source: HDF5 file
    # ------------------------------------------------------------------
    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load OPD source field',
            '', 'HDF5 files (*.h5 *.hdf5);;All files (*)')
        if not path:
            return
        self.edit_path.setText(path)
        self._loaded_path = path
        self.summary.append(f'Loaded path: {path}')

    def _load_opd_from_file(self):
        """Read a complex field from ``self._loaded_path`` and convert
        to a 2-D OPD map via ``wave_opd_2d``.

        Sets ``self._opd``, ``self._dx``, ``self._wavelength_m``,
        ``self._aperture_m`` on success and returns ``True``; returns
        ``False`` on failure (with an error message appended to the
        summary panel).
        """
        if not self._loaded_path:
            self.summary.append('No file selected -- click Browse...')
            return False
        try:
            from lumenairy.io import load_field_h5
            from lumenairy.analysis import wave_opd_2d
            E, meta = load_field_h5(self._loaded_path)
            dx = float(meta.get('dx', 0.0))
            if not np.isfinite(dx) or dx <= 0:
                self.summary.append(
                    "File missing or has invalid 'dx' attribute.")
                return False
            wv = float(meta.get('wavelength',
                                self.sm.wavelength_m or 0.0))
            if not np.isfinite(wv) or wv <= 0:
                self.summary.append(
                    "File missing 'wavelength'; falling back to "
                    "system model wavelength.")
                wv = float(self.sm.wavelength_m or 1.31e-6)
            ap_m = float(meta.get('aperture',
                                  self.sm.epd_m or 0.0)) or None
            _, _, opd_map = wave_opd_2d(
                E, dx, wv, aperture=ap_m)
            self._opd = opd_map
            self._dx = dx
            self._wavelength_m = wv
            self._aperture_m = ap_m
            self._opd_label = (
                f'File: {self._loaded_path}\n'
                f'  shape={opd_map.shape}, dx={dx*1e6:.2f} um, '
                f'wv={wv*1e9:.2f} nm')
            self.summary.append(self._opd_label)
            return True
        except Exception as e:
            self.summary.append(
                f'load_field_h5 / wave_opd_2d failed: '
                f'{type(e).__name__}: {e}')
            return False

    # ------------------------------------------------------------------
    # Live optimiser subscription
    # ------------------------------------------------------------------
    def _subscribe_optimizer(self):
        """Connect to the OptimizerDock's per-iteration ``progress``
        signal so this dock auto-recomputes the OPD after each step.

        The optimizer dock isn't guaranteed to exist yet at dock-
        creation time -- ``MainWindow`` instantiates docks lazily.  We
        walk up through ``self.window()`` and look for a worker on the
        OptimizerDock instance.  Pre-v5.4 the worker only existed
        during a run, so we also listen for the dock\'s eventual
        creation of a worker (no signal for that -- best-effort poll
        on click).
        """
        try:
            mw = self.window()
            opt = getattr(mw, 'optimizer_widget', None)
            if opt is None:
                self.lbl_live_status.setText('(optimizer dock not found)')
                return
            worker = getattr(opt, '_worker', None)
            if worker is None:
                self.lbl_live_status.setText(
                    '(no active optimizer worker -- click again after '
                    'starting an optimisation)')
                return
            # ``progress(it, merit)``.  We don't care about merit; we
            # only use the signal as a "system updated" notification
            # and recompute the OPD off the fresh prescription.
            worker.progress.connect(self._on_optimizer_step)
            self.lbl_live_status.setText(
                'Subscribed to optimizer iteration signal.')
            self.summary.append(
                'Live optimiser subscription active -- OPD will '
                'redraw on each iteration.')
        except Exception as e:
            self.lbl_live_status.setText(
                f'(subscription failed: {type(e).__name__})')

    def _on_optimizer_step(self, it, merit):
        """Slot connected to OptimizeWorker.progress.

        Only redraws when the live page is active; otherwise the
        signal stays connected but no-ops so the user can switch back
        without re-subscribing.
        """
        if self.stacked.currentIndex() != 2:
            return
        # Reuse the current-system trace path with the active
        # (field, wavelength) selection.
        self._compute_current_system(quiet=True, label_suffix=f'iter {it}')

    # ------------------------------------------------------------------
    # Source: current system
    # ------------------------------------------------------------------
    def _compute_current_system(self, quiet=False, label_suffix=''):
        """Launch the background OPD worker for the current system."""
        if self._worker is not None and self._worker.isRunning():
            if not quiet:
                self.summary.append(
                    'A previous OPD trace is still running -- wait '
                    'for it to finish.')
            return
        try:
            fa_idx = max(0, self.combo_field.currentIndex())
            fa = float(self.sm.field_angles_deg[fa_idx])
        except Exception:
            fa = 0.0
        try:
            wv_idx = max(0, self.combo_wavelength.currentIndex())
            wv = float(self.sm.wavelengths_nm[wv_idx]) * 1e-9
        except Exception:
            wv = float(self.sm.wavelength_m or 1.31e-6)
        N = int(self.spin_grid.value())
        self._opd_label = (
            f'Current system: field={fa:.3f} deg, '
            f'wv={wv*1e9:.2f} nm, N={N}'
            + (f'  ({label_suffix})' if label_suffix else ''))
        # ``fa_x`` is left zero -- the dropdown collapses to a
        # single (meridional) angle to keep parity with run_trace.
        self._worker = _OPDWorker(self.sm, 0.0, fa, wv, grid_N=N)
        self._worker.finished_result.connect(self._on_worker_done)
        self.progress.setVisible(True)
        self.btn_run.setEnabled(False)
        self._worker.start()

    def _on_worker_done(self, result):
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        if not result.get('success'):
            self.summary.append(
                f'OPD computation failed: {result.get("error")}')
            return
        self._opd = result['opd']
        self._dx = result['dx']
        self._aperture_m = result['aperture_m']
        self._wavelength_m = result['wavelength_m']
        self.summary.append(
            f'OPD computed: {self._opd.shape}, '
            f'dx={self._dx*1e6:.2f} um, '
            f'ap={self._aperture_m*1e3:.2f} mm, '
            f'wv={self._wavelength_m*1e9:.2f} nm')
        self._redraw(self._opd, self._dx, self._wavelength_m)

    # ------------------------------------------------------------------
    # Run + display
    # ------------------------------------------------------------------
    def _on_run(self):
        """Top-level dispatcher: picks the right source path."""
        self.summary.clear()
        src_idx = self.stacked.currentIndex()
        if src_idx == 0:
            self._compute_current_system()
        elif src_idx == 1:
            if self._load_opd_from_file():
                self._redraw(self._opd, self._dx, self._wavelength_m)
        elif src_idx == 2:
            # Live page: do a one-shot snapshot now, then let the
            # subscribed signal drive subsequent updates.
            self._compute_current_system()

    def _on_display_changed(self):
        """Re-render the cached OPD when only display knobs changed.

        Avoids a re-trace when the user toggles cmap / units / show-
        stats / aperture.  No-op if nothing has been computed yet.
        """
        if self._opd is None or not HAS_MPL:
            return
        self._redraw(self._opd, self._dx, self._wavelength_m)

    def _redraw(self, opd, dx, wavelength):
        """Hand the OPD off to ``plot_wavefront`` with current knobs.

        The library function takes an ``ax`` argument so we can plot
        directly onto our embedded ``FigureCanvasQTAgg``; we wipe the
        figure first so the colour-bar from a prior call doesn't
        stack into a second axis.
        """
        if not HAS_MPL:
            return
        if opd is None:
            self.summary.append('No OPD available to plot.')
            return
        try:
            from lumenairy.analysis import plot_wavefront
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            # Build the aperture mask only when the checkbox is set
            # AND we know the system's EPD.  ``plot_wavefront``
            # otherwise renders the full grid (including any padding
            # outside the circular pupil).
            ap_mask = None
            if self.chk_aperture.isChecked() and self._aperture_m:
                Ny, Nx = opd.shape
                x = (np.arange(Nx) - Nx / 2) * dx
                y = (np.arange(Ny) - Ny / 2) * dx
                X, Y = np.meshgrid(x, y)
                ap_mask = (X ** 2 + Y ** 2) <= (0.5 * self._aperture_m) ** 2
            units = self.combo_units.currentText()
            kwargs = dict(
                dx=dx,
                aperture=ap_mask,
                units=units,
                cmap=self.combo_cmap.currentText(),
                show_stats=self.chk_stats.isChecked(),
                ax=ax,
                fig=self.fig,
                title=self._opd_label or 'Wavefront map',
            )
            # ``units='waves'`` requires a wavelength.  Pass it
            # unconditionally when known so users can switch back to
            # 'waves' after computing without re-tracing.
            if wavelength and np.isfinite(wavelength) and wavelength > 0:
                kwargs['wavelength'] = wavelength
            elif units == 'waves':
                self.summary.append(
                    "units='waves' but no wavelength is known -- "
                    "falling back to 'um'.")
                kwargs['units'] = 'um'
            plot_wavefront(opd, **kwargs)
            self.canvas.draw()
        except Exception as e:
            self.summary.append(
                f'plot_wavefront failed: {type(e).__name__}: {e}')
