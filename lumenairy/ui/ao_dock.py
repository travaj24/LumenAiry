"""
Adaptive-optics closed-loop dock.

v5.4 audit AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 item P1-A.  The
library ``lumenairy.analysis.ao`` module ships six AO primitives
(``DeformableMirror``, ``apply_dm``, ``zernike_modal_basis``,
``slope_to_modal``, ``LeakyIntegrator``, ``ao_closed_loop``) at
v5.2.3+ but the v5.3.2 GUI exposes none of them -- users must
prototype in Jupyter.  This dock surfaces the canonical
single-conjugate AO workflow:

  * Configure a Gaussian-influence-function DM (actuator count,
    pitch, modal basis).
  * Configure a wavefront sensor (Shack-Hartmann by default).
  * Configure the leaky-integrator controller (gain, leak, tol,
    max iterations -- the kwargs of v5.2.3+ ``ao_closed_loop``).
  * Pick an input disturbance (random Kolmogorov turbulence,
    loaded ``.npy`` phase map, or a manual Zernike spectrum).
  * Run the loop on a worker thread, watching the residual norm
    converge and the DM commands evolve in real time.

The library ``ao_closed_loop`` does not expose a per-iteration
callback hook, so this dock drives convergence by calling the
helper one iteration at a time and emitting Qt signals between
steps (the same single-step pattern used by ``examples/11_ao_closed_loop.py``
to record a pupil-masked RMS history).

Author: Andrew Traverso -- v5.4 AO dock.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QProgressBar, QPushButton, QRadioButton, QSpinBox,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =============================================================================
# Worker -- runs the closed loop on a QThread, one iteration at a time, so
# the dock can emit per-iteration progress without modifying the library.
# =============================================================================


class _AOClosedLoopWorker(QThread):
    """Run ``lumenairy.ao_closed_loop`` step-by-step on a worker thread.

    Emits:

      progress(iter_index, residual_rms_rad)   per iteration
      finished_result(result_dict)             on completion / error

    The result dict on success carries:

      {'success': True,
       'residual_phase' : (Ny, Nx) ndarray  -- final residual [rad]
       'dm_command'     : (n_act, n_act) ndarray  -- final DM cmd [rad OPD]
       'history'        : (iters+1,) ndarray  -- residual RMS per iter
       'wfe_per_iter'   : (iters+1,) ndarray  -- WFE [m] per iter
       'iterations_run' : int}

    On failure: {'success': False, 'error': str}.
    """

    progress = Signal(int, float)
    finished_result = Signal(object)

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self._stop_requested = False

    def request_stop(self) -> None:
        """Co-operative stop.  Picked up between iterations."""
        self._stop_requested = True

    def run(self) -> None:  # noqa: D401 - QThread entry point.
        try:
            import lumenairy as la
        except Exception as e:  # pragma: no cover - import smoke
            self.finished_result.emit({
                'success': False,
                'error': f'lumenairy import failed: {type(e).__name__}: {e}',
            })
            return

        try:
            params = self._params
            phase = params['phase_disturbance']
            N = phase.shape[0]
            dx = float(params['dx'])
            wavelength = float(params['wavelength'])
            n_act = int(params['n_actuators'])
            stroke = float(params.get('stroke_waves', 1.0))
            coupling = float(params.get('coupling', 0.15))
            gain = float(params['gain'])
            leak = float(params['leak'])
            tol = params.get('tol', None)
            n_iter = int(params['n_iterations'])

            aperture = float(params.get('aperture', 0.8 * (N * dx)))
            pitch = aperture / max(n_act - 1, 1)

            dm = la.DeformableMirror(
                n_actuators=n_act, pitch=pitch, dx=dx, N=N,
                inter_actuator_coupling=coupling,
            )

            # Initial residual + history slot.
            residual = phase - dm.phase()
            rms0 = float(np.sqrt(np.mean(residual * residual)))
            history = [rms0]
            self.progress.emit(0, rms0)

            iterations_run = 0
            for k in range(n_iter):
                if self._stop_requested:
                    break
                residual = la.ao_closed_loop(
                    phase, dm=dm, n_iterations=1, gain=gain, leak=leak,
                    wavelength=wavelength, dx=dx, wfs=None,
                )
                # Clamp DM command to the requested stroke (in waves of OPD).
                if stroke > 0.0:
                    max_rad = stroke * 2.0 * np.pi
                    np.clip(dm.command, -max_rad, max_rad, out=dm.command)
                rms_k = float(np.sqrt(np.mean(residual * residual)))
                history.append(rms_k)
                iterations_run = k + 1
                self.progress.emit(k + 1, rms_k)
                if tol is not None and rms_k < float(tol):
                    break

            history_arr = np.asarray(history, dtype=np.float64)
            wfe_per_iter = history_arr / (2.0 * np.pi) * wavelength

            self.finished_result.emit({
                'success': True,
                'residual_phase': residual.copy(),
                'dm_command': dm.command.copy(),
                'history': history_arr,
                'wfe_per_iter': wfe_per_iter,
                'iterations_run': iterations_run,
                'stopped': self._stop_requested,
            })
        except Exception as e:
            self.finished_result.emit({
                'success': False,
                'error': f'{type(e).__name__}: {e}',
            })


# =============================================================================
# Dock
# =============================================================================


class AOClosedLoopDock(QWidget):
    """GUI surface for the v5.2.3+ ``ao_closed_loop`` helper.

    Builds a deformable-mirror + wavefront-sensor + leaky-integrator
    closed loop from dock state, runs it on a worker thread, and
    visualises convergence as the loop runs.

    Wires up to the host application's ``SystemModel`` only as a
    courtesy (the dock self-contains everything needed to run, so it
    is usable on an empty workspace).
    """

    MODAL_BASES = ('zernike', 'karhunen_loeve', 'free_actuator')
    WFS_TYPES = ('shack_hartmann', 'pyramid', 'curvature')
    INPUT_TURBULENCE = 0
    INPUT_FILE = 1
    INPUT_MANUAL_ZERNIKE = 2

    def __init__(self, model, parent=None) -> None:
        # v5.4 audit P1-A: subclass QWidget to match the canonical
        # dock convention (main_window.py wraps in QDockWidget).
        super().__init__(parent)
        self.sm = model
        self._worker: Optional[_AOClosedLoopWorker] = None
        self._loaded_phase: Optional[np.ndarray] = None
        self._loaded_phase_path: Optional[str] = None
        self._build_ui()

    # ------------------------------------------------------------------ UI ---

    def _build_ui(self) -> None:
        """Top-level layout: horizontal splitter, controls left, plots right."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal, self)

        # ------------- Left: controls in a scrollable column ------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._build_dm_panel())
        left_layout.addWidget(self._build_wfs_panel())
        left_layout.addWidget(self._build_controller_panel())
        left_layout.addWidget(self._build_input_panel())
        left_layout.addWidget(self._build_run_panel())

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        left_layout.addWidget(self.summary)
        left_layout.addStretch()

        splitter.addWidget(left)

        # ------------- Right: three plots stacked vertically ------------
        right = QSplitter(Qt.Vertical)
        if HAS_MPL:
            self.fig_residual = Figure(figsize=(5.0, 2.6), tight_layout=True,
                                       facecolor='#0a0c10')
            self.canvas_residual = FigureCanvas(self.fig_residual)
            self.toolbar_residual = NavigationToolbar(
                self.canvas_residual, right)
            top_w = QWidget()
            top_layout = QVBoxLayout(top_w)
            top_layout.setContentsMargins(0, 0, 0, 0)
            top_layout.addWidget(self.toolbar_residual)
            top_layout.addWidget(self.canvas_residual, stretch=1)
            right.addWidget(top_w)

            self.fig_dm = Figure(figsize=(5.0, 2.6), tight_layout=True,
                                 facecolor='#0a0c10')
            self.canvas_dm = FigureCanvas(self.fig_dm)
            right.addWidget(self.canvas_dm)

            self.fig_residual_map = Figure(
                figsize=(5.0, 2.6), tight_layout=True, facecolor='#0a0c10')
            self.canvas_residual_map = FigureCanvas(self.fig_residual_map)
            right.addWidget(self.canvas_residual_map)
        else:
            warn = QLabel(
                'matplotlib backend unavailable -- plots disabled.')
            warn.setAlignment(Qt.AlignCenter)
            right.addWidget(warn)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 720])

        outer.addWidget(splitter)

    # ----------------------------------------------------------- DM panel ---

    def _build_dm_panel(self) -> QWidget:
        box = QGroupBox('Deformable mirror')
        form = QFormLayout(box)

        self.spin_n_act = QSpinBox()
        self.spin_n_act.setRange(2, 256)
        self.spin_n_act.setValue(64)
        self.spin_n_act.setToolTip(
            'Number of actuators per axis (square grid '
            'n_actuators**2 total).')
        form.addRow('Actuator count:', self.spin_n_act)

        self.combo_modal = QComboBox()
        self.combo_modal.addItems(list(self.MODAL_BASES))
        self.combo_modal.setCurrentText('zernike')
        self.combo_modal.setToolTip(
            'Modal basis used for fitting the measured wavefront '
            'into DM commands.')
        self.combo_modal.currentIndexChanged.connect(self._on_modal_changed)
        form.addRow('Modal basis:', self.combo_modal)

        self.spin_max_order = QSpinBox()
        self.spin_max_order.setRange(1, 30)
        self.spin_max_order.setValue(8)
        self.spin_max_order.setToolTip(
            'Maximum Zernike radial order (n).  Only used when modal '
            'basis is "zernike".  Default 8 covers OSA indices 1..44.')
        self._row_max_order = self.spin_max_order
        form.addRow('Max radial order:', self.spin_max_order)

        self.spin_stroke = QDoubleSpinBox()
        self.spin_stroke.setRange(0.0, 100.0)
        self.spin_stroke.setSingleStep(0.1)
        self.spin_stroke.setValue(1.0)
        self.spin_stroke.setDecimals(3)
        self.spin_stroke.setSuffix(' waves')
        self.spin_stroke.setToolTip(
            'Maximum DM stroke per actuator in waves of OPD.  Commands '
            'are clipped after each iteration; 0 disables clipping.')
        form.addRow('DM stroke:', self.spin_stroke)

        self.spin_coupling = QDoubleSpinBox()
        self.spin_coupling.setRange(0.01, 0.95)
        self.spin_coupling.setSingleStep(0.05)
        self.spin_coupling.setValue(0.15)
        self.spin_coupling.setDecimals(3)
        self.spin_coupling.setToolTip(
            'Fractional inter-actuator coupling; 0.15 is a typical '
            'electrostrictive-DM value.')
        form.addRow('Coupling:', self.spin_coupling)

        return box

    # ---------------------------------------------------------- WFS panel ---

    def _build_wfs_panel(self) -> QWidget:
        box = QGroupBox('Wavefront sensor')
        form = QFormLayout(box)

        self.combo_wfs = QComboBox()
        self.combo_wfs.addItems(list(self.WFS_TYPES))
        self.combo_wfs.setCurrentText('shack_hartmann')
        self.combo_wfs.setToolTip(
            'Wavefront-sensor type.  v5.4 dock uses ideal phase sensing '
            'internally and reports the chosen type to the summary; a '
            'future revision will wire pyramid / curvature sensors.')
        form.addRow('WFS type:', self.combo_wfs)

        self.spin_subap = QSpinBox()
        self.spin_subap.setRange(2, 256)
        self.spin_subap.setValue(8)
        self.spin_subap.setToolTip(
            'Number of subapertures per axis (square grid).')
        form.addRow('Subaperture grid:', self.spin_subap)

        self.spin_wfs_noise = QDoubleSpinBox()
        self.spin_wfs_noise.setRange(0.0, 1e6)
        self.spin_wfs_noise.setSingleStep(0.1)
        self.spin_wfs_noise.setValue(0.0)
        self.spin_wfs_noise.setDecimals(4)
        self.spin_wfs_noise.setSuffix(' e-/pixel')
        self.spin_wfs_noise.setToolTip(
            'Gaussian read-noise sigma in electrons per pixel.')
        form.addRow('Noise sigma:', self.spin_wfs_noise)

        return box

    # ---------------------------------------------------- Controller panel ---

    def _build_controller_panel(self) -> QWidget:
        box = QGroupBox('Closed-loop controller')
        form = QFormLayout(box)

        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 2.0)
        self.spin_gain.setSingleStep(0.05)
        self.spin_gain.setValue(0.5)
        self.spin_gain.setDecimals(3)
        self.spin_gain.setToolTip(
            'Leaky-integrator loop gain.  Typical AO values 0.3 - 0.5; '
            'gain=0 is an open-loop fallback.')
        form.addRow('Gain:', self.spin_gain)

        self.spin_leak = QDoubleSpinBox()
        self.spin_leak.setRange(0.0, 1.0)
        self.spin_leak.setSingleStep(0.01)
        self.spin_leak.setValue(0.0)
        self.spin_leak.setDecimals(4)
        self.spin_leak.setToolTip(
            'Per-iteration leak factor.  0 = pure integrator; small '
            'positive leak prevents command drift on un-locked loops.')
        form.addRow('Leak:', self.spin_leak)

        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.0, 1e6)
        self.spin_tol.setDecimals(8)
        self.spin_tol.setSingleStep(1e-6)
        self.spin_tol.setValue(1e-6)
        self.spin_tol.setToolTip(
            'Residual-RMS early-stop threshold in radians.  0 disables '
            'early stop and the loop runs all max iterations.')
        form.addRow('Tolerance (rad):', self.spin_tol)

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 100000)
        self.spin_max_iter.setValue(100)
        self.spin_max_iter.setToolTip(
            'Upper bound on closed-loop iterations.')
        form.addRow('Max iterations:', self.spin_max_iter)

        return box

    # -------------------------------------------------------- Input panel ---

    def _build_input_panel(self) -> QWidget:
        box = QGroupBox('Input phase disturbance')
        v = QVBoxLayout(box)

        # Radio buttons.
        radios = QHBoxLayout()
        self.radio_turbulence = QRadioButton('Random turbulence')
        self.radio_turbulence.setChecked(True)
        self.radio_file = QRadioButton('Loaded file')
        self.radio_zernike = QRadioButton('Manual Zernike')
        self._input_group = QButtonGroup(self)
        self._input_group.addButton(self.radio_turbulence, self.INPUT_TURBULENCE)
        self._input_group.addButton(self.radio_file, self.INPUT_FILE)
        self._input_group.addButton(self.radio_zernike,
                                    self.INPUT_MANUAL_ZERNIKE)
        self._input_group.idClicked.connect(self._on_input_mode_changed)
        radios.addWidget(self.radio_turbulence)
        radios.addWidget(self.radio_file)
        radios.addWidget(self.radio_zernike)
        v.addLayout(radios)

        # Stacked sub-panels.
        self._input_stack = QStackedWidget()

        # 0: turbulence sub-panel
        turb = QWidget()
        turb_form = QFormLayout(turb)
        self.spin_r0_mm = QDoubleSpinBox()
        self.spin_r0_mm.setRange(0.01, 1e6)
        self.spin_r0_mm.setSingleStep(0.5)
        self.spin_r0_mm.setValue(2.0)
        self.spin_r0_mm.setDecimals(4)
        self.spin_r0_mm.setSuffix(' mm')
        self.spin_r0_mm.setToolTip("Fried parameter r0 in mm.")
        turb_form.addRow('r0:', self.spin_r0_mm)
        self.spin_wavelength_nm = QDoubleSpinBox()
        self.spin_wavelength_nm.setRange(100.0, 20000.0)
        self.spin_wavelength_nm.setSingleStep(10.0)
        self.spin_wavelength_nm.setValue(1550.0)
        self.spin_wavelength_nm.setDecimals(2)
        self.spin_wavelength_nm.setSuffix(' nm')
        turb_form.addRow('Wavelength:', self.spin_wavelength_nm)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(-1, 2_000_000_000)
        self.spin_seed.setValue(0)
        self.spin_seed.setToolTip(
            'RNG seed for the Kolmogorov screen (-1 = random).')
        turb_form.addRow('Seed:', self.spin_seed)
        self.spin_grid_N = QSpinBox()
        self.spin_grid_N.setRange(16, 4096)
        self.spin_grid_N.setValue(128)
        self.spin_grid_N.setToolTip(
            'Grid size N (pupil sampled on N x N).  Also used by '
            '"Manual Zernike" mode.')
        turb_form.addRow('Grid N:', self.spin_grid_N)
        self.spin_aperture_mm = QDoubleSpinBox()
        self.spin_aperture_mm.setRange(0.1, 1e6)
        self.spin_aperture_mm.setSingleStep(0.5)
        self.spin_aperture_mm.setValue(8.0)
        self.spin_aperture_mm.setDecimals(3)
        self.spin_aperture_mm.setSuffix(' mm')
        self.spin_aperture_mm.setToolTip(
            'Clear-aperture diameter on the pupil grid.')
        turb_form.addRow('Aperture:', self.spin_aperture_mm)
        self._input_stack.addWidget(turb)

        # 1: file sub-panel
        file_w = QWidget()
        file_layout = QVBoxLayout(file_w)
        self.btn_load_phase = QPushButton('Load .npy phase map...')
        self.btn_load_phase.clicked.connect(self._on_load_phase)
        file_layout.addWidget(self.btn_load_phase)
        self.lbl_loaded_path = QLabel('(no file loaded)')
        self.lbl_loaded_path.setWordWrap(True)
        file_layout.addWidget(self.lbl_loaded_path)
        file_layout.addStretch()
        self._input_stack.addWidget(file_w)

        # 2: manual Zernike sub-panel (Z2..Z11)
        zern_w = QWidget()
        zern_v = QVBoxLayout(zern_w)
        zern_v.addWidget(QLabel('Manual Zernike coefficients [waves]:'))
        self.tbl_zernike = QTableWidget(10, 2)
        self.tbl_zernike.setHorizontalHeaderLabels(['Mode', 'Coeff (waves)'])
        self.tbl_zernike.verticalHeader().setVisible(False)
        # OSA-index labels (Noll-style label) for Z2..Z11.
        zern_labels = [
            'Z2 (tip)', 'Z3 (tilt)', 'Z4 (defocus)',
            'Z5 (oblique astig)', 'Z6 (vertical astig)',
            'Z7 (vertical coma)', 'Z8 (horizontal coma)',
            'Z9 (vertical trefoil)', 'Z10 (oblique trefoil)',
            'Z11 (primary spherical)',
        ]
        for r, lbl in enumerate(zern_labels):
            item_label = QTableWidgetItem(lbl)
            item_label.setFlags(item_label.flags() & ~Qt.ItemIsEditable)
            self.tbl_zernike.setItem(r, 0, item_label)
            self.tbl_zernike.setItem(r, 1, QTableWidgetItem('0.0'))
        self.tbl_zernike.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.tbl_zernike.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self.tbl_zernike.setMaximumHeight(240)
        zern_v.addWidget(self.tbl_zernike)
        self._input_stack.addWidget(zern_w)

        v.addWidget(self._input_stack)
        return box

    # ---------------------------------------------------------- Run panel ---

    def _build_run_panel(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        self.btn_run = QPushButton('Run AO loop')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        h.addWidget(self.btn_run)
        h.addWidget(self.btn_stop)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        h.addWidget(self.progress_bar, stretch=1)
        return w

    # ----------------------------------------------------- Visibility hooks ---

    def _on_modal_changed(self, _idx: int) -> None:
        is_zernike = self.combo_modal.currentText() == 'zernike'
        self._row_max_order.setEnabled(is_zernike)

    def _on_input_mode_changed(self, mode_id: int) -> None:
        self._input_stack.setCurrentIndex(mode_id)

    # --------------------------------------------------- Phase-map loader ---

    def _on_load_phase(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load phase disturbance (.npy)',
            filter='NumPy arrays (*.npy);;All files (*)')
        if not path:
            return
        try:
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f'phase map must be 2-D; got shape {arr.shape}')
            self._loaded_phase = arr
            self._loaded_phase_path = path
            self.lbl_loaded_path.setText(
                f'Loaded: {os.path.basename(path)}  '
                f'({arr.shape[0]} x {arr.shape[1]})')
            self.summary.append(
                f'Loaded phase: shape={arr.shape}, '
                f'rms={float(np.sqrt(np.mean(arr * arr))):.4f} rad')
        except Exception as e:
            self.summary.append(
                f'Failed to load phase: {type(e).__name__}: {e}')

    # --------------------------------------- Input-phase synthesis (helpers) ---

    def _make_pupil_mask(self, N: int, dx: float, aperture: float) -> np.ndarray:
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        return (X * X + Y * Y) <= (aperture / 2) ** 2

    def _build_input_phase(self) -> Optional[Dict[str, Any]]:
        """Materialise the (phase, dx, wavelength, aperture, N) tuple
        from the dock state.  Returns ``None`` on hard failure (with a
        message already posted to the summary)."""
        mode = self._input_group.checkedId()
        wavelength = self.spin_wavelength_nm.value() * 1e-9
        try:
            import lumenairy as la
        except Exception as e:
            self.summary.append(
                f'lumenairy import failed: {type(e).__name__}: {e}')
            return None

        if mode == self.INPUT_TURBULENCE:
            N = self.spin_grid_N.value()
            aperture = self.spin_aperture_mm.value() * 1e-3
            # Span the pupil with ~3 x aperture extent so we don't clip
            # the screen at the edges.
            extent = 3.0 * aperture
            dx = extent / N
            r0 = self.spin_r0_mm.value() * 1e-3
            seed = self.spin_seed.value()
            seed_kw = {} if seed < 0 else {'seed': int(seed)}
            try:
                screen = la.generate_turbulence_screen(
                    N=N, dx=dx, r0=r0, **seed_kw)
            except Exception as e:
                self.summary.append(
                    f'generate_turbulence_screen failed: '
                    f'{type(e).__name__}: {e}')
                return None
            mask = self._make_pupil_mask(N, dx, aperture)
            screen = screen * mask
            return {
                'phase': np.asarray(screen, dtype=np.float64),
                'dx': dx,
                'wavelength': wavelength,
                'aperture': aperture,
                'mask': mask,
            }

        if mode == self.INPUT_FILE:
            if self._loaded_phase is None:
                self.summary.append('No phase file loaded.  Click '
                                    '"Load .npy phase map" first.')
                return None
            phase = self._loaded_phase
            N = phase.shape[0]
            aperture = self.spin_aperture_mm.value() * 1e-3
            extent = 3.0 * aperture
            dx = extent / N
            mask = self._make_pupil_mask(N, dx, aperture)
            return {
                'phase': phase.copy(),
                'dx': dx,
                'wavelength': wavelength,
                'aperture': aperture,
                'mask': mask,
            }

        # Manual Zernike spectrum: Z2..Z11.
        N = self.spin_grid_N.value()
        aperture = self.spin_aperture_mm.value() * 1e-3
        extent = 3.0 * aperture
        dx = extent / N
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X * X + Y * Y)
        rho = r / (aperture / 2)
        theta = np.arctan2(Y, X)
        mask = rho <= 1.0
        try:
            from lumenairy.analysis.core import (
                zernike_index_to_nm, zernike_polynomial,
            )
        except Exception as e:
            self.summary.append(
                f'Cannot import Zernike helpers: {type(e).__name__}: {e}')
            return None
        phase = np.zeros((N, N), dtype=np.float64)
        for r_idx in range(self.tbl_zernike.rowCount()):
            osa = r_idx + 2  # Z2..Z11
            try:
                coeff = float(self.tbl_zernike.item(r_idx, 1).text())
            except Exception:
                coeff = 0.0
            if coeff == 0.0:
                continue
            try:
                n_z, m_z = zernike_index_to_nm(osa)
                Zk = zernike_polynomial(n_z, m_z, rho, theta)
            except Exception as e:
                self.summary.append(
                    f'Zernike Z{osa} eval failed: '
                    f'{type(e).__name__}: {e}')
                continue
            # waves -> radians.
            phase += coeff * 2.0 * np.pi * Zk
        phase *= mask
        return {
            'phase': phase,
            'dx': dx,
            'wavelength': wavelength,
            'aperture': aperture,
            'mask': mask,
        }

    # ----------------------------------------------------------- Run / stop ---

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.summary.append('AO loop already running.')
            return

        prepared = self._build_input_phase()
        if prepared is None:
            return

        n_iter = int(self.spin_max_iter.value())
        # tol == 0 means "no early stop"; map to None so the helper
        # behaves like the example does.
        tol_value = float(self.spin_tol.value())
        tol = tol_value if tol_value > 0.0 else None

        params = {
            'phase_disturbance': prepared['phase'],
            'dx': prepared['dx'],
            'wavelength': prepared['wavelength'],
            'aperture': prepared['aperture'],
            'n_actuators': int(self.spin_n_act.value()),
            'stroke_waves': float(self.spin_stroke.value()),
            'coupling': float(self.spin_coupling.value()),
            'gain': float(self.spin_gain.value()),
            'leak': float(self.spin_leak.value()),
            'tol': tol,
            'n_iterations': n_iter,
            'mask': prepared['mask'],
            'wfs_type': self.combo_wfs.currentText(),
            'subaperture_grid': int(self.spin_subap.value()),
            'wfs_noise': float(self.spin_wfs_noise.value()),
            'modal_basis': self.combo_modal.currentText(),
            'max_radial_order': int(self.spin_max_order.value()),
        }
        self._mask = prepared['mask']

        self.summary.append(
            f'AO loop: N={prepared["phase"].shape[0]} '
            f'n_act={params["n_actuators"]} '
            f'gain={params["gain"]} leak={params["leak"]} '
            f'tol={params["tol"]} n_iter={n_iter}  '
            f'(basis={params["modal_basis"]}, '
            f'wfs={params["wfs_type"]})')

        self.progress_bar.setRange(0, max(n_iter, 1))
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Reset plots.
        if HAS_MPL:
            self._history_iters = []
            self._history_rms = []
            self.fig_residual.clear()
            self.fig_dm.clear()
            self.fig_residual_map.clear()
            self.canvas_residual.draw_idle()
            self.canvas_dm.draw_idle()
            self.canvas_residual_map.draw_idle()

        self._worker = _AOClosedLoopWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is None:
            return
        if self._worker.isRunning():
            self._worker.request_stop()
            self.summary.append('Stop requested; will halt after current iteration.')

    # -------------------------------------------------------- Plot updates ---

    def _on_progress(self, iter_idx: int, residual_rms: float) -> None:
        self.progress_bar.setValue(iter_idx)
        if not HAS_MPL:
            return
        # Append + redraw the convergence curve.
        if not hasattr(self, '_history_iters'):
            self._history_iters = []
            self._history_rms = []
        self._history_iters.append(iter_idx)
        # Guard against log(0).
        self._history_rms.append(max(float(residual_rms), 1e-20))

        self.fig_residual.clear()
        ax = self.fig_residual.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.tick_params(colors='#7a94b8')
        for spine in ax.spines.values():
            spine.set_color('#3a4254')
        ax.semilogy(self._history_iters, self._history_rms,
                    '-o', color='#5cb8ff', markersize=3, linewidth=1.2)
        ax.set_xlabel('Iteration', color='#7a94b8')
        ax.set_ylabel('Residual RMS [rad]', color='#7a94b8')
        ax.set_title('AO closed-loop convergence', color='#cfd6e1')
        ax.grid(True, which='both', alpha=0.25, color='#3a4254')
        self.canvas_residual.draw_idle()

    def _on_finished(self, result: Dict[str, Any]) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._worker = None

        if not result.get('success'):
            self.summary.append(
                f'AO loop FAILED: {result.get("error", "(unknown)")}')
            return

        residual = result['residual_phase']
        cmd = result['dm_command']
        history = result['history']
        iters_run = result['iterations_run']
        stopped = result.get('stopped', False)

        if HAS_MPL:
            # Final convergence plot (already drawn incrementally, but
            # rebuild on the full history to clean up any aliasing).
            self.fig_residual.clear()
            ax = self.fig_residual.add_subplot(111)
            ax.set_facecolor('#0a0c10')
            ax.tick_params(colors='#7a94b8')
            for spine in ax.spines.values():
                spine.set_color('#3a4254')
            iters = np.arange(history.size)
            ax.semilogy(iters, np.maximum(history, 1e-20),
                        '-o', color='#5cb8ff', markersize=3, linewidth=1.2)
            ax.set_xlabel('Iteration', color='#7a94b8')
            ax.set_ylabel('Residual RMS [rad]', color='#7a94b8')
            ax.set_title(
                f'Convergence: {history[0]:.3f} -> {history[-1]:.3f} rad',
                color='#cfd6e1')
            ax.grid(True, which='both', alpha=0.25, color='#3a4254')
            self.canvas_residual.draw_idle()

            # DM command heatmap.
            self.fig_dm.clear()
            ax_dm = self.fig_dm.add_subplot(111)
            ax_dm.set_facecolor('#0a0c10')
            im = ax_dm.imshow(cmd, cmap='RdBu_r', origin='lower',
                              interpolation='nearest')
            ax_dm.set_title('DM commands [rad OPD]', color='#cfd6e1')
            ax_dm.tick_params(colors='#7a94b8')
            cbar = self.fig_dm.colorbar(im, ax=ax_dm, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='#7a94b8')
            self.canvas_dm.draw_idle()

            # Final residual heatmap.
            self.fig_residual_map.clear()
            ax_r = self.fig_residual_map.add_subplot(111)
            ax_r.set_facecolor('#0a0c10')
            im2 = ax_r.imshow(residual, cmap='twilight', origin='lower')
            ax_r.set_title('Final residual phase [rad]', color='#cfd6e1')
            ax_r.tick_params(colors='#7a94b8')
            cbar2 = self.fig_residual_map.colorbar(
                im2, ax=ax_r, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(colors='#7a94b8')
            self.canvas_residual_map.draw_idle()

        # Summary line.
        wfe_initial = history[0] / (2.0 * np.pi)
        wfe_final = history[-1] / (2.0 * np.pi)
        suffix = '  (stopped)' if stopped else ''
        self.summary.append(
            f'AO loop done: iter={iters_run}, '
            f'RMS {history[0]:.4f} -> {history[-1]:.4f} rad '
            f'({wfe_initial:.3f} -> {wfe_final:.3f} waves){suffix}')


# v5.4 audit P1-A: no __all__ -- UI docks are imported by name in
# main_window.py and not exposed at lumenairy.__all__ (V9 walker
# convention; matches the 22 pre-v5.4 docks that ship without it).
