# v5.4 (audit P1-E): expand from 41-LOC stub to full algorithm-dispatched dock
"""
Phase-retrieval dock -- algorithm-dispatched runner.

Wraps the six phase-retrieval entry points exposed by
:mod:`lumenairy.analysis.phase_retrieval`:

    - ``gerchberg_saxton``           (NumPy, source -> target amplitudes)
    - ``error_reduction``            (NumPy, single-intensity + support)
    - ``hybrid_input_output``        (NumPy, Fienup HIO)
    - ``gerchberg_saxton_jax``       (JAX-jit, same physics as GS)
    - ``error_reduction_jax``        (JAX-jit, same physics as ER)
    - ``hybrid_input_output_jax``    (JAX-jit, same physics as HIO)

The dock is intentionally generic: an algorithm dropdown, an iteration
budget, a convergence tolerance, amplitude and phase-wrap constraints,
input-image file pickers, and an initial-phase strategy.  Two
matplotlib canvases (convergence trace + reconstruction preview) update
live as the worker walks the iteration.

Author: Andrew Traverso
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSizePolicy, QSpinBox, QComboBox, QFileDialog, QTextEdit,
    QDoubleSpinBox, QProgressBar, QSplitter, QFormLayout,
    QLineEdit, QCheckBox,
)
from PySide6.QtGui import QFont

# v5.4 (audit P1-F): cooperative-cancel via the library protocol.
# Note: the underlying phase-retrieval functions do NOT take a
# ``progress=`` kwarg and do NOT poll cancellation; we honour Stop by
# chunking the iteration budget at this dock layer and polling
# ``CancellableProgress.should_stop`` between chunks.
from lumenairy.progress import CancellableProgress

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from lumenairy.backend import JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------
#
# The dropdown is populated from this list.  Each entry is
# (display_label, method_id).  Method IDs match the function names in
# :mod:`lumenairy.analysis.phase_retrieval` exactly so the worker can
# dispatch directly from ``self.method`` without a second lookup table.
# JAX twins are appended only when ``JAX_AVAILABLE`` is True.

_NUMPY_METHODS = [
    ('Gerchberg-Saxton (NumPy)', 'gerchberg_saxton'),
    ('Error reduction (NumPy)', 'error_reduction'),
    ('Hybrid input-output (NumPy)', 'hybrid_input_output'),
]

_JAX_METHODS = [
    ('Gerchberg-Saxton (JAX)', 'gerchberg_saxton_jax'),
    ('Error reduction (JAX)', 'error_reduction_jax'),
    ('Hybrid input-output (JAX)', 'hybrid_input_output_jax'),
]

# Methods that require a support mask + measured amplitude (single-
# intensity reconstruction) rather than source + target amplitudes.
_SUPPORT_METHODS = {
    'error_reduction', 'error_reduction_jax',
    'hybrid_input_output', 'hybrid_input_output_jax',
}

_GS_METHODS = {'gerchberg_saxton', 'gerchberg_saxton_jax'}


def _load_intensity_image(path: str) -> np.ndarray:
    """Load an intensity image as a 2-D float64 numpy array.

    Supports ``.npy``, ``.npz`` (uses first array), ``.h5`` / ``.hdf5``
    (uses the first 2-D dataset), and ``.zarr`` (uses the first 2-D
    array).  Other extensions fall through to PIL grayscale.  Negative
    pixels are clamped to zero so the array can be interpreted as an
    amplitude (sqrt of intensity).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.npz':
        with np.load(path) as f:
            arr = f[list(f.keys())[0]]
    elif ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            def _find(g):
                for key in g:
                    item = g[key]
                    if hasattr(item, 'shape') and len(item.shape) == 2:
                        return np.asarray(item)
                    if hasattr(item, 'keys'):
                        sub = _find(item)
                        if sub is not None:
                            return sub
                return None
            arr = _find(f)
            if arr is None:
                raise ValueError(
                    f'No 2-D dataset found in HDF5 file {path!r}')
    elif ext == '.zarr':
        import zarr
        root = zarr.open(path, mode='r')
        for key in root:
            item = root[key]
            if hasattr(item, 'shape') and len(item.shape) == 2:
                arr = np.asarray(item)
                break
        else:
            raise ValueError(
                f'No 2-D array found in Zarr store {path!r}')
    else:
        from PIL import Image
        arr = np.asarray(
            Image.open(path).convert('L'), dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f'Expected 2-D intensity image, got shape {arr.shape}')
    return np.clip(arr, 0.0, None)


def _apply_phase_wrap(phase: np.ndarray, mode: str) -> np.ndarray:
    """Apply a phase-wrap policy to a real phase array."""
    if mode == 'principal_value':
        return np.angle(np.exp(1j * phase))
    if mode == 'unwrap':
        try:
            from skimage.restoration import unwrap_phase
            return np.asarray(unwrap_phase(
                np.angle(np.exp(1j * phase))), dtype=np.float64)
        except ImportError:
            # Fallback: 1-D unwrap row-by-row then column-by-column.
            p = np.unwrap(np.angle(np.exp(1j * phase)), axis=1)
            return np.unwrap(p, axis=0)
    return phase  # 'none'


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _PhaseRetrievalWorker(QThread):
    """Generalised background worker dispatching on ``self.method``.

    Emits:
        progress(iteration: int, error: float)   -- per outer chunk
        preview(amplitude, phase)                 -- every N iters
        finished_result(dict)                     -- final state
    """

    progress = Signal(int, float)
    preview = Signal(object, object)
    finished_result = Signal(object)
    cancelled = Signal()
    # Back-compat: ``_GSWorker`` exposed ``finished_result``; keep it.

    PREVIEW_EVERY = 10   # emit a preview every N outer chunks
    CHUNK_SIZE = 10      # iterations per outer chunk (re-enters Python)

    def __init__(self, method, source_amp, target_amp, support,
                 n_iter, tol, amp_min, amp_max, phase_wrap,
                 initial_phase, beta=0.9):
        super().__init__()
        self.method = str(method)
        self.source_amp = source_amp
        self.target_amp = target_amp
        self.support = support
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.amp_min = float(amp_min)
        self.amp_max = float(amp_max)
        self.phase_wrap = str(phase_wrap)
        self.initial_phase = initial_phase
        self.beta = float(beta)
        # v5.4 (audit P1-F): replace ad-hoc ``_stop_requested`` flag
        # with the library's CancellableProgress so the cancellation
        # protocol matches the other docks (OptimizerDock, ToleranceDock,
        # MultiConfigDock).  ``should_stop`` is polled between chunks.
        self._cancel_progress = CancellableProgress()

    def request_stop(self):
        """Cooperative stop (back-compat shim, checked between chunks)."""
        self._cancel_progress.cancel()

    @Slot()
    def cancel(self):
        """Canonical v5.4 cancel slot -- alias of request_stop()."""
        self._cancel_progress.cancel()

    @property
    def _stop_requested(self) -> bool:
        # Preserve the property name used elsewhere in this file's
        # iteration loops so we don't churn unrelated code.
        return self._cancel_progress.should_stop

    def run(self):
        try:
            if self.method in _GS_METHODS:
                result = self._run_gs()
            elif self.method in _SUPPORT_METHODS:
                result = self._run_support()
            else:
                raise ValueError(
                    f'Unknown phase-retrieval method {self.method!r}')
            # v5.4 (audit P1-F): if user cancelled, emit cancelled
            # before finished so the dock can mark the run as aborted
            # while still showing the partial reconstruction.
            if self._cancel_progress.should_stop:
                self.cancelled.emit()
            self.finished_result.emit(result)
        except Exception as e:
            self.finished_result.emit({
                'success': False,
                'error': f'{type(e).__name__}: {e}',
            })

    # ----- GS dispatch (source + target amplitudes) -----
    def _run_gs(self):
        from lumenairy.analysis import phase_retrieval as pr

        is_jax = self.method.endswith('_jax')
        fn = getattr(pr, self.method)

        # Chunk through the iteration budget so we can emit live
        # progress / preview events without piping a callback into the
        # core function (which has no progress hook today).
        history = []
        phase = self.initial_phase
        cumulative = 0
        last_err = float('inf')
        while cumulative < self.n_iter and not self._stop_requested:
            this_chunk = min(self.CHUNK_SIZE, self.n_iter - cumulative)
            if is_jax:
                # JAX twin only returns (phase, err); no history.
                phase_out, err = fn(
                    self.source_amp, self.target_amp,
                    n_iter=this_chunk, initial_phase=phase)
                chunk_hist = [float(err)]
            else:
                phase_out, err, chunk_hist = fn(
                    self.source_amp, self.target_amp,
                    n_iter=this_chunk, initial_phase=phase,
                    return_history=True)
            phase = phase_out
            history.extend([float(v) for v in chunk_hist])
            cumulative += this_chunk

            err_f = float(err)
            self.progress.emit(cumulative, err_f)

            # Reconstruction preview: amplitude is far-field amplitude
            # given the current source-plane phase.
            if (cumulative // self.CHUNK_SIZE) % self.PREVIEW_EVERY == 0:
                far = np.fft.fftshift(
                    np.fft.fft2(np.fft.ifftshift(
                        self.source_amp * np.exp(1j * phase))))
                self.preview.emit(
                    np.abs(far).astype(np.float64),
                    np.angle(far).astype(np.float64))

            # Tolerance check (only meaningful with NumPy history).
            if abs(last_err - err_f) < self.tol and cumulative > self.CHUNK_SIZE:
                history.append(err_f)
                break
            last_err = err_f

        # Apply phase-wrap policy + amplitude clamp to the recovered phase.
        phase = _apply_phase_wrap(phase, self.phase_wrap)
        far = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(
                self.source_amp * np.exp(1j * phase))))
        amp = np.clip(np.abs(far), self.amp_min, self.amp_max)
        return {
            'success': True,
            'method': self.method,
            'phase': phase,
            'amplitude': amp,
            'history': np.asarray(history, dtype=np.float64),
            'iterations': cumulative,
            'stopped': self._stop_requested,
        }

    # ----- ER / HIO dispatch (single-intensity + support) -----
    def _run_support(self):
        from lumenairy.analysis import phase_retrieval as pr

        is_jax = self.method.endswith('_jax')
        is_hio = self.method.startswith('hybrid')
        fn = getattr(pr, self.method)

        history = []
        obj: Optional[np.ndarray] = None
        cumulative = 0
        last_err = float('inf')

        # JAX twins of ER / HIO take ``init_phase`` (real Fourier-plane
        # phase array), not ``initial_guess``.  We fold the user's
        # ``initial_phase`` into the right kwarg per backend.
        while cumulative < self.n_iter and not self._stop_requested:
            this_chunk = min(self.CHUNK_SIZE, self.n_iter - cumulative)
            kwargs = {'n_iter': this_chunk}
            if is_hio:
                kwargs['beta'] = self.beta
            if is_jax:
                if cumulative == 0 and self.initial_phase is not None:
                    kwargs['init_phase'] = self.initial_phase
                obj_out, err = fn(
                    self.target_amp, self.support, **kwargs)
                chunk_hist = [float(err)]
            else:
                if obj is None and self.initial_phase is not None:
                    obj = (self.support.astype(np.float64)
                           * np.exp(1j * self.initial_phase))
                kwargs['initial_guess'] = obj
                kwargs['return_history'] = True
                obj_out, err, chunk_hist = fn(
                    self.target_amp, self.support, **kwargs)
            obj = obj_out
            history.extend([float(v) for v in chunk_hist])
            cumulative += this_chunk

            err_f = float(err)
            self.progress.emit(cumulative, err_f)

            if (cumulative // self.CHUNK_SIZE) % self.PREVIEW_EVERY == 0:
                self.preview.emit(
                    np.abs(obj).astype(np.float64),
                    np.angle(obj).astype(np.float64))

            if abs(last_err - err_f) < self.tol and cumulative > self.CHUNK_SIZE:
                history.append(err_f)
                break
            last_err = err_f

        phase = _apply_phase_wrap(np.angle(obj), self.phase_wrap)
        amp = np.clip(np.abs(obj), self.amp_min, self.amp_max)
        return {
            'success': True,
            'method': self.method,
            'phase': phase,
            'amplitude': amp,
            'field': obj,
            'history': np.asarray(history, dtype=np.float64),
            'iterations': cumulative,
            'stopped': self._stop_requested,
        }


# Back-compat alias.  The old dock and any external callers expected
# ``_GSWorker``; preserve the symbol so imports don't break.
_GSWorker = _PhaseRetrievalWorker


# ---------------------------------------------------------------------------
# Dock
# ---------------------------------------------------------------------------

class PhaseRetrievalDock(QWidget):
    """Algorithm-dispatched phase-retrieval surface.

    Left pane: controls (algorithm, iteration budget, convergence
    tolerance, amplitude clamp, phase wrap, initial phase strategy,
    file pickers, run/stop).  Right pane: live convergence plot
    (top) and reconstruction preview (bottom)."""

    TARGET_PRESETS = ('Gaussian spot', 'Top-hat', 'Ring',
                      'Dammann grid 4x4')
    PHASE_WRAP_MODES = ('principal_value', 'unwrap', 'none')
    INIT_PHASE_STRATEGIES = ('zeros', 'random', 'from_file')

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker: Optional[_PhaseRetrievalWorker] = None
        self._target_image: Optional[np.ndarray] = None
        self._source_image: Optional[np.ndarray] = None
        self._init_phase_image: Optional[np.ndarray] = None
        self._history_x: list = []
        self._history_y: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Horizontal splitter: controls (left) + plots (right).
        h_split = QSplitter(Qt.Horizontal)
        outer.addWidget(h_split, stretch=1)

        # ----- Left panel: controls -----
        controls = QWidget()
        controls.setMinimumWidth(360)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(2, 2, 2, 2)
        ctrl_layout.setSpacing(6)

        # Algorithm + iteration budget
        algo_group = QGroupBox('Algorithm')
        algo_form = QFormLayout(algo_group)

        self.combo_algo = QComboBox()
        for label, mid in _NUMPY_METHODS:
            self.combo_algo.addItem(label, mid)
        if JAX_AVAILABLE:
            for label, mid in _JAX_METHODS:
                self.combo_algo.addItem(label, mid)
        self.combo_algo.currentIndexChanged.connect(self._on_method_changed)
        algo_form.addRow('Method:', self.combo_algo)

        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 100000)
        self.spin_iter.setValue(200)
        self.spin_iter.setSingleStep(10)
        algo_form.addRow('Max iterations:', self.spin_iter)

        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setDecimals(12)
        self.spin_tol.setRange(0.0, 1.0)
        self.spin_tol.setValue(1e-6)
        self.spin_tol.setSingleStep(1e-7)
        self.spin_tol.setToolTip(
            'Stop when the per-chunk change in error falls below '
            'this value.  Set 0 to disable early-exit.')
        algo_form.addRow('Convergence tol:', self.spin_tol)

        self.spin_beta = QDoubleSpinBox()
        self.spin_beta.setRange(0.0, 2.0)
        self.spin_beta.setValue(0.9)
        self.spin_beta.setSingleStep(0.05)
        self.spin_beta.setDecimals(3)
        self.spin_beta.setToolTip(
            'HIO feedback parameter.  Only used for hybrid input-'
            'output methods.  Typical values are 0.5--1.0.')
        algo_form.addRow('HIO beta:', self.spin_beta)

        ctrl_layout.addWidget(algo_group)

        # Constraint group
        cons_group = QGroupBox('Constraints')
        cons_form = QFormLayout(cons_group)

        self.spin_amp_min = QDoubleSpinBox()
        self.spin_amp_min.setRange(-1e9, 1e9)
        self.spin_amp_min.setValue(0.0)
        self.spin_amp_min.setDecimals(4)
        cons_form.addRow('Amplitude min:', self.spin_amp_min)

        self.spin_amp_max = QDoubleSpinBox()
        self.spin_amp_max.setRange(-1e9, 1e9)
        self.spin_amp_max.setValue(1.0)
        self.spin_amp_max.setDecimals(4)
        cons_form.addRow('Amplitude max:', self.spin_amp_max)

        self.combo_phase_wrap = QComboBox()
        self.combo_phase_wrap.addItems(self.PHASE_WRAP_MODES)
        self.combo_phase_wrap.setCurrentText('principal_value')
        cons_form.addRow('Phase wrap:', self.combo_phase_wrap)

        ctrl_layout.addWidget(cons_group)

        # Input + initial-phase group
        input_group = QGroupBox('Inputs')
        input_form = QFormLayout(input_group)

        self.edit_source = QLineEdit()
        self.edit_source.setPlaceholderText(
            '(optional .npy/.h5/.zarr) pupil-plane intensity')
        self.btn_source = QPushButton('Browse...')
        self.btn_source.clicked.connect(
            lambda: self._pick_file(self.edit_source, 'source'))
        src_row = QHBoxLayout()
        src_row.addWidget(self.edit_source, stretch=1)
        src_row.addWidget(self.btn_source)
        src_wrap = QWidget()
        src_wrap.setLayout(src_row)
        input_form.addRow('Pupil intensity:', src_wrap)

        self.edit_target = QLineEdit()
        self.edit_target.setPlaceholderText(
            '(optional .npy/.h5/.zarr) focal-plane intensity')
        self.btn_target = QPushButton('Browse...')
        self.btn_target.clicked.connect(
            lambda: self._pick_file(self.edit_target, 'target'))
        tgt_row = QHBoxLayout()
        tgt_row.addWidget(self.edit_target, stretch=1)
        tgt_row.addWidget(self.btn_target)
        tgt_wrap = QWidget()
        tgt_wrap.setLayout(tgt_row)
        input_form.addRow('PSF intensity:', tgt_wrap)

        self.spin_N = QSpinBox()
        self.spin_N.setRange(32, 4096)
        self.spin_N.setValue(256)
        self.spin_N.setToolTip(
            'Grid size used when generating synthetic source/target '
            'amplitudes (when no file is loaded).')
        input_form.addRow('Synthetic grid N:', self.spin_N)

        self.combo_target_preset = QComboBox()
        self.combo_target_preset.addItems(self.TARGET_PRESETS)
        input_form.addRow('Synthetic target:', self.combo_target_preset)

        self.spin_sz = QDoubleSpinBox()
        self.spin_sz.setRange(0.01, 1.0)
        self.spin_sz.setValue(0.2)
        self.spin_sz.setDecimals(3)
        input_form.addRow('Synthetic size:', self.spin_sz)

        ctrl_layout.addWidget(input_group)

        # Initial-phase strategy
        init_group = QGroupBox('Initial phase')
        init_form = QFormLayout(init_group)

        self.combo_init = QComboBox()
        self.combo_init.addItems(self.INIT_PHASE_STRATEGIES)
        self.combo_init.setCurrentText('zeros')
        self.combo_init.currentTextChanged.connect(self._on_init_changed)
        init_form.addRow('Strategy:', self.combo_init)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 1_000_000_000)
        self.spin_seed.setValue(0)
        init_form.addRow('Random seed:', self.spin_seed)

        self.edit_init = QLineEdit()
        self.edit_init.setEnabled(False)
        self.edit_init.setPlaceholderText('(only for "from_file")')
        self.btn_init = QPushButton('Browse...')
        self.btn_init.setEnabled(False)
        self.btn_init.clicked.connect(
            lambda: self._pick_file(self.edit_init, 'init'))
        init_row = QHBoxLayout()
        init_row.addWidget(self.edit_init, stretch=1)
        init_row.addWidget(self.btn_init)
        init_wrap = QWidget()
        init_wrap.setLayout(init_row)
        init_form.addRow('Init phase file:', init_wrap)

        ctrl_layout.addWidget(init_group)

        # Run / stop buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton('Run phase retrieval')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        btn_row.addWidget(self.btn_run)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        # Slot left intentionally empty by spec; WAVE 2A wires
        # CancellableProgress separately.  Provide a cooperative-stop
        # local fallback so the button isn't a no-op even before
        # CancellableProgress integration lands.
        self.btn_stop.clicked.connect(self._stop)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        ctrl_layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        ctrl_layout.addWidget(self.progress_bar)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        ctrl_layout.addWidget(self.summary)

        ctrl_layout.addStretch()
        h_split.addWidget(controls)

        # ----- Right panel: vertical splitter of two matplotlib canvases -----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)

        v_split = QSplitter(Qt.Vertical)

        # Top: convergence plot
        conv_wrap = QWidget()
        conv_layout = QVBoxLayout(conv_wrap)
        conv_layout.setContentsMargins(0, 0, 0, 0)
        if HAS_MPL:
            self.fig_conv = Figure(figsize=(6, 3), tight_layout=True)
            self.ax_conv = self.fig_conv.add_subplot(1, 1, 1)
            self.ax_conv.set_xlabel('iteration')
            self.ax_conv.set_ylabel('error')
            self.ax_conv.set_yscale('log')
            self.ax_conv.grid(alpha=0.2)
            self.canvas_conv = FigureCanvas(self.fig_conv)
            # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
            self.canvas_conv.setMinimumSize(0, 0)
            self.canvas_conv.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.toolbar_conv = NavigationToolbar(self.canvas_conv, self)
            conv_layout.addWidget(self.toolbar_conv)
            conv_layout.addWidget(self.canvas_conv, stretch=1)
            self._line_conv, = self.ax_conv.plot(
                [], [], color='#5cb8ff', lw=1.2)
        else:
            conv_layout.addWidget(QLabel(
                'matplotlib unavailable -- convergence plot disabled'))
        v_split.addWidget(conv_wrap)

        # Bottom: reconstruction preview (amplitude + phase side by side)
        recon_wrap = QWidget()
        recon_layout = QVBoxLayout(recon_wrap)
        recon_layout.setContentsMargins(0, 0, 0, 0)
        if HAS_MPL:
            self.fig_recon = Figure(figsize=(6, 3), tight_layout=True)
            self.ax_amp = self.fig_recon.add_subplot(1, 2, 1)
            self.ax_amp.set_title('|reconstruction|')
            self.ax_amp.set_xticks([]); self.ax_amp.set_yticks([])
            self.ax_phase = self.fig_recon.add_subplot(1, 2, 2)
            self.ax_phase.set_title('arg(reconstruction)')
            self.ax_phase.set_xticks([]); self.ax_phase.set_yticks([])
            self.canvas_recon = FigureCanvas(self.fig_recon)
            # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
            self.canvas_recon.setMinimumSize(0, 0)
            self.canvas_recon.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.toolbar_recon = NavigationToolbar(self.canvas_recon, self)
            recon_layout.addWidget(self.toolbar_recon)
            recon_layout.addWidget(self.canvas_recon, stretch=1)
            self._im_amp = None
            self._im_phase = None
        else:
            recon_layout.addWidget(QLabel(
                'matplotlib unavailable -- preview disabled'))
        v_split.addWidget(recon_wrap)

        v_split.setStretchFactor(0, 1)
        v_split.setStretchFactor(1, 1)
        right_layout.addWidget(v_split)
        h_split.addWidget(right)

        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)

        self._on_method_changed()

    # ---------------- helpers ----------------

    def _current_method(self) -> str:
        return self.combo_algo.currentData()

    def _on_method_changed(self):
        method = self._current_method() or ''
        is_hio = method.startswith('hybrid')
        self.spin_beta.setEnabled(is_hio)

    def _on_init_changed(self, value: str):
        is_file = (value == 'from_file')
        self.edit_init.setEnabled(is_file)
        self.btn_init.setEnabled(is_file)
        self.spin_seed.setEnabled(value == 'random')

    def _pick_file(self, line_edit: QLineEdit, role: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f'Select {role} intensity image',
            filter=(
                'Numeric arrays (*.npy *.npz);;'
                'HDF5 (*.h5 *.hdf5);;'
                'Zarr stores (*.zarr);;'
                'Images (*.png *.jpg *.tif);;'
                'All files (*)'))
        if not path:
            return
        try:
            arr = _load_intensity_image(path)
        except Exception as e:
            self.summary.append(
                f'Load failed ({role}): {type(e).__name__}: {e}')
            return
        line_edit.setText(path)
        if role == 'source':
            self._source_image = arr
        elif role == 'target':
            self._target_image = arr
        elif role == 'init':
            self._init_phase_image = arr
        self.summary.append(
            f'Loaded {role}: shape={arr.shape}, '
            f'max={arr.max():.3e}, dtype={arr.dtype}')

    # ---------------- input assembly ----------------

    def _build_inputs(self):
        """Assemble (source_amp, target_amp, support, initial_phase)."""
        method = self._current_method()

        # ----- target / measured amplitude -----
        if self._target_image is not None:
            target = np.sqrt(self._target_image)
            N = target.shape[0]
        else:
            N = int(self.spin_N.value())
            y, x = np.indices((N, N)) - N / 2.0
            r = np.sqrt(x * x + y * y) / (N / 2.0)
            sz = float(self.spin_sz.value())
            preset = self.combo_target_preset.currentIndex()
            if preset == 0:
                target = np.exp(-r * r / (2.0 * sz * sz))
            elif preset == 1:
                target = (r <= sz).astype(np.float64)
            elif preset == 2:
                target = ((r >= sz * 0.8) & (r <= sz)).astype(np.float64)
            else:
                target = np.zeros_like(r)
                spacing = N // 5
                for ix in range(1, 5):
                    for iy in range(1, 5):
                        target[iy * spacing - 1:iy * spacing + 2,
                               ix * spacing - 1:ix * spacing + 2] = 1.0

        # ----- source amplitude -----
        if self._source_image is not None:
            src = np.sqrt(self._source_image)
            if src.shape != target.shape:
                from scipy.ndimage import zoom
                src = zoom(src, target.shape[0] / src.shape[0], order=1)
        else:
            y, x = np.indices(target.shape) - target.shape[0] / 2.0
            r = np.sqrt(x * x + y * y) / (target.shape[0] / 2.0)
            src = np.where(r <= 0.8, 1.0, 0.0).astype(np.float64)

        # ----- power normalisation -----
        src_p = float(np.sum(src ** 2))
        tgt_p = float(np.sum(target ** 2))
        if tgt_p > 0:
            target = target * np.sqrt(src_p / tgt_p)

        # ----- support (for ER/HIO) -----
        if method in _SUPPORT_METHODS:
            support = (src > 0).astype(bool)
        else:
            support = None

        # ----- initial phase -----
        strategy = self.combo_init.currentText()
        if strategy == 'zeros':
            init_phase = np.zeros(target.shape, dtype=np.float64)
        elif strategy == 'random':
            rng = np.random.default_rng(int(self.spin_seed.value()))
            init_phase = rng.uniform(
                -np.pi, np.pi, size=target.shape).astype(np.float64)
        elif strategy == 'from_file' and self._init_phase_image is not None:
            init_phase = self._init_phase_image
            if init_phase.shape != target.shape:
                from scipy.ndimage import zoom
                init_phase = zoom(
                    init_phase,
                    target.shape[0] / init_phase.shape[0], order=1)
        else:
            init_phase = np.zeros(target.shape, dtype=np.float64)

        return src, target, support, init_phase

    # ---------------- run / stop ----------------

    def _run(self):
        if self._worker is not None and self._worker.isRunning():
            self.summary.append('Worker already running -- ignoring.')
            return

        try:
            src, tgt, sup, init_phase = self._build_inputs()
        except Exception as e:
            self.summary.append(
                f'Input build failed: {type(e).__name__}: {e}')
            return

        method = self._current_method()
        n_iter = int(self.spin_iter.value())
        tol = float(self.spin_tol.value())
        amp_min = float(self.spin_amp_min.value())
        amp_max = float(self.spin_amp_max.value())
        phase_wrap = self.combo_phase_wrap.currentText()
        beta = float(self.spin_beta.value())

        # Reset live state.
        self._history_x = []
        self._history_y = []
        if HAS_MPL:
            self._line_conv.set_data([], [])
            self.ax_conv.relim(); self.ax_conv.autoscale_view()
            self.canvas_conv.draw_idle()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setRange(0, n_iter)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.summary.append(
            f'Starting {method}  n_iter={n_iter}  tol={tol:.1e}')

        self._worker = _PhaseRetrievalWorker(
            method=method, source_amp=src, target_amp=tgt,
            support=sup, n_iter=n_iter, tol=tol,
            amp_min=amp_min, amp_max=amp_max,
            phase_wrap=phase_wrap, initial_phase=init_phase,
            beta=beta)
        self._worker.progress.connect(self._on_progress)
        self._worker.preview.connect(self._on_preview)
        self._worker.finished_result.connect(self._on_finished)
        # v5.4 (audit P1-F): also note user cancellation so the summary
        # text can mark a partial reconstruction.
        self._worker.cancelled.connect(
            lambda: self.summary.append('Cancelled by user.'))
        self._worker.start()

    def _stop(self):
        # v5.4 (audit P1-F): cooperative cancel via CancellableProgress.
        # The worker chunks the iteration budget so should_stop is
        # polled every CHUNK_SIZE (=10) iterations, giving responsive
        # cancellation despite the core phase_retrieval functions not
        # accepting a ``progress=`` kwarg.
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.btn_stop.setEnabled(False)
            self.summary.append('Cancellation requested -- waiting '
                                'for current chunk to finish...')

    # ---------------- live signal slots ----------------

    def _on_progress(self, iteration: int, error: float):
        self._history_x.append(int(iteration))
        self._history_y.append(float(error))
        self.progress_bar.setValue(int(iteration))
        if HAS_MPL:
            self._line_conv.set_data(self._history_x, self._history_y)
            self.ax_conv.relim()
            self.ax_conv.autoscale_view()
            self.canvas_conv.draw_idle()

    def _on_preview(self, amplitude, phase):
        if not HAS_MPL:
            return
        if self._im_amp is None:
            self._im_amp = self.ax_amp.imshow(
                amplitude, cmap='viridis', origin='lower')
            self._im_phase = self.ax_phase.imshow(
                phase, cmap='twilight', origin='lower',
                vmin=-np.pi, vmax=np.pi)
        else:
            self._im_amp.set_data(amplitude)
            self._im_amp.set_clim(amplitude.min(), max(amplitude.max(), 1e-30))
            self._im_phase.set_data(phase)
        self.canvas_recon.draw_idle()

    def _on_finished(self, result):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._worker = None
        if not result.get('success'):
            self.summary.append(f'Failed: {result.get("error")}')
            return
        hist = result.get('history')
        iters = int(result.get('iterations', 0))
        if hist is not None and len(hist):
            self.summary.append(
                f'Done ({result["method"]})  iters={iters}  '
                f'err[-1]={float(hist[-1]):.3e}  '
                f'min={float(hist.min()):.3e}')
        else:
            self.summary.append(
                f'Done ({result["method"]})  iters={iters}')
        if result.get('stopped'):
            self.summary.append('  (stopped by user)')
        # Final preview refresh in case the last chunk wasn't on the
        # preview-emit cadence.
        if HAS_MPL and 'amplitude' in result and 'phase' in result:
            self._on_preview(result['amplitude'], result['phase'])

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
