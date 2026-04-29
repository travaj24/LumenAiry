"""
Phase-retrieval dock — Gerchberg-Saxton / error-reduction runner.

Primary use case: design a phase-only element (source amplitude is
given, target amplitude is the desired far-field pattern).  A
secondary mode runs error-reduction for CDI-style retrieval from a
single far-field intensity with a support constraint.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSpinBox, QComboBox, QFileDialog, QTextEdit,
    QDoubleSpinBox, QProgressBar,
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


class _GSWorker(QThread):
    progress = Signal(int, float)
    finished_result = Signal(object)

    def __init__(self, algo, source_amp, target_amp, support, n_iter):
        super().__init__()
        self.algo = algo
        self.source_amp = source_amp
        self.target_amp = target_amp
        self.support = support
        self.n_iter = n_iter

    def run(self):
        try:
            if self.algo == 'gs':
                from lumenairy.phase_retrieval import gerchberg_saxton
                phase, history = gerchberg_saxton(
                    self.source_amp, self.target_amp,
                    n_iter=self.n_iter, return_history=True)
                self.finished_result.emit({
                    'success': True, 'phase': phase,
                    'history': np.asarray(history)})
            else:
                from lumenairy.phase_retrieval import error_reduction
                field, history = error_reduction(
                    self.target_amp, self.support,
                    n_iter=self.n_iter, return_history=True)
                self.finished_result.emit({
                    'success': True, 'field': field,
                    'history': np.asarray(history)})
        except Exception as e:
            self.finished_result.emit({
                'success': False,
                'error': f'{type(e).__name__}: {e}'})


class PhaseRetrievalDock(QWidget):
    """UI for Gerchberg-Saxton and error-reduction retrieval."""

    TARGET_PRESETS = ('Gaussian spot', 'Top-hat', 'Ring',
                       'Dammann grid 4x4')

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Algorithm selector
        algo_box = QGroupBox('Algorithm')
        algo_row = QHBoxLayout(algo_box)
        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            'Gerchberg-Saxton (amplitude -> amplitude)',
            'Error reduction (|FT|^2 + support)',
        ])
        algo_row.addWidget(self.combo_algo)
        algo_row.addWidget(QLabel('iterations:'))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 10000)
        self.spin_iter.setValue(200)
        algo_row.addWidget(self.spin_iter)
        algo_row.addStretch()
        layout.addWidget(algo_box)

        # Source / target
        src_box = QGroupBox('Source + target')
        src_layout = QVBoxLayout(src_box)
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel('Grid N:'))
        self.spin_N = QSpinBox()
        self.spin_N.setRange(32, 2048)
        self.spin_N.setValue(256)
        src_row.addWidget(self.spin_N)
        src_row.addWidget(QLabel('Target preset:'))
        self.combo_target = QComboBox()
        self.combo_target.addItems(self.TARGET_PRESETS)
        src_row.addWidget(self.combo_target)
        src_row.addWidget(QLabel('size param:'))
        self.spin_sz = QDoubleSpinBox()
        self.spin_sz.setRange(0.01, 1.0)
        self.spin_sz.setValue(0.2)
        self.spin_sz.setDecimals(3)
        self.spin_sz.setToolTip(
            'For Gaussian: sigma in grid-fraction units.\n'
            'For top-hat/ring: radius in grid-fraction units.')
        src_row.addWidget(self.spin_sz)
        src_layout.addLayout(src_row)

        btn_row = QHBoxLayout()
        self.btn_load_img = QPushButton('Load target image...')
        self.btn_load_img.clicked.connect(self._load_image)
        btn_row.addWidget(self.btn_load_img)
        self.btn_run = QPushButton('Run')
        self.btn_run.clicked.connect(self._run)
        btn_row.addWidget(self.btn_run)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        src_layout.addLayout(btn_row)
        layout.addWidget(src_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        if HAS_MPL:
            self.fig = Figure(figsize=(7, 4), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, stretch=1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

        self._target_image = None  # custom image loaded from file

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load target amplitude image', filter='Images (*.png *.jpg *.tif)')
        if not path:
            return
        try:
            from PIL import Image
            img = np.asarray(Image.open(path).convert('L'), dtype=np.float64)
            img /= max(img.max(), 1.0)
            self._target_image = img
            self.summary.append(
                f'Target image loaded: {img.shape}, max={img.max():.3f}')
        except Exception as e:
            self.summary.append(
                f'Image load failed: {type(e).__name__}: {e}')

    def _build_source_target(self):
        N = self.spin_N.value()
        y, x = np.indices((N, N)) - N / 2
        r = np.sqrt(x * x + y * y) / (N / 2)

        # source: uniform unit-disk amplitude, 80% fill
        source = np.where(r <= 0.8, 1.0, 0.0).astype(np.float64)

        # target: preset or image
        if self._target_image is not None:
            from scipy.ndimage import zoom
            tgt = self._target_image
            if tgt.shape[0] != N:
                tgt = zoom(tgt, N / tgt.shape[0], order=1)
            target = tgt.astype(np.float64)
        else:
            sz = self.spin_sz.value()
            preset = self.combo_target.currentIndex()
            if preset == 0:   # Gaussian
                target = np.exp(-r * r / (2 * sz * sz))
            elif preset == 1: # top-hat
                target = (r <= sz).astype(np.float64)
            elif preset == 2: # ring
                target = ((r >= sz * 0.8) & (r <= sz)).astype(np.float64)
            else:             # Dammann 4x4
                target = np.zeros_like(r)
                spacing = N // 5
                for ix in range(1, 5):
                    for iy in range(1, 5):
                        target[iy * spacing - 1:iy * spacing + 2,
                               ix * spacing - 1:ix * spacing + 2] = 1.0
        # Normalise power
        target *= np.sqrt(np.sum(source ** 2) / max(np.sum(target ** 2), 1e-30))
        # Support: where source is nonzero
        support = source > 0
        return source, target, support

    def _run(self):
        src, tgt, sup = self._build_source_target()
        algo = 'gs' if self.combo_algo.currentIndex() == 0 else 'er'
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self._worker = _GSWorker(
            algo, src, tgt, sup, self.spin_iter.value())
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._on_finished({'success': False, 'error': 'Stopped.'})

    def _on_finished(self, res):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._worker = None
        if not res.get('success'):
            self.summary.append(f'Failed: {res.get("error")}')
            return
        history = res['history']
        if HAS_MPL:
            self.fig.clear()
            ax_hist = self.fig.add_subplot(1, 2, 1)
            ax_hist.plot(history, color='#5cb8ff', lw=1.2)
            ax_hist.set_xlabel('iteration')
            ax_hist.set_ylabel('error')
            ax_hist.set_yscale('log')
            ax_hist.grid(alpha=0.2)
            ax_hist.set_title(f'Error history ({len(history)} iters)')

            ax_res = self.fig.add_subplot(1, 2, 2)
            if 'phase' in res:
                ax_res.imshow(res['phase'], cmap='twilight',
                              origin='lower')
                ax_res.set_title('Retrieved phase')
            else:
                ax_res.imshow(np.abs(res['field']), cmap='viridis',
                              origin='lower')
                ax_res.set_title('|field|')
            ax_res.set_xticks([]); ax_res.set_yticks([])
            self.canvas.draw()
        self.summary.append(
            f'Converged: error[-1] = {history[-1]:.3e} '
            f'(history min = {history.min():.3e})')
