# v5.4 (audit P3-A): expand from 162-LOC Schell-source-only to 4 tabs
"""Partial-coherence imaging dock.

Tab 1 preserves the legacy Schell / Koehler source UI verbatim.
Tabs 2-4 wrap koehler_image / extended_source_image /
mutual_coherence from lumenairy.analysis.coherence.  Heavy work
runs on a shared _CoherenceAnalysisWorker QThread.

Author: Andrew Traverso
"""

import os
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QComboBox,
    QSizePolicy, QTextEdit, QTabWidget, QFileDialog, QLineEdit,
)
from PySide6.QtGui import QFont
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


# Legacy Schell / Koehler worker (preserved for Tab 1).
class _KoehlerWorker(QThread):
    finished_result = Signal(object)

    def __init__(self, sm, params):
        super().__init__()
        self.sm = sm
        self.params = params

    def run(self):
        try:
            import lumenairy as la
            pres = self.sm.to_prescription()
            wv = self.sm.wavelength_m
            res = la.koehler_image(
                prescription=pres, wavelength=wv,
                source_sigma=self.params['sigma'],
                N=self.params['N'], dx=self.params['dx'],
                n_modes=self.params['n_modes'])
            self.finished_result.emit(res)
        except Exception as exc:
            self.finished_result.emit(
                {'error': f'{type(exc).__name__}: {exc}'})


# Shared analysis worker for tabs 2-4.  mode in {'koehler','extended','mcf'}.
class _CoherenceAnalysisWorker(QThread):
    """Dispatches Koehler / extended-source / MCF library calls."""

    finished_result = Signal(object)

    def __init__(self, sm, mode, params):
        super().__init__()
        self.sm = sm
        self.mode = mode
        self.params = params

    @staticmethod
    def _load_arrays(path):
        """Return (list_of_arrays, dx_or_None) from .npy / .npz / .h5."""
        if not path or not os.path.exists(path):
            return [], None
        ext = os.path.splitext(path)[1].lower()
        arrays, dx = [], None
        if ext == '.npy':
            arrays.append(np.load(path))
        elif ext == '.npz':
            with np.load(path) as z:
                dx = float(z['dx']) if 'dx' in z.files else None
                arrays.extend(z[k] for k in z.files if k != 'dx')
        elif ext in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(path, 'r') as f:
                dx = float(f.attrs['dx']) if 'dx' in f.attrs else None
                arrays.extend(np.asarray(f[k][...]) for k in f.keys()
                              if isinstance(f[k], h5py.Dataset))
        return arrays, dx

    @classmethod
    def _load_object(cls, path, N):
        """Load an (N, N) complex object amplitude; demo grating on miss."""
        arr = None
        try:
            arrays, _ = cls._load_arrays(path)
            if arrays:
                arr = arrays[0]
        except Exception:
            arr = None
        if arr is None:
            x = (np.arange(N) - N / 2) / N
            X, Y = np.meshgrid(x, x)
            R = np.sqrt(X ** 2 + Y ** 2)
            arr = (0.5 + 0.5 * np.cos(2 * np.pi * 6 * X)) * (R < 0.45)
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape[0] != N or arr.shape[1] != N:
            ix = np.linspace(0, arr.shape[0] - 1, N).astype(int)
            iy = np.linspace(0, arr.shape[1] - 1, N).astype(int)
            arr = arr[np.ix_(ix, iy)]
        return arr.astype(np.complex128)

    @staticmethod
    def _build_source_angles(shape, fwhm_rad, n):
        """Return (angles, weights) for a parametric source distribution."""
        hw = max(fwhm_rad, 1e-9)
        ax = np.linspace(-hw, hw, n)
        AX, AY = np.meshgrid(ax, ax)
        R = np.sqrt(AX ** 2 + AY ** 2)
        if shape == 'Annular':
            W = ((R <= hw) & (R >= 0.6 * hw)).astype(np.float64)
        elif shape == 'Gaussian':
            W = np.exp(-(R ** 2) / (2.0 * (hw / 2.355) ** 2))
        elif shape == 'Custom':
            W = np.clip(1.0 - (R / hw), 0.0, 1.0)
        else:  # 'Circular'
            W = (R <= hw).astype(np.float64)
        mask = W > 0
        return (list(zip(AX[mask].ravel(), AY[mask].ravel())),
                W[mask].ravel().tolist())

    @classmethod
    def _load_ensemble(cls, path):
        """Load complex-field ensemble; synthetic dataset on miss."""
        if not path or not os.path.exists(path):
            rng = np.random.default_rng(0)
            N = 128
            x = np.arange(N) - N / 2
            X, Y = np.meshgrid(x, x)
            fields = []
            for _ in range(16):
                kx, ky = rng.normal(0, 0.05), rng.normal(0, 0.05)
                f = (np.exp(-(X ** 2 + Y ** 2) / (2 * (N / 6) ** 2))
                     * np.exp(1j * (kx * X + ky * Y)))
                fields.append(f.astype(np.complex128))
            return fields, 1e-6
        arrays, dx = cls._load_arrays(path)
        fields = []
        for arr in arrays:
            arr = np.asarray(arr)
            if arr.ndim == 3:
                fields.extend(np.asarray(s, dtype=np.complex128) for s in arr)
            elif arr.ndim == 2:
                fields.append(np.asarray(arr, dtype=np.complex128))
        if not fields:
            raise ValueError('No 2-D / 3-D complex fields in file.')
        return fields, (dx if dx is not None else 1e-6)

    def run(self):
        try:
            import lumenairy as la
            p = self.params
            pres = self.sm.to_prescription()
            wv = float(p.get('wavelength', self.sm.wavelength_m))
            if self.mode == 'koehler':
                N, dx = int(p['N']), float(p['dx'])
                obj = self._load_object(p.get('object_path', ''), N)
                img = la.koehler_image(
                    object_field=obj, prescription=pres, wavelength=wv,
                    dx=dx, condenser_NA=float(p['cond_na']),
                    n_source_points=int(p['n_source']),
                    focal_length=p.get('focal_length'))
                self.finished_result.emit({
                    'mode': 'koehler', 'image': np.asarray(img), 'dx': dx,
                    'cond_na': p['cond_na'], 'obj_na': p['obj_na']})
            elif self.mode == 'extended':
                N, dx = int(p['N']), float(p['dx'])
                angles, weights = self._build_source_angles(
                    p['shape'], float(p['fwhm']), int(p['source_n']))
                obj = self._load_object(p.get('object_path', ''), N)
                img = la.extended_source_image(
                    object_field=obj, prescription=pres, wavelength=wv,
                    dx=dx, source_angles=angles, source_weights=weights,
                    focal_length=p.get('focal_length'))
                self.finished_result.emit({
                    'mode': 'extended', 'image': np.asarray(img), 'dx': dx,
                    'n_source': len(angles), 'shape': p['shape']})
            elif self.mode == 'mcf':
                fields, dx = self._load_ensemble(p.get('ensemble_path', ''))
                Gamma, x = la.mutual_coherence(fields=fields, dx=dx)
                i1 = int(np.argmin(np.abs(x - float(p['x1']))))
                i2 = int(np.argmin(np.abs(x - float(p['x2']))))
                self.finished_result.emit({
                    'mode': 'mcf', 'Gamma': Gamma, 'x': x, 'i1': i1, 'i2': i2,
                    'value': complex(Gamma[i1, i2]),
                    'n_ensemble': len(fields)})
            else:
                self.finished_result.emit(
                    {'error': f'Unknown mode {self.mode!r}'})
        except Exception as exc:
            self.finished_result.emit(
                {'error': f'{type(exc).__name__}: {exc}'})


class CoherenceDock(QWidget):
    """Partial-coherence imaging (4-tab): Schell, Koehler, Extended, MCF."""

    _BG = '#0a0c10'
    _FG = '#dde8f8'
    _MUTED = '#7a94b8'
    _AXIS = '#334054'

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None              # legacy worker for Tab 1
        self._analysis_worker = None     # shared worker for Tabs 2-4

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_schell_tab(), 'Schell source')
        self.tabs.addTab(self._build_koehler_tab(), 'Koehler imaging')
        self.tabs.addTab(self._build_extended_tab(), 'Extended source')
        self.tabs.addTab(self._build_mcf_tab(), 'Mutual coherence')
        outer.addWidget(self.tabs, stretch=1)

    # Tab 1: legacy Schell / Koehler source UI (preserved verbatim).
    def _build_schell_tab(self):
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        param = QGroupBox('Source / illumination')
        form = QFormLayout(param)
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(
            ['Circular', 'Annular', 'Dipole', 'Quadrupole'])
        form.addRow('Source shape:', self.combo_shape)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.01, 1.0)
        self.spin_sigma.setSingleStep(0.05)
        self.spin_sigma.setValue(0.5)
        self.spin_sigma.setToolTip(
            'Partial-coherence factor sigma = NA_source / NA_pupil.  '
            '0 = fully coherent, 1 = critical illumination.')
        form.addRow('sigma (partial coherence):', self.spin_sigma)
        self.spin_modes = QSpinBox()
        self.spin_modes.setRange(1, 256)
        self.spin_modes.setValue(16)
        self.spin_modes.setToolTip(
            'Number of mutually-incoherent modes to sum in the '
            'Koehler decomposition.  More = closer to a continuous '
            'source.')
        form.addRow('Source modes:', self.spin_modes)
        self.spin_N = QSpinBox()
        self.spin_N.setRange(64, 4096)
        self.spin_N.setValue(256)
        form.addRow('Image grid N:', self.spin_N)
        self.spin_dx_um = QDoubleSpinBox()
        self.spin_dx_um.setRange(0.01, 1000.0)
        self.spin_dx_um.setValue(0.5)
        self.spin_dx_um.setSuffix(' um')
        form.addRow('Image dx:', self.spin_dx_um)
        outer.addWidget(param)
        self.btn_run = QPushButton('Run: Compute partial-coherence image')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        outer.addWidget(self.btn_run)
        self.fig = Figure(figsize=(6, 3.4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas, stretch=1)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(100)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        outer.addWidget(self.summary)
        self._draw_empty()
        return tab

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Sum incoherent images from a Koehler-decomposed\n'
                'extended source.  Useful for lithography NA-sigma\n'
                'analysis and microscope condenser illumination.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _run(self):
        params = dict(
            sigma=float(self.spin_sigma.value()),
            n_modes=int(self.spin_modes.value()),
            N=int(self.spin_N.value()),
            dx=float(self.spin_dx_um.value()) * 1e-6,
        )
        self.btn_run.setEnabled(False)
        self.summary.setPlainText('Computing...')
        self._worker = _KoehlerWorker(self.sm, params)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, res):
        self.btn_run.setEnabled(True)
        self._worker = None
        if isinstance(res, dict) and 'error' in res:
            self.summary.setPlainText(
                f'koehler_image failed:\n  {res["error"]}')
            return
        try:
            I = np.asarray(getattr(res, 'image', res))
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor('#0a0c10')
            ax.imshow(I, cmap='inferno', origin='lower')
            ax.set_title('Partial-coherence image',
                         color='#dde8f8', fontfamily='monospace')
            ax.tick_params(colors='#7a94b8', labelsize=8)
            for s in ax.spines.values():
                s.set_color('#334054')
            self.canvas.draw_idle()
            self.summary.setPlainText(
                f'Image shape: {I.shape}; peak {I.max():.4e}')
        except Exception as exc:
            self.summary.setPlainText(
                f'Could not display result: {type(exc).__name__}: {exc}')

    # Tab helpers shared by tabs 2-4.
    def _make_canvas(self):
        fig = Figure(figsize=(6, 3.4), dpi=100, facecolor=self._BG)
        canvas = FigureCanvasQTAgg(fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        canvas.setMinimumSize(0, 0)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return fig, canvas

    def _make_summary(self):
        s = QTextEdit()
        s.setReadOnly(True)
        s.setMaximumHeight(100)
        s.setFont(QFont('Consolas', 10))
        s.setStyleSheet(
            f'QTextEdit{{background:{self._BG};color:{self._MUTED};'
            'border:none}}')
        return s

    def _make_file_picker(self, label, attr_name):
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText('(leave blank for demo data)')
        btn = QPushButton('Browse...')
        btn.clicked.connect(lambda: self._pick_file(edit, label))
        row.addWidget(edit, stretch=1)
        row.addWidget(btn)
        setattr(self, attr_name, edit)
        return row

    def _pick_file(self, edit, label):
        path, _ = QFileDialog.getOpenFileName(
            self, label, '', 'Arrays (*.npy *.npz *.h5 *.hdf5)')
        if path:
            edit.setText(path)

    def _placeholder(self, fig, canvas, text):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self._BG)
        ax.text(0.5, 0.5, text, color=self._MUTED, ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors=self._MUTED)
        for s in ax.spines.values():
            s.set_color(self._AXIS)
        canvas.draw_idle()

    @staticmethod
    def _dspin(lo, hi, val, suffix=None, decimals=None, step=None, tip=None):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        if decimals is not None: sb.setDecimals(decimals)
        if step is not None: sb.setSingleStep(step)
        sb.setValue(val)
        if suffix: sb.setSuffix(suffix)
        if tip: sb.setToolTip(tip)
        return sb

    @staticmethod
    def _ispin(lo, hi, val, tip=None):
        sb = QSpinBox(); sb.setRange(lo, hi); sb.setValue(val)
        if tip: sb.setToolTip(tip)
        return sb

    def _attach_runner(self, outer, btn_label, callback, placeholder_text):
        """Append run-button + canvas + summary; return (btn, fig, canvas, sum)."""
        btn = QPushButton(btn_label)
        btn.setObjectName('run_button')
        btn.clicked.connect(callback)
        outer.addWidget(btn)
        fig, canvas = self._make_canvas()
        outer.addWidget(canvas, stretch=1)
        summary = self._make_summary()
        outer.addWidget(summary)
        self._placeholder(fig, canvas, placeholder_text)
        return btn, fig, canvas, summary

    def _new_tab(self):
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        return tab, outer

    def _file_box(self, title, attr_name, label):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.addLayout(self._make_file_picker(label, attr_name))
        return box

    # Tab 2: Koehler imaging (lumenairy.koehler_image).
    def _build_koehler_tab(self):
        tab, outer = self._new_tab()
        param = QGroupBox('Koehler illumination parameters')
        form = QFormLayout(param)
        self.koehler_cond_na = self._dspin(
            0.001, 0.999, 0.1, decimals=3, step=0.01,
            tip='Condenser numerical aperture.')
        form.addRow('Condenser NA:', self.koehler_cond_na)
        self.koehler_obj_na = self._dspin(
            0.001, 0.999, 0.25, decimals=3, step=0.01,
            tip='Objective NA (summary only).')
        form.addRow('Objective NA:', self.koehler_obj_na)
        self.koehler_wv = QComboBox()
        self._populate_wavelength_combo(self.koehler_wv)
        form.addRow('Wavelength:', self.koehler_wv)
        self.koehler_N = self._ispin(32, 4096, 128)
        form.addRow('Grid N:', self.koehler_N)
        self.koehler_n_src = self._ispin(
            3, 64, 9, tip='Source samples per axis (masked to disk).')
        form.addRow('Source points (per axis):', self.koehler_n_src)
        self.koehler_dx_um = self._dspin(0.01, 1000.0, 1.0, suffix=' um')
        form.addRow('Pixel dx:', self.koehler_dx_um)
        outer.addWidget(param)
        outer.addWidget(self._file_box(
            'Object amplitude file (npy / npz / h5)',
            'koehler_obj_edit', 'Select object amplitude'))
        (self.btn_koehler, self.fig_koehler,
         self.canvas_koehler, self.summary_koehler) = self._attach_runner(
            outer, 'Run: Koehler imaging', self._run_koehler,
            'Koehler imaging: integrate the coherent PSF\n'
            'over the condenser-NA pupil.\n'
            'Provide an object amplitude or use the built-in demo.')
        return tab

    def _populate_wavelength_combo(self, combo):
        combo.clear()
        wvs = getattr(self.sm, 'wavelengths_nm', None) or [
            self.sm.wavelength_m * 1e9]
        for wv in wvs:
            combo.addItem(f'{float(wv):.1f} nm', float(wv))
        combo.setCurrentIndex(0)

    def _dispatch_worker(self, mode, params, btn, summary, msg):
        btn.setEnabled(False)
        summary.setPlainText(msg)
        self._analysis_worker = _CoherenceAnalysisWorker(self.sm, mode, params)
        self._analysis_worker.finished_result.connect(
            self._on_analysis_finished)
        self._analysis_worker.start()

    def _run_koehler(self):
        wv_nm = self.koehler_wv.currentData() or self.sm.wavelength_m * 1e9
        self._dispatch_worker('koehler', dict(
            cond_na=float(self.koehler_cond_na.value()),
            obj_na=float(self.koehler_obj_na.value()),
            wavelength=float(wv_nm) * 1e-9,
            N=int(self.koehler_N.value()),
            n_source=int(self.koehler_n_src.value()),
            dx=float(self.koehler_dx_um.value()) * 1e-6,
            object_path=self.koehler_obj_edit.text().strip(),
        ), self.btn_koehler, self.summary_koehler,
            'Computing Koehler image...')

    # Tab 3: Extended source (lumenairy.extended_source_image).
    def _build_extended_tab(self):
        tab, outer = self._new_tab()
        param = QGroupBox('Extended-source distribution')
        form = QFormLayout(param)
        self.ext_shape = QComboBox()
        self.ext_shape.addItems(['Circular', 'Annular', 'Gaussian', 'Custom'])
        form.addRow('Source shape:', self.ext_shape)
        self.ext_fwhm_mrad = self._dspin(
            0.01, 500.0, 20.0, suffix=' mrad',
            tip='Full-width at half-max of source angular spread.')
        form.addRow('Source FWHM:', self.ext_fwhm_mrad)
        self.ext_source_n = self._ispin(
            3, 64, 9, tip='Source samples per axis.')
        form.addRow('Source N (per axis):', self.ext_source_n)
        self.ext_prop_m = self._dspin(
            0.0, 100.0, 0.0, suffix=' m', decimals=4,
            tip='Image-plane propagation distance (0 = at lens exit).')
        form.addRow('Propagation distance:', self.ext_prop_m)
        self.ext_focal_m = self._dspin(
            0.0, 100.0, 0.1, suffix=' m', decimals=4,
            tip='Back focal length (focal_length library kwarg).')
        form.addRow('Lens focal length:', self.ext_focal_m)
        self.ext_N = self._ispin(32, 4096, 128)
        form.addRow('Grid N:', self.ext_N)
        self.ext_dx_um = self._dspin(0.01, 1000.0, 1.0, suffix=' um')
        form.addRow('Pixel dx:', self.ext_dx_um)
        outer.addWidget(param)
        outer.addWidget(self._file_box(
            'Object amplitude file (optional)',
            'ext_obj_edit', 'Select object amplitude'))
        (self.btn_ext, self.fig_ext,
         self.canvas_ext, self.summary_ext) = self._attach_runner(
            outer, 'Run: Extended-source PSF', self._run_extended,
            'Extended-source PSF: arbitrary angular source\n'
            'distribution summed incoherently through the lens.')
        return tab

    def _run_extended(self):
        focal = float(self.ext_focal_m.value())
        self._dispatch_worker('extended', dict(
            shape=self.ext_shape.currentText(),
            fwhm=float(self.ext_fwhm_mrad.value()) * 1e-3,
            source_n=int(self.ext_source_n.value()),
            wavelength=self.sm.wavelength_m,
            N=int(self.ext_N.value()),
            dx=float(self.ext_dx_um.value()) * 1e-6,
            focal_length=(focal if focal > 0 else None),
            propagation=float(self.ext_prop_m.value()),
            object_path=self.ext_obj_edit.text().strip(),
        ), self.btn_ext, self.summary_ext,
            'Computing extended-source PSF...')

    # Tab 4: Mutual coherence (lumenairy.mutual_coherence).
    def _build_mcf_tab(self):
        tab, outer = self._new_tab()
        outer.addWidget(self._file_box(
            'Source ensemble file (npy / npz / h5)',
            'mcf_ensemble_edit', 'Select field ensemble'))
        ref_box = QGroupBox('Reference points')
        ref_form = QFormLayout(ref_box)
        self.mcf_x1_um = self._dspin(-1e6, 1e6, 0.0, suffix=' um')
        ref_form.addRow('x1 (reference):', self.mcf_x1_um)
        self.mcf_x2_um = self._dspin(-1e6, 1e6, 10.0, suffix=' um')
        ref_form.addRow('x2 (sweep / probe):', self.mcf_x2_um)
        outer.addWidget(ref_box)
        self.btn_mcf = QPushButton('Run: Evaluate mutual coherence')
        self.btn_mcf.setObjectName('run_button')
        self.btn_mcf.clicked.connect(self._run_mcf)
        outer.addWidget(self.btn_mcf)
        self.mcf_value_label = QLabel('Gamma(x1, x2) = (not yet evaluated)')
        self.mcf_value_label.setFont(QFont('Consolas', 10))
        self.mcf_value_label.setStyleSheet(
            f'QLabel{{background:{self._BG};color:{self._FG};'
            'padding:4px;border:1px solid #334054}}')
        outer.addWidget(self.mcf_value_label)
        self.fig_mcf, self.canvas_mcf = self._make_canvas()
        outer.addWidget(self.canvas_mcf, stretch=1)
        self.summary_mcf = self._make_summary()
        outer.addWidget(self.summary_mcf)
        self._placeholder(
            self.fig_mcf, self.canvas_mcf,
            'Mutual coherence: |Gamma(x1, x2)| with x1 fixed and\n'
            'x2 swept across the grid.')
        return tab

    def _run_mcf(self):
        self.mcf_value_label.setText('Gamma(x1, x2) = computing...')
        self._dispatch_worker('mcf', dict(
            ensemble_path=self.mcf_ensemble_edit.text().strip(),
            x1=float(self.mcf_x1_um.value()) * 1e-6,
            x2=float(self.mcf_x2_um.value()) * 1e-6,
            wavelength=self.sm.wavelength_m,
        ), self.btn_mcf, self.summary_mcf, 'Computing mutual coherence...')

    # Shared result dispatcher for tabs 2-4.
    def _on_analysis_finished(self, res):
        for b in (self.btn_koehler, self.btn_ext, self.btn_mcf):
            b.setEnabled(True)
        self._analysis_worker = None
        if isinstance(res, dict) and 'error' in res:
            err = res['error']
            for box in (self.summary_koehler, self.summary_ext,
                        self.summary_mcf):
                box.setPlainText(f'Analysis failed:\n  {err}')
            return
        {'koehler': self._show_koehler,
         'extended': self._show_extended,
         'mcf': self._show_mcf}.get(res.get('mode'), lambda _r: None)(res)

    def _show_image(self, fig, canvas, I, title, cmap):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self._BG)
        ax.imshow(I, cmap=cmap, origin='lower')
        ax.set_title(title, color=self._FG, fontfamily='monospace')
        ax.tick_params(colors=self._MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(self._AXIS)
        canvas.draw_idle()

    def _show_koehler(self, res):
        I = np.asarray(res['image'])
        self._show_image(self.fig_koehler, self.canvas_koehler, I,
                         'Koehler-illumination image', 'inferno')
        self.summary_koehler.setPlainText(
            f'Image shape: {I.shape}\nPeak: {I.max():.4e}\n'
            f'Mean: {I.mean():.4e}\n'
            f'NA_cond / NA_obj: {res["cond_na"]:.3f} / {res["obj_na"]:.3f}')

    def _show_extended(self, res):
        I = np.asarray(res['image'])
        self._show_image(
            self.fig_ext, self.canvas_ext, I,
            f'Extended-source PSF ({res["shape"]})', 'magma')
        self.summary_ext.setPlainText(
            f'Image shape: {I.shape}\nSource samples: {res["n_source"]}\n'
            f'Source shape: {res["shape"]}\nPeak: {I.max():.4e}')

    def _show_mcf(self, res):
        Gamma = np.asarray(res['Gamma'])
        x = np.asarray(res['x'])
        i1 = res['i1']
        val = res['value']
        self.mcf_value_label.setText(
            f'Gamma(x1, x2) = {val.real:+.4e} {val.imag:+.4e}j   '
            f'|Gamma| = {abs(val):.4e}   '
            f'arg = {np.degrees(np.angle(val)):+.2f} deg')
        self.fig_mcf.clear()
        ax = self.fig_mcf.add_subplot(111)
        ax.set_facecolor(self._BG)
        slice_mag = np.abs(Gamma[i1, :])
        ax.plot(x * 1e6, slice_mag, color='#4ec9b0')
        ax.set_xlabel('x2 [um]', color=self._MUTED, fontfamily='monospace')
        ax.set_ylabel('|Gamma(x1, x2)|', color=self._MUTED,
                      fontfamily='monospace')
        ax.set_title('Mutual coherence slice (x1 fixed)',
                     color=self._FG, fontfamily='monospace')
        ax.tick_params(colors=self._MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(self._AXIS)
        ax.grid(alpha=0.2, color=self._AXIS)
        self.canvas_mcf.draw_idle()
        self.summary_mcf.setPlainText(
            f'Ensemble size: {res["n_ensemble"]}\n'
            f'Gamma shape: {Gamma.shape}\n'
            f'x range: [{x.min() * 1e6:+.2f}, {x.max() * 1e6:+.2f}] um\n'
            f'|Gamma| peak along slice: {slice_mag.max():.4e}')
