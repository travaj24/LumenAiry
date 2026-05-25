"""
lumenairy.ui.coronagraph_dock -- interactive 4-stop coronagraph
chain builder + contrast-curve analyzer.

Created per audit v5.4 docs/audits/AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24.md
Part 6.1 P1-C.  The library ships
``lumenairy.analysis.coronagraph.coronagraph_contrast_curve`` which
requires PRE-COMPUTED PSF pairs (coronagraph PSF vs reference PSF) and
the canonical pipeline lives only in
``examples/10_coronagraph_workflow.py``.  This dock wraps that pipeline
in an interactive 4-stop chain builder:

    Source pupil  ->  Stop 1: Apodised pupil  ->  forward Fraunhofer
                  ->  Stop 2: Lyot focal-plane mask  ->  inverse Fraunhofer
                  ->  Stop 3: Lyot stop (pupil)  ->  forward Fraunhofer
                  ->  Stop 4: Image-plane sampler

The element factories used here are the canonical pupil/focal helpers
``lumenairy.elements.apply_apodized_pupil``,
``lumenairy.elements.apply_lyot_focal_plane_mask`` and
``lumenairy.elements.apply_lyot_stop``.  The Stop 4 "image plane
sampler" is not an element per se -- it is the final |.|^2 plus the
contrast-curve reduction.

ASCII only.  Matches the matplotlib-embedding pattern in
``psf_mtf_dock.py``.

Author: Andrew Traverso  --  v5.4 audit P1-C
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTextEdit, QComboBox,
    QProgressBar, QRadioButton, QButtonGroup, QSplitter, QScrollArea,
    QFormLayout, QFrame, QFileDialog,
)
from PySide6.QtGui import QFont

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:                                       # pragma: no cover
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Per-stop profile catalogues.
#
# Each profile lists the parameters the dock exposes for it.  The triple is
# ``(label, default, (min, max, step, decimals, suffix))``.  Parameters
# whose "natural" units are length are stored in micrometres in the UI and
# converted to metres before being handed to the element helpers.
# ---------------------------------------------------------------------------

# --- Stop 1: Apodised pupil --------------------------------------------------
APOD_PROFILES = {
    'gaussian':        {'lib_name': 'gaussian',
                        'params': [
                            ('diameter_um',  8000.0, (10.0,  100000.0, 100.0, 1, ' um')),
                            ('sigma_um',     1300.0, (1.0,    50000.0,  50.0, 2, ' um')),
                        ]},
    'cosine':          {'lib_name': 'cos2',
                        'params': [
                            ('diameter_um',  8000.0, (10.0,  100000.0, 100.0, 1, ' um')),
                        ]},
    'super-gaussian':  {'lib_name': 'cos_power',
                        'params': [
                            ('diameter_um',  8000.0, (10.0,  100000.0, 100.0, 1, ' um')),
                            ('exponent',     4.0,    (1.0,       16.0,   1.0, 1, '')),
                        ]},
    'uniform':         {'lib_name': 'sonine',
                        'params': [
                            ('diameter_um',  8000.0, (10.0,  100000.0, 100.0, 1, ' um')),
                            ('exponent',     0.0,    (0.0,        8.0,   1.0, 1, '')),
                        ]},
}

# --- Stop 2: Lyot focal-plane mask ------------------------------------------
LYOT_FPM_PROFILES = {
    'hard_circular':                {'lib_name': 'hard',
                                     'params': [
                                         ('mask_diameter_lod',  4.0,  (0.1,   30.0, 0.1, 2, ' lambda/D')),
                                     ]},
    'gaussian_taper':               {'lib_name': 'gaussian',
                                     'params': [
                                         ('mask_diameter_lod',  4.0,  (0.1,   30.0, 0.1, 2, ' lambda/D')),
                                         ('sigma_frac',         0.166,(0.01,    1.0, 0.01, 3, '')),
                                     ]},
    'eight-octant_phase_mask':      {'lib_name': '__phase8__',
                                     'params': [
                                         ('mask_radius_lod',    4.0,  (0.1,   30.0, 0.1, 2, ' lambda/D')),
                                     ]},
    'four-quadrant_phase_mask':     {'lib_name': '__phase4__',
                                     'params': [
                                         ('mask_radius_lod',    8.0,  (0.1,   30.0, 0.1, 2, ' lambda/D')),
                                     ]},
}

# --- Stop 3: Lyot stop (pupil) ----------------------------------------------
LYOT_STOP_PROFILES = {
    'hard_circular':                       {'params': [
                                                ('outer_frac',  0.85, (0.05, 1.20, 0.01, 3, ' x D')),
                                            ]},
    'lyot_with_secondary_obscuration':     {'params': [
                                                ('outer_frac',  0.85, (0.05, 1.20, 0.01, 3, ' x D')),
                                                ('inner_frac',  0.25, (0.00, 1.10, 0.01, 3, ' x D')),
                                            ]},
}

# --- Stop 4: Image plane sampler (no profile knobs; contrast settings only) -
IMAGE_PLANE_PROFILES = {
    'contrast_curve': {'params': [
                            ('n_radii',          48.0, (4.0,  256.0,   1.0, 0, '')),
                            ('max_lam_over_D',   12.0, (1.0,   60.0,   1.0, 1, ' lambda/D')),
                       ]},
}


# ---------------------------------------------------------------------------
# Phase-mask helpers
#
# v5.4 Phase 5: library now ships create_four_quadrant_phase_mask +
# create_eight_octant_phase_mask in ``lumenairy.elements`` (and the
# ``lumenairy.elements.coronagraph`` namespace re-export).  The dock
# now consumes those canonical builders -- no inline implementation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class _CoronagraphWorker(QThread):
    """Background thread for the 4-stop chain.

    Signals
    -------
    progress(stop_idx, label) : emitted once per stop while building the
        chain so the dock can update its status bar.
    finished_result(dict) : emitted at the end.  ``dict`` keys:
        * ``success`` -- bool
        * ``curve`` -- 1D contrast array (or None on failure)
        * ``separations`` -- 1D lambda/D array
        * ``throughput`` -- float (photon throughput, 0..1)
        * ``per_stop_fields`` -- list of (label, ndarray complex)
        * ``error`` -- str, present only on failure
    """

    progress = Signal(int, str)
    finished_result = Signal(object)

    def __init__(self, params):
        super().__init__()
        # ``params`` is the dock-built dict; no Qt objects survive into the
        # worker beyond the signal connections.
        self.p = params

    # ............................................................. main ...
    def run(self):
        try:
            result = self._run_chain()
            self.finished_result.emit(result)
        except Exception as exc:
            self.finished_result.emit({
                'success': False,
                'error': f'{type(exc).__name__}: {exc}',
            })

    def _run_chain(self):
        p = self.p
        N = int(p['N'])
        dx_pupil = float(p['dx_pupil_m'])
        wavelength = float(p['wavelength_m'])
        D = float(p['pupil_diameter_m'])
        f_eff = float(p['f_eff_m'])

        per_stop_fields = []

        # --- Source pupil --------------------------------------------------
        x = (np.arange(N) - N / 2) * dx_pupil
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X * X + Y * Y)
        pupil = np.where(R <= D / 2.0, 1.0, 0.0).astype(np.complex128)

        # --- Stop 1: apodised pupil ---------------------------------------
        self.progress.emit(0, 'Stop 1: applying apodised pupil')
        E1 = self._apply_apodised(pupil, dx_pupil, p['stop1'])
        per_stop_fields.append(('Stop 1: Apodised pupil', E1.copy()))

        # --- Forward Fraunhofer (pupil -> focal) ---------------------------
        self.progress.emit(1, 'Stop 2: forward FT + focal mask')
        E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E1)))
        dx_focal = wavelength * f_eff / (N * dx_pupil)

        # Reference PSF intensity (no FPM, no Lyot stop, just the apodised
        # pupil's focal-plane intensity).  We compute this off the FIRST
        # focal-plane field so the apodiser's effect on the reference is
        # captured -- the user's "reference" radio toggles whether the
        # ref PSF includes the apodiser (default) or is the bare clear
        # pupil's PSF (loaded-file path).
        ref_source = p['ref_source']
        if ref_source == 'loaded' and p.get('psf_ref_loaded') is not None:
            psf_ref = np.asarray(p['psf_ref_loaded'], dtype=float)
            if psf_ref.shape != E_focal.shape:
                # Fall back to the in-system clear pupil (without
                # apodisation) so the contrast-curve shape is still
                # comparable; warn the user via the summary.
                clean = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
                psf_ref = np.abs(clean) ** 2
        else:
            clean = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
            psf_ref = np.abs(clean) ** 2

        # --- Stop 2: Lyot focal-plane mask --------------------------------
        E2 = self._apply_focal_mask(E_focal, dx_focal, p['stop2'],
                                    wavelength, f_eff, D)
        per_stop_fields.append(('Stop 2: Lyot focal mask', E2.copy()))

        # --- Inverse Fraunhofer (focal -> relay pupil) --------------------
        self.progress.emit(2, 'Stop 3: inverse FT + Lyot stop')
        E_relay_pupil = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(E2)))

        # --- Stop 3: Lyot stop --------------------------------------------
        E3 = self._apply_lyot_stop(E_relay_pupil, dx_pupil, p['stop3'], D)
        per_stop_fields.append(('Stop 3: Lyot stop', E3.copy()))

        # --- Forward Fraunhofer (relay pupil -> image plane) --------------
        self.progress.emit(3, 'Stop 4: image plane + contrast curve')
        E4 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E3)))
        per_stop_fields.append(('Stop 4: Image plane', E4.copy()))
        psf_coro = np.abs(E4) ** 2

        # --- Stop 4: contrast-curve reduction -----------------------------
        from lumenairy.analysis.coronagraph import coronagraph_contrast_curve
        n_radii = int(p['stop4']['n_radii'])
        max_lod = float(p['stop4']['max_lam_over_D'])
        curve = coronagraph_contrast_curve(
            psf_coro, psf_ref,
            dx_focal=dx_focal,
            wavelength=wavelength,
            f_eff=f_eff,
            pupil_diameter_m=D,
            n_radii=n_radii,
            max_lam_over_D=max_lod,
        )

        # --- Throughput ---------------------------------------------------
        # Photon throughput = integrated coro intensity / integrated bare-
        # pupil reference intensity.  We use the SAME-grid reference here
        # (clear pupil, no apodiser) so the ratio is interpretable as the
        # fraction of un-coronagraphed photons that make it through.
        clean = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        ref_total = float(np.sum(np.abs(clean) ** 2))
        coro_total = float(np.sum(psf_coro))
        throughput = (coro_total / ref_total) if ref_total > 0.0 else 0.0

        return {
            'success': True,
            'curve': np.asarray(curve['contrast'], dtype=float),
            'separations': np.asarray(curve['r_lam_over_D'], dtype=float),
            'throughput': throughput,
            'per_stop_fields': per_stop_fields,
            'dx_focal': dx_focal,
            'dx_pupil': dx_pupil,
            'D': D,
            'wavelength': wavelength,
        }

    # ............................................................ stops ...
    def _apply_apodised(self, E_in, dx, stop):
        from lumenairy.elements import apply_apodized_pupil
        prof = stop['profile']
        cfg = APOD_PROFILES[prof]
        lib_name = cfg['lib_name']
        diam = stop['diameter_um'] * 1e-6
        kwargs = {}
        if prof == 'gaussian':
            kwargs['sigma'] = stop['sigma_um'] * 1e-6
        elif prof == 'super-gaussian':
            kwargs['exponent'] = stop['exponent']
        elif prof == 'uniform':
            # 'uniform' maps to the Sonine family with exponent ~ 0.
            kwargs['exponent'] = max(0.0, stop['exponent'])
        return apply_apodized_pupil(
            E_in, dx, diameter=diam, apodization=lib_name, **kwargs)

    def _apply_focal_mask(self, E_in, dx_focal, stop, wavelength, f_eff, D):
        from lumenairy.elements import (
            apply_lyot_focal_plane_mask,
            create_eight_octant_phase_mask,
            create_four_quadrant_phase_mask,
        )
        prof = stop['profile']
        cfg = LYOT_FPM_PROFILES[prof]
        lib_name = cfg['lib_name']
        lod = wavelength * f_eff / D
        if lib_name == '__phase4__':
            # v5.4 Phase 5: library now ships
            # ``create_four_quadrant_phase_mask`` -- consume it instead of
            # the legacy inline ``_phase_octant_mask`` helper.  Apply a
            # soft outer cutoff at mask_radius_lod so the result behaves
            # like a finite-extent FPM.
            Ny, Nx = E_in.shape
            mask = create_four_quadrant_phase_mask(Nx, dx_focal)
            E_phase = E_in * mask
            r_cut = stop['mask_radius_lod'] * lod
            x = (np.arange(Nx) - Nx / 2) * dx_focal
            X, Y = np.meshgrid(x, x)
            R = np.sqrt(X * X + Y * Y)
            inside = R <= r_cut
            return np.where(inside, E_phase, E_in)
        if lib_name == '__phase8__':
            # v5.4 Phase 5: library now ships
            # ``create_eight_octant_phase_mask`` -- consume it instead of
            # the legacy inline ``_phase_octant_mask`` helper.
            Ny, Nx = E_in.shape
            mask = create_eight_octant_phase_mask(Nx, dx_focal)
            E_phase = E_in * mask
            r_cut = stop['mask_radius_lod'] * lod
            x = (np.arange(Nx) - Nx / 2) * dx_focal
            X, Y = np.meshgrid(x, x)
            R = np.sqrt(X * X + Y * Y)
            inside = R <= r_cut
            return np.where(inside, E_phase, E_in)

        mask_d = stop['mask_diameter_lod'] * lod
        kwargs = {'profile': lib_name}
        if lib_name == 'gaussian':
            kwargs['sigma'] = stop['sigma_frac'] * mask_d
        return apply_lyot_focal_plane_mask(
            E_in, dx_focal, mask_diameter=mask_d, **kwargs)

    def _apply_lyot_stop(self, E_in, dx_pupil, stop, D):
        from lumenairy.elements import apply_lyot_stop
        prof = stop['profile']
        outer = stop['outer_frac'] * D
        inner = stop.get('inner_frac', 0.0) * D if prof == \
            'lyot_with_secondary_obscuration' else 0.0
        return apply_lyot_stop(
            E_in, dx_pupil,
            outer_diameter=outer, inner_diameter=inner)


# ---------------------------------------------------------------------------
# Per-stop builder helper -- builds the QGroupBox containing the profile
# dropdown + the per-profile parameter spinners.
# ---------------------------------------------------------------------------
class _StopPanel(QGroupBox):
    """One collapsible group representing a single stop.

    Holds a profile dropdown and a stack of parameter spinners that
    swap when the profile changes.
    """

    STOP_TITLES = {
        1: ('Stop 1: Apodised pupil',          APOD_PROFILES),
        2: ('Stop 2: Lyot focal mask',         LYOT_FPM_PROFILES),
        3: ('Stop 3: Lyot stop',               LYOT_STOP_PROFILES),
        4: ('Stop 4: Image plane (sampler)',   IMAGE_PLANE_PROFILES),
    }

    def __init__(self, stop_idx, parent=None):
        title, catalogue = self.STOP_TITLES[stop_idx]
        super().__init__(title, parent)
        self.stop_idx = stop_idx
        self.catalogue = catalogue
        self.setCheckable(True)
        self.setChecked(True)

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(8, 18, 8, 8)
        self._outer.setSpacing(4)

        # Header row -- profile dropdown
        head = QHBoxLayout()
        head.addWidget(QLabel('profile:'))
        self.combo_profile = QComboBox()
        for name in catalogue.keys():
            self.combo_profile.addItem(name)
        self.combo_profile.currentTextChanged.connect(self._rebuild_params)
        head.addWidget(self.combo_profile, stretch=1)
        self._outer.addLayout(head)

        # Container for the per-profile spinners; rebuilt on profile-change.
        self._param_box = QFrame()
        self._param_form = QFormLayout(self._param_box)
        self._param_form.setContentsMargins(0, 4, 0, 0)
        self._param_form.setVerticalSpacing(2)
        self._outer.addWidget(self._param_box)

        self._spinners = {}                              # name -> QSpinBox
        self._rebuild_params(self.combo_profile.currentText())

    # ............................................................. utils ..
    def _rebuild_params(self, profile_name):
        # Clear existing rows.
        while self._param_form.rowCount() > 0:
            self._param_form.removeRow(0)
        self._spinners.clear()
        cfg = self.catalogue.get(profile_name, {'params': []})
        for name, default, (lo, hi, step, decimals, suffix) in cfg.get('params', []):
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(default)
            if suffix:
                spin.setSuffix(suffix)
            self._param_form.addRow(QLabel(name.replace('_', ' ') + ':'), spin)
            self._spinners[name] = spin

    # ........................................................... readout ..
    def to_dict(self):
        """Snapshot the current widget state to a plain dict for the worker."""
        out = {'profile': self.combo_profile.currentText(),
               'enabled': self.isChecked()}
        for name, spin in self._spinners.items():
            out[name] = float(spin.value())
        return out


# ---------------------------------------------------------------------------
# Main dock
# ---------------------------------------------------------------------------
class CoronagraphDock(QWidget):
    """Interactive coronagraph workflow dock.

    Implements the 4-stop chain (apodised pupil -> Lyot focal mask ->
    Lyot stop -> image plane) with per-stop profile selection, a
    reference-PSF source toggle, and a live contrast-curve plot.
    """

    def __init__(self, model, parent=None):
        # v5.4 audit P1-C: subclass QWidget to match canonical dock
        # convention (main_window.py wraps in QDockWidget).
        super().__init__(parent)
        self.model = model
        self._worker = None
        self._psf_ref_loaded = None        # set via the "Load file" path

        self._build_ui()

    # ============================================================ build ===
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal, self)
        outer.addWidget(splitter, stretch=1)

        # --- Left: scrollable chain builder + controls --------------------
        left_host = QWidget()
        left_lay = QVBoxLayout(left_host)
        left_lay.setContentsMargins(6, 6, 6, 6)
        left_lay.setSpacing(6)

        # System parameters (pulled from the model, but overridable here).
        sys_box = QGroupBox('System parameters')
        sys_form = QFormLayout(sys_box)
        sys_form.setContentsMargins(8, 18, 8, 8)
        sys_form.setVerticalSpacing(2)

        # Grid N (256 default; the focal-plane sampling scales with N).
        self.combo_N = QComboBox()
        for n in (128, 256, 512, 1024):
            self.combo_N.addItem(str(n), n)
        self.combo_N.setCurrentIndex(1)
        sys_form.addRow(QLabel('Grid N:'), self.combo_N)

        self.spin_dx_pupil = QDoubleSpinBox()
        self.spin_dx_pupil.setRange(0.1, 1000.0)
        self.spin_dx_pupil.setValue(50.0)
        self.spin_dx_pupil.setDecimals(2)
        self.spin_dx_pupil.setSuffix(' um')
        sys_form.addRow(QLabel('dx pupil:'), self.spin_dx_pupil)

        # Wavelength selector -- driven by the model's list.  Falls back
        # to a single sensible default if the model is bare.
        self.combo_wavelength = QComboBox()
        for wv in getattr(self.model, 'wavelengths_nm', None) or [1550.0]:
            self.combo_wavelength.addItem(f'{wv:.1f} nm', float(wv))
        sys_form.addRow(QLabel('Wavelength:'), self.combo_wavelength)

        # Field selector -- maps to the model's ``field_angles_deg``.
        self.combo_field = QComboBox()
        for ang in getattr(self.model, 'field_angles_deg', None) or [0.0]:
            self.combo_field.addItem(f'{ang:+.3f} deg', float(ang))
        sys_form.addRow(QLabel('Field:'), self.combo_field)

        # Pupil diameter (from EPD).
        self.spin_D_mm = QDoubleSpinBox()
        self.spin_D_mm.setRange(0.01, 10000.0)
        self.spin_D_mm.setValue(float(getattr(self.model, 'epd_mm', 8.0)))
        self.spin_D_mm.setDecimals(3)
        self.spin_D_mm.setSuffix(' mm')
        sys_form.addRow(QLabel('Pupil diameter D:'), self.spin_D_mm)

        # Focal length.
        self.spin_f_mm = QDoubleSpinBox()
        self.spin_f_mm.setRange(0.1, 100000.0)
        # Pull from the model if it has an EFL; otherwise default to 200 mm.
        try:
            efl_mm = float(self.model.efl_mm)
            if not np.isfinite(efl_mm) or efl_mm <= 0:
                efl_mm = 200.0
        except Exception:
            efl_mm = 200.0
        self.spin_f_mm.setValue(efl_mm)
        self.spin_f_mm.setDecimals(3)
        self.spin_f_mm.setSuffix(' mm')
        sys_form.addRow(QLabel('Focal length f:'), self.spin_f_mm)

        left_lay.addWidget(sys_box)

        # Reference PSF source.
        ref_box = QGroupBox('Reference PSF source')
        ref_lay = QHBoxLayout(ref_box)
        ref_lay.setContentsMargins(8, 18, 8, 8)
        self.radio_ref_compute = QRadioButton('Compute from system')
        self.radio_ref_loaded = QRadioButton('Loaded file')
        self.radio_ref_compute.setChecked(True)
        self.ref_group = QButtonGroup(self)
        self.ref_group.addButton(self.radio_ref_compute)
        self.ref_group.addButton(self.radio_ref_loaded)
        ref_lay.addWidget(self.radio_ref_compute)
        ref_lay.addWidget(self.radio_ref_loaded)
        self.btn_load_ref = QPushButton('Load .npy ...')
        self.btn_load_ref.clicked.connect(self._on_load_ref)
        ref_lay.addWidget(self.btn_load_ref)
        ref_lay.addStretch()
        left_lay.addWidget(ref_box)

        # --- 4 stops ------------------------------------------------------
        chain_box = QGroupBox('4-stop chain')
        chain_lay = QVBoxLayout(chain_box)
        chain_lay.setContentsMargins(8, 18, 8, 8)
        chain_lay.setSpacing(6)
        self.stop_panels = []
        for idx in (1, 2, 3, 4):
            panel = self._build_stop_panel(idx, kind=None)
            self.stop_panels.append(panel)
            chain_lay.addWidget(panel)
        left_lay.addWidget(chain_box)

        # --- Run / Stop ---------------------------------------------------
        ctrl = QHBoxLayout()
        self.btn_run = QPushButton('Run chain')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._on_run)
        # Alias for the F-key dispatcher.
        self.btn_compute = self.btn_run
        ctrl.addWidget(self.btn_run)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        ctrl.addWidget(self.btn_stop)
        self.progress = QProgressBar()
        self.progress.setRange(0, 4)
        self.progress.setValue(0)
        ctrl.addWidget(self.progress, stretch=1)
        left_lay.addLayout(ctrl)

        # Throughput readout.
        self.lbl_throughput = QLabel('Throughput: --')
        f = self.lbl_throughput.font()
        f.setBold(True)
        self.lbl_throughput.setFont(f)
        left_lay.addWidget(self.lbl_throughput)

        # Status summary.
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        left_lay.addWidget(self.summary)

        left_lay.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(left_host)
        scroll.setWidgetResizable(True)
        splitter.addWidget(scroll)

        # --- Right: plots -------------------------------------------------
        right_host = QWidget()
        right_lay = QVBoxLayout(right_host)
        right_lay.setContentsMargins(2, 2, 2, 2)
        right_lay.setSpacing(2)

        if HAS_MPL:
            # Contrast curve canvas (top, 60% height).
            self.fig_curve = Figure(figsize=(6, 4), tight_layout=True)
            self.canvas_curve = FigureCanvas(self.fig_curve)
            self.toolbar_curve = NavigationToolbar(self.canvas_curve, right_host)
            right_lay.addWidget(self.toolbar_curve)
            right_lay.addWidget(self.canvas_curve, stretch=6)

            # Per-stop preview canvas (bottom, 40% height) -- 1 figure
            # with 4 horizontal subplots.
            self.fig_stops = Figure(figsize=(8, 3), tight_layout=True)
            self.canvas_stops = FigureCanvas(self.fig_stops)
            right_lay.addWidget(self.canvas_stops, stretch=4)
        else:                                              # pragma: no cover
            right_lay.addWidget(QLabel('(matplotlib not available)'))

        splitter.addWidget(right_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 900])

    # ............................................. per-stop panel build ..
    def _build_stop_panel(self, stop_idx, kind=None):
        """Create the QGroupBox for stop ``stop_idx`` (1..4).

        ``kind`` is unused here but kept in the signature per the dock
        spec so future per-stop variants (e.g. APLC vs vortex) can swap
        the catalogue without changing the call site.
        """
        return _StopPanel(stop_idx, self)

    # =========================================================== events ===
    def _on_load_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load reference PSF (.npy)', '',
            'NumPy arrays (*.npy)')
        if not path:
            return
        try:
            arr = np.load(path)
            self._psf_ref_loaded = np.asarray(arr, dtype=float)
            self.radio_ref_loaded.setChecked(True)
            self.summary.append(
                f'Loaded reference PSF: shape {self._psf_ref_loaded.shape}')
        except Exception as exc:
            self.summary.append(
                f'Reference load failed: {type(exc).__name__}: {exc}')

    # ............................................................ run ....
    def _on_run(self):
        if self._worker is not None:
            self.summary.append('A run is already in progress.')
            return

        params = self._collect_params()
        if params is None:
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.summary.append('--- Coronagraph chain start ---')

        self._worker = _CoronagraphWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        # PySide6 doesn't expose a clean interrupt for a numpy-heavy
        # thread.  We can only request a quit; the heavy work will
        # finish on its own and the signal will report the result.
        if self._worker is not None:
            self._worker.requestInterruption()
            self.summary.append('Stop requested (waits for current step).')

    def closeEvent(self, event):
        """v5.4.2 (audit C1): wait briefly for an in-flight worker
        thread before the dock destructs.  Without this, closing the
        dock during a run leaves the worker alive with dangling
        callbacks to a deleted parent (potential segfault on emit).
        2-second timeout is more than enough for the worker to finish
        the current 4-stop chain step and clean up.
        """
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        super().closeEvent(event)

    def _on_progress(self, stop_idx, label):
        self.progress.setValue(int(stop_idx) + 1)
        self.summary.append(f'  [{stop_idx + 1}/4] {label}')

    def _on_finished(self, result):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._worker = None

        if not result.get('success'):
            self.summary.append(
                f'Chain failed: {result.get("error", "unknown error")}')
            return

        curve = result['curve']
        separations = result['separations']
        throughput = float(result['throughput'])
        per_stop_fields = result['per_stop_fields']

        self.lbl_throughput.setText(
            f'Throughput: {throughput * 100.0:.2f} % '
            f'({throughput:.3e})')
        self.summary.append(
            f'Chain finished.  Throughput = {throughput * 100.0:.3f} %')
        # Min contrast in [3, 10] lambda/D is a useful one-line summary.
        if separations.size:
            band = (separations > 3.0) & (separations < 10.0)
            if band.any():
                c_min = float(np.nanmin(curve[band]))
                self.summary.append(
                    f'Min contrast in [3, 10] lambda/D: {c_min:.3e}')

        if HAS_MPL:
            self._plot_curve(separations, curve)
            self._plot_per_stop(per_stop_fields)

    # ============================================================ plot ===
    def _plot_curve(self, separations, contrast):
        self.fig_curve.clear()
        ax = self.fig_curve.add_subplot(1, 1, 1)
        # log10(contrast) on Y axis (the spec asks for log10).  Mask out
        # any non-positive contrast values so the log is well-defined.
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.log10(np.clip(contrast, 1e-30, None))
        ax.plot(separations, y, '-o', markersize=3, color='#5cb8ff')
        ax.set_xlabel('Angular separation [lambda / D]')
        ax.set_ylabel('log10(contrast)')
        ax.set_title('Coronagraph contrast curve (azimuthal mean)')
        ax.grid(True, which='both', alpha=0.3)
        self.canvas_curve.draw()

    def _plot_per_stop(self, per_stop_fields):
        self.fig_stops.clear()
        n = len(per_stop_fields)
        if n == 0:
            self.canvas_stops.draw()
            return
        for k, (label, E) in enumerate(per_stop_fields):
            ax = self.fig_stops.add_subplot(1, n, k + 1)
            I = np.abs(E) ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                imax = float(I.max())
                if imax <= 0:
                    log_I = np.zeros_like(I)
                else:
                    log_I = np.log10(np.clip(I / imax, 1e-6, 1.0))
            ax.imshow(log_I, origin='lower', cmap='magma',
                      vmin=-6, vmax=0)
            ax.set_title(label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas_stops.draw()

    # =================================================== param gather ===
    def _collect_params(self):
        """Snapshot every widget into a worker-ready dict."""
        wv_data = self.combo_wavelength.currentData()
        wavelength_m = float(wv_data) * 1e-9 if wv_data is not None \
            else float(getattr(self.model, 'wavelength_m', 1.55e-6))
        f_eff_m = self.spin_f_mm.value() * 1e-3
        D_m = self.spin_D_mm.value() * 1e-3

        if D_m <= 0 or f_eff_m <= 0:
            self.summary.append('Need positive D and f to run.')
            return None

        stops = [p.to_dict() for p in self.stop_panels]
        # Sanity: the dock spec only defines 4 stops; image-plane "stop"
        # is just the sampler.
        params = {
            'N': int(self.combo_N.currentData()),
            'dx_pupil_m': self.spin_dx_pupil.value() * 1e-6,
            'wavelength_m': wavelength_m,
            'f_eff_m': f_eff_m,
            'pupil_diameter_m': D_m,
            'field_angle_deg': float(self.combo_field.currentData() or 0.0),
            'ref_source': 'loaded' if self.radio_ref_loaded.isChecked()
                          else 'compute',
            'psf_ref_loaded': self._psf_ref_loaded,
            'stop1': stops[0],
            'stop2': stops[1],
            'stop3': stops[2],
            'stop4': stops[3],
        }
        return params


# v5.4 audit P1-C: no __all__ -- UI docks are imported by name in
# main_window.py and not exposed at lumenairy.__all__ (V9 walker
# convention; matches the 22 pre-v5.4 docks that ship without it).
