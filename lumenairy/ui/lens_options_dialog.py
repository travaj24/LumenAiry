"""
Lens-function options dialog — tabbed UI for the optional kwargs of
``apply_real_lens``, ``apply_real_lens_traced``, and
``apply_real_lens_maslov``.

Opens from the top-level &Options menu in the main window.  Stores
the user's choices on ``SystemModel.lens_options`` (a dict of dicts,
one per function) so that the next Wave-Optics run picks them up
automatically.

The kwarg registry below is the single source of truth for what the
dialog exposes.  Each entry specifies the widget kind (`bool`, `int`,
`float`, `enum`, `str`), the default, an optional range / enum list,
a short human-readable label, and a tooltip.  Dropping a new kwarg
into the registry is enough to make it appear in the dialog;
``WaveOpticsDock._run`` will pass it through automatically because
it just splats ``**lens_options[func_name]`` onto the lens call.

Author: Andrew Traverso
"""
from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QTabWidget, QWidget, QPushButton, QCheckBox, QSpinBox,
    QDoubleSpinBox, QComboBox, QDialogButtonBox, QGroupBox,
)


# ---------------------------------------------------------------------------
# Kwarg registry: the single source of truth for the dialog
# ---------------------------------------------------------------------------
#
# Per function, a list of (kwarg, widget_kind, default, extra) tuples.
#
#   widget_kind  ::= 'bool' | 'int' | 'float' | 'enum' | 'str'
#   extra        :: dict with any of:
#       'label'   -- human-readable label (default: kwarg name)
#       'tooltip' -- multi-line tooltip
#       'min'     -- numeric min (int/float)
#       'max'     -- numeric max (int/float)
#       'step'    -- numeric step (float)
#       'decimals'-- DoubleSpinBox decimals (float)
#       'suffix'  -- DoubleSpinBox suffix
#       'choices' -- list of strings (enum)
#
# Anything not listed here is intentionally NOT exposed (advanced /
# Newton-internals / Maslov-internals knobs whose defaults we never
# want users to override from the GUI).

LENS_KWARG_REGISTRY = {
    # ------------------------------------------------------------------
    'apply_real_lens': [
        ('bandlimit', 'bool', True,
         {'label': 'ASM bandlimit',
          'tooltip': 'Apply the Matsushima-Shimobaba band-limit '
                     'kernel during the through-glass ASM step.  '
                     'Recommended on for any z > 0.'}),
        ('fresnel', 'bool', False,
         {'label': 'Fresnel transmission losses',
          'tooltip': 'Apply unpolarised Fresnel coefficients at each '
                     'glass interface.  Removes ~4-15% of power per '
                     'BK7 lens at 1.31 µm; turn on for power-budget '
                     'estimates.'}),
        ('absorption', 'bool', False,
         {'label': 'Bulk absorption',
          'tooltip': 'Apply exp(-2π·κ·t/λ₀) bulk absorption from the '
                     'imaginary part of the glass index.  Matters in '
                     'the IR or for very long path lengths.'}),
        ('slant_correction', 'bool', False,
         {'label': 'Slant-correction OPD',
          'tooltip': 'Use the generalised n2·sag/cos(θ_t) - n1·sag/cos(θ_i) '
                     'OPD formula instead of the paraxial (n2-n1)·sag.  '
                     'Useful for strongly off-axis sims; the default '
                     'paraxial form is empirically equal-or-better for '
                     'most setups because the in-glass ASM already '
                     'encodes most obliquity.'}),
        ('seidel_correction', 'bool', False,
         {'label': 'Seidel correction',
          'tooltip': 'Add a ray-trace-derived radial phase correction '
                     'on top of the analytic thin-element model.  ~3-5x '
                     'OPD-RMS improvement on cemented doublets at '
                     'essentially zero cost.  Off by default because it '
                     'can inject ~100 nm fit artifacts on well-corrected '
                     'singlets.'}),
        ('seidel_poly_order', 'int', 6,
         {'label': 'Seidel poly order',
          'min': 4, 'max': 12, 'step': 2,
          'tooltip': 'Highest even power in the radial polynomial fit '
                     'used for Seidel correction.  4 = classical r^4; '
                     '6/8 add 6th and 8th-order spherical terms.  Higher '
                     'is rarely beneficial because the 1-D ray-trace '
                     'sample density limits the fit.'}),
        ('wave_propagator', 'enum', 'asm',
         {'label': 'Through-glass propagator',
          'choices': ['asm', 'sas', 'fresnel', 'rayleigh_sommerfeld'],
          'tooltip': 'Which propagator to use for the in-glass leg '
                     'between surfaces.  ASM is correct at mm-scale; '
                     'the others are exposed for cross-validation and '
                     'pipelines that want a single propagator used '
                     'consistently throughout.  Expert override.'}),
    ],

    # ------------------------------------------------------------------
    'apply_real_lens_traced': [
        ('bandlimit', 'bool', True,
         {'label': 'ASM bandlimit',
          'tooltip': 'Same as apply_real_lens; applied during the '
                     'amplitude pass.'}),
        ('ray_subsample', 'int', 8,
         {'label': 'Ray subsample',
          'min': 1, 'max': 16,
          'tooltip': 'Newton runs on a (N/sub)² coarse grid then '
                     'splines back to N².  1 = exact (slowest), '
                     '4 = older default, 8 = current default with '
                     '<1 nm fidelity loss.'}),
        ('preserve_input_phase', 'bool', True,
         {'label': 'Preserve input phase',
          'tooltip': 'Subtract the analytic lens-phase reference from '
                     'the input phase before adding the ray-traced OPL.  '
                     'Default True is physically correct; turning off '
                     'is an expert debugging move.'}),
        ('tilt_aware_rays', 'bool', False,
         {'label': 'Tilt-aware ray launch',
          'tooltip': 'Launch rays along the local input phase gradient '
                     'rather than parallel to z.  Off by default — has '
                     'a reference-frame mismatch with preserve_input_phase '
                     'on multi-mode inputs (post-DOE diffraction patterns). '
                     'See 3.1.4 changelog.'}),
        ('fast_analytic_phase', 'bool', False,
         {'label': 'Fast analytic phase',
          'tooltip': 'Skip the full ASM-through-glass reference phase '
                     'pass; compute the geometric lens phase analytically '
                     'from per-surface sag.  ~25% speedup with <10 nm '
                     'OPL error on typical refractive prescriptions.'}),
        ('parallel_amp', 'bool', True,
         {'label': 'Parallel amp pass',
          'tooltip': 'Run the apply_real_lens(input) and '
                     'apply_real_lens(plane wave) calls concurrently '
                     'on a thread pool.  ~1.7x speedup; auto-disables '
                     'when free RAM is tight.'}),
        ('inversion_method', 'enum', 'newton',
         {'label': 'Inversion method',
          'choices': ['newton', 'backward_trace'],
          'tooltip': 'newton = forward trace + per-pixel Newton inversion '
                     '(default, fully validated).  backward_trace = direct '
                     'backward ray trace through reversed prescription '
                     '(experimental, ~3x faster, sub-pm single-ray '
                     'agreement).'}),
        ('newton_fit', 'enum', 'polynomial',
         {'label': 'Newton fit',
          'choices': ['polynomial', 'spline'],
          'tooltip': 'polynomial = 2-D Chebyshev tensor product fit '
                     '(default 3.1.7+, ~12x faster on hot loop with '
                     'numba JIT).  spline = pre-3.1.7 SciPy '
                     'RectBivariateSpline; flip back if the polynomial '
                     'fit struggles on high-order freeforms.'}),
        ('newton_poly_order', 'int', 6,
         {'label': 'Newton poly order',
          'min': 2, 'max': 12, 'step': 1,
          'tooltip': 'Order of the Chebyshev tensor-product fit.  Higher '
                     'captures more aberration; lower is faster.  6 is '
                     'the default for typical prescriptions.'}),
        ('on_undersample', 'enum', 'error',
         {'label': 'Undersample policy',
          'choices': ['error', 'warn', 'silent'],
          'tooltip': 'What to do when ray_subsample yields fewer than '
                     'min_coarse_samples_per_aperture across an aperture.  '
                     'Default raises ValueError to prevent silently bad '
                     'results.'}),
        ('wave_propagator', 'enum', 'asm',
         {'label': 'Through-glass propagator',
          'choices': ['asm', 'sas', 'fresnel', 'rayleigh_sommerfeld'],
          'tooltip': 'Forwarded to the inner apply_real_lens calls.  '
                     'ASM is correct; others are research overrides.'}),
    ],

    # ------------------------------------------------------------------
    'apply_real_lens_maslov': [
        ('integration_method', 'enum', 'quadrature',
         {'label': 'Integration method',
          'choices': ['quadrature', 'stationary_phase', 'local'],
          'tooltip': 'quadrature = uniform Tukey quadrature over (v2x, v2y) '
                     '(default, extended-source regime).  '
                     'stationary_phase = closed-form per-pixel evaluation '
                     '(recommended on caustic-near outputs).  '
                     'local = Hessian-oriented local quadrature '
                     '(asymptotic corrections beyond leading stationary '
                     'phase).'}),
        ('poly_order', 'int', 4,
         {'label': 'Chebyshev poly order',
          'min': 2, 'max': 8,
          'tooltip': 'Order of the 4-D Chebyshev tensor-product fit to '
                     'the canonical map (s1(s2,v2), OPD(s2,v2)).  Higher '
                     'captures more aberration at sub-millisecond fit '
                     'cost.'}),
        ('n_v2', 'int', 32,
         {'label': 'v2 quadrature points',
          'min': 8, 'max': 128, 'step': 8,
          'tooltip': 'Number of (v2x, v2y) grid samples for the '
                     'quadrature integral.  Higher = more accurate, '
                     'quadratic in cost.  Only used in quadrature mode.'}),
        ('ray_field_samples', 'int', 16,
         {'label': 'Ray field samples',
          'min': 8, 'max': 64, 'step': 4,
          'tooltip': 'Field-side ray density used to fit the canonical '
                     'map.  16 covers most refractive prescriptions; '
                     'increase for strongly aberrated systems.'}),
        ('ray_pupil_samples', 'int', 16,
         {'label': 'Ray pupil samples',
          'min': 8, 'max': 64, 'step': 4,
          'tooltip': 'Pupil-side ray density used to fit the canonical '
                     'map.'}),
        ('extract_linear_phase', 'bool', True,
         {'label': 'Extract linear phase',
          'tooltip': 'Subtract the linear part of the OPD before '
                     'fitting the Chebyshev polynomial.  Recommended on '
                     'except for debugging.'}),
        ('collimated_input', 'bool', False,
         {'label': 'Collimated input',
          'tooltip': 'Specialise for plane-wave inputs.  Skips '
                     'per-pixel input-phase tilt-extraction.'}),
        ('output_subsample', 'int', 1,
         {'label': 'Output subsample',
          'min': 1, 'max': 8,
          'tooltip': 'Compute the Maslov integral on a (N/sub)² grid '
                     'then bilinearly upsample.  Useful when only the '
                     'envelope matters.'}),
        ('normalize_output', 'enum', 'power',
         {'label': 'Normalisation',
          'choices': ['power', 'amplitude', 'none'],
          'tooltip': 'Output normalisation: power conservation (default), '
                     'amplitude conservation, or no normalisation '
                     '(report raw integrand).'}),
    ],
}


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class LensOptionsDialog(QDialog):
    """Tabbed dialog for the optional kwargs of the real-lens functions.

    Reads / writes ``system_model.lens_options`` (a dict-of-dicts) which
    the Wave-Optics dock then splats onto the actual lens calls.  Library
    defaults are used for any kwarg the user hasn't touched.
    """

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self.setWindowTitle('Lens function options')
        self.setMinimumSize(640, 520)

        # Pull the current state (or initialise from library defaults).
        # Stored as a deep copy so Cancel discards changes.
        existing = getattr(self.sm, 'lens_options', None) or {}
        self._working = {
            fn: dict(existing.get(fn, {}))
            for fn in LENS_KWARG_REGISTRY
        }

        layout = QVBoxLayout(self)

        # -- Top blurb
        blurb = QLabel(
            'Each tab below configures one of the real-lens wave-optics '
            'pipelines.  Settings persist for the current session and '
            'are applied automatically on the next Wave-Optics run.')
        blurb.setWordWrap(True)
        blurb.setStyleSheet('color: #7a94b8; padding: 4px;')
        layout.addWidget(blurb)

        # -- Tabs
        self.tabs = QTabWidget()
        # Each tab maps kwarg -> widget so we can read values back later.
        self._widgets = {}
        for fn_name, kwargs in LENS_KWARG_REGISTRY.items():
            tab = QWidget()
            self._widgets[fn_name] = {}
            self._build_tab(tab, fn_name, kwargs)
            self.tabs.addTab(tab, fn_name)
        layout.addWidget(self.tabs, stretch=1)

        # -- Buttons
        btn_row = QHBoxLayout()
        self.btn_reset_tab = QPushButton('Reset this tab')
        self.btn_reset_tab.setToolTip(
            'Clear overrides for the current tab; library defaults '
            'will be used.')
        self.btn_reset_tab.clicked.connect(self._reset_current_tab)
        btn_row.addWidget(self.btn_reset_tab)

        self.btn_reset_all = QPushButton('Reset all to defaults')
        self.btn_reset_all.setToolTip(
            'Clear overrides for every tab; library defaults will be '
            'used everywhere.')
        self.btn_reset_all.clicked.connect(self._reset_all)
        btn_row.addWidget(self.btn_reset_all)

        btn_row.addStretch()

        bb = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        btn_row.addWidget(bb)
        layout.addLayout(btn_row)

    # -- Tab construction ---------------------------------------------------

    def _build_tab(self, tab, fn_name, kwargs):
        """Populate one tab.  Each kwarg becomes a labelled widget."""
        layout = QVBoxLayout(tab)

        # Header note about library defaults
        hdr = QLabel(
            f'<b>{fn_name}</b><br>'
            f'Empty / default-valued fields use the library default.')
        hdr.setStyleSheet('color: #c0d0e8; padding: 4px;')
        layout.addWidget(hdr)

        # Form
        group = QGroupBox('Options')
        form = QFormLayout(group)
        for kwarg, kind, default, extra in kwargs:
            current = self._working[fn_name].get(kwarg, default)
            widget = self._make_widget(kind, current, extra)
            label = extra.get('label', kwarg)
            tooltip = extra.get('tooltip', '')
            if tooltip:
                widget.setToolTip(tooltip)
            form.addRow(f'{label}:', widget)
            self._widgets[fn_name][kwarg] = (widget, kind, default, extra)
        layout.addWidget(group)
        layout.addStretch()

    def _make_widget(self, kind, value, extra):
        """Map a (kind, value, extra) spec to a Qt widget."""
        if kind == 'bool':
            w = QCheckBox()
            w.setChecked(bool(value))
            return w
        if kind == 'int':
            w = QSpinBox()
            w.setRange(int(extra.get('min', 0)), int(extra.get('max', 1024)))
            if 'step' in extra:
                w.setSingleStep(int(extra['step']))
            w.setValue(int(value))
            return w
        if kind == 'float':
            w = QDoubleSpinBox()
            w.setRange(float(extra.get('min', 0)),
                       float(extra.get('max', 1e9)))
            w.setDecimals(int(extra.get('decimals', 4)))
            if 'step' in extra:
                w.setSingleStep(float(extra['step']))
            if 'suffix' in extra:
                w.setSuffix(' ' + extra['suffix'])
            w.setValue(float(value))
            return w
        if kind == 'enum':
            w = QComboBox()
            choices = extra.get('choices', [])
            w.addItems(choices)
            try:
                w.setCurrentIndex(choices.index(value))
            except ValueError:
                w.setCurrentIndex(0)
            return w
        # 'str' fallback — not currently used but available.
        from PySide6.QtWidgets import QLineEdit
        w = QLineEdit(str(value) if value is not None else '')
        return w

    # -- Read-back ---------------------------------------------------------

    def _read_widget(self, widget, kind, default, extra):
        """Pull the current value from a Qt widget."""
        if kind == 'bool':
            return bool(widget.isChecked())
        if kind == 'int':
            return int(widget.value())
        if kind == 'float':
            return float(widget.value())
        if kind == 'enum':
            return widget.currentText()
        return widget.text()

    def _collect(self):
        """Build the full dict-of-dicts from current widget state."""
        out = {}
        for fn_name, kwarg_map in self._widgets.items():
            d = {}
            for kwarg, (w, kind, default, extra) in kwarg_map.items():
                val = self._read_widget(w, kind, default, extra)
                # Only persist values that differ from the library
                # default — keeps the stored dict minimal and means a
                # caller that omits the kwarg gets the library default
                # automatically.
                if val != default:
                    d[kwarg] = val
            out[fn_name] = d
        return out

    # -- Reset buttons -----------------------------------------------------

    def _reset_current_tab(self):
        idx = self.tabs.currentIndex()
        if idx < 0:
            return
        fn_name = self.tabs.tabText(idx)
        self._working[fn_name] = {}
        # Rebuild the tab in place
        tab = self.tabs.widget(idx)
        # Wipe + re-build
        old_layout = tab.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            QWidget().setLayout(old_layout)  # detach
        self._widgets[fn_name] = {}
        self._build_tab(tab, fn_name, LENS_KWARG_REGISTRY[fn_name])

    def _reset_all(self):
        for i in range(self.tabs.count()):
            fn_name = self.tabs.tabText(i)
            self._working[fn_name] = {}
        # Rebuild every tab
        for i in range(self.tabs.count()):
            fn_name = self.tabs.tabText(i)
            tab = self.tabs.widget(i)
            old_layout = tab.layout()
            if old_layout is not None:
                while old_layout.count():
                    item = old_layout.takeAt(0)
                    w = item.widget()
                    if w is not None:
                        w.deleteLater()
                QWidget().setLayout(old_layout)
            self._widgets[fn_name] = {}
            self._build_tab(tab, fn_name, LENS_KWARG_REGISTRY[fn_name])

    # -- Accept ------------------------------------------------------------

    def accept(self):
        """Persist the user's choices on the SystemModel."""
        self.sm.lens_options = self._collect()
        super().accept()
