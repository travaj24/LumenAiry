"""
Optimizer dock -- variable selection, merit function, optimization control.

# v5.4 (audit P1-F): wire CancellableProgress + Stop button
# v5.4 (audit P1-D): parameter surface expansion for v4.16.0 optimisation framework

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QTextEdit, QCheckBox, QGroupBox, QComboBox,
    QDialog, QDialogButtonBox, QScrollArea, QFormLayout,
    QFileDialog, QLineEdit, QDoubleSpinBox,
)
from PySide6.QtGui import QFont, QColor

import numpy as np

from .model import SystemModel, SurfaceRow
from ..progress import CancellableProgress, is_cancelled


# v5.4 (audit P1-D): canonical scipy / design_optimize method tokens
# surfaced via the Advanced-parameters dropdown.  Order matters --
# QComboBox.addItems lands on index 0 by default.  Per the backward-
# compat note in the prompt, the local geometric path's
# model.run_optimization() hardcoded 'Nelder-Mead' pre-v5.4, so we
# keep it as the initial selection for byte-identical untouched-
# control behaviour.  Users wanting the v4.16.0 default ('L-BFGS-B',
# better for smooth problems with many free vars) pick it explicitly.
ADVANCED_METHODS = (
    'Nelder-Mead', 'L-BFGS-B', 'SLSQP', 'trust-constr', 'trust-ncg',
    'Powell', 'COBYLA',
    'differential_evolution', 'basin_hopping', 'dual_annealing',
    'newton',
)

# Methods that accept hard constraints via scipy.optimize.NonlinearConstraint.
# Mirrors lumenairy.optimize.context._METHODS_SUPPORTING_CONSTRAINTS so a
# users' Constraint editor + method= mismatch surfaces at the dock layer
# (clearer message) before reaching design_optimize().
_CONSTRAINT_METHODS = ('SLSQP', 'trust-constr')

# Methods that accept a Hessian (hess=) kwarg.  Mirrors the dispatch
# branch in lumenairy/optimize/driver.py around line 1156.
_HESS_METHODS = ('trust-ncg', 'trust-constr', 'newton')

# Default wave-propagator names.  Resolved lazily at dock construction
# from lumenairy.optimize.WAVE_PROPAGATOR_REGISTRY so user-registered
# propagators appear in the dropdown alongside the built-ins.
_DEFAULT_WAVE_PROPAGATORS = (
    'real_lens', 'gbd', 'hf', 'hfpi', 'asymptotic',
)


class OptimizeWorker(QThread):
    """Run optimization in a background thread.

    v5.4 (audit P1-D): accepts an ``advanced_kwargs`` dict from the
    dock so the user's dropdown / spinner choices (method, max_iter,
    ...) flow through to model.run_optimization() instead of being
    silently dropped.  Pre-v5.4 the worker only forwarded ``max_iter``
    and hardcoded method='Nelder-Mead' inside the model.
    """
    progress = Signal(int, float)
    finished = Signal(bool, str)
    cancelled = Signal()

    def __init__(self, model, max_iter, advanced_kwargs=None):
        super().__init__()
        self.model = model
        self.max_iter = max_iter
        # v5.4 (audit P1-D): dock-supplied advanced parameter dict.
        # Defaults to empty -- model.run_optimization will then keep
        # its pre-v5.4 Nelder-Mead behaviour.
        self.advanced_kwargs = dict(advanced_kwargs or {})
        # v5.4 (audit P1-F): cancellation flag polled by the scipy
        # callback below.  run_optimization() doesn't take a
        # CancellableProgress so we sentinel via StopIteration in the
        # callback and catch it in run().
        self._cancel_progress = CancellableProgress()
        # v5.24.4 (audit S4-7): the worker runs the optimization with
        # ``apply_result=False`` so it never mutates the shared live model
        # off the GUI thread; it hands the solution vector back here for
        # the dock's finished-handler to apply on the MAIN thread.
        self.result_x = None

    def run(self):
        # v5.4 (audit P1-D): validate dock kwarg combinations BEFORE
        # spawning the scipy run so the user sees an immediate error
        # message instead of a deep-stack KeyError / ValueError from
        # scipy.  The local geometric path runs scipy.minimize
        # directly -- only ``method`` is meaningful here.
        # hess / constraints / state_file / wave_propagator /
        # precision are wave-leg parameters and are silently ignored
        # (the Advanced-group help text documents this); but the
        # combination hess= + non-Hessian-method is genuinely
        # inconsistent and is reported up-front.
        method = self.advanced_kwargs.get('method', 'Nelder-Mead')
        hess = self.advanced_kwargs.get('hess')
        if hess and hess != 'auto' and method not in _HESS_METHODS:
            self.finished.emit(
                False,
                f"hess={hess!r} requires method in {_HESS_METHODS}; "
                f"got method={method!r}.  Either change the method "
                f"or set hess to 'auto'.")
            return
        constraints = self.advanced_kwargs.get('constraints') or ()
        if constraints and method not in _CONSTRAINT_METHODS:
            self.finished.emit(
                False,
                f"constraints= requires method in {_CONSTRAINT_METHODS}; "
                f"got method={method!r}.  Switch to SLSQP / trust-constr "
                f"or clear the constraints table.")
            return

        def cb(it, merit):
            self.progress.emit(it, merit)
            if self._cancel_progress.should_stop:
                # Nelder-Mead's callback path lacks a clean abort
                # contract; raise StopIteration which scipy surfaces
                # via OptimizeResult or raises into the caller.
                raise StopIteration('cancelled by user')
        try:
            # v5.24.4 (audit S4-7): apply_result=False -- the model runs
            # the solve without writing back into the shared live model
            # from this worker thread; it restores itself to x0 and
            # exposes the solution via ``model._last_optimization_x``.
            success, msg = self.model.run_optimization(
                self.max_iter, cb, method=method, apply_result=False)
        except StopIteration:
            self.cancelled.emit()
            self.finished.emit(False, 'Cancelled by user')
            return
        # Carry the solution to the MAIN-thread finished handler.
        self.result_x = list(getattr(self.model, '_last_optimization_x', None)
                             or [])
        self.finished.emit(success, msg)

    @Slot()
    def cancel(self):
        self._cancel_progress.cancel()


class OptimizerDock(QWidget):
    """Optimization control panel."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Variable selection ──
        var_group = QGroupBox('Variables')
        var_layout = QVBoxLayout(var_group)

        self.var_table = QTableWidget(0, 4)
        self.var_table.setHorizontalHeaderLabels(['Surf#', 'Parameter', 'Value', 'Active'])
        self.var_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.var_table.setMaximumHeight(150)
        var_layout.addWidget(self.var_table)

        var_btn_layout = QHBoxLayout()
        btn_add = QPushButton('+ Add Variable')
        btn_add.clicked.connect(self._add_variable)
        btn_clear = QPushButton('Clear All')
        btn_clear.clicked.connect(self._clear_variables)
        var_btn_layout.addWidget(btn_add)
        var_btn_layout.addWidget(btn_clear)
        var_layout.addLayout(var_btn_layout)

        layout.addWidget(var_group)

        # ── Config ──
        config_group = QGroupBox('Wavelengths & Fields')
        config_layout = QVBoxLayout(config_group)

        wv_row = QHBoxLayout()
        wv_row.addWidget(QLabel('Wavelengths (nm):'))
        self.wv_input = QLabel(str(self.sm.wavelength_nm))
        wv_row.addWidget(self.wv_input)
        btn_add_wv = QPushButton('+λ')
        btn_add_wv.setFixedWidth(30)
        btn_add_wv.clicked.connect(self._add_wavelength)
        wv_row.addWidget(btn_add_wv)
        btn_wt_wv = QPushButton('weights...')
        btn_wt_wv.setToolTip(
            'Edit per-wavelength weights (photopic, equal, or custom) '
            'for multi-wavelength merits.')
        btn_wt_wv.clicked.connect(self._edit_wavelength_weights)
        wv_row.addWidget(btn_wt_wv)
        config_layout.addLayout(wv_row)

        field_row = QHBoxLayout()
        field_row.addWidget(QLabel('Fields (deg):'))
        self.field_input = QLabel('0.0')
        field_row.addWidget(self.field_input)
        btn_add_f = QPushButton('+F')
        btn_add_f.setFixedWidth(30)
        btn_add_f.clicked.connect(self._add_field)
        field_row.addWidget(btn_add_f)
        btn_wt_f = QPushButton('weights...')
        btn_wt_f.setToolTip(
            'Edit per-field weights for multi-field merits (axial-heavy '
            'default, uniform, or custom).')
        btn_wt_f.clicked.connect(self._edit_field_weights)
        field_row.addWidget(btn_wt_f)
        config_layout.addLayout(field_row)

        layout.addWidget(config_group)

        # ── Merit function selector ──
        merit_group = QGroupBox('Merit Function')
        merit_layout = QVBoxLayout(merit_group)

        merit_row1 = QHBoxLayout()
        merit_row1.addWidget(QLabel('Geometric:'))
        self.combo_merit_geo = QComboBox()
        self.combo_merit_geo.addItems([
            'RMS Spot (default)',
            'EFL Target',
            'BFL Target',
            'Seidel Spherical',
            'Min Thickness',
            'Max F-Number',
            'Chromatic Focal Shift',
            'Tolerance-aware (robust)',
        ])
        self.combo_merit_geo.setToolTip(
            'Geometric merit (fast, ray-trace based).\n'
            '  Chromatic Focal Shift: minimises EFL variation across the '
            'current wavelength list.\n'
            '  Tolerance-aware: wraps any merit in a Monte-Carlo mean so '
            'the optimum is robust to manufacturing scatter.')
        merit_row1.addWidget(self.combo_merit_geo)
        merit_layout.addLayout(merit_row1)

        merit_row2 = QHBoxLayout()
        merit_row2.addWidget(QLabel('Wave (slow):'))
        self.combo_merit_wave = QComboBox()
        self.combo_merit_wave.addItems([
            'None',
            'Strehl > target',
            'RMS Wavefront < target',
            'Match Ideal Thin Lens',
            'Match Ideal System (full)',
            'Zernike Coefficients',
        ])
        self.combo_merit_wave.setToolTip(
            'Wave-optics merit (slower, runs apply_real_lens + through-focus).\n'
            '  Match Ideal System: drives the full radiation pattern and '
            'relative phase toward a reference thin-lens system\n'
            '  (field-overlap metric, invariant to global phase).')
        merit_row2.addWidget(self.combo_merit_wave)
        merit_layout.addLayout(merit_row2)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel('Target value:'))
        from PySide6.QtWidgets import QDoubleSpinBox
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(-1e6, 1e6)
        self.spin_target.setDecimals(4)
        self.spin_target.setValue(100.0)
        self.spin_target.setToolTip(
            'Target value for the selected merit (EFL in mm, Strehl 0-1, etc.)')
        target_row.addWidget(self.spin_target)
        target_row.addWidget(QLabel('mm / ratio'))
        merit_layout.addLayout(target_row)

        layout.addWidget(merit_group)

        # ── Optimization control ──
        opt_group = QGroupBox('Optimization')
        opt_layout = QVBoxLayout(opt_group)

        # ── Compute backend (3.6) ──
        # JAX is a *backend* choice; group it with iterations rather
        # than placing it between two run-config rows.
        try:
            import jax  # noqa
            _jax_ok = True
        except Exception:
            _jax_ok = False
        backend_group = QGroupBox('Compute backend')
        backend_layout = QFormLayout(backend_group)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 5000)
        self.spin_iter.setValue(200)
        backend_layout.addRow('Max iterations:', self.spin_iter)
        self.chk_jax = QCheckBox(
            'Use JAX wave propagator (faster gradients)')
        self.chk_jax.setChecked(False)
        self.chk_jax.setEnabled(_jax_ok)
        self.chk_jax.setToolTip(
            'Route the wave leg through apply_real_lens_traced_jax '
            'and let design_optimize use jax.grad-derived analytic '
            'Jacobians for JAX-aware merit terms.  Falls back to FD '
            'for non-JAX merit terms.  Significant speedup on systems '
            'with many free variables.'
            + ('' if _jax_ok else
               '\n\n(JAX not detected — install via '
               'pip install jax jaxlib)'))
        backend_layout.addRow(self.chk_jax)
        opt_layout.addWidget(backend_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        opt_layout.addWidget(self.progress_bar)

        # ── Run buttons (3.6) ──
        # Promote a single primary action; the other two move under a
        # disclosure so novices aren't forced to choose between three
        # similar-looking buttons before pressing Run.
        primary_row = QHBoxLayout()
        self.btn_optimize = QPushButton('▶ Optimize')
        self.btn_optimize.setObjectName('run_button')
        self.btn_optimize.setToolTip(
            'Run the local Nelder-Mead geometric optimizer.  When a '
            'wave merit is selected the run automatically reroutes '
            'to the hybrid wave/ray design_optimize engine.')
        self.btn_optimize.clicked.connect(self._start_optimize)
        # Alias so F-key dispatcher / tests can always find btn_run.
        self.btn_run = self.btn_optimize
        primary_row.addWidget(self.btn_optimize)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_optimize)
        primary_row.addWidget(self.btn_stop)
        opt_layout.addLayout(primary_row)

        # Disclosure for advanced run modes.
        adv_row = QHBoxLayout()
        self.btn_global = QPushButton('Global Search…')
        self.btn_global.setToolTip(
            'Random-restart optimization (finds different lens forms)')
        self.btn_global.clicked.connect(self._start_global)
        adv_row.addWidget(self.btn_global)
        self.btn_wave = QPushButton('Wave Optimize…')
        self.btn_wave.setToolTip(
            'Force the hybrid wave/ray engine even when only '
            'geometric merits are selected.  Use for Strehl / RMS '
            'wavefront / Zernike-coefficient targets.')
        self.btn_wave.clicked.connect(self._start_wave_optimize)
        adv_row.addWidget(self.btn_wave)
        adv_row.addStretch()
        opt_layout.addLayout(adv_row)

        # v5.4 (audit P1-D): "Advanced parameters" collapsible group
        # surfaces the 8 dock-relevant design_optimize() kwargs that
        # were hardcoded pre-v5.4.  Sits beneath the run-button row so
        # casual users still see the run controls without scrolling;
        # power users tick the checkable group to expose method /
        # constraints / state_file / hess / wave_propagator /
        # precision / multi-objective / max_iter.  Built in
        # _build_advanced_group() to keep __init__ readable.
        self._build_advanced_group(opt_layout)

        layout.addWidget(opt_group)

        # ── Log ──
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setFont(QFont('Consolas', 10))
        self.log.setStyleSheet("QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.log)

        # ── Convergence plot (merit vs iteration) ──
        # Live visualisation of the optimization trajectory -- useful
        # to spot stagnation/divergence without parsing the log.
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas)
            self._conv_fig = Figure(figsize=(4, 1.6), facecolor='#0a0c10',
                                    tight_layout=True)
            self._conv_ax = self._conv_fig.add_subplot(111)
            self._conv_ax.set_facecolor('#0a0c10')
            for spine in self._conv_ax.spines.values():
                spine.set_color('#334054')
            self._conv_ax.tick_params(colors='#7a94b8', labelsize=8)
            self._conv_ax.set_xlabel('iteration', color='#7a94b8', fontsize=8)
            self._conv_ax.set_ylabel('merit', color='#7a94b8', fontsize=8)
            self._conv_ax.set_yscale('log')
            self._conv_line, = self._conv_ax.plot([], [], '-o',
                                                   color='#5cb8ff',
                                                   markersize=3, linewidth=1)
            self._conv_canvas = FigureCanvas(self._conv_fig)
            self._conv_canvas.setFixedHeight(130)
            layout.addWidget(self._conv_canvas)
            self._conv_history = []
        except Exception:
            self._conv_fig = None
            self._conv_canvas = None
            self._conv_history = None

        layout.addStretch()

        self.sm.system_changed.connect(self._refresh_variables)

    def _add_variable(self):
        """Open a single grid dialog to tick every (element, parameter)
        pair that should be an optimization variable.  Replaces the
        old two-popup dance."""
        dlg = _VariableGridDialog(self.sm, self)
        if dlg.exec() != QDialog.Accepted:
            return
        # Replace the current variable list with the ticked set.
        new_vars = dlg.checked_variables()
        self.sm.opt_variables = new_vars
        self._refresh_variables()
        self.log.append(
            f'{len(new_vars)} variable(s) selected: '
            + ', '.join(f'E{e}.S{s}.{f}' for (e, s, f) in new_vars))

    def _clear_variables(self):
        self.sm.opt_variables.clear()
        self._refresh_variables()

    def _refresh_variables(self):
        self.var_table.setRowCount(len(self.sm.opt_variables))
        for i, (elem_idx, surf_idx, field) in enumerate(self.sm.opt_variables):
            elem = self.sm.elements[elem_idx] if elem_idx < len(self.sm.elements) else None
            val = '?'
            if elem and field == 'distance':
                val = f'{elem.distance_mm:.4g}'
            elif elem and surf_idx < len(elem.surfaces):
                val = f'{getattr(elem.surfaces[surf_idx], field, 0):.4g}'
            self.var_table.setItem(i, 0, QTableWidgetItem(f'E{elem_idx}'))
            self.var_table.setItem(i, 1, QTableWidgetItem(f'S{surf_idx}.{field}'))
            self.var_table.setItem(i, 2, QTableWidgetItem(val))
            self.var_table.setItem(i, 3, QTableWidgetItem('OK'))

        self.wv_input.setText(', '.join(f'{w:.1f}' for w in self.sm.wavelengths_nm))
        self.field_input.setText(', '.join(f'{f:.1f}' for f in self.sm.field_angles_deg))

    def _add_wavelength(self):
        from PySide6.QtWidgets import QInputDialog
        wv, ok = QInputDialog.getDouble(self, 'Add Wavelength', 'Wavelength (nm):',
                                         550.0, 200, 20000, 1)
        if ok:
            if wv not in self.sm.wavelengths_nm:
                self.sm.wavelengths_nm.append(wv)
                self._refresh_variables()

    def _add_field(self):
        from PySide6.QtWidgets import QInputDialog
        fa, ok = QInputDialog.getDouble(self, 'Add Field', 'Field angle (deg):',
                                         1.0, -45, 45, 2)
        if ok:
            if fa not in self.sm.field_angles_deg:
                self.sm.field_angles_deg.append(fa)
                self._refresh_variables()

    def _edit_wavelength_weights(self):
        dlg = _WeightsDialog(
            'Wavelength weights',
            [f'{w:.1f} nm' for w in self.sm.wavelengths_nm],
            getattr(self.sm, 'wavelength_weights', None),
            presets={
                'uniform': [1.0] * len(self.sm.wavelengths_nm),
                'photopic 555 nm': _photopic_weights(
                    self.sm.wavelengths_nm),
            }, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.sm.wavelength_weights = dlg.weights()
            self.log.append(
                f'Wavelength weights: '
                f'{", ".join(f"{w:.3f}" for w in self.sm.wavelength_weights)}')

    def _edit_field_weights(self):
        dlg = _WeightsDialog(
            'Field weights',
            [f'{f:.2f} deg' for f in self.sm.field_angles_deg],
            getattr(self.sm, 'field_weights', None),
            presets={
                'uniform': [1.0] * len(self.sm.field_angles_deg),
                'axial-heavy (cos^4)': _axial_heavy_weights(
                    self.sm.field_angles_deg),
            }, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.sm.field_weights = dlg.weights()
            self.log.append(
                f'Field weights: '
                f'{", ".join(f"{w:.3f}" for w in self.sm.field_weights)}')

    def _apply_merit_type(self):
        """Set the model's geometric merit type from the UI combo."""
        geo_map = {
            0: 'rms_spot',
            1: 'efl_target',
            2: 'bfl_target',
            3: 'seidel_spherical',
            4: 'min_thickness',
            5: 'max_fnumber',
        }
        self.sm.geo_merit_type = geo_map.get(
            self.combo_merit_geo.currentIndex(), 'rms_spot')
        self.sm.geo_merit_target = self.spin_target.value()

    # ----------------------------------------------------------------
    # v5.4 (audit P1-D): Advanced-parameters group.
    # ----------------------------------------------------------------

    def _build_advanced_group(self, parent_layout):
        """Construct the collapsible Advanced-parameters group.

        v5.4 (audit P1-D): surfaces 8 design_optimize() kwargs that
        were hardcoded pre-v5.4.  Children of the group are:

        * combo_method   -- method dropdown (default Nelder-Mead for
                            backward compatibility with pre-v5.4
                            local geometric path; the wave path
                            falls back to L-BFGS-B when the user
                            leaves the dropdown on Nelder-Mead AND
                            picks a wave merit, since Nelder-Mead is
                            a poor fit for the smooth wave-leg merit
                            landscape).
        * spin_max_iter_adv -- max_iter override (mirrors the
                            existing Compute-backend spin_iter; we
                            still read spin_iter as the fallback so
                            the dock keeps a single visible
                            iteration knob for casual users).
        * combo_hess     -- hess dropdown.  Auto disables the row
                            when method is not in _HESS_METHODS.
        * combo_wp       -- wave_propagator dropdown.  Populated
                            lazily from the registry so user-
                            registered propagators appear.
        * combo_precision -- 'double' / 'single' dropdown.
        * chk_mo + spin_n_gen + spin_pop -- multi-objective Pareto
                            checkbox + NSGA-II generation / pop
                            spinners.  Disabled when PYMOO_AVAILABLE
                            is False.
        * edit_state_file + btn_resume + btn_save -- checkpoint
                            picker.
        * constraints_editor -- _ConstraintsEditor (QTableWidget +
                            Add / Remove buttons).
        """
        self._adv_group = QGroupBox('Advanced parameters')
        self._adv_group.setCheckable(True)
        self._adv_group.setChecked(False)
        self._adv_group.setToolTip(
            'Surfaces the v4.16.0 design_optimize() parameter '
            'surface.  Untick to use the pre-v5.4 defaults '
            '(Nelder-Mead for the local geometric path, L-BFGS-B for '
            'the wave-optimize path, no constraints, no checkpoint).')
        adv_layout = QFormLayout(self._adv_group)

        # --- method dropdown ---
        self.combo_method = QComboBox()
        self.combo_method.addItems(ADVANCED_METHODS)
        # Pre-v5.4 hardcoded Nelder-Mead in model.run_optimization;
        # keep that as the initial dropdown selection so users that
        # don't touch the control see byte-identical behaviour.
        # 'L-BFGS-B' is the v4.16.0 design_optimize default and
        # generally a better choice -- power users pick it
        # explicitly.  See module-level ADVANCED_METHODS docstring.
        self.combo_method.setCurrentText('Nelder-Mead')
        self.combo_method.setToolTip(
            'Optimization algorithm.  Nelder-Mead (default, pre-v5.4 '
            'behaviour): derivative-free, slow but robust.  '
            'L-BFGS-B: gradient-based, fast for smooth merits.  '
            'SLSQP / trust-constr: support hard constraints.  '
            'differential_evolution / basin_hopping / dual_annealing: '
            'global stochastic.  newton: FD-Hessian Newton (small N '
            'problems).')
        self.combo_method.currentTextChanged.connect(
            self._on_method_changed)
        adv_layout.addRow('method:', self.combo_method)

        # --- max_iter override ---
        # Mirrors the existing Compute-backend max-iter spinner.  We
        # surface a second copy here for proximity to the other
        # Advanced controls; when the group is unchecked we read
        # spin_iter (the original) instead.
        self.spin_max_iter_adv = QSpinBox()
        self.spin_max_iter_adv.setRange(10, 50000)
        self.spin_max_iter_adv.setValue(self.spin_iter.value())
        self.spin_max_iter_adv.setToolTip(
            'Maximum scipy iterations / generations.  Overrides the '
            'Compute-backend spinner when this Advanced group is '
            'expanded.  Unchecked: the original spinner wins.')
        adv_layout.addRow('max_iter:', self.spin_max_iter_adv)

        # --- hess dropdown ---
        self.combo_hess = QComboBox()
        self.combo_hess.addItems(['auto', '2-point', '3-point', 'cs'])
        self.combo_hess.setCurrentText('auto')
        self.combo_hess.setToolTip(
            'Hessian estimator.  Only honoured for method in '
            f'{_HESS_METHODS}.  auto: design_optimize chooses '
            '(default).  2-point / 3-point / cs: scipy '
            'FiniteDifferenceHessian schemes.')
        adv_layout.addRow('hess:', self.combo_hess)

        # --- wave_propagator dropdown ---
        self.combo_wp = QComboBox()
        try:
            from lumenairy.optimize import WAVE_PROPAGATOR_REGISTRY
            wp_names = sorted(WAVE_PROPAGATOR_REGISTRY.keys())
            if not wp_names:
                wp_names = list(_DEFAULT_WAVE_PROPAGATORS)
        except Exception:
            wp_names = list(_DEFAULT_WAVE_PROPAGATORS)
        self.combo_wp.addItems(wp_names)
        self.combo_wp.setCurrentText(
            'real_lens' if 'real_lens' in wp_names else wp_names[0])
        self.combo_wp.setToolTip(
            'Wave-leg propagator (Wave Optimize path only).  '
            'real_lens: default lens-then-Fresnel.  gbd: Gaussian '
            'beam decomposition.  hf / hfpi: Huygens-Fresnel (with '
            'phase integral).  asymptotic: canonical-polynomial '
            'modal asymptotic.')
        adv_layout.addRow('wave_propagator:', self.combo_wp)

        # --- precision dropdown ---
        self.combo_precision = QComboBox()
        self.combo_precision.addItems(['double', 'single'])
        self.combo_precision.setCurrentText('double')
        self.combo_precision.setToolTip(
            'Complex-array precision.  double (default): complex128, '
            'best accuracy.  single: complex64, ~2x FFT throughput '
            'and ~2x memory headroom; ~80 dB cumulative dynamic-'
            'range noise floor.')
        adv_layout.addRow('precision:', self.combo_precision)

        # --- multi-objective Pareto row ---
        try:
            from lumenairy.optimize import PYMOO_AVAILABLE
        except Exception:
            PYMOO_AVAILABLE = False
        mo_row = QHBoxLayout()
        self.chk_mo = QCheckBox('Multi-objective (NSGA-II)')
        self.chk_mo.setChecked(False)
        self.chk_mo.setEnabled(bool(PYMOO_AVAILABLE))
        self.chk_mo.setToolTip(
            'Route through design_optimize_multi_objective (pymoo '
            'NSGA-II).  Wave Optimize path only.'
            + ('' if PYMOO_AVAILABLE else
               '\n\n(pymoo not detected -- '
               'pip install lumenairy[multi_objective])'))
        mo_row.addWidget(self.chk_mo)
        mo_row.addWidget(QLabel('n_gen:'))
        self.spin_n_gen = QSpinBox()
        self.spin_n_gen.setRange(1, 10000)
        self.spin_n_gen.setValue(100)
        self.spin_n_gen.setEnabled(False)
        mo_row.addWidget(self.spin_n_gen)
        mo_row.addWidget(QLabel('pop:'))
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(4, 10000)
        self.spin_pop.setValue(50)
        self.spin_pop.setEnabled(False)
        mo_row.addWidget(self.spin_pop)
        mo_row.addStretch()
        self.chk_mo.toggled.connect(self.spin_n_gen.setEnabled)
        self.chk_mo.toggled.connect(self.spin_pop.setEnabled)
        adv_layout.addRow('multi-objective:', mo_row)

        # --- state_file row ---
        state_row = QHBoxLayout()
        self.edit_state_file = QLineEdit()
        self.edit_state_file.setPlaceholderText(
            '(no checkpoint)')
        self.edit_state_file.setToolTip(
            'JSON file used to checkpoint and resume optimisation. '
            'Pre-existing files are loaded on the next run; new '
            'runs create the file on first eval.')
        state_row.addWidget(self.edit_state_file, stretch=1)
        btn_resume = QPushButton('Resume from...')
        btn_resume.setToolTip('Pick an existing checkpoint JSON to '
                              'resume from.')
        btn_resume.clicked.connect(self._pick_state_file_resume)
        btn_save = QPushButton('Save to...')
        btn_save.setToolTip('Pick a destination JSON for periodic '
                            'state saves.')
        btn_save.clicked.connect(self._pick_state_file_save)
        state_row.addWidget(btn_resume)
        state_row.addWidget(btn_save)
        adv_layout.addRow('state_file:', state_row)

        # --- constraints editor (sub-group) ---
        self.constraints_editor = _ConstraintsEditor(self)
        adv_layout.addRow(self.constraints_editor)

        # Initial method-driven enable / disable sweep.
        self._on_method_changed(self.combo_method.currentText())

        parent_layout.addWidget(self._adv_group)

    def _on_method_changed(self, method):
        """Enable / disable hess row + constraint editor based on
        method compatibility.  Mirrors design_optimize's dispatch
        rules so the user sees grayed-out controls instead of a
        downstream ValueError.
        """
        # hess is only honoured by trust-* and the 'newton' alias.
        self.combo_hess.setEnabled(method in _HESS_METHODS)
        if method not in _HESS_METHODS:
            self.combo_hess.setCurrentText('auto')
        # Constraints only supported by SLSQP / trust-constr.
        if hasattr(self, 'constraints_editor'):
            self.constraints_editor.setEnabled(
                method in _CONSTRAINT_METHODS)
            if method not in _CONSTRAINT_METHODS:
                self.constraints_editor.setToolTip(
                    f'Constraints disabled: method={method!r} does '
                    f'not support hard constraints.  Switch to '
                    f'{list(_CONSTRAINT_METHODS)}.')
            else:
                self.constraints_editor.setToolTip('')

    def _pick_state_file_resume(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Resume from checkpoint', '',
            'JSON files (*.json);;All files (*.*)')
        if path:
            self.edit_state_file.setText(path)

    def _pick_state_file_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save checkpoint to', 'optimize_state.json',
            'JSON files (*.json);;All files (*.*)')
        if path:
            self.edit_state_file.setText(path)

    def _collect_advanced_kwargs(self):
        """Gather Advanced-group widget values into a dict.

        Returns an empty dict if the Advanced group is unchecked --
        the worker then preserves the pre-v5.4 hardcoded behaviour
        (Nelder-Mead in the local path, L-BFGS-B + no extras in the
        wave path).
        """
        if not getattr(self, '_adv_group', None):
            return {}
        if not self._adv_group.isChecked():
            return {}
        kwargs = {
            'method': self.combo_method.currentText(),
            'max_iter': int(self.spin_max_iter_adv.value()),
            'hess': self.combo_hess.currentText(),
            'wave_propagator': self.combo_wp.currentText(),
            'precision': self.combo_precision.currentText(),
            'multi_objective': bool(self.chk_mo.isChecked()),
            'n_generations': int(self.spin_n_gen.value()),
            'pop_size': int(self.spin_pop.value()),
        }
        state = self.edit_state_file.text().strip()
        if state:
            kwargs['state_file'] = state
        constraints = self.constraints_editor.to_constraints()
        if constraints:
            kwargs['constraints'] = constraints
        return kwargs

    def _start_optimize(self):
        if not self.sm.opt_variables:
            self.log.append('No variables defined -- add variables first.')
            return

        # Guard: if the user picked a wave-optics merit but is about to
        # fire the geometric Local Optimize path, redirect to the Wave
        # Optimize path where wave merits are actually honoured.
        if self.combo_merit_wave.currentIndex() != 0:
            self.log.append(
                'Wave merit selected -- routing through Wave Optimize '
                '(Local uses geometric merits only).')
            self._start_wave_optimize()
            return

        self._apply_merit_type()
        self.btn_optimize.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)

        initial_merit = self.sm.merit_function(self.sm.get_variable_values())
        self.log.append(f'Starting optimization ({len(self.sm.opt_variables)} variables)')
        self.log.append(f'Initial merit: {initial_merit*1e6:.3f} µm')
        self._reset_convergence()
        self._append_convergence(0, initial_merit)

        # v5.4 (audit P1-D): forward the Advanced-parameters dock
        # selections (method / max_iter override / hess / constraints)
        # to the worker.  Empty dict when the group is unchecked --
        # OptimizeWorker then preserves the pre-v5.4 Nelder-Mead path.
        adv = self._collect_advanced_kwargs()
        max_iter = adv.pop('max_iter', self.spin_iter.value())
        self._worker = OptimizeWorker(self.sm, max_iter, advanced_kwargs=adv)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        # v5.4 (audit P1-F): also reset UI on cooperative cancel.
        self._worker.cancelled.connect(
            lambda: self._on_finished(False, 'Cancelled by user'))
        self._worker.start()

    def _stop_optimize(self):
        # v5.4 (audit P1-F): cooperative cancellation -- workers poll
        # CancellableProgress.should_stop and return partial results.
        # The finished/cancelled signal handlers re-enable the UI.
        if self._worker and self._worker.isRunning():
            try:
                self._worker.cancel()
            except AttributeError:
                # Defensive: legacy worker without cancel() -- fall
                # back to terminate().  Should not happen post-v5.4.
                self._worker.terminate()
                self._on_finished(False, 'Stopped by user')
            self.log.append('Cancellation requested -- waiting for '
                            'current iteration to finish...')

    def _on_progress(self, iteration, merit):
        self.log.append(f'  iter {iteration}: merit = {merit*1e6:.4f} µm')
        self._append_convergence(iteration, merit)

    def _append_convergence(self, iteration, merit):
        if self._conv_canvas is None or self._conv_history is None:
            return
        try:
            self._conv_history.append((int(iteration), float(abs(merit))))
            xs = [p[0] for p in self._conv_history]
            ys = [p[1] for p in self._conv_history]
            self._conv_line.set_data(xs, ys)
            self._conv_ax.relim()
            self._conv_ax.autoscale_view()
            self._conv_canvas.draw_idle()
        except Exception:
            pass

    def _reset_convergence(self):
        if self._conv_history is None:
            return
        self._conv_history.clear()
        if self._conv_canvas is not None:
            try:
                self._conv_line.set_data([], [])
                self._conv_canvas.draw_idle()
            except Exception:
                pass

    def _on_finished(self, success, msg):
        # v5.24.4 (audit S4-7): the background OptimizeWorker no longer
        # writes its solution into the live model off-thread -- it restored
        # the model to its pre-run state and exposed the solution vector on
        # ``worker.result_x``.  Apply it HERE, on the GUI thread, so
        # self.elements is mutated and the rebuild signal is emitted from
        # the main thread only.  Workers without a ``result_x`` (the global
        # search, cancel/failure paths) leave the model untouched.
        worker = self._worker
        if success and worker is not None:
            result_x = getattr(worker, 'result_x', None)
            if result_x:
                self.sm.set_variable_values(result_x)
                self.sm.system_changed.emit()
        self.btn_optimize.setEnabled(True)
        self.btn_global.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        status = '✓ Done' if success else '✗ Failed'
        self.log.append(f'{status}: {msg}')
        self._refresh_variables()
        self._worker = None

    def _start_global(self):
        """Global search: random restarts around the current design."""
        if not self.sm.opt_variables:
            self.log.append('No variables defined -- add variables first.')
            return

        self._apply_merit_type()
        self.btn_optimize.setEnabled(False)
        self.btn_global.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)

        x0 = self.sm.get_variable_values()
        self.log.append(f'Global search ({len(x0)} variables, 20 restarts)')

        self._worker = GlobalSearchWorker(self.sm, self.spin_iter.value(), 20)
        self._worker.progress.connect(self._on_global_progress)
        self._worker.finished.connect(self._on_finished)
        # v5.4 (audit P1-F): map cancel signal to the same UI reset.
        self._worker.cancelled.connect(
            lambda: self._on_finished(False, 'Cancelled by user'))
        self._worker.start()

    def _on_global_progress(self, restart, merit):
        self.log.append(f'  restart {restart}: best merit = {merit*1e6:.4f} um')

    def _start_wave_optimize(self):
        """Launch the hybrid wave/ray optimizer from optimize.py."""
        if not self.sm.opt_variables:
            self.log.append('No variables defined -- add variables first.')
            return
        self.log.append('Building hybrid wave/ray optimizer...')
        try:
            from lumenairy.optimize import (
                DesignParameterization, design_optimize,
                FocalLengthMerit, BackFocalLengthMerit,
                SphericalSeidelMerit, StrehlMerit,
                RMSWavefrontMerit, MatchIdealThinLensMerit,
                MatchIdealSystemMerit,
                MinThicknessMerit, MaxFNumberMerit,
                ZernikeCoefficientMerit,
                ChromaticFocalShiftMerit, ToleranceAwareMerit,
            )

            pres = self.sm.to_prescription()
            # The prescription's ``surfaces`` list is a flattened sequence
            # of refracting surfaces across ALL lens elements (no Source
            # or Detector).  To map a UI (elem_idx, surf_idx, field)
            # triple to a prescription index we need the *absolute*
            # surface position in that flat list, not the per-element
            # surf_idx.  Same for ``thicknesses``: one entry per gap
            # between prescription surfaces (glass thicknesses + air
            # gaps).  Build the forward map from the current element
            # list.
            flat_surf_map = {}   # (elem_idx, surf_idx) -> flat index
            thickness_map = {}   # elem_idx -> air-gap thickness index
            # v5.24.x (audit S4-6): (elem_idx, surf_idx) -> internal-gap
            # thickness index.  The flattened ``surfaces`` dict emitted by
            # ``to_prescription`` has NO ``thickness`` key, so a surface
            # thickness variable must be routed to the top-level
            # ``thicknesses`` slot; a ('surfaces', fs, 'thickness') path
            # otherwise KeyErrors when DesignParameterization reads x0.
            surf_thk_map = {}
            flat_surf = 0
            flat_thk = 0
            for ei, elem in enumerate(self.sm.elements):
                if elem.elem_type in ('Source', 'Detector'):
                    continue
                # The air gap that precedes this element (distance_mm)
                # is the thickness between the previous surface and
                # this one in the flattened prescription.
                if flat_surf > 0:
                    thickness_map[ei] = flat_thk
                    flat_thk += 1   # consume the air-gap slot
                for si in range(len(elem.surfaces)):
                    flat_surf_map[(ei, si)] = flat_surf
                    flat_surf += 1
                    # Internal thicknesses on all-but-last surface.
                    if si < len(elem.surfaces) - 1:
                        surf_thk_map[(ei, si)] = flat_thk
                        flat_thk += 1

            # v5.24.x (audit S4-6): a LAST-surface ``thickness`` is the air
            # gap to the FOLLOWING lens element -- the same slot that
            # element's ``distance`` occupies.  Return its thickness index,
            # or None at the tail (the gap to the detector is not a legacy
            # ``thicknesses`` slot).
            def _next_air_gap_idx(from_elem):
                for ej in range(from_elem + 1, len(self.sm.elements)):
                    if self.sm.elements[ej].elem_type in (
                            'Source', 'Detector'):
                        continue
                    return thickness_map.get(ej)
                return None

            # Surface-dict fields the legacy wave-leg prescription carries
            # (see ModelState.to_prescription -- radius/conic and their
            # anamorphic partners).  ``thickness`` routes to ``thicknesses``
            # below; anything else (``glass``, ``semi_diameter``) has no
            # home there and is skipped rather than emitting a path that
            # KeyErrors when DesignParameterization reads x0.
            wave_surf_fields = ('radius', 'conic', 'radius_y', 'conic_y')

            free_vars = []
            bounds_list = []
            seen_paths = set()   # S4-6: de-dup thickness/distance clashes
            for i, (elem_idx, surf_idx, field) in enumerate(self.sm.opt_variables):
                if field == 'distance':
                    tk_idx = thickness_map.get(elem_idx)
                    if tk_idx is None:
                        self.log.append(
                            f'  (skipped distance for element {elem_idx}: '
                            f'first/source element has no preceding gap)')
                        continue
                    path = ('thicknesses', tk_idx)
                elif field == 'thickness':
                    # v5.24.x (audit S4-6): route a surface thickness to its
                    # top-level ``thicknesses`` slot -- the internal gap for
                    # a non-last surface, else the air gap to the next
                    # element.  The surface dict has no ``thickness`` key.
                    tk_idx = surf_thk_map.get((elem_idx, surf_idx))
                    if tk_idx is None:
                        tk_idx = _next_air_gap_idx(elem_idx)
                    if tk_idx is None:
                        self.log.append(
                            f'  (skipped thickness for element {elem_idx} '
                            f'surface {surf_idx}: last surface, no '
                            f'following gap in the wave prescription)')
                        continue
                    path = ('thicknesses', tk_idx)
                elif field in wave_surf_fields:
                    fs = flat_surf_map.get((elem_idx, surf_idx))
                    if fs is None:
                        continue
                    path = ('surfaces', fs, field)
                else:
                    # v5.24.x (audit S4-6): glass / semi_diameter etc. have
                    # no numeric slot in the legacy wave prescription.
                    self.log.append(
                        f'  (skipped {field} for element {elem_idx} '
                        f'surface {surf_idx}: not a wave-optimizable field)')
                    continue
                # v5.24.x (audit S4-6): a last-surface thickness and the
                # next element's distance address the SAME gap; emitting
                # both trips DesignParameterization's duplicate-path guard.
                if path in seen_paths:
                    self.log.append(
                        f'  (skipped duplicate {field} for element '
                        f'{elem_idx}: gap already mapped to {path})')
                    continue
                seen_paths.add(path)
                free_vars.append(path)
                # v5.24.3 (audit S4-2): the bounds' centre must be in the
                # SAME units as x0.  DesignParameterization.initial_values()
                # reads x0 from ``pres`` (to_prescription converts mm -> m),
                # so read the centre from the metre-unit template at the same
                # ``path`` -- NOT from get_variable_values() (millimetres),
                # which put x0 outside every box and made scipy clip the
                # start to a garbage (e.g. 25-metre-radius) design.
                if path[0] == 'thicknesses':
                    val = float(pres['thicknesses'][path[1]])
                else:  # ('surfaces', fs, field)
                    val = float(
                        pres['surfaces'][path[1]].get(path[2], 0.0) or 0.0)
                # Sensible bounds: conic is absolute; others fractional.
                if field == 'conic':
                    bounds_list.append((val - 2.0, val + 2.0))
                elif field in ('thickness', 'distance'):
                    bounds_list.append((max(0.0, val * 0.5),
                                        max(val * 2.0, 1e-4)))
                else:
                    lo = val * 0.5 if val > 0 else val * 2.0
                    hi = val * 2.0 if val > 0 else val * 0.5
                    bounds_list.append((min(lo, hi), max(lo, hi)))

            if not free_vars:
                self.log.append('No mappable variables for wave optimizer.')
                return

            param = DesignParameterization(
                template=pres, free_vars=free_vars, bounds=bounds_list)

            # Build merit list from UI combos
            merit_terms = []
            target = self.spin_target.value()

            geo_idx = self.combo_merit_geo.currentIndex()
            if geo_idx == 0:
                pass  # RMS spot handled by existing geometric optimizer
            elif geo_idx == 1:
                merit_terms.append(FocalLengthMerit(target=target * 1e-3, weight=1.0))
            elif geo_idx == 2:
                merit_terms.append(BackFocalLengthMerit(target=target * 1e-3, weight=1.0))
            elif geo_idx == 3:
                merit_terms.append(SphericalSeidelMerit(weight=1e-10))
            elif geo_idx == 4:
                merit_terms.append(MinThicknessMerit(min_thickness=1e-3, weight=1e6))
            elif geo_idx == 5:
                merit_terms.append(MaxFNumberMerit(max_f_number=target, weight=1.0))
            elif geo_idx == 6:
                # Chromatic focal shift across the UI's wavelength list
                wls = sorted(set(
                    float(w) * 1e-9 for w in self.sm.wavelengths_nm))
                if len(wls) < 2:
                    self.log.append(
                        '  (chromatic: need >=2 wavelengths -- '
                        'using current wavelength only, merit trivial)')
                merit_terms.append(ChromaticFocalShiftMerit(
                    wavelengths=wls or [self.sm.wavelength_nm * 1e-9],
                    weight=1.0))
            elif geo_idx == 7:
                # Tolerance-aware wrapper -- user sets the wrapped merit
                # implicitly (we wrap the currently-selected wave merit
                # if any, otherwise EFL target).  ``target`` here is the
                # number of Monte-Carlo trials.
                inner = FocalLengthMerit(target=target * 1e-3, weight=1.0)
                merit_terms.append(ToleranceAwareMerit(
                    inner_merit=inner,
                    n_trials=16, radius_sigma_frac=0.002,
                    thickness_sigma=5e-6, seed=1, weight=1.0))

            wave_idx = self.combo_merit_wave.currentIndex()
            if wave_idx == 1:
                merit_terms.append(StrehlMerit(min_strehl=target / 100.0 if target > 1 else target, weight=10.0))
            elif wave_idx == 2:
                merit_terms.append(RMSWavefrontMerit(max_rms_waves=target / 1000.0 if target > 1 else target, weight=50.0))
            elif wave_idx == 3:
                from lumenairy.raytrace import surfaces_from_prescription, system_abcd
                surfs = surfaces_from_prescription(pres)
                _, efl, _, _ = system_abcd(surfs, self.sm.wavelength_nm * 1e-9)
                merit_terms.append(MatchIdealThinLensMerit(
                    target_focal_length=efl, weight=10.0))
            elif wave_idx == 4:
                # Match full ideal system via field-overlap metric.  The
                # ideal system is built as a single thin lens at the
                # current EFL; users who want a bespoke reference can
                # edit the prescription returned by the optimizer.
                from lumenairy.raytrace import (
                    surfaces_from_prescription, system_abcd)
                surfs = surfaces_from_prescription(pres)
                _, efl, _, _ = system_abcd(surfs, self.sm.wavelength_nm * 1e-9)
                merit_terms.append(MatchIdealSystemMerit.single_lens(
                    f=float(efl), weight=10.0))
            elif wave_idx == 5:
                merit_terms.append(ZernikeCoefficientMerit(
                    targets={12: 0.0}, weight=100.0))  # minimize primary spherical

            if not merit_terms:
                self.log.append('No merit terms selected -- pick at least one.')
                return

            self.log.append(f'  {len(free_vars)} variables, {len(merit_terms)} merit terms')
            self.log.append(f'  Running design_optimize (L-BFGS-B)...')
            self.btn_optimize.setEnabled(False)
            self.btn_global.setEnabled(False)
            self.btn_wave.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.progress_bar.setVisible(True)
            # Determinate mode -- driven by the core progress hook.
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat('%p%')

            # Run in background thread
            use_jax = bool(self.chk_jax.isChecked())
            # v5.4 (audit P1-D): forward the full Advanced-parameters
            # dock surface to the wave worker.  The wave worker passes
            # method / hess / wave_propagator / precision / constraints
            # / state_file straight to design_optimize().  If the user
            # also enabled the multi-objective checkbox we re-route
            # via design_optimize_multi_objective(...) inside the
            # worker.
            adv = self._collect_advanced_kwargs()
            adv_max_iter = adv.pop('max_iter', self.spin_iter.value())
            self._worker = WaveOptimizeWorker(
                param, merit_terms, self.sm.wavelength_nm * 1e-9,
                adv_max_iter, use_jax=use_jax,
                advanced_kwargs=adv)
            if use_jax:
                self.log.append(
                    '  JAX wave propagator: ON (jac="auto" will use '
                    'analytic Jacobians for JAX merit terms)')
            # Log the advanced choices so the user sees the effective
            # call shape in the log.
            if adv:
                _amsg = ', '.join(f'{k}={v!r}' for k, v in adv.items()
                                  if v not in (None, '', 'auto'))
                if _amsg:
                    self.log.append(f'  Advanced: {_amsg}')
            self._worker.finished_result.connect(self._on_wave_finished)
            self._worker.fine_progress.connect(self._on_wave_progress)
            # v5.4 (audit P1-F): wave worker emits its own cancelled
            # signal; the finished_result already carries success=False
            # on cancel so we don't need a second UI handler.
            self._worker.start()

        except Exception as e:
            self.log.append(f'Wave optimizer setup failed: {type(e).__name__}: {e}')

    def _on_wave_progress(self, fraction, message):
        """Route core ``design_optimize`` progress into the bar."""
        self.progress_bar.setValue(
            int(1000 * max(0.0, min(1.0, fraction))))
        if message:
            # Hover the bar to see which iteration / message is current.
            self.progress_bar.setToolTip(message)
        # Try to extract merit=<val> from the message for the
        # convergence plot; messages look like
        # "iter N: merit=1.23e-4  efl=..." or "eval N: merit=..."
        if message and 'merit=' in message and self._conv_history is not None:
            try:
                tail = message.split('merit=', 1)[1]
                val = float(tail.split()[0])
                it = len(self._conv_history) + 1
                self._append_convergence(it, val)
            except Exception:
                pass

    def _on_wave_finished(self, result_dict):
        self.btn_optimize.setEnabled(True)
        self.btn_global.setEnabled(True)
        self.btn_wave.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        if result_dict.get('success'):
            self.log.append(
                f'Wave optimize done: merit={result_dict["merit"]:.3e}, '
                f'EFL={result_dict["efl_mm"]:.3f}mm, '
                f'Strehl={result_dict["strehl"]:.4f}, '
                f'{result_dict["iterations"]} iters, '
                f'{result_dict["time_sec"]:.1f}s')
            # Apply the optimised prescription back to the model
            if result_dict.get('prescription'):
                self.sm.load_prescription(
                    result_dict['prescription'],
                    wavelength_nm=self.sm.wavelength_nm)
                self.log.append('Prescription updated from optimizer result.')
        else:
            self.log.append(f'Wave optimize failed: {result_dict.get("msg", "unknown")}')
        self._worker = None

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


class WaveOptimizeWorker(QThread):
    """Background thread for hybrid wave/ray optimization."""
    finished_result = Signal(dict)
    fine_progress = Signal(float, str)   # fraction in [0, 1], label
    cancelled = Signal()

    def __init__(self, param, merit_terms, wavelength, max_iter,
                 use_jax=False, advanced_kwargs=None):
        super().__init__()
        self.param = param
        self.merit_terms = merit_terms
        self.wavelength = wavelength
        self.max_iter = max_iter
        self.use_jax = use_jax
        # v5.4 (audit P1-D): full advanced-parameter dict from the
        # dock.  Recognised keys: method, hess, wave_propagator,
        # precision, constraints, state_file, state_save_every,
        # multi_objective (bool), n_generations, pop_size.
        # Default empty -> preserves the pre-v5.4 'L-BFGS-B' /
        # double-precision / no-constraints behaviour.
        self.advanced_kwargs = dict(advanced_kwargs or {})
        # v5.4 (audit P1-F): CancellableProgress wraps the existing
        # Qt-emit callback.  design_optimize polls should_stop in all
        # 4 scipy callbacks and stops cleanly with a partial result.
        self._cancel_progress = CancellableProgress(self._on_progress)

    def _on_progress(self, stage, fraction, message=''):
        # Route the core's callback into a Qt signal the dock can
        # connect to its progress bar.
        self.fine_progress.emit(fraction, message)

    def _validate_kwargs(self):
        """v5.4 (audit P1-D): pre-flight check on dock-supplied kwargs.

        Raises ValueError on combinations that ``design_optimize``
        would reject downstream with a less-readable message.  Returns
        the cleaned kwarg dict ready to splat into ``design_optimize``.
        """
        adv = self.advanced_kwargs
        method = adv.get('method', 'L-BFGS-B')
        hess = adv.get('hess')
        if hess and hess != 'auto' and method not in _HESS_METHODS:
            raise ValueError(
                f"hess={hess!r} requires method in {_HESS_METHODS}; "
                f"got method={method!r}.")
        constraints = adv.get('constraints') or ()
        if constraints and method not in _CONSTRAINT_METHODS:
            raise ValueError(
                f"constraints= requires method in {_CONSTRAINT_METHODS}; "
                f"got method={method!r}.  Switch to SLSQP / trust-constr "
                f"or clear the constraints table.")
        precision = adv.get('precision', 'double')
        if precision not in ('double', 'single'):
            raise ValueError(
                f"precision must be 'double' or 'single', got "
                f"{precision!r}.")

        kwargs = {
            'method': method,
            'precision': precision,
        }
        if hess and hess != 'auto':
            kwargs['hess'] = hess
        if constraints:
            kwargs['constraints'] = list(constraints)
        wp = adv.get('wave_propagator')
        if wp and wp != 'real_lens':
            kwargs['wave_propagator'] = wp
        sf = adv.get('state_file')
        if sf:
            kwargs['state_file'] = sf
            ssev = adv.get('state_save_every')
            if ssev:
                kwargs['state_save_every'] = int(ssev)
        return kwargs

    def _run_multi_objective(self):
        """v5.4 (audit P1-D): NSGA-II Pareto front via pymoo.

        Wraps each MeritTerm.evaluate(...) into a scalar callable so
        ``design_optimize_multi_objective`` can score the population.
        The Pareto result is logged but the prescription returned is
        the first (and any) point on the front -- the user can run a
        single-objective refinement on a specific solution later.
        """
        from lumenairy.optimize import design_optimize_multi_objective
        from lumenairy.optimize import EvaluationContext

        adv = self.advanced_kwargs
        n_gen = int(adv.get('n_generations', 100))
        pop = int(adv.get('pop_size', 50))

        # Wrap each merit term as a scalar callable f(x) -> float by
        # building a transient EvaluationContext.  Ray-only merits
        # work; wave merits will fail (need_wave) -- we let the user
        # discover that via the wrapper's exception path.
        wavelength = self.wavelength
        param = self.param

        def make_merit(term):
            def _f(x):
                pres = param.build(x)
                ctx = EvaluationContext(
                    prescription=pres, wavelength=wavelength,
                    N=256, dx=16e-6, x=np.asarray(x, dtype=np.float64))
                try:
                    return float(term.evaluate(ctx))
                except Exception:
                    return 1e18
            return _f
        merits = [make_merit(t) for t in self.merit_terms]

        # Initial x0 from parameterization defaults.
        try:
            x0 = np.asarray(param.x0, dtype=np.float64)
        except AttributeError:
            x0 = np.zeros(len(param.bounds), dtype=np.float64)
        bounds = list(param.bounds)
        result = design_optimize_multi_objective(
            merits, x0, bounds,
            n_generations=n_gen, pop_size=pop,
            progress=self._cancel_progress,
            verbose=False,
        )
        return result

    def run(self):
        try:
            # v5.4 (audit P1-D): NSGA-II Pareto front branch.
            if self.advanced_kwargs.get('multi_objective'):
                pareto = self._run_multi_objective()
                self.finished_result.emit({
                    'success': True,
                    'merit': float(np.min(pareto.F[:, 0]))
                              if pareto.F.size else 0.0,
                    'efl_mm': 0.0,
                    'strehl': 0.0,
                    'iterations': int(pareto.n_generations),
                    'time_sec': 0.0,
                    'prescription': None,
                    'pareto_F': pareto.F,
                    'pareto_X': pareto.X,
                })
                return

            from lumenairy.optimize import design_optimize
            # JAX wave propagator (3.5.0+): apply_real_lens_traced_jax
            # is JAX-traceable so design_optimize's jac='auto' default
            # can construct analytic Jacobians for JAX-aware merit
            # terms.  The kwarg is forwarded only when requested so
            # the NumPy default path is unchanged for existing users.
            extra = {}
            if self.use_jax:
                extra['wave_propagator'] = 'real_lens_traced_jax'

            # v5.4 (audit P1-D): merge dock-supplied advanced kwargs
            # over the worker defaults.  Method defaults to 'L-BFGS-B'
            # here (was hardcoded pre-v5.4); user picks at dock level.
            adv_clean = self._validate_kwargs()
            # If the dock picked a specific wave_propagator that
            # disagrees with use_jax, the dock-supplied value wins
            # (the user explicitly picked it -- documented behaviour).
            base_kwargs = {
                'method': 'L-BFGS-B',
            }
            base_kwargs.update(extra)
            base_kwargs.update(adv_clean)

            result = design_optimize(
                parameterization=self.param,
                merit_terms=self.merit_terms,
                wavelength=self.wavelength,
                N=256, dx=16e-6,
                max_iter=self.max_iter,
                verbose=False,
                progress=self._cancel_progress,
                **base_kwargs)
            if self._cancel_progress.should_stop:
                self.cancelled.emit()
                self.finished_result.emit({
                    'success': False,
                    'msg': 'Cancelled by user',
                })
                return
            self.finished_result.emit({
                'success': True,
                'merit': result.merit,
                'efl_mm': result.context_final.efl * 1e3,
                'strehl': result.context_final.strehl_best,
                'iterations': result.iterations,
                'time_sec': result.time_sec,
                'prescription': result.prescription,
            })
        except Exception as e:
            self.finished_result.emit({
                'success': False,
                'msg': f'{type(e).__name__}: {e}',
            })

    @Slot()
    def cancel(self):
        self._cancel_progress.cancel()


class GlobalSearchWorker(QThread):
    """Random-restart global optimization (inspired by CODE V Global Synthesis)."""
    progress = Signal(int, float)  # restart number, best merit
    finished = Signal(bool, str)
    cancelled = Signal()

    def __init__(self, model, max_iter_per_restart, n_restarts):
        super().__init__()
        self.model = model
        self.max_iter = max_iter_per_restart
        self.n_restarts = n_restarts
        # v5.4 (audit P1-F): polled between restarts (and inside each
        # restart's Nelder-Mead callback) for clean cancellation.
        self._cancel_progress = CancellableProgress()

    def run(self):
        from scipy.optimize import minimize

        x0 = self.model.get_variable_values()
        best_x = x0.copy()
        best_merit = self.model.merit_function(x0)
        rng = np.random.default_rng()

        def _inner_cb(xk):
            if self._cancel_progress.should_stop:
                raise StopIteration('cancelled')

        for restart in range(self.n_restarts):
            if self._cancel_progress.should_stop:
                break
            # Perturb starting point: ±30% for radius/thickness, ±1 for conic
            x_start = x0.copy()
            for i, (row_idx, col_idx) in enumerate(self.model.opt_variables):
                if col_idx == 7:  # conic
                    x_start[i] = x0[i] + rng.uniform(-1, 1)
                else:
                    x_start[i] = x0[i] * (1 + rng.uniform(-0.3, 0.3))

            try:
                result = minimize(
                    self.model.merit_function, x_start,
                    method='Nelder-Mead',
                    options={'maxiter': self.max_iter, 'xatol': 1e-8, 'fatol': 1e-12},
                    callback=_inner_cb,
                )
                if result.fun < best_merit:
                    best_merit = result.fun
                    best_x = result.x.copy()
            except StopIteration:
                break
            except Exception:
                pass

            self.progress.emit(restart + 1, best_merit)

        # Apply best result (best-so-far on cancel)
        self.model.set_variable_values(best_x)
        self.model._invalidate()
        self.model.system_changed.emit()
        if self._cancel_progress.should_stop:
            self.cancelled.emit()
            self.finished.emit(
                False,
                f'Cancelled -- best so far: {best_merit*1e6:.3f} um')
            return
        msg = f'Best merit: {best_merit*1e6:.3f} um from {self.n_restarts} restarts'
        self.finished.emit(True, msg)

    @Slot()
    def cancel(self):
        self._cancel_progress.cancel()


# ---------------------------------------------------------------------------
# Variable-grid dialog (replaces the two-popup dance).
# ---------------------------------------------------------------------------

class _VariableGridDialog(QDialog):
    """Grid of checkboxes: rows = (element, surface), cols = parameter.

    Current values are shown next to each checkbox so the user knows
    what they're freeing up.  OK applies the whole selection at once.
    """

    PARAMS = ('radius', 'thickness', 'conic', 'radius_y', 'conic_y')

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self.setWindowTitle('Choose optimization variables')
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout(self)

        intro = QLabel(
            'Tick every parameter that should be free during optimization. '
            'The current value is shown beside each checkbox so you know '
            'the starting point.  radius_y / conic_y only appear on '
            'biconic surfaces.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#7a94b8;')
        layout.addWidget(intro)

        # Scrollable grid (long systems need it)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid_layout = QVBoxLayout(inner)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setSpacing(2)

        self._checks = []   # list of (elem_idx, surf_idx, field, QCheckBox)

        existing = set(tuple(v) for v in self.sm.opt_variables)

        for ei, elem in enumerate(self.sm.elements):
            if elem.elem_type in ('Source', 'Detector'):
                continue
            box = QGroupBox(f'E{ei}  {elem.elem_type}: {elem.name}')
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(6, 2, 6, 4)

            # Per-element "distance" row
            if ei > 0:
                row = QHBoxLayout()
                chk = QCheckBox('distance')
                chk.setToolTip(
                    'Axial distance from the previous element.  '
                    'Freeing this lets the optimizer move the element '
                    'along the optical axis.')
                if (ei, 0, 'distance') in existing:
                    chk.setChecked(True)
                val_lbl = QLabel(f'= {elem.distance_mm:.4g} mm')
                val_lbl.setStyleSheet('color:#5cb8ff; font-family:Consolas;')
                row.addWidget(chk)
                row.addWidget(val_lbl)
                row.addStretch()
                box_layout.addLayout(row)
                self._checks.append((ei, 0, 'distance', chk))

            # One row per surface, showing every editable parameter.
            for si, s in enumerate(elem.surfaces):
                surf_row = QHBoxLayout()
                surf_row.addWidget(QLabel(f'  S{si}:'))
                for field in self.PARAMS:
                    val = getattr(s, field, None)
                    if val is None and field not in ('radius', 'thickness',
                                                     'conic'):
                        continue    # hide biconic-only fields on symmetric surf
                    chk = QCheckBox(field)
                    if (ei, si, field) in existing:
                        chk.setChecked(True)
                    if np.isinf(val):
                        txt = '\u221e'
                    else:
                        txt = f'{val:.4g}'
                    lbl = QLabel(f'= {txt}')
                    lbl.setStyleSheet('color:#5cb8ff; font-family:Consolas;')
                    surf_row.addWidget(chk)
                    surf_row.addWidget(lbl)
                    self._checks.append((ei, si, field, chk))
                surf_row.addStretch()
                box_layout.addLayout(surf_row)

            grid_layout.addWidget(box)

        grid_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        # Bulk toggles
        bulk = QHBoxLayout()
        btn_all_radii = QPushButton('Free all radii')
        btn_all_radii.clicked.connect(lambda: self._set_all('radius', True))
        btn_all_thick = QPushButton('Free all thicknesses')
        btn_all_thick.clicked.connect(
            lambda: self._set_all('thickness', True))
        btn_clear = QPushButton('Clear all')
        btn_clear.clicked.connect(self._clear_all)
        bulk.addWidget(btn_all_radii)
        bulk.addWidget(btn_all_thick)
        bulk.addWidget(btn_clear)
        bulk.addStretch()
        layout.addLayout(bulk)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _set_all(self, field, state):
        for ei, si, f, chk in self._checks:
            if f == field:
                chk.setChecked(state)

    def _clear_all(self):
        for _, _, _, chk in self._checks:
            chk.setChecked(False)

    def checked_variables(self):
        return [(ei, si, f) for ei, si, f, chk in self._checks
                if chk.isChecked()]


# ---------------------------------------------------------------------------
# Weight-editing helpers.
# ---------------------------------------------------------------------------

def _photopic_weights(wavelengths_nm):
    """Approximate CIE 1931 V(lambda) photopic luminosity, peak at 555 nm."""
    import numpy as np
    w = np.asarray(wavelengths_nm, dtype=float)
    # Gaussian-ish fit centred at 555 nm, sigma ~60 nm
    weights = np.exp(-((w - 555.0) / 60.0) ** 2)
    if weights.sum() <= 0:
        weights = np.ones_like(w)
    return list(weights / weights.max())


def _axial_heavy_weights(field_angles_deg):
    """cos^4(theta) weighting -- overweights axial, tracks relative
    illumination in a typical imaging system."""
    import numpy as np
    f = np.asarray(field_angles_deg, dtype=float)
    th = np.deg2rad(np.abs(f))
    return list(np.cos(th) ** 4)


class _WeightsDialog(QDialog):
    """Tiny editor for per-wavelength / per-field weights with presets."""

    def __init__(self, title, labels, current, presets=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel('Preset:'))
        self.combo_preset = QComboBox()
        self.combo_preset.addItem('(custom)')
        if presets:
            for k in presets.keys():
                self.combo_preset.addItem(k)
        self._presets = presets or {}
        self.combo_preset.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self.combo_preset)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        self._rows = []
        from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout
        form = QFormLayout()
        if current is None or len(current) != len(labels):
            current = [1.0] * len(labels)
        for lab, val in zip(labels, current):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1e6)
            sp.setDecimals(4)
            sp.setValue(float(val))
            form.addRow(lab, sp)
            self._rows.append(sp)
        layout.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _apply_preset(self, text):
        vals = self._presets.get(text)
        if not vals:
            return
        for sp, v in zip(self._rows, vals):
            sp.setValue(float(v))

    def weights(self):
        return [sp.value() for sp in self._rows]


# ---------------------------------------------------------------------------
# v5.4 (audit P1-D): Constraint editor sub-panel.
# ---------------------------------------------------------------------------

class _ConstraintsEditor(QGroupBox):
    """Sub-panel for editing the Constraint sequence passed to
    design_optimize(constraints=...).

    Layout: a QTableWidget with one row per Constraint and five
    columns (label, fun expression, lb, ub, kind) plus Add / Remove
    row buttons.  ``fun`` is entered as a string expression in the
    parameter vector ``x``; on submission it is compiled to a
    callable via ``eval('lambda x: ' + expr, {'np': np})`` so users
    can reference numpy helpers.  Empty rows are silently skipped.

    Output via :meth:`to_constraints` -- builds a list of
    :class:`lumenairy.optimize.Constraint` instances ready to splat
    into design_optimize(constraints=...).  Returns an empty list
    when the editor is empty so callers can pass the result
    unconditionally.
    """

    COLS = ('label', 'fun (lambda x: ...)', 'lb', 'ub', 'kind')

    def __init__(self, parent=None):
        super().__init__('Constraints (SLSQP / trust-constr only)', parent)
        self.setToolTip(
            'Hard non-linear constraints applied by design_optimize via '
            'scipy.optimize.NonlinearConstraint.  Each row evaluates '
            '``f(x)`` and enforces ``lb <= f(x) <= ub``.  Leave lb or '
            'ub blank for one-sided constraints (-inf / +inf).')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(list(self.COLS))
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table.setMaximumHeight(120)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton('+ Add constraint')
        btn_add.clicked.connect(self._add_row)
        btn_remove = QPushButton('- Remove selected')
        btn_remove.clicked.connect(self._remove_row)
        btn_clear = QPushButton('Clear')
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        # Pre-fill helpful placeholders so the user knows the format.
        defaults = ['c{}'.format(r), 'np.sum(x)', '-inf', '+inf', 'ineq']
        for c, default in enumerate(defaults):
            if c == 4:
                # kind: dropdown via setCellWidget
                combo = QComboBox()
                combo.addItems(['ineq', 'eq'])
                combo.setCurrentText(default)
                self.table.setCellWidget(r, c, combo)
            else:
                self.table.setItem(r, c, QTableWidgetItem(default))

    def _remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _clear(self):
        self.table.setRowCount(0)

    def to_constraints(self):
        """Compile the table rows into Constraint instances.

        Returns an empty list when the table is empty.  Skips rows
        with a blank ``fun`` cell silently (an empty row from
        accidental Add clicks shouldn't blow up the run).  Raises
        ``ValueError`` on rows whose ``fun`` expression doesn't
        compile or whose lb / ub aren't parseable floats.
        """
        try:
            from lumenairy.optimize import Constraint
        except Exception:
            return []
        out = []
        for r in range(self.table.rowCount()):
            label_item = self.table.item(r, 0)
            fun_item = self.table.item(r, 1)
            lb_item = self.table.item(r, 2)
            ub_item = self.table.item(r, 3)
            kind_widget = self.table.cellWidget(r, 4)

            label = (label_item.text() if label_item else '').strip()
            fun_expr = (fun_item.text() if fun_item else '').strip()
            if not fun_expr:
                continue
            lb_str = (lb_item.text() if lb_item else '').strip().lower()
            ub_str = (ub_item.text() if ub_item else '').strip().lower()
            kind = (kind_widget.currentText() if kind_widget else 'ineq')

            # Parse bounds.  Empty / 'inf' / '-inf' map to None so
            # Constraint(lb=None, ub=...) is one-sided.
            def _parse_bound(s):
                if not s or s == 'none':
                    return None
                if s in ('inf', '+inf', 'infinity', '+infinity'):
                    return float('inf')
                if s in ('-inf', '-infinity'):
                    return float('-inf')
                try:
                    return float(s)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Constraint row {r}: cannot parse bound "
                        f"{s!r} as a float.  Use a numeric literal, "
                        f"'inf', '-inf', or leave blank.")
            lb = _parse_bound(lb_str)
            ub = _parse_bound(ub_str)
            # Treat +/-inf endpoints as None so Constraint accepts
            # them as "no bound on this side".  Constraint.__post_init__
            # raises if both endpoints are None, so the user is forced
            # to give at least one finite bound.
            if lb is not None and np.isinf(lb) and lb < 0:
                lb = None
            if ub is not None and np.isinf(ub) and ub > 0:
                ub = None
            if kind == 'eq':
                # scipy NonlinearConstraint encodes equality as lb==ub.
                # If only one side is provided we treat it as the
                # equality target.
                target = lb if lb is not None else ub
                if target is None:
                    raise ValueError(
                        f"Constraint row {r}: kind='eq' requires lb "
                        f"or ub (the equality target).")
                lb = ub = target

            # Compile the expression to a callable f(x) -> float.  We
            # accept either a bare expression in ``x`` (e.g.
            # ``np.sum(x)``) or a full ``lambda x: ...`` literal.
            # The expression namespace is locked down to numpy +
            # builtins to avoid arbitrary-name leakage.
            try:
                if fun_expr.lstrip().startswith('lambda'):
                    fun = eval(fun_expr, {'np': np, '__builtins__': {}})
                else:
                    fun = eval(
                        'lambda x: ' + fun_expr,
                        {'np': np, '__builtins__': {}})
            except Exception as e:
                raise ValueError(
                    f"Constraint row {r}: cannot compile fun "
                    f"expression {fun_expr!r}: {type(e).__name__}: "
                    f"{e}")
            out.append(Constraint(
                fun=fun, lb=lb, ub=ub, label=label or f'row{r}'))
        return out
