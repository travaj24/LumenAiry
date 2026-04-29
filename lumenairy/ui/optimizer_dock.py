"""
Optimizer dock — variable selection, merit function, optimization control.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QTextEdit, QCheckBox, QGroupBox, QComboBox,
    QDialog, QDialogButtonBox, QScrollArea,
)
from PySide6.QtGui import QFont, QColor

import numpy as np

from .model import SystemModel, SurfaceRow


class OptimizeWorker(QThread):
    """Run optimization in a background thread."""
    progress = Signal(int, float)
    finished = Signal(bool, str)

    def __init__(self, model, max_iter):
        super().__init__()
        self.model = model
        self.max_iter = max_iter

    def run(self):
        def cb(it, merit):
            self.progress.emit(it, merit)
        success, msg = self.model.run_optimization(self.max_iter, cb)
        self.finished.emit(success, msg)


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

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel('Max iterations:'))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 5000)
        self.spin_iter.setValue(200)
        iter_row.addWidget(self.spin_iter)
        opt_layout.addLayout(iter_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        opt_layout.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.btn_optimize = QPushButton('Local Optimize')
        self.btn_optimize.clicked.connect(self._start_optimize)
        self.btn_global = QPushButton('Global Search')
        self.btn_global.setToolTip('Random restart optimization (finds different lens forms)')
        self.btn_global.clicked.connect(self._start_global)
        self.btn_wave = QPushButton('Wave Optimize')
        self.btn_wave.setToolTip(
            'Hybrid wave/ray optimization using the design_optimize engine. '
            'Slower but uses wave-optics merits (Strehl, wavefront, etc.)')
        self.btn_wave.clicked.connect(self._start_wave_optimize)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_optimize)
        btn_row.addWidget(self.btn_optimize)
        btn_row.addWidget(self.btn_global)
        btn_row.addWidget(self.btn_wave)
        btn_row.addWidget(self.btn_stop)
        opt_layout.addLayout(btn_row)

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

        self._worker = OptimizeWorker(self.sm, self.spin_iter.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _stop_optimize(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._on_finished(False, 'Stopped by user')

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
                        flat_thk += 1

            free_vars = []
            bounds_list = []
            values = self.sm.get_variable_values()
            for i, (elem_idx, surf_idx, field) in enumerate(self.sm.opt_variables):
                if field == 'distance':
                    tk_idx = thickness_map.get(elem_idx)
                    if tk_idx is None:
                        self.log.append(
                            f'  (skipped distance for element {elem_idx}: '
                            f'first/source element has no preceding gap)')
                        continue
                    path = ('thicknesses', tk_idx)
                else:
                    fs = flat_surf_map.get((elem_idx, surf_idx))
                    if fs is None:
                        continue
                    path = ('surfaces', fs, field)
                free_vars.append(path)
                val = values[i] if i < len(values) else 0.0
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
            self._worker = WaveOptimizeWorker(
                param, merit_terms, self.sm.wavelength_nm * 1e-9,
                self.spin_iter.value())
            self._worker.finished_result.connect(self._on_wave_finished)
            self._worker.fine_progress.connect(self._on_wave_progress)
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


class WaveOptimizeWorker(QThread):
    """Background thread for hybrid wave/ray optimization."""
    finished_result = Signal(dict)
    fine_progress = Signal(float, str)   # fraction in [0, 1], label

    def __init__(self, param, merit_terms, wavelength, max_iter):
        super().__init__()
        self.param = param
        self.merit_terms = merit_terms
        self.wavelength = wavelength
        self.max_iter = max_iter

    def _on_progress(self, stage, fraction, message=''):
        # Route the core's callback into a Qt signal the dock can
        # connect to its progress bar.
        self.fine_progress.emit(fraction, message)

    def run(self):
        try:
            from lumenairy.optimize import design_optimize
            result = design_optimize(
                parameterization=self.param,
                merit_terms=self.merit_terms,
                wavelength=self.wavelength,
                N=256, dx=16e-6,
                method='L-BFGS-B',
                max_iter=self.max_iter,
                verbose=False,
                progress=self._on_progress)
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


class GlobalSearchWorker(QThread):
    """Random-restart global optimization (inspired by CODE V Global Synthesis)."""
    progress = Signal(int, float)  # restart number, best merit
    finished = Signal(bool, str)

    def __init__(self, model, max_iter_per_restart, n_restarts):
        super().__init__()
        self.model = model
        self.max_iter = max_iter_per_restart
        self.n_restarts = n_restarts

    def run(self):
        from scipy.optimize import minimize

        x0 = self.model.get_variable_values()
        best_x = x0.copy()
        best_merit = self.model.merit_function(x0)
        rng = np.random.default_rng()

        for restart in range(self.n_restarts):
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
                )
                if result.fun < best_merit:
                    best_merit = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            self.progress.emit(restart + 1, best_merit)

        # Apply best result
        self.model.set_variable_values(best_x)
        self.model._invalidate()
        self.model.system_changed.emit()
        msg = f'Best merit: {best_merit*1e6:.3f} µm from {self.n_restarts} restarts'
        self.finished.emit(True, msg)


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
