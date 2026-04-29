"""
Multi-configuration designer dock.

Exposes ``MultiPrescriptionParameterization``: lets the user clone
the current system into N configurations (zoom-step, day/night,
laser / imaging, etc.) and jointly optimise all of them at once.
Each configuration gets its own thickness / radius overrides, but
the free-variable set is shared so the final design is a single
set of parts that works across every configuration.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QGroupBox, QTextEdit, QProgressBar,
)
from PySide6.QtGui import QFont


class _MultiConfigWorker(QThread):
    fine_progress = Signal(float, str)
    finished_result = Signal(dict)

    def __init__(self, templates, free_vars, bounds, merit_terms,
                 wavelength, max_iter):
        super().__init__()
        self.templates = templates
        self.free_vars = free_vars
        self.bounds = bounds
        self.merit_terms = merit_terms
        self.wavelength = wavelength
        self.max_iter = max_iter

    def _progress(self, stage, frac, msg=''):
        self.fine_progress.emit(float(frac), str(msg))

    def run(self):
        try:
            from lumenairy.optimize import (
                MultiPrescriptionParameterization, design_optimize)
            par = MultiPrescriptionParameterization(
                templates=self.templates,
                free_vars=self.free_vars,
                bounds=self.bounds)
            res = design_optimize(
                par, self.merit_terms, wavelength=self.wavelength,
                method='L-BFGS-B', max_iter=self.max_iter,
                verbose=False, progress=self._progress)
            self.finished_result.emit({
                'success': True,
                'merit': res.merit,
                'iterations': res.iterations,
                'time_sec': res.time_sec,
                'prescriptions': res.prescriptions,
            })
        except Exception as e:
            self.finished_result.emit({
                'success': False,
                'error': f'{type(e).__name__}: {e}'})


class MultiConfigDock(QWidget):
    """Multi-configuration joint-optimisation control panel."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        intro = QLabel(
            'Joint optimisation across multiple configurations: each '
            'config is a variant of the current prescription with its '
            'own thickness / radius overrides, but free variables are '
            'shared.  The result is a single set of surface radii that '
            'works in every configuration (used for zoom steps, '
            'day/night modes, laser + imaging duty-cycle, etc.).')
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#7a94b8;')
        layout.addWidget(intro)

        cfg_box = QGroupBox('Configurations')
        cfg_layout = QVBoxLayout(cfg_box)
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton('+ Add configuration from current')
        self.btn_add.clicked.connect(self._add_config)
        btn_row.addWidget(self.btn_add)
        self.btn_remove = QPushButton('Remove selected')
        self.btn_remove.clicked.connect(self._remove_config)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch()
        cfg_layout.addLayout(btn_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            'Name', 'Wavelength (nm)', 'Notes'])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table.setMaximumHeight(160)
        cfg_layout.addWidget(self.table)

        layout.addWidget(cfg_box)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel('Max iter:'))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 5000)
        self.spin_iter.setValue(100)
        run_row.addWidget(self.spin_iter)
        self.btn_run = QPushButton('Jointly optimise')
        self.btn_run.setToolTip(
            'Run design_optimize on the MultiPrescriptionParameterization.'
            '  Uses the wavelengths and variable list set in the '
            'Optimizer dock.')
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(140)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

        self._configs = []   # list of {name, prescription, wavelength}

    def _add_config(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, 'New configuration', 'Name:',
            text=f'Config {len(self._configs) + 1}')
        if not ok or not name:
            return
        try:
            pres = self.sm.to_prescription()
        except Exception as e:
            self.summary.append(f'Snapshot failed: {e}')
            return
        self._configs.append({
            'name': name,
            'prescription': pres,
            'wavelength': self.sm.wavelength_nm,
        })
        self._refresh_table()

    def _remove_config(self):
        idx = self.table.currentRow()
        if 0 <= idx < len(self._configs):
            del self._configs[idx]
            self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self._configs))
        for i, c in enumerate(self._configs):
            self.table.setItem(i, 0, QTableWidgetItem(c['name']))
            self.table.setItem(i, 1,
                QTableWidgetItem(f'{c["wavelength"]:.1f}'))
            self.table.setItem(i, 2, QTableWidgetItem(''))

    def _run(self):
        if len(self._configs) < 1:
            self.summary.append('Add at least one configuration first.')
            return
        if not self.sm.opt_variables:
            self.summary.append(
                'No variables -- define them in the Optimizer dock first.')
            return
        templates = [c['prescription'] for c in self._configs]
        # Map (elem_idx, surf_idx, field) triples to flat free-var paths
        # keyed by the configuration index so the parameterisation can
        # apply the same value to all templates.
        free_vars = []
        bounds = []
        values = self.sm.get_variable_values()
        for i, (elem_idx, surf_idx, field) in enumerate(self.sm.opt_variables):
            # Use the Optimizer's path logic -- here we just pass
            # ('surfaces', surf_idx, field) against every template
            # (the free-var list is shared across prescriptions by
            # index 0 convention).
            path = (0, 'surfaces', surf_idx, field)
            free_vars.append(path)
            v = values[i]
            if field == 'conic':
                bounds.append((v - 2, v + 2))
            else:
                lo = v * 0.5 if v > 0 else v * 2
                hi = v * 2 if v > 0 else v * 0.5
                bounds.append((min(lo, hi), max(lo, hi)))
        try:
            from lumenairy.optimize import (
                FocalLengthMerit, BackFocalLengthMerit)
            merits = [FocalLengthMerit(target=0.05, weight=1.0)]
            self.btn_run.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self._worker = _MultiConfigWorker(
                templates, free_vars, bounds, merits,
                self.sm.wavelength_nm * 1e-9,
                self.spin_iter.value())
            self._worker.fine_progress.connect(self._on_progress)
            self._worker.finished_result.connect(self._on_finished)
            self._worker.start()
        except Exception as e:
            self.summary.append(f'Start failed: {type(e).__name__}: {e}')
            self.btn_run.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_progress(self, frac, msg):
        self.progress_bar.setValue(int(1000 * max(0.0, min(1.0, frac))))
        if msg:
            self.progress_bar.setToolTip(msg)

    def _on_finished(self, res):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._worker = None
        if not res.get('success'):
            self.summary.append(f'Failed: {res.get("error")}')
            return
        self.summary.append(
            f'Joint optimisation: merit={res["merit"]:.3e}, '
            f'{res["iterations"]} iters, {res["time_sec"]:.1f}s')
        for i, p in enumerate(res['prescriptions'] or []):
            self.summary.append(
                f'  config {i} ({self._configs[i]["name"]}) -- '
                f'{len(p.get("surfaces", []))} surfaces')
