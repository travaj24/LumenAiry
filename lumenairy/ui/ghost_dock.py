"""
Ghost-path analysis dock — enumerate internal reflections from each
surface pair and rank them by R_i * R_j.

Wraps :func:`lumenairy.ghost.ghost_analysis` in a sortable
Qt table with a one-click "Run" button.  Stays idle until the user
requests a run (analysis is O(N_surfaces^2) and hits the glass-index
catalog repeatedly; not worth doing on every edit).

Author: Andrew Traverso
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

import numpy as np

from .model import SystemModel


class GhostDock(QWidget):
    """Ghost-path table: (i, j, R_i, R_j, intensity) sorted brightest-first.

    One row per ordered surface pair that can support a double-bounce
    ghost; the intensity column is ``R_i * R_j`` from bare-Fresnel
    reflectance at the relevant interfaces.  Coatings are not
    considered — this is a pessimistic first-pass estimate that lens
    designers use to decide where AR coatings will do the most good.
    """

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -- Toolbar
        toolbar = QHBoxLayout()
        self.btn_run = QPushButton('Run ghost analysis')
        self.btn_run.setToolTip(
            'Enumerate all ordered surface pairs (i, j) with j > i+1 '
            'and compute the ghost intensity R_i * R_j from bare-Fresnel '
            'reflectance at each interface.  Stays cheap; re-run after '
            'editing the prescription.')
        self.btn_run.clicked.connect(self._run)
        toolbar.addWidget(self.btn_run)

        self.lbl_summary = QLabel('Idle.  Click "Run" to analyze.')
        self.lbl_summary.setStyleSheet(
            'color: #7a94b8; font-family: monospace;')
        toolbar.addWidget(self.lbl_summary, stretch=1)

        layout.addLayout(toolbar)

        # -- Ghost table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ['i', 'j', 'Path', 'R_i', 'R_j', 'Intensity'])
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

    def _run(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)

        # Build the prescription directly from the model — ghost_analysis
        # takes the same dict shape as apply_real_lens.
        pres = self.sm.to_prescription()
        if not pres.get('surfaces'):
            self.lbl_summary.setText('No surfaces in current system.')
            self.table.setSortingEnabled(True)
            return

        wv = self.sm.wavelength_m
        try:
            from ..ghost import ghost_analysis
            ghosts = ghost_analysis(pres, wv, verbose=False)
        except Exception as e:
            self.lbl_summary.setText(f'Error: {e}')
            try:
                from .diagnostics import diag
                diag.report('ghost-dock', e,
                            context=f'nsurf={len(pres["surfaces"])}')
            except Exception:
                pass
            self.table.setSortingEnabled(True)
            return

        if not ghosts:
            self.lbl_summary.setText('No ghost paths found.')
            self.table.setSortingEnabled(True)
            return

        self.table.setRowCount(len(ghosts))
        for row, g in enumerate(ghosts):
            i = int(g.get('i', -1))
            j = int(g.get('j', -1))
            R_i = float(g.get('R_i', 0.0))
            R_j = float(g.get('R_j', 0.0))
            I = float(g.get('intensity', R_i * R_j))
            path = g.get('path', f'S{i} -> S{j}')

            def _set_numeric(row_, col, val, fmt):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, val)
                item.setText(fmt.format(val))
                self.table.setItem(row_, col, item)

            _set_numeric(row, 0, i, '{:d}')
            _set_numeric(row, 1, j, '{:d}')
            item = QTableWidgetItem(str(path))
            self.table.setItem(row, 2, item)
            _set_numeric(row, 3, R_i, '{:.4e}')
            _set_numeric(row, 4, R_j, '{:.4e}')
            _set_numeric(row, 5, I, '{:.4e}')

        self.table.setSortingEnabled(True)
        # Default sort: brightest ghost first (intensity column desc)
        self.table.sortByColumn(5, Qt.DescendingOrder)

        brightest = ghosts[0]
        self.lbl_summary.setText(
            f'{len(ghosts)} ghost paths.  Brightest = '
            f'I = {brightest.get("intensity", 0):.3e} '
            f'(S{brightest.get("i")} -> S{brightest.get("j")}).')
