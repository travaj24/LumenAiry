# v5.4 (audit P2-C): expand from 141-LOC single-function wrapper
"""
Ghost-path analysis dock -- enumerate, filter, rank, and inspect
2-bounce ghost reflections through the current system.

Library functions wired (audit P2-C):

  * ``enumerate_ghost_paths``      -- combinatorial (i, j) pair list
  * ``ghost_analysis``             -- per-path Fresnel intensity UB
  * ``non_sequential_stray_light`` -- conservative budget total

Author: Andrew Traverso  -- v5.4 audit expansion.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QSizePolicy, QFileDialog, QMessageBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class _GhostWorker(QThread):
    """Run ``ghost_analysis`` + ``non_sequential_stray_light`` off-thread."""

    finished_result = Signal(object)

    def __init__(self, prescription, wavelength, semi_aperture,
                 n_rays, top_n):
        super().__init__()
        self.prescription = prescription
        self.wavelength = wavelength
        self.semi_aperture = semi_aperture
        self.n_rays = n_rays
        self.top_n = top_n

    def run(self):
        try:
            from ..analysis.ghost import (
                enumerate_ghost_paths, ghost_analysis,
                non_sequential_stray_light,
            )
            ghosts = ghost_analysis(
                self.prescription, self.wavelength,
                semi_aperture=self.semi_aperture,
                n_rays=self.n_rays, verbose=False)
            report = non_sequential_stray_light(
                self.prescription, self.wavelength,
                semi_aperture=self.semi_aperture,
                n_rays=self.n_rays, top_n=max(self.top_n, 1),
                bsdf_model=None, verbose=False)
            n_surfaces = self._n_surfaces(self.prescription)
            all_paths = enumerate_ghost_paths(n_surfaces)
            self.finished_result.emit({
                'ghosts': ghosts, 'report': report,
                'n_surfaces': n_surfaces, 'all_paths': all_paths,
            })
        except Exception as exc:
            self.finished_result.emit(
                {'error': f'{type(exc).__name__}: {exc}'})

    @staticmethod
    def _n_surfaces(pres):
        if 'surfaces' in pres:
            return len(pres['surfaces'])
        if 'elements' in pres:
            return sum(
                1 for e in pres['elements']
                if e.get('element_type', e.get('type')) in
                ('surface', 'mirror', 'real_lens'))
        return 0


# ---------------------------------------------------------------------------
# Dock
# ---------------------------------------------------------------------------

class GhostDock(QWidget):
    """Ghost-path enumeration + filter UI.

    Constructor signature preserved from the original 141-LOC wrapper
    so ``main_window.py`` registration keeps working unmodified.
    """

    COL_PATH, COL_ORDER, COL_T, COL_EPPM, COL_FOCUS_MM = 0, 1, 2, 3, 4

    SORT_ENERGY = 'energy_fraction (desc)'
    SORT_ORDER = 'order'
    SORT_T = 'transmittance (desc)'

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker: Optional[_GhostWorker] = None
        self._all_ghosts: List[Dict[str, Any]] = []
        self._filtered_ghosts: List[Dict[str, Any]] = []
        self._stray_light_fraction: float = 0.0
        self._n_surfaces: int = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_run = QPushButton('Run ghost analysis')
        self.btn_run.setObjectName('run_button')
        self.btn_run.setToolTip(
            'Enumerate every 2-bounce ghost path, compute the '
            'Fresnel R_i * R_j upper bound, and feed the result '
            'through non_sequential_stray_light for a conservative '
            'budget total.')
        self.btn_run.clicked.connect(self._run)
        toolbar.addWidget(self.btn_run)
        self.btn_clear = QPushButton('Clear')
        self.btn_clear.clicked.connect(self._clear)
        toolbar.addWidget(self.btn_clear)
        self.btn_export = QPushButton('Export CSV')
        self.btn_export.setToolTip(
            'Write the currently visible (post-filter) ghost table '
            'to a .csv file with the on-screen columns.')
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_export.setEnabled(False)
        toolbar.addWidget(self.btn_export)
        self.lbl_summary = QLabel('Idle.  Click "Run" to analyze.')
        self.lbl_summary.setStyleSheet(
            'color: #7a94b8; font-family: monospace;')
        toolbar.addWidget(self.lbl_summary, stretch=1)
        outer.addLayout(toolbar)

        outer.addWidget(self._build_filters())

        # Table on left, bar chart on right
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        self.table = self._build_table()
        splitter.addWidget(self.table)
        self.fig_bar = Figure(figsize=(4.0, 3.0), dpi=100,
                              facecolor='#0a0c10')
        self.canvas_bar = FigureCanvasQTAgg(self.fig_bar)
        self.canvas_bar.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.canvas_bar)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, stretch=2)

        # Budget label + per-path spot preview
        bottom = QHBoxLayout()
        self.lbl_budget = QLabel('Total visible stray light: -- ppm')
        self.lbl_budget.setStyleSheet(
            'color: #dde8f8; font-family: monospace; '
            'background: #0a0c10; padding: 4px;')
        bottom.addWidget(self.lbl_budget, stretch=1)
        self.fig_spot = Figure(figsize=(3.0, 2.5), dpi=100,
                               facecolor='#0a0c10')
        self.canvas_spot = FigureCanvasQTAgg(self.fig_spot)
        self.canvas_spot.setMinimumHeight(200)
        self.canvas_spot.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred)
        bottom.addWidget(self.canvas_spot, stretch=1)
        outer.addLayout(bottom)

        self._draw_empty_bar()
        self._draw_empty_spot()

    # ----------------------------------------------------------------
    # Layout builders
    # ----------------------------------------------------------------

    def _build_filters(self) -> QGroupBox:
        """Return the 4-knob filter group box."""
        group = QGroupBox('Filters')
        form = QFormLayout(group)
        form.setContentsMargins(8, 4, 8, 4)
        form.setSpacing(4)

        self.spin_max_bounces = QSpinBox()
        self.spin_max_bounces.setRange(1, 10)
        self.spin_max_bounces.setValue(2)
        self.spin_max_bounces.setToolTip(
            'Maximum bounces to display.  Library currently '
            'enumerates 2-bounce ghosts only; higher values are '
            'a no-op upper bound.  Default 2.')
        self.spin_max_bounces.valueChanged.connect(self._reapply_filters)
        form.addRow('Max bounces:', self.spin_max_bounces)

        self.spin_min_log_t = QDoubleSpinBox()
        self.spin_min_log_t.setRange(-30.0, 0.0)
        self.spin_min_log_t.setDecimals(2)
        self.spin_min_log_t.setSingleStep(0.5)
        self.spin_min_log_t.setValue(-10.0)
        self.spin_min_log_t.setSuffix(' (log10)')
        self.spin_min_log_t.setToolTip(
            'Reject ghosts whose Fresnel transmittance is below 10^x.')
        self.spin_min_log_t.valueChanged.connect(self._reapply_filters)
        form.addRow('Min transmittance:', self.spin_min_log_t)

        self.spin_min_eppm = QDoubleSpinBox()
        self.spin_min_eppm.setRange(0.0, 1e9)
        self.spin_min_eppm.setDecimals(4)
        self.spin_min_eppm.setSingleStep(0.1)
        self.spin_min_eppm.setValue(0.0)
        self.spin_min_eppm.setSuffix(' ppm')
        self.spin_min_eppm.setToolTip(
            'Hide ghosts whose energy fraction is below this ppm value.')
        self.spin_min_eppm.valueChanged.connect(self._reapply_filters)
        form.addRow('Min energy fraction:', self.spin_min_eppm)

        self.combo_sort = QComboBox()
        self.combo_sort.addItems([
            self.SORT_ENERGY, self.SORT_ORDER, self.SORT_T])
        self.combo_sort.setCurrentText(self.SORT_ENERGY)
        self.combo_sort.currentTextChanged.connect(self._reapply_filters)
        form.addRow('Sort by:', self.combo_sort)

        return group

    def _build_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            'path', 'order', 'transmittance',
            'energy fraction (ppm)', 'peak position (mm)',
        ])
        table.setSortingEnabled(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        hdr = table.horizontalHeader()
        hdr.setSectionResizeMode(self.COL_PATH, QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_ORDER,
                                 QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_T,
                                 QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_EPPM,
                                 QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_FOCUS_MM,
                                 QHeaderView.ResizeToContents)
        table.itemSelectionChanged.connect(self._on_selection_changed)
        return table

    # ----------------------------------------------------------------
    # Run path
    # ----------------------------------------------------------------

    def _run(self):
        try:
            pres = self.sm.to_prescription()
        except Exception as exc:
            self.lbl_summary.setText(
                f'Prescription export failed: {type(exc).__name__}.')
            return
        # Library helper _ghost_surfaces tolerates both schemas.
        if not (pres.get('surfaces') or pres.get('elements')):
            self.lbl_summary.setText('No surfaces in current system.')
            return

        ap = pres.get('aperture_diameter', 25.4e-3)
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.lbl_summary.setText('Running ghost analysis...')
        self._worker = _GhostWorker(
            prescription=pres, wavelength=self.sm.wavelength_m,
            semi_aperture=(ap / 2.0 if ap else None),
            n_rays=21, top_n=20)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, result):
        self.btn_run.setEnabled(True)
        self._worker = None
        if 'error' in result:
            self.lbl_summary.setText(f'Error: {result["error"]}')
            try:
                from .diagnostics import diag
                diag.report('ghost-dock', RuntimeError(result['error']),
                            context='non_sequential_stray_light')
            except Exception:
                pass
            return

        self._all_ghosts = list(result.get('ghosts') or [])
        self._n_surfaces = int(result.get('n_surfaces') or 0)
        report = result.get('report') or {}
        self._stray_light_fraction = float(
            report.get('stray_light_fraction', 0.0))
        if not self._all_ghosts:
            self.lbl_summary.setText('No ghost paths found.')
            return

        total = float(sum(g.get('intensity', 0.0)
                          for g in self._all_ghosts))
        denom = total if total > 0 else 1.0
        for g in self._all_ghosts:
            g['energy_fraction'] = float(g.get('intensity', 0.0)) / denom
            g['energy_ppm'] = g['energy_fraction'] * 1e6
            g['order'] = 2  # library currently enumerates 2-bounce only

        self._reapply_filters()
        self.btn_export.setEnabled(True)
        self.lbl_summary.setText(
            f'{len(self._all_ghosts)} ghosts; '
            f'N_surfaces={self._n_surfaces}; '
            f'total flux UB = {total:.3e}.')

    def _clear(self):
        self._all_ghosts = []
        self._filtered_ghosts = []
        self._stray_light_fraction = 0.0
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.table.setSortingEnabled(True)
        self.btn_export.setEnabled(False)
        self.lbl_summary.setText('Idle.  Click "Run" to analyze.')
        self.lbl_budget.setText('Total visible stray light: -- ppm')
        self._draw_empty_bar()
        self._draw_empty_spot()

    # ----------------------------------------------------------------
    # Filter + sort + repopulate
    # ----------------------------------------------------------------

    def _reapply_filters(self):
        if not self._all_ghosts:
            return
        max_bounces = int(self.spin_max_bounces.value())
        min_t = 10.0 ** float(self.spin_min_log_t.value())
        min_eppm = float(self.spin_min_eppm.value())
        sort_key = self.combo_sort.currentText()

        kept = []
        for g in self._all_ghosts:
            order = int(g.get('order', 2))
            if order > max_bounces:
                continue
            t_val = float(g.get('intensity', 0.0))
            if t_val < min_t:
                continue
            eppm = float(g.get('energy_ppm', 0.0))
            if eppm < min_eppm:
                continue
            kept.append(g)

        if sort_key == self.SORT_ENERGY:
            kept.sort(key=lambda g: -g.get('energy_fraction', 0.0))
        elif sort_key == self.SORT_ORDER:
            kept.sort(key=lambda g: (g.get('order', 2),
                                     -g.get('energy_fraction', 0.0)))
        elif sort_key == self.SORT_T:
            kept.sort(key=lambda g: -g.get('intensity', 0.0))

        self._filtered_ghosts = kept
        self._populate_table()
        self._draw_bar_chart()
        self._update_budget()

    def _populate_table(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._filtered_ghosts))
        for row, g in enumerate(self._filtered_ghosts):
            path = g.get('path')
            if isinstance(path, tuple) and len(path) == 2:
                path_str = f'S{path[0]} -> S{path[1]} -> Image'
            else:
                path_str = str(path)
            t_val = float(g.get('intensity', 0.0))
            order = int(g.get('order', 2))
            eppm = float(g.get('energy_ppm', 0.0))
            focus_mm = float(g.get('focus_z_estimate', 0.0)) * 1e3

            self._set_text_item(row, self.COL_PATH, path_str)
            self._set_numeric_item(row, self.COL_ORDER, order, '{:d}')
            self._set_numeric_item(row, self.COL_T, t_val, '{:.4e}')
            self._set_numeric_item(row, self.COL_EPPM, eppm, '{:.4f}')
            self._set_numeric_item(row, self.COL_FOCUS_MM, focus_mm,
                                   '{:.3f}')
        self.table.setSortingEnabled(True)

    def _set_text_item(self, row: int, col: int, text: str):
        item = QTableWidgetItem(text)
        self.table.setItem(row, col, item)

    def _set_numeric_item(self, row: int, col: int, value: float,
                          fmt: str):
        item = QTableWidgetItem()
        item.setData(Qt.DisplayRole, value)
        item.setText(fmt.format(value))
        item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
        self.table.setItem(row, col, item)

    # ----------------------------------------------------------------
    # Bar chart
    # ----------------------------------------------------------------

    def _draw_empty_bar(self):
        self.fig_bar.clear()
        ax = self.fig_bar.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Top-N ghost energy bar chart\n'
                '(populated after Run)',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace',
                fontsize=9)
        ax.tick_params(colors='#7a94b8')
        for spine in ax.spines.values():
            spine.set_color('#334054')
        self.canvas_bar.draw_idle()

    def _draw_bar_chart(self):
        self.fig_bar.clear()
        ax = self.fig_bar.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        items = self._filtered_ghosts[:10]
        if not items:
            ax.text(0.5, 0.5, 'No ghosts pass current filters',
                    color='#ff6b35', ha='center', va='center',
                    transform=ax.transAxes, fontfamily='monospace',
                    fontsize=9)
        else:
            labels, values = [], []
            for g in items:
                p = g.get('path')
                labels.append(f'S{p[0]}->S{p[1]}'
                              if isinstance(p, tuple) and len(p) == 2
                              else str(p))
                values.append(float(g.get('energy_ppm', 0.0)))
            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, values, color='#5cb8ff',
                           edgecolor='#7a94b8', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8,
                               fontfamily='monospace', color='#dde8f8')
            ax.invert_yaxis()  # brightest at top
            ax.set_xlabel('Energy fraction (ppm)',
                          color='#dde8f8', fontsize=8,
                          fontfamily='monospace')
            ax.set_title(f'Top {len(items)} ghosts',
                         color='#dde8f8', fontsize=9,
                         fontfamily='monospace')
            if max(values) > 0:
                ax.set_xscale('log')
            for b, v in zip(bars, values):
                ax.text(v, b.get_y() + b.get_height() / 2,
                        f'  {v:.2f}', color='#7a94b8',
                        va='center', fontsize=7,
                        fontfamily='monospace')
        ax.tick_params(colors='#7a94b8', labelsize=8)
        ax.grid(True, axis='x', alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color('#334054')
        self.fig_bar.tight_layout()
        self.canvas_bar.draw_idle()

    # ----------------------------------------------------------------
    # Spot diagram preview
    # ----------------------------------------------------------------

    def _draw_empty_spot(self):
        self.fig_spot.clear()
        ax = self.fig_spot.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Select a row to preview\nthe ghost spot diagram',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace',
                fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#334054')
        self.canvas_spot.draw_idle()

    def _on_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self._draw_empty_spot()
            return
        # Map the visible (sorted) row index back to the underlying
        # filtered-ghost item using the path string.
        row = rows[0].row()
        path_item = self.table.item(row, self.COL_PATH)
        if path_item is None:
            return
        path_str = path_item.text()
        ghost = self._lookup_ghost_by_path_str(path_str)
        if ghost is None:
            return
        self._draw_spot_preview(ghost)

    def _lookup_ghost_by_path_str(
            self, path_str: str) -> Optional[Dict[str, Any]]:
        for g in self._filtered_ghosts:
            p = g.get('path')
            if isinstance(p, tuple) and len(p) == 2:
                if f'S{p[0]} -> S{p[1]} -> Image' == path_str:
                    return g
        return None

    def _draw_spot_preview(self, ghost):
        """Render the per-path ghost spot diagram for the selected row.

        v5.4 Phase 5 (audit P2-C closure): replaced the synthesised
        Gaussian relative-magnitude visual with a REAL per-path
        retrace via ``analysis.ghost.retrace_ghost_path``.  The dock
        now displays the actual ray-cloud histogram at the image
        plane along with the centroid + RMS legend.

        Falls back to the legacy Gaussian if the retrace raises (e.g.
        an unsupported prescription schema or a numerically-degenerate
        path).  The fallback path keeps the dock usable while still
        clearly labelling the fallback in the title.
        """
        self.fig_spot.clear()
        ax = self.fig_spot.add_subplot(111)
        ax.set_facecolor('#0a0c10')

        path = ghost.get('path')
        eppm = float(ghost.get('energy_ppm', 0.0))

        # Attempt a real retrace.  All inputs come from the dock's
        # current model so we don't need to re-validate the
        # prescription here.
        retrace_ok = False
        rays_xy = None
        rms_mm = float('nan')
        peak_xy_mm = (0.0, 0.0)
        try:
            if isinstance(path, tuple) and len(path) == 2:
                from ..analysis.ghost import (
                    _path_from_pair, retrace_ghost_path)
                pres = self.sm.to_prescription()
                ap = pres.get('aperture_diameter', 25.4e-3)
                semi_ap = (ap / 2.0) if ap else 12.7e-3
                # Use the system's image-plane z if the prescription
                # carries one, else default to None (library falls
                # back to the last surface's thickness).
                image_z = pres.get('image_distance')
                expl_path = _path_from_pair(
                    self._n_surfaces, int(path[0]), int(path[1]))
                rr = retrace_ghost_path(
                    pres, expl_path,
                    wavelength=self.sm.wavelength_m,
                    semi_aperture=float(semi_ap),
                    n_rays=64,
                    image_plane_z=image_z,
                )
                rays_xy = np.asarray(rr['rays_image_plane'])
                rms_mm = float(rr['rms_radius_mm'])
                peak_xy_mm = tuple(rr['peak_xy_mm'])
                retrace_ok = rays_xy is not None and rays_xy.shape[0] >= 2
        except Exception:
            retrace_ok = False

        if retrace_ok and rays_xy.shape[0] >= 2:
            # v5.4 Phase 5: real per-path retrace -- ray-cloud scatter
            # plus a 2-D histogram underlay for density.  Coordinates
            # in millimetres relative to the image-plane vertex.
            x_mm = rays_xy[:, 0] * 1e3
            y_mm = rays_xy[:, 1] * 1e3
            # Choose extent: 3-sigma RMS, with a minimum window so we
            # never zoom into a single-pixel artefact.
            half = max(3.0 * rms_mm, 1e-3)
            extent = (-half, half, -half, half)
            try:
                ax.hist2d(x_mm, y_mm,
                          bins=min(48, max(8, int(np.sqrt(len(x_mm))))),
                          range=[[-half, half], [-half, half]],
                          cmap='inferno')
            except (ValueError, RuntimeError):
                # hist2d can choke on edge cases (all rays at one
                # point); fall through to a scatter.
                pass
            ax.scatter(x_mm, y_mm, s=4, alpha=0.4, c='#5cb8ff',
                       edgecolors='none')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            title = (f'Ghost S{path[0]}->S{path[1]} '
                     f'({eppm:.3f} ppm, RMS {rms_mm:.3f} mm)')
            ax.set_xlabel('x [mm]', color='#dde8f8', fontsize=8,
                          fontfamily='monospace')
            ax.set_ylabel('y [mm]', color='#dde8f8', fontsize=8,
                          fontfamily='monospace')
        else:
            # Fallback: synthesised Gaussian (legacy behaviour) so the
            # dock stays usable on prescriptions where the retrace
            # raises.
            n = 64
            x = np.linspace(-1.0, 1.0, n)
            xx, yy = np.meshgrid(x, x)
            rr = np.sqrt(xx * xx + yy * yy)
            f_est = float(ghost.get('focus_z_estimate', 0.05))
            sigma = max(min(0.04 / max(abs(f_est) * 10, 1e-6), 0.6),
                         0.05)
            spot = (np.exp(-(rr ** 2) / (2.0 * sigma * sigma))
                    * (eppm + 1e-12))
            im = ax.imshow(np.log10(spot + spot.max() * 1e-6 + 1e-30),
                           cmap='inferno', origin='lower',
                           extent=(-1.0, 1.0, -1.0, 1.0))
            title = (f'Ghost S{path[0]}->S{path[1]} (fallback) '
                     if isinstance(path, tuple) else 'Ghost spot ')
            title += f'({eppm:.3f} ppm)'
            try:
                self.fig_spot.colorbar(im, ax=ax, fraction=0.046,
                                       shrink=0.8, pad=0.04)
            except Exception:
                pass
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(title, color='#dde8f8', fontsize=8,
                     fontfamily='monospace')
        ax.tick_params(colors='#7a94b8', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#334054')
        self.fig_spot.tight_layout()
        self.canvas_spot.draw_idle()

    # ----------------------------------------------------------------
    # Budget label
    # ----------------------------------------------------------------

    def _update_budget(self):
        if not self._filtered_ghosts:
            self.lbl_budget.setText('Total visible stray light: -- ppm')
            return
        total_ppm = float(sum(g.get('energy_ppm', 0.0)
                              for g in self._filtered_ghosts))
        slf_ppm = self._stray_light_fraction * 1e6
        self.lbl_budget.setText(
            f'Total visible stray light: {total_ppm:.3f} ppm '
            f'(filtered) | conservative full budget: {slf_ppm:.3f} ppm '
            f'| N_visible={len(self._filtered_ghosts)} / '
            f'{len(self._all_ghosts)}')

    # ----------------------------------------------------------------
    # CSV export
    # ----------------------------------------------------------------

    def _export_csv(self):
        if not self._filtered_ghosts:
            QMessageBox.information(
                self, 'Ghost export',
                'No filtered ghost paths to export.  Run the analysis '
                'and adjust filters first.')
            return
        default_path = os.path.join(os.path.expanduser('~'),
                                    'ghost_paths.csv')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export ghost table', default_path,
            'CSV files (*.csv);;All files (*.*)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='ascii', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    'path', 'i', 'j', 'order', 'transmittance',
                    'energy_fraction_ppm', 'peak_position_mm',
                ])
                for g in self._filtered_ghosts:
                    p = g.get('path')
                    if isinstance(p, tuple) and len(p) == 2:
                        path_str = f'S{p[0]} -> S{p[1]} -> Image'
                        i_idx, j_idx = int(p[0]), int(p[1])
                    else:
                        path_str, i_idx, j_idx = str(p), -1, -1
                    writer.writerow([
                        path_str, i_idx, j_idx, int(g.get('order', 2)),
                        f'{float(g.get("intensity", 0.0)):.6e}',
                        f'{float(g.get("energy_ppm", 0.0)):.6f}',
                        f'{float(g.get("focus_z_estimate", 0.0))*1e3:.4f}',
                    ])
        except OSError as exc:
            QMessageBox.warning(
                self, 'Ghost export',
                f'Could not write {path}:\n{type(exc).__name__}: {exc}')
            return
        self.lbl_summary.setText(
            f'Exported {len(self._filtered_ghosts)} paths to '
            f'{os.path.basename(path)}.')
