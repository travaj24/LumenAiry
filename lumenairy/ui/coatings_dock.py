"""
Thin-film coating-stack design dock.

v5.4 audit AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 item P2-D.  The
library ships ``lumenairy.elements.coatings`` (re-exported at the
top level) with three primitives:

  * ``coating_reflectance(layers, wavelengths, angle, n_substrate,
    n_ambient, polarization)``  -- transfer-matrix multilayer solver
    returning ``(R, T, phase_r)`` arrays.
  * ``quarter_wave_ar(n_substrate, wavelength_center)``  -- single-
    layer quarter-wave AR design helper.
  * ``broadband_ar_v_coat(n_substrate, wavelength_center)``  -- two-
    layer V-coat AR design helper.

The v5.3.2 GUI exposes none of them.  This dock surfaces the
canonical stack-design workflow without dropping to Jupyter:

  * Interactive layer table (material + n + thickness).
  * Substrate / incident-medium / wavelength-sweep / AOI /
    polarisation controls.
  * R(lambda) plot on an embedded matplotlib canvas.
  * Quick-template buttons that populate the table from the library
    design helpers (single-layer MgF2 QWL, broadband AR V-coat,
    HR Bragg stack).
  * R metrics summary (R_mean, R_min, R_max, lambda_min).

Limitation: the library's ``coating_reflectance`` does not consume
material names -- it wants explicit refractive indices.  No material
lookup exists in ``elements.coatings``, so this dock hardcodes the
common-coating short list documented in MATERIAL_INDEX below.  When
the user picks 'Custom n', the table's ``n`` cell is edited directly
and the material column reads 'Custom'.

Author: Andrew Traverso -- v5.4 coatings dock.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:  # pragma: no cover - matplotlib is a hard dep
    HAS_MPL = False


# =============================================================================
# Material database -- common thin-film coating indices.
#
# v5.4 Phase 5: library now ships COATING_MATERIAL_REGISTRY in
# ``lumenairy.elements.coatings``, so the dock consumes the canonical
# registry instead of carrying its own hardcoded short list.  Each
# material's reference index (at its ``ref_wavelength``, typically
# 550 nm) is materialised into ``MATERIAL_INDEX`` at module-load
# time for back-compat with the dock's existing ``MATERIAL_INDEX[
# name]`` call sites.  ``get_coating_material_index(material,
# wavelength)`` is the canonical accessor for wavelength-aware
# lookups -- the four most common AR/HR materials (MgF2, SiO2, TiO2,
# Ta2O5) carry Sellmeier coefficients, the rest fall back to a flat
# index.  Documented limitation: the dock's table view still treats
# n as dispersionless within a single layer entry (the user can
# override by editing the ``n`` cell directly).
# =============================================================================

from lumenairy.elements.coatings import (
    COATING_MATERIAL_REGISTRY,
    get_coating_material_index,
)

MATERIAL_INDEX: Dict[str, float] = {
    name: float(get_coating_material_index(name, entry['ref_wavelength']))
    for name, entry in sorted(COATING_MATERIAL_REGISTRY.items())
}

SUBSTRATE_INDEX: Dict[str, float] = {
    'BK7':          1.515,
    'Fused Silica': 1.460,
    'CaF2':         1.434,
    'ZnSe':         2.403,
    'Si':           3.882,
    'Sapphire':     1.768,
    'Custom n':     1.500,  # placeholder -- replaced by spin_n_sub
}

AMBIENT_INDEX: Dict[str, float] = {
    'air':      1.0003,
    'vacuum':   1.0000,
    'Custom n': 1.0000,
}

POLARIZATION_OPTIONS: List[Tuple[str, str]] = [
    ('Unpolarized (avg)', 'avg'),
    ('s-pol',             's'),
    ('p-pol',             'p'),
]


# =============================================================================
# CoatingsDock -- the dock widget.
# =============================================================================


class CoatingsDock(QWidget):
    """Interactive thin-film coating-stack designer.

    Wraps :func:`lumenairy.coating_reflectance` for forward analysis
    and :func:`lumenairy.quarter_wave_ar` /
    :func:`lumenairy.broadband_ar_v_coat` for quick-template design.

    Parameters
    ----------
    model : object, optional
        SystemModel reference.  Currently unused (the coating
        designer is standalone) but accepted for parity with other
        docks and forward-compatible with a future
        "apply coating to surface" hook.
    parent : QWidget, optional
        Parent widget; the main window passes a QDockWidget here.
    """

    # Column indices for the stack-editor table.
    COL_INDEX   = 0
    COL_MAT     = 1
    COL_N       = 2
    COL_THICK   = 3

    def __init__(self, model=None, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self._last_R: Optional[np.ndarray] = None
        self._last_wavelengths_m: Optional[np.ndarray] = None
        self._suspend_table_signals = False
        self._build_ui()
        self._draw_empty()

    # --------------------------------------------------------------- UI ---

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: controls + table ----
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)

        # Substrate / ambient panel
        media_box = QGroupBox('Media')
        media_form = QFormLayout(media_box)
        self.combo_substrate = QComboBox()
        self.combo_substrate.addItems(list(SUBSTRATE_INDEX.keys()))
        self.combo_substrate.setCurrentText('BK7')
        self.combo_substrate.currentIndexChanged.connect(
            self._on_substrate_changed)
        media_form.addRow('Substrate:', self.combo_substrate)
        self.spin_n_sub = QDoubleSpinBox()
        self.spin_n_sub.setRange(1.0, 6.0)
        self.spin_n_sub.setDecimals(4)
        self.spin_n_sub.setSingleStep(0.01)
        self.spin_n_sub.setValue(SUBSTRATE_INDEX['BK7'])
        self.spin_n_sub.setToolTip(
            'Substrate refractive index.  Auto-filled from the '
            'preset; editable when "Custom n" is selected.')
        media_form.addRow('Substrate n:', self.spin_n_sub)

        self.combo_ambient = QComboBox()
        self.combo_ambient.addItems(list(AMBIENT_INDEX.keys()))
        self.combo_ambient.setCurrentText('air')
        self.combo_ambient.currentIndexChanged.connect(
            self._on_ambient_changed)
        media_form.addRow('Incident medium:', self.combo_ambient)
        self.spin_n_amb = QDoubleSpinBox()
        self.spin_n_amb.setRange(1.0, 6.0)
        self.spin_n_amb.setDecimals(4)
        self.spin_n_amb.setSingleStep(0.01)
        self.spin_n_amb.setValue(AMBIENT_INDEX['air'])
        self.spin_n_amb.setToolTip(
            'Incident-medium refractive index.')
        media_form.addRow('Ambient n:', self.spin_n_amb)
        left_v.addWidget(media_box)

        # Sweep / AOI / polarisation panel
        sweep_box = QGroupBox('Sweep / geometry')
        sweep_form = QFormLayout(sweep_box)
        self.spin_wl_min = QDoubleSpinBox()
        self.spin_wl_min.setRange(50.0, 50000.0)
        self.spin_wl_min.setValue(400.0)
        self.spin_wl_min.setDecimals(2)
        self.spin_wl_min.setSuffix(' nm')
        sweep_form.addRow('Wavelength min:', self.spin_wl_min)
        self.spin_wl_max = QDoubleSpinBox()
        self.spin_wl_max.setRange(50.0, 50000.0)
        self.spin_wl_max.setValue(800.0)
        self.spin_wl_max.setDecimals(2)
        self.spin_wl_max.setSuffix(' nm')
        sweep_form.addRow('Wavelength max:', self.spin_wl_max)
        self.spin_wl_n = QSpinBox()
        self.spin_wl_n.setRange(2, 4096)
        self.spin_wl_n.setValue(100)
        sweep_form.addRow('# samples:', self.spin_wl_n)
        self.spin_aoi = QDoubleSpinBox()
        self.spin_aoi.setRange(0.0, 89.9)
        self.spin_aoi.setValue(0.0)
        self.spin_aoi.setDecimals(2)
        self.spin_aoi.setSuffix(' deg')
        self.spin_aoi.setToolTip(
            'Angle of incidence in the ambient medium (degrees).')
        sweep_form.addRow('Angle of incidence:', self.spin_aoi)
        self.combo_pol = QComboBox()
        for label, _ in POLARIZATION_OPTIONS:
            self.combo_pol.addItem(label)
        self.combo_pol.setCurrentIndex(0)
        sweep_form.addRow('Polarisation:', self.combo_pol)
        left_v.addWidget(sweep_box)

        # Quick-template panel
        tpl_box = QGroupBox('Quick templates')
        tpl_layout = QVBoxLayout(tpl_box)
        self.btn_tpl_mgf2 = QPushButton(
            'Single-layer MgF2 QWL @ 550 nm')
        self.btn_tpl_mgf2.setToolTip(
            'Populate the stack with one MgF2 layer at quarter-wave '
            'optical thickness for 550 nm.')
        self.btn_tpl_mgf2.clicked.connect(self._populate_mgf2_qwl)
        tpl_layout.addWidget(self.btn_tpl_mgf2)
        self.btn_tpl_vcoat = QPushButton(
            'Broadband AR V-coat (450-650 nm)')
        self.btn_tpl_vcoat.setToolTip(
            'Call broadband_ar_v_coat() at the band center and '
            'populate the stack with the resulting two layers.')
        self.btn_tpl_vcoat.clicked.connect(self._populate_vcoat)
        tpl_layout.addWidget(self.btn_tpl_vcoat)
        self.btn_tpl_bragg = QPushButton(
            'HR Bragg stack (TiO2/SiO2 N pairs)')
        self.btn_tpl_bragg.setToolTip(
            'Populate the stack with N quarter-wave bilayer pairs '
            '(high-low-high-low ...) using TiO2 / SiO2 at the '
            'centre wavelength.  N is taken from the spinbox below.')
        self.btn_tpl_bragg.clicked.connect(self._populate_bragg)
        bragg_row = QHBoxLayout()
        bragg_row.addWidget(QLabel('Pairs:'))
        self.spin_bragg_pairs = QSpinBox()
        self.spin_bragg_pairs.setRange(1, 64)
        self.spin_bragg_pairs.setValue(8)
        bragg_row.addWidget(self.spin_bragg_pairs)
        bragg_row.addWidget(QLabel('Center:'))
        self.spin_bragg_center = QDoubleSpinBox()
        self.spin_bragg_center.setRange(100.0, 20000.0)
        self.spin_bragg_center.setValue(550.0)
        self.spin_bragg_center.setSuffix(' nm')
        bragg_row.addWidget(self.spin_bragg_center)
        bragg_row.addStretch()
        tpl_layout.addLayout(bragg_row)
        tpl_layout.addWidget(self.btn_tpl_bragg)
        left_v.addWidget(tpl_box)

        # Stack-editor table
        stack_box = QGroupBox('Layer stack (outer -> substrate)')
        stack_v = QVBoxLayout(stack_box)
        row_buttons = QHBoxLayout()
        self.btn_add = QPushButton('+ Add layer')
        self.btn_add.clicked.connect(self._add_layer)
        row_buttons.addWidget(self.btn_add)
        self.btn_remove = QPushButton('- Remove selected')
        self.btn_remove.clicked.connect(self._remove_layer)
        row_buttons.addWidget(self.btn_remove)
        self.btn_clear = QPushButton('Clear all')
        self.btn_clear.clicked.connect(self._clear_layers)
        row_buttons.addWidget(self.btn_clear)
        row_buttons.addStretch()
        stack_v.addLayout(row_buttons)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            '#', 'Material', 'n', 'Thickness (nm)'])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(self.COL_INDEX, 28)
        self.table.setColumnWidth(self.COL_MAT, 110)
        self.table.setColumnWidth(self.COL_N, 70)
        self.table.itemChanged.connect(self._on_table_item_changed)
        stack_v.addWidget(self.table, stretch=1)
        left_v.addWidget(stack_box, stretch=1)

        # Run button
        self.btn_run = QPushButton('Compute R(lambda)')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._on_run)
        left_v.addWidget(self.btn_run)

        splitter.addWidget(left)

        # ---- Right: plot + metrics ----
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(6)

        if HAS_MPL:
            self.fig = Figure(figsize=(6.0, 3.6), tight_layout=True,
                              facecolor='#0a0c10')
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.toolbar = NavigationToolbar(self.canvas, self)
            right_v.addWidget(self.toolbar)
            right_v.addWidget(self.canvas, stretch=1)
        else:
            warn = QLabel('matplotlib backend unavailable -- '
                          'plots disabled.')
            warn.setAlignment(Qt.AlignCenter)
            right_v.addWidget(warn, stretch=1)

        metrics_box = QGroupBox('R metrics')
        metrics_v = QVBoxLayout(metrics_box)
        self.lbl_metrics = QLabel('No computation yet.')
        self.lbl_metrics.setFont(QFont('Consolas', 10))
        self.lbl_metrics.setStyleSheet(
            'QLabel{color:#cad6e8;background:#0a0c10;padding:4px;'
            'border:1px solid #334054}')
        self.lbl_metrics.setWordWrap(True)
        metrics_v.addWidget(self.lbl_metrics)
        right_v.addWidget(metrics_box)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(80)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        right_v.addWidget(self.summary)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 720])
        outer.addWidget(splitter)

    # ----------------------------------------------- table row helpers ---

    def _make_index_item(self, row: int) -> QTableWidgetItem:
        item = QTableWidgetItem(str(row + 1))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    def _make_material_combo(self, material: str = 'MgF2') -> QComboBox:
        combo = QComboBox()
        combo.addItems(list(MATERIAL_INDEX.keys()))
        combo.addItem('Custom')
        idx = combo.findText(material)
        combo.setCurrentIndex(idx if idx >= 0 else combo.count() - 1)
        combo.currentIndexChanged.connect(
            lambda _: self._on_material_combo_changed(combo))
        return combo

    def _set_index_column(self) -> None:
        """Renumber the # column after add / remove / clear."""
        self._suspend_table_signals = True
        try:
            for r in range(self.table.rowCount()):
                item = self.table.item(r, self.COL_INDEX)
                if item is None:
                    self.table.setItem(
                        r, self.COL_INDEX, self._make_index_item(r))
                else:
                    item.setText(str(r + 1))
        finally:
            self._suspend_table_signals = False

    def _append_row(self, material: str, n: float,
                    thickness_nm: float) -> None:
        """Internal helper: add one row with given material / n / d."""
        self._suspend_table_signals = True
        try:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(
                r, self.COL_INDEX, self._make_index_item(r))
            self.table.setCellWidget(
                r, self.COL_MAT, self._make_material_combo(material))
            n_item = QTableWidgetItem(f'{float(n):.4f}')
            n_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, self.COL_N, n_item)
            d_item = QTableWidgetItem(f'{float(thickness_nm):.3f}')
            d_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, self.COL_THICK, d_item)
        finally:
            self._suspend_table_signals = False

    # ----------------------------------------------- public-ish slots ---

    def _add_layer(self) -> None:
        """Add a default MgF2 layer to the end of the stack."""
        self._append_row(
            material='MgF2',
            n=MATERIAL_INDEX['MgF2'],
            thickness_nm=99.6,  # ~lambda/(4n) for 550 nm / 1.38
        )
        self._set_index_column()

    def _remove_layer(self) -> None:
        """Remove the currently-selected row from the table."""
        r = self.table.currentRow()
        if r < 0 or r >= self.table.rowCount():
            return
        self.table.removeRow(r)
        self._set_index_column()

    def _clear_layers(self) -> None:
        """Remove every row from the stack editor."""
        self._suspend_table_signals = True
        try:
            self.table.setRowCount(0)
        finally:
            self._suspend_table_signals = False

    def _set_substrate(self, idx: int) -> None:
        """Programmatic substrate change (named per spec).

        Equivalent to selecting an item in ``combo_substrate``; the
        index is interpreted against ``SUBSTRATE_INDEX``'s ordered
        keys.  Out-of-range indices are silently ignored.
        """
        if 0 <= idx < self.combo_substrate.count():
            self.combo_substrate.setCurrentIndex(idx)

    def _on_substrate_changed(self) -> None:
        name = self.combo_substrate.currentText()
        n = SUBSTRATE_INDEX.get(name, 1.5)
        if name != 'Custom n':
            self.spin_n_sub.setValue(n)
            self.spin_n_sub.setEnabled(False)
        else:
            self.spin_n_sub.setEnabled(True)

    def _on_ambient_changed(self) -> None:
        name = self.combo_ambient.currentText()
        n = AMBIENT_INDEX.get(name, 1.0)
        if name != 'Custom n':
            self.spin_n_amb.setValue(n)
            self.spin_n_amb.setEnabled(False)
        else:
            self.spin_n_amb.setEnabled(True)

    def _on_material_combo_changed(self, combo: QComboBox) -> None:
        """When the user changes a layer's material, auto-update n."""
        if self._suspend_table_signals:
            return
        # Find the row that owns this combo.
        row = -1
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, self.COL_MAT) is combo:
                row = r
                break
        if row < 0:
            return
        name = combo.currentText()
        if name == 'Custom':
            return  # leave n cell alone for manual editing
        n = MATERIAL_INDEX.get(name)
        if n is None:
            return
        self._suspend_table_signals = True
        try:
            n_item = self.table.item(row, self.COL_N)
            if n_item is None:
                n_item = QTableWidgetItem()
                n_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, self.COL_N, n_item)
            n_item.setText(f'{n:.4f}')
        finally:
            self._suspend_table_signals = False

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        """Validate user edits to n / thickness cells."""
        if self._suspend_table_signals:
            return
        col = item.column()
        if col not in (self.COL_N, self.COL_THICK):
            return
        text = item.text().strip()
        try:
            val = float(text)
        except (TypeError, ValueError):
            self._suspend_table_signals = True
            try:
                item.setText('1.0000' if col == self.COL_N else '100.000')
            finally:
                self._suspend_table_signals = False
            return
        if col == self.COL_N and val < 1.0:
            val = 1.0
        if col == self.COL_THICK and val < 0.0:
            val = 0.0
        self._suspend_table_signals = True
        try:
            item.setText(
                f'{val:.4f}' if col == self.COL_N else f'{val:.3f}')
        finally:
            self._suspend_table_signals = False

    # ------------------------------------------------ quick templates ---

    def _populate_mgf2_qwl(self) -> None:
        """Populate the stack from quarter_wave_ar() at 550 nm."""
        try:
            import lumenairy as la
        except ImportError as e:
            self.summary.setPlainText(
                f'Cannot import lumenairy: {type(e).__name__}: {e}')
            return
        # The library helper ignores the n_substrate input for the
        # design index (it computes sqrt(n_sub) ideal) -- we drive it
        # with the current dock setting for parity with the user's
        # selected substrate.
        n_sub = float(self.spin_n_sub.value())
        wl_center_m = 550e-9
        layers = la.quarter_wave_ar(n_sub, wl_center_m)
        # Override the helper's ideal sqrt(n_sub) with the canonical
        # MgF2 index (1.38) so the displayed material name is honest.
        # Recompute thickness for the same QWL condition.
        n_layer = MATERIAL_INDEX['MgF2']
        d_nm = (wl_center_m / (4.0 * n_layer)) * 1e9
        self._clear_layers()
        self._append_row('MgF2', n_layer, d_nm)
        self._set_index_column()
        self.summary.setPlainText(
            f'Loaded single-layer MgF2 QWL at 550 nm '
            f'(n={n_layer:.3f}, d={d_nm:.2f} nm).  '
            f'Library returned {len(layers)} layer(s).')

    def _populate_vcoat(self) -> None:
        """Populate the stack from broadband_ar_v_coat() at band center."""
        try:
            import lumenairy as la
        except ImportError as e:
            self.summary.setPlainText(
                f'Cannot import lumenairy: {type(e).__name__}: {e}')
            return
        n_sub = float(self.spin_n_sub.value())
        wl_center_m = 0.5 * (450e-9 + 650e-9)
        layers = la.broadband_ar_v_coat(n_sub, wl_center_m)
        self._clear_layers()
        # The V-coat helper hardcodes n_L = 1.38 (MgF2-like) and
        # n_H = 2.3 (TiO2-like).  Map those to material names so the
        # GUI display lines up with the library's intent.
        material_for_n = {1.38: 'MgF2', 2.30: 'TiO2'}
        for (n_layer, d_m) in layers:
            n_val = float(np.real(n_layer))
            mat = material_for_n.get(round(n_val, 2), 'Custom')
            self._append_row(mat, n_val, d_m * 1e9)
        self._set_index_column()
        self.summary.setPlainText(
            f'Loaded broadband AR V-coat (450-650 nm) via '
            f'broadband_ar_v_coat() -- {len(layers)} layers, '
            f'n_sub={n_sub:.3f}, center={wl_center_m * 1e9:.1f} nm.')

    def _populate_bragg(self) -> None:
        """Populate the stack with N quarter-wave bilayer pairs."""
        n_pairs = int(self.spin_bragg_pairs.value())
        wl_center_nm = float(self.spin_bragg_center.value())
        wl_center_m = wl_center_nm * 1e-9
        n_H = MATERIAL_INDEX['TiO2']
        n_L = MATERIAL_INDEX['SiO2']
        d_H_nm = (wl_center_m / (4.0 * n_H)) * 1e9
        d_L_nm = (wl_center_m / (4.0 * n_L)) * 1e9
        self._clear_layers()
        # HR Bragg stack starts with the high-index layer outermost
        # to maximize reflectance.
        for _ in range(n_pairs):
            self._append_row('TiO2', n_H, d_H_nm)
            self._append_row('SiO2', n_L, d_L_nm)
        self._set_index_column()
        self.summary.setPlainText(
            f'Loaded HR Bragg stack: {n_pairs} TiO2/SiO2 pairs '
            f'({2 * n_pairs} layers) at center {wl_center_nm:.1f} nm.')

    # ------------------------------------------------ compute / draw ---

    def _collect_stack(self) -> List[Tuple[float, float]]:
        """Read the layer table into a ``[(n, d_meters), ...]`` list."""
        layers: List[Tuple[float, float]] = []
        for r in range(self.table.rowCount()):
            try:
                n_item = self.table.item(r, self.COL_N)
                d_item = self.table.item(r, self.COL_THICK)
                if n_item is None or d_item is None:
                    continue
                n_val = float(n_item.text())
                d_nm = float(d_item.text())
            except (TypeError, ValueError):
                continue
            d_m = d_nm * 1e-9
            layers.append((n_val, d_m))
        return layers

    def _collect_inputs(self) -> Dict[str, Any]:
        """Snapshot every UI widget's value into a plain dict."""
        pol_label = self.combo_pol.currentText()
        pol_key = 'avg'
        for label, key in POLARIZATION_OPTIONS:
            if label == pol_label:
                pol_key = key
                break
        return {
            'layers': self._collect_stack(),
            'wl_min_m': float(self.spin_wl_min.value()) * 1e-9,
            'wl_max_m': float(self.spin_wl_max.value()) * 1e-9,
            'wl_n':     int(self.spin_wl_n.value()),
            'angle_rad': float(self.spin_aoi.value()) * np.pi / 180.0,
            'n_substrate': float(self.spin_n_sub.value()),
            'n_ambient':   float(self.spin_n_amb.value()),
            'polarization': pol_key,
        }

    def _on_run(self) -> None:
        """Compute R(lambda) and refresh the plot + metrics."""
        try:
            import lumenairy as la
        except ImportError as e:
            self.summary.setPlainText(
                f'Cannot import lumenairy: {type(e).__name__}: {e}')
            return
        inputs = self._collect_inputs()
        if inputs['wl_max_m'] <= inputs['wl_min_m']:
            self.summary.setPlainText(
                'Wavelength max must exceed wavelength min.')
            return
        wavelengths = np.linspace(
            inputs['wl_min_m'], inputs['wl_max_m'], inputs['wl_n'])
        try:
            R, T, phase_r = la.coating_reflectance(
                inputs['layers'],
                wavelengths,
                angle=inputs['angle_rad'],
                n_substrate=inputs['n_substrate'],
                n_ambient=inputs['n_ambient'],
                polarization=inputs['polarization'],
            )
        except (TypeError, ValueError, RuntimeError, ImportError) as exc:
            self.summary.setPlainText(
                f'coating_reflectance failed: '
                f'{type(exc).__name__}: {exc}')
            return
        R = np.asarray(R)
        self._last_R = R
        self._last_wavelengths_m = wavelengths
        self._redraw(wavelengths, R)
        self._update_metrics(wavelengths, R, T)
        self.summary.setPlainText(
            f'Computed R(lambda) over {wavelengths.size} samples; '
            f'{len(inputs["layers"])} layer(s); '
            f'AOI={np.degrees(inputs["angle_rad"]):.2f} deg; '
            f'pol={inputs["polarization"]}.')

    def _update_metrics(self, wavelengths: np.ndarray, R: np.ndarray,
                        T: np.ndarray) -> None:
        R = np.asarray(R)
        T = np.asarray(T)
        if R.size == 0:
            self.lbl_metrics.setText('No samples.')
            return
        i_min = int(np.argmin(R))
        i_max = int(np.argmax(R))
        wl_nm = wavelengths * 1e9
        msg = (
            f'  R_mean   = {R.mean():.5f}\n'
            f'  R_min    = {R.min():.5f}  at  {wl_nm[i_min]:8.2f} nm\n'
            f'  R_max    = {R.max():.5f}  at  {wl_nm[i_max]:8.2f} nm\n'
            f'  T_mean   = {T.mean():.5f}\n'
            f'  T_min    = {T.min():.5f}\n'
            f'  T_max    = {T.max():.5f}'
        )
        self.lbl_metrics.setText(msg)

    # --------------------------------------------------------- plotting ---

    def _draw_empty(self) -> None:
        if not HAS_MPL:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Configure a layer stack and click '
                '"Compute R(lambda)".\nQuick-template buttons populate '
                'common AR / HR designs.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _redraw(self, wavelengths: np.ndarray, R: np.ndarray) -> None:
        """Refresh the matplotlib canvas with the new R(lambda) curve."""
        if not HAS_MPL:
            return
        wl_nm = np.asarray(wavelengths) * 1e9
        R = np.asarray(R)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.plot(wl_nm, R, color='#7ec1ff', linewidth=1.6,
                label='R (reflectance)')
        ax.set_xlabel('Wavelength [nm]', color='#dde8f8',
                      fontfamily='monospace')
        ax.set_ylabel('Reflectance', color='#dde8f8',
                      fontfamily='monospace')
        ax.set_xlim(wl_nm.min(), wl_nm.max())
        ymax = max(0.05, float(R.max()) * 1.10)
        ax.set_ylim(0.0, min(1.0, ymax))
        ax.grid(True, linestyle=':', color='#334054', alpha=0.6)
        ax.tick_params(colors='#7a94b8', labelsize=8)
        for s in ax.spines.values():
            s.set_color('#334054')
        # Highlight the R_min point.
        if R.size:
            i_min = int(np.argmin(R))
            ax.scatter([wl_nm[i_min]], [R[i_min]], s=40,
                       color='#ffb066', zorder=5,
                       label=f'R_min @ {wl_nm[i_min]:.1f} nm')
        leg = ax.legend(loc='best', fontsize=8, frameon=False,
                        labelcolor='#dde8f8')
        for txt in leg.get_texts():
            txt.set_color('#dde8f8')
        self.fig.tight_layout()
        self.canvas.draw_idle()


# =============================================================================
# Module smoke-test (matches the dock spec's smoke-test command):
#   python -c "from lumenairy.ui.coatings_dock import CoatingsDock; print('ok')"
# =============================================================================
