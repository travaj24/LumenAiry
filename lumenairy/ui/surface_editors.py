"""
Advanced surface-form editors: asphere / biconic / freeform / coating.

These dialogs open from the surface sub-table's right-click menu and
expose surface attributes that aren't reachable from the main table
(aspheric even-power coefficients, biconic y-radius / y-conic, XY
freeform polynomial, coating wavelength-range & reflectance target).

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QGroupBox, QFormLayout, QMessageBox,
    QComboBox, QCheckBox, QTextEdit,
)


class AsphereEditorDialog(QDialog):
    """Edit even-power asphere coefficients a4, a6, a8, a10, ...

    The surface shape is

        z(r) = r^2/(R*(1+sqrt(1-(1+k)*(r/R)^2))) + sum_i a_2i * r^(2i)

    (standard Code V even-power asphere, which is the form the core
    raytracer consumes via ``Surface.aspheric_coeffs``).
    """

    MAX_ORDER = 20   # coefficients up to r^20 (a20)

    def __init__(self, surface, parent=None):
        super().__init__(parent)
        self.surface = surface
        self.setWindowTitle('Asphere coefficients')
        self.setMinimumSize(520, 420)

        layout = QVBoxLayout(self)
        intro = QLabel(
            'Even-power asphere polynomial: z(r) = conic base + '
            'a4*r^4 + a6*r^6 + a8*r^8 + ...  Units: SI (meters^-n for '
            'the r^n coefficient).  Leave unused rows at 0.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#7a94b8;')
        layout.addWidget(intro)

        # Base conic row (read-only; edit it in the main table)
        base = QFormLayout()
        self.lbl_radius = QLabel(
            f'{surface.radius:.6g} m (edit via main table)')
        base.addRow('Radius of curvature:', self.lbl_radius)
        self.lbl_conic = QLabel(
            f'{surface.conic:.6g} (edit via main table)')
        base.addRow('Conic constant k:', self.lbl_conic)
        layout.addLayout(base)

        # Coefficients table.  The library stores aspheric_coeffs as a
        # dict keyed by r^power (int), e.g. ``{4: A4, 6: A6, ...}`` --
        # that's what Zemax import produces, what surface_sag_general
        # consumes, and what the raytracer's .items() loop expects.
        # We normalise any legacy list format on load so old session
        # files keep working.
        self.table = QTableWidget(self.MAX_ORDER // 2, 2)
        self.table.setHorizontalHeaderLabels(['Order (r^n)', 'Coefficient'])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        existing = surface.aspheric_coeffs or {}
        if isinstance(existing, (list, tuple)):
            # Legacy list form: [A4, A6, A8, ...] starting at r^4.
            existing = {2 * (i + 2): v for i, v in enumerate(existing)}
        for i in range(self.MAX_ORDER // 2):
            order = 2 * (i + 2)  # start at r^4
            self.table.setItem(i, 0, QTableWidgetItem(f'r^{order}'))
            self.table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            val = float(existing.get(order, 0.0))
            self.table.setItem(i, 1, QTableWidgetItem(f'{val:.12g}'))
        layout.addWidget(self.table, stretch=1)

        # Plot of sag (quick sanity)
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas)
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(4.5, 1.6), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setFixedHeight(150)
            layout.addWidget(self.canvas)
            self.btn_preview = QPushButton('Preview sag profile')
            self.btn_preview.clicked.connect(self._plot_sag)
            layout.addWidget(self.btn_preview)
        except Exception:
            self.fig = None

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._apply_and_accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _gather_coefficients(self):
        """Return ``{power: coeff}`` dict, dropping zeros.

        Matches the library's canonical aspheric_coeffs convention
        (same as Zemax import, surface_sag_general, and the raytracer).
        """
        coeffs = {}
        for i in range(self.MAX_ORDER // 2):
            order = 2 * (i + 2)
            try:
                v = float(self.table.item(i, 1).text())
            except Exception:
                v = 0.0
            if v != 0.0:
                coeffs[order] = v
        return coeffs

    def _plot_sag(self):
        if self.fig is None:
            return
        coeffs = self._gather_coefficients()
        R = float(self.surface.radius)
        k = float(self.surface.conic)
        if R == 0 or not np.isfinite(R):
            R = 1e30  # flat
        r_max = abs(R) * 0.5 if np.isfinite(R) else 1e-2
        r = np.linspace(0, r_max, 200)
        denom = 1 + np.sqrt(np.clip(1 - (1 + k) * (r / R) ** 2, 0, None))
        z_conic = r * r / (R * denom)
        z_asph = np.zeros_like(r)
        for power, c in coeffs.items():
            z_asph += c * r ** power
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(r * 1e3, z_conic * 1e6, '-', color='#7a94b8',
                label='conic base')
        ax.plot(r * 1e3, (z_conic + z_asph) * 1e6, '-',
                color='#5cb8ff', label='full sag')
        ax.set_xlabel('r (mm)')
        ax.set_ylabel('z (um)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        self.canvas.draw()

    def _apply_and_accept(self):
        self.surface.aspheric_coeffs = self._gather_coefficients() or None
        self.accept()


class BiconicEditorDialog(QDialog):
    """Edit radius_y / conic_y for anamorphic (biconic) surfaces."""

    def __init__(self, surface, parent=None):
        super().__init__(parent)
        self.surface = surface
        self.setWindowTitle('Biconic / anamorphic surface')
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        intro = QLabel(
            'A biconic has different curvature in X and Y.  Leave empty '
            'to keep the surface rotationally symmetric.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#7a94b8;')
        layout.addWidget(intro)

        form = QFormLayout()
        self.lbl_rx = QLabel(f'{surface.radius:.6g} m')
        form.addRow('Rx (from main table):', self.lbl_rx)
        self.lbl_kx = QLabel(f'{surface.conic:.6g}')
        form.addRow('kx (from main table):', self.lbl_kx)

        self.spin_ry = QDoubleSpinBox()
        self.spin_ry.setRange(-1e9, 1e9)
        self.spin_ry.setDecimals(9)
        self.spin_ry.setSingleStep(1e-3)
        self.spin_ry.setSpecialValueText('(symmetric)')
        ry = surface.radius_y
        if ry is None or not np.isfinite(ry):
            self.spin_ry.setValue(self.spin_ry.minimum())
        else:
            self.spin_ry.setValue(float(ry))
        form.addRow('Ry (m):', self.spin_ry)

        self.spin_ky = QDoubleSpinBox()
        self.spin_ky.setRange(-10, 10)
        self.spin_ky.setDecimals(4)
        self.spin_ky.setSingleStep(0.01)
        self.spin_ky.setSpecialValueText('(symmetric)')
        ky = surface.conic_y
        if ky is None:
            self.spin_ky.setValue(self.spin_ky.minimum())
        else:
            self.spin_ky.setValue(float(ky))
        form.addRow('ky:', self.spin_ky)

        layout.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._apply_and_accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _apply_and_accept(self):
        ry = self.spin_ry.value()
        self.surface.radius_y = None if ry <= self.spin_ry.minimum() + 1e-12 else ry
        ky = self.spin_ky.value()
        self.surface.conic_y = None if ky <= self.spin_ky.minimum() + 1e-12 else ky
        self.accept()


class FreeformEditorDialog(QDialog):
    """Edit XY polynomial freeform sag coefficients.

    Sag is

        z(x, y) = sum over (i, j) :  c_{i,j} * x^i * y^j

    stored on the surface as a dict keyed by ``(i, j)``.  This dialog
    supports all non-negative integer pairs with i + j <= order_max.
    """

    MAX_ORDER = 8

    def __init__(self, surface, parent=None):
        super().__init__(parent)
        self.surface = surface
        self.setWindowTitle('Freeform surface (XY polynomial)')
        self.setMinimumSize(560, 520)

        layout = QVBoxLayout(self)
        intro = QLabel(
            'XY polynomial sag: z(x, y) = sum c_ij * x^i * y^j.  Grid '
            'shows coefficients by order i+j (diagonals).  Units: '
            'SI (metres^(1-i-j) for c_ij).  Only coefficients with '
            'i+j >= 2 contribute to curvature; c_00/c_10/c_01 are '
            'offset/tilt and should be 0 in normal use.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#7a94b8;')
        layout.addWidget(intro)

        max_order_row = QHBoxLayout()
        max_order_row.addWidget(QLabel('max i+j:'))
        self.spin_order = QSpinBox()
        self.spin_order.setRange(2, self.MAX_ORDER)
        self.spin_order.setValue(min(self.MAX_ORDER,
                                      max(4, self._detect_order(surface))))
        self.spin_order.valueChanged.connect(self._rebuild_table)
        max_order_row.addWidget(self.spin_order)
        max_order_row.addStretch()
        layout.addLayout(max_order_row)

        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        layout.addWidget(self.table, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._apply_and_accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self._rebuild_table()

    def _detect_order(self, surface):
        fc = getattr(surface, 'freeform_coeffs', None) or {}
        if not fc:
            return 4
        return max(i + j for (i, j) in fc.keys())

    def _rebuild_table(self):
        order = self.spin_order.value()
        # Columns = index i, Rows = index j
        self.table.setRowCount(order + 1)
        self.table.setColumnCount(order + 1)
        self.table.setHorizontalHeaderLabels([f'i={i}' for i in range(order + 1)])
        self.table.setVerticalHeaderLabels([f'j={j}' for j in range(order + 1)])
        fc = getattr(self.surface, 'freeform_coeffs', None) or {}
        for j in range(order + 1):
            for i in range(order + 1):
                cell = QTableWidgetItem()
                if i + j > order:
                    cell.setFlags(Qt.NoItemFlags)
                    cell.setText('-')
                else:
                    cell.setText(f'{fc.get((i, j), 0.0):.6g}')
                self.table.setItem(j, i, cell)

    def _apply_and_accept(self):
        order = self.spin_order.value()
        new_fc = {}
        for j in range(order + 1):
            for i in range(order + 1):
                if i + j > order:
                    continue
                item = self.table.item(j, i)
                if item is None:
                    continue
                try:
                    val = float(item.text())
                except Exception:
                    val = 0.0
                if val != 0.0:
                    new_fc[(i, j)] = val
        # Store as a plain dict; raytracer looks up by tuple key.
        self.surface.freeform_coeffs = new_fc or None
        self.accept()


class CoatingEditorDialog(QDialog):
    """Per-surface AR coating: wavelength range + target reflectance.

    This dialog edits the surface's ``coating`` attribute (a dict);
    the raytracer's Fresnel / transmission routines consult it for
    AR reflectance multipliers.
    """

    def __init__(self, surface, parent=None):
        super().__init__(parent)
        self.surface = surface
        self.setWindowTitle('AR coating')
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        intro = QLabel(
            'Simple AR-coating model: constant reflectance Rc inside '
            '[lambda_min, lambda_max], 4% glass value outside.  Use '
            "'none' to strip any existing coating.  A full thin-film "
            'stack can be added later via a material from the Library '
            'dock.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#7a94b8;')
        layout.addWidget(intro)

        form = QFormLayout()
        coating = getattr(surface, 'coating', None) or {}

        self.combo_type = QComboBox()
        self.combo_type.addItems(['none', 'broadband_AR',
                                   'narrowband_AR', 'custom'])
        ctype = coating.get('type', 'none')
        if ctype in ('none', 'broadband_AR', 'narrowband_AR', 'custom'):
            self.combo_type.setCurrentText(ctype)
        form.addRow('Type:', self.combo_type)

        self.spin_wmin = QDoubleSpinBox()
        self.spin_wmin.setRange(100.0, 20000.0)
        self.spin_wmin.setDecimals(1)
        self.spin_wmin.setSuffix(' nm')
        self.spin_wmin.setValue(
            float(coating.get('lambda_min_m', 400e-9)) * 1e9)
        form.addRow('Wavelength min:', self.spin_wmin)

        self.spin_wmax = QDoubleSpinBox()
        self.spin_wmax.setRange(100.0, 20000.0)
        self.spin_wmax.setDecimals(1)
        self.spin_wmax.setSuffix(' nm')
        self.spin_wmax.setValue(
            float(coating.get('lambda_max_m', 700e-9)) * 1e9)
        form.addRow('Wavelength max:', self.spin_wmax)

        self.spin_R = QDoubleSpinBox()
        self.spin_R.setRange(0.0, 1.0)
        self.spin_R.setDecimals(4)
        self.spin_R.setSingleStep(0.001)
        self.spin_R.setValue(float(coating.get('reflectance', 0.005)))
        form.addRow('R in band:', self.spin_R)

        layout.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._apply_and_accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _apply_and_accept(self):
        ctype = self.combo_type.currentText()
        if ctype == 'none':
            self.surface.coating = None
        else:
            self.surface.coating = {
                'type': ctype,
                'lambda_min_m': self.spin_wmin.value() * 1e-9,
                'lambda_max_m': self.spin_wmax.value() * 1e-9,
                'reflectance': self.spin_R.value(),
            }
        self.accept()
