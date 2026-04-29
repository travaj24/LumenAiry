"""
SurfaceTableEditor — prescription spreadsheet widget.

A QTableView backed by a QAbstractTableModel that displays and edits
the surface data from SystemModel.  Editing any cell triggers a
system_changed signal, which propagates to the layout and analysis views.

The equivalent of a Lens Data Editor (LDE) in commercial tools.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView,
    QPushButton, QLabel, QLineEdit, QComboBox, QAbstractItemView,
    QStyledItemDelegate,
)
from PySide6.QtGui import QColor, QFont

import numpy as np

from .model import SystemModel, SurfaceRow


# ────────────────────────────────────────────────────────────────────────
# Table model
# ────────────────────────────────────────────────────────────────────────

class SurfaceTableModel(QAbstractTableModel):
    """Qt table model wrapping SystemModel.surfaces."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self.sm.system_changed.connect(self._on_system_changed)

    def _on_system_changed(self):
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return len(self.sm.surfaces)

    def columnCount(self, parent=QModelIndex()):
        return SurfaceRow.N_COLS

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return SurfaceRow.COLUMNS[section]
            else:
                # Row header: OBJ, 1, 2, ..., IMA
                if section == 0:
                    return 'OBJ'
                elif section == len(self.sm.surfaces) - 1:
                    return 'IMA'
                else:
                    return str(section)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = self.sm.surfaces[index.row()]
        col = index.column()

        if role == Qt.DisplayRole or role == Qt.EditRole:
            return row.display_value(col)

        elif role == Qt.BackgroundRole:
            # Tint glass surfaces
            if row.glass and col == 5:
                return QColor(40, 60, 90)
            # Tint mirror surfaces
            if row.surf_type == 'Mirror':
                return QColor(50, 50, 70)
            # OBJ and IMA rows
            if index.row() == 0 or index.row() == len(self.sm.surfaces) - 1:
                return QColor(25, 30, 40)
            # Grouped surfaces get a tinted background
            if row.group and col == 0:
                # Hash the group name to a colour
                h = hash(row.group) % 360
                return QColor.fromHsv(h, 80, 50)

        elif role == Qt.ForegroundRole:
            # Infinity values in grey
            if col in (3, 6) and np.isinf(getattr(row, 'radius' if col == 3 else 'semi_diameter')):
                return QColor(120, 140, 170)
            return QColor(220, 232, 248)

        elif role == Qt.FontRole:
            if col == 0:
                f = QFont('Consolas', 10)
                f.setBold(True)
                return f

        elif role == Qt.TextAlignmentRole:
            if col in (3, 4, 6, 7):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter

        return None

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        col = index.column()
        row_idx = index.row()

        # Surf number is read-only
        if col == 0:
            return base

        # OBJ row: only thickness is editable
        if row_idx == 0 and col != 4:
            return base

        # IMA row: nothing editable
        if row_idx == len(self.sm.surfaces) - 1:
            return base

        return base | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole:
            return False
        changed = self.sm.set_surface_field(
            index.row(), index.column(), str(value))
        return changed


# ────────────────────────────────────────────────────────────────────────
# Type column delegate — dropdown for Standard / Mirror / Image
# ────────────────────────────────────────────────────────────────────────

class TypeDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(['Standard', 'Mirror', 'MLA', 'DOE', 'Dammann'])
        combo.setStyleSheet(
            "QComboBox { background: #1a2030; color: #dde8f8; "
            "border: 1px solid #3a4a60; }")
        return combo

    def setEditorData(self, editor, index):
        val = index.data(Qt.EditRole)
        idx = editor.findText(val)
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


# ────────────────────────────────────────────────────────────────────────
# Surface table widget
# ────────────────────────────────────────────────────────────────────────

class SurfaceTableEditor(QWidget):
    """Complete surface editor widget with toolbar and system info."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)

        btn_insert = QPushButton('+ Insert')
        btn_insert.setToolTip('Insert surface before selection')
        btn_insert.clicked.connect(self._insert_surface)

        btn_delete = QPushButton('− Delete')
        btn_delete.setToolTip('Delete selected surface')
        btn_delete.clicked.connect(self._delete_surface)

        lbl_wv = QLabel('λ (nm):')
        self.inp_wv = QLineEdit(str(self.sm.wavelength_nm))
        self.inp_wv.setFixedWidth(70)
        self.inp_wv.editingFinished.connect(self._set_wavelength)

        lbl_epd = QLabel('EPD (mm):')
        self.inp_epd = QLineEdit(str(self.sm.epd_mm))
        self.inp_epd.setFixedWidth(70)
        self.inp_epd.editingFinished.connect(self._set_epd)

        btn_up = QPushButton('▲')
        btn_up.setFixedWidth(28)
        btn_up.setToolTip('Move surface up')
        btn_up.clicked.connect(self._move_up)

        btn_down = QPushButton('▼')
        btn_down.setFixedWidth(28)
        btn_down.setToolTip('Move surface down')
        btn_down.clicked.connect(self._move_down)

        btn_group = QPushButton('Group')
        btn_group.setToolTip('Group selected surfaces as a cemented optic')
        btn_group.clicked.connect(self._group_selected)

        btn_ungroup = QPushButton('Ungroup')
        btn_ungroup.setToolTip('Remove selected surface from its group')
        btn_ungroup.clicked.connect(self._ungroup_selected)

        toolbar.addWidget(btn_insert)
        toolbar.addWidget(btn_delete)
        toolbar.addWidget(btn_up)
        toolbar.addWidget(btn_down)
        toolbar.addWidget(btn_group)
        toolbar.addWidget(btn_ungroup)
        toolbar.addStretch()
        toolbar.addWidget(lbl_wv)
        toolbar.addWidget(self.inp_wv)
        toolbar.addWidget(lbl_epd)
        toolbar.addWidget(self.inp_epd)

        layout.addLayout(toolbar)

        # ── Table ──
        self.table_model = SurfaceTableModel(self.sm)
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)

        # Set column delegate for Type column
        self.table.setItemDelegateForColumn(2, TypeDelegate(self.table))

        # Column widths — all Interactive so user can resize any column
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setStretchLastSection(True)

        # Set sensible defaults
        default_widths = {
            0: 40,   # Surf
            1: 100,  # Comment
            2: 70,   # Type
            3: 90,   # Radius
            4: 80,   # Thickness
            5: 75,   # Glass
            6: 85,   # Semi-Diameter
            7: 60,   # Conic
            8: 55,   # Tilt X
            9: 55,   # Tilt Y
            10: 65,  # Decenter X
            11: 65,  # Decenter Y
        }
        for col, width in default_widths.items():
            self.table.setColumnWidth(col, width)

        layout.addWidget(self.table, stretch=1)

        # ── System info bar ──
        self.info_bar = QLabel()
        self.info_bar.setStyleSheet(
            "QLabel { background: #12161e; color: #7a94b8; "
            "padding: 4px 8px; font-size: 12px; }")
        layout.addWidget(self.info_bar)

        # Connect
        self.sm.system_changed.connect(self._update_info)
        self.sm.surface_selected.connect(self._select_row)
        self._update_info()

    def _insert_surface(self):
        idx = self.table.currentIndex().row()
        if idx < 1:
            idx = len(self.sm.surfaces) - 1  # before IMA
        self.sm.insert_surface(idx)

    def _delete_surface(self):
        idx = self.table.currentIndex().row()
        self.sm.delete_surface(idx)

    def _set_wavelength(self):
        try:
            wv = float(self.inp_wv.text())
            if wv > 0:
                self.sm.set_wavelength(wv)
        except ValueError:
            pass

    def _set_epd(self):
        try:
            epd = float(self.inp_epd.text())
            if epd > 0:
                self.sm.set_epd(epd)
        except ValueError:
            pass

    def _move_up(self):
        idx = self.table.currentIndex().row()
        self.sm.move_surface_up(idx)
        if idx > 1:
            self.table.selectRow(idx - 1)

    def _move_down(self):
        idx = self.table.currentIndex().row()
        self.sm.move_surface_down(idx)
        if idx < len(self.sm.surfaces) - 2:
            self.table.selectRow(idx + 1)

    def _group_selected(self):
        """Group the currently selected rows as a cemented optic."""
        selection = self.table.selectionModel().selectedRows()
        if len(selection) < 2:
            return
        indices = sorted(idx.row() for idx in selection)
        # Generate a group name from the first surface comment or index
        first = self.sm.surfaces[indices[0]]
        group_name = first.comment or f'Group_{indices[0]}'
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, 'Group Name', 'Name for this optic group:', text=group_name)
        if ok and name:
            self.sm.group_surfaces(indices, name)

    def _ungroup_selected(self):
        idx = self.table.currentIndex().row()
        self.sm.ungroup_surface(idx)

    def _select_row(self, surface_index):
        """Highlight the row corresponding to a surface clicked in the layout."""
        if 0 <= surface_index < self.table_model.rowCount():
            self.table.selectRow(surface_index)
            self.table.scrollTo(self.table_model.index(surface_index, 0))

    def _update_info(self):
        try:
            abcd, efl, bfl = self.sm.get_abcd()
            efl_str = f'{efl * 1e3:.2f}' if np.isfinite(efl) else '∞'
            bfl_str = f'{bfl * 1e3:.2f}' if np.isfinite(bfl) else '∞'
            na = self.sm.epd_m / (2 * abs(efl)) if np.isfinite(efl) and efl != 0 else 0
            fnum = abs(efl) / self.sm.epd_m if np.isfinite(efl) and self.sm.epd_m > 0 else np.inf
            fnum_str = f'{fnum:.2f}' if np.isfinite(fnum) else '∞'
            src_desc = self.sm.source.describe() if self.sm.source else 'Plane wave'
            self.info_bar.setText(
                f'EFL: {efl_str} mm  |  BFL: {bfl_str} mm  |  '
                f'f/#: {fnum_str}  |  '
                f'λ: {self.sm.wavelength_nm:.1f} nm  |  '
                f'Surfaces: {self.sm.num_surfaces - 2}  |  '  # exclude OBJ/IMA
                f'Source: {src_desc}'
            )
        except Exception as e:
            self.info_bar.setText(f'Error: {e}')
