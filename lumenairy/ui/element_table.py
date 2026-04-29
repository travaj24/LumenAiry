"""
ElementTableEditor — element-based prescription editor.

Each row is an optical element (lens, mirror, DOE), not a raw surface.
Clicking an element shows its internal surfaces in a detail panel below.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView,
    QPushButton, QLabel, QLineEdit, QComboBox, QAbstractItemView,
    QStyledItemDelegate, QSplitter, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QDialog,
)
from PySide6.QtGui import QColor, QFont

import numpy as np

from .model import SystemModel, Element, SurfaceRow, SourceDefinition


# ────────────────────────────────────────────────────────────────────────
# Element table model
# ────────────────────────────────────────────────────────────────────────

class ElementTableModel(QAbstractTableModel):
    """Qt model wrapping SystemModel.elements."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self.sm.system_changed.connect(self._on_changed)
        self.sm.display_changed.connect(self._on_changed)

    def _on_changed(self):
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return len(self.sm.elements)

    def columnCount(self, parent=QModelIndex()):
        return Element.N_COLS

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if self.sm.coordinate_mode == 'absolute':
                return Element.COLUMNS_ABSOLUTE[section]
            return Element.COLUMNS_RELATIVE[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        elem = self.sm.elements[index.row()]
        col = index.column()

        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 3:
                return elem.display_value(col, self.sm.get_display_distance(index.row()))
            return elem.display_value(col)

        elif role == Qt.BackgroundRole:
            if elem.elem_type == 'Source':
                return QColor(30, 50, 40)
            elif elem.elem_type == 'Detector':
                return QColor(50, 30, 30)
            elif elem.elem_type == 'Mirror':
                return QColor(35, 35, 55)
            elif elem.elem_type in ('MLA', 'DOE', 'Dammann'):
                return QColor(45, 35, 55)

        elif role == Qt.ForegroundRole:
            if col == 2:
                # Type in accent color
                return QColor(92, 184, 255)
            # Highlight Elem# (col 0) gold when this element has any
            # optimization variable on it.  Distance variable cell
            # (col 3) is highlighted directly below.
            if col == 0:
                row = index.row()
                if any(v[0] == row for v in self.sm.opt_variables):
                    return QColor(255, 200, 80)  # amber
            if col == 3:
                row = index.row()
                if any(v[0] == row and v[2] == 'distance'
                       for v in self.sm.opt_variables):
                    return QColor(255, 200, 80)
            return QColor(220, 232, 248)

        elif role == Qt.FontRole:
            if col == 0:
                f = QFont('Consolas', 10)
                f.setBold(True)
                return f

        elif role == Qt.TextAlignmentRole:
            if col in (3, 4, 5, 6, 7):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter

        elif role == Qt.ToolTipRole:
            return elem.summary()

        return None

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        col = index.column()
        row = index.row()
        elem = self.sm.elements[row]

        # Elem# is always read-only
        if col == 0:
            return base
        # Source: only Name is editable (source params edited in detail panel)
        if elem.elem_type == 'Source':
            return base | Qt.ItemIsEditable if col == 1 else base
        # Detector: Name and Distance
        if elem.elem_type == 'Detector':
            return base | Qt.ItemIsEditable if col in (1, 3) else base
        # Everything else: all editable except elem#
        return base | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole:
            return False
        return self.sm.set_element_field(index.row(), index.column(), str(value))


# ────────────────────────────────────────────────────────────────────────
# Surface flat model — Zemax-style view of all surfaces
# ────────────────────────────────────────────────────────────────────────

class SurfaceFlatModel(QAbstractTableModel):
    """Flat surface view showing all surfaces plus air gaps between elements.

    Includes tilt/decenter from the parent element on each surface row,
    and explicit air-gap rows between consecutive elements.
    """

    COLUMNS = ['Surf#', 'Element', 'Type', 'Radius', 'Thickness', 'Glass',
               'Semi-Diam', 'Conic', 'Tilt X', 'Tilt Y', 'Decenter X', 'Decenter Y']

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self.sm.system_changed.connect(lambda: self.layoutChanged.emit())
        self.sm.display_changed.connect(lambda: self.layoutChanged.emit())
        self._flat = []
        # Each entry: (elem_idx, surf_idx, SurfaceRow_or_None, row_kind, label)
        # row_kind: 'source', 'surface', 'airgap', 'detector'

    def _rebuild_flat(self):
        self._flat = []
        elements = self.sm.elements

        for ei, elem in enumerate(elements):
            if elem.elem_type == 'Source':
                self._flat.append((ei, -1, None, 'source', 'Source'))
                continue
            if elem.elem_type == 'Detector':
                self._flat.append((ei, -1, None, 'detector', 'Detector'))
                continue

            # Air gap before this element (from previous element)
            if elem.distance_mm > 0:
                self._flat.append((ei, -1, None, 'airgap',
                                   f'Air gap to {elem.name}'))

            # Surfaces of this element
            for si, srow in enumerate(elem.surfaces):
                self._flat.append((ei, si, srow, 'surface', elem.name))

    def rowCount(self, parent=QModelIndex()):
        self._rebuild_flat()
        return len(self._flat)

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._flat):
            return None
        ei, si, srow, kind, label = self._flat[index.row()]
        col = index.column()
        elem = self.sm.elements[ei]

        if role in (Qt.DisplayRole, Qt.EditRole):
            if kind == 'source':
                return {0: 'OBJ', 1: 'Source', 2: elem.source.describe() if elem.source else ''}.get(col, '')
            elif kind == 'detector':
                return {0: 'IMA', 1: 'Detector', 4: f'{elem.distance_mm:.4g}'}.get(col, '')
            elif kind == 'airgap':
                return {0: '', 1: label, 2: 'Air', 4: f'{elem.distance_mm:.4g}'}.get(col, '')
            else:  # surface
                if col == 0: return str(index.row())
                elif col == 1: return label
                elif col == 2: return srow.surf_type
                elif col == 3: return 'Infinity' if np.isinf(srow.radius) else f'{srow.radius:.6g}'
                elif col == 4: return f'{srow.thickness:.6g}'
                elif col == 5: return srow.glass
                elif col == 6: return 'Infinity' if np.isinf(srow.semi_diameter) else f'{srow.semi_diameter:.6g}'
                elif col == 7: return f'{srow.conic:.6g}'
                elif col == 8: return f'{elem.tilt_x:.4g}' if elem.tilt_x else '0'
                elif col == 9: return f'{elem.tilt_y:.4g}' if elem.tilt_y else '0'
                elif col == 10: return f'{elem.decenter_x:.4g}' if elem.decenter_x else '0'
                elif col == 11: return f'{elem.decenter_y:.4g}' if elem.decenter_y else '0'

        elif role == Qt.BackgroundRole:
            if kind == 'source': return QColor(25, 40, 30)
            elif kind == 'detector': return QColor(40, 25, 25)
            elif kind == 'airgap': return QColor(20, 25, 35)
            elif srow and srow.glass: return QColor(35, 50, 75)
            elif srow and (srow.surf_type == 'Mirror' or elem.elem_type == 'Mirror'):
                return QColor(35, 35, 55)

        elif role == Qt.ForegroundRole:
            if kind == 'airgap': return QColor(100, 130, 170)
            return QColor(220, 232, 248)

        elif role == Qt.FontRole:
            if kind == 'airgap':
                f = QFont('Consolas', 10)
                f.setItalic(True)
                return f

        elif role == Qt.TextAlignmentRole:
            if col in (3, 4, 6, 7, 8, 9, 10, 11):
                return Qt.AlignRight | Qt.AlignVCenter

        return None

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.row() >= len(self._flat):
            return base
        ei, si, srow, kind, _ = self._flat[index.row()]
        if kind in ('source', 'detector'):
            return base
        if kind == 'airgap':
            # Only the thickness (distance) column is editable for air gaps
            return base | Qt.ItemIsEditable if index.column() == 4 else base
        # Surface row: radius, thickness, glass, semi-diam, conic editable
        if index.column() in (0, 1, 2):
            return base
        return base | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or index.row() >= len(self._flat):
            return False
        ei, si, srow, kind, _ = self._flat[index.row()]
        col = index.column()

        if kind == 'airgap' and col == 4:
            # Edit the element's distance
            try:
                self.sm.elements[ei].distance_mm = max(0, float(value))
                self.sm._invalidate()
                self.sm.system_changed.emit()
                return True
            except ValueError:
                return False

        if kind != 'surface' or srow is None:
            return False

        # Surface fields
        field_map = {3: 'radius', 4: 'thickness', 5: 'glass', 6: 'semi_diameter', 7: 'conic'}
        field = field_map.get(col)
        if field:
            return self.sm.set_surface_field(ei, si, field, value)

        # Tilt/decenter — these modify the parent element
        elem = self.sm.elements[ei]
        try:
            val = float(value) if value else 0.0
            if col == 8 and val != elem.tilt_x:
                elem.tilt_x = val
            elif col == 9 and val != elem.tilt_y:
                elem.tilt_y = val
            elif col == 10 and val != elem.decenter_x:
                elem.decenter_x = val
            elif col == 11 and val != elem.decenter_y:
                elem.decenter_y = val
            else:
                return False
            self.sm._invalidate()
            self.sm.system_changed.emit()
            return True
        except ValueError:
            return False


# ────────────────────────────────────────────────────────────────────────
# Type column delegate
# ────────────────────────────────────────────────────────────────────────

class TypeDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(Element.TYPES)
        combo.setStyleSheet(
            "QComboBox { background: #1a2030; color: #dde8f8; border: 1px solid #3a4a60; }")
        return combo

    def setEditorData(self, editor, index):
        idx = editor.findText(index.data(Qt.EditRole))
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


# ────────────────────────────────────────────────────────────────────────
# Surface detail panel (shown when an element is selected)
# ────────────────────────────────────────────────────────────────────────

class SurfaceDetailPanel(QWidget):
    """Shows and edits the internal surfaces of the selected element."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._elem_idx = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Visible "banner" title that makes the selection link between
        # the element table and the detail panel unambiguous.  Colored
        # bar on the left + bold label in accent.
        banner = QWidget()
        banner.setStyleSheet(
            "background: #12161e; border-left: 4px solid #5cb8ff;")
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(8, 4, 8, 4)
        self.title_label = QLabel('\u25b6  Select an element to see its surfaces')
        self.title_label.setStyleSheet(
            "color: #5cb8ff; font-size: 13px; font-weight: bold; "
            "padding: 2px; background: transparent; border: none;")
        self.title_label.setToolTip(
            'The surfaces shown below belong to the element currently '
            'selected in the table above.')
        banner_layout.addWidget(self.title_label)
        layout.addWidget(banner)

        # Surface sub-table
        self.surf_table = QTableView()
        self.surf_table.setAlternatingRowColors(True)
        self.surf_table.setMaximumHeight(150)
        self.surf_model = SurfaceSubModel(system_model)
        self.surf_table.setModel(self.surf_model)
        # Right-click context menu: quick actions on the selected surface.
        self.surf_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.surf_table.customContextMenuRequested.connect(
            self._on_surface_context_menu)

        hdr = self.surf_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setStretchLastSection(True)

        layout.addWidget(self.surf_table)

        # Source config panel (shown only for Source element).
        # Irrelevant rows are hidden per-source-type so the form doesn't
        # confront the user with ten unrelated parameters at once.
        self.source_panel = QWidget()
        src_layout = QFormLayout(self.source_panel)
        src_layout.setContentsMargins(4, 4, 4, 4)

        self.src_type_combo = QComboBox()
        self.src_type_combo.addItems(SourceDefinition.TYPES)
        self.src_type_combo.setToolTip(
            'Choose the source model.  The parameters shown below '
            'change to match -- unused ones are hidden.')
        self.src_type_combo.currentTextChanged.connect(self._on_source_type_changed)
        src_layout.addRow('Source type:', self.src_type_combo)

        # Param metadata: (key, label, default, tooltip,
        #                  {source_types that use this param})
        _SRC_PARAMS = [
            ('beam_diameter_mm',    'Beam diameter (mm):',    '1.0',
             '1/e\u00b2 diameter of the incoming Gaussian beam',
             {'gaussian'}),
            ('na',                  'NA:',                    '0.1',
             'Numerical aperture of the Gaussian source',
             {'gaussian'}),
            ('sigma_mm',            'Sigma (mm):',            '5.0',
             'Soft-edge standard deviation of the Gaussian aperture',
             {'gaussian_aperture'}),
            ('object_distance_mm',  'Object distance (mm):',  '1000',
             'Distance from the point source to the entrance pupil',
             {'point_source'}),
            ('emitter_pitch_mm',    'Emitter pitch (mm):',    '0.05',
             'Center-to-center spacing of emitters in the array',
             {'emitter_array'}),
            ('emitter_nx',          'Array NX:',              '12',
             'Number of emitters along X',
             {'emitter_array'}),
            ('emitter_ny',          'Array NY:',              '12',
             'Number of emitters along Y',
             {'emitter_array'}),
            ('emitter_waist_mm',    'Emitter waist (mm):',    '0.009',
             '1/e\u00b2 waist radius of each emitter',
             {'emitter_array'}),
            ('field_angle_x_deg',   'Field angle X (deg):',   '0.0',
             'Off-axis tilt about the X axis (applies to any source)',
             None),  # always shown
            ('field_angle_y_deg',   'Field angle Y (deg):',   '0.0',
             'Off-axis tilt about the Y axis (applies to any source)',
             None),
        ]
        self.src_params = {}
        self._src_param_meta = {}         # key -> {types} or None
        self._src_param_rows = {}         # key -> (label_widget, inp_widget)

        # Debounce to avoid retracing on every keystroke.
        self._src_timer = QTimer(self)
        self._src_timer.setSingleShot(True)
        self._src_timer.setInterval(200)
        self._src_timer.timeout.connect(self._apply_source_params)

        for key, label, default, tip, uses_in in _SRC_PARAMS:
            inp = QLineEdit(default)
            inp.setFixedWidth(100)
            inp.setToolTip(tip)
            # textChanged -> debounce + re-apply once settled
            inp.textChanged.connect(lambda _t: self._src_timer.start())
            lbl_widget = QLabel(label)
            lbl_widget.setToolTip(tip)
            src_layout.addRow(lbl_widget, inp)
            self.src_params[key] = inp
            self._src_param_meta[key] = uses_in
            self._src_param_rows[key] = (lbl_widget, inp)

        layout.addWidget(self.source_panel)
        self.source_panel.hide()

    def show_element(self, elem_idx):
        """Display the internal surfaces of the element at elem_idx."""
        self._elem_idx = elem_idx
        if elem_idx < 0 or elem_idx >= len(self.sm.elements):
            self.title_label.setText('Select an element')
            self.surf_model.set_element(-1)
            self.source_panel.hide()
            return

        elem = self.sm.elements[elem_idx]
        self.title_label.setText(
            f'\u25b6  Row {elem_idx}: {elem.elem_type} "{elem.name}" \u2014 '
            f'{len(elem.surfaces)} surface(s)')

        if elem.elem_type == 'Source':
            self.surf_table.hide()
            self.source_panel.show()
            self._load_source_ui(elem.source)
        else:
            self.source_panel.hide()
            self.surf_table.show()
            self.surf_model.set_element(elem_idx)
            # Set default column widths
            for c in range(self.surf_model.columnCount()):
                self.surf_table.setColumnWidth(c, 85)

    def _load_source_ui(self, src):
        if not src:
            return
        self.src_type_combo.blockSignals(True)
        self.src_type_combo.setCurrentText(src.source_type)
        self.src_type_combo.blockSignals(False)
        for key, inp in self.src_params.items():
            inp.blockSignals(True)
            inp.setText(str(getattr(src, key, '')))
            inp.blockSignals(False)
        self._show_relevant_source_rows(src.source_type)

    def _on_source_type_changed(self, text):
        self._show_relevant_source_rows(text)
        self._apply_source_params()

    def _show_relevant_source_rows(self, source_type):
        """Hide source-parameter rows that aren't used by this type."""
        for key, (label_widget, inp_widget) in self._src_param_rows.items():
            uses_in = self._src_param_meta[key]
            show = (uses_in is None) or (source_type in uses_in)
            label_widget.setVisible(show)
            inp_widget.setVisible(show)

    def _on_surface_context_menu(self, pos):
        """Right-click context menu on the surface sub-table."""
        from PySide6.QtWidgets import QMenu
        index = self.surf_table.indexAt(pos)
        if not index.isValid():
            return
        surf_idx = index.row()
        if self._elem_idx < 0:
            return
        elem = self.sm.elements[self._elem_idx]
        if surf_idx >= len(elem.surfaces):
            return
        s = elem.surfaces[surf_idx]

        menu = QMenu(self.surf_table)
        if s.glass:
            act = menu.addAction(
                f'Propagate "{s.glass}" to all cemented faces in this element')
            act.triggered.connect(
                lambda: self._propagate_cemented_glass(
                    self._elem_idx, surf_idx, s.glass))
        menu.addAction(
            'Copy this surface to clipboard (radius / thickness / glass / conic)',
            lambda: self._copy_surface_info(s))
        menu.addSeparator()
        menu.addAction(
            'Edit asphere coefficients...',
            lambda: self._edit_surface(s, 'asphere'))
        menu.addAction(
            'Edit biconic (Ry, ky)...',
            lambda: self._edit_surface(s, 'biconic'))
        menu.addAction(
            'Edit freeform XY polynomial...',
            lambda: self._edit_surface(s, 'freeform'))
        menu.addAction(
            'Edit AR coating...',
            lambda: self._edit_surface(s, 'coating'))
        menu.exec(self.surf_table.viewport().mapToGlobal(pos))

    def _edit_surface(self, surface, which):
        """Bridge to the advanced surface-form editor dialogs."""
        from .surface_editors import (
            AsphereEditorDialog, BiconicEditorDialog,
            FreeformEditorDialog, CoatingEditorDialog,
        )
        dlg_map = {
            'asphere': AsphereEditorDialog,
            'biconic': BiconicEditorDialog,
            'freeform': FreeformEditorDialog,
            'coating': CoatingEditorDialog,
        }
        cls = dlg_map.get(which)
        if cls is None:
            return
        self.sm._checkpoint()
        dlg = cls(surface, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.sm._invalidate()
            self.sm.system_changed.emit()

    def _propagate_cemented_glass(self, ei, si, glass_name):
        """Apply glass_name to every subsequent surface in this element
        that has a non-empty glass field (the cemented-interface chain).
        """
        elem = self.sm.elements[ei]
        self.sm._checkpoint()
        changed = 0
        # Move forward from (si) until we hit an air-facing surface.
        for k in range(si, len(elem.surfaces) - 1):
            if elem.surfaces[k].glass:
                if elem.surfaces[k].glass != glass_name:
                    elem.surfaces[k].glass = glass_name
                    changed += 1
            else:
                break
        self.sm._invalidate()
        self.sm.system_changed.emit()
        try:
            from .diagnostics import diag
            diag.info('glass-propagate',
                      f'Set {changed} cemented surface(s) to {glass_name}')
        except Exception:
            pass

    def _copy_surface_info(self, surface):
        from PySide6.QtGui import QGuiApplication
        txt = (f'radius = {surface.radius}\n'
               f'thickness = {surface.thickness}\n'
               f'glass = {surface.glass}\n'
               f'conic = {surface.conic}')
        QGuiApplication.clipboard().setText(txt)

    def _apply_source_params(self):
        if self._elem_idx != 0:
            return
        kwargs = {}
        for key, inp in self.src_params.items():
            try:
                val = inp.text().strip()
                if val == '':
                    continue
                kwargs[key] = float(val)
            except ValueError:
                pass
        src = SourceDefinition(self.src_type_combo.currentText(), **kwargs)
        self.sm.set_source(src)


class SurfaceSubModel(QAbstractTableModel):
    """Table model for the internal surfaces of a single element."""

    COLUMNS = ['Surf#', 'Radius', 'Thickness', 'Glass', 'Semi-Diam', 'Conic',
                'Radius Y', 'Conic Y']

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._elem_idx = -1

    def set_element(self, elem_idx):
        self.beginResetModel()
        self._elem_idx = elem_idx
        self.endResetModel()

    def _surfaces(self):
        if 0 <= self._elem_idx < len(self.sm.elements):
            return self.sm.elements[self._elem_idx].surfaces
        return []

    def rowCount(self, parent=QModelIndex()):
        return len(self._surfaces())

    def columnCount(self, parent=QModelIndex()):
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        surfs = self._surfaces()
        if not index.isValid() or index.row() >= len(surfs):
            return None
        s = surfs[index.row()]
        col = index.column()

        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 0:
                return str(index.row() + 1)
            elif col == 1:
                return 'Infinity' if np.isinf(s.radius) else f'{s.radius:.6g}'
            elif col == 2:
                return f'{s.thickness:.6g}'
            elif col == 3:
                return s.glass
            elif col == 4:
                return 'Infinity' if np.isinf(s.semi_diameter) else f'{s.semi_diameter:.6g}'
            elif col == 5:
                return f'{s.conic:.6g}'
            elif col == 6:
                if s.radius_y is None:
                    return ''
                return 'Infinity' if np.isinf(s.radius_y) else f'{s.radius_y:.6g}'
            elif col == 7:
                if s.conic_y is None:
                    return ''
                return f'{s.conic_y:.6g}'

        elif role == Qt.ForegroundRole:
            return QColor(220, 232, 248)

        elif role == Qt.BackgroundRole:
            if s.glass:
                return QColor(35, 50, 75)

        elif role == Qt.TextAlignmentRole:
            if col in (1, 2, 4, 5, 6, 7):
                return Qt.AlignRight | Qt.AlignVCenter

        return None

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole:
            return False
        col = index.column()
        field_map = {1: 'radius', 2: 'thickness', 3: 'glass',
                     4: 'semi_diameter', 5: 'conic',
                     6: 'radius_y', 7: 'conic_y'}
        field = field_map.get(col)
        if not field:
            return False
        return self.sm.set_surface_field(self._elem_idx, index.row(), field, value)


# ────────────────────────────────────────────────────────────────────────
# Main editor widget
# ────────────────────────────────────────────────────────────────────────

class ElementTableEditor(QWidget):
    """Complete element editor with toolbar, table, and detail panel."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar ──
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)

        self.btn_up = QPushButton('\u25b2')
        self.btn_up.setFixedWidth(28)
        self.btn_up.setToolTip('Move element up (Alt+\u2191)')
        self.btn_up.clicked.connect(self._move_up)
        btn_up = self.btn_up

        self.btn_down = QPushButton('\u25bc')
        self.btn_down.setFixedWidth(28)
        self.btn_down.setToolTip('Move element down (Alt+\u2193)')
        self.btn_down.clicked.connect(self._move_down)
        btn_down = self.btn_down

        self.btn_delete = QPushButton('Delete')
        self.btn_delete.setToolTip('Delete selected element (Ctrl+D)')
        self.btn_delete.clicked.connect(self._delete_element)
        btn_delete = self.btn_delete

        self.btn_group = QPushButton('Group')
        self.btn_group.setToolTip(
            'Merge two or more selected elements into a single compound '
            'optic.  Cemented interfaces are detected automatically.')
        self.btn_group.clicked.connect(self._group_selected)
        self.btn_group.setEnabled(False)
        btn_group = self.btn_group

        self.btn_ungroup = QPushButton('Ungroup')
        self.btn_ungroup.setToolTip(
            'Split the selected compound element back into its component '
            'lenses.  Disabled on elements with fewer than 2 surfaces.')
        self.btn_ungroup.clicked.connect(self._ungroup_selected)
        self.btn_ungroup.setEnabled(False)
        btn_ungroup = self.btn_ungroup

        # Coordinate mode: a checkbox-style toggle that reads
        # "Absolute coordinates" and goes pressed-in when active.  No
        # more ambiguity about whether the label shows current or next
        # state.
        self.btn_coord = QPushButton('Absolute coordinates')
        self.btn_coord.setToolTip(
            'When off (default): Distance column shows the gap from the '
            'previous element.\n'
            'When on: Distance column shows each element\'s absolute Z '
            'position along the optical axis.')
        self.btn_coord.setCheckable(True)
        self.btn_coord.setChecked(False)
        self.btn_coord.clicked.connect(self._toggle_coord_mode)

        lbl_wv = QLabel('wv(nm):')
        lbl_wv.setToolTip('Design wavelength.  Drives glass index lookup, '
                          'Airy disc, OPD.')
        self.inp_wv = QDoubleSpinBox()
        self.inp_wv.setRange(100.0, 20000.0)
        self.inp_wv.setDecimals(2)
        self.inp_wv.setSingleStep(10.0)
        self.inp_wv.setSuffix(' nm')
        self.inp_wv.setFixedWidth(90)
        self.inp_wv.setValue(self.sm.wavelength_nm)
        self.inp_wv.setToolTip(lbl_wv.toolTip())
        self.inp_wv.valueChanged.connect(self._queue_set_wavelength)

        lbl_epd = QLabel('EPD(mm):')
        lbl_epd.setToolTip('Entrance pupil diameter.  Caps the input ray '
                           'bundle + sets the default semi-aperture.')
        self.inp_epd = QDoubleSpinBox()
        self.inp_epd.setRange(0.001, 1000.0)
        self.inp_epd.setDecimals(3)
        self.inp_epd.setSingleStep(0.5)
        self.inp_epd.setSuffix(' mm')
        self.inp_epd.setFixedWidth(90)
        self.inp_epd.setValue(self.sm.epd_mm)
        self.inp_epd.setToolTip(lbl_epd.toolTip())
        self.inp_epd.valueChanged.connect(self._queue_set_epd)

        # Debounce applies so dragging a spinbox doesn't retrace 10x.
        self._wv_timer = QTimer(self)
        self._wv_timer.setSingleShot(True)
        self._wv_timer.setInterval(150)
        self._wv_timer.timeout.connect(self._set_wavelength)
        self._epd_timer = QTimer(self)
        self._epd_timer.setSingleShot(True)
        self._epd_timer.setInterval(150)
        self._epd_timer.timeout.connect(self._set_epd)

        # Surface/element view toggle
        self.btn_view = QPushButton('Element View')
        self.btn_view.setToolTip('Toggle between element view and surface view')
        self.btn_view.setCheckable(True)
        self.btn_view.clicked.connect(self._toggle_view_mode)
        self._view_mode = 'element'  # or 'surface'

        # Search box: filter rows whose Name doesn't match a substring.
        self.inp_search = QLineEdit()
        self.inp_search.setPlaceholderText('Search elements...')
        self.inp_search.setFixedWidth(160)
        self.inp_search.setClearButtonEnabled(True)
        self.inp_search.setToolTip(
            'Hide rows whose Name does not contain this substring '
            '(case-insensitive).  Clear the box to show all rows.')
        self.inp_search.textChanged.connect(self._apply_search)

        toolbar.addWidget(btn_up)
        toolbar.addWidget(btn_down)
        toolbar.addWidget(btn_delete)
        toolbar.addWidget(btn_group)
        toolbar.addWidget(btn_ungroup)
        toolbar.addWidget(self.btn_coord)
        toolbar.addWidget(self.btn_view)
        toolbar.addWidget(self.inp_search)
        toolbar.addStretch()
        toolbar.addWidget(lbl_wv)
        toolbar.addWidget(self.inp_wv)
        toolbar.addWidget(lbl_epd)
        toolbar.addWidget(self.inp_epd)

        layout.addLayout(toolbar)

        # ── Splitter: element table + detail panel ──
        splitter = QSplitter(Qt.Vertical)

        # Table models (both created, only one active at a time)
        self.table_model = ElementTableModel(self.sm)
        self.surface_model = SurfaceFlatModel(self.sm)
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setItemDelegateForColumn(2, TypeDelegate(self.table))

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setStretchLastSection(True)
        widths = {0: 45, 1: 140, 2: 75, 3: 90, 4: 55, 5: 55, 6: 70, 7: 70}
        for c, w in widths.items():
            self.table.setColumnWidth(c, w)

        # Connect selection to detail panel AND to button-enable logic
        self.table.selectionModel().currentRowChanged.connect(self._on_row_selected)
        self.table.selectionModel().selectionChanged.connect(
            self._update_toolbar_enabled)

        # Right-click context menu on element rows.
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            self._on_table_context_menu)
        # Re-apply the search filter whenever rows change (otherwise
        # newly-inserted elements would appear regardless of filter).
        self.sm.system_changed.connect(self._apply_search)

        splitter.addWidget(self.table)

        # Detail panel
        self.detail_panel = SurfaceDetailPanel(self.sm)
        splitter.addWidget(self.detail_panel)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, stretch=1)

        # ── Info bar ──
        self.info_bar = QLabel()
        self.info_bar.setStyleSheet(
            "QLabel { background: #12161e; color: #7a94b8; "
            "padding: 4px 8px; font-size: 12px; }")
        self.info_bar.setToolTip(
            'System summary.\n'
            '  EFL / BFL: paraxial focal lengths\n'
            '  f/# = |EFL| / EPD\n'
            '  rays = alive / total in latest trace\n'
            '  STALE = an edit has been made since the last trace finished')
        layout.addWidget(self.info_bar)

        # Connect
        self.sm.system_changed.connect(self._update_info)
        self.sm.system_changed.connect(self._mark_stale)
        self.sm.trace_ready.connect(self._on_trace_ready)
        self.sm.display_changed.connect(self._update_info)
        self.sm.element_selected.connect(self._select_row)
        self._last_trace = None   # most recent TraceResult, for stats
        self._stale = False
        self._update_info()

    def _on_row_selected(self, current, previous):
        self.detail_panel.show_element(current.row())
        self._update_toolbar_enabled()

    def _update_toolbar_enabled(self, *args):
        """Enable/disable Group/Ungroup/Delete based on current selection."""
        rows = self.table.selectionModel().selectedRows()
        n_sel = len(rows)
        # Group needs >=2 selected; they must all be editable rows.
        editable = []
        for r in rows:
            if 0 < r.row() < len(self.sm.elements) - 1:
                editable.append(r.row())
        self.btn_group.setEnabled(len(editable) >= 2)

        # Ungroup: one editable element with >=2 surfaces.
        cur = self.table.currentIndex().row()
        can_ungroup = False
        if 0 < cur < len(self.sm.elements) - 1:
            elem = self.sm.elements[cur]
            can_ungroup = len(getattr(elem, 'surfaces', [])) >= 2
        self.btn_ungroup.setEnabled(can_ungroup)

        # Delete: any row except Source (0) and Detector (last).
        self.btn_delete.setEnabled(
            0 < cur < len(self.sm.elements) - 1)

    def _move_up(self):
        idx = self.table.currentIndex().row()
        self.sm.move_element(idx, -1)
        if idx > 1:
            self.table.selectRow(idx - 1)

    def _move_down(self):
        idx = self.table.currentIndex().row()
        self.sm.move_element(idx, 1)
        if idx < len(self.sm.elements) - 2:
            self.table.selectRow(idx + 1)

    def _delete_element(self):
        idx = self.table.currentIndex().row()
        self.sm.delete_element(idx)

    # --- Context menu + search ---------------------------------------

    def _on_table_context_menu(self, pos):
        """Right-click menu on the element table."""
        from PySide6.QtWidgets import QMenu
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        if row < 0 or row >= len(self.sm.elements):
            return
        elem = self.sm.elements[row]
        is_endpoint = elem.elem_type in ('Source', 'Detector')

        m = QMenu(self)
        act_dup    = m.addAction('Duplicate Element')
        act_del    = m.addAction('Delete Element')
        m.addSeparator()
        act_up     = m.addAction('Move Up')
        act_down   = m.addAction('Move Down')
        m.addSeparator()
        # Toggle "distance" optimization variable for this element.
        has_dist_var = any(
            v[0] == row and v[2] == 'distance'
            for v in self.sm.opt_variables)
        act_var = m.addAction(
            'Unset Distance as Variable' if has_dist_var
            else 'Set Distance as Variable')

        # Endpoints can't be moved / duplicated / deleted.
        for a in (act_dup, act_del, act_up, act_down):
            a.setEnabled(not is_endpoint)
        # Source has no meaningful distance variable.
        if elem.elem_type == 'Source':
            act_var.setEnabled(False)

        chosen = m.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_dup:
            self._duplicate_element(row)
        elif chosen is act_del:
            self.sm.delete_element(row)
        elif chosen is act_up:
            self.sm.move_element(row, -1)
        elif chosen is act_down:
            self.sm.move_element(row, 1)
        elif chosen is act_var:
            self._toggle_distance_variable(row)

    def _duplicate_element(self, row):
        """Insert a deep-copy of the element at row+1 (before Detector)."""
        if row <= 0 or row >= len(self.sm.elements) - 1:
            return
        try:
            import copy
            src = self.sm.elements[row]
            new = copy.deepcopy(src)
            new.name = src.name + ' copy'
            self.sm.insert_element(row + 1, new)
        except Exception:
            pass

    def _toggle_distance_variable(self, row):
        """Add or remove (row, -1, 'distance') from opt_variables."""
        try:
            self.sm._checkpoint()
            key = (row, -1, 'distance')
            existing = [v for v in self.sm.opt_variables
                        if v[0] == row and v[2] == 'distance']
            if existing:
                self.sm.opt_variables = [
                    v for v in self.sm.opt_variables
                    if not (v[0] == row and v[2] == 'distance')]
            else:
                self.sm.opt_variables.append(key)
            self.sm.system_changed.emit()
        except Exception:
            pass

    def _apply_search(self, *args):
        """Hide rows whose Name doesn't contain the search substring."""
        try:
            text = self.inp_search.text().strip().lower()
        except Exception:
            return
        # Resolve which model is currently shown.
        model = self.table.model()
        n_rows = model.rowCount() if model else 0
        for r in range(n_rows):
            if not text:
                self.table.setRowHidden(r, False)
                continue
            try:
                # Element view: name in col 1.  Surface view: name in col 1 too.
                name = model.data(model.index(r, 1), Qt.DisplayRole) or ''
                self.table.setRowHidden(r, text not in str(name).lower())
            except Exception:
                self.table.setRowHidden(r, False)

    def _group_selected(self):
        """Group selected elements into a single compound element."""
        selection = self.table.selectionModel().selectedRows()
        if len(selection) < 2:
            return
        indices = sorted(idx.row() for idx in selection)

        from PySide6.QtWidgets import QInputDialog
        # Default name from first selected element
        first = self.sm.elements[indices[0]] if indices[0] < len(self.sm.elements) else None
        default_name = first.name if first else 'Compound'
        name, ok = QInputDialog.getText(
            self, 'Group Elements',
            'Name for compound element:', text=default_name)
        if ok and name:
            self.sm.group_elements(indices, name)

    def _ungroup_selected(self):
        """Split the selected element back into individual elements."""
        idx = self.table.currentIndex().row()
        if 0 < idx < len(self.sm.elements) - 1:
            elem = self.sm.elements[idx]
            if len(elem.surfaces) >= 2:
                self.sm.ungroup_element(idx)
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, 'Ungroup',
                    f'Element "{elem.name}" has only {len(elem.surfaces)} surface(s) '
                    f'— nothing to ungroup.')

    def _toggle_coord_mode(self):
        if self.btn_coord.isChecked():
            self.sm.set_coordinate_mode('absolute')
        else:
            self.sm.set_coordinate_mode('relative')

    def _toggle_view_mode(self):
        """Toggle between element view and surface (flat) view."""
        if self._view_mode == 'element':
            self._view_mode = 'surface'
            self.table.setModel(self.surface_model)
            self.btn_view.setText('Surface View')
            # Adjust column widths for surface view
            for c in range(self.surface_model.columnCount()):
                self.table.setColumnWidth(c, 90)
        else:
            self._view_mode = 'element'
            self.table.setModel(self.table_model)
            self.btn_view.setText('Element View')
            widths = {0: 45, 1: 140, 2: 75, 3: 90, 4: 55, 5: 55, 6: 70, 7: 70}
            for c, w in widths.items():
                self.table.setColumnWidth(c, w)
        # Reconnect selection
        self.table.selectionModel().currentRowChanged.connect(self._on_row_selected)

    def _queue_set_wavelength(self, _):
        self._wv_timer.start()

    def _queue_set_epd(self, _):
        self._epd_timer.start()

    def _set_wavelength(self):
        wv = float(self.inp_wv.value())
        if wv > 0:
            self.sm.set_wavelength(wv)

    def _set_epd(self):
        epd = float(self.inp_epd.value())
        if epd > 0:
            self.sm.set_epd(epd)

    def _select_row(self, elem_index):
        if 0 <= elem_index < self.table_model.rowCount():
            self.table.selectRow(elem_index)
            self.table.scrollTo(self.table_model.index(elem_index, 0))

    def _mark_stale(self):
        self._stale = True
        self._update_info()

    def _on_trace_ready(self, result):
        self._last_trace = result
        self._stale = False
        self._update_info()

    def _update_info(self):
        try:
            abcd, efl, bfl = self.sm.get_abcd()
            efl_str = f'{efl * 1e3:.2f}' if np.isfinite(efl) else 'inf'
            bfl_str = f'{bfl * 1e3:.2f}' if np.isfinite(bfl) else 'inf'
            fnum = abs(efl) / self.sm.epd_m if np.isfinite(efl) and self.sm.epd_m > 0 else np.inf
            fnum_str = f'{fnum:.2f}' if np.isfinite(fnum) else 'inf'
            mode = self.sm.coordinate_mode.capitalize()

            # Trace throughput / stale indicator.
            rays_str = '(no trace)'
            if self._last_trace is not None:
                try:
                    r = self._last_trace.image_rays
                    n_alive = int(np.sum(r.alive))
                    n_tot = r.n_rays
                    vign = 100.0 * (1 - n_alive / n_tot) if n_tot > 0 else 0
                    rays_str = f'rays: {n_alive}/{n_tot} ({vign:.0f}% vign.)'
                except Exception:
                    pass
            state = ('[STALE \u25CF]' if self._stale
                     else '[OK \u2713]' if self._last_trace is not None
                     else '')

            self.info_bar.setText(
                f'{state}  EFL: {efl_str} mm | BFL: {bfl_str} mm | '
                f'f/#: {fnum_str} | wv: {self.sm.wavelength_nm:.1f} nm | '
                f'Elements: {self.sm.num_elements - 2} | {rays_str} | '
                f'Coords: {mode}'
            )
        except Exception as e:
            try:
                from .diagnostics import diag
                diag.report('info-bar', e)
            except Exception:
                pass
            self.info_bar.setText(f'Error: {e}')
