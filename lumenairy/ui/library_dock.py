"""
Library dock — browse and manage saved materials, lenses, and phase masks.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QListWidget, QListWidgetItem, QGroupBox,
    QFormLayout, QLineEdit, QDoubleSpinBox, QInputDialog,
    QMessageBox, QTextEdit, QComboBox,
)
from PySide6.QtGui import QFont

from .model import SystemModel
from ..user_library import (
    save_material, load_material, list_materials, delete_material,
    register_fixed_glass,
    save_lens, load_lens, list_lenses, delete_lens,
    save_phase_mask, load_phase_mask, load_phase_mask_info,
    list_phase_masks, delete_phase_mask,
    get_library_path,
)


class LibraryDock(QWidget):
    """Browse and manage the user library of materials, lenses, and masks."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Library path label
        path_label = QLabel(f'Library: {get_library_path()}')
        path_label.setStyleSheet("color: #7a94b8; font-size: 10px;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_materials_tab(), 'Materials')
        tabs.addTab(self._build_lenses_tab(), 'Lenses')
        tabs.addTab(self._build_masks_tab(), 'Phase Masks')
        layout.addWidget(tabs)

    # ── Materials tab ─────────────────────────────────────────────

    def _build_materials_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        self.mat_list = QListWidget()
        self.mat_list.currentItemChanged.connect(self._on_mat_selected)
        layout.addWidget(self.mat_list)

        self.mat_info = QLabel('')
        self.mat_info.setWordWrap(True)
        self.mat_info.setStyleSheet("color: #a0b4d0; font-size: 12px; padding: 4px;")
        layout.addWidget(self.mat_info)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn('+ Fixed Index', self._add_fixed_material))
        btn_row.addWidget(self._btn('+ From Catalog', self._add_catalog_material))
        btn_row.addWidget(self._btn('Delete', self._delete_material))
        btn_row.addWidget(self._btn('Use', self._use_material))
        layout.addLayout(btn_row)

        self._refresh_materials()
        return w

    def _refresh_materials(self):
        self.mat_list.clear()
        for name in list_materials():
            self.mat_list.addItem(name)

    def _on_mat_selected(self, current, prev):
        if current is None:
            self.mat_info.setText('')
            return
        name = current.text()
        try:
            data = load_material(name)
            if data.get('type') == 'fixed':
                self.mat_info.setText(f'{name}: n = {data["n"]:.5f} (fixed)')
            elif data.get('type') == 'catalog':
                from ..glass import get_glass_index
                n = get_glass_index(name, 550e-9)
                self.mat_info.setText(
                    f'{name}: n(550nm) = {n:.5f}\n'
                    f'Source: {data["shelf"]}/{data["book"]}/{data["page"]}')
            else:
                self.mat_info.setText(f'{name}: {data.get("type", "?")}')
        except Exception as e:
            self.mat_info.setText(f'Error: {e}')

    def _add_fixed_material(self):
        name, ok = QInputDialog.getText(self, 'New Material', 'Material name:')
        if not ok or not name:
            return
        n, ok = QInputDialog.getDouble(self, 'Refractive Index',
                                        'n:', 1.5, 1.0, 5.0, 5)
        if not ok:
            return
        save_material(name, n=n)
        self._refresh_materials()

    def _add_catalog_material(self):
        from ..glass import GLASS_REGISTRY
        glasses = sorted(GLASS_REGISTRY.keys())
        glass, ok = QInputDialog.getItem(self, 'Add from Catalog',
                                          'Glass:', glasses, 0, False)
        if not ok:
            return
        shelf, book, page = GLASS_REGISTRY[glass]
        name, ok = QInputDialog.getText(self, 'Save As', 'Name:', text=glass)
        if ok and name:
            save_material(name, shelf=shelf, book=book, page=page)
            self._refresh_materials()

    def _delete_material(self):
        item = self.mat_list.currentItem()
        if item:
            delete_material(item.text())
            self._refresh_materials()

    def _use_material(self):
        """Load the selected material into the glass registry."""
        item = self.mat_list.currentItem()
        if item:
            load_material(item.text())
            self.mat_info.setText(f'{item.text()} loaded into glass registry.')

    # ── Lenses tab ────────────────────────────────────────────────

    def _build_lenses_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        self.lens_list = QListWidget()
        self.lens_list.currentItemChanged.connect(self._on_lens_selected)
        layout.addWidget(self.lens_list)

        self.lens_info = QLabel('')
        self.lens_info.setWordWrap(True)
        self.lens_info.setStyleSheet("color: #a0b4d0; font-size: 12px; padding: 4px;")
        layout.addWidget(self.lens_info)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn('Save Current', self._save_current_lens))
        btn_row.addWidget(self._btn('Load', self._load_lens))
        btn_row.addWidget(self._btn('Delete', self._delete_lens))
        layout.addLayout(btn_row)

        self._refresh_lenses()
        return w

    def _refresh_lenses(self):
        self.lens_list.clear()
        for name in list_lenses():
            self.lens_list.addItem(name)

    def _on_lens_selected(self, current, prev):
        if current is None:
            self.lens_info.setText('')
            return
        name = current.text()
        try:
            rx = load_lens(name)
            n_surf = len(rx.get('surfaces', []))
            ap = rx.get('aperture_diameter', 0)
            self.lens_info.setText(
                f'{rx.get("name", name)}\n'
                f'{n_surf} surfaces, aperture = {ap*1e3:.1f} mm')
        except Exception as e:
            self.lens_info.setText(f'Error: {e}')

    def _save_current_lens(self):
        """Save the current system prescription to the library."""
        rx = self.sm.to_prescription()
        name, ok = QInputDialog.getText(self, 'Save Lens', 'Name:',
                                         text=rx.get('name', 'My Lens'))
        if ok and name:
            save_lens(name, rx)
            self._refresh_lenses()

    def _load_lens(self):
        """Load a lens prescription into the current system."""
        item = self.lens_list.currentItem()
        if not item:
            return
        try:
            rx = load_lens(item.text())
            self.sm.load_prescription(rx, self.sm.wavelength_nm)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def _delete_lens(self):
        item = self.lens_list.currentItem()
        if item:
            delete_lens(item.text())
            self._refresh_lenses()

    # ── Phase masks tab ───────────────────────────────────────────

    def _build_masks_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        self.mask_list = QListWidget()
        self.mask_list.currentItemChanged.connect(self._on_mask_selected)
        layout.addWidget(self.mask_list)

        self.mask_info = QLabel('')
        self.mask_info.setWordWrap(True)
        self.mask_info.setStyleSheet("color: #a0b4d0; font-size: 12px; padding: 4px;")
        layout.addWidget(self.mask_info)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn('+ Expression', self._add_expression_mask))
        btn_row.addWidget(self._btn('+ Glass Block', self._add_glass_block))
        btn_row.addWidget(self._btn('Delete', self._delete_mask))
        layout.addLayout(btn_row)

        self._refresh_masks()
        return w

    def _refresh_masks(self):
        self.mask_list.clear()
        for name in list_phase_masks():
            self.mask_list.addItem(name)

    def _on_mask_selected(self, current, prev):
        if current is None:
            self.mask_info.setText('')
            return
        try:
            info = load_phase_mask_info(current.text())
            mtype = info.get('type', '?')
            desc = info.get('description', '')
            if mtype == 'expression':
                self.mask_info.setText(
                    f'Type: expression\n'
                    f'Formula: {info["expression"]}\n'
                    f'{desc}')
            elif mtype == 'glass_block':
                self.mask_info.setText(
                    f'Type: glass block\n'
                    f'n = {info["n"]:.5f}, t = {info["thickness"]*1e3:.2f} mm\n'
                    f'{desc}')
            elif mtype == 'array':
                shape = info.get('shape', [])
                self.mask_info.setText(
                    f'Type: array {shape}\n'
                    f'dx = {info.get("dx", 0)*1e6:.2f} um\n'
                    f'{desc}')
            else:
                self.mask_info.setText(f'Type: {mtype}\n{desc}')
        except Exception as e:
            self.mask_info.setText(f'Error: {e}')

    def _add_expression_mask(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle('Define Phase Mask Expression')
        dlg.setMinimumWidth(400)
        form = QFormLayout(dlg)

        inp_name = QLineEdit('my_mask')
        form.addRow('Name:', inp_name)

        inp_expr = QLineEdit('arctan2(Y, X) * 2')
        inp_expr.setToolTip(
            'Variables: X, Y (metres), R (radius), THETA (angle), k (wavenumber), pi\n'
            'Functions: sin, cos, sqrt, exp, log, arctan2, mod, floor, ceil')
        form.addRow('Expression:', inp_expr)

        inp_desc = QLineEdit('')
        form.addRow('Description:', inp_desc)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() == QDialog.Accepted:
            save_phase_mask(inp_name.text(), expression=inp_expr.text(),
                           description=inp_desc.text())
            self._refresh_masks()

    def _add_glass_block(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle('Define Glass Block')
        dlg.setMinimumWidth(350)
        form = QFormLayout(dlg)

        inp_name = QLineEdit('glass_window')
        form.addRow('Name:', inp_name)

        inp_n = QDoubleSpinBox()
        inp_n.setRange(1.0, 5.0)
        inp_n.setValue(1.517)
        inp_n.setDecimals(5)
        form.addRow('Refractive index:', inp_n)

        inp_t = QDoubleSpinBox()
        inp_t.setRange(0.001, 1000)
        inp_t.setValue(5.0)
        inp_t.setDecimals(3)
        inp_t.setSuffix(' mm')
        form.addRow('Thickness:', inp_t)

        inp_desc = QLineEdit('')
        form.addRow('Description:', inp_desc)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() == QDialog.Accepted:
            save_phase_mask(inp_name.text(), n=inp_n.value(),
                           thickness=inp_t.value() * 1e-3,
                           description=inp_desc.text())
            self._refresh_masks()

    def _delete_mask(self):
        item = self.mask_list.currentItem()
        if item:
            delete_phase_mask(item.text())
            self._refresh_masks()

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _btn(text, callback):
        b = QPushButton(text)
        b.clicked.connect(callback)
        return b
