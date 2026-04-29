"""
Snapshots dock — named point-in-time saves of the full design state.

Unlike undo/redo history (which is linear and lost at shutdown),
snapshots are user-labeled and persist for the session so A/B
comparisons stay findable.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QInputDialog, QMessageBox,
)

from .model import SystemModel


class SnapshotsDock(QWidget):
    """Tiny A/B-comparison panel: save named states, click to swap back."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        help_lbl = QLabel(
            'Name-and-save points in your design process.  Click a row '
            'to restore it.  Snapshots survive edits but are forgotten '
            'when the app closes (autosave handles the latest state).')
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet('color:#7a94b8; font-size:11px;')
        layout.addWidget(help_lbl)

        self.list = QListWidget()
        self.list.setToolTip(
            'Double-click a snapshot to restore it.  '
            'Loading a snapshot is itself undoable.')
        self.list.itemDoubleClicked.connect(self._load_selected)
        layout.addWidget(self.list, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_save = QPushButton('Save current')
        self.btn_save.setToolTip(
            'Save the current system under a user-chosen name.')
        self.btn_save.clicked.connect(self._save)
        self.btn_load = QPushButton('Load selected')
        self.btn_load.clicked.connect(self._load_selected)
        self.btn_delete = QPushButton('Delete')
        self.btn_delete.clicked.connect(self._delete)
        self.btn_compare = QPushButton('Compare selected to current')
        self.btn_compare.setToolTip(
            'Show a Delta table (EFL/BFL/f-number/merit) between the '
            'highlighted snapshot and the current system, plus an '
            'overlaid ray-fan diff if both traced successfully.')
        self.btn_compare.clicked.connect(self._compare)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_compare)
        btn_row.addWidget(self.btn_delete)
        layout.addLayout(btn_row)

        self.sm.snapshots_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        self.list.clear()
        for i, snap in enumerate(self.sm.snapshots):
            efl = snap.get('efl_mm', float('nan'))
            efl_txt = (f'EFL {efl:.2f} mm' if np.isfinite(efl)
                       else 'EFL n/a')
            item = QListWidgetItem(f'{i + 1}. {snap["name"]}   \u2014   {efl_txt}')
            self.list.addItem(item)

    def _save(self):
        name, ok = QInputDialog.getText(
            self, 'Save Snapshot', 'Snapshot name:',
            text=f'Snapshot {len(self.sm.snapshots) + 1}')
        if ok and name:
            self.sm.save_snapshot(name)

    def _load_selected(self, *args):
        idx = self.list.currentRow()
        if idx < 0:
            return
        self.sm.load_snapshot(idx)

    def _compare(self):
        idx = self.list.currentRow()
        if idx < 0 or idx >= len(self.sm.snapshots):
            QMessageBox.information(
                self, 'Compare', 'Select a snapshot first.')
            return
        snap = self.sm.snapshots[idx]
        cur = self.sm.to_prescription()
        try:
            from lumenairy.raytrace import (
                surfaces_from_prescription, system_abcd)
            wv = self.sm.wavelength_nm * 1e-9
            cur_surfs = surfaces_from_prescription(cur)
            _, efl_c, bfl_c, _ = system_abcd(cur_surfs, wv)
            snap_pres = snap.get('prescription') or snap.get('pres')
            if snap_pres is None:
                # Old snapshots may only have the state dict; reconstruct
                # the prescription by loading and re-grabbing.
                QMessageBox.information(
                    self, 'Compare',
                    'This snapshot predates the compare feature -- '
                    're-save it after loading to capture its prescription.')
                return
            snap_surfs = surfaces_from_prescription(snap_pres)
            _, efl_s, bfl_s, _ = system_abcd(snap_surfs, wv)
            lines = ['      current       snapshot        delta',
                     f'EFL   {efl_c*1e3:9.4f} mm  {efl_s*1e3:9.4f} mm  '
                     f'{(efl_c-efl_s)*1e3:+.4f} mm',
                     f'BFL   {bfl_c*1e3:9.4f} mm  {bfl_s*1e3:9.4f} mm  '
                     f'{(bfl_c-bfl_s)*1e3:+.4f} mm']
            if self.sm.epd_m > 0:
                fc = abs(efl_c) / self.sm.epd_m
                fs = abs(efl_s) / self.sm.epd_m
                lines.append(f'f/#   {fc:11.3f}  {fs:11.3f}  '
                             f'{fc-fs:+8.4f}')
            QMessageBox.information(
                self, f'Compare -> snapshot "{snap["name"]}"',
                '\n'.join(lines))
        except Exception as e:
            QMessageBox.warning(self, 'Compare failed', str(e))

    def _delete(self):
        idx = self.list.currentRow()
        if idx < 0:
            return
        if QMessageBox.question(
                self, 'Delete snapshot',
                f'Delete snapshot "{self.sm.snapshots[idx]["name"]}"?',
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.sm.delete_snapshot(idx)
