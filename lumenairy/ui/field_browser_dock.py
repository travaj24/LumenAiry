"""
HDF5 / Zarr plane browser dock.

Lists every plane in a saved field file and previews the selected
one (intensity + wrapped phase).  Lets the user push any plane back
into the Zernike, Interferometry, or PSF/MTF docks without writing
a one-off script.

Author: Andrew Traverso
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QFileDialog, QTextEdit, QGroupBox, QComboBox,
)
from PySide6.QtGui import QFont

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class FieldBrowserDock(QWidget):
    """Browse and reload saved wave-field planes (HDF5 / Zarr)."""

    def __init__(self, system_model, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._filepath = None
        self._planes = []
        self._current_field = None
        self._current_dx = None
        self._current_wavelength = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # File picker
        file_row = QHBoxLayout()
        self.btn_open = QPushButton('Open HDF5/Zarr...')
        self.btn_open.clicked.connect(self._open_file)
        file_row.addWidget(self.btn_open)
        self.lbl_file = QLabel('(no file loaded)')
        self.lbl_file.setStyleSheet('color:#7a94b8;')
        file_row.addWidget(self.lbl_file, stretch=1)
        layout.addLayout(file_row)

        # Plane list
        self.list_planes = QListWidget()
        self.list_planes.itemSelectionChanged.connect(self._preview_selected)
        self.list_planes.setMaximumHeight(160)
        layout.addWidget(self.list_planes)

        # Route target
        route_box = QGroupBox('Send selected plane to')
        route_row = QHBoxLayout(route_box)
        self.combo_target = QComboBox()
        self.combo_target.addItems([
            'Zernike dock (decompose OPD)',
            'Interferometry dock',
            'PSF/MTF dock (use as pupil)',
        ])
        route_row.addWidget(self.combo_target)
        self.btn_route = QPushButton('Send')
        self.btn_route.clicked.connect(self._route_selected)
        route_row.addWidget(self.btn_route)
        route_row.addStretch()
        layout.addWidget(route_box)

        # Preview
        if HAS_MPL:
            self.fig = Figure(figsize=(6, 3), tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, stretch=1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(100)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            "QTextEdit{background:#0a0c10;color:#7a94b8;border:none}")
        layout.addWidget(self.summary)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open saved field file',
            filter='Wave files (*.h5 *.hdf5 *.zarr);;All (*)')
        if not path:
            return
        try:
            from lumenairy.storage import list_planes
            self._filepath = path
            self._planes = list_planes(path)
        except Exception as e:
            self.summary.append(f'list_planes failed: {e}')
            return
        self.lbl_file.setText(Path(path).name)
        self.list_planes.clear()
        for p in self._planes:
            label = p.get('label', '?')
            z = p.get('z', 0.0)
            dx = p.get('dx', 0.0)
            self.list_planes.addItem(
                f'{label}  @ z={z*1e3:.3f} mm  dx={dx*1e6:.2f} um')
        self.summary.append(f'{len(self._planes)} planes in {path}')

    def _preview_selected(self):
        sel = self.list_planes.currentRow()
        if sel < 0 or sel >= len(self._planes):
            return
        plane = self._planes[sel]
        try:
            from lumenairy.storage import load_plane_by_label
            field, dx, meta = load_plane_by_label(
                self._filepath, plane['label'])
            self._current_field = field
            self._current_dx = float(dx)
            self._current_wavelength = float(
                meta.get('wavelength', 1.0e-6))
            if HAS_MPL:
                self.fig.clear()
                ax1 = self.fig.add_subplot(1, 2, 1)
                ax2 = self.fig.add_subplot(1, 2, 2)
                I = np.abs(field) ** 2
                ax1.imshow(I / max(I.max(), 1e-30), cmap='magma',
                           origin='lower')
                ax1.set_title(f'|E|^2 @ {plane["label"]}')
                ax1.set_xticks([]); ax1.set_yticks([])
                ax2.imshow(np.angle(field), cmap='twilight',
                           origin='lower', vmin=-np.pi, vmax=np.pi)
                ax2.set_title('phase')
                ax2.set_xticks([]); ax2.set_yticks([])
                self.canvas.draw()
            self.summary.append(
                f'Loaded {plane["label"]}: {field.shape}, '
                f'lambda={self._current_wavelength*1e9:.1f} nm')
        except Exception as e:
            self.summary.append(f'load_plane failed: {type(e).__name__}: {e}')

    def _route_selected(self):
        if self._current_field is None:
            self.summary.append('No plane selected.')
            return
        tgt = self.combo_target.currentIndex()
        mw = self.window()
        try:
            if tgt == 0:
                dock = getattr(mw, 'zernike_widget', None)
                if dock is None:
                    self.summary.append('Zernike dock not present.')
                    return
                dock.set_field(
                    self._current_field, self._current_dx,
                    self._current_wavelength,
                    aperture=float(self.sm.epd_m))
                self.summary.append('Sent to Zernike dock.')
            elif tgt == 1:
                dock = getattr(mw, 'interferometry_widget', None)
                if dock is None:
                    self.summary.append('Interferometry dock not present.')
                    return
                try:
                    from lumenairy.analysis import wave_opd_2d
                    _, _, opd = wave_opd_2d(
                        self._current_field, self._current_dx,
                        self._current_wavelength,
                        aperture=float(self.sm.epd_m))
                    dock._opd = opd
                    dock._dx = self._current_dx
                    dock._aperture = float(self.sm.epd_m)
                    dock._wavelength = self._current_wavelength
                    dock.summary.append('OPD installed from field browser.')
                    self.summary.append('Sent to Interferometry dock.')
                except Exception as e:
                    self.summary.append(f'OPD build failed: {e}')
            else:
                dock = getattr(mw, 'psfmtf_widget', None)
                if dock is None:
                    self.summary.append('PSF/MTF dock not present.')
                    return
                dock._pupil = self._current_field
                dock._dx = self._current_dx
                dock._wavelength = self._current_wavelength
                dock._focal_length = 0.0
                dock.summary.append('Pupil installed from field browser.')
                self.summary.append('Sent to PSF/MTF dock.')
        except Exception as e:
            self.summary.append(
                f'Route failed: {type(e).__name__}: {e}')
