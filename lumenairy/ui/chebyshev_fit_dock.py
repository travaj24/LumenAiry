# v5.4 Phase 5: library now ships chebyshev_fit_2d.
#
# AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24.md Part 6.3 P3-C originally
# noted that six library sites consume the shared Chebyshev helpers
# extracted into ``lumenairy/_math/chebyshev.py`` in v5.2.0, but no UI
# offered a "fit measured freeform data to Chebyshev polynomials"
# workflow.  This dock fills that gap: it loads a measured z(x, y)
# height map, fits it to a 2-D first-kind Chebyshev product basis,
# and either exports the coefficients or stamps them into the current
# system's prescription as a ``freeform_type='chebyshev'`` surface
# (matching the convention of
# ``lumenairy.elements.freeform.surface_sag_chebyshev``).
#
# v5.4 Phase 5 follow-up: the inline ``chebvander2d`` + ``lstsq``
# block has been promoted into the library as
# ``lumenairy._math.chebyshev.chebyshev_fit_2d`` (re-exported at the
# top level as ``lumenairy.chebyshev_fit_2d``).  The dock now defers
# the LS solve to that helper so every UI / notebook / batch tool
# shares a single tested fit primitive instead of each rolling its
# own Vandermonde + lstsq block.  The basis is identical to the one
# ``surface_sag_chebyshev`` evaluates, so the emitted
# ``{(i, j): c_ij}`` coefficients drop straight into the existing
# prescription dict format.
"""
Chebyshev freeform-surface fitter dock.

A small specialised metrology tool: load a measured 2-D height
map (e.g. interferometer or profilometer output) from
``.npy`` / ``.h5`` / ``.csv``, optionally mask outliers, fit it
to a tensor-product Chebyshev basis up to a chosen radial order,
and either export the coefficients or apply them to the current
``SystemModel`` prescription.

Author: Andrew Traverso
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QCheckBox, QFileDialog, QGroupBox, QTableWidget,
    QTableWidgetItem, QMessageBox, QHeaderView,
)
from PySide6.QtGui import QFont

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ----------------------------------------------------------------------
# File loader -- supports .npy / .h5 / .csv 2-D height maps.
# ----------------------------------------------------------------------

def _load_height_map(path: str) -> np.ndarray:
    """Load a 2-D height map from disk.

    Supports ``.npy``, ``.csv`` (delimiter auto-detected), and ``.h5``
    / ``.hdf5`` (first 2-D dataset found).  Returns a float64 ndarray
    of shape ``(Ny, Nx)``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.csv':
        # Try comma first, fall back to whitespace.
        try:
            arr = np.loadtxt(path, delimiter=',')
        except ValueError:
            arr = np.loadtxt(path)
    elif ext in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            arr = None
            for key in f.keys():
                d = f[key]
                if hasattr(d, 'shape') and d.ndim == 2:
                    arr = np.array(d)
                    break
            if arr is None:
                raise ValueError(
                    'no 2-D dataset found in HDF5 file')
    else:
        raise ValueError(f'unsupported extension: {ext}')
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f'expected a 2-D array, got shape {arr.shape}')
    return arr


def _fit_chebyshev_2d(z: np.ndarray, max_order: int,
                      mask: Optional[np.ndarray] = None,
                      normalize_aperture: bool = True
                      ) -> tuple:
    """Fit a tensor-product Chebyshev surface to a 2-D height map.

    v5.4 Phase 5: this wrapper now defers the LS solve to
    ``lumenairy._math.chebyshev.chebyshev_fit_2d`` (re-exported at
    the top level as ``lumenairy.chebyshev_fit_2d``).  It still
    returns the dock's historical 4-tuple
    ``(coeff_matrix, fit_map, residual, x_axis, y_axis)`` so the
    rest of the dock (table, plot, export, apply) keeps working
    unchanged; the coefficient matrix is reconstructed from the
    sparse ``{(i, j): c_ij}`` dict that the library helper returns.
    """
    from lumenairy._math.chebyshev import chebyshev_fit_2d
    from numpy.polynomial.chebyshev import chebval2d

    ny, nx = z.shape
    # Build pixel-centre coordinate axes.  When normalize_aperture is
    # True we pre-scale to [-1, 1] and pass normalize_xy=False so the
    # library doesn't redundantly rescale; otherwise we hand it raw
    # pixel-indexed axes and let it normalise.
    if normalize_aperture:
        x = np.linspace(-1.0, 1.0, nx)
        y = np.linspace(-1.0, 1.0, ny)
        normalize_xy = False
    else:
        x = np.linspace(-(nx - 1) / 2.0, (nx - 1) / 2.0, nx)
        y = np.linspace(-(ny - 1) / 2.0, (ny - 1) / 2.0, ny)
        normalize_xy = True

    # The library helper accepts a per-pixel weight array.  We build
    # one from the dock's binary mask (1 on keep, 0 on drop) and also
    # zero out non-finite z entries so the LS solve never sees a NaN.
    finite = np.isfinite(z)
    if mask is not None:
        if mask.shape != z.shape:
            raise ValueError(
                f'mask shape {mask.shape} does not match data '
                f'shape {z.shape}')
        weight = (finite & mask.astype(bool)).astype(np.float64)
    else:
        weight = finite.astype(np.float64)

    coeffs_dict, residual = chebyshev_fit_2d(
        x, y, z,
        n_max_x=max_order, n_max_y=max_order,
        weight=weight,
        normalize_xy=normalize_xy,
        return_residual=True,
    )

    # Reconstruct the (max_order+1, max_order+1) dense coefficient
    # matrix the dock's table / export code expects.  The library
    # helper drops near-zero terms from the dict; missing entries are
    # therefore exactly zero, so the reconstruction below is loss-
    # less and matches the historical ``coeffs_flat.reshape(...)``
    # output of the inline implementation.
    coeff_matrix = np.zeros(
        (max_order + 1, max_order + 1), dtype=np.float64)
    for (i, j), c in coeffs_dict.items():
        coeff_matrix[i, j] = c

    # Evaluate fit_map on the full grid for the residual plot.  We
    # don't reuse ``z - residual`` here because residual is NaN-
    # masked outside the weight support, but the user wants to see
    # the smooth extrapolated fit there.
    X, Y = np.meshgrid(x, y)
    fit_map = chebval2d(X, Y, coeff_matrix)
    return coeff_matrix, fit_map, residual, x, y


# ----------------------------------------------------------------------
# Main dock.
# ----------------------------------------------------------------------

class ChebyshevFitDock(QWidget):
    """Fit measured freeform-surface height maps to Chebyshev modes.

    The dock is decoupled from the wave-optics pipeline: it consumes
    its own height-map file and writes its result back via the
    ``Apply to prescription`` button (if a model is attached) or
    via the ``Export coefficients`` button.

    Parameters
    ----------
    model : SystemModel-like, optional
        Used by ``_on_apply`` to add a Chebyshev surface to the
        current prescription.  If ``None``, Apply is disabled.
    parent : QWidget, optional
    """

    def __init__(self, model: Any = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.sm = model
        self._z: Optional[np.ndarray] = None
        self._z_path: Optional[str] = None
        self._mask: Optional[np.ndarray] = None
        self._mask_path: Optional[str] = None
        self._coeffs: Optional[np.ndarray] = None
        self._fit_map: Optional[np.ndarray] = None
        self._residual: Optional[np.ndarray] = None
        self._x_axis: Optional[np.ndarray] = None
        self._y_axis: Optional[np.ndarray] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Data loader -----------------------------------------
        load_box = QGroupBox('Measured height map')
        load_row = QHBoxLayout(load_box)
        self.btn_load = QPushButton('Load data...')
        self.btn_load.setToolTip(
            'Load a 2-D height map from .npy, .csv, or .h5 / .hdf5.\n'
            'Expected units: arbitrary (the fit is unitless in z).')
        self.btn_load.clicked.connect(self._on_load)
        load_row.addWidget(self.btn_load)
        self.lbl_data = QLabel('(no file)')
        self.lbl_data.setStyleSheet('color:#7a94b8')
        load_row.addWidget(self.lbl_data, stretch=1)
        layout.addWidget(load_box)

        # --- Fit controls ----------------------------------------
        ctl_box = QGroupBox('Fit controls')
        ctl_row = QHBoxLayout(ctl_box)
        ctl_row.addWidget(QLabel('Max radial order:'))
        self.spin_order = QSpinBox()
        self.spin_order.setRange(2, 20)
        self.spin_order.setValue(8)
        self.spin_order.setToolTip(
            'Maximum polynomial order n (and m) -- fit builds the '
            'tensor product T_i(x) T_j(y) for 0 <= i, j <= n.')
        ctl_row.addWidget(self.spin_order)

        self.chk_normalize = QCheckBox('Normalize aperture')
        self.chk_normalize.setChecked(True)
        self.chk_normalize.setToolTip(
            'Rescale (x, y) pixel coordinates to [-1, 1].  '
            'Chebyshev polynomials are only defined there.')
        ctl_row.addWidget(self.chk_normalize)

        self.btn_mask = QPushButton('Load mask...')
        self.btn_mask.setToolTip(
            'Optional binary mask (.npy / .csv) -- non-zero pixels '
            'are kept, zero pixels are excluded from the fit.')
        self.btn_mask.clicked.connect(self._on_load_mask)
        ctl_row.addWidget(self.btn_mask)
        self.lbl_mask = QLabel('(no mask)')
        self.lbl_mask.setStyleSheet('color:#7a94b8')
        ctl_row.addWidget(self.lbl_mask, stretch=1)
        layout.addWidget(ctl_box)

        # --- Action row ------------------------------------------
        act_row = QHBoxLayout()
        self.btn_fit = QPushButton('Fit')
        self.btn_fit.setObjectName('run_button')
        self.btn_fit.setToolTip(
            'Solve the least-squares Chebyshev fit on the loaded '
            'data with the current settings.')
        self.btn_fit.clicked.connect(self._on_fit)
        act_row.addWidget(self.btn_fit)

        self.btn_export = QPushButton('Export coefficients...')
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip(
            'Write the fitted coefficient table to .json or .csv.')
        self.btn_export.clicked.connect(self._on_export)
        act_row.addWidget(self.btn_export)

        self.btn_apply = QPushButton('Apply to prescription')
        self.btn_apply.setEnabled(False)
        if self.sm is None:
            self.btn_apply.setToolTip(
                'No model attached.  Open the dock from the main '
                'window to enable this action.')
        else:
            self.btn_apply.setToolTip(
                'Insert a new chebyshev-freeform surface in the '
                'current prescription using these coefficients.  '
                'Per surface_sag_chebyshev the surface is stored as '
                'freeform_type="chebyshev" with cheb_coeffs={(i,j):c}.\n'
                'If the model rejects the surface, the coefficients '
                'are saved to a system-snapshot JSON instead.')
        self.btn_apply.clicked.connect(self._on_apply)
        act_row.addWidget(self.btn_apply)

        act_row.addStretch()
        layout.addLayout(act_row)

        # --- Residual summary ------------------------------------
        self.lbl_summary = QLabel('Residual: --')
        self.lbl_summary.setFont(QFont('Consolas', 10))
        self.lbl_summary.setStyleSheet('color:#bcd6ff')
        layout.addWidget(self.lbl_summary)

        # --- Plot canvas (raw / fit / residual) ------------------
        if HAS_MPL:
            self.fig = Figure(figsize=(7, 3), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.fig)
            layout.addWidget(self.canvas, stretch=1)
        else:
            self.canvas = None
            layout.addWidget(QLabel('(matplotlib unavailable)'))

        # --- Coefficient table -----------------------------------
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['n', 'm', 'c_nm'])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(180)
        layout.addWidget(self.table)

    # ------------------------------------------------------------
    # Load handlers.
    # ------------------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load measured height map', '',
            'Height map (*.npy *.csv *.h5 *.hdf5);;All files (*)')
        if not path:
            return
        try:
            self._z = _load_height_map(path)
        except Exception as exc:
            QMessageBox.warning(
                self, 'Load failed', f'{type(exc).__name__}: {exc}')
            return
        self._z_path = path
        self.lbl_data.setText(
            f'{os.path.basename(path)}  '
            f'shape {self._z.shape}')
        self._refresh_plot(raw_only=True)

    def _on_load_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load fit mask', '',
            'Mask (*.npy *.csv);;All files (*)')
        if not path:
            return
        try:
            arr = _load_height_map(path)
        except Exception as exc:
            QMessageBox.warning(
                self, 'Mask load failed',
                f'{type(exc).__name__}: {exc}')
            return
        self._mask = arr
        self._mask_path = path
        self.lbl_mask.setText(
            f'{os.path.basename(path)}  '
            f'kept {int((arr != 0).sum())} / {arr.size}')

    # ------------------------------------------------------------
    # Fit + display.
    # ------------------------------------------------------------

    def _on_fit(self) -> None:
        if self._z is None:
            QMessageBox.information(
                self, 'No data', 'Load a height map first.')
            return
        order = int(self.spin_order.value())
        try:
            coeffs, fit_map, residual, x, y = _fit_chebyshev_2d(
                self._z, order, mask=self._mask,
                normalize_aperture=self.chk_normalize.isChecked())
        except Exception as exc:
            QMessageBox.warning(
                self, 'Fit failed',
                f'{type(exc).__name__}: {exc}')
            return
        self._coeffs = coeffs
        self._fit_map = fit_map
        self._residual = residual
        self._x_axis = x
        self._y_axis = y
        # Residual stats (mask NaN out for PV/RMS).
        valid = residual[np.isfinite(residual)]
        if valid.size:
            rms = float(np.sqrt(np.mean(valid ** 2)))
            pv = float(np.max(valid) - np.min(valid))
        else:
            rms = float('nan')
            pv = float('nan')
        self.lbl_summary.setText(
            f'Residual:  RMS = {rms:.4g}    PV = {pv:.4g}    '
            f'(order {order}, {coeffs.size} coefficients)')
        self._refresh_table()
        self._refresh_plot()
        self.btn_export.setEnabled(True)
        self.btn_apply.setEnabled(self.sm is not None)

    def _refresh_table(self) -> None:
        self.table.setRowCount(0)
        if self._coeffs is None:
            return
        n_max, m_max = self._coeffs.shape
        # List in (n + m) ascending then (n, m) lex order so dominant
        # low-order terms appear first.
        rows = sorted(
            ((i, j, float(self._coeffs[i, j]))
             for i in range(n_max) for j in range(m_max)),
            key=lambda r: (r[0] + r[1], r[0], r[1]))
        self.table.setRowCount(len(rows))
        for k, (i, j, c) in enumerate(rows):
            self.table.setItem(k, 0, QTableWidgetItem(str(i)))
            self.table.setItem(k, 1, QTableWidgetItem(str(j)))
            self.table.setItem(k, 2, QTableWidgetItem(f'{c:.6e}'))

    def _refresh_plot(self, raw_only: bool = False) -> None:
        if not HAS_MPL or self.canvas is None:
            return
        self.fig.clear()
        if raw_only:
            ax = self.fig.add_subplot(1, 1, 1)
            if self._z is not None:
                im = ax.imshow(self._z, origin='lower', cmap='viridis')
                self.fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title('Raw data')
        else:
            axes = [self.fig.add_subplot(1, 3, k) for k in (1, 2, 3)]
            titles = ['Raw data', 'Fit', 'Residual']
            arrays = [self._z, self._fit_map, self._residual]
            for ax, title, arr in zip(axes, titles, arrays):
                if arr is None:
                    continue
                im = ax.imshow(arr, origin='lower', cmap='viridis')
                self.fig.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(title)
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # Export + apply.
    # ------------------------------------------------------------

    def _on_export(self) -> None:
        if self._coeffs is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Chebyshev coefficients', '',
            'JSON (*.json);;CSV (*.csv)')
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.csv':
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['n', 'm', 'c_nm'])
                    n_max, m_max = self._coeffs.shape
                    for i in range(n_max):
                        for j in range(m_max):
                            writer.writerow(
                                [i, j, float(self._coeffs[i, j])])
            else:
                payload = {
                    'source': self._z_path,
                    'mask': self._mask_path,
                    'max_order': int(self.spin_order.value()),
                    'normalize_aperture':
                        bool(self.chk_normalize.isChecked()),
                    'cheb_coeffs': {
                        f'{i},{j}': float(self._coeffs[i, j])
                        for i in range(self._coeffs.shape[0])
                        for j in range(self._coeffs.shape[1])
                    },
                }
                with open(path, 'w') as f:
                    json.dump(payload, f, indent=2)
        except Exception as exc:
            QMessageBox.warning(
                self, 'Export failed',
                f'{type(exc).__name__}: {exc}')
            return
        QMessageBox.information(
            self, 'Exported',
            f'Wrote {self._coeffs.size} coefficients to\n{path}')

    def _on_apply(self) -> None:
        if self._coeffs is None:
            return
        # Build the cheb_coeffs dict that surface_sag_chebyshev expects.
        cheb_dict = {
            (int(i), int(j)): float(self._coeffs[i, j])
            for i in range(self._coeffs.shape[0])
            for j in range(self._coeffs.shape[1])
        }
        surface_dict = {
            'freeform_type': 'chebyshev',
            'cheb_coeffs': cheb_dict,
            'norm_x': 1.0,
            'norm_y': 1.0,
            'radius': float('inf'),
            'conic': 0.0,
            'source': self._z_path,
        }
        # Try the model.  If it doesn't accept Chebyshev surfaces in
        # the prescription, fall back to a JSON snapshot file.
        applied = False
        if self.sm is not None:
            for attr in ('add_chebyshev_surface',
                         'add_freeform_surface'):
                method = getattr(self.sm, attr, None)
                if callable(method):
                    try:
                        method(surface_dict)
                        applied = True
                        break
                    except Exception as exc:
                        QMessageBox.warning(
                            self, 'Apply failed',
                            f'{type(exc).__name__}: {exc}')
                        return
        if applied:
            QMessageBox.information(
                self, 'Applied',
                'Chebyshev surface added to the current prescription.')
            return
        # Fallback: stash as a JSON snapshot so the user can wire it
        # in manually via the surface editor.
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Chebyshev surface snapshot', '',
            'JSON (*.json)')
        if not path:
            return
        try:
            json_safe = dict(surface_dict)
            json_safe['cheb_coeffs'] = {
                f'{i},{j}': v
                for (i, j), v in cheb_dict.items()
            }
            json_safe['radius'] = None  # JSON has no inf literal
            with open(path, 'w') as f:
                json.dump(json_safe, f, indent=2)
        except Exception as exc:
            QMessageBox.warning(
                self, 'Snapshot failed',
                f'{type(exc).__name__}: {exc}')
            return
        QMessageBox.information(
            self, 'Saved snapshot',
            'Model did not accept a Chebyshev surface directly.\n'
            f'Surface snapshot written to\n{path}\n\n'
            'To use it: open the snapshot dock or load the file '
            'into a custom surface entry with freeform_type='
            '"chebyshev".')
