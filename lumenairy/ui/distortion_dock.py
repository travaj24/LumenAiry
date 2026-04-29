"""
Distortion dock — chief-ray distortion vs field angle.

Two plots side by side:

* **Distortion %** — for each field angle theta in [0, max], trace the
  chief ray, find its image-plane height ``h_chief``, and compare to
  the paraxial f * tan(theta) prediction.  Distortion is reported as
  ``100 * (h_chief - f tan(theta)) / (f tan(theta))``.  Pincushion =
  positive %, barrel = negative %.

* **Distortion grid** — a square reference grid (paraxial chief-ray
  positions) overlaid on the actual chief-ray positions.  Standard
  visualization of how a real lens warps imagery vs the f-tan-theta
  prediction.

Both plots use the system's defined field angles for the axis range
when present, falling back to a 0..max(field_angles_deg) sweep
otherwise (or a default 5° sweep if no fields are defined).

Author: Andrew Traverso
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox,
)

import numpy as np

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from .model import SystemModel


class DistortionDock(QWidget):
    """Distortion vs field angle + distortion-grid plot."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Toolbar
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel('Max field (°):'))
        self.spin_max_deg = QDoubleSpinBox()
        self.spin_max_deg.setRange(0.1, 90.0)
        self.spin_max_deg.setDecimals(2)
        self.spin_max_deg.setValue(5.0)
        self.spin_max_deg.setSuffix(' °')
        self.spin_max_deg.valueChanged.connect(self._replot)
        toolbar.addWidget(self.spin_max_deg)

        toolbar.addWidget(QLabel('Sweep points:'))
        self.spin_points = QSpinBox()
        self.spin_points.setRange(7, 51)
        self.spin_points.setValue(21)
        self.spin_points.valueChanged.connect(self._replot)
        toolbar.addWidget(self.spin_points)

        toolbar.addStretch()
        btn_refresh = QPushButton('Refresh')
        btn_refresh.clicked.connect(self._replot)
        toolbar.addWidget(btn_refresh)

        layout.addLayout(toolbar)

        # Matplotlib canvas
        self.fig = Figure(figsize=(11, 5), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.mpl_toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # Status line beneath the plot for max-distortion summary.
        self.lbl_status = QLabel('')
        self.lbl_status.setStyleSheet(
            'color: #7a94b8; font-family: monospace; padding: 4px;')
        layout.addWidget(self.lbl_status)

        self.sm.system_changed.connect(self._replot)
        self._replot()

    def _replot(self):
        from ..raytrace import (
            surfaces_from_prescription, find_paraxial_focus, trace,
            _make_bundle, system_abcd, Surface)
        self.fig.clear()

        surfaces = self.sm.build_trace_surfaces()
        if not surfaces:
            self._draw_message('No surfaces in current system.')
            return

        wv = self.sm.wavelength_m

        # Need EFL + BFL for the f*tan(theta) reference and to put
        # a flat image plane at the paraxial focus.
        try:
            _, efl, bfl, _ = system_abcd(surfaces, wv)
        except Exception as e:
            self._draw_message(f'Cannot compute EFL: {e}')
            return
        if not (np.isfinite(efl) and efl > 0
                and np.isfinite(bfl) and bfl > 0):
            self._draw_message('Need a converging system with finite EFL/BFL.')
            return

        # Build the surfaces-to-image-plane stack
        surfs_img = surfaces[:]
        surfs_img[-1].thickness = bfl
        surfs_img.append(Surface(
            radius=np.inf, semi_diameter=np.inf,
            glass_before=surfs_img[-1].glass_after,
            glass_after=surfs_img[-1].glass_after))

        # Sweep field angles: 0 → max
        max_deg = float(self.spin_max_deg.value())
        n_pts = int(self.spin_points.value())
        thetas_deg = np.linspace(0, max_deg, n_pts)
        thetas_rad = np.radians(thetas_deg)

        h_chief = []
        for theta in thetas_rad:
            # Chief ray: launched from the entrance-pupil center (x=y=0 for
            # an on-axis stop) at angle theta in the y-direction.
            try:
                rays = _make_bundle(
                    x=np.array([0.0]), y=np.array([0.0]),
                    L=np.array([0.0]), M=np.array([np.sin(theta)]),
                    wavelength=wv)
                result = trace(rays, surfs_img, wv)
                if result.image_rays.alive[0]:
                    h_chief.append(float(result.image_rays.y[0]))
                else:
                    h_chief.append(np.nan)
            except Exception:
                h_chief.append(np.nan)

        h_chief = np.array(h_chief)
        h_paraxial = efl * np.tan(thetas_rad)

        # Distortion %
        # Avoid divide-by-zero at theta=0 by taking the limit
        with np.errstate(divide='ignore', invalid='ignore'):
            distortion_pct = np.where(
                np.abs(h_paraxial) > 1e-15,
                100.0 * (h_chief - h_paraxial) / h_paraxial,
                0.0)

        # ---- Plot 1: Distortion % vs field angle
        ax1 = self.fig.add_subplot(121)
        ax1.set_facecolor('#0a0c10')
        ax1.tick_params(colors='#7a94b8', labelsize=9)
        for s in ax1.spines.values():
            s.set_color('#2a3548')
        ax1.grid(True, color='#1a2535', linewidth=0.5)
        ax1.axhline(0, color='#2a3548', linewidth=1)

        valid = ~np.isnan(distortion_pct)
        if valid.any():
            ax1.plot(thetas_deg[valid], distortion_pct[valid],
                     color='#5cb8ff', linewidth=1.5, marker='o',
                     markersize=3)
        ax1.set_xlabel('Field angle (deg)', color='#dde8f8', fontsize=10,
                       fontfamily='monospace')
        ax1.set_ylabel('Distortion (%)', color='#dde8f8', fontsize=10,
                       fontfamily='monospace')
        ax1.set_title('Distortion vs Field', color='#5cb8ff',
                      fontsize=11, fontfamily='monospace')
        # Mark defined field points
        for fa in (self.sm.field_angles_deg or []):
            if 0 <= fa <= max_deg:
                ax1.axvline(fa, color='#ff6b35', linewidth=0.8,
                            linestyle='--', alpha=0.5)

        # ---- Plot 2: Distortion grid
        # Build a square paraxial reference grid in field-angle space,
        # then trace the chief ray for each cell to get the actual
        # image-plane position.  Show paraxial as red, actual as blue.
        ax2 = self.fig.add_subplot(122)
        ax2.set_facecolor('#0a0c10')
        ax2.tick_params(colors='#7a94b8', labelsize=9)
        for s in ax2.spines.values():
            s.set_color('#2a3548')
        ax2.set_aspect('equal')
        ax2.set_xlabel('x (mm)', color='#dde8f8', fontsize=10,
                       fontfamily='monospace')
        ax2.set_ylabel('y (mm)', color='#dde8f8', fontsize=10,
                       fontfamily='monospace')
        ax2.set_title('Distortion grid (red=paraxial, blue=actual)',
                      color='#5cb8ff', fontsize=11,
                      fontfamily='monospace')

        n_grid = 7
        ths = np.linspace(-max_deg, max_deg, n_grid)
        ths_rad = np.radians(ths)
        actual_x = np.full((n_grid, n_grid), np.nan)
        actual_y = np.full((n_grid, n_grid), np.nan)
        for ix, tx in enumerate(ths_rad):
            for iy, ty in enumerate(ths_rad):
                try:
                    rays = _make_bundle(
                        x=np.array([0.0]), y=np.array([0.0]),
                        L=np.array([np.sin(tx)]),
                        M=np.array([np.sin(ty)]),
                        wavelength=wv)
                    r = trace(rays, surfs_img, wv)
                    if r.image_rays.alive[0]:
                        actual_x[iy, ix] = r.image_rays.x[0]
                        actual_y[iy, ix] = r.image_rays.y[0]
                except Exception:
                    pass

        # Paraxial reference grid
        para_x = (efl * np.tan(ths_rad))[None, :].repeat(n_grid, axis=0)
        para_y = (efl * np.tan(ths_rad))[:, None].repeat(n_grid, axis=1)

        # Draw paraxial
        for i in range(n_grid):
            ax2.plot(para_x[i, :] * 1e3, para_y[i, :] * 1e3,
                     color='#aa3333', linewidth=0.7, alpha=0.6)
            ax2.plot(para_x[:, i] * 1e3, para_y[:, i] * 1e3,
                     color='#aa3333', linewidth=0.7, alpha=0.6)

        # Draw actual
        for i in range(n_grid):
            ax2.plot(actual_x[i, :] * 1e3, actual_y[i, :] * 1e3,
                     color='#5cb8ff', linewidth=1.0)
            ax2.plot(actual_x[:, i] * 1e3, actual_y[:, i] * 1e3,
                     color='#5cb8ff', linewidth=1.0)

        # Status: max distortion
        if valid.any():
            max_d = float(np.nanmax(np.abs(distortion_pct)))
            tag = ('Pincushion' if np.nanmax(distortion_pct) > 0
                   else 'Barrel')
            self.lbl_status.setText(
                f'EFL = {efl*1e3:.3f} mm | '
                f'Max distortion = {max_d:.4f} % ({tag}) | '
                f'BFL = {bfl*1e3:.3f} mm')
        else:
            self.lbl_status.setText('No valid trace at the requested fields.')

        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_message(self, text):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5, text, ha='center', va='center',
                color='#7a94b8', transform=ax.transAxes,
                fontfamily='monospace')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#2a3548')
        self.canvas.draw()
        self.lbl_status.setText('')
