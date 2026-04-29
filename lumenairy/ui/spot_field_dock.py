"""
Spot-vs-field dock — array of spot diagrams at the configured field
angles, all on a shared scale so cross-field comparison is direct.

The existing :mod:`analysis.SpotDiagramWidget` shows ONE spot at the
current trace (typically on-axis).  This dock complements it by
plotting every defined ``model.field_angles_deg`` in a single figure
with matched axes, so the user can see at a glance how aberrations
grow off-axis.

Author: Andrew Traverso
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QCheckBox,
)

import numpy as np

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from .model import SystemModel


class SpotFieldDock(QWidget):
    """N×1 (or N×M) array of spot diagrams across configured fields."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('Rings:'))
        self.spin_rings = QSpinBox()
        self.spin_rings.setRange(2, 12)
        self.spin_rings.setValue(6)
        self.spin_rings.valueChanged.connect(self._replot)
        toolbar.addWidget(self.spin_rings)

        toolbar.addWidget(QLabel('Per ring:'))
        self.spin_per_ring = QSpinBox()
        self.spin_per_ring.setRange(6, 36)
        self.spin_per_ring.setValue(24)
        self.spin_per_ring.valueChanged.connect(self._replot)
        toolbar.addWidget(self.spin_per_ring)

        self.chk_airy = QCheckBox('Airy disc')
        self.chk_airy.setChecked(True)
        self.chk_airy.setToolTip(
            'Overlay the diffraction-limited Airy disc (1.22*lam*f/D) '
            'on each subplot for diffraction-vs-aberration context.')
        self.chk_airy.toggled.connect(self._replot)
        toolbar.addWidget(self.chk_airy)

        self.chk_share = QCheckBox('Shared scale')
        self.chk_share.setChecked(True)
        self.chk_share.setToolTip(
            'When ON, all subplots share the same axis range so you can '
            'see the relative aberration growth across fields.  When OFF, '
            'each subplot autoscales independently.')
        self.chk_share.toggled.connect(self._replot)
        toolbar.addWidget(self.chk_share)

        toolbar.addStretch()
        btn_refresh = QPushButton('Refresh')
        btn_refresh.clicked.connect(self._replot)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        # Matplotlib canvas
        self.fig = Figure(figsize=(11, 4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.mpl_toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.lbl_status = QLabel('')
        self.lbl_status.setStyleSheet(
            'color: #7a94b8; font-family: monospace; padding: 4px;')
        layout.addWidget(self.lbl_status)

        self.sm.system_changed.connect(self._replot)
        self._replot()

    def _replot(self):
        from ..raytrace import (
            find_paraxial_focus, trace, make_rings, spot_rms,
            system_abcd, Surface)
        self.fig.clear()

        surfaces = self.sm.build_trace_surfaces()
        if not surfaces:
            self._draw_message('No surfaces.')
            return

        wv = self.sm.wavelength_m
        semi_ap = self.sm.epd_m / 2.0
        rings = self.spin_rings.value()
        per_ring = self.spin_per_ring.value()
        fields_deg = list(self.sm.field_angles_deg) or [0.0]

        # Add a flat image plane at paraxial focus
        try:
            _, efl, bfl, _ = system_abcd(surfaces, wv)
            if not (np.isfinite(bfl) and bfl > 0):
                bfl = find_paraxial_focus(surfaces, wv)
        except Exception:
            efl = bfl = np.nan
        if not (np.isfinite(bfl) and bfl > 0):
            self._draw_message('Cannot compute BFL.')
            return

        surfs_img = surfaces[:]
        surfs_img[-1].thickness = bfl
        surfs_img.append(Surface(
            radius=np.inf, semi_diameter=np.inf,
            glass_before=surfs_img[-1].glass_after,
            glass_after=surfs_img[-1].glass_after))

        # Trace each field, collect spots
        results = []
        for fa_deg in fields_deg:
            fa_rad = np.radians(fa_deg)
            try:
                rays = make_rings(semi_ap, rings, per_ring, fa_rad, wv)
                r = trace(rays, surfs_img, wv)
            except Exception:
                r = None
            results.append((fa_deg, r))

        # Determine shared scale if requested
        shared_extent = 0.0
        if self.chk_share.isChecked():
            for _, r in results:
                if r is None:
                    continue
                ir = r.image_rays
                alive = ir.alive
                if not alive.any():
                    continue
                cx = float(np.mean(ir.x[alive]))
                cy = float(np.mean(ir.y[alive]))
                e = max(float(np.max(np.abs(ir.x[alive] - cx))),
                        float(np.max(np.abs(ir.y[alive] - cy))))
                shared_extent = max(shared_extent, e)
            shared_extent *= 1.2  # 20% margin

        # Compute Airy radius (constant across fields)
        airy_um = 0.0
        if self.chk_airy.isChecked() and np.isfinite(efl) and efl > 0 \
                and semi_ap > 0:
            airy_um = 1.22 * wv * efl / (2 * semi_ap) * 1e6

        n = len(results)
        n_cols = min(n, 5)
        n_rows = (n + n_cols - 1) // n_cols
        axes = self.fig.subplots(n_rows, n_cols, squeeze=False)

        rms_lines = []
        for i, (fa_deg, r) in enumerate(results):
            ax = axes[i // n_cols][i % n_cols]
            ax.set_facecolor('#0a0c10')
            ax.tick_params(colors='#7a94b8', labelsize=8)
            for s in ax.spines.values():
                s.set_color('#2a3548')
            ax.set_aspect('equal')
            ax.grid(True, color='#1a2535', linewidth=0.5)

            if r is None:
                ax.text(0.5, 0.5, 'trace failed', ha='center', va='center',
                        color='#ff5555', transform=ax.transAxes,
                        fontfamily='monospace')
                ax.set_title(f'{fa_deg:.2f}°', color='#dde8f8',
                             fontsize=10, fontfamily='monospace')
                continue

            ir = r.image_rays
            alive = ir.alive
            if not alive.any():
                ax.text(0.5, 0.5, 'all rays vignetted', ha='center',
                        va='center', color='#ff5555',
                        transform=ax.transAxes, fontfamily='monospace')
                ax.set_title(f'{fa_deg:.2f}°', color='#dde8f8',
                             fontsize=10, fontfamily='monospace')
                continue

            cx = float(np.mean(ir.x[alive]))
            cy = float(np.mean(ir.y[alive]))
            xs = (ir.x[alive] - cx) * 1e6
            ys = (ir.y[alive] - cy) * 1e6
            ax.scatter(xs, ys, s=6, c='#5cb8ff', alpha=0.7)

            try:
                rms_m, _ = spot_rms(r)
                rms_um = rms_m * 1e6
                rms_lines.append((fa_deg, rms_um))
            except Exception:
                rms_um = 0.0

            if airy_um > 0:
                theta = np.linspace(0, 2 * np.pi, 80)
                ax.plot(airy_um * np.cos(theta), airy_um * np.sin(theta),
                        color='#ffd166', linewidth=1.0, alpha=0.6)

            if shared_extent > 0:
                lim_um = shared_extent * 1e6
                ax.set_xlim(-lim_um, lim_um)
                ax.set_ylim(-lim_um, lim_um)

            ax.set_title(f'{fa_deg:.2f}°  (RMS={rms_um:.2f} µm)',
                         color='#dde8f8', fontsize=10,
                         fontfamily='monospace')
            ax.set_xlabel('x (µm)', color='#7a94b8', fontsize=8)
            ax.set_ylabel('y (µm)', color='#7a94b8', fontsize=8)

        # Hide unused
        for j in range(n, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].set_visible(False)

        # Status
        if rms_lines:
            tag = ', '.join(f'{fa:.1f}°={r:.2f}µm' for fa, r in rms_lines)
            airy_tag = (f'  Airy={airy_um:.2f}µm' if airy_um > 0 else '')
            self.lbl_status.setText(f'RMS by field: {tag}{airy_tag}')
        else:
            self.lbl_status.setText('No valid spots produced.')

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
