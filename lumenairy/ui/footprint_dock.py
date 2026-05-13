"""
Footprint dock — ray-bundle outline drawn on every surface.

For each surface in the system, plots the (x, y) coordinates of the
ALIVE rays at that surface, with the surface clear-aperture circle
drawn as a reference.  Standard tool for verifying surface diameters,
stop placement, and vignetting at every interface (not just the
image plane).

Author: Andrew Traverso
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QComboBox,
)

import numpy as np

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from .model import SystemModel


class FootprintDock(QWidget):
    """Per-surface ray-footprint plots (one subplot per surface)."""

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
        self.spin_rings.setToolTip(
            'Number of rays-per-aperture ring; 6 is the standard '
            'spot-diagram convention.')
        self.spin_rings.valueChanged.connect(self._replot)
        toolbar.addWidget(self.spin_rings)

        toolbar.addWidget(QLabel('Per ring:'))
        self.spin_per_ring = QSpinBox()
        self.spin_per_ring.setRange(6, 36)
        self.spin_per_ring.setValue(12)
        self.spin_per_ring.setToolTip(
            'Rays per ring; 12 gives a clean azimuthal sampling.')
        self.spin_per_ring.valueChanged.connect(self._replot)
        toolbar.addWidget(self.spin_per_ring)

        toolbar.addWidget(QLabel('Field:'))
        self.combo_field = QComboBox()
        self.combo_field.addItems(['On-axis', 'All defined fields'])
        self.combo_field.currentIndexChanged.connect(self._replot)
        toolbar.addWidget(self.combo_field)

        toolbar.addStretch()

        btn_refresh = QPushButton('Refresh')
        btn_refresh.clicked.connect(self._replot)
        toolbar.addWidget(btn_refresh)

        layout.addLayout(toolbar)

        # Matplotlib canvas
        self.fig = Figure(figsize=(10, 4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.mpl_toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.sm.system_changed.connect(self._replot)
        self._replot()

    def _replot(self):
        from ..analysis.field import footprint_per_surface
        self.fig.clear()

        # 3.7.7: world-frame surfaces so folded designs render with
        # correct per-surface ray positions.  4.4: the actual
        # ring-trace math lives in
        # ``lumenairy.analysis.field.footprint_per_surface`` -- this
        # dock just builds the world-frame surface list and renders
        # the returned per-surface dicts.
        surfaces = self.sm.build_run_trace_world_surfaces()
        if not surfaces:
            self._draw_message('No surfaces in current system.')
            return

        wv = self.sm.wavelength_m
        semi_ap = self.sm.epd_m / 2.0
        rings = self.spin_rings.value()
        per_ring = self.spin_per_ring.value()

        # Decide field set
        if self.combo_field.currentIndex() == 0:
            fields_deg = [0.0]
        else:
            fields_deg = list(self.sm.field_angles_deg) or [0.0]

        try:
            fp = footprint_per_surface(
                surfaces, wv,
                semi_aperture=semi_ap,
                fields_deg=tuple(fields_deg),
                num_rings=rings, rays_per_ring=per_ring,
            )
        except Exception as e:
            self._draw_message(f'footprint_per_surface failed: {e}')
            return

        # Lay out subplots: one per refracting/image surface.
        n_surf = len(fp)
        n_cols = min(4, n_surf)
        n_rows = (n_surf + n_cols - 1) // n_cols
        axes = self.fig.subplots(n_rows, n_cols, squeeze=False)

        colors = ['#5cb8ff', '#ff6b35', '#3ddc84', '#ffd166', '#ff5555']

        # Render the per-surface footprints returned by the analysis.
        for si, entry in enumerate(fp):
            ax = axes[si // n_cols][si % n_cols]
            for fi, field_entry in enumerate(entry['fields']):
                alive = field_entry['alive']
                if not alive.any():
                    continue
                color = colors[fi % len(colors)]
                fa_deg = field_entry['field_deg']
                ax.scatter(field_entry['x'][alive] * 1e3,
                           field_entry['y'][alive] * 1e3,
                           s=4, c=color, alpha=0.7,
                           label=f'{fa_deg:.1f}°' if si == 0 else None)

        # Annotate every subplot with the surface circle, axis labels,
        # face colors, and a title.
        for si in range(n_surf):
            ax = axes[si // n_cols][si % n_cols]
            ax.set_facecolor('#0a0c10')
            ax.tick_params(colors='#7a94b8', labelsize=8)
            for s in ax.spines.values():
                s.set_color('#2a3548')
            ax.grid(True, color='#1a2535', linewidth=0.5)
            ax.set_aspect('equal')

            # Surface clear aperture
            sd = fp[si]['semi_diameter']
            if np.isfinite(sd) and sd > 0:
                theta = np.linspace(0, 2 * np.pi, 128)
                ax.plot(sd * 1e3 * np.cos(theta), sd * 1e3 * np.sin(theta),
                        color='#cc6600', linewidth=1.0, linestyle='--',
                        label='SD' if si == 0 else None)

            label = fp[si]['label']
            if si == n_surf - 1 and n_surf > 1:
                label = 'Image' if 'Image' not in label else label
            ax.set_title(label, color='#dde8f8', fontsize=10,
                         fontfamily='monospace')
            ax.set_xlabel('x (mm)', color='#7a94b8', fontsize=8)
            ax.set_ylabel('y (mm)', color='#7a94b8', fontsize=8)

        # Hide unused subplot cells
        for si in range(n_surf, n_rows * n_cols):
            axes[si // n_cols][si % n_cols].set_visible(False)

        # One legend on the first subplot only
        if axes[0][0].get_legend_handles_labels()[0]:
            leg = axes[0][0].legend(
                loc='upper right', fontsize=7, ncol=1,
                facecolor='#12161e', edgecolor='#2a3548',
                labelcolor='#dde8f8')

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
