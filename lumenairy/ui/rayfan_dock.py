"""
Ray fan and OPD dock — matplotlib-based transverse aberration plots.

Author: Andrew Traverso
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from PySide6.QtCore import Qt

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel
from ..raytrace import (
    ray_fan_data, opd_fan_data, make_fan, trace, spot_rms,
    Surface, find_paraxial_focus,
)


class RayFanDock(QWidget):
    """Transverse ray aberration and OPD fan plots."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Toolbar
        toolbar = QHBoxLayout()
        self.combo_plot = QComboBox()
        self.combo_plot.addItems(['Ray Fan (EY/EX)', 'OPD Fan', 'Field Curvature'])
        self.combo_plot.setToolTip(
            'Which plot to show.\n'
            'Ray-fan and OPD-fan update live on every system change.\n'
            'Field Curvature does a 21-field sweep (slow), so it only '
            'runs on demand \u2013 click "Compute Field Curvature".')
        self.combo_plot.currentIndexChanged.connect(self._on_plot_type_changed)
        toolbar.addWidget(QLabel('Plot:'))
        toolbar.addWidget(self.combo_plot)
        toolbar.addStretch()
        self.btn_field_curve = QPushButton('Compute Field Curvature')
        self.btn_field_curve.setToolTip(
            'Run the 21-point field sweep.  Only needed when the '
            'Field Curvature plot is selected; otherwise skipped to '
            'keep editing responsive.')
        self.btn_field_curve.setVisible(False)
        self.btn_field_curve.clicked.connect(self._replot)
        toolbar.addWidget(self.btn_field_curve)
        btn_refresh = QPushButton('Refresh')
        btn_refresh.setToolTip('Redraw the current plot.')
        btn_refresh.clicked.connect(self._replot)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        # Matplotlib canvas with the standard navigation toolbar so
        # every plot in this dock gets pan / zoom / save-PNG for free.
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.mpl_toolbar)
        layout.addWidget(self.canvas, stretch=1)

        self.sm.system_changed.connect(self._on_system_changed)
        self._replot()

    def _on_plot_type_changed(self):
        is_fc = self.combo_plot.currentIndex() == 2
        self.btn_field_curve.setVisible(is_fc)
        if not is_fc:
            self._replot()
        else:
            # Clear the canvas and wait for the explicit compute.
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor('#0a0c10')
            ax.text(0.5, 0.5,
                    'Click "Compute Field Curvature" to run the 21-field sweep.',
                    ha='center', va='center', color='#7a94b8',
                    transform=ax.transAxes, fontfamily='monospace')
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw()

    def _on_system_changed(self):
        # Only auto-refresh the fast plots; field curvature waits for
        # an explicit click.
        if self.combo_plot.currentIndex() != 2:
            self._replot()

    def _replot(self):
        self.fig.clear()
        plot_type = self.combo_plot.currentIndex()

        surfaces = self.sm.build_trace_surfaces()
        if not surfaces:
            self.canvas.draw()
            return

        wv = self.sm.wavelength_m
        semi_ap = self.sm.epd_m / 2.0
        fields = self.sm.field_angles_deg

        # Style
        style = {'color': '#dde8f8', 'fontsize': 10, 'fontfamily': 'monospace'}
        grid_kw = {'color': '#1a2535', 'linewidth': 0.5}

        if plot_type == 0:
            # Ray fan: tangential + sagittal
            ax_t = self.fig.add_subplot(121)
            ax_s = self.fig.add_subplot(122)
            for ax in (ax_t, ax_s):
                ax.set_facecolor('#0a0c10')
                ax.tick_params(colors='#7a94b8', labelsize=9)
                ax.spines[:].set_color('#2a3548')
                ax.grid(True, **grid_kw)
                ax.axhline(0, color='#2a3548', linewidth=1)

            colors = ['#5cb8ff', '#ff6b35', '#3ddc84', '#ffd166', '#ff5555']
            for i, fa_deg in enumerate(fields):
                fa_rad = np.radians(fa_deg)
                try:
                    py, ey, px, ex = ray_fan_data(surfaces, wv, semi_ap, fa_rad, 101)
                    c = colors[i % len(colors)]
                    ax_t.plot(py, ey * 1e6, color=c, linewidth=1.2,
                              label=f'{fa_deg:.1f}°')
                    ax_s.plot(px, ex * 1e6, color=c, linewidth=1.2,
                              label=f'{fa_deg:.1f}°')
                except Exception:
                    pass

            ax_t.set_xlabel('PY', **style)
            ax_t.set_ylabel('EY (µm)', **style)
            ax_t.set_title('Tangential', color='#5cb8ff', fontsize=11, fontfamily='monospace')
            ax_t.legend(fontsize=8, facecolor='#12161e', edgecolor='#2a3548',
                        labelcolor='#dde8f8')

            ax_s.set_xlabel('PX', **style)
            ax_s.set_ylabel('EX (µm)', **style)
            ax_s.set_title('Sagittal', color='#5cb8ff', fontsize=11, fontfamily='monospace')
            ax_s.legend(fontsize=8, facecolor='#12161e', edgecolor='#2a3548',
                        labelcolor='#dde8f8')

        elif plot_type == 1:
            # OPD fan
            ax_t = self.fig.add_subplot(121)
            ax_s = self.fig.add_subplot(122)
            for ax in (ax_t, ax_s):
                ax.set_facecolor('#0a0c10')
                ax.tick_params(colors='#7a94b8', labelsize=9)
                ax.spines[:].set_color('#2a3548')
                ax.grid(True, **grid_kw)
                ax.axhline(0, color='#2a3548', linewidth=1)

            colors = ['#5cb8ff', '#ff6b35', '#3ddc84', '#ffd166']
            for i, fa_deg in enumerate(fields):
                fa_rad = np.radians(fa_deg)
                try:
                    py, opd_y, px, opd_x = opd_fan_data(surfaces, wv, semi_ap, fa_rad, 101)
                    c = colors[i % len(colors)]
                    ax_t.plot(py, opd_y, color=c, linewidth=1.2, label=f'{fa_deg:.1f}°')
                    ax_s.plot(px, opd_x, color=c, linewidth=1.2, label=f'{fa_deg:.1f}°')
                except Exception:
                    pass

            ax_t.set_xlabel('PY', **style)
            ax_t.set_ylabel('OPD (waves)', **style)
            ax_t.set_title('Tangential OPD', color='#5cb8ff', fontsize=11, fontfamily='monospace')
            ax_t.legend(fontsize=8, facecolor='#12161e', edgecolor='#2a3548',
                        labelcolor='#dde8f8')
            ax_s.set_xlabel('PX', **style)
            ax_s.set_ylabel('OPD (waves)', **style)
            ax_s.set_title('Sagittal OPD', color='#5cb8ff', fontsize=11, fontfamily='monospace')

        elif plot_type == 2:
            # Field curvature: RMS spot vs field angle
            ax = self.fig.add_subplot(111)
            ax.set_facecolor('#0a0c10')
            ax.tick_params(colors='#7a94b8', labelsize=9)
            ax.spines[:].set_color('#2a3548')
            ax.grid(True, **grid_kw)

            fa_sweep = np.linspace(0, max(max(fields), 5.0), 21)
            rms_values = []

            for fa_deg in fa_sweep:
                fa_rad = np.radians(fa_deg)
                try:
                    from ..raytrace import make_rings
                    rays = make_rings(semi_ap, 6, 24, fa_rad, wv)
                    surfs = [Surface(
                        radius=s.radius, conic=s.conic,
                        aspheric_coeffs=s.aspheric_coeffs,
                        semi_diameter=s.semi_diameter,
                        glass_before=s.glass_before, glass_after=s.glass_after,
                        is_mirror=s.is_mirror, thickness=s.thickness,
                    ) for s in surfaces]
                    try:
                        bfl = find_paraxial_focus(surfs, wv)
                        if np.isfinite(bfl) and bfl > 0:
                            surfs[-1].thickness = bfl
                            surfs.append(Surface(radius=np.inf, semi_diameter=np.inf,
                                glass_before=surfs[-1].glass_after,
                                glass_after=surfs[-1].glass_after))
                    except Exception:
                        pass
                    result = trace(rays, surfs, wv)
                    rms, _ = spot_rms(result)
                    rms_values.append(rms * 1e6)
                except Exception:
                    rms_values.append(np.nan)

            ax.plot(fa_sweep, rms_values, color='#5cb8ff', linewidth=1.5, marker='o', markersize=3)
            # Mark the defined field points
            for fa_deg in fields:
                ax.axvline(fa_deg, color='#ff6b35', linewidth=0.8, linestyle='--', alpha=0.6)

            ax.set_xlabel('Field angle (deg)', **style)
            ax.set_ylabel('RMS spot (µm)', **style)
            ax.set_title('Field Curvature', color='#5cb8ff', fontsize=11, fontfamily='monospace')

        self.fig.tight_layout()
        self.canvas.draw()
