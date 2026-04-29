"""
Layout3DView — interactive 3-D optical system visualization.

Uses pyvistaqt.QtInteractor to embed a fully interactive VTK widget
with orbit, zoom, and pan.  Falls back to off-screen rendering if
pyvistaqt is unavailable.

Author: Andrew Traverso
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt

import numpy as np

from .model import SystemModel
from ..lenses import surface_sag_biconic

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    from pyvistaqt import QtInteractor
    PYVISTAQT_AVAILABLE = True
except ImportError:
    PYVISTAQT_AVAILABLE = False


class Layout3DView(QWidget):
    """Interactive 3-D system layout with orbit/zoom/pan controls."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._plotter = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Toolbar
        toolbar = QHBoxLayout()
        btn_refresh = QPushButton('Refresh')
        btn_refresh.clicked.connect(self.rebuild)
        toolbar.addWidget(btn_refresh)

        btn_reset_cam = QPushButton('Reset View')
        btn_reset_cam.clicked.connect(self._reset_camera)
        toolbar.addWidget(btn_reset_cam)

        # View axis buttons
        for label, view in [('Front', 'xy'), ('Side', 'xz'), ('Top', 'yz'),
                             ('Iso', 'iso')]:
            btn = QPushButton(label)
            btn.setFixedWidth(40)
            btn.setToolTip(f'Snap to {label.lower()} view')
            btn.clicked.connect(lambda checked, v=view: self._snap_to_view(v))
            toolbar.addWidget(btn)

        toolbar.addStretch()

        help_label = QLabel('Left-drag: rotate | Scroll: zoom | Middle-drag: pan')
        help_label.setStyleSheet("color: #7a94b8; font-size: 10px;")
        toolbar.addWidget(help_label)

        layout.addLayout(toolbar)

        # 3D widget
        if PYVISTA_AVAILABLE and PYVISTAQT_AVAILABLE:
            self._plotter = QtInteractor(self)
            self._plotter.set_background('#050709')
            self._plotter.enable_anti_aliasing('ssaa')
            # Add orientation axes in the corner
            self._plotter.add_axes(
                line_width=2, color='#aabbcc',
                xlabel='X', ylabel='Y', zlabel='Z',
            )
            layout.addWidget(self._plotter.interactor, stretch=1)
        elif PYVISTA_AVAILABLE:
            # Fallback: static image label
            self._fallback_label = QLabel('pyvistaqt not available — static render only.\npip install pyvistaqt')
            self._fallback_label.setAlignment(Qt.AlignCenter)
            self._fallback_label.setStyleSheet(
                "QLabel { background: #050709; color: #a0b4d0; "
                "font-family: Consolas; font-size: 12px; }")
            self._fallback_label.setMinimumSize(300, 200)
            layout.addWidget(self._fallback_label, stretch=1)
        else:
            lbl = QLabel('PyVista not installed.\npip install pyvista pyvistaqt')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "QLabel { background: #050709; color: #a0b4d0; "
                "font-family: Consolas; font-size: 12px; }")
            layout.addWidget(lbl, stretch=1)

        self.sm.system_changed.connect(self.rebuild)
        self.sm.trace_ready.connect(lambda _: self._draw_rays())

    def rebuild(self):
        """Rebuild the entire 3D scene from the model."""
        if self._plotter is None:
            if PYVISTA_AVAILABLE and not PYVISTAQT_AVAILABLE:
                self._render_static()
            return

        self._plotter.clear()

        elements = self.sm.elements
        if len(elements) < 2:
            return

        z_positions = self.sm.element_z_positions_mm()

        # Optical axis
        z_min = z_positions[0] - 20
        z_max = z_positions[-1] + 20
        axis_line = pv.Line((0, 0, z_min), (0, 0, z_max))
        self._plotter.add_mesh(axis_line, color='#3a4a60', line_width=1,
                               name='axis')

        # Draw elements
        for ei, elem in enumerate(elements):
            if elem.elem_type in ('Source', 'Detector'):
                continue

            z = z_positions[ei]
            for si, srow in enumerate(elem.surfaces):
                internal_offset = sum(s.thickness for s in elem.surfaces[:si])
                z_s = z + internal_offset
                sd = srow.semi_diameter if np.isfinite(srow.semi_diameter) else self.sm.epd_mm / 2

                if srow.surf_type == 'Mirror' or elem.elem_type == 'Mirror':
                    self._draw_mirror_3d(z_s, sd, srow, ei * 100 + si)
                else:
                    self._draw_surface_3d(z_s, sd, srow, ei * 100 + si, elem.surfaces, z_positions)

        # Image plane
        z_ima = z_positions[-1]
        ima_disc = pv.Disc(center=(0, 0, z_ima), normal=(0, 0, 1),
                           inner=0, outer=self.sm.epd_mm / 2)
        self._plotter.add_mesh(ima_disc, color='#884444', opacity=0.3,
                               name='image_plane')

        # Draw rays if we have a trace result
        self._draw_rays()

        self._reset_camera()

    def _draw_surface_3d(self, z, sd, row, idx, all_surfaces, z_positions):
        """Draw a refractive surface as a curved disc.

        Sag is evaluated with the core ``surface_sag_biconic`` so the 3D
        view honours conic, polynomial-aspheric, and biconic-Y terms and
        cannot drift from the ray tracer's geometry.  Coordinates here
        are in mm (scene units); the core works in metres, so we scale
        in then back out.
        """
        R = row.radius
        n_theta = 48
        n_radial = 16
        theta = np.linspace(0, 2 * np.pi, n_theta)
        radii = np.linspace(0, sd, n_radial)

        # Vectorised sag eval: build (n_radial, n_theta) grids in metres,
        # call the core, then convert back to scene mm.
        T, Rg = np.meshgrid(theta, radii)        # shape (n_radial, n_theta)
        x_mm = Rg * np.cos(T)
        y_mm = Rg * np.sin(T)
        try:
            sag_m = surface_sag_biconic(
                x_mm.ravel() * 1e-3, y_mm.ravel() * 1e-3,
                R_x=R * 1e-3 if np.isfinite(R) else np.inf,
                R_y=(row.radius_y * 1e-3
                     if (getattr(row, 'radius_y', None) is not None
                         and np.isfinite(row.radius_y))
                     else None),
                conic_x=getattr(row, 'conic', 0.0) or 0.0,
                conic_y=getattr(row, 'conic_y', None),
                aspheric_coeffs=getattr(row, 'aspheric_coeffs', None),
                aspheric_coeffs_y=getattr(row, 'aspheric_coeffs_y', None),
            )
            sag_mm = sag_m.reshape(n_radial, n_theta) * 1e3
        except Exception:
            sag_mm = np.zeros((n_radial, n_theta))

        points = np.column_stack([
            x_mm.ravel(), y_mm.ravel(), (z + sag_mm).ravel(),
        ])

        # Build faces
        faces = []
        for ir in range(n_radial - 1):
            for it in range(n_theta - 1):
                p0 = ir * n_theta + it
                p1 = ir * n_theta + it + 1
                p2 = (ir + 1) * n_theta + it + 1
                p3 = (ir + 1) * n_theta + it
                faces.append([4, p0, p1, p2, p3])

        if faces:
            mesh = pv.PolyData(points, np.array(faces))
            color = '#5588cc' if row.glass else '#aaaaaa'
            self._plotter.add_mesh(mesh, color=color, opacity=0.35,
                                   name=f'surf_{idx}')

        # Glass volume between this surface and the next
        if row.glass and idx < len(all_surfaces) - 1:
            z_next = z + (row.thickness if np.isfinite(row.thickness) else 0)
            if abs(z_next - z) > 0.01:
                cyl = pv.Cylinder(center=(0, 0, (z + z_next) / 2),
                                  direction=(0, 0, 1),
                                  radius=sd, height=abs(z_next - z),
                                  resolution=48)
                self._plotter.add_mesh(cyl, color='#334466', opacity=0.08,
                                       name=f'glass_{idx}')

        # Edge ring for visibility
        ring_pts = np.column_stack([
            sd * np.cos(theta), sd * np.sin(theta),
            np.full(n_theta, z),
        ])
        ring = pv.lines_from_points(np.vstack([ring_pts, ring_pts[:1]]))
        self._plotter.add_mesh(ring, color=color if row.glass else '#888888',
                               line_width=1.5, name=f'ring_{idx}')

    def _draw_mirror_3d(self, z, sd, row, idx):
        """Draw a mirror as an opaque disc with edge ring."""
        disc = pv.Disc(center=(0, 0, z), normal=(0, 0, 1),
                       inner=0, outer=sd, r_res=1, c_res=48)
        self._plotter.add_mesh(disc, color='#7799cc', opacity=0.5,
                               name=f'mirror_{idx}')

    def _draw_rays(self):
        """Overlay traced rays on the 3D scene."""
        if self._plotter is None:
            return

        result = self.sm.trace_result
        if result is None:
            return

        # Remove old rays
        actors_to_remove = [name for name in self._plotter.actors
                           if name.startswith('ray_')]
        for name in actors_to_remove:
            self._plotter.remove_actor(name)

        z_positions = self._compute_z_positions()
        n_rays = result.input_rays.n_rays
        step = max(1, n_rays // 40)
        history = result.ray_history
        if self.sm.prefs.get('ray_use_wavelength', True):
            wv_color = self._wv_color(self.sm.wavelength_nm)
        else:
            wv_color = self.sm.prefs.get('ray_color', '#5cb8ff')

        for r in range(0, n_rays, step):
            if not result.input_rays.alive[r]:
                continue

            pts = []
            x_in = result.input_rays.x[r] * 1e3
            y_in = result.input_rays.y[r] * 1e3
            z0 = z_positions[1] if len(z_positions) > 1 else 0
            pts.append([x_in, y_in, z0])

            for si, rb in enumerate(history):
                if not rb.alive[r]:
                    break
                x = rb.x[r] * 1e3
                y = rb.y[r] * 1e3
                zi = si + 1
                z = z_positions[zi] if zi < len(z_positions) else pts[-1][2] + 20
                pts.append([x, y, z])

            if len(pts) >= 2:
                ray_line = pv.lines_from_points(np.array(pts))
                self._plotter.add_mesh(ray_line, color=wv_color,
                                       line_width=1.2, opacity=0.5,
                                       name=f'ray_{r}')

    def _compute_z_positions(self):
        """Cumulative z positions from element distances."""
        return self.sm.element_z_positions_mm()

    def _reset_camera(self):
        """Reset camera to a side view with slight elevation."""
        self._snap_to_view('iso')

    def _snap_to_view(self, view):
        """Snap camera to an axis-aligned or isometric view."""
        if self._plotter is None:
            return
        if view == 'xy':
            self._plotter.camera_position = 'xy'
        elif view == 'xz':
            self._plotter.camera_position = 'xz'
        elif view == 'yz':
            self._plotter.camera_position = 'yz'
        elif view == 'iso':
            self._plotter.camera_position = 'xz'
            self._plotter.camera.azimuth = 25
            self._plotter.camera.elevation = 15
        self._plotter.reset_camera()

    def _render_static(self):
        """Fallback: off-screen render to an image (no pyvistaqt)."""
        elements = self.sm.elements
        if len(elements) < 2:
            self._fallback_label.setText('No elements to render.')
            return

        try:
            pl = pv.Plotter(off_screen=True, window_size=[600, 400])
            pl.set_background('#050709')

            z_positions = self._compute_z_positions()

            # Simplified rendering for static mode
            for ei, elem in enumerate(elements):
                if elem.elem_type in ('Source', 'Detector'):
                    continue
                z = z_positions[ei]
                for srow in elem.surfaces:
                    sd = srow.semi_diameter if np.isfinite(srow.semi_diameter) else self.sm.epd_mm / 2
                    disc = pv.Disc(center=(0, 0, z), normal=(0, 0, 1),
                                   inner=sd * 0.05, outer=sd)
                    color = '#6688cc' if elem.elem_type == 'Mirror' else '#4488cc'
                    pl.add_mesh(disc, color=color, opacity=0.4)

            pl.camera_position = 'xz'
            pl.camera.azimuth = 25
            pl.camera.elevation = 15

            img = pl.screenshot(return_img=True)
            pl.close()

            from PySide6.QtGui import QImage, QPixmap
            img = np.ascontiguousarray(img)
            h, w, ch = img.shape
            qimg = QImage(img.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self._fallback_label.setPixmap(pixmap.scaled(
                self._fallback_label.size(), Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
        except Exception as e:
            self._fallback_label.setText(f'3D render error:\n{e}')

    @staticmethod
    def _wv_color(wv_nm):
        if 380 <= wv_nm <= 780:
            if wv_nm < 490:
                return '#4488ff'
            elif wv_nm < 580:
                return '#44ff88'
            elif wv_nm < 645:
                return '#ffaa44'
            else:
                return '#ff4444'
        return '#8888cc'

    def closeEvent(self, event):
        if self._plotter is not None:
            self._plotter.close()
        super().closeEvent(event)
