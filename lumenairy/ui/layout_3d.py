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
from ..elements.lenses import surface_sag_biconic

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
        # 3.6.1: bidirectional table <-> layout selection.  See
        # the matching machinery in layout_2d.py for the full
        # explanation; here we just track the index and re-add a
        # gold highlight ring after every rebuild.
        self._selected_elem_idx = -1
        # 3.6.1 hotfix-3: camera-position persistence.  The first
        # rebuild snaps to an iso view; subsequent rebuilds
        # (triggered by every parameter edit) preserve whatever
        # orientation the user has rotated to, instead of fighting
        # them by re-snapping every time.  The toolbar's "Reset
        # View" button still calls _reset_camera() directly.
        self._camera_initialized = False

        # 3.7.6: allow the layout dock to shrink freely.  Previous
        # attempts (3.7.2 setMinimumSize + minimumSizeHint override)
        # didn't help because Qt's QDockWidget reads the layout's
        # effective minimum size (sum of child minimums for a
        # QVBoxLayout) regardless of the widget's own setMinimumSize.
        # The correct decoupling is
        # ``setSizeConstraint(QLayout.SetNoConstraint)`` on the main
        # QVBoxLayout, paired with a QToolBar (Qt's built-in toolbar
        # with overflow handling) instead of the QHBoxLayout +
        # QPushButton row that summed ~720 px of hard minimums and
        # pinned the dock open.  When the dock is narrower than the
        # toolbar's natural width, QToolBar shows a ">>" overflow
        # button and pushes hidden actions into a popup -- the user
        # never loses access to Refresh / View axes / camera roll
        # even when the dock is 80 px wide.
        from PySide6.QtWidgets import QSizePolicy, QLayout, QToolBar
        from PySide6.QtGui import QAction
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSizeConstraint(QLayout.SetNoConstraint)

        # 3.7.6: QToolBar replaces the QHBoxLayout-with-buttons --
        # QToolBar's built-in overflow handler shows a ">>" popup
        # when narrower than its actions need.  Use Expanding
        # horizontally (so it fills the dock width) + Fixed
        # vertically; ``minimumWidth(0)`` lets it accept clipping.
        toolbar = QToolBar(self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.setMinimumWidth(0)
        toolbar.setStyleSheet(
            'QToolBar { spacing: 4px; border: 0; }'
            'QToolButton { padding: 2px 8px; }')

        act_refresh = toolbar.addAction('Refresh')
        act_refresh.triggered.connect(self.rebuild)
        act_refresh.setToolTip('Rebuild the 3D scene from the model.')

        act_reset = toolbar.addAction('Reset View')
        act_reset.triggered.connect(self._reset_camera)
        act_reset.setToolTip('Fit the entire system in view '
                              '(isometric).')

        toolbar.addSeparator()

        act_roll_ccw = toolbar.addAction('⟲')
        act_roll_ccw.triggered.connect(lambda: self._roll_view(-15))
        act_roll_ccw.setToolTip(
            'Rotate the view 15° counter-clockwise in the screen '
            'plane (camera roll).')

        act_roll_cw = toolbar.addAction('⟳')
        act_roll_cw.triggered.connect(lambda: self._roll_view(+15))
        act_roll_cw.setToolTip(
            'Rotate the view 15° clockwise in the screen plane '
            '(camera roll).')

        toolbar.addSeparator()

        for label, view_id in [('Front', 'xy'), ('Side', 'side2d'),
                                ('Top', 'yz'), ('Iso', 'iso')]:
            act = toolbar.addAction(label)
            act.triggered.connect(
                lambda checked=False, v=view_id: self._snap_to_view(v))
            act.setToolTip(f'Snap the camera to the {label.lower()} '
                            'view.')

        layout.addWidget(toolbar)

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
            # 3.7.6: let the VTK widget shrink with the dock --
            # ``Expanding`` so it fills available space (the
            # natural default for a viewport widget),
            # ``setMinimumSize(0, 0)`` so it can shrink to zero
            # when the dock is dragged narrow.  Using
            # ``Ignored`` here is wrong: it tells Qt the viewport
            # doesn't want any space, and the dock area allocates
            # minimal space at launch instead of letting the user
            # shrink the dock from a sensible default.
            self._plotter.interactor.setMinimumSize(0, 0)
            self._plotter.interactor.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(self._plotter.interactor, stretch=1)
        elif PYVISTA_AVAILABLE:
            # Fallback: static image label
            self._fallback_label = QLabel('pyvistaqt not available — static render only.\npip install pyvistaqt')
            self._fallback_label.setAlignment(Qt.AlignCenter)
            self._fallback_label.setStyleSheet(
                "QLabel { background: #050709; color: #a0b4d0; "
                "font-family: Consolas; font-size: 12px; }")
            self._fallback_label.setMinimumSize(0, 0)
            self._fallback_label.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(self._fallback_label, stretch=1)
        else:
            lbl = QLabel('PyVista not installed.\npip install pyvista pyvistaqt')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "QLabel { background: #050709; color: #a0b4d0; "
                "font-family: Consolas; font-size: 12px; }")
            lbl.setMinimumSize(0, 0)
            lbl.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(lbl, stretch=1)

        self.sm.system_changed.connect(self.rebuild)
        self.sm.trace_ready.connect(lambda _: self._draw_rays())

    def rebuild(self):
        """Rebuild the entire 3D scene from the model.

        3.6.1 hotfix-4: uses ``element_frames_3d_mm`` to position and
        orient each element in the world frame so mirror folds and
        coord-break tilts actually bend the geometry.  Local surface
        primitives are placed at the element's world origin and
        rotated by the element's rotation matrix.
        """
        if self._plotter is None:
            if PYVISTA_AVAILABLE and not PYVISTAQT_AVAILABLE:
                self._render_static()
            return

        # 3.6.1 hotfix-3: snapshot the camera state before we tear
        # down the scene so the user's orbit/zoom/pan survive a
        # parameter edit.  Only restored when _camera_initialized
        # is True (i.e., not the first rebuild).
        saved_cam = None
        if self._camera_initialized:
            try:
                saved_cam = self._plotter.camera_position
            except Exception:
                saved_cam = None

        self._plotter.clear()

        elements = self.sm.elements
        if len(elements) < 2:
            return

        # Per-element 3D frames (origin_xyz, 3x3 R) carrying mirror
        # reflections and absorbed coord-break tilt/decenter info.
        frames = self.sm.element_frames_3d_mm()

        # ── Optical-axis polyline (traces the bent path) ──
        # 3.6.1 hotfix-6: post-mirror Rflip on R[:,2] alone gives
        # the correct world direction for internal-thickness
        # advances; do NOT apply mirror_sign here.
        axis_pts = []
        if frames:
            o0, R0 = frames[0]
            axis_pts.append(o0 - 20.0 * R0[:, 2])
        for ei, elem in enumerate(elements):
            o, R = frames[ei]
            if elem.elem_type == 'Source':
                axis_pts.append(o.copy())
                continue
            axis_pts.append(o.copy())
            if elem.surfaces:
                internal = sum(
                    float(s.thickness) for s in elem.surfaces)
                axis_pts.append(o + internal * R[:, 2])
        if frames:
            o_last, R_last = frames[-1]
            axis_pts.append(o_last + 20.0 * R_last[:, 2])
        if len(axis_pts) >= 2:
            try:
                axis_poly = pv.lines_from_points(np.array(axis_pts))
                self._plotter.add_mesh(
                    axis_poly, color='#3a4a60', line_width=1,
                    name='axis')
            except Exception:
                pass

        # ── Draw each element in its rotated frame ──
        for ei, elem in enumerate(elements):
            o, R = frames[ei]
            if elem.elem_type == 'Source':
                src = getattr(elem, 'source', None) or self.sm.source
                self._draw_source_3d(o, R, src, ei)
                continue
            if elem.elem_type == 'Detector':
                continue

            cum_t = 0.0
            for si, srow in enumerate(elem.surfaces):
                surf_origin = o + cum_t * R[:, 2]
                sd = srow.semi_diameter \
                    if np.isfinite(srow.semi_diameter) \
                    else self.sm.epd_mm / 2

                if srow.surf_type == 'Mirror' \
                        or elem.elem_type == 'Mirror':
                    self._draw_mirror_3d(
                        surf_origin, R, sd, srow,
                        ei * 100 + si)
                else:
                    self._draw_surface_3d(
                        surf_origin, R, sd, srow,
                        ei * 100 + si, elem.surfaces)
                cum_t += float(srow.thickness)

        # ── Image plane (detector) in its frame ──
        if frames:
            o_det, R_det = frames[-1]
            ima_disc = pv.Disc(
                center=tuple(o_det),
                normal=tuple(R_det[:, 2]),
                inner=0, outer=self.sm.epd_mm / 2)
            self._plotter.add_mesh(
                ima_disc, color='#884444', opacity=0.3,
                name='image_plane')

        # Draw rays if we have a trace result
        self._draw_rays()

        # 3.6.1: redraw selection highlight on top after rebuild.
        self._redraw_highlight_3d()

        # 3.6.1 hotfix-3: restore camera state if we have one;
        # otherwise (first rebuild) snap to the default iso view
        # and remember that the camera has been initialised.
        if saved_cam is not None:
            try:
                self._plotter.camera_position = saved_cam
            except Exception:
                # If restoration fails for any reason, fall back
                # to the iso-view snap so the user still gets a
                # usable view.
                self._reset_camera()
        else:
            self._reset_camera()
            self._camera_initialized = True

    def _draw_surface_3d(self, surf_origin, R_world, sd, row, idx,
                          all_surfaces):
        """Draw a refractive surface as a curved disc.

        ``surf_origin`` is the surface vertex in world mm; ``R_world``
        is the 3x3 rotation matrix mapping local axes (x, y, z=optical
        axis) into world coordinates.  Sag is evaluated in the local
        frame with ``surface_sag_biconic``, then the resulting points
        are rotated and translated into world coordinates so a tilted
        or folded surface draws in its proper orientation.
        """
        Rrad = row.radius
        n_theta = 48
        n_radial = 16
        theta = np.linspace(0, 2 * np.pi, n_theta)
        radii = np.linspace(0, sd, n_radial)

        # Vectorised sag eval in the local frame (mm → m → mm).
        T, Rg = np.meshgrid(theta, radii)        # (n_radial, n_theta)
        x_mm = Rg * np.cos(T)
        y_mm = Rg * np.sin(T)
        try:
            sag_m = surface_sag_biconic(
                x_mm.ravel() * 1e-3, y_mm.ravel() * 1e-3,
                R_x=Rrad * 1e-3 if np.isfinite(Rrad) else np.inf,
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

        # Local-frame points: (x, y, sag).  Rotate into world and add
        # the surface origin.  All meshes built downstream use the
        # transformed points directly.
        local_pts = np.column_stack([
            x_mm.ravel(), y_mm.ravel(), sag_mm.ravel(),
        ])
        world_pts = local_pts @ R_world.T + surf_origin[None, :]

        faces = []
        for ir in range(n_radial - 1):
            for it in range(n_theta - 1):
                p0 = ir * n_theta + it
                p1 = ir * n_theta + it + 1
                p2 = (ir + 1) * n_theta + it + 1
                p3 = (ir + 1) * n_theta + it
                faces.append([4, p0, p1, p2, p3])

        if faces:
            mesh = pv.PolyData(world_pts, np.array(faces))
            color = '#5588cc' if row.glass else '#aaaaaa'
            self._plotter.add_mesh(mesh, color=color, opacity=0.35,
                                   name=f'surf_{idx}')

        # Glass volume between this surface and the next, oriented
        # along the local +z axis (world: R_world[:, 2]).
        if row.glass and idx < len(all_surfaces) - 1:
            t = row.thickness if np.isfinite(row.thickness) else 0
            if abs(t) > 0.01:
                axis_w = R_world[:, 2]
                center = surf_origin + 0.5 * t * axis_w
                cyl = pv.Cylinder(
                    center=tuple(center),
                    direction=tuple(axis_w),
                    radius=sd, height=abs(t),
                    resolution=48)
                self._plotter.add_mesh(
                    cyl, color='#334466', opacity=0.08,
                    name=f'glass_{idx}')

        # Edge ring at the surface vertex, in the local xy plane.
        ring_local = np.column_stack([
            sd * np.cos(theta), sd * np.sin(theta), np.zeros(n_theta),
        ])
        ring_world = ring_local @ R_world.T + surf_origin[None, :]
        ring = pv.lines_from_points(
            np.vstack([ring_world, ring_world[:1]]))
        ring_color = '#5588cc' if row.glass else '#888888'
        self._plotter.add_mesh(
            ring, color=ring_color, line_width=1.5,
            name=f'ring_{idx}')

    def _draw_mirror_3d(self, surf_origin, R_world, sd, row, idx):
        """Draw a mirror as an opaque disc with edge ring at the
        surface vertex, oriented by the element's rotation matrix.
        """
        disc = pv.Disc(
            center=tuple(surf_origin),
            normal=tuple(R_world[:, 2]),
            inner=0, outer=sd, r_res=1, c_res=48)
        self._plotter.add_mesh(disc, color='#7799cc', opacity=0.5,
                               name=f'mirror_{idx}')

    def _draw_source_3d(self, origin, R_world, src, idx):
        """3.6.1: render the optical source in the 3D view.

        Mirrors layout_2d.py._draw_source.  Per-source-type
        primitives are added with name=f'source_{idx}' so subsequent
        rebuilds replace cleanly.  All sizes are in mm (the scene's
        native unit).
        """
        if self._plotter is None or src is None:
            return
        accent = '#78dc8c'   # mint green, matches 2D layout
        epd_h = max(2.0, self.sm.epd_mm / 2)
        st = src.source_type
        # World-frame helpers: source origin and the local +z axis
        # in world coordinates.  All glyphs are drawn at ``origin``
        # with their natural local-frame orientation, then their
        # normal / direction vectors are mapped through R_world so
        # a folded source still renders correctly.
        o = np.asarray(origin, dtype=float)
        z_axis_w = R_world[:, 2]

        def to_world(local_xyz):
            """Map a length-3 local point into world coords."""
            return tuple(o + R_world @ np.asarray(local_xyz, dtype=float))

        try:
            if st == 'plane_wave':
                # Disc cap perpendicular to the optical axis +
                # cone glyph pointing into the system to indicate
                # propagation direction.
                cap = pv.Disc(center=tuple(o),
                              normal=tuple(z_axis_w),
                              inner=0, outer=epd_h, c_res=48)
                self._plotter.add_mesh(
                    cap, color=accent, opacity=0.45,
                    name=f'source_{idx}')
                arrow = pv.Cone(center=to_world((0, 0, epd_h * 0.3)),
                                direction=tuple(z_axis_w),
                                height=epd_h * 0.4,
                                radius=epd_h * 0.15)
                self._plotter.add_mesh(
                    arrow, color=accent, opacity=0.7,
                    name=f'source_arrow_{idx}')

            elif st in ('gaussian', 'gaussian_aperture'):
                if st == 'gaussian':
                    w_mm = max(0.05, src.beam_diameter_mm)
                else:
                    w_mm = max(0.05, src.sigma_mm * 2)
                # Oblate-sphere proxy for a Gaussian waist; rotation
                # is approximate (sphere scaling is axis-aligned),
                # so we accept a slight orientation mismatch when
                # the source itself is tilted.
                sphere = pv.Sphere(center=tuple(o), radius=w_mm / 2)
                sphere.scale([1.0, 1.0, 0.25], inplace=True)
                self._plotter.add_mesh(
                    sphere, color=accent, opacity=0.55,
                    name=f'source_{idx}')

            elif st == 'top_hat':
                half = max(0.5, src.top_hat_diameter_mm / 2)
                cap = pv.Disc(center=tuple(o),
                              normal=tuple(z_axis_w),
                              inner=0, outer=half, c_res=48)
                self._plotter.add_mesh(
                    cap, color=accent, opacity=0.65,
                    name=f'source_{idx}')

            elif st == 'fiber_mode':
                half = max(0.05, src.fiber_mfd_um * 1e-3 / 2)
                outer = pv.Disc(center=tuple(o),
                                normal=tuple(z_axis_w),
                                inner=half, outer=half * 2.5,
                                c_res=48)
                inner = pv.Sphere(center=tuple(o), radius=half)
                self._plotter.add_mesh(
                    outer, color=accent, opacity=0.35,
                    name=f'source_{idx}')
                self._plotter.add_mesh(
                    inner, color=accent, opacity=0.85,
                    name=f'source_core_{idx}')

            elif st == 'point_source':
                # Small bright sphere; size scales with EPD.
                r = max(0.4, epd_h * 0.04)
                sphere = pv.Sphere(center=tuple(o), radius=r)
                self._plotter.add_mesh(
                    sphere, color=accent, opacity=0.95,
                    name=f'source_{idx}')

            elif st == 'emitter_array':
                # Grid of small spheres at the emitter pitch in the
                # local x-y plane, mapped to world via R_world.
                nx = max(1, min(7, src.emitter_nx))
                ny = max(1, min(7, src.emitter_ny))
                pitch = max(0.005, src.emitter_pitch_mm)
                w0 = max(0.001, src.emitter_waist_mm)
                blocks = pv.MultiBlock()
                for ix in range(nx):
                    for iy in range(ny):
                        cx = (ix - (nx - 1) / 2) * pitch
                        cy = (iy - (ny - 1) / 2) * pitch
                        blocks.append(pv.Sphere(
                            center=to_world((cx, cy, 0.0)),
                            radius=w0))
                self._plotter.add_mesh(
                    blocks, color=accent, opacity=0.9,
                    name=f'source_{idx}')

            else:
                # Unknown future type: marker sphere.
                sphere = pv.Sphere(center=tuple(o), radius=0.5)
                self._plotter.add_mesh(
                    sphere, color=accent, opacity=0.6,
                    name=f'source_{idx}')
        except Exception:
            # Don't let an unusual source type crash the whole
            # layout build; fall back to a placeholder marker.
            try:
                self._plotter.add_mesh(
                    pv.Sphere(center=tuple(o), radius=0.5),
                    color=accent, opacity=0.6,
                    name=f'source_{idx}')
            except Exception:
                pass

    def set_selected_element(self, idx: int):
        """Public slot wired in main_window.py to the new
        element_selected_in_table signal.  Stores the selected index
        and repaints the highlight ring.
        """
        try:
            idx = int(idx)
        except Exception:
            idx = -1
        self._selected_elem_idx = idx
        self._redraw_highlight_3d()

    def _redraw_highlight_3d(self):
        """Repaint the gold selection ring on the highlighted element.

        3.6.1 hotfix-4: ring is placed at the element's world-frame
        origin and oriented by its rotation matrix so a folded
        element shows its highlight ring tilted with the element.
        """
        if self._plotter is None:
            return
        try:
            self._plotter.remove_actor('selection_highlight')
        except Exception:
            pass
        idx = self._selected_elem_idx
        if idx < 0:
            return
        elements = self.sm.elements
        if idx >= len(elements):
            return
        try:
            frames = self.sm.element_frames_3d_mm()
        except Exception:
            return
        if idx >= len(frames):
            return
        o, R = frames[idx]
        elem = elements[idx]
        # Determine the element's lateral extent for the ring radius.
        if elem.surfaces:
            sd = elem.surfaces[0].semi_diameter
            if not np.isfinite(sd):
                sd = self.sm.epd_mm / 2
        else:
            sd = self.sm.epd_mm / 2
        ring_r = max(2.0, sd * 1.15)
        try:
            ring = pv.Disc(center=tuple(o),
                           normal=tuple(R[:, 2]),
                           inner=ring_r, outer=ring_r * 1.06,
                           c_res=64)
            self._plotter.add_mesh(
                ring, color='gold', opacity=0.85,
                name='selection_highlight')
        except Exception:
            pass

    def _draw_rays(self):
        """Overlay traced rays on the 3D scene.

        3.6.1 hotfix-4: rays are placed in the per-element 3D world
        frames returned by ``element_frames_3d_mm``.  Each surface
        contributes a ``(origin, R)`` pair; a ray's local (x, y)
        height at that surface maps to world via
        ``origin + x*R[:,0] + y*R[:,1]`` so rays bend correctly
        through mirrors and coord-break tilts in 3D.
        """
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

        # 3.6.1 hotfix-6: per-traced-surface 3D world frames in trace
        # order (incl. image plane), with mirror_sign tracking so
        # post-mirror surfaces sit on the correct side of the mirror.
        surf_frames = self.sm.surface_frames_3d_mm()
        if not surf_frames:
            return
        elem_frames = self.sm.element_frames_3d_mm()
        if not elem_frames:
            return

        n_rays = result.input_rays.n_rays
        step = max(1, n_rays // 40)
        history = result.ray_history
        # 3.7.5: per-segment direction-based colouring for folded
        # systems (matches the 2D layout palette).  Pre-Mirror-1
        # = blue, post-Mirror-1 = orange, post-Mirror-2 = green,
        # subsequent folds cycle through magenta / yellow / cyan.
        # 3.7.5: walk the world-frame surface list that was actually
        # traced (``result.surfaces``) so mirror-count indices align
        # with ``ray_history``.  The legacy ``build_trace_surfaces``
        # still emits cb_pre entries that the world trace skips.
        trace_surfs = result.surfaces
        has_mirrors = any(getattr(s, 'is_mirror', False)
                          for s in trace_surfs)
        mirrors_before = [0]
        running = 0
        for s in trace_surfs:
            if getattr(s, 'is_mirror', False):
                running += 1
            mirrors_before.append(running)
        direction_palette = [
            '#7cbcff',   # 0 mirrors: forward (blue)
            '#ff9966',   # 1 mirror: return (orange)
            '#88ff66',   # 2 mirrors: post-fold (green)
            '#dc82ff',   # 3 mirrors: magenta
            '#ffdc64',   # 4 mirrors: yellow
            '#64ffdc',   # 5+ mirrors: cyan
        ]
        if self.sm.prefs.get('ray_use_wavelength', True):
            wv_color = self._wv_color(self.sm.wavelength_nm)
        else:
            wv_color = self.sm.prefs.get('ray_color', '#5cb8ff')

        def color_for_segment(seg_idx):
            if not has_mirrors:
                return wv_color
            mc = min(mirrors_before[seg_idx],
                     len(direction_palette) - 1)
            return direction_palette[mc]

        # Source frame for the launch point.
        o_src, R_src = elem_frames[0]

        def to_world(si, x_local, y_local, z_local=0.0):
            """3.7.2: include z_local so coord-break frames where
            the rotation transform leaves a non-zero ray.z render
            at the correct world position instead of jumping."""
            o, R = surf_frames[si]
            return (o + x_local * R[:, 0]
                    + y_local * R[:, 1]
                    + z_local * R[:, 2])

        for r in range(0, n_rays, step):
            if not result.input_rays.alive[r]:
                continue

            pts = []
            x_in = result.input_rays.x[r] * 1e3   # m → mm
            y_in = result.input_rays.y[r] * 1e3

            # Source launch point in world coords.
            pts.append(o_src + x_in * R_src[:, 0]
                       + y_in * R_src[:, 1])

            # Pre-lens segment: end at surface 0 using history[0]'s
            # local (x, y, z) so a tilted launch terminates at the
            # right height (and a cb's non-zero z is honoured).
            if history and len(history) > 0 and history[0].alive[r]:
                z0 = history[0].z[r] * 1e3 if hasattr(
                    history[0], 'z') else 0.0
                pts.append(to_world(0,
                                    history[0].x[r] * 1e3,
                                    history[0].y[r] * 1e3,
                                    z0))
            else:
                pts.append(to_world(0, x_in, y_in))

            # Through-system + image plane.
            for si in range(1, len(history)):
                if not history[si].alive[r]:
                    break
                x_l = history[si].x[r] * 1e3
                y_l = history[si].y[r] * 1e3
                z_l = (history[si].z[r] * 1e3
                        if hasattr(history[si], 'z') else 0.0)
                if si < len(surf_frames):
                    pts.append(to_world(si, x_l, y_l, z_l))
                else:
                    o_last, R_last = surf_frames[-1]
                    pts.append(pts[-1] + 20.0 * R_last[:, 2])

            # 3.7.5: draw each segment as its own mesh so direction-
            # based colouring works.  Group consecutive segments
            # that share the same colour into a single polyline to
            # keep the actor count down (one actor per
            # direction-colour transition).
            if len(pts) >= 2:
                run_start = 0
                run_color = color_for_segment(0)
                for j in range(1, len(pts) - 1):
                    seg_color = color_for_segment(j)
                    if seg_color != run_color:
                        # Flush the previous run as one polyline.
                        run_pts = np.asarray(pts[run_start:j + 1])
                        ray_line = pv.lines_from_points(run_pts)
                        self._plotter.add_mesh(
                            ray_line, color=run_color,
                            line_width=1.2, opacity=0.55,
                            name=f'ray_{r}_{run_start}')
                        run_start = j
                        run_color = seg_color
                # Trailing run.
                run_pts = np.asarray(pts[run_start:])
                if len(run_pts) >= 2:
                    ray_line = pv.lines_from_points(run_pts)
                    self._plotter.add_mesh(
                        ray_line, color=run_color,
                        line_width=1.2, opacity=0.55,
                        name=f'ray_{r}_{run_start}')

    def _compute_z_positions(self):
        """Cumulative z positions from element distances.

        Retained for backward compatibility with older callers
        (e.g., the now-removed ``_draw_rays`` body that used to
        consume this) but no longer used internally; the new
        world-frame surface-z calculation lives inline in
        ``_draw_rays``.
        """
        return self.sm.element_z_positions_mm()

    def _roll_view(self, deg):
        """3.6.1: rotate the camera in the plane of the screen by
        ``deg`` degrees (positive = clockwise from the user's
        perspective).  Implemented as a roll about the camera's
        forward (view) axis so the perceived orientation rotates
        without changing what's facing the viewer.
        """
        if self._plotter is None:
            return
        try:
            cam = self._plotter.camera
            cam.roll = (cam.roll or 0.0) + float(deg)
            self._plotter.render()
        except Exception:
            pass

    def _reset_camera(self):
        """3.7.2: Reset to a 2D-matching side view by default
        (optical axis +z horizontal-right, +y vertical-up).  Use
        the Iso toolbar button for the previous slight-elevation
        oblique view.
        """
        self._snap_to_view('side2d')

    def _snap_to_view(self, view):
        """Snap camera to an axis-aligned or isometric view.

        3.7.2: ``'side2d'`` is the new default view that matches the
        2D layout's side-view orientation (camera looks from -x;
        on-screen, +z is right and +y is up).  PyVista's stock
        ``'xz'`` / ``'yz'`` strings have the wrong axis swap (z up
        instead of z right), so a custom camera position is set.
        """
        if self._plotter is None:
            return
        if view == 'xy':
            self._plotter.camera_position = 'xy'
        elif view == 'xz':
            self._plotter.camera_position = 'xz'
        elif view == 'yz':
            self._plotter.camera_position = 'yz'
        elif view == 'side2d':
            # Match the 2D side-view: optical axis +z horizontal,
            # +y up.  Camera at -x looking through origin.
            try:
                bounds = self._plotter.bounds
                cx = 0.5 * (bounds[0] + bounds[1])
                cy = 0.5 * (bounds[2] + bounds[3])
                cz = 0.5 * (bounds[4] + bounds[5])
                # Camera distance: enough that reset_camera can
                # tighten the fit while preserving orientation.
                dx = max(50.0, abs(bounds[1] - bounds[0]) * 2)
            except Exception:
                cx = cy = cz = 0.0
                dx = 200.0
            self._plotter.camera_position = [
                (cx - dx, cy, cz),  # camera at -x
                (cx, cy, cz),       # focal at scene centre
                (0.0, 1.0, 0.0),    # +y up
            ]
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

    def minimumSizeHint(self):
        """3.7.2: report a tiny minimum so the QDockWidget will let
        the user shrink the layout pane.  When 2D and 3D layouts
        are tabbed in the same dock, the dock honours the LARGER
        of their minimumSizeHints, so 3D needs the same override
        as 2D for the dock to actually shrink.
        """
        from PySide6.QtCore import QSize
        return QSize(40, 40)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(400, 200)

    def closeEvent(self, event):
        if self._plotter is not None:
            self._plotter.close()
        super().closeEvent(event)
