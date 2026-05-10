"""
Layout2DView — interactive 2-D optical system layout.

Renders lenses, mirrors, and traced rays on a QGraphicsScene.
Automatically updates when the SystemModel changes.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsTextItem, QWidget, QVBoxLayout,
)
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPainterPath, QFont, QTransform,
)

import numpy as np

from .model import SystemModel
from ..elements.lenses import surface_sag_biconic


class Layout2DView(QWidget):
    """2-D cross-section of the optical system with ray overlay."""

    # Scale: 1 mm = SCALE pixels in scene coords
    SCALE = 4.0

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        # 3.6.1 hotfix-6: allow the layout dock to shrink freely.
        # Override minimumSizeHint as well -- the inherited
        # implementation walks the layout (QGraphicsView etc.) and
        # produces a much larger value than setMinimumSize alone
        # can defeat.
        from PySide6.QtWidgets import QSizePolicy, QPushButton, QHBoxLayout
        from PySide6.QtCore import QSize
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3.6.1 hotfix-6: thin toolbar with Reset View + zoom hints.
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 2, 2, 2)
        btn_reset = QPushButton('Reset View')
        btn_reset.setFixedHeight(20)
        btn_reset.setToolTip(
            'Fit the entire optical system in view (Ctrl+0 also works '
            'when the dock has focus).')
        btn_reset.clicked.connect(self._reset_view)
        toolbar.addWidget(btn_reset)
        toolbar.addStretch()
        from PySide6.QtWidgets import QLabel
        hint = QLabel('Wheel: zoom · Drag: pan')
        hint.setStyleSheet('color: #7a94b8; font-size: 10px;')
        toolbar.addWidget(hint)
        layout.addLayout(toolbar)

        self.scene = QGraphicsScene()
        bg = self.sm.prefs.get('bg_2d', '#050709')
        self.scene.setBackgroundBrush(QBrush(QColor(bg)))

        self.view = QGraphicsView(self.scene)
        from PySide6.QtGui import QPainter as _P
        self.view.setRenderHint(_P.RenderHint.Antialiasing, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setMinimumSize(0, 0)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # 3.6.1 hotfix-6: install our own wheelEvent on the view so
        # the wheel always zooms regardless of scrollbar state.  At
        # high zoom the QGraphicsView's default handler routes the
        # wheel to its scrollbars instead of bubbling up to our
        # parent-widget handler -- the user gets stuck panning when
        # they wanted to keep zooming.
        self.view.wheelEvent = self._view_wheel_event

        layout.addWidget(self.view)

        # Connect
        # Map z-position (scene coords) to surface index for click detection
        self._surface_zones = []  # list of (z_start, z_end, surface_index)
        # 3.6.1: index of the element highlighted in the layout (kept in
        # sync with the prescription-table selection via the new
        # element_selected_in_table signal in main_window).  -1 means
        # no selection.  Highlight items live in self._highlight_items
        # so they can be removed/redrawn without a full rebuild.
        self._selected_elem_idx = -1
        self._highlight_items = []

        self.sm.system_changed.connect(self.rebuild)
        self.sm.trace_ready.connect(self._draw_rays)

        # Click on the layout → select the corresponding surface row
        self.view.setMouseTracking(True)
        self.view.mousePressEvent = self._on_view_click

        self.rebuild()

    def rebuild(self):
        """Rebuild the entire scene from the model."""
        self.scene.clear()
        self._surface_zones = []
        S = self.SCALE

        elements = self.sm.elements
        if len(elements) < 2:
            return

        # Empty-state onboarding: only Source + Detector present.
        # Draw a friendly hint in the middle of the scene so the first
        # launch has a clear next step.
        if all(e.elem_type in ('Source', 'Detector') for e in elements):
            hint = QGraphicsTextItem(
                'Empty optical system.\n\n'
                '  \u2022 Insert \u2192 Lens \u2192 Plano-Convex Singlet    (Ctrl+L)\n'
                '  \u2022 Insert \u2192 Thorlabs Catalog\n'
                '  \u2022 File \u2192 Open Prescription\n\n'
                'Ctrl+E focuses the element table above.')
            hint.setDefaultTextColor(QColor(180, 200, 230))
            hint.setFont(QFont('Consolas', 11))
            # Place at origin, a bit above the axis
            hint.setPos(40, -80)
            self.scene.addItem(hint)
            self.view.fitInView(
                self.scene.sceneRect().adjusted(-40, -40, 40, 40),
                Qt.KeepAspectRatio,
            )
            return

        # 3.6.1: per-element 2D frames in mm + radians.  Carry
        # mirror reflections and tilt_x / decenter_y (the absorbed
        # COORDBRK information from .zmx import) so a folded
        # geometry actually shows up bent in the layout.
        frames = self.sm.element_frames_2d_mm()

        # ── Draw the optical-axis polyline ──
        # Each segment goes from one element's front vertex to the
        # next, plus the in-element internal-thickness span.  This
        # replaces the single horizontal dashed line: the axis now
        # actually traces the path through the system.
        axis_pen = QPen(QColor(40, 50, 65), 1, Qt.DashLine)
        path_pts = []
        # Start a bit upstream of the source for context.
        if frames:
            z0, y0, t0 = frames[0]
            path_pts.append((z0 - 20 * np.cos(t0),
                             y0 - 20 * np.sin(t0)))
        for ei, elem in enumerate(elements):
            z_w, y_w, theta = frames[ei]
            if elem.elem_type == 'Source':
                # Source plane sits at the source's frame; advance
                # path through the source plane and into the gap to
                # the next element handled by the next element.
                path_pts.append((z_w, y_w))
                continue
            # Append the front vertex of this element ...
            path_pts.append((z_w, y_w))
            # ... and the back vertex (front + internal thickness
            # along this element's local axis).
            internal = sum(s.thickness for s in elem.surfaces) \
                if elem.surfaces else 0.0
            path_pts.append((z_w + internal * np.cos(theta),
                             y_w + internal * np.sin(theta)))
        # Extend a bit past the last element along its outgoing
        # direction so the final segment doesn't terminate abruptly
        # at the detector.
        if frames:
            z_last, y_last, t_last = frames[-1]
            path_pts.append((z_last + 20 * np.cos(t_last),
                             y_last + 20 * np.sin(t_last)))
        for j in range(len(path_pts) - 1):
            self.scene.addLine(
                path_pts[j][0] * S, path_pts[j][1] * S,
                path_pts[j + 1][0] * S, path_pts[j + 1][1] * S,
                axis_pen)

        # ── Draw each element in its rotated frame ──
        from PySide6.QtGui import QTransform
        for ei, elem in enumerate(elements):
            z_w, y_w, theta = frames[ei]
            if elem.elem_type == 'Source':
                src = getattr(elem, 'source', None) or self.sm.source
                z_src = z_w * S
                # Source glyph still drawn axis-aligned; rotation at
                # the source itself is 0 in the unfolded case.
                self._draw_source(z_src, src)
                zone_w = max(8, self.sm.epd_mm / 2 * S * 0.3)
                self._surface_zones.append(
                    (z_src - zone_w, z_src + zone_w, ei))
                continue
            if elem.elem_type == 'Detector':
                # Image plane is drawn in its own frame below.
                continue

            # Track which scene items belong to this element so we
            # can group them and apply the rotation transform in one
            # shot (Qt's createItemGroup walks an item list).
            items_before = set(self.scene.items())

            for si, srow in enumerate(elem.surfaces):
                # Local z of this surface inside the element.
                z_local = sum(s.thickness
                               for s in elem.surfaces[:si]) * S
                sd = (srow.semi_diameter
                      if np.isfinite(srow.semi_diameter)
                      else self.sm.epd_mm / 2)
                h = sd * S

                if srow.surf_type == 'Mirror' \
                        or elem.elem_type == 'Mirror':
                    self._draw_mirror(z_local, h, srow)
                else:
                    self._draw_surface(z_local, h, srow,
                                        si, elem.surfaces)

            new_items = [it for it in self.scene.items()
                         if it not in items_before]
            if new_items:
                group = self.scene.createItemGroup(new_items)
                t = QTransform()
                t.translate(z_w * S, y_w * S)
                t.rotate(float(np.degrees(theta)))
                group.setTransform(t)

            # Register click zone at the element's world-frame
            # position.  Zone is axis-aligned in scene coords (the
            # click test is point-in-rect against z and y), so we
            # use a slightly enlarged box around the element's
            # rotated extent.
            sd = (elem.surfaces[0].semi_diameter
                  if elem.surfaces else self.sm.epd_mm / 2)
            if not np.isfinite(sd):
                sd = self.sm.epd_mm / 2
            h = sd * S
            zone_w = max(10, h * 0.3)
            self._surface_zones.append(
                (z_w * S - zone_w, z_w * S + zone_w, ei))

        # ── Draw image plane (detector) in its frame ──
        det_idx = len(elements) - 1
        z_ima_w, y_ima_w, t_ima = frames[-1] if frames else (0, 0, 0)
        ima_pen = QPen(QColor(180, 100, 100), 1.5, Qt.DashDotLine)
        sd_ima = self.sm.epd_mm / 2
        # Rotated image-plane line: from (z_ima, y_ima - h) to
        # (z_ima, y_ima + h) in the local frame, then rotated.
        ux, uy = -np.sin(t_ima), np.cos(t_ima)  # local +y in world
        self.scene.addLine(
            (z_ima_w + ux * (-sd_ima)) * S,
            (y_ima_w + uy * (-sd_ima)) * S,
            (z_ima_w + ux * sd_ima) * S,
            (y_ima_w + uy * sd_ima) * S,
            ima_pen)

        label = QGraphicsTextItem('Image')
        label.setDefaultTextColor(QColor(200, 130, 130))
        label.setFont(QFont('Consolas', 8))
        label.setPos((z_ima_w + ux * sd_ima) * S - 10,
                     (y_ima_w + uy * sd_ima) * S + 5)
        self.scene.addItem(label)

        # Detector click zone.
        if det_idx >= 0 and elements[det_idx].elem_type == 'Detector':
            zone_w = max(8.0, sd_ima * S * 0.3)
            self._surface_zones.append(
                (z_ima_w * S - zone_w, z_ima_w * S + zone_w, det_idx))

        # ── Fit view ──
        self.view.fitInView(
            self.scene.sceneRect().adjusted(-40, -40, 40, 40),
            Qt.KeepAspectRatio,
        )

        # 3.6.1: re-draw the selection highlight on top after a full
        # rebuild so the gold outline survives auto-retraces.
        self._redraw_highlight()

        # Auto-trace
        self.sm.run_trace()

    def _draw_surface(self, z, h, row, idx, elem_surfaces):
        """Draw a refractive surface as a curved polyline.

        Uses the core ``surface_sag_biconic`` so that conic, aspheric
        polynomial, and biconic-Y contributions show up in the layout
        and stay in lockstep with the ray tracer.  ``h`` is the
        half-aperture in **scene pixels** (mm * SCALE); we convert back
        to metres to evaluate the sag and back to pixels for drawing.
        """
        S = self.SCALE
        R = row.radius
        has_glass = bool(row.glass)

        if np.isinf(R) and not getattr(row, 'aspheric_coeffs', None):
            pen = QPen(QColor(200, 200, 200), 1.5)
            self.scene.addLine(z, -h, z, h, pen)
        else:
            # Sample y across the aperture (scene pixels -> mm -> m).
            n_samp = 32
            y_px = np.linspace(-h, h, n_samp)
            y_m = (y_px / S) * 1e-3
            x_m = np.zeros_like(y_m)
            try:
                sag_m = surface_sag_biconic(
                    x_m, y_m,
                    R_x=R * 1e-3 if np.isfinite(R) else np.inf,
                    R_y=(row.radius_y * 1e-3
                         if (getattr(row, 'radius_y', None) is not None
                             and np.isfinite(row.radius_y))
                         else None),
                    conic_x=getattr(row, 'conic', 0.0) or 0.0,
                    conic_y=getattr(row, 'conic_y', None),
                    aspheric_coeffs=getattr(row, 'aspheric_coeffs', None),
                    aspheric_coeffs_y=getattr(row,
                                              'aspheric_coeffs_y', None),
                )
                sag_px = sag_m * 1e3 * S  # m -> mm -> pixels
                # Cap visual extent so over-aperture sampling can't
                # blow the layout up; doesn't affect physics.
                cap = 0.8 * abs(h)
                sag_px = np.clip(sag_px, -cap, cap)
            except Exception:
                # Fall back to a flat line on any sag failure.
                sag_px = np.zeros(n_samp)

            path = QPainterPath()
            path.moveTo(z + sag_px[0], y_px[0])
            for k in range(1, n_samp):
                path.lineTo(z + sag_px[k], y_px[k])
            pen = QPen(QColor(200, 210, 230), 1.5)
            self.scene.addPath(path, pen)

        # Glass fill between this surface and next surface in the element
        if has_glass and idx < len(elem_surfaces) - 1:
            t = row.thickness
            if np.isfinite(t) and t > 0:
                z_next = z + t * S
                fill_brush = QBrush(QColor(40, 60, 100, 60))
                self.scene.addRect(
                    min(z, z_next), -h, abs(z_next - z), 2 * h,
                    QPen(Qt.NoPen), fill_brush)

        if has_glass:
            lbl = QGraphicsTextItem(row.glass)
            lbl.setDefaultTextColor(QColor(130, 170, 230))
            lbl.setFont(QFont('Consolas', 7))
            lbl.setPos(z + 2, -h - 16)
            self.scene.addItem(lbl)

    def _draw_mirror(self, z, h, row):
        """Draw a mirror surface with hatch marks."""
        pen = QPen(QColor(140, 160, 220), 2)
        self.scene.addLine(z, -h, z, h, pen)

        # Hatch marks behind the mirror
        hatch_pen = QPen(QColor(60, 70, 100), 1)
        n_hatches = 8
        for i in range(n_hatches + 1):
            y = -h + (2 * h / n_hatches) * i
            self.scene.addLine(z, y, z + 3, y - 4, hatch_pen)

    def _draw_source(self, z, src):
        """3.6.1: draw a visible glyph for the optical source.

        Dispatches on ``src.source_type`` (model.SourceDefinition.TYPES)
        to draw a different shape per source kind.  Uses the EPD as
        a fallback height when the source has no intrinsic size
        (plane wave) so the glyph stays in proportion to the rest
        of the layout.

        Parameters
        ----------
        z : float
            Source z-position in scene pixels.
        src : SourceDefinition or None
            The current source.  If ``None``, falls back to the
            model's source.
        """
        S = self.SCALE
        if src is None:
            return
        accent = QColor(120, 220, 140)        # mint green
        pen = QPen(accent, 1.6)
        fill = QBrush(QColor(120, 220, 140, 70))
        epd_h_px = max(8.0, self.sm.epd_mm / 2 * S)
        st = src.source_type

        if st == 'plane_wave':
            # Vertical bar across the EPD + 3 short propagation
            # arrows pointing into the system.
            self.scene.addLine(z, -epd_h_px, z, epd_h_px, pen)
            for ay in (-epd_h_px * 0.6, 0.0, epd_h_px * 0.6):
                self.scene.addLine(z - 6, ay, z + 4, ay, pen)
                self.scene.addLine(z + 4, ay, z + 1, ay - 2, pen)
                self.scene.addLine(z + 4, ay, z + 1, ay + 2, pen)

        elif st in ('gaussian', 'gaussian_aperture'):
            # Filled ellipse sized by beam diameter or sigma.
            if st == 'gaussian':
                w_mm = max(0.05, src.beam_diameter_mm)
            else:
                w_mm = max(0.05, src.sigma_mm * 2)
            half = w_mm / 2 * S
            half = max(half, 6.0)
            self.scene.addEllipse(z - 3, -half, 6, 2 * half, pen, fill)

        elif st == 'top_hat':
            # Hard-edge rectangle at the source plane.
            half = max(6.0, src.top_hat_diameter_mm / 2 * S)
            self.scene.addRect(z - 3, -half, 6, 2 * half, pen, fill)

        elif st == 'fiber_mode':
            # Annular ring outline (MFD-derived radius).  No fill.
            half = max(4.0, src.fiber_mfd_um * 1e-3 / 2 * S)
            outer = max(half * 1.4, half + 4)
            self.scene.addEllipse(z - outer, -outer,
                                  2 * outer, 2 * outer, pen,
                                  QBrush(Qt.NoBrush))
            self.scene.addEllipse(z - half, -half,
                                  2 * half, 2 * half, pen, fill)

        elif st == 'point_source':
            # Small filled circle.
            r = 4.0
            self.scene.addEllipse(z - r, -r, 2 * r, 2 * r,
                                  pen, fill)

        elif st == 'emitter_array':
            # Grid of dots; cap visible count at 7x7 so the layout
            # doesn't drown in micro-dots for large arrays.
            nx = max(1, min(7, src.emitter_nx))
            ny = max(1, min(7, src.emitter_ny))
            pitch_px = max(2.0, src.emitter_pitch_mm * S)
            x0 = z
            y0 = 0.0
            for ix in range(nx):
                for iy in range(ny):
                    cx = x0 + (ix - (nx - 1) / 2) * pitch_px * 0.0
                    cy = y0 + (iy - (ny - 1) / 2) * pitch_px
                    self.scene.addEllipse(cx - 1.5, cy - 1.5,
                                          3, 3, pen, fill)

        else:
            # Unknown future source-type: a question-mark marker.
            self.scene.addLine(z - 4, -4, z + 4, -4, pen)
            self.scene.addLine(z, -4, z, 4, pen)

        # Source description label above the glyph.
        try:
            label_text = src.describe()
        except Exception:
            label_text = st
        lbl = QGraphicsTextItem(label_text)
        lbl.setDefaultTextColor(accent)
        lbl.setFont(QFont('Consolas', 7))
        lbl.setPos(z - 30, -epd_h_px - 24)
        self.scene.addItem(lbl)

        # 3.6.1 (Stage C.1): optional preview rays drawn DOWNSTREAM
        # of the source -- from the source plane to the first
        # optical surface -- to make the propagation direction
        # visually unambiguous.  The original (3.6.1 first cut)
        # drew them upstream of the source which read backward.
        # Default OFF in 3.6.1 hotfix; opt-in via the preference key
        # so users who want this overlay can enable it without
        # confusing ray directions for everyone else.
        if not self.sm.prefs.get('show_source_preview', False):
            return
        # Find the first optical surface's z to bound the rays.
        try:
            elem_z = self.sm.element_z_positions_mm()
        except Exception:
            return
        if len(elem_z) < 2:
            return
        z_first = elem_z[1] * S  # first optical-element vertex
        # Don't draw if the gap is degenerate.
        if z_first - z < 4:
            return
        try:
            wv = self.sm.wavelength_nm
            rc = self._wavelength_to_color(wv)
            ray_pen = QPen(QColor(*rc, 110), 0.8)
        except Exception:
            ray_pen = QPen(QColor(180, 200, 255, 90), 0.8)

        if st == 'plane_wave':
            # Parallel rays from source plane through first surface.
            for ay in np.linspace(-epd_h_px * 0.85,
                                   epd_h_px * 0.85, 5):
                self.scene.addLine(z, ay, z_first, ay, ray_pen)
        elif st in ('gaussian', 'gaussian_aperture'):
            # Diverging from waist at the source plane.
            if st == 'gaussian':
                ap_far = max(8.0,
                              src.beam_diameter_mm / 2 * 1.5 * S)
            else:
                ap_far = max(8.0, src.sigma_mm * 1.5 * S)
            ap_near = ap_far * 0.25
            for k, ay_far in enumerate(
                    np.linspace(-ap_far, ap_far, 5)):
                ay_near = -ap_near + (k * 2 * ap_near / 4)
                self.scene.addLine(
                    z, ay_near, z_first, ay_far, ray_pen)
        elif st == 'top_hat':
            # Hard-edged parallel from the rim of the aperture.
            ap = max(6.0, src.top_hat_diameter_mm / 2 * S)
            for ay in np.linspace(-ap, ap, 5):
                self.scene.addLine(z, ay, z_first, ay, ray_pen)
        elif st == 'fiber_mode':
            # Diverging cone with NA-derived half-angle.
            try:
                na = max(0.001, float(src.fiber_NA))
            except Exception:
                na = 0.14
            ap_far = max(8.0,
                          (z_first - z) * np.tan(np.arcsin(na)))
            for ay in np.linspace(-ap_far, ap_far, 5):
                self.scene.addLine(z, 0.0, z_first, ay, ray_pen)
        elif st == 'point_source':
            # Diverging fan from the source point at z=0.
            for ay in np.linspace(
                    -epd_h_px * 0.6, epd_h_px * 0.6, 7):
                self.scene.addLine(z, 0.0, z_first, ay, ray_pen)
        elif st == 'emitter_array':
            # One short ray per visible emitter (parallel beam).
            nx = max(1, min(5, src.emitter_nx))
            ny = max(1, min(5, src.emitter_ny))
            pitch_px = max(2.0, src.emitter_pitch_mm * S)
            for iy in range(ny):
                cy = (iy - (ny - 1) / 2) * pitch_px
                self.scene.addLine(z, cy, z_first, cy, ray_pen)

    # 3.6.1: bidirectional table <-> layout selection.
    # The layout already emits sm.element_selected on click (handled
    # by ElementTableEditor), but the reverse direction (table click
    # -> layout highlight) was missing.  set_selected_element is the
    # public slot wired in main_window.py to the new
    # element_selected_in_table signal on ElementTableEditor.

    def set_selected_element(self, idx: int):
        """Public slot: set the currently-highlighted element index.

        Negative values clear the highlight.  Repaint is incremental
        (clears and adds the highlight items only); it does not
        trigger a full rebuild.
        """
        try:
            idx = int(idx)
        except Exception:
            idx = -1
        self._selected_elem_idx = idx
        self._redraw_highlight()

    def _redraw_highlight(self):
        """Repaint the selection-highlight overlay.

        Cleared and redrawn every time the selection changes or the
        scene is rebuilt.  Looks up the element's z-zone in
        self._surface_zones (registered during rebuild) and overlays
        a translucent amber rectangle there.
        """
        # Clear prior highlight items.
        for it in self._highlight_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._highlight_items.clear()
        idx = self._selected_elem_idx
        if idx < 0 or not self._surface_zones:
            return
        # Find the zone for this element.  _surface_zones is a list
        # of (z_start, z_end, element_index); pick the matching
        # entry.
        for z_start, z_end, ei in self._surface_zones:
            if ei != idx:
                continue
            S = self.SCALE
            h = max(8.0, self.sm.epd_mm / 2 * S * 1.1)
            pad_z = max(6.0, (z_end - z_start) * 0.15)
            outline_pen = QPen(QColor(255, 200, 80), 2)
            fill = QBrush(QColor(255, 200, 80, 60))
            rect = self.scene.addRect(
                z_start - pad_z, -h,
                (z_end - z_start) + 2 * pad_z, 2 * h,
                outline_pen, fill)
            # Push the highlight to the front so it sits on top of
            # rays / glass fills without obscuring them entirely
            # (alpha=60 keeps everything underneath visible).
            rect.setZValue(100)
            self._highlight_items.append(rect)
            break

    def _draw_rays(self, trace_result):
        """Draw traced rays on the scene.

        3.6.1 hotfix-4: rays are now plotted in the per-element 2D
        world frames returned by
        :meth:`SystemModel.element_frames_2d_mm`.  Each surface
        contributes a ``(z_world, y_world, theta)`` triple; a ray's
        local-frame ``y`` (perpendicular to the local optical axis)
        is converted to world coordinates via the surface's local
        +y direction ``(-sin theta, cos theta)`` so rays follow the
        bent path through mirrors and coord-break tilts.  Falls back
        to a straight-axis layout when no folds are present.

        3.6.1 hotfix-2 (kept) — the world-frame z accounts for the
        source-to-first-surface air gap; previously the entire ray
        fan was squished into the first ~10 mm of the system.
        """
        if trace_result is None:
            return

        S = self.SCALE

        # 3.6.1 hotfix-6: per-traced-surface 2D world frames in
        # trace order (incl. image plane), with mirror_sign tracking
        # so post-mirror surfaces sit on the correct side of the
        # mirror in the world frame.
        surf_frames = self.sm.surface_frames_2d_mm()
        if not surf_frames:
            return
        # Source frame (always frames[0] from element_frames).
        elem_frames = self.sm.element_frames_2d_mm()
        if not elem_frames:
            return

        # Ray colour — preference or wavelength-based.
        if self.sm.prefs.get('ray_use_wavelength', True):
            wv = self.sm.wavelength_nm
            rc = self._wavelength_to_color(wv)
            ray_pen = QPen(QColor(*rc, 120), 0.8)
        else:
            rc_hex = self.sm.prefs.get('ray_color', '#5cb8ff')
            c = QColor(rc_hex)
            c.setAlpha(120)
            ray_pen = QPen(c, 0.8)

        n_rays = trace_result.input_rays.n_rays
        history = trace_result.ray_history
        step = max(1, n_rays // 50)

        # Source frame for the launch point.
        z_src, y_src, t_src = elem_frames[0]
        src_ux, src_uy = -np.sin(t_src), np.cos(t_src)

        def to_world(si, y_local):
            """Map (surface index, local-y mm) → world (z_mm, y_mm)."""
            zw, yw, th = surf_frames[si]
            ux, uy = -np.sin(th), np.cos(th)
            return (zw + y_local * ux, yw + y_local * uy)

        for r in range(0, n_rays, step):
            if not trace_result.input_rays.alive[r]:
                continue

            pts = []
            y_in = trace_result.input_rays.y[r] * 1e3   # m -> mm

            # Source launch point in world coords.
            pts.append((z_src + y_in * src_ux,
                        y_src + y_in * src_uy))

            # Pre-lens segment: end at the first surface.  Use
            # history[0].y (the ray's height at surface 0 after
            # propagation through the air gap) when available, so
            # tilted launches still terminate at the right height.
            if history and len(history) > 0 and history[0].alive[r]:
                pts.append(to_world(0, history[0].y[r] * 1e3))
            else:
                # Ray died before surface 0 — draw a straight pre-lens
                # segment using the launch y projected onto surf 0's
                # local +y, which is exact for an unfolded source.
                pts.append(to_world(0, y_in))

            # Through-system + image plane: history[si] is AFTER
            # surface si.  Each entry's ray.y is the local-frame
            # height at surf_frames[si].
            for si in range(1, len(history)):
                if not history[si].alive[r]:
                    break
                y_local = history[si].y[r] * 1e3   # m -> mm
                if si < len(surf_frames):
                    pts.append(to_world(si, y_local))
                else:
                    # History longer than known frames — extrapolate
                    # 20 mm forward along the last frame's axis so
                    # the ray has somewhere to go.
                    zw, yw, th = surf_frames[-1]
                    pts.append((pts[-1][0] + 20 * np.cos(th),
                                pts[-1][1] + 20 * np.sin(th)))

            # Draw connected segments (scaled to scene pixels).
            for j in range(len(pts) - 1):
                self.scene.addLine(
                    pts[j][0] * S, pts[j][1] * S,
                    pts[j + 1][0] * S, pts[j + 1][1] * S,
                    ray_pen,
                )

    @staticmethod
    def _wavelength_to_color(wv_nm):
        """Convert wavelength (nm) to RGB tuple."""
        wv = wv_nm
        if 380 <= wv <= 440:
            r = (440 - wv) / 60
            g = 0
            b = 1
        elif 440 < wv <= 490:
            r = 0
            g = (wv - 440) / 50
            b = 1
        elif 490 < wv <= 510:
            r = 0
            g = 1
            b = (510 - wv) / 20
        elif 510 < wv <= 580:
            r = (wv - 510) / 70
            g = 1
            b = 0
        elif 580 < wv <= 645:
            r = 1
            g = (645 - wv) / 65
            b = 0
        elif 645 < wv <= 780:
            r = 1
            g = 0
            b = 0
        else:
            # IR / UV — use white-ish
            r = 0.7
            g = 0.7
            b = 0.9
        return (int(r * 255), int(g * 255), int(b * 255))

    def _on_view_click(self, event):
        """Handle click on the 2D layout — find and select the nearest surface."""
        # Convert screen position to scene coordinates
        scene_pos = self.view.mapToScene(event.pos())
        click_z = scene_pos.x()  # z-axis is horizontal

        # Find the nearest surface zone
        best_dist = float('inf')
        best_idx = -1
        for z_start, z_end, surf_idx in self._surface_zones:
            if z_start <= click_z <= z_end:
                # Inside the zone — exact match
                best_idx = surf_idx
                break
            dist = min(abs(click_z - z_start), abs(click_z - z_end))
            if dist < best_dist:
                best_dist = dist
                best_idx = surf_idx

        if best_idx >= 0:
            self.sm.element_selected.emit(best_idx)

        # Still allow drag-to-pan via the original handler
        QGraphicsView.mousePressEvent(self.view, event)

    def wheelEvent(self, event):
        """Forward bubbled-up wheel events to the view zoom handler.

        Most wheel events go directly to the QGraphicsView via
        ``_view_wheel_event``; this fallback only fires when the
        wheel happens over the toolbar / margins.
        """
        self._view_wheel_event(event)

    def _view_wheel_event(self, event):
        """3.6.1 hotfix-6: always zoom on wheel; never let the
        QGraphicsView's default handler route to scrollbars.

        Without this override, once the scene exceeds the viewport
        (zoomed in), the wheel scrolls instead of zooming.
        """
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self.view.scale(factor, factor)
        event.accept()

    def _reset_view(self):
        """Fit the entire scene with a small margin (toolbar button)."""
        if self.scene.items():
            self.view.fitInView(
                self.scene.sceneRect().adjusted(-40, -40, 40, 40),
                Qt.KeepAspectRatio,
            )

    def minimumSizeHint(self):
        """3.6.1 hotfix-6: report a tiny minimum so the QDockWidget
        will let the user drag the layout pane down to almost
        nothing.  The inherited implementation walks the
        QGraphicsView size hint and reserves several hundred px,
        which is what was preventing the dock from collapsing.
        """
        from PySide6.QtCore import QSize
        return QSize(40, 40)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(400, 200)
