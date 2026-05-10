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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scene = QGraphicsScene()
        bg = self.sm.prefs.get('bg_2d', '#050709')
        self.scene.setBackgroundBrush(QBrush(QColor(bg)))

        self.view = QGraphicsView(self.scene)
        from PySide6.QtGui import QPainter as _P
        self.view.setRenderHint(_P.RenderHint.Antialiasing, True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

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

        z_positions = self.sm.element_z_positions_mm()

        # ── Draw optical axis ──
        z_min = -20
        z_max = max(z_positions) + 20
        axis_pen = QPen(QColor(40, 50, 65), 1, Qt.DashLine)
        self.scene.addLine(z_min * S, 0, z_max * S, 0, axis_pen)

        # ── Draw each element ──
        for ei, elem in enumerate(elements):
            # 3.6.1: render the source as a visible glyph instead of
            # skipping it.  Detector is still skipped here because the
            # image-plane line below covers it.
            if elem.elem_type == 'Source':
                src = getattr(elem, 'source', None) or self.sm.source
                z_src = z_positions[ei] * S
                self._draw_source(z_src, src)
                # Register a click zone at the source so users can
                # click it in the layout and have the table jump to
                # element 0.
                zone_w = max(8, self.sm.epd_mm / 2 * S * 0.3)
                self._surface_zones.append(
                    (z_src - zone_w, z_src + zone_w, ei))
                continue
            if elem.elem_type == 'Detector':
                continue

            z_front = z_positions[ei] * S

            for si, srow in enumerate(elem.surfaces):
                # z of this surface within the element
                internal_offset = sum(s.thickness for s in elem.surfaces[:si])
                z = z_front + internal_offset * S
                sd = srow.semi_diameter if np.isfinite(srow.semi_diameter) else self.sm.epd_mm / 2
                h = sd * S

                if srow.surf_type == 'Mirror' or elem.elem_type == 'Mirror':
                    self._draw_mirror(z, h, srow)
                else:
                    self._draw_surface(z, h, srow, si, elem.surfaces)

            # Register clickable zone for this element
            sd = elem.surfaces[0].semi_diameter if elem.surfaces else self.sm.epd_mm / 2
            if not np.isfinite(sd):
                sd = self.sm.epd_mm / 2
            h = sd * S
            zone_width = max(10, h * 0.3)
            self._surface_zones.append((z_front - zone_width, z_front + zone_width, ei))

        # ── Draw image plane (detector) ──
        z_ima = z_positions[-1] * S
        ima_pen = QPen(QColor(180, 100, 100), 1.5, Qt.DashDotLine)
        sd_ima = self.sm.epd_mm / 2
        self.scene.addLine(z_ima, -sd_ima * S, z_ima, sd_ima * S, ima_pen)

        # ── Label ──
        label = QGraphicsTextItem('Image')
        label.setDefaultTextColor(QColor(200, 130, 130))
        label.setFont(QFont('Consolas', 8))
        label.setPos(z_ima - 10, self.sm.epd_mm / 2 * S + 5)
        self.scene.addItem(label)

        # 3.6.1 hotfix: register a click zone for the detector so it
        # is selectable from the layout (parity with the other
        # elements + the new source glyph).
        det_idx = len(elements) - 1
        if det_idx >= 0 and elements[det_idx].elem_type == 'Detector':
            zone_w = max(8.0, sd_ima * S * 0.3)
            self._surface_zones.append(
                (z_ima - zone_w, z_ima + zone_w, det_idx))

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

        Rays are plotted in world-frame z (mm), computed from the
        per-element positions returned by ``element_z_positions_mm``
        plus each surface's internal thickness within its element.
        Bug fix in 3.6.1 hotfix-2: previously this method built a
        flat cumulative-thickness ``z_positions`` from
        ``build_trace_surfaces``, which only sums *in-glass*
        thicknesses and ignores the source-to-first-surface air gap
        entirely.  That made the entire ray fan render at z ~ 0-10
        mm instead of spanning the actual ~150 mm system length.
        """
        if trace_result is None:
            return

        S = self.SCALE

        # World-frame z (mm) of every traced surface.  Walk the
        # element list in order, skipping the Source (z=0, no
        # surfaces) and the Detector (handled separately as the
        # image plane below); every other element contributes one
        # entry per internal surface, offset from the element's
        # front vertex by the cumulative in-glass thickness.
        elem_z = self.sm.element_z_positions_mm()
        surf_world_z = []
        for ei, elem in enumerate(self.sm.elements):
            if elem.elem_type in ('Source', 'Detector'):
                continue
            z0 = elem_z[ei]
            cum_t = 0.0
            # elem.surfaces[i].thickness is already in mm (GUI
            # convention); do NOT multiply by 1e3 -- doing so blew
            # the lens out by 1000x in the first cut of this fix.
            for srow in elem.surfaces:
                surf_world_z.append(z0 + cum_t)
                cum_t += srow.thickness
        # Image-plane / detector z follows the last optical surface.
        # The trace's history may include the image plane as an
        # extra entry beyond the optical surfaces; we tack on the
        # detector's world z so the final segment lands correctly.
        if elem_z and self.sm.elements \
                and self.sm.elements[-1].elem_type == 'Detector':
            surf_world_z.append(elem_z[-1])

        if not surf_world_z:
            return

        # Ray color — preference or wavelength-based.
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

        # Source plane = z = 0 in world coordinates (matches
        # SystemModel.element_z_positions_mm convention).
        z_source = 0.0
        z_first_surf = surf_world_z[0]

        for r in range(0, n_rays, step):
            if not trace_result.input_rays.alive[r]:
                continue

            pts = []
            y_in = trace_result.input_rays.y[r] * 1e3   # m -> mm

            # Pre-lens segment: from source plane to the first
            # optical surface, at the input ray's launch y.  For a
            # plane wave this gives a horizontal line; for a point
            # source it gives the divergent fan we expect.
            pts.append((z_source * S, y_in * S))
            pts.append((z_first_surf * S, y_in * S))

            # Through-system segments: history[si] is the ray bundle
            # AFTER surface si, so its y is the height at
            # surf_world_z[si] (refraction is in-place; z doesn't
            # change at the refraction event itself, only between
            # surfaces).
            for si, rb in enumerate(history):
                if not rb.alive[r]:
                    break
                y = rb.y[r] * 1e3   # m -> mm
                if si < len(surf_world_z):
                    z = surf_world_z[si] * S
                else:
                    # Beyond the last known surface -- extrapolate
                    # 20 mm forward so the ray has somewhere to go.
                    z = pts[-1][0] + 20 * S
                pts.append((z, y * S))

            # Draw connected segments.
            for j in range(len(pts) - 1):
                self.scene.addLine(
                    pts[j][0], pts[j][1],
                    pts[j + 1][0], pts[j + 1][1],
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
        """Zoom with scroll wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self.view.scale(factor, factor)
