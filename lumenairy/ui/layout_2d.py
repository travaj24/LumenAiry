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
from ..lenses import surface_sag_biconic


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
            if elem.elem_type in ('Source', 'Detector'):
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

        # ── Fit view ──
        self.view.fitInView(
            self.scene.sceneRect().adjusted(-40, -40, 40, 40),
            Qt.KeepAspectRatio,
        )

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

    def _draw_rays(self, trace_result):
        """Draw traced rays on the scene."""
        if trace_result is None:
            return

        S = self.SCALE

        # Compute z positions from the flattened trace surfaces
        trace_surfs = self.sm.build_trace_surfaces()
        z_positions = [0.0]
        for ts in trace_surfs:
            z_positions.append(z_positions[-1] + ts.thickness * 1e3)  # m -> mm

        # Ray color — use preference or wavelength-based
        if self.sm.prefs.get('ray_use_wavelength', True):
            wv = self.sm.wavelength_nm
            rc = self._wavelength_to_color(wv)
            ray_pen = QPen(QColor(*rc, 120), 0.8)
        else:
            rc_hex = self.sm.prefs.get('ray_color', '#5cb8ff')
            c = QColor(rc_hex)
            c.setAlpha(120)
            ray_pen = QPen(c, 0.8)

        # For each ray, draw from surface to surface
        n_rays = trace_result.input_rays.n_rays
        history = trace_result.ray_history

        # Subsample rays for display (limit to ~50 for clarity)
        step = max(1, n_rays // 50)

        for r in range(0, n_rays, step):
            if not trace_result.input_rays.alive[r]:
                continue

            # Start: input ray at z=z_positions[1] (first optical surface)
            # The input rays are at z=0 of the first surface
            pts = []

            # Entry point: y from input ray
            y_in = trace_result.input_rays.y[r] * 1e3  # m → mm
            pts.append((z_positions[1] * S, y_in * S))

            # Each surface in history
            for si, rb in enumerate(history):
                if not rb.alive[r]:
                    break
                y = rb.y[r] * 1e3  # m → mm
                # Surface index in the table: si corresponds to
                # optical surface si, which is table row si+1
                if si + 1 < len(z_positions):
                    z = z_positions[si + 1] * S
                else:
                    # Image plane or beyond
                    z = pts[-1][0] + 20 * S
                pts.append((z, y * S))

            # Draw the ray as connected line segments
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
