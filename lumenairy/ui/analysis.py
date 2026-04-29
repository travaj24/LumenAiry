"""
Analysis dock widgets — spot diagram, ray fan, system summary.

Each analysis is a QWidget designed to live inside a QDockWidget in
the main window.  They connect to SystemModel.trace_ready and update
automatically when the prescription changes.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy, QTextEdit,
)
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush

import numpy as np

from .model import SystemModel
from ..raytrace import spot_rms, spot_geo_radius, TraceResult


# ────────────────────────────────────────────────────────────────────────
# Spot Diagram Widget
# ────────────────────────────────────────────────────────────────────────

class SpotDiagramWidget(QWidget):
    """Interactive spot diagram that redraws when trace_ready fires."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._result = None
        self._spot_data = None
        self._rms = 0
        self._geo = 0
        self._airy = 0

        self.setMinimumSize(250, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.sm.trace_ready.connect(self._on_trace)

    def _on_trace(self, result):
        if result is None:
            self._result = None
            self._spot_data = None
            self.update()
            return

        self._result = result
        r = result.image_rays
        alive = r.alive

        if not np.any(alive):
            self._spot_data = None
            self.update()
            return

        cx = np.mean(r.x[alive])
        cy = np.mean(r.y[alive])
        self._spot_data = {
            'x': (r.x[alive] - cx) * 1e6,  # µm
            'y': (r.y[alive] - cy) * 1e6,
            'n_alive': int(np.sum(alive)),
            'n_total': r.n_rays,
        }

        self._rms, _ = spot_rms(result)
        self._geo = spot_geo_radius(result)

        # Airy radius at the focal plane: 1.22 * lambda * f / D, where D
        # is the entrance-pupil diameter (= 2 * semi_diameter of S1) and
        # f is the system EFL.  Without f the formula gives the angular
        # Airy radius (radians), not a length, and the displayed micron
        # value would be meaningless.
        sd = result.surfaces[0].semi_diameter if result.surfaces else np.inf
        try:
            _, efl, _ = self.sm.get_abcd()
        except Exception:
            efl = np.nan
        if np.isfinite(sd) and sd > 0 and np.isfinite(efl) and efl != 0:
            self._airy = 1.22 * result.wavelength * abs(efl) / (2 * sd)
        else:
            self._airy = 0

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        W = self.width()
        H = self.height()

        # Background
        painter.fillRect(0, 0, W, H, QColor(10, 12, 18))

        if self._spot_data is None:
            painter.setPen(QColor(120, 140, 170))
            painter.setFont(QFont('Consolas', 11))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             'No trace data.\nEdit surfaces to trace.')
            return

        # Determine scale
        x = self._spot_data['x']
        y = self._spot_data['y']
        max_extent = max(np.max(np.abs(x)), np.max(np.abs(y)), 1)
        # Add margin
        max_extent *= 1.3

        plot_size = min(W, H) - 60
        cx = W / 2
        cy = H / 2 - 10
        scale = plot_size / (2 * max_extent)

        # Grid
        painter.setPen(QPen(QColor(30, 40, 55), 1))
        painter.drawLine(int(cx - plot_size / 2), int(cy),
                         int(cx + plot_size / 2), int(cy))
        painter.drawLine(int(cx), int(cy - plot_size / 2),
                         int(cx), int(cy + plot_size / 2))

        # Airy disc
        if self._airy > 0:
            airy_um = self._airy * 1e6
            airy_r = airy_um * scale
            if airy_r > 2 and airy_r < plot_size:
                painter.setPen(QPen(QColor(200, 60, 60), 1, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(
                    int(cx - airy_r), int(cy - airy_r),
                    int(2 * airy_r), int(2 * airy_r),
                )

        # Spot points
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(92, 184, 255, 180)))
        for i in range(len(x)):
            px = cx + x[i] * scale
            py = cy - y[i] * scale  # flip y
            painter.drawEllipse(int(px - 2), int(py - 2), 4, 4)

        # Labels
        painter.setPen(QColor(180, 200, 230))
        painter.setFont(QFont('Consolas', 10))

        rms_str = f'RMS: {self._rms * 1e6:.2f} µm'
        geo_str = f'GEO: {self._geo * 1e6:.2f} µm'
        airy_str = f'Airy: {self._airy * 1e6:.2f} µm' if self._airy > 0 else ''
        rays_str = f'{self._spot_data["n_alive"]}/{self._spot_data["n_total"]} rays'

        y_text = H - 45
        painter.drawText(10, y_text, rms_str)
        painter.drawText(10, y_text + 15, geo_str)
        painter.drawText(150, y_text, airy_str)
        painter.drawText(150, y_text + 15, rays_str)

        # Title
        painter.setPen(QColor(92, 184, 255))
        painter.setFont(QFont('Consolas', 11, QFont.Bold))
        painter.drawText(10, 20, 'Spot Diagram')

        # Scale bar
        painter.setPen(QColor(120, 140, 170))
        painter.setFont(QFont('Consolas', 8))
        bar_um = self._nice_scale(max_extent)
        bar_px = bar_um * scale
        bx = cx + plot_size / 2 - bar_px - 10
        by = cy + plot_size / 2 + 15
        painter.drawLine(int(bx), int(by), int(bx + bar_px), int(by))
        painter.drawText(int(bx), int(by + 12), f'{bar_um:.0f} µm')

        painter.end()

    @staticmethod
    def _nice_scale(max_val):
        """Pick a nice round number for the scale bar."""
        order = 10 ** np.floor(np.log10(max(max_val, 1e-6)))
        if max_val / order < 2:
            return order
        elif max_val / order < 5:
            return 2 * order
        else:
            return 5 * order


# ────────────────────────────────────────────────────────────────────────
# System Summary Widget
# ────────────────────────────────────────────────────────────────────────

class SystemSummaryWidget(QWidget):
    """Text summary of system parameters, ABCD matrix, and trace results."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont('Consolas', 11))
        self.text.setStyleSheet(
            "QTextEdit { background: #0a0c10; color: #dde8f8; "
            "border: none; }")
        layout.addWidget(self.text)

        self.sm.system_changed.connect(self._update)
        self.sm.trace_ready.connect(self._update_trace)

        self._update()

    def _update(self):
        lines = []
        lines.append('═══ System Parameters ═══')
        lines.append(f'  Wavelength:  {self.sm.wavelength_nm:.1f} nm')
        wls = list(self.sm.wavelengths_nm or [])
        if len(wls) > 1:
            lines.append(f'  Wavelengths: {", ".join(f"{w:.1f}" for w in wls)} nm')
        lines.append(f'  EPD:         {self.sm.epd_mm:.2f} mm')
        fields = list(self.sm.field_angles_deg or [])
        if fields:
            field_max = max(abs(f) for f in fields)
            lines.append(f'  Field angles: {", ".join(f"{f:.2f}" for f in fields)}°')
            lines.append(f'  FOV (full):  {2 * field_max:.2f}°')
        lines.append(f'  Surfaces:    {self.sm.num_elements - 2}')
        lines.append('')

        try:
            abcd, efl, bfl = self.sm.get_abcd()
            lines.append('═══ Paraxial Data ═══')
            efl_mm = efl * 1e3 if np.isfinite(efl) else np.inf
            bfl_mm = bfl * 1e3 if np.isfinite(bfl) else np.inf
            lines.append(f'  EFL:   {efl_mm:.4f} mm' if np.isfinite(efl_mm)
                         else '  EFL:   ∞')
            lines.append(f'  BFL:   {bfl_mm:.4f} mm' if np.isfinite(bfl_mm)
                         else '  BFL:   ∞')

            if np.isfinite(efl) and self.sm.epd_m > 0:
                fnum = abs(efl) / self.sm.epd_m
                na = 1 / (2 * fnum)
                lines.append(f'  f/#:   {fnum:.2f}')
                lines.append(f'  NA:    {na:.4f}')

                # Working f/# for an object at infinity equals the
                # image-space f/#, which equals f/# above only when m=0.
                # For finite-conjugate systems the working f/# differs:
                #   working f/# = (1 + |m|) * f/#   where m = -BFL / EFL
                # is the lateral magnification of the system imaging
                # the back-focal-plane source.  We don't have an
                # explicit object distance; assume infinity and report
                # working f/# == f/# in that limit.
                lines.append(f'  Working f/#: {fnum:.2f}  (object at inf.)')

                # Image-space NA (matches NA above for paraxial system)
                # Reported separately for clarity in mixed contexts.
                lines.append(f'  Image NA:    {na:.4f}')

                # Diffraction-limited spot (Airy radius)
                airy_um = 1.22 * self.sm.wavelength_m * abs(efl) \
                          / self.sm.epd_m * 1e6
                lines.append(f'  Airy radius: {airy_um:.2f} µm')

            # Principal planes (Welford convention):
            #   H  (front pp)  = (D - 1) / C  measured from front vertex
            #   H' (rear pp)   = (1 - A) / C  measured from rear vertex
            C = abcd[1, 0]
            if abs(C) > 1e-30:
                H_mm = ((abcd[1, 1] - 1.0) / C) * 1e3
                Hp_mm = ((1.0 - abcd[0, 0]) / C) * 1e3
                lines.append(f'  Front PP (H):   {H_mm:+.4f} mm '
                             f'(from front vertex)')
                lines.append(f'  Rear  PP (H\'):  {Hp_mm:+.4f} mm '
                             f'(from rear vertex)')

            # Stop position
            try:
                from ..raytrace import find_stop
                stop_idx = find_stop(self.sm.build_trace_surfaces())
                if stop_idx is not None:
                    lines.append(f'  Stop surface:  S{stop_idx + 1}')
            except Exception:
                pass

            lines.append('')
            lines.append('═══ ABCD Matrix ═══')
            lines.append(f'  A = {abcd[0, 0]:+.6f}   '
                         f'B = {abcd[0, 1] * 1e3:+.6f} mm')
            lines.append(f'  C = {abcd[1, 0] / 1e3:+.6f} /mm   '
                         f'D = {abcd[1, 1]:+.6f}')
            det = abcd[0, 0] * abcd[1, 1] - abcd[0, 1] * abcd[1, 0]
            lines.append(f'  det = {det:.6f}')

            # Pupils (best-effort; needs surfaces and a stop to compute)
            try:
                from ..raytrace import compute_pupils
                pupils = compute_pupils(
                    self.sm.build_trace_surfaces(), self.sm.wavelength_m)
                if pupils is not None:
                    lines.append('')
                    lines.append('═══ Pupils (paraxial) ═══')
                    if pupils.entrance_pupil_z is not None \
                            and np.isfinite(pupils.entrance_pupil_z):
                        ep_mm = pupils.entrance_pupil_z * 1e3
                        lines.append(f'  EP at z = {ep_mm:+.4f} mm '
                                     f'(from S0)')
                    if pupils.exit_pupil_z is not None \
                            and np.isfinite(pupils.exit_pupil_z):
                        xp_mm = pupils.exit_pupil_z * 1e3
                        lines.append(f'  XP at z = {xp_mm:+.4f} mm '
                                     f'(from last vertex)')
                    if pupils.entrance_pupil_radius is not None:
                        ep_r_mm = pupils.entrance_pupil_radius * 1e3
                        lines.append(f'  EP radius: {ep_r_mm:.4f} mm')
                    if pupils.exit_pupil_radius is not None:
                        xp_r_mm = pupils.exit_pupil_radius * 1e3
                        lines.append(f'  XP radius: {xp_r_mm:.4f} mm')
            except Exception:
                pass

        except Exception as e:
            lines.append(f'  Error computing ABCD: {e}')

        self.text.setPlainText('\n'.join(lines))

    def _update_trace(self, result):
        if result is None:
            return

        current = self.text.toPlainText()
        lines = [current, '']
        lines.append('═══ Ray Trace ═══')

        rms, (cx, cy) = spot_rms(result)
        geo = spot_geo_radius(result)
        n_alive = int(np.sum(result.image_rays.alive))
        n_total = result.image_rays.n_rays
        vignetting = 100 * (1 - n_alive / n_total) if n_total > 0 else 0

        lines.append(f'  Rays:      {n_alive}/{n_total} '
                     f'({vignetting:.1f}% vignetted)')
        lines.append(f'  Centroid:  ({cx * 1e6:.2f}, {cy * 1e6:.2f}) µm')
        lines.append(f'  RMS spot:  {rms * 1e6:.2f} µm')
        lines.append(f'  GEO radius: {geo * 1e6:.2f} µm')

        # Airy disc at the focal plane: 1.22 * lambda * f / D.
        sd = result.surfaces[0].semi_diameter if result.surfaces else np.inf
        try:
            _, efl, _ = self.sm.get_abcd()
        except Exception:
            efl = np.nan
        if np.isfinite(sd) and sd > 0 and np.isfinite(efl) and efl != 0:
            airy = 1.22 * result.wavelength * abs(efl) / (2 * sd)
            lines.append(f'  Airy radius: {airy * 1e6:.2f} µm')
            if rms > 0 and airy > 0:
                lines.append(f'  Spot/Airy: {rms / airy:.2f}')

        self.text.setPlainText('\n'.join(lines))
