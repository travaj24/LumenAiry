"""
Live parameter slider dock — Quadoa-inspired real-time exploration.

Generates a slider for each optimization variable.  Dragging a slider
updates the surface parameter and re-traces in real time, giving
immediate feedback on how each variable affects the spot size.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QScrollArea, QGroupBox, QComboBox,
)
from PySide6.QtGui import QFont

import numpy as np

from .model import SystemModel


class ParameterSlider(QWidget):
    """A single parameter slider with label, value readout, and a
    per-variable range selector (\u00b15 / 10 / 20 / 50 %).

    ``field`` drives the range mode: conic swings absolute \u00b12, distance
    and thickness hold a zero floor, everything else is fractional.
    """

    def __init__(self, label, value, vmin, vmax, callback, parent=None,
                 field='radius'):
        super().__init__(parent)
        self._callback = callback
        self._vmin = vmin
        self._vmax = vmax
        self._center = value
        self._field = field

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self.label = QLabel(label)
        self.label.setFixedWidth(120)
        self.label.setStyleSheet("color: #7a94b8; font-size: 11px;")
        self.label.setToolTip(f'Variable: {label}\nCenter: {value:.6g}')
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self._set_slider_from_value(value)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider, stretch=1)

        self.readout = QLabel(f'{value:.4g}')
        self.readout.setFixedWidth(80)
        self.readout.setAlignment(Qt.AlignRight)
        self.readout.setStyleSheet(
            "color: #5cb8ff; font-size: 11px; font-family: Consolas;")
        layout.addWidget(self.readout)

        # Per-slider range selector -- much more useful than a fixed
        # \u00b150% that's either too wide or too narrow for every variable.
        self.range_combo = QComboBox()
        if field == 'conic':
            self.range_combo.addItems(['\u00b10.2', '\u00b10.5', '\u00b11', '\u00b12'])
            self._range_map = {'\u00b10.2': 0.2, '\u00b10.5': 0.5,
                               '\u00b11': 1.0, '\u00b12': 2.0}
            self.range_combo.setCurrentText('\u00b12')
        else:
            self.range_combo.addItems(['\u00b15%', '\u00b110%', '\u00b120%',
                                        '\u00b150%'])
            self._range_map = {'\u00b15%': 0.05, '\u00b110%': 0.10,
                               '\u00b120%': 0.20, '\u00b150%': 0.50}
            self.range_combo.setCurrentText('\u00b150%')
        self.range_combo.setFixedWidth(70)
        self.range_combo.setToolTip(
            'Range the slider spans around the starting value.\n'
            'Narrower = finer control for a near-optimum variable.')
        self.range_combo.currentTextChanged.connect(self._on_range_changed)
        layout.addWidget(self.range_combo)

    def _set_slider_from_value(self, value):
        if self._vmax == self._vmin:
            self.slider.setValue(500)
            return
        frac = (value - self._vmin) / (self._vmax - self._vmin)
        self.slider.setValue(int(np.clip(frac, 0.0, 1.0) * 1000))

    def _value_from_slider(self):
        frac = self.slider.value() / 1000.0
        return self._vmin + frac * (self._vmax - self._vmin)

    def _on_slider(self):
        val = self._value_from_slider()
        self.readout.setText(f'{val:.4g}')
        self._callback(val)

    def _on_range_changed(self, text):
        """User picked a different \u00b1 span; recompute vmin/vmax around the
        current center and preserve the current value position."""
        span = self._range_map.get(text)
        if span is None:
            return
        # Snapshot current value before we redo the mapping
        current = self._value_from_slider()
        if self._field == 'conic':
            self._vmin = self._center - span
            self._vmax = self._center + span
        elif self._field in ('thickness', 'distance'):
            self._vmin = max(0.0, self._center * (1 - span))
            self._vmax = (self._center * (1 + span)
                          if self._center > 0 else max(span, 1e-3))
        else:
            # Radius/semi-diam: handle negative values too.
            lo = self._center * (1 - span) if self._center > 0 else self._center * (1 + span)
            hi = self._center * (1 + span) if self._center > 0 else self._center * (1 - span)
            self._vmin, self._vmax = min(lo, hi), max(lo, hi)
        self.slider.blockSignals(True)
        self._set_slider_from_value(current)
        self.slider.blockSignals(False)

    def set_value(self, value):
        """Update without triggering callback."""
        self.slider.blockSignals(True)
        self._set_slider_from_value(value)
        self.readout.setText(f'{value:.4g}')
        self.slider.blockSignals(False)


class SliderDock(QWidget):
    """Live parameter exploration panel."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._sliders = []
        # Debounce merit-evaluation while dragging so the trace doesn't
        # run on every pixel of motion (cf. Tier-4 audit item 4.2).
        self._merit_timer = QTimer(self)
        self._merit_timer.setSingleShot(True)
        self._merit_timer.setInterval(80)
        self._merit_timer.timeout.connect(self._update_merit)

        # v5.4.2 (audit B-P2 belt-and-suspenders): inner 40 ms
        # debounce on the system_changed emit itself.  The
        # main_window auto-retrace timer (200 ms) already coalesces
        # the downstream retrace, but the inner debounce defends
        # against the case where the auto-retrace path is ever
        # disabled or rewired.  Net effect: slider drag at 60+ Hz
        # produces ~25 system_changed emits/sec instead of 60+.
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(40)
        self._emit_timer.timeout.connect(self._emit_system_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        btn_generate = QPushButton('Generate Sliders')
        btn_generate.setToolTip(
            'Create a slider per optimization variable.\n'
            'Define variables first in the Optimizer dock.')
        btn_generate.clicked.connect(self._generate_sliders)
        toolbar.addWidget(btn_generate)

        btn_reset = QPushButton('Reset to Original')
        btn_reset.setToolTip(
            'Restore every slider to the value it had when Generate '
            'was clicked.')
        btn_reset.clicked.connect(self._reset)
        toolbar.addWidget(btn_reset)

        btn_snapshot = QPushButton('Snapshot')
        btn_snapshot.setToolTip(
            'Save the current parameter values as a named snapshot '
            '(shows up in the Snapshots dock).')
        btn_snapshot.clicked.connect(self._save_snapshot)
        toolbar.addWidget(btn_snapshot)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Scroll area for sliders
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(2)
        self.scroll.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll, stretch=1)

        # Live metrics read-out -- more useful than just the merit value.
        self.metrics_label = QLabel('')
        self.metrics_label.setStyleSheet(
            "color: #ffd166; font-size: 12px; font-family: Consolas; "
            "padding: 4px;")
        self.metrics_label.setToolTip(
            'Merit = currently-selected optimizer objective.\n'
            'EFL / BFL update live as you drag sliders.')
        layout.addWidget(self.metrics_label)

        self._original_values = None

    def _generate_sliders(self):
        """Create sliders for all current optimization variables."""
        # Clear existing
        for s in self._sliders:
            s.setParent(None)
            s.deleteLater()
        self._sliders = []

        if not self.sm.opt_variables:
            empty = QWidget()
            empty_layout = QVBoxLayout(empty)
            empty_layout.setAlignment(Qt.AlignCenter)
            lbl = QLabel(
                'No optimization variables defined.\n\n'
                'Sliders let you drag any free parameter live while '
                'ray-tracing updates in real time.')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                "color: #7a94b8; font-size: 12px; padding: 20px;")
            empty_layout.addWidget(lbl)
            btn = QPushButton('Define variables...')
            btn.setToolTip(
                'Open the Optimizer dock to pick which radii, '
                'thicknesses, conics etc. become sliders.')
            btn.clicked.connect(self._open_variable_picker)
            empty_layout.addWidget(btn, alignment=Qt.AlignCenter)
            self.scroll_layout.addWidget(empty)
            self._sliders.append(empty)
            return

        self._original_values = self.sm.get_variable_values().copy()

        for i, var in enumerate(self.sm.opt_variables):
            # The optimizer dock stores (elem_idx, surf_idx, field) 3-tuples;
            # older serialisations stored 2-tuples.  Accept both.
            if len(var) == 3:
                elem_idx, surf_idx, field = var
            else:
                elem_idx, surf_idx = var
                field = 'radius'

            value = self._original_values[i]
            if field == 'distance':
                label = f'E{elem_idx} distance'
            else:
                label = f'E{elem_idx}.S{surf_idx} {field}'

            # Set per-variable range based on the parameter meaning, not
            # a brittle column-index.  Conic is absolute \u00b12; thickness
            # and distance get a floor at zero; everything else \u00b150%.
            if field == 'conic':
                vmin, vmax = value - 2.0, value + 2.0
            elif field in ('thickness', 'distance'):
                vmin = max(0.0, value * 0.5)
                vmax = value * 1.5 if value > 0 else 10.0
            else:
                vmin = value * 0.5 if value > 0 else value * 1.5
                vmax = value * 1.5 if value > 0 else value * 0.5

            def make_callback(idx):
                def cb(val):
                    self._on_slider_change(idx, val)
                return cb

            slider = ParameterSlider(
                label, value, vmin, vmax, make_callback(i), field=field)
            self.scroll_layout.addWidget(slider)
            self._sliders.append(slider)

        self.scroll_layout.addStretch()
        self._update_merit()

    def attach_variable(self, elem_idx, surf_idx, field):
        """3.6.1 (Stage C.2): public slot called from main_window when
        the user picks "Attach slider to this parameter…" on the
        prescription editor.  By the time we get here, the variable
        has already been appended to ``sm.opt_variables`` by the
        SystemModel helper, so ``_generate_sliders`` will pick it up.
        Locate the matching slider in the regenerated list and pulse
        it amber for ~3 seconds so the user can see which one is new.
        """
        # Force a regeneration so the new variable's slider exists.
        self._generate_sliders()
        target = (int(elem_idx), int(surf_idx), str(field))
        # Find the matching slider widget by aligning indices into
        # opt_variables with self._sliders (skipping the empty-state
        # placeholder, which is also stored in self._sliders when
        # the list was empty).
        match_slider = None
        for var, slider in zip(self.sm.opt_variables, self._sliders):
            if isinstance(slider, ParameterSlider):
                triple = tuple(var) if len(var) == 3 \
                    else (var[0], var[1], 'radius')
                if triple == target:
                    match_slider = slider
                    break
        if match_slider is None:
            return
        # Pulse: temporarily set a bright stylesheet, restore after
        # 3 seconds.  Uses QTimer so the UI thread is not blocked.
        try:
            prior = match_slider.styleSheet() or ''
            match_slider.setStyleSheet(
                prior + '\n* { background: #4a3a16; }')
            from PySide6.QtCore import QTimer
            QTimer.singleShot(
                3000,
                lambda: match_slider.setStyleSheet(prior))
        except Exception:
            pass

    def _on_slider_change(self, var_index, new_value):
        """Called when any slider is moved.

        v5.4.2 (audit B-P2 belt-and-suspenders): defer the
        ``system_changed`` emission via a 40 ms debounce timer so a
        drag at 60+ Hz coalesces into ~25 emits/sec instead of one
        emit per pixel.  The main_window's auto-retrace timer
        (200 ms) ALSO debounces this path, so retraces fire only
        on idle in both cases -- but the inner debounce defends
        against the case where the main_window debounce is ever
        bypassed (e.g., a future caller wiring ``system_changed``
        to an undebounced consumer).
        """
        values = self.sm.get_variable_values()
        values[var_index] = new_value
        self.sm.set_variable_values(values)
        self.sm._invalidate()
        # v5.4.2: emit system_changed via 40 ms inner debounce
        # instead of synchronously.  Coalesces 60 Hz slider drag
        # into ~25 Hz emit cadence.
        self._emit_timer.start()
        # Debounce the merit + EFL/BFL update so a drag at 60 Hz doesn't
        # trigger 60 ray-traces per second on a large system.
        self._merit_timer.start()

    def _emit_system_changed(self):
        """v5.4.2: deferred system_changed emit slot."""
        self.sm.system_changed.emit()

    def _update_merit(self):
        if not self.sm.opt_variables:
            self.metrics_label.setText('')
            return
        parts = []
        try:
            values = self.sm.get_variable_values()
            merit = self.sm.merit_function(values)
            parts.append(f'Merit: {merit*1e6:.3f} \u00b5m')
        except Exception as e:
            try:
                from .diagnostics import diag
                diag.report('slider-merit', e)
            except Exception:
                pass
            parts.append('Merit: error')
        try:
            _, efl, bfl = self.sm.get_abcd()
            if np.isfinite(efl):
                parts.append(f'EFL: {efl*1e3:.3f} mm')
            if np.isfinite(bfl):
                parts.append(f'BFL: {bfl*1e3:.3f} mm')
            if np.isfinite(efl) and self.sm.epd_m > 0:
                parts.append(f'f/#: {abs(efl)/self.sm.epd_m:.2f}')
        except Exception:
            pass
        self.metrics_label.setText('   |   '.join(parts))

    def _open_variable_picker(self):
        """Bridge the empty-state CTA into the Optimizer dock's picker
        dialog so users don't have to hunt for the right tab."""
        try:
            from .optimizer_dock import _VariableGridDialog
            dlg = _VariableGridDialog(self.sm, self)
            from PySide6.QtWidgets import QDialog
            if dlg.exec() == QDialog.Accepted:
                self.sm.opt_variables = dlg.checked_variables()
                self.sm.system_changed.emit()
                self._generate_sliders()
        except Exception as e:
            try:
                from .diagnostics import diag
                diag.report('slider-cta', e)
            except Exception:
                pass

    def _save_snapshot(self):
        """Save a named snapshot of the current system to the model."""
        from PySide6.QtWidgets import QInputDialog
        n = len(self.sm.snapshots)
        name, ok = QInputDialog.getText(
            self, 'Save Snapshot', 'Name for this configuration:',
            text=f'Snapshot {n + 1}')
        if ok and name:
            self.sm.save_snapshot(name)

    def _reset(self):
        """Reset all variables to their original values."""
        if self._original_values is not None and len(self._original_values) > 0:
            self.sm.set_variable_values(self._original_values)
            self.sm._invalidate()
            self.sm.system_changed.emit()
            # Update slider positions
            for i, slider in enumerate(self._sliders):
                if isinstance(slider, ParameterSlider) and i < len(self._original_values):
                    slider.set_value(self._original_values[i])
            self._update_merit()
