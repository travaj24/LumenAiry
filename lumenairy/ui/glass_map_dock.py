"""
Glass map dock — interactive Abbe diagram (nd vs Vd).

Displays all glasses in the registry as a scatter plot. Clicking a
glass selects it for use in the surface table. Glasses currently used
in the system are highlighted.

Inspired by CODE V's glass map display.

Author: Andrew Traverso
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox,
)
from PySide6.QtCore import Qt, Signal

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel
from ..glass import GLASS_REGISTRY, get_glass_index


class GlassMapDock(QWidget):
    """Interactive Abbe diagram with glass selection."""

    glass_selected = Signal(str)  # emitted when user clicks a glass

    # Reference wavelengths for nd and Vd (Fraunhofer lines)
    WV_D = 587.6e-9   # d-line (yellow)
    WV_F = 486.1e-9   # F-line (blue)
    WV_C = 656.3e-9   # C-line (red)

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._glass_data = {}  # name -> (nd, vd)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('Filter:'))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText('Glass name...')
        self.filter_input.setToolTip(
            'Show only glasses whose name contains this substring.')
        self.filter_input.textChanged.connect(self._replot)
        toolbar.addWidget(self.filter_input)

        btn_refresh = QPushButton('Refresh')
        btn_refresh.setToolTip('Re-scan the glass registry.')
        btn_refresh.clicked.connect(self._load_glasses)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        # Apply-to-surface row: makes the click action explicit.
        apply_row = QHBoxLayout()
        apply_row.addWidget(QLabel('Apply selected glass to:'))
        self.combo_target = QComboBox()
        self.combo_target.setToolTip(
            'Surface that will receive the selected glass when you '
            'press Apply.  List refreshes as the system changes.')
        apply_row.addWidget(self.combo_target, stretch=1)
        self.btn_apply = QPushButton('Apply')
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip(
            'Set the selected surface\'s glass to the last-clicked '
            'glass on the diagram.')
        self.btn_apply.clicked.connect(self._apply_glass_to_target)
        apply_row.addWidget(self.btn_apply)
        layout.addLayout(apply_row)

        self._selected_glass = None

        # Matplotlib canvas
        self.fig = Figure(figsize=(6, 4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        layout.addWidget(self.canvas, stretch=1)

        # Info label
        self.info_label = QLabel('Click a glass to select it.')
        self.info_label.setStyleSheet("color: #7a94b8; font-size: 11px; padding: 4px;")
        layout.addWidget(self.info_label)

        self.sm.system_changed.connect(self._replot)
        self.sm.system_changed.connect(self._refresh_targets)
        self._load_glasses()
        self._refresh_targets()

    def _refresh_targets(self):
        self.combo_target.blockSignals(True)
        self.combo_target.clear()
        for ei, elem in enumerate(self.sm.elements):
            if elem.elem_type in ('Source', 'Detector'):
                continue
            for si, s in enumerate(elem.surfaces):
                # Only offer surfaces that front onto a glass region
                # (glass side of the interface, i.e. ``s.glass`` is
                # already set).  Other surfaces would need a different
                # meaning (glass_before vs glass_after), so skip to
                # keep the action unambiguous.
                if s.glass:
                    label = f'E{ei}.S{si}  {elem.name}  (now: {s.glass})'
                    self.combo_target.addItem(label, (ei, si))
        self.combo_target.blockSignals(False)

    def _apply_glass_to_target(self):
        if not self._selected_glass:
            return
        data = self.combo_target.currentData()
        if not data:
            return
        ei, si = data
        self.sm.set_surface_field(ei, si, 'glass', self._selected_glass)
        self._refresh_targets()

    def _load_glasses(self):
        """Compute nd and Vd for all glasses in the registry."""
        self._glass_data = {}
        for name in GLASS_REGISTRY:
            if name.lower() in ('air',):
                continue
            try:
                nd = get_glass_index(name, self.WV_D)
                nf = get_glass_index(name, self.WV_F)
                nc = get_glass_index(name, self.WV_C)
                vd = (nd - 1) / (nf - nc) if abs(nf - nc) > 1e-10 else 0
                self._glass_data[name] = (nd, vd)
            except Exception:
                pass
        self._replot()

    def _replot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.tick_params(colors='#7a94b8', labelsize=9)
        ax.spines[:].set_color('#2a3548')
        ax.grid(True, color='#1a2535', linewidth=0.5, alpha=0.5)

        filt = self.filter_input.text().upper() if hasattr(self, 'filter_input') else ''

        # Glasses currently in the system
        used_glasses = set()
        for elem in self.sm.elements:
            for s in elem.surfaces:
                if s.glass:
                    used_glasses.add(s.glass)

        # Plot all glasses
        names = []
        nds = []
        vds = []
        colors = []
        sizes = []

        for name, (nd, vd) in self._glass_data.items():
            if filt and filt not in name.upper():
                continue
            names.append(name)
            nds.append(nd)
            vds.append(vd)
            if name in used_glasses:
                colors.append('#ff6b35')
                sizes.append(80)
            else:
                colors.append('#5cb8ff')
                sizes.append(30)

        if vds:
            ax.scatter(vds, nds, c=colors, s=sizes, alpha=0.7, edgecolors='none')

            # Label used glasses
            for i, name in enumerate(names):
                if name in used_glasses:
                    ax.annotate(name, (vds[i], nds[i]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, color='#ff6b35',
                                fontfamily='monospace')

        ax.set_xlabel('Abbe number Vd', color='#dde8f8', fontsize=10, fontfamily='monospace')
        ax.set_ylabel('Refractive index nd', color='#dde8f8', fontsize=10, fontfamily='monospace')
        ax.set_title('Glass Map', color='#5cb8ff', fontsize=11, fontfamily='monospace')

        # Invert x-axis (conventional Abbe diagram has high Vd on left)
        if vds:
            ax.invert_xaxis()

        # Region labels
        ax.text(0.95, 0.95, f'{len(names)} glasses', transform=ax.transAxes,
                fontsize=9, color='#7a94b8', ha='right', va='top',
                fontfamily='monospace')

        self.fig.tight_layout()
        self.canvas.draw()

        # Store for click detection
        self._plot_names = names
        self._plot_vds = np.array(vds) if vds else np.array([])
        self._plot_nds = np.array(nds) if nds else np.array([])

    def _on_click(self, event):
        """Handle click on the glass map — find nearest glass."""
        if event.inaxes is None or len(self._plot_vds) == 0:
            return

        # Find nearest glass to click point
        vd_click = event.xdata
        nd_click = event.ydata

        # Normalise distances (Vd range ~20-90, nd range ~1.4-2.0)
        vd_range = max(self._plot_vds.max() - self._plot_vds.min(), 1)
        nd_range = max(self._plot_nds.max() - self._plot_nds.min(), 0.1)

        dist = ((self._plot_vds - vd_click) / vd_range) ** 2 + \
               ((self._plot_nds - nd_click) / nd_range) ** 2
        idx = np.argmin(dist)
        name = self._plot_names[idx]
        nd, vd = self._glass_data[name]

        self._selected_glass = name
        self.btn_apply.setEnabled(True)
        self.info_label.setText(
            f'Selected: {name}   nd = {nd:.5f}   Vd = {vd:.2f}   '
            f'\u2192 press Apply to assign'
        )
        self.glass_selected.emit(name)
