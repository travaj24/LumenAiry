"""
Jones-pupil visualization dock.

Probes the current lens system with orthogonal x- and y-polarized
plane-wave inputs (via :func:`compute_jones_pupil`) and renders the
spatially-resolved 2x2 exit-pupil Jones matrix as the canonical 2x4
grid (amplitude + phase for each of Jxx / Jxy / Jyx / Jyy), using
:func:`plot_jones_pupil`.

Scalar (non-polarizing) lens systems should show exactly diagonal
Jones pupils.  Polarization-sensitive coatings or birefringent
materials are where the off-diagonal maps become interesting.

Author: Andrew Traverso
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox,
)

import numpy as np

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from .model import SystemModel


class JonesPupilDock(QWidget):
    """Dock that renders the 2x4 Jones-pupil amplitude + phase grid."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -- Toolbar: N / dx / Run
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel('N:'))
        self.combo_N = QComboBox()
        for n in (128, 256, 512, 1024):
            self.combo_N.addItem(str(n), n)
        self.combo_N.setCurrentIndex(1)  # 256
        toolbar.addWidget(self.combo_N)

        toolbar.addWidget(QLabel('dx (µm):'))
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(0.1, 500.0)
        self.spin_dx.setValue(16.0)
        self.spin_dx.setDecimals(2)
        self.spin_dx.setSuffix(' µm')
        toolbar.addWidget(self.spin_dx)

        self.btn_run = QPushButton('Compute Jones pupil')
        self.btn_run.setToolTip(
            'Probe the current prescription with pure-x and pure-y '
            'plane-wave inputs, record both outgoing JonesFields, and '
            'plot the 2×2 exit-pupil Jones matrix as amplitude + phase.')
        self.btn_run.clicked.connect(self._run)
        toolbar.addWidget(self.btn_run)

        toolbar.addStretch()

        self.lbl_status = QLabel('Idle.  Click "Compute" to run.')
        self.lbl_status.setStyleSheet(
            'color: #7a94b8; font-family: monospace;')
        toolbar.addWidget(self.lbl_status)

        layout.addLayout(toolbar)

        # -- Matplotlib canvas + navigation toolbar
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.mpl_toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # Placeholder on first draw
        self._show_placeholder()

    def _show_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Click "Compute Jones pupil" to probe the current system.',
                ha='center', va='center', color='#7a94b8',
                transform=ax.transAxes, fontfamily='monospace')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#2a3548')
        self.canvas.draw()

    def _run(self):
        pres = self.sm.to_prescription()
        if not pres.get('surfaces'):
            self.lbl_status.setText('No surfaces in current system.')
            return

        N = int(self.combo_N.currentData())
        dx = float(self.spin_dx.value()) * 1e-6
        wv = self.sm.wavelength_m

        # Apply-function: run the model's prescription on a JonesField
        # in place, matching the compute_jones_pupil contract.  Any
        # errors here (non-scalar glass, unsupported surface type) bubble
        # up to the status bar.
        def apply_fn(jf):
            jf.apply_real_lens(pres, wv)
            return jf

        try:
            from .. import compute_jones_pupil, plot_jones_pupil
            J, dx_out, dy_out = compute_jones_pupil(apply_fn, N, dx, wv)
        except Exception as e:
            self.lbl_status.setText(f'Error: {e}')
            try:
                from .diagnostics import diag
                diag.report('jones-pupil-dock', e,
                            context=f'N={N}, dx={dx}')
            except Exception:
                pass
            return

        # Re-draw using plot_jones_pupil's figure into our canvas.  The
        # library helper builds its own figure; we re-render into our
        # Figure instance by reusing its axes layout so the canvas stays
        # live in the dock.
        self.fig.clear()
        # 2 rows x 4 cols: amplitude (cols 0..1) + phase (cols 2..3)
        axes = self.fig.subplots(2, 4)

        labels = [['J_xx', 'J_xy'], ['J_yx', 'J_yy']]
        amp = np.abs(J)
        phase = np.angle(J)
        amp_max = float(amp.max()) if amp.max() > 0 else 1.0
        mask = amp > 0.01 * amp_max

        extent_um = (-N/2 * dx_out * 1e6, N/2 * dx_out * 1e6,
                     -N/2 * dy_out * 1e6, N/2 * dy_out * 1e6)

        for r in range(2):
            for c in range(2):
                ax_a = axes[r, c]
                ax_p = axes[r, c + 2]
                ax_a.set_facecolor('#0a0c10')
                ax_p.set_facecolor('#0a0c10')

                im_a = ax_a.imshow(amp[..., r, c], extent=extent_um,
                                    origin='lower', cmap='inferno',
                                    vmin=0, vmax=amp_max,
                                    aspect='equal')
                self.fig.colorbar(im_a, ax=ax_a, shrink=0.75)
                ax_a.set_title(f'|{labels[r][c]}|',
                               color='#5cb8ff', fontsize=10,
                               fontfamily='monospace')

                ph = np.where(mask[..., r, c], phase[..., r, c], np.nan)
                im_p = ax_p.imshow(ph, extent=extent_um,
                                    origin='lower', cmap='twilight',
                                    vmin=-np.pi, vmax=np.pi,
                                    aspect='equal')
                cb = self.fig.colorbar(im_p, ax=ax_p, shrink=0.75,
                                        ticks=[-np.pi, 0, np.pi])
                cb.ax.set_yticklabels(['-π', '0', 'π'])
                ax_p.set_title(f'arg({labels[r][c]})',
                               color='#5cb8ff', fontsize=10,
                               fontfamily='monospace')

                for ax in (ax_a, ax_p):
                    ax.tick_params(colors='#7a94b8', labelsize=8)
                    for s in ax.spines.values():
                        s.set_color('#2a3548')

        self.fig.suptitle('Jones pupil (exit pupil)', color='#dde8f8',
                          fontsize=12, fontfamily='monospace')
        self.fig.tight_layout()
        self.canvas.draw()

        jxx = float(np.abs(J[..., 0, 0]).max())
        jyy = float(np.abs(J[..., 1, 1]).max())
        jxy = float(np.abs(J[..., 0, 1]).max())
        self.lbl_status.setText(
            f'Computed.  |Jxx|={jxx:.3e}  |Jyy|={jyy:.3e}  '
            f'|Jxy|={jxy:.3e} (cross-pol)')
