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

# v5.4 (audit P2-E): add Stokes + DOP tabs
The single Jones-pupil plot is now wrapped in a QTabWidget alongside
two new analysis tabs:
  - Stokes:                S0 / S1 / S2 / S3 derived from J under an
                           unpolarized-input convention.
  - Polarisation derived:  DOP / DOLP / DOCP from those Stokes maps.
The existing run button refreshes all three tabs in one shot.

Author: Andrew Traverso
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QDoubleSpinBox, QSizePolicy, QTabWidget,
)

import numpy as np

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure

from .model import SystemModel


# ---------------------------------------------------------------------------
# Pure-numerical helpers (no Qt; testable in isolation)
# ---------------------------------------------------------------------------

def _jones_to_stokes_unpolarized(J: np.ndarray) -> dict:
    """Derive per-pixel Stokes maps from a Jones pupil under
    unpolarized-input convention.

    The library's ``stokes_parameters(field)`` operates on a single
    JonesField (one input polarization).  A Jones *pupil* is the full
    2x2 transfer matrix at each spatial point, so to collapse it to a
    Stokes image we must specify the input polarization.  We use the
    canonical unpolarized-input formulas (Mueller-matrix row-0):

        S0 = |J00|^2 + |J01|^2 + |J10|^2 + |J11|^2
        S1 = |J00|^2 - |J01|^2 + |J10|^2 - |J11|^2
        S2 = 2 Re(J00 J01* + J10 J11*)
        S3 = 2 Im(J01 J00* + J11 J10*)

    Parameters
    ----------
    J : ndarray (complex, Ny, Nx, 2, 2)
        Jones pupil as returned by :func:`compute_jones_pupil`.

    Returns
    -------
    dict with keys 'S0', 'S1', 'S2', 'S3' -- each a real (Ny, Nx) array.
    """
    J00 = J[..., 0, 0]
    J01 = J[..., 0, 1]
    J10 = J[..., 1, 0]
    J11 = J[..., 1, 1]
    a00 = np.abs(J00) ** 2
    a01 = np.abs(J01) ** 2
    a10 = np.abs(J10) ** 2
    a11 = np.abs(J11) ** 2
    S0 = a00 + a01 + a10 + a11
    S1 = a00 - a01 + a10 - a11
    S2 = 2.0 * np.real(J00 * np.conj(J01) + J10 * np.conj(J11))
    S3 = 2.0 * np.imag(J01 * np.conj(J00) + J11 * np.conj(J10))
    return {'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3}


def _dop_dolp_docp(stokes: dict) -> dict:
    """Per-pixel DOP / DOLP / DOCP from a Stokes dict.

    DOP  = sqrt(S1^2 + S2^2 + S3^2) / S0
    DOLP = sqrt(S1^2 + S2^2)        / S0
    DOCP = |S3|                     / S0

    Pixels with S0 <= 1e-30 (background) are set to 0.
    """
    S0 = stokes['S0']
    S1 = stokes['S1']
    S2 = stokes['S2']
    S3 = stokes['S3']
    safe = np.maximum(S0, 1e-30)
    mask = S0 > 1e-30
    dop = np.where(mask, np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / safe, 0.0)
    dolp = np.where(mask, np.sqrt(S1 ** 2 + S2 ** 2) / safe, 0.0)
    docp = np.where(mask, np.abs(S3) / safe, 0.0)
    return {'DOP': dop, 'DOLP': dolp, 'DOCP': docp}


# ---------------------------------------------------------------------------
# Dock widget
# ---------------------------------------------------------------------------

class JonesPupilDock(QWidget):
    """Dock that renders the 2x4 Jones-pupil amplitude + phase grid
    plus Stokes / polarization-derived tabs."""

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

        toolbar.addWidget(QLabel('dx (um):'))
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(0.1, 500.0)
        self.spin_dx.setValue(16.0)
        self.spin_dx.setDecimals(2)
        self.spin_dx.setSuffix(' um')
        toolbar.addWidget(self.spin_dx)

        self.btn_run = QPushButton('Compute Jones pupil')
        self.btn_run.setObjectName('run_button')
        self.btn_run.setToolTip(
            'Probe the current prescription with pure-x and pure-y '
            'plane-wave inputs, record both outgoing JonesFields, and '
            'plot the 2x2 exit-pupil Jones matrix as amplitude + phase. '
            'Also refreshes the Stokes and Polarisation-derived tabs.')
        self.btn_run.clicked.connect(self._run)
        toolbar.addWidget(self.btn_run)

        toolbar.addStretch()

        self.lbl_status = QLabel('Idle.  Click "Compute" to run.')
        self.lbl_status.setStyleSheet(
            'color: #7a94b8; font-family: monospace;')
        toolbar.addWidget(self.lbl_status)

        layout.addLayout(toolbar)

        # -- Tabbed plot area: Jones / Stokes / DOP-DOLP-DOCP
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        # Tab 1: Jones pupil (original 2x4 grid)
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        tab_jones = QWidget()
        lay_jones = QVBoxLayout(tab_jones)
        lay_jones.setContentsMargins(0, 0, 0, 0)
        lay_jones.addWidget(self.mpl_toolbar)
        lay_jones.addWidget(self.canvas, stretch=1)
        self.tabs.addTab(tab_jones, 'Jones pupil')

        # Tab 2: Stokes 2x2 grid
        self.fig_stokes = Figure(figsize=(10, 8), dpi=100, facecolor='#0a0c10')
        self.canvas_stokes = FigureCanvasQTAgg(self.fig_stokes)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas_stokes.setMinimumSize(0, 0)
        self.canvas_stokes.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_toolbar_stokes = NavigationToolbar2QT(self.canvas_stokes, self)
        tab_stokes = QWidget()
        lay_stokes = QVBoxLayout(tab_stokes)
        lay_stokes.setContentsMargins(0, 0, 0, 0)
        lay_stokes.addWidget(self.mpl_toolbar_stokes)
        lay_stokes.addWidget(self.canvas_stokes, stretch=1)
        self.tabs.addTab(tab_stokes, 'Stokes')

        # Tab 3: Polarisation-derived (DOP / DOLP / DOCP)
        self.fig_dop = Figure(figsize=(12, 4), dpi=100, facecolor='#0a0c10')
        self.canvas_dop = FigureCanvasQTAgg(self.fig_dop)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas_dop.setMinimumSize(0, 0)
        self.canvas_dop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_toolbar_dop = NavigationToolbar2QT(self.canvas_dop, self)
        tab_dop = QWidget()
        lay_dop = QVBoxLayout(tab_dop)
        lay_dop.setContentsMargins(0, 0, 0, 0)
        lay_dop.addWidget(self.mpl_toolbar_dop)
        lay_dop.addWidget(self.canvas_dop, stretch=1)
        self.tabs.addTab(tab_dop, 'Polarisation derived')

        # Placeholder on first draw
        self._show_placeholder()

    def _show_placeholder(self):
        for fig, msg in (
            (self.fig, 'Click "Compute Jones pupil" to probe the current system.'),
            (self.fig_stokes, 'Stokes (S0/S1/S2/S3) -- click "Compute" to populate.'),
            (self.fig_dop, 'DOP / DOLP / DOCP -- click "Compute" to populate.'),
        ):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0a0c10')
            ax.text(0.5, 0.5, msg,
                    ha='center', va='center', color='#7a94b8',
                    transform=ax.transAxes, fontfamily='monospace')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('#2a3548')
        self.canvas.draw()
        self.canvas_stokes.draw()
        self.canvas_dop.draw()

    # ------------------------------------------------------------------
    # Per-tab redraw helpers
    # ------------------------------------------------------------------

    def _style_axes(self, ax):
        ax.set_facecolor('#0a0c10')
        ax.tick_params(colors='#7a94b8', labelsize=8)
        for s in ax.spines.values():
            s.set_color('#2a3548')

    def _draw_jones_tab(self, J, N, dx_out, dy_out):
        self.fig.clear()
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
                self._style_axes(ax_a)
                self._style_axes(ax_p)

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
                cb.ax.set_yticklabels(['-pi', '0', 'pi'])
                ax_p.set_title(f'arg({labels[r][c]})',
                               color='#5cb8ff', fontsize=10,
                               fontfamily='monospace')

        self.fig.suptitle('Jones pupil (exit pupil)', color='#dde8f8',
                          fontsize=12, fontfamily='monospace')
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_stokes_tab(self, stokes, N, dx_out, dy_out):
        self.fig_stokes.clear()
        axes = self.fig_stokes.subplots(2, 2)

        extent_um = (-N/2 * dx_out * 1e6, N/2 * dx_out * 1e6,
                     -N/2 * dy_out * 1e6, N/2 * dy_out * 1e6)

        # S0: viridis, [0, max].  S1/S2/S3: RdBu_r centred on zero.
        S0 = stokes['S0']
        s0_max = float(S0.max()) if S0.max() > 0 else 1.0
        # Symmetric scale shared across S1/S2/S3 so colours are
        # comparable across panels.
        sym = max(
            float(np.abs(stokes['S1']).max()),
            float(np.abs(stokes['S2']).max()),
            float(np.abs(stokes['S3']).max()),
            1e-30,
        )

        plots = [
            ('S0', S0, 'viridis', 0.0, s0_max),
            ('S1', stokes['S1'], 'RdBu_r', -sym, sym),
            ('S2', stokes['S2'], 'RdBu_r', -sym, sym),
            ('S3', stokes['S3'], 'RdBu_r', -sym, sym),
        ]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for (name, arr, cmap, vmin, vmax), (r, c) in zip(plots, positions):
            ax = axes[r, c]
            self._style_axes(ax)
            im = ax.imshow(arr, extent=extent_um, origin='lower',
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            aspect='equal')
            self.fig_stokes.colorbar(im, ax=ax, shrink=0.75)
            ax.set_title(name, color='#5cb8ff', fontsize=11,
                         fontfamily='monospace')

        self.fig_stokes.suptitle(
            'Stokes parameters (unpolarized input convention)',
            color='#dde8f8', fontsize=12, fontfamily='monospace')
        self.fig_stokes.tight_layout()
        self.canvas_stokes.draw()

    def _draw_dop_tab(self, derived, N, dx_out, dy_out):
        self.fig_dop.clear()
        axes = self.fig_dop.subplots(1, 3)

        extent_um = (-N/2 * dx_out * 1e6, N/2 * dx_out * 1e6,
                     -N/2 * dy_out * 1e6, N/2 * dy_out * 1e6)

        plots = [
            ('DOP',  derived['DOP']),
            ('DOLP', derived['DOLP']),
            ('DOCP', derived['DOCP']),
        ]
        for ax, (name, arr) in zip(axes, plots):
            self._style_axes(ax)
            im = ax.imshow(arr, extent=extent_um, origin='lower',
                            cmap='viridis', vmin=0.0, vmax=1.0,
                            aspect='equal')
            self.fig_dop.colorbar(im, ax=ax, shrink=0.75)
            ax.set_title(name, color='#5cb8ff', fontsize=11,
                         fontfamily='monospace')

        self.fig_dop.suptitle(
            'Polarisation degree maps (DOP / DOLP / DOCP)',
            color='#dde8f8', fontsize=12, fontfamily='monospace')
        self.fig_dop.tight_layout()
        self.canvas_dop.draw()

    # ------------------------------------------------------------------
    # Run slot: compute J, derive Stokes / DOP, refresh all three tabs
    # ------------------------------------------------------------------

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
            from .. import compute_jones_pupil
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

        # Tab 1: Jones pupil (original 2x4 grid)
        self._draw_jones_tab(J, N, dx_out, dy_out)

        # v5.4 (audit P2-E): Stokes and polarisation-derived tabs
        try:
            stokes = _jones_to_stokes_unpolarized(J)
            derived = _dop_dolp_docp(stokes)
            self._draw_stokes_tab(stokes, N, dx_out, dy_out)
            self._draw_dop_tab(derived, N, dx_out, dy_out)
        except Exception as e:
            # A failed Stokes/DOP render shouldn't tear down the Jones
            # tab the user came for -- surface the message and bail.
            self.lbl_status.setText(f'Stokes/DOP error: {e}')
            try:
                from .diagnostics import diag
                diag.report('jones-pupil-dock-stokes', e,
                            context=f'N={N}, dx={dx}')
            except Exception:
                pass
            return

        jxx = float(np.abs(J[..., 0, 0]).max())
        jyy = float(np.abs(J[..., 1, 1]).max())
        jxy = float(np.abs(J[..., 0, 1]).max())
        dop_mean = float(np.nanmean(derived['DOP']))
        self.lbl_status.setText(
            f'Computed.  |Jxx|={jxx:.3e}  |Jyy|={jyy:.3e}  '
            f'|Jxy|={jxy:.3e} (cross-pol)  <DOP>={dop_mean:.3f}')
