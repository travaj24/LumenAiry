"""
Laguerre-Gauss aberration tensor dock (3.6).

Exposes the modal asymptotic aberration analysis from
``lumenairy.aberration_tensor`` / ``lumenairy.decompose_lg`` /
``lumenairy.lg_seidel_label`` -- a complementary view to the
Zernike decomposition that captures aberration coupling between
LG mode pairs and labels each tensor element with its Seidel
equivalent.

Author: Andrew Traverso
"""

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QTextEdit,
    QSizePolicy,
)
from PySide6.QtGui import QFont
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


class LGAberrationDock(QWidget):
    """LG-tensor heat-map of the system's modal aberrations."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        param = QGroupBox('Mode basis')
        form = QFormLayout(param)
        self.spin_pmax = QSpinBox()
        self.spin_pmax.setRange(0, 12)
        self.spin_pmax.setValue(4)
        self.spin_pmax.setToolTip('Max radial index p.')
        form.addRow('Max p:', self.spin_pmax)
        self.spin_lmax = QSpinBox()
        self.spin_lmax.setRange(0, 12)
        self.spin_lmax.setValue(4)
        self.spin_lmax.setToolTip('Max azimuthal index |ℓ|.')
        form.addRow('Max |ℓ|:', self.spin_lmax)
        self.spin_w0_um = QDoubleSpinBox()
        self.spin_w0_um.setRange(1.0, 1e6)
        self.spin_w0_um.setValue(100.0)
        self.spin_w0_um.setSuffix(' µm')
        self.spin_w0_um.setToolTip('LG basis waist.')
        form.addRow('Basis waist:', self.spin_w0_um)
        outer.addWidget(param)

        self.btn_run = QPushButton('▶ Compute LG aberration tensor')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        outer.addWidget(self.btn_run)

        self.fig = Figure(figsize=(6, 3.4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas, stretch=1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        outer.addWidget(self.summary)
        self._draw_empty()

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        ax.text(0.5, 0.5,
                'Compute the Laguerre-Gauss aberration tensor for\n'
                'this system.  Captures inter-mode coupling that\n'
                'a single Zernike spectrum cannot represent.\n'
                'Each cell labelled with its Seidel equivalent.',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _run(self):
        try:
            import lumenairy as la
            pres = self.sm.to_prescription()
            wv = self.sm.wavelength_m
            w0 = float(self.spin_w0_um.value()) * 1e-6
            pmax = int(self.spin_pmax.value())
            lmax = int(self.spin_lmax.value())
            T = la.aberration_tensor(
                pres, wavelength=wv, w0=w0,
                p_max=pmax, l_max=lmax)
        except Exception as exc:
            self.summary.setPlainText(
                f'aberration_tensor failed: {type(exc).__name__}: {exc}')
            return
        self._draw_tensor(T)

    def _draw_tensor(self, T):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        try:
            arr = np.asarray(getattr(T, 'tensor', T))
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            im = ax.imshow(np.abs(arr), cmap='magma', origin='lower')
            ax.set_title(
                'LG aberration tensor (|element|)',
                color='#dde8f8', fontfamily='monospace')
            ax.set_xlabel('mode index n', color='#dde8f8',
                          fontfamily='monospace')
            ax.set_ylabel('mode index m', color='#dde8f8',
                          fontfamily='monospace')
            ax.tick_params(colors='#7a94b8', labelsize=8)
            for s in ax.spines.values():
                s.set_color('#334054')
            # Try to surface a Seidel-equivalent label list.
            try:
                import lumenairy as la
                P, L = arr.shape[:2]
                lines = ['Largest tensor elements (|coeff|, mode pair → Seidel):']
                idx = np.dstack(np.unravel_index(
                    np.argsort(np.abs(arr).ravel())[::-1][:8], arr.shape))
                for (i, j) in idx[0]:
                    label = la.lg_seidel_label(i, j) \
                        if hasattr(la, 'lg_seidel_label') else ''
                    lines.append(
                        f'  |T[{i},{j}]| = {np.abs(arr[i,j]):.4e}  {label}')
                self.summary.setPlainText('\n'.join(lines))
            except Exception:
                self.summary.setPlainText(
                    f'Tensor shape: {arr.shape}; '
                    f'max |element|: {np.max(np.abs(arr)):.4e}')
        except Exception as exc:
            self.summary.setPlainText(
                f'Could not render tensor: {type(exc).__name__}: {exc}')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def minimumSizeHint(self):
        """v5.4.4 (audit GUI-resize round 2): report a tiny minimum so
        the QDockWidget will let the user drag this dock pane down to
        almost nothing.  Inherited Qt implementation walks layout
        children (matplotlib canvas, tables, toolbars) and adds up
        their hints, producing a floor that locks the bottom dock
        area on non-Design tabs.  Matches the v3.6.1 fix in
        layout_2d.py / layout_3d.py.
        """
        from PySide6.QtCore import QSize
        return QSize(40, 40)

    def sizeHint(self):
        """v5.4.4: companion to minimumSizeHint() above.  Provides a
        reasonable initial size when the dock is first shown."""
        from PySide6.QtCore import QSize
        return QSize(400, 200)
