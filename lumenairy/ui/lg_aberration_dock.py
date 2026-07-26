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
        # v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A
        # UI pass): this called ``aberration_tensor(pres, wavelength=,
        # w0=, p_max=, l_max=)``, but that function takes
        # ``(fit: CanonicalPolyFit, s2_image, *, source_point=,
        # source_modes=, pupil_modes=, output_modes=, w_s=, w_p=, ...)``
        # -- a prescription is not a fit and none of the four kwargs
        # exist, so Run could only report "aberration_tensor failed:
        # TypeError: missing a required argument: 's2_image'" (measured by
        # signature bind).  Routed through the public one-shot wrapper
        # ``aberration_summary``, which owns the
        # fit_canonical_polynomials -> solve_envelope_stationary ->
        # aberration_tensor chain (the same sequence
        # analysis/aberration.py uses) and reports why in ``notes`` when
        # the asymptotic fit does not converge.  The dock's p / |l| grid
        # becomes the OUTPUT mode list -- the rows of the returned L.
        try:
            import lumenairy as la
            pres = self.sm.to_prescription()
            wv = self.sm.wavelength_m
            w0 = float(self.spin_w0_um.value()) * 1e-6
            pmax = int(self.spin_pmax.value())
            lmax = int(self.spin_lmax.value())
            out_modes = [(p, ell)
                         for p in range(pmax + 1)
                         for ell in range(-lmax, lmax + 1)]
            summary = la.aberration_summary(
                pres, wv, output_modes=out_modes, w_s=w0)
            T = summary.lg_tensor
            if T is None:
                self.summary.setPlainText(
                    'LG aberration tensor unavailable:\n  '
                    + '\n  '.join(summary.notes or ['(no diagnostics)']))
                return
        except Exception as exc:
            self.summary.setPlainText(
                f'aberration_summary failed: {type(exc).__name__}: {exc}')
            return
        self._draw_tensor(T)

    def _draw_tensor(self, T):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#0a0c10')
        try:
            # v5.30: ``AberrationTensorResult`` carries the matrix on
            # ``.L`` (rows = output modes, columns = source modes); the
            # pre-fix ``getattr(T, 'tensor', T)`` fell through to the
            # dataclass itself and np.asarray'd an object.
            arr = np.asarray(getattr(T, 'L', getattr(T, 'tensor', T)))
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            im = ax.imshow(np.abs(arr), cmap='magma', origin='lower',
                           aspect='auto')
            ax.set_title(
                'LG aberration tensor (|element|)',
                color='#dde8f8', fontfamily='monospace')
            ax.set_xlabel('source mode index', color='#dde8f8',
                          fontfamily='monospace')
            ax.set_ylabel('output mode index', color='#dde8f8',
                          fontfamily='monospace')
            ax.tick_params(colors='#7a94b8', labelsize=8)
            for s in ax.spines.values():
                s.set_color('#334054')
            # Try to surface a Seidel-equivalent label list.
            try:
                import lumenairy as la
                # v5.30: label rows by their (p, ell) OUTPUT mode -- the
                # pre-fix code fed the raw matrix indices (i, j) to
                # lg_seidel_label(p, ell), mislabelling every row.
                out_modes = list(getattr(T, 'output_modes', []) or [])
                lines = ['Largest tensor elements (|coeff|, output mode → Seidel):']
                idx = np.dstack(np.unravel_index(
                    np.argsort(np.abs(arr).ravel())[::-1][:8], arr.shape))
                for (i, j) in idx[0]:
                    if i < len(out_modes):
                        p, ell = out_modes[i]
                        label = f'(p={p}, l={ell:+d}) {la.lg_seidel_label(p, ell)}'
                    else:
                        label = ''
                    lines.append(
                        f'  |L[{i},{j}]| = {np.abs(arr[i,j]):.4e}  {label}')
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
