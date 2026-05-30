"""
Richards-Wolf high-NA focusing dock (3.6).

Wraps :func:`lumenairy.richards_wolf_focus` /
:func:`lumenairy.debye_wolf_psf` for vector-diffraction focal-spot
analysis at NA > 0.4 where the scalar PSF model breaks down.
Outputs the |Ex|², |Ey|², |Ez|² components and the total intensity.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QComboBox,
    QSizePolicy, QTextEdit,
)
from PySide6.QtGui import QFont
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


class _RWResult:
    """Lightweight container so the dock can read ``res.Ex/.Ey/.Ez`` --
    v5.4.6 (audit F-18): ``richards_wolf_focus`` returns a TUPLE
    ``(Ex, Ey, Ez, x_f, y_f)``, not an object with attributes."""

    def __init__(self, Ex, Ey, Ez, x_f, y_f):
        self.Ex, self.Ey, self.Ez = Ex, Ey, Ez
        self.x_f, self.y_f = x_f, y_f


def _rw_compute(NA, wavelength, polarization, N, dx_m, z_offset_m):
    """v5.4.6 (audit F-18): build a uniform circular pupil and call the
    real ``richards_wolf_focus(pupil, wavelength, NA, f, dx_pupil, ...)``
    signature.  The pre-fix dock called a fabricated signature
    (``NA=, n_im=, N=, dx=, z=``) that ALWAYS raised TypeError, so the
    feature never produced output.  Factored out of the QThread so it can
    be exercised without a Qt event loop."""
    import lumenairy as la
    # Map the dock's polarization labels to the library's supported set.
    pol_map = {'linear_x': 'x', 'x': 'x', 'linear_y': 'y', 'y': 'y',
               'rcp': 'circular', 'lcp': 'circular', 'circular': 'circular'}
    pol = pol_map.get(str(polarization).lower(), 'x')
    N = int(N)
    # f is arbitrary (a valid focal length); the focal-plane pitch is set
    # via dx_pupil so it honours the dock's requested dx, and the pupil is
    # filled out to the requested NA.
    f = 1.0e-3
    dx_pupil = float(wavelength) * f / (N * float(dx_m))
    xp = (np.arange(N) - N / 2) * dx_pupil
    Xp, Yp = np.meshgrid(xp, xp)
    rho = np.sqrt(Xp ** 2 + Yp ** 2)
    pupil = (rho <= f * float(NA)).astype(np.complex128)
    Ex, Ey, Ez, x_f, y_f = la.richards_wolf_focus(
        pupil, float(wavelength), float(NA), f, dx_pupil,
        N_focal=N, z_planes=[float(z_offset_m)], polarization=pol)
    return _RWResult(Ex, Ey, Ez, x_f, y_f)


class RichardsWolfWorker(QThread):
    """Background worker for the (potentially expensive) RW integral."""
    finished_result = Signal(object)

    def __init__(self, NA, wavelength, n_im, polarization, N, dx_m,
                 z_offset_m):
        super().__init__()
        self.NA = NA
        self.wavelength = wavelength
        self.n_im = n_im  # immersion index (NA already folds it in)
        self.polarization = polarization
        self.N = N
        self.dx_m = dx_m
        self.z_offset_m = z_offset_m

    def run(self):
        try:
            res = _rw_compute(self.NA, self.wavelength, self.polarization,
                              self.N, self.dx_m, self.z_offset_m)
            self.finished_result.emit(res)
        except Exception as exc:
            self.finished_result.emit(
                {'error': f'{type(exc).__name__}: {exc}'})


class RichardsWolfDock(QWidget):
    """High-NA focal-spot analysis via the Richards-Wolf integral."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None
        self._last_result = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        param = QGroupBox('Focusing geometry')
        form = QFormLayout(param)
        self.spin_NA = QDoubleSpinBox()
        self.spin_NA.setRange(0.01, 0.999)
        self.spin_NA.setValue(0.7)
        self.spin_NA.setDecimals(3)
        form.addRow('NA (focusing):', self.spin_NA)
        self.spin_n = QDoubleSpinBox()
        self.spin_n.setRange(1.0, 3.0)
        self.spin_n.setValue(1.0)
        self.spin_n.setDecimals(3)
        self.spin_n.setToolTip('Image-side refractive index '
                                '(1.0 air, 1.33 water, 1.518 oil).')
        form.addRow('Image-side n:', self.spin_n)
        self.combo_pol = QComboBox()
        self.combo_pol.addItems([
            'Linear x', 'Linear y', 'Circular (RCP)',
            'Circular (LCP)', 'Radial', 'Azimuthal',
        ])
        form.addRow('Pupil polarization:', self.combo_pol)
        self.spin_N = QSpinBox()
        self.spin_N.setRange(32, 1024)
        self.spin_N.setValue(128)
        form.addRow('Output grid N:', self.spin_N)
        self.spin_dx_nm = QDoubleSpinBox()
        self.spin_dx_nm.setRange(1.0, 10000.0)
        self.spin_dx_nm.setValue(20.0)
        self.spin_dx_nm.setSuffix(' nm')
        form.addRow('Output dx:', self.spin_dx_nm)
        self.spin_z_um = QDoubleSpinBox()
        self.spin_z_um.setRange(-100.0, 100.0)
        self.spin_z_um.setValue(0.0)
        self.spin_z_um.setSuffix(' µm')
        self.spin_z_um.setToolTip('Defocus from geometric focus.')
        form.addRow('z-offset:', self.spin_z_um)
        outer.addWidget(param)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton('▶ Compute Richards-Wolf focus')
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        run_row.addStretch()
        outer.addLayout(run_row)

        self.fig = Figure(figsize=(6, 3.4), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self.canvas, stretch=1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(80)
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
                'Compute |Ex|², |Ey|², |Ez|² at the focal spot for\n'
                'high-NA imaging where the scalar PSF model breaks.\n'
                'Reference: Richards & Wolf (1959).',
                color='#7a94b8', ha='center', va='center',
                transform=ax.transAxes, fontfamily='monospace')
        ax.tick_params(colors='#7a94b8')
        for s in ax.spines.values():
            s.set_color('#334054')
        self.canvas.draw_idle()

    def _run(self):
        polmap = {
            'Linear x': 'linear_x',
            'Linear y': 'linear_y',
            'Circular (RCP)': 'rcp',
            'Circular (LCP)': 'lcp',
            'Radial': 'radial',
            'Azimuthal': 'azimuthal',
        }
        self.btn_run.setEnabled(False)
        self.summary.setPlainText('Computing…')
        self._worker = RichardsWolfWorker(
            NA=float(self.spin_NA.value()),
            wavelength=self.sm.wavelength_m,
            n_im=float(self.spin_n.value()),
            polarization=polmap.get(self.combo_pol.currentText(),
                                     'linear_x'),
            N=int(self.spin_N.value()),
            dx_m=float(self.spin_dx_nm.value()) * 1e-9,
            z_offset_m=float(self.spin_z_um.value()) * 1e-6)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, res):
        self.btn_run.setEnabled(True)
        self._worker = None
        if isinstance(res, dict) and 'error' in res:
            self.summary.setPlainText(
                f'Richards-Wolf failed:\n  {res["error"]}')
            return
        self._last_result = res
        self._draw_result(res)
        self._summarise(res)

    def _draw_result(self, res):
        self.fig.clear()
        # res has Ex, Ey, Ez fields (complex) + total intensity.
        try:
            comps = [('|Ex|²', np.abs(res.Ex) ** 2),
                     ('|Ey|²', np.abs(res.Ey) ** 2),
                     ('|Ez|²', np.abs(res.Ez) ** 2),
                     ('|E|²',  np.abs(res.Ex) ** 2
                                + np.abs(res.Ey) ** 2
                                + np.abs(res.Ez) ** 2)]
        except Exception:
            comps = [('|E|²', getattr(res, 'I', None))]
        for i, (label, arr) in enumerate(comps):
            if arr is None:
                continue
            ax = self.fig.add_subplot(2, 2, i + 1)
            ax.set_facecolor('#0a0c10')
            ax.imshow(arr, cmap='inferno', origin='lower')
            ax.set_title(label, color='#dde8f8',
                         fontfamily='monospace', fontsize=9)
            ax.tick_params(colors='#7a94b8', labelsize=7)
            for s in ax.spines.values():
                s.set_color('#334054')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _summarise(self, res):
        lines = []
        try:
            Ix = np.abs(res.Ex) ** 2
            Iy = np.abs(res.Ey) ** 2
            Iz = np.abs(res.Ez) ** 2
            Itot = Ix + Iy + Iz
            depol = (Iz.sum() / max(Itot.sum(), 1e-30))
            lines.append(
                f'Peak |E|²: {Itot.max():.4e}')
            lines.append(
                f'Longitudinal fraction (∫|Ez|² / ∫|E|²): '
                f'{depol*100:.1f}%')
        except Exception:
            pass
        self.summary.setPlainText('\n'.join(lines))

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
