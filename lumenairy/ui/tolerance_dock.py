"""
Tolerance analysis dock — Monte Carlo sensitivity analysis.

Perturbs surface parameters within user-defined tolerances and traces
each perturbed system to build a statistical distribution of performance
metrics (RMS spot, EFL, BFL).

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QProgressBar, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
)
from PySide6.QtGui import QFont

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel
from ..raytrace import (
    Surface, trace, make_rings, spot_rms, system_abcd,
    find_paraxial_focus,
)


class ToleranceWorker(QThread):
    """Run Monte Carlo tolerance analysis in background."""
    progress = Signal(int, int)       # current, total (coarse)
    fine_progress = Signal(float, str)  # fraction, stage msg
    trial_result = Signal(float, float)  # per-trial (rms_um, efl_mm)
    finished = Signal(object)         # results dict

    def __init__(self, model, n_trials, tol_radius_pct, tol_thickness_mm,
                 tol_decenter_mm):
        super().__init__()
        self.model = model
        self.n_trials = n_trials
        self.tol_radius_pct = tol_radius_pct
        self.tol_thickness_mm = tol_thickness_mm
        self.tol_decenter_mm = tol_decenter_mm

    def run(self):
        rms_list = []
        efl_list = []
        rng = np.random.default_rng()
        base_surfaces = self.model.build_trace_surfaces()
        wv = self.model.wavelength_m
        semi_ap = self.model.epd_m / 2.0

        for trial in range(self.n_trials):
            self.progress.emit(trial + 1, self.n_trials)
            self.fine_progress.emit(
                trial / max(self.n_trials, 1),
                f'trial {trial + 1}/{self.n_trials}')

            # Decenter is a lateral ray perturbation applied to the
            # whole incoming bundle: a random offset draws the ray
            # starting point in (x, y).  Physically equivalent to
            # perturbing one lens element's decenter when only one lens
            # is present; for multi-element systems it's a first-order
            # approximation until per-element decenters are plumbed
            # through Surface.
            if self.tol_decenter_mm > 0:
                dec_x = rng.normal(0, self.tol_decenter_mm * 1e-3)
                dec_y = rng.normal(0, self.tol_decenter_mm * 1e-3)
            else:
                dec_x = dec_y = 0.0

            # Perturb surfaces
            surfs = []
            for s in base_surfaces:
                R = s.radius
                if np.isfinite(R) and self.tol_radius_pct > 0:
                    R *= (1 + rng.normal(0, self.tol_radius_pct / 100))

                # Anamorphic: independently perturb the y-axis radius so
                # that biconic / cylindrical surfaces are toleranced rather
                # than silently snapping to rotational symmetry.
                Ry = s.radius_y
                if (Ry is not None and np.isfinite(Ry)
                        and self.tol_radius_pct > 0):
                    Ry *= (1 + rng.normal(0, self.tol_radius_pct / 100))

                t = s.thickness
                if t > 0 and self.tol_thickness_mm > 0:
                    t += rng.normal(0, self.tol_thickness_mm * 1e-3)
                    t = max(t, 0)

                surfs.append(Surface(
                    radius=R, conic=s.conic,
                    aspheric_coeffs=s.aspheric_coeffs,
                    semi_diameter=s.semi_diameter,
                    glass_before=s.glass_before, glass_after=s.glass_after,
                    is_mirror=s.is_mirror, thickness=t,
                    label=s.label,
                    surf_num=s.surf_num,
                    radius_y=Ry,
                    conic_y=s.conic_y,
                    aspheric_coeffs_y=s.aspheric_coeffs_y,
                ))

            try:
                # ABCD
                _, efl, bfl, _ = system_abcd(surfs, wv)
                efl_list.append(efl * 1e3)  # mm

                # Trace -- offset the whole bundle by the sampled decenter.
                rays = make_rings(semi_ap, 6, 24, 0.0, wv)
                if dec_x or dec_y:
                    rays.x = rays.x + dec_x
                    rays.y = rays.y + dec_y
                if np.isfinite(bfl) and bfl > 0:
                    surfs[-1].thickness = bfl
                    surfs.append(Surface(radius=np.inf, semi_diameter=np.inf,
                        glass_before=surfs[-1].glass_after,
                        glass_after=surfs[-1].glass_after))
                result = trace(rays, surfs, wv)
                rms, _ = spot_rms(result)
                rms_list.append(rms * 1e6)  # µm
                # Emit the running trial so the dock can update its
                # live histogram.
                self.trial_result.emit(
                    float(rms * 1e6), float(efl * 1e3))
            except Exception as e:
                try:
                    from .diagnostics import diag
                    diag.report('tolerance-trial', e,
                                context=f'trial {trial}')
                except Exception:
                    pass
                rms_list.append(np.nan)
                efl_list.append(np.nan)

        self.finished.emit({
            'rms': np.array(rms_list),
            'efl': np.array(efl_list),
        })


class ToleranceDock(QWidget):
    """Monte Carlo tolerance analysis panel."""

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Tolerances ──
        tol_group = QGroupBox('Tolerances (1σ)')
        tol_layout = QVBoxLayout(tol_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel('Radius (%):'))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0, 10)
        self.spin_radius.setValue(0.1)
        self.spin_radius.setDecimals(3)
        row1.addWidget(self.spin_radius)
        tol_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Thickness (mm):'))
        self.spin_thick = QDoubleSpinBox()
        self.spin_thick.setRange(0, 1)
        self.spin_thick.setValue(0.01)
        self.spin_thick.setDecimals(4)
        row2.addWidget(self.spin_thick)
        tol_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel('Decenter (mm):'))
        self.spin_decenter = QDoubleSpinBox()
        self.spin_decenter.setRange(0, 1)
        self.spin_decenter.setValue(0.01)
        self.spin_decenter.setDecimals(4)
        self.spin_decenter.setToolTip(
            '1\u03c3 lateral offset of the incoming ray bundle (applied '
            'uniformly in X and Y).  Captures first-order effects of '
            'mechanical mounting decenter.')
        row3.addWidget(self.spin_decenter)
        tol_layout.addLayout(row3)

        layout.addWidget(tol_group)

        # ── Run control ──
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel('Trials:'))
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(10, 10000)
        self.spin_trials.setValue(500)
        run_row.addWidget(self.spin_trials)

        self.btn_run = QPushButton('▶ Run MC')
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)
        layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Results plot ──
        self.fig = Figure(figsize=(6, 3), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, stretch=1)

        # ── Summary ──
        self.summary = QLabel('')
        self.summary.setStyleSheet("color: #7a94b8; font-size: 11px;")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def _run(self):
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.spin_trials.value())
        self.progress_bar.setValue(0)
        self._live_rms = []
        self._live_efl = []
        # Update cadence: redraw the histogram every K trials so we
        # don't kill perf with matplotlib overhead on 10k-trial runs.
        self._live_redraw_every = max(
            1, self.spin_trials.value() // 40)

        self._worker = ToleranceWorker(
            self.sm, self.spin_trials.value(),
            self.spin_radius.value(),
            self.spin_thick.value(),
            self.spin_decenter.value(),
        )
        self._worker.progress.connect(
            lambda c, t: self.progress_bar.setValue(c))
        self._worker.fine_progress.connect(self._on_fine_progress)
        self._worker.trial_result.connect(self._on_trial_result)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_trial_result(self, rms_um, efl_mm):
        """Append to the live histogram; redraw every N trials."""
        self._live_rms.append(rms_um)
        self._live_efl.append(efl_mm)
        if len(self._live_rms) % self._live_redraw_every == 0:
            self._draw_histograms(
                np.asarray(self._live_rms),
                np.asarray(self._live_efl),
                title_suffix=f' (live: {len(self._live_rms)} trials)')

    def _draw_histograms(self, rms, efl, title_suffix=''):
        self.fig.clear()
        style = {'color': '#dde8f8', 'fontsize': 9, 'fontfamily': 'monospace'}
        if len(rms) > 0:
            ax1 = self.fig.add_subplot(121)
            ax1.set_facecolor('#0a0c10')
            ax1.tick_params(colors='#7a94b8', labelsize=8)
            ax1.spines[:].set_color('#2a3548')
            ax1.hist(rms, bins=min(30, max(5, len(rms) // 5)),
                     color='#5cb8ff', alpha=0.7, edgecolor='#2a3548')
            ax1.axvline(np.median(rms), color='#ff6b35', linewidth=1.5, linestyle='--')
            ax1.set_xlabel('RMS Spot (um)', **style)
            ax1.set_ylabel('Count', **style)
            ax1.set_title(f'RMS Distribution{title_suffix}',
                          color='#5cb8ff', fontsize=10,
                          fontfamily='monospace')
        if len(efl) > 0:
            ax2 = self.fig.add_subplot(122)
            ax2.set_facecolor('#0a0c10')
            ax2.tick_params(colors='#7a94b8', labelsize=8)
            ax2.spines[:].set_color('#2a3548')
            ax2.hist(efl, bins=min(30, max(5, len(efl) // 5)),
                     color='#3ddc84', alpha=0.7, edgecolor='#2a3548')
            ax2.axvline(np.median(efl), color='#ff6b35', linewidth=1.5, linestyle='--')
            ax2.set_xlabel('EFL (mm)', **style)
            ax2.set_ylabel('Count', **style)
            ax2.set_title('EFL Distribution', color='#3ddc84',
                          fontsize=10, fontfamily='monospace')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_fine_progress(self, fraction, message):
        # The progress bar is already driven by the integer trial count
        # above; use the message to tag tooltips so the user can see
        # which trial is running on hover.
        self.progress_bar.setToolTip(message)

    def _on_finished(self, results):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._worker = None

        rms = results['rms']
        efl = results['efl']
        valid = ~np.isnan(rms)
        rms = rms[valid]
        efl = efl[~np.isnan(efl)]

        self._draw_histograms(rms, efl)

        # Summary text
        if len(rms) > 0:
            p95 = np.percentile(rms, 95)
            self.summary.setText(
                f'RMS: median={np.median(rms):.2f} µm, '
                f'σ={np.std(rms):.2f} µm, 95th={p95:.2f} µm  |  '
                f'EFL: median={np.median(efl):.3f} mm, '
                f'σ={np.std(efl):.4f} mm  |  '
                f'{len(rms)}/{self.spin_trials.value()} converged'
            )
