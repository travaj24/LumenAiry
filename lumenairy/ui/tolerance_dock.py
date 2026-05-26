"""
Tolerance analysis dock — Monte Carlo sensitivity analysis.

Perturbs surface parameters within user-defined tolerances and traces
each perturbed system to build a statistical distribution of performance
metrics (RMS spot, EFL, BFL).

# v5.4 (audit P1-F): wire CancellableProgress + Stop button

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QProgressBar, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QFileDialog, QMessageBox,
)
from PySide6.QtGui import QFont

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel
from ..progress import CancellableProgress
from ..raytrace import (
    Surface, trace, make_rings, spot_rms, system_abcd,
    find_paraxial_focus,
)
from ..raytrace.core import trace_world


class ToleranceWorker(QThread):
    """Run Monte Carlo tolerance analysis in background."""
    progress = Signal(int, int)       # current, total (coarse)
    fine_progress = Signal(float, str)  # fraction, stage msg
    trial_result = Signal(float, float)  # per-trial (rms_um, efl_mm)
    finished = Signal(object)         # results dict
    cancelled = Signal()

    def __init__(self, model, n_trials, tol_radius_pct, tol_thickness_mm,
                 tol_decenter_mm):
        super().__init__()
        self.model = model
        self.n_trials = n_trials
        self.tol_radius_pct = tol_radius_pct
        self.tol_thickness_mm = tol_thickness_mm
        self.tol_decenter_mm = tol_decenter_mm
        # v5.4 (audit P1-F): polled between trials in the run() loop.
        # Per-trial trace_world is fast (~ms), so per-trial granularity
        # gives a responsive Stop without library-side support.
        self._cancel_progress = CancellableProgress()

    def run(self):
        rms_list = []
        efl_list = []
        rng = np.random.default_rng()
        # 3.7.8: world-frame trace + perturb the WORLD positions of
        # downstream surfaces when a thickness is perturbed (so
        # folded designs see the correct downstream shift along the
        # perturbed gap's local +z axis).  The legacy code only
        # perturbed Surface.thickness, which the world trace ignores
        # entirely -- thickness perturbations would have had zero
        # effect post-3.7.6.  This rewrite makes the Monte Carlo
        # actually tolerance the gaps in the world frame.
        base_world = self.model.build_run_trace_world_surfaces()
        # Legacy local list -- still used for ABCD / paraxial EFL/BFL
        # because the world list has Surface.thickness only carried
        # for compat and system_abcd needs real per-surface
        # thicknesses for the paraxial transfer matrices.
        base_local = self.model.build_trace_surfaces()
        wv = self.model.wavelength_m
        semi_ap = self.model.epd_m / 2.0

        for trial in range(self.n_trials):
            # v5.4 (audit P1-F): poll for cooperative cancel between
            # trials so the user's Stop click is honoured promptly.
            if self._cancel_progress.should_stop:
                break
            self.progress.emit(trial + 1, self.n_trials)
            self.fine_progress.emit(
                trial / max(self.n_trials, 1),
                f'trial {trial + 1}/{self.n_trials}')

            if self.tol_decenter_mm > 0:
                dec_x = rng.normal(0, self.tol_decenter_mm * 1e-3)
                dec_y = rng.normal(0, self.tol_decenter_mm * 1e-3)
            else:
                dec_x = dec_y = 0.0

            # --- Perturb world surfaces ---
            # Copy base world surfaces with fresh world_origin / world_R
            # arrays so this trial's mutations don't leak across trials.
            trial_surfs = []
            for s in base_world:
                trial_surfs.append(Surface(
                    radius=s.radius, conic=s.conic,
                    aspheric_coeffs=s.aspheric_coeffs,
                    semi_diameter=s.semi_diameter,
                    glass_before=s.glass_before,
                    glass_after=s.glass_after,
                    is_mirror=s.is_mirror, thickness=s.thickness,
                    label=s.label, surf_num=s.surf_num,
                    radius_y=s.radius_y, conic_y=s.conic_y,
                    aspheric_coeffs_y=s.aspheric_coeffs_y,
                    world_origin=(s.world_origin.copy()
                                  if s.world_origin is not None
                                  else None),
                    world_R=(s.world_R.copy()
                             if s.world_R is not None else None),
                ))

            # Radius perturbations are per-surface, no downstream shift.
            for s in trial_surfs:
                if np.isfinite(s.radius) and self.tol_radius_pct > 0:
                    s.radius *= (1 + rng.normal(
                        0, self.tol_radius_pct / 100))
                if (s.radius_y is not None
                        and np.isfinite(s.radius_y)
                        and self.tol_radius_pct > 0):
                    s.radius_y *= (1 + rng.normal(
                        0, self.tol_radius_pct / 100))

            # Thickness perturbations: shift all downstream
            # world_origins by delta_t * surface_i.world_R[:,2].
            # Only positive-thickness gaps are perturbed (matches
            # the legacy filter -- skip post-mirror Zemax-signed
            # negative thicknesses to avoid double-counting).
            if self.tol_thickness_mm > 0:
                for i, s in enumerate(trial_surfs[:-1]):
                    if s.thickness > 0 and s.world_R is not None:
                        delta_t = rng.normal(
                            0, self.tol_thickness_mm * 1e-3)
                        shift = delta_t * s.world_R[:, 2]
                        for s2 in trial_surfs[i + 1:]:
                            if s2.world_origin is not None:
                                s2.world_origin = (
                                    s2.world_origin + shift)

            try:
                # ABCD on the legacy local list -- world thickness
                # perturbations are first-order on EFL, so we use
                # the un-perturbed paraxial EFL/BFL as a stable
                # reference for the live histogram.  For a tighter
                # tolerance histogram on EFL itself, a future
                # release can re-derive paraxial from the perturbed
                # world list.
                _, efl, bfl, _ = system_abcd(base_local, wv)
                efl_list.append(efl * 1e3)  # mm

                rays = make_rings(semi_ap, 6, 24, 0.0, wv)
                if dec_x or dec_y:
                    rays.x = rays.x + dec_x
                    rays.y = rays.y + dec_y

                result = trace_world(rays, trial_surfs, wv)
                rms, _ = spot_rms(result)
                rms_list.append(rms * 1e6)
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

        # v5.4 (audit P1-F): emit cancelled before finished so the
        # dock can distinguish a partial run.  finished still fires
        # so the histogram is drawn from the partial sample.
        if self._cancel_progress.should_stop:
            self.cancelled.emit()
        self.finished.emit({
            'rms': np.array(rms_list),
            'efl': np.array(efl_list),
        })

    @Slot()
    def cancel(self):
        self._cancel_progress.cancel()


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
        self.btn_run.setObjectName('run_button')
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run)

        # v5.4 (audit P1-F): Stop button -- cooperative cancel via
        # CancellableProgress polled between MC trials.
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(
            'Cancel the running MC sweep.  The histogram will be '
            'drawn from the trials completed so far.')
        self.btn_stop.clicked.connect(self._stop)
        run_row.addWidget(self.btn_stop)

        # 3.5.9: structured-report export (uses tolerancing_report
        # when Strehl data is available; falls back to a JSON/text
        # summary built from the locally-collected RMS + EFL arrays
        # when only the lightweight MC engine has run).
        self.btn_export = QPushButton('Export Report…')
        self.btn_export.setToolTip(
            'Export a structured Monte Carlo report (JSON or text).\n'
            'Includes per-knob tolerances, per-trial RMS / EFL '
            'arrays, and summary statistics (median, sigma, p95, '
            'yield).  When Strehl data becomes available '
            '(future Lumenairy core-MC integration) the export '
            'will route through la.tolerancing_report for full '
            'Strehl yield curves.')
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip(
            'Run an MC analysis first to enable export.')
        self.btn_export.clicked.connect(self._export_report)
        run_row.addWidget(self.btn_export)
        layout.addLayout(run_row)

        # 3.6: surface the dock's MC limitations up-front so users
        # know the simple ray-trace MC isn't a substitute for the
        # core's Strehl-yield monte_carlo_tolerancing.
        warn = QLabel(
            'Note: this dock runs a lightweight ray-trace MC that '
            'collects RMS spot + EFL only.  Decenter is applied as a '
            'bundle-level lateral shift, not per-element.  For Strehl '
            'yield curves and per-element decenter / tilt / form-error '
            'MC, use lumenairy.monte_carlo_tolerancing in a script.')
        warn.setWordWrap(True)
        warn.setStyleSheet(
            'color: #ffd178; padding: 4px 6px; '
            'border: 1px solid #2a3548; border-radius: 3px;')
        layout.addWidget(warn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Results plot ──
        self.fig = Figure(figsize=(6, 3), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        # v5.4.3 (audit GUI-resize): override matplotlib canvas sizeHint so the dock can shrink
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)

        # ── Summary ──
        self.summary = QLabel('')
        self.summary.setStyleSheet("color: #7a94b8; font-size: 11px;")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def _run(self):
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
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
        # v5.4 (audit P1-F): partial-result still flushed via
        # finished; the cancelled signal is informational so the
        # summary line below can mark the run as user-aborted.
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _stop(self):
        # v5.4 (audit P1-F): cooperative cancel via CancellableProgress.
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.btn_stop.setEnabled(False)
            self.progress_bar.setToolTip('Cancelling...')

    def _on_cancelled(self):
        # The finished handler still fires after cancelled, so just
        # tag the histogram title for the next redraw.
        self._cancelled_by_user = True

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
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._worker = None

        rms = results['rms']
        efl = results['efl']
        valid = ~np.isnan(rms)
        rms = rms[valid]
        efl = efl[~np.isnan(efl)]

        self._draw_histograms(rms, efl)

        # Cache for the structured-report export.  Stash both the
        # filtered arrays and the per-knob tolerances so the report
        # is self-contained and reproducible without keeping a
        # reference to the dock.
        self._last_results = {
            'rms_um': np.asarray(rms),
            'efl_mm': np.asarray(efl),
            'tolerances': {
                'radius_pct_1sigma': float(self.spin_radius.value()),
                'thickness_mm_1sigma': float(self.spin_thick.value()),
                'decenter_mm_1sigma': float(self.spin_decenter.value()),
            },
            'n_trials_requested': int(self.spin_trials.value()),
            'n_trials_converged': int(len(rms)),
            'wavelength_m': float(self.sm.wavelength_m),
            'epd_m': float(self.sm.epd_m),
            'efl_mm_nominal': float(self.sm.efl_mm),
        }
        self.btn_export.setEnabled(len(rms) > 0)
        if len(rms) > 0:
            self.btn_export.setToolTip(
                'Export the last MC results as JSON or text.')

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

    # ── Structured report export (3.5.9) ─────────────────────────────

    def _build_report_dict(self):
        """Build a structured report dict from the cached MC results.

        Includes summary statistics, percentile breakdowns, and the
        full per-trial arrays so downstream tools (or a future
        Strehl-aware path through ``la.tolerancing_report``) can
        re-derive yield curves.
        """
        rs = getattr(self, '_last_results', None)
        if rs is None or len(rs.get('rms_um', [])) == 0:
            return None
        rms = np.asarray(rs['rms_um'])
        efl = np.asarray(rs['efl_mm'])

        def _stats(a, label, units):
            if len(a) == 0:
                return {'label': label, 'units': units}
            return {
                'label': label, 'units': units,
                'mean': float(np.mean(a)),
                'std': float(np.std(a)),
                'median': float(np.median(a)),
                'min': float(np.min(a)),
                'max': float(np.max(a)),
                'p05': float(np.percentile(a, 5)),
                'p95': float(np.percentile(a, 95)),
            }

        return {
            'kind': 'lumenairy_tolerance_report',
            'version': '1',
            'generator': 'lumenairy.ui.ToleranceDock',
            'limitations': [
                'Decenter is applied as a bundle-level lateral '
                'shift; per-element decenter/tilt/form-error MC '
                'requires lumenairy.monte_carlo_tolerancing.',
                'Strehl yield curves are NOT included; this report '
                'is built from a lightweight ray-trace MC engine '
                'that captures RMS spot + EFL only.',
                'The dock does not (yet) consume tolerancing_report; '
                'expected to migrate when the core MC integration '
                'lands.',
            ],
            'tolerances_1sigma': rs['tolerances'],
            'n_trials_requested': rs['n_trials_requested'],
            'n_trials_converged': rs['n_trials_converged'],
            'system': {
                'wavelength_m': rs['wavelength_m'],
                'epd_m': rs['epd_m'],
                'efl_mm_nominal': rs['efl_mm_nominal'],
            },
            'rms_spot': _stats(rms, 'RMS spot size', 'um'),
            'efl': _stats(efl, 'Effective focal length', 'mm'),
            'arrays': {
                'rms_um': rms.tolist(),
                'efl_mm': efl.tolist(),
            },
            'note': (
                'Strehl-based yield curves require the core '
                'monte_carlo_tolerancing path; this report is built '
                'from the lightweight ray-trace MC engine.  Future '
                'releases will route through la.tolerancing_report '
                'when Strehl arrays are available.'),
        }

    def _format_report_text(self, report):
        """Plain-text rendering for human-readable export."""
        lines = []
        lines.append('LumenAiry — Monte Carlo Tolerance Report')
        lines.append('=' * 56)
        tol = report['tolerances_1sigma']
        sys_ = report['system']
        lines.append(f"Trials:    {report['n_trials_converged']} converged "
                     f"of {report['n_trials_requested']} requested")
        lines.append(f"Wavelength: {sys_['wavelength_m'] * 1e9:.1f} nm")
        lines.append(f"EPD:        {sys_['epd_m'] * 1e3:.3f} mm")
        lines.append(f"EFL:        {sys_['efl_mm_nominal']:.3f} mm")
        lines.append('')
        lines.append('Tolerances (1σ):')
        lines.append(f"  Radius:    {tol['radius_pct_1sigma']:.3f} %")
        lines.append(f"  Thickness: {tol['thickness_mm_1sigma']:.4f} mm")
        lines.append(f"  Decenter:  {tol['decenter_mm_1sigma']:.4f} mm")
        lines.append('')
        for key in ('rms_spot', 'efl'):
            s = report[key]
            if 'median' not in s:
                continue
            lines.append(f"{s['label']} [{s['units']}]:")
            lines.append(f"  median {s['median']:.4f}, "
                         f"sigma {s['std']:.4f}")
            lines.append(f"  p05 {s['p05']:.4f}  "
                         f"p95 {s['p95']:.4f}  "
                         f"min {s['min']:.4f}  max {s['max']:.4f}")
            lines.append('')
        lines.append('Note: ' + report['note'])
        return '\n'.join(lines)

    def _export_report(self):
        report = self._build_report_dict()
        if report is None:
            QMessageBox.information(self, 'Export Report',
                                     'No MC data to export.  Run an '
                                     'MC analysis first.')
            return
        path, sel = QFileDialog.getSaveFileName(
            self, 'Export Tolerance Report', 'tolerance_report',
            'JSON (*.json);;Text (*.txt);;All Files (*)')
        if not path:
            return
        try:
            if path.lower().endswith('.txt') or 'Text' in sel:
                if not path.lower().endswith('.txt'):
                    path += '.txt'
                with open(path, 'w', encoding='utf-8') as fp:
                    fp.write(self._format_report_text(report))
            else:
                if not path.lower().endswith('.json'):
                    path += '.json'
                import json as _json
                with open(path, 'w', encoding='utf-8') as fp:
                    _json.dump(report, fp, indent=2)
        except Exception as e:
            QMessageBox.warning(self, 'Export Report',
                                 f'Could not write report:\n{e}')

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
