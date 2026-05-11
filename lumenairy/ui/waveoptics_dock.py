"""
Wave-optics dock — full ASM/Fresnel/Fraunhofer simulation control panel.

Runs coherent wave propagation through the current design with configurable
grid, method, compute backend, output file, plane selection, memory limits,
and pre-run forecasts.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QProgressBar, QGroupBox, QComboBox,
    QTextEdit, QCheckBox, QFileDialog, QLineEdit, QScrollArea,
    QFormLayout, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox,
)
from PySide6.QtGui import QFont

import numpy as np
import time
import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .model import SystemModel


# ════════════════════════════════════════════════════════════════════════
# Forecast helpers
# ════════════════════════════════════════════════════════════════════════

# Hardware self-calibration: rather than hardcode a single 12 ms ASM
# reference, run a small ASM call once on the local box and cache the
# result.  All subsequent forecasts scale from THIS machine's measured
# ASM throughput, so a fast 32-core workstation gets shorter forecasts
# than a 4-core laptop without any user configuration.
#
# The cache is module-global; ``_invalidate_asm_calibration`` lets the
# UI's "Recalibrate" button force a fresh measurement (e.g. after the
# user switches FFT backend from NumPy to pyFFTW).
_CALIBRATED_ASM_MS_AT_1024 = None


def _local_asm_baseline_ms(force=False):
    """Measured ASM-at-N=1024 wall-clock time on this machine, in ms.

    First call times one warmup + one timed ASM at N=512 (faster than
    N=1024, so the calibration adds <300 ms even on slow machines),
    then extrapolates to N=1024 via the standard ``N^2 log N`` cost.
    Subsequent calls hit the cache.

    Pass ``force=True`` to re-time (used by the UI's Recalibrate
    button when, e.g., the user has switched FFT backend or moved
    the process to a different machine via a hibernate / VM
    migration).
    """
    global _CALIBRATED_ASM_MS_AT_1024
    if _CALIBRATED_ASM_MS_AT_1024 is not None and not force:
        return _CALIBRATED_ASM_MS_AT_1024
    try:
        # Local import to avoid a hard dependency at module import time
        # (this file is imported eagerly during QMainWindow construction;
        # we don't want a slow propagation import to block startup).
        from ..propagators.propagation import angular_spectrum_propagate
        N0 = 512
        E = np.ones((N0, N0), dtype=np.complex128)
        # Warmup (FFT-plan caches, JIT, etc.)
        angular_spectrum_propagate(E, 0.01, 1.31e-6, 4e-6)
        t0 = time.perf_counter()
        # Two-call average smooths out one-shot OS jitter
        angular_spectrum_propagate(E, 0.01, 1.31e-6, 4e-6)
        angular_spectrum_propagate(E, 0.01, 1.31e-6, 4e-6)
        dt_512_ms = (time.perf_counter() - t0) / 2.0 * 1000.0
        # FFT cost ~ N^2 log N → N=1024 is 4 * (10/9) ≈ 4.44× N=512.
        scale = (1024 / 512) ** 2 * (np.log2(1024) / np.log2(512))
        _CALIBRATED_ASM_MS_AT_1024 = float(dt_512_ms * scale)
    except Exception:
        # Fallback: if propagation isn't importable for any reason
        # (broken cupy install, etc.), revert to the historical
        # reference 12 ms so the forecast still produces something.
        _CALIBRATED_ASM_MS_AT_1024 = 12.0
    return _CALIBRATED_ASM_MS_AT_1024


def _invalidate_asm_calibration():
    """Force the next ``forecast_resources`` call to re-time ASM."""
    global _CALIBRATED_ASM_MS_AT_1024
    _CALIBRATED_ASM_MS_AT_1024 = None


def forecast_resources(N, n_surfaces, n_save_planes,
                       lens_model='asm', ray_subsample=8,
                       method='asm'):
    """Estimate memory, disk, and time for a wave-optics simulation.

    Recalibrated for library v3.2.x with all of the recent perf
    improvements included:

    * ``apply_real_lens`` — numexpr-fused phase screens (3.1.3),
      pre-resolved glass indices (3.1.11), decenter-aliased
      entrance grids (3.1.3).  Cost is dominated by (N_surf - 1)
      ASM-through-glass FFTs plus a small phase-screen overhead,
      NOT the old "6 FFTs per surface" overestimate.
    * ``apply_real_lens_traced`` — polynomial-Newton default
      (3.1.7, ~12x faster on the hot loop), parallel_amp default
      (3.1.3, amp+amp(pw) overlapped), amplitude-masked Newton
      (3.1.3), ray_subsample=8 default (3.1.7, scales as 1/sub^2).
    * ``apply_real_lens_maslov`` — phase-space propagator added 3.1.7,
      merged into ``lenses`` 3.2.2.  Dominated by 2-D quadrature
      integration; cost scales with N**2 and is weakly surface-count
      dependent (the ray trace is a small fraction).
    * ``scalable_angular_spectrum_propagate`` — added 3.2.0.  Uses a
      3-FFT kernel at the 2N padded grid so per-call cost is about
      5x an N-ASM.
    * Multi-threaded SciPy FFT default (``workers=-1``).

    Ratios to ASM, calibrated against a 3-surface Thorlabs AC254-100-C
    doublet at N=1024, averaged over 2-3 runs:

        Propagator alone (N=1024 free-space step):
          ASM                    1.0  (reference)
          Fresnel                0.8
          Fraunhofer             0.6
          Rayleigh-Sommerfeld    3.3
          SAS                    5.0

        Full optical train (doublet, 3 refracting surfaces, N=1024):
          apply_real_lens               2.2 ASM  total
          apply_real_lens_traced sub=8  22  ASM  total   (3.1.7 default)
          apply_real_lens_traced sub=4  80  ASM  total
          apply_real_lens_maslov        600 ASM  total   (quadrature)

    All coefficients below are in units of "ASM-equivalent time", so
    the formula is hardware-agnostic: pick ``ref_asm_ms`` to match the
    target CPU's actual ASM throughput at N=1024 and the rest
    follows.  Default 12 ms/step is a fast-8-core-SciPy-FFT
    workstation; on a laptop or offscreen sandbox scale up.
    """
    bytes_per_field = N * N * 16  # complex128

    # Peak memory:
    # * base field
    # * scipy FFT plan (2-4x, pocketfft shares buffers)
    # * 2-3 work arrays inside apply_real_lens
    # * amplitude cache for the traced variant
    # * extra Chebyshev / quadrature buffers for Maslov
    # Net: ~6-8x the raw field for the heavier paths, ~5x for analytic.
    if lens_model == 'real_lens_traced':
        mem_mult = 7
    elif lens_model == 'real_lens_maslov':
        mem_mult = 6
    else:
        mem_mult = 5
    peak_memory = bytes_per_field * mem_mult

    # Disk: complex128 planes compress ~0.55-0.65 with gzip/zstd.
    # Use the actual count (not max(count, 1)) so an off-by-one doesn't
    # inflate disk-zero forecasts by 10 MB of "phantom" storage.
    disk_per_plane = int(bytes_per_field * 0.6)
    total_disk = disk_per_plane * max(n_save_planes, 0)

    # --- time model --------------------------------------------------
    # Base per-step FFT cost: measured on the local CPU rather than
    # hardcoded.  ``_local_asm_baseline_ms`` runs a one-time ASM
    # benchmark at N=512 and extrapolates to N=1024; all other
    # coefficients in this function are ratios against that baseline,
    # so a faster (or slower) machine just rescales every prediction
    # without changing the relative cost of different code paths.
    ref_N = 1024
    ref_asm_ms = _local_asm_baseline_ms()
    fft_scale = (N / ref_N) ** 2 * (np.log2(max(N, 2)) / np.log2(ref_N))
    per_fft_sec = (ref_asm_ms * 1e-3) * fft_scale

    # Hardware-speed factor: every non-FFT term (Newton inner loop,
    # array-allocation setup, glass index resolves) was originally
    # calibrated against a 12 ms ASM-1024 reference machine.  On a
    # different machine, those CPU-bound costs scale proportionally,
    # so multiply by (local / reference) to get a faithful prediction.
    # (FFT terms are already scaled via per_fft_sec.)
    _HW_REF_ASM_MS = 12.0
    hw_scale = ref_asm_ms / _HW_REF_ASM_MS

    # Free-space propagator multipliers (ratio to one N-ASM call).
    # Calibrated from a 2026-04 benchmark; Fresnel is actually FASTER
    # than ASM (single FFT, no bandlimit kernel); SAS is 3 FFTs at
    # 2N padded so ~5x an N-ASM.
    method_mult = {'asm': 1.0,
                   'fresnel': 0.8,
                   'fraunhofer': 0.6,
                   'rayleigh-sommerfeld': 3.3,
                   'sas': 5.0,
                   # MFT siblings: between-element steps fall back to
                   # the natural-grid base method; only the to-focus
                   # call hits the Bluestein chirp-Z path, so the
                   # per-surface cost matches the base method to
                   # within FFT scheduling jitter.
                   'asm-mft': 1.0,
                   'fresnel-mft': 0.8,
                   'fraunhofer-mft': 0.6}.get(method, 1.0)

    # Per-surface / per-system cost depends on which lens path is used.
    if lens_model == 'real_lens_traced':
        # Amplitude pass calls apply_real_lens twice (main + plane-
        # wave reference).  With parallel_amp=True (default 3.1.3+)
        # these overlap ~1.7x on a multi-core machine, so effective
        # amp cost ≈ 1.2 * apply_real_lens rather than 2x.
        analytic_cost = per_fft_sec * _apply_real_lens_asm_equiv(n_surfaces)
        amp_cost = 1.2 * analytic_cost

        # Newton inversion: polynomial fit is the default since 3.1.7
        # and runs ~2-3x faster than the old RectBivariateSpline path
        # (with combined value+gradient eval + optional Numba jit).
        # Calibrate at 6 us per pixel for the polynomial hot loop on
        # the 12 ms-ASM reference machine; scale to local HW.
        launch_N = max(16, N // max(1, ray_subsample))
        newton_cost = 6.0e-6 * launch_N * launch_N * hw_scale

        # Setup: scatter + polynomial fit + glass-interval prep.
        # Smaller than the old spline path which had a ~0.15 s base.
        # Same hw_scale applies (CPU-bound).
        setup_cost = (0.05 + 0.012 * max(n_surfaces, 1)) * hw_scale

        total_lens_time = amp_cost + newton_cost + setup_cost
        # Traced pipeline credits the full pipeline once (not
        # per-surface — the lens-router delegates the whole chain).
        time_for_lens = total_lens_time
    elif lens_model == 'real_lens':
        # Analytic apply_real_lens: numexpr-fused phase screens +
        # (n_surfaces - 1) ASM-through-glass calls.  Calibration on
        # a doublet (3 surfaces) gives ~2.2 ASM total, scaling ~1.0
        # ASM per inter-surface glass gap plus ~0.3 ASM per phase
        # screen.
        time_for_lens = per_fft_sec * _apply_real_lens_asm_equiv(n_surfaces)
    elif lens_model == 'real_lens_maslov':
        # Phase-space Maslov: ray trace + 4-D Chebyshev fit + 2-D
        # quadrature integration.  The quadrature dominates; cost is
        # almost N**2 * n_v2**2 * poly_order** with n_v2=32 default.
        # Benchmark on N=1024 doublet defaults = ~600 × ASM_time.
        # Scales with N^2 like any other full-grid op and has a small
        # fixed ray-trace setup.
        setup_cost = (0.08 + 0.012 * max(n_surfaces, 1)) * hw_scale
        time_for_lens = per_fft_sec * 600.0 + setup_cost
    else:
        # Pure propagation (user placed phase screens manually).  For
        # this branch, the "lens" isn't really a thing; each surface
        # corresponds to one propagation step plus a phase multiply.
        per_step = per_fft_sec * method_mult
        time_for_lens = per_step * max(n_surfaces, 1)

    # Total = lens time + in-between free-space propagations.  For
    # the ASM / pure-propagation branch the surface loop already
    # covers everything; for lens-router branches the model already
    # accounts for the full system so we just add the post-lens
    # propagate-to-focus leg.
    total_time = time_for_lens
    if lens_model in ('real_lens_traced', 'real_lens', 'real_lens_maslov'):
        total_time += per_fft_sec * method_mult  # to-focus leg

    # Add I/O: saving a plane is ~40 ms fixed + 80 ns/byte on a fast SSD.
    # Only when the caller actually wants planes saved.
    if n_save_planes > 0:
        total_time += 0.04 * n_save_planes
        total_time += 80e-9 * disk_per_plane * n_save_planes

    return peak_memory, total_disk, total_time


def _apply_real_lens_asm_equiv(n_surfaces):
    """ASM-equivalent time for one ``apply_real_lens`` call.

    Empirically: 2 surfaces -> 1.1 ASM, 3 surfaces -> 2.2 ASM.
    Fits a simple (n - 1) glass propagations + per-surface phase-
    screen overhead model.
    """
    n = max(int(n_surfaces), 1)
    return max(n - 1, 1) * 1.0 + 0.2 * n


def format_bytes(n):
    """Human-readable byte count."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def format_time(seconds):
    """Human-readable time."""
    if seconds < 1:
        return f'{seconds*1000:.0f} ms'
    elif seconds < 60:
        return f'{seconds:.1f} sec'
    elif seconds < 3600:
        return f'{seconds/60:.1f} min'
    else:
        return f'{seconds/3600:.1f} hr'


# ════════════════════════════════════════════════════════════════════════
# Worker thread
# ════════════════════════════════════════════════════════════════════════

class WaveOpticsWorker(QThread):
    """Run wave-optics propagation in a background thread."""
    progress = Signal(int, int, str)          # step, total, label (coarse)
    fine_progress = Signal(float, str)        # overall fraction, stage msg
    finished = Signal(object)                 # results dict

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config  # dict with all config
        # Per-stage (lo, hi) window for fine_progress so a sub-callback
        # inside apply_real_lens_traced can report 0-1 locally and we
        # map it into the overall timeline.
        self._stage_lo = 0.0
        self._stage_hi = 1.0

    def _set_stage(self, lo, hi):
        self._stage_lo = lo
        self._stage_hi = hi

    def _core_progress(self, stage, frac, msg=''):
        """Adapter so core functions can drive our fine_progress signal."""
        overall = self._stage_lo + (self._stage_hi - self._stage_lo) * frac
        self.fine_progress.emit(overall,
                                f'{stage}: {msg}' if msg else stage)

    def run(self):
        from ..propagators.propagation import (
            angular_spectrum_propagate,
            fresnel_propagate, fraunhofer_propagate,
            rayleigh_sommerfeld_propagate,
            angular_spectrum_propagate_mft,
            fresnel_propagate_mft, fraunhofer_propagate_mft,
            apply_fresnel_curvature,
            resample_field, PYFFTW_AVAILABLE, CUPY_AVAILABLE,
        )
        from ..elements.lenses import surface_sag_general
        from ..glass import get_glass_index
        from ..analysis import beam_d4sigma, beam_power

        cfg = self.cfg
        N = cfg['N']
        dx = cfg['dx_m']
        wv = self.model.wavelength_m
        method = cfg['method']  # 'asm', 'fresnel', 'fraunhofer', etc.
        bandlimit = cfg.get('bandlimit', True)
        # MFT methods: between-element steps fall back to the
        # natural-grid base method; only the to-focus step uses the
        # arbitrary-output-grid Bluestein chirp-Z.  Keep the focal-zoom
        # config alongside the dispatch so all three to-focus branches
        # see consistent parameters.
        mft_methods = ('asm-mft', 'fresnel-mft', 'fraunhofer-mft')
        is_mft = method in mft_methods
        if is_mft:
            base_method = {
                'asm-mft': 'asm',
                'fresnel-mft': 'fresnel',
                'fraunhofer-mft': 'fraunhofer',
            }[method]
        else:
            base_method = method
        mft_dx_out = cfg.get('mft_dx_out_m', dx)
        mft_N_out = int(cfg.get('mft_N_out', N))
        mft_centre = cfg.get('mft_centre_out_m', (0.0, 0.0))
        chief_relative_focal = bool(cfg.get('chief_relative_focal', False))
        use_gpu = cfg.get('use_gpu', False)
        output_path = cfg.get('output_path', '')
        save_plane_flags = cfg.get('save_planes', {})  # {label: bool}
        start_idx = cfg.get('start_elem', 0)
        end_idx = cfg.get('end_elem', len(self.model.elements) - 1)

        # Apply memory limit
        mem_limit = cfg.get('memory_limit_gb')
        if mem_limit and mem_limit > 0:
            from ..memory import set_max_ram
            set_max_ram(mem_limit)

        # Set FFT backend
        backend = cfg.get('backend', 'numpy')
        import lumenairy.propagation as _prop
        _prop.USE_PYFFTW = False
        _prop.USE_SCIPY_FFT = False
        if backend == 'pyfftw':
            _prop.USE_PYFFTW = True
        elif backend == 'scipy':
            _prop.USE_SCIPY_FFT = True

        trace_surfs = self.model.build_trace_surfaces()
        if not trace_surfs:
            self.finished.emit({'error': 'No optical surfaces.'})
            return

        total_steps = len(trace_surfs) + 3
        step = 0
        results = {}
        t_start = time.time()

        try:
            # ── Step 1: Create source field ──
            step += 1
            self.progress.emit(step, total_steps, 'Creating source field')

            # Precision: complex64 halves memory and gives ~2x FFT / phase-
            # screen throughput; library functions preserve this dtype
            # end-to-end and apply mod-2pi kernel-phase reduction so the
            # only residual cost is the FFT's single-precision floor.
            precision = cfg.get('precision', 'complex128')
            cdtype = (np.complex64 if precision == 'complex64'
                      else np.complex128)

            x = (np.arange(N) - N / 2) * dx
            X, Y = np.meshgrid(x, x)
            R_sq = X ** 2 + Y ** 2
            epd_m = self.model.epd_m

            # Source construction (3.5.9): prefer the Source factories
            # (3.5.0) via SourceDefinition.to_source so the dock's
            # source path stays in lockstep with the rest of the
            # library's source story.  Source.gaussian, plane_wave,
            # and point_source all return Source instances; we copy
            # E off the wrapper and recast to the user-selected
            # precision (Source factories build complex128 so the
            # dtype control still happens here).  Falls back to the
            # legacy hand-rolled construction if to_source raises
            # (e.g. an unknown source_type added in a future model
            # version).
            src = self.model.source
            try:
                if src is None:
                    raise ValueError('no source defined')
                src_inst = src.to_source(N=N, dx_m=dx, epd_m=epd_m)
                E = np.asarray(src_inst.E, dtype=cdtype)
            except Exception:
                E = np.ones((N, N), dtype=cdtype)
                if src and src.source_type == 'gaussian':
                    w0 = src.beam_diameter_mm * 1e-3 / 2
                    E = np.exp(-(R_sq) / (w0 ** 2)).astype(cdtype)
                elif src and src.source_type == 'gaussian_aperture':
                    sigma = src.sigma_mm * 1e-3
                    E = np.exp(-R_sq / (2 * sigma ** 2)).astype(cdtype)
                else:
                    # Plane wave clipped by EPD
                    E[R_sq > (epd_m / 2) ** 2] = 0.0

            # Off-axis field angle: apply linear phase tilt to the source.
            # Direction cosines: kx = k0 * sin(theta_x), ky = k0 * sin(theta_y).
            # Carrier phase = exp(i * (kx*X + ky*Y)).
            #
            # Note: Source.plane_wave already accepts angle_x/angle_y
            # and the to_source() path forwards these.  But the
            # gaussian / gaussian_aperture / point_source factories
            # don't carry tilt, so we still apply the carrier here
            # uniformly for non-plane-wave types so the angle setting
            # always behaves the same way.
            if (src is not None
                    and src.source_type != 'plane_wave'
                    and (src.field_angle_x_deg
                         or src.field_angle_y_deg)):
                k0 = 2 * np.pi / wv
                kx = k0 * np.sin(np.radians(src.field_angle_x_deg))
                ky = k0 * np.sin(np.radians(src.field_angle_y_deg))
                E = E * np.exp(1j * (kx * X + ky * Y))

            planes = []
            current_dx = dx

            def maybe_save(label, field, z):
                if save_plane_flags.get(label, True):
                    planes.append({'label': label, 'field': field.copy(),
                                   'dx': current_dx, 'z': z})

            maybe_save('Source', E, 0.0)
            z_cum = 0.0

            # ── Step 2a: lens-model router ─────────────────────────────
            # If the user asked for apply_real_lens[_traced] we delegate
            # the ENTIRE optical train to the core function, then
            # propagate from the exit vertex to focus below.  This is
            # mutually exclusive with the per-surface inline loop.
            lens_model = cfg.get('lens_model', 'asm')
            ray_sub = int(cfg.get('ray_subsample', 1))
            used_lens_router = False

            if lens_model in ('real_lens', 'real_lens_traced',
                              'real_lens_maslov') and trace_surfs:
                from ..elements.lenses import (apply_real_lens,
                                       apply_real_lens_traced,
                                       apply_real_lens_maslov)
                pres = self.model.to_prescription()
                # Per-function kwarg overrides chosen via the &Options
                # menu's Lens Options dialog.  Only kwargs the user
                # actually changed are present; library defaults apply
                # for everything else.
                opts_all = getattr(self.model, 'lens_options', {}) or {}
                # Allocate 70% of the bar to the lens call (dominant
                # cost when traced is selected).
                self._set_stage(lo=step / total_steps,
                                hi=(step + 0.7 * len(trace_surfs)) / total_steps)
                try:
                    if lens_model == 'real_lens_traced':
                        opts = dict(opts_all.get(
                            'apply_real_lens_traced', {}))
                        # Honour the dock's own controls when the dialog
                        # didn't override them (the dock is the
                        # primary source for ray_subsample +
                        # tilt_aware_rays so users see them in the
                        # main view).
                        opts.setdefault('bandlimit', True)
                        opts.setdefault('ray_subsample', ray_sub)
                        opts.setdefault('tilt_aware_rays',
                                         cfg.get('tilt_aware_rays', True))
                        E = apply_real_lens_traced(
                            E, pres, wv, current_dx,
                            progress=self._core_progress, **opts)
                    elif lens_model == 'real_lens_maslov':
                        opts = dict(opts_all.get(
                            'apply_real_lens_maslov', {}))
                        E = apply_real_lens_maslov(
                            E, pres, wv, current_dx, **opts)
                    else:
                        opts = dict(opts_all.get('apply_real_lens', {}))
                        opts.setdefault('bandlimit', True)
                        E = apply_real_lens(
                            E, pres, wv, current_dx,
                            progress=self._core_progress, **opts)
                    used_lens_router = True
                    step += len(trace_surfs)   # credit all surfaces at once
                    # Sum thicknesses so the focus step has a correct z.
                    z_cum += float(sum(
                        p_thk for p_thk in pres.get('thicknesses', [])))
                    # Save the exit plane as a single "LensExit" plane.
                    maybe_save('LensExit', E, z_cum)
                except Exception as e:
                    try:
                        from .diagnostics import diag
                        diag.report('waveoptics-lens-router', e,
                                    context=f'lens_model={lens_model}')
                    except Exception:
                        pass
                    used_lens_router = False

            # ── Step 2b: per-surface inline pipeline (fallback / default)
            # Surfaces cover the [1/total .. (total-2)/total] band of
            # overall progress (source was step 1, focus+analysis the
            # last two).
            n_surf = max(1, len(trace_surfs))

            # 3.6: whole-prescription propagators (GBD, HFPI,
            # Huygens-Fresnel, Subaperture) take the full prescription
            # rather than per-surface fields, so we short-circuit the
            # per-element loop and call them directly.  Result is the
            # focal-plane field (output_dx defaulting to dx_in).
            whole_prescription = {
                'gbd', 'hfpi', 'huygens-fresnel', 'subaperture',
            }
            if method in whole_prescription:
                try:
                    pres = self.model.to_prescription()
                except Exception as exc:
                    self.finished.emit({
                        'error': f'Cannot export prescription: {exc}'})
                    return
                step = total_steps - 2
                self.progress.emit(step, total_steps,
                                    f'Running {method} (whole-prescription)')
                try:
                    if method == 'gbd':
                        from ..propagators.propagation import (
                            propagate_gbd_through_prescription)
                        E_focus = propagate_gbd_through_prescription(
                            E, dx, pres, wavelength=wv)
                    elif method == 'hfpi':
                        from ..propagators.propagation import (
                            propagate_hfpi_through_prescription)
                        E_focus = propagate_hfpi_through_prescription(
                            E, dx, pres, wavelength=wv,
                            n_paths=cfg.get('hfpi_n_paths', 4096))
                    elif method == 'huygens-fresnel':
                        from ..propagators.propagation import (
                            propagate_huygens_fresnel_through_prescription)
                        E_focus = (
                            propagate_huygens_fresnel_through_prescription(
                                E, dx, pres, wavelength=wv))
                    elif method == 'subaperture':
                        from ..propagators.propagation import (
                            propagate_subaperture_asymptotic)
                        E_focus = propagate_subaperture_asymptotic(
                            E, dx, pres, wavelength=wv)
                except Exception as exc:
                    self.finished.emit({
                        'error': f'{method} failed: '
                                 f'{type(exc).__name__}: {exc}'})
                    return
                current_dx = dx
                # Build a minimal "planes" record so the rest of the
                # finalisation code (intensity, beam D4-sigma, save)
                # works unchanged.
                planes = [{
                    'label': f'{method.upper()} focus',
                    'field': E_focus, 'dx': current_dx, 'z': 0.0,
                }]
                I_focus = np.abs(E_focus) ** 2
                from ..analysis import beam_d4sigma, beam_power
                power_in = beam_power(E, dx)
                power_focus = beam_power(E_focus, current_dx)
                try:
                    dx_b, dy_b = beam_d4sigma(E_focus, current_dx)
                    d4sig = (dx_b + dy_b) / 2
                except Exception:
                    d4sig = 0
                elapsed = time.time() - t_start
                self.progress.emit(total_steps, total_steps, 'done')
                results.update({
                    'planes': planes,
                    'I_focus': I_focus,
                    'dx': current_dx,
                    'wavelength': wv,
                    'power_in': power_in,
                    'power_focus': power_focus,
                    'peak_intensity': np.max(I_focus),
                    'd4sigma': d4sig,
                    'N': N,
                    'elapsed': elapsed,
                    'n_planes_saved': len(planes),
                    'output_path': '',
                    'propagation_result': None,
                })
                self.finished.emit(results)
                return

            if used_lens_router:
                trace_surfs = []   # skip the inline loop
            else:
                pass
            for i, ts in enumerate(trace_surfs):
                step += 1
                self.progress.emit(step, total_steps,
                                   f'Surface {i+1}/{len(trace_surfs)}: {ts.label}')
                # Fine-grained fraction in [0, 1] across the whole run.
                self._set_stage(
                    lo=(step - 1) / total_steps,
                    hi=step / total_steps,
                )
                self._core_progress(
                    'surface', 0.0,
                    f'{i + 1}/{n_surf} {ts.label}')

                # Refraction phase screen
                n1 = get_glass_index(ts.glass_before, wv)
                n2 = get_glass_index(ts.glass_after, wv)

                if abs(n2 - n1) > 1e-10 and np.isfinite(ts.radius):
                    h_sq = X ** 2 + Y ** 2
                    sag = surface_sag_general(h_sq, ts.radius, ts.conic)
                    k = 2 * np.pi / wv
                    phase = -k * (n2 - n1) * sag
                    E = E * np.exp(1j * phase)

                # Aperture
                if np.isfinite(ts.semi_diameter):
                    E[R_sq > ts.semi_diameter ** 2] = 0.0

                # Propagate through thickness
                if ts.thickness > 0:
                    n_med = n2 if n2 > 1 else 1.0
                    lam_med = wv / n_med

                    # Between-surface dispatch.  MFT methods use the
                    # natural-grid base method here -- the MFT chirp-Z
                    # is only applied at the to-focus step below.
                    if base_method == 'fresnel':
                        E, dx_new, _ = fresnel_propagate(
                            E, ts.thickness, lam_med, current_dx)
                        if abs(dx_new - current_dx) > current_dx * 1e-6:
                            E, _ = resample_field(E, dx_new, current_dx,
                                                   N_out=N)
                    elif base_method == 'fraunhofer' and i == len(trace_surfs) - 1:
                        E, dx_new, _ = fraunhofer_propagate(
                            E, ts.thickness, lam_med, current_dx)
                        current_dx = dx_new
                    elif base_method == 'rayleigh-sommerfeld':
                        E = rayleigh_sommerfeld_propagate(
                            E, ts.thickness, lam_med, current_dx,
                            bandlimit=bandlimit, use_gpu=use_gpu)
                    elif base_method == 'sas':
                        from ..propagators.propagation import (
                            scalable_angular_spectrum_propagate)
                        E, dx_new, _ = scalable_angular_spectrum_propagate(
                            E, ts.thickness, lam_med, current_dx,
                            use_gpu=use_gpu)
                        if abs(dx_new - current_dx) > current_dx * 1e-6:
                            E, _ = resample_field(E, dx_new, current_dx,
                                                   N_out=N)
                    else:
                        E = angular_spectrum_propagate(
                            E, ts.thickness, lam_med, current_dx,
                            bandlimit=bandlimit, use_gpu=use_gpu)

                    z_cum += ts.thickness

                maybe_save(ts.label, E, z_cum)

            # ── Step 3: Propagate to focus ──
            step += 1
            self.progress.emit(step, total_steps, 'Propagating to focus')

            bfl_mm = self.model.bfl_mm
            if np.isfinite(bfl_mm) and bfl_mm > 0:
                bfl_m = bfl_mm * 1e-3
                if is_mft:
                    # MFT focal-zoom dispatch (3.5.7).  Output grid
                    # entirely user-specified -- decoupled from input
                    # grid and propagation distance.
                    if method == 'fraunhofer-mft':
                        E_focus = fraunhofer_propagate_mft(
                            E, bfl_m, wv, current_dx, mft_dx_out,
                            mft_N_out, centre_out=mft_centre)
                    elif method == 'fresnel-mft':
                        E_focus = fresnel_propagate_mft(
                            E, bfl_m, wv, current_dx, mft_dx_out,
                            mft_N_out, centre_out=mft_centre)
                    else:  # 'asm-mft'
                        E_focus = angular_spectrum_propagate_mft(
                            E, bfl_m, wv, current_dx, mft_dx_out,
                            mft_N_out, centre_out=mft_centre,
                            bandlimit=bandlimit, use_gpu=use_gpu)
                    current_dx = mft_dx_out
                elif method == 'fraunhofer':
                    E_focus, dx_focus, _ = fraunhofer_propagate(
                        E, bfl_m, wv, current_dx)
                    current_dx = dx_focus
                elif method == 'fresnel':
                    E_focus, dx_focus, _ = fresnel_propagate(
                        E, bfl_m, wv, current_dx)
                    current_dx = dx_focus
                elif method == 'rayleigh-sommerfeld':
                    E_focus = rayleigh_sommerfeld_propagate(
                        E, bfl_m, wv, current_dx,
                        bandlimit=bandlimit, use_gpu=use_gpu)
                elif method == 'sas':
                    from ..propagators.propagation import (
                        scalable_angular_spectrum_propagate)
                    E_focus, dx_focus, _ = scalable_angular_spectrum_propagate(
                        E, bfl_m, wv, current_dx, use_gpu=use_gpu)
                    current_dx = dx_focus
                else:
                    E_focus = angular_spectrum_propagate(
                        E, bfl_m, wv, current_dx,
                        bandlimit=bandlimit, use_gpu=use_gpu)
                z_cum += bfl_m

                # Optional 3.5.7 chief-relative-OPD conversion on the
                # focal field.  Bridges the absolute-phase convention
                # used by Lumenairy / Fresnel / ASM family against the
                # ray-trace-rooted form used by OPDPy and Zemax OPD
                # operands.  Skipped on MFT-Fraunhofer (the natural
                # output is already a far-field amplitude where this
                # conversion is ill-defined).
                # 3.6: optional detector model (applied to E_focus).
                if cfg.get('detector_apply', False):
                    try:
                        from ..detector import apply_detector
                        E_focus = apply_detector(
                            E_focus, current_dx,
                            pixel_pitch=cfg['detector_pixel_um'] * 1e-6,
                            quantum_efficiency=cfg['detector_qe'],
                            read_noise_e=cfg['detector_read_noise_e'],
                            dark_current_e_per_s=cfg['detector_dark_e_per_s'],
                            exposure_time=cfg['detector_exposure_s'])
                    except Exception:
                        pass

                if (chief_relative_focal
                        and method != 'fraunhofer'
                        and method != 'fraunhofer-mft'):
                    efl_mm = float(self.model.efl_mm)
                    if np.isfinite(efl_mm) and efl_mm > 0:
                        R = (bfl_mm - efl_mm) * 1e-3
                        if abs(R) > 1e-9:
                            E_focus = apply_fresnel_curvature(
                                E_focus, current_dx, wv, R=R, sign=-1)

                maybe_save('Focus', E_focus, z_cum)
            else:
                E_focus = E

            # ── Step 4: Analysis ──
            step += 1
            self.progress.emit(step, total_steps, 'Computing analysis')

            I_focus = np.abs(E_focus) ** 2
            power_in = beam_power(planes[0]['field'] if planes else E, dx)
            power_focus = beam_power(E_focus, current_dx)

            try:
                dx_b, dy_b = beam_d4sigma(E_focus, current_dx)
                d4sig = (dx_b + dy_b) / 2
            except Exception:
                d4sig = 0

            elapsed = time.time() - t_start

            # ── Save to file ──
            if output_path:
                self.progress.emit(step, total_steps, f'Saving to {os.path.basename(output_path)}')
                try:
                    ext = os.path.splitext(output_path)[1].lower()
                    if ext == '.zarr':
                        from ..io.storage import set_storage_backend, append_plane, write_metadata
                        set_storage_backend('zarr')
                    else:
                        from ..io.storage import set_storage_backend, append_plane, write_metadata
                        set_storage_backend('hdf5')

                    for p in planes:
                        append_plane(output_path, p['field'], p['dx'],
                                     z=p['z'], label=p['label'])
                    write_metadata(output_path, {
                        'wavelength': wv,
                        'grid_N': N,
                        'dx': dx,
                        'method': method,
                        'n_planes': len(planes),
                    })
                except Exception as e:
                    results['save_error'] = str(e)

            # 3.5.9: also surface the run as a PropagationResult
            # (3.5.0 unified return type) so downstream consumers
            # (other docks, scripted callers) can use the same
            # wrapper they get from la.propagate().  Existing
            # consumers of `results['planes']`, `results['I_focus']`,
            # etc. continue to work untouched.
            try:
                from ..propagators import PropagationResult
                history = [(p['label'], p['field'], p['z'])
                           for p in planes]
                propagation_result = PropagationResult(
                    field=E_focus, dx=current_dx, wavelength=wv,
                    z=z_cum, method=method,
                    history=history,
                    metadata={
                        'N': N, 'dx_in': dx,
                        'lens_model': cfg.get('lens_model', 'asm'),
                        'bandlimit': bandlimit,
                        'is_mft': is_mft,
                        'mft_dx_out_m': mft_dx_out if is_mft else None,
                        'mft_N_out': mft_N_out if is_mft else None,
                        'mft_centre_out_m':
                            mft_centre if is_mft else None,
                        'chief_relative_focal':
                            chief_relative_focal,
                        'power_in': power_in,
                        'power_focus': power_focus,
                        'peak_intensity': float(np.max(I_focus)),
                        'd4sigma': d4sig,
                        'elapsed_sec': elapsed,
                    },
                    intermediates=None)
            except Exception:
                propagation_result = None

            results.update({
                'planes': planes,
                'I_focus': I_focus,
                'dx': current_dx,
                'wavelength': wv,
                'power_in': power_in,
                'power_focus': power_focus,
                'peak_intensity': np.max(I_focus),
                'd4sigma': d4sig,
                'N': N,
                'elapsed': elapsed,
                'n_planes_saved': len(planes),
                'output_path': output_path,
                'propagation_result': propagation_result,
            })

        except Exception as e:
            results = {'error': str(e)}

        self.finished.emit(results)


# ════════════════════════════════════════════════════════════════════════
# Dock widget
# ════════════════════════════════════════════════════════════════════════

class WaveOpticsDock(QWidget):
    """Production-grade wave-optics simulation control panel."""

    # Emitted after a successful run with the full results dict so other
    # docks (e.g. Zernike) can pick up the focal-plane field.
    run_finished = Signal(object)

    def __init__(self, system_model: SystemModel, parent=None):
        super().__init__(parent)
        self.sm = system_model
        self._worker = None

        # ── Tabbed layout (3.5.9) ──
        # The existing per-element propagation flow lives in the
        # "Per-element propagation" tab; the new "Custom MHS chain"
        # tab exposes the Multi-Huygens-Surface framework
        # (3.5.0) for advanced subdomain chaining via
        # MhsPipeline.from_prescription / .run.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        scroll.setWidget(inner)
        self._tabs.addTab(scroll, 'Per-element propagation')

        # MHS-chain tab is built lazily after the per-element tab is
        # populated so we can reuse the dock's N / dx / save settings
        # without duplicating widgets.
        self._mhs_tab = None  # filled in _build_mhs_tab() at end of __init__

        # ── Quick-run presets (3.6) ──
        # One-click sane defaults for the three most common runs.
        # Each preset writes a complete config (N / dx / method /
        # lens_model / precision / bandlimit) so a new user can
        # press Run with confidence immediately.  Power users can
        # then tweak below.
        quick_group = QGroupBox('Quick run')
        quick_layout = QHBoxLayout(quick_group)
        quick_layout.setContentsMargins(6, 6, 6, 6)
        for label, key, tip in [
            ('Fast preview', 'fast',
             'N=512, dx=4 µm, ASM phase-screen, complex64.\n'
             'Sub-second on 1k grids.  Fine for sanity checks.'),
            ('Production', 'production',
             'N=1024, dx=2 µm, apply_real_lens, ASM, complex128, '
             'bandlimit=ON.  Default for design reviews.'),
            ('Sub-nm validation', 'validation',
             'N=2048, dx=1 µm, apply_real_lens_traced sub=1, '
             'complex128.  Sub-nm OPD; minutes per run.'),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _c=False, k=key:
                                self._apply_quick_preset(k))
            quick_layout.addWidget(btn)
        layout.addWidget(quick_group)

        # ── Simulation Parameters ──
        sim_group = QGroupBox('Simulation Parameters')
        sim_layout = QFormLayout(sim_group)

        row_grid = QHBoxLayout()
        self.spin_N = QComboBox()
        # Powers of 2, plus 2^a*3 and 2^a*5 up to 131072
        n_values = sorted(set(
            [2**a * m for a in range(7, 18) for m in (1, 3, 5)
             if 2**a * m <= 131072]
        ))
        for n in n_values:
            self.spin_N.addItem(str(n), n)
        self.spin_N.setCurrentText('1024')
        self.spin_N.currentIndexChanged.connect(self._update_forecast)
        row_grid.addWidget(QLabel('N:'))
        row_grid.addWidget(self.spin_N)

        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(0.001, 500)
        self.spin_dx.setValue(2.0)
        self.spin_dx.setDecimals(3)
        self.spin_dx.setSuffix(' um')
        self.spin_dx.valueChanged.connect(self._update_forecast)
        row_grid.addWidget(QLabel('dx:'))
        row_grid.addWidget(self.spin_dx)
        sim_layout.addRow(row_grid)

        row_method = QHBoxLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(['ASM', 'Fresnel', 'Fraunhofer',
                                    'Rayleigh-Sommerfeld', 'SAS',
                                    'Fresnel MFT', 'Fraunhofer MFT',
                                    'ASM MFT',
                                    # 3.6: standalone propagators
                                    # exposed at the dock level.
                                    'GBD',
                                    'HFPI',
                                    'Huygens-Fresnel',
                                    'Subaperture'])
        self.combo_method.setToolTip(
            'Free-space propagator used BETWEEN elements (MFT variants '
            'are applied at the FOCAL plane only; between-surface steps '
            'fall back to the corresponding base method on the natural '
            'grid).\n'
            '  ASM:     exact band-limited, fixed grid (default).\n'
            '  Fresnel: single-FFT paraxial; auto-resampled back to dx.\n'
            '  Fraunhofer: far-field only (last surface).\n'
            '  R-S:     Rayleigh-Sommerfeld convolution (slowest).\n'
            '  SAS:     Scalable Angular Spectrum (Heintzmann 2023); '
            'right for long z where plain ASM needs too many samples. '
            'Auto-resampled back to dx.\n'
            '  Fresnel MFT / Fraunhofer MFT / ASM MFT (3.5.7): same '
            'physics as their non-MFT siblings, but the focal plane is '
            'sampled on a user-specified grid (dx_out, N_out, '
            'centre_out) via Bluestein chirp-Z.  Standard tool for '
            'focal-plane zoom.')
        self.combo_method.currentIndexChanged.connect(self._update_forecast)
        self.combo_method.currentIndexChanged.connect(
            self._update_mft_visibility)
        row_method.addWidget(QLabel('Method:'))
        row_method.addWidget(self.combo_method)
        # 3.7.9: standalone "Use MFT" checkbox + "Options" button.
        # Picks the *-mft variant of the chosen base propagator
        # (ASM / Fresnel / Fraunhofer) without forcing users to
        # find the "ASM MFT" etc. entries in the long Method
        # dropdown.  The Options button exposes the same dx_out /
        # N_out / centre fields that 3.5.7 had as a always-inline
        # group -- now collapsed behind a dialog so the dock isn't
        # cluttered when MFT is off.
        self.chk_mft = QCheckBox('Use MFT')
        self.chk_mft.setToolTip(
            'Route the focal-plane step through the Matrix Fourier '
            'Transform (Bluestein chirp-Z) variant of the selected '
            'method.  Lets you sample the focal plane on a user-'
            'specified grid (dx_out, N_out, centre) decoupled from '
            'the input grid and propagation distance -- standard '
            'tool for focal-plane zoom on high-NA systems.\n\n'
            'Applies to ASM, Fresnel, and Fraunhofer base methods. '
            'No-op for SAS / R-S / GBD / HFPI / Huygens-Fresnel / '
            'Subaperture, which have no MFT analogue.')
        self.chk_mft.toggled.connect(self._update_forecast)
        self.chk_mft.toggled.connect(self._update_mft_visibility)
        row_method.addWidget(self.chk_mft)
        self.btn_mft_options = QPushButton('Options\u2026')
        self.btn_mft_options.setToolTip(
            'MFT focal-zoom options: output pixel pitch, output '
            'grid size, output centre.  Only relevant when "Use MFT" '
            'is checked and the base method supports MFT.')
        self.btn_mft_options.clicked.connect(
            self._open_mft_options_dialog)
        row_method.addWidget(self.btn_mft_options)
        self.btn_recommend = QPushButton('Recommend')
        self.btn_recommend.setToolTip(
            'Auto-size N and dx from system NA, aperture, and the '
            'OPD-Nyquist rule (dx \u2264 \u03bb*f/aperture).')
        self.btn_recommend.clicked.connect(self._recommend_grid)
        row_method.addWidget(self.btn_recommend)
        sim_layout.addRow(row_method)

        # \u2500\u2500 Bandlimit checkbox (3.5.8): Matsushima cutoff applied to
        # ASM / RS / ASM-MFT transfer functions.  Default ON matches
        # the historical hardcoded behaviour for ASM; on Rayleigh-
        # Sommerfeld the core library defaults to OFF, but exposing
        # the toggle here lets users opt in at the dock level.
        row_bandlimit = QHBoxLayout()
        self.chk_bandlimit = QCheckBox('Bandlimit (Matsushima)')
        self.chk_bandlimit.setChecked(True)
        self.chk_bandlimit.setToolTip(
            'Apply the Matsushima-Shimobaba frequency cutoff to the '
            'propagator transfer function.  Suppresses aliasing on '
            'coarse grids at long propagation distances.  Default ON.\n'
            'Affects ASM, ASM-MFT, and Rayleigh-Sommerfeld between-'
            'element + to-focus calls.  Fresnel/Fraunhofer ignore this '
            'flag (their kernels are not band-limit-able in the same '
            'way).')
        self.chk_bandlimit.toggled.connect(self._update_forecast)
        row_bandlimit.addWidget(self.chk_bandlimit)
        row_bandlimit.addStretch()
        sim_layout.addRow(row_bandlimit)

        # \u2500\u2500 Focal-plane MFT zoom group (revealed when an MFT method
        # is selected) \u2500\u2500
        # Lets the user oversample the focal plane on a user-specified
        # grid, decoupled from the input grid.  The non-MFT siblings
        # would require a much larger N to achieve the same effective
        # focal-plane sampling.
        self.grp_mft = QWidget()
        mft_layout = QFormLayout(self.grp_mft)
        mft_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_dx_out = QDoubleSpinBox()
        self.spin_dx_out.setRange(1e-4, 1e6)
        self.spin_dx_out.setDecimals(4)
        self.spin_dx_out.setValue(1.0)
        self.spin_dx_out.setSuffix(' \u00b5m')
        self.spin_dx_out.setToolTip(
            'Output-plane pixel pitch.  Independent of dx_in and z; '
            'standard tool for sampling a tightly-focused region at '
            'sub-FFT-pitch resolution without padding the input grid.')
        mft_layout.addRow('Output dx:', self.spin_dx_out)
        self.spin_N_out = QSpinBox()
        self.spin_N_out.setRange(16, 32768)
        self.spin_N_out.setValue(256)
        self.spin_N_out.setToolTip(
            'Output grid size (square).  Independent of input N.')
        mft_layout.addRow('Output N:', self.spin_N_out)
        row_centre = QHBoxLayout()
        self.spin_cx = QDoubleSpinBox()
        self.spin_cx.setRange(-1e6, 1e6)
        self.spin_cx.setDecimals(3)
        self.spin_cx.setValue(0.0)
        self.spin_cx.setSuffix(' \u00b5m')
        self.spin_cy = QDoubleSpinBox()
        self.spin_cy.setRange(-1e6, 1e6)
        self.spin_cy.setDecimals(3)
        self.spin_cy.setValue(0.0)
        self.spin_cy.setSuffix(' \u00b5m')
        row_centre.addWidget(QLabel('x:'))
        row_centre.addWidget(self.spin_cx)
        row_centre.addWidget(QLabel('y:'))
        row_centre.addWidget(self.spin_cy)
        mft_centre_widget = QWidget()
        mft_centre_widget.setLayout(row_centre)
        mft_layout.addRow('Output centre:', mft_centre_widget)
        # 3.7.9: do NOT add grp_mft as an inline row.  The MFT
        # Options button on the method row re-parents grp_mft into
        # a modal dialog on demand, so the dock isn't cluttered
        # with focal-zoom fields when MFT is off (which is the
        # common case).
        self.grp_mft.setVisible(False)

        # \u2500\u2500 Convert focal field to chief-relative (3.5.7
        # apply_fresnel_curvature) \u2500\u2500
        # Bridges the absolute-phase convention used by Lumenairy
        # against ray-trace-rooted aberration tools (OPDPy, Zemax
        # OPD operands).  Optional post-processing applied to the
        # focal-plane field only.
        # \u2500\u2500 Detector model toggle (3.6) \u2500\u2500
        # Optional pixel-array sensor model applied to the focal-plane
        # field via apply_detector.  Adds gain / QE / read-noise /
        # dark-current realism on top of the wave-optics PSF.
        det_group = QGroupBox('Detector model (optional)')
        det_layout = QFormLayout(det_group)
        self.chk_detector = QCheckBox('Apply detector to focal field')
        self.chk_detector.setChecked(False)
        self.chk_detector.setToolTip(
            'After propagation, sample the focal-plane field on a '
            'pixel grid and add Poisson + read + dark-current noise.')
        det_layout.addRow(self.chk_detector)
        self.spin_pixel_um = QDoubleSpinBox()
        self.spin_pixel_um.setRange(0.1, 1000.0)
        self.spin_pixel_um.setValue(5.0)
        self.spin_pixel_um.setDecimals(2)
        self.spin_pixel_um.setSuffix(' \u00b5m')
        det_layout.addRow('Pixel pitch:', self.spin_pixel_um)
        self.spin_qe = QDoubleSpinBox()
        self.spin_qe.setRange(0.0, 1.0)
        self.spin_qe.setSingleStep(0.05)
        self.spin_qe.setValue(0.7)
        det_layout.addRow('Quantum efficiency:', self.spin_qe)
        self.spin_read_noise = QDoubleSpinBox()
        self.spin_read_noise.setRange(0.0, 1000.0)
        self.spin_read_noise.setValue(3.0)
        det_layout.addRow('Read noise (e\u207b):', self.spin_read_noise)
        self.spin_dark = QDoubleSpinBox()
        self.spin_dark.setRange(0.0, 1e6)
        self.spin_dark.setValue(0.0)
        det_layout.addRow('Dark current (e\u207b/s):', self.spin_dark)
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(1e-6, 1e6)
        self.spin_exposure.setValue(1.0)
        self.spin_exposure.setDecimals(4)
        self.spin_exposure.setSuffix(' s')
        det_layout.addRow('Exposure:', self.spin_exposure)
        sim_layout.addRow(det_group)

        row_chief = QHBoxLayout()
        self.chk_chief_relative = QCheckBox(
            'Convert focal field to chief-relative OPD '
            '(R = v \u2212 f)')
        self.chk_chief_relative.setChecked(False)
        self.chk_chief_relative.setToolTip(
            'After propagation, divide out the natural Gaussian-beam '
            'wavefront curvature exp(i\u00b7k\u00b7r\u00b2/(2R)) at the focal plane.\n'
            'Useful for comparing fields against OPDPy / Zemax OPD '
            'operands, which store the chief-relative form.\n'
            'R defaults to bfl \u2212 efl (thin-lens approximation); for '
            'multi-element systems this is the predicted image-plane '
            'wavefront radius from Gaussian-beam ABCD propagation.')
        row_chief.addWidget(self.chk_chief_relative)
        row_chief.addStretch()
        sim_layout.addRow(row_chief)

        # Lens-model selector: picks HOW each lens element is treated.
        row_lens = QHBoxLayout()
        self.combo_lens_model = QComboBox()
        self.combo_lens_model.addItems([
            'ASM phase-screen (fastest)',
            'apply_real_lens (analytic thin element, fresnel / absorption)',
            'apply_real_lens_traced (sub-nm OPD, slowest)',
            'apply_real_lens_maslov (phase-space, caustic-safe)',
        ])
        self.combo_lens_model.setToolTip(
            'How each lens element is propagated:\n'
            '  \u2022 ASM phase-screen: inline sag phase + ASM between '
            'surfaces.  Matches apply_real_lens defaults, ~6 FFTs/surface.\n'
            '  \u2022 apply_real_lens: same math but delegated to the core '
            'function; adds Fresnel transmission + absorption when enabled.\n'
            '  \u2022 apply_real_lens_traced: hybrid wave/ray OPD.  Sub-nm '
            'agreement with the geometric ray trace on cemented doublets '
            'and freeform surfaces.  ~10-30\u00d7 slower.')
        self.combo_lens_model.currentIndexChanged.connect(
            self._update_forecast)
        row_lens.addWidget(QLabel('Lens model:'))
        row_lens.addWidget(self.combo_lens_model)

        # 3.7.9: bumped default 4 -> 8 (fastest with <1 nm fidelity
        # loss; the prior 4 was conservative for early traced-lens
        # work but production use now standardises on 8 per the
        # library's own apply_real_lens_traced default).  Prefix
        # renamed from terse "sub=" to "Ray subsample 1:N (N=)" so
        # the column it's stored in -- the per-pixel ray-trace
        # decimation factor -- is discoverable from the toolbar
        # without a tooltip.
        row_lens.addWidget(QLabel('Ray subsample 1:N'))
        self.spin_raysub = QSpinBox()
        self.spin_raysub.setRange(1, 16)
        self.spin_raysub.setValue(8)
        self.spin_raysub.setPrefix('N=')
        self.spin_raysub.setToolTip(
            'apply_real_lens_traced ray-subsample factor.  Newton-'
            'inverted OPD is computed on every Nth pixel of the '
            'lens grid and spline-interpolated to the rest, giving '
            'an N² speedup at <1 nm fidelity loss.\n\n'
            '  N=1  Newton at every pixel (slowest, exact)\n'
            '  N=4  legacy production default (~4³ = 64× '
            'faster than N=1, ~10 pm RMS departure)\n'
            '  N=8  current default (~256× faster, <1 nm RMS)\n'
            '  N=16 maximum (~1024× faster, ~few nm RMS on '
            'aggressive aspherics)')
        self.spin_raysub.valueChanged.connect(self._update_forecast)
        row_lens.addWidget(self.spin_raysub)
        sim_layout.addRow(row_lens)

        # Tilt-aware ray launch: traced-lens-only advanced knob.
        # Default OFF (3.1.4) because with preserve_input_phase=True
        # the per-pixel tilt introduces a reference-frame mismatch
        # with the plane-wave phase_analytic_lens used in the
        # delta_phase subtraction, which produces wrong OPL on
        # multi-mode inputs (post-DOE diffraction patterns).  The
        # plane-wave launch is the reference-consistent choice that
        # the pre-3.1.2 code used and that gives correct results for
        # any input the wave model can represent.
        self.chk_tilt_aware_rays = QCheckBox('Tilt-aware ray launch')
        self.chk_tilt_aware_rays.setChecked(False)
        self.chk_tilt_aware_rays.setToolTip(
            'Only affects apply_real_lens_traced.\n\n'
            'When OFF (default): all rays are launched parallel to z '
            '(classical collimated / plane-wave launch).  '
            'Reference-consistent with the `preserve_input_phase=True` '
            'subtraction, works correctly on any input the wave model '
            'can represent.\n\n'
            'When ON: each ray\'s launch direction is derived from the '
            'local phase gradient of the input field.  Produces wrong '
            'output on multi-mode inputs (post-DOE diffraction '
            'patterns, compound superpositions) because of the '
            'reference-frame mismatch described in the 3.1.4 changelog '
            'entry.  Only turn on for rigorous off-axis characterisation '
            'of a specifically known small-tilt / single-mode input, '
            'and validate against the default first.')
        sim_layout.addRow(self.chk_tilt_aware_rays)

        layout.addWidget(sim_group)

        # ── Execution Range ──
        range_group = QGroupBox('Execution Range')
        range_layout = QFormLayout(range_group)

        self.combo_start = QComboBox()
        self.combo_end = QComboBox()
        range_layout.addRow('Start at:', self.combo_start)
        range_layout.addRow('End at:', self.combo_end)

        layout.addWidget(range_group)

        # ── Compute ──
        compute_group = QGroupBox('Compute')
        comp_layout = QFormLayout(compute_group)

        self.combo_backend = QComboBox()
        backends = ['NumPy FFT']
        from ..propagators.propagation import PYFFTW_AVAILABLE, CUPY_AVAILABLE, SCIPY_FFT_AVAILABLE
        if SCIPY_FFT_AVAILABLE:
            backends.append('SciPy FFT')
        if PYFFTW_AVAILABLE:
            backends.append('pyFFTW')
        if CUPY_AVAILABLE:
            backends.append('CuPy GPU')
        self.combo_backend.addItems(backends)
        comp_layout.addRow('Backend:', self.combo_backend)

        self.combo_mem = QComboBox()
        self.combo_mem.addItems(['Auto', '2 GB', '4 GB', '8 GB',
                                 '16 GB', '32 GB', '64 GB', '128 GB',
                                 '256 GB', '512 GB', '1 TB'])
        self.combo_mem.currentIndexChanged.connect(self._update_forecast)
        comp_layout.addRow('Memory limit:', self.combo_mem)

        # Precision selector: complex128 (default, double precision) vs
        # complex64 (single precision, half memory + ~2x throughput).
        # The library's phase-screen and ASM kernel mitigations (mod-2pi
        # reduction in float64 before cast, added in 3.1.3) keep
        # complex64 accurate at large kernel-phase magnitudes, so the
        # only residual cost is the FFT's single-precision round-off
        # floor (~-80 dB cumulative vs ~-140 dB at double).  Fine for
        # most design work; stay at complex128 for deep-null or
        # stray-light analysis below -60 dB.
        self.combo_precision = QComboBox()
        self.combo_precision.addItems([
            'complex128 (double, default)',
            'complex64 (single, half memory + ~2x speed)',
        ])
        self.combo_precision.setToolTip(
            'Complex field dtype used for the whole simulation.\n\n'
            'complex128 (default): double-precision real + imag, '
            '~-140 dB cumulative dynamic range.  Bit-compatible with '
            'all previous runs.\n\n'
            'complex64: single-precision real + imag, halves memory '
            'and gives ~2x FFT / phase-screen throughput.  Effective '
            'dynamic range ~-80 dB cumulative (FFT round-off floor).  '
            'The library computes kernel phase + per-surface OPD in '
            'float64 with modulo-2pi reduction before casting to '
            'float32, so accuracy is NOT degraded by the phase '
            'magnitude.  Fine for power / magnification / crosstalk '
            'at typical dB levels; keep double for deep-null or '
            'stray-light analysis below -60 dB.')
        self.combo_precision.currentIndexChanged.connect(
            self._update_forecast)
        comp_layout.addRow('Precision:', self.combo_precision)

        layout.addWidget(compute_group)

        # ── Output ──
        output_group = QGroupBox('Output')
        out_layout = QFormLayout(output_group)

        self.chk_save = QCheckBox('Save field data to file')
        self.chk_save.setChecked(False)
        self.chk_save.stateChanged.connect(self._toggle_save)
        out_layout.addRow(self.chk_save)

        self.save_container = QWidget()
        save_inner = QFormLayout(self.save_container)
        save_inner.setContentsMargins(0, 0, 0, 0)

        self.combo_format = QComboBox()
        self.combo_format.addItems(['HDF5 (.h5)', 'Zarr (.zarr)'])
        save_inner.addRow('Format:', self.combo_format)

        folder_row = QHBoxLayout()
        self.inp_folder = QLineEdit()
        self.inp_folder.setPlaceholderText('Output folder...')
        folder_row.addWidget(self.inp_folder)
        btn_browse_folder = QPushButton('...')
        btn_browse_folder.setFixedWidth(30)
        btn_browse_folder.clicked.connect(self._browse_folder)
        folder_row.addWidget(btn_browse_folder)
        save_inner.addRow('Folder:', folder_row)

        self.inp_filename = QLineEdit('simulation')
        self.inp_filename.setPlaceholderText('Filename (no extension)')
        save_inner.addRow('Filename:', self.inp_filename)

        # Plane checkboxes
        self.plane_check_area = QWidget()
        self.plane_check_layout = QVBoxLayout(self.plane_check_area)
        self.plane_check_layout.setContentsMargins(0, 0, 0, 0)
        self.plane_check_layout.setSpacing(1)
        self.plane_checks = []
        save_inner.addRow('Planes:', self.plane_check_area)

        out_layout.addRow(self.save_container)
        self.save_container.setVisible(False)

        layout.addWidget(output_group)

        # ── Calibration strip ──
        # The forecast time model is calibrated against a single
        # measured ASM-at-N=1024 sample on THIS box (auto-measured on
        # first forecast, cached after that).  The strip shows the
        # current baseline + lets users force a re-measurement after
        # switching FFT backend, etc.
        cal_row = QHBoxLayout()
        cal_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_calibration = QLabel('Forecast calibration: pending')
        self.lbl_calibration.setStyleSheet(
            "color: #7a94b8; font-size: 11px; "
            "font-family: Consolas;")
        self.lbl_calibration.setToolTip(
            'Forecast time predictions are scaled by a one-shot ASM '
            'measurement on this CPU.  Faster machines get faster '
            'forecasts; slower machines get longer forecasts.  '
            'Click Recalibrate after switching FFT backend (numpy / '
            'scipy / pyfftw / cupy) or moving the process to a '
            'different machine.')
        cal_row.addWidget(self.lbl_calibration, stretch=1)
        self.btn_calibrate = QPushButton('Recalibrate')
        self.btn_calibrate.setToolTip(
            'Re-measure the local ASM baseline.  Costs ~50-300 ms '
            '(one warmup ASM + two timed ASMs at N=512).')
        self.btn_calibrate.clicked.connect(self._recalibrate)
        cal_row.addWidget(self.btn_calibrate)
        layout.addLayout(cal_row)

        # ── Forecast ──
        self.forecast_label = QLabel('')
        self.forecast_label.setWordWrap(True)
        self.forecast_label.setStyleSheet(
            "color: #c0d0e8; font-size: 13px; padding: 8px; "
            "background: #0e1118; border: 1px solid #2a3548; "
            "font-family: Consolas; line-height: 1.5;")
        self.forecast_label.setMinimumHeight(80)
        layout.addWidget(self.forecast_label)

        self.warning_label = QLabel('')
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "color: #ff5555; font-size: 13px; padding: 6px; "
            "font-family: Consolas; font-weight: bold;")
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        # ── Run controls ──
        # Save-planes toggle promoted to a prominent segmented control
        # right next to the Run button -- accidentally saving a huge
        # simulation is a painful mistake to make silently.
        run_row = QHBoxLayout()
        self.btn_save_toggle = QPushButton('Save planes: ON')
        self.btn_save_toggle.setCheckable(True)
        self.btn_save_toggle.setChecked(True)
        self.btn_save_toggle.setToolTip(
            'When ON, intermediate fields are saved to disk per the '
            'plane checkboxes above.  When OFF, the simulation runs but '
            'only a summary is kept (useful for big-N exploration).')
        self.btn_save_toggle.toggled.connect(self._on_save_toggle)
        run_row.addWidget(self.btn_save_toggle)

        self.btn_run = QPushButton('\u25B6 Run Wave-Optics  (F5)')
        self.btn_run.setObjectName('run_button')
        self.btn_run.setToolTip(
            'Start a background simulation.  Press F5 from anywhere to '
            'trigger this.')
        self.btn_run.clicked.connect(self._run)
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_stop)
        layout.addLayout(run_row)

        # Forecast strip right above the progress bar so it's always
        # in view when the user is about to click Run.
        self.run_forecast_label = QLabel('')
        self.run_forecast_label.setStyleSheet(
            "color: #c0d0e8; font-size: 12px; padding: 4px 8px; "
            "background: #0e1118; border-left: 3px solid #5cb8ff; "
            "font-family: Consolas;")
        self.run_forecast_label.setToolTip(
            'Memory / disk / time forecast for the CURRENT settings.  '
            'Click Run only when the numbers are sane.')
        layout.addWidget(self.run_forecast_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat('%p%  %v / %m steps')
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel('')
        self.progress_label.setStyleSheet("color: #7a94b8; font-size: 11px;")
        layout.addWidget(self.progress_label)

        # ── Results ──
        self.fig = Figure(figsize=(6, 3.5), dpi=100, facecolor='#0a0c10')
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, stretch=1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(120)
        self.summary.setFont(QFont('Consolas', 10))
        self.summary.setStyleSheet("QTextEdit{background:#0a0c10;color:#a0b4d0;border:none}")
        layout.addWidget(self.summary)

        # Connect model changes to refresh plane list
        self.sm.system_changed.connect(self._refresh_planes)
        self._refresh_planes()
        self._update_forecast()

    # ── UI helpers ─────────────────────────────────────────────────

    def _toggle_save(self, state=None):
        self.save_container.setVisible(self.chk_save.isChecked())
        self._update_forecast()

    def _recalibrate(self):
        """Force a fresh ASM-baseline measurement and refresh the forecast.

        Triggered by the Recalibrate button.  Disables the button while
        the (sub-300ms) measurement runs so a double-click can't kick
        off two timed propagations at once, and reports the new value
        in the calibration strip.
        """
        from PySide6.QtWidgets import QApplication
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setText('Recalibrating...')
        self.lbl_calibration.setText('Forecast calibration: measuring...')
        QApplication.processEvents()
        try:
            _invalidate_asm_calibration()
            ref_ms = _local_asm_baseline_ms(force=True)
            self.lbl_calibration.setText(
                f'Forecast calibration: ASM-1024 = {ref_ms:.1f} ms '
                f'(self-measured)')
        except Exception as e:
            self.lbl_calibration.setText(
                f'Forecast calibration: failed -- {e}')
        finally:
            self.btn_calibrate.setText('Recalibrate')
            self.btn_calibrate.setEnabled(True)
        self._update_forecast()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.inp_folder.setText(folder)

    def _get_output_path(self):
        """Build full output path from folder + filename + format."""
        folder = self.inp_folder.text().strip()
        fname = self.inp_filename.text().strip() or 'simulation'
        if not folder:
            return ''
        ext = '.zarr' if 'Zarr' in self.combo_format.currentText() else '.h5'
        return os.path.join(folder, fname + ext)

    def _refresh_planes(self):
        """Rebuild plane checkboxes and range dropdowns."""
        # Clear old checkboxes
        for cb in self.plane_checks:
            cb.setParent(None)
            cb.deleteLater()
        self.plane_checks = []

        # Build element label list for range dropdowns
        elem_labels = []
        for elem in self.sm.elements:
            elem_labels.append(f'{elem.elem_num}: {elem.name} ({elem.elem_type})')

        self.combo_start.blockSignals(True)
        self.combo_end.blockSignals(True)
        self.combo_start.clear()
        self.combo_end.clear()
        for lbl in elem_labels:
            self.combo_start.addItem(lbl)
            self.combo_end.addItem(lbl)
        self.combo_start.setCurrentIndex(0)
        self.combo_end.setCurrentIndex(len(elem_labels) - 1)
        self.combo_start.blockSignals(False)
        self.combo_end.blockSignals(False)

        # Add Source checkbox
        cb = QCheckBox('Source')
        cb.setChecked(True)
        cb.stateChanged.connect(self._update_forecast)
        self.plane_check_layout.addWidget(cb)
        self.plane_checks.append(cb)

        # Add each surface
        for elem in self.sm.elements:
            if elem.elem_type in ('Source', 'Detector'):
                continue
            for si, srow in enumerate(elem.surfaces):
                label = f'{elem.name} S{si+1}'
                cb = QCheckBox(label)
                cb.setChecked(True)
                cb.stateChanged.connect(self._update_forecast)
                self.plane_check_layout.addWidget(cb)
                self.plane_checks.append(cb)

        # Focus
        cb = QCheckBox('Focus')
        cb.setChecked(True)
        cb.stateChanged.connect(self._update_forecast)
        self.plane_check_layout.addWidget(cb)
        self.plane_checks.append(cb)

        self._update_forecast()

        # Now that the per-element tab is fully populated, build the
        # MHS-chain tab.  Done here (end of __init__) so we can read
        # back any settings already established on the per-element
        # side -- e.g. wavelength + N + dx defaults.
        self._build_mhs_tab()

    # ── MHS chain tab (3.5.9) ─────────────────────────────────────

    def _build_mhs_tab(self):
        """Add the 'Custom MHS chain' tab for advanced users who want
        to drive :class:`lumenairy.MhsPipeline` directly.

        The pipeline is constructed via ``MhsPipeline.from_prescription``
        -- that path takes a prescription dict, the same wavelength /
        N / dx as the per-element tab, plus optional pre/post
        free-space distances and a propagator-method override.
        After construction the resulting subdomains are listed as a
        read-only table (z, label, dx, N) so the user can see the
        plane layout the pipeline will use before pressing Run.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        scroll.setWidget(inner)
        self._tabs.addTab(scroll, 'Custom MHS chain')

        # Description / context.
        desc = QLabel(
            'Drive lumenairy.MhsPipeline directly: build a sequence '
            'of MhsSubdomains from the current prescription with '
            'one chosen propagator (method) per subdomain, optionally '
            'adding free-space pre- and post-distances.  Output is '
            'a list of (HuygensSurface, field) pairs you can save to '
            'HDF5 / Zarr the same way the per-element tab does.')
        desc.setWordWrap(True)
        desc.setStyleSheet('color: #7a94b8; font-size: 11px;')
        layout.addWidget(desc)

        # Pipeline construction parameters.
        ppgrp = QGroupBox('Pipeline parameters')
        ppform = QFormLayout(ppgrp)

        self.combo_mhs_method = QComboBox()
        # Methods recognised by MhsPipeline.from_prescription.  GBD is
        # the natural choice for thick + curved subdomains; the others
        # are exposed for cross-validation / scripted overrides.
        self.combo_mhs_method.addItems(['gbd', 'asm', 'fresnel',
                                         'rayleigh_sommerfeld'])
        self.combo_mhs_method.setToolTip(
            'Per-subdomain propagator.  GBD (Gaussian beamlet '
            'decomposition) is the natural choice for thick / curved '
            'subdomains; ASM / Fresnel / RS are exposed for cross-'
            'validation against the per-element tab.')
        ppform.addRow('Method:', self.combo_mhs_method)

        self.spin_mhs_pre = QDoubleSpinBox()
        self.spin_mhs_pre.setRange(0.0, 1e6)
        self.spin_mhs_pre.setDecimals(3)
        self.spin_mhs_pre.setValue(0.0)
        self.spin_mhs_pre.setSuffix(' mm')
        self.spin_mhs_pre.setToolTip(
            'Free-space pre-distance prepended before the first '
            'refractive surface.  Useful when the source plane is '
            'separated from the first lens.')
        ppform.addRow('Pre-distance:', self.spin_mhs_pre)

        self.spin_mhs_post = QDoubleSpinBox()
        self.spin_mhs_post.setRange(0.0, 1e6)
        self.spin_mhs_post.setDecimals(3)
        self.spin_mhs_post.setValue(0.0)
        self.spin_mhs_post.setSuffix(' mm')
        self.spin_mhs_post.setToolTip(
            'Free-space post-distance appended after the last '
            'refractive surface.  Set to e.g. the BFL to reach the '
            'focal plane within the pipeline rather than as a '
            'separate to-focus call.')
        ppform.addRow('Post-distance:', self.spin_mhs_post)

        self.lbl_mhs_grid = QLabel('Grid: inherits N / dx from the '
                                    'per-element tab.')
        self.lbl_mhs_grid.setStyleSheet('color: #7a94b8;')
        ppform.addRow(self.lbl_mhs_grid)

        layout.addWidget(ppgrp)

        # Build / inspect / run.
        btn_row = QHBoxLayout()
        self.btn_mhs_build = QPushButton('Build pipeline')
        self.btn_mhs_build.setToolTip(
            'Construct MhsPipeline.from_prescription with the '
            'current settings and populate the subdomain table '
            'below.  No propagation happens until you press Run.')
        self.btn_mhs_build.clicked.connect(self._mhs_build_pipeline)
        btn_row.addWidget(self.btn_mhs_build)
        self.btn_mhs_run = QPushButton('▶ Run pipeline')
        self.btn_mhs_run.setEnabled(False)
        self.btn_mhs_run.setToolTip(
            'Run the previously-built pipeline on the current '
            'source field.  Intermediate planes are kept in memory; '
            'use the per-element tab\'s save plumbing for HDF5/Zarr '
            'export of the full chain.')
        self.btn_mhs_run.clicked.connect(self._mhs_run_pipeline)
        btn_row.addWidget(self.btn_mhs_run)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Subdomain inventory.
        self.tbl_mhs = QTableWidget(0, 4)
        self.tbl_mhs.setHorizontalHeaderLabels(
            ['Subdomain', 'In z [mm]', 'Out z [mm]', 'Label'])
        self.tbl_mhs.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        from PySide6.QtWidgets import QAbstractItemView
        self.tbl_mhs.setSelectionBehavior(
            QAbstractItemView.SelectRows)
        self.tbl_mhs.setEditTriggers(
            QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.tbl_mhs, stretch=1)

        # Status + summary.
        self.lbl_mhs_status = QTextEdit()
        self.lbl_mhs_status.setReadOnly(True)
        self.lbl_mhs_status.setMaximumHeight(120)
        self.lbl_mhs_status.setFont(QFont('Consolas', 10))
        self.lbl_mhs_status.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        layout.addWidget(self.lbl_mhs_status)

        self._mhs_pipeline = None
        self._mhs_last_planes = None
        self._mhs_tab = inner

    def _mhs_build_pipeline(self):
        try:
            import lumenairy as _la
            pres = self.sm.to_prescription()
        except Exception as exc:
            self.lbl_mhs_status.setPlainText(
                f'Could not export prescription: '
                f'{type(exc).__name__}: {exc}')
            return

        N = self.spin_N.currentData() or 1024
        dx_m = self.spin_dx.value() * 1e-6
        wv = self.sm.wavelength_m
        method = self.combo_mhs_method.currentText()
        pre_m = float(self.spin_mhs_pre.value()) * 1e-3
        post_m = float(self.spin_mhs_post.value()) * 1e-3

        try:
            pipeline = _la.MhsPipeline.from_prescription(
                pres, wavelength=wv, dx=dx_m, Ny=N, Nx=N,
                pre_distance=pre_m, post_distance=post_m,
                method=method)
        except Exception as exc:
            self.lbl_mhs_status.setPlainText(
                f'MhsPipeline.from_prescription failed: '
                f'{type(exc).__name__}: {exc}')
            return
        self._mhs_pipeline = pipeline

        try:
            surfaces = pipeline.surfaces()
        except Exception:
            surfaces = []
        n_sub = pipeline.n_subdomains()
        self.tbl_mhs.setRowCount(n_sub)
        for i, sub in enumerate(getattr(pipeline, 'subdomains', []) or []):
            self.tbl_mhs.setItem(i, 0, QTableWidgetItem(f'{i}'))
            self.tbl_mhs.setItem(
                i, 1, QTableWidgetItem(f'{sub.in_surface.z * 1e3:.4f}'))
            self.tbl_mhs.setItem(
                i, 2, QTableWidgetItem(f'{sub.out_surface.z * 1e3:.4f}'))
            self.tbl_mhs.setItem(
                i, 3, QTableWidgetItem(sub.label or ''))

        self.btn_mhs_run.setEnabled(n_sub > 0)
        self.lbl_mhs_status.setPlainText(
            f'Built MhsPipeline with {n_sub} subdomain(s); '
            f'{len(surfaces)} Huygens surface(s).\n'
            f'Method: {method}; pre {pre_m * 1e3:.3f} mm, '
            f'post {post_m * 1e3:.3f} mm.\n'
            f'Press "Run pipeline" to propagate the current source '
            f'field through every subdomain.')

    def _mhs_run_pipeline(self):
        pipeline = self._mhs_pipeline
        if pipeline is None:
            return
        # Build a source field consistent with the per-element tab.
        # Plane-wave-clipped-by-EPD is the safe default; users who
        # want a more nuanced source can run the per-element tab
        # first and use the same model.
        N = self.spin_N.currentData() or 1024
        dx_m = self.spin_dx.value() * 1e-6
        x = (np.arange(N) - N / 2) * dx_m
        X, Y = np.meshgrid(x, x)
        E = np.ones((N, N), dtype=np.complex128)
        epd_m = self.sm.epd_m
        if epd_m > 0:
            E[X * X + Y * Y > (epd_m / 2) ** 2] = 0.0

        try:
            planes = pipeline.run(E, return_intermediate=True)
        except Exception as exc:
            self.lbl_mhs_status.setPlainText(
                f'MhsPipeline.run failed: '
                f'{type(exc).__name__}: {exc}')
            return
        self._mhs_last_planes = planes

        # Summary: peak / centroid per plane.
        from ..analysis import beam_d4sigma
        lines = [f'Pipeline run: {len(planes)} plane(s).',
                 '       z [mm]   |E|_peak    label']
        for surf, field in planes:
            try:
                peak = float(np.max(np.abs(field)))
            except Exception:
                peak = float('nan')
            label = getattr(surf, 'label', '') or ''
            lines.append(
                f'  {surf.z * 1e3:10.4f}  {peak:10.4e}    {label}')
        self.lbl_mhs_status.setPlainText('\n'.join(lines))
        self.btn_save_toggle.setText(
            'Save planes: ON' if checked else 'Save planes: OFF')
        # Keep the main save-to-file checkbox in sync so the two
        # controls never disagree.
        self.chk_save.setChecked(checked)
        self._update_forecast()

    def _on_save_toggle(self, checked):
        """Sync the save-planes pill with the main save-to-file
        checkbox so the two controls never disagree.  Connected
        from `__init__` to `self.btn_save_toggle.toggled`.
        """
        self.btn_save_toggle.setText(
            'Save planes: ON' if checked else 'Save planes: OFF')
        self.chk_save.setChecked(checked)
        self._update_forecast()

    def _current_lens_model(self):
        idx = self.combo_lens_model.currentIndex()
        return {0: 'asm', 1: 'real_lens',
                2: 'real_lens_traced',
                3: 'real_lens_maslov'}.get(idx, 'asm')

    def _apply_quick_preset(self, key: str):
        """3.6: write a complete dock config for one of three named
        production presets.  The user is then one click away from
        Run.  Implemented by setting the existing widgets, so the
        run path is unchanged.
        """
        presets = {
            'fast': {
                'N': 512, 'dx_um': 4.0, 'method_text': 'ASM',
                'lens_idx': 0, 'precision_idx': 1, 'bandlimit': True,
            },
            'production': {
                'N': 1024, 'dx_um': 2.0, 'method_text': 'ASM',
                'lens_idx': 1, 'precision_idx': 0, 'bandlimit': True,
            },
            'validation': {
                'N': 2048, 'dx_um': 1.0, 'method_text': 'ASM',
                'lens_idx': 2, 'precision_idx': 0, 'bandlimit': True,
                'ray_subsample': 1,
            },
        }
        p = presets.get(key)
        if p is None:
            return
        idx = self.spin_N.findData(p['N'])
        if idx >= 0:
            self.spin_N.setCurrentIndex(idx)
        self.spin_dx.setValue(p['dx_um'])
        idx = self.combo_method.findText(p['method_text'])
        if idx >= 0:
            self.combo_method.setCurrentIndex(idx)
        self.combo_lens_model.setCurrentIndex(p['lens_idx'])
        self.combo_precision.setCurrentIndex(p['precision_idx'])
        self.chk_bandlimit.setChecked(p['bandlimit'])
        if 'ray_subsample' in p:
            self.spin_raysub.setValue(p['ray_subsample'])
        self._update_forecast()

    def _current_method_key(self):
        """Map the combo-box label to the canonical method key used
        downstream (config dict, dispatch, forecast lookup).  Kept
        here as a single source of truth so the run-path and the
        forecast see the same key."""
        text = self.combo_method.currentText().lower()
        # Resolve the base propagator + whether the legacy
        # combo-style MFT items were picked.
        if 'asm mft' in text:
            base, legacy_mft = 'asm', True
        elif 'fresnel mft' in text:
            base, legacy_mft = 'fresnel', True
        elif 'fraunhofer mft' in text:
            base, legacy_mft = 'fraunhofer', True
        elif 'rayleigh' in text or 'sommerfeld' in text:
            base, legacy_mft = 'rayleigh-sommerfeld', False
        elif 'sas' in text:
            base, legacy_mft = 'sas', False
        elif 'gbd' in text:
            base, legacy_mft = 'gbd', False
        elif 'hfpi' in text:
            base, legacy_mft = 'hfpi', False
        elif 'huygens' in text:
            base, legacy_mft = 'huygens-fresnel', False
        elif 'subaperture' in text:
            base, legacy_mft = 'subaperture', False
        elif 'fresnel' in text:
            base, legacy_mft = 'fresnel', False
        elif 'fraunhofer' in text:
            base, legacy_mft = 'fraunhofer', False
        else:
            base, legacy_mft = 'asm', False
        # 3.7.9: append '-mft' when the new "Use MFT" checkbox is
        # set AND the base method supports MFT (ASM / Fresnel /
        # Fraunhofer).  Falls back to legacy combo-item dispatch
        # so prior snapshots with "ASM MFT" etc. still work.
        mft_supported = base in ('asm', 'fresnel', 'fraunhofer')
        chk_on = (hasattr(self, 'chk_mft')
                  and self.chk_mft.isChecked())
        use_mft = legacy_mft or (chk_on and mft_supported)
        return f'{base}-mft' if use_mft else base

    def _update_mft_visibility(self):
        """3.7.9: grp_mft is now driven from the MFT Options dialog,
        not inline visibility -- so this is a no-op kept for the
        existing signal connections.  The MFT options button is
        always available; ``Use MFT`` checkbox just toggles whether
        the dispatch uses the -mft variant of the base method.
        """
        return

    def _open_mft_options_dialog(self):
        """3.7.9: show the MFT focal-zoom options (dx_out, N_out,
        centre x/y) as a modal dialog.

        Re-parents the existing ``self.grp_mft`` form into the
        dialog for the duration so the same widget references the
        run path reads from (``self.spin_dx_out``,
        ``self.spin_N_out``, ``self.spin_cx``, ``self.spin_cy``)
        stay valid -- no duplication / write-back gymnastics
        needed.
        """
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QDialogButtonBox, QLabel,
        )
        dlg = QDialog(self)
        dlg.setWindowTitle('MFT focal-plane zoom options')
        dlg.setModal(True)
        lay = QVBoxLayout(dlg)
        hdr = QLabel(
            'Output-plane grid for the Matrix Fourier Transform '
            'focal step.  Decouples the focal sampling from the '
            'input grid + propagation distance; only used when '
            '"Use MFT" is checked.')
        hdr.setWordWrap(True)
        hdr.setStyleSheet('color: #a0b4d0; font-size: 11px;')
        lay.addWidget(hdr)
        # Re-parent grp_mft for the duration of the dialog.
        original_parent = self.grp_mft.parentWidget()
        self.grp_mft.setParent(dlg)
        self.grp_mft.setVisible(True)
        lay.addWidget(self.grp_mft)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.accept)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        try:
            dlg.exec()
        finally:
            # Return grp_mft to its original parent and hide it
            # again.  The run path reads spinbox values directly
            # via ``self.spin_*``, so visibility / layout
            # placement after close don't affect behaviour.
            self.grp_mft.setParent(original_parent)
            self.grp_mft.setVisible(False)

    def _update_forecast(self):
        N = self.spin_N.currentData() or 1024
        n_surfs = len(self.sm.build_trace_surfaces())
        saving = (self.chk_save.isChecked()
                  and self.btn_save_toggle.isChecked())
        n_save = (sum(1 for cb in self.plane_checks if cb.isChecked())
                  if saving else 0)

        lens_model = self._current_lens_model()
        ray_sub = int(self.spin_raysub.value())
        method_key = self._current_method_key()

        peak_mem, disk, est_time = forecast_resources(
            N, n_surfs, n_save,
            lens_model=lens_model,
            ray_subsample=ray_sub,
            method=method_key)

        # If not saving, memory is lower (no plane storage in RAM)
        if not saving:
            bytes_per_field = N * N * 16
            mem_mult = 4 if lens_model != 'real_lens_traced' else 5
            peak_mem = bytes_per_field * mem_mult

        # Available memory
        try:
            from ..memory import available_memory_bytes, get_ram_budget
            avail = available_memory_bytes()
            budget = get_ram_budget()
        except Exception:
            avail = 4 * 1024**3
            budget = avail

        # Memory limit override
        mem_text = self.combo_mem.currentText()
        if mem_text != 'Auto':
            parts = mem_text.split()
            val = int(parts[0])
            if 'TB' in mem_text:
                budget = val * 1024**4
            else:
                budget = val * 1024**3

        # Check disk space
        disk_ok = True
        disk_avail = 0
        if saving:
            folder = self.inp_folder.text().strip()
            if folder and os.path.isdir(folder):
                try:
                    import shutil
                    disk_avail = shutil.disk_usage(folder).free
                    disk_ok = disk < disk_avail
                except Exception:
                    disk_avail = 0

        mem_ok = peak_mem < budget
        field_mm = N * self.spin_dx.value() * 1e-3

        # Build forecast text
        lens_desc = {
            'asm':              'ASM phase-screen (fast)',
            'real_lens':        'apply_real_lens (analytic)',
            'real_lens_traced': f'apply_real_lens_traced (sub={ray_sub})',
            'real_lens_maslov': 'apply_real_lens_maslov (phase-space)',
        }.get(lens_model, lens_model)

        lines = []
        lines.append(f'Lens model: {lens_desc}')
        lines.append(f'Memory:  ~{format_bytes(peak_mem)} peak  '
                     f'(budget: {format_bytes(budget)})')
        if saving and n_save > 0:
            lines.append(f'Disk:    ~{format_bytes(disk)}  '
                         f'({n_save} planes, gzip)')
        else:
            lines.append(f'Disk:    none (not saving)')
        lines.append(f'Time:    ~{format_time(est_time)}  '
                     f'({n_surfs} surface step{"s" if n_surfs != 1 else ""})')
        lines.append(f'Grid:    {N} x {N} at {self.spin_dx.value():.3f} um  '
                     f'= {field_mm:.2f} mm field')

        self.forecast_label.setText('\n'.join(lines))

        # Refresh the calibration strip with the actual measured value.
        ref_ms = _local_asm_baseline_ms()
        self.lbl_calibration.setText(
            f'Forecast calibration: ASM-1024 = {ref_ms:.1f} ms '
            f'(self-measured)')

        # Concise one-liner for the always-visible strip above Run.
        # Colored by feasibility: green = ok, amber = marginal, red = fail.
        marginal = est_time > 120 or peak_mem > 0.7 * budget
        fatal = peak_mem > budget or (saving and not disk_ok
                                      and disk_avail > 0) \
                or est_time > 86400
        if fatal:
            tag_color = '#ff6b6b'
            tag = 'CHECK BEFORE RUN'
        elif marginal:
            tag_color = '#ffd166'
            tag = 'HEADS-UP'
        else:
            tag_color = '#3ddc84'
            tag = 'ok'
        self.run_forecast_label.setText(
            f'<span style="color:{tag_color};font-weight:bold;">'
            f'[{tag}]</span>  '
            f'{lens_desc}   \u2502   N={N}^2  dx={self.spin_dx.value():.3g}\u00b5m'
            f'   \u2502   mem ~{format_bytes(peak_mem)}'
            f'   \u2502   time ~{format_time(est_time)}'
            f'   \u2502   '
            f'{"disk ~" + format_bytes(disk) if saving else "no save"}')

        # Warnings
        warnings = []
        if not mem_ok:
            warnings.append(
                f'MEMORY: peak ~{format_bytes(peak_mem)} exceeds '
                f'budget {format_bytes(budget)}. '
                f'Reduce N or increase memory limit.')
        if saving and not disk_ok and disk_avail > 0:
            warnings.append(
                f'DISK: estimated {format_bytes(disk)} exceeds '
                f'available {format_bytes(disk_avail)}.')
        if est_time > 86400:
            warnings.append(
                f'TIME: estimated {format_time(est_time)} '
                f'(> 24 hours). Simulation will still run if started.')

        if warnings:
            self.warning_label.setText('\n'.join(warnings))
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

    # ── Recommend ─────────────────────────────────────────────────

    def _recommend_grid(self):
        try:
            N, dx_um = self.sm.recommend_grid()
            # Set N in combo
            idx = self.spin_N.findText(str(N))
            if idx >= 0:
                self.spin_N.setCurrentIndex(idx)
            else:
                self.spin_N.addItem(str(N), N)
                self.spin_N.setCurrentText(str(N))
            self.spin_dx.setValue(dx_um)
            self._update_forecast()
        except Exception as e:
            self.summary.setPlainText(f'Recommend error: {e}')

    # ── Run ───────────────────────────────────────────────────────

    def _run(self):
        N = self.spin_N.currentData() or 1024
        dx_m = self.spin_dx.value() * 1e-6
        method = self._current_method_key()

        backend_text = self.combo_backend.currentText()
        if 'CuPy' in backend_text:
            backend = 'numpy'
            use_gpu = True
        elif 'pyFFTW' in backend_text:
            backend = 'pyfftw'
            use_gpu = False
        elif 'SciPy' in backend_text:
            backend = 'scipy'
            use_gpu = False
        else:
            backend = 'numpy'
            use_gpu = False

        mem_text = self.combo_mem.currentText()
        mem_limit = None
        if mem_text != 'Auto':
            parts = mem_text.split()
            val = int(parts[0])
            if 'TB' in mem_text:
                mem_limit = val * 1024
            else:
                mem_limit = val

        # Output path
        output_path = ''
        if self.chk_save.isChecked():
            output_path = self._get_output_path()

        # Plane flags
        save_planes = {}
        if self.chk_save.isChecked():
            for cb in self.plane_checks:
                save_planes[cb.text()] = cb.isChecked()

        # Execution range
        start_idx = self.combo_start.currentIndex()
        end_idx = self.combo_end.currentIndex()

        config = {
            'N': N,
            'dx_m': dx_m,
            'method': method,
            'backend': backend,
            'use_gpu': use_gpu,
            'memory_limit_gb': mem_limit,
            'output_path': output_path,
            'save_planes': save_planes,
            'start_elem': start_idx,
            'end_elem': end_idx,
            'lens_model': self._current_lens_model(),
            'ray_subsample': int(self.spin_raysub.value()),
            'tilt_aware_rays': bool(self.chk_tilt_aware_rays.isChecked()),
            'precision': ('complex64'
                          if self.combo_precision.currentIndex() == 1
                          else 'complex128'),
            # 3.5.8 propagator standardisation: Matsushima bandlimit
            # exposed as a single dock-wide flag (passed to ASM, RS,
            # ASM-MFT).
            'bandlimit': bool(self.chk_bandlimit.isChecked()),
            # 3.5.7 MFT propagators: focal-plane output grid
            # parameters.  Only consulted when method endswith '-mft'.
            'mft_dx_out_m': float(self.spin_dx_out.value()) * 1e-6,
            'mft_N_out': int(self.spin_N_out.value()),
            'mft_centre_out_m': (
                float(self.spin_cx.value()) * 1e-6,
                float(self.spin_cy.value()) * 1e-6),
            # 3.5.7 apply_fresnel_curvature: optional post-processing
            # applied to the focal field for chief-relative-OPD
            # comparison against ray-trace-rooted libraries.
            'chief_relative_focal': bool(
                self.chk_chief_relative.isChecked()),
            # 3.6: detector model post-processing.
            'detector_apply': bool(self.chk_detector.isChecked()),
            'detector_pixel_um': float(self.spin_pixel_um.value()),
            'detector_qe': float(self.spin_qe.value()),
            'detector_read_noise_e': float(self.spin_read_noise.value()),
            'detector_dark_e_per_s': float(self.spin_dark.value()),
            'detector_exposure_s': float(self.spin_exposure.value()),
        }

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        # Switch the bar into DETERMINATE mode (0-1000) so we can report
        # fine-grained progress from the core progress hooks.
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)

        self._worker = WaveOpticsWorker(self.sm, config)
        self._worker.progress.connect(self._on_progress)
        self._worker.fine_progress.connect(self._on_fine_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._on_finished({'error': 'Stopped by user'})

    def _on_progress(self, step, total, label):
        # Coarse-grained per-stage progress -- complements the
        # fine_progress signal which drives the 0-1000 bar.
        self.progress_label.setText(label)
        # If fine progress is never emitted (e.g. the old inline path
        # with no sub-stages), approximate from step/total.
        if self.progress_bar.maximum() == 1000:
            self.progress_bar.setValue(
                int(1000 * step / max(total, 1)))

    def _on_fine_progress(self, fraction, msg):
        self.progress_bar.setValue(int(1000 * max(0.0, min(1.0, fraction))))
        if msg:
            self.progress_label.setText(msg)

    def _on_finished(self, results):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText('Done')
        self._worker = None

        if 'error' in results:
            self.summary.setPlainText(f'Error: {results["error"]}')
            return

        # ── Plot results ──
        self.fig.clear()
        I_focus = results['I_focus']
        dx = results['dx']
        N = results['N']

        # PSF (log scale)
        ax = self.fig.add_subplot(121)
        ax.set_facecolor('#0a0c10')
        ax.tick_params(colors='#7a94b8', labelsize=8)
        ax.spines[:].set_color('#2a3548')

        c = N // 2
        w = N // 8
        I_log = np.log10(I_focus / max(I_focus.max(), 1e-30) + 1e-10)
        crop = I_log[c - w:c + w, c - w:c + w]
        ext = w * dx * 1e6

        ax.imshow(crop, extent=[-ext, ext, -ext, ext],
                  cmap='inferno', origin='lower', aspect='equal')
        ax.set_xlabel('x (um)', color='#dde8f8', fontsize=9, fontfamily='monospace')
        ax.set_ylabel('y (um)', color='#dde8f8', fontsize=9, fontfamily='monospace')
        ax.set_title('PSF (log)', color='#5cb8ff', fontsize=10, fontfamily='monospace')

        # Cross-section
        ax2 = self.fig.add_subplot(122)
        ax2.set_facecolor('#0a0c10')
        ax2.tick_params(colors='#7a94b8', labelsize=8)
        ax2.spines[:].set_color('#2a3548')
        ax2.grid(True, color='#1a2535', linewidth=0.5)

        x_um = (np.arange(N) - N / 2) * dx * 1e6
        I_slice = I_focus[c, :]
        I_norm = I_slice / max(I_slice.max(), 1e-30)
        ax2.semilogy(x_um[c - w:c + w], I_norm[c - w:c + w],
                     color='#5cb8ff', linewidth=1.2)
        ax2.set_xlabel('x (um)', color='#dde8f8', fontsize=9, fontfamily='monospace')
        ax2.set_ylabel('Intensity (norm)', color='#dde8f8', fontsize=9, fontfamily='monospace')
        ax2.set_title('X cross-section', color='#5cb8ff', fontsize=10, fontfamily='monospace')
        ax2.set_ylim(1e-6, 2)

        self.fig.tight_layout()
        self.canvas.draw()

        # ── Summary text ──
        lines = []
        lines.append(f'Grid: {N}x{N}, dx = {dx*1e6:.3f} um')
        lines.append(f'Wavelength: {results["wavelength"]*1e9:.1f} nm')
        lines.append(f'Method: {self.combo_method.currentText()}')
        lines.append(f'Backend: {self.combo_backend.currentText()}')
        lines.append(f'Power in: {results["power_in"]:.4e}')
        lines.append(f'Power at focus: {results["power_focus"]:.4e}')
        eff = results["power_focus"] / max(results["power_in"], 1e-30) * 100
        lines.append(f'Throughput: {eff:.1f}%')
        lines.append(f'D4sigma: {results["d4sigma"]*1e6:.2f} um')
        lines.append(f'Elapsed: {format_time(results.get("elapsed", 0))}')
        lines.append(f'Planes saved: {results.get("n_planes_saved", 0)}')
        if results.get('output_path'):
            lines.append(f'Output: {results["output_path"]}')
        if results.get('save_error'):
            lines.append(f'Save error: {results["save_error"]}')
        self.summary.setPlainText('\n'.join(lines))

        # Notify any external listeners (e.g. ZernikeDock) that a fresh
        # focal-plane field is available.
        self.run_finished.emit(results)
