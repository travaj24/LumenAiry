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
    QFormLayout,
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
        from ..propagation import angular_spectrum_propagate
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
                   'sas': 5.0}.get(method, 1.0)

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
        from ..propagation import (
            angular_spectrum_propagate,
            fresnel_propagate, fraunhofer_propagate,
            rayleigh_sommerfeld_propagate,
            resample_field, PYFFTW_AVAILABLE, CUPY_AVAILABLE,
        )
        from ..lenses import surface_sag_general
        from ..glass import get_glass_index
        from ..analysis import beam_d4sigma, beam_power

        cfg = self.cfg
        N = cfg['N']
        dx = cfg['dx_m']
        wv = self.model.wavelength_m
        method = cfg['method']  # 'asm', 'fresnel', 'fraunhofer'
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

            E = np.ones((N, N), dtype=cdtype)
            x = (np.arange(N) - N / 2) * dx
            X, Y = np.meshgrid(x, x)
            R_sq = X ** 2 + Y ** 2
            epd_m = self.model.epd_m

            # Source type
            src = self.model.source
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
            # Carrier phase = exp(i * (kx*X + ky*Y)).  Applied to all source
            # types (plane / gaussian / gaussian_aperture).
            if src is not None and (src.field_angle_x_deg
                                    or src.field_angle_y_deg):
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
                from ..lenses import (apply_real_lens,
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

                    if method == 'fresnel':
                        E, dx_new, _ = fresnel_propagate(
                            E, ts.thickness, lam_med, current_dx)
                        if abs(dx_new - current_dx) > current_dx * 1e-6:
                            E, _ = resample_field(E, dx_new, current_dx,
                                                   N_out=N)
                    elif method == 'fraunhofer' and i == len(trace_surfs) - 1:
                        E, dx_new, _ = fraunhofer_propagate(
                            E, ts.thickness, lam_med, current_dx)
                        current_dx = dx_new
                    elif method == 'rayleigh-sommerfeld':
                        E = rayleigh_sommerfeld_propagate(
                            E, ts.thickness, lam_med, current_dx,
                            use_gpu=use_gpu)
                    elif method == 'sas':
                        from ..propagation import (
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
                            bandlimit=True, use_gpu=use_gpu)

                    z_cum += ts.thickness

                maybe_save(ts.label, E, z_cum)

            # ── Step 3: Propagate to focus ──
            step += 1
            self.progress.emit(step, total_steps, 'Propagating to focus')

            bfl_mm = self.model.bfl_mm
            if np.isfinite(bfl_mm) and bfl_mm > 0:
                bfl_m = bfl_mm * 1e-3
                if method == 'fraunhofer':
                    E_focus, dx_focus, _ = fraunhofer_propagate(
                        E, bfl_m, wv, current_dx)
                    current_dx = dx_focus
                elif method == 'fresnel':
                    E_focus, dx_focus, _ = fresnel_propagate(
                        E, bfl_m, wv, current_dx)
                    current_dx = dx_focus
                elif method == 'rayleigh-sommerfeld':
                    E_focus = rayleigh_sommerfeld_propagate(
                        E, bfl_m, wv, current_dx, use_gpu=use_gpu)
                elif method == 'sas':
                    from ..propagation import (
                        scalable_angular_spectrum_propagate)
                    E_focus, dx_focus, _ = scalable_angular_spectrum_propagate(
                        E, bfl_m, wv, current_dx, use_gpu=use_gpu)
                    current_dx = dx_focus
                else:
                    E_focus = angular_spectrum_propagate(
                        E, bfl_m, wv, current_dx,
                        bandlimit=True, use_gpu=use_gpu)
                z_cum += bfl_m
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
                        from ..storage import set_storage_backend, append_plane, write_metadata
                        set_storage_backend('zarr')
                    else:
                        from ..storage import set_storage_backend, append_plane, write_metadata
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

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
                                    'Rayleigh-Sommerfeld', 'SAS'])
        self.combo_method.setToolTip(
            'Free-space propagator used BETWEEN elements.\n'
            '  ASM:     exact band-limited, fixed grid (default).\n'
            '  Fresnel: single-FFT paraxial; auto-resampled back to dx.\n'
            '  Fraunhofer: far-field only (last surface).\n'
            '  R-S:     Rayleigh-Sommerfeld convolution (slowest).\n'
            '  SAS:     Scalable Angular Spectrum (Heintzmann 2023); '
            'right for long z where plain ASM needs too many samples. '
            'Auto-resampled back to dx.')
        self.combo_method.currentIndexChanged.connect(self._update_forecast)
        row_method.addWidget(QLabel('Method:'))
        row_method.addWidget(self.combo_method)
        self.btn_recommend = QPushButton('Recommend')
        self.btn_recommend.setToolTip(
            'Auto-size N and dx from system NA, aperture, and the '
            'OPD-Nyquist rule (dx \u2264 \u03bb*f/aperture).')
        self.btn_recommend.clicked.connect(self._recommend_grid)
        row_method.addWidget(self.btn_recommend)
        sim_layout.addRow(row_method)

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

        self.spin_raysub = QSpinBox()
        self.spin_raysub.setRange(1, 16)
        self.spin_raysub.setValue(4)
        self.spin_raysub.setPrefix('sub=')
        self.spin_raysub.setToolTip(
            'ray_subsample for apply_real_lens_traced.  '
            '1 = Newton at every pixel (exact), 4 = production default, '
            '8 = fastest with <1 nm fidelity loss.')
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
        from ..propagation import PYFFTW_AVAILABLE, CUPY_AVAILABLE, SCIPY_FFT_AVAILABLE
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

        self.btn_run = QPushButton('\u25B6 Run Wave-Optics')
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

    # ── Forecast ──────────────────────────────────────────────────

    def _on_save_toggle(self, checked):
        self.btn_save_toggle.setText(
            'Save planes: ON' if checked else 'Save planes: OFF')
        # Keep the main save-to-file checkbox in sync so the two
        # controls never disagree.
        self.chk_save.setChecked(checked)
        self._update_forecast()

    def _current_lens_model(self):
        idx = self.combo_lens_model.currentIndex()
        return {0: 'asm', 1: 'real_lens',
                2: 'real_lens_traced',
                3: 'real_lens_maslov'}.get(idx, 'asm')

    def _update_forecast(self):
        N = self.spin_N.currentData() or 1024
        n_surfs = len(self.sm.build_trace_surfaces())
        saving = (self.chk_save.isChecked()
                  and self.btn_save_toggle.isChecked())
        n_save = (sum(1 for cb in self.plane_checks if cb.isChecked())
                  if saving else 0)

        lens_model = self._current_lens_model()
        ray_sub = int(self.spin_raysub.value())
        method = self.combo_method.currentText().lower()
        if 'rayleigh' in method or 'sommerfeld' in method:
            method_key = 'rayleigh-sommerfeld'
        elif 'sas' in method:
            method_key = 'sas'
        else:
            method_key = method

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
        method = self.combo_method.currentText().lower()

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
