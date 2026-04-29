"""Cross-cutting / API / end-to-end integration tests.

Contains:
- Library self-consistency: required symbols are exported at top level,
  long-running funcs accept progress=, traced lens has guardrail kwargs.
- ProgressScaler two-call forms (2-arg and 3-arg protocol).
- End-to-end compositions: coating on real-lens glass, interferogram
  from real-lens OPD, fiber->lens->detector, telescope through-focus.
- Memory helpers.
- Plotting smoke tests.

Taken from hammer_test.py (sections 1, 8), deep_audit.py composite tests,
new_features_deep_test.py (compositions), physics_remaining_test.py
(memory + plotting).
"""
from __future__ import annotations

import inspect
import sys
import matplotlib
matplotlib.use('Agg')

import numpy as np

from _harness import Harness

import lumenairy as op


H = Harness('integration')

lam = 1.31e-6


# ---------------------------------------------------------------------
H.section('Library self-consistency')


MUST_EXPORT = [
    'apply_real_lens', 'apply_real_lens_traced',
    'propagate_through_system',
    'through_focus_scan', 'tolerancing_sweep',
    'monte_carlo_tolerancing',
    'design_optimize', 'ProgressCallback', 'ProgressScaler',
    'call_progress',
    'MultiPrescriptionParameterization', 'MatchIdealSystemMerit',
    'ChromaticFocalShiftMerit', 'ToleranceAwareMerit',
    'compute_psf', 'compute_mtf', 'polychromatic_strehl',
    'simulate_interferogram', 'phase_shift_extract',
    'gerchberg_saxton', 'error_reduction',
]


def t_symbols_present():
    missing = [n for n in MUST_EXPORT if not hasattr(op, n)]
    return len(missing) == 0, \
        f'missing attrs: {missing}' if missing else 'all present'


H.run('all expected symbols present at top-level', t_symbols_present)


def t_symbols_in_all():
    missing = [n for n in MUST_EXPORT if n not in op.__all__]
    return len(missing) == 0, \
        (f'missing in __all__: {missing}'
         if missing else 'all listed')


H.run('all expected symbols in __all__', t_symbols_in_all)


for fname in ['apply_real_lens', 'apply_real_lens_traced',
              'propagate_through_system', 'through_focus_scan',
              'tolerancing_sweep', 'monte_carlo_tolerancing',
              'design_optimize']:
    def _make(name):
        def _fn():
            fn = getattr(op, name)
            sig = inspect.signature(fn)
            return 'progress' in sig.parameters, f'params include progress'
        return _fn
    H.run(f'{fname} accepts progress=', _make(fname))


def t_traced_lens_guardrail_kwargs():
    sig = inspect.signature(op.apply_real_lens_traced)
    ok = ('min_coarse_samples_per_aperture' in sig.parameters
          and 'on_undersample' in sig.parameters)
    return ok, 'both kwargs present' if ok else 'missing'


H.run('apply_real_lens_traced has guardrail kwargs',
      t_traced_lens_guardrail_kwargs)


# ---------------------------------------------------------------------
H.section('ProgressScaler signatures')


def t_progress_scaler_2arg():
    calls = []
    def parent(stage, frac, msg):
        calls.append((stage, frac, msg))
    scaler = op.ProgressScaler(parent, 'wrap', 0.2, 0.8)
    scaler(0.5, 'inline')
    return len(calls) == 1 and abs(calls[0][1] - 0.5) < 1e-12, \
        f'{len(calls)} calls, first frac={calls[0][1] if calls else None}'


H.run('ProgressScaler 2-arg form fires', t_progress_scaler_2arg)


def t_progress_scaler_3arg():
    calls = []
    def parent(stage, frac, msg):
        calls.append((stage, frac, msg))
    scaler = op.ProgressScaler(parent, 'wrap', 0.2, 0.8)
    scaler('inner_stage', 0.5, 'protocol')
    ok = (len(calls) == 1 and calls[0][0] == 'wrap'
          and abs(calls[0][1] - 0.5) < 1e-12)
    return ok, f'{len(calls)} calls, stage={calls[0][0] if calls else None}'


H.run('ProgressScaler 3-arg (protocol) form fires & uses self.stage',
      t_progress_scaler_3arg)


# ---------------------------------------------------------------------
H.section('End-to-end compositions')


def t_coating_with_real_lens():
    pres = op.thorlabs_lens('AC254-100-C')
    n_baf10 = op.get_glass_index('N-BAF10', lam)
    layers = op.quarter_wave_ar(n_baf10, lam)
    R, _, _ = op.coating_reflectance(layers, [lam], n_substrate=n_baf10)
    return R[0] < 1e-6, f'AR on N-BAF10: R = {R[0]:.2e}'


H.run('Composition: AR coating on real lens glass',
      t_coating_with_real_lens)


def t_interferogram_of_real_lens():
    N = 128; dx = 8e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=2e-3)
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_real_lens(E_in, pres, lam, dx)
    x_w, opl = op.wave_opd_1d(E_out, dx, lam, aperture=1.8e-3)
    opd_2d = np.zeros((N, N))
    for i in range(N):
        opd_2d[i, :] = np.interp((np.arange(N)-N//2)*dx, x_w, opl,
                                 left=0, right=0)
    fringe = op.simulate_interferogram(opd_2d, lam, tilt_x=500, dx=dx)
    return fringe.max() > 0 and fringe.min() >= 0, \
        f'fringe range [{fringe.min():.3f}, {fringe.max():.3f}]'


H.run('Composition: interferogram from real-lens OPD',
      t_interferogram_of_real_lens)


def t_fiber_to_detector():
    N = 128; dx = 8e-6
    from lumenairy.raytrace import (
        surfaces_from_prescription, system_abcd)
    E_fib, _, _ = op.create_fiber_mode(N, dx, 10e-6, lam)
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=2e-3)
    E_exit = op.apply_real_lens(E_fib, pres, lam, dx)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, lam)
    E_focus = op.angular_spectrum_propagate(E_exit, bfl, lam, dx)
    img, _, _ = op.apply_detector(E_focus, dx, pixel_pitch=20e-6,
                                  n_pixels=30, read_noise_e=5, seed=0)
    return img.max() > 0, f'detector max = {img.max():.1f} electrons'


H.run('Composition: fiber -> lens -> detector', t_fiber_to_detector)


def t_telescope_through_focus():
    from lumenairy.raytrace import (
        surfaces_from_prescription, system_abcd)
    pres = op.keplerian_telescope(100e-3, 25e-3)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, lam)
    N = 128; dx = 16e-6
    E_in = np.ones((N, N), dtype=np.complex128)
    E_exit = op.apply_real_lens(E_in, pres, lam, dx)
    z = np.linspace(max(abs(bfl)-20e-3, 1e-3), abs(bfl)+20e-3, 11)
    scan = op.through_focus_scan(E_exit, dx, lam, z)
    return scan.peak_I.max() > 0, \
        (f'peak I range = [{scan.peak_I.min():.2e}, '
         f'{scan.peak_I.max():.2e}]')


H.run('Composition: telescope through-focus scan',
      t_telescope_through_focus)


# ---------------------------------------------------------------------
H.section('Memory helpers')


def t_memory_info():
    info = op.memory_info()
    return 'total_bytes' in info or 'total' in str(info), \
        f'info keys: {list(info.keys())[:5]}'


H.run('memory_info: returns dict with memory data', t_memory_info)


def t_total_memory_bytes():
    m = op.total_memory_bytes()
    return m > 1e8, f'total memory = {m/1e9:.2f} GB'


H.run('total_memory_bytes: > 100 MB', t_total_memory_bytes)


def t_should_split():
    total = 16384**2 * 16
    result = op.should_split(total, available=2e9, safety=0.5)
    return isinstance(result, bool) and result is True, \
        f'should_split(4GB, avail=2GB) = {result}'


H.run('should_split: 4 GB field on 2 GB limit -> True', t_should_split)


def t_pick_batch_size():
    bs = op.pick_batch_size(100, 100e6, available=2e9, safety=0.5)
    return bs > 0 and bs <= 100, f'batch_size = {bs}'


H.run('pick_batch_size: returns positive int', t_pick_batch_size)


# ---------------------------------------------------------------------
H.section('Plotting smoke (matplotlib Agg backend)')


def _plot_smoke(plot_fn_name, builder, *args, **kwargs):
    def _fn():
        fn = getattr(op, plot_fn_name)
        import matplotlib.pyplot as plt
        arg = builder()
        result = fn(arg, *args, **kwargs)
        fig = result[0] if isinstance(result, tuple) else result
        plt.close('all')
        return fig is not None, 'fig created'
    return _fn


N_plot = 64
dx_plot = 16e-6


def _rand_complex():
    return np.random.default_rng(0).standard_normal(
        (N_plot, N_plot)).astype(np.complex128)


def _exp_phase():
    return np.exp(1j * np.random.default_rng(0).standard_normal(
        (N_plot, N_plot)))


H.run('plot_intensity: returns figure',
      _plot_smoke('plot_intensity', _rand_complex, dx_plot))
H.run('plot_phase: returns figure',
      _plot_smoke('plot_phase', _exp_phase, dx_plot))
H.run('plot_field: returns figure',
      _plot_smoke('plot_field', _rand_complex, dx_plot))
H.run('plot_cross_section: returns figure',
      _plot_smoke('plot_cross_section', _rand_complex, dx_plot))


def t_plot_beam_profile():
    E, _, _ = op.create_gaussian_beam(N_plot, dx_plot, 20e-6)
    fig, _ = op.plot_beam_profile(E, dx_plot)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fig is not None, 'fig created'


H.run('plot_beam_profile: returns figure', t_plot_beam_profile)


def t_plot_psf():
    psf = np.zeros((N_plot, N_plot))
    psf[N_plot//2, N_plot//2] = 1.0
    fig, _ = op.plot_psf(psf, 1e-6)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fig is not None, 'fig created'


H.run('plot_psf: returns figure', t_plot_psf)


def t_plot_through_focus():
    from lumenairy.through_focus import ThroughFocusResult
    scan = ThroughFocusResult(
        z=np.linspace(0.09, 0.11, 11),
        peak_I=np.ones(11), strehl=np.ones(11)*0.9,
        d4sigma_x=np.ones(11)*1e-5, d4sigma_y=np.ones(11)*1e-5,
        rms_radius=np.ones(11)*5e-6,
        power_in_bucket=np.full(11, np.nan),
        wavelength=lam, dx=16e-6)
    fig, _ = op.plot_through_focus(scan)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fig is not None, 'fig created'


H.run('plot_through_focus: returns figure', t_plot_through_focus)


# ---------------------------------------------------------------------
# Additional cross-pipeline integration hammer tests (3.2.13)
# ---------------------------------------------------------------------
H.section('End-to-end: prescription -> trace -> wave-optics agreement')


def t_singlet_paraxial_focus_matches_wave_peak_position():
    """Build a singlet, find its paraxial focus via raytrace, propagate
    the wave optics to that distance, verify on-axis intensity is
    higher there than at +-1 mm offsets."""
    import lumenairy as op
    from lumenairy.raytrace import (
        surfaces_from_prescription, find_paraxial_focus,
    )
    N, dx, lam = 512, 8e-6, 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    bfl = find_paraxial_focus(surfs, lam)
    E = np.ones((N, N), dtype=np.complex128)
    E_lens = op.apply_real_lens(E, pres, lam, dx)
    I_focus = float(np.abs(
        op.angular_spectrum_propagate(E_lens, bfl, lam, dx)[N//2, N//2])**2)
    I_pre = float(np.abs(
        op.angular_spectrum_propagate(E_lens, bfl - 2e-3, lam, dx)[N//2, N//2])**2)
    I_post = float(np.abs(
        op.angular_spectrum_propagate(E_lens, bfl + 2e-3, lam, dx)[N//2, N//2])**2)
    return I_focus > I_pre and I_focus > I_post, \
        (f'I(bfl-2)={I_pre:.3e}, I(bfl)={I_focus:.3e}, '
         f'I(bfl+2)={I_post:.3e}')


H.run('Singlet end-to-end: wave-optics peak at paraxial focus',
      t_singlet_paraxial_focus_matches_wave_peak_position)


def t_propagate_through_system_matches_manual_real_lens_plus_asm():
    """propagate_through_system with [real_lens, propagate(z=bfl)] gives
    the same on-axis intensity at the focus as apply_real_lens followed
    by ASM propagation."""
    import lumenairy as op
    from lumenairy.system import propagate_through_system
    from lumenairy.raytrace import (
        surfaces_from_prescription, find_paraxial_focus,
    )
    N, dx, lam = 256, 8e-6, 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    bfl = find_paraxial_focus(surfs, lam)
    E = np.ones((N, N), dtype=np.complex128)
    elements = [
        {'type': 'real_lens', 'prescription': pres},
        {'type': 'propagate', 'z': bfl},
    ]
    out = propagate_through_system(E, elements, lam, dx)
    E_sys = out[0] if isinstance(out, tuple) else out
    E_lens = op.apply_real_lens(E, pres, lam, dx)
    E_prop = op.angular_spectrum_propagate(E_lens, bfl, lam, dx)
    rel = (abs(np.abs(E_sys[N//2, N//2])**2 - np.abs(E_prop[N//2, N//2])**2)
           / max(np.abs(E_prop[N//2, N//2])**2, 1e-30))
    return rel < 1e-6, f'on-axis I rel diff = {rel:.2e}'


H.run('propagate_through_system matches manual real_lens+ASM',
      t_propagate_through_system_matches_manual_real_lens_plus_asm)


def t_real_lens_traced_subsample_finite_for_low_NA_singlet():
    """apply_real_lens_traced via the subsample path produces a finite
    output field for a low-NA singlet (smoke check covering the
    parallelization branch)."""
    import lumenairy as op
    N, dx, lam = 256, 8e-6, 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    Et = op.apply_real_lens_traced(
        E, pres, lam, dx, n_workers=1,
        min_coarse_samples_per_aperture=8)
    return np.all(np.isfinite(Et)) and np.abs(Et).max() > 0, \
        f'finite={np.all(np.isfinite(Et))}, peak={np.abs(Et).max():.3e}'


H.run('apply_real_lens_traced (subsampling path): finite output',
      t_real_lens_traced_subsample_finite_for_low_NA_singlet)


def t_glass_dispersion_normal_for_N_BK7():
    """N-BK7 normal dispersion: n decreases monotonically F -> d -> C."""
    import lumenairy as op
    n_F = float(op.get_glass_index('N-BK7', 0.486e-6))
    n_d = float(op.get_glass_index('N-BK7', 0.587e-6))
    n_C = float(op.get_glass_index('N-BK7', 0.656e-6))
    monotone = n_F > n_d > n_C
    in_range = 1.50 < n_C < 1.55 and 1.50 < n_F < 1.55
    return monotone and in_range, \
        f'n(F)={n_F:.4f}, n(d)={n_d:.4f}, n(C)={n_C:.4f}'


H.run('N-BK7: normal dispersion across F/d/C lines',
      t_glass_dispersion_normal_for_N_BK7)


def t_propagate_through_system_aperture_then_asm_keeps_disk_area():
    """propagate_through_system with [circular aperture, propagate]
    keeps the total transmitted power equal to pi (D/2)^2."""
    import lumenairy as op
    from lumenairy.system import propagate_through_system
    N, dx, lam = 256, 8e-6, 1.31e-6
    D = 1e-3
    E = np.ones((N, N), dtype=np.complex128)
    elements = [
        {'type': 'aperture', 'shape': 'circular',
         'params': {'diameter': D}},
        {'type': 'propagate', 'z': 1e-3},
    ]
    out = propagate_through_system(E, elements, lam, dx)
    E_out = out[0] if isinstance(out, tuple) else out
    P = float(np.sum(np.abs(E_out)**2) * dx**2)
    P_expect = np.pi * (D / 2)**2
    rel = abs(P - P_expect) / P_expect
    return rel < 0.05, \
        f'P={P*1e9:.3f}nm^2, expect={P_expect*1e9:.3f}nm^2, rel={rel*100:.2f}%'


H.run('propagate_through_system: aperture+ASM keeps power = disk area',
      t_propagate_through_system_aperture_then_asm_keeps_disk_area)


def t_jones_field_real_lens_propagate_preserves_component_powers():
    """A JonesField passed through real_lens + propagation preserves
    total power on each component (within wave-optics tolerance)."""
    import lumenairy as op
    from lumenairy.polarization import JonesField
    N, dx, lam = 128, 16e-6, 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    jf = JonesField(Ex=E.copy(), Ey=0.5 * E.copy(), dx=dx)
    P_x_in = float(np.sum(np.abs(jf.Ex)**2) * dx**2)
    P_y_in = float(np.sum(np.abs(jf.Ey)**2) * dx**2)
    jf.apply_real_lens(pres, lam)
    jf.propagate(z=5e-3, wavelength=lam)
    P_x_out = float(np.sum(np.abs(jf.Ex)**2) * dx**2)
    P_y_out = float(np.sum(np.abs(jf.Ey)**2) * dx**2)
    rel_x = abs(P_x_out - P_x_in) / P_x_in
    rel_y = abs(P_y_out - P_y_in) / P_y_in
    return rel_x < 0.05 and rel_y < 0.05, \
        f'rel_x={rel_x:.3e}, rel_y={rel_y:.3e}'


H.run('JonesField: real_lens + propagate preserves component powers',
      t_jones_field_real_lens_propagate_preserves_component_powers)


if __name__ == '__main__':
    sys.exit(H.summary())
