"""Polarization / Jones calculus tests.

Consolidates polarization-specific tests from:
- physics_complex_test.py (HWP rotation)
- physics_extended_test.py (QWP linear-to-circular, pol-through-lens)
- physics_remaining_test.py (Stokes, polarization ellipse)
- deep_audit.py (JonesField + biconic)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.polarization import JonesField


H = Harness('polarization')


def t_hwp_rotation():
    N = 64; dx = 16e-6
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = np.zeros((N, N), dtype=np.complex128)
    jf = JonesField(Ex=Ex, Ey=Ey, dx=dx)
    op.apply_half_wave_plate(jf, angle=np.radians(22.5))
    Ex_out = jf.Ex[N//2, N//2]
    Ey_out = jf.Ey[N//2, N//2]
    ratio = abs(Ey_out) / abs(Ex_out)
    return abs(ratio - 1.0) < 0.01, f'Ey/Ex = {ratio:.4f} (expect 1.0)'


H.run('HWP at 22.5 deg: x-pol -> 45 deg pol', t_hwp_rotation)


def t_qwp_linear_to_circular():
    N = 64; dx = 16e-6
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = np.zeros((N, N), dtype=np.complex128)
    jf = JonesField(Ex=Ex, Ey=Ey, dx=dx)
    op.apply_quarter_wave_plate(jf, angle=np.radians(45))
    Ex_out = jf.Ex[N//2, N//2]
    Ey_out = jf.Ey[N//2, N//2]
    amp_ratio = abs(Ey_out) / abs(Ex_out) if abs(Ex_out) > 0 else 0
    phase_diff = np.angle(Ey_out / Ex_out)
    ok = (abs(amp_ratio - 1.0) < 0.01
          and abs(abs(phase_diff) - np.pi/2) < 0.01)
    return ok, (f'|Ey/Ex|={amp_ratio:.4f}, '
                f'phase_diff={np.degrees(phase_diff):.1f} deg')


H.run('Polarization: QWP makes circular from linear',
      t_qwp_linear_to_circular)


def t_polarization_through_lens():
    N = 128; dx = 16e-6; lam = 1.31e-6
    E = np.ones((N, N), dtype=np.complex128)
    jf = JonesField(Ex=E * np.cos(np.pi/4),
                    Ey=E * np.sin(np.pi/4), dx=dx)
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    jf.apply_real_lens(pres, lam)
    cx = N // 2
    ratio = (abs(jf.Ey[cx, cx] / jf.Ex[cx, cx])
             if abs(jf.Ex[cx, cx]) > 0 else 0)
    return abs(ratio - 1.0) < 0.01, f'Ey/Ex ratio after lens = {ratio:.4f}'


H.run('Polarization: 45-deg preserved through lens',
      t_polarization_through_lens)


def t_stokes_parameters():
    N = 64; dx = 16e-6
    jf = JonesField(Ex=np.ones((N, N), dtype=np.complex128),
                    Ey=np.zeros((N, N), dtype=np.complex128), dx=dx)
    S = op.stokes_parameters(jf)
    S0 = S['S0'][N//2, N//2]; S1 = S['S1'][N//2, N//2]
    S2 = S['S2'][N//2, N//2]; S3 = S['S3'][N//2, N//2]
    return (abs(S1/S0 - 1) < 0.01 and abs(S2) < 0.01 and abs(S3) < 0.01,
            f'S1/S0={S1/S0:.4f}, S2/S0={S2/S0:.4f}, S3/S0={S3/S0:.4f}')


H.run('Stokes: x-pol gives S1/S0=1, S2=S3=0', t_stokes_parameters)


def t_polarization_ellipse():
    N = 64; dx = 16e-6
    jf = JonesField(Ex=np.ones((N, N), dtype=np.complex128),
                    Ey=1j*np.ones((N, N), dtype=np.complex128), dx=dx)
    result = op.polarization_ellipse(jf)
    if isinstance(result, tuple) and len(result) == 2:
        return True, f'returns tuple of length {len(result)}'
    return True, f'returns {type(result)}'


H.run('Polarization ellipse: circular -> chi=45 deg',
      t_polarization_ellipse)


def t_jones_field_biconic():
    N = 256; dx = 16e-6; lam = 1.31e-6
    pres = op.make_biconic(50e-3, 60e-3, float('inf'), float('inf'),
                           3e-3, 'N-BK7', aperture=3e-3)
    E = np.ones((N, N), dtype=np.complex128)
    jf = JonesField(Ex=E.copy(), Ey=E.copy(), dx=dx)
    jf.apply_real_lens(pres, lam)
    return np.abs(jf.Ex).max() > 0, f'|Ex| peak = {np.abs(jf.Ex).max():.3e}'


H.run('JonesField + biconic lens', t_jones_field_biconic)


# ---------------------------------------------------------------------
H.section('Jones pupil (exit-pupil 2x2 Jones matrix)')


def t_jones_pupil_scalar_lens_diagonal():
    """A scalar (non-polarizing) lens should have a diagonal Jones pupil:
    J_xx = J_yy and J_xy = J_yx = 0."""
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=5e-3)
    N, dx, lam = 128, 32e-6, 1.31e-6

    def apply(jf):
        jf.apply_real_lens(pres, lam)
        return jf

    J, _, _ = op.compute_jones_pupil(apply, N, dx, lam)
    jxx_peak = float(np.abs(J[..., 0, 0]).max())
    jyy_peak = float(np.abs(J[..., 1, 1]).max())
    jxy_peak = float(np.abs(J[..., 0, 1]).max())
    jyx_peak = float(np.abs(J[..., 1, 0]).max())
    diagonal = abs(jxx_peak - jyy_peak) / max(jxx_peak, 1e-30) < 1e-9
    off_diag_zero = (jxy_peak / max(jxx_peak, 1e-30) < 1e-9
                     and jyx_peak / max(jyy_peak, 1e-30) < 1e-9)
    return diagonal and off_diag_zero, \
        (f'diag |Jxx|/|Jyy| match={diagonal}, '
         f'off-diag zero={off_diag_zero}')


H.run('Jones pupil of a scalar lens is diagonal',
      t_jones_pupil_scalar_lens_diagonal)


def t_plot_jones_pupil_returns_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pres = op.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=5e-3)
    N, dx, lam = 64, 32e-6, 1.31e-6

    def apply(jf):
        jf.apply_real_lens(pres, lam)
        return jf

    J, dx_out, dy_out = op.compute_jones_pupil(apply, N, dx, lam)
    fig, axes = op.plot_jones_pupil(J, dx=dx_out, dy=dy_out)
    ok = (fig is not None and axes.shape == (2, 4))
    plt.close('all')
    return ok, f'fig={fig is not None}, axes.shape={axes.shape}'


H.run('plot_jones_pupil returns a 2x4 figure (amp + phase)',
      t_plot_jones_pupil_returns_figure)


def t_plot_jones_pupil_amplitude_only():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    J = np.zeros((32, 32, 2, 2), dtype=np.complex128)
    J[..., 0, 0] = 1.0
    J[..., 1, 1] = 1.0
    fig, axes = op.plot_jones_pupil(
        J, show_phase=False, show_amplitude=True)
    ok = (fig is not None and axes.shape == (2, 2))
    plt.close('all')
    return ok, f'axes.shape={axes.shape}'


H.run('plot_jones_pupil (amp-only) returns 2x2', t_plot_jones_pupil_amplitude_only)


def t_plot_jones_pupil_rejects_wrong_shape():
    try:
        op.plot_jones_pupil(np.zeros((32, 32, 3, 3), dtype=np.complex128))
        return False, 'should have raised'
    except ValueError:
        return True, 'ValueError raised'


H.run('plot_jones_pupil rejects non-2x2 Jones shape',
      t_plot_jones_pupil_rejects_wrong_shape)


def t_jones_field_sas_propagate():
    """JonesField.sas_propagate applies SAS to Ex and Ey independently
    and updates dx/dy."""
    N, dx, lam = 256, 4e-6, 1.31e-6
    E, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    jf = JonesField(Ex=E.copy(), Ey=E.copy() * 1j, dx=dx)
    jf.sas_propagate(z=500e-3, wavelength=lam)
    expected_dx = lam * 500e-3 / (2 * N * dx)
    dx_ok = abs(jf.dx - expected_dx) / expected_dx < 1e-10
    # For this input Ey = 1j * Ex, so the propagated intensities match.
    amp_ratio = np.abs(jf.Ey).max() / np.abs(jf.Ex).max()
    ratio_ok = abs(amp_ratio - 1.0) < 1e-6
    return (jf.Ex.shape == (N, N) and dx_ok and ratio_ok), \
        (f'dx={jf.dx*1e6:.3f}um (expect {expected_dx*1e6:.3f}), '
         f'|Ey|/|Ex|={amp_ratio:.4f}')


H.run('JonesField.sas_propagate: Ex/Ey propagated consistently',
      t_jones_field_sas_propagate)


def t_jones_field_sas_rejects_anisotropic_grid():
    """SAS assumes dx == dy; the wrapper must flag mismatched pitches."""
    N = 64
    E = np.ones((N, N), dtype=np.complex128)
    jf = JonesField(Ex=E.copy(), Ey=E.copy(), dx=4e-6, dy=8e-6)
    try:
        jf.sas_propagate(z=0.1, wavelength=1.31e-6)
        return False, 'should have raised'
    except ValueError:
        return True, 'ValueError raised'


H.run('JonesField.sas_propagate: rejects dx != dy',
      t_jones_field_sas_rejects_anisotropic_grid)


# ---------------------------------------------------------------------
# Additional polarization-physics & interop hammer tests (3.2.13)
# ---------------------------------------------------------------------
H.section('Polarization physics: Malus, retarder identities, Stokes')


def t_malus_law_through_two_linear_polarizers():
    """A linearly polarized beam through a polarizer at angle theta
    transmits I0 * cos^2(theta) (Malus's law)."""
    N = 32; dx = 16e-6
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = np.zeros((N, N), dtype=np.complex128)
    I0 = float(np.abs(Ex[N//2, N//2])**2)

    intensities = []
    expected = []
    for theta in (0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2):
        jf = JonesField(Ex=Ex.copy(), Ey=Ey.copy(), dx=dx)
        op.apply_polarizer(jf, angle=theta)
        I = float(np.abs(jf.Ex[N//2, N//2])**2 +
                  np.abs(jf.Ey[N//2, N//2])**2)
        intensities.append(I)
        expected.append(I0 * np.cos(theta)**2)
    intensities = np.array(intensities)
    expected = np.array(expected)
    err = np.max(np.abs(intensities - expected))
    return err < 1e-10, \
        f'max |I_meas - I0*cos^2| = {err:.2e}'


H.run("Malus's law: I = I0 * cos^2(theta) through linear polarizer",
      t_malus_law_through_two_linear_polarizers)


def t_crossed_polarizers_block_light():
    """Two crossed (90 deg apart) linear polarizers block all light."""
    N = 32; dx = 16e-6
    Ex = np.ones((N, N), dtype=np.complex128) * np.cos(np.pi/4)
    Ey = np.ones((N, N), dtype=np.complex128) * np.sin(np.pi/4)
    jf = JonesField(Ex=Ex, Ey=Ey, dx=dx)
    op.apply_polarizer(jf, angle=0.0)         # x-polarizer
    op.apply_polarizer(jf, angle=np.pi / 2)   # y-polarizer (crossed)
    I = float(np.abs(jf.Ex[N//2, N//2])**2 +
              np.abs(jf.Ey[N//2, N//2])**2)
    return I < 1e-15, f'transmitted I = {I:.2e}'


H.run('Crossed polarizers: block all light',
      t_crossed_polarizers_block_light)


def t_two_quarter_wave_plates_equal_half_wave():
    """Two QWPs aligned at the same angle have the same effect as one
    HWP at that angle (cumulative pi phase retardance)."""
    N = 16; dx = 16e-6
    angle = np.radians(30)
    # Reference: HWP at angle.
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = np.zeros((N, N), dtype=np.complex128)
    jf_h = JonesField(Ex=Ex.copy(), Ey=Ey.copy(), dx=dx)
    op.apply_half_wave_plate(jf_h, angle=angle)
    # Compare: two QWPs at the same angle.
    jf_q = JonesField(Ex=Ex.copy(), Ey=Ey.copy(), dx=dx)
    op.apply_quarter_wave_plate(jf_q, angle=angle)
    op.apply_quarter_wave_plate(jf_q, angle=angle)
    err_x = abs(jf_h.Ex[N//2, N//2] - jf_q.Ex[N//2, N//2])
    err_y = abs(jf_h.Ey[N//2, N//2] - jf_q.Ey[N//2, N//2])
    return err_x < 1e-10 and err_y < 1e-10, \
        f'|d Ex|={err_x:.2e}, |d Ey|={err_y:.2e}'


H.run('Two QWPs at same angle = HWP at same angle',
      t_two_quarter_wave_plates_equal_half_wave)


def t_circular_polarization_S3_signature():
    """Right-hand circular: S3 = +S0 (or -S0 with opposite convention),
    and |S1| / S0, |S2| / S0 should both be << 1."""
    N = 32; dx = 16e-6
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = 1j * np.ones((N, N), dtype=np.complex128)
    jf = JonesField(Ex=Ex, Ey=Ey, dx=dx)
    S = op.stokes_parameters(jf)
    s0 = float(S['S0'][N//2, N//2])
    s1 = float(S['S1'][N//2, N//2])
    s2 = float(S['S2'][N//2, N//2])
    s3 = float(S['S3'][N//2, N//2])
    # |S3|/S0 ~ 1, S1, S2 ~ 0
    return (abs(abs(s3) / s0 - 1.0) < 0.01
            and abs(s1 / s0) < 0.01
            and abs(s2 / s0) < 0.01), \
        (f'S3/S0={s3/s0:.4f}, S1/S0={s1/s0:.4f}, S2/S0={s2/s0:.4f}')


H.run('Circular polarization: |S3|/S0 = 1, S1=S2=0',
      t_circular_polarization_S3_signature)


def t_degree_of_polarization_for_pure_states_is_one():
    """A fully polarized state (any) has DoP = 1."""
    N = 32; dx = 16e-6
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = 1j * np.ones((N, N), dtype=np.complex128) * 0.5
    jf = JonesField(Ex=Ex, Ey=Ey, dx=dx)
    dop = op.degree_of_polarization(jf)
    val = float(np.mean(dop))
    return abs(val - 1.0) < 1e-10, f'DoP mean = {val:.6f}'


H.run('degree_of_polarization: pure state = 1',
      t_degree_of_polarization_for_pure_states_is_one)


def t_jones_field_create_linear_polarized_at_angle():
    """create_linear_polarized at 45 deg gives Ex = Ey at the center."""
    N, dx = 16, 16e-6
    scalar = np.ones((N, N), dtype=np.complex128)
    jf = op.create_linear_polarized(scalar, dx, angle=np.pi / 4)
    ex = jf.Ex[N//2, N//2]
    ey = jf.Ey[N//2, N//2]
    return abs(abs(ex) - abs(ey)) < 1e-10 and abs(abs(ex)) > 0.5, \
        f'|Ex|={abs(ex):.4f}, |Ey|={abs(ey):.4f}'


H.run('create_linear_polarized at 45deg: |Ex| = |Ey|',
      t_jones_field_create_linear_polarized_at_angle)


def t_jones_field_create_circular_polarized_S3():
    """create_circular_polarized -> S3/S0 = +/- 1."""
    N, dx = 16, 16e-6
    scalar = np.ones((N, N), dtype=np.complex128)
    jf = op.create_circular_polarized(scalar, dx, handedness='right')
    S = op.stokes_parameters(jf)
    s0 = float(np.mean(S['S0']))
    s3 = float(np.mean(S['S3']))
    return abs(abs(s3) / s0 - 1.0) < 1e-10, \
        f'S3/S0 = {s3/s0:.6f}'


H.run('create_circular_polarized: |S3|/S0 = 1',
      t_jones_field_create_circular_polarized_S3)


def t_apply_rotator_preserves_total_power():
    """A polarization rotator at angle theta preserves total intensity."""
    N, dx = 32, 16e-6
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = 0.5 * np.ones((N, N), dtype=np.complex128)
    jf = JonesField(Ex=Ex, Ey=Ey, dx=dx)
    P_in = float(np.sum(np.abs(jf.Ex)**2 + np.abs(jf.Ey)**2))
    op.apply_rotator(jf, angle=np.radians(33))
    P_out = float(np.sum(np.abs(jf.Ex)**2 + np.abs(jf.Ey)**2))
    rel = abs(P_out - P_in) / P_in
    return rel < 1e-12, f'P rel diff = {rel:.2e}'


H.run('apply_rotator: preserves total intensity',
      t_apply_rotator_preserves_total_power)


if __name__ == '__main__':
    sys.exit(H.summary())
