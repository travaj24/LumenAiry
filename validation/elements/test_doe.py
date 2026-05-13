"""Tests for the 4.3.0 diffractive-lens trio:

* :func:`lumenairy.create_diffractive_lens`
* :func:`lumenairy.create_kinoform`
* :func:`lumenairy.create_fresnel_zone_plate`

Validation strategy:

* API contracts (shape, dtype, unitarity for phase-only types).
* Phase-at-center == 0 for the continuous lens (sign convention).
* Kinoform converges to the continuous lens as ``n_levels`` grows
  (max pointwise complex-transmission error scales with the
  quantization step).
* Each variant focuses a clipped plane wave to ``z = f`` with a
  detectable on-axis peak.
* Binary phase FZP focuses ~3-4x stronger than binary amplitude FZP
  at the design focus (theory: efficiency ratio = 4 / 1 = 4 for
  ideal infinite-aperture FZPs; finite-aperture ratio is somewhat
  less because both designs share equivalent off-axis losses).
* ``n_zones`` aperture crops outside the requested zone.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import numpy as np

import lumenairy as la


H = Harness('doe')

# Common test parameters
N_TEST = 256
DX_TEST = 10e-6      # 10 um pixel
WL_TEST = 1.31e-6    # near-IR
F_TEST = 50e-3       # 50 mm focal length
APERTURE_M = 4e-3    # 4 mm pupil (within paraxial regime at f = 50 mm)


def _build_pupil():
    return la.apply_aperture(
        np.ones((N_TEST, N_TEST), dtype=np.complex128),
        DX_TEST, shape='circular',
        params={'diameter': APERTURE_M},
    )


def _peak_ratio_at_focus(T):
    pupil = _build_pupil()
    E = la.angular_spectrum_propagate(pupil * T, F_TEST, WL_TEST, DX_TEST)
    I = np.abs(E) ** 2
    peak = float(I.max())
    edge = float(np.mean(I[:8, :8]))
    return peak / max(edge, 1e-30)


# ----------------------------------------------------------------------
# create_diffractive_lens
# ----------------------------------------------------------------------

def t_diffractive_lens_shape_dtype():
    T = la.create_diffractive_lens(N_TEST, DX_TEST, F_TEST, WL_TEST)
    ok = (T.shape == (N_TEST, N_TEST) and T.dtype == np.complex128)
    return ok, f'shape={T.shape}, dtype={T.dtype}'


H.run('create_diffractive_lens: shape and dtype', t_diffractive_lens_shape_dtype)


def t_diffractive_lens_unitarity():
    T = la.create_diffractive_lens(N_TEST, DX_TEST, F_TEST, WL_TEST)
    mag_dev = float(np.max(np.abs(np.abs(T) - 1.0)))
    return mag_dev < 1e-12, f'max |T|-1 deviation = {mag_dev:.2e}'


H.run('create_diffractive_lens: phase-only (|T| == 1)',
      t_diffractive_lens_unitarity)


def t_diffractive_lens_center_phase_zero():
    T = la.create_diffractive_lens(N_TEST, DX_TEST, F_TEST, WL_TEST)
    phi = float(np.angle(T[N_TEST // 2, N_TEST // 2]))
    return abs(phi) < 1e-12, f'phi(center) = {phi:.2e} rad'


H.run('create_diffractive_lens: phase(center) == 0 (sign convention)',
      t_diffractive_lens_center_phase_zero)


def t_diffractive_lens_focuses_plane_wave():
    T = la.create_diffractive_lens(N_TEST, DX_TEST, F_TEST, WL_TEST)
    ratio = _peak_ratio_at_focus(T)
    return ratio > 100, f'peak/edge = {ratio:.2e}'


H.run('create_diffractive_lens: focuses a plane wave at z=f',
      t_diffractive_lens_focuses_plane_wave)


def t_diffractive_lens_negative_f():
    T = la.create_diffractive_lens(N_TEST, DX_TEST, -F_TEST, WL_TEST)
    # Diverging lens: |T| == 1 still, but phase reverses sign vs converging
    T_pos = la.create_diffractive_lens(N_TEST, DX_TEST, F_TEST, WL_TEST)
    phase_neg = np.angle(T)
    phase_pos = np.angle(T_pos)
    # Should have opposite sign at every point
    delta = np.mod(phase_neg + phase_pos + np.pi, 2 * np.pi) - np.pi
    max_dev = float(np.max(np.abs(delta)))
    return max_dev < 1e-9, f'max |phi(+f) + phi(-f)| mod 2pi = {max_dev:.2e}'


H.run('create_diffractive_lens: negative focal length inverts phase',
      t_diffractive_lens_negative_f)


# ----------------------------------------------------------------------
# create_kinoform
# ----------------------------------------------------------------------

def t_kinoform_shape_dtype():
    T = la.create_kinoform(N_TEST, DX_TEST, F_TEST, WL_TEST, n_levels=8)
    ok = (T.shape == (N_TEST, N_TEST) and T.dtype == np.complex128
          and np.allclose(np.abs(T), 1.0))
    return ok, (f'shape={T.shape}, dtype={T.dtype}, '
                f'max |T|-1 = {np.max(np.abs(np.abs(T) - 1.0)):.2e}')


H.run('create_kinoform: shape / dtype / phase-only',
      t_kinoform_shape_dtype)


def t_kinoform_high_n_approaches_continuous():
    """High n_levels should converge pointwise to the continuous lens."""
    T_cont = la.create_diffractive_lens(N_TEST, DX_TEST, F_TEST, WL_TEST)
    T_kino = la.create_kinoform(N_TEST, DX_TEST, F_TEST, WL_TEST, n_levels=256)
    step = 2 * np.pi / 256
    # |exp(j*a) - exp(j*b)| <= 2 |sin((a-b)/2)| <= |a-b|
    expected_bound = 2 * np.sin(step / 2)
    max_err = float(np.max(np.abs(T_cont - T_kino)))
    # Allow 2x slack for the wrap offset.
    return max_err < 2 * expected_bound, \
        f'max |T_cont - T_kino_256| = {max_err:.3e}, bound ~ {expected_bound:.3e}'


H.run('create_kinoform: n_levels=256 converges to continuous lens',
      t_kinoform_high_n_approaches_continuous)


def t_kinoform_focuses_plane_wave():
    T = la.create_kinoform(N_TEST, DX_TEST, F_TEST, WL_TEST, n_levels=16)
    ratio = _peak_ratio_at_focus(T)
    return ratio > 100, f'peak/edge = {ratio:.2e}'


H.run('create_kinoform: n_levels=16 focuses a plane wave at z=f',
      t_kinoform_focuses_plane_wave)


def t_kinoform_efficiency_grows_with_levels():
    """Higher n_levels should give a brighter focus."""
    ratios = []
    for n in (2, 4, 8, 16):
        T = la.create_kinoform(N_TEST, DX_TEST, F_TEST, WL_TEST, n_levels=n)
        ratios.append(_peak_ratio_at_focus(T))
    # Monotonically increasing
    ok = all(ratios[i] <= ratios[i + 1] * 1.1 for i in range(len(ratios) - 1))
    return ok, f'peak/edge for n_levels (2,4,8,16) = {[f"{r:.2e}" for r in ratios]}'


H.run('create_kinoform: focal efficiency rises with n_levels',
      t_kinoform_efficiency_grows_with_levels)


def t_kinoform_rejects_n_levels_lt_2():
    try:
        la.create_kinoform(N_TEST, DX_TEST, F_TEST, WL_TEST, n_levels=1)
    except ValueError:
        return True, 'raised ValueError as expected'
    return False, 'should have raised for n_levels=1'


H.run('create_kinoform: rejects n_levels < 2',
      t_kinoform_rejects_n_levels_lt_2)


# ----------------------------------------------------------------------
# create_fresnel_zone_plate
# ----------------------------------------------------------------------

def t_fzp_amplitude_binary_open_fraction():
    """Binary amplitude FZP: roughly half-open over a symmetric grid."""
    T = la.create_fresnel_zone_plate(
        N_TEST, DX_TEST, F_TEST, WL_TEST, binary=True)
    open_frac = float(np.mean(np.abs(T) > 0.5))
    return 0.3 < open_frac < 0.7, f'open fraction = {open_frac:.3f}'


H.run('create_fresnel_zone_plate (binary amp): ~half open',
      t_fzp_amplitude_binary_open_fraction)


def t_fzp_phase_binary_unitary():
    """Binary phase FZP: |T| = 1 everywhere when no aperture given."""
    T = la.create_fresnel_zone_plate(
        N_TEST, DX_TEST, F_TEST, WL_TEST, binary=False)
    mag_dev = float(np.max(np.abs(np.abs(T) - 1.0)))
    return mag_dev < 1e-12, f'max |T|-1 = {mag_dev:.2e}'


H.run('create_fresnel_zone_plate (binary phase): |T|=1',
      t_fzp_phase_binary_unitary)


def t_fzp_n_zones_crop():
    n_zones = 5
    T = la.create_fresnel_zone_plate(
        N_TEST, DX_TEST, F_TEST, WL_TEST, binary=True, n_zones=n_zones)
    r5 = np.sqrt(n_zones * WL_TEST * F_TEST)
    x = (np.arange(N_TEST) - N_TEST / 2) * DX_TEST
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X * X + Y * Y)
    outside = r > r5 * 1.01
    inside = r < r5 * 0.99
    ok = (np.all(T[outside] == 0)
          and np.any(np.abs(T[inside]) > 0.5))
    return ok, (f'r_n_zones = {r5*1e3:.3f}mm, '
                f'outside zero={np.all(T[outside] == 0)}, '
                f'inside open={np.any(np.abs(T[inside]) > 0.5)}')


H.run('create_fresnel_zone_plate: n_zones aperture crops cleanly',
      t_fzp_n_zones_crop)


def t_fzp_amplitude_focuses():
    T = la.create_fresnel_zone_plate(
        N_TEST, DX_TEST, F_TEST, WL_TEST, binary=True)
    ratio = _peak_ratio_at_focus(T)
    return ratio > 10, f'peak/edge = {ratio:.2e}'


H.run('create_fresnel_zone_plate (binary amp): focuses plane wave',
      t_fzp_amplitude_focuses)


def t_fzp_phase_focuses_better_than_amplitude():
    """Phase FZP should be brighter at the focus than the same-design
    amplitude FZP (the classical 4x diffraction-efficiency improvement;
    finite-aperture peak ratio is typically 2-4x)."""
    T_amp = la.create_fresnel_zone_plate(
        N_TEST, DX_TEST, F_TEST, WL_TEST, binary=True)
    T_phase = la.create_fresnel_zone_plate(
        N_TEST, DX_TEST, F_TEST, WL_TEST, binary=False)
    r_amp = _peak_ratio_at_focus(T_amp)
    r_phase = _peak_ratio_at_focus(T_phase)
    ratio = r_phase / r_amp
    return 1.5 < ratio < 8, \
        f'phase/amp peak ratio = {ratio:.2f} (theory ~4x ideal, 2-4x finite-aperture)'


H.run('create_fresnel_zone_plate: phase ~3x stronger than amplitude',
      t_fzp_phase_focuses_better_than_amplitude)


def t_fzp_center_offset():
    """Center kwarg should shift the focusing peak off-axis by the
    same amount in the focal plane."""
    offset = 50e-6  # shift the lens 50 um in X
    T = la.create_fresnel_zone_plate(
        N_TEST, DX_TEST, F_TEST, WL_TEST, binary=False,
        center=(offset, 0.0))
    pupil = _build_pupil()
    E = la.angular_spectrum_propagate(pupil * T, F_TEST, WL_TEST, DX_TEST)
    I = np.abs(E) ** 2
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    x_peak_m = (ix - N_TEST / 2) * DX_TEST
    # Expect peak at +offset (lens centered there, so chief comes to
    # the off-axis focus).
    err = abs(x_peak_m - offset)
    return err < 2 * DX_TEST, \
        f'peak x = {x_peak_m*1e6:.2f} um, expected {offset*1e6:.2f} um'


H.run('create_fresnel_zone_plate: center kwarg shifts focus',
      t_fzp_center_offset)


if __name__ == '__main__':
    import sys
    sys.exit(H.summary())
