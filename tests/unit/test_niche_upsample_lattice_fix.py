"""Regression pins for the coarse->fine upsample LATTICE bug (audit
AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24, walk-hunter / lattice-probe
sections): ``apply_real_lens_traced``'s OPL / ray-density / valid-mask maps
were displaced diagonally toward the (-x, -y) grid corner (and radially
mis-scaled) whenever ``ray_subsample`` did NOT divide ``N``, because the
``map_coordinates`` coordinate stack used ``ii * Ns / N`` (``Ns =
ceil(N/sub)``) instead of the exact ``ii / sub``.  The displacement is
``(N/2) * (Ns*sub - N) / N`` fine pixels -- measured on the design-121
final-leg conditions: -6.100 um at N=8192 / sub=50 (predicted -6.11),
-12.187 at sub=48 (predicted -12.22), -14.467 at sub=51 (predicted -14.51),
and EXACTLY 0 for divisor subs.  This drove the traced carrier chain's
diagonal focus walk: the F-C fine-retrace rescale (``rs_fine =
round(rs * cur_dx / dx_fine)``) routinely produces non-divisor values.

These tests are CI-safe (synthetic singlet, no external assets) and pin the
exit-field centroid to the axis for BOTH a divisor and a non-divisor
``ray_subsample``.  Pre-fix, the non-divisor case shifts by ~3 pixels here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la

_WL = 1.31e-6


def _singlet(R1, R2, d, glass, ap, name='s'):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


def _centroid(E, dx):
    E = np.asarray(E)
    n = E.shape[-1]
    x = (np.arange(n) - n // 2) * dx
    I = np.abs(E) ** 2
    P = I.sum()
    return (float((I.sum(axis=0) * x).sum() / P),
            float((I.sum(axis=1) * x).sum() / P))


@pytest.fixture(scope='module')
def _setup():
    N, dx = 512, 10e-6
    w, R_in = 0.6e-3, 50e-3
    presc = _singlet(9.0e-3, -9.0e-3, 1.5e-3, 'N-BK7', 2.4e-3, 'lat')
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    k0 = 2.0 * np.pi / _WL
    S = np.sign(R_in) * (np.sqrt(r2 + R_in * R_in) - abs(R_in))
    E_in = (np.exp(-r2 / w ** 2) * np.exp(1j * k0 * S)
            ).astype(np.complex128)
    return N, dx, R_in, presc, E_in


def _traced_centroid(_setup, rs, **extra):
    N, dx, R_in, presc, E_in = _setup
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_out = la.apply_real_lens_traced(
            E_in, prescription=presc, wavelength=_WL, dx=dx, carrier=R_in,
            ray_subsample=rs, n_workers=1, on_undersample='silent',
            on_noncollimated='silent', **extra)
    return _centroid(E_out, dx)


@pytest.mark.parametrize('rs', [8, 7])
def test_ray_density_exit_centroid_on_axis(rs, _setup):
    """ray_density amplitude + pure traced phase: exit centroid must stay on
    the axis for divisor (8 | 512) AND non-divisor (7) ray_subsample.
    Pre-fix, rs=7 displaced the exit by (256*(74*7-512)/512) = 3 px = 30 um
    diagonally; the gate is 0.25 px."""
    dx = _setup[1]
    cx, cy = _traced_centroid(_setup, rs, amplitude_model='ray_density',
                              preserve_input_phase=False)
    assert abs(cx) < 0.25 * dx, (rs, cx / dx)
    assert abs(cy) < 0.25 * dx, (rs, cy / dx)


@pytest.mark.parametrize('rs', [8, 7])
def test_default_screen_exit_centroid_on_axis(rs, _setup):
    """Default (screen-amplitude, preserve_input_phase=True) path: the OPL
    map displacement alone must not walk the exit centroid off-axis for a
    non-divisor ray_subsample."""
    dx = _setup[1]
    cx, cy = _traced_centroid(_setup, rs)
    assert abs(cx) < 0.25 * dx, (rs, cx / dx)
    assert abs(cy) < 0.25 * dx, (rs, cy / dx)


def test_remap_no_double_count_on_pure_sphere(_setup):
    """preserve_input_phase='remap' (audit S6.7): for a PURE carrier-sphere
    input the de-chirped residual is ~0, so 'remap' must coincide with
    preserve_input_phase=False to high accuracy (the no-double-count pin,
    same guard class as the R8 F3 pins)."""
    N, dx, R_in, presc, E_in = _setup
    outs = {}
    for pip in (False, 'remap'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outs[pip] = np.asarray(la.apply_real_lens_traced(
                E_in, prescription=presc, wavelength=_WL, dx=dx,
                carrier=R_in, ray_subsample=8, n_workers=1,
                on_undersample='silent', on_noncollimated='silent',
                amplitude_model='ray_density', preserve_input_phase=pip))
    num = np.linalg.norm(outs['remap'] - outs[False])
    den = np.linalg.norm(outs[False])
    assert num / den < 1e-3, num / den


def test_remap_requires_ray_density(_setup):
    N, dx, R_in, presc, E_in = _setup
    with pytest.raises(ValueError, match='remap'):
        la.apply_real_lens_traced(
            E_in, prescription=presc, wavelength=_WL, dx=dx, carrier=R_in,
            ray_subsample=8, n_workers=1, preserve_input_phase='remap')


def test_remap_carries_injected_residual(_setup):
    """A gentle KNOWN residual multiplied onto the sphere input must appear
    in the 'remap' exit but not the False exit: the difference field's phase
    must correlate with the injected residual (transported), while
    pip=False discards it entirely."""
    N, dx, R_in, presc, E_in = _setup
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    w = 0.6e-3
    resid = 0.5 * (r2 / w ** 2) ** 2 * np.exp(-r2 / (2 * w * w))  # gentle r^4
    E_res = (E_in * np.exp(1j * resid)).astype(np.complex128)
    outs = {}
    for pip in (False, 'remap'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outs[pip] = np.asarray(la.apply_real_lens_traced(
                E_res, prescription=presc, wavelength=_WL, dx=dx,
                carrier=R_in, ray_subsample=8, n_workers=1,
                on_undersample='silent', on_noncollimated='silent',
                amplitude_model='ray_density', preserve_input_phase=pip))
    # the two exits must differ by a PHASE-ONLY factor of ~the residual's
    # magnitude: same |E| (same amplitude model), different phase
    a_f, a_r = np.abs(outs[False]), np.abs(outs['remap'])
    assert np.allclose(a_f, a_r, atol=1e-12 + 1e-6 * a_f.max())
    m = a_f > 0.05 * a_f.max()
    dphi = np.angle(outs['remap'][m] * np.conj(outs[False][m]))
    # injected residual rms over the same support (entrance~exit for this
    # gentle singlet): the carried phase must be nonzero and of the same
    # order as the injected content
    inj = resid[m]
    assert np.std(dphi) > 0.3 * np.std(inj)
    assert np.std(dphi) < 3.0 * np.std(inj)


def test_divisor_and_nondivisor_agree():
    """The exit fields at rs=8 (divisor) and rs=7 (non-divisor) sample the
    same smooth OPL at slightly different ray pitches -- post-fix they must
    agree closely (pre-fix the rs=7 field is displaced by 3 px, driving the
    normalized difference to O(1))."""
    N, dx = 512, 10e-6
    w, R_in = 0.6e-3, 50e-3
    presc = _singlet(9.0e-3, -9.0e-3, 1.5e-3, 'N-BK7', 2.4e-3, 'lat')
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    k0 = 2.0 * np.pi / _WL
    S = np.sign(R_in) * (np.sqrt(r2 + R_in * R_in) - abs(R_in))
    E_in = (np.exp(-r2 / w ** 2) * np.exp(1j * k0 * S)
            ).astype(np.complex128)
    outs = {}
    for rs in (8, 7):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outs[rs] = np.asarray(la.apply_real_lens_traced(
                E_in, prescription=presc, wavelength=_WL, dx=dx,
                carrier=R_in, ray_subsample=rs, n_workers=1,
                on_undersample='silent', on_noncollimated='silent',
                amplitude_model='ray_density',
                preserve_input_phase=False))
    num = np.linalg.norm(outs[8] - outs[7])
    den = np.linalg.norm(outs[8])
    assert num / den < 0.05, num / den


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
