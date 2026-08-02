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
from lumenairy.elements import _lens_traced as LT

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


#: The rms RELATIVE |E| difference between the two EXACT ray constructions of
#: the fixture below -- rays along ``grad(W)`` (what ``preserve_input_phase=
#: False`` builds) versus rays along ``grad(W + a/k0)`` (Fermat's stationary
#: point of the total entrance eikonal, i.e. the truth).  Pure raytrace plus
#: closed-form input phase, exit amplitude ``|E_in|/sqrt(|det dX/dp|)`` from
#: the exact landing map: no fit, no Newton inverse, no coarse lattice, no
#: upsample, no ``preserve_input_phase`` code path.  Measured 2026-08-01 by
#: ``validation/repro_traced_carrier_121/recon_remap_residual_oracle.py``;
#: the worst pointwise value over the same mask is 2.921e-04 of peak.
_EXACT_REMAP_DAMP_RMS_REL = 3.353e-04


def test_remap_carries_injected_residual(_setup):
    """A gentle KNOWN residual multiplied onto the sphere input must appear
    in the 'remap' exit but not the False exit: the difference field's phase
    must correlate with the injected residual (transported), while
    pip=False discards it entirely.

    2026-08-01 -- NICHE C6 CHANGED THE MECHANISM, AND THE PIN'S PREMISE WAS
    WRONG PHYSICS.

    PIN WAS: ``np.allclose(|E_false|, |E_remap|)`` -- "the two exits must
    differ by a PHASE-ONLY factor".  ``REMAP_STATIONARY_PHASE_LAUNCH``
    (niche C6) launches 'remap' along ``grad(W + a_fit)`` rather than
    ``grad(W)``, so the ray TUBE differs between the two modes and
    ``ray_density``'s ``1/sqrt(|det J|)`` follows it -- by design
    ("the ``ray_density`` Jacobian follows the augmented map automatically").

    THE ORACLE THAT ADJUDICATED, sharing no code with the element's fit,
    Newton inverse, lattice or upsample: two EXACT skew ray traces of this
    same fixture, one launched along ``grad(W)`` and one along
    ``grad(W + a/k0)``, each with the exit amplitude
    ``|E_in| / sqrt(|det dX/dp|)`` taken from its own exact landing map.
    They differ in |E| by ``_EXACT_REMAP_DAMP_RMS_REL`` = 3.353e-04 rms
    relative (worst 2.921e-04 of peak).  **So an amplitude difference is
    REQUIRED here; a phase-only operator would be the wrong answer.**
    The library delivers 2.740e-04 rms relative (worst 2.764e-04 of peak) --
    within 18 % of the exact prediction in rms and 5 % at the worst point --
    while with ``REMAP_STATIONARY_PHASE_LAUNCH = False`` it delivers
    1.6e-16, i.e. exactly the phase-only behaviour the old pin asserted.

    PIN IS NOW: the old assertion verbatim on the C6-off arm (fail-before),
    and on the shipped path the amplitude change must be PRESENT and within a
    factor of two of the exact-ray prediction -- which fails both if C6 is
    reverted (0) and if the ray-density amplitude stops tracking the augmented
    map (wrong magnitude).  The phase assertions are unchanged and still pass
    at ``std(dphi)/std(inj)`` = 1.0376 (C6 on) / 1.0459 (C6 off).

    NOTE the exact ray trace also prices C6 on this fixture: complex relL2
    against the true (stationary) construction is 5.554e-03 with C6 on and
    5.214e-03 with it off -- a 6.5 % cost, because this residual is gentle
    (the two exact constructions differ by only 3.3e-04 rad of phase, so
    there is almost nothing for C6 to restore and its degree-4 fit of an
    ``r^4 x Gaussian`` residual is the larger term).  Both are 31x better
    than ``pip=False``'s 1.746e-01.  On a fixture where ``grad a`` is large
    the same oracle scores C6 140x BETTER (see
    ``test_niche_s12_remap_sampling.py``)."""
    N, dx, R_in, presc, E_in = _setup
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    w = 0.6e-3
    resid = 0.5 * (r2 / w ** 2) ** 2 * np.exp(-r2 / (2 * w * w))  # gentle r^4
    E_res = (E_in * np.exp(1j * resid)).astype(np.complex128)

    def _call(pip, launch=None):
        old = LT.REMAP_STATIONARY_PHASE_LAUNCH
        if launch is not None:
            LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(la.apply_real_lens_traced(
                    E_res, prescription=presc, wavelength=_WL, dx=dx,
                    carrier=R_in, ray_subsample=8, n_workers=1,
                    on_undersample='silent', on_noncollimated='silent',
                    amplitude_model='ray_density', preserve_input_phase=pip))
        finally:
            LT.REMAP_STATIONARY_PHASE_LAUNCH = old

    outs = {False: _call(False), 'remap': _call('remap')}
    a_f, a_r = np.abs(outs[False]), np.abs(outs['remap'])

    # (a) FAIL-BEFORE: with the C6 launch off, 'remap' really is phase-only,
    #     to the last bit -- so the amplitude motion below is C6's and nothing
    #     else's.
    a_r0 = np.abs(_call('remap', launch=False))
    assert np.allclose(a_f, a_r0, atol=1e-12 + 1e-6 * a_f.max()), (
        float(np.abs(a_f - a_r0).max()))

    # (b) the shipped path: the ray tube DOES change, by the amount two exact
    #     ray traces of the two congruences say it must.
    msk = a_f > 0.05 * a_f.max()
    d_rms = float(np.sqrt((a_f[msk] * (a_r[msk] - a_f[msk]) ** 2).sum()
                          / (a_f[msk] ** 3).sum()))
    assert d_rms > 0.5 * _EXACT_REMAP_DAMP_RMS_REL, d_rms
    assert d_rms < 2.0 * _EXACT_REMAP_DAMP_RMS_REL, d_rms

    m = msk
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
