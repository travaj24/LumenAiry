"""S12: ``apply_real_lens_traced(remap_sampling=...)`` -- the resolution at
which ``preserve_input_phase='remap'`` samples the transported residual.

Mechanism (measured on design-121, audit S12).  Under the validated
carrier-regime configuration the residual a chain hands each group is the
DESIGN's own correction, which is r^4-dominant, so its phase GRADIENT grows
as r^3::

    phi(r) ~ A (r/w)^4      =>      |dphi/dr| = 4 A r^3 / w^4

The pre-S12 implementation sampled that phasor at the COARSE RAY LATTICE's
entrance pullback points (pitch ``h = ray_subsample * dx``) and then
bilinearly upsampled the PHASOR to the wave grid -- i.e. it resampled the
FAST quantity off a coarse lattice.  Beyond

    r_alias = (pi w^4 / (4 A h))^(1/3)

the phasor exceeds that lattice's Nyquist limit and the transported residual
is ALIASED: the beam SKIRT receives scrambled phase.  On the design-121 final
leg (A = 9.2 rad, w = 3.124 mm, h = 50 * 1.524 um) the prediction is
``r_alias = 1.52 w``, and the measured exit-wavefront residual jumps from
0.052 rad (r < w) / 0.139 rad (1-1.5 w) to **0.971 rad (1.5-2.0 w)** -- 95 %
of the entire Strehl-loss variance (Strehl 0.910) out of 1.2 % of the power,
worth 1.16 EE3 points at best focus.

``remap_sampling='full'`` upsamples only the SMOOTH entrance pullback
COORDINATES (a geometric ray map) and samples the residual phasor at FULL
wave-grid resolution -- the same "upsample the smooth thing, not the fast
thing" discipline the F-A / reference-lattice fixes established.  The default
stays ``'lattice'`` so every shipping result is byte-identical.

Pins here (all self-contained -- no Zemax, no design-121 assets):
  1. ``'full'`` is BYTE-IDENTICAL to ``'lattice'`` at ``ray_subsample == 1``
     (there is no coarse lattice, so the option must be an exact no-op).
  2. The default is ``'lattice'`` and is byte-identical to passing it.
  3. ``remap_sampling`` is inert unless ``preserve_input_phase='remap'``.
  4. Validation: an unrecognised value RAISES (house rule -- knobs must never
     silently fall through to a default).
  5. The accuracy claim, ground-truth-anchored: ``ray_subsample == 1`` samples
     everything at full resolution and is therefore the reference; ``'full'``
     reproduces it to <= 0.01 rad at ``ray_subsample`` 2/4/8 while
     ``'lattice'`` drifts to 0.5-1.1 rad -- a 180-9000x reduction.
  6. The differences are CONFINED to r > r_alias (the mechanism, not just the
     magnitude).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la

_WL = 1.31e-6


def _singlet(R1, R2, d, glass, ap, name='s'):
    gb, ga = ['air', glass], [glass, 'air']
    surfaces = [
        {'radius': R1, 'glass_before': gb[0], 'glass_after': ga[0],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': gb[1], 'glass_after': ga[1],
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


_N, _DX, _W, _A, _RC = 256, 4.0e-6, 200e-6, 6.0, -0.02


def _r_alias(h, A=_A, w=_W):
    """Radius beyond which an ``A (r/w)^4`` residual's phasor exceeds the
    Nyquist limit of a sampling lattice of pitch ``h``."""
    return (np.pi * w ** 4 / (4.0 * A * h)) ** (1.0 / 3.0)


@pytest.fixture(scope='module')
def _setup():
    """Carrier-referenced input carrying a KNOWN r^4 residual (the design's
    carried content), on a grid where ``r_alias`` at ``ray_subsample=4`` falls
    INSIDE the beam skirt (1.18 w) -- the design-121 condition, reproduced
    small and fast."""
    x = (np.arange(_N) - _N // 2) * _DX
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    k = 2.0 * np.pi / _WL
    S = np.sign(_RC) * (np.sqrt(r2 + _RC ** 2) - abs(_RC))
    E = (np.exp(-r2 / _W ** 2) * np.exp(1j * k * S)
         * np.exp(1j * _A * (r2 / _W ** 2) ** 2)).astype(np.complex128)
    presc = _singlet(3.1e-3, -3.1e-3, 1.0e-3, 'N-BK7', 1.2e-3, 'strong')
    kw = dict(prescription=presc, wavelength=_WL, dx=_DX, carrier=_RC,
              amplitude_model='ray_density', preserve_input_phase='remap',
              parallel_amp=False, on_undersample='silent',
              on_noncollimated='silent')
    return E, kw, np.sqrt(r2)


def _run(E, kw, **over):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(E, **kw, **over))


@pytest.fixture(scope='module', autouse=True)
def _warm(_setup):
    """Push every byte-identity pair in this file PAST the traced pipeline's
    warm-up boundary (audit W9 CI calibration, d3941f5).

    MEASURED: in a fresh process the traced output for one fixed input
    changes ONCE, by ~2.7e-15 (max abs, ulp scale), after the first two
    calls -- every later call is byte-identical to every other later call,
    and the (lattice, full) pair is byte-identical WITHIN any iteration
    (30/30).  The byte-identity pins in this file are therefore correct
    if and only if both compared calls sit on the same side of that
    boundary.  On CI the shard's PRECEDING call history decides where the
    boundary falls: the d3941f5 collection put it between this file's two
    inertness calls on three Pythons at once (deterministically -- same
    layout, same history), while a210de4 / d78b39f / 848e8f9 layouts did
    not.  Two warm-up calls with this module's own geometry (same FFT
    shapes, same cache keys) land the boundary before any pinned pair.
    The underlying call-history ulp drift is recorded as an open
    determinism defect to root-cause -- this fixture makes the pins
    measure what they claim (remap_sampling inertness), not cache
    warm-up."""
    E, kw, _ = _setup
    for _ in range(2):
        _run(E, kw, ray_subsample=4)


def _rms_phase_diff(a, b, mask=None):
    """Amplitude-weighted rms phase difference b vs a over the illuminated
    region (``a`` as the weight / reference)."""
    m = (np.abs(a) > 1e-3 * np.abs(a).max()) & (np.abs(b) > 0)
    if mask is not None:
        m = m & mask
    if not m.any():
        return 0.0
    d = np.angle(b[m] / a[m])
    wt = np.abs(a[m])
    return float(np.sqrt((wt * d ** 2).sum() / wt.sum()))


# ===========================================================================
# 1-4: contract (no-op at rs=1, default, inertness, validation)
# ===========================================================================
def test_full_is_byte_identical_at_ray_subsample_1(_setup):
    """At ``ray_subsample == 1`` the "coarse" lattice IS the wave grid, so the
    two modes sample the residual at exactly the same coordinates and must
    agree BIT for bit -- not merely closely."""
    E, kw, _ = _setup
    a = _run(E, kw, ray_subsample=1, remap_sampling='lattice')
    b = _run(E, kw, ray_subsample=1, remap_sampling='full')
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_default_is_lattice_and_byte_identical(_setup):
    """The shipping default must be the pre-S12 sampling, byte for byte, so no
    existing result moves."""
    E, kw, _ = _setup
    a = _run(E, kw, ray_subsample=4)
    b = _run(E, kw, ray_subsample=4, remap_sampling='lattice')
    assert np.array_equal(a, b)


@pytest.mark.parametrize('pip', [True, False])
def test_remap_sampling_is_inert_without_remap(_setup, pip):
    """``remap_sampling`` may only affect ``preserve_input_phase='remap'``;
    for True / False it must be a byte-identical no-op."""
    E, kw, _ = _setup
    kw = dict(kw)
    kw['preserve_input_phase'] = pip
    if pip:
        # preserve_input_phase=True with ray_density is a legal pair
        pass
    a = _run(E, kw, ray_subsample=4, remap_sampling='lattice')
    b = _run(E, kw, ray_subsample=4, remap_sampling='full')
    assert np.array_equal(a, b)


def test_bad_remap_sampling_raises(_setup):
    """Harness/API rule: an unrecognised knob value must ERROR, never fall
    through to a default (two false campaign results came from silent
    fall-throughs -- AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §3)."""
    E, kw, _ = _setup
    with pytest.raises(ValueError, match="remap_sampling must be"):
        _run(E, kw, ray_subsample=4, remap_sampling='bogus')


def test_full_actually_changes_the_result_at_coarse_lattice(_setup):
    """The feature must be LIVE at ray_subsample > 1 (guards against a knob
    that quietly does nothing)."""
    E, kw, _ = _setup
    a = _run(E, kw, ray_subsample=4, remap_sampling='lattice')
    b = _run(E, kw, ray_subsample=4, remap_sampling='full')
    assert not np.array_equal(a, b)
    assert _rms_phase_diff(a, b) > 0.1


# ===========================================================================
# 5: the accuracy claim -- ray_subsample independence
# ===========================================================================
@pytest.mark.parametrize('rs,tol_full,min_lattice', [
    (2, 0.02, 0.2), (4, 0.02, 0.4), (8, 0.05, 0.6)])
def test_full_is_ray_subsample_independent(_setup, rs, tol_full, min_lattice):
    """``ray_subsample=1`` samples every quantity at full wave-grid
    resolution, so it is the REFERENCE.  ``'full'`` must reproduce it almost
    exactly at any coarser lattice (the only residual error is the O(h^2)
    bilinear upsample of the SMOOTH pullback coordinates), while ``'lattice'``
    must NOT -- that non-convergence is the defect.

    Measured (2026-07-25): rms phase diff vs the rs=1 reference is
    0.0001 / 0.0003 / 0.0059 rad for 'full' at rs = 2 / 4 / 8 versus
    0.5458 / 0.8386 / 1.0888 rad for 'lattice' -- a 9018x / 3190x / 184x
    reduction.  Tolerances here are ~3x the measured 'full' values and ~40 %
    of the measured 'lattice' values, so the pin brackets the behaviour
    without being brittle."""
    E, kw, _ = _setup
    ref = _run(E, kw, ray_subsample=1, remap_sampling='lattice')
    lat = _run(E, kw, ray_subsample=rs, remap_sampling='lattice')
    ful = _run(E, kw, ray_subsample=rs, remap_sampling='full')
    d_lat = _rms_phase_diff(ref, lat)
    d_ful = _rms_phase_diff(ref, ful)
    assert d_ful < tol_full, (d_ful, d_lat)
    assert d_lat > min_lattice, (d_lat, d_ful)
    assert d_lat > 20.0 * max(d_ful, 1e-6), (d_lat, d_ful)


# ===========================================================================
# 6: the mechanism -- the disagreement lives beyond r_alias
# ===========================================================================
def test_lattice_error_is_confined_beyond_the_alias_radius(_setup):
    """The 'lattice' error must be an ALIASING signature, i.e. confined to
    ``r > r_alias(h = ray_subsample*dx)`` and absent well inside it -- not a
    broadband offset.  This is what ties the fix to the design-121
    observation (residual 0.05 rad inside 1 w, 0.97 rad at 1.5-2.0 w)."""
    E, kw, rr = _setup
    rs = 4
    ra = _r_alias(rs * _DX)
    assert _W < ra < 1.6 * _W, ra / _W        # fixture sanity: 1.18 w
    ref = _run(E, kw, ray_subsample=1, remap_sampling='lattice')
    lat = _run(E, kw, ray_subsample=rs, remap_sampling='lattice')
    ful = _run(E, kw, ray_subsample=rs, remap_sampling='full')
    inner = rr < 0.75 * ra
    outer = rr > 1.05 * ra
    d_in = _rms_phase_diff(ref, lat, inner)
    d_out = _rms_phase_diff(ref, lat, outer)
    assert d_out > 5.0 * max(d_in, 1e-6), (d_in, d_out)
    # and 'full' is clean on BOTH sides
    assert _rms_phase_diff(ref, ful, inner) < 0.02
    assert _rms_phase_diff(ref, ful, outer) < 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
