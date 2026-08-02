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

==========================================================================
2026-08-01 -- NICHE C6 COLLAPSED THE FULL-vs-LATTICE DIFFERENCE, AND THE
EXACT RAY TRACE SAYS THE COLLAPSE IS THE FIX, NOT A LOST CAPABILITY.
==========================================================================

``REMAP_STATIONARY_PHASE_LAUNCH`` (niche C6) launches ``'remap'``'s rays along
``grad(W + a_fit)`` instead of ``grad(W)``, and adds ``a_fit`` to the traced
OPL, so the transported phasor carries only the LEFTOVER ``a - a_fit``.  On
this fixture the injected residual is ``A (r^2/w^2)^2`` -- an exactly
degree-4 polynomial -- so ``a_fit`` absorbs essentially all of it inside the
fit disc and there is almost nothing left to sample on any lattice.  The
aliasing mechanism items 5 and 6 pin is therefore STRUCTURALLY ABSENT, not
merely small.

MEASURED on this exact fixture (``validation/repro_traced_carrier_121/
recon_s12_measure.py``), rms phase difference against the ``rs=1`` reference,
'lattice' / 'full' / and the two modes against EACH OTHER:

    rs        C6 OFF (= pre-C6)              C6 ON (= shipped)
     2   5.4576e-01 / 6.0518e-05 / 6.7401e-01   1.6340e-02 / 1.6340e-02 / 6.06e-07
     4   8.3863e-01 / 2.6291e-04 / 9.2267e-01   5.6227e-02 / 5.6227e-02 / 2.69e-06
     8   1.0888e+00 / 5.9216e-03 / 1.1504e+00   9.1257e-02 / 9.1257e-02 / 1.09e-05

The C6-OFF column reproduces this file's own 2026-07-25 numbers to every
printed digit, so C6 is the whole of the change.  The two modes now differ by
6e-07 to 1e-05 rad instead of 0.67 to 1.15 rad -- a collapse of ~6 orders of
magnitude, matching ``docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md``'s
``remap_sampling`` row on design 121 (full-vs-lattice EE3 -17.73 -> +0.0988,
the sign flipped).

THE ORACLE THAT ADJUDICATED, and it shares no code with any of it
(``recon_s12_oracle.py``): an EXACT skew ray trace of this fixture --
``lumenairy.raytrace`` plus the closed-form input phase, launched along
``grad(k0 W + a)/k0`` (Fermat's stationary point of the total entrance
eikonal), exit phase ``k0 (V + W) + a``, exit amplitude
``|E_in| / sqrt(|det dX/dp|)`` from the exact landing map, scattered onto the
wave grid.  No forward-map fit, no Newton inverse, no coarse launch lattice,
no bilinear upsample, no ``remap_sampling`` code path.  VALIDATED on a control
(``A = 0``, ``preserve_input_phase=False``, ``rs=1``) where the library must
reproduce it: agreement 1.2671e-03 rad after removing ONE global piston (the
raw difference is a constant 1.428842 rad with standard deviation 1.27e-03 and
no radial structure -- an eikonal is defined up to a constant).

    rms phase vs the EXACT ray trace, same pixels for every row:

      rs / mode          C6 OFF (pre-C6)     C6 ON (shipped)
      1   lattice          1.2538e+00          8.8177e-03
      2   lattice          1.2532e+00          8.7784e-03
      2   full             1.2538e+00          8.7785e-03
      4   lattice          1.1926e+00          8.5979e-03
      4   full             1.2538e+00          8.5983e-03
      8   lattice          1.2429e+00          1.0524e-02
      8   full             1.2538e+00          1.0526e-02

**The shipped library is 140x closer to the exact ray trace than the state
these pins were written in, and 'full' is no longer distinguishable from
'lattice' because there is nothing left for either to sample.**  Note what
that table also says about the ORIGINAL claim: pre-C6, ``'full'`` converged
to a ``ray_subsample=1`` REFERENCE THAT WAS ITSELF 1.2538 rad from the truth
-- the "9018x reduction" was convergence to the wrong answer.  Two exact ray
traces differing only in the launch eikonal (``grad(W)`` vs ``grad(W + a)``)
are 1.2516 rad apart on this fixture, which is the entire pre-C6 error.

WHAT CHANGED IN THE TESTS.  Items 5 and 6, and the "the knob is live" pin,
are kept WORD FOR WORD in an arm that sets ``REMAP_STATIONARY_PHASE_LAUNCH =
False`` -- the library state they were calibrated in, where they are still
true and still discriminating -- and each grows a second arm pinning the
shipped behaviour: the two modes agree to 1e-04 rad, are still not
byte-identical (so the knob has not quietly died), and their residual
disagreement with the ``rs=1`` reference is IDENTICAL for both modes, i.e.
it is the C6 launch's own ``ray_subsample`` dependence and not a sampling
effect.  No tolerance was loosened on any pin that still applies.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

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


def _run(E, kw, launch=None, **over):
    """One element call.  ``launch`` forces ``REMAP_STATIONARY_PHASE_LAUNCH``
    (niche C6) and restores it; ``None`` leaves the shipped default, which is
    what every pin in sections 1-4 runs at."""
    old = LT.REMAP_STATIONARY_PHASE_LAUNCH
    if launch is not None:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(E, **kw, **over))
    finally:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = old


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
    RESOLVED (same wave): the drift was fft_infra's ESTIMATE->MEASURE
    plan auto-promote tripping its 5-transform per-key counter inside
    the traced pipeline's second call; auto-promote now ships OFF
    (audit W9, tests/unit/test_niche_audit_w9_traced_determinism.py
    pins the contract), which makes this fixture REDUNDANT (measured
    10/10 green with it neutered).  Kept anyway as defense-in-depth:
    if any test earlier in a shard opts back in without restoring
    (set_fft_auto_promote(True) / set_pyfftw_planner), the warm-up
    keeps these byte-identity pins measuring remap_sampling inertness
    rather than that leak."""
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
    that quietly does nothing).

    PIN WAS: ``_rms_phase_diff(lattice, full) > 0.1`` rad at the shipped
    default.  PIN IS NOW: that bar, verbatim, on the ``REMAP_STATIONARY_PHASE_
    LAUNCH = False`` arm (measured 9.2267e-01), plus a shipped-default arm
    that pins the knob still MOVES the field (not byte-identical) while the
    difference has collapsed to 2.7e-06 rad.

    WHY IT MOVED: niche C6 absorbs this fixture's exactly-degree-4 residual
    into the launch eikonal ``a_fit``, so the transported phasor is
    ``a - a_fit ~ 0`` and neither sampling mode has anything left to alias --
    see the module docstring for the exact-ray-trace adjudication (the shipped
    library is 140x closer to an exact skew ray trace than the C6-off state
    this bar was calibrated in)."""
    E, kw, _ = _setup
    # (a) the original pin, in the library state it was measured in.
    a0 = _run(E, kw, ray_subsample=4, remap_sampling='lattice', launch=False)
    b0 = _run(E, kw, ray_subsample=4, remap_sampling='full', launch=False)
    assert not np.array_equal(a0, b0)
    assert _rms_phase_diff(a0, b0) > 0.1
    # (b) the shipped path: still a real knob, but a collapsed one.
    a = _run(E, kw, ray_subsample=4, remap_sampling='lattice')
    b = _run(E, kw, ray_subsample=4, remap_sampling='full')
    assert not np.array_equal(a, b), (
        'remap_sampling has stopped touching the field entirely -- the knob '
        'is dead, not merely quiet')
    d = _rms_phase_diff(a, b)
    assert 1e-8 < d < 1e-4, d           # measured 2.6853e-06


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
    without being brittle.

    2026-08-01 (niche C6).  Those numbers, and the reasoning above them, are
    kept VERBATIM in arm (a) at ``REMAP_STATIONARY_PHASE_LAUNCH = False`` --
    the library state they were measured in, where they reproduce to every
    printed digit.  They do not survive the shipped launch, and the exact ray
    trace says that is a fix rather than a loss: pre-C6, ``'full'`` was
    converging to a ``rs=1`` reference that is itself 1.2538 rad from an exact
    skew ray trace of this fixture, while the shipped library sits at
    8.6e-03 rad at every ``rs`` and mode (module docstring; oracle in
    ``validation/repro_traced_carrier_121/recon_s12_oracle.py``).

    PIN IS NOW, on the shipped path (arm b): the two modes must agree with
    each other to 1e-04 rad and must sit at the SAME distance from the
    ``rs=1`` reference to within 1 %, because the residual ``rs`` dependence
    is the C6 launch's own and not a sampling effect.  Measured
    ``d_lat``/``d_ful`` = 1.6340e-02 / 5.6227e-02 / 9.1257e-02 rad (equal to
    5 figures) with ``d(lat, ful)`` = 6.06e-07 / 2.69e-06 / 1.09e-05."""
    E, kw, _ = _setup
    # (a) the original claim, in the library state it was calibrated in.
    ref0 = _run(E, kw, ray_subsample=1, remap_sampling='lattice', launch=False)
    lat0 = _run(E, kw, ray_subsample=rs, remap_sampling='lattice', launch=False)
    ful0 = _run(E, kw, ray_subsample=rs, remap_sampling='full', launch=False)
    d_lat0 = _rms_phase_diff(ref0, lat0)
    d_ful0 = _rms_phase_diff(ref0, ful0)
    assert d_ful0 < tol_full, (d_ful0, d_lat0)
    assert d_lat0 > min_lattice, (d_lat0, d_ful0)
    assert d_lat0 > 20.0 * max(d_ful0, 1e-6), (d_lat0, d_ful0)
    # (b) the shipped path: the aliasing channel is gone, so the two modes
    #     must be indistinguishable AND equally distant from the reference.
    ref = _run(E, kw, ray_subsample=1, remap_sampling='lattice')
    lat = _run(E, kw, ray_subsample=rs, remap_sampling='lattice')
    ful = _run(E, kw, ray_subsample=rs, remap_sampling='full')
    d_lat = _rms_phase_diff(ref, lat)
    d_ful = _rms_phase_diff(ref, ful)
    assert _rms_phase_diff(lat, ful) < 1e-4, _rms_phase_diff(lat, ful)
    assert abs(d_lat - d_ful) < 0.01 * max(d_lat, d_ful), (d_lat, d_ful)
    assert d_lat < 0.12 and d_ful < 0.12, (d_lat, d_ful)


# ===========================================================================
# 6: the mechanism -- the disagreement lives beyond r_alias
# ===========================================================================
def test_lattice_error_is_confined_beyond_the_alias_radius(_setup):
    """The 'lattice' error must be an ALIASING signature, i.e. confined to
    ``r > r_alias(h = ray_subsample*dx)`` and absent well inside it -- not a
    broadband offset.  This is what ties the fix to the design-121
    observation (residual 0.05 rad inside 1 w, 0.97 rad at 1.5-2.0 w).

    2026-08-01 (niche C6).  PIN WAS: that signature at the shipped default,
    with ``'full'`` clean on both sides (< 0.02 / < 0.05 rad).  It is kept
    VERBATIM in arm (a) at ``REMAP_STATIONARY_PHASE_LAUNCH = False``, where it
    still holds exactly (lattice inner 5.166e-02 / outer 1.896e+00, ratio
    36.7; full inner 3.920e-05 / outer 8.029e-04).

    PIN IS NOW, on the shipped path (arm b): there is no aliasing signature to
    find, because the C6 launch absorbs this fixture's degree-4 residual into
    ``a_fit`` and leaves the transported phasor ~0 -- so the two modes must
    show the SAME inner and outer deviations (measured lattice 3.6215e-04 /
    1.1284e-01, full 3.6080e-04 / 1.1284e-01).  What is left in the outer
    band is the C6 launch's own ``ray_subsample`` dependence, and pinning that
    the two modes share it is what distinguishes "no aliasing" from "aliasing
    that both modes now suffer".  Adjudicated by an exact skew ray trace
    (module docstring): the shipped field is 8.6e-03 rad from it, the C6-off
    field 1.19-1.25 rad."""
    E, kw, rr = _setup
    rs = 4
    ra = _r_alias(rs * _DX)
    assert _W < ra < 1.6 * _W, ra / _W        # fixture sanity: 1.18 w
    inner = rr < 0.75 * ra
    outer = rr > 1.05 * ra
    # (a) the aliasing signature, in the library state it was measured in.
    ref0 = _run(E, kw, ray_subsample=1, remap_sampling='lattice', launch=False)
    lat0 = _run(E, kw, ray_subsample=rs, remap_sampling='lattice', launch=False)
    ful0 = _run(E, kw, ray_subsample=rs, remap_sampling='full', launch=False)
    d_in0 = _rms_phase_diff(ref0, lat0, inner)
    d_out0 = _rms_phase_diff(ref0, lat0, outer)
    assert d_out0 > 5.0 * max(d_in0, 1e-6), (d_in0, d_out0)
    assert _rms_phase_diff(ref0, ful0, inner) < 0.02
    assert _rms_phase_diff(ref0, ful0, outer) < 0.05
    # (b) the shipped path: no aliasing channel, so the modes cannot differ.
    ref = _run(E, kw, ray_subsample=1, remap_sampling='lattice')
    lat = _run(E, kw, ray_subsample=rs, remap_sampling='lattice')
    ful = _run(E, kw, ray_subsample=rs, remap_sampling='full')
    for m in (inner, outer):
        d_l = _rms_phase_diff(ref, lat, m)
        d_f = _rms_phase_diff(ref, ful, m)
        assert abs(d_l - d_f) < 0.01 * max(d_l, d_f, 1e-12), (d_l, d_f)
    assert _rms_phase_diff(ref, lat, inner) < 0.01
    assert _rms_phase_diff(ref, ful, inner) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
