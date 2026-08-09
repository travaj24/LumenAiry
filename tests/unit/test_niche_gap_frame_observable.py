"""Gap-transport FRAME observable + arm C (spec
``docs/audits/SPEC_EXACT_SPHERE_GAP_TRANSPORT_2026_08_05.md`` Stages 0-1).

WHAT THIS PINS, AND WHY IT IS NOT THE SPEC'S OWN FRAMING
--------------------------------------------------------
The spec asks for an "exact sphere-referenced" gap transport (its M1) to
replace the paraxial Sziklas-Siegman step, and asserts in its Sec 4 that "**No
guard covers B**" (B = the paraxial FRAME approximation, as opposed to A = the
Fresnel kernel).  That assertion is **false against the code**:
``_check_gap_paraxial`` already had TWO arms, and arm B trips on the gap NA
with a calibration measured against an independent band-limited ASM
(``_GAP_NA_TOL``, and ``test_niche_c3_gap_paraxial_guard.py``'s inline
Matsushima oracle).  Niche C3 shipped that guard in direct response to the same
roadmap P7 sentence the spec quotes as unfulfilled.

What IS genuinely missing is narrower: arm B's trip variable is the CARRIER's
geometry (``w/|R|``), which is only a PROXY for the quantity the frame's
validity actually rests on -- the ENVELOPE's own residual angular content.  For
a slowly-varying envelope the two track; for an envelope carrying real
non-spherical content (an aberrated intermediate wavefront, or a carrier
mismatched to the beam) the proxy under-reports.  So these tests pin the
DIRECT observable and the arm that trips on it, and deliberately do NOT pin any
claim about M1, which was not implemented (see the audit's "declined" section).

Self-contained; no ``.zmx``, no design-121 dependency.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import carrier as C

_WL = 1.31e-6
_K = 2.0 * np.pi / _WL


def _gauss(n, dx, w):
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128), X, Y


def _singlet(R1, R2, d, glass, ap, name='s'):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


_RELAY = [
    {'prescription': _singlet(200e-3, -200e-3, 3e-3, 'N-BK7', 8e-3, 'g1'),
     'gap_before': 0.0},
    {'prescription': _singlet(200e-3, -200e-3, 3e-3, 'N-BK7', 8e-3, 'g2'),
     'gap_before': 40e-3},
]

# A LEADING gap, so the leg is genuinely collimated (R = inf) rather than
# carrying the first group's convergence.  Needed because the 40 mm leg in
# ``_RELAY`` sits AFTER a +200 mm singlet, where R ~ -198 mm.
_LEADING_GAP = [
    {'prescription': _singlet(200e-3, -200e-3, 3e-3, 'N-BK7', 8e-3, 'g1'),
     'gap_before': 30e-3},
]

# Grid sized so the beam FITS: half-width N*dx/2 must comfortably exceed w0,
# else the envelope is truncated at the grid edge and the hard edge's spectral
# content is (correctly) read as angular content -- a real effect, but not the
# one these tests are about.  768 * 4 um / 2 = 1.536 mm against w0 = 1.5 mm was
# marginal; 1024 gives 2.048 mm = 1.37x w0.
_N, _DX, _W0 = 1024, 4e-6, 1.5e-3


def _chain(env0, dx, groups=None, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain(
            env0, _RELAY if groups is None else groups, _WL, dx,
            r_in=np.inf, ray_subsample=4, n_workers=1,
            traced_kwargs=dict(parallel_amp=False), final_distance=0.0, **kw)
    frame = [w for w in rec
             if 'SZIKLAS-SIEGMAN FRAME' in str(w.message)]
    return res, frame


# ===========================================================================
# The observable itself
# ===========================================================================
def test_flat_envelope_spread_is_the_diffractive_floor_not_zero():
    """A real-valued Gaussian envelope has ZERO phase gradient but NON-zero
    angular content -- its own diffractive spectrum, ~lambda/(pi w).  The
    shipped (spectral) estimator must report that floor rather than 0, because
    that content is real and does drive the frame term.  This is the concrete
    difference from a phase-difference estimator, which reads 0 here."""
    n, dx, w = 256, 2e-6, 200e-6
    env, _, _ = _gauss(n, dx, w)
    theta, frac = C._gap_envelope_angular_spread(env, dx, _WL)
    floor = _WL / (np.pi * w)                       # Gaussian far-field angle
    assert theta > 0.0, "a finite Gaussian is not angularly a delta"
    # 99.9 % power radius sits a few x above the 1/e amplitude angle
    assert floor < theta < 12.0 * floor, (theta, floor)
    assert 0.0 <= frac < 0.5


def test_spread_grows_with_injected_non_spherical_content():
    """Monotone response to a deliberately injected quartic residual -- the
    envelope content the carrier does NOT describe."""
    n, dx, w = 256, 2e-6, 200e-6
    env0, X, Y = _gauss(n, dx, w)
    r2 = X ** 2 + Y ** 2
    prev = -1.0
    for a4 in (0.0, 1e8, 4e8, 1.6e9):
        env = (np.abs(env0) * np.exp(1j * _K * a4 * r2 ** 2)).astype(
            np.complex128)
        theta, _ = C._gap_envelope_angular_spread(env, dx, _WL)
        assert theta > prev, (a4, theta, prev)
        prev = theta


def test_spread_flags_its_own_undersampling():
    """The estimator cannot see past the grid Nyquist tilt -- that is a
    property of the SAMPLING, not of the estimator.  It must therefore publish
    a Nyquist fraction that rises into the flag band rather than returning a
    confident small number.  (The audit records the failure mode this avoids:
    a wrapped gradient folds over-Nyquist content back to a small reading.)"""
    n, dx, w = 256, 2e-6, 200e-6
    env0, X, _ = _gauss(n, dx, w)
    nyq = _WL / (2.0 * dx)
    # a tilt at ~0.9x Nyquist is representable but marginal -> must flag
    env = (np.abs(env0) * np.exp(1j * _K * 0.9 * nyq * X)).astype(
        np.complex128)
    _, frac = C._gap_envelope_angular_spread(env, dx, _WL)
    assert frac > C._GAP_ENV_NYQUIST_FRAC, frac


def test_spread_route_is_reported():
    """``return_kind`` says which estimator ran, so a consumer can tell a
    spectrum from a difference.  Small grids take the spectral route."""
    n, dx, w = 256, 2e-6, 200e-6
    env, _, _ = _gauss(n, dx, w)
    out = C._gap_envelope_angular_spread(env, dx, _WL, return_kind=True)
    assert len(out) == 3
    assert out[2] is True, "a 256-grid must take the affordable spectral route"


def test_difference_fallback_recovers_a_known_tilt():
    """The large-grid fallback (wrapped nearest-neighbour difference) is still
    exercised, via the size switch, and must recover a known linear tilt.  This
    pins the fallback path independently of the spectral one."""
    n, dx, w = 256, 2e-6, 200e-6
    env0, X, _ = _gauss(n, dx, w)
    L = 5e-3
    env = (np.abs(env0) * np.exp(1j * _K * L * X)).astype(np.complex128)
    old = C._GAP_ENV_SPECTRAL_MAX_N
    try:
        C._GAP_ENV_SPECTRAL_MAX_N = 8      # force the fallback
        theta, _, spectral = C._gap_envelope_angular_spread(
            env, dx, _WL, return_kind=True)
    finally:
        C._GAP_ENV_SPECTRAL_MAX_N = old
    assert spectral is False
    assert theta == pytest.approx(L, rel=0.02), theta


# ===========================================================================
# Publication into stages
# ===========================================================================
def test_every_gap_leg_publishes_the_frame_diagnostic():
    """Read-off-without-catching-a-warning, matching the convention arms A/B
    already follow."""
    n, dx, w0 = _N, _DX, _W0
    env0, _, _ = _gauss(n, dx, w0)
    res, _ = _chain(env0, dx)
    legs = [s for s in res.stages if 'gap_env_theta' in s]
    assert legs, "the 40 mm inter-group leg must publish a frame diagnostic"
    for s in legs:
        for key in ('gap_env_theta', 'gap_env_nyq_frac', 'gap_env_phi_drop',
                    'gap_z_eff', 'gap_env_spectral'):
            assert key in s, key
        assert s['gap_env_theta'] >= 0.0
        assert np.isfinite(s['gap_env_phi_drop'])


def test_healthy_relay_does_not_trip_arm_c():
    """No false alarm on a benign near-collimated relay -- the regime the
    library is validated in.  (Measured margin at the audit date: frame drop
    ~2e-4 rad against the 0.30 rad threshold.)"""
    n, dx, w0 = _N, _DX, _W0
    env0, _, _ = _gauss(n, dx, w0)
    res, frame = _chain(env0, dx)
    assert not frame, [str(w.message)[:200] for w in frame]
    legs = [s for s in res.stages if 'gap_env_theta' in s]
    assert all(s['gap_env_phi_drop'] < 0.30 for s in legs), \
        [s['gap_env_phi_drop'] for s in legs]


# ===========================================================================
# The point of the whole exercise: the arm-B blind spot
# ===========================================================================
def test_arm_c_catches_what_the_carrier_na_proxy_misses():
    """THE load-bearing test.  A carrier-mismatched envelope (strong injected
    quartic, ``r_in=inf`` so the carrier does not describe it) leaves the
    CARRIER NA -- arm B's trip variable -- small, while the ENVELOPE's own
    angular content is large.  Arm B stays silent; arm C must fire.  This is
    what makes the direct observable worth having rather than a duplicate of
    arm B."""
    n, dx, w0 = _N, _DX, _W0
    env0, X, Y = _gauss(n, dx, w0)
    r2 = X ** 2 + Y ** 2
    env = (np.abs(env0) * np.exp(1j * _K * 2e7 * r2 ** 2)).astype(
        np.complex128)
    res, frame = _chain(env, dx)
    legs = [s for s in res.stages if 'gap_env_theta' in s]
    assert legs
    s = legs[0]
    # arm B's proxy is small ...
    assert s['gap_na'] < C._GAP_NA_TOL, s['gap_na']
    # ... while the measured envelope content is large, and arm C fired.
    assert s['gap_env_theta'] > 10.0 * 1e-3, s['gap_env_theta']
    assert s['gap_env_phi_drop'] > 0.30, s['gap_env_phi_drop']
    assert frame, "arm C must fire where arm B's carrier-NA proxy cannot"


# ===========================================================================
# Knob separation + collimated coverage (two defects found in review)
# ===========================================================================
def test_arm_c_has_its_own_knob_and_does_not_silence_arms_a_b():
    """Arm C is uncalibrated by construction (its threshold is carried across
    from ``gap_sag_tol`` by dimensional analogy), so silencing it must NOT also
    silence the two ASM-calibrated carrier-geometry arms.  Pinning the knob
    separation directly: ``on_gap_frame='ignore'`` suppresses arm C while
    ``on_gap_paraxial`` is untouched."""
    n, dx, w0 = _N, _DX, _W0
    env0, X, Y = _gauss(n, dx, w0)
    r2 = X ** 2 + Y ** 2
    env = (np.abs(env0) * np.exp(1j * _K * 2e7 * r2 ** 2)).astype(
        np.complex128)
    _, frame_on = _chain(env, dx)
    assert frame_on, "precondition: arm C fires by default here"
    res_off, frame_off = _chain(env, dx, on_gap_frame='ignore')
    assert not frame_off, "on_gap_frame='ignore' must silence arm C"
    # ... and the diagnostic is still published, so silencing != going blind
    legs = [s for s in res_off.stages if 'gap_env_theta' in s]
    assert legs and legs[0]['gap_env_theta'] > 0.0


def test_collimated_leg_is_covered_by_arm_c():
    """A COLLIMATED inter-group leg (R = inf) took no arm at all before this
    work: the whole guard was gated on ``isfinite(R)``.  That is exactly
    backwards for the frame arm -- with no co-moving reduction ``z_eff = z``,
    its LARGEST value, so the frame term is maximal there, and roadmap P8
    names "a fast final group after a collimated space" as the most common
    relay architecture.  ``r_in=inf`` + the 40 mm leg is that geometry."""
    n, dx, w0 = _N, _DX, _W0
    env0, _, _ = _gauss(n, dx, w0)
    res, _ = _chain(env0, dx, groups=_LEADING_GAP)
    legs = [s for s in res.stages if 'gap_env_theta' in s]
    assert legs, "a collimated leg must still publish the frame diagnostic"
    s = legs[0]
    # collimated => no reduction => z_eff is the full geometric gap
    assert s['gap_z_eff'] == pytest.approx(30e-3, rel=1e-9), s['gap_z_eff']
    # arms A/B self-silence on a collimated leg; the diagnostic still runs
    assert s['gap_na'] == 0.0
    assert s['gap_phi_drop'] == 0.0


def test_gap_env_phi_tol_zero_disables_the_trip_but_keeps_the_number():
    """``0`` is the documented escape hatch: report, do not trip."""
    n, dx, w0 = _N, _DX, _W0
    env0, X, Y = _gauss(n, dx, w0)
    r2 = X ** 2 + Y ** 2
    env = (np.abs(env0) * np.exp(1j * _K * 2e7 * r2 ** 2)).astype(
        np.complex128)
    res, frame = _chain(env, dx, gap_env_phi_tol=0.0)
    assert not frame
    legs = [s for s in res.stages if 'gap_env_theta' in s]
    assert legs and legs[0]['gap_env_phi_drop'] > 0.0


def test_invalid_knobs_are_refused():
    n, dx, w0 = 128, _DX, 100e-6
    env0, _, _ = _gauss(n, dx, w0)
    with pytest.raises(ValueError):
        la.propagate_traced_carrier_chain(
            env0, _RELAY, _WL, dx, r_in=np.inf, on_gap_frame='shout')
    with pytest.raises(ValueError):
        la.propagate_traced_carrier_chain(
            env0, _RELAY, _WL, dx, r_in=np.inf, gap_env_phi_tol=-1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
