"""Niche C5 (2026-07-30) -- the chain's TILTED carrier reference.

``propagate_traced_carrier_chain`` used to store a tilted congruence's
envelope against an on-axis SPHERE PLUS A LINEAR RAMP.  That is not a solution
of the eikonal equation, so the "envelope" carried a coma/astigmatism term the
Sziklas-Siegman step cannot transport (it dilates ``du -> m du`` while the term
wants an ``R``-weighted rescaling).  ``_tilt_exactness_phase`` upgrades the
reference to the EXACT displaced-point-source eikonal.

Everything here is self-contained: closed-form congruences and the library's
own carrier leg.  No ``.zmx``, no ray trace, no diffraction integral, no phase
unwrap and no FFT derivative -- the three instrument classes that have
produced false findings on this problem.  Every wave figure states its own
sampling adequacy (the wrapped nearest-neighbour step of the compared
difference, which must sit far below ``pi``).
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT
from lumenairy.elements._lens_traced import TiltedCarrier
from lumenairy.propagators.carrier import (
    _exact_sphere_eikonal,
    _radial_carrier_phase,
    _sphere_parab_conversion,
    _tilt_exactness_phase,
    _tilt_ramp,
    propagate_carrier_referenced,
)

LAM = 1.31e-6
K0 = 2.0 * np.pi / LAM


def _w_true(u, v, R, L, M):
    """Exact eikonal of a point source at signed AXIAL distance ``R`` whose
    chief ray leaves ``(0, 0)`` along ``(L, M, N)``."""
    N = np.sqrt(1.0 - L * L - M * M)
    sgn = 1.0 if R > 0 else -1.0
    uu = u + R * L / N
    vv = v + R * M / N
    return sgn * (np.sqrt(uu * uu + vv * vv + R * R) - abs(R) / N)


def _w_sphere_ramp(u, v, R, L, M):
    sgn = 1.0 if R > 0 else -1.0
    return sgn * (np.sqrt(u * u + v * v + R * R) - abs(R)) + L * u + M * v


def _wrms(ph, wgt):
    """Weighted rms of ``ph`` with piston removed, in waves."""
    w = wgt / max(float(wgt.sum()), 1e-300)
    mean = float(np.sum(w * ph))
    return float(np.sqrt(max(float(np.sum(w * (ph - mean) ** 2)), 0.0))) \
        / (2.0 * np.pi)


def _nn_step(ph, amp):
    """99.9th-percentile amplitude-weighted wrapped nearest-neighbour step of
    ``ph`` (rad) -- the licence to treat a wrapped phase difference as a
    smooth field.  A percentile and not a MAX on purpose: a single skirt pixel
    sets the max, and this study has already had a max-reported nn-step read
    ``pi`` on a field whose core was clean to 0.02 rad."""
    a = amp / max(float(amp.max()), 1e-300)
    st = np.concatenate([
        (np.abs(np.diff(ph, axis=0)) * np.minimum(a[1:], a[:-1])).ravel(),
        (np.abs(np.diff(ph, axis=1)) * np.minimum(a[:, 1:], a[:, :-1])
         ).ravel()])
    return float(np.percentile(st, 99.9))


# ---------------------------------------------------------------------------
# 1.  The algebra: what the factory returns, and that it is a pure no-op
#     wherever the fix must not reach.
# ---------------------------------------------------------------------------

def test_the_factor_is_exactly_the_gap_between_the_two_eikonals():
    """``sphere x ramp x factor`` must equal ``exp(i k W_true)`` pointwise,
    everywhere the band-limit taper is not engaged."""
    n, dx, R, L, M = 128, 2.0e-5, -21.139e-3, 0.0490735, 0.0245367
    cx, cy = 1.1e-3, -0.7e-3
    t = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(t, t)
    S = _exact_sphere_eikonal((n, n), dx, dx, LAM, R, centre=(cx, cy))
    rp = _tilt_ramp((n, n), dx, LAM, L, M, cx, cy, +1)
    xf = _tilt_exactness_phase((n, n), dx, dx, LAM, R, L, M, +1,
                               centre=(cx, cy))
    assert xf is not None
    ref = np.exp(1j * K0 * _w_true(X - cx, Y - cy, R, L, M))
    # the taper only touches r > 0.75 r_safe; check where it does not
    s = L * L + M * M
    a = 1.5 * np.sqrt(s) / (R * R)
    b = s / abs(R)
    r_safe = (np.sqrt(b * b + 4.0 * a * LAM / (2.0 * dx)) - b) / (2.0 * a)
    inside = np.hypot(X - cx, Y - cy) <= 0.75 * r_safe
    assert inside.mean() > 0.5
    err = np.abs(np.exp(1j * K0 * S) * rp * xf - ref)
    assert err[inside].max() < 1e-9, err[inside].max()


def test_the_round_trip_is_exact_whatever_the_taper_does():
    """``+1`` and ``-1`` build the SAME taper, so the chain's entrance
    multiply and exit divide cancel even in the tapered skirt."""
    n, dx, R, L, M = 96, 1.5e-4, -21.139e-3, 0.049, 0.0245
    up = _tilt_exactness_phase((n, n), dx, dx, LAM, R, L, M, +1)
    dn = _tilt_exactness_phase((n, n), dx, dx, LAM, R, L, M, -1)
    assert np.abs(up * dn - 1.0).max() < 1e-14


@pytest.mark.parametrize('args', [
    (0.0, 0.0),          # untilted -- the whole point of the on-axis pin
    (0.0, -0.0),
])
def test_the_factor_is_none_for_an_untilted_congruence(args):
    assert _tilt_exactness_phase((32, 32), 1e-5, 1e-5, LAM, -20e-3,
                                 args[0], args[1], +1) is None


def test_the_factor_is_none_for_a_collimated_or_degenerate_carrier():
    for R in (np.inf, -np.inf, np.nan, 0.0):
        assert _tilt_exactness_phase((32, 32), 1e-5, 1e-5, LAM, R,
                                     0.05, 0.02, +1) is None


def test_a_non_propagating_tilt_is_refused_by_name():
    with pytest.raises(ValueError, match='DIRECTION COSINES'):
        _tilt_exactness_phase((16, 16), 1e-5, 1e-5, LAM, -20e-3, 0.9, 0.9, +1)


def test_the_fail_before_switch_disables_it_completely():
    old = LT.TILTED_CARRIER_EXACT_EIKONAL
    try:
        LT.TILTED_CARRIER_EXACT_EIKONAL = False
        assert _tilt_exactness_phase((32, 32), 1e-5, 1e-5, LAM, -20e-3,
                                     0.05, 0.02, +1) is None
    finally:
        LT.TILTED_CARRIER_EXACT_EIKONAL = old
    assert _tilt_exactness_phase((32, 32), 1e-5, 1e-5, LAM, -20e-3,
                                 0.05, 0.02, +1) is not None


def test_the_gap_has_the_predicted_coma_plus_astigmatism_size():
    """Fail-before, sized: the term the pre-C5 reference dropped is
    ``-(n.du) rho^2/(2R^2) - (n.du)^2/(2R)``.  On design 121's last coarse leg
    (R = -24.46 mm, |n| = 0.0549, w = 3.63 mm, lambda 1.31 um) that is -0.73
    waves one beam radius ALONG the tilt, +2.53 waves one radius AGAINST it
    (the coma is odd), and 15.8 waves at two radii.  If this ever reads ~0 the
    whole niche is moot."""
    R, L, M, w = -24.4625e-3, 0.0490735, 0.0245367, 3.6253e-3
    n_t = np.hypot(L, M)
    u, v = w * L / n_t, w * M / n_t          # one beam radius ALONG the tilt
    gap = (_w_true(u, v, R, L, M) - _w_sphere_ramp(u, v, R, L, M)) / LAM
    gap_m = (_w_true(-u, -v, R, L, M)
             - _w_sphere_ramp(-u, -v, R, L, M)) / LAM
    assert gap == pytest.approx(-0.730, abs=0.02), gap
    assert gap_m == pytest.approx(+2.531, abs=0.02), gap_m
    # the odd part is the COMA and is LINEAR in the field angle; the even part
    # is the astigmatism and is quadratic
    odd = 0.5 * (gap - gap_m)
    even = 0.5 * (gap + gap_m)
    g10 = (_w_true(u, v, R, L / 10, M / 10)
           - _w_sphere_ramp(u, v, R, L / 10, M / 10)) / LAM
    g10m = (_w_true(-u, -v, R, L / 10, M / 10)
            - _w_sphere_ramp(-u, -v, R, L / 10, M / 10)) / LAM
    assert 0.5 * (g10 - g10m) / odd == pytest.approx(0.1, rel=0.1)
    assert 0.5 * (g10 + g10m) / even == pytest.approx(0.01, rel=0.1)


# ---------------------------------------------------------------------------
# 2.  THE ACCEPTANCE.  A closed-form congruence through the library's own
#     carrier leg, scored against the closed-form answer.
# ---------------------------------------------------------------------------

def _leg_error(L, M, exact, R0=-24.4625e-3, z=3.3233e-3, w=3.6253e-3,
               n=1024, dx=3.84324e-5):
    """Push an EXACT point-source congruence across one free-space carrier
    leg the way the chain does, and return (rms error in waves, nn-step).

    Free-space transport of a point-source congruence is closed form in the
    geometric limit -- dilation by ``m = (R+z)/R`` about the source
    projection, chief ray advancing by ``z n / cos(theta)``, amplitude
    carrying ``1/m``.  Diffraction is 4 orders below the effect measured here
    (the Rayleigh range of a 3.6 mm 1.31 um Gaussian is 31 m against a 3.3 mm
    gap), and the untilted arm of this same function measures that floor.
    """
    t = (np.arange(n) - n / 2) * dx
    U, V = t[None, :], t[:, None]
    A = np.exp(-(U * U + V * V) / (w * w))
    ref = _w_true if exact else _w_sphere_ramp
    # the chain's stored envelope: the field with ITS reference divided out
    env = (A * np.exp(1j * K0 * (_w_true(U, V, R0, L, M)
                                 - ref(U, V, R0, L, M)))).astype(complex)
    cr = propagate_carrier_referenced(env, R0, z, LAM, dx)
    R1, dx1 = float(cr.R), float(cr.dx)
    N = np.sqrt(1.0 - L * L - M * M)
    t1 = (np.arange(n) - n / 2) * dx1
    U1, V1 = t1[None, :], t1[:, None]
    # rebuild the full field against the same reference, plus the chain's own
    # obliquity piston
    E = np.asarray(cr.env) * np.exp(
        1j * K0 * (ref(U1, V1, R1, L, M) + z * (1.0 / N - 1.0)))
    m = R1 / R0
    A1 = np.exp(-((U1 / m) ** 2 + (V1 / m) ** 2) / (w * w)) / m
    E_true = A1 * np.exp(1j * K0 * _w_true(U1, V1, R1, L, M))
    d = np.angle(E * np.conj(E_true))
    return _wrms(d, np.abs(E_true) ** 2), _nn_step(d, np.abs(E_true))


def test_the_exact_reference_makes_the_tilted_leg_exact():
    """THE acceptance.  At design 121's leg-5 geometry and its 54.9 mrad
    carried tilt, the pre-C5 reference costs 0.13 waves rms of spurious
    wavefront and the C5 one costs the same 1e-5 as an UNTILTED leg."""
    L, M = 0.0490735, 0.0245367
    rms_0, step_0 = _leg_error(0.0, 0.0, exact=True)       # untilted control
    rms_x, step_x = _leg_error(L, M, exact=True)
    rms_s, step_s = _leg_error(L, M, exact=False)          # fail-before
    for s in (step_0, step_x, step_s):
        assert s < 0.6, s          # sampling adequacy; pi = 3.1416
    assert rms_0 < 1e-4, rms_0
    assert rms_x < 3e-4, rms_x                       # == the untilted floor
    assert rms_s > 0.10, rms_s                       # fail-before
    assert rms_s / max(rms_x, 1e-12) > 100.0


def test_it_stays_exact_at_a_tilt_three_times_the_design_and_scales_right():
    """The pre-C5 error is ~linear in the field angle (the dropped term's
    leading piece is coma), and the C5 reference is flat in it."""
    prev = None
    for f in (0.25, 0.5, 1.0, 3.0):
        L, M = 0.0490735 * f, 0.0245367 * f
        rms_x, step_x = _leg_error(L, M, exact=True)
        rms_s, _ = _leg_error(L, M, exact=False)
        assert step_x < 0.6
        assert rms_x < 5e-4, (f, rms_x)
        if prev is not None and f <= 1.0:
            assert rms_s / prev == pytest.approx(2.0, rel=0.25), (f, rms_s)
        if f <= 1.0:
            prev = rms_s


def test_the_carrier_leg_is_untouched_for_an_untilted_congruence():
    """C5 changes the REFERENCE, never the transport: the leg distance, the
    magnification and the piston are all as shipped."""
    n, dx, R0, z = 128, 5e-5, -24.4625e-3, 3.3233e-3
    env = np.ones((n, n), dtype=complex)
    cr = propagate_carrier_referenced(env, R0, z, LAM, dx)
    assert float(cr.R) == R0 + z
    assert float(cr.dx) == (R0 + z) / R0 * dx


# ---------------------------------------------------------------------------
# 3.  BYTE IDENTITY of every path C5 must not reach (the D7 pattern).
# ---------------------------------------------------------------------------

def _singlet(f_mm=30.0, t_mm=3.0):
    return {'name': 'C5 singlet', 'aperture_diameter': 16e-3,
            'thicknesses': [t_mm * 1e-3],
            'surfaces': [
                {'radius': f_mm * 1e-3, 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -f_mm * 1e-3, 'conic': 0.0,
                 'glass_before': 'N-BK7', 'glass_after': 'air'}]}


def _gauss(n, dx, w):
    t = (np.arange(n) - n / 2) * dx
    return np.exp(-(t[None, :] ** 2 + t[:, None] ** 2) / (w * w)
                  ).astype(np.complex128)


_G1 = [{'prescription': _singlet(), 'gap_before': 12e-3}]
_G2 = [{'prescription': _singlet(40.0), 'gap_before': 12e-3},
       {'prescription': _singlet(25.0), 'gap_before': 9e-3}]
_GDOE = [{'doe': {'period': 1.2e-4, 'order': 0, 'angle_deg': 0.0,
                  'gap_before': 4e-3, 'gap_after': 8e-3}},
         {'prescription': _singlet(), 'gap_before': 0.0}]
_FR = {'dx_out': 0.4e-6, 'N_out': 48}

_UNTILTED_CASES = [
    # (groups, r_in, final_distance, N, dx, w, ray_subsample, extra kwargs)
    (_G1, np.inf, 25e-3, 128, 4.0e-5, 1.2e-3, 4, {}),
    (_G1, np.inf, 30e-3, 192, 3.0e-5, 1.0e-3, 4, {}),
    (_G1, 0.20, 25e-3, 128, 4.0e-5, 1.2e-3, 4, {}),
    (_G1, -0.35, 20e-3, 128, 4.0e-5, 1.0e-3, 8, {}),
    (_G1, TiltedCarrier(np.inf, 0.0, 0.0), 25e-3, 128, 4.0e-5, 1.2e-3, 4, {}),
    (_G1, np.inf, 25e-3, 128, 4.0e-5, 1.2e-3, 2, {}),
    (_G2, np.inf, 22e-3, 160, 3.5e-5, 1.0e-3, 4, {}),
    (_G2, 0.15, 22e-3, 160, 3.5e-5, 1.0e-3, 4, {}),
    # the DOE branch (order 0 -> no tilt), which has its own deferred-gap
    # bookkeeping
    (_GDOE, np.inf, 25e-3, 128, 4.0e-5, 1.2e-3, 4, {}),
    # the two readout paths: paraxial focus readout, and the EXACT high-NA
    # final leg (which C5 also touches, in _fine_trace_group_exit and in
    # carrier_referenced_exact_focus_readout)
    (_G1, np.inf, 25e-3, 128, 4.0e-5, 1.2e-3, 4,
     {'focus_readout': _FR, 'final_leg': 'paraxial'}),
    (_G1, np.inf, 25e-3, 128, 4.0e-5, 1.2e-3, 4,
     {'focus_readout': dict(_FR, n_fine_cap=2048), 'final_leg': 'exact'}),
    (_G1, 0.20, 25e-3, 128, 4.0e-5, 1.2e-3, 4,
     {'focus_readout': dict(_FR, n_fine_cap=2048), 'final_leg': 'exact'}),
]


@pytest.mark.parametrize('case', _UNTILTED_CASES,
                         ids=[str(i) for i in range(len(_UNTILTED_CASES))])
def test_every_untilted_configuration_is_byte_identical(case):
    """NON-NEGOTIABLE.  ``np.array_equal``, not a tolerance: the shipped
    single-beam design-121 acceptance (3.450 um / EE3 88.8 / EE6 99.6 / EE12
    99.8) must not re-baseline, and neither may any other on-axis result.

    Twelve configurations x two toggle states, covering a collimated /
    diverging / converging carrier, an explicit zero-tilt ``TiltedCarrier``,
    two ray_subsamples, two grids, one- and two-group chains, an order-0 DOE
    entry (the deferred-gap bookkeeping), the paraxial focus readout and the
    EXACT high-NA final leg -- i.e. every code path C5 inserted into."""
    groups, r_in, fd, n, dx, w, rs, extra = case
    E = _gauss(n, dx, w)
    old = LT.TILTED_CARRIER_EXACT_EIKONAL
    kw = dict(final_leg='paraxial', on_decentred_fit='ignore',
              on_gap_paraxial='ignore', on_na_proximity='ignore',
              on_tilt_exact_grid='ignore', on_rs_fine_clamp='ignore')
    kw.update(extra)
    try:
        outs = []
        for flag in (True, False):
            LT.TILTED_CARRIER_EXACT_EIKONAL = flag
            res = la.propagate_traced_carrier_chain(
                E, groups, LAM, dx, r_in=r_in, ray_subsample=rs, n_workers=1,
                final_distance=fd, **kw)
            outs.append(np.asarray(res.field))
    finally:
        LT.TILTED_CARRIER_EXACT_EIKONAL = old
    assert outs[0].dtype == outs[1].dtype
    assert np.array_equal(outs[0], outs[1]), (
        float(np.abs(outs[0] - outs[1]).max()))


def test_a_pure_input_decentre_is_NOT_in_the_untilted_set_and_why():
    """Guard against a plausible-looking addition to the list above.

    A congruence entering with zero tilt but a transverse DECENTRE does not
    stay untilted: a chief ray at height ``x0`` leaves a lens with
    ``L_out = C x0``, so from the first group onward it IS tilted and C5
    legitimately applies to it.  Measured here, not assumed."""
    n, dx, w = 128, 4.0e-5, 1.0e-3
    res = la.propagate_traced_carrier_chain(
        _gauss(n, dx, w), _G1, LAM, dx,
        r_in=TiltedCarrier(np.inf, 0.0, 0.0, 0.6e-3, -0.4e-3),
        ray_subsample=4, n_workers=1, final_distance=25e-3,
        final_leg='paraxial', on_decentred_fit='ignore',
        on_gap_paraxial='ignore')
    st = [s for s in res.stages if not s.get('target')][-1]
    assert abs(st['L_out']) > 1e-3, st['L_out']
    assert abs(st['M_out']) > 1e-3, st['M_out']


def test_a_tilted_run_is_byte_identical_with_the_switch_off():
    """The fail-before switch is exact, not approximate: with it off a TILTED
    chain reproduces the pre-C5 field bit for bit, so any archived tilted
    result can be regenerated."""
    n, dx, w = 160, 4.0e-5, 1.0e-3
    groups = [{'prescription': _singlet(), 'gap_before': 12e-3}]
    E = _gauss(n, dx, w)
    old = LT.TILTED_CARRIER_EXACT_EIKONAL
    try:
        LT.TILTED_CARRIER_EXACT_EIKONAL = False
        a = np.asarray(la.propagate_traced_carrier_chain(
            E, groups, LAM, dx, r_in=TiltedCarrier(0.20, 0.03, 0.015),
            ray_subsample=4, n_workers=1, final_distance=25e-3,
            final_leg='paraxial', on_decentred_fit='ignore',
            on_gap_paraxial='ignore').field)
        b = np.asarray(la.propagate_traced_carrier_chain(
            E, groups, LAM, dx, r_in=TiltedCarrier(0.20, 0.03, 0.015),
            ray_subsample=4, n_workers=1, final_distance=25e-3,
            final_leg='paraxial', on_decentred_fit='ignore',
            on_gap_paraxial='ignore').field)
        LT.TILTED_CARRIER_EXACT_EIKONAL = True
        c = np.asarray(la.propagate_traced_carrier_chain(
            E, groups, LAM, dx, r_in=TiltedCarrier(0.20, 0.03, 0.015),
            ray_subsample=4, n_workers=1, final_distance=25e-3,
            final_leg='paraxial', on_decentred_fit='ignore',
            on_gap_paraxial='ignore').field)
    finally:
        LT.TILTED_CARRIER_EXACT_EIKONAL = old
    assert np.array_equal(a, b)              # determinism control
    assert not np.array_equal(a, c)          # ... and C5 really did something


def test_the_element_and_the_chain_read_the_SAME_flag():
    """A MIXED pair is worse than either.  ``_tilt_exactness_phase`` resolves
    the flag from ``_lens_traced`` at CALL TIME, so the chain's reconstruct
    and the element's ``TiltedCarrier`` can never disagree about which
    congruence the field is written against.

    Fail-before, measured on the D1 46 mrad two-singlet relay: with the chain
    exact and the element left on sphere-plus-ramp, the element de-chirps with
    a reference the field is no longer written against, the whole coma term
    lands in the residual that ``preserve_input_phase='remap'`` transports
    along the CARRIER rays, and the tilted spot's peak falls to 0.989 of the
    on-axis one (matched pair: above 0.99) while the image centroid moves
    0.116 um off the exact ray trace (matched pair: 0.014 um)."""
    from lumenairy.propagators.carrier import _tilt_exactness_phase as _xf
    old = LT.TILTED_CARRIER_EXACT_EIKONAL
    try:
        LT.TILTED_CARRIER_EXACT_EIKONAL = False
        assert _xf((32, 32), 1e-5, 1e-5, LAM, -20e-3, 0.05, 0.02, +1) is None
        W_off, _, _ = LT._tilted_carrier_parts(
            TiltedCarrier(-20e-3, 0.05, 0.02), np.array([[2e-3]]),
            np.array([[1e-3]]))
        LT.TILTED_CARRIER_EXACT_EIKONAL = True
        assert _xf((32, 32), 1e-5, 1e-5, LAM, -20e-3, 0.05, 0.02,
                   +1) is not None
        W_on, _, _ = LT._tilted_carrier_parts(
            TiltedCarrier(-20e-3, 0.05, 0.02), np.array([[2e-3]]),
            np.array([[1e-3]]))
    finally:
        LT.TILTED_CARRIER_EXACT_EIKONAL = old
    assert float(W_off[0, 0]) == pytest.approx(
        _w_sphere_ramp(2e-3, 1e-3, -20e-3, 0.05, 0.02), rel=1e-14)
    assert float(W_on[0, 0]) == pytest.approx(
        _w_true(2e-3, 1e-3, -20e-3, 0.05, 0.02), rel=1e-14)


def test_the_element_eikonal_is_the_exact_congruence():
    """``TiltedCarrier`` now evaluates the exact displaced-point-source
    eikonal, with ``W(x0,y0) == 0`` and ``grad W(x0,y0) == (L, M)`` exactly,
    and reduces to the plain decentred sphere when the tilt is zero."""
    R, L, M, x0, y0 = -21.139e-3, 0.049, 0.0245, 1.1e-3, -0.7e-3
    W, gx, gy = LT._tilted_carrier_parts(
        TiltedCarrier(R, L, M, x0, y0), np.array([[x0]]), np.array([[y0]]))
    assert float(W[0, 0]) == pytest.approx(0.0, abs=1e-18)
    assert float(gx[0, 0]) == pytest.approx(L, rel=1e-14)
    assert float(gy[0, 0]) == pytest.approx(M, rel=1e-14)
    xq = np.array([[2.0e-3]])
    yq = np.array([[-1.0e-3]])
    W2, _, _ = LT._tilted_carrier_parts(TiltedCarrier(R, L, M), xq, yq)
    assert float(W2[0, 0]) == pytest.approx(
        _w_true(2.0e-3, -1.0e-3, R, L, M), rel=1e-14)
    # untilted -> the historical decentred sphere, term for term
    W0, g0x, g0y = LT._tilted_carrier_parts(
        TiltedCarrier(R, 0.0, 0.0, x0, y0), xq, yq)
    rho = np.sqrt((2.0e-3 - x0) ** 2 + (-1.0e-3 - y0) ** 2 + R * R)
    assert float(W0[0, 0]) == -(rho - abs(R))
    assert float(g0x[0, 0]) == -(2.0e-3 - x0) / rho
    assert float(g0y[0, 0]) == -(-1.0e-3 - y0) / rho


def test_a_non_propagating_tilt_is_refused_by_the_element_too():
    with pytest.raises(ValueError, match='DIRECTION COSINES'):
        LT._tilted_carrier_parts(TiltedCarrier(-20e-3, 0.9, 0.9),
                                 np.array([[0.0]]), np.array([[0.0]]))


def test_the_sphere_parabola_conversion_is_untouched():
    """C5 adds a SEPARATE factor; it must not have perturbed the historical
    parabola<->sphere conversion (which the on-axis chain uses on every leg)."""
    n, dx, R = 64, 3e-5, -30e-3
    cf = _sphere_parab_conversion((n, n), dx, LAM, R, +1)
    t = (np.arange(n) - n / 2) * dx
    r2 = t[None, :] ** 2 + t[:, None] ** 2
    diff = _exact_sphere_eikonal((n, n), dx, dx, LAM, R) - r2 / (2.0 * R)
    r_safe = (abs(R) ** 3 * LAM / dx) ** (1.0 / 3.0)
    tp = np.clip((np.sqrt(r2) - 0.75 * r_safe) / (0.25 * r_safe), 0.0, 1.0)
    ref = np.exp(1j * K0 * diff * np.cos(0.5 * np.pi * tp) ** 2)
    assert np.array_equal(cf, ref)
    # and the parabola screen the chain reconstructs against
    ph = _radial_carrier_phase((n, n), dx, dx, LAM, R, +1)
    assert np.array_equal(ph, np.exp(1j * K0 * r2 / (2.0 * R)))
