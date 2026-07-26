"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 Territory R -- grazing-ray guards
(R-3, R-4), degenerate-bundle handling (R-7) and the two OPD/LG merit
sign defects (R-5, R-6).

Each test is written so it FAILS on the pre-fix tree (verified against a
``git worktree`` of 5c9f7c3) and states the measured pre-fix number in its
assertion message.

R-3  ``jax_trace._intersect_jax``, flat-WITH-aspherics branch: missing the
     grazing-miss guard (``miss |= |N| <= eps``) that both the pure-flat
     branch above it and the ``_intersect_jax_param`` twin already carry.
     Measured on a ``{4: 300.0}`` Schmidt-like plate under the DEFAULT
     float32: the grazing ray survived with ``N`` flipped to -1 and a
     NEGATIVE OPL of -8.053754e-03 m, while NumPy killed it
     (RAY_MISSED_SURFACE) and ``_intersect_jax_param`` killed it too.

R-4  ``intersection.py`` NumPy flat fast path: a ray parallel to the plane
     (|N| <= 1e-30) got ``t = 0``, stayed alive and stayed RAY_OK -- an
     immortal phantom that walked a 4-flat stack with ``opd = 0.0`` and was
     counted "3/3 alive" in ``trace_summary`` (centroid 5.0252 mm, RMS spot
     7.1067 mm, both contaminated).  The guard is scoped to the flat
     INTERSECTION so the P3-58 DOE-order case (a diffraction order landing
     exactly on the propagation cone, L^2+M^2 == 1 -> N == 0, kept alive by
     the strict ``sumsq > 1.0`` evanescence test) is bit-unchanged.

R-5  ``LGAberrationMerit`` summed ``|L|^2`` over ALL targets including
     ``(0, 0)``, which is the piston/STREHL AMPLITUDE channel -- so
     ``design_optimize`` (a MINIMISER) drove the Strehl toward 0.  The JAX
     twin ``make_lg_aberration_merit_jax`` already carried the documented
     OPT-1 fix ``1 - |res|^2``; the NumPy merit now matches it.

R-6  All three OPD merits zero-filled non-finite OPD before
     ``zernike_decompose``, which masks non-finite samples itself.  An
     in-pupil vignetted annulus (rho > 0.8 NaN) therefore read as genuine
     0-waves OPD: injected defocus 0.300 -> 0.1259 waves and primary
     spherical +0.1000 -> -0.0174 waves (a SIGN FLIP -- wrong magnitude AND
     wrong descent direction).

R-7  ``ray_transfer_jacobian_analytic`` raised a bare ``ZeroDivisionError``
     on a degenerate bundle (a slope large/non-finite enough that
     ``N = 1/sqrt(1+u^2)`` underflows to 0): the numba kernel compiles with
     numba's default ``error_model='python'``, unlike the ``_AdrtDual``
     sibling it is bit-identical to (which runs under
     ``np.errstate(divide='ignore')``) and unlike the FD
     ``ray_transfer_jacobian``, both of which return the documented masked
     result.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

WL = 1.31e-6


# =========================================================================
# helpers
# =========================================================================

def _bundle(x, y, L, M, N):
    from lumenairy.raytrace import RayBundle
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.zeros_like(x)
    return RayBundle(x=x.copy(), y=y.copy(), z=z.copy(),
                     L=np.asarray(L, float), M=np.asarray(M, float),
                     N=np.asarray(N, float), wavelength=WL,
                     alive=np.ones_like(x, bool), opd=z.copy())


def _surfaces_from(rx):
    from lumenairy.raytrace import Surface
    ths = rx['thicknesses']
    out = []
    for i, s in enumerate(rx['surfaces']):
        out.append(Surface(
            radius=s.get('radius', np.inf), conic=s.get('conic', 0.0),
            aspheric_coeffs=s.get('aspheric_coeffs'),
            glass_before=s.get('glass_before', 'air'),
            glass_after=s.get('glass_after', 'air'),
            thickness=ths[i] if i < len(ths) else 0.0,
            semi_diameter=s.get('semi_diameter', np.inf)))
    return out


def _jax_f32_or_skip():
    """Import jax, skipping when ``jax_enable_x64`` is on process-wide.

    R-3 is a DEFAULT-precision (float32) finding -- the audit records it as
    "masked under x64" -- so follow the repo's existing pattern for
    dtype-sensitive JAX pins (see ``test_audit_jax_c64_propagator_precision``
    ``_jnp_or_skip``) and require the JAX default here.
    """
    jax = pytest.importorskip('jax')
    if jax.config.jax_enable_x64:
        pytest.skip(
            'jax_enable_x64 is on process-wide; R-3 (the negative-OPL '
            'grazing survivor) reproduces at the JAX-default float32.')
    return jax


# =========================================================================
# R-3 -- jax_trace flat-with-aspherics grazing-miss guard
# =========================================================================

_PLATE_RX = {
    'surfaces': [
        {'radius': np.inf, 'aspheric_coeffs': {4: 300.0},
         'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': np.inf, 'glass_before': 'N-BK7', 'glass_after': 'air'}],
    'thicknesses': [4e-3, 0.05],
}


def test_r3_jax_flat_with_aspherics_never_returns_negative_opl():
    """R-3, DTYPE-INDEPENDENT: launch the grazing ray AT THE VERTEX so its
    Newton residual ``z - sag(0, 0)`` is exactly 0.  The P3-59 residual
    check then cannot mask the missing grazing guard, and the ray accrues a
    NEGATIVE OPL at the intersect step in BOTH float32 and float64.

    Measured pre-fix: opd = -8.053754e-03 m under f32 AND under x64
    (x64 killed the ray one stage later, but the negative OPL was already
    banked).  Post-fix the guard zeroes ``t`` at intersect -> opd = 0.
    """
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    z2 = np.zeros(2)
    st = make_jax_ray_state(z2, z2, z2, np.array([1.0, 0.0]), z2,
                            np.array([0.0, 1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = trace_jax(st, _PLATE_RX, WL)
    opd = np.asarray(s.opd, float)
    assert opd[0] >= 0.0, (
        f'R-3: grazing-ray OPL = {opd[0]:+.6e} m must never be negative '
        f'(pre-fix: -8.053754e-03 m -- with no grazing-miss guard the '
        f'flat+aspherics Newton converged on a BACKWARD intersection and '
        f'the ray accumulated negative optical path).')
    assert bool(np.asarray(s.alive, bool)[1]), (
        'the axial companion ray must still survive')
    assert opd[1] > 0.0


def test_r3_jax_flat_with_aspherics_kills_grazing_ray():
    """R-3: the flat+aspherics branch must apply the same grazing-miss guard
    as its pure-flat sibling -- the ray must not SURVIVE either.

    This is the audit's exact construction (grazing ray offset to
    y = 1 mm) and is float32-only: under x64 the P3-59 Newton-residual
    check happens to kill the same ray for an unrelated reason, so the
    alive flag stops discriminating (the OPL pin above covers x64).
    """
    _jax_f32_or_skip()
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax

    # ray 0 grazes the base plane exactly (L = 1, N = 0); ray 1 is axial.
    b = _bundle([0.0, 0.0], [1e-3, 1e-3], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0])
    st = make_jax_ray_state(b.x, b.y, b.z, b.L, b.M, b.N,
                            opd=b.opd, alive=b.alive)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = trace_jax(st, _PLATE_RX, WL)
    alive = np.asarray(s.alive, bool)
    opd = np.asarray(s.opd, float)

    assert not bool(alive[0]), (
        'R-3: the grazing (N=0) ray SURVIVED the flat+aspherics branch of '
        '_intersect_jax.  Pre-fix it did exactly this, refracting backwards '
        'to N = -1; both the NumPy path and _intersect_jax_param kill it.')
    assert bool(alive[1]), 'the axial companion ray must still survive'
    assert opd[0] >= 0.0, (
        f'R-3: grazing-ray OPL = {opd[0]:+.6e} m must never be negative '
        f'(pre-fix: -8.053754e-03 m -- the ray refracted backwards and '
        f'accumulated NEGATIVE optical path).')


def test_r3_grazing_alive_parity_numpy_jax_static_jax_param():
    """R-3: restore the numpy <-> jax-static <-> jax-param alive triangle on
    the grazing probe, for the pure-flat AND flat+aspherics surfaces."""
    _jax_f32_or_skip()
    from lumenairy.raytrace import trace
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax, trace_jax_with_params
    flat_rx = {
        'surfaces': [
            {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air'},
            {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [1e-3, 0.0]}

    for name, rx in (('pure flat', flat_rx), ('flat+asph', _PLATE_RX)):
        b = _bundle([0.0, 0.0], [1e-3, 1e-3], [1.0, 0.0], [0.0, 0.0],
                    [0.0, 1.0])
        st = make_jax_ray_state(b.x, b.y, b.z, b.L, b.M, b.N,
                                opd=b.opd, alive=b.alive)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a_np = np.asarray(
                trace(b.copy(), _surfaces_from(rx), WL).ray_history[-1].alive,
                bool)
            a_js = np.asarray(trace_jax(st, rx, WL).alive, bool)
            a_jp = np.asarray(trace_jax_with_params(st, rx, WL).alive, bool)
        assert a_np.tolist() == [False, True], (
            f'{name}: NumPy reference must kill the grazing ray, got {a_np}')
        assert a_js.tolist() == a_np.tolist(), (
            f'{name}: jax-static alive {a_js} != numpy {a_np} (R-3)')
        assert a_jp.tolist() == a_np.tolist(), (
            f'{name}: jax-param alive {a_jp} != numpy {a_np}')


# =========================================================================
# R-4 -- NumPy flat fast path grazing phantom, and the P3-58 DOE case
# =========================================================================

def test_r4_numpy_flat_fast_path_kills_grazing_phantom():
    """R-4: a ray parallel to a flat surface cannot intersect it -- it must
    not survive the fast path as an ``opd = 0`` RAY_OK phantom."""
    from lumenairy.raytrace import RAY_MISSED_SURFACE, RAY_OK, Surface, trace
    surfs = [Surface(radius=np.inf, glass_before='air', glass_after='air',
                     thickness=0.05, semi_diameter=0.02) for _ in range(4)]
    # ray 0: grazing (N = 0);  ray 1: axial;  ray 2: mildly tilted.
    b = _bundle([0.0, 0.0, 0.0], [1e-3, 1e-3, 1e-3], [1.0, 0.0, 0.1],
                [0.0, 0.0, 0.0], [0.0, 1.0, np.sqrt(1 - 0.01)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        o = trace(b, surfs, WL).ray_history[-1]
    alive = np.asarray(o.alive, bool)
    ec = np.asarray(o.error_code, int)
    opd = np.asarray(o.opd, float)

    assert not bool(alive[0]), (
        f'R-4: the grazing (N=0) ray is still alive after a 4-flat stack '
        f'with opd = {opd[0]:.3e} -- pre-fix it was reported alive AND '
        f'RAY_OK, so trace_summary counted "3/3 alive" and folded a '
        f'never-propagated ray into the centroid / RMS spot.')
    assert ec[0] == RAY_MISSED_SURFACE, (
        f'R-4: grazing ray error_code = {ec[0]}, expected '
        f'RAY_MISSED_SURFACE = {RAY_MISSED_SURFACE} (pre-fix: '
        f'RAY_OK = {RAY_OK})')
    # The legitimate rays are untouched: 0.05 * 3 gaps = 0.15 m of air.
    assert alive[1] and alive[2] and ec[1] == RAY_OK and ec[2] == RAY_OK
    assert opd[1] == pytest.approx(0.15, rel=1e-12)
    assert opd[2] == pytest.approx(0.15 / np.sqrt(1 - 0.01), rel=1e-12)


def test_r4_doe_order_grazing_case_is_unchanged():
    """R-4 scope guard: the P3-58 DOE-order case (L^2+M^2 == 1 exactly, so
    N == 0) is kept alive BY DESIGN by the strict ``sumsq > 1.0``
    evanescence test.  That kick is applied AFTER the surface's own
    intersection (where the ray still had N = 1), so the new flat-
    intersection guard must not touch it."""
    from lumenairy.raytrace import RAY_OK, Surface, trace
    period = WL     # order +1 gives dL = wavelength/period = 1.0 exactly
    doe = {0: (1.0, 0.0, period, np.inf)}

    # (a) the DOE surface alone: the grazing order stays alive, RAY_OK.
    one = [Surface(radius=np.inf, glass_before='air', glass_after='air',
                   thickness=0.0, semi_diameter=0.05)]
    b = _bundle([0.0], [0.0], [0.0], [0.0], [1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        o = trace(b, one, WL, surface_diffraction=doe).ray_history[-1]
    assert bool(np.asarray(o.alive, bool)[0]), (
        'R-4 scope: the P3-58 DOE-order grazing case must remain ALIVE at '
        'the surface that carries the kick -- the guard is on the flat '
        'INTERSECTION, which happens before the kick.')
    assert int(np.asarray(o.error_code, int)[0]) == RAY_OK
    assert float(np.asarray(o.L, float)[0]) == pytest.approx(1.0, abs=0.0)
    assert float(np.asarray(o.N, float)[0]) == pytest.approx(0.0, abs=0.0)

    # (b) with FOLLOWING flat surfaces the order provably cannot reach them.
    #     trace_jax already kills it there; NumPy must now agree.
    three = [Surface(radius=np.inf, glass_before='air', glass_after='air',
                     thickness=t, semi_diameter=0.05)
             for t in (0.02, 0.02, 0.0)]
    b2 = _bundle([0.0], [0.0], [0.0], [0.0], [1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = trace(b2, three, WL, surface_diffraction=doe)
    hist = res.ray_history
    assert bool(np.asarray(hist[0].alive, bool)[0]), (
        'the DOE surface itself must still pass the order (see (a))')
    assert not bool(np.asarray(hist[1].alive, bool)[0]), (
        'R-4: the grazing order survived the NEXT flat surface -- pre-fix it '
        'reached the image plane alive with opd = 0.0 while trace_jax and '
        'trace_jax_with_params both killed it.')

    jax = pytest.importorskip('jax')
    del jax
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax, trace_jax_with_params
    rx = {'surfaces': [
        {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air'}] * 3,
        'thicknesses': [0.02, 0.02, 0.0]}
    z = np.array([0.0])
    st = make_jax_ray_state(z, z, z, z, z, np.array([1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a_js = bool(np.asarray(
            trace_jax(st, rx, WL, surface_diffraction=doe).alive)[0])
        a_jp = bool(np.asarray(trace_jax_with_params(
            st, rx, WL, surface_diffraction=doe).alive)[0])
    a_np = bool(np.asarray(hist[-1].alive, bool)[0])
    assert a_np == a_js == a_jp is False, (
        f'R-4: DOE + 2 flats alive parity broken -- numpy={a_np} '
        f'jax-static={a_js} jax-param={a_jp}')


# =========================================================================
# R-5 -- LGAberrationMerit (0, 0) is the Strehl DEFICIT
# =========================================================================

_LG_WL = 1.30e-6


def _lg_singlet():
    import lumenairy
    pres = lumenairy.make_singlet(R1=500e-3, R2=float('inf'), d=3e-3,
                                  glass='N-BK7', aperture=4e-3)
    pres['object_distance'] = 0.0
    return pres


def _lg_merit(pres, targets, image_points=None, w_s=20e-6, weight=1.0):
    from lumenairy.optimize.core import EvaluationContext, LGAberrationMerit
    m = LGAberrationMerit(targets=targets, field_points=[(0.0, 0.0)],
                          image_points=image_points, w_s=w_s, w_p=0.05,
                          weight=weight)
    ctx = EvaluationContext(prescription=pres, wavelength=_LG_WL, N=64,
                           dx=10e-6, x=np.array([w_s]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(m.evaluate(ctx))


def _lg_chan_sq(pres, s2, chan=(0, 0), w_s=20e-6):
    """``|L_{chan, (0,0)}|^2`` straight from the merit's own documented
    dependency (``asymptotic.aberration_tensor``), so the pin does not
    re-derive the merit's arithmetic from the merit itself."""
    from lumenairy.propagators.asymptotic import aberration_tensor, fit_canonical_polynomials
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = fit_canonical_polynomials(pres, wavelength=_LG_WL)
        t = aberration_tensor(fit, s2_image=s2, source_point=(0.0, 0.0),
                              source_modes=[(0, 0)], pupil_modes=[(0, 0)],
                              output_modes=[tuple(chan)], w_s=w_s, w_p=0.05,
                              w_o=None)
    v = complex(np.asarray(t.L).ravel()[0])
    return v.real * v.real + v.imag * v.imag


def _lg_strehl_sq(pres, s2, w_s=20e-6):
    return _lg_chan_sq(pres, s2, (0, 0), w_s)


def test_r5_lg_merit_piston_term_is_strehl_deficit():
    """R-5: the (0, 0) contribution must be ``w * (1 - |L|^2)``, not
    ``w * |L|^2`` -- ``design_optimize`` MINIMISES, so the old form drove
    |Strehl| toward 0."""
    pres = _lg_singlet()
    s_sq = _lg_strehl_sq(pres, (0.0, 0.0))
    got = _lg_merit(pres, {(0, 0): 1.0})
    assert got == pytest.approx(1.0 - s_sq, rel=0.0, abs=1e-12), (
        f'R-5: LGAberrationMerit (0,0) = {got:.12f}, expected the Strehl '
        f'DEFICIT 1 - |L|^2 = {1.0 - s_sq:.12f}.  Pre-fix it returned '
        f'|L|^2 = {s_sq:.12e} (measured 3.205062e-03 on this design), which '
        f'a minimiser drives to zero == MAXIMUM aberration.')
    # weight scaling still linear
    assert _lg_merit(pres, {(0, 0): 2.5}) == pytest.approx(
        2.5 * (1.0 - s_sq), rel=1e-12)


def test_r5_lg_merit_descends_as_aberration_falls():
    """R-5: correct descent direction.  Walking the image point off best
    focus makes the coupling |L|^2 fall monotonically; the merit must RISE
    monotonically (pre-fix it fell -- the optimiser was rewarded for
    defocusing)."""
    pres = _lg_singlet()
    offsets = (0.0, 20e-6, 50e-6, 100e-6)
    strehl = [_lg_strehl_sq(pres, (0.0, dy)) for dy in offsets]
    merit = [_lg_merit(pres, {(0, 0): 1.0}, image_points=[(0.0, dy)])
             for dy in offsets]
    assert all(strehl[i] > strehl[i + 1] for i in range(len(offsets) - 1)), (
        f'construction check: |L|^2 must fall monotonically off focus, '
        f'got {strehl}')
    assert all(merit[i] < merit[i + 1] for i in range(len(offsets) - 1)), (
        f'R-5: merit {merit} must RISE as the aberration grows (|L|^2 = '
        f'{strehl}).  Pre-fix the merit tracked |L|^2 itself and therefore '
        f'FELL -- the wrong descent direction for a minimiser.')


def test_r5_non_piston_channels_unchanged():
    """R-5 scope guard: every non-(0, 0) channel keeps ``|L|^2`` (driving a
    named aberration channel to zero IS the intent there)."""
    pres = _lg_singlet()
    for chan in ((2, 0), (1, 0)):
        ref = _lg_chan_sq(pres, (0.0, 0.0), chan)
        assert ref > 0.0, f'construction check: |L_{chan}|^2 must be nonzero'
        got = _lg_merit(pres, {chan: 1.0})
        assert got == pytest.approx(ref, rel=1e-9), (
            f'R-5 scope: channel {chan} must still contribute |L|^2, got '
            f'{got:.12e} vs {ref:.12e}')
        # ... and emphatically NOT the deficit form.
        assert got != pytest.approx(1.0 - ref, rel=1e-6, abs=1e-9)


def test_r5_numpy_lg_merit_matches_jax_twin():
    """R-5: NumPy ``LGAberrationMerit`` and the JAX
    ``make_lg_aberration_merit_jax`` now agree on the (0, 0) channel.
    Pre-fix they differed by ~0.99 -- they were literally ``x`` vs ``1-x``."""
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from lumenairy.optimize.core import EvaluationContext, make_lg_aberration_merit_jax
    pres = _lg_singlet()
    for w_s in (5e-6, 20e-6, 50e-6):
        merit = make_lg_aberration_merit_jax(
            pres, wavelength=_LG_WL, targets={(0, 0): 1.0},
            build_args=lambda x: (None, None, None, None, x[0], None),
            field_points=[(0.0, 0.0)])
        ctx = EvaluationContext(prescription=pres, wavelength=_LG_WL, N=64,
                               dx=10e-6, x=np.array([w_s]))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                jv = float(merit.evaluate(ctx))
        except (RuntimeError, ValueError, ZeroDivisionError,
                np.linalg.LinAlgError) as exc:
            pytest.skip(f'LG-tensor JAX eval unstable on this runtime: {exc}')
        if not np.isfinite(jv):
            pytest.skip('LG-tensor JAX eval returned non-finite.')
        nv = _lg_merit(pres, {(0, 0): 1.0}, w_s=w_s)
        assert nv == pytest.approx(jv, rel=1e-6, abs=1e-6), (
            f'R-5: w_s={w_s:.1e}: NumPy merit {nv:.9e} != JAX twin '
            f'{jv:.9e}.  Pre-fix NumPy returned |L|^2 while JAX returned '
            f'1 - |L|^2, so the two summed to 1.0 instead of matching.')


# =========================================================================
# R-6 -- the three OPD merits must not zero-fill vignetted pupil samples
# =========================================================================

_R6_N = 128
_R6_AP = 4e-3
_R6_DX = _R6_AP / (_R6_N - 8)
_R6_DEFOCUS_W = 0.300
_R6_SPHER_W = 0.100


def _r6_fields():
    """A pupil OPD carrying exactly ``defocus = 0.300`` and ``primary
    spherical = 0.100`` waves (orthonormal OSA), plus the same map with an
    IN-PUPIL vignetted annulus (rho > 0.8 set to NaN)."""
    x = (np.arange(_R6_N) - _R6_N / 2) * _R6_DX
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X ** 2 + Y ** 2) / (0.5 * _R6_AP)
    pupil = rho <= 1.0
    W = (_R6_DEFOCUS_W * np.sqrt(3.0) * (2 * rho ** 2 - 1.0)
         + _R6_SPHER_W * np.sqrt(5.0)
         * (6 * rho ** 4 - 6 * rho ** 2 + 1.0)) * WL
    full = np.where(pupil, W, np.nan)
    vig = np.where(pupil & (rho > 0.8), np.nan, full)
    return X, Y, full, vig


def _r6_ctx(pres, opd):
    from lumenairy.optimize.core import EvaluationContext
    return EvaluationContext(prescription=pres, wavelength=WL, N=_R6_N,
                            dx=_R6_DX, x=np.array([0.0]), opd_map=opd)


def _r6_pres():
    import lumenairy
    pres = lumenairy.make_singlet(R1=500e-3, R2=float('inf'), d=3e-3,
                                  glass='N-BK7', aperture=_R6_AP)
    pres['aperture_diameter'] = _R6_AP
    return pres


def test_r6_zernike_decompose_masks_nan_itself():
    """R-6 premise: ``zernike_decompose`` drops non-finite pupil samples
    from BOTH the design matrix and the RHS, so the merits' zero-fill was
    not merely redundant -- it was the defect."""
    from lumenairy.analysis import zernike_decompose
    _X, _Y, full, vig = _r6_fields()
    for tag, arr in (('unvignetted', full), ('vignetted', vig)):
        c, _ = zernike_decompose(arr, _R6_DX, _R6_AP, n_modes=21)
        assert c[4] / WL == pytest.approx(_R6_DEFOCUS_W, abs=2e-3), tag
        assert c[12] / WL == pytest.approx(_R6_SPHER_W, abs=2e-3), tag
    # ... and the zero-fill the merits used to apply DOES corrupt it.
    zf = np.where(np.isfinite(vig), vig, 0.0)
    c_zf, _ = zernike_decompose(zf, _R6_DX, _R6_AP, n_modes=21)
    assert c_zf[12] / WL < 0.0, (
        'construction check: the pre-fix zero-fill must SIGN-FLIP the '
        f'spherical coefficient (measured -0.0174 waves), got '
        f'{c_zf[12] / WL:+.4f}')


def test_r6_match_ideal_thin_lens_merit_survives_vignetting():
    """R-6 site 1 (``MatchIdealThinLensMerit``)."""
    from lumenairy.optimize.core import MatchIdealThinLensMerit
    X, Y, full, vig = _r6_fields()
    pres = _r6_pres()
    fl = 1.0
    ideal = -(X ** 2 + Y ** 2) / (2.0 * fl)
    exact = _R6_DEFOCUS_W ** 2 + _R6_SPHER_W ** 2   # 0.10 waves^2
    m = MatchIdealThinLensMerit(target_focal_length=fl, exclude_low_order=1,
                                n_modes=21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clean = m.evaluate(_r6_ctx(pres, ideal + full))
        vign = m.evaluate(_r6_ctx(pres, ideal + vig))
    assert clean == pytest.approx(exact, rel=2e-3)
    assert vign == pytest.approx(exact, rel=2e-2), (
        f'R-6: MatchIdealThinLensMerit = {vign:.6f} on a rho>0.8-vignetted '
        f'pupil, expected {exact:.6f} (pre-fix: 0.016145 -- the zero-filled '
        f'annulus read as genuine 0-waves OPD).')


def test_r6_match_target_opd_merit_survives_vignetting():
    """R-6 site 2 (``MatchTargetOPDMerit``)."""
    from lumenairy.optimize.core import MatchTargetOPDMerit
    _X, _Y, full, vig = _r6_fields()
    pres = _r6_pres()
    exact = _R6_DEFOCUS_W ** 2 + _R6_SPHER_W ** 2
    m = MatchTargetOPDMerit(target_opd=np.zeros((_R6_N, _R6_N)),
                            exclude_low_order=1, n_modes=21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clean = m.evaluate(_r6_ctx(pres, full))
        vign = m.evaluate(_r6_ctx(pres, vig))
    assert clean == pytest.approx(exact, rel=2e-3)
    assert vign == pytest.approx(exact, rel=2e-2), (
        f'R-6: MatchTargetOPDMerit = {vign:.6f} on a rho>0.8-vignetted '
        f'pupil, expected {exact:.6f} (pre-fix: 0.016145).')


def test_r6_zernike_coefficient_merit_survives_vignetting():
    """R-6 site 3 (``ZernikeCoefficientMerit``) -- the sharpest of the
    three: the spherical coefficient is AT its target, so the merit must be
    ~0; the zero-fill made it a large POSITIVE penalty pointing the
    optimiser the wrong way."""
    from lumenairy.optimize.core import ZernikeCoefficientMerit
    _X, _Y, full, vig = _r6_fields()
    pres = _r6_pres()
    m = ZernikeCoefficientMerit(targets={12: _R6_SPHER_W * WL}, n_modes=21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clean = m.evaluate(_r6_ctx(pres, full))
        vign = m.evaluate(_r6_ctx(pres, vig))
    assert clean < 1e-6
    assert vign < 1e-6, (
        f'R-6: ZernikeCoefficientMerit = {vign:.6e} with the targeted '
        f'spherical coefficient sitting exactly ON target; expected ~0 '
        f'(pre-fix: 1.377200e-02, from a +0.1000 -> -0.0174 waves sign '
        f'flip in the fitted coefficient).')


# =========================================================================
# R-7 -- ray_transfer_jacobian_analytic on a degenerate bundle
# =========================================================================

_R7_DEGENERATE = {'underflowing slope (N->0)': 1e300,
                  'infinite slope': np.inf,
                  'nan slope': np.nan}


def _r7_surfaces():
    from lumenairy.raytrace import Surface
    return [Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                    thickness=3e-3, semi_diameter=10e-3),
            Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                    thickness=45e-3, semi_diameter=10e-3)]


@pytest.mark.parametrize('label', sorted(_R7_DEGENERATE))
def test_r7_analytic_jacobian_handles_degenerate_bundle(label):
    """R-7: the analytic twin must not raise where the FD sibling copes."""
    from lumenairy.raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )
    surfs = _r7_surfaces()
    z = np.array([0.0])
    u = np.array([_R7_DEGENERATE[label]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fd = ray_transfer_jacobian(z, z, u, z, surfs, WL)
        try:
            an = ray_transfer_jacobian_analytic(z, z, u, z, surfs, WL)
        except ZeroDivisionError as exc:      # pragma: no cover - the bug
            pytest.fail(
                f'R-7: ray_transfer_jacobian_analytic raised a bare '
                f'ZeroDivisionError ({exc}) on a {label} bundle that the '
                f'documented FD sibling ray_transfer_jacobian handles.  '
                f'The numba fast path compiles with numba\'s default '
                f'error_model="python", unlike the _AdrtDual sibling it is '
                f'meant to be bit-identical to.')
    for tag, dt in (('FD', fd), ('analytic', an)):
        j = np.asarray(dt.jacobian)
        assert j.shape == (1, 4, 4), f'{tag} jacobian shape {j.shape}'
        assert np.all(np.isfinite(j)), (
            f'{tag}: degenerate bundle must yield a finite (masked) '
            f'Jacobian, got {j}')


@pytest.mark.parametrize('label', sorted(_R7_DEGENERATE))
def test_r7_degenerate_fallback_matches_the_dual_reference(label):
    """R-7: the fallback must land on the documented pure-NumPy dual path
    (the same result ``per_surface=True`` already returned), not on some
    third answer."""
    from lumenairy.raytrace import differential as D
    surfs = _r7_surfaces()
    z = np.array([0.0])
    u = np.array([_R7_DEGENERATE[label]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = D.ray_transfer_jacobian_analytic(z, z, u, z, surfs, WL)
        ref = D._adrt_numpy(z, z, u, z, surfs, WL, False)
    np.testing.assert_array_equal(np.asarray(got.jacobian),
                                  np.asarray(ref.jacobian))
    np.testing.assert_array_equal(np.asarray(got.alive),
                                  np.asarray(ref.alive))


def test_r7_non_degenerate_bundle_still_takes_the_numba_fast_path():
    """R-7 scope guard: the ZeroDivisionError fallback must not shadow the
    numba fast path for ordinary bundles (it stays ULP-identical to the
    dual reference)."""
    from lumenairy.raytrace import differential as D
    if D._adrt_numba_kernel() is None:
        pytest.skip('numba unavailable; the fast path is not in play')
    surfs = _r7_surfaces()
    x = np.zeros(5)
    y = np.linspace(-5e-3, 5e-3, 5)
    ux = np.zeros(5)
    uy = np.linspace(-0.02, 0.02, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nb = D._adrt_numba(x, y, ux, uy, surfs, WL)
        pub = D.ray_transfer_jacobian_analytic(x, y, ux, uy, surfs, WL)
    assert nb is not None
    np.testing.assert_array_equal(np.asarray(pub.jacobian),
                                  np.asarray(nb.jacobian))
