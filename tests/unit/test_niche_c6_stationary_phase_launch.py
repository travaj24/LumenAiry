"""Niche C6: ``preserve_input_phase='remap'`` launches along the TOTAL
entrance eikonal ``grad(W + a_fit)``, not the carrier's ``grad(W)`` alone.

THE DERIVATION THESE PINS ENCODE
--------------------------------
Write the input as ``E_in = |E_in| exp(i k0 (W + a))`` -- ``W`` the carrier
eikonal, ``a`` the residual -- and let ``V(x, X)`` be the element's point
characteristic (the traced optical path from entrance ``x`` to exit ``X``).
Geometrical optics gives the exit eikonal as a STATIONARY value::

    Psi(X) = stat_x [ W(x) + a(x) + V(x, X) ]  ==  F(x_*)

and Hamilton's ``dV/dx = -p_in`` makes ``x_*`` the foot of the ray launched
along ``grad(W + a)``.

The shipped 'remap' evaluates the SAME ``F`` at the foot ``x_W`` of the
``grad(W)`` ray: ``opl_map`` carries ``W + V`` and the transported residual
phasor multiplies ``exp(i k0 a)`` on at that foot.  Since ``x_W`` is
stationary for ``W + V`` alone, ``grad F(x_W) = grad a(x_W)``, so::

    Phi_remap - Psi = F(x_W) - F(x_*)
                    = 1/2 * grad a^T H^-1 grad a + O(|grad a|^3)

with ``H = Hess_x (W + V)``.  Two consequences are pinned below and they are
what make this a MECHANISM rather than a correlation:

  * the shipped error is QUADRATIC in the residual's ray slope (test 5), and
  * launching along ``grad(W + a_fit)`` removes it to the order the model
    captures ``a`` (tests 4 and 5).

Everything is scored against an EXACT-RAY ORACLE built in this file: rays
launched along the analytic ``grad(W + a)`` of a KNOWN residual, traced by
``lumenairy.raytrace``, exit-vertex corrected, and Newton-inverted onto the
element's own exit nodes with the exact trace as the residual.  No design-121
asset, no Zemax, no FFT derivative and no phase unwrap is involved -- the
comparison is pointwise and its sampling adequacy is asserted as the
power-weighted p99 of the wrapped nearest-neighbour step against pi (test 0),
because on this study a bare max saturates at pi on a clean core and an
unwrap-based rms then reads a measurement artefact.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT
from lumenairy.glass import get_glass_index
from lumenairy.raytrace import trace
from lumenairy.raytrace.trace import _make_bundle, surfaces_from_prescription

# THE FIXTURE, and why it is a FREE LEG.
#
# The dropped term is ``1/2 grad a^T H^-1 grad a`` with ``H = Hess_x(W + V)``,
# so its size is set by ``H^-1`` -- an effective LENGTH.  For a free leg of
# length ``z`` under a carrier sphere of radius ``R`` that is exactly
# ``z R / (z + R)``, which makes the whole effect predictable in closed form
# and the fixture a quantitative pin rather than a direction-of-change one.
#
# It also rules out a fixture that cannot see the effect at all: a short
# singlet measured AT ITS EXIT VERTEX has ``H^-1`` of order a millimetre, and
# the same 2 mrad residual slope then costs 4e-4 waves -- below this study's
# own noise.  Design 121's failing group has ``H^-1 ~ 32 mm``, which is the
# regime reproduced here.
_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_N, _DX, _W = 384, 25e-6, 1.5e-3
_RC = -0.06                      # carrier conjugate (converging), metres
_Z = 15e-3                       # free-leg length, metres
_AP = 12e-3
_ALPHA = 2.0                     # r^4 residual: phase in RADIANS at r = w
_RS = 4


# ---------------------------------------------------------------------------
# fixtures: a free leg, a carrier sphere and a KNOWN analytic residual
# ---------------------------------------------------------------------------
def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _free_leg(z=_Z, ap=_AP):
    """A pure free leg of length ``z`` expressed as a two-flat-surface
    'prescription' -- zero optical power, an EXACT ray map, and an ``H^-1``
    that is known in closed form."""
    return {'name': 'c6_freeleg', 'aperture_diameter': ap,
            'surfaces': [_flat(), _flat()], 'thicknesses': [z]}


def _h_inv(z=_Z, R=_RC):
    """``H^-1`` for a free leg under a carrier sphere -- the paraxial
    ``z R / (z + R)``."""
    return abs(z * R / (z + R))


def _carrier_WLM(x, y, s=_RC):
    """Exact on-axis point-source sphere -- the same closed form
    ``_compute_carrier`` uses for a scalar conjugate.

    "The same closed form" is the whole point of this helper and it is a
    LOAD-BEARING claim, not a convenience: ``test_pure_congruence_input_is_
    inert_to_rounding`` builds its input as ``exp(i k0 W)`` from here and then
    asserts the element finds no residual in it.  An input built from a
    DIFFERENT arithmetic route than the element's own is not that congruence,
    and the difference shows up as a residual the element then models.

    So this tracks the library form exactly.  It was
    ``sgn * (rho - abs(s))`` until 2026-08-12, when ``_compute_carrier`` was
    rationalized to ``sgn * r^2 / (rho + |s|)`` to close the
    ``sqrt(r^2+R^2) - |R|`` cancellation class.  The two differ by up to
    6.586e-18 m here (eps*|R| = 1.332e-17 m), whose GRID GRADIENT is 4.01e-13
    -- and the congruence test's ``grad_a_rms`` duly read 1.338e-13 against a
    floor of 3.459e-17 until this helper followed.  The subtraction form is
    the WORSE of the two (a 60-digit oracle puts it at k0*eps*|R| rad, ~951x
    the rationalized form on the campaign's own fixture), so following it is
    also the more accurate oracle."""
    sgn = 1.0 if s > 0 else -1.0
    r2 = x * x + y * y
    rho = np.sqrt(r2 + s * s)
    return sgn * (r2 / (rho + abs(s))), sgn * x / rho, sgn * y / rho


def _resid_aLM(x, y, alpha=_ALPHA, w=_W):
    """``a`` (metres) and its transverse ray slope for the r^4 residual
    ``phase = alpha * (r/w)^4`` -- the r^4-dominant content a
    carrier-referenced relay actually carries."""
    r2 = x * x + y * y
    a = (alpha / _K0) * (r2 / (w * w)) ** 2
    g = (4.0 * alpha / (_K0 * w ** 4)) * r2
    return a, g * x, g * y


def _field(alpha=_ALPHA, n=_N, dx=_DX, w=_W):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    W, _l, _m = _carrier_WLM(X, Y)
    a, _la, _ma = _resid_aLM(X, Y, alpha=alpha, w=w)
    amp = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    return (amp * np.exp(1j * _K0 * (W + a))).astype(np.complex128), X, Y


_BASE_KW = dict(wavelength=_WL, dx=_DX, amplitude_model='ray_density',
                preserve_input_phase='remap', remap_sampling='full',
                parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent', ray_subsample=_RS,
                fit_radius_beam_factor=2.0)


def _run(E, flag, presc=None, carrier=_RC, out=None, **over):
    old = LT.REMAP_STATIONARY_PHASE_LAUNCH
    LT.REMAP_STATIONARY_PHASE_LAUNCH = flag
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kw = dict(_BASE_KW)
            kw.update(over)
            return np.asarray(la.apply_real_lens_traced(
                E, prescription=presc or _free_leg(), carrier=carrier,
                _remap_launch_out=out, **kw))
    finally:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = old


# ---------------------------------------------------------------------------
# the exact-ray oracle
# ---------------------------------------------------------------------------
def _surfaces(presc):
    """The surface list the element itself traces (it pops the prescription
    aperture before building them)."""
    p = dict(presc)
    p.pop('aperture_diameter', None)
    return surfaces_from_prescription(p)


def _trace_total(xe, ye, surfs, alpha):
    """Exact trace launched along ``grad(W + a)``; returns exit position, the
    TOTAL eikonal ``W + a + OPL`` at the flat exit VERTEX plane, and the exit
    direction cosines."""
    xe = np.asarray(xe, dtype=np.float64).ravel()
    ye = np.asarray(ye, dtype=np.float64).ravel()
    W, L, M = _carrier_WLM(xe, ye)
    a, la_, ma_ = _resid_aLM(xe, ye, alpha=alpha)
    rays = _make_bundle(x=xe.copy(), y=ye.copy(), L=(L + la_).copy(),
                        M=(M + ma_).copy(), wavelength=_WL)
    fin = trace(rays, surfs, _WL, output_filter='last').image_rays
    n_exit = get_glass_index(surfs[-1].glass_after, _WL)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    opd = np.asarray(fin.opd) + n_exit * t + W + a
    xo = np.asarray(fin.x) + np.asarray(fin.L) * t
    yo = np.asarray(fin.y) + np.asarray(fin.M) * t
    return (xo, yo, opd, np.asarray(fin.L), np.asarray(fin.M),
            np.asarray(fin.alive, dtype=bool))


def _exact_phase(Xt, Yt, surfs, alpha, n_iter=14, tol=1e-12, h=5e-8):
    """Newton-invert the EXACT total-eikonal map onto the requested exit nodes.

    The Newton RESIDUAL is the exact trace (the Jacobian is a central
    difference of the same trace, so the Jacobian's error cannot move the
    fixed point), and a first-order exit-direction correction removes the
    leftover landing residual to O(resid^2)."""
    sh = Xt.shape
    xt, yt = Xt.ravel(), Yt.ravel()
    xe, ye = xt.copy(), yt.copy()
    for _ in range(n_iter):
        xo, yo, psi, Lo, Mo, alive = _trace_total(xe, ye, surfs, alpha)
        rx, ry = xo - xt, yo - yt
        if np.nanmax(np.hypot(rx, ry)) < tol:
            break
        x1, y1 = _trace_total(xe + h, ye, surfs, alpha)[:2]
        x2, y2 = _trace_total(xe, ye + h, surfs, alpha)[:2]
        jxx, jyx = (x1 - xo) / h, (y1 - yo) / h
        jxy, jyy = (x2 - xo) / h, (y2 - yo) / h
        det = jxx * jyy - jxy * jyx
        g = np.isfinite(det) & (np.abs(det) > 1e-16)
        dxe = np.where(g, (jyy * rx - jxy * ry) / np.where(g, det, 1.0), 0.0)
        dye = np.where(g, (-jyx * rx + jxx * ry) / np.where(g, det, 1.0), 0.0)
        xe, ye = xe - dxe, ye - dye
    xo, yo, psi, Lo, Mo, alive = _trace_total(xe, ye, surfs, alpha)
    rx, ry = xo - xt, yo - yt
    psi_node = psi - (Lo * rx + Mo * ry)
    ok = alive & np.isfinite(psi_node) & (np.hypot(rx, ry) < 1e-9)
    return (_K0 * psi_node).reshape(sh), ok.reshape(sh)


def _sigma_and_step(E_patch, phi_exact, keep):
    """UNWRAP-FREE equivalent rms wavefront error (waves) from the
    piston-removed coherent fidelity, plus the POWER-WEIGHTED p99 of the
    wrapped nearest-neighbour step of the residual (the sampling-adequacy
    statement; a bare max is set by a handful of skirt pixels)."""
    amp = np.abs(E_patch)
    keep = keep & (amp > 0) & np.isfinite(phi_exact)
    z = np.asarray(E_patch) * np.exp(-1j * np.asarray(phi_exact))
    psi = np.angle(np.where(keep, z, 1.0))
    w = np.where(keep, amp ** 2, 0.0)
    F = float(np.abs((w * np.exp(1j * psi))[keep].sum()) ** 2
              / max(float(w[keep].sum()) ** 2, 1e-300))
    sigma = float(np.sqrt(max(-np.log(max(F, 1e-16)), 0.0)) / (2 * np.pi))
    dx1 = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1])))
    dy1 = np.angle(np.exp(1j * (psi[1:, :] - psi[:-1, :])))
    kx = keep[:, 1:] & keep[:, :-1]
    ky = keep[1:, :] & keep[:-1, :]
    v = np.concatenate([np.abs(dx1[kx]), np.abs(dy1[ky])])
    ww = np.concatenate([np.minimum(w[:, 1:], w[:, :-1])[kx],
                         np.minimum(w[1:, :], w[:-1, :])[ky]])
    o = np.argsort(v)
    cw = np.cumsum(ww[o])
    cw = cw / max(cw[-1], 1e-300)
    p999 = float(v[o][min(np.searchsorted(cw, 0.999), v.size - 1)])
    return sigma, p999, int(keep.sum())


def _score(alpha, flag, half=None):
    """Exit wavefront error of the element against the exact-ray oracle."""
    E, X, Y = _field(alpha=alpha)
    presc = _free_leg()
    F = _run(E, flag, presc=presc)
    surfs = _surfaces(presc)
    c = _N // 2
    half = half if half is not None else int(2.0 * _W / _DX)
    sl = slice(max(c - half, 0), min(c + half + 1, _N))
    Xp, Yp, Ep = X[sl, sl], Y[sl, sl], F[sl, sl]
    phi, ok = _exact_phase(Xp, Yp, surfs, alpha)
    amp = np.abs(Ep)
    keep = ok & (amp > 0.02 * float(amp.max()))
    return _sigma_and_step(Ep, phi, keep)


@pytest.fixture(scope='module')
def _warm():
    """Push every byte-identity pair in this file past the traced pipeline's
    warm-up boundary (the W9 determinism calibration: in a fresh process the
    traced output for one fixed input can change ONCE, at ulp scale, after the
    first calls).  Both members of every compared pair must sit on the same
    side of it."""
    E, _X, _Y = _field()
    for _ in range(2):
        _run(E, False)
    return True


# ---------------------------------------------------------------------------
# 0. the instrument itself: sampling adequacy, and the oracle's own closure
# ---------------------------------------------------------------------------
def test_oracle_sampling_is_adequate(_warm):
    """The pointwise comparison is only legitimate while the residual's
    wrapped nearest-neighbour step stays far below pi.  Asserted as the
    POWER-weighted p99.9, not a max."""
    for flag in (False, True):
        _sig, p999, npix = _score(_ALPHA, flag)
        assert npix > 400, npix
        assert p999 < 0.25 * np.pi, (flag, p999)


def test_oracle_is_self_consistent_at_zero_residual(_warm):
    """With NO residual the two launches are the same congruence, so the
    oracle must agree with the element to well below the effect being
    measured -- this bounds the oracle's own error, not the element's."""
    sig, _p, _n = _score(0.0, False)
    assert sig < 2e-3, sig


# ---------------------------------------------------------------------------
# 1. the fail-before switch, and what it must NOT touch
# ---------------------------------------------------------------------------
def test_flag_off_is_deterministic_and_flag_on_acts(_warm):
    E, _X, _Y = _field()
    a = _run(E, False)
    b = _run(E, False)
    assert np.array_equal(a, b)
    c = _run(E, True)
    assert not np.array_equal(a, c)
    assert float(np.abs(a - c).max()) > 0.0


@pytest.mark.parametrize('pip,amod', [
    (True, 'screen'), (False, 'screen'), (True, 'ray_density'),
    (False, 'ray_density')])
def test_inert_outside_remap(_warm, pip, amod):
    """Only ``preserve_input_phase='remap'`` transports the residual
    GEOMETRICALLY, so only there may the launch direction move.  Every other
    mode must be BYTE-identical with the flag on and off."""
    E, _X, _Y = _field()
    a = _run(E, False, preserve_input_phase=pip, amplitude_model=amod)
    b = _run(E, True, preserve_input_phase=pip, amplitude_model=amod)
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_inert_without_an_engaged_carrier(_warm):
    """``a`` is only a small RESIDUAL relative to an engaged carrier; with no
    carrier the de-chirp is the identity and the 'residual' would be the
    input's whole phase.  The launch stays the shipped one there."""
    E, _X, _Y = _field()
    out_off, out_on = {}, {}
    a = _run(E, False, carrier=None, out=out_off)
    b = _run(E, True, carrier=None, out=out_on)
    assert out_on['engaged'] is False
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_pure_congruence_input_is_inert_to_rounding(_warm):
    """An input that IS the carrier congruence (real, positive envelope times
    ``exp(i k0 W)``) has a residual of 1, so there is nothing to model and the
    augmentation must vanish.

    It is NOT bit-for-bit, and that is stated rather than hidden -- C6 cannot
    be byte-identical on any input carrying a residual, however small:

      * the de-chirp ``E * exp(-i k0 W)`` reconstructs ``exp(i k0 W)`` by a
        different floating-point route than the input did, so the measured
        slope floors at ~3e-17 rad rather than exactly 0;
      * a launch perturbation of that size still moves the Newton pullback of
        a FAR SKIRT pixel across the ``0.99 * launch_radius`` ray-domain
        threshold, and such a pixel flips between its (tiny) value and exactly
        zero.  Measured: the largest single-pixel departure is 1.9e-5 of peak
        and sits at r = 2.5 w, while the total energy moved is 3.7e-12 of the
        field.

    So the pin is on ENERGY, where a boundary flip cannot masquerade as a
    change: 9 orders below the effect the mode exists to correct.

    AND THE SLOPE BARS ARE NOT ABSOLUTE (re-stated 2026-08-12).  They used to
    be ``grad_a_rms < 1e-14`` and ``grad_a_fit_max_launch < 1e-12``: absolute
    floors on a ROUNDING RESIDUE, i.e. on a magnitude a different BLAS, a
    different libm or a different summation order is entitled to move, and
    which carries no scale of its own.  They are now scored two ways that do
    carry one:

      * against the FLOAT64 CEILING the reconstruction cannot beat.  The
        de-chirp rebuilds ``exp(i k0 W)`` by a different route than the input
        did, so its eikonal error is ~``eps * |W|`` metres; spread over one
        pixel that is at most ``eps * max|W| / dx`` in SLOPE units, which is
        3.400e-15 here (max|W| = 3.828e-04 m).  The fit smooths it further
        and the measured value is 3.459e-17, 98x under.  No BLAS can move a
        bound set by the mantissa.
      * against the SAME quantity on the deliberately NON-congruent fixture
        measured in the same process, which is what "inert" is inert compared
        to: 3.459e-17 / 6.261e-04 = 5.5e-14 and 8.330e-16 / 1.737e-02 =
        4.8e-14, against a 1e-11 bar.

    Both arms have a live fail-before, and it is the one that actually
    happened: with the fixture built from the pre-2026-08-12 subtraction-form
    sphere (see ``_carrier_WLM``) the readings are 1.338e-13 and 4.786e-12 --
    39x over the float64 ceiling and, as ratios, 2.14e-10 and 2.76e-10
    against the 1e-11 bar."""
    ax = (np.arange(_N) - _N // 2) * _DX
    X, Y = np.meshgrid(ax, ax)
    W, _l, _m = _carrier_WLM(X, Y)
    E = (np.exp(-(X ** 2 + Y ** 2) / _W ** 2)
         * np.exp(1j * _K0 * W)).astype(np.complex128)
    out = {}
    a = _run(E, False)
    b = _run(E, True, out=out)
    assert out['engaged'] is True

    # the control: the same diagnostics on an input that DOES carry a
    # residual, measured in this process so no cross-build number is involved
    ctl = {}
    _run(_field()[0], True, out=ctl)
    assert ctl['grad_a_rms'] > 0.0 and ctl['grad_a_fit_max_launch'] > 0.0

    # (a) the float64 ceiling on rebuilding exp(i k0 W) by another route
    ceiling = float(np.finfo(np.float64).eps * np.abs(W).max() / _DX)
    assert out['grad_a_rms'] <= ceiling, (
        f"grad_a_rms {out['grad_a_rms']:.4e} is above the float64 "
        f"reconstruction ceiling eps*max|W|/dx = {ceiling:.4e}: the input is "
        f"not the congruence the element is referencing")
    # (b) inert COMPARED TO an input that is not a pure congruence
    assert out['grad_a_rms'] <= 1e-11 * ctl['grad_a_rms'], (
        out['grad_a_rms'], ctl['grad_a_rms'])
    assert (out['grad_a_fit_max_launch']
            <= 1e-11 * ctl['grad_a_fit_max_launch']), (
        out['grad_a_fit_max_launch'], ctl['grad_a_fit_max_launch'])
    d2 = float((np.abs(a - b) ** 2).sum())
    p = float((np.abs(a) ** 2).sum())
    assert d2 / p < 1e-9, d2 / p
    assert float(np.abs(a - b).max()) < 1e-4 * float(np.abs(a).max())


def test_diagnostic_sink_contract(_warm):
    E, _X, _Y = _field()
    out = {}
    _run(E, True, out=out)
    assert out['engaged'] is True and out['flag'] is True
    assert out['remap'] is True
    for k in ('degree', 'n_terms', 'n_samples', 'grad_a_rms',
              'grad_a_residual_rms', 'w_beam', 'centre', 'r_fit',
              'grad_a_fit_max_launch'):
        assert k in out, k
    assert 1 <= out['degree'] <= LT._REMAP_RESID_DEGREE_CAP
    # the model must explain most of the slope it measured
    assert out['grad_a_residual_rms'] < 0.15 * out['grad_a_rms']


# ---------------------------------------------------------------------------
# 2. the residual-eikonal model is a genuine POTENTIAL, and it is BOUNDED
# ---------------------------------------------------------------------------
def _fit_model(alpha=_ALPHA):
    E, X, Y = _field(alpha=alpha)
    W, _l, _m = _carrier_WLM(X, Y)
    return LT._fit_residual_eikonal(E, W, _WL, _DX, _DX, (0.0, 0.0), _W,
                                    stride=1)


def test_model_gradient_is_the_gradient_of_its_value():
    """``grad`` must be the true gradient of ``value`` -- INSIDE the fit disc
    and, the part that is easy to get wrong, across and beyond the C^1 radial
    freeze.  A launch direction that is not the gradient of the eikonal the
    H6 term adds would not be a congruence at all."""
    eik = _fit_model()
    assert eik is not None
    rng = np.random.default_rng(7)
    r1 = eik.r_fit
    for rad in (0.3 * r1, 0.9 * r1, 1.05 * r1, 2.0 * r1, 6.0 * r1):
        th = rng.uniform(0, 2 * np.pi, 24)
        x = rad * np.cos(th)
        y = rad * np.sin(th)
        h = 1e-4 * r1
        gx, gy = eik.grad(x, y)
        nx = (eik.value(x + h, y) - eik.value(x - h, y)) / (2 * h)
        ny = (eik.value(x, y + h) - eik.value(x, y - h)) / (2 * h)
        sc = max(float(np.abs(np.concatenate([gx, gy])).max()), 1e-12)
        assert np.allclose(gx, nx, atol=2e-5 * sc, rtol=2e-4), rad
        assert np.allclose(gy, ny, atol=2e-5 * sc, rtol=2e-4), rad


def test_model_slope_is_bounded_far_outside_the_fit_disc():
    """The freeze exists so the model cannot deflect a marginal launch ray by
    a polynomial extrapolation.  Its gradient must stay bounded as r grows
    without limit -- a windowed or bare polynomial does not."""
    eik = _fit_model()
    r1 = eik.r_fit
    th = np.linspace(0, 2 * np.pi, 33)[:-1]
    g_in = float(np.hypot(*eik.grad(r1 * np.cos(th), r1 * np.sin(th))).max())
    prev = None
    for k in (2.0, 10.0, 100.0, 1e4):
        g = float(np.hypot(*eik.grad(k * r1 * np.cos(th),
                                     k * r1 * np.sin(th))).max())
        assert np.isfinite(g)
        assert g < 4.0 * g_in, (k, g, g_in)
        if prev is not None:
            assert g < 1.5 * prev + 1e-12, (k, g, prev)
        prev = g


def test_model_has_zero_radial_curvature_outside_the_freeze():
    """The freeze is a LINEAR continuation in r, so the second radial
    derivative is exactly 0 beyond it.  That is what makes the continuation
    unable to focus -- the ring-caustic failure mode of the multiplicative
    window this replaced."""
    eik = _fit_model()
    r1 = eik.r_fit
    h = 1e-3 * r1
    for th in (0.0, 0.7, 2.1, 4.4):
        for rad in (1.5 * r1, 3.0 * r1):
            r = np.array([rad - h, rad, rad + h])
            v = eik.value(r * np.cos(th), r * np.sin(th))
            d2 = (v[0] - 2 * v[1] + v[2]) / (h * h)
            scale = abs(v[1]) / (rad * rad) + 1e-30
            assert abs(d2) < 1e-6 * max(scale, 1e-12) + 1e-9 * abs(v[1]) / h,\
                (th, rad, d2)


def test_the_freeze_circle_clears_the_ray_fit_disc():
    """The model's SAMPLE disc and its RADIAL FREEZE circle are different
    radii, and the freeze has to clear the caller's RAY-fit disc.

    They were the same number until 2026-07-31, and that coincidence broke the
    element's two ``newton_fit`` backends apart.  The polynomial forward-map
    fit is restricted to the ray-fit disc and EXTRAPOLATES outside it; putting
    the freeze's curvature discontinuity on that same circle makes the
    extrapolation invalid from its first pixel outward.  Scored against the
    EXACT skew ray trace of the same augmented launch, the polynomial backend's
    map then carried 5.6 um of error in the 2-4 w skirt and 15.1 um at the
    entrance-aperture rim where the spline backend carried 0.006 and 0.002 --
    and 0.089 um with the C6 launch off.  See ``_REMAP_RESID_FREEZE_MARGIN``.

    The FIT itself must not move: only the continuation does.
    """
    E, X, Y = _field()
    W, _l, _m = _carrier_WLM(X, Y)

    def fit(rr):
        return LT._fit_residual_eikonal(E, W, _WL, _DX, _DX, (0.0, 0.0), _W,
                                        stride=1, ray_fit_radius=rr)

    base = fit(None)
    assert base is not None
    r_samp = LT._REMAP_RESID_FIT_W * _W
    # no ray-fit disc -> the historical single radius, unchanged
    assert base.scale == pytest.approx(r_samp)
    assert base.r_fit == pytest.approx(r_samp)
    assert base.diag['r_freeze'] == pytest.approx(r_samp)
    assert base.diag['ray_fit_radius'] is None
    for rr in (0.4 * _W, r_samp, 2.4 * _W, 20.0 * _W):
        e = fit(rr)
        rz = e.diag['r_freeze']
        assert e.r_fit == rz, 'r_fit IS the freeze circle'
        assert rz >= r_samp - 1e-15, (rr, rz)       # never inside the data
        assert rz <= LT._REMAP_RESID_FREEZE_MAX_W * _W + 1e-15, (rr, rz)
        if (LT._REMAP_RESID_FREEZE_MARGIN * rr
                <= LT._REMAP_RESID_FREEZE_MAX_W * _W):
            assert rz > rr, (rr, rz)                # ... and it clears the disc
        # the sample disc, the basis normalisation and the coefficients are
        # untouched -- this moves the continuation, never the fit
        assert e.scale == pytest.approx(r_samp)
        assert e.diag['r_fit'] == pytest.approx(r_samp)
        assert np.array_equal(e.coef, base.coef)
        assert e.diag['n_samples'] == base.diag['n_samples']


def test_the_two_newton_fit_backends_still_describe_the_same_map(monkeypatch):
    """``newton_fit`` is an interpolant choice, not a physics choice: the
    polynomial and the spline read the SAME traced samples and must return the
    same field.  With the freeze circle collapsed back onto the ray-fit disc
    they do not -- which is what niche D7's
    ``test_the_off_centre_field_tracks_the_unrestricted_spline_map`` caught.

    SCORED ON THE FORWARD PATH (``inverse_map=False``), because that is where
    the freeze circle lives and where its fail-before is observable.  The
    stronger statement is asserted separately below it: with the
    inverse-characteristic evaluator engaged -- the shipped default since
    ``FIX_G8_PROBE_2026_08_12`` -- the two backends do not merely agree to
    5e-04, they return the SAME BYTES, because the model that produces the
    returned OPL / entrance / ``det J`` is basis-independent by construction
    and its acceptance no longer depends on which interpolant the incumbent
    is.  That also makes the freeze-circle fail-before INERT on the shipped
    default (both arms come from the same model), which is the map removing
    the field's dependence on the forward fit, not the defect going away --
    hence both arms are kept, each measured where it is real.
    """
    E, _X, _Y = _field()

    def spread(imap):
        ref = np.abs(_run(E, True, newton_fit='spline', inverse_map=imap))
        got = np.abs(_run(E, True, inverse_map=imap))
        scale = float(ref.max())
        assert scale > 0.0
        return float(np.abs(got - ref).max()) / scale

    d_map = spread(True)
    assert d_map == 0.0, (
        'the two newton_fit backends no longer return byte-identical fields '
        f'with the inverse map engaged: {d_map}')

    d_ok = spread(False)
    monkeypatch.setattr(LT, '_REMAP_RESID_FREEZE_MARGIN', 1.0)
    monkeypatch.setattr(LT, '_REMAP_RESID_FREEZE_MAX_W', LT._REMAP_RESID_FIT_W)
    d_bad = spread(False)
    assert d_ok < 5.0e-4, d_ok
    assert d_bad > 3.0 * d_ok, (d_ok, d_bad)


def test_degree_cap_is_enforced():
    old = LT._REMAP_RESID_EIKONAL_DEGREE
    LT._REMAP_RESID_EIKONAL_DEGREE = 40
    try:
        eik = _fit_model()
    finally:
        LT._REMAP_RESID_EIKONAL_DEGREE = old
    assert eik.diag['degree'] <= LT._REMAP_RESID_DEGREE_CAP


# ---------------------------------------------------------------------------
# 3. the aliased-sample rejection (the reading, not the physics)
# ---------------------------------------------------------------------------
def test_aliased_increments_are_rejected_not_fitted():
    """A wrapped nearest-neighbour increment past ~pi/2 is not a slope
    reading, it is a fold.  Feeding one to the fit would import a wrong ray
    direction; the sampler must drop it instead."""
    E, X, Y = _field()
    W, _l, _m = _carrier_WLM(X, Y)
    old = LT._REMAP_RESID_MAX_STEP_RAD
    try:
        LT._REMAP_RESID_MAX_STEP_RAD = 0.5 * np.pi
        good = LT._fit_residual_eikonal(E, W, _WL, _DX, _DX, (0.0, 0.0), _W,
                                        stride=1)
        LT._REMAP_RESID_MAX_STEP_RAD = 1e-6
        starved = LT._fit_residual_eikonal(E, W, _WL, _DX, _DX, (0.0, 0.0),
                                          _W, stride=1)
    finally:
        LT._REMAP_RESID_MAX_STEP_RAD = old
    assert good is not None and good.diag['n_samples'] > 1000
    # with the acceptance driven to zero there is nothing left to fit
    assert starved is None or starved.diag['n_samples'] < 64


# ---------------------------------------------------------------------------
# 4/5. THE ACCURACY CLAIM, and the QUADRATIC scaling that names the mechanism
# ---------------------------------------------------------------------------
def test_stationary_phase_launch_removes_the_residual_model_error(_warm):
    """The headline: against an exact-ray oracle the shipped ``grad(W)``
    launch carries a real exit wavefront error, and launching along
    ``grad(W + a_fit)`` removes essentially ALL of it.

    On this fixture the residual is exactly ``alpha (r/w)^4``, which the
    shipping degree-4 potential spans EXACTLY, so the leftover is the
    numerical floor rather than a modelling remainder -- 0.021 -> 2e-5 waves,
    a factor of ~1000.

    This holds on the SHIPPED (hard-mask) ray fit.  Turning on the opt-in
    ``REMAP_STATIONARY_PHASE_FIT_GUARD`` takes it to 6.93e-04 waves -- still a
    30x removal, but 30x worse than this -- because this fixture is the best
    case for the hard mask (the free leg's augmented map is a CUBIC inside the
    fit disc, which order 6 fits exactly, while outside it the residual model's
    radial freeze continues ``a`` linearly in r, so a weighted fit imports a
    shape the Chebyshev basis cannot represent).  Pinned in
    test_niche_c6_fit_guard.py; see that flag's note for why it is opt-in."""
    off, p_off, _n = _score(_ALPHA, False)
    on, p_on, _n2 = _score(_ALPHA, True)
    assert p_off < 0.5 * np.pi and p_on < 0.25 * np.pi
    assert off > 0.01, off                 # the defect is real and measurable
    assert on < 0.02 * off, (off, on)      # and removed, not merely reduced


def test_shipped_error_matches_the_closed_form_second_order_term(_warm):
    """QUANTITATIVE, not directional.  For a free leg under a carrier sphere
    ``H^-1 = z R / (z + R)`` in closed form, so the dropped term
    ``1/2 grad a^T H^-1 grad a`` predicts the MAGNITUDE of the shipped
    launch's exit error, up to a shape factor relating the pointwise term to
    an amplitude-weighted, piston-and-tilt-removed rms over the beam.

    Pinned as: the ratio measured/predicted stays O(1) across an 11x span of
    ``H^-1`` -- a wrong mechanism does not track an 11x lever to within a
    factor of 2.  The ratio DRIFTS downward (2.6 -> 1.4) because the leading
    term over-predicts once the error stops being small; that is a property of
    the truncation, and the quadratic-scaling test below is taken in the
    regime where it holds."""
    ratios = []
    for z in (5e-3, 15e-3, 30e-3):
        presc = _free_leg(z=z)
        E, X, Y = _field(alpha=_ALPHA)
        F = _run(E, False, presc=presc)
        surfs = _surfaces(presc)
        c = _N // 2
        h = int(2.0 * _W / _DX)
        sl = slice(max(c - h, 0), min(c + h + 1, _N))
        phi, ok = _exact_phase(X[sl, sl], Y[sl, sl], surfs, _ALPHA)
        amp = np.abs(F[sl, sl])
        sig, p999, _n = _sigma_and_step(
            F[sl, sl], phi, ok & (amp > 0.02 * float(amp.max())))
        assert p999 < 0.35 * np.pi, (z, p999)
        g_w = 4.0 * _ALPHA / (_K0 * _W)          # |grad a| at r = w
        pred = 0.5 * g_w * g_w * _h_inv(z=z) / _WL
        ratios.append(sig / pred)
    # magnitude right across an 11x lever, and monotone in H^-1
    assert all(0.7 < r < 4.0 for r in ratios), ratios
    assert max(ratios) / min(ratios) < 2.2, ratios


def test_shipped_error_is_quadratic_in_the_residual_ray_slope(_warm):
    """``1/2 grad a^T H^-1 grad a`` is QUADRATIC in ``a``.  Doubling the
    residual amplitude must therefore roughly QUADRUPLE the shipped launch's
    exit error -- the signature that separates this mechanism from a linear
    sampling / interpolation error, which would merely double.

    (The r^4 residual used here has ``grad a`` exactly proportional to
    ``alpha``, so the scaling is in the amplitude alone.  Taken at small
    amplitudes, where the leading term IS the whole story: pushing to
    alpha = 4 the ratio falls to 2.4 as the next order arrives.)"""
    e = {}
    for al in (0.5, 1.0, 2.0):
        e[al] = _score(al, False)[0]
    r1 = e[1.0] / e[0.5]
    r2 = e[2.0] / e[1.0]
    assert 3.0 < r1 < 4.6, e
    assert 3.0 < r2 < 4.6, e
    # ... and the corrected launch does NOT follow that law
    on = {al: _score(al, True)[0] for al in (0.5, 2.0)}
    assert on[2.0] < 0.05 * e[2.0], (on, e)


def test_amplitude_stays_energy_consistent(_warm):
    """The ray-density amplitude uses the SAME augmented map's Jacobian, so
    the exit power must not move measurably -- an unbounded launch
    perturbation shows up here first, as a D1-class ghost lobe."""
    E, X, Y = _field()
    p_in = float((np.abs(E) ** 2).sum())
    a = _run(E, False)
    b = _run(E, True)
    ra = float((np.abs(a) ** 2).sum()) / p_in
    rb = float((np.abs(b) ** 2).sum()) / p_in
    assert abs(rb - ra) < 5e-3, (ra, rb)
    # and no power appears far from the beam that was not there before
    R = np.hypot(X, Y)
    far = R > 2.5 * _W
    fa = float((np.abs(a[far]) ** 2).sum()) / p_in
    fb = float((np.abs(b[far]) ** 2).sum()) / p_in
    assert fb < fa + 1e-4, (fa, fb)
