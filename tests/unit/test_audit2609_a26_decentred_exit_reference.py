"""WP-A26 -- the DECENTRED ray fit's order was calibrated against a ray set the
tracer was falsely truncating, and WP-A1 stopped truncating it.

Finding A26 of ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
WP-A26_REPORT.md``.

WHAT IS GOING ON
----------------
``apply_real_lens_traced`` launches its ray lattice over a SQUARE of half-width
``launch_radius = 0.75 * aperture_diameter`` -- 1.5 clear-aperture radii on the
axes, 2.12 at the corners -- and pops ``aperture_diameter`` before
``surfaces_from_prescription`` so that margin is traced unvignetted and no
field energy is clipped.  On a DECENTRED beam D1's weighted restriction then
keeps every one of those samples in the least squares, down-weighted to
``_FIT_DISC_OUTSIDE_WEIGHT_REL`` of the in-disc Gram contribution, so that the
directions the fit disc leaves free stay pinned to the traced map instead of to
fit noise.  The order that fit is given, ``_DECENTRED_FIT_POLY_ORDER``, has to
be enough to FOLLOW that data -- all of it, not just the disc -- because what a
weighted least squares trades away where it cannot follow is accuracy where it
can.

Until WP-A1 it never had to.  ``_intersect_surface`` seeded its Newton branch
from the ray-SPHERE quadratic and used THAT discriminant as the miss test, so
on a conic every ray beyond ``h = |R|`` came back
``alive=False, error_code=RAY_MISSED_SURFACE, t=0`` although it genuinely hits
the surface, and the fit's data stopped there.  On the fixture below that was
``|R| = (n-1) f = 1.5106 mm`` -- 0.59 launch radii, and by coincidence just
inside the 1.70 mm clear-aperture radius.  A1's R4 replaced the seed and the
miss test with the exact conic quadratic (correctly -- see
``test_a1s_resurrected_conic_rays_hit_the_conic``, which re-measures the
resurrected rays against an inline exact trace) and the same degree-10 fit was
handed data out to the launch square's corner at 3.6062 mm, 2.4x further.

The CONCENTRIC branch never noticed: it restricts by a hard NaN mask at the fit
disc, so its sample set is the disc either way and the on-axis figure below
does not move by one bit.

MEASURED ON THIS FIXTURE (this box, 2026-09-13; the pre-A1 library extracted
read-only with ``git archive 0067d63b lumenairy`` and the CURRENT test driving
it, so only the library varies):

    tree                            fit rows   data |h|max   exit-slope error
                                                             vs the analytic
                                                             oracle, 0.5w / 1.0w
    0067d63b (the commit before A1)  111 525     1.5106 mm    2.162 / 1.958 urad
    f602b72c (WP-A1) .. 6345d99d     405 769     3.6062 mm   44.457 / 31.556
    ... at the re-derived order 16   405 769     3.6062 mm    2.371 / 1.683

against **41.089 urad on axis, identical to the last digit on all three**.
The order ladder at the enlarged domain, same fixture, same process:

    order      10       12       14       16       18       20       24
    0.5 w   44.457   10.841    3.718    2.371    1.044    0.321    0.071 urad
    1.0 w   31.556   11.829    5.419    1.683    0.655    0.301    0.057

Monotone, with no fold and no ghost at any of them on D1's own adversarial
fixture (off-beam amplitude 1.76e-04 of peak and 0 sign changes in
``d(x_out)/dx`` at every order, measured).  16 is the LOWEST order on that
ladder that reaches the pre-truncation scale on both decentres -- 2.371 urad
against 2.162 at 0.5 w and 1.683 against 1.958 at 1.0 w -- for 1.56x of the
order-10 wall clock on the off-centre branch, which is the whole of the
cost/accuracy argument in ``_DECENTRED_FIT_POLY_ORDER``'s own note.

THE ORACLE
----------
A flat-entrance / ``K = -n^2`` conic-exit singlet images a collimated bundle in
air stigmatically, exactly, to all orders, so every ray leaves the exit VERTEX
plane pointing at one focus and Fermat fixes the exit-plane path exactly::

    OPL(x, y) - OPL(0, 0) = f_b - sqrt(x^2 + y^2 + f_b^2)

That closed form is the SAME for every decentre -- it knows nothing about where
the beam sits -- which is what lets it score a decentre-dependent defect at
all.  ``test_the_oracle_is_exactly_stigmatic`` pins ``f_b`` and the stigmatism
with an inline exact conic raytrace, so the oracle cannot drift either.  Shared
with ``tests/unit/test_niche_d7_decentred_fit.py``, which owns the D7
acceptance decision; this file owns the A26 mechanism.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced as _lt
from lumenairy import get_glass_index
from lumenairy.elements._lens_traced import _Cheb2DEvaluator

# Slow lane: the exit-wavefront arms run at ``ray_subsample=1`` on a 512 grid,
# i.e. ~406k rays per call and three calls per parametrisation.
pytestmark = pytest.mark.slow

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_GLASS = 'N-BK7'
_F = 3.00e-3
_THICK = 1.5e-3
_APER = 3.40e-3             # clear aperture: semi 1.700 mm
_W = 0.60e-3                # collimated beam 1/e^2 radius
_N, _DX = 512, 8.0e-6
_FRBF = 1.5                 # ray-fit disc: 0.900 mm about the beam
_LAUNCH_R = 0.75 * _APER    # 2.550 mm; square corners at 3.6062 mm

#: The order the fit took before A26 re-derived it -- the fail-before, driven
#: through the public ``decentred_fit_poly_order`` knob rather than by patching
#: a module global.
_PRE_A26_ORDER = 10

_MIN_FREE_GIB = 3.0


def _ram_guard():
    try:
        import psutil
    except ImportError:
        return
    free = psutil.virtual_memory().available / (1024 ** 3)
    if free < _MIN_FREE_GIB:
        pytest.skip(f"needs ~{_MIN_FREE_GIB} GiB available, saw {free:.1f}")


def _n_glass() -> float:
    return float(get_glass_index(_GLASS, _WL))


def _surface(radius, glass_before, glass_after, conic):
    return {'radius': float(radius), 'glass_before': glass_before,
            'glass_after': glass_after, 'conic': float(conic),
            'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _prescription(aperture=_APER):
    """The Fermat singlet: flat entrance, ``K = -n^2`` conic exit."""
    n = _n_glass()
    p = {'name': 'A26 Fermat singlet', 'thicknesses': [_THICK],
         'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0),
                      _surface(-(n - 1.0) * _F, _GLASS, 'air', -n * n)]}
    if aperture is not None:
        p['aperture_diameter'] = float(aperture)
    return p


def _gauss(cx=0.0, cy=0.0, n=_N, dx=_DX, w=_W):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2) / (w * w)
                  ).astype(np.complex128)


def _apply(cx=0.0, cy=0.0, rs=8, aperture=_APER, **kw):
    """One element call, phase-only output (``k0 * OPL`` exactly)."""
    opts = dict(prescription=_prescription(aperture), wavelength=_WL, dx=_DX,
                ray_subsample=rs, n_workers=1, fit_radius_beam_factor=_FRBF,
                beam_centre=(cx, cy), preserve_input_phase=False,
                amplitude_model='screen', newton_amp_mask_rel=0.0,
                on_undersample='silent', on_noncollimated='silent',
                on_aperture_beam='silent')
    opts.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(_gauss(cx, cy), **opts))


# ---------------------------------------------------------------------------
# The inline exact conic trace -- the oracle's own foundation.  No lumenairy.
# ---------------------------------------------------------------------------
def _inline_conic_exit(h):
    """``(x_out, opl, z_axis)`` at the exit VERTEX plane for a collimated ray
    entering at height ``h``.

    Flat entrance (normal incidence, no refraction), straight leg through the
    glass to the exact even-conic sag, vector Snell on the exact gradient
    normal, then the straight exit leg.  ``z_axis`` is where that leg crosses
    the axis, measured from the REAR vertex.
    """
    n = _n_glass()
    R = -(n - 1.0) * _F
    K = -n * n
    h = np.atleast_1d(np.asarray(h, dtype=np.float64))
    r2 = h * h
    q = np.sqrt(1.0 - (1.0 + K) * r2 / (R * R))
    sag = r2 / (R * (1.0 + q))
    dsag = h / (R * q)
    # F(x, z) = z - THICK - sag(x); the normal taken AGAINST the incident ray
    nrm = np.sqrt(1.0 + dsag * dsag)
    n1x, n1z = dsag / nrm, -1.0 / nrm
    mu = n
    c1 = -n1z
    c2 = np.sqrt(1.0 - mu * mu * (1.0 - c1 * c1))
    dx_o = (mu * c1 - c2) * n1x
    dz_o = mu + (mu * c1 - c2) * n1z
    t_v = -sag / dz_o                     # sag -> the exit vertex plane
    x_v = h + dx_o * t_v
    opl_v = n * (_THICK + sag) + t_v      # from the ENTRANCE plane
    with np.errstate(divide='ignore', invalid='ignore'):
        z_axis = sag - h * dz_o / dx_o    # from the REAR vertex
    return x_v, opl_v, z_axis


def _oracle_opl(X, Y, f_b):
    """Exit-vertex-plane OPL referenced to the axis.  Exact, closed form."""
    return f_b - np.sqrt(X * X + Y * Y + f_b * f_b)


def _exit_slope_rms(E, cx, f_b, cy=0.0):
    """Aliasing-free rms exit-slope error over the beam CORE [rad].

    ``arg(E) - k0 * OPL_exact`` differenced between NEIGHBOURING pixels with a
    2*pi wrap (per-pixel steps are ~1e-2 rad, four orders inside pi, so the
    wrap is unambiguous), then piston and tilt removed -- pointing is not an
    aberration.  Restricted to ``r <= w`` about the beam: outside the ray-FIT
    disc the low-order fit legitimately extrapolates.
    """
    x = (np.arange(_N) - _N // 2) * _DX
    X, Y = np.meshgrid(x, x, indexing='xy')
    psi = np.angle(E * np.exp(-1j * _K0 * _oracle_opl(X, Y, f_b)))
    amp = np.abs(E)
    d = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1]))) / (_K0 * _DX)
    Xm = 0.5 * (X[:, 1:] + X[:, :-1])
    Ym = 0.5 * (Y[:, 1:] + Y[:, :-1])
    thr = 1e-3 * amp.max()
    keep = ((amp[:, 1:] > thr) & (amp[:, :-1] > thr)
            & (np.hypot(Xm - cx, Ym - cy) <= _W))
    assert keep.sum() > 500, 'the illuminated core vanished'
    wt = (0.5 * (amp[:, 1:] + amp[:, :-1])) ** 2
    B = np.stack([np.ones(int(keep.sum())), Xm[keep] - cx, Ym[keep] - cy],
                 axis=1)
    q, ww = d[keep], wt[keep]
    Bw = B * ww[:, None]
    res = q - B @ np.linalg.solve(Bw.T @ B, Bw.T @ q)
    return float(np.sqrt((ww * res ** 2).sum() / ww.sum()))


def _estimator_floor(cx, f_b):
    """What ``_exit_slope_rms`` reads on a field that IS the oracle.

    The derived floor under every bar below: a field whose exit-slope error is
    zero by construction, carried through the same wrap, the same core mask and
    the same weighted piston+tilt removal.  Anything reported at this level is
    the estimator's own arithmetic, not the element's.
    """
    x = (np.arange(_N) - _N // 2) * _DX
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = _gauss(cx, 0.0) * np.exp(1j * _K0 * _oracle_opl(X, Y, f_b))
    return _exit_slope_rms(E, cx, f_b)


def _f_b():
    """Back focal distance from the rear vertex, by the inline trace."""
    return float(_inline_conic_exit(np.linspace(0.05e-3, 1.70e-3, 41))[2].mean())


def _fit_orders(**kw):
    """The orders of the LAST THREE ``_Cheb2DEvaluator`` builds of one call.

    Those three are the fits the Newton inversion is handed (``x_out``,
    ``y_out``, ``opl``); niche C11's arbiter builds trial OPL fits ahead of
    them.  Returned with each build's weighted/unweighted flag.
    """
    seen = []
    orig = _Cheb2DEvaluator.__init__

    def spy(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
        orig(self, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        seen.append((int(order), weights is not None))

    _Cheb2DEvaluator.__init__ = spy
    try:
        _apply(**kw)
    finally:
        _Cheb2DEvaluator.__init__ = orig
    return seen[-3:]


# ===========================================================================
# 1.  The oracle's own pin, and WP-A1's half of the mechanism.
# ===========================================================================
def test_the_oracle_is_exactly_stigmatic():
    """Every collimated ray of the ``K = -n^2`` conic crosses the axis at the
    SAME point, so the exit wavefront is exactly the sphere ``_oracle_opl``
    assumes -- for any sub-aperture, hence for any decentre."""
    z = _inline_conic_exit(np.linspace(0.05e-3, 1.70e-3, 41))[2]
    assert np.ptp(z) < 1e-9, (
        f"the stand-in stopped being stigmatic: axis crossings span "
        f"{np.ptp(z) * 1e9:.3f} nm")
    assert abs(float(z.mean()) - _F) < 1e-9, (
        f"back focal distance {float(z.mean()) * 1e6:.4f} um vs f = "
        f"{_F * 1e6:.4f} um")


def test_a1s_resurrected_conic_rays_hit_the_conic():
    """The rays WP-A1's R4 brought back are RIGHT, so the cure for A26 is not
    to restore the miss test that killed them.

    ``|R| = (n-1) f = 1.5106 mm`` here and the launch square runs to
    3.6062 mm, so the whole band ``|R| < h <= 3.6062 mm`` used to come back
    ``alive=False, error_code=RAY_MISSED_SURFACE, t=0``.  It is scored against
    the inline exact conic trace above, which shares no code with the library.
    """
    from lumenairy import raytrace as rt

    h = np.array([0.30, 0.90, 1.50, 1.5106, 1.55, 1.70, 2.00, 2.55, 3.00,
                  3.6062]) * 1e-3
    pres = _prescription(aperture=None)      # unvignetted, as the element does
    rays = rt._make_bundle(x=h.copy(), y=np.zeros_like(h), L=np.zeros_like(h),
                           M=np.zeros_like(h), wavelength=_WL)
    res = rt.trace(rays, rt.surfaces_from_prescription(pres), _WL,
                   output_filter='last')
    fin = res.at_exit_vertex(1.0)
    assert np.asarray(fin.alive).all(), (
        f"a conic marginal ray is being killed again: alive="
        f"{np.asarray(fin.alive)} at h = {h * 1e3} mm")
    x_o, opl_o, _z = _inline_conic_exit(h)
    d_x = float(np.max(np.abs(np.asarray(fin.x) - x_o)))
    d_o = float(np.max(np.abs(np.asarray(fin.opd) - opl_o)))
    # DERIVED bar: these are two float64 evaluations of the same closed form,
    # so the floor is ULP-scale on the quantities themselves -- eps*max|x_out| =
    # 4.59e-19 m and eps*max|OPL| = 4.98e-19 m on this band.  MEASURED
    # 2026-09-13: 2.17e-19 m and 6.51e-19 m, i.e. AT that floor (0.5 and 1.3 of
    # it).  Bar 1e-15 m sits three decades over the measurement and twelve
    # decades under the 8.10e-4 m the exit coordinate moves over the resurrected
    # band alone, so it cannot be passed by an element that is merely close.
    assert d_x < 1e-15, f"exit coordinate departs by {d_x:.3e} m"
    assert d_o < 1e-15, f"exit OPL departs by {d_o:.3e} m"


# ===========================================================================
# 2.  THE MECHANISM -- deterministic, no solver in it.
# ===========================================================================
def test_the_decentred_fit_takes_the_re_derived_order():
    """The applied decentred fit runs at ``_DECENTRED_FIT_POLY_ORDER``, and
    that order is the one A26 re-derived (>= 16), not D7's original 10.

    The concentric branch is asserted in the same call pair, because the whole
    reason the on-axis figure never moved is that it never leaves
    ``newton_poly_order``."""
    _ram_guard()
    assert _lt._DECENTRED_FIT_POLY_ORDER >= 16, (
        f"_DECENTRED_FIT_POLY_ORDER is {_lt._DECENTRED_FIT_POLY_ORDER}; at 10 "
        f"the decentred exit wavefront reads 44.5 / 31.6 urad against the "
        f"analytic oracle (see the module docstring's ladder)")
    off = _fit_orders(cx=_W, rs=2)
    assert off and all(o == _lt._DECENTRED_FIT_POLY_ORDER and w
                       for o, w in off), off
    on = _fit_orders(cx=0.0, rs=2)
    assert on and all(o == 6 and not w for o, w in on), on


def test_the_fit_is_handed_data_out_to_the_launch_square_corner():
    """A1's half of the mechanism, at the FIT rather than at the tracer: every
    launch node of the square carries a live ray, so the decentred weighted fit
    sees data out to 2.12 clear-aperture radii.

    Pure counting -- no solver, no build to depend on.  Before A1 the same
    census stopped at ``|R| = 1.5106 mm`` with 111 525 of 405 769 nodes alive.
    """
    _ram_guard()
    from lumenairy import raytrace as rt

    n_launch = 2 * int(_LAUNCH_R / (_DX * 2)) + 1
    xs = np.linspace(-_LAUNCH_R, _LAUNCH_R, n_launch)
    Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
    pres = _prescription(aperture=None)
    rays = rt._make_bundle(x=Xg.ravel().copy(), y=Yg.ravel().copy(),
                           L=np.zeros(Xg.size), M=np.zeros(Xg.size),
                           wavelength=_WL)
    res = rt.trace(rays, rt.surfaces_from_prescription(pres), _WL,
                   output_filter='last')
    alive = np.asarray(res.image_rays.alive)
    r = np.hypot(Xg.ravel(), Yg.ravel())
    frac = float(alive.sum()) / alive.size
    assert frac > 0.999, (
        f"only {frac:.4f} of the launch square survives the conic -- the "
        f"false sphere-discriminant miss test is back (it left {111525 / 405769:.4f})")
    corner = _LAUNCH_R * np.sqrt(2.0)
    assert float(r[alive].max()) > 0.999 * corner, (
        f"the live data stops at {float(r[alive].max()) * 1e3:.4f} mm, short "
        f"of the launch square's corner at {corner * 1e3:.4f} mm")


# ===========================================================================
# 3.  The consequence, against the analytic decentre-invariant oracle.
# ===========================================================================
#: Ceiling on ``slope_decentred / slope_on_axis``.
#:
#: DERIVED, not chosen.  The two populations this bar separates were measured
#: on this fixture 2026-09-13, in one process, with only the decentred fit
#: order varying:
#:
#:     0.5 w   order 16  2.371 urad -> 0.0577     order 10  44.457 urad -> 1.0820
#:     1.0 w   order 16  1.683 urad -> 0.0410     order 10  31.556 urad -> 0.7680
#:
#: (the on-axis figure, 41.089 urad, is the same to the digit on every arm --
#: it is the concentric order-6 fit, which neither D7 nor A26 touches -- so the
#: ratio removes the box and removes nothing real.)  The worst shipped reading
#: is 0.0577 and the smallest defect reading is 0.7680; the geometric mean of
#: those is 0.2105, i.e. **3.65x clear of each side**.  That is the whole 13x
#: gap the mechanism offers, split evenly -- more than that cannot be had, since
#: the two populations are what they are.
#:
#: The D7 ACCEPTANCE gate is looser (0.25) and belongs to
#: ``test_niche_d7_decentred_fit.py``; this bar brackets the A26 defect, and the
#: shipped arm clears the D7 gate by 4.3x and 6.1x as well.
_SLOPE_RATIO_MAX = 0.21

#: How far over the estimator's own floor a reading must sit to count as a
#: measurement of the element.  The floor is computed at runtime by
#: ``_estimator_floor``; MEASURED 2026-09-13 it is ~1e-12 urad against readings
#: of 1.7-44 urad, i.e. twelve decades down, so 1e3 is not a bar anyone can
#: trip by accident.
_FLOOR_MARGIN = 1.0e3


@pytest.mark.parametrize('frac', [0.5, 1.0])
def test_the_decentred_exit_wavefront_returns_to_the_analytic_oracle(frac):
    """The exit slope of a decentred beam against the closed-form Fermat
    sphere, with the defect arm measured in the same process.

    The oracle is decentre-INVARIANT, so a correct element reads the same off
    axis as on it.  Two-sided: above the estimator's own derived floor (else
    the test is scoring arithmetic) and under ``_SLOPE_RATIO_MAX``.  The
    fail-before is D7's own public knob pinned back to the order it shipped
    with, not a claim quoted from a changelog.
    """
    _ram_guard()
    f_b = _f_b()
    cx = frac * _W
    floor = _estimator_floor(cx, f_b)
    on_axis = _exit_slope_rms(_apply(0.0, rs=1), 0.0, f_b)
    fixed = _exit_slope_rms(_apply(cx, rs=1), cx, f_b)
    defect = _exit_slope_rms(
        _apply(cx, rs=1, decentred_fit_poly_order=_PRE_A26_ORDER), cx, f_b)

    assert on_axis > _FLOOR_MARGIN * floor, (
        f"the estimator is reading its own floor ({floor * 1e6:.4e} urad "
        f"against an on-axis {on_axis * 1e6:.3f} urad) -- it is not scoring "
        f"the element")
    assert on_axis < 1.0e-4, (
        f"the on-axis path regressed: {on_axis * 1e6:.3f} urad")
    assert fixed > _FLOOR_MARGIN * floor, (
        f"the decentred reading {fixed * 1e6:.4e} urad is at the estimator "
        f"floor {floor * 1e6:.4e} urad")
    assert fixed <= _SLOPE_RATIO_MAX * on_axis, (
        f"the decentred exit wavefront left the analytic oracle: "
        f"{fixed * 1e6:.3f} urad against an on-axis {on_axis * 1e6:.3f} urad "
        f"(ratio {fixed / on_axis:.4f}, bar {_SLOPE_RATIO_MAX})")
    # ... and the order is what does it
    assert defect > _SLOPE_RATIO_MAX * on_axis, (
        f"the fail-before stopped failing: at the pre-A26 order "
        f"{_PRE_A26_ORDER} the decentred exit slope reads "
        f"{defect * 1e6:.3f} urad (ratio {defect / on_axis:.4f}), inside the "
        f"bar this test asserts -- either the traced ray set is truncated "
        f"again or the fixture no longer reaches the enlarged domain")
    # ... by more than the bars alone require.  MEASURED 2026-09-13: 18.75x at
    # 0.5 w and 18.75x at 1.0 w, so 4.0 keeps 4.7x on both arms.
    assert defect > 4.0 * fixed, (defect, fixed)


# ===========================================================================
# 4.  What the re-derived order must NOT touch.
# ===========================================================================
@pytest.mark.parametrize('kw,why', [
    (dict(cx=0.0), 'the concentric branch keeps newton_poly_order exactly'),
    (dict(cx=0.0, carrier=60e-3), 'a concentric disc stays concentric under a '
                                  'carrier'),
])
def test_the_concentric_path_is_untouched(kw, why):
    """Byte identity between the shipped order and the pre-A26 one on the
    paths the raise does not reach.  ``array_equal``, not a tolerance: a path
    that is unaffected BY CONSTRUCTION cannot become "nearly unaffected"."""
    _ram_guard()
    a = _apply(**kw)
    b = _apply(decentred_fit_poly_order=_PRE_A26_ORDER, **kw)
    assert np.array_equal(a, b), (
        f"{why}, yet the decentred fit order moved it by "
        f"{float(np.max(np.abs(a - b))):.3e}")


def test_a_caller_asking_for_more_still_gets_more():
    """The raise only ever RAISES: ``newton_poly_order`` above the constant
    wins, exactly as D7 shipped it."""
    _ram_guard()
    hi = _lt._DECENTRED_FIT_POLY_ORDER + 4
    off = _fit_orders(cx=_W, rs=2, newton_poly_order=hi)
    assert off and all(o == hi for o, _w in off), off


def test_the_step_down_still_protects_an_undersampled_disc():
    """The re-derived order needs 3 samples per basis term like every other:
    on a disc that cannot hold them the order steps back down rather than
    handing the solver an under-determined normal matrix, and the returned
    field stays finite."""
    _ram_guard()
    off = _fit_orders(cx=_W, rs=16)
    orders = [o for o, _w in off]
    assert orders, 'no polynomial fit was built'
    assert max(orders) < _lt._DECENTRED_FIT_POLY_ORDER, (
        f"order {max(orders)} survived a disc that cannot constrain "
        f"{(max(orders) + 1) * (max(orders) + 2) // 2} terms")
    assert min(orders) >= 6, orders
    assert np.isfinite(_apply(cx=_W, rs=16)).all()
