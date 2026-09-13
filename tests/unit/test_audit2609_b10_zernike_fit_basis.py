"""WP-B10 -- the opt-in DISC-ORTHOGONAL (Zernike) basis for the traced lens's
ray fits (audit 2026-09-11 sec. 15.9).

The audit's line is that "the entire fit-radius / arbiter / predictor apparatus
exists because a square Chebyshev basis couples marginal rays into defocus on a
disc".  This file is where that claim is measured rather than assumed, and it
pins the three things the measurement found:

1. **A change of basis inside one span cannot move a least-squares answer.**
   The Zernike set of total degree ``<= order`` and the tensor-Chebyshev
   total-degree set are two bases of the SAME space (``(order+1)(order+2)/2``
   terms, proved here by rank and by projection), so the same samples with the
   same weights return the same polynomial.  Niche D7 already refused the
   affine re-map of the Chebyshev domain on exactly this ground
   (``test_niche_d7_decentred_fit.py::
   test_the_basis_domain_is_affine_invariant_so_remapping_it_is_a_no_op``);
   this is that refusal generalised to a rotation of the basis inside the span,
   and it is what makes the audit's claim false as stated.
2. **Marginal rays DO couple into defocus -- identically in both bases.**  The
   coupling is a property of the weighted problem (D1's skirt hands the fit
   every sample out to the launch square's corners at 4.0 disc radii) and of
   the ORDER that has to follow it, which is what WP-A26 re-derived 10 -> 16.
   Measured here as the shift of the fitted map's ``Z(2,0)`` on the fit disc
   between the disc-only and the skirt-weighted fit of the same samples.
3. **What the disc basis DOES buy is conditioning, and only where the samples
   are a disc.**  Where the retained samples ARE the disc -- the concentric
   branch, restricted by a hard NaN mask -- the Gram becomes nearly the
   identity and the niche-C13 step-down stops firing.  Where they are not --
   the decentred branch, where D1's skirt keeps every sample out to 4.0 disc
   radii -- the advantage decays as ``(r/R)^2`` per degree and has crossed over
   by the shipped order.  That decay is what makes this opt-in, and it is
   pinned here.

The oracles are the ones niche D7 uses and they share no code with the element:
an inline exact conic raytrace (flat entrance, exact even-conic sag, vector
Snell on the gradient normal) and the closed-form Fermat exit sphere of the
``K = -n^2`` stand-in, both rebuilt here.  Nothing is imported from another
test module.

No wall clock is asserted anywhere: the cost of the basis is reported in
WP-B10_REPORT.md and what is pinned here instead is the step-down firing count
and the arbiter's engagement.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced as LT
from lumenairy import get_glass_index

_EPS = float(np.finfo(np.float64).eps)

# --------------------------------------------------------------------------
# Niche D7's Fermat singlet, rebuilt.  A flat entrance and a ``K = -n^2``
# conic exit image a collimated bundle stigmatically and exactly, so the exit
# wavefront is a sphere about the focus for ANY sub-aperture -- which is what
# makes the oracle decentre-invariant.
# --------------------------------------------------------------------------
_WL = 1.31e-6
_GLASS = 'N-BK7'
_F = 3.00e-3
_THICK = 1.5e-3
_APER = 3.40e-3
_W = 0.60e-3                  # beam 1/e^2 radius
_FRBF = 1.5                   # fit_radius_beam_factor -> disc radius 0.90 mm
_LAUNCH_R = 0.75 * _APER      # 2.55 mm, the element's own launch radius


def _n_glass() -> float:
    return float(get_glass_index(_GLASS, _WL))


def _exact_conic_exit(h):
    """``(x_out, OPL)`` at the exit VERTEX plane for a collimated ray at
    entrance height ``h``, by an exact conic trace.  No lumenairy.

    Flat entrance (normal incidence, no refraction), straight leg to the exact
    even-conic sag, vector Snell on the exact gradient normal, straight leg to
    the rear vertex plane.  ``OPL`` is referenced to the axial ray.
    """
    n = _n_glass()
    R = -(n - 1.0) * _F
    K = -n * n
    h = np.asarray(h, dtype=np.float64)
    r2 = h * h
    q = np.sqrt(1.0 - (1.0 + K) * r2 / (R * R))
    sag = r2 / (R * (1.0 + q))
    dsag = h / (R * q)
    nrm = np.sqrt(1.0 + dsag * dsag)
    n1x, n1z = dsag / nrm, -1.0 / nrm
    mu = n
    c1 = -n1z
    c2 = np.sqrt(1.0 - mu * mu * (1.0 - c1 * c1))
    dx_o = (mu * c1 - c2) * n1x
    dz_o = mu * 1.0 + (mu * c1 - c2) * n1z
    z_hit = _THICK + sag
    s = (_THICK - z_hit) / dz_o          # back to the rear vertex plane
    return h + s * dx_o, n * z_hit + s


def _exact_bfd():
    """The axis crossing, from the same inline trace -- the oracle's ``f_b``."""
    n = _n_glass()
    R = -(n - 1.0) * _F
    K = -n * n
    h = np.linspace(0.05e-3, 1.70e-3, 41)
    r2 = h * h
    q = np.sqrt(1.0 - (1.0 + K) * r2 / (R * R))
    sag = r2 / (R * (1.0 + q))
    dsag = h / (R * q)
    nrm = np.sqrt(1.0 + dsag * dsag)
    n1x, n1z = dsag / nrm, -1.0 / nrm
    mu = n
    c1 = -n1z
    c2 = np.sqrt(1.0 - mu * mu * (1.0 - c1 * c1))
    dx_o = (mu * c1 - c2) * n1x
    dz_o = mu * 1.0 + (mu * c1 - c2) * n1z
    z_axis = (_THICK + sag) - h * dz_o / dx_o
    return float(np.mean(z_axis - _THICK)), float(np.ptp(z_axis))


def _fermat_opl(x, y, f_b):
    """The closed-form exit-vertex-plane OPL of the stigmatic stand-in."""
    return f_b - np.sqrt(x * x + y * y + f_b * f_b)


# --------------------------------------------------------------------------
# The ray-fit fixture: the element's own launch lattice, the exact conic map
# on it, and D1's weighted restriction to a BEAM-CENTRED disc -- built through
# the library's own ``_decentred_fit_restriction`` so the weights under test
# are the shipped ones and not a copy of them.
# --------------------------------------------------------------------------
_DECENTRE = 1.0 * _W          # one beam radius, niche D7's own arm
_DISC_R = _FRBF * _W          # 0.90 mm


def _fit_fixture(n_side=129, order=16):
    """``(xs, x_out, opl, weights, disc, applied_order)`` on the launch
    lattice, from the exact conic trace alone."""
    xs = np.linspace(-_LAUNCH_R, _LAUNCH_R, int(n_side))
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    h = np.hypot(X, Y)
    xo_r, opl = _exact_conic_exit(h)
    with np.errstate(invalid='ignore', divide='ignore'):
        scale = np.where(h > 0.0, xo_r / np.where(h > 0.0, h, 1.0), 1.0)
    x_out = X * scale
    disc = ((X - _DECENTRE) ** 2 + Y ** 2) <= _DISC_R ** 2
    w, o = LT._decentred_fit_restriction(disc, True, 6, int(order))
    return xs, x_out, opl, w, disc, int(o)


def _both_bases(xs, vals, weights, order, disc_geom):
    """The same fit expressed in each basis."""
    ev_c = LT._Cheb2DEvaluator(xs, xs, vals, order=order, weights=weights)
    ev_z = LT._Cheb2DEvaluator(xs, xs, vals, order=order, weights=weights,
                               basis='zernike', disc=disc_geom)
    return ev_c, ev_z


def _disc_quadrature(order_q=64, n_th=64):
    """Nodes and weights of the EXACT quadrature for the unit-disc mean.

    ``s = 2 rho^2 - 1`` turns the radial integral into a Gauss-Legendre one
    (exact for a polynomial of degree ``< 2 * order_q`` in ``s``) and the
    azimuthal integral is a trapezoid on equispaced angles, which is exact for
    every harmonic below ``n_th``.  So this is an integration with no
    truncation error of its own on any polynomial this file builds.
    """
    s, ws = np.polynomial.legendre.leggauss(int(order_q))
    rho = np.sqrt(0.5 * (s + 1.0))
    th = 2.0 * np.pi * np.arange(int(n_th)) / int(n_th)
    RR, TT = np.meshgrid(rho, th, indexing='ij')
    W = (np.repeat((0.5 * ws)[:, None], int(n_th), axis=1) / int(n_th)).ravel()
    return (RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel(), W


def _zernike_spectrum(ev, disc_geom, order_out=4):
    """The fitted map's low-order Zernike content ON THE FIT DISC.

    This is the basis being used as an INSTRUMENT rather than as a design: the
    coefficients are projections of the evaluated map, so they can be taken of
    a fit built in either basis and compared.
    """
    cx, cy, R = disc_geom
    ux, uy, W = _disc_quadrature()
    f = np.asarray(ev.ev(cx + R * ux, cy + R * uy)).ravel()
    A = LT._zernike_design(ux, uy, int(order_out))
    return {nm: float(np.sum(W * A[:, j] * f))
            for j, nm in enumerate(LT._zernike_terms(int(order_out)))}


# ===========================================================================
# 1.  The basis itself: the same span, the textbook polynomials, orthonormal
#     on the disc, and a gradient that is the derivative of the value.
# ===========================================================================
@pytest.mark.parametrize('order', [0, 1, 4, 6, 12, 16, 20])
def test_the_two_bases_have_the_same_term_count(order):
    """``(order+1)(order+2)/2`` either way -- the arithmetic precondition for
    the span claim below.  A Zernike degree ``n`` shell holds ``n+1`` terms
    (``m`` from ``-n`` to ``n`` in steps of 2) and a Chebyshev total-degree
    shell holds ``n+1`` (``kx + ky == n``)."""
    zern = LT._zernike_terms(order)
    cheb = [(a, b) for a in range(order + 1) for b in range(order + 1 - a)]
    assert len(zern) == len(cheb) == (order + 1) * (order + 2) // 2
    assert len(set(zern)) == len(zern), 'a (n, m) pair is repeated'
    for n, m in zern:
        assert 0 <= n <= order and abs(m) <= n and (n - abs(m)) % 2 == 0


@pytest.mark.parametrize('order', [6, 12, 16])
def test_the_two_bases_span_the_same_space(order):
    """THE PREMISE OF THE WHOLE WORK PACKAGE, as a rank and a projection.

    On a lattice with far more rows than terms, the two design matrices have
    the same rank as their concatenation -- so neither basis reaches a
    direction the other does not -- and every Zernike column is reproduced by
    the Chebyshev columns to the conditioning floor of that solve.

    The bound is DERIVED, not fitted: the projection is a least-squares solve
    of a well-conditioned system, so its residual is bounded by
    ``eps * cond(A_cheb)`` and nothing smaller can be claimed.  Measured
    2026-09-13: 5.4e-15 / 3.9e-15 / 3.9e-15 at orders 6 / 12 / 16, against a
    bound of 1e-10 -- five decades of room, and the failure it is there to
    catch (a term dropped, a degree mismatched) is a rank deficit, not a
    tolerance.
    """
    xs = np.linspace(-1.0, 1.0, 61)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    fx, fy = X.ravel(), Y.ravel()
    Az = LT._zernike_design(fx, fy, order)
    mi = [(a, b) for a in range(order + 1) for b in range(order + 1 - a)]
    Tu = LT._cheb_vand_2d(fx, order, np)
    Tv = LT._cheb_vand_2d(fy, order, np)
    Ac = np.stack([Tu[a] * Tv[b] for a, b in mi], axis=1)
    n_terms = Ac.shape[1]
    assert np.linalg.matrix_rank(Ac) == n_terms
    assert np.linalg.matrix_rank(Az) == n_terms
    assert np.linalg.matrix_rank(np.concatenate([Ac, Az], axis=1)) == n_terms
    coef, *_ = np.linalg.lstsq(Ac, Az, rcond=None)
    rel = float(np.max(np.abs(Ac @ coef - Az))) / float(np.max(np.abs(Az)))
    assert rel < 1e-10, f'the Zernike columns left the Chebyshev span: {rel:.3e}'


def test_the_radial_recurrence_reproduces_the_closed_form_zernikes():
    """Independent oracle for the Jacobi recurrence: the textbook polynomials,
    written out in ``(rho, theta)`` and compared to what the generator emits in
    Cartesian coordinates.

    The bound is the evaluation floor of the closed forms themselves --
    ``eps * max|Z|`` times three decades -- because both sides are
    float64 evaluations of the same polynomial in different orders."""
    u = np.linspace(-1.0, 1.0, 17)
    v = np.linspace(-0.7, 0.9, 17)
    rho, th = np.hypot(u, v), np.arctan2(v, u)
    closed = {
        (0, 0): np.ones_like(rho),
        (1, 1): 2.0 * rho * np.cos(th),
        (1, -1): 2.0 * rho * np.sin(th),
        (2, 0): np.sqrt(3.0) * (2.0 * rho ** 2 - 1.0),
        (2, 2): np.sqrt(6.0) * rho ** 2 * np.cos(2 * th),
        (2, -2): np.sqrt(6.0) * rho ** 2 * np.sin(2 * th),
        (3, 1): np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.cos(th),
        (3, -3): np.sqrt(8.0) * rho ** 3 * np.sin(3 * th),
        (4, 0): np.sqrt(5.0) * (6.0 * rho ** 4 - 6.0 * rho ** 2 + 1.0),
        (4, 2): np.sqrt(10.0) * (4 * rho ** 4 - 3 * rho ** 2) * np.cos(2 * th),
        (5, -1): (np.sqrt(12.0) * (10 * rho ** 5 - 12 * rho ** 3 + 3 * rho)
                  * np.sin(th)),
        (6, 0): np.sqrt(7.0) * (20 * rho ** 6 - 30 * rho ** 4
                                + 12 * rho ** 2 - 1.0),
    }
    A = LT._zernike_design(u, v, 6)
    idx = {nm: j for j, nm in enumerate(LT._zernike_terms(6))}
    worst = 0.0
    for nm, want in closed.items():
        worst = max(worst, float(np.max(np.abs(A[:, idx[nm]] - want))))
    floor = _EPS * max(float(np.max(np.abs(w))) for w in closed.values())
    assert worst < 1e3 * floor, (
        f'the radial recurrence left the textbook Zernikes: {worst:.3e} '
        f'against an evaluation floor of {floor:.3e}')


@pytest.mark.parametrize('order', [6, 12, 20])
def test_the_basis_is_orthonormal_on_the_unit_disc(order):
    """What ``fit_basis='zernike'`` IS: the Gram of the columns under the mean
    over the unit disc is the identity.

    Integrated by the exact quadrature (see ``_disc_quadrature``), so the only
    error is float64 accumulation over ``64 * 64`` nodes: ``sqrt(4096) * eps``
    per entry at worst, and the bound below sits four decades above it.  This
    is the property every conditioning claim in this file rests on -- if it
    fails, the basis is not the one the report measured."""
    ux, uy, W = _disc_quadrature()
    A = LT._zernike_design(ux, uy, order)
    G = (A * W[:, None]).T @ A
    off = float(np.max(np.abs(G - np.eye(A.shape[1]))))
    assert off < 1e-10, f'the disc Gram is not the identity: {off:.3e}'


def test_the_gradient_is_the_derivative_of_the_value():
    """Central differences of the evaluator's own value, INSIDE and OUTSIDE
    the disc (the Newton loop evaluates the fitted map over the whole launch
    square, which reaches 4.0 disc radii on the D7 fixture).

    The bar is the central difference's own error -- ``h^2 |f'''| / 6`` plus
    ``eps |f| / h`` -- evaluated from the measured ``|f|`` at each point rather
    than assumed, so it stays honest where the polynomial is large."""
    rng = np.random.default_rng(20260913)
    order = 12
    c = rng.standard_normal(len(LT._zernike_terms(order)))
    pu = np.array([0.0, 0.3, -0.8, 0.95, 1.7, -2.4, 4.0])
    pv = np.array([0.0, -0.5, 0.2, 0.31, -1.1, 2.0, -3.5])
    h = 1e-5
    _f, fu, fv = LT._zernike_value_and_grad(c, pu, pv, order)
    for arg, got in ((0, fu), (1, fv)):
        du = np.where(np.arange(2) == arg, h, 0.0)
        hi = LT._zernike_value_and_grad(c, pu + du[0], pv + du[1], order)[0]
        lo = LT._zernike_value_and_grad(c, pu - du[0], pv - du[1], order)[0]
        fd = (hi - lo) / (2 * h)
        # the difference's own floor, per point
        floor = _EPS * np.maximum(np.abs(hi), np.abs(lo)) / h
        rel = np.abs(got - fd) / np.maximum(np.abs(fd), 1e-300)
        assert np.all(np.abs(got - fd) <= 1e-4 * np.abs(fd) + 1e3 * floor), (
            f'gradient axis {arg}: worst relative {float(np.max(rel)):.3e}')


def test_the_two_oracles_agree_with_each_other():
    """The oracles' own pin, before either is used to grade anything.

    The `K = -n^2` conic images a collimated bundle stigmatically and exactly,
    so (a) every ray crosses the axis at the same point -- which is `f_b` --
    and (b) the exit-vertex-plane OPL is the closed-form Fermat sphere about
    that focus, for any sub-aperture and therefore for any decentre.  The
    inline exact raytrace and the closed form share no arithmetic, so their
    agreement is a statement about both.

    The bound is the float64 evaluation floor of the OPL itself, `eps *
    max|OPL|`, times three decades."""
    f_b, spread = _exact_bfd()
    assert spread < 1e-9, f'the stand-in is not stigmatic: {spread * 1e9:.3f} nm'
    assert abs(f_b - _F) < 1e-9, f'back focal distance {f_b * 1e6:.4f} um'
    h = np.linspace(0.0, _LAUNCH_R, 257)
    x_out, opl = _exact_conic_exit(h)
    d = np.abs((opl - opl[0]) - _fermat_opl(x_out, 0.0, f_b))
    floor = _EPS * float(np.max(np.abs(opl)))
    assert float(np.max(d)) < 1e3 * floor, (
        f'the exact conic trace and the Fermat sphere disagree by '
        f'{float(np.max(d)):.3e} m against a float64 floor of {floor:.3e} m')


@pytest.mark.parametrize('order', [6, 16])
def test_the_fitted_opl_misses_the_oracle_by_the_same_amount_on_both_bases(
        order):
    """The accuracy claim, held out from the fit and scored on the ORACLE.

    Both bases fit the same exact-conic OPL samples under D1's skirt weights,
    and both are then evaluated at points that are NOT on the fit lattice,
    inside the beam disc, against the closed-form Fermat sphere.  What the
    element's own exit-slope estimator measures at `ray_subsample=1` is this
    quantity with a Newton inversion and a phase unwrap in front of it; here it
    is measured directly and it says the same thing.

    Measured 2026-09-13, rms over the held-out lattice: **4.0349e-08 m** at
    order 6 and **1.8857e-10 m** at the shipped 16 -- the order buying two
    decades -- with the two bases agreeing to **2.3e-12** and **1.2e-10** OF
    THAT ERROR.  The bar is 1e-3 of the error, i.e. seven decades of room at
    the shipped order, and the quantity it guards is not a tolerance but a
    claim: the two bases miss the oracle by the same amount because they fit
    the same polynomial."""
    f_b, _spread = _exact_bfd()
    xs, _x_out, opl, w, _disc, o = _fit_fixture(order=order)
    geom = (_DECENTRE, 0.0, _DISC_R)
    ev_c, ev_z = _both_bases(xs, opl, w, o, geom)
    # a held-out lattice inside the beam disc, deliberately off the fit's own
    q = np.linspace(-_DISC_R, _DISC_R, 37) * 0.97
    QX, QY = np.meshgrid(q + _DECENTRE, q, indexing='ij')
    keep = ((QX - _DECENTRE) ** 2 + QY ** 2) <= (0.97 * _DISC_R) ** 2
    h = np.hypot(QX[keep], QY[keep])
    xo_r, opl_exact = _exact_conic_exit(h)
    err = {}
    for name, ev in (('chebyshev', ev_c), ('zernike', ev_z)):
        got = np.asarray(ev.ev(QX[keep], QY[keep]))
        err[name] = float(np.sqrt(np.mean((got - opl_exact) ** 2)))
    # the oracle the samples themselves were built from is the Fermat sphere,
    # so scoring against it is scoring against the closed form
    sphere = _fermat_opl(xo_r * QX[keep] / np.where(h > 0, h, 1.0),
                         xo_r * QY[keep] / np.where(h > 0, h, 1.0), f_b)
    assert float(np.max(np.abs((opl_exact - opl_exact.min())
                               - (sphere - sphere.min())))) < 1e-12
    assert err['chebyshev'] > 0.0
    assert abs(err['chebyshev'] - err['zernike']) <= 1e-3 * err['chebyshev'], (
        f"the two bases miss the oracle by different amounts: "
        f"{err['chebyshev']:.6e} m against {err['zernike']:.6e} m")


# ===========================================================================
# 2.  THE HEADLINE: a change of basis inside one span cannot move the fit.
# ===========================================================================
@pytest.mark.parametrize('order', [6, 10, 16])
def test_changing_the_basis_does_not_change_the_fitted_polynomial(order):
    """Niche D7's affine-invariance refusal, generalised.

    The same samples, the same D1 skirt weights and the same total degree,
    expressed in the square-Chebyshev basis and in the disc-Zernike basis:
    least squares minimises the same residual over the same space, so the
    fitted FUNCTION is the same one and the audit's "the basis couples
    marginal rays into defocus" cannot be a statement about the basis.

    The bar is what float64 can promise of two different parametrisations of
    one solve: the agreement of two solutions of a system whose Gram screens
    singular is ``cond``-limited, not ``eps``-limited.  Measured 2026-09-13 on
    this fixture: **2.843e-13 / 1.187e-12 / 1.289e-12** of the peak map value
    at orders 6 / 10 / 16, against a 1e-3 bar -- nine decades of room -- while
    two degrees of ORDER move the same map by 2.344 / 1.010 / 6.213e-02 of its
    peak, i.e. by nine to eleven decades more."""
    xs, x_out, _opl, w, _disc, o = _fit_fixture(order=order)
    geom = (_DECENTRE, 0.0, _DISC_R)
    ev_c, ev_z = _both_bases(xs, x_out, w, o, geom)
    qx = np.linspace(-_LAUNCH_R, _LAUNCH_R, 51)
    QX, QY = np.meshgrid(qx, qx, indexing='ij')
    fc = np.asarray(ev_c.ev(QX, QY))
    fz = np.asarray(ev_z.ev(QX, QY))
    peak = float(np.max(np.abs(fc)))
    rel = float(np.max(np.abs(fc - fz))) / peak
    # ... and one order lower is a MATERIALLY different polynomial, so the bar
    # above is not passing on a quantity that cannot move
    ev_c1, _ = _both_bases(xs, x_out, w, max(1, o - 2), geom)
    moved = float(np.max(np.abs(np.asarray(ev_c1.ev(QX, QY)) - fc))) / peak
    assert rel < 1e-3, (
        f'the two bases returned different polynomials: {rel:.3e} of peak')
    assert moved > 1e2 * rel, (
        f'the fail-before is dead: dropping two degrees moves the map by '
        f'{moved:.3e} of peak against a cross-basis {rel:.3e}')


def test_the_arbiter_has_exactly_the_same_thing_to_arbitrate():
    """The audit expects the fit-radius arbiter to have "nothing to
    arbitrate" on a disc-orthogonal basis.  It has the same thing: its two
    candidates are two different DISCS -- different retained samples and
    different weights -- and no change of basis can equalise two different
    weighted problems.

    Scored through the library's own ``_decentred_fit_score``, the function
    niche C11 arbitrates on, on both candidate discs and in both bases.
    Measured 2026-09-13: off-centre **3.373791140779e-09** m against
    3.373791140493e-09, concentric **3.938165482888e-08** m against
    3.938165479822e-08 -- ten digits either way, while the GAP the arbiter
    ranks is 3.6e-08 m, nine decades above what the basis moves."""
    xs, _x_out, opl, w_off, disc_off, o = _fit_fixture(order=16)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    disc_c = (X ** 2 + Y ** 2) <= _DISC_R ** 2          # concentric candidate
    w_c, o_c = LT._decentred_fit_restriction(disc_c, True, 6, 16)
    wgt = LT._decentred_fit_score_weight(xs, _DECENTRE, 0.0, _W)
    g_off, g_conc = (_DECENTRE, 0.0, _DISC_R), (0.0, 0.0, _DISC_R)
    got = {}
    for basis in ('chebyshev', 'zernike'):
        kw = {} if basis == 'chebyshev' else {'basis': basis}
        got[basis] = (
            LT._decentred_fit_score(xs, opl, wgt, disc_off, w_off, o,
                                    basis_disc=g_off, **kw),
            LT._decentred_fit_score(xs, opl, wgt, disc_c, w_c, o_c,
                                    basis_disc=g_conc, **kw))
    moved = 0.0
    for i, name in enumerate(('off-centre', 'concentric')):
        a, b = got['chebyshev'][i], got['zernike'][i]
        assert a > 0.0 and np.isfinite(a)
        moved = max(moved, abs(a - b))
        assert abs(a - b) <= 1e-3 * a, (
            f'the {name} candidate scores differently on the two bases: '
            f'{a:.6e} vs {b:.6e}')
    # and the verdict -- which candidate wins -- is the same either way
    assert ((got['chebyshev'][0] < got['chebyshev'][1])
            == (got['zernike'][0] < got['zernike'][1])), got
    # ... and it is not a coin toss that the basis could flip: what the
    # arbiter RANKS is far larger than what the basis MOVES
    gap = abs(got['chebyshev'][0] - got['chebyshev'][1])
    assert gap > 1e2 * max(moved, 1e-300), (
        f'the two candidates are now separated by {gap:.3e} against a '
        f'cross-basis movement of {moved:.3e}; the arbiter would be ranking '
        f'float noise and this pin would mean nothing')


# ===========================================================================
# 3.  THE FAIL-BEFORE: marginal rays DO couple into defocus.
# ===========================================================================
@pytest.mark.parametrize('order', [6, 10])
def test_the_marginal_rays_couple_into_defocus_in_both_bases_alike(order):
    """The audit's mechanism, measured -- and measured in both bases.

    D1's weighted restriction keeps every traced sample, so the fit is handed
    the launch square out to 4.0 fit-disc radii; the same samples restricted to
    the disc alone give a different polynomial, and the difference between the
    two, read as the ``Z(2,0)`` (defocus) coefficient of the fitted map ON THE
    FIT DISC, is the coupling the audit names.

    It is real, it is large at a degree that cannot follow the skirt, and it is
    the SAME NUMBER in both bases -- which is the finding: the coupling is a
    property of the weighted problem and of the order, not of the basis.  That
    is also why WP-A26's cure was the order (10 -> 16) and not a basis.

    Measured 2026-09-13, coupling and cross-basis agreement of it, in metres of
    exit coordinate: order 6 **2.083e-10 m**, the two bases agreeing to
    1.3e-10 OF IT; order 10 **-2.317e-11 m**, agreeing to 4.7e-09.  The orders
    where the agreement degrades are the orders where the SOLVE does (5.4e-04
    at 16, 1.1e-02 at 20, both recorded in WP-B10_REPORT.md section 6 and
    neither asserted here) -- that is the conditioning finding, not a
    disagreement about the coupling, so the claim is pinned where it has
    decades of room."""
    xs, x_out, _opl, w, disc, o = _fit_fixture(order=order)
    geom = (_DECENTRE, 0.0, _DISC_R)
    shift = {}
    for basis in ('chebyshev', 'zernike'):
        kw = ({} if basis == 'chebyshev'
              else {'basis': basis, 'disc': geom})
        ev_skirt = LT._Cheb2DEvaluator(xs, xs, x_out, order=o, weights=w, **kw)
        ev_disc = LT._Cheb2DEvaluator(xs, xs, np.where(disc, x_out, np.nan),
                                      order=o, **kw)
        a_s = _zernike_spectrum(ev_skirt, geom)
        a_d = _zernike_spectrum(ev_disc, geom)
        shift[basis] = {k: a_s[k] - a_d[k] for k in a_s}
    d20 = abs(shift['chebyshev'][(2, 0)])
    # The two floors this has to clear, both DERIVED on this run rather than
    # assumed: the float64 evaluation floor of an exit coordinate, and what a
    # change of basis alone moves the same fit by.
    floor = _EPS * float(np.max(np.abs(x_out)))
    same = abs(shift['zernike'][(2, 0)] - shift['chebyshev'][(2, 0)])
    assert d20 > 1e2 * max(same, floor), (
        f'the fail-before is dead: the skirt moves the fitted map\'s defocus '
        f'by {d20:.3e} m at order {o}, against a float64 floor of '
        f'{floor:.3e} m and a cross-basis movement of {same:.3e} m')
    # ... and the disc basis does not reduce it: the coupling is the same
    # number in both, to four decades better than the bar
    assert same <= 1e-4 * d20, (
        f'the two bases disagree about the coupling: '
        f'{shift["chebyshev"][(2, 0)]:.6e} vs {shift["zernike"][(2, 0)]:.6e}')


def test_the_coupling_falls_with_the_order_and_not_with_the_basis():
    """The other half of the same statement, as a ladder: what removes the
    marginal-ray coupling is giving the fit the terms to follow the skirt,
    which is WP-A26's re-derivation of ``_DECENTRED_FIT_POLY_ORDER``.

    Measured 2026-09-13 on the Chebyshev arm, |Z(2,0)| shift in metres:
    2.083e-10 (order 6) -> 2.317e-11 (10) -> 1.934e-13 (16), i.e. the shipped
    order carries **1/1077** of the coupling the pre-D7 degree does.  The bar
    is a factor of 10, so it has two decades of room, and the fail-before is
    the order-6 arm being large rather than any number being small."""
    seen = {}
    for order in (6, 10, 16):
        xs, x_out, _opl, w, disc, o = _fit_fixture(order=order)
        geom = (_DECENTRE, 0.0, _DISC_R)
        row = []
        for basis in ('chebyshev', 'zernike'):
            kw = ({} if basis == 'chebyshev'
                  else {'basis': basis, 'disc': geom})
            ev_s = LT._Cheb2DEvaluator(xs, xs, x_out, order=o, weights=w, **kw)
            ev_d = LT._Cheb2DEvaluator(xs, xs, np.where(disc, x_out, np.nan),
                                       order=o, **kw)
            row.append(abs(_zernike_spectrum(ev_s, geom)[(2, 0)]
                           - _zernike_spectrum(ev_d, geom)[(2, 0)]))
        seen[order] = row
    assert seen[16][0] < 0.1 * seen[6][0], seen
    assert seen[16][1] < 0.1 * seen[6][1], seen
    # the two bases fall TOGETHER, asserted where the solve still resolves the
    # difference (see the sibling's docstring for why not at 16)
    for order in (6, 10):
        c, z = seen[order]
        assert abs(c - z) <= 1e-4 * max(c, z), (order, c, z)


# ===========================================================================
# 4.  WHAT THE DISC BASIS DOES BUY: conditioning, in both directions.
# ===========================================================================
def test_a_disc_shaped_fit_is_conditioned_by_the_disc_basis():
    """Where the retained samples ARE the disc -- the concentric branch, whose
    restriction is a hard NaN mask -- the Zernike Gram is nearly the identity
    and the niche-C13 screen passes it, where the square basis's Gram screens
    SINGULAR at the same order on the same samples.

    Both readings are taken from the library's own ``_gram_rcond``, i.e. the
    quantity the step-down actually tests, and the bar is the step-down's own
    threshold ``_LSTSQ_GRAM_RCOND_MIN`` = 1e-8 with the two answers on opposite
    sides of it by decades: measured 2026-09-13, **0.000e+00** on the square
    basis (its Gram does not even stay positive-definite on a disc of samples
    at this degree) against **3.326e-01** on the disc basis, i.e. the identity
    to within the lattice's own discretisation of the disc."""
    xs = np.linspace(-_LAUNCH_R, _LAUNCH_R, 129)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    h = np.hypot(X, Y)
    _xo, opl = _exact_conic_exit(h)
    disc = h <= _DISC_R
    vals = np.where(disc, opl, np.nan)
    rc = {'chebyshev': _gram_rcond_of(xs, vals, 12),
          'zernike': _gram_rcond_of(xs, vals, 12,
                                    geom=(0.0, 0.0, _DISC_R))}
    assert rc['chebyshev'] < LT._LSTSQ_GRAM_RCOND_MIN, (
        f'the square basis no longer screens singular on a disc of samples '
        f'({rc["chebyshev"]:.3e}); the comparison has lost its fail-before')
    assert rc['zernike'] > 1e3 * LT._LSTSQ_GRAM_RCOND_MIN, (
        f'the disc basis did not condition a disc-shaped fit: '
        f'{rc["zernike"]:.3e} against {rc["chebyshev"]:.3e}')


def _gram_rcond_of(xs, vals, order, weights=None, geom=None):
    """The equilibrated Gram rcond the C13 screen reads, for one fit."""
    kw = {} if geom is None else {'basis': 'zernike', 'disc': geom}
    rows = []
    orig = LT._solve_lstsq_thread_safe

    def _spy(A, b, **k):
        A64 = np.ascontiguousarray(A, dtype=np.float64)
        rows.append(LT._gram_rcond(A64.T @ A64))
        return orig(A, b, **k)

    LT._solve_lstsq_thread_safe = _spy
    try:
        LT._Cheb2DEvaluator(xs, xs, vals, order=int(order), weights=weights,
                            **kw)
    finally:
        LT._solve_lstsq_thread_safe = orig
    return min(rows)


def test_the_disc_basis_advantage_decays_with_the_order_on_a_square_of_data():
    """WHY THIS SHIPS OPT-IN, and A26-7 made quantitative.

    D1's weighted skirt hands the DECENTRED fit every launch sample, out to
    4.0 fit-disc radii on this fixture, so a disc-normalised column is
    ``(r/R)^n`` out there where a square-normalised one is bounded by 1.  Each
    added degree multiplies the skirt's share of the Gram by ``(r/R)^2 ~ 16``,
    so the disc basis's conditioning advantage decays geometrically -- about
    1.2 decades per degree -- and crosses the square basis's roughly flat
    ``1e-11`` near the shipped order 16.

    What is asserted is the DECAY, which is a geometric law with decades of
    room and not a knife edge: measured 2026-09-13 on this fixture, Zernike
    **5.930e-02** at order 6 against **3.563e-11** at order 14 -- 9.2 decades
    over eight degrees, 1.15 per degree, against the 1.20 that
    ``(R_data/R_disc)^2 = 16.1`` predicts -- while the square basis moves only
    2.047e-10 -> 1.725e-11.  The bars are three decades and one, against 9.2
    and 1.2 measured.  The crossover order itself is RECORDED in
    WP-B10_REPORT.md and asserted nowhere: it is a ratio of two near-singular
    rconds, which is exactly the shape ``docs/TESTING_STANDARDS.md`` S1 forbids
    a test to carry."""
    geom = (_DECENTRE, 0.0, _DISC_R)
    rc = {}
    for order in (6, 14):
        xs, x_out, _opl, w, _disc, o = _fit_fixture(order=order)
        assert o == order, 'the step-down capped the fixture'
        rc[order] = (_gram_rcond_of(xs, x_out, o, w, None),
                     _gram_rcond_of(xs, x_out, o, w, geom))
    # the advantage is real at low order ...
    assert rc[6][1] > 1e3 * rc[6][0], rc
    # ... and decays by decades over eight degrees, because the data is a
    # SQUARE and the basis is normalised to a disc inside it
    assert rc[6][1] > 1e3 * rc[14][1], (
        f'the (r/R)^n decay of the disc basis over a square of data is gone: '
        f'{rc[6][1]:.3e} at order 6 against {rc[14][1]:.3e} at order 14 -- '
        f're-measure the ladder in WP-B10_REPORT.md')
    # the square basis has no such decay to speak of (it is bounded by 1
    # everywhere on its own domain), which is what the crossover is made of
    assert rc[6][0] < 1e3 * rc[14][0], rc


# ===========================================================================
# 5.  The plumbing: the default is untouched, the opt-in reaches the fit, and
#     the combinations that have no meaning are refused.
# ===========================================================================
def test_the_default_basis_ships_the_state_it_always_shipped():
    """The byte-identity discipline, at the one place a new field could have
    leaked into a shipped payload: the Newton pool's fit state is the nine keys
    it has always been on the default basis, and carries the disc only on the
    opt-in one."""
    xs = np.linspace(-1.0, 1.0, 33)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    vals = np.exp(0.3 * X) * np.cos(0.7 * Y)
    ev_c = LT._Cheb2DEvaluator(xs, xs, vals, order=6)
    assert set(LT._cheb_fit_state(ev_c)) == {
        'order', 'mi', 'coeffs', 'K1', 'K2',
        'xmin', 'xmax', 'ymin', 'ymax'}
    assert ev_c.basis == 'chebyshev'
    ev_z = LT._Cheb2DEvaluator(xs, xs, vals, order=6, basis='zernike',
                               disc=(0.1, -0.2, 0.8))
    st = LT._cheb_fit_state(ev_z)
    assert set(st) == {'order', 'mi', 'coeffs', 'K1', 'K2',
                       'xmin', 'xmax', 'ymin', 'ymax', 'basis', 'disc'}
    assert st['basis'] == 'zernike' and st['disc'] == (0.1, -0.2, 0.8)
    # a worker rebuilt from that state evaluates the parent's polynomial
    # BITWISE -- the property the whole payload exists for
    back = LT._Cheb2DEvaluator.from_state(st, xp=np, backend='numpy')
    q = np.linspace(-0.9, 0.9, 11)
    QX, QY = np.meshgrid(q, q, indexing='ij')
    for a, b in zip(ev_z.ev_value_and_grad(QX, QY),
                    back.ev_value_and_grad(QX, QY)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_a_zernike_fit_without_a_disc_is_refused():
    """The disc IS the basis; inventing one would answer a different question
    than the caller asked."""
    xs = np.linspace(-1.0, 1.0, 9)
    vals = np.zeros((9, 9))
    for bad in ({}, {'disc': None}, {'disc': (0.0, 0.0, 0.0)},
                {'disc': (0.0, 0.0, -1.0)}, {'disc': (np.nan, 0.0, 1.0)},
                {'disc': (0.0, 0.0)}):
        with pytest.raises(ValueError, match='disc'):
            LT._Cheb2DEvaluator(xs, xs, vals, order=2, basis='zernike', **bad)
    with pytest.raises(ValueError, match='fit_basis'):
        LT._Cheb2DEvaluator(xs, xs, vals, order=2, basis='legendre',
                            disc=(0.0, 0.0, 1.0))


def test_the_entry_point_refuses_the_combinations_that_have_no_meaning():
    """A knob that is silently inert is the defect the ``on_fit_domain_basis``
    ledger in this module records; ``fit_basis`` is gated for every call, not
    only for the calls that build a polynomial fit."""
    pres = {'name': 'x', 'aperture_diameter': 2e-3, 'thicknesses': [1e-3],
            'surfaces': [
                {'radius': np.inf, 'glass_before': 'air',
                 'glass_after': _GLASS, 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -2e-3, 'glass_before': _GLASS,
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}
    E = np.ones((32, 32), dtype=np.complex128)
    common = dict(prescription=pres, wavelength=_WL, dx=8e-6,
                  on_undersample='silent', on_noncollimated='silent')
    with pytest.raises(ValueError, match='fit_basis'):
        la.apply_real_lens_traced(E, fit_basis='bessel', **common)
    with pytest.raises(ValueError, match="newton_fit='polynomial'"):
        la.apply_real_lens_traced(E, fit_basis='zernike',
                                  newton_fit='spline', **common)
    with pytest.raises(ValueError, match="inversion_method='newton'"):
        la.apply_real_lens_traced(E, fit_basis='zernike',
                                  inversion_method='fit', **common)
