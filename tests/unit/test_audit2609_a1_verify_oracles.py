"""VERIFY-A1 -- two independent-oracle tests that close gaps in WP-A1's own
regression set (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, §15.1 and R3).

Gap 1 -- the exit-vertex helper's oracle is not independent.
    ``test_audit2609_a1_exit_vertex.py::
    test_at_exit_vertex_is_the_analytic_vertex_plane_opl`` compares
    ``at_exit_vertex()`` against ``img.opd + n * (-img.z / img.N)``, i.e. the
    helper's own formula re-evaluated on the helper's own inputs.  That
    discriminates a sign error, an ``abs()``, a wrong alive mask and a wrong
    ``n_exit`` -- but it cannot see the tracer leaving the rays anywhere other
    than ``z = sag``, because both sides read ``img.z``.  The test below
    re-derives the whole ray path in 60-digit :mod:`decimal` from the CLOSED
    FORM of the conic (exact quadratic, exact gradient, exact vector Snell)
    and never reads a single traced quantity.

    It is also the only test in the set with ``glass_after != 'air'`` on the
    last surface, so it is the only one that exercises ``resolve_exit_index``
    returning something other than 1.0 on a refracting exit.

Gap 2 -- the aspheric S3 scaling exponent is asserted, never measured.
    WP-A1 chose ``dS3 = dS1 * (y_c/y_m)**2`` (Welford §8.5) over the audit
    snippet's ``dS3 = dS1 * (y_c/y_m)``, correctly, but no test distinguishes
    them: every R3 test in the set is a STOP-AT-SURFACE-0 fixture where the
    chief height at the aspheric surface is ~0, so both exponents give
    essentially the same number.  The test below displaces the stop so that
    ``y_c/y_m`` is far from 0 and 1, and measures S3 from real rays.
"""
from __future__ import annotations

from decimal import Decimal as D, getcontext

import numpy as np
import pytest

from lumenairy.glass import get_glass_index
from lumenairy.raytrace import Surface, seidel_coefficients, trace
from lumenairy.raytrace.surface import RayBundle

WL = 1.31e-6

# 60 significant digits: the quantities compared are ~1e-2 m, so the oracle's
# own arithmetic floor is ~1e-62 m -- 44 decades below the float64 result it
# is judging (whose own ULP at 1e-2 m is ~2e-18 m).
getcontext().prec = 60


# ---------------------------------------------------------------------------
# Gap 1: a Decimal closed-form oracle for the exit-vertex transfer
# ---------------------------------------------------------------------------

def _intersect_conic_D(P, d, R, k):
    """Near root of the exact implicit conic, in Decimal.

    ``F = c(x^2+y^2) - 2z + (1+k) c z^2 = 0`` substituted with ``X = P + t d``.
    Stable Spencer & Murty form; at 60 digits the choice of form is cosmetic.
    """
    c = D(1) / D(float(R))
    k1 = D(1) + D(float(k))
    x, y, z = P
    L, M, N = d
    a = c * (L * L + M * M) + k1 * c * N * N
    b = D(2) * (c * (x * L + y * M + k1 * z * N) - N)
    e = c * (x * x + y * y) + k1 * c * z * z - D(2) * z
    disc = b * b - D(4) * a * e
    assert disc >= 0, 'oracle fixture ray misses the conic'
    sq = disc.sqrt()
    q = D('-0.5') * (b + (D(1) if b >= 0 else D(-1)) * sq)
    t1 = e / q
    t2 = q / a if a != 0 else t1
    return t1 if abs(t1) <= abs(t2) else t2


def _normal_D(P, R, k):
    """Unit normal = normalised grad F of the same implicit conic."""
    c = D(1) / D(float(R))
    k1 = D(1) + D(float(k))
    x, y, z = P
    gx, gy, gz = D(2) * c * x, D(2) * c * y, D(-2) + D(2) * k1 * c * z
    n = (gx * gx + gy * gy + gz * gz).sqrt()
    return (gx / n, gy / n, gz / n)


def _snell_D(d, nrm, n1, n2):
    """Exact vector Snell, normal oriented against the incident ray."""
    dot = sum(a * b for a, b in zip(d, nrm))
    if dot > 0:
        nrm = tuple(-v for v in nrm)
        dot = -dot
    cos_i = -dot
    eta = D(float(n1)) / D(float(n2))
    disc = D(1) - eta * eta * (D(1) - cos_i * cos_i)
    assert disc >= 0, 'oracle fixture ray TIRs'
    cos_t = disc.sqrt()
    return tuple(eta * di + (eta * cos_i - cos_t) * ni
                 for di, ni in zip(d, nrm))


def _oracle_vertex_plane(y0, R1, k1, R2, k2, thick, n1, n2, n3):
    """OPL / x / y at the LAST surface's vertex plane, in Decimal.

    Launches ``(0, y0, 0)`` along ``+z`` with ``opd = 0`` (the library's
    default ``opd_seed='plane'``), refracts through both conics, then applies
    the SIGNED straight-line leg to ``z = 0`` in the last surface's frame.
    Reads nothing from the tracer; the only library inputs are the three
    refractive-index VALUES, which are an input to the geometry question
    rather than part of it.
    """
    P = (D(0), D(float(y0)), D(0))
    d = (D(0), D(0), D(1))
    opl = D(0)

    t = _intersect_conic_D(P, d, R1, k1)
    P = tuple(p + di * t for p, di in zip(P, d))
    opl += D(float(n1)) * t
    d = _snell_D(d, _normal_D(P, R1, k1), n1, n2)

    t = (D(float(thick)) - P[2]) / d[2]
    P = tuple(p + di * t for p, di in zip(P, d))
    opl += D(float(n2)) * t
    P = (P[0], P[1], P[2] - D(float(thick)))          # into surface 2's frame

    t = _intersect_conic_D(P, d, R2, k2)
    P = tuple(p + di * t for p, di in zip(P, d))
    opl += D(float(n2)) * t
    d = _snell_D(d, _normal_D(P, R2, k2), n2, n3)

    t = (D(0) - P[2]) / d[2]                          # SIGNED leg to z = 0
    P = tuple(p + di * t for p, di in zip(P, d))
    opl += D(float(n3)) * t
    return float(opl), float(P[0]), float(P[1])


@pytest.mark.parametrize('R2, k2, g3, tag', [
    (+40e-3, 0.0, 'N-SF11', 'concave rear (sag > 0, t < 0), exit into glass'),
    (-35e-3, -2.5, 'air', 'convex hyperbolic rear (sag < 0, t > 0)'),
    (+60e-3, +2.0, 'N-BK7', 'oblate-ellipsoid rear, exit into glass'),
])
def test_at_exit_vertex_matches_a_60_digit_closed_form_oracle(R2, k2, g3, tag):
    """``at_exit_vertex()`` == a Decimal trace built from the conic closed form.

    ORACLE: :func:`_oracle_vertex_plane` -- exact conic quadratic, exact
    gradient normal, exact vector Snell, 60 significant digits, reading NO
    traced quantity.  This is the independence the in-set test lacks: it
    would fail if ``trace`` left the rays anywhere other than ``z = sag``,
    if the transfer used ``abs(t)``, or if ``n_exit`` were taken from the
    wrong medium.

    Two of the three fixtures have ``glass_after != 'air'`` on the last
    surface, so a wrong ``resolve_exit_index`` shows up directly: using 1.0
    instead of ``n(N-SF11) = 1.7480`` would move the OPL by
    ``(n-1) * |sag| / |N| ~ 0.75 * 1.37e-3 = 1.0e-3 m``, 15 decades above
    the bar below.

    BAR: 1e-16 m on the OPL, 1e-16 m on x and y.  Derivation -- the oracle's
    own arithmetic floor is ~1e-62 m (60 digits on a ~1e-2 m quantity), so
    the entire gap is float64 rounding in the library.  A ~1e-2 m OPL
    accumulated over five legs has a rounding envelope of a few ULP,
    i.e. ~1e-17 m; the audit's own 60-digit OPL probe
    (repro/RAYTRACE/p2b_opl.py) measured 1.39e-17 m worst case over five
    systems and re-measures at 1.388e-17 m on this commit.  Measured here:
    max |d opd| = 4.3e-18 m, max |d y| = 1.7e-18 m -- 1.4 decades below the
    bar, and the bar is 13 decades below the transferred term itself
    (n_exit * sag / N, asserted to exceed 1e-4 m below).
    """
    R1, k1, thick = 30e-3, -0.7, 6e-3
    g1, g2 = 'air', 'N-BK7'
    n1 = float(get_glass_index(g1, WL))
    n2 = float(get_glass_index(g2, WL))
    n3 = float(get_glass_index(g3, WL))
    surfs = [Surface(radius=R1, conic=k1, glass_before=g1, glass_after=g2,
                     thickness=thick, semi_diameter=14e-3, is_stop=True),
             Surface(radius=R2, conic=k2, glass_before=g2, glass_after=g3,
                     thickness=0.0, semi_diameter=14e-3)]
    ys = np.array([1e-3, 4e-3, 8e-3, 11e-3])
    n = ys.size
    b = RayBundle(x=np.zeros(n), y=ys.copy(), z=np.zeros(n), L=np.zeros(n),
                  M=np.zeros(n), N=np.ones(n), wavelength=WL,
                  alive=np.ones(n, bool), opd=np.zeros(n),
                  error_code=np.zeros(n, np.uint8))
    res = trace(b, surfs, WL)
    img = res.image_rays
    ex = res.at_exit_vertex()

    assert img.alive.all(), f'{tag}: fixture vignettes -- pick a wider surface'
    # The fixture must actually exercise the class: a CURVED exit whose
    # transferred OPL term is macroscopic.
    term = np.abs(n3 * img.z / img.N)
    assert term.max() > 1e-4, (
        f'{tag}: transferred term only {term.max():.2e} m -- the whole bug '
        f'class is invisible on a flat exit, which is why the audit found it')

    d_opd = d_x = d_y = 0.0
    for i, y0 in enumerate(ys):
        o_opd, o_x, o_y = _oracle_vertex_plane(
            y0, R1, k1, R2, k2, thick, n1, n2, n3)
        d_opd = max(d_opd, abs(o_opd - ex.opd[i]))
        d_x = max(d_x, abs(o_x - ex.x[i]))
        d_y = max(d_y, abs(o_y - ex.y[i]))
    assert d_opd < 1e-16, f'{tag}: max |d opd| = {d_opd:.3e} m'
    assert d_x < 1e-16, f'{tag}: max |d x| = {d_x:.3e} m'
    assert d_y < 1e-16, f'{tag}: max |d y| = {d_y:.3e} m'
    assert np.all(ex.z == 0.0)


# ---------------------------------------------------------------------------
# Gap 2: the aspheric S3 exponent, measured with a DISPLACED stop
# ---------------------------------------------------------------------------

def _displaced_stop_system(stop_gap, A4, conic, r_stop=6e-3):
    """flat STOP -- air gap -- aspherized N-BK7 plano-convex -- image plane.

    The gap puts the chief ray well off axis at the aspheric surface, so
    ``y_c / y_m`` there is far from both 0 and 1 and the S2 / S3 / S5
    exponents become distinguishable.
    """
    from lumenairy.raytrace import system_abcd
    lens = [Surface(radius=np.inf, glass_before='air', glass_after='air',
                    thickness=stop_gap),
            Surface(radius=60e-3, conic=conic,
                    aspheric_coeffs=({4: A4} if A4 else None),
                    glass_before='air', glass_after='N-BK7', thickness=5e-3),
            Surface(radius=np.inf, glass_before='N-BK7', glass_after='air',
                    thickness=0.0)]
    _, _, bfl, _ = system_abcd(lens, WL)
    return [Surface(radius=np.inf, semi_diameter=r_stop, glass_before='air',
                    glass_after='air', thickness=stop_gap, is_stop=True),
            Surface(radius=60e-3, conic=conic,
                    aspheric_coeffs=({4: A4} if A4 else None),
                    semi_diameter=30e-3, glass_before='air',
                    glass_after='N-BK7', thickness=5e-3),
            Surface(radius=np.inf, semi_diameter=30e-3, glass_before='N-BK7',
                    glass_after='air', thickness=bfl),
            Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                    glass_after='air', label='image')], bfl


def _ref_sphere_radius(surfs):
    """XP-to-image distance from a paraxial trace written here (y-nu)."""
    y, u = 0.0, 1.0                      # ray from the stop centre
    for i, sf in enumerate(surfs[:-1]):
        if i > 0:
            n1 = float(get_glass_index(sf.glass_before, WL))
            n2 = float(get_glass_index(sf.glass_after, WL))
            c = 0.0 if not np.isfinite(sf.radius) else 1.0 / sf.radius
            u = (n1 * u - y * c * (n2 - n1)) / n2
        y = y + u * sf.thickness
    return abs(y / u)


def _fan_wavefront(surfs, theta, r_stop, axis, R_ref, nray=61):
    """Chief-referenced wavefront [m] of one fan, reference sphere written here.

    Uses ``trace`` only as the ray primitive (the audit verified its OPL to
    1.4e-17 m against a 60-digit oracle); the wavefront definition, the
    entrance eikonal and the sphere solve are local.
    """
    rho = np.linspace(-1, 1, nray)
    h = rho * r_stop

    def _b(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        n = xs.size
        L = np.zeros(n)
        M = np.full(n, np.sin(theta))
        return RayBundle(x=xs.copy(), y=ys.copy(), z=np.zeros(n), L=L, M=M,
                         N=np.sqrt(1 - M ** 2), wavelength=WL,
                         alive=np.ones(n, bool), opd=L * xs + M * ys,
                         error_code=np.zeros(n, np.uint8))

    zz = np.zeros(nray)
    im = trace(_b(zz, h) if axis == 'y' else _b(h, zz), surfs, WL).image_rays
    ch = trace(_b([0.0], [0.0]), surfs, WL).image_rays
    vx, vy, vz = im.x - ch.x[0], im.y - ch.y[0], im.z - ch.z[0]
    vd = vx * im.L + vy * im.M + vz * im.N
    v2 = vx ** 2 + vy ** 2 + vz ** 2
    leg = (v2 - 2 * R_ref * vd) / (R_ref - vd
                                   + np.sqrt(R_ref ** 2 + vd ** 2 - v2))
    return rho, np.where(im.alive, (im.opd - ch.opd[0]) + leg, np.nan)


def _rho2_coefficient(rho, W):
    g = np.isfinite(W)
    V = np.vander(rho[g], 7, increasing=True)
    c, *_ = np.linalg.lstsq(V, W[g], rcond=None)
    return float(c[2])


@pytest.mark.parametrize('A4, conic, tag', [
    (-4000.0, 0.0, 'A4 = -4000 m^-3'),
    (+4000.0, 0.0, 'A4 = +4000 m^-3'),
    (0.0, -3.0, 'conic k = -3'),
])
def test_r3_aspheric_s3_uses_the_squared_chief_marginal_ratio(A4, conic, tag):
    """The aspheric S3 term scales as ``(y_c/y_m)**2``, not ``(y_c/y_m)``.

    The audit's §4 snippet writes ``S3[i] += dS * (y_val_c / y_val_m)`` -- the
    same power as S2.  WP-A1 used the square (Welford §8.5) and argued for it
    in prose; nothing measured it, because every other R3 fixture has the stop
    AT surface 0, where the chief height at the asphere is ~0 and the two
    exponents are indistinguishable.

    ORACLE: real rays.  With the library's own Seidel wavefront convention
    (``seidel_analysis.seidel_wfe``:
    ``W = -[S1 r^4/8 + S2 r^3 cos/2 + S3 r^2 cos^2/2 + S3 r^2/4
    + S4 H^2 r^2/4 + S5 r cos/2]``) the TANGENTIAL-minus-SAGITTAL ``rho^2``
    coefficient is ``-(1/2) S3`` exactly: defocus, Petzval and the S4 term are
    all rotationally symmetric and cancel in the difference.  So
    ``S3_rays = -2 * (a2_tangential - a2_sagittal)``, with the fans traced and
    the reference sphere solved in this file.  Differencing the aspherized
    system against its base sphere isolates ``dS3`` (A4 and the conic do not
    change the paraxial focus, so the defocus is common to both).

    BAR: ``|dS3_library / dS3_rays - 1| < 0.10``.  Derivation -- third-order
    theory cannot do better than the fifth-order content of the fan, which
    this fixture's own rho^6 fit puts at ~1-2 %; measured ratios 0.9895 /
    0.9846 / 0.9891 (i.e. 1.0-1.5 % low), so the bar has ~6x headroom.  The
    FIRST-POWER hypothesis predicts a ratio of
    ``1 / (y_c/y_m) = 1/0.2618 = 3.82`` on this fixture -- measured 3.78 /
    3.76 / 3.78 -- which is 28 bar-widths away.  The two hypotheses are
    therefore separated by a factor 3.8, not by a tolerance.
    """
    stop_gap, theta, r_stop = 30e-3, np.radians(3.0), 6e-3

    base, _ = _displaced_stop_system(stop_gap, 0.0, 0.0, r_stop)
    asph, _ = _displaced_stop_system(stop_gap, A4, conic, r_stop)

    sd_b, _ = seidel_coefficients(base, WL, np.inf, 0, theta)
    sd_a, _ = seidel_coefficients(asph, WL, np.inf, 0, theta)
    ratio = float(sd_a['y_chief'][1] / sd_a['y_marginal'][1])
    # The fixture is only a discriminator if the chief/marginal ratio at the
    # aspheric surface is far from 1 (and from 0).
    assert 0.1 < abs(ratio) < 0.6, (
        f'{tag}: y_c/y_m = {ratio:.4f} at the asphere -- this fixture cannot '
        f'separate the two exponents')

    dS3_lib = float(np.sum(sd_a['S3']) - np.sum(sd_b['S3']))

    def _s3_rays(surfs):
        R_ref = _ref_sphere_radius(surfs)
        rho, wt = _fan_wavefront(surfs, theta, r_stop, 'y', R_ref)
        _, ws = _fan_wavefront(surfs, theta, r_stop, 'x', R_ref)
        return -2.0 * (_rho2_coefficient(rho, wt)
                       - _rho2_coefficient(rho, ws))

    dS3_rays = _s3_rays(asph) - _s3_rays(base)
    assert abs(dS3_rays) > 1e-7, (
        f'{tag}: the aspheric dS3 is only {dS3_rays:.3e} -- fixture too weak')

    rel = dS3_lib / dS3_rays
    assert abs(rel - 1.0) < 0.10, (
        f'{tag}: library dS3 {dS3_lib:+.6e} vs real-ray {dS3_rays:+.6e} '
        f'(ratio {rel:+.5f}); the first-power hypothesis would give '
        f'{rel / ratio:+.5f}')
    # ...and the first power is excluded by a wide margin, not by a tolerance.
    assert abs(rel / ratio - 1.0) > 1.0, (
        f'{tag}: the (y_c/y_m)^1 and (y_c/y_m)^2 hypotheses are not separated '
        f'on this fixture (ratio {ratio:.4f})')


def test_r3_aspheric_petzval_s4_is_untouched():
    """``S4`` is curvature-only: an aspheric departure must not move it.

    WP-A1's ``_aspheric_seidel`` deliberately returns no S4 term.  That is
    only consistent if the library's wavefront expansion carries the
    field-curvature DC term as ``(1/4)(S3 + S4 H^2)`` -- which
    ``seidel_analysis.seidel_wfe`` does.  Expanding the aspheric plate's
    ``dn A4 |y_m rho + y_c H|^4`` in that basis fixes ``dS3`` and ``dS4``
    simultaneously: the ``cos 2 theta`` half gives ``dS3 = dS1 (y_c/y_m)^2``
    and the DC half is then satisfied with ``dS4 = 0`` exactly.

    BAR: exact equality.  ``S4`` is a closed-form function of curvature and
    index only (``S4_code = -c (n2-n1) / (n1 n2)`` per surface), and neither
    ``conic`` nor ``aspheric_coeffs`` enters it, so any difference at all
    would mean the aspheric block had leaked into the Petzval sum.
    """
    base, _ = _displaced_stop_system(30e-3, 0.0, 0.0)
    for A4, conic in ((-4000.0, 0.0), (0.0, -3.0), (2000.0, +2.0)):
        asph, _ = _displaced_stop_system(30e-3, A4, conic)
        sb, _ = seidel_coefficients(base, WL, np.inf, 0, np.radians(3.0))
        sa, _ = seidel_coefficients(asph, WL, np.inf, 0, np.radians(3.0))
        assert np.array_equal(sa['S4'], sb['S4']), (
            f'A4={A4}, k={conic}: S4 moved from {sb["S4"]} to {sa["S4"]}')
