"""VERIFY-A7 / A4 -- ``eval_image_plane_wfe`` at the infinite conjugate,
checked against a sequential ray trace written IN THIS FILE.

The oracle imports nothing from ``lumenairy.raytrace``: exact sphere
intersection, vector Snell, OPL accumulated as ``n * t``, an externally
sourced Schott Sellmeier index and the textbook thick-lens BFL.  It covers
the OFF-AXIS ``field_max_rad`` launch, which the WP's own A4 tests only
check for finiteness, and it pins the field-sign convention trap.
"""
import warnings

import numpy as np
import pytest

from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe


def _n_bk7(lam_um):
    """Schott N-BK7 Sellmeier from the manufacturer's data sheet -- an
    external constant, not the library's glass table."""
    B = (1.03961212, 0.231792344, 1.01046945)
    C = (0.00600069867, 0.0200179144, 103.560653)
    l2 = lam_um ** 2
    return float(np.sqrt(1 + sum(b * l2 / (l2 - c) for b, c in zip(B, C))))


def _oracle_opd_waves(surfs, n_exit, rho, ep_r, theta_y, lam, img_d_m):
    """Reference-sphere OPD [waves] of a collimated bundle, from scratch.

    ``surfs``: list of ``(R, z_vertex, n_before, n_after)``.  Sphere centred
    on the chief's landing point at ``z = z_last + img_d_m`` and passing
    through the chief's crossing of the last-surface vertex plane; NEAR
    intersection root; ``rayoptics`` sign (OPD > 0 = the wavefront leads),
    chief re-zeroed -- the conventions ``_ray_sphere_opd`` documents.
    """
    L0, M0, N0 = 0.0, float(np.sin(theta_y)), float(np.cos(theta_y))
    d = np.array([L0, M0, N0])
    back = 0.05                       # arbitrary: a common launch wavefront
    P = np.stack([np.zeros_like(rho), rho * ep_r - back * M0,
                  np.full_like(rho, -back * N0)], axis=-1)
    D = np.broadcast_to(d, P.shape).copy()
    opl = P @ d                       # entrance eikonal (constant here)
    for (R, zv, n1, n2) in surfs:
        if np.isfinite(R):
            C = np.array([0.0, 0.0, zv + R])
            dP = P - C
            b = 2.0 * np.einsum('...i,...i->...', dP, D)
            c = np.einsum('...i,...i->...', dP, dP) - R * R
            s = np.sqrt(np.maximum(b * b - 4 * c, 0.0))
            t1, t2 = 0.5 * (-b - s), 0.5 * (-b + s)
            P1, P2 = P + t1[..., None] * D, P + t2[..., None] * D
            near = np.abs(P1[..., 2] - zv) <= np.abs(P2[..., 2] - zv)
            t = np.where(near, t1, t2)
            Pn = P + t[..., None] * D
            Nn = (Pn - C) / R
        else:
            t = (zv - P[..., 2]) / D[..., 2]
            Pn = P + t[..., None] * D
            Nn = np.broadcast_to(np.array([0.0, 0.0, 1.0]), Pn.shape).copy()
        opl = opl + n1 * t
        Nn = Nn / np.linalg.norm(Nn, axis=-1, keepdims=True)
        cosi = -np.einsum('...i,...i->...', Nn, D)
        Nl = np.where(cosi[..., None] < 0, -Nn, Nn)
        cosi = np.abs(cosi)
        eta = n1 / n2
        k = 1 - eta * eta * (1 - cosi * cosi)
        D = eta * D + (eta * cosi - np.sqrt(k))[..., None] * Nl
        P = Pn
    ic = int(np.argmin(rho ** 2))
    zv_last = surfs[-1][1]
    Cimg = P[ic] + ((zv_last + img_d_m - P[ic, 2]) / D[ic, 2]) * D[ic]
    Pvc = P[ic] + ((zv_last - P[ic, 2]) / D[ic, 2]) * D[ic]
    Rs = float(np.linalg.norm(Cimg - Pvc))
    dP = P - Cimg
    b = 2.0 * np.einsum('...i,...i->...', dP, D)
    c = np.einsum('...i,...i->...', dP, dP) - Rs * Rs
    s = np.sqrt(np.maximum(b * b - 4 * c, 0.0))
    t1, t2 = 0.5 * (-b - s), 0.5 * (-b + s)
    t = np.where(np.abs(t1) < np.abs(t2), t1, t2)
    tot = opl + n_exit * t
    return -(tot - tot[ic]) / lam


# Bars on "the library's infinite-conjugate OPD IS the ray trace".
#
# ON-AXIS both sides solve the identical geometry -- same launch wavefront,
# same surfaces, same reference sphere -- so in exact arithmetic they are
# equal and the residual is float64 cancellation in the ray-sphere
# quadratic (|P - C|^2 ~ 2.3e-3 m^2 against R^2 = 2.29e-3, i.e.
# eps |P-C|^2 / (2 R) ~ 5e-18 m = 1e-11 waves per surface).  Measured:
# 1.9e-11 waves over an 81-ray pupil on a map of PV 3.5944 waves.
#
# OFF-AXIS the two take the reference-sphere tangent point at O(theta^2)
# different places -- both legitimate definitions of "the chief's path
# length from the last-surface vertex plane" -- which shows up as a PURE
# DEFOCUS term and nothing else.  Measured on this singlet, fitting the
# difference to piston + tilt + rho^2 + rho^3 + rho^4:
#
#   theta     max |diff|     rho^2 coeff    everything else
#   0.0 deg   1.9e-11 w      8.4e-12 w      < 2.0e-11 w
#   1.0 deg   1.0e-04 w     -1.06e-04 w     < 5.5e-09 w
#   2.0 deg   4.1e-04 w     -4.24e-04 w     < 3.0e-08 w
#
# So the test asserts BOTH: the raw agreement (bar 5e-3 waves -- 1.1
# decades above the worst measurement and 4.6 decades below the ~210-wave
# error this audit row is about), and, after removing piston + tilt +
# defocus, the SHAPE (bar 1e-5 waves -- 2.5 decades above the worst
# measured 3.0e-8 and 7 decades below the same defect).  A real error in
# the collimated launch, the trace or the sphere would move the shape, not
# just the focus; `image_plane='best_rms'` exists precisely because the
# defocus term is a choice of reference plane.
RAYTRACE_TOL_WAVES = 5e-3
RAYTRACE_SHAPE_TOL_WAVES = 1e-5


@pytest.fixture(scope='module')
def singlet():
    import lumenairy as la
    lam = 587.6e-9
    R1, R2, d, ap = 50e-3, -50e-3, 3e-3, 10e-3
    nl = _n_bk7(lam * 1e6)
    surfs = [(R1, 0.0, 1.0, nl), (R2, d, nl, 1.0)]
    efl = 1.0 / ((nl - 1) * (1 / R1 - 1 / R2 + (nl - 1) * d / (nl * R1 * R2)))
    bfl = efl * (1 - (nl - 1) * d / (nl * R1))
    pres = dict(la.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7', aperture=ap))
    return pres, surfs, lam, ap, efl, bfl


def test_infinite_conjugate_image_distance_is_the_thick_lens_bfl(singlet):
    """Independent oracle: the textbook thick-lens BFL
    ``f (1 - (n-1) d / (n R1))`` with an externally sourced Sellmeier n.
    Measured agreement 0 .. 3.9e-16 relative across biconvex, plano-convex
    and meniscus shapes at 587.6 and 486.1 nm.  Bar 1e-12: four decades
    above that, ten below the ~2 % a dropped principal-plane offset or a
    mis-threaded terminal index would produce.
    """
    pres, surfs, lam, ap, efl, bfl = singlet
    p = dict(pres)
    p['object_distance'] = float('inf')
    w = eval_image_plane_wfe(p, lam, n_pupil=31)
    assert w.img_d_m == pytest.approx(bfl, rel=1e-12), (
        f'img_d {w.img_d_m * 1e3:.6f} mm vs thick-lens BFL '
        f'{bfl * 1e3:.6f} mm')


@pytest.mark.parametrize('th_deg', [0.0, 0.5, 1.0, 2.0])
def test_infinite_conjugate_opd_matches_an_independent_ray_trace(singlet,
                                                                 th_deg):
    """Ray by ray, not PV against PV: the whole map must reproduce a trace
    written in this file.  ``th_deg > 0`` exercises the new
    ``field_max_rad`` collimated launch off-axis."""
    pres, surfs, lam, ap, efl, bfl = singlet
    n = 81
    rho = np.linspace(-1.0, 1.0, n)
    p = dict(pres)
    p['object_distance'] = float('inf')
    th = float(np.deg2rad(th_deg))
    if th_deg:
        p['field_max_rad'] = th
        w = eval_image_plane_wfe(p, lam, field=(0.0, 1.0),
                                 pupil_grid=(np.zeros(n), rho))
    else:
        w = eval_image_plane_wfe(p, lam, pupil_grid=(np.zeros(n), rho))
    # ``_oracle_opd_waves`` takes the ray DIRECTION angle; ``field`` scales
    # an OBJECT POSITION, so a positive field is a negative direction.
    ora = _oracle_opd_waves(surfs, 1.0, rho, ap / 2, -th, lam, w.img_d_m)
    g = np.isfinite(w.opd_w) & np.isfinite(ora)
    assert int(g.sum()) > n // 2
    mid = int(g.sum() // 2)
    a = w.opd_w[g] - w.opd_w[g][mid]
    b = ora[g] - ora[g][mid]
    span = float(b.max() - b.min())
    assert span > 1.0, 'fixture must carry a real aberration'
    diff = a - b
    assert float(np.max(np.abs(diff))) < RAYTRACE_TOL_WAVES, (
        f'{th_deg} deg: library PV {a.max() - a.min():.4f} w, oracle PV '
        f'{span:.4f} w, max |diff| {float(np.max(np.abs(diff))):.3e} w')
    # ... and the SHAPE, with the O(theta^2) reference-sphere-tangent
    # difference (piston + tilt + defocus) projected out.
    r = rho[g]
    basis = np.vstack([np.ones_like(r), r, r ** 2]).T
    resid = diff - basis @ np.linalg.lstsq(basis, diff, rcond=None)[0]
    assert float(np.max(np.abs(resid))) < RAYTRACE_SHAPE_TOL_WAVES, (
        f'{th_deg} deg: shape residual {float(np.max(np.abs(resid))):.3e} w '
        f'after removing piston/tilt/defocus (PV {span:.4f} w)')


def test_chief_ray_opd_is_exactly_zero_off_axis(singlet):
    """The chief defines the reference sphere, so its OPD is 0 by
    construction at every field.  A nonzero value means the off-axis
    launch or the sphere centre drifted."""
    pres, surfs, lam, ap, efl, bfl = singlet
    p = dict(pres)
    p['object_distance'] = float('inf')
    p['field_max_rad'] = float(np.deg2rad(2.0))
    w = eval_image_plane_wfe(p, lam, field=(0.0, 1.0), n_pupil=11)
    chief = int(np.argmin(w.px ** 2 + w.py ** 2))
    assert abs(float(w.opd_w[chief])) < 1e-12


@pytest.mark.parametrize('th_deg', [0.5, 1.0, 2.0])
def test_field_means_the_same_object_point_at_both_conjugates(singlet, th_deg):
    """``field`` scales an OBJECT POSITION at both conjugates (ruling on
    open item VERIFY-A7/A4-N1), so ``field = (0, +1)`` must name the same
    physical field point whether the object is 1e3 m away or at infinity,
    and the two must agree in the limit.

    Bar, derived: at 1e3 m the object subtends the same angle to within
    ``aperture / object_distance = 1e-5`` rad of the collimated case, and
    the residual difference is the entrance-pupil parallax over that
    angle.  Measured 0.0025 / 0.0033 / 0.0033 waves at 0.5 / 1 / 2 deg,
    i.e. <= 0.07 % of the map's span; the bar at 1 % of span sits one
    decade above that.  With the field sign REVERSED -- which is what the
    infinite branch did before the ruling -- the same comparison reads
    0.854 / 1.706 / 3.404 waves, 23 / 40 / 59 % of span, so the two arms
    of this test are separated by 2.5 decades and nothing between them is
    a legal reading.  PV and RMS are sign-blind, which is why this has to
    be a per-ray comparison.
    """
    pres, surfs, lam, ap, efl, bfl = singlet
    n = 61
    rho = np.linspace(-0.95, 0.95, n)
    grid = (np.zeros(n), rho)
    th = float(np.deg2rad(th_deg))
    p_inf = dict(pres)
    p_inf['object_distance'] = float('inf')
    p_inf['field_max_rad'] = th
    w_inf = eval_image_plane_wfe(p_inf, lam, field=(0.0, +1.0), pupil_grid=grid)
    p_fin = dict(pres)
    p_fin['object_distance'] = 1e3
    p_fin['field_max_m'] = 1e3 * float(np.tan(th))
    maps = {}
    for sgn in (+1.0, -1.0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            maps[sgn] = eval_image_plane_wfe(p_fin, lam, field=(0.0, sgn),
                                             pupil_grid=grid)
    g = np.isfinite(w_inf.opd_w)
    ref = w_inf.opd_w[g] - w_inf.opd_w[g][n // 2]
    span = float(ref.max() - ref.min())
    same = maps[+1.0].opd_w[g] - maps[+1.0].opd_w[g][n // 2]
    flip = maps[-1.0].opd_w[g] - maps[-1.0].opd_w[g][n // 2]
    d_same = float(np.max(np.abs(ref - same)))
    d_flip = float(np.max(np.abs(ref - flip)))
    assert d_same < 0.01 * span, (
        f'inf field=(0,+1) must equal finite field=(0,+1); differs by '
        f'{d_same:.4f} waves ({100 * d_same / span:.2f} % of span) -- the '
        f'infinite-conjugate launch has the field sign of a ray DIRECTION '
        f'angle again instead of an object position.')
    assert d_flip > 0.1 * span, (
        f'inf field=(0,+1) also matches finite field=(0,-1) to '
        f'{100 * d_flip / span:.2f} % of span, so this fixture carries no '
        f'sign-sensitive aberration and the test above proves nothing.')


def test_chief_at_infinity_travels_towards_minus_y_for_a_positive_field(singlet):
    """The mechanism behind the test above, read directly off the launch:
    a positive ``field`` is an object ABOVE the axis, so its chief ray
    must travel DOWNWARD.  ``raytrace.ray_fan``'s ``field_angle`` is the
    ray-direction convention and is therefore this one's negative."""
    import lumenairy.analysis.image_plane_wfe as ipw
    pres, surfs, lam, ap, efl, bfl = singlet
    th = float(np.deg2rad(1.0))
    cap = {}
    real_trace = ipw.trace

    def spy(bundle, surfaces, wavelength, **kw):
        cap['M'] = float(np.asarray(bundle.M).ravel()[0])
        return real_trace(bundle, surfaces, wavelength, **kw)

    p = dict(pres)
    p['object_distance'] = float('inf')
    p['field_max_rad'] = th
    ipw.trace = spy
    try:
        eval_image_plane_wfe(p, lam, field=(0.0, +1.0), n_pupil=5)
        m_inf = cap['M']
        p_fin = dict(pres)
        p_fin['object_distance'] = 1e3
        p_fin['field_max_m'] = 1e3 * float(np.tan(th))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            eval_image_plane_wfe(p_fin, lam, field=(0.0, +1.0), n_pupil=5)
        m_fin = cap['M']
    finally:
        ipw.trace = real_trace
    assert m_inf == pytest.approx(-np.sin(th), rel=1e-12), (
        f'launch M = {m_inf:+.9f}, expected {-np.sin(th):+.9f}')
    assert m_fin < 0.0 and m_inf < 0.0, (m_inf, m_fin)


def test_field_max_rad_beyond_ninety_degrees_is_refused(singlet):
    """Ruling on open item VERIFY-A7/A4-N2: the guard is on the ANGLE.

    A direction-cosine guard (``N = sqrt(1 - sin^2 x - sin^2 y) <= 0``)
    cannot catch this, because ``sin`` is not monotonic past pi/2:
    ``field_max_rad = 2.0`` rad is 114.6 deg but gives
    ``|M| = sin(2.0) = 0.909`` and ``N = +0.417``, so it used to be
    launched silently at 65.4 deg while the message it never emitted said
    'at or beyond 90 deg from the axis'.  Both arms below are exact
    decisions, not tolerances.
    """
    pres, surfs, lam, ap, efl, bfl = singlet
    p = dict(pres)
    p['object_distance'] = float('inf')
    for bad in (2.0, float(np.pi / 2), 3.0):
        p['field_max_rad'] = bad
        with pytest.raises(ValueError, match=r'eval_image_plane_wfe: .*90 deg'):
            eval_image_plane_wfe(p, lam, field=(0.0, 1.0), n_pupil=9)
    # ... and an ordinary field is still accepted.
    p['field_max_rad'] = float(np.deg2rad(2.0))
    assert np.isfinite(
        eval_image_plane_wfe(p, lam, field=(0.0, 1.0), n_pupil=9).pv_waves)
