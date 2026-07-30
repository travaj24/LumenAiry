"""Tilted chain carriers -- niche D1 (roadmap
ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P1a).

Why it exists.  ``propagate_traced_carrier_chain`` reduced every hand-off to a
scalar sphere (``R = float(R_carrier)``), so a tilted congruence
``W = S_R(r) + L x + M y`` could not be carried chain-level even though the
ELEMENT already accepted an arbitrary ``W``.  A +-46 mrad DOE order therefore
had no way to be carried as its own congruence: its residual after the (scalar)
carrier is the FULL split angle, 2.3x outside the documented
``_NONCOLLIMATED_RESID_THRESH = 0.02 rad`` envelope, and the traced correction
is applied to a reference the beam does not follow.

``r_in=TiltedCarrier(R, L, M)`` carries ``(R, L, M)`` plus the chief-ray
position instead.  The state is closed under the group transfer -- ``R`` by
the paraxial wavefront Moebius law, and the CHIEF RAY ``(x_c, y_c, L, M)`` by
an EXACT ray trace through that group's own surfaces (niche C3, 2026-07-30;
see ``_group_chief_transfer``, which keeps the lumped paraxial ABCD only as
its paraxial limit and as a fall-back for a group that cannot be traced) --
and, in the chief-ray-tracking frame, the ENVELOPE transport is the untouched
scalar code, which is what makes the ``L = M = 0`` path byte-identical.

Everything here is SELF-CONTAINED (synthetic N-BK7 singlets built inline, no
prescription asset) and every physical claim is checked against an inline
oracle that does not share code with the chain: an exact meridional ray trace
through the real spherical surfaces, the measured WAVE centroid, and the group
ABCD.

Pins, in order:

* the ``TiltedCarrier`` eikonal / gradient are ANALYTIC (and strictly better
  than the pre-existing ndarray-carrier route, whose gradient is a finite
  difference sampled nearest-neighbour);
* ``_tilt_obliquity`` is the exact ``1/cos(theta)``, whose ``z L^3/2``
  correction is the whole reason the wave centroid lands on the ray trace;
* ``L = M = 0`` reproduces the scalar chain to <= 1e-10 * scale (the shipped
  design-121 acceptance cannot move);
* the per-group chief-ray closure reproduces a WHOLE-SYSTEM exact ray trace at
  every intermediate plane (measured residual 0.0), where the lumped paraxial
  ABCD it replaced read 0.044 um out at gA and 0.288 um at gB;
* a 46 mrad congruence through a 2-group relay lands on the EXACT meridional
  ray trace and reaches the same diffraction-limited spot the on-axis run
  does, while the scalar chain fed the same physical field smears it;
* energy is conserved;
* the per-order residual angular spread is inside the 0.02 rad envelope
  (tilted) and outside it (scalar) -- the point of the whole exercise.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements as _elements
from lumenairy.elements._lens_traced import (
    _NONCOLLIMATED_RESID_THRESH,
    _carrier_residual_rms,
    _compute_carrier,
    _input_beam_amp_radius,
)
from lumenairy.propagators.carrier import (
    _group_abcd,
    _group_chief_transfer,
    _shift_envelope,
    _tilt_obliquity,
)
from lumenairy.raytrace import Surface, make_ray, trace
from lumenairy.raytrace.trace import surfaces_from_prescription

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL

# --- synthetic 2-group relay -------------------------------------------------
_N, _DX = 1024, 12e-6          # 12.288 mm window; lambda/(2 dx) = 54.6 mrad,
_W_IN = 1.0e-3                 # so the 46 mrad ramp is RESOLVED -- the scalar
_TILT = 0.046                  # control gets a fair, unaliased input
_GAP = 25e-3
_DXO, _NO = 0.4e-6, 256
_TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _singlet(R1, R2, d, glass, ap, name):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


def _relay_prescriptions():
    return (_singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'gA'),
            _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'gB'))


def _paraxial_state(gA, gB, tilt):
    """The PARAXIAL-ABCD version of the D1 closure, computed by hand: ``R`` by
    the Moebius law, ``(x_c, L)`` as a paraxial ray through each GROUP, and the
    exact ``1/cos(theta)`` obliquity on each FREE leg.

    Two of these are still the chain's contract and are pinned as such:
    ``R_A`` / ``R_B`` (the sphere DOES follow the ABCD Moebius law) and the
    exact free-leg obliquity.  The chief-ray entries ``x_A``/``L_A``/
    ``x_B``/``L_B``/``x_img`` are NO LONGER what the chain computes -- since
    niche C3 (2026-07-30) it TRACES the chief ray through each group's own
    surfaces -- so they are kept here as the superseded predictor, used only as
    fail-before witnesses (measured 0.044 um out at gA, 0.288 um at gB and
    0.121 um at the image plane against an exact trace).

    ``x_img_paraxial`` is the same ABCD trace with the paraxial free-leg
    advance ``L z`` instead of ``z L / cos(theta)`` -- kept so the tests can
    show they discriminate the ``z L^3/2`` term."""
    A1, B1, C1, D1 = _group_abcd(gA, _WL)
    A2, B2, C2, D2 = _group_abcd(gB, _WL)
    R_A, x_A, L_A = A1 / C1, B1 * tilt, D1 * tilt        # R_in = inf
    ob_A = 1.0 / np.sqrt(1.0 - L_A ** 2)
    R_g, x_g = R_A + _GAP, x_A + L_A * _GAP * ob_A
    R_B = (A2 * R_g + B2) / (C2 * R_g + D2)
    x_B, L_B = A2 * x_g + B2 * L_A, C2 * x_g + D2 * L_A
    fd = -R_B
    ob_B = 1.0 / np.sqrt(1.0 - L_B ** 2)
    x_par = ((A2 * (x_A + L_A * _GAP) + B2 * L_A)
             + (C2 * (x_A + L_A * _GAP) + D2 * L_A) * fd)
    return dict(R_A=R_A, x_A=x_A, L_A=L_A, R_g=R_g, x_g=x_g,
                R_B=R_B, x_B=x_B, L_B=L_B, fd=fd,
                x_img_abcd=x_B + L_B * fd * ob_B, x_img_paraxial=x_par)


def _relay_surfaces(gA, gB, image_distance):
    """Both groups + the air gap + a flat image surface, as ONE surface list
    for the exact ray trace (the chain sees the gap as a chain-level leg)."""
    sa = surfaces_from_prescription(gA)
    sb = surfaces_from_prescription(gB)
    sa[-1] = dataclasses.replace(sa[-1], thickness=_GAP)
    sb[-1] = dataclasses.replace(sb[-1], thickness=image_distance)
    return sa + sb + [Surface(
        radius=np.inf, conic=0.0, semi_diameter=np.inf, glass_before='air',
        glass_after='air', is_mirror=False, thickness=0.0, label='img')]


def _exact_chief_height(gA, gB, tilt, image_distance):
    """EXACT meridional trace of the chief ray (entrance height 0, direction
    cosine ``tilt``) through the real spherical surfaces -- no paraxial
    approximation anywhere, so it is a genuinely independent oracle for the
    chain's chief-ray bookkeeping."""
    res = trace(make_ray(0.0, 0.0, tilt, 0.0, wavelength=_WL),
                _relay_surfaces(gA, gB, image_distance), _WL)
    return float(res.image_rays.x[0])


_FLAT = dict(radius=np.inf, conic=0.0, semi_diameter=np.inf,
             glass_before='air', glass_after='air', is_mirror=False)


def _chief_state(surfaces, x=0.0, y=0.0, L=0.0, M=0.0):
    """``(x, y, L, M)`` of ONE exactly-traced ray at the end of ``surfaces``,
    angles as direction cosines (what :func:`make_ray` takes and what the chain
    carries)."""
    r = trace(make_ray(x, y, L, M, wavelength=_WL), surfaces, _WL).image_rays
    return (float(r.x[0]), float(r.y[0]), float(r.L[0]), float(r.M[0]))


def _leg_surfaces(presc, pre=0.0, post=0.0):
    """``pre`` metres of air, one GROUP, ``post`` metres of air, ending on a
    flat plane -- so a single exact trace covers a whole gap+group+leg chain
    and can be compared with the chain's own composed bookkeeping.

    ``post = pre = 0`` is the group's front-vertex -> back-vertex transfer, the
    plane pair the chain's per-group closure is stated on."""
    sf = surfaces_from_prescription(presc)
    sf[-1] = dataclasses.replace(sf[-1], thickness=float(post))
    sf = sf + [Surface(thickness=0.0, label='img', **_FLAT)]
    if pre:
        sf = [Surface(thickness=float(pre), label='entry', **_FLAT)] + sf
    return sf


def _exact_ray_centroid(gA, gB, tilt, image_distance, n=15):
    """Gaussian-weighted meridional ray centroid -- the geometric analogue of
    the wave centroid (it carries the coma-side weighting the single chief ray
    does not)."""
    xs = np.linspace(-2.0 * _W_IN, 2.0 * _W_IN, n)
    surfs = _relay_surfaces(gA, gB, image_distance)
    hits, wgt = [], []
    for xe in xs:
        r = trace(make_ray(float(xe), 0.0, tilt, 0.0, wavelength=_WL),
                  surfs, _WL).image_rays
        if np.isfinite(r.x[0]):
            hits.append(float(r.x[0]))
            wgt.append(np.exp(-2.0 * xe ** 2 / _W_IN ** 2))
    hits, wgt = np.asarray(hits), np.asarray(wgt)
    return float((hits * wgt).sum() / wgt.sum())


def _gauss_env(n=_N, dx=_DX, w=_W_IN):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2
                  ).astype(np.complex128)


def _spot(field, dx_out, centre_abs):
    """(absolute centroid, x FWHM, encircled fraction within 25 um, window
    power) of a focal field on a grid centred at ``centre_abs``."""
    inten = np.abs(np.asarray(field)) ** 2
    tot = float(inten.sum())
    n = inten.shape[-1]
    ax = (np.arange(n) - n / 2) * dx_out
    cx = float((inten.sum(axis=0) * ax).sum() / tot)
    cy = float((inten.sum(axis=1) * ax).sum() / tot)
    row = inten[np.unravel_index(np.argmax(inten), inten.shape)[0]]
    idx = np.where(row >= 0.5 * row.max())[0]
    fwhm = float(idx[-1] - idx[0] + 1) * dx_out
    rr = np.hypot(ax[None, :] - cx, ax[:, None] - cy)
    return (centre_abs + cx, fwhm, float(inten[rr <= 25e-6].sum() / tot),
            tot * dx_out * dx_out)


# ===========================================================================
# 1.  the TiltedCarrier spec itself
# ===========================================================================

def test_tilted_carrier_eikonal_and_gradient_are_analytic():
    """``W = S_R(rho) + L u + M v`` about ``(x0, y0)``, with the gradient the
    exact derivative -- checked against the closed form and against a finite
    difference of ``W`` itself."""
    n, dx, R = 64, 20e-6, -30e-3
    L, M, x0, y0 = 0.046, -0.017, 1.3e-4, -2.1e-4
    ax = (np.arange(n) - n / 2) * dx
    Y, X = np.meshgrid(ax, ax, indexing='ij')
    spec = la.TiltedCarrier(R, L, M, x0, y0)
    W, grad, wfn = _compute_carrier(spec, None, _WL, dx, X, Y)
    u, v = X - x0, Y - y0
    ref = -(np.sqrt(u ** 2 + v ** 2 + R * R) - abs(R)) + L * u + M * v
    np.testing.assert_allclose(W, ref, rtol=0, atol=1e-18)
    assert abs(wfn(np.array([x0]), np.array([y0]))[0]) < 1e-18   # W(x0,y0)=0
    h = 1e-9
    xq, yq = np.array([3e-4]), np.array([-1e-4])
    gx, gy = grad(xq, yq)
    np.testing.assert_allclose(
        gx, (wfn(xq + h, yq) - wfn(xq - h, yq)) / (2 * h), rtol=2e-6)
    np.testing.assert_allclose(
        gy, (wfn(xq, yq + h) - wfn(xq, yq - h)) / (2 * h), rtol=2e-6)


def test_tilted_carrier_collimated_is_the_pure_tilted_plane():
    """``R = +/-inf`` must give ``W = L u + M v`` with a CONSTANT gradient --
    not the all-NaN ``inf - inf`` sentinel that silently disables the engage
    test (the trap the scalar branch documents)."""
    n, dx = 32, 20e-6
    ax = (np.arange(n) - n / 2) * dx
    Y, X = np.meshgrid(ax, ax, indexing='ij')
    for R in (np.inf, -np.inf):
        W, grad, _ = _compute_carrier(la.TiltedCarrier(R, 0.046, -0.01),
                                      None, _WL, dx, X, Y)
        assert np.isfinite(W).all()
        np.testing.assert_allclose(W, 0.046 * X - 0.01 * Y, atol=1e-18)
        gx, gy = grad(np.array([1e-4, -2e-4]), np.array([0.0, 3e-4]))
        np.testing.assert_allclose(gx, 0.046, atol=1e-15)
        np.testing.assert_allclose(gy, -0.01, atol=1e-15)


def test_tilted_carrier_beats_the_equivalent_ndarray_wavefront():
    """The element already accepted an ``ndarray`` W -- but that branch
    differentiates by ``np.gradient`` and samples NEAREST-NEIGHBOUR, so the
    ray-launch cosines are quantised to the grid.  The analytic spec is exact
    everywhere; the ndarray one is wrong OFF the grid lattice, which is
    exactly where the Newton entrance heights land."""
    n, dx, R = 64, 20e-6, -30e-3
    spec = la.TiltedCarrier(R, 0.046, 0.0)
    ax = (np.arange(n) - n / 2) * dx
    Y, X = np.meshgrid(ax, ax, indexing='ij')
    W, grad_a, _ = _compute_carrier(spec, None, _WL, dx, X, Y)
    _, grad_n, _ = _compute_carrier(np.asarray(W), None, _WL, dx, X, Y)
    xq = np.array([0.37 * dx, 4.5 * dx])          # deliberately off-lattice
    yq = np.zeros_like(xq)
    exact = 0.046 - xq / np.sqrt(xq ** 2 + R * R)
    np.testing.assert_allclose(grad_a(xq, yq)[0], exact, rtol=1e-12)
    assert np.abs(grad_n(xq, yq)[0] - exact).max() > 1e-5


def test_tilted_carrier_guards():
    ax = (np.arange(8) - 4) * 1e-5
    Y, X = np.meshgrid(ax, ax, indexing='ij')
    with pytest.raises(ValueError, match='own focus'):
        _compute_carrier(la.TiltedCarrier(0.0, 0.1), None, _WL, 1e-5, X, Y)
    assert la.TiltedCarrier(1.0).is_tilted is False
    assert la.TiltedCarrier(1.0, 0.0, 0.0, 1e-9).is_tilted is True


# ===========================================================================
# 2.  the analytic transport pieces
# ===========================================================================

def test_tilt_obliquity_is_exact_secant_not_the_paraxial_one():
    """``1/sqrt(1-L^2-M^2)``: the ``z L^3/2`` term it adds to the chief-ray
    advance is what puts the wave centroid on the ray trace (measured 2.73 um
    on the D1 single-group case, 0.15 of the spot FWHM)."""
    for th in (0.0, 5e-3, 0.046, 0.3):
        L, M = np.sin(th) * 0.6, np.sin(th) * 0.8
        assert _tilt_obliquity(L, M, 'fn') == pytest.approx(
            1.0 / np.cos(th), rel=1e-14)
    # paraxial limit, and the cubic term is the whole correction
    L = 0.046
    assert _tilt_obliquity(L, 0.0, 'fn') - 1.0 == pytest.approx(
        0.5 * L ** 2, rel=2e-3)
    with pytest.raises(ValueError, match='DIRECTION COSINES'):
        _tilt_obliquity(0.8, 0.7, 'fn')


def test_shift_envelope_is_exact_for_a_band_limited_envelope():
    """Sub-pixel, band-limited and lossless -- the hand-off relies on it to
    place the beam at its physical position without quantising the offset."""
    n, dx, w = 128, 5e-6, 6e-5
    ax = (np.arange(n) - n / 2) * dx
    env = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / w ** 2
                 ).astype(np.complex128)
    s = 3.37 * dx
    out = _shift_envelope(env, s, 0.0, dx)
    ref = np.exp(-((ax[None, :] - s) ** 2 + ax[:, None] ** 2) / w ** 2)
    np.testing.assert_allclose(np.abs(out), ref, atol=1e-9)
    back = _shift_envelope(out, -s, 0.0, dx)
    np.testing.assert_allclose(back, env, atol=1e-12)
    assert np.array_equal(_shift_envelope(env, 0.0, 0.0, dx), env)


# ===========================================================================
# 3.  the byte-identity pin -- the shipped acceptance must not move
# ===========================================================================

@pytest.mark.parametrize('final_distance', [0.0, 8e-3])
def test_zero_tilt_reproduces_the_scalar_chain(final_distance):
    """``TiltedCarrier(R, 0, 0)`` REDUCES to the scalar path: the new entry
    point must not perturb the shipped scalar default.  A 2-group relay with a
    gap and a final leg, compared with a TOLERANCE (<= 1e-10 * scale) rather
    than array_equal, since both arms are live FFT work.  (Measured margin
    here: exactly 0.)

    WHAT THIS DOES NOT COVER (niche C1 item 2, 2026-07-30).
    ``_parse_chain_carrier(TiltedCarrier(R, 0.0, 0.0), fn)`` reports
    ``tilted=False``, so both arms below are literally the SAME code path and
    the zero margin says nothing at all about the tilted branch -- not about
    the tilted transport, not about the obliquity piston, not about the
    chief-ray bookkeeping.  That is deliberate and it is this test's whole
    scope: the REDUCTION.  The tilted branch itself is pinned by
    ``test_tilted_relay_lands_on_the_exact_ray_trace`` /
    ``test_chief_ray_closure_matches_the_exact_chief_trace`` here (meridional)
    and by ``tests/unit/test_niche_c1_consolidation.py``'s SKEW pins, which
    score a
    (44, 26) mrad congruence against an exact skew ray trace and demonstrate a
    fail-before for each of the three mechanisms."""
    n, dx, w, R_in = 512, 30e-6, 4.5e-3, 60e-3
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    env = _gauss_env(n, dx, w)
    groups = [{'prescription': presc, 'gap_before': 20e-3},
              {'prescription': presc, 'gap_before': 10e-3}]
    kw = dict(ray_subsample=8, n_workers=1, final_distance=final_distance,
              traced_kwargs=_TKW)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = la.propagate_traced_carrier_chain(env, groups, _WL, dx,
                                              r_in=R_in, **kw)
        b = la.propagate_traced_carrier_chain(
            env, groups, _WL, dx, r_in=la.TiltedCarrier(R_in, 0.0, 0.0), **kw)
    A, B = np.asarray(a.field), np.asarray(b.field)
    assert A.shape == B.shape and A.dtype == B.dtype
    margin = float(np.abs(A - B).max())
    assert margin <= 1e-10 * float(np.abs(A).max()), margin
    # and no tilt bookkeeping leaks into the scalar result contract.  Pinned
    # as "the tilt keys are absent, and the two runs agree on the key set"
    # rather than as a closed list: niche C3 added the per-leg
    # ``gap_*`` transport diagnostics to EVERY stage, and a closed list makes
    # any future scalar-side diagnostic look like a tilt leak.
    assert len(a.stages) == len(b.stages) == 2
    _TILT_KEYS = {'L_out', 'M_out', 'x_c_out', 'y_c_out', 'L', 'M', 'x_c',
                  'y_c'}
    for sa, sb in zip(a.stages, b.stages):
        assert set(sa) == set(sb)
        assert {'name', 'R_in', 'R_out', 'dx', 'w', 'power'} <= set(sa)
        assert not (set(sa) & _TILT_KEYS)


# ===========================================================================
# 4.  the 46 mrad relay -- position, spot, energy, residual
# ===========================================================================

@pytest.fixture(scope='module')
def _relay():
    gA, gB = _relay_prescriptions()
    st = _paraxial_state(gA, gB, _TILT)
    x_exact = _exact_chief_height(gA, gB, _TILT, st['fd'])
    x_ray_centroid = _exact_ray_centroid(gA, gB, _TILT, st['fd'])
    env = _gauss_env()
    groups = [{'prescription': gA, 'gap_before': 0.0},
              {'prescription': gB, 'gap_before': _GAP}]
    return dict(gA=gA, gB=gB, st=st, x_exact=x_exact,
                x_ray_centroid=x_ray_centroid, env=env, groups=groups)


def _run_chain(relay, field, r_in, centre_abs, n_out=_NO):
    fr = dict(dx_out=_DXO, N_out=n_out, centre_out=(centre_abs, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain(
            field, relay['groups'], _WL, _DX, r_in=r_in, ray_subsample=8,
            n_workers=1, final_distance=relay['st']['fd'], focus_readout=fr,
            traced_kwargs=_TKW)


@pytest.fixture(scope='module')
def _runs(_relay):
    tilted = _run_chain(_relay, _relay['env'],
                        la.TiltedCarrier(np.inf, _TILT, 0.0),
                        _relay['x_exact'])
    on_axis = _run_chain(_relay, _relay['env'], np.inf, 0.0)
    x = (np.arange(_N) - _N // 2) * _DX
    scalar_in = _relay['env'] * np.exp(1j * _K0 * _TILT * x[None, :])
    scalar = _run_chain(_relay, scalar_in, np.inf, _relay['x_exact'],
                        n_out=1024)
    return dict(tilted=tilted, on_axis=on_axis, scalar=scalar,
                scalar_in=scalar_in)


def test_chief_ray_closure_matches_the_exact_chief_trace(_relay, _runs):
    """The per-group ``(x_c_out, L_out)`` the chain reports must be the EXACT
    chief-ray trace through that group's own surfaces, front vertex -> back
    vertex -- the closure that makes the tilted state carryable at all.

    RENAMED 2026-07-30 (niche C3).  This test used to be
    ``..._matches_the_group_abcd`` and pinned the group's lumped paraxial
    ABCD, BY CONSTRUCTION.  That premise is obsolete: the chain deliberately no
    longer uses the ABCD for the chief ray, because a lumped group ABCD is
    neither a sine nor a tangent convention (refraction is linear in SINES,
    intra-group transfer in TANGENTS) while the chain carries ``(L, M)`` as
    DIRECTION COSINES.  Measured on this fixture against an exact trace: the
    ABCD predictor is 0.044 um out at gA's back vertex, 0.288 um at gB's and
    +0.1214 um at the image plane; a cosine<->slope conversion of the same
    ABCD is +1.1208 um, i.e. 9x WORSE; the exact trace is 0.0.

    The oracle here is the same whole-system exact trace
    ``test_tilted_relay_lands_on_the_exact_ray_trace`` uses, stopped one plane
    early -- so gB's pin is ONE five-surface trace against the chain's
    (trace gA) + (obliquity free leg) + (trace gB) composition, which is a
    genuinely different computation and is what validates the composition.

    The ABCD is NOT dropped: it is pinned in its two remaining documented
    roles, the PARAXIAL LIMIT of the exact transfer and the FALL-BACK for a
    group the ray engine cannot build.
    """
    st, stages = _relay['st'], _runs['tilted'].stages
    gA, gB = _relay['gA'], _relay['gB']
    assert [s['name'] for s in stages] == ['gA', 'gB', '<target>']
    # --- gA back vertex: exact trace of the same 46 mrad chief ray
    xa, _ya, La, _Ma = _chief_state(_leg_surfaces(gA), L=_TILT)
    assert stages[0]['x_c_out'] == pytest.approx(xa, rel=1e-12)
    assert stages[0]['L_out'] == pytest.approx(La, rel=1e-12)
    # --- gB back vertex: the WHOLE relay, stopped there (image_distance = 0)
    xb, _yb, Lb, _Mb = _chief_state(_relay_surfaces(gA, gB, 0.0), L=_TILT)
    assert stages[1]['x_c_out'] == pytest.approx(xb, rel=1e-12)
    assert stages[1]['L_out'] == pytest.approx(Lb, rel=1e-12)
    # the SPHERE still follows the paraxial ABCD Moebius law -- only the chief
    # ray moved, and that separation is the whole D1 closure
    assert stages[0]['R_out'] == pytest.approx(st['R_A'], rel=1e-12)
    assert stages[1]['R_out'] == pytest.approx(st['R_B'], rel=1e-12)
    assert stages[0]['y_c_out'] == 0.0 and stages[0]['M_out'] == 0.0
    # --- fail-before: the superseded ABCD chief ray is measurably wrong, and
    # this test used to assert exactly those two numbers
    assert abs(st['x_A'] - xa) == pytest.approx(0.0440e-6, rel=0.05)
    assert abs(st['x_B'] - xb) == pytest.approx(0.2881e-6, rel=0.05)
    # --- the ABCD is the exact transfer's PARAXIAL LIMIT ...
    abcd = _group_abcd(gA, _WL)
    tiny = 1e-7
    x_t, _, L_t, _ = _group_chief_transfer(gA, abcd, 0.0, 0.0, tiny, 0.0,
                                           _WL, 'fn')
    assert x_t == pytest.approx(abcd[1] * tiny, rel=1e-9)
    assert L_t == pytest.approx(abcd[3] * tiny, rel=1e-9)
    # ... and its documented FALL-BACK, for a group the ray engine cannot
    # build (here an unknown glass): the predictor degrades to the ABCD
    # rather than killing the propagation
    bad = dict(gA, surfaces=[dict(s) for s in gA['surfaces']])
    bad['surfaces'][0]['glass_after'] = 'NOT-A-REAL-GLASS'
    bad['surfaces'][1]['glass_before'] = 'NOT-A-REAL-GLASS'
    assert _group_chief_transfer(bad, abcd, 0.0, 0.0, _TILT, 0.0, _WL,
                                 'fn') == (abcd[1] * _TILT, 0.0,
                                           abcd[3] * _TILT, 0.0)
    # ... and the untilted short-circuit that keeps the on-axis path
    # byte-identical returns the zeros the ABCD returned too
    assert _group_chief_transfer(gA, abcd, 0.0, 0.0, 0.0, 0.0, _WL, 'fn') == \
        (0.0, 0.0, 0.0, 0.0)


def test_tilted_relay_lands_on_the_exact_ray_trace(_relay, _runs):
    """THE acceptance for P1a's position claim.  The measured WAVE centroid
    must land on the EXACT meridional ray trace of the 46 mrad chief ray
    through the real spherical surfaces.

    Discriminating power (all measured on this fixture): dropping the
    element's ``C x_c`` chief-ray bending moves the image by ~430 um, the
    ``B L`` term by ~92 um, and using the PARAXIAL advance ``L z`` instead of
    the exact ``z L / cos(theta)`` by 0.77 um -- against a 0.30 um tolerance
    and a measured 0.014 um wave residual (the image height itself is
    1.783 mm, so that is 8e-6 relative).

    The chain's own analytic tracking used to be a SEPARATE number here: it
    pushed the chief ray through the lumped paraxial group ABCD and landed
    +0.1214 um from the exact trace.  Since niche C3 (2026-07-30) it TRACES
    the chief ray through each group, so it now EQUALS the exact trace
    (measured residual 0.0, i.e. the two agree to the last bit), and the
    measured wave centroid is 0.014 um from it instead of 0.107 um -- a
    7.7x closer landing judged by a diffraction calculation that shares no
    code with either predictor.
    """
    tol = 0.30e-6
    x_meas, _, _, _ = _spot(_runs['tilted'].field, _DXO, _relay['x_exact'])
    # the test must be able to SEE the exact-obliquity term at all: a
    # fully-paraxial chain lands 0.77 um away, 2.6x the tolerance
    assert abs(_relay['st']['x_img_paraxial'] - _relay['x_exact']) > 2 * tol
    assert abs(x_meas - _relay['x_exact']) < tol, (
        x_meas, _relay['x_exact'])
    # ... and it is nearer the Gaussian-weighted ray centroid than a full
    # spot radius, i.e. the whole geometric image position is reproduced
    assert abs(x_meas - _relay['x_ray_centroid']) < 1.0e-6
    # the chain's own analytic chief-ray tracking IS the exact ray trace
    assert _runs['tilted'].stages[-1]['x_c'] == pytest.approx(
        _relay['x_exact'], rel=1e-9)
    # fail-before: the superseded ABCD predictor is 0.1214 um out, and the
    # WAVE (an independent judge of both) is 7.7x nearer the new one
    x_abcd = _relay['st']['x_img_abcd']
    assert abs(x_abcd - _relay['x_exact']) == pytest.approx(0.1214e-6,
                                                            rel=0.05)
    assert abs(x_meas - _relay['x_exact']) < 0.2 * abs(x_abcd - x_meas)


def test_tilted_relay_reaches_the_on_axis_diffraction_limit(_relay, _runs):
    """Same relay, same beam, 46 mrad off axis: the spot must be the SAME
    diffraction-limited spot the on-axis run makes."""
    _, fwhm_t, ee_t, _ = _spot(_runs['tilted'].field, _DXO, _relay['x_exact'])
    _, fwhm_0, ee_0, _ = _spot(_runs['on_axis'].field, _DXO, 0.0)
    assert fwhm_t == pytest.approx(fwhm_0, abs=1.5 * _DXO)
    assert ee_t == pytest.approx(ee_0, abs=0.01)
    pk_t = float(np.abs(np.asarray(_runs['tilted'].field)).max() ** 2)
    pk_0 = float(np.abs(np.asarray(_runs['on_axis'].field)).max() ** 2)
    assert pk_t / pk_0 > 0.99
    # and it is the ABCD/Gaussian diffraction width, not just "same as x"
    w_exit = _runs['tilted'].stages[1]['w']
    na = w_exit / abs(_relay['st']['R_B'])
    fwhm_gauss = np.sqrt(2 * np.log(2)) * _WL / (np.pi * na)
    assert fwhm_t == pytest.approx(fwhm_gauss, rel=0.10), (fwhm_t, fwhm_gauss)


def test_scalar_chain_smears_the_same_physical_field(_relay, _runs):
    """Fail-before.  Handing the SAME physical (tilted) field to the scalar
    chain -- unaliased, ``dx`` resolves the 46 mrad ramp -- leaves the traced
    correction referenced to a plane wave the beam does not follow."""
    _, fwhm_t, _, pw_t = _spot(_runs['tilted'].field, _DXO, _relay['x_exact'])
    _, fwhm_s, _, pw_s = _spot(_runs['scalar'].field, _DXO, _relay['x_exact'])
    assert fwhm_s > 10 * fwhm_t, (fwhm_s, fwhm_t)
    assert pw_s < 1e-3 * pw_t, (pw_s, pw_t)
    pk_t = float(np.abs(np.asarray(_runs['tilted'].field)).max() ** 2)
    pk_s = float(np.abs(np.asarray(_runs['scalar'].field)).max() ** 2)
    assert pk_s < 1e-4 * pk_t


def test_energy_is_conserved_through_the_tilted_relay(_relay, _runs):
    """No energy is created or lost by the tilt bookkeeping (all of it is
    unit-modulus phase plus one band-limited shift)."""
    p_in = float((np.abs(_relay['env']) ** 2).sum()) * _DX * _DX
    _, _, _, pw_t = _spot(_runs['tilted'].field, _DXO, _relay['x_exact'])
    _, _, _, pw_0 = _spot(_runs['on_axis'].field, _DXO, 0.0)
    assert pw_t <= p_in * (1.0 + 1e-9)
    assert pw_t / p_in > 0.98
    assert pw_t == pytest.approx(pw_0, rel=0.02)
    for st in _runs['tilted'].stages[:2]:
        assert st['power'] == pytest.approx(p_in, rel=0.02)
    assert np.isfinite(np.asarray(_runs['tilted'].field)).all()


def test_per_order_residual_stays_inside_the_documented_envelope(_relay):
    """The whole point of P1a: with its own ``(R, L, M)`` an order's residual
    after the carrier is the ordinary small diffraction residual, INSIDE
    ``_NONCOLLIMATED_RESID_THRESH``; the scalar chain's is the full split
    angle and worse downstream."""
    x = (np.arange(_N) - _N // 2) * _DX
    scalar_in = _relay['env'] * np.exp(1j * _K0 * _TILT * x[None, :])
    seen = {}

    def _measure(tag, field, r_in):
        rec = []
        orig = _elements.apply_real_lens_traced

        def _spy(E_in, **kw):
            rec.append((np.asarray(E_in), kw['carrier'], kw['dx']))
            return orig(E_in, **kw)

        _elements.apply_real_lens_traced = _spy
        try:
            _run_chain(_relay, field, r_in, 0.0)
        finally:
            _elements.apply_real_lens_traced = orig
        out = []
        for E_f, carr, dxg in rec:
            n = E_f.shape[-1]
            ax = (np.arange(n) - n / 2) * dxg
            Yg, Xg = np.meshgrid(ax, ax, indexing='ij')
            W = None
            if not (carr is None
                    or (np.isscalar(carr) and not np.isfinite(carr))):
                W, _, _ = _compute_carrier(carr, E_f, _WL, dxg, Xg, Yg)
            out.append(_carrier_residual_rms(E_f, W, _WL, dxg))
        seen[tag] = out

    _measure('tilted', _relay['env'], la.TiltedCarrier(np.inf, _TILT, 0.0))
    _measure('scalar', scalar_in, np.inf)
    assert len(seen['tilted']) == len(seen['scalar']) == 2
    for r in seen['tilted']:
        assert r < 0.1 * _NONCOLLIMATED_RESID_THRESH, seen
    for r in seen['scalar']:
        assert r > 2.0 * _NONCOLLIMATED_RESID_THRESH, seen


# ===========================================================================
# 5.  guards
# ===========================================================================

def test_sequence_shorthand_matches_the_dataclass():
    a = la.propagate_traced_carrier_chain.__doc__
    assert 'TiltedCarrier' in a
    from lumenairy.propagators.carrier import _parse_chain_carrier
    assert _parse_chain_carrier((-30e-3, 0.01, 0.02), 'fn') == (
        -30e-3, 0.01, 0.02, 0.0, 0.0, True)
    assert _parse_chain_carrier(
        la.TiltedCarrier(-30e-3, 0.01, 0.02, 1e-4, 2e-4), 'fn') == (
        -30e-3, 0.01, 0.02, 1e-4, 2e-4, True)
    assert _parse_chain_carrier(np.inf, 'fn') == (
        np.inf, 0.0, 0.0, 0.0, 0.0, False)
    with pytest.raises(ValueError, match=r'\(R, L, M\)'):
        _parse_chain_carrier((1.0, 2.0), 'fn')
    with pytest.raises(ValueError, match='own focus'):
        _parse_chain_carrier((0.0, 0.01, 0.0), 'fn')


def test_non_propagating_tilt_raises_before_any_work():
    with pytest.raises(ValueError, match='DIRECTION COSINES'):
        la.propagate_traced_carrier_chain(
            _gauss_env(32, 1e-5, 5e-5), [], _WL, 1e-5,
            r_in=la.TiltedCarrier(np.inf, 0.9, 0.5))


def test_offgrid_chief_ray_raises_instead_of_wrapping():
    """The band-limited shift is periodic, so a chief ray further off axis
    than the grid can hold would wrap the beam to the far edge -- a
    plausible-looking wrong answer.  Raise, naming the grid it would need."""
    gA, _ = _relay_prescriptions()
    n, dx = 128, 20e-6                      # half-extent 1.28 mm
    with pytest.raises(ValueError, match='does not fit on the co-moving grid'):
        la.propagate_traced_carrier_chain(
            _gauss_env(n, dx, 2e-4),
            [{'prescription': gA, 'gap_before': 0.0}], _WL, dx,
            r_in=la.TiltedCarrier(np.inf, 0.0, 0.0, 5.0e-3, 0.0),
            ray_subsample=8, n_workers=1, traced_kwargs=_TKW)


def test_exact_final_leg_now_runs_for_a_tilted_carrier(_relay, _runs):
    """D1's documented limitation is CLOSED by niche D6.

    The exact high-NA final leg used to raise ``NotImplementedError`` for any
    tilted congruence: its pre-readout re-trace cropped to a few beam radii
    about the GRID CENTRE, which cannot also hold an off-axis chief ray.  D6
    made that crop tilt-aware (the retrace window grows to
    ``2*max(|x_c|,|y_c|) + window_factor*w`` so ONE axis-centred grid holds
    both, and the readout takes its own crop about the chief ray), so the leg
    now runs and lands on the same chief ray the paraxial route reports.  Full
    coverage, including the fail-before/pass-after against an independent
    Kirchhoff oracle, lives in ``test_niche_d6_exact_tilted_leg.py``."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            _relay['env'], _relay['groups'], _WL, _DX,
            r_in=la.TiltedCarrier(np.inf, _TILT, 0.0), ray_subsample=8,
            n_workers=1, final_distance=_relay['st']['fd'],
            final_leg='exact', traced_kwargs=_TKW,
            on_tilt_exact_grid='warn',
            focus_readout=dict(dx_out=_DXO, N_out=_NO,
                               centre_out=(_relay['x_exact'], 0.0)))
    assert np.isfinite(np.asarray(res.field)).all()
    assert any(s.get('exact_final') for s in res.stages)
    tgt = res.stages[-1]
    assert tgt.get('target') and tgt.get('exact_final')
    # the exact leg reports the SAME chief ray as the paraxial route -- the
    # closure is shared, only the readout changed
    assert tgt['x_c'] == pytest.approx(
        _runs['tilted'].stages[-1]['x_c'], rel=1e-12)


# ===========================================================================
# 6.  the remaining tilted routes (no readout, legacy reference, overrides)
# ===========================================================================

def _small_setup():
    n, dx, w = 256, 20e-6, 4e-4
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 3e-3, 'p')
    return n, dx, w, presc, _gauss_env(n, dx, w)


@pytest.mark.parametrize('carrier_reference', ['sphere', 'parabola'])
def test_tilted_run_without_a_focus_readout(carrier_reference):
    """The plain-reconstruct landing: the returned grid is CENTRED ON THE
    CHIEF RAY, its absolute position is reported, and the tilt ramp is put
    back on the field so the result is the physical field, not an envelope.
    Exercised under BOTH carrier references (the legacy escape hatch shares
    the tilt plumbing but skips the sphere conversions)."""
    n, dx, w, presc, env = _small_setup()
    tkw = dict(_TKW)
    if carrier_reference == 'parabola':
        tkw.update(amplitude_model='screen', preserve_input_phase=True)
    L, M = 0.02, 0.01
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env, [{'prescription': presc, 'gap_before': 2e-3}], _WL, dx,
            r_in=la.TiltedCarrier(np.inf, L, M), ray_subsample=8,
            n_workers=1, final_distance=5e-3,
            carrier_reference=carrier_reference, traced_kwargs=tkw)
    field = np.asarray(res.field)
    assert np.isfinite(field).all()
    assert res.R is not None and np.isfinite(res.R)
    tgt = res.stages[-1]
    assert tgt['name'] == '<target>' and tgt['target'] is True
    # ORACLE: ONE exact skew trace of the whole thing -- 2 mm of air, the real
    # spherical surfaces, then the 5 mm final leg -- against the chain's
    # composed (obliquity leg) + (traced group) + (obliquity leg) bookkeeping.
    # Before niche C3 the group step was the lumped paraxial ABCD, which lands
    # 0.0037 um short in x and 0.0018 um in y here (rel 2.1e-5); the traced
    # step reproduces the single trace to 2e-16 relative.
    xt, yt, Lt, Mt = _chief_state(_leg_surfaces(presc, pre=2e-3, post=5e-3),
                                  L=L, M=M)
    assert tgt['x_c'] == pytest.approx(xt, rel=1e-12)
    assert tgt['y_c'] == pytest.approx(yt, rel=1e-12)
    assert tgt['L'] == pytest.approx(Lt, rel=1e-12)
    assert tgt['M'] == pytest.approx(Mt, rel=1e-12)
    # fail-before: the ABCD group step really is a different, worse answer
    A, B, C, D = _group_abcd(presc, _WL)
    ob0 = _tilt_obliquity(L, M, 'fn')
    x1, y1 = L * 2e-3 * ob0, M * 2e-3 * ob0
    x2, L2a = A * x1 + B * L, C * x1 + D * L
    y2, M2a = A * y1 + B * M, C * y1 + D * M
    ob2 = _tilt_obliquity(L2a, M2a, 'fn')
    assert abs((x2 + L2a * 5e-3 * ob2) - xt) == pytest.approx(0.00369e-6,
                                                              rel=0.05)
    assert abs((y2 + M2a * 5e-3 * ob2) - yt) == pytest.approx(0.00184e-6,
                                                              rel=0.05)
    # the tilt ramp really is on the returned FIELD: its mean transverse
    # phase gradient is the carried tilt, not zero
    row = field[field.shape[0] // 2]
    lo, hi = 3 * len(row) // 8, 5 * len(row) // 8
    grad = np.angle(row[lo + 1:hi] * np.conj(row[lo:hi - 1])) \
        / (_K0 * res.dx)
    assert float(np.median(grad)) == pytest.approx(Lt, abs=2e-3)


def test_per_group_tilted_override_reseeds_the_congruence():
    """A group-level ``'r_in'`` may itself be a TiltedCarrier -- that is how a
    DOE plane hands each order its own congruence mid-chain.

    The reseeded congruence enters at the group's FRONT VERTEX (the override
    replaces the state the gap left), so the oracle is one exact trace of a
    10 mrad chief ray from that vertex to the back vertex.  Before niche C3
    the chain reported the ABCD's ``(B t, D t)`` instead, which is 0.00045 um
    out in height and 8.5e-9 out in direction cosine here."""
    n, dx, w, presc, env = _small_setup()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env, [{'prescription': presc, 'gap_before': 2e-3,
                   'r_in': la.TiltedCarrier(np.inf, 0.01, 0.0)}],
            _WL, dx, r_in=np.inf, ray_subsample=8, n_workers=1,
            traced_kwargs=_TKW)
    A, B, C, D = _group_abcd(presc, _WL)
    xo, _yo, Lo, _Mo = _chief_state(_leg_surfaces(presc), L=0.01)
    tgt = res.stages[-1]
    assert tgt['name'] == '<target>'
    assert tgt['L'] == pytest.approx(Lo, rel=1e-12)
    assert tgt['x_c'] == pytest.approx(xo, rel=1e-12)
    assert np.isfinite(np.asarray(res.field)).all()
    # fail-before: the superseded ABCD closure is a different answer
    assert abs(B * 0.01 - xo) == pytest.approx(0.000452e-6, rel=0.05)
    assert abs(D * 0.01 - Lo) == pytest.approx(8.545e-9, rel=0.05)


# ---------------------------------------------------------------------------
# The P2 aperture:beam cliff guard under a DECENTRED beam (D1 fixer,
# adversarial kill 2026-07-28).
#
# The chain defaults ``fit_radius_beam_factor=2.0`` (audit
# AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 S4) because the aperture:beam
# cliff is SILENT -- measured exit-wavefront Strehl 0.998 -> 0.105 -> 0.039.
# Both halves of that guard were referenced to the GRID ORIGIN: the beam radius
# is an intensity second moment about the origin, and the ray-fit disc is a
# disc about the origin.  D1 hands the element a beam at its physical chief-ray
# position, so the "beam radius" became sqrt(2 x_c^2 + w^2) and the disc it
# sizes grew with the DECENTRE until it covered the whole launch domain -- the
# guard silently stopped guarding on exactly the path D1 added.
#
# Oracle: ``newton_fit='spline'`` is a LOCAL bicubic interpolant of the same
# traced map.  It needs no fit-domain restriction and is immune to the global
# tensor-Chebyshev aliasing the discs exist to prevent (see
# ``_CARRIER_FIT_RADIUS_FRAC``: "the local bicubic spline is immune"), so it is
# the reference both polynomial guards must reproduce.
# ---------------------------------------------------------------------------

_OFFAX_AP = 12e-3               # 30x the beam 1/e^2 diameter -- cliff regime
_OFFAX_N, _OFFAX_DX = 1024, 20e-6
_OFFAX_W = 0.40e-3
_OFFAX_XC = 3.2e-3              # 8 w off axis, still 3.2 mm inside the grid
_OFFAX_FD = 30e-3


def _offax_prescription():
    return _singlet(32e-3, -32e-3, 1.6e-3, 'N-BK7', _OFFAX_AP, 'offax')


def _offax_focus_centroid(extra_kw, x_c=_OFFAX_XC, r_in=np.inf):
    """x centroid (m) of the focal spot of a beam launched ``x_c`` off axis
    through one fast singlet.  ``r_in`` finite ENGAGES the carrier, which also
    switches on the aperture-relative ``_CARRIER_FIT_RADIUS_FRAC`` disc, so the
    two discs have to be intersected rather than min-combined."""
    env = _gauss_env(_OFFAX_N, _OFFAX_DX, _OFFAX_W)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain(
            env, [{'prescription': _offax_prescription(), 'gap_before': 0.0}],
            _WL, _OFFAX_DX,
            r_in=la.TiltedCarrier(r_in, 0.0, 0.0, x_c, 0.0),
            ray_subsample=8, n_workers=1, final_distance=_OFFAX_FD,
            final_leg='paraxial',
            focus_readout=dict(dx_out=0.5e-6, N_out=1024,
                               centre_out=(0.0, 0.0)),
            traced_kwargs=dict(_TKW, on_aperture_beam='silent', **extra_kw))
    inten = np.abs(np.asarray(res.field)) ** 2
    axis = (np.arange(1024) - 1024 // 2) * 0.5e-6
    return (float((inten.sum(axis=0) * axis).sum() / inten.sum()),
            [str(m.message) for m in rec])


def test_beam_radius_is_measured_about_the_beam_not_the_grid_origin():
    """The second moment that SIZES the cliff guard must follow the beam.

    About the origin a beam of radius ``w`` at ``x_c`` reads
    ``sqrt(2 x_c^2 + w^2)``: at 8 w off axis that is 11.4x the true radius, so
    the disc the guard builds from it is 11.4x too big.  Closed form, no
    tolerance slop."""
    n, dx, w, x_c = 512, 20e-6, 4.0e-4, 3.0e-3
    x = (np.arange(n) - n // 2) * dx
    env = np.exp(-((x[None, :] - x_c) ** 2 + x[:, None] ** 2) / w ** 2
                 ).astype(np.complex128)
    about_beam = _input_beam_amp_radius(env, dx, centre=(x_c, 0.0))
    about_origin = _input_beam_amp_radius(env, dx)
    assert about_beam == pytest.approx(w, rel=2e-3)
    assert about_origin == pytest.approx(np.sqrt(2 * x_c ** 2 + w ** 2),
                                         rel=2e-3)
    assert about_origin > 7.0 * about_beam
    # y decentre, and the on-axis no-op
    env_y = np.exp(-(x[None, :] ** 2 + (x[:, None] - x_c) ** 2) / w ** 2
                   ).astype(np.complex128)
    assert _input_beam_amp_radius(env_y, dx, centre=(0.0, x_c)) == \
        pytest.approx(w, rel=2e-3)
    on_axis = _gauss_env(n, dx, w)
    assert _input_beam_amp_radius(on_axis, dx, centre=(0.0, 0.0)) == \
        _input_beam_amp_radius(on_axis, dx)
    assert _input_beam_amp_radius(on_axis, dx, centre=None) == \
        _input_beam_amp_radius(on_axis, dx)


def test_off_axis_fit_disc_tracks_the_local_spline_map():
    """FAIL-BEFORE / PASS-AFTER for the kill.

    ``beam_centre=(0, 0)`` reproduces the origin-referenced guard exactly (what
    D1 shipped): its focus centroid walks away from the aliasing-immune local
    spline map.  The beam-referenced default lands on it."""
    c_spline, _ = _offax_focus_centroid({'newton_fit': 'spline'})
    c_origin, _ = _offax_focus_centroid({'beam_centre': (0.0, 0.0)})
    c_beam, _ = _offax_focus_centroid({})
    # the beam-referenced disc reproduces the local map to well under a
    # thousandth of the spot (rms spot radius ~31 um here)
    assert abs(c_beam - c_spline) < 5.0e-8
    # the origin-referenced disc does not -- and by >10x the pass tolerance
    assert abs(c_origin - c_spline) > 5.0e-7
    assert abs(c_origin - c_spline) > 10.0 * abs(c_beam - c_spline)


@pytest.mark.parametrize('x_c_mm', [4.4, 5.6])
def test_engaged_carrier_intersects_the_two_fit_discs(x_c_mm):
    """With the carrier ENGAGED both discs are live and they are NOT
    concentric, so ``min`` of the two radii is not their intersection.

    ``x_c = 4.4 mm`` puts the beam disc partly outside the 4.50 mm
    aperture-relative disc (the intersection tier); ``5.6 mm`` puts it wholly
    outside (the beam-disc-alone fallback, which must NOT degrade to the whole
    launch square).  Both must still reproduce the local spline map."""
    x_c, r_in = x_c_mm * 1e-3, 60e-3
    c_spline, _ = _offax_focus_centroid({'newton_fit': 'spline'},
                                        x_c=x_c, r_in=r_in)
    c_beam, msgs = _offax_focus_centroid({}, x_c=x_c, r_in=r_in)
    assert np.isfinite(c_beam)
    assert abs(c_beam - c_spline) < 5.0e-8
    assert not [m for m in msgs if 'ray-fit domain' in m], msgs


def test_tilted_carrier_supplies_the_beam_centre_by_default():
    """A TiltedCarrier already STATES the chief-ray position, so the element
    takes the beam centre from it -- the chain needs no extra plumbing, and a
    direct ``apply_real_lens_traced`` call with a decentred TiltedCarrier is
    guarded too.  An explicit ``beam_centre`` still wins."""
    n, dx, w = 256, 20e-6, 3e-4
    x_c = 1.2e-3
    x = (np.arange(n) - n // 2) * dx
    env = np.exp(-((x[None, :] - x_c) ** 2 + x[:, None] ** 2) / w ** 2
                 ).astype(np.complex128)
    presc = _singlet(20e-3, -20e-3, 1.2e-3, 'N-BK7', 4e-3, 'p')
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
              n_workers=1, fit_radius_beam_factor=2.0,
              amplitude_model='ray_density', newton_amp_mask_rel=0.0,
              on_undersample='silent', on_noncollimated='silent',
              on_aperture_beam='silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tc = la.TiltedCarrier(np.inf, 0.0, 0.0, x_c, 0.0)
        derived = np.asarray(la.apply_real_lens_traced(env, carrier=tc, **kw))
        explicit = np.asarray(la.apply_real_lens_traced(
            env, carrier=tc, beam_centre=(x_c, 0.0), **kw))
        origin = np.asarray(la.apply_real_lens_traced(
            env, carrier=tc, beam_centre=(0.0, 0.0), **kw))
    scale = float(np.max(np.abs(derived)))
    assert scale > 0.0
    assert np.max(np.abs(derived - explicit)) <= 1e-10 * scale
    # and the origin-referenced guard really is a DIFFERENT computation
    assert np.max(np.abs(derived - origin)) > 1e-6 * scale


def test_on_axis_beam_centre_is_inert():
    """``beam_centre`` must not perturb the shipped on-axis path: omitted,
    ``None`` and ``(0, 0)`` are the same computation (tolerance pin, not
    array_equal -- live FFT/cache calls)."""
    n, dx, w = 256, 20e-6, 3e-4
    env = _gauss_env(n, dx, w)
    presc = _singlet(20e-3, -20e-3, 1.2e-3, 'N-BK7', 4e-3, 'p')
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
              n_workers=1, fit_radius_beam_factor=2.0,
              amplitude_model='ray_density', newton_amp_mask_rel=0.0,
              on_undersample='silent', on_noncollimated='silent',
              on_aperture_beam='silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base = np.asarray(la.apply_real_lens_traced(env, **kw))
        none_ = np.asarray(la.apply_real_lens_traced(
            env, beam_centre=None, **kw))
        zero = np.asarray(la.apply_real_lens_traced(
            env, beam_centre=(0.0, 0.0), **kw))
    scale = float(np.max(np.abs(base)))
    assert scale > 0.0
    assert np.max(np.abs(base - none_)) <= 1e-10 * scale
    assert np.max(np.abs(base - zero)) <= 1e-10 * scale


def test_beam_centre_validation():
    """Bad ``beam_centre`` raises naming the argument, before any ray work."""
    n, dx, w = 64, 20e-6, 3e-4
    env = _gauss_env(n, dx, w)
    presc = _singlet(20e-3, -20e-3, 1.2e-3, 'N-BK7', 1e-3, 'p')
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
              n_workers=1, on_undersample='silent',
              on_noncollimated='silent', on_aperture_beam='silent')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for bad in (1.0, (1e-3,), (1e-3, 2e-3, 3e-3), 'x'):
            with pytest.raises(ValueError, match='beam_centre'):
                la.apply_real_lens_traced(env, beam_centre=bad, **kw)
        for bad in ((np.nan, 0.0), (0.0, np.inf)):
            with pytest.raises(ValueError, match='beam_centre'):
                la.apply_real_lens_traced(env, beam_centre=bad, **kw)


def test_off_axis_fit_domain_shortfall_names_the_off_axis_disc():
    """When the beam-relative disc is too thin to constrain the fit the
    restriction is abandoned -- and the message must say the disc is OFF AXIS,
    or the advice ("lower ray_subsample") reads as a sampling problem on a
    centred disc that does not exist."""
    n, dx, w = 256, 20e-6, 3e-4
    x_c = 1.2e-3
    x = (np.arange(n) - n // 2) * dx
    env = np.exp(-((x[None, :] - x_c) ** 2 + x[:, None] ** 2) / w ** 2
                 ).astype(np.complex128)
    presc = _singlet(20e-3, -20e-3, 1.2e-3, 'N-BK7', 4e-3, 'p')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        la.apply_real_lens_traced(
            env, prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
            n_workers=1, fit_radius_beam_factor=0.02,
            beam_centre=(x_c, 0.0), on_undersample='silent',
            on_noncollimated='silent', on_aperture_beam='warn')
    msgs = [str(m.message) for m in rec if 'ray-fit domain' in str(m.message)]
    assert msgs, [str(m.message)[:80] for m in rec]
    assert 'off the grid centre' in msgs[0]
    assert '1.2000' in msgs[0]        # the decentre, in mm


# ---------------------------------------------------------------------------
# The off-centre fit disc must be applied as WEIGHTS, not as a hard sample mask
# (adversarial kill 2026-07-28, second round).
#
# Restricting a GLOBAL tensor-Chebyshev fit to a small disc leaves the rest of
# the launch domain unconstrained.  While the disc is CONCENTRIC with the
# basis's own domain the leftover freedom inherits the map's radial symmetry
# and the extrapolation stays monotone; off centre it does not, the fitted
# forward map FOLDS, the Newton inverse sends far exit pixels back into the
# bright beam and ``amplitude_model='ray_density'`` gives them real amplitude.
# That is a bright spurious lobe in the RETURNED FIELD -- invisible to any
# centroid metric, and exactly the "populated, credible-looking, power-
# scrambled" failure the roadmap's P1/P3 exist to prevent.
#
# ``_FIT_DISC_OUTSIDE_WEIGHT_REL = 0`` degenerates the weights back to the hard
# mask, so it is the fail-before switch for every test below.
# ---------------------------------------------------------------------------

_GHOST_AP, _GHOST_N, _GHOST_DX = 12e-3, 1024, 20e-6
_GHOST_W, _GHOST_XC, _GHOST_RIN = 0.40e-3, 5.6e-3, 60e-3


def _ghost_metrics(**kw):
    """(off-beam power / input power, max off-beam amplitude / on-beam peak,
    warning messages) for one decentred beam through one fast singlet.

    ``x_c = 5.6 mm`` puts the beam disc wholly outside the aperture-relative
    ``_CARRIER_FIT_RADIUS_FRAC`` disc, i.e. the beam-disc-alone tier."""
    x = (np.arange(_GHOST_N) - _GHOST_N // 2) * _GHOST_DX
    env = np.exp(-((x[None, :] - _GHOST_XC) ** 2 + x[:, None] ** 2)
                 / _GHOST_W ** 2).astype(np.complex128)
    p_in = float(np.sum(np.abs(env) ** 2))
    near = ((x[None, :] - _GHOST_XC) ** 2 + x[:, None] ** 2) <= (3 * _GHOST_W) ** 2
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = np.asarray(la.apply_real_lens_traced(
            env, prescription=_singlet(32e-3, -32e-3, 1.6e-3, 'N-BK7',
                                       _GHOST_AP, 'ghost'),
            wavelength=_WL, dx=_GHOST_DX, ray_subsample=8, n_workers=1,
            carrier=la.TiltedCarrier(_GHOST_RIN, 0.0, 0.0, _GHOST_XC, 0.0),
            fit_radius_beam_factor=2.0, amplitude_model='ray_density',
            preserve_input_phase='remap', remap_sampling='full',
            on_aperture_beam='silent', **_TKW))
    amp = np.abs(out)
    peak = float(amp[near].max())
    assert peak > 0.0
    off_p = float(np.sum(amp[~near] ** 2)) / p_in
    return off_p, float(amp[~near].max()) / peak, [str(m.message) for m in rec]


def test_off_centre_fit_disc_does_not_ghost_the_exit_field(monkeypatch):
    """FAIL-BEFORE / PASS-AFTER for the exit FIELD (not just its centroid).

    Oracle: ``newton_fit='spline'``, the local bicubic interpolant of the same
    traced map -- it needs no fit-domain restriction at all, so whatever halo
    it shows is the real one."""
    import lumenairy.elements._lens_traced as _lt

    x = (np.arange(_GHOST_N) - _GHOST_N // 2) * _GHOST_DX
    env = np.exp(-((x[None, :] - _GHOST_XC) ** 2 + x[:, None] ** 2)
                 / _GHOST_W ** 2).astype(np.complex128)
    p_in = float(np.sum(np.abs(env) ** 2))
    near = ((x[None, :] - _GHOST_XC) ** 2 + x[:, None] ** 2) <= (3 * _GHOST_W) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.abs(np.asarray(la.apply_real_lens_traced(
            env, prescription=_singlet(32e-3, -32e-3, 1.6e-3, 'N-BK7',
                                       _GHOST_AP, 'ghost'),
            wavelength=_WL, dx=_GHOST_DX, ray_subsample=8, n_workers=1,
            carrier=la.TiltedCarrier(_GHOST_RIN, 0.0, 0.0, _GHOST_XC, 0.0),
            fit_radius_beam_factor=2.0, amplitude_model='ray_density',
            preserve_input_phase='remap', remap_sampling='full',
            newton_fit='spline', on_aperture_beam='silent', **_TKW)))
    ref_off_p = float(np.sum(ref[~near] ** 2)) / p_in
    ref_rel = float(ref[~near].max()) / float(ref[near].max())
    assert ref_off_p < 1e-5 and ref_rel < 0.01     # the true halo is nothing

    off_p, rel, _ = _ghost_metrics()
    assert off_p < 10.0 * ref_off_p
    assert rel < 10.0 * ref_rel

    # fail-before: weight 0 outside the disc IS the hard sample mask
    monkeypatch.setattr(_lt, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 0.0)
    bad_p, bad_rel, _ = _ghost_metrics()
    assert bad_p > 1000.0 * ref_off_p, (bad_p, ref_off_p)
    assert bad_rel > 0.1, bad_rel        # a lobe at >10% of the on-beam peak


def test_the_fold_warning_on_an_off_centre_disc_was_a_true_positive(monkeypatch):
    """The ray-density caustic scan is NOT masked away for a decentred beam.

    It fired on exactly the calls that returned the spurious lobe above, so it
    was the only diagnostic the library emitted about a wrong field.  With the
    fit regularised the fold is gone at the source and the same unmasked scan
    is silent -- which is the only acceptable way for that warning to stop."""
    import lumenairy.elements._lens_traced as _lt

    def _folds(msgs):
        return sum('fold caustic' in m for m in msgs)

    _, _, msgs = _ghost_metrics()
    assert _folds(msgs) == 0, msgs

    monkeypatch.setattr(_lt, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 0.0)
    _, bad_rel, bad_msgs = _ghost_metrics()
    assert bad_rel > 0.1
    assert _folds(bad_msgs) >= 1, \
        'the fold detector no longer flags the folded fit -- retune the case'
    assert not hasattr(_lt, '_RAY_DENSITY_SCAN_SUPPORT_REL'), \
        'the caustic scan must not be masked to the beam support again'


def test_cheb2d_weights_are_a_weighted_least_squares():
    """Unit pin on the fit primitive, no optics.

    ``weights=None`` must take the historical unweighted branch exactly, and a
    weight array must solve ``min sum (w_i r_i)^2`` -- checked against an
    independent ``np.linalg.lstsq`` on the row-scaled system."""
    from lumenairy.elements._lens_traced import _Cheb2DEvaluator

    rng = np.random.default_rng(20260728)
    xs = np.linspace(-1.0, 1.0, 21)
    Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
    vals = (0.3 + 1.1 * Xg - 0.7 * Yg + 0.4 * Xg ** 2 * Yg
            + 0.05 * rng.standard_normal(Xg.shape))
    ev_none = _Cheb2DEvaluator(xs, xs, vals, order=3)
    ev_ones = _Cheb2DEvaluator(xs, xs, vals, order=3,
                               weights=np.ones_like(vals))
    scale = float(np.max(np.abs(vals)))
    assert np.max(np.abs(np.asarray(ev_none.coeffs)
                         - np.asarray(ev_ones.coeffs))) <= 1e-10 * scale
    # a genuinely weighted case, against an independent lstsq
    w = np.where(Xg > 0.0, 1.0, 1e-3)
    ev_w = _Cheb2DEvaluator(xs, xs, vals, order=3, weights=w)
    mi = [(a, b) for a in range(4) for b in range(4 - a)]
    A = np.column_stack([np.polynomial.chebyshev.chebval(Xg.ravel(),
                                                         np.eye(4)[a])
                         * np.polynomial.chebyshev.chebval(Yg.ravel(),
                                                           np.eye(4)[b])
                         for a, b in mi])
    ref, *_ = np.linalg.lstsq(A * w.ravel()[:, None], vals.ravel() * w.ravel(),
                              rcond=None)
    assert np.max(np.abs(np.asarray(ev_w.coeffs) - ref)) <= 1e-8 * scale
    assert np.max(np.abs(np.asarray(ev_w.coeffs)
                         - np.asarray(ev_none.coeffs))) > 1e-3 * scale
    with pytest.raises(ValueError, match='weights shape'):
        _Cheb2DEvaluator(xs, xs, vals, order=3, weights=np.ones(3))


def test_multi_broadcasts_one_shared_tilted_carrier():
    """``TiltedCarrier`` is a ``NamedTuple``, i.e. a ``tuple``.

    ``apply_real_lens_traced_multi`` picks 'per-emitter list' vs 'broadcast'
    with an ``isinstance(..., (list, tuple))`` test, so one shared spec used to
    be UNPACKED into its five fields as five per-emitter scalar carriers --
    silently, and only for n == 5 (any other n raised naming a list the caller
    never wrote)."""
    n_grid, dx, w = 256, 20e-6, 3e-4
    x = (np.arange(n_grid) - n_grid // 2) * dx
    presc = _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 6e-3, 'm')
    tc = la.TiltedCarrier(0.200, 0.020, 0.010, 0.5e-3, 0.25e-3)
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
              n_workers=1, **_TKW)
    for n in (2, 5):
        fields = [np.exp(-((x[None, :] - 0.4e-3 * k) ** 2 + x[:, None] ** 2)
                         / w ** 2).astype(np.complex128) for k in range(n)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            one = np.asarray(la.apply_real_lens_traced_multi(
                fields, carriers=tc, **kw))
            many = np.asarray(la.apply_real_lens_traced_multi(
                fields, carriers=[tc] * n, **kw))
        scale = float(np.max(np.abs(many)))
        assert scale > 0.0
        assert np.max(np.abs(one - many)) <= 1e-10 * scale
    # a genuine per-emitter list of the WRONG length still raises
    fields = [fields[0], fields[1]]
    with pytest.raises(ValueError, match='carriers list length'):
        la.apply_real_lens_traced_multi(fields, carriers=[tc] * 3, **kw)
