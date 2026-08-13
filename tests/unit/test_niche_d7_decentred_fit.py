"""The OFF-CENTRE ray fit -- niche D7 (roadmap
ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27, D6 retraction + D7 correction).

Why it exists.  Niche D6 reported that ``apply_real_lens_traced`` under a
DECENTRED carrier invents 3.7 -> 408 urad of exit slope from 0 to 0.97 beam
radii of decentre, and localised it to "the ray-FIT region follows the beam
while the fit BASIS domain stays the axis-centred launch square".  D7 measured
that claim three ways and it does not survive:

* the 408 urad curve was the D6 repro script's own FFT-derivative artefact --
  the same estimator reads 400.51 urad on a SYNTHETIC field whose exit-slope
  error is 0.36 urad by construction;
* measured aliasing-free the element carried 1.28 urad on axis and 7.16 urad at
  0.97 w, i.e. 0.055 um of blur against a 3.5 um FWHM;
* re-mapping the basis domain onto the off-centre disc cannot change anything:
  the total-degree tensor-Chebyshev space is AFFINE-INVARIANT, so the weighted
  least-squares solution is the same polynomial (measured identical to 4-5
  significant figures), while the re-mapped basis blows up outside its own
  domain -- which the Newton loop evaluates.  Refused, with numbers.

What IS real: an off-centre disc of radius ``r`` about a chief ray ``|c|`` off
axis covers the aperture out to ``|c| + r``, so the same total-degree budget
buys a worse fit over strictly more aberrated territory.  D7 raises the fit
order on exactly that branch (``_DECENTRED_FIT_POLY_ORDER``, exposed as
``decentred_fit_poly_order``), stepping back down when the disc cannot
constrain the extra terms.

The oracle here is ANALYTIC and decentre-INVARIANT, and needs no ray trace at
all.  A flat-entrance / ``K = -n^2`` conic-exit singlet images a collimated
bundle in air stigmatically, exactly, to all orders (the Fermat solution), so
every ray of it -- on axis or decentred -- leaves the exit VERTEX plane
pointing at the same focus.  Fermat then fixes the exit-plane optical path
exactly::

    OPL(x, y) - OPL(0, 0) = f_b - sqrt(x^2 + y^2 + f_b^2)

with ``f_b`` the back focal distance from that vertex.  That is the whole
oracle: a closed form, identical for every decentre, sharing no code with the
element.  ``test_the_conic_standin_is_exactly_stigmatic`` pins ``f_b`` and the
stigmatism itself with an inline exact conic raytrace so the oracle cannot
drift either.

The element's phase is compared against it with LOCAL wrapped
nearest-neighbour differences (per-pixel steps are ~1e-2 rad, four orders
inside pi), never a global transform -- that is precisely the estimator error
D6 tripped over.

``decentred_fit_poly_order=<newton_poly_order>`` restores the pre-D7 behaviour
exactly and is the fail-before switch throughout.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced as _lt
from lumenairy import get_glass_index
from lumenairy.elements._lens_traced import _Cheb2DEvaluator

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_GLASS = 'N-BK7'
_F = 3.00e-3                # exit NA = w / f = 0.20
_THICK = 1.5e-3
_APER = 3.40e-3
_W = 0.60e-3                # collimated beam 1/e^2 radius
_X0 = 0.60e-3               # chief-ray decentre == one beam radius
_N, _DX = 512, 8.0e-6       # 4.096 mm grid
_RS = 2                     # coarse pitch 16 um -> ~9.9k in-disc samples
_FRBF = 1.5

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


def _prescription(radius=None, aperture=_APER):
    n = _n_glass()
    r2 = -(n - 1.0) * _F if radius is None else float(radius)
    k2 = -n * n if radius is None else 0.0
    return {'name': 'D7 Fermat singlet', 'aperture_diameter': float(aperture),
            'thicknesses': [_THICK],
            'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0),
                         _surface(r2, _GLASS, 'air', k2)]}


def _gauss(cx=0.0, cy=0.0, n=_N, dx=_DX, w=_W):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2) / (w * w)
                  ).astype(np.complex128)


def _apply(cx=0.0, cy=0.0, pre_d7=False, n=_N, dx=_DX, w=_W, rs=_RS,
           aperture=_APER, radius=None, **kw):
    """One element call, phase-only output (``k0 * OPL`` exactly)."""
    opts = dict(prescription=_prescription(radius, aperture), wavelength=_WL,
                dx=dx, ray_subsample=rs, n_workers=1,
                fit_radius_beam_factor=_FRBF, beam_centre=(cx, cy),
                preserve_input_phase=False, amplitude_model='screen',
                newton_amp_mask_rel=0.0, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent')
    if pre_d7:
        opts['decentred_fit_poly_order'] = int(kw.get('newton_poly_order', 6))
    opts.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            _gauss(cx, cy, n, dx, w), **opts))


# ---------------------------------------------------------------------------
# The analytic, decentre-invariant oracle -- and its own pin.
# ---------------------------------------------------------------------------
def _bfd_by_inline_raytrace(heights):
    """Axis crossing of each collimated ray, by an inline exact conic trace.

    Flat entrance (no refraction at normal incidence), march to the conic by
    Newton on its exact sag, refract with vector Snell, then intersect the
    axis.  No lumenairy anywhere.
    """
    n = _n_glass()
    R = -(n - 1.0) * _F
    K = -n * n
    h = np.atleast_1d(np.asarray(heights, dtype=np.float64))
    # exact even-conic sag and its radial derivative
    r2 = h * h
    q = np.sqrt(1.0 - (1.0 + K) * r2 / (R * R))
    sag = r2 / (R * (1.0 + q))
    dsag = h / (R * q)                       # d(sag)/dh
    # F(x, z) = z - THICK - sag(x) -> grad F = (-dsag, 1); take the normal
    # pointing AGAINST the incident ray, N' = -grad F / |grad F|.
    nrm = np.sqrt(1.0 + dsag * dsag)
    n1x, n1z = dsag / nrm, -1.0 / nrm
    # vector Snell for d = (0, 1) inside the glass leaving into air
    mu = n / 1.0
    c1 = -n1z                                 # = -d . N' > 0
    c2 = np.sqrt(1.0 - mu * mu * (1.0 - c1 * c1))
    dx_o = (mu * c1 - c2) * n1x
    dz_o = mu * 1.0 + (mu * c1 - c2) * n1z
    z_hit = _THICK + sag
    with np.errstate(divide='ignore', invalid='ignore'):
        z_axis = z_hit - h * dz_o / dx_o      # where x reaches 0
    return z_axis - _THICK                    # from the REAR VERTEX


def _oracle_opl(X, Y, f_b):
    """Exit-vertex-plane OPL, referenced to the axis.  Exact, closed form."""
    return f_b - np.sqrt(X * X + Y * Y + f_b * f_b)


def _exit_slope_rms(E, cx, f_b, cy=0.0, dx=_DX, n=_N, w=_W):
    """Aliasing-free rms exit-slope error over the beam CORE (rad).

    ``arg(E) - k0 * OPL_exact`` differenced between NEIGHBOURING pixels with a
    2*pi wrap, then piston + tilt removed (pointing is not an aberration).
    Restricted to ``r <= w`` about the beam: outside the ray-FIT disc the
    low-order fit legitimately extrapolates, and the element documents that it
    does -- measuring there would grade the extrapolation, not the fit.
    """
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    psi = np.angle(E * np.exp(-1j * _K0 * _oracle_opl(X, Y, f_b)))
    amp = np.abs(E)
    d = np.angle(np.exp(1j * (psi[:, 1:] - psi[:, :-1]))) / (_K0 * dx)
    Xm = 0.5 * (X[:, 1:] + X[:, :-1])
    Ym = 0.5 * (Y[:, 1:] + Y[:, :-1])
    thr = 1e-3 * amp.max()
    keep = ((amp[:, 1:] > thr) & (amp[:, :-1] > thr)
            & (np.hypot(Xm - cx, Ym - cy) <= w))
    assert keep.sum() > 500, 'the illuminated core vanished'
    wt = (0.5 * (amp[:, 1:] + amp[:, :-1])) ** 2
    B = np.stack([np.ones(int(keep.sum())), Xm[keep] - cx, Ym[keep] - cy],
                 axis=1)
    q, ww = d[keep], wt[keep]
    Bw = B * ww[:, None]
    res = q - B @ np.linalg.solve(Bw.T @ B, Bw.T @ q)
    return float(np.sqrt((ww * res ** 2).sum() / ww.sum()))


def test_the_conic_standin_is_exactly_stigmatic():
    """The oracle's own pin: every collimated ray of the ``K = -n^2`` conic
    crosses the axis at the SAME point, so the exit wavefront is exactly the
    sphere ``_oracle_opl`` assumes -- for any sub-aperture, hence for any
    decentre.  Inline exact conic raytrace, no lumenairy."""
    z = _bfd_by_inline_raytrace(np.linspace(0.05e-3, 1.70e-3, 41))
    assert np.ptp(z) < 1e-9, (
        f"the stand-in stopped being stigmatic: axis crossings span "
        f"{np.ptp(z) * 1e9:.3f} nm")
    # ... and that point is the back focal distance the oracle uses
    assert abs(float(z.mean()) - _F) < 1e-9, (
        f"back focal distance {float(z.mean()) * 1e6:.4f} um vs f = "
        f"{_F * 1e6:.4f} um")


# ===========================================================================
# 1.  FAIL-BEFORE / PASS-AFTER on the exit wavefront itself.
# ===========================================================================
@pytest.mark.parametrize('frac', [0.5, 1.0])
def test_the_off_centre_fit_order_raise_flattens_the_exit_wavefront(frac):
    """The element's exit slope against the ANALYTIC sphere, at decentre and
    on axis, pre-D7 and post-D7.

    Measured 2026-07-29 (``ray_subsample=1``, so no coarse-grid OPL upsample
    sits between the fit and the phase): **41.1 urad on axis** -- the
    concentric order-6 fit, which D7 deliberately does not touch -- against
    **134.8 / 118.3 urad pre-D7** and **2.4 / 3.4 urad post-D7** at 0.5 w and
    1.0 w of decentre.  Pinned as ratios to the on-axis figure so the test
    does not encode this machine's exact float noise.

    ``ray_subsample=1`` matters: at ``sub > 1`` without an engaged carrier the
    OPL is LINEARLY upsampled from the coarse Newton grid, and that
    interpolation's own sawtooth derivative (~1300 urad here) swamps the fit
    error a per-pixel slope estimator is trying to see.  That is a different
    quantity, and R7's cubic upsample already addresses it."""
    _ram_guard()
    f_b = float(_bfd_by_inline_raytrace([0.3e-3]).mean())
    cx = frac * _W
    on_axis = _exit_slope_rms(_apply(0.0, rs=1), 0.0, f_b)
    before = _exit_slope_rms(_apply(cx, pre_d7=True, rs=1), cx, f_b)
    after = _exit_slope_rms(_apply(cx, rs=1), cx, f_b)
    assert on_axis < 1.0e-4, (
        f"the on-axis path regressed: {on_axis * 1e6:.3f} urad")
    # fail-before: the pre-D7 off-centre fit is materially worse than on axis
    assert before > 2.0 * on_axis, (
        f"the fail-before switch stopped failing: decentred {before * 1e6:.3f} "
        f"urad vs on-axis {on_axis * 1e6:.3f} urad")
    # pass-after: the raised order beats even the concentric order-6 fit
    assert after < 0.25 * on_axis, (
        f"decentred exit slope {after * 1e6:.3f} urad against on-axis "
        f"{on_axis * 1e6:.3f} urad")
    assert after < 0.1 * before


def test_the_fit_order_actually_rises_only_off_centre():
    """Instrument the fit itself: the order handed to ``_Cheb2DEvaluator`` is
    ``newton_poly_order`` on the concentric path and
    ``_DECENTRED_FIT_POLY_ORDER`` off centre.

    ``seen[-3:]`` rather than ``seen``: niche C11's arbiter builds two TRIAL
    OPL fits before the three the Newton inversion is handed, and this pin is
    about the latter.  Nothing else changed -- the assertions and their
    thresholds are the originals."""
    _ram_guard()
    seen = []
    orig = _Cheb2DEvaluator.__init__

    def spy(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
        orig(self, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        seen.append((int(order), weights is not None))

    _Cheb2DEvaluator.__init__ = spy
    try:
        seen.clear()
        _apply(0.0)
        seen[:] = seen[-3:]
        assert seen and all(o == 6 for o, _w in seen), seen
        assert not any(w for _o, w in seen), 'the on-axis disc used weights'
        seen.clear()
        _apply(_X0)
        seen[:] = seen[-3:]
        assert seen and all(o == _lt._DECENTRED_FIT_POLY_ORDER
                            for o, _w in seen), seen
        assert all(w for _o, w in seen), 'the off-centre disc lost its weights'
        seen.clear()
        _apply(_X0, pre_d7=True)
        seen[:] = seen[-3:]
        assert seen and all(o == 6 for o, _w in seen), seen
        # a caller asking for MORE still gets more
        seen.clear()
        _apply(_X0, newton_poly_order=14)
        seen[:] = seen[-3:]
        assert seen and all(o == 14 for o, _w in seen), seen
    finally:
        _Cheb2DEvaluator.__init__ = orig


def test_the_order_steps_down_when_the_disc_cannot_constrain_it():
    """Order 10 needs 66 basis terms.  When the off-centre disc holds fewer
    than 3 coarse samples per term the raise must step back down instead of
    handing the solver an under-determined normal matrix -- otherwise
    ``_CARRIER_FIT_MIN_SAMPLES`` (64) would admit an order-10 fit on 64 rows."""
    _ram_guard()
    seen = []
    orig = _Cheb2DEvaluator.__init__

    def spy(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
        orig(self, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        seen.append(int(order))

    _Cheb2DEvaluator.__init__ = spy
    try:
        # ray_subsample 16 leaves ~O(100) samples inside the same disc
        seen.clear()
        E = _apply(_X0, rs=16)
        n_terms = [(o + 1) * (o + 2) // 2 for o in seen]
        assert seen, 'no polynomial fit was built'
        assert max(seen) < _lt._DECENTRED_FIT_POLY_ORDER, (
            f"order {max(seen)} survived a disc that cannot constrain it "
            f"({max(n_terms)} terms)")
        assert min(seen) >= 6, seen
        assert np.isfinite(E).all()
    finally:
        _Cheb2DEvaluator.__init__ = orig


# ===========================================================================
# 2.  The concentric / on-axis path is untouched.
# ===========================================================================
@pytest.mark.parametrize('kw', [
    {},
    {'newton_poly_order': 10},
    {'inversion_method': 'fit'},
    {'carrier': 60e-3},
])
def test_the_concentric_path_is_unchanged_by_d7(kw):
    """Tolerance pin (not ``array_equal`` -- live FFT / cache calls): with the
    fit disc CONCENTRIC, the D7 default and the fail-before switch are the same
    computation."""
    _ram_guard()
    a = _apply(0.0, **kw)
    b = _apply(0.0, pre_d7=True, **kw)
    scale = float(np.max(np.abs(a)))
    assert scale > 0.0
    assert np.max(np.abs(a - b)) <= 1e-10 * scale


def test_a_tilted_but_centred_carrier_is_still_the_concentric_path():
    """The raise keys on the DISC being off centre, not on the carrier being
    tilted: a 20 mrad tilt with the beam on the grid centre must not move."""
    _ram_guard()
    car = la.TiltedCarrier(np.inf, 0.02, 0.0, 0.0, 0.0)
    a = _apply(0.0, carrier=car)
    b = _apply(0.0, carrier=car, pre_d7=True)
    scale = float(np.max(np.abs(a)))
    assert scale > 0.0
    assert np.max(np.abs(a - b)) <= 1e-10 * scale


def test_the_spline_fit_takes_no_fit_disc_so_d7_cannot_touch_it():
    """``newton_fit='spline'`` skips the fit-domain restriction entirely, so
    even a DECENTRED disc must be byte-identical across the switch.

    Run on the WEAK singlet: ``RectBivariateSpline`` needs a NaN-free regular
    grid, and the Fermat conic total-internal-reflects the outer launch rays
    (the launch square is 1.5x the aperture by design), which NaNs it out.
    That is pre-existing behaviour, not a D7 effect."""
    _ram_guard()
    a = _ghost_apply(newton_fit='spline')
    b = _ghost_apply(newton_fit='spline', pre_d7=True)
    scale = float(np.max(np.abs(a)))
    assert scale > 0.0
    assert np.max(np.abs(a - b)) <= 1e-10 * scale


def test_the_decentred_path_really_did_change():
    """The complement of the pins above -- otherwise they would pass on a
    no-op."""
    _ram_guard()
    a = _apply(_X0)
    b = _apply(_X0, pre_d7=True)
    scale = float(np.max(np.abs(a)))
    assert np.max(np.abs(a - b)) > 1e-8 * scale


# ===========================================================================
# 3.  No fold, no ghost -- D1's failure must not come back with the terms.
# ===========================================================================
# D1's own ghost geometry: a WEAK singlet with a large aperture:beam ratio, so
# the off-beam region really is empty and a fold shows up as a bright lobe
# rather than as the legitimate ray-density spike a fast conic puts at its own
# aperture edge.  ``ray_subsample=2`` keeps ~1.2k coarse samples inside the
# off-centre disc, enough to constrain the D7 order.
_GA, _GN, _GDX, _GW, _GXC = 12e-3, 1024, 20e-6, 0.40e-3, 5.6e-3


def _ghost_apply(cx=_GXC, cy=0.0, pre_d7=False, rs=2, radius=32e-3, **kw):
    opts = dict(prescription=_prescription(radius, _GA), wavelength=_WL,
                dx=_GDX, ray_subsample=rs, n_workers=1,
                fit_radius_beam_factor=2.0,
                carrier=la.TiltedCarrier(60e-3, 0.0, 0.0, cx, cy),
                amplitude_model='ray_density', preserve_input_phase='remap',
                remap_sampling='full', on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent')
    if pre_d7:
        opts['decentred_fit_poly_order'] = int(kw.get('newton_poly_order', 6))
    opts.update(kw)
    return np.asarray(la.apply_real_lens_traced(
        _gauss(cx, cy, _GN, _GDX, _GW), **opts))


_ADVERSARIAL = [
    ('x+', dict(cx=_GXC)),
    ('x-', dict(cx=-_GXC)),
    ('y+', dict(cx=0.0, cy=_GXC)),
    ('diag', dict(cx=4.0e-3, cy=4.0e-3)),
    ('rs4', dict(cx=4.4e-3, rs=4)),
    ('rs8', dict(cx=_GXC, rs=8)),
    ('rs16', dict(cx=_GXC, rs=16)),
    ('R+200', dict(cx=_GXC, radius=200e-3)),
    ('R-200', dict(cx=_GXC, radius=-200e-3)),
    ('edge', dict(cx=5.95e-3)),
    ('workers4', dict(cx=_GXC, n_workers=4)),
    ('spline', dict(cx=_GXC, newton_fit='spline')),
]


@pytest.mark.parametrize('label,kw', _ADVERSARIAL,
                         ids=[c[0] for c in _ADVERSARIAL])
def test_no_fold_and_no_ghost_across_the_adversarial_geometries(label, kw):
    """D1's kill re-run at the D7 order.  A folded forward map shows up three
    ways and all three must stay clean: a sign change in ``d(x_out)/d(x_in)``
    over the fitted map, a spurious off-beam lobe in the returned field (D1
    measured 0.17-0.91 of the on-beam peak against the 6e-4 the unrestricted
    spline map shows), and the ray-density caustic scan."""
    _ram_guard()
    cx = kw.get('cx', 0.0)
    cy = kw.get('cy', 0.0)
    seen = []
    orig = _Cheb2DEvaluator.__init__

    def spy(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
        orig(self, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        seen.append((self, np.asarray(xs_in)))

    _Cheb2DEvaluator.__init__ = spy
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            E = _ghost_apply(**kw)
    finally:
        _Cheb2DEvaluator.__init__ = orig
    msgs = [str(m.message) for m in rec]
    assert not [m for m in msgs if 'fold caustic' in m], msgs
    x = (np.arange(_GN) - _GN // 2) * _GDX
    near = ((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2) <= (3 * _GW) ** 2
    amp = np.abs(E)
    peak = float(amp[near].max())
    assert peak > 0.0
    assert float(amp[~near].max()) / peak < 0.02, (
        f"{label}: off-beam lobe at {float(amp[~near].max()) / peak:.3f} of "
        f"the on-beam peak")
    p_in = float(np.sum(np.abs(_gauss(cx, cy, _GN, _GDX, _GW)) ** 2))
    assert float(np.sum(amp[~near] ** 2)) / p_in < 1e-5
    if kw.get('newton_fit', 'polynomial') == 'polynomial':
        # The APPLIED forward-map fit, which is the LAST THREE builds
        # (``x_out``, ``y_out``, ``opl``).  It used to be ``seen[0]`` because
        # a call built exactly three evaluators; niche C11's arbiter builds two
        # TRIAL OPL fits ahead of them, and an OPL has a vertex, so reading
        # ``seen[0]`` scores the sign changes of ``d(OPL)/dx`` -- ~450-570 of
        # them on this lattice, by construction and about nothing.  Measured on
        # the ``x+`` case: ``seen[0]`` reports 491 sign changes and the applied
        # ``x_out`` fit reports 0.  The claim is unchanged; the handle moved.
        assert len(seen) >= 3, seen
        Sx, xs = seen[-3]
        Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
        _v, jxx, _jy = Sx.ev_value_and_grad(Xg.ravel(), Yg.ravel())
        sgn = np.sign(jxx.reshape(Xg.shape))
        n_sc = int(np.sum(sgn[1:, :] * sgn[:-1, :] < 0)
                   + np.sum(sgn[:, 1:] * sgn[:, :-1] < 0))
        assert n_sc == 0, f"{label}: the fitted forward map folds ({n_sc})"


# ===========================================================================
# 3b.  THE SOLVE CENSUS -- the deterministic handle on the hard-mask arm
# ===========================================================================
# WHY THE OFF-BEAM MAGNITUDE IS NOT AN ASSERTABLE OBSERVABLE, and what is.
#
# The three tests below all degenerate the ray-fit restriction to D1's hard
# NaN mask.  That leaves the 28-term decentred coordinate fit EXACTLY singular
# in the normal equations (``_gram_rcond(A^T A) == 0.0`` -- measured, not
# estimated), so the returned coefficients are a NULL-SPACE DRAW.  The size of
# the ghost such a draw makes is then decided by the last bits of the data,
# and that is measurable rather than asserted.  Measured on this fixture,
# ``LSTSQ_CONDITIONING_STEPDOWN`` off, degree 4, off-beam fraction of peak,
# nudging the carrier decentre by whole ULPs (1 ULP on 5.6 mm = 8.674e-19 m,
# i.e. sub-femtometre):
#
#     +0 ULP  1.7549e-04      +1 ULP  9.4155e-01      +2 ULP  9.7849e-01
#     +3 ULP  6.7412e-01      +4 ULP  9.4576e-01        spread 5576x
#
# and with ``LSTSQ_CONDITIONING_STEPDOWN`` ON, the same five nudges read
# 1.7548444178e-04 .. 1.7548444187e-04 -- stable to NINE significant figures.
# The same experiment run by adding IID noise of 1e-18 m (0.075 of eps*|R|)
# to the carrier eikonal flips the answer across the 0.1 bar on a change of
# RNG SEED alone.  The historical bar ``r_old > 0.1`` was therefore a bar on a
# coin flip: it recorded four build answers spanning four orders of magnitude
# in the docstrings below, and this branch's `sqrt(r^2+R^2)-|R|`
# rationalization -- worth 1.21e-17 m on this fixture -- produced a fifth,
# 1.75e-04, which is under the floor.  The magnitude is now RECORDED
# everywhere and asserted NOWHERE.
#
# What IS deterministic, on every build, is the draw itself.  ``_solve_census``
# scores each least-squares solve the call makes on the quantity the fit is
# DEFINED by, against a backward-stable QR answer to the same system:
#
#   config                          min rcond     max ||b-Ax||/||b-Ax_qr||
#   hard mask, pre-C13               0.0e+00           36972.83
#   hard mask, pre-C13, +1 ULP       0.0e+00          233325.49
#   hard mask, C13 ON                0.0e+00               1.000000
#   weighted disc, pre-C13           1.3e-11               1.000003
#   weighted disc, C13 ON (shipped)  1.3e-11               1.000000
#
# Four orders of separation, no ties, and it says exactly what the ghost was
# always a proxy for: the hard mask makes the solve return an answer that is
# not the least-squares answer, and EITHER cure -- the weighted disc (D1) or
# the C13 step-down -- makes it return one that is.


def _solve_census(**kw):
    """One ``_ghost_apply`` with the least-squares solver instrumented.

    Returns ``(field, folds, rows)`` where each row is
    ``(gram_rcond, ||b - A x_returned|| / ||b - A x_qr||, A.shape)`` for one
    solve.  The instrument only OBSERVES: it delegates to the shipped solver
    and returns its answer untouched, then scores it against an independent
    QR solve of the same system.  A ratio of 1.0 means the shipped path
    returned the least-squares answer; >> 1 means it returned a null-space
    draw instead."""
    rows = []
    orig = _lt._solve_lstsq_thread_safe

    def _instrumented(A, b):
        x = orig(A, b)
        A64 = np.ascontiguousarray(A, dtype=np.float64)
        b64 = np.asarray(b, dtype=np.float64)
        r_x = _lt._lstsq_residual(A64, b64, x)
        r_q = _lt._lstsq_residual(A64, b64, _lt._solve_lstsq_qr(A64, b64))
        rows.append((_lt._gram_rcond(A64.T @ A64),
                     (r_x / r_q) if r_q > 0.0 else np.inf, A64.shape))
        return x

    _lt._solve_lstsq_thread_safe = _instrumented
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            a = np.abs(_ghost_apply(**kw))
    finally:
        _lt._solve_lstsq_thread_safe = orig
    folds = sum('fold caustic' in str(w.message) for w in rec)
    return a, folds, rows


def _worst_draw(rows):
    """``(min rcond, max residual ratio)`` over a census."""
    assert rows, ('the call made no least-squares solve at all -- the census '
                  'is instrumenting a function the fit no longer goes through')
    return (min(r[0] for r in rows), max(r[1] for r in rows))


def _offbeam_ratio(a, cx=_GXC, cy=0.0):
    x = (np.arange(_GN) - _GN // 2) * _GDX
    near = ((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2) <= (3 * _GW) ** 2
    return float(a[~near].max()) / float(a[near].max())


#: A returned fit that misses the least-squares residual by more than this is
#: a null-space draw, not a fit.  Measured minimum on the arms below is
#: 3240.5x and the maximum is 233325.5x, so this floor sits 324x under the
#: smallest draw ever observed here, while the cured arms read 1.000000 and
#: 1.000003 -- i.e. the bar has three decades of room on BOTH sides and does
#: not need to know which draw a given build lands on.
_DRAW_RESID_RATIO = 10.0

#: The cured arms return the QR answer or something that ties with it.  The
#: library's own tie margin is ``_LSTSQ_RESID_MARGIN`` = 1e-6; 1.001 is three
#: decades above it and four decades under the smallest draw.
_CURED_RESID_RATIO = 1.001


def test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order(
        monkeypatch):
    """The extra terms must not have made ``_FIT_DISC_OUTSIDE_WEIGHT_REL``
    redundant OR insufficient: with the weights degenerated back to D1's hard
    mask the same call folds and ghosts, and with them on it does not.

    ERA-PINNED to ``_REMAP_RESID_EIKONAL_DEGREE = 4`` (niche C10) AND to
    ``LSTSQ_CONDITIONING_STEPDOWN = False`` (niche C13, 2026-08-03), which
    together are the library state this case was calibrated in.  On the C13
    pin specifically: this fixture degenerates the restriction to D1's hard NaN
    mask, whose normal matrix its own sibling below calls "ill-conditioned BY
    CONSTRUCTION" -- and that is exactly the solve C13 made stable, so with the
    step-down on THE FOLD DOES NOT HAPPEN AT ALL and the witness has nothing to
    witness.  Measured on this fixture at the pinned degree 4, hard mask,
    off-beam fraction of peak: **0.5213 with the step-down off, 0.0002 with it
    on**, and the fold-caustic warning goes from 1 to 0.  The cure is asserted
    directly in ``test_c13_cures_the_hard_mask_fold_at_the_d7_order`` below;
    what is pinned HERE is unchanged.
    which is the library state this case was calibrated in.  C10 raised that
    degree to 6, and a better model of the input residual removes THIS
    FIXTURE's fold even on the hard mask -- the witness stops witnessing
    because the thing it witnesses has partly gone away.  The assertions below
    are kept WORD FOR WORD; the SHIPPED era is scored separately, as a
    measurement rather than a requirement, in the sibling test below.  Nothing
    is relaxed, and the guard is still load-bearing where it matters: on
    design 121's real chain, degenerating this weight from 1e-8 to 1e-4 costs
    41 EE3 points (docs/audits/D121_RESIDUAL_CLOSURE_2026_08_02.md S5.2).

    RE-ARMED 2026-08-12 (fix/verify-arch CI reconciliation).  The fail-before
    used to be ``r_bad > 0.1`` -- the off-beam magnitude of the hard-mask arm.
    That magnitude is a NULL-SPACE DRAW and is decided at 1e-18 m; see the
    ULP ladder in section 3b above, where five sub-femtometre nudges of the
    same fixture read 1.75e-04 / 0.942 / 0.978 / 0.674 / 0.946.  It is now
    recorded and not asserted, and the load-bearing claim is made where it is
    deterministic: WITH the weighted restriction the normal equations return
    the least-squares answer (residual ratio 1.000003 against QR); with the
    mask degenerated they return one that fits 3.7e+04 times worse.  That is
    what "the regularisation is load-bearing" means, stated so that no build
    can pass it by landing on the lucky side of the instability."""
    _ram_guard()
    monkeypatch.setattr(_lt, '_REMAP_RESID_EIKONAL_DEGREE', 4)
    monkeypatch.setattr(_lt, 'LSTSQ_CONDITIONING_STEPDOWN', False)
    good, _f_good, rows_good = _solve_census()
    monkeypatch.setattr(_lt, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 0.0)
    bad, folds_bad, rows_bad = _solve_census()
    r_good = _offbeam_ratio(good)
    r_bad = _offbeam_ratio(bad)
    rc_good, ratio_good = _worst_draw(rows_good)
    rc_bad, ratio_bad = _worst_draw(rows_bad)

    # the shipped restriction still produces a clean field ...
    assert r_good < 0.01, r_good
    # ... and, deterministically, a fit that IS the least-squares fit
    assert ratio_good <= _CURED_RESID_RATIO, (
        f"the weighted restriction stopped conditioning the solve: its worst "
        f"returned fit misses the least-squares residual by {ratio_good:.6f}x "
        f"(rcond {rc_good:.3e})")

    # degenerate it to D1's hard NaN mask and the same solve goes singular and
    # answers with a draw.  THIS is the fail-before.
    assert rc_bad < _lt._LSTSQ_GRAM_RCOND_MIN, (
        f"the hard mask no longer makes the decentred fit singular "
        f"(rcond {rc_bad:.3e} >= {_lt._LSTSQ_GRAM_RCOND_MIN:.1e}) -- the "
        f"restriction has nothing left to regularise, so retune the case")
    assert ratio_bad > _DRAW_RESID_RATIO, (
        f"the hard-mask fail-before stopped failing at the D7 order: its "
        f"worst returned fit misses the least-squares residual by only "
        f"{ratio_bad:.4f}x (off-beam {r_bad:.4f} of peak) -- retune the case")
    assert ratio_bad > 100.0 * ratio_good, (ratio_bad, ratio_good)
    assert folds_bad >= 1, 'the hard-mask arm stopped folding'
    # r_bad is RECORDED, never bounded -- section 3b.
    assert np.isfinite(r_bad), r_bad


def test_the_hard_mask_arm_ghosts_on_every_build(monkeypatch):
    """The era-pinned sibling above degenerates the regularisation and asserts
    the result ghosts.  This checks that PREMISE independently of the era pin,
    and records what the shipped residual-eikonal degree does to the same
    fixture -- which is why that sibling is era-pinned in the first place.

    It says nothing about whether the restriction is still needed (it is --
    see the sibling's docstring, and the 41-72 EE3 points it costs on design
    121 when degenerated there).

    Niche C11 (2026-08-03) REMOVED the shrink assertion entirely, and the
    reason is a measurement across three builds rather than a judgement.

    This fixture DELIBERATELY degenerates the regularisation to D1's hard NaN
    mask, whose normal matrix is ill-conditioned BY CONSTRUCTION.  The size of
    the ghost that survives such a solve is therefore not a stable observable:
    it is set by which side of the instability that build's LAPACK lands on.
    Measured on the same tree, same fixture, same source:

        Windows / MKL           0.35   -> 1.8e-04     shrink ~1900x
        WSL Linux / OpenBLAS    0.9970 -> 0.5216      shrink    1.9x
        CI  Linux / OpenBLAS    ~1.0   -> 0.9998      shrink    1.0x

    Three builds, three answers spanning FOUR ORDERS OF MAGNITUDE, including
    one that shows no shrink at all.  Two successive numeric bars (``0.1x``,
    then ``0.8x``) each passed the builds they were calibrated on and failed
    the next one, which is the definition of a quantity that must not carry an
    assertion.  **No bar on this magnitude can be both meaningful and true**,
    so the magnitude is now RECORDED here and asserted nowhere.

    What is still asserted is the part that is build-invariant and that the
    era-pinned sibling above actually depends on -- but it is no longer the
    ghost.  RE-ARMED 2026-08-12 (fix/verify-arch CI reconciliation): the
    remaining bar ``r_old > 0.1`` was itself a bar on the same unstable
    magnitude, and a FOURTH build answer has now arrived under it.  This
    branch's ``sqrt(r^2+R^2) - |R|`` rationalization moves the carrier eikonal
    by 1.214e-17 m on this fixture (eps*|R| = 1.332e-17 m) and the arm read
    1.7549e-04.  Bisected to one line, ``_lens_traced._tilted_carrier_parts``
    line 3848; reverting that line alone restores 5.2129148080e-01 to ten
    digits, and reverting the sibling ``_compute_carrier`` hunk changes
    nothing.  Sub-femtometre ULP nudges of the fixture's own decentre
    reproduce the whole four-order spread IN ONE PROCESS (section 3b).

    So the test name is kept -- it is the ledger entry for this arm -- and
    what it asserts is the mechanism the ghost was always a proxy for: on
    every build, the degree-4 hard-mask arm returns a fit that is NOT the
    least-squares fit.  That is a property of the singular normal equations,
    not of which side of them a build lands on, and it is the sibling's real
    premise.

    (This was one of the v5.32.1 CI failures at ``5af1edf`` and it PREDATES
    niche C11: driven directly with the flag pinned each way in one process it
    fails identically with ``DECENTRED_FIT_ARBITER`` True and False.)

    ERA-PINNED to ``LSTSQ_CONDITIONING_STEPDOWN = False`` (niche C13,
    2026-08-03).  The docstring above diagnosed the mechanism one step short of
    the answer: "whose normal matrix is ill-conditioned BY CONSTRUCTION ... set
    by which side of the instability that build's LAPACK lands on" IS C13's
    finding, and C13 removed the instability.  With the step-down on, the
    degree-4 hard-mask arm no longer ghosts at all (0.5213 -> 0.0002 off-beam),
    so the four-orders-of-magnitude build spread recorded above is not a
    property of the library any more -- it is a property of the library BEFORE
    C13, which is what this arm now pins.  The cure is asserted separately
    below; the assertions here are unchanged.
    """
    _ram_guard()
    monkeypatch.setattr(_lt, 'LSTSQ_CONDITIONING_STEPDOWN', False)
    monkeypatch.setattr(_lt, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 0.0)
    # This test DELIBERATELY builds the halo the v5.32 self-check exists to
    # report, so its firing here is the fixture working, not a finding -- and
    # leaving it on would put the campaign's "zero HALO self-check firings
    # across the niche suites" property at the mercy of a test that asserts
    # the halo is there.
    monkeypatch.setattr(_lt, 'RAY_DENSITY_HALO_CHECK', 'silent')
    with monkeypatch.context() as m:
        m.setattr(_lt, '_REMAP_RESID_EIKONAL_DEGREE', 4)
        old, _folds_old, rows_old = _solve_census()
    new = np.abs(_ghost_apply())
    r_old = _offbeam_ratio(old)
    r_new = _offbeam_ratio(new)
    rc_old, ratio_old = _worst_draw(rows_old)

    # THE BUILD-INVARIANT PART: the degree-4 hard-mask arm is live because its
    # normal equations are singular and answer with a draw -- not because the
    # draw happened to be big on this build.
    assert rc_old < _lt._LSTSQ_GRAM_RCOND_MIN, (
        f"the degree-4 hard-mask arm is not live: its worst Gram is no "
        f"longer singular (rcond {rc_old:.3e}), so there is no instability "
        f"for the era pin to sit on")
    assert ratio_old > _DRAW_RESID_RATIO, (
        f"the degree-4 hard-mask arm is not live: the fit it returns misses "
        f"the least-squares residual by only {ratio_old:.4f}x, i.e. the "
        f"normal equations are answering it correctly "
        f"(off-beam {r_old:.6f} of peak)")
    # r_old AND r_new are RECORDED, not bounded -- see the docstring and
    # section 3b.  The only thing asserted about the fields is that the arm
    # still returns a usable one on this deliberately degenerate fixture
    # rather than NaN or nothing.
    assert np.isfinite(r_old) and np.isfinite(r_new), (r_old, r_new)
    assert float(new.max()) > 0.0 and float(old.max()) > 0.0


def test_c13_cures_the_hard_mask_fold_at_the_d7_order(monkeypatch):
    """The PASS-AFTER for the two era-pinned witnesses above, asserted rather
    than only recorded.

    Both of them degenerate the restriction to D1's hard NaN mask and require
    the result to ghost.  It no longer does, and the reason is niche C13: that
    hard-mask design matrix is near-singular, the normal equations answered it
    with a null-space draw, and the draw is what folded the inverse map.  With
    ``LSTSQ_CONDITIONING_STEPDOWN`` on, the same call is solved by a
    backward-stable QR and there is no fold to make a ghost from.

    So D1's weighted restriction and C13's stable solve are INDEPENDENT cures
    for the same defect, and this fixture now shows the second one.  It is a
    strictly stronger statement than the era-pinned arms make, which is why it
    is asserted here instead of being folded into them.

    (This says nothing about whether the restriction is still needed -- it is;
    see the siblings' docstrings and the 41-72 EE3 points it costs on design
    121 when degenerated there.)

    RE-ARMED 2026-08-12 (fix/verify-arch CI reconciliation).  The cure used to
    be scored as ``r_pre > 0.1`` and ``r_pre > 100 r_post`` -- a RATIO OF TWO
    GHOST MAGNITUDES, one of which is a null-space draw (section 3b).  It is
    now scored on the thing C13 actually changes, which is deterministic and
    which the docstring above already names: the SOLVE.  Both arms leave the
    Gram equally singular (rcond 0.0 either way -- C13 does not condition the
    matrix, it stops trusting the normal equations); pre-C13 the returned fit
    misses the least-squares residual by 3.7e+04x, and with C13 on it reads
    1.000000.  The fold and ghost assertions on the POST arm are unchanged,
    because those are the arm where the answer is stable to nine figures."""
    _ram_guard()
    monkeypatch.setattr(_lt, 'RAY_DENSITY_HALO_CHECK', 'silent')
    monkeypatch.setattr(_lt, '_REMAP_RESID_EIKONAL_DEGREE', 4)
    monkeypatch.setattr(_lt, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 0.0)

    with monkeypatch.context() as m:
        m.setattr(_lt, 'LSTSQ_CONDITIONING_STEPDOWN', False)
        a_pre, folds_pre, rows_pre = _solve_census()
    a_post, folds_post, rows_post = _solve_census()
    r_pre, r_post = _offbeam_ratio(a_pre), _offbeam_ratio(a_post)
    rc_pre, ratio_pre = _worst_draw(rows_pre)
    rc_post, ratio_post = _worst_draw(rows_post)

    # the fail-before is live: the pre-C13 solve answers a singular system
    # with a draw, and the draw folds the fitted forward map
    assert rc_pre < _lt._LSTSQ_GRAM_RCOND_MIN, (
        f'the pre-C13 arm is no longer singular (rcond {rc_pre:.3e})')
    assert ratio_pre > _DRAW_RESID_RATIO, (
        f'the pre-C13 arm stopped drawing: its returned fit misses the '
        f'least-squares residual by only {ratio_pre:.4f}x '
        f'(off-beam {r_pre:.6f} of peak)')
    assert folds_pre >= 1, 'the pre-C13 arm stopped folding'

    # ... and the shipped solve returns the least-squares answer to the SAME
    # singular system, and does neither
    assert rc_post < _lt._LSTSQ_GRAM_RCOND_MIN, (
        f'C13 is being credited with conditioning the matrix, which it does '
        f'not do (rcond {rc_post:.3e}) -- it re-solves, and that is the claim')
    assert ratio_post <= _CURED_RESID_RATIO, (
        f'C13 no longer returns the least-squares answer: {ratio_post:.6f}x')
    assert ratio_pre > 100.0 * ratio_post, (ratio_pre, ratio_post)
    assert r_post < 0.01, f'C13 no longer removes the ghost ({r_post:.4f})'
    assert folds_post == 0, 'the fold detector still fires with C13 on'
    # r_pre is RECORDED, never bounded -- section 3b.
    assert np.isfinite(r_pre), r_pre


def test_the_off_centre_field_tracks_the_unrestricted_spline_map():
    """Oracle that needs no fit restriction at all: ``newton_fit='spline'`` is
    the local bicubic interpolant of the SAME traced samples, so whatever halo
    it shows is the real one.  D7 must move the field TOWARD it, never away."""
    _ram_guard()
    ref = np.abs(_ghost_apply(newton_fit='spline'))
    new = np.abs(_ghost_apply())
    old = np.abs(_ghost_apply(pre_d7=True))
    scale = float(ref.max())
    d_new = float(np.max(np.abs(new - ref))) / scale
    d_old = float(np.max(np.abs(old - ref))) / scale
    assert d_new < 1e-3, d_new
    assert d_new <= d_old * 1.05, (
        f"D7 moved the field AWAY from the unrestricted spline map: "
        f"{d_old:.3e} -> {d_new:.3e}")


# ===========================================================================
# 4.  The basis-domain re-map that was measured and REFUSED.
# ===========================================================================
def test_the_basis_domain_is_affine_invariant_so_remapping_it_is_a_no_op():
    """Pins the reason D7 did not re-centre / re-scale the fit's basis domain.

    ``_Cheb2DEvaluator`` spans total degree <= ``order`` in ``(x, y)``, and that
    space is closed under an affine change of variables -- so fitting the same
    samples with the same weights on a SHIFTED, RESCALED domain returns the
    same polynomial.  Here the shift/scale is applied to the sample AXES (which
    is what sets the basis domain) and to the query points together."""
    rng = np.random.default_rng(20260729)
    xs = np.linspace(-1.0, 1.0, 25)
    Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
    vals = (0.4 + 1.3 * Xg - 0.8 * Yg + 0.5 * Xg ** 2 * Yg - 0.2 * Yg ** 3
            + 0.02 * rng.standard_normal(Xg.shape))
    w = np.where(((Xg - 0.5) ** 2 + Yg ** 2) <= 0.25, 1.0, 1e-4)
    a, b = 3.7, -0.9                      # u = a * x + b
    ev0 = _Cheb2DEvaluator(xs, xs, vals, order=4, weights=w)
    ev1 = _Cheb2DEvaluator(a * xs + b, a * xs + b, vals, order=4, weights=w)
    q = np.linspace(-0.9, 0.9, 17)
    Q1, Q2 = np.meshgrid(q, q, indexing='ij')
    f0 = np.asarray(ev0.ev(Q1.ravel(), Q2.ravel()))
    f1 = np.asarray(ev1.ev(a * Q1.ravel() + b, a * Q2.ravel() + b))
    scale = float(np.max(np.abs(f0)))
    assert np.max(np.abs(f0 - f1)) <= 1e-9 * scale, (
        "the tensor-Chebyshev fit stopped being affine-invariant -- the D7 "
        "refusal of the basis-domain re-map rests on this")


def test_a_shrunken_basis_domain_is_a_liability_outside_itself():
    """The other half of the refusal: the Newton loop evaluates these fits over
    the WHOLE launch square, and a basis re-mapped onto a small off-centre disc
    runs to |u| >> 1 there, where the Chebyshev terms explode.  Two
    mathematically IDENTICAL fits must then disagree numerically."""
    rng = np.random.default_rng(20260729)
    xs = np.linspace(-1.0, 1.0, 41)
    Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
    vals = np.exp(0.7 * Xg) * np.cos(1.3 * Yg) + 1e-9 * rng.standard_normal(
        Xg.shape)
    w = np.where(((Xg - 0.5) ** 2 + Yg ** 2) <= 0.09, 1.0, 1e-4)
    ev_wide = _Cheb2DEvaluator(xs, xs, vals, order=12, weights=w)
    # the SAME fit expressed on the disc's own domain: u = (x - 0.5) / 0.3
    a, b = 1.0 / 0.3, -0.5 / 0.3
    ev_tight = _Cheb2DEvaluator(a * xs + b, a * xs + b, vals, order=12,
                                weights=w)
    inside = np.array([[0.5, 0.0], [0.6, 0.1], [0.4, -0.1]])
    corner = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    d = {}
    for name, pts in (('inside', inside), ('corner', corner)):
        f0 = np.asarray(ev_wide.ev(pts[:, 0], pts[:, 1]))
        f1 = np.asarray(ev_tight.ev(a * pts[:, 0] + b, a * pts[:, 1] + b))
        d[name] = float(np.max(np.abs(f0 - f1)))
    assert d['inside'] < 1e-10, d
    assert d['corner'] > 1e3 * max(d['inside'], 1e-16), (
        f"the shrunken basis stopped diverging outside its domain "
        f"({d}) -- re-check the D7 refusal")


# ===========================================================================
# 5.  API surface.
# ===========================================================================
@pytest.mark.parametrize('bad', [0, -1, 2.5, 'ten', np.nan])
def test_decentred_fit_poly_order_is_validated(bad):
    env = _gauss()
    with pytest.raises(ValueError, match='decentred_fit_poly_order'):
        la.apply_real_lens_traced(
            env, prescription=_prescription(), wavelength=_WL, dx=_DX,
            ray_subsample=8, n_workers=1, decentred_fit_poly_order=bad,
            on_undersample='silent', on_noncollimated='silent',
            on_aperture_beam='silent')


def test_none_means_the_module_default():
    """``None`` and the module constant are the same call."""
    _ram_guard()
    a = _apply(_X0)
    b = _apply(_X0, decentred_fit_poly_order=None)
    c = _apply(_X0, decentred_fit_poly_order=_lt._DECENTRED_FIT_POLY_ORDER)
    scale = float(np.max(np.abs(a)))
    assert np.max(np.abs(a - b)) <= 1e-10 * scale
    assert np.max(np.abs(a - c)) <= 1e-10 * scale


def test_the_prepared_screen_path_refuses_the_knob_with_a_reason():
    """A prepared screen is built on a flat ``ones`` placeholder, so it has no
    beam to place a disc around: the multi entry point must say so rather than
    raise ``TypeError`` three frames down."""
    env = _gauss()
    with pytest.raises(ValueError, match='decentred_fit_poly_order'):
        la.apply_real_lens_traced_multi(
            [env, env], prescription=_prescription(), wavelength=_WL, dx=_DX,
            carriers=60e-3, reuse_prepared=True,
            decentred_fit_poly_order=10, ray_subsample=8, n_workers=1,
            on_undersample='silent', on_noncollimated='silent')
