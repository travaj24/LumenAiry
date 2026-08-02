"""Niche D6 -- the EXACT high-NA final leg under a TILTED carrier.

Roadmap ``ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27`` P1a, closing the
last capability gap left by D1/D2: until v5.32
:func:`~lumenairy.propagate_traced_carrier_chain` RAISED ``NotImplementedError``
whenever a tilted congruence routed onto the exact high-NA final leg, so a
design-121-class DOE fan had to run ``final_leg='paraxial'`` and every per-order
spot was capped far below the single-beam acceptance.

**The obstacle, measured.**  ``_fine_trace_group_exit`` re-traces the last group
on a grid cropped to ``window_factor * w_entrance`` about the GRID CENTRE.  The
chain stores its envelope in the CHIEF-RAY-TRACKING frame, so that crop is
already chief-ray-centred in absolute terms -- but ``apply_real_lens_traced``
builds its own grid symmetrically about the OPTICAL AXIS (it traces the beam
through the surfaces at the beam's physical transverse position), so one
axis-centred window has to hold the axis AND the displaced beam.  On design 121
at N=1024 that is a chief ray 3.373 mm off axis against an entrance beam radius
3.126 mm: the window grows 12.50 -> 18.54 mm (1.48x) and the fine grid must grow
with it (``n_fine_cap`` 8192 -> 12288) to keep ``dx_fine`` below the exit
sphere's Nyquist pitch.

**The fix.**  ``_fine_trace_group_exit`` and
:func:`~lumenairy.propagators.carrier.carrier_referenced_exact_focus_readout`
both take ``centre`` (the chief ray) and ``tilt`` ``(L, M)``:

* the RETRACE widens its axis-centred window to
  ``2*max(|x_c|,|y_c|) + window_factor*w``, band-limit-shifts the envelope to
  the beam's physical position, reconstructs against the DECENTRED sphere plus
  the tilt ramp, and hands the element the matching
  :class:`~lumenairy.TiltedCarrier` -- the same hand-off the chain already
  performs on its coarse legs;
* the READOUT references the exact sphere AND the tilt about the chief ray,
  takes its ``window_factor`` crop ABOUT THE CHIEF RAY (so its internal fine
  grid costs exactly what an on-axis beam of the same radius costs, however far
  off axis it sits), and asks the band-limited ASM Bluestein zoom for the window
  at ``centre_out - chief``.  The ASM is translation-covariant, so the returned
  field is the physical field at the ABSOLUTE ``centre_out``, with the tilt's own
  transverse advance and path piston carried exactly rather than reimposed by
  paraxial bookkeeping.

Where the growth cannot be paid, ``on_tilt_exact_grid`` (default ``'error'``)
REFUSES, naming the chief-ray offset, the beam radius, the window, the required
``n_fine`` and the cap that bound it -- never a silent fall-back and never a leg
whose outer NA has been discarded without saying so.

**The oracle here is lumenairy-FREE and inline**: an exact conic raytrace to the
lens's exit surface (entrance eikonal included -- the hammer-H6 term) followed by
a direct Rayleigh-Sommerfeld/Kirchhoff sum over that surface with the exact
surface area element and obliquity factor.  It shares no FFT grid, no
``window_factor``, no carrier convention and no propagator with the code under
test, and its own FWHM is checked against the analytic Gaussian focus
``1.1774 * lambda * f / (pi * w)`` so the configuration itself is pinned.

**Stand-in.**  ONE flat-entrance / ``K = -n^2`` conic-exit singlet (the Fermat
solution: a collimated bundle in air imaged stigmatically, exactly, to all
orders), fed by a COLLIMATED Gaussian entering at a transverse DECENTRE
comparable to its own radius.  Because the conic is stigmatic for the whole
collimated bundle, ANY sub-aperture of it is stigmatic too -- so the decentred
beam is diffraction-limited by construction and the exact leg has a hard target
to hit, while the paraxial readout (~200 rad of wavefront wrong at this exit NA)
demonstrably misses it.

**The caveat this stand-in exposed is now FIXED** (niche C3, 2026-07-30).  The
chain carries ``(L, M)`` as DIRECTION COSINES -- advancing the chief ray by
``z L / cos(theta)``, which is exact for a free leg -- but used to obtain them
from the group's PARAXIAL ABCD, where they are slopes.  At this stand-in's exit
tilt that mixed ``tan`` with ``sin`` and put the predicted chief ray **12.4 um**
from the Fermat focus at ``f = 3 mm``, while the exact leg's SPOT landed on the
Fermat focus.  The predictor now TRACES the chief ray through the group's own
surfaces (:func:`~lumenairy.propagators.carrier._group_chief_transfer`) and
lands on the Fermat focus to 1e-19 m, i.e. 0.0005 um from the measured spot
centroid -- pinned by ``test_the_exact_leg_is_reachable_under_a_tilted_carrier``
below.

The OBVIOUS repair was refuted before that one shipped, and the refutation is
worth keeping: converting cosine -> slope into the ABCD and back makes the group
transfer formally consistent with the free legs, and on the D1 two-singlet relay
at 46 mrad it moves the predicted image height the WRONG WAY -- against an exact
meridional trace the raw-cosine ABCD is +0.1214 um out and the converted one
+1.1208 um, 9x worse.  A lumped group ABCD is refraction-dominated and Snell is
linear in SINES while intra-group free transfer is linear in TANGENTS, so no
scalar angle convention is right for it.  Exact tracing is +0.0000 um.  Design
121's final group leaves ``L_out = 8.3e-5``, where the whole term was 6 pm --
invisible, which is why nothing in that study saw it before this stand-in.

The tests below still centre the readout on the analytically-known Fermat focus
and check that the EXACT leg's spot lands THERE, which remains a stronger
statement than agreeing with the predictor; what changed is that the predictor
is now checked against the measured spot centroid as well.

Cost: 6 chain runs on a 2048-square fine leg, measured ~60 s total on Windows
(the exact leg is 6 s, the paraxial 2 s).  Every chain-running test is
RAM-guarded at 4 GiB available.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import get_glass_index
from lumenairy.propagators.carrier import (
    _chain_chief_ray_at_target,
    _exact_sphere_eikonal,
    _paraxial_group_r_out,
    carrier_referenced_exact_focus_readout,
)

# ---------------------------------------------------------------------------
# The synthetic stand-in (no .zmx, no data file).
# ---------------------------------------------------------------------------
_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_GLASS = 'N-BK7'
_W = 0.60e-3                # collimated beam 1/e^2 radius at the lens
_F = 3.00e-3                # focal length  ->  exit NA = _W / _F = 0.20
_X0 = 0.60e-3               # chief-ray DECENTRE at the lens (== _W, so the
#                             axis-centred retrace window must grow 1.5x)
_APER = 3.40e-3            # clear aperture (holds the beam to 1.7 mm,
#                            i.e. inside the retrace window's own half-extent)
_THICK = 1.5e-3
_NGRID = 1024               # chain grid points
_EXTENT = 6.0e-3            # chain grid physical extent (holds axis + beam)
_DXO = 0.15e-6              # readout pitch
_NOUT = 96                  # readout points  (+/- 7.2 um)
_WFAC = 4.0                 # window_factor (no-op regime, >= 4)
_RADII = (2.0, 4.0, 6.0)    # EE radii in um

# Oracle sampling.  The Huygens integrand is stationary near the focus, so a
# coarse pupil is exact here; 141 across 4.4 w is ~0.03 rad/sample of residual
# phase at the window edge.
_N_PUPIL = 141
_PUPIL_HALF = 2.2 * _W

_MIN_FREE_GIB = 4.0


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


def _prescription():
    """Flat entrance + ``K = -n^2`` conic exit: the exact Fermat singlet for a
    collimated bundle (``R = -(n-1) f``, ``K = -n^2``)."""
    n = _n_glass()
    return {'name': 'D6 fast singlet', 'aperture_diameter': _APER,
            'thicknesses': [_THICK],
            'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0),
                         _surface(-(n - 1.0) * _F, _GLASS, 'air', -n * n)]}


def _groups():
    return [{'prescription': _prescription(), 'gap_before': 0.0}]


def _launch(x0=0.0, y0=0.0):
    """Collimated Gaussian ENVELOPE in the chain's chief-ray-tracking frame
    (the beam sits on the grid centre; ``(x0, y0)`` is where the congruence
    says it physically is)."""
    dx = _EXTENT / _NGRID
    x = (np.arange(_NGRID) - _NGRID // 2) * dx
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / (_W * _W)
                 ).astype(np.complex128)
    return env, float(dx)


def _final_distance():
    """Back-vertex -> paraxial image distance for the collimated launch."""
    return float(-_paraxial_group_r_out(_prescription(), np.inf, _WL))


def _run_chain(carrier, final_leg='auto', n_fine_cap=16384,
               on_tilt_exact_grid='error', N_out=_NOUT, dx_out=_DXO,
               centre_out=None, groups=None, final_distance=None):
    groups = _groups() if groups is None else groups
    fd = _final_distance() if final_distance is None else final_distance
    env, dx = _launch()
    if centre_out is None:
        cx, cy, _L, _M = _chain_chief_ray_at_target(
            groups, _WL, carrier, fd, 'test_niche_d6')
        centre_out = (cx, cy)
    fr = {'dx_out': dx_out, 'N_out': int(N_out), 'n_fine_cap': int(n_fine_cap),
          'window_factor': _WFAC, 'centre_out': tuple(centre_out)}
    return la.propagate_traced_carrier_chain(
        env, groups, _WL, dx, r_in=carrier, ray_subsample=4, n_workers=1,
        final_distance=fd, focus_readout=fr, final_leg=final_leg,
        on_tilt_exact_grid=on_tilt_exact_grid,
        traced_kwargs={'fit_radius_beam_factor': 1.5,
                       'on_undersample': 'silent'}), centre_out


# ---------------------------------------------------------------------------
# The lumenairy-FREE oracle: exact conic raytrace + Kirchhoff surface integral.
# ---------------------------------------------------------------------------
def _sag(r2, R, K):
    """Even conic sag ``r^2 / (R (1 + sqrt(1 - (1+K) r^2/R^2)))``."""
    return r2 / (R * (1.0 + np.sqrt(np.maximum(
        1.0 - (1.0 + K) * r2 / (R * R), 0.0))))


def _dsag(r2, R, K):
    """``d(sag)/dr`` divided by ``r`` (so the gradient is ``(gx, gy) = this *
    (x, y)``) -- the analytic derivative of :func:`_sag`."""
    q = np.sqrt(np.maximum(1.0 - (1.0 + K) * r2 / (R * R), 0.0))
    return 1.0 / (R * q)


def _oracle_field(xs, ys, x0=0.0, y0=0.0, L=0.0, M=0.0, z_img=None,
                  n_pupil=_N_PUPIL):
    """Complex field at the ABSOLUTE points ``(xs, ys)`` on the image plane.

    Independent of every lumenairy propagator: an exact raytrace of the
    COLLIMATED congruence ``W_in = L (x-x0) + M (y-y0)`` from the flat entrance
    surface, through the glass, to the conic exit surface (Newton on the exact
    sag), then a direct Rayleigh-Sommerfeld sum over that surface with its
    exact area element ``sqrt(1+|grad sag|^2)`` and obliquity factor.  No
    Snell at the exit is needed -- the Kirchhoff kernel produces the refracted
    wave from the exit-surface field alone.
    """
    n = _n_glass()
    R2 = -(n - 1.0) * _F
    K2 = -n * n
    z_img = (_THICK + _final_distance()) if z_img is None else z_img
    # --- entrance pupil samples (a square patch centred on the beam) --------
    h = np.linspace(-_PUPIL_HALF, _PUPIL_HALF, int(n_pupil))
    dA = float(h[1] - h[0]) ** 2
    HX, HY = np.meshgrid(h + x0, h + y0, indexing='ij')
    HX, HY = HX.ravel(), HY.ravel()
    amp = np.exp(-((HX - x0) ** 2 + (HY - y0) ** 2) / (_W * _W))
    inside = (HX ** 2 + HY ** 2) <= (0.5 * _APER) ** 2
    # --- refract at the FLAT entrance, march to the conic ------------------
    lg, mg = L / n, M / n
    ng = np.sqrt(max(1.0 - lg * lg - mg * mg, 0.0))
    s = np.full(HX.shape, _THICK / ng)
    for _ in range(60):
        X = HX + lg * s
        Y = HY + mg * s
        r2 = X * X + Y * Y
        f_ = ng * s - _THICK - _sag(r2, R2, K2)
        g = _dsag(r2, R2, K2)
        df = ng - (g * (X * lg + Y * mg) * 2.0) * 0.5
        s = s - f_ / df
    X = HX + lg * s
    Y = HY + mg * s
    r2 = X * X + Y * Y
    Z = _THICK + _sag(r2, R2, K2)
    opl = L * (HX - x0) + M * (HY - y0) + n * s
    # exact surface element + normal (F = z - t - sag -> grad = (-gx, -gy, 1))
    g = _dsag(r2, R2, K2)
    gx, gy = g * X, g * Y
    nrm = np.sqrt(1.0 + gx * gx + gy * gy)
    w = amp * np.where(inside, 1.0, 0.0) * nrm * dA
    keep = w > 1e-9 * w.max()
    X, Y, Z, opl, w = X[keep], Y[keep], Z[keep], opl[keep], w[keep]
    nx, ny, nz = (-gx[keep] / nrm[keep], -gy[keep] / nrm[keep],
                  1.0 / nrm[keep])
    # --- Rayleigh-Sommerfeld sum to each field point -----------------------
    xs = np.atleast_1d(np.asarray(xs, dtype=np.float64)).ravel()
    ys = np.atleast_1d(np.asarray(ys, dtype=np.float64)).ravel()
    out = np.empty(xs.size, dtype=np.complex128)
    phase0 = _K0 * opl
    chunk = max(1, int(4e7 // max(X.size, 1)))
    for i0 in range(0, xs.size, chunk):
        sl = slice(i0, min(i0 + chunk, xs.size))
        dxp = xs[sl][:, None] - X[None, :]
        dyp = ys[sl][:, None] - Y[None, :]
        dzp = (z_img - Z)[None, :]
        rr = np.sqrt(dxp * dxp + dyp * dyp + dzp * dzp)
        cosang = (dxp * nx[None, :] + dyp * ny[None, :]
                  + dzp * nz[None, :]) / rr
        out[sl] = ((w[None, :] * cosang / rr)
                   * np.exp(1j * (phase0[None, :] + _K0 * rr))).sum(axis=1)
    return out


def _oracle_on_grid(centre, x0=0.0, y0=0.0, L=0.0, M=0.0, N_out=_NOUT,
                    dx_out=_DXO):
    u = (np.arange(N_out) - N_out / 2) * dx_out
    XS, YS = np.meshgrid(u + centre[0], u + centre[1], indexing='xy')
    E = _oracle_field(XS.ravel(), YS.ravel(), x0=x0, y0=y0, L=L, M=M)
    return E.reshape(N_out, N_out)


# ---------------------------------------------------------------------------
# Metrics (shared by the chain and the oracle, so no metric bias).
# ---------------------------------------------------------------------------
def _metrics(E, dx_out=_DXO, radii=_RADII):
    E = np.asarray(E)
    n = E.shape[-1]
    I = np.abs(E) ** 2
    tot = float(I.sum())
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    xx = (np.arange(n) - ix) * dx_out
    yy = (np.arange(n) - iy) * dx_out
    rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
    nb = n // 2
    ring = np.clip((rr / dx_out).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cnt = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(cnt[:nb], 1)) / I[iy, ix]
    rb = (np.arange(nb) + 0.5) * dx_out
    idx = np.where(prof < 0.5)[0]
    fwhm = float(2 * rb[idx[0]]) if len(idx) else np.nan
    ee = {r: float(I[rr <= r * 1e-6].sum()) / tot for r in radii}
    ax = (np.arange(n) - n / 2) * dx_out
    cx = float((I.sum(axis=0) * ax).sum() / tot)
    cy = float((I.sum(axis=1) * ax).sum() / tot)
    return {'fwhm': fwhm, 'ee': ee, 'peak_off': ((ix - n / 2) * dx_out,
                                                 (iy - n / 2) * dx_out),
            'centroid': (cx, cy)}


# ===========================================================================
# 1.  The on-axis path is untouched.
# ===========================================================================
def test_untilted_tiltedcarrier_takes_the_scalar_path_byte_identically():
    """``TiltedCarrier(R, 0, 0)`` is not tilted, so it must reproduce the
    scalar carrier BIT for bit -- the whole point of the ``is_tilted``
    short-circuit that keeps the shipped default path unchanged."""
    _ram_guard()
    a, _ = _run_chain(np.inf, final_leg='exact')
    b, _ = _run_chain(la.TiltedCarrier(np.inf, 0.0, 0.0), final_leg='exact')
    assert np.array_equal(np.asarray(a.field), np.asarray(b.field))


def test_a_negligible_tilt_reproduces_the_on_axis_exact_leg():
    """The NEW tilted exact leg must REDUCE to the on-axis one.

    ``L = 1e-15`` engages every D6 branch (widened window, band-limited shift,
    decentred sphere, tilt ramp, chief-ray-centred readout crop) while being
    physically nothing, so any structural mistake in the tilted algebra shows
    up as a finite difference rather than roundoff.  Tolerance, not
    ``array_equal``: the two routes reach the same field through different
    (mathematically identical) expressions.

    The residue that IS there is the tilt's own ramp, not noise: at ``L`` the
    ramp reaches ``k0 * L * (N dx / 2)`` = 1.4e-8 rad per 1e-12 of ``L``, and
    the measured difference tracks it linearly (1.2e-9 relative at
    ``L = 1e-12``, ~1e-12 at ``L = 1e-15``)."""
    _ram_guard()
    a, c = _run_chain(np.inf, final_leg='exact')
    b, _ = _run_chain(la.TiltedCarrier(np.inf, 1e-15, 0.0), final_leg='exact',
                      centre_out=c)
    A, B = np.asarray(a.field), np.asarray(b.field)
    scale = float(np.abs(A).max())
    assert scale > 0
    assert float(np.abs(A - B).max()) <= 1e-10 * scale


def test_readout_centre_and_tilt_defaults_are_byte_identical():
    """The readout's new ``centre`` / ``tilt`` kwargs must short-circuit."""
    N, dx = 256, 2.0e-6
    x = (np.arange(N) - N // 2) * dx
    R = -2.0e-3
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    E = (np.exp(-r2 / (0.25e-3 ** 2))
         * np.exp(1j * _K0 * (-(np.sqrt(r2 + R * R) - abs(R))))
         ).astype(np.complex128)
    kw = dict(dx_out=0.1e-6, N_out=64, window_factor=4.0)
    a = carrier_referenced_exact_focus_readout(E, R, -R, _WL, dx, **kw)
    b = carrier_referenced_exact_focus_readout(
        E, R, -R, _WL, dx, centre=(0.0, 0.0), tilt=(0.0, 0.0), **kw)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_readout_is_translation_covariant():
    """The decentred readout is the on-axis one, translated.

    Pure unit test -- no element, no chain.  A converging field shifted by an
    INTEGER number of pixels, read out with ``centre`` and ``centre_out``
    shifted by the same amount, must return the on-axis answer.  This is what
    lets the exact leg take its crop about the chief ray instead of about the
    optical axis and still land the window on the ABSOLUTE lattice."""
    # Sized so BOTH runs take the same window: the beam is compact enough that
    # the shifted copy does not wrap, and ``window_factor * w`` fits inside the
    # grid at the decentre (else the decentred run would silently take a
    # NARROWER window and the comparison would be of two different crops).
    N, dx = 512, 2.0e-6
    shift_px = 64
    sh = shift_px * dx
    x = (np.arange(N) - N // 2) * dx
    R = -2.0e-3
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2 / (0.10e-3 ** 2))
    E0 = (env * np.exp(-1j * _K0 * (np.sqrt(r2 + R * R) - abs(R)))
          ).astype(np.complex128)
    E1 = np.roll(E0, (shift_px, shift_px), axis=(0, 1))
    kw = dict(dx_out=0.5e-6, N_out=64, window_factor=4.0)
    a = np.asarray(carrier_referenced_exact_focus_readout(
        E0, R, -R, _WL, dx, centre_out=(0.0, 0.0), **kw))
    b = np.asarray(carrier_referenced_exact_focus_readout(
        E1, R, -R, _WL, dx, centre=(sh, sh), centre_out=(sh, sh), **kw))
    scale = float(np.abs(a).max())
    assert scale > 0
    assert float(np.abs(a - b).max()) <= 1e-9 * scale


# ===========================================================================
# 2.  The oracle pins itself, then the exact leg.
# ===========================================================================
def test_the_inline_oracle_is_diffraction_limited_and_grid_converged():
    """Pin the ORACLE before using it to judge the propagator.

    The stand-in is stigmatic BY CONSTRUCTION (``K = -n^2`` is the exact
    Fermat conic for a collimated bundle, so any SUB-APERTURE of that bundle
    is stigmatic too -- the geometric spot is identically zero, hence any width
    the oracle reports IS diffraction).  Both the on-axis bundle and the
    DECENTRED sub-aperture focus to the SAME on-axis Fermat point.

    Measured: FWHM 3.150 um in both cases, 1.283x the PARAXIAL Gaussian focus
    ``1.1774 lambda f / (pi w)`` = 2.455 um -- the paraxial formula is not the
    right reference at NA 0.20 off a conic (the exact ``r -> sin(theta)`` map
    reads 0.189 at ``r = w`` where ``r/f`` reads 0.200, and a converging
    SPHERE off a curved surface carries its own obliquity apodisation), so the
    band is a sanity band, not a tolerance.  The teeth are the convergence
    check underneath it."""
    fwhm_ideal = 1.1774 * _WL * _F / (np.pi * _W)
    fw = {}
    for x0 in (0.0, _X0):
        m = _metrics(_oracle_on_grid((0.0, 0.0), x0=x0))
        fw[x0] = m['fwhm']
        assert 0.9 < m['fwhm'] / fwhm_ideal < 1.6, (
            f"x0={x0 * 1e3:.3f} mm: oracle FWHM {m['fwhm'] * 1e6:.4f} um vs "
            f"the paraxial Gaussian {fwhm_ideal * 1e6:.4f} um -- outside the "
            f"diffraction-limited sanity band")
        assert m['ee'][4.0] > 0.95
        # the Fermat focus is ON AXIS for both bundles
        assert float(np.hypot(*m['centroid'])) < 0.2e-6
    # ... and the two agree, as a stigmatic system's sub-aperture must
    assert abs(fw[_X0] / fw[0.0] - 1.0) < 0.05
    # grid convergence in the pupil sampling (1-D cut, cheap)
    u = (np.arange(_NOUT) - _NOUT / 2) * _DXO
    z = np.zeros_like(u)
    a = np.abs(_oracle_field(u, z, x0=_X0, n_pupil=_N_PUPIL)) ** 2
    b = np.abs(_oracle_field(u, z, x0=_X0, n_pupil=2 * _N_PUPIL - 1)) ** 2
    assert float(np.abs(a - b).max()) < 1e-3 * float(a.max())


def test_the_exact_leg_is_reachable_under_a_tilted_carrier():
    """The headline: no ``NotImplementedError``, and the run reports the exact
    leg plus the chief ray it landed on -- which is now the chief ray the SPOT
    is actually measured at.

    STRENGTHENED 2026-07-30 (niche C3).  This used to assert only that the
    chain echoed back the ABCD prediction the test had seeded ``centre_out``
    with, which was true by construction on both sides and said nothing about
    where the light went; the roadmap separately RECORDED that the predictor
    sat 12.4 um from the Fermat focus while the exact leg's spot landed on it.
    The predictor is an exact chief-ray trace now, so the statement worth
    pinning is the one that was previously only recorded.

    Measured here, readout centred on the analytically-known Fermat focus:
    predictor ``x_c`` = -1.1e-19 m (the Fermat focus itself, to double
    precision) against a measured spot centroid of 4.7e-10 m -- the two agree
    to 0.0005 um, 1/6700 of the 3.15 um FWHM.  The superseded ABCD predictor
    sat 12.372 um away, 3.9 FWHM, far enough that a readout tile centred on it
    clipped the spot on this +/-7.2 um window (EE2 0.363 against 0.703).
    """
    _ram_guard()
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0)
    res, centre = _run_chain(car, final_leg='auto', centre_out=(0.0, 0.0))
    last_group = [s for s in res.stages if s.get('exact_final')
                  and not s.get('target')]
    assert last_group, 'the exact final leg did not run'
    assert last_group[0]['na_exit'] > 0.15
    tgt = res.stages[-1]
    assert tgt.get('target') and tgt.get('exact_final')
    F = np.asarray(res.field)
    assert F.shape == (_NOUT, _NOUT) and np.isfinite(F).all()
    # the predicted chief ray is ON the Fermat focus -- the stand-in's
    # analytically exact answer for any sub-aperture of the collimated bundle
    assert abs(float(tgt['x_c'])) < 1e-12, tgt['x_c']
    assert abs(float(tgt['y_c'])) < 1e-12, tgt['y_c']
    # ... and ON the measured spot centroid, to a fiftieth of the spot width
    m = _metrics(F)
    cx = centre[0] + m['centroid'][0]
    cy = centre[1] + m['centroid'][1]
    assert abs(float(tgt['x_c']) - cx) < 0.02 * m['fwhm'], (tgt['x_c'], cx)
    assert abs(float(tgt['y_c']) - cy) < 0.02 * m['fwhm'], (tgt['y_c'], cy)
    assert float(np.hypot(cx, cy)) < 0.05e-6


def test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle():
    """FAIL-BEFORE / PASS-AFTER, on the same congruence and the same grid.

    The decentred collimated beam is diffraction-limited by construction, so
    the inline oracle IS the right answer.  The exact leg must land on it; the
    paraxial readout -- the only route available before D6 -- must not.

    THE FAIL-BEFORE RATIO MOVED, 2026-07-30 (niche C3), and this is the reason
    so the next reader is not surprised again.  The paraxial leg used to read
    **3.19x** the oracle FWHM (EE2 6.5 % against 71.9 %); it now reads
    **1.857x** (EE2 10.6 % against 71.6 %).  Nothing about the paraxial
    readout changed -- its wavefront is still ~200 rad wrong at this exit NA.
    What changed is the chief-ray PREDICTOR that places its spot: it is an
    exact trace now instead of a lumped paraxial ABCD, so the paraxial leg's
    (still wrong) spot is at least in the right PLACE, and less of it falls
    off the +/-7.2 um readout window.  The exact leg improved on the same
    change, from 9.5 % / 0.951x / 0.991x on FWHM / EE2 / EE4 to
    **0.0 % / 0.982x / 0.995x**.

    Because the FWHM bar had to come down, the discrimination is carried by
    two measurements that did NOT weaken, both on the shipped window: the
    paraxial leg keeps 14.8 % of the oracle's EE2 (pinned below 25 %), and its
    brightest pixel sits at the window EDGE while the exact leg's sits at the
    centre.  On a +/-38 um window -- measured, not asserted, since this test
    deliberately keeps the shipped readout -- the paraxial peak is 8.25 um
    from the Fermat focus and only 2.8 % of its power lands within 2 um of it,
    against the exact leg's 70.2 % and the oracle's 71.6 %."""
    _ram_guard()
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0)
    # Window centred on the ANALYTICALLY KNOWN Fermat focus (on axis for any
    # sub-aperture of a collimated bundle through this conic).  The chain's
    # own chief-ray predictor now agrees with it to 1e-19 m, but the point of
    # centring here is that this centre is known WITHOUT the chain.
    ex, _ = _run_chain(car, final_leg='exact', centre_out=(0.0, 0.0))
    px, _ = _run_chain(car, final_leg='paraxial', centre_out=(0.0, 0.0))
    orc = _metrics(_oracle_on_grid((0.0, 0.0), x0=_X0))
    m_ex = _metrics(np.asarray(ex.field))
    m_px = _metrics(np.asarray(px.field))
    # the exact leg tracks the independent oracle (measured 0.0 % on FWHM,
    # 0.982x on EE2, 0.995x on EE4)
    assert abs(m_ex['fwhm'] / orc['fwhm'] - 1.0) < 0.15, (
        f"exact FWHM {m_ex['fwhm'] * 1e6:.4f} um vs oracle "
        f"{orc['fwhm'] * 1e6:.4f} um")
    assert m_ex['ee'][2.0] > 0.90 * orc['ee'][2.0]
    assert m_ex['ee'][4.0] > 0.97 * orc['ee'][4.0]
    # ... and the paraxial one does not, by a wide margin.  Measured 1.857x
    # the oracle FWHM (was 3.19x -- see the docstring for why it moved).
    # ERODING AXIS, floored not chased (2026-08-02): this ratio has fallen
    # 3.19x -> 1.857x (C3, bar lowered 2.0 -> 1.70) -> 1.571x on a main-CI
    # runner post-C9, because every physics fix (C5/C6/C9 chief-ray and
    # reference improvements) also better-PLACES the deliberately-inferior
    # paraxial route's spot.  Its wavefront is still ~200 rad wrong; the
    # discriminators that measure THAT are the two below (EE2 < 0.25x the
    # oracle's, peak >= 4 um off focus), which have never eroded.  The FWHM
    # ratio keeps only a sanity floor with real cross-runner headroom.
    assert m_px['fwhm'] > 1.25 * orc['fwhm'], (
        f"paraxial FWHM {m_px['fwhm'] * 1e6:.4f} um should be far wide of the "
        f"oracle's {orc['fwhm'] * 1e6:.4f} um -- the fail-before half of this "
        f"test has stopped failing")
    # the two discriminators that did NOT weaken with the predictor fix
    assert m_px['ee'][2.0] < 0.25 * orc['ee'][2.0], (
        f"paraxial EE2 {m_px['ee'][2.0]:.4f} against the oracle's "
        f"{orc['ee'][2.0]:.4f} (measured ratio 0.148)")
    assert abs(m_px['peak_off'][0]) > 4.0e-6, (
        f"the paraxial leg's brightest pixel is {m_px['peak_off'][0] * 1e6:+.3f}"
        f" um off the Fermat focus -- it used to be on the window edge")
    assert abs(m_ex['peak_off'][0]) < 0.5e-6
    assert m_ex['fwhm'] < 0.60 * m_px['fwhm']
    assert m_ex['ee'][2.0] > 5.0 * m_px['ee'][2.0]
    # the spot lands on the FERMAT focus, i.e. where the physics puts it
    assert float(np.hypot(*m_ex['centroid'])) < 0.5e-6


# ===========================================================================
# 2b.  The DECENTRED-carrier penalty -- measured envelope, not a claim.
#      (2026-07-29 adversarial verification.  An earlier revision of this
#      module's roadmap note called the per-order loss on design 121 "genuine
#      field-angle aberration, which is the physics".  It is not: an
#      independent skew-ray + Debye oracle of the same .zmx says every order is
#      equally diffraction-limited, EE6 99.9 %.
#
#      This block used to be `test_decentred_carrier_fit_defect_envelope`,
#      written to FAIL LOUDLY the day the defect it pinned moved.  Niche D7
#      moved it: the 3.7 -> 408 urad calibration it was built on turned out to
#      be the repro script's own FFT-derivative artefact, and raising the
#      OFF-CENTRE ray fit's polynomial order took the stand-in's decentred EE2
#      ratio 0.9498 -> 0.9828 at one full beam radius.  The envelope below is
#      the POST-FIX measurement; the fail-before switch is
#      `decentred_fit_poly_order=6`.  A residual penalty remains, which is why
#      `on_decentred_fit` still has a job -- see `_check_decentred_fit`.)
# ===========================================================================
def test_decentred_carrier_decentre_penalty_envelope():
    """The chain tracks the oracle on axis and slightly worse when the same
    beam is decentred -- pinned as an envelope so neither side can drift.

    The stand-in makes the comparison airtight: the ``K = -n^2`` conic is the
    exact Fermat solution for the WHOLE collimated bundle, so every
    sub-aperture of it -- on axis or decentred -- is stigmatic to all orders.
    Whatever the oracle reports for each is therefore the right answer for
    each, and the ratio chain/oracle removes the aperture clipping that makes
    the two oracles differ in their own right.

    Measured 2026-07-29 at ``|c|/w = 1.0``: 0.9498 pre-D7, **0.9828** post-D7,
    against 0.9966 on axis.  If the decentred ratio improves again, update this
    envelope, ``_check_decentred_fit``'s calibration and the roadmap's D6/D7
    block, and re-measure design 121's per-order rows -- they are still a
    LOWER BOUND."""
    _ram_guard()
    on_axis, _ = _run_chain(la.TiltedCarrier(np.inf, 0.0, 0.0, 0.0, 0.0),
                            final_leg='exact', centre_out=(0.0, 0.0))
    off_axis, _ = _run_chain(la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0),
                             final_leg='exact', centre_out=(0.0, 0.0))
    o_on = _metrics(_oracle_on_grid((0.0, 0.0), x0=0.0))
    o_off = _metrics(_oracle_on_grid((0.0, 0.0), x0=_X0))
    m_on = _metrics(np.asarray(on_axis.field))
    m_off = _metrics(np.asarray(off_axis.field))
    # The oracle says the two are the same class of spot (both stigmatic):
    # same FWHM to the readout pitch.
    assert abs(o_off['fwhm'] / o_on['fwhm'] - 1.0) < 0.10, (
        f"the stand-in stopped being decentre-invariant: oracle FWHM "
        f"{o_on['fwhm'] * 1e6:.4f} um on axis vs {o_off['fwhm'] * 1e6:.4f} um "
        f"decentred")
    r_on = m_on['ee'][2.0] / o_on['ee'][2.0]
    r_off = m_off['ee'][2.0] / o_off['ee'][2.0]
    # measured 2026-07-29 (post-D7): 0.9966 on axis, 0.9828 decentred
    assert r_on > 0.97, f"the ON-AXIS path regressed: EE2 ratio {r_on:.4f}"
    assert 0.965 < r_off < 1.005, (
        f"decentred EE2 ratio {r_off:.4f} is outside the MEASURED post-D7 "
        f"envelope (0.9828 at |c|/w = 1.0).  Above it: the decentred path "
        f"improved again -- update this envelope, _check_decentred_fit's "
        f"calibration and the roadmap's D6/D7 block.  Below it: it regressed.")
    # ... and the decentred spot is now the oracle's WIDTH, which it was not
    # pre-D7 (FWHM ratio 1.0952 there, 1.0000 here).
    assert abs(m_off['fwhm'] / o_off['fwhm'] - 1.0) < 0.05, (
        f"decentred FWHM ratio {m_off['fwhm'] / o_off['fwhm']:.4f}")


def test_on_decentred_fit_warns_and_names_the_lower_bound():
    """The guard must fire on the very congruence the shipped fan runs, and
    say in as many words that the reported metric is a LOWER BOUND."""
    _ram_guard()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        _run_chain(la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0),
                   final_leg='exact', centre_out=(0.0, 0.0))
    hits = [str(w.message) for w in wl if 'decentre_fit_frac' in str(w.message)]
    assert hits, 'the decentred-fit guard did not fire on a 1.0 w decentre'
    msg = hits[0]
    for token in ('LOWER BOUND', 'beam amplitude radii', 'decentre-INVARIANT',
                  'on_decentred_fit', 'ray trace'):
        assert token in msg, f"{token!r} missing from the guard message"


def test_on_decentred_fit_is_silent_on_axis_and_below_the_threshold():
    """No warning for the on-axis path (the shipped single-beam acceptance),
    and none for a decentre the measured curve says is harmless."""
    _ram_guard()
    for car in (np.inf, la.TiltedCarrier(np.inf, 0.0, 0.0, 0.3 * _W, 0.0)):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            _run_chain(car, final_leg='exact', centre_out=(0.0, 0.0))
        assert not [w for w in wl if 'decentre_fit_frac' in str(w.message)], (
            f"the decentred-fit guard fired for carrier {car!r}")


def test_on_decentred_fit_can_refuse():
    _ram_guard()
    with pytest.raises(RuntimeError, match='LOWER BOUND'):
        la.propagate_traced_carrier_chain(
            _launch()[0], _groups(), _WL, _EXTENT / _NGRID,
            r_in=la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0),
            ray_subsample=4, n_workers=1,
            final_distance=_final_distance(), on_decentred_fit='error')


@pytest.mark.parametrize('bad', ['ERROR', 'raise', '', None, 1])
def test_on_decentred_fit_is_validated(bad):
    env, dx = _launch()
    with pytest.raises(ValueError, match='on_decentred_fit'):
        la.propagate_traced_carrier_chain(
            env, _groups(), _WL, dx, r_in=np.inf, on_decentred_fit=bad)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1.0])
def test_decentre_fit_frac_is_validated(bad):
    env, dx = _launch()
    with pytest.raises(ValueError, match='decentre_fit_frac'):
        la.propagate_traced_carrier_chain(
            env, _groups(), _WL, dx, r_in=np.inf, decentre_fit_frac=bad)


def test_multi_forwards_the_decentred_fit_guard():
    env, dx = _launch()
    with pytest.raises(ValueError, match='on_decentred_fit'):
        la.propagate_traced_carrier_chain_multi(
            [{'field': env, 'carrier': np.inf}], _groups(), _WL, dx,
            output_grid={'dx_out': _DXO, 'N_out': 64},
            on_decentred_fit='nope')


# ===========================================================================
# 3.  Honest refusal where the tilt cannot be accommodated.
# ===========================================================================
def test_refusal_when_the_fine_grid_cannot_sample_the_widened_window():
    """``on_tilt_exact_grid='error'`` (the default) refuses a leg whose outer
    NA would be discarded, and names every measured quantity."""
    _ram_guard()
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0)
    with pytest.raises(RuntimeError) as e:
        _run_chain(car, final_leg='exact', n_fine_cap=256)
    msg = str(e.value)
    for token in ('chief ray', 'axis-centred window', 'n_fine_cap=256',
                  'Nyquist', 'on_tilt_exact_grid', "final_leg='paraxial'"):
        assert token in msg, f"refusal message is missing {token!r}:\n{msg}"


def test_the_refusal_can_be_downgraded_and_then_it_is_worse():
    """``'warn'`` accepts the degraded leg -- and the degradation is REAL, so
    the guard is not bureaucratic: the same run at the honest ``n_fine_cap``
    is materially sharper.

    Measured here: EE2 63.4 % at ``n_fine_cap=256`` against 68.0 % uncapped
    (FWHM unchanged at 3.450 um, and flat across caps 256 / 512 / 1024).  That
    the loss is only ~4.6 points is itself informative and is WHY the guard
    defaults to ``'error'`` rather than being tuned to a tolerance: the band
    the coarse retrace throws away carries the RESIDUAL, and this stand-in is
    stigmatic, so its residual is ~0 and only the resample noise is lost.  A
    real design's residual is not zero (design 121 carries 9.2 rad of it on
    its final leg), so the same discard is worth far more there -- and a
    warning the caller has to size for themselves is exactly the
    silent-degradation trap the roadmap's P3/P5 exist to close."""
    _ram_guard()
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0)
    with pytest.warns(RuntimeWarning, match='TILTED congruence'):
        bad, _ = _run_chain(car, final_leg='exact', n_fine_cap=256,
                            on_tilt_exact_grid='warn', centre_out=(0.0, 0.0))
    good, _ = _run_chain(car, final_leg='exact', centre_out=(0.0, 0.0))
    m_bad = _metrics(np.asarray(bad.field))
    m_good = _metrics(np.asarray(good.field))
    assert m_good['ee'][2.0] > m_bad['ee'][2.0] + 0.03
    assert m_good['ee'][4.0] > m_bad['ee'][4.0] + 0.01


def test_refusal_when_the_chief_ray_does_not_fit_the_grid():
    """A decentre the co-moving grid cannot hold is a hard error naming the
    window it would have needed -- never a wrapped-round, plausible-looking
    field."""
    _ram_guard()
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, 2.6e-3, 0.0)
    with pytest.raises(ValueError) as e:
        _run_chain(car, final_leg='exact')
    msg = str(e.value)
    assert ('chief' in msg.lower()) or ('does not fit' in msg)


@pytest.mark.parametrize('bad', ['ERROR', 'raise', '', None, 1])
def test_on_tilt_exact_grid_is_validated(bad):
    env, dx = _launch()
    with pytest.raises(ValueError, match='on_tilt_exact_grid'):
        la.propagate_traced_carrier_chain(
            env, _groups(), _WL, dx, r_in=np.inf, on_tilt_exact_grid=bad)


def _decentred_sphere(n, dx, w, R, cx):
    """Gaussian x exact decentred sphere on a centred grid -- the readout's
    own documented input convention."""
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    S = _exact_sphere_eikonal((n, n), dx, dx, _WL, R, centre=(cx, 0.0))
    return (np.exp(-((X - cx) ** 2 + Y ** 2) / (w * w))
            * np.exp(1j * _K0 * S)).astype(np.complex128)


def test_the_decentred_readout_window_clamp_is_no_longer_silent():
    """A decentred ``centre`` shrinks what the grid can hold to
    ``N*dx - 2*|c|``.  That bound is geometrically forced, but applying it
    SILENTLY returned a truncated beam as if it were the requested window --
    measured 0.919 of the power at 0.906 of the peak, and 0.279 at 0.435 one
    step further out, with an empty warning list."""
    n, dx, w, R, z = 1024, 0.5e-6, 40e-6, -400e-6, 400e-6
    for cx in (200e-6, 230e-6):
        E = _decentred_sphere(n, dx, w, R, cx)
        with pytest.raises(RuntimeError) as e:
            carrier_referenced_exact_focus_readout(
                E, R, z, _WL, dx, dx_out=0.05e-6, N_out=128,
                centre_out=(cx, 0.0), window_factor=6.0, centre=(cx, 0.0))
        msg = str(e.value)
        for token in ('window_factor', 'available', 'TRUNCATES',
                      'on_readout_window'):
            assert token in msg, f"{token!r} missing from {msg[:200]!r}"


def test_a_harmless_readout_window_clamp_stays_silent():
    """The guard is sized by the power the clamp actually costs, not by the
    window ratio -- so a clamp that only trims empty guard band (measured
    3e-4 of the peak against a plain ASM on the same grid) must not fire."""
    from lumenairy.propagators.mft import angular_spectrum_propagate_mft
    n, dx, w, R, z, cx = 1024, 0.5e-6, 40e-6, -400e-6, 400e-6, 150e-6
    E = _decentred_sphere(n, dx, w, R, cx)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        got = carrier_referenced_exact_focus_readout(
            E, R, z, _WL, dx, dx_out=0.05e-6, N_out=128,
            centre_out=(cx, 0.0), window_factor=6.0, centre=(cx, 0.0))
    assert not [x for x in wl if 'on_readout_window' in str(x.message)]
    ref = angular_spectrum_propagate_mft(E, z, _WL, dx, 0.05e-6, 128,
                                         centre_out=(cx, 0.0), bandlimit=True)
    sc = float(np.abs(ref).max())
    assert float(np.abs(np.asarray(got) - np.asarray(ref)).max()) < 1e-3 * sc


def test_the_readout_window_refusal_can_be_downgraded():
    n, dx, w, R, z, cx = 1024, 0.5e-6, 40e-6, -400e-6, 400e-6, 230e-6
    E = _decentred_sphere(n, dx, w, R, cx)
    with pytest.warns(RuntimeWarning, match='TRUNCATES'):
        got = carrier_referenced_exact_focus_readout(
            E, R, z, _WL, dx, dx_out=0.05e-6, N_out=128,
            centre_out=(cx, 0.0), window_factor=6.0, centre=(cx, 0.0),
            on_readout_window='warn')
    assert np.isfinite(np.asarray(got)).all()


@pytest.mark.parametrize('bad', ['ERROR', 'raise', '', None, 1])
def test_on_readout_window_is_validated(bad):
    E = _decentred_sphere(64, 1e-6, 8e-6, -400e-6, 0.0)
    with pytest.raises(ValueError, match='on_readout_window'):
        carrier_referenced_exact_focus_readout(
            E, -400e-6, 400e-6, _WL, 1e-6, dx_out=0.05e-6, N_out=32,
            on_readout_window=bad)


def test_multi_forwards_the_guard():
    """The orchestrator a production fan run actually calls must expose and
    validate the same knob."""
    env, dx = _launch()
    with pytest.raises(ValueError, match='on_tilt_exact_grid'):
        la.propagate_traced_carrier_chain_multi(
            [{'field': env, 'carrier': np.inf}], _groups(), _WL, dx,
            output_grid={'dx_out': _DXO, 'N_out': 64},
            on_tilt_exact_grid='nope')


# ===========================================================================
# 4.  The energy bookkeeping the fan's power acceptance rests on.
# ===========================================================================
def test_the_tilted_exact_leg_conserves_power_like_the_paraxial_one():
    """D6 must improve the SPOT without disturbing what D2's per-frame power
    acceptance reads: the chain-exit power the last stage reports."""
    _ram_guard()
    car = la.TiltedCarrier(np.inf, 0.0, 0.0, _X0, 0.0)
    ex, _ = _run_chain(car, final_leg='exact', centre_out=(0.0, 0.0))
    px, _ = _run_chain(car, final_leg='paraxial', centre_out=(0.0, 0.0))
    p_ex = [s['power'] for s in ex.stages if 'power' in s][-1]
    p_px = [s['power'] for s in px.stages if 'power' in s][-1]
    env, dx = _launch()
    p_in = float(np.sum(np.abs(env) ** 2)) * dx * dx
    # The exact leg reports its exit power on the FINE retrace grid and the
    # paraxial one on the co-moving grid; both go through the ray-density
    # amplitude, whose own energy self-check tolerates a small band.  What the
    # fan's power acceptance needs is that the two AGREE (measured 0.65 % here
    # -- 0.9959 exact vs 0.9894 paraxial of the launched power).
    assert 0.95 < p_ex / p_in < 1.02
    assert abs(p_ex / p_px - 1.0) < 0.02


if __name__ == '__main__':  # pragma: no cover
    warnings.simplefilter('ignore')
    pytest.main([__file__, '-x', '-q'])
