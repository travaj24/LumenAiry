"""Regression pins for the S9 (review4a) sibling-pattern findings in
``lumenairy/raytrace/**`` and ``lumenairy/analysis/**``.

S9-RT1 -- PATTERN #4 (an ``inf`` sentinel silently disabling logic).
    ``intersection._intersect_surface`` tested its FLAT fast path
    (``np.isinf(radius) and not asph and not field_frame``) BEFORE the
    ``is_pure_spherical`` check, so it swallowed every surface whose power
    lives on the OTHER axis.  ``radius = inf`` does not mean "flat" -- the
    library documents a y-focusing cylinder as exactly
    ``radius = inf, radius_y = <finite>`` (``lenses.surface_sag_biconic``:
    "Cylindrical: pass R_y = inf (focusing in x only) or R_x = inf (focusing
    in y only)"), and a phase plate on a flat base as
    ``radius = inf, freeform = ...``.  Those were intersected at ``z = 0``
    with the sag DISCARDED while ``_surface_normal`` still refracted off the
    true biconic/freeform gradient -- right bend, wrong point, and the
    vertex->sag OPL leg dropped.  Measured on
    ``Surface(radius=inf, radius_y=50e-3)``: true sag 1010.205 um at
    y = 10 mm, intersection returned z = 0.000 um / opd = 0.000 um (771 waves
    of missing OPL at 1.31 um).

S9-RT2 -- degenerate ``None`` default (pattern #3 class).
    ``surface._base_surface_sag_derivatives_xy`` fed ``surface.conic_y``
    straight into ``(1 + K)``, so any biconic left at the ``conic_y=None``
    dataclass default raised ``TypeError`` from the surface-NORMAL path --
    while its sag twin ``surface_sag_biconic`` documents and implements
    ``conic_y is None -> conic_x``.  ``Surface(radius=50e-3, radius_y=60e-3)``
    could not be traced at all.

S9-RT3 -- PATTERN #2 (a pixel-count parameter whose physical meaning flips
    with the grid).  ``from_field._place_uniform`` divided the
    orientation-BLIND ``aspect = max(Nx,Ny)/min(Nx,Ny)`` into the ``y`` count
    only, so tall grids got FEWER samples on the LONGER axis.  Measured at
    ``n_rays=64``: ``(Nx,Ny)=(64,256)`` -> 4.20 px x-pitch vs 85.00 px
    y-pitch (20.2x anisotropic), ``(32,512)`` -> 1.00 vs 511.00 px (511x),
    while the transposes came out 1.24x / 1.88x.  Same n_rays, same physical
    field, ray density per metre changing 20-500x with orientation alone.

S9-AN1 -- PATTERN #1 (index-vs-metric registration).
    ``psf_mtf_otf._radial_profile_subpixel`` built its polar sample ring in
    pixel-INDEX space (an ELLIPSE in metres for ``dy != dx``) yet labelled the
    radius with the isotropic ``sqrt(dx*dy)``, disagreeing with its Euclidean-
    binning sibling ``_psf_1d_profile(axis='radial')`` and with the documented
    "true Euclidean distance" contract of ``rayleigh_resolution`` /
    ``sparrow_resolution``.  Measured on a metric-isotropic Airy PSF
    (500 nm, f/4; analytic first zero 2.4400 um, Sparrow 1.8940 um): first
    zero 2.4400 -> 3.3658 -> 4.7600 um and Sparrow 1.8943 -> 1.5971 ->
    1.1376 um as ``dy/dx`` went 1 -> 2 -> 4, i.e. up to 1.95x / 0.60x error
    from the grid aspect ratio alone.

CI-safe: analytic surfaces / analytic Airy PSF, no external assets.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.analysis.psf_mtf_otf import (
    _psf_1d_profile,
    _radial_profile_subpixel,
    sparrow_resolution,
)
from lumenairy.raytrace.from_field import _place_uniform
from lumenairy.raytrace.intersection import _intersect_surface, _surface_sag_xy
from lumenairy.raytrace.surface import Surface
from lumenairy.raytrace.trace import make_fan, make_ray, trace

_WL = 1.31e-6


# ===========================================================================
# S9-RT1 / S9-RT2: raytrace surface geometry
# ===========================================================================

def _sag_at(surface, x, y):
    return float(np.atleast_1d(
        _surface_sag_xy(np.array([float(x)]), np.array([float(y)]),
                        surface))[0])


@pytest.mark.parametrize('kw,h,expect_flat', [
    (dict(radius=np.inf), 10e-3, True),                       # truly flat
    (dict(radius=np.inf, radius_y=np.inf), 10e-3, True),      # explicit flat y
    (dict(radius=np.inf, radius_y=50e-3), 10e-3, False),      # y cylinder
    (dict(radius=np.inf, radius_y=-50e-3), 10e-3, False),     # neg y cylinder
    (dict(radius=np.inf, radius_y=50e-3, conic_y=-1.0), 10e-3, False),
    (dict(radius=50e-3), 10e-3, False),                       # sphere control
])
def test_intersection_lands_on_the_true_sag(kw, h, expect_flat):
    """The intersection point must equal the surface's own sag -- the flat
    fast path may only be taken when the surface really is flat."""
    surface = Surface(**kw)
    sag = _sag_at(surface, 0.0, h)
    assert (sag == 0.0) is expect_flat, (kw, sag)
    rays = make_ray(0.0, h, wavelength=_WL)
    _intersect_surface(rays, surface, 1.0)
    z = float(np.atleast_1d(rays.z)[0])
    opd = float(np.atleast_1d(rays.opd)[0])
    assert z == pytest.approx(sag, abs=1e-11), (kw, z, sag)
    # an axial ray travels exactly the sag distance to reach the surface
    assert opd == pytest.approx(sag, abs=1e-11), (kw, opd, sag)


def _cyl_stack(**kw):
    return [Surface(glass_before='air', glass_after='N-BK7',
                    thickness=3e-3, **kw),
            Surface(radius=np.inf, glass_before='N-BK7', glass_after='air',
                    thickness=0.0)]


def _fan_exit(surfs, axis='y'):
    rays = make_fan(axis=axis, semi_aperture=8e-3, n_rays=5, wavelength=_WL)
    out = trace(rays, surfs, _WL).ray_history[-1]
    return (np.asarray(getattr(out, axis), dtype=float).copy(),
            np.asarray(out.opd, dtype=float).copy())


@pytest.mark.parametrize('conic_y', [0.0, None])
def test_y_cylinder_matches_the_sphere_oracle_in_a_meridional_fan(conic_y):
    """A y-only cylinder of radius R must be INDISTINGUISHABLE from a sphere
    of radius R for a fan confined to the y = meridional plane (x == 0), and
    ``conic_y=None`` must behave as ``conic_y = conic`` (S9-RT2)."""
    R = 60e-3
    kw = dict(radius=np.inf, radius_y=R)
    if conic_y is not None:
        kw['conic_y'] = conic_y
    y_cyl, opd_cyl = _fan_exit(_cyl_stack(**kw))
    y_sph, opd_sph = _fan_exit(_cyl_stack(radius=R))
    assert np.allclose(y_cyl, y_sph, atol=1e-12), (y_cyl, y_sph)
    assert np.allclose(opd_cyl, opd_sph, atol=1e-12), (opd_cyl, opd_sph)
    # sanity: the sphere oracle really does bend (so the pin is not vacuous);
    # pre-fix the cylinder returned the FLAT answer (|y| == 8 mm, flat OPD).
    assert abs(y_sph).max() < 7.95e-3
    assert np.ptp(opd_sph) > 100e-6


def test_y_cylinder_has_no_power_in_the_sagittal_fan():
    """The same y cylinder must remain FLAT for an x fan (y == 0): power on
    one axis only."""
    y_flat, opd_flat = _fan_exit(_cyl_stack(radius=np.inf, radius_y=60e-3),
                                 axis='x')
    assert np.allclose(np.abs(y_flat), [8e-3, 4e-3, 0.0, 4e-3, 8e-3],
                       atol=1e-12)
    assert np.ptp(opd_flat) == pytest.approx(0.0, abs=1e-15)


def test_biconic_conic_y_none_defaults_to_conic_x():
    """``conic_y=None`` must reproduce ``conic_y=conic`` exactly (both the sag
    and the surface-NORMAL path); pre-fix the normal path raised TypeError."""
    from lumenairy.raytrace.surface import _surface_sag_derivatives_xy
    x = np.array([1e-3, 3e-3])
    y = np.array([2e-3, 4e-3])
    for conic in (0.0, -0.6):
        a = Surface(radius=50e-3, radius_y=60e-3, conic=conic)
        b = Surface(radius=50e-3, radius_y=60e-3, conic=conic, conic_y=conic)
        for va, vb in zip(_surface_sag_derivatives_xy(x, y, a),
                          _surface_sag_derivatives_xy(x, y, b)):
            assert np.allclose(va, vb, atol=0.0, rtol=0.0), conic
        assert np.allclose(_surface_sag_xy(x, y, a),
                           _surface_sag_xy(x, y, b), atol=0.0, rtol=0.0)


# ===========================================================================
# S9-RT3: ray placement must not depend on the grid's ORIENTATION
# ===========================================================================

def _pitches(Nx, Ny, n_rays=64):
    E = np.ones((Ny, Nx), dtype=np.complex128)
    _, _, ix, iy = _place_uniform(E, 2e-6, 2e-6, n_rays, 0.0)
    ux, uy = np.unique(ix), np.unique(iy)
    px = float(np.diff(ux).mean()) if ux.size > 1 else float('nan')
    py = float(np.diff(uy).mean()) if uy.size > 1 else float('nan')
    return px, py


@pytest.mark.parametrize('Nx,Ny', [(64, 256), (32, 512), (100, 200),
                                   (128, 128), (200, 100)])
def test_uniform_ray_placement_is_transpose_symmetric(Nx, Ny):
    """Transposing the grid must transpose the ray pitch -- nothing else.  The
    pre-fix code was correct only for wide grids, so this pins the tall ones
    to their own transposes.  The tolerance absorbs the +-1 rounding of the
    asymmetric ``nx_grid = ceil(n_rays / ny_grid)`` closure (11 vs 12 cells on
    a 100x200 grid); it is far tighter than the 20x-511x pre-fix breach."""
    px, py = _pitches(Nx, Ny)
    qx, qy = _pitches(Ny, Nx)
    assert px == pytest.approx(qy, rel=0.15), (Nx, Ny, px, qy)
    assert py == pytest.approx(qx, rel=0.15), (Nx, Ny, py, qx)


@pytest.mark.parametrize('Nx,Ny', [(64, 256), (32, 512), (100, 200)])
def test_uniform_ray_placement_stays_near_isotropic(Nx, Ny):
    """The sub-grid must be near-isotropic in PIXELS (hence in metres for a
    square pitch): pre-fix the tall grids ran 20x-511x anisotropic."""
    px, py = _pitches(Nx, Ny)
    assert max(px, py) / min(px, py) < 2.5, (Nx, Ny, px, py)


# ===========================================================================
# S9-AN1: radial PSF profile must use a METRIC circle
# ===========================================================================

_LAM_A, _FNUM = 500e-9, 4.0


def _airy(N, dx, dy):
    from scipy.special import j1
    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dy
    R = np.hypot(*np.meshgrid(x, y))
    v = np.pi * R / (_LAM_A * _FNUM)
    vs = np.where(v == 0.0, 1.0, v)
    return np.where(v == 0.0, 1.0, 2.0 * j1(vs) / vs) ** 2


def _first_min(r, p):
    for i in range(1, p.size - 1):
        if p[i] < p[i - 1] and p[i] <= p[i + 1]:
            return float(r[i])
    return float('nan')


@pytest.mark.parametrize('ratio', [1.0, 2.0, 4.0, 0.25])
def test_radial_profile_first_zero_is_grid_aspect_invariant(ratio):
    """A metric-isotropic Airy PSF has ONE first zero (1.22 lam f/#); the
    sub-pixel radial profile must recover it independently of dy/dx."""
    dx = 80e-9
    dy = dx * ratio
    psf = _airy(256, dx, dy)
    r, p = _radial_profile_subpixel(psf, dx, dy)
    fz = _first_min(r, p)
    truth = 1.22 * _LAM_A * _FNUM
    assert fz == pytest.approx(truth, rel=0.03), (ratio, fz, truth)
    # cross-check against the Euclidean-binning sibling
    rb, pb = _psf_1d_profile(psf, dx, dy=dy, axis='radial')
    assert _first_min(rb, pb) == pytest.approx(truth, rel=0.05)


@pytest.mark.parametrize('ratio', [1.0, 2.0, 4.0, 0.25])
def test_sparrow_radial_is_grid_aspect_invariant(ratio):
    """``sparrow_resolution(axis='radial')`` consumes the profile above; it
    must return the analytic 0.947 lam f/# for every grid aspect (pre-fix it
    read 1.8943 / 1.5971 / 1.1376 um at ratio 1 / 2 / 4)."""
    dx = 80e-9
    dy = dx * ratio
    got = sparrow_resolution(_airy(256, dx, dy), dx, axis='radial', dy=dy)
    assert got == pytest.approx(0.947 * _LAM_A * _FNUM, rel=0.02), (ratio, got)


def test_square_grid_radial_profile_is_bit_identical():
    """The square-grid branch must be BYTE-identical to the historical
    index-space construction (the fix is opt-in on ``dy != dx`` only)."""
    dx = 80e-9
    psf = _airy(128, dx, dx)
    Ny, Nx = psf.shape
    py, px = divmod(int(np.argmax(psf)), Nx)
    r_max = float(min(py, Ny - 1 - py, px, Nx - 1 - px))
    n_r = int(r_max * 4)
    r_pixels = np.linspace(0.0, r_max, n_r + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    from scipy.ndimage import map_coordinates
    xs = px + r_pixels[:, None] * np.cos(phi)[None, :]
    ys = py + r_pixels[:, None] * np.sin(phi)[None, :]
    legacy = map_coordinates(psf.astype(np.float64),
                             [ys.ravel(), xs.ravel()], order=3,
                             mode='constant', cval=0.0
                             ).reshape(n_r + 1, 64).mean(axis=1)
    r_got, p_got = _radial_profile_subpixel(psf, dx, dx)
    assert np.array_equal(r_got, r_pixels * dx)
    assert np.array_equal(p_got, legacy)


# ===========================================================================
# S9-AN2: the all-NaN distortion guard must not be short-circuited by the
# forced theta=0 zero
# ===========================================================================

def _singlet_surfaces(sd):
    return [Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                    thickness=4e-3, semi_diameter=sd),
            Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                    thickness=0.0, semi_diameter=sd)]


def test_distortion_reports_nan_when_every_chief_ray_is_vignetted():
    """``thetas_deg`` always starts at 0, where ``distortion_pct`` is FORCED to
    0.0, so ``isfinite(distortion_pct).any()`` was unconditionally True and the
    all-NaN branch was dead: a fully-clipped system reported
    ``max_distortion_pct = 0.0`` ("no distortion")."""
    import warnings as _w
    from lumenairy.analysis.field import distortion_vs_field
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        r = distortion_vs_field(_singlet_surfaces(1e-9), 550e-9, 10.0,
                                n_points=7)
    # only the on-axis y=0 chief survives, and it carries no distortion info
    assert int(np.isfinite(r.h_chief).sum()) == 1
    assert np.isnan(r.max_distortion_pct), r.max_distortion_pct
    assert r.sign == 'unknown'


def test_distortion_healthy_case_is_unchanged():
    """Bit-identity companion: with a real aperture the reported value is the
    pre-fix one (0.16461028308262293 % barrel on this singlet)."""
    import warnings as _w
    from lumenairy.analysis.field import distortion_vs_field
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        r = distortion_vs_field(_singlet_surfaces(12e-3), 550e-9, 10.0,
                                n_points=7)
    assert int(np.isfinite(r.h_chief).sum()) == 7
    assert r.max_distortion_pct == pytest.approx(0.16461028308262293,
                                                 rel=1e-9)
    assert r.sign == 'barrel'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
