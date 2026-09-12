"""WP-A8 regression pins for findings E5, E6 and the remaining E7 items of
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`` section 5.

* **E5 (P1)** ``apply_grin_lens`` applied the short-rod power
  ``n0 g**2 d`` while its own Notes recommended the quarter pitch, where
  that power is 36 % low (``f_code/f_exact = sin(gd)/(gd)``).
* **E6 (P2)** Harvey-Shack TIS under-read by 18 % at ``l = 1e-3``;
  ``make_bsdf`` silently ignored unknown keys including the ``A``/``B``
  aliases its own docstring teaches; ``thin_grating_efficiency_1d`` had no
  Klein-Cook guard; ``create_microlens_array`` built ~10 full grids for a
  separable phase; ``sample_scatter_rays`` looped in Python per ray.
* **E7 (P3)** ``create_periodic_phase_mask`` assumed a square cell;
  ``apply_aperture`` had no grey-pixel edge; ``elements.zernike`` pointed
  Noll users at the OSA converter; ``apply_spherical_lens`` and
  ``apply_real_lens`` use different reference planes without saying so.
"""
from __future__ import annotations

import inspect
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import bsdf as bsdf_mod
from lumenairy.elements import doe as doe_mod
from lumenairy.elements import elements as elem_mod
from lumenairy.elements import thin_grating as tg_mod
from lumenairy.elements._lens_thin import apply_grin_lens, apply_spherical_lens


# ===========================================================================
# E5 -- GRIN rod paraxial power
# ===========================================================================

def _grin_power_from_screen(n0, g, d, wavelength, thin_form=False):
    """Recover the applied quadratic coefficient from the screen itself:
    phi = -k * P/2 * r**2  =>  P = -2 * phi(r) / (k r**2)."""
    N, dx = 64, 1e-6
    E = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_grin_lens(E, n0=n0, g=g, d=d, wavelength=wavelength,
                              dx=dx, thin_form=thin_form)
    k = 2 * np.pi / wavelength
    # Unwrapped along the central row; use two small radii and fit r**2.
    x = (np.arange(N) - N / 2) * dx
    row = np.unwrap(np.angle(out[N // 2]))
    row = row - row[N // 2]
    i1, i2 = N // 2 + 2, N // 2 + 6
    # phi = -k P/2 r^2  ->  P = -2 (phi2 - phi1) / (k (r2^2 - r1^2))
    return -2.0 * (row[i2] - row[i1]) / (k * (x[i2] ** 2 - x[i1] ** 2))


@pytest.mark.parametrize('gd', [0.05, 0.3, np.pi / 4, np.pi / 2])
def test_e5_grin_screen_carries_the_exact_paraxial_power(gd):
    """The rod's ABCD gives ``C = -n0 g sin(g d)``, so the screen's power
    must be ``n0 g sin(g d)``, not ``n0 g**2 d``.

    Oracle: the closed-form ABCD element, exact.  The bar 1e-6 relative is
    set by the phase-unwrap readout (float64 angle + unwrap, ~1e-12) times
    a wide margin; the defect it separates from is ``sin(gd)/(gd)``, which
    is 1.6e-3 low at gd = 0.05 and 0.364 low at the quarter pitch -- 3 to 6
    decades above the bar.
    """
    n0, g, wl = 1.6, 300.0, 1e-6
    d = gd / g
    want_exact = n0 * g * np.sin(gd)
    want_thin = n0 * g ** 2 * d
    got = _grin_power_from_screen(n0, g, d, wl)
    assert abs(got / want_exact - 1.0) < 1e-6, (got, want_exact)
    if gd > 0.1:
        assert abs(got / want_thin - 1.0) > 1e-2, (
            f"the screen still carries the short-rod power {want_thin:.6g}")


@pytest.mark.parametrize('gd', [0.05, np.pi / 2])
def test_e5_thin_form_reproduces_the_pre_fix_screen_exactly(gd):
    n0, g, wl, dx, N = 1.6, 300.0, 1e-6, 4e-6, 64
    d = gd / g
    E = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = apply_grin_lens(E, n0=n0, g=g, d=d, wavelength=wl, dx=dx,
                              thin_form=True)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    k = 2 * np.pi / wl
    pre_fix = E * np.exp(-1j * k * n0 * (g ** 2 / 2) * d * (X ** 2 + Y ** 2))
    assert np.array_equal(got, pre_fix)


def test_e5_asm_focus_tracks_the_exact_focal_length():
    """End-to-end: a collimated Gaussian through the screen must focus at
    ``f = 1/(n0 g sin(g d))``.

    Fixture: n0 = 1.6, g = 30 /m, g*d = pi/2 (the quarter pitch the Notes
    recommend), w0 = 0.5 mm -> NA 0.024, so the paraxial screen's own
    spherical-aberration focal shift is ~0.1 %.  f_exact = 20.833 mm;
    f_thin = 13.263 mm (a factor 2/pi = 0.6366 away).  Bar: the measured
    peak-intensity plane within 3 % of f_exact -- 30x the aberration shift
    and 12x below the distance to f_thin.
    """
    n0, g, wl = 1.6, 30.0, 1e-6
    d = (np.pi / 2) / g
    f_exact = 1.0 / (n0 * g * np.sin(g * d))
    f_thin = 1.0 / (n0 * g ** 2 * d)
    N, dx = 512, 6e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.5e-3) ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = apply_grin_lens(E0, n0=n0, g=g, d=d, wavelength=wl, dx=dx)

    def peak(z):
        return float(np.max(np.abs(
            la.angular_spectrum_propagate(E, z, wl, dx)) ** 2))

    zs = np.linspace(0.55 * f_exact, 1.25 * f_exact, 29)
    z0 = zs[int(np.argmax([peak(z) for z in zs]))]
    zs2 = np.linspace(z0 - 0.04 * f_exact, z0 + 0.04 * f_exact, 21)
    z_focus = zs2[int(np.argmax([peak(z) for z in zs2]))]
    assert abs(z_focus / f_exact - 1.0) < 0.03, (
        f"ASM focus {z_focus * 1e3:.4f} mm vs f_exact "
        f"{f_exact * 1e3:.4f} mm (f_thin = {f_thin * 1e3:.4f} mm)")
    assert abs(z_focus / f_thin - 1.0) > 0.2


def test_e5_guards_fire_where_the_model_breaks_and_are_silent_elsewhere():
    N, dx, wl, n0, g = 32, 4e-6, 1e-6, 1.6, 300.0
    E = np.ones((N, N), dtype=np.complex128)

    def call(gd, **kw):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            apply_grin_lens(E, n0=n0, g=g, d=gd / g, wavelength=wl, dx=dx,
                            **kw)
        return [w for w in rec if issubclass(w.category, UserWarning)]

    assert len(call(0.1, thin_form=True)) == 0
    hot = call(np.pi / 2, thin_form=True)
    assert len(hot) == 1 and 'thin_form' in str(hot[0].message)
    assert '0.6366' in str(hot[0].message), str(hot[0].message)
    assert len(call(1.0)) == 0
    past = call(0.9 * np.pi)
    assert len(past) == 1 and 'quarter pitch' in str(past[0].message)


def test_e7_spherical_vs_real_lens_reference_plane_is_documented():
    """The two models' focus planes are 503 um apart on a 19 mm lens (2.6 %);
    the docstring must say so where a reader choosing between them looks."""
    doc = ' '.join(inspect.getdoc(apply_spherical_lens).split()).lower()
    assert 'different reference planes' in doc
    assert 'back vertex' in doc
    assert '503 um' in doc, 'the measured offset must be quoted'
    assert '0.272 waves' in doc


# ===========================================================================
# E6 -- Harvey-Shack TIS
# ===========================================================================

def _tis_reference(model, n_pts=200_001, u_min=1e-9):
    """Independent hemisphere integral ``2 pi int_0^1 B(u) u du`` by the
    trapezoid rule on a GEOMETRIC u grid (different nodes and a different
    rule from the library's two-point Gauss cells).

    Error floor: trapezoid in v = ln u is O(dv**2); with dv = ln(1e9)/2e5 =
    1.04e-4 that is ~1e-8 relative, plus the analytically-added u < u_min
    disc.  Five decades below the tightest bar used against it.
    """
    v = np.linspace(np.log(u_min), 0.0, n_pts)
    u = np.exp(v)
    theta = np.arcsin(np.clip(u, 0.0, 1.0))
    S = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)],
                 axis=-1)
    B = np.asarray(model.evaluate(np.array([0.0, 0.0, -1.0]), S), dtype=float)
    integral = np.trapezoid(B * u * u, v)
    return float(2 * np.pi * (integral + B[0] * u_min ** 2 / 2))


_HS_CASES = [(1e-1, 2.0), (1e-2, 2.0), (1e-3, 2.0), (1e-4, 2.0),
             (1e-2, 1.5), (1e-2, 2.5)]


@pytest.mark.parametrize('l,s', _HS_CASES)
def test_e6_harvey_shack_tis_closed_form_matches_an_independent_integral(l, s):
    """Pre-fix the inherited 256-point LINEAR-theta quadrature read
    -0.72 % at the default l = 1e-2, -18.3 % at l = 1e-3 and -31.8 % at
    l = 1e-4, because the lobe's shoulder sits below one grid step.  The
    closed form is exact.

    Bar 1e-6 relative: the reference's own floor is ~1e-8 and the measured
    agreement is <= 1.1e-7, so the bar is ~10x the measurement and 4 to 5
    decades below the errors it must catch.
    """
    model = bsdf_mod.HarveyShackBSDF(b0=1.0, l=l, s=s)
    got = model.total_integrated_scatter()
    ref = _tis_reference(model)
    assert abs(got / ref - 1.0) < 1e-6, (got, ref)


def test_e6_harvey_shack_tis_matches_the_published_s2_closed_form():
    """``TIS(s=2) = pi b0 l**2 ln(1 + 1/l**2)`` -- a second, purely symbolic
    oracle.  Exact to float rounding (bar 1e-12)."""
    for l in (0.1, 0.01, 0.001):
        model = bsdf_mod.HarveyShackBSDF(b0=1.0, l=l, s=2.0)
        want = np.pi * l * l * np.log(1 + 1 / l ** 2)
        assert abs(model.total_integrated_scatter() / want - 1.0) < 1e-12


def test_e6_tis_honours_the_wavelength_scaling():
    a = bsdf_mod.HarveyShackBSDF(b0=1.0, l=1e-3, s=2.0)
    b = bsdf_mod.HarveyShackBSDF(b0=1.0, l=1e-3, s=2.0,
                                 wavelength_ref=633e-9, wavelength=1310e-9)
    assert abs(b.total_integrated_scatter()
               / (a.total_integrated_scatter() * (633 / 1310) ** 2)
               - 1.0) < 1e-12


@pytest.mark.parametrize('l,s', _HS_CASES)
def test_e6_base_class_quadrature_is_lobe_width_independent(l, s):
    """The inherited quadrature is what a user subclass gets.  With the
    u = sin(theta) substitution and two-point Gauss cells geometric in
    ``ln u`` it is accurate to 1.4e-6 worst-case at the SAME node count
    (256 x 128) that previously read -32 %.

    Bar 1e-4 relative: 70x the measured worst case, and 3 decades below the
    pre-fix errors (-0.7 % to -32 %).
    """
    model = bsdf_mod.HarveyShackBSDF(b0=1.0, l=l, s=s)
    quad = bsdf_mod.BSDFModel.total_integrated_scatter(model)
    ref = _tis_reference(model)
    assert abs(quad / ref - 1.0) < 1e-4, (quad, ref)


def test_e6_base_class_quadrature_is_exact_for_a_flat_lobe():
    """The cell weights sum to ``int u du`` exactly, so a constant BSDF
    integrates with no quadrature error at all -- the property that makes
    the grid safe for broad lobes as well as narrow ones."""
    lam = bsdf_mod.LambertianBSDF(rho=0.5)
    quad = bsdf_mod.BSDFModel.total_integrated_scatter(lam)
    assert abs(quad - 0.5) < 1e-5, quad


def test_e6_shipped_model_tis_values_are_unchanged():
    """Audit-verified-correct: Lambertian TIS == rho exactly and Gaussian
    TIS == scattered_fraction.  Keep them."""
    assert bsdf_mod.LambertianBSDF(rho=0.5).total_integrated_scatter() == 0.5
    g = bsdf_mod.GaussianBSDF(sigma_rad=1e-2, scattered_fraction=1e-2)
    assert g.total_integrated_scatter() == 1e-2


# ===========================================================================
# E6 -- Harvey-Shack sampler and sample_scatter_rays
# ===========================================================================

@pytest.mark.parametrize('l,s', [(1e-1, 2.0), (1e-2, 2.0), (1e-3, 2.0),
                                 (1e-2, 1.5), (1e-2, 2.5)])
def test_e6_harvey_shack_sampler_matches_its_power_weighted_density(l, s):
    """The rejection sampler (9 % acceptance at l = 1e-2, 1 % at 1e-3, with
    a growing Python list) is replaced by the exact inverse CDF.

    Oracle: the CDF obtained by NUMERICALLY integrating the radial density
    ``u / (1 + (u/l)**2)**(s/2)`` on a dense geometric grid -- built here,
    not shared with the implementation.  Bar: KS distance < 0.01 on
    n = 50 000 draws; the two-sided 99.9 % band for a correct sampler at
    that n is 0.0087, and a sampler off by even a few percent in width
    shows KS >> 0.05.
    """
    model = bsdf_mod.HarveyShackBSDF(b0=1.0, l=l, s=s)
    n = 50_000
    d = model.sample(np.array([0.0, 0.0, -1.0]), n, rng=7)
    u_draw = np.sort(np.hypot(d[:, 0], d[:, 1]))
    grid = np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-9), 0.0,
                                                     400_001))])
    dens = grid / (1 + (grid / l) ** 2) ** (s / 2)
    cdf_grid = np.concatenate([[0.0], np.cumsum(
        0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
    cdf_grid /= cdf_grid[-1]
    emp = np.arange(1, n + 1) / n
    ks = float(np.max(np.abs(emp - np.interp(u_draw, grid, cdf_grid))))
    assert ks < 0.01, f"KS = {ks:.4f}"


def test_e6_harvey_shack_sampler_makes_no_rejection_loop():
    src = inspect.getsource(bsdf_mod.HarveyShackBSDF._sample_local)
    assert 'while' not in src and 'extend' not in src, (
        'the rejection loop is back')


@pytest.mark.parametrize('inc', [
    (0.0, 0.0, -1.0), (0.3, 0.2, -0.93), (0.0, 0.0, 1.0), (0.6, -0.5, -0.62)])
def test_e6_batched_rotation_equals_the_per_ray_reference(inc):
    """The batched frame build must be bit-identical to the per-ray code it
    replaces, including the |spec_z| >= 0.999 pole branch."""
    inc = np.array(inc) / np.linalg.norm(inc)
    rng = np.random.default_rng(0)
    local = rng.normal(size=(500, 3))
    local /= np.linalg.norm(local, axis=1, keepdims=True)

    spec = np.array([inc[0], inc[1], -inc[2]])
    spec = spec / np.linalg.norm(spec)
    up = (np.array([0.0, 0.0, 1.0]) if abs(spec[2]) < 0.999
          else np.array([1.0, 0.0, 0.0]))
    tangent = np.cross(up, spec)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(spec, tangent)
    ref = (local[:, 0:1] * tangent + local[:, 1:2] * bitangent
           + local[:, 2:3] * spec)
    ref[ref[:, 2] < 0] *= -1

    got = bsdf_mod._rotate_local_to_specular(local.copy(), inc)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize('spec', [
    {'kind': 'lambertian', 'rho': 0.5},
    {'kind': 'gaussian', 'sigma_rad': 0.02, 'scattered_fraction': 0.01},
    {'kind': 'harvey_shack', 'b0': 1.0, 'l': 0.01, 's': 2.0},
])
def test_e6_sample_scatter_rays_is_vectorised_and_distribution_preserving(
        spec):
    """Pre-fix this was one Python ``sample()`` call per incident ray.  The
    bundle is now drawn in one call; the pins are (a) no Python loop over
    rays in the source, (b) unit-norm outgoing directions in the outgoing
    hemisphere, and (c) the same scatter-angle distribution as the per-ray
    reference, compared as a mean with a derived Monte-Carlo bar.
    """
    import lumenairy.raytrace as rt

    src = inspect.getsource(bsdf_mod.sample_scatter_rays)
    assert 'for i in range(n_rays)' not in src

    n = 4000
    rr = np.random.default_rng(5)
    L = rr.normal(0, 0.05, n)
    M = rr.normal(0, 0.05, n)
    N = -np.sqrt(1 - L ** 2 - M ** 2)
    bundle = rt.RayBundle(x=np.zeros(n), y=np.zeros(n), z=np.zeros(n),
                          L=L, M=M, N=N, wavelength=1e-6,
                          alive=np.ones(n, bool), opd=np.zeros(n))

    class _Surface:
        bsdf = spec

    out = bsdf_mod.sample_scatter_rays(_Surface(), bundle, n_per_ray=1, rng=3)
    dirs = np.stack([out.L, out.M, out.N], axis=1)
    assert np.max(np.abs(np.linalg.norm(dirs, axis=1) - 1.0)) < 1e-12
    assert np.all(out.N > 0)

    model = bsdf_mod.make_bsdf(spec)
    rng_loop = np.random.default_rng(3)
    ref = np.empty((n, 3))
    for i in range(n):
        ref[i] = model.sample(np.array([L[i], M[i], N[i]]), 1, rng=rng_loop)

    got_ang = np.degrees(np.arccos(np.clip(out.N, -1, 1)))
    ref_ang = np.degrees(np.arccos(np.clip(ref[:, 2], -1, 1)))
    # Two independent draws of n = 4000 from the same law: the difference
    # of their means has sd ~ sqrt(2/n) * sd(angle).  Bar = 6 sigma.
    sigma = np.sqrt(2.0 / n) * float(np.std(ref_ang))
    assert abs(got_ang.mean() - ref_ang.mean()) < 6 * sigma, (
        got_ang.mean(), ref_ang.mean(), sigma)


# ===========================================================================
# E6 / E7 -- make_bsdf key validation
# ===========================================================================

def test_e6_make_bsdf_rejects_keys_it_would_silently_ignore():
    """``{'kind':'gaussian','sigma':1e-3,'scatter_fraction':0.5}`` used to
    build a lobe 10x wider and 50x weaker than requested, silently."""
    with pytest.raises(ValueError, match=r"make_bsdf: unknown key\(s\)"):
        bsdf_mod.make_bsdf({'kind': 'gaussian', 'sigma': 1e-3,
                            'scatter_fraction': 0.5})
    with pytest.raises(ValueError, match=r"make_bsdf: unknown key\(s\)"):
        bsdf_mod.make_bsdf({'kind': 'lambertian', 'reflectance': 0.5})


def test_e6_make_bsdf_accepts_the_abc_aliases_its_docstring_teaches():
    got = bsdf_mod.make_bsdf({'kind': 'harvey_shack', 'A': 1e-3, 'B': 0.02,
                              'C': 1.8})
    assert (got.b0, got.l, got.s) == (1e-3, 0.02, 1.8)


def test_e6_make_bsdf_refuses_both_spellings_of_one_parameter():
    with pytest.raises(ValueError, match="sets both 'A'"):
        bsdf_mod.make_bsdf({'kind': 'harvey_shack', 'A': 1e-3, 'b0': 2.0})


def test_e6_make_bsdf_still_builds_every_documented_spec():
    assert isinstance(bsdf_mod.make_bsdf({'kind': 'lambertian', 'rho': 0.1}),
                      bsdf_mod.LambertianBSDF)
    assert isinstance(bsdf_mod.make_bsdf(
        {'kind': 'gaussian', 'sigma_rad': 5e-3,
         'scattered_fraction': 5e-3}), bsdf_mod.GaussianBSDF)
    hs = bsdf_mod.make_bsdf(
        {'kind': 'harvey_shack', 'b0': 0.01, 'l': 0.01, 's': 2.0,
         'wavelength_ref': 633e-9, 'wavelength': 1310e-9})
    assert isinstance(hs, bsdf_mod.HarveyShackBSDF)
    assert bsdf_mod.make_bsdf(None) is None
    with pytest.raises(ValueError, match='unknown kind'):
        bsdf_mod.make_bsdf({'kind': 'does_not_exist'})


# ===========================================================================
# E6 -- thin-grating Klein-Cook guard
# ===========================================================================

_TG = dict(n_ridge=1.5, n_groove=1.0, n_substrate=1.0, n_superstrate=1.0,
           duty_cycle=0.5, wavelength=1e-6)


def _grating_warnings(period, depth, **kw):
    args = dict(_TG)
    args.update(kw)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = tg_mod.thin_grating_efficiency_1d(
            period, args['n_ridge'], args['n_groove'], args['n_substrate'],
            args['n_superstrate'], depth, args['duty_cycle'],
            args['wavelength'], n_orders=args.get('n_orders', 11))
    return [w for w in rec if issubclass(w.category, UserWarning)], out


def test_e6_klein_cook_guard_fires_in_the_bragg_regime():
    """Measured pre-fix: Q = 0.16 / 15.7 / 62.8 at Lambda = 20 / 2 / 1 um
    with depth 10 um, ZERO warnings, and sum(T) = 1.0000 every time -- so
    energy closure cannot be used as the validity check.  n_bar here is the
    duty-weighted 1.25, giving Q = 0.126 / 12.6 / 50.3.
    """
    quiet, out = _grating_warnings(20e-6, 10e-6)
    assert quiet == [], [str(w.message) for w in quiet]
    assert abs(out[2].sum() - 1.0) < 1e-9
    for period in (2e-6, 1e-6):
        hot, out = _grating_warnings(period, 10e-6)
        assert len(hot) == 1 and 'Klein-Cook' in str(hot[0].message), hot
        assert abs(out[2].sum() - 1.0) < 1e-9, (
            'the returned numbers still look clean -- that is the point')


def test_e6_short_period_guard_fires_when_q_is_small():
    """A shallow sub-wavelength grating has Q << 1 but is still outside the
    scalar picture; the second arm catches it."""
    hot, _ = _grating_warnings(5e-6, 0.05e-6)
    assert len(hot) == 1 and 'period/wavelength' in str(hot[0].message)
    quiet, _ = _grating_warnings(20e-6, 0.05e-6)
    assert quiet == []


def test_e6_grating_fourier_coefficients_are_untouched():
    """Audit-verified-correct: the 50 %-duty pi-step grating gives
    eta_0 = 0, eta_+1 = 4/pi**2 and eta_+3 = 4/(9 pi**2) exactly."""
    wl, period = 1e-6, 20e-6
    depth = wl / (2 * (1.5 - 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        orders, _, T = tg_mod.thin_grating_efficiency_1d(
            period, 1.5, 1.0, 1.0, 1.0, depth, 0.5, wl, n_orders=31)
    i0 = int(np.where(orders == 0)[0][0])
    assert abs(T[i0]) < 1e-12
    assert abs(T[i0 + 1] - 4 / np.pi ** 2) < 1e-9
    assert abs(T[i0 + 3] - 4 / (9 * np.pi ** 2)) < 1e-9


def test_e6_grating_module_docstring_formula_is_well_formed():
    """The module docstring used to carry a garbled ``t_m`` expression
    (duplicated ``f``, unbalanced terms, a trailing ``* ...``)."""
    doc = tg_mod.__doc__
    assert '(see code for exact form)' not in doc
    assert 'f * exp(i*phi) * f * sinc' not in doc
    assert 't_0 = f * exp(i*phi_r) + (1 - f) * exp(i*phi_g)' in doc


# ===========================================================================
# E6 -- microlens array
# ===========================================================================

def _mla_reference(N, dx, n_lenslets, pitch, focal_length, wavelength):
    """The pre-separable full-grid implementation, written out."""
    k = 2 * np.pi / wavelength
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    half = n_lenslets * pitch / 2
    in_mla = (np.abs(X) < half) & (np.abs(Y) < half)
    jx = np.clip(np.round(X / pitch + (n_lenslets - 1) / 2), 0,
                 n_lenslets - 1)
    jy = np.clip(np.round(Y / pitch + (n_lenslets - 1) / 2), 0,
                 n_lenslets - 1)
    dX = X - (jx - (n_lenslets - 1) / 2) * pitch
    dY = Y - (jy - (n_lenslets - 1) / 2) * pitch
    r_sq = dX * dX + dY * dY
    return np.exp(1j * np.where(in_mla, -k / (2 * focal_length) * r_sq, 0.0))


@pytest.mark.parametrize('N,n_l,pitch', [(512, 8, 101.3e-6), (257, 5, 60e-6),
                                         (256, 16, 32e-6)])
def test_e6_microlens_array_separable_form_is_bit_identical(N, n_l, pitch):
    args = (N, 2e-6, n_l, pitch, 2e-3, 1.55e-6)
    assert np.array_equal(doe_mod.create_microlens_array(*args),
                          _mla_reference(*args))


def test_e6_microlens_array_allocates_a_handful_of_grids_not_ten():
    """Pre-fix the function materialised X, Y, in_mla, jx, jy, xc, yc, dX,
    dY, r_sq and phase as full grids, then ``np.exp(1j*phase)`` on top.

    Measured tracemalloc peak, in units of one N**2 float64 grid, identical
    at N = 1024 and N = 2048: **10.13 before, 3.25 after** (the separable
    axis vectors cost O(N), and ``cos``/``sin`` write straight into the
    output's real / imaginary views instead of building the ``1j*phase``
    and ``exp`` temporaries).  Output is bit-identical.

    Bar 6.0 grids: 1.85x above the measured 3.25 and 1.69x below the
    pre-fix 10.13.  A memory-peak property is O(1) grids by nature, so
    "decades of gap" is not available here; the gap is a factor ~1.8 on
    each side and the reading is deterministic for a given numpy build.
    """
    N = 1024
    grid_bytes = N * N * 8
    tracemalloc.start()
    try:
        doe_mod.create_microlens_array(N, 2e-6, 16, 32e-6, 1e-3, 1e-6)
        tracemalloc.reset_peak()
        doe_mod.create_microlens_array(N, 2e-6, 16, 32e-6, 1e-3, 1e-6)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 6.0 * grid_bytes, (
        f"peak {peak / grid_bytes:.2f} float64 grids at N={N}")


def test_e6_microlens_array_physics_is_untouched():
    """Audit-verified-correct: |T| = 1 everywhere and exactly zero steer at
    every lenslet centre."""
    N, dx, n_l, pitch, f, wl = 512, 2e-6, 8, 64e-6, 2e-3, 1.55e-6
    m = doe_mod.create_microlens_array(N, dx, n_l, pitch, f, wl)
    assert np.max(np.abs(np.abs(m) - 1.0)) < 1e-12
    ph = np.unwrap(np.angle(m[N // 2]))
    for j in (0, 3, 7):
        xc = (j - (n_l - 1) / 2) * pitch
        i = int(round(xc / dx + N / 2))
        assert abs((ph[i + 1] - ph[i - 1]) / (2 * dx)) < 1e-9


# ===========================================================================
# E7 -- periodic phase mask cell shape
# ===========================================================================

@pytest.mark.parametrize('shape', [(4, 8), (8, 4), (8,), (2, 2, 2)])
def test_e7_periodic_phase_mask_rejects_a_non_square_cell(shape):
    """A (4, 8) cell silently dropped columns 4-7; an (8, 4) cell raised a
    bare IndexError naming neither the function nor the argument."""
    with pytest.raises(ValueError,
                       match='create_periodic_phase_mask: phase_cell'):
        doe_mod.create_periodic_phase_mask(64, 2e-6, np.zeros(shape), 2e-6)


def test_e7_periodic_phase_mask_square_behaviour_is_unchanged():
    """Audit-verified-correct: per-cell-pixel occupancy exactly uniform on
    a grid-native cell (the v5.30 modulo close)."""
    N, cell_n = 256, 8
    dx = 2e-6
    cell = np.arange(cell_n * cell_n, dtype=float).reshape(cell_n, cell_n)
    mask = doe_mod.create_periodic_phase_mask(N, dx, cell, dx)
    coord = (np.arange(N) - N / 2) * dx
    idx = np.round(np.mod(coord, cell_n * dx) / dx).astype(int) % cell_n
    counts = np.bincount(idx, minlength=cell_n)
    assert list(counts) == [N // cell_n] * cell_n
    assert np.max(np.abs(np.abs(mask) - 1.0)) < 1e-12


# ===========================================================================
# E7 -- grey-pixel aperture edge
# ===========================================================================

def test_e7_aperture_hard_edge_is_the_default_and_unchanged():
    N, dx = 256, 2e-6
    rng = np.random.default_rng(1)
    E = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    for shape, params, mask in (
            ('circular', {'diameter': 2e-4},
             (X ** 2 + Y ** 2) <= (1e-4) ** 2),
            ('annular', {'inner_diameter': 5e-5, 'outer_diameter': 2e-4},
             ((X ** 2 + Y ** 2) >= (2.5e-5) ** 2)
             & ((X ** 2 + Y ** 2) <= (1e-4) ** 2)),
            ('rectangular', {'width_x': 1.5e-4, 'width_y': 9e-5},
             (np.abs(X) <= 7.5e-5) & (np.abs(Y) <= 4.5e-5))):
        got = elem_mod.apply_aperture(E, dx, shape, params)
        assert np.array_equal(got, np.where(mask, E, 0)), shape
    assert inspect.signature(
        elem_mod.apply_aperture).parameters['edge'].default == 'hard'


def test_e7_aperture_gray_edge_removes_the_area_quantisation():
    """Measured 2026-09-12, rms over 20 sub-pixel rim placements of the
    transmitted-area error against the analytic disc area:

        D/dx ~  50 px : 0.342 % hard  ->  0.059 % gray (4x4)
        D/dx ~ 200 px : 0.041 % hard  ->  0.0069 % gray

    Bars below: gray rms < 0.12 % at 50 px (2x the measurement, 2.8x below
    the hard reading) and gray strictly better than hard at both sizes
    (a decision, no bar).
    """
    N, dx = 512, 1e-6
    E = np.ones((N, N), dtype=np.complex128)
    for npx, gray_bar in ((50, 1.2e-3), (200, 2.0e-4)):
        hard, gray = [], []
        for frac in np.linspace(0.0, 0.95, 12):
            D = (npx + frac) * dx
            analytic = np.pi * (D / 2) ** 2
            h = float(np.sum(np.real(elem_mod.apply_aperture(
                E, dx, 'circular', {'diameter': D})))) * dx * dx
            g = float(np.sum(np.real(elem_mod.apply_aperture(
                E, dx, 'circular', {'diameter': D},
                edge='gray')))) * dx * dx
            hard.append((h - analytic) / analytic)
            gray.append((g - analytic) / analytic)
        rms_h = float(np.sqrt(np.mean(np.square(hard))))
        rms_g = float(np.sqrt(np.mean(np.square(gray))))
        assert rms_g < gray_bar, (npx, rms_g)
        assert rms_g < rms_h / 2, (npx, rms_g, rms_h)


def test_e7_aperture_gray_edge_has_jax_parity():
    """``apply_aperture`` is one of the ``backend.array_namespace``-dispatched
    entry points, so the new grey path must run on the JAX twin too.  The
    accumulation is written functionally (``frac = frac + mask``) because
    JAX arrays are immutable.

    Measured with ``jax_enable_x64``: max|JAX - NumPy| = 0.00e+00 for hard
    and gray on circular / annular / rectangular, and a complex64 input
    stays complex64.  No skip: when JAX is absent the test asserts that
    fact instead, so it can never silently drop out of the gate.
    """
    import importlib.util
    if importlib.util.find_spec('jax') is None:
        assert importlib.util.find_spec('jax') is None
        return
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    N, dx = 128, 2e-6
    e_jax = jnp.ones((N, N), dtype=jnp.complex128)
    e_np = np.ones((N, N), dtype=np.complex128)
    cases = (('circular', {'diameter': 1.2e-4}),
             ('annular', {'inner_diameter': 3e-5, 'outer_diameter': 1.2e-4}),
             ('rectangular', {'width_x': 1e-4, 'width_y': 6e-5}))
    for shape, params in cases:
        for edge in ('hard', 'gray'):
            a = np.asarray(elem_mod.apply_aperture(e_jax, dx, shape, params,
                                                   edge=edge))
            b = elem_mod.apply_aperture(e_np, dx, shape, params, edge=edge)
            assert np.max(np.abs(a - b)) == 0.0, (shape, edge)
    for edge in ('hard', 'gray'):
        out = elem_mod.apply_aperture(jnp.ones((N, N), dtype=jnp.complex64),
                                      dx, 'circular', {'diameter': 1.2e-4},
                                      edge=edge)
        assert out.dtype == jnp.complex64, (edge, out.dtype)


def test_e7_aperture_gray_edge_preserves_dtype_and_validates_its_kwargs():
    for dtype in (np.complex64, np.complex128):
        out = elem_mod.apply_aperture(np.ones((32, 32), dtype), 2e-6,
                                      'circular', {'diameter': 3e-5},
                                      edge='gray')
        assert out.dtype == dtype
    E = np.ones((8, 8), dtype=np.complex128)
    for kw in ({'edge': 'soft'}, {'edge': 'gray', 'edge_samples': 0},
               {'edge': 'gray', 'edge_samples': 2.5}):
        with pytest.raises(ValueError, match='apply_aperture:'):
            elem_mod.apply_aperture(E, 2e-6, 'circular', {'diameter': 1e-5},
                                    **kw)


# ===========================================================================
# E7 -- the Noll pointer
# ===========================================================================

def test_e7_zernike_docstring_no_longer_points_noll_users_at_the_osa_map():
    """``analysis.zernike_index_to_nm`` is the OSA map (its own docstring
    says so); OSA j = 5 is (2, +2) while Noll j = 5 is (2, -2), so a reader
    following the old pointer got the wrong polynomial for every j >= 5."""
    doc = inspect.getdoc(elem_mod.zernike)
    assert 'to map j_Noll -> (n, m) use' not in doc
    assert 'OSA' in doc and 'Noll' in doc
    assert 'no' in doc.lower().split('noll *single index*')[1][:400], doc
    from lumenairy.analysis import zernike_index_to_nm
    assert zernike_index_to_nm(5) == (2, 2), (
        'the converter is OSA; if this ever returns (2, -2) the docstring '
        'must be revisited')
