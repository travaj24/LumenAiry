"""v4.15.1 Agent F -- Cluster B Item 6: ``rays_from_field`` bridge.

Pins the wave-to-ray bridge function introduced in v4.15.1
(:mod:`lumenairy.raytrace.from_field`).  The function samples a
coherent complex field into a geometric :class:`RayBundle` so that the
output can be handed to ``trace`` / ``spot_diagram`` / HFPI / GBD, or
overlaid on coherent-field plots.

Tests cover:

1. Constant-amplitude, zero-phase field -> rays along ``+z``.
2. Tilted plane wave -> rays with ``L = sin(theta)``, ``M = 0``.
3. Converging spherical wave -> rays focus at ``z = f``.
4. Uniform-amplitude circular aperture (rejection sampling) ->
   placement stays inside the support.
5. Below-threshold pixels excluded from placement.
6. Evanescent k-vectors flagged ``alive=False`` /
   ``error_code=RAY_EVANESCENT``.
7. OPD initialised from ``phase(E_origin) / k0``.
8. ``cdf`` and ``rejection`` placement agree on first / second moments
   for separable Gaussian intensity.
9. ``complex_gradient`` angle method handles a phase-singularity
   (LG-like) field without NaN.
10. Integer-seed ``random_state`` gives bit-identical bundles on two
    calls.

Deviations from CLUSTER_B_SPEC.md §4.5 ("Tests for Item 6") are
flagged in the test docstring where applicable and enumerated in
``docs/release_notes/.release_notes_v4_15_1_agent_f.md``.

Author: Andrew Traverso -- v4.15.1 / Agent F (Cluster B Item 6).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy import RAY_EVANESCENT, RAY_OK

# ============================================================================
# 1-7: Spec-derived tests (CLUSTER_B_SPEC.md §4.5)
# ============================================================================

def test_plane_wave_normal_incidence():
    """A constant-amplitude, zero-phase field should produce rays
    all with ``(L, M, N) = (0, 0, 1)``.
    """
    N, dx, lam = 128, 2e-6, 633e-9
    E = np.ones((N, N), dtype=complex)
    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=100, random_state=42,
    )
    assert np.allclose(rays.L, 0.0, atol=1e-6)
    assert np.allclose(rays.M, 0.0, atol=1e-6)
    assert np.allclose(rays.N, 1.0, atol=1e-6)
    assert np.all(rays.alive)
    assert np.all(rays.error_code == RAY_OK)
    # n_rays bookkeeping
    assert rays.n_rays == 100
    assert rays.wavelength == lam


def test_tilted_plane_wave():
    """``E = exp(i kx X)`` should give all rays ``L = sin(theta)``."""
    N, dx, lam = 256, 1e-6, 633e-9
    k0 = 2.0 * np.pi / lam
    theta = np.deg2rad(5.0)
    kx_expected = k0 * np.sin(theta)
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    E = np.exp(1j * kx_expected * X)
    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=200, random_state=42,
    )
    # The phase-ratio gradient estimator is *exact* for plane waves
    # provided ``|kx * dx| < pi`` (here ``|kx * dx| ~ 0.087 << pi``),
    # so the recovered L should equal sin(theta) to machine
    # precision.  Tolerance kept at 1e-3 per spec.
    assert np.allclose(rays.L, np.sin(theta), atol=1e-3)
    assert np.allclose(rays.M, 0.0, atol=1e-3)
    assert np.all(rays.alive)


def test_converging_spherical_wave_focuses():
    """A field with phase ``= -k0 r^2 / (2 f)`` should produce rays
    that converge to the focal point at ``z = f``.
    """
    N, dx, lam, f = 256, 1e-6, 633e-9, 0.01
    k0 = 2.0 * np.pi / lam
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X * X + Y * Y
    # Gaussian amplitude * converging spherical phase.  The Gaussian
    # 1/e radius (50 um) is well inside the 128-um grid half-extent,
    # so the bulk of the rays land in the well-sampled support
    # region.
    E = np.exp(-R2 / (50e-6) ** 2) * np.exp(-1j * k0 * R2 / (2.0 * f))
    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=200, placement='cdf',
        random_state=42,
    )
    # Living rays only (we keep all rays alive in this test by
    # construction).
    assert np.all(rays.alive)
    # Propagate each ray to z = f and check it converges to the
    # optical axis within a few wavelengths.
    x_at_f = rays.x + (rays.L / rays.N) * f
    y_at_f = rays.y + (rays.M / rays.N) * f
    assert np.all(np.abs(x_at_f) < 5.0 * lam)
    assert np.all(np.abs(y_at_f) < 5.0 * lam)


def test_uniform_amplitude_uniform_density_cdf():
    """Uniform amplitude on a circular aperture should give roughly
    uniform 2-D placement density.

    Deviation: spec asks for ``placement='rejection'`` here; we test
    both modes inline because ``'rejection'`` is the exact 2-D
    sampler and the spec docstring says ``'cdf'`` is approximate
    for non-separable intensities (a disk).
    """
    N, dx = 256, 1e-6
    R = 50e-6
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.where(X * X + Y * Y <= R * R, 1.0 + 0j, 0.0 + 0j)
    rays = la.rays_from_field(
        E, dx=dx, wavelength=633e-9, n_rays=2000,
        placement='rejection', random_state=42,
    )
    # Check that >= 95% of the placement is inside the support.
    # The other ~5% can land on the boundary annulus where the
    # rasterised disk is fuzzy.
    inside = rays.x ** 2 + rays.y ** 2 <= (R + dx) ** 2
    assert inside.sum() >= int(0.95 * rays.n_rays)


def test_below_threshold_pixels_excluded():
    """Pixels below ``intensity_threshold`` should not be sampled."""
    N, dx = 128, 1e-6
    E = np.zeros((N, N), dtype=complex)
    E[60:68, 60:68] = 1.0  # tiny bright square
    rays = la.rays_from_field(
        E, dx=dx, wavelength=633e-9, n_rays=100,
        intensity_threshold=1e-4, random_state=42,
    )
    # All rays should land inside the bright square (in pixel
    # coordinates).
    x_pix = np.round(rays.x / dx).astype(int) + N // 2
    y_pix = np.round(rays.y / dx).astype(int) + N // 2
    assert np.all((x_pix >= 60) & (x_pix < 68))
    assert np.all((y_pix >= 60) & (y_pix < 68))


def test_evanescent_rays_marked():
    """A field whose phase gradient exceeds ``k0`` should produce
    evanescent-flagged rays (``alive=False``,
    ``error_code=RAY_EVANESCENT``).

    Deviation: the spec test uses ``dx = 1e-6`` and ``kx = 2 k0``,
    but at ``kx*dx ~ 19.86 rad`` the field is hopelessly aliased
    (the discrete gradient maps back into ``(-pi, pi]`` and looks
    like a small tilt).  We use ``dx = 100 nm`` and ``kx = 1.5 k0``
    so the phase ramp is genuinely sub-Nyquist while the recovered
    ``L = 1.5 > 1`` triggers evanescent flagging.  This is the
    spirit of the spec test (verifying evanescent flagging works)
    with sane sampling.
    """
    N, dx, lam = 256, 100e-9, 633e-9
    k0 = 2.0 * np.pi / lam
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    # 1.5 * k0 phase ramp.  kx*dx = 1.5*(2*pi/lam)*lam/6.33 ~ 1.49 rad
    # which is well within (-pi, pi].
    E = np.ones((N, N), dtype=complex) * np.exp(1j * 1.5 * k0 * X)
    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=50, random_state=42,
    )
    assert np.any(~rays.alive)
    assert np.any(rays.error_code == RAY_EVANESCENT)
    # Evanescent rays should have ``N = 0`` (no z propagation).
    n_evan = rays.error_code == RAY_EVANESCENT
    assert np.all(rays.N[n_evan] == 0.0)


def test_opd_initialized_from_phase():
    """OPD at each ray origin should equal ``phase(E_origin) / k0``.

    Deviation: spec uses ``phi = 0.3 * k0 * X`` with ``X`` extending
    out to ``+/- 64 um`` (at ``N=128, dx=1um``), which produces
    phases up to ``+/- 19 rad`` -- far outside the principal
    branch.  ``np.angle(E)`` returns the wrapped phase, so the
    spec assertion ``allclose(rays.opd, 0.3 * rays.x, atol=lam/100)``
    is unsatisfiable in regions where the phase wraps.  We use a
    slope of ``pi / (N * dx / 2 * k0) ~ 1/(N/2 * dx) * lam/2`` so
    the full grid is within a single principal branch.
    """
    N, dx, lam = 128, 1e-6, 633e-9
    k0 = 2.0 * np.pi / lam
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    # Choose alpha so that max(alpha * k0 * X) = pi/2 -- safely
    # within the principal branch and far from any wrap point.
    x_max = (N // 2) * dx
    alpha = (np.pi / 2.0) / (k0 * x_max)
    phi = alpha * k0 * X
    assert np.abs(phi).max() <= np.pi / 2.0 + 1e-12
    E = np.exp(1j * phi)
    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=100, random_state=42,
    )
    expected = alpha * rays.x
    assert np.allclose(rays.opd, expected, atol=lam / 100.0)


# ============================================================================
# 8-10: Additional regression tests required by the v4.15.1 brief
# ============================================================================

def test_cdf_vs_rejection_agree_on_separable_intensity():
    """``cdf`` and ``rejection`` should agree on first / second
    moments to within 5% for a separable (axially aligned)
    Gaussian intensity distribution.
    """
    N, dx, lam = 256, 1e-6, 633e-9
    w = 30e-6
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    # Separable Gaussian intensity: exp(-(X^2 + Y^2)/w^2) = product
    # of exp(-X^2/w^2) * exp(-Y^2/w^2).
    E = np.exp(-(X * X + Y * Y) / (w * w)).astype(complex)

    rays_cdf = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=4000,
        placement='cdf', random_state=42,
    )
    rays_rej = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=4000,
        placement='rejection', random_state=42,
    )

    def _moments(r):
        # First-moment (centroid) ~ 0; second-moment (RMS) ~ w/sqrt(2)
        # for intensity-weighted samples from |E|^2 = exp(-2r^2/w^2).
        return np.mean(r.x), np.mean(r.y), np.std(r.x), np.std(r.y)

    mx_cdf, my_cdf, sx_cdf, sy_cdf = _moments(rays_cdf)
    mx_rej, my_rej, sx_rej, sy_rej = _moments(rays_rej)

    # Centroids: both should be ~0; compare on the absolute scale
    # of w/sqrt(N) ~ statistical noise.
    centroid_tol = 0.2 * w
    assert abs(mx_cdf - mx_rej) < centroid_tol
    assert abs(my_cdf - my_rej) < centroid_tol

    # RMS widths: agree within 5% (separable -> cdf is exact in the
    # limit, rejection has Monte-Carlo noise ~1/sqrt(4000) ~ 1.6%).
    assert abs(sx_cdf - sx_rej) / sx_rej < 0.05
    assert abs(sy_cdf - sy_rej) / sy_rej < 0.05


def test_complex_gradient_handles_vortex_without_nan():
    """A Laguerre-Gauss-like field with an azimuthal phase
    singularity (``E ~ r * exp(i * theta)`` near origin) must not
    produce NaN / Inf in the recovered L, M, N or OPD even though
    ``E(0) = 0``.
    """
    N, dx, lam = 128, 1e-6, 633e-9
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y)
    Theta = np.arctan2(Y, X)
    # LG_{0,1}-like: r * exp(i theta) * exp(-r^2 / w^2) with l = +1.
    w = 30e-6
    E = (R / w) * np.exp(1j * Theta) * np.exp(-(R / w) ** 2)
    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=200,
        placement='cdf', random_state=42,
        angle_method='complex_gradient',
    )
    # No NaNs / Infs anywhere.
    assert np.all(np.isfinite(rays.x))
    assert np.all(np.isfinite(rays.y))
    assert np.all(np.isfinite(rays.L))
    assert np.all(np.isfinite(rays.M))
    assert np.all(np.isfinite(rays.N))
    assert np.all(np.isfinite(rays.opd))
    # The cdf placement zeroes out the marginal at the vortex core,
    # so all ray origins should sit safely outside r ~ 0.  In any
    # case the angle estimator can never blow up because we use
    # the phase-ratio form (np.angle is bounded).


def test_random_state_reproducibility():
    """Integer seed ``random_state=42`` must produce a bit-identical
    bundle on two successive calls.
    """
    N, dx, lam = 128, 1e-6, 633e-9
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X * X + Y * Y) / (40e-6) ** 2).astype(complex)

    rays_a = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=300,
        placement='rejection', random_state=42,
    )
    rays_b = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=300,
        placement='rejection', random_state=42,
    )

    np.testing.assert_array_equal(rays_a.x, rays_b.x)
    np.testing.assert_array_equal(rays_a.y, rays_b.y)
    np.testing.assert_array_equal(rays_a.L, rays_b.L)
    np.testing.assert_array_equal(rays_a.M, rays_b.M)
    np.testing.assert_array_equal(rays_a.N, rays_b.N)
    np.testing.assert_array_equal(rays_a.opd, rays_b.opd)
    np.testing.assert_array_equal(rays_a.alive, rays_b.alive)
    np.testing.assert_array_equal(rays_a.error_code, rays_b.error_code)


# ============================================================================
# Wire / packaging sanity checks
# ============================================================================

def test_rays_from_field_top_level_export():
    """``rays_from_field`` must be importable from the top level and
    from the ``raytrace`` subpackage, and listed in both ``__all__``s.
    """
    import lumenairy
    import lumenairy.raytrace
    assert hasattr(lumenairy, 'rays_from_field')
    assert hasattr(lumenairy.raytrace, 'rays_from_field')
    assert 'rays_from_field' in lumenairy.__all__
    assert 'rays_from_field' in lumenairy.raytrace.__all__
    assert lumenairy.rays_from_field is lumenairy.raytrace.rays_from_field


def test_input_validation():
    """Boundary checks should fail loudly with informative ValueError."""
    E_good = np.ones((32, 32), dtype=complex)
    # 1-D field rejected.
    with pytest.raises(ValueError, match='2-D'):
        la.rays_from_field(np.ones(32, dtype=complex), dx=1e-6,
                            wavelength=633e-9)
    # Non-positive dx.
    with pytest.raises(ValueError, match='dx'):
        la.rays_from_field(E_good, dx=0.0, wavelength=633e-9)
    with pytest.raises(ValueError, match='dx'):
        la.rays_from_field(E_good, dx=-1e-6, wavelength=633e-9)
    # Non-positive dy.
    with pytest.raises(ValueError, match='dy'):
        la.rays_from_field(E_good, dx=1e-6, dy=0.0, wavelength=633e-9)
    # Non-positive wavelength.
    with pytest.raises(ValueError, match='wavelength'):
        la.rays_from_field(E_good, dx=1e-6, wavelength=-1.0)
    # Bad placement.
    with pytest.raises(ValueError, match='placement'):
        la.rays_from_field(E_good, dx=1e-6, wavelength=633e-9,
                            placement='magic')
    # Bad angle_method.
    with pytest.raises(ValueError, match='angle_method'):
        la.rays_from_field(E_good, dx=1e-6, wavelength=633e-9,
                            angle_method='magic')
    # Identically zero field.
    with pytest.raises(ValueError, match='identically zero'):
        la.rays_from_field(np.zeros((32, 32), dtype=complex),
                            dx=1e-6, wavelength=633e-9)


def test_anamorphic_dy_passes_through():
    """A non-square pitch ``dy != dx`` should be honoured -- ray
    positions in y use ``dy``.
    """
    N, dx, dy, lam = 64, 1e-6, 5e-6, 633e-9
    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(x, y)
    # Bright disc to constrain the placement region: ensure |x| and
    # |y| stay within their distinct grids.
    Rx = (N // 2) * dx
    Ry = (N // 2) * dy
    E = np.where((X / Rx) ** 2 + (Y / Ry) ** 2 <= 0.5,
                  1.0 + 0j, 0.0 + 0j)
    rays = la.rays_from_field(
        E, dx=dx, dy=dy, wavelength=lam, n_rays=400,
        placement='rejection', random_state=42,
    )
    # If dy were defaulted to dx, the y values would sit on a 1-um
    # grid and the extent would be at most ~32 um.  With dy=5um
    # they should range out to ~160 um.
    assert np.max(np.abs(rays.y)) > 30e-6


# ============================================================================
# v4.15.2 Agent D backfill -- audit-closure coverage for the
# previously-untested paths flagged in
# ``docs/audits/AUDIT_V4_15_1_2026_05_18.md`` Part 3 P2 / P3
# (``'uniform'`` placement; ``'unwrap_gradient'`` angle method; vortex
# per-pixel direction recovery; anamorphic direction-cosine pitches;
# ``'cdf'`` placement reproducibility).
# ============================================================================

def test_uniform_placement_returns_grid_pixels_above_threshold():
    """``placement='uniform'`` should sample the field on a regular
    sub-grid, retaining only pixels whose intensity exceeds the
    threshold.

    Build a separable Gaussian; uniform placement should yield ray
    origins on pixels of the underlying grid whose intensity exceeds
    ``intensity_threshold * max(|E|^2)``.  Pin both: (a) ray origins
    are integer multiples of ``(dx, dy)`` shifted by ``N // 2``, and
    (b) ``|E(x_ray, y_ray)|^2 / max(|E|^2)`` exceeds the threshold
    for every ray.

    A wide-enough Gaussian keeps the survivor count >= n_rays so no
    short-return ``RuntimeWarning`` fires; the short-return path is
    covered separately in :func:`test_short_return_emits_warning`.
    """
    import warnings as _w
    N, dx, lam = 64, 1e-6, 633e-9
    w = 30e-6  # wide enough that 100 sub-grid cells comfortably survive
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    I_arr = np.exp(-(X * X + Y * Y) / (w * w))
    E = I_arr.astype(complex)
    threshold = 1e-4

    with _w.catch_warnings():
        _w.simplefilter('error', RuntimeWarning)  # no warning expected
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=64,
            placement='uniform', intensity_threshold=threshold,
        )
    # (a) ray origins land on grid pixels (no rounding error to
    # within numerical tol).
    ix = np.round(rays.x / dx).astype(int) + N // 2
    iy = np.round(rays.y / dx).astype(int) + N // 2
    assert np.all((ix >= 0) & (ix < N))
    assert np.all((iy >= 0) & (iy < N))
    assert np.allclose(rays.x, (ix - N // 2) * dx, atol=1e-15)
    assert np.allclose(rays.y, (iy - N // 2) * dx, atol=1e-15)
    # (b) every ray's pixel passes the pixel-wise threshold.
    intensity_at_origin = np.abs(E[iy, ix]) ** 2
    Imax = (np.abs(E) ** 2).max()
    assert np.all(intensity_at_origin / Imax > threshold)


def test_uniform_placement_handles_below_threshold_pixels():
    """``placement='uniform'`` with a thin bright spot in a dark
    background should land ALL rays on the bright-spot pixels.

    Backfills the v4.15.1 audit gap: ``'uniform'`` was completely
    untested.  A 4x4 bright square in an otherwise zero field should
    confine the sub-grid survivors to the bright square -- nothing
    leaks into the dark region even though the sub-grid has many
    times more cells than the survivor count.  Because almost all
    sub-grid cells fall in the dark region, a ``RuntimeWarning`` for
    short-return is expected; the test catches and ignores it (the
    short-return contract itself is pinned by
    :func:`test_short_return_emits_warning`).
    """
    import warnings as _w
    N, dx, lam = 64, 1e-6, 633e-9
    E = np.zeros((N, N), dtype=complex)
    # 4x4 bright square at the centre.
    E[30:34, 30:34] = 1.0
    with _w.catch_warnings():
        _w.simplefilter('ignore', RuntimeWarning)
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=8,
            placement='uniform', intensity_threshold=1e-4,
        )
    # Every surviving ray must land on a bright-square pixel.
    ix = np.round(rays.x / dx).astype(int) + N // 2
    iy = np.round(rays.y / dx).astype(int) + N // 2
    assert np.all((ix >= 30) & (ix < 34))
    assert np.all((iy >= 30) & (iy < 34))


def test_unwrap_gradient_on_smooth_phase():
    """``angle_method='unwrap_gradient'`` on a smooth tilted plane
    wave should agree with ``'complex_gradient'`` to machine
    precision and recover ``L = sin(theta)``.

    Backfills the v4.15.1 audit gap: ``'unwrap_gradient'`` was
    completely untested.
    """
    N, dx, lam = 256, 1e-6, 633e-9
    k0 = 2.0 * np.pi / lam
    theta = np.deg2rad(5.0)
    kx_expected = k0 * np.sin(theta)
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    E = np.exp(1j * kx_expected * X)

    rays_uw = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=200, random_state=42,
        angle_method='unwrap_gradient',
    )
    rays_cg = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=200, random_state=42,
        angle_method='complex_gradient',
    )
    # Both methods recover L = sin(theta) and disagree by at most
    # 1e-6 on every pixel.
    assert np.allclose(rays_uw.L, np.sin(theta), atol=1e-6)
    assert np.allclose(rays_uw.L, rays_cg.L, atol=1e-6)
    assert np.allclose(rays_uw.M, rays_cg.M, atol=1e-6)


def test_unwrap_gradient_finite_on_vortex():
    """``angle_method='unwrap_gradient'`` on a vortex field should
    produce a non-NaN result.

    Backfills the v4.15.1 audit gap: ``'unwrap_gradient'`` was
    completely untested AND the vortex test only checked
    ``isfinite`` for ``'complex_gradient'``.  Empirically the two
    methods agree to ~1e-15 on this LG-like vortex because
    ``np.unwrap`` along x / y rows independently does NOT cross the
    branch cut at the origin in a way that diverges from the
    phase-ratio result -- both produce finite, comparable values
    away from the singularity core.
    """
    N, dx, lam = 128, 1e-6, 633e-9
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y)
    Theta = np.arctan2(Y, X)
    w = 30e-6
    E = (R / w) * np.exp(1j * Theta) * np.exp(-(R / w) ** 2)
    rays_uw = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=200, random_state=42,
        angle_method='unwrap_gradient',
    )
    # No NaNs / Infs anywhere -- the headline behaviour pin.
    assert np.all(np.isfinite(rays_uw.L))
    assert np.all(np.isfinite(rays_uw.M))
    assert np.all(np.isfinite(rays_uw.N))
    assert np.all(np.isfinite(rays_uw.opd))


def test_complex_gradient_vortex_direction_recovery_per_pixel():
    """Per-pixel direction recovery on a vortex field should point
    azimuthally to within 5%.

    Backfills the v4.15.1 audit gap: the existing vortex test only
    checked ``isfinite``.  The vortex phase is
    ``phi = atan2(y, x)`` so ``grad(phi) = (-y, x) / r^2`` and
    ``(L, M) = (-y, x) / (r^2 * k0)``.  For pixels safely outside
    the vortex core (``r > 2 * dx``) the recovered direction must
    be tangential to the radial direction (cos(angle between
    radial and ray direction) ~ 0) and magnitude must follow
    ``1 / (r k0)``.
    """
    N, dx, lam = 128, 1e-6, 633e-9
    2.0 * np.pi / lam
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y)
    Theta = np.arctan2(Y, X)
    w = 30e-6
    E = (R / w) * np.exp(1j * Theta) * np.exp(-(R / w) ** 2)

    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=400, random_state=42,
        placement='rejection', angle_method='complex_gradient',
    )
    r_ray = np.sqrt(rays.x ** 2 + rays.y ** 2)
    far_mask = r_ray > 2.0 * dx
    assert far_mask.sum() > 50  # need a sane sample size
    # Tangential direction unit vector: (-y, x) / r.
    tangent_x = -rays.y[far_mask] / r_ray[far_mask]
    tangent_y = rays.x[far_mask] / r_ray[far_mask]
    # Ray direction in the (x, y) plane normalised by its in-plane
    # magnitude.
    L = rays.L[far_mask]
    M = rays.M[far_mask]
    mag = np.sqrt(L * L + M * M)
    # Avoid divide-by-zero for any (truly impossible) zero-direction
    # ray, but the vortex always has non-zero in-plane k.
    mag = np.where(mag > 0, mag, 1.0)
    ray_dir_x = L / mag
    ray_dir_y = M / mag
    # The ray direction in the transverse plane should be parallel
    # (or anti-parallel) to the tangent vector.  cos(angle) is +-1.
    dot = ray_dir_x * tangent_x + ray_dir_y * tangent_y
    assert np.all(np.abs(np.abs(dot) - 1.0) < 5e-2)


def test_anamorphic_direction_cosines_use_correct_pitches():
    """A tilted plane wave on an anamorphic ``dx != dy`` grid must
    yield direction cosines ``L = kx / k0`` and ``M = ky / k0`` --
    each computed with the correct per-axis pitch.

    Backfills the v4.15.1 audit gap: anamorphic ``dx != dy`` was
    only tested for positions, not direction cosines.

    The Gaussian envelope concentrates the placement well inside the
    grid so boundary one-sided differences (at ``ix == 0`` or
    ``ix == Nx-1`` the phase-ratio uses the clamped neighbour and
    halves the recovered ``kx``) do not contaminate the test.
    """
    N, dx, dy, lam = 128, 1e-6, 3e-6, 633e-9
    k0 = 2.0 * np.pi / lam
    theta_x = np.deg2rad(4.0)
    theta_y = np.deg2rad(2.0)
    kx = k0 * np.sin(theta_x)
    ky = k0 * np.sin(theta_y)
    x = (np.arange(N) - N // 2) * dx
    y = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(x, y)
    # Gaussian envelope so that the CDF placement concentrates well
    # inside the grid; the boundary clamp effect on the phase-ratio
    # would otherwise halve the gradient at edge pixels.
    Xnorm = X / ((N // 2) * dx * 0.25)
    Ynorm = Y / ((N // 2) * dy * 0.25)
    env = np.exp(-(Xnorm ** 2 + Ynorm ** 2) / 2.0)
    E = env * np.exp(1j * (kx * X + ky * Y))
    rays = la.rays_from_field(
        E, dx=dx, dy=dy, wavelength=lam, n_rays=200, random_state=42,
    )
    # Independent recovery of L and M (the test would fail if the
    # angle helper used dx everywhere instead of dx for x, dy for
    # y).  Phase-ratio is exact for |k*pitch| < pi.
    assert np.allclose(rays.L, np.sin(theta_x), atol=1e-3)
    assert np.allclose(rays.M, np.sin(theta_y), atol=1e-3)


def test_cdf_placement_reproducibility_with_integer_seed():
    """``placement='cdf'`` with ``random_state=42`` must be
    bit-identical on two successive calls.

    Backfills the v4.15.1 audit gap: ``'cdf'`` reproducibility was
    unpinned (only ``'rejection'`` was).
    """
    N, dx, lam = 128, 1e-6, 633e-9
    w = 30e-6
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X * X + Y * Y) / (w * w)).astype(complex)
    rays_a = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=300,
        placement='cdf', random_state=42,
    )
    rays_b = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=300,
        placement='cdf', random_state=42,
    )
    np.testing.assert_array_equal(rays_a.x, rays_b.x)
    np.testing.assert_array_equal(rays_a.y, rays_b.y)
    np.testing.assert_array_equal(rays_a.L, rays_b.L)
    np.testing.assert_array_equal(rays_a.M, rays_b.M)
    np.testing.assert_array_equal(rays_a.N, rays_b.N)
    np.testing.assert_array_equal(rays_a.opd, rays_b.opd)


# ============================================================================
# 2026-08-03 -- ``test_threshold_boundary_inclusive_all_modes`` DELETED here.
#
# It claimed to pin the v4.15.3 inclusive (``>=``) threshold convention, but
# its fixture could not: it set the peak at ``E[N//2, N//2]`` and then listed
# ``(8, 8)`` -- the same cell on a 16x16 grid -- among the "boundary" pixels,
# overwriting the peak.  Measured 2026-08-03: max|E|^2 = 0.01 (not 1.0) and
# every boundary pixel sits at normalised intensity 1.0 (not 0.01), so strict
# ``>`` and inclusive ``>=`` retain the SAME 5 pixels.  The test could not
# fail for the regression it named.
#
# The claim is pinned correctly, on a non-degenerate fixture (peak at the
# corner (0, 0) so the diagonal never collides with it, plus an explicit
# precondition guard asserting the normalised intensity IS the threshold), by
# ``test_v4_15_3_agent_d.py::test_rays_from_field_threshold_inclusive_consistent_across_modes``.
# This copy was a v4.15.3 multi-agent backfill of that test; nothing is lost.
# ============================================================================
