"""v4.15.2 Agent D -- close the docstring-vs-code drift + missing
coverage on ``lumenairy.raytrace.from_field.rays_from_field`` flagged
in ``docs/audits/AUDIT_V4_15_1_2026_05_18.md`` Part 3.

Audit findings closed here (full enumeration in the v4.15.2 / Agent D
release notes):

* **P1-NEW-D** (audit Part 3) -- ``_place_cdf`` applied
  ``intensity_threshold`` to the marginal sums rather than per-pixel,
  so a thin bright streak surrounded by sub-threshold noise survived
  the threshold incorrectly and spread the placement over the full
  y-extent.  After v4.15.2 the threshold is applied pixel-wise BEFORE
  the marginal sums are formed -- consistent with ``'rejection'`` and
  ``'uniform'``.
* **P1-NEW-H** (audit Part 3) -- ``_place_rejection`` capped at
  ``50 * n_rays`` candidate tries and ``_place_uniform`` returned only
  surviving sub-grid pixels; both could silently under-deliver
  ``n_rays``.  After v4.15.2 the main ``rays_from_field`` emits a
  ``RuntimeWarning`` when ``n_actual < n_rays`` instead of returning
  silently.
* **P2** (audit Part 3 highlights) -- the previously-untested
  ``'uniform'`` placement and ``'unwrap_gradient'`` angle method now
  have direct pins; the vortex test pins per-pixel direction
  recovery (was only ``isfinite``); the anamorphic ``dx != dy`` test
  pins direction cosines as well as positions; ``'cdf'`` is now
  pinned for reproducibility under integer ``random_state`` (was
  only pinned for ``'rejection'``).
* **P3** (audit Part 3 highlights) -- ``n_rays = 0`` and
  ``n_rays > N^2`` for ``'uniform'`` are now pinned.

Author: Andrew Traverso -- v4.15.2 / Agent D.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import RAY_OK


# ============================================================================
# D.4 -- 'uniform' placement test coverage
# ============================================================================

def test_uniform_placement_returns_grid_pixels_above_threshold():
    """``placement='uniform'`` should place rays at pixels of the
    underlying grid whose intensity exceeds the threshold.

    Pins: (a) ray origins are integer multiples of ``(dx, dy)``
    shifted by ``N // 2``, (b) ``|E(x_ray, y_ray)|^2 / max(|E|^2)``
    strictly exceeds the threshold for every ray.  A wide-enough
    Gaussian keeps the survivor count >= n_rays so no short-return
    ``RuntimeWarning`` fires.
    """
    N, dx, lam = 64, 1e-6, 633e-9
    w = 30e-6
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    I_arr = np.exp(-(X * X + Y * Y) / (w * w))
    E = I_arr.astype(complex)
    threshold = 1e-4

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=64,
            placement='uniform', intensity_threshold=threshold,
        )
    # (a) ray origins land on grid pixels exactly.
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
    """A bright 4x4 square in a dark background: all rays must land
    inside the bright square; nothing leaks into the dark region.
    """
    N, dx, lam = 64, 1e-6, 633e-9
    E = np.zeros((N, N), dtype=complex)
    E[30:34, 30:34] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=8,
            placement='uniform', intensity_threshold=1e-4,
        )
    ix = np.round(rays.x / dx).astype(int) + N // 2
    iy = np.round(rays.y / dx).astype(int) + N // 2
    assert np.all((ix >= 30) & (ix < 34))
    assert np.all((iy >= 30) & (iy < 34))


# ============================================================================
# D.5 -- 'unwrap_gradient' angle method test coverage
# ============================================================================

def test_unwrap_gradient_on_smooth_phase():
    """``'unwrap_gradient'`` on a smooth tilted plane wave should
    agree with ``'complex_gradient'`` to 1e-6 and recover
    ``L = sin(theta)``.
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
    assert np.allclose(rays_uw.L, np.sin(theta), atol=1e-6)
    assert np.allclose(rays_uw.L, rays_cg.L, atol=1e-6)
    assert np.allclose(rays_uw.M, rays_cg.M, atol=1e-6)


def test_unwrap_gradient_finite_on_vortex():
    """``'unwrap_gradient'`` on a vortex field must produce a
    non-NaN finite result.

    Measured behaviour on the LG-like ``E = (r/w)*exp(i*theta)*
    exp(-r^2/w^2)`` vortex: ``np.unwrap`` along x / y rows
    independently does NOT diverge from the phase-ratio result --
    both methods agree to ~1e-15 away from the singularity core.
    Neither method emits a warning about the singularity (the
    docstring already calls ``'unwrap_gradient'`` fragile; this test
    only pins finite-ness, the headline behavioural contract).
    """
    N, dx, lam = 128, 1e-6, 633e-9
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y)
    Theta = np.arctan2(Y, X)
    w = 30e-6
    E = (R / w) * np.exp(1j * Theta) * np.exp(-(R / w) ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)  # no warn expected
        rays_uw = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=200, random_state=42,
            angle_method='unwrap_gradient',
        )
    assert np.all(np.isfinite(rays_uw.L))
    assert np.all(np.isfinite(rays_uw.M))
    assert np.all(np.isfinite(rays_uw.N))
    assert np.all(np.isfinite(rays_uw.opd))


# ============================================================================
# D.6 -- Vortex direction-recovery pin (per-pixel)
# ============================================================================

def test_complex_gradient_vortex_direction_recovery_per_pixel():
    """Per-pixel direction recovery on a vortex field should point
    azimuthally (perpendicular to radial) to within 5e-2 for rays
    placed at ``r > 2 * dx``.

    Vortex phase ``phi = atan2(y, x)`` so ``grad(phi) = (-y/r^2,
    x/r^2)`` and the ray direction (L, M) is tangential to the
    radial direction.
    """
    N, dx, lam = 128, 1e-6, 633e-9
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
    assert far_mask.sum() > 50
    tangent_x = -rays.y[far_mask] / r_ray[far_mask]
    tangent_y = rays.x[far_mask] / r_ray[far_mask]
    L = rays.L[far_mask]
    M = rays.M[far_mask]
    mag = np.sqrt(L * L + M * M)
    mag = np.where(mag > 0, mag, 1.0)
    ray_dir_x = L / mag
    ray_dir_y = M / mag
    dot = ray_dir_x * tangent_x + ray_dir_y * tangent_y
    # Tangent vs ray direction: |cos(angle)| ~ 1 if perfectly
    # azimuthal.  5e-2 tolerance accounts for finite-difference
    # error and Gaussian-envelope-induced radial phase contribution.
    assert np.all(np.abs(np.abs(dot) - 1.0) < 5e-2)


# ============================================================================
# D.7 -- Anamorphic dx != dy direction-cosines pin
# ============================================================================

def test_anamorphic_direction_cosines_use_correct_pitches():
    """Anamorphic tilted plane wave: ``L = kx/k0`` and ``M = ky/k0``
    independently, confirming the direction-cosine computation uses
    the correct per-axis pitch ``(dx, dy)``.
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
    # Gaussian envelope to confine placement away from the boundary
    # one-sided difference effect.
    Xnorm = X / ((N // 2) * dx * 0.25)
    Ynorm = Y / ((N // 2) * dy * 0.25)
    env = np.exp(-(Xnorm ** 2 + Ynorm ** 2) / 2.0)
    E = env * np.exp(1j * (kx * X + ky * Y))
    rays = la.rays_from_field(
        E, dx=dx, dy=dy, wavelength=lam, n_rays=200, random_state=42,
    )
    assert np.allclose(rays.L, np.sin(theta_x), atol=1e-3)
    assert np.allclose(rays.M, np.sin(theta_y), atol=1e-3)


# ============================================================================
# D.8 -- 'cdf' random_state reproducibility
# ============================================================================

def test_cdf_placement_reproducibility_with_integer_seed():
    """``placement='cdf'`` with ``random_state=42`` is bit-identical
    on two successive calls.
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
    np.testing.assert_array_equal(rays_a.alive, rays_b.alive)


# ============================================================================
# D.9 -- 'uniform' edge cases (n_rays = 0; n_rays > N^2)
# ============================================================================

def test_uniform_placement_n_rays_zero():
    """``n_rays = 0`` returns an empty RayBundle cleanly (no
    exception, no spurious dimensions).
    """
    N, dx, lam = 32, 1e-6, 633e-9
    E = np.ones((N, N), dtype=complex)
    rays = la.rays_from_field(
        E, dx=dx, wavelength=lam, n_rays=0,
        placement='uniform',
    )
    assert rays.n_rays == 0
    assert rays.x.shape == (0,)
    assert rays.y.shape == (0,)
    assert rays.L.shape == (0,)
    assert rays.M.shape == (0,)
    assert rays.N.shape == (0,)
    assert rays.opd.shape == (0,)
    assert rays.alive.shape == (0,)
    assert rays.error_code.shape == (0,)
    assert rays.wavelength == lam


def test_uniform_placement_n_rays_zero_works_for_cdf_and_rejection():
    """``n_rays = 0`` is honoured by every placement mode."""
    N, dx, lam = 32, 1e-6, 633e-9
    E = np.ones((N, N), dtype=complex)
    for mode in ('cdf', 'rejection', 'uniform'):
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=0,
            placement=mode, random_state=42,
        )
        assert rays.n_rays == 0, f'{mode}: n_rays mismatch'
        assert rays.x.shape == (0,), f'{mode}: x shape'


def test_uniform_placement_n_rays_exceeds_grid_size():
    """``n_rays = 2 * N^2`` on an ``N x N`` grid returns at most
    ``N^2`` rays after threshold and pixel deduplication; the
    short-return ``RuntimeWarning`` fires.
    """
    N, dx, lam = 16, 1e-6, 633e-9
    E = np.ones((N, N), dtype=complex)
    target = 2 * N * N
    with pytest.warns(RuntimeWarning, match='returned'):
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=target,
            placement='uniform',
        )
    # At most one ray per pixel after dedup.
    assert rays.n_rays <= N * N
    # Confirm: positions are unique (no duplicate pixels).
    ix = np.round(rays.x / dx).astype(int) + N // 2
    iy = np.round(rays.y / dx).astype(int) + N // 2
    composite = iy.astype(np.int64) * N + ix.astype(np.int64)
    assert len(np.unique(composite)) == len(composite)


# ============================================================================
# Regression tests (P1-NEW-D and P1-NEW-H pins)
# ============================================================================

def test_cdf_placement_pixel_threshold_excludes_thin_streak():
    """Pin: ``'cdf'`` applies ``intensity_threshold`` pixel-wise, not
    against the marginal sum.

    Build a field where:
    * a single bright column at ``ix = N // 2`` has 5 bright pixels at
      rows ``iy in [Ny//2 - 2, Ny//2 + 3)`` (short streak)
    * everything else is sub-threshold noise

    With pixel-wise thresholding (the new behaviour), placement is
    confined to the 5 bright pixels.  With the OLD marginal-sum
    behaviour, the y-marginal of the noise rows is comparable to the
    y-marginal of the streak rows so rays would scatter across the
    full y-extent.

    This is the audit's P1-NEW-D regression pin.
    """
    N, dx, lam = 64, 1e-6, 633e-9
    cx, cy = N // 2, N // 2
    # Sub-threshold-noise floor: per-pixel intensity 1e-3 << threshold
    # 1e-2 so each pixel individually fails the pixel-wise test.
    E = np.full((N, N), np.sqrt(1e-3), dtype=complex)
    # Five bright pixels at (col=cx, row=cy-2..cy+2): amplitude 1.0
    # so intensity 1.0 -- safely above threshold.
    E[cy - 2:cy + 3, cx] = 1.0
    threshold = 1e-2

    # Run cdf placement.
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)  # don't short
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=200,
            placement='cdf', intensity_threshold=threshold,
            random_state=42,
        )

    ix = np.round(rays.x / dx).astype(int) + N // 2
    iy = np.round(rays.y / dx).astype(int) + N // 2
    # All rays must land on the bright column.
    assert np.all(ix == cx), (
        f'Expected all rays at column {cx}; got columns '
        f'{np.unique(ix)}'
    )
    # All rays must land in the short y-streak.  This is the
    # pixel-wise-threshold pin.  With the old marginal-threshold
    # behaviour, y would scatter across full [0, N) since every
    # row's Iy is dominated by the noise sum 1e-3 * N ~ 0.064 (vs
    # 1e-3 * (N-1) + 1.0 ~ 1.063 at streak rows) -- so
    # threshold * max(Iy) ~ 1e-2 * 1.063 ~ 0.0106 and noise rows
    # have Iy = 0.064 > 0.0106 -> they incorrectly pass.
    assert np.all((iy >= cy - 2) & (iy <= cy + 2)), (
        f'Expected all rays at rows [{cy-2}, {cy+2}]; '
        f'got rows {np.unique(iy)}'
    )


def test_short_return_emits_warning():
    """Pin: a too-aggressive ``intensity_threshold`` (or other budget
    exhaustion) causes ``rays_from_field`` to emit a
    ``RuntimeWarning`` indicating ``n_actual`` vs requested
    ``n_rays`` -- it does NOT raise.

    This is the audit's P1-NEW-H regression pin.
    """
    N, dx, lam = 16, 1e-6, 633e-9
    E = np.zeros((N, N), dtype=complex)
    # Single isolated bright pixel: only ONE pixel can survive
    # placement.  Asking for 100 rays under 'uniform' forces a
    # short-return.
    E[8, 8] = 1.0
    with pytest.warns(RuntimeWarning, match='returned'):
        rays = la.rays_from_field(
            E, dx=dx, wavelength=lam, n_rays=100,
            placement='uniform', intensity_threshold=1e-4,
        )
    # The bundle still has fewer rays than requested but is valid
    # (i.e. did NOT raise).
    assert rays.n_rays < 100
    assert rays.n_rays >= 1
    # All rays are valid (RAY_OK) -- short-return doesn't poison
    # the survivors.
    assert np.all(rays.error_code == RAY_OK)
    assert np.all(rays.alive)
