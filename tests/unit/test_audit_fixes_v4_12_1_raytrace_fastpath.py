"""Pinning tests for the v4.12.1 Track C raytrace pure-spherical
Newton-skip fast-path (``lumenairy/raytrace/core.py:_intersect_surface``).

Background
==========

The legacy ``_intersect_surface`` routes every curved-surface
intersection through a 10-iteration Newton refinement of an analytical
sphere-quadratic initial guess.  For pure-spherical surfaces (conic=0,
no aspheric / biconic / freeform extension) the initial guess **is**
the exact intersection (modulo LSB rounding), so the Newton loop costs
~2 wasted iterations per surface.  On a 1k-ray doublet trace this is
the bulk of the per-surface cost.

v4.12.0 attempted both halves of the optimisation:

* (a) **Skip Newton** for pure-spherical surfaces (the perf win).
* (b) **Switch the surface normal** to the analytic
  ``(x/R, y/R, (z - R)/R)`` form (matching ``jax_trace.py``).

The validation suite caught a 1.17e-3 cross-backend rel error in
``validation/propagators/test_asymptotic.py::aberration_tensor_lg00_jax
matches NumPy (0,0) element``.  The drift came from (b): switching the
spherical normal changed the LSB-level rounding path, which
compounded ~50x through the Maslov asymptotic field summation into a
1e-3 phase drift -- enough to cross the 1e-3 threshold of the
cross-backend test.

v4.12.1 (this fix) ships **only (a)** -- the Newton-skip fast path --
and keeps the surface normal on the legacy
``_surface_sag_derivatives_xy``-derived path so the LSB rounding
behaviour through :func:`_refract` / :func:`_reflect` is bit-identical
to v4.11.2.  The skipped Newton step would have made an ~1e-17 LSB
correction at most; per-ray drift after the skip is ~5e-17 m (the OPD
field is the worst-case ~5e-17).

The tests below pin:

* **Bit-near-exact parity** vs the legacy pre-fix output (per-ray
  (x, y, z, L, M, N, opd) within 1e-15 absolute).
* **Cross-backend asymptotic correctness** -- the
  ``aberration_tensor_lg00_jax matches NumPy (0,0)`` test from the
  asymptotic validation must still pass with rel_err comfortably below
  the 1e-3 cross-backend tolerance (we assert < 1e-3 to match the
  validation gate; baseline is 4.83e-4, post-fix 4.53e-4).
* **Speed** -- the 1k-ray doublet trace must be at least 1.3x faster
  than the pre-fix legacy Newton-loop path.

If any of these regress (especially the asymptotic cross-backend
check) we have re-introduced the v4.12.0 LSB-rounding drift that
forced the revert.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.raytrace.core import (
    Surface, _intersect_surface, _surface_sag_xy,
    _surface_sag_derivatives_xy,
)
from lumenairy.raytrace import RayBundle


# ============================================================================
# Helper builders
# ============================================================================

def _make_doublet():
    """100 mm Thorlabs AC254-100-C achromat (three pure-spherical
    surfaces)."""
    pres = lm.thorlabs_lens('AC254-100-C')
    surfaces = lm.surfaces_from_prescription(pres)
    return pres, surfaces


def _make_1k_rays(wavelength=1310e-9):
    """1024-ray (32x32) square grid filling a 12.5 mm aperture."""
    return lm.make_grid(
        semi_aperture=6.25e-3, n_across=32, wavelength=wavelength
    )


def _legacy_newton_intersect(rays, surface, n_medium=1.0):
    """Reference implementation of the pre-v4.12.1 Newton intersect.

    Mirrors the legacy ``_intersect_surface`` body sans the fast-path
    branch so we can compare against it bit-for-bit.  Kept as a
    private helper -- not exported.

    Returns the post-intersect (x, y, z, opd, alive) without mutating
    ``rays``.
    """
    from lumenairy.raytrace.core import RAY_OK, RAY_MISSED_SURFACE

    rays = RayBundle(
        x=rays.x.copy(), y=rays.y.copy(), z=rays.z.copy(),
        L=rays.L.copy(), M=rays.M.copy(), N=rays.N.copy(),
        opd=rays.opd.copy(),
        alive=rays.alive.copy(),
        error_code=(rays.error_code.copy()
                    if rays.error_code is not None else None),
        wavelength=rays.wavelength,
    )
    R = surface.radius
    asph = surface.aspheric_coeffs
    if np.isinf(R) and not asph:
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(rays.alive & (np.abs(rays.N) > 1e-30),
                         -rays.z / rays.N, 0.0)
    else:
        t = np.zeros(rays.n_rays)
        if not np.isinf(R):
            x0, y0, z0 = rays.x, rays.y, rays.z
            Ld, Md, Nd = rays.L, rays.M, rays.N
            dx, dy, dz = x0, y0, z0 - R
            b = 2.0 * (Ld * dx + Md * dy + Nd * dz)
            c = dx ** 2 + dy ** 2 + dz ** 2 - R ** 2
            disc = b ** 2 - 4 * c
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            t1 = (-b - sqrt_disc) / 2.0
            t2 = (-b + sqrt_disc) / 2.0
            t = t1 if R > 0 else t2
            missed_init = (disc < 0) & rays.alive
            t = np.where(disc > 0, t, 0.0)
        else:
            missed_init = np.zeros(rays.n_rays, dtype=bool)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(np.abs(rays.N) > 1e-30, -rays.z / rays.N, 0.0)

        converged = np.zeros(rays.n_rays, dtype=bool)
        for _ in range(10):
            xi = rays.x + rays.L * t
            yi = rays.y + rays.M * t
            zi = rays.z + rays.N * t
            sag_i = _surface_sag_xy(xi, yi, surface)
            F = zi - sag_i
            dz_dx, dz_dy = _surface_sag_derivatives_xy(xi, yi, surface)
            dF_dt = rays.N - dz_dx * rays.L - dz_dy * rays.M
            stuck = np.abs(dF_dt) <= 1e-30
            dt = np.where(stuck, 0.0, F / np.where(stuck, 1.0, dF_dt))
            t = t - dt
            converged = (np.abs(dt) < 1e-15) & (~stuck | (np.abs(F) < 1e-12))
            if converged.all():
                break

        missed_final = (~converged | missed_init) & rays.alive
        if missed_final.any():
            rays.alive = rays.alive & ~missed_final
            if rays.error_code is not None:
                first_failure = missed_final & (rays.error_code == RAY_OK)
                rays.error_code = np.where(
                    first_failure, RAY_MISSED_SURFACE, rays.error_code
                )

    t = np.where(rays.alive, t, 0.0)
    rays.x = rays.x + rays.L * t
    rays.y = rays.y + rays.M * t
    rays.z = rays.z + rays.N * t
    rays.opd = rays.opd + n_medium * t
    if np.isfinite(surface.semi_diameter):
        h_sq = rays.x ** 2 + rays.y ** 2
        clipped = (h_sq > surface.semi_diameter ** 2) & rays.alive
        if clipped.any():
            from lumenairy.raytrace.core import RAY_APERTURE
            rays.alive = rays.alive & ~clipped
            if rays.error_code is not None:
                rays.error_code = np.where(
                    clipped, RAY_APERTURE, rays.error_code
                )
    return rays


# ============================================================================
# Per-ray bit-near-exact pin against the legacy Newton path
# ============================================================================

class TestPerSurfaceBitNearExact:
    """Single-surface intersect (and full trace) within 1e-15 of legacy.

    The fast-path skips the legacy Newton refinement loop.  On a pure
    sphere the initial-guess quadratic IS the exact root, so the
    skipped Newton iterations would have made at most one
    ``F / dF_dt ~ 1e-17`` correction.  Drift between the two paths
    is therefore at the LSB scale.
    """

    @pytest.mark.parametrize('R', [62.8e-3, -46.5e-3, -184.5e-3,
                                     0.5, -0.1])
    def test_single_pure_sphere_matches_newton(self, R):
        """Single-surface intersect matches legacy Newton to 1e-15."""
        surf = Surface(radius=R, conic=0.0, glass_before='air',
                       glass_after='N-BK7', semi_diameter=25e-3)

        # Random rays approaching the surface (vary direction & origin)
        rng = np.random.default_rng(seed=12345)
        n = 256
        x = rng.uniform(-10e-3, 10e-3, n)
        y = rng.uniform(-10e-3, 10e-3, n)
        z = np.full(n, -5e-3)
        # Direction cosines (mostly forward, small NA)
        L = rng.uniform(-0.1, 0.1, n)
        M = rng.uniform(-0.1, 0.1, n)
        N = np.sqrt(1.0 - L ** 2 - M ** 2)
        opd = np.zeros(n)
        alive = np.ones(n, dtype=bool)
        rays_fast = RayBundle(x=x.copy(), y=y.copy(), z=z.copy(),
                              L=L.copy(), M=M.copy(), N=N.copy(),
                              opd=opd.copy(), alive=alive.copy(),
                              error_code=np.zeros(n, dtype=np.uint8),
                              wavelength=1310e-9)
        rays_ref = _legacy_newton_intersect(rays_fast, surf, n_medium=1.0)
        _intersect_surface(rays_fast, surf, n_medium=1.0)
        for field in ['x', 'y', 'z', 'L', 'M', 'N', 'opd']:
            cur = getattr(rays_fast, field)
            ref = getattr(rays_ref, field)
            max_abs = np.abs(cur - ref).max()
            assert max_abs < 1e-15, (
                f'pure-sphere R={R} field={field!r}: max_abs={max_abs:.3e}'
                f' exceeds 1e-15 -- the fast-path is drifting beyond LSB.')

    def test_full_doublet_trace_matches_newton(self):
        """1k-ray doublet trace matches a re-traced Newton reference
        (per-ray (x, y, z, L, M, N, opd) within 1e-15 absolute).

        We can't easily revert the fast path inside an already-imported
        :func:`_intersect_surface`, so we compare the current trace to
        a hand-rebuilt trace via :func:`_legacy_newton_intersect` for
        the three doublet surfaces -- this exercises the same pathway
        the legacy Newton loop took.
        """
        wavelength = 1310e-9
        pres, surfaces = _make_doublet()
        rays_in = _make_1k_rays(wavelength=wavelength)

        # Run the official trace -- uses fast path on the three spheres.
        res = lm.trace(rays_in, surfaces, wavelength)
        img_fast = res.image_rays

        # Hand-rebuild a Newton-only doublet trace.  This mimics
        # what the legacy code did surface-by-surface (transfer +
        # intersect + refract).
        from lumenairy.raytrace.core import (
            _transfer, _refract, _reflect,
        )
        rays = RayBundle(
            x=rays_in.x.copy(), y=rays_in.y.copy(), z=rays_in.z.copy(),
            L=rays_in.L.copy(), M=rays_in.M.copy(), N=rays_in.N.copy(),
            opd=rays_in.opd.copy(),
            alive=rays_in.alive.copy(),
            error_code=(rays_in.error_code.copy()
                        if rays_in.error_code is not None else None),
            wavelength=rays_in.wavelength,
        )

        # Mimic the trace loop manually but swap _intersect_surface
        # for _legacy_newton_intersect.  Use the same n1/n2 lookup
        # logic.
        from lumenairy.glass import get_glass_index
        # The user-facing `trace` builds nL of each surface from
        # glass_before/glass_after; we replicate that here.
        for surf in surfaces:
            n_before = get_glass_index(surf.glass_before, wavelength)
            n_after = get_glass_index(surf.glass_after, wavelength)
            # Replace _intersect_surface with the legacy Newton version
            rays_after_intersect = _legacy_newton_intersect(
                rays, surf, n_medium=n_before)
            # Mutate `rays` to match
            rays.x[:] = rays_after_intersect.x
            rays.y[:] = rays_after_intersect.y
            rays.z[:] = rays_after_intersect.z
            rays.opd[:] = rays_after_intersect.opd
            rays.alive[:] = rays_after_intersect.alive
            if rays.error_code is not None and (
                    rays_after_intersect.error_code is not None):
                rays.error_code[:] = rays_after_intersect.error_code
            if surf.is_mirror:
                _reflect(rays, surf)
            else:
                _refract(rays, surf, n_before, n_after)
            _transfer(rays, surf.thickness, n_after)

        img_ref = rays

        # Per-ray comparison
        for field in ['x', 'y', 'z', 'L', 'M', 'N', 'opd']:
            cur = getattr(img_fast, field)
            ref = getattr(img_ref, field)
            max_abs = np.abs(cur - ref).max()
            assert max_abs < 1e-15, (
                f'doublet trace field={field!r}: max_abs={max_abs:.3e}'
                f' exceeds 1e-15.  Pre-fix Newton output vs post-fix '
                f'fast-path should agree to LSB.')


# ============================================================================
# Fast-path correctness vs analytical sphere equation
# ============================================================================

class TestFastPathOnPureSphere:
    """The fast path must produce points satisfying the sphere equation
    ``x^2 + y^2 + (z - R)^2 = R^2`` to within rounding noise."""

    @pytest.mark.parametrize('R', [62.8e-3, -46.5e-3, 0.5, -0.1])
    def test_intersection_satisfies_sphere_equation(self, R):
        surf = Surface(radius=R, conic=0.0, glass_before='air',
                       glass_after='air', semi_diameter=20e-3)

        rng = np.random.default_rng(seed=42)
        n = 200
        x = rng.uniform(-5e-3, 5e-3, n)
        y = rng.uniform(-5e-3, 5e-3, n)
        z = np.full(n, -2e-3)
        L = rng.uniform(-0.05, 0.05, n)
        M = rng.uniform(-0.05, 0.05, n)
        N = np.sqrt(1.0 - L ** 2 - M ** 2)
        rays = RayBundle(
            x=x.copy(), y=y.copy(), z=z.copy(),
            L=L.copy(), M=M.copy(), N=N.copy(),
            opd=np.zeros(n), alive=np.ones(n, dtype=bool),
            error_code=np.zeros(n, dtype=np.uint8),
            wavelength=1310e-9,
        )
        _intersect_surface(rays, surf, n_medium=1.0)

        residual = (rays.x ** 2 + rays.y ** 2
                    + (rays.z - R) ** 2 - R ** 2)
        max_residual = np.abs(residual).max()
        # 1e-17 absolute is good; allow 1e-15 for the L*t accumulation
        # round-off at large t.
        assert max_residual < 1e-15, (
            f'pure-sphere intersection residual max = {max_residual:.3e}, '
            f'expected at LSB level (< 1e-15).')


# ============================================================================
# Negative test: fast-path NOT taken on non-spherical surfaces
# ============================================================================

class TestFastPathGuards:
    """The fast path must only kick in on surfaces whose intersection
    *is* the spherical quadratic.  Aspherics, biconics, freeforms,
    and conic >= 0 != 0 must continue to use Newton."""

    def test_conic_aspheric_routes_to_newton(self):
        """Conic surface (k != 0) traces correctly through Newton."""
        surf = Surface(radius=0.1, conic=-1.0, glass_before='air',
                       glass_after='air', semi_diameter=20e-3)
        rays = RayBundle(
            x=np.array([1e-3, -2e-3]),
            y=np.array([0.0, 1e-3]),
            z=np.array([-1e-3, -1e-3]),
            L=np.array([0.0, 0.0]),
            M=np.array([0.0, 0.0]),
            N=np.array([1.0, 1.0]),
            opd=np.zeros(2), alive=np.ones(2, dtype=bool),
            error_code=np.zeros(2, dtype=np.uint8),
            wavelength=1310e-9,
        )
        _intersect_surface(rays, surf, n_medium=1.0)
        # Parabolic sag: z = h^2 / (2R) at conic = -1
        h_sq = rays.x ** 2 + rays.y ** 2
        expected_z = h_sq / (2.0 * 0.1)
        assert np.allclose(rays.z, expected_z, atol=1e-12, rtol=1e-10)

    def test_biconic_routes_to_newton(self):
        """Biconic surface (radius_y set) traces correctly."""
        surf = Surface(radius=0.1, conic=0.0, radius_y=0.2, conic_y=0.0,
                       glass_before='air', glass_after='air',
                       semi_diameter=20e-3)
        rays = RayBundle(
            x=np.array([1e-3]),
            y=np.array([2e-3]),
            z=np.array([-1e-3]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            opd=np.zeros(1), alive=np.ones(1, dtype=bool),
            error_code=np.zeros(1, dtype=np.uint8),
            wavelength=1310e-9,
        )
        _intersect_surface(rays, surf, n_medium=1.0)
        # The biconic sag for an on-axis-incidence ray is the surface
        # height at (x, y) computed via :func:`_surface_sag_xy`.
        # We just check the ray ends up on the surface, i.e.
        # residual zi - sag(xi, yi) is small.
        sag = _surface_sag_xy(rays.x, rays.y, surf)
        assert np.abs(rays.z - sag).max() < 1e-12

    def test_flat_aspheric_routes_to_newton(self):
        """Flat surface with aspheric coeffs (radius == inf, asph) uses
        the existing flat-with-Newton branch, not the fast path."""
        surf = Surface(radius=np.inf, conic=0.0,
                       aspheric_coeffs={4: 1e6},
                       glass_before='air', glass_after='air',
                       semi_diameter=20e-3)
        rays = RayBundle(
            x=np.array([1e-3]),
            y=np.array([0.0]),
            z=np.array([0.0]),
            L=np.array([0.0]),
            M=np.array([0.0]),
            N=np.array([1.0]),
            opd=np.zeros(1), alive=np.ones(1, dtype=bool),
            error_code=np.zeros(1, dtype=np.uint8),
            wavelength=1310e-9,
        )
        _intersect_surface(rays, surf, n_medium=1.0)
        # Expected sag at h = 1e-3: 1e6 * (1e-3)^4 = 1e-6 m
        assert abs(rays.z[0] - 1e-6) < 1e-12


# ============================================================================
# THE CRITICAL TEST: asymptotic cross-backend rel_err must not regress
# ============================================================================

class TestAsymptoticCrossBackend:
    """Mirror of ``validation/propagators/test_asymptotic.py::
    t_aberration_tensor_lg00_jax_matches_numpy``.

    This is the test that revealed the v4.12.0 LSB drift -- the post-
    fix rel_err must stay comfortably below the 1e-3 cross-backend
    threshold.  Baseline (pre-v4.12.1) is ~4.8e-4, well below 1e-3.
    v4.12.0 ran at ~1.17e-3 (failing).  v4.12.1 conservative fast-path
    is ~4.5e-4.
    """

    def test_aberration_tensor_lg00_jax_matches_numpy_0_0(self):
        pytest.importorskip('jax', reason='JAX not installed')
        import lumenairy as la
        from lumenairy.propagators.asymptotic import (
            fit_canonical_polynomials, solve_envelope_stationary,
            aberration_tensor, aberration_tensor_lg00_jax,
        )

        # 100 mm BK7 singlet (same prescription as the validation test)
        pres = la.make_singlet(
            51.5e-3, np.inf, 4.1e-3, 'N-BK7', aperture=12.0e-3,
        )
        pres['object_distance'] = 200e-3
        fit = fit_canonical_polynomials(
            pres, wavelength=1.31e-6,
            source_box_half=20e-6, pupil_box_half=0.02,
            n_field=8, n_pupil=8, poly_order=6,
        )
        s2_image = (fit.s2x_centre, fit.s2y_centre)
        v_star, _, _ = solve_envelope_stationary(
            fit, s2_image, (0.0, 0.0),
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
        )
        res = aberration_tensor(
            fit, s2_image=s2_image,
            source_point=(0.0, 0.0),
            source_modes=[(0, 0)], pupil_modes=[(0, 0)],
            output_modes=[(0, 0)],
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
        )
        L_np = complex(res.L[0, 0])
        L_jax = aberration_tensor_lg00_jax(
            fit, s2_image, v_star,
            source_point=(0.0, 0.0),
            w_s=20e-6, w_p=0.02, w_o=res.w_o,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
        )
        L_jax_c = complex(L_jax)
        rel = abs(L_np - L_jax_c) / max(abs(L_np), 1e-30)

        # Validation test uses 1e-3 threshold; we assert the same.
        # The v4.12.0 broken implementation hit 1.17e-3 (failing).
        assert rel < 1e-3, (
            f'aberration_tensor_lg00_jax (0,0) cross-backend rel_err '
            f'= {rel:.3e} (must be < 1e-3).  L_np={L_np:.4e}, '
            f'L_jax={L_jax_c:.4e}.  '
            f'v4.12.0 regression value was 1.17e-3; v4.11.2 baseline '
            f'4.83e-4; v4.12.1 expected ~4.5e-4.')


# ============================================================================
# Speed pin: doublet trace at least 1.3x faster than legacy
# ============================================================================

class TestFastPathSpeedup:
    """The post-fix doublet trace must be at least 1.3x faster than
    the legacy 10-iter Newton path on a 1k-ray bundle.

    The reference is built by re-tracing the doublet via the legacy
    Newton intersect helper used in :class:`TestPerSurfaceBitNearExact`.

    Note: this is a perf test -- under heavy CPU contention (e.g. a
    busy CI host) the timings can jitter.  We use ``min(times)`` over
    5 batches to filter out worst-case stalls, and the threshold
    (1.3x) is conservative against the typical 1.5x measurement so
    contention doesn't false-fail us.
    """

    def _time_fast_path_trace(self, surfaces, rays_in, wavelength,
                                  n_iter=50):
        """Time the production :func:`lm.trace` (which uses the fast
        path on pure spheres)."""
        # Warmup
        for _ in range(5):
            lm.trace(rays_in, surfaces, wavelength)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                lm.trace(rays_in, surfaces, wavelength)
            times.append((time.perf_counter() - t0) / n_iter)
        return min(times)

    def _time_legacy_newton_trace(self, surfaces, rays_in, wavelength,
                                    n_iter=50):
        """Time a re-traced doublet using the legacy Newton intersect
        helper (same code path the legacy `_intersect_surface` took)."""
        from lumenairy.raytrace.core import (
            _transfer, _refract, _reflect,
        )
        from lumenairy.glass import get_glass_index

        def trace_once():
            rays = RayBundle(
                x=rays_in.x.copy(), y=rays_in.y.copy(),
                z=rays_in.z.copy(),
                L=rays_in.L.copy(), M=rays_in.M.copy(),
                N=rays_in.N.copy(),
                opd=rays_in.opd.copy(),
                alive=rays_in.alive.copy(),
                error_code=(rays_in.error_code.copy()
                            if rays_in.error_code is not None else None),
                wavelength=rays_in.wavelength,
            )
            for surf in surfaces:
                n_before = get_glass_index(surf.glass_before, wavelength)
                n_after = get_glass_index(surf.glass_after, wavelength)
                # Run the legacy Newton intersect (returns a new rays).
                rays_after = _legacy_newton_intersect(
                    rays, surf, n_medium=n_before)
                rays.x[:] = rays_after.x
                rays.y[:] = rays_after.y
                rays.z[:] = rays_after.z
                rays.opd[:] = rays_after.opd
                rays.alive[:] = rays_after.alive
                if (rays.error_code is not None
                        and rays_after.error_code is not None):
                    rays.error_code[:] = rays_after.error_code
                if surf.is_mirror:
                    _reflect(rays, surf)
                else:
                    _refract(rays, surf, n_before, n_after)
                _transfer(rays, surf.thickness, n_after)
            return rays

        for _ in range(5):
            trace_once()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                trace_once()
            times.append((time.perf_counter() - t0) / n_iter)
        return min(times)

    def test_doublet_trace_at_least_1_3x_faster(self):
        """1k-ray doublet trace: fast path / legacy >= 1.3x."""
        wavelength = 1310e-9
        pres, surfaces = _make_doublet()
        rays_in = _make_1k_rays(wavelength=wavelength)
        fast = self._time_fast_path_trace(surfaces, rays_in, wavelength)
        legacy = self._time_legacy_newton_trace(
            surfaces, rays_in, wavelength)
        speedup = legacy / fast
        # Surface the timing for diagnostic logging.
        print(f'\n  fast={fast*1e6:.1f} us, legacy={legacy*1e6:.1f} us, '
              f'speedup={speedup:.2f}x')
        # Allow timing jitter -- the conservative target is 1.3x.
        # In a clean run the speedup measures ~1.45-1.5x.
        assert speedup >= 1.3, (
            f'Post-fix doublet trace = {fast*1e6:.1f} us, '
            f'legacy Newton = {legacy*1e6:.1f} us, '
            f'speedup = {speedup:.2f}x (target >= 1.3x).  '
            f'If speedup regressed, the fast path may have been '
            f'short-circuited or the legacy Newton may have become '
            f'cheaper -- investigate.')
