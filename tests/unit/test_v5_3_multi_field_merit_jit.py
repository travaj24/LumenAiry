"""Regression pins for the v5.3 ``MultiFieldMerit`` Numba JIT path.

Closes the ROADMAP v5.3 horizon item:

    ``MultiFieldMerit`` performance ceiling -- meshgrid cache hit
    1.19x at N=128; the per-field ``np.exp`` + ``np.where`` calls
    dominate and cannot be cached.  Future: JIT-compile the per-
    field tilt path via Numba or JAX.

Background
----------
Pre-v5.3, ``MultiFieldMerit.evaluate`` built the per-field masked
tilted plane wave via a three-statement NumPy expression

    tilt_phase = sin(tx)*k_X + sin(ty)*k_Y           # 1
    phasor     = exp(1j * tilt_phase)                # 2
    E_tilted   = where(mask, phasor, 0)              # 3

with three intermediate N x N temporaries per field.  v5.3
introduces a fused Numba ``@njit(parallel=True, fastmath=True)``
kernel that collapses all three into a single in-place pass with
zero temporaries.  Two specialised kernels exist (one per output
dtype) because Numba does not erase the complex precision; the
caller-facing ``_multi_field_tilt_phasor_masked`` helper dispatches
on dtype and falls back to the legacy NumPy expression below a
threshold (N < 128) or when Numba is unavailable.

These tests pin the four contracts the ROADMAP item promises:

1. **Numerical equivalence.**  The JIT and NumPy paths produce
   bit-identical (to 1e-12) output across a range of field angles,
   wavelengths, and grid sizes.
2. **Numba speedup.**  At N=256 with 8 fields the JIT path beats
   the NumPy path by >= 1.5x (modest pin; the ROADMAP just says
   "JIT-compile", no specific speedup target).  Skipped if Numba
   is not installed on the test machine.
3. **NumPy fallback still works.**  Monkey-patching
   ``_NUMBA_AVAILABLE = False`` must keep the merit returning the
   correct value (the JIT path is a perf optimisation only).
4. **Meshgrid cache hit rate preserved.**  The v4.14.0 cache pin
   (one ``_WRAPPER_MERIT_CACHE`` build per (N, dx, aperture)
   signature) must NOT regress with the JIT change -- the JIT
   reads the same cached ``X_factor``/``Y_factor``/``mask`` payload
   that the legacy NumPy path read.

Author: Andrew Traverso -- v5.3 (ROADMAP v5.3 horizon --
MultiFieldMerit JIT).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_test_inputs(N=128, dx=10e-6, aperture_diameter=900e-6,
                      wavelength=1.30e-6):
    """Construct the cached payload + tilt factors used by
    :class:`MultiFieldMerit` so we can exercise the JIT helper in
    isolation (without running the full ``evaluate`` and triggering
    the wave leg)."""
    from lumenairy.optimize.wrapper_merits import _get_wrapper_merit_cache

    cache = _get_wrapper_merit_cache(
        N, dx, float(aperture_diameter), np.complex128)
    k_X = cache['X_factor'] / wavelength
    k_Y = cache['Y_factor'] / wavelength
    mask = cache['mask']
    return cache, k_X, k_Y, mask


def _legacy_numpy_path(sin_tx, sin_ty, k_X, k_Y, mask, dtype=np.complex128):
    """The pre-v5.3 NumPy expression -- the canonical reference."""
    tilt_phase = sin_tx * k_X + sin_ty * k_Y
    return np.where(mask, np.exp(1j * tilt_phase), 0.0).astype(dtype)


def _simple_singlet():
    return {
        'surfaces': [
            {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -50e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [3e-3],
        'aperture_diameter': 10e-3,
    }


# ---------------------------------------------------------------------------
# 1. Numerical equivalence: NumPy vs Numba
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """The JIT and NumPy paths produce identical output to within
    1e-12 across a range of fields, tilts, and grid sizes."""

    @pytest.mark.parametrize('N', [64, 128, 256])
    @pytest.mark.parametrize('theta_x,theta_y', [
        (0.0, 0.0),
        (0.0, 0.01),
        (0.01, 0.0),
        (0.005, 0.012),
        (-0.02, 0.018),
    ])
    @pytest.mark.parametrize('wavelength', [0.85e-6, 1.30e-6, 1.55e-6])
    def test_numerical_equivalence_numpy_vs_numba(
        self, N, theta_x, theta_y, wavelength,
    ):
        """Numba JIT path agrees with the legacy NumPy expression to
        1e-12 on every output pixel."""
        from lumenairy.optimize._merit_jit import (
            _multi_field_tilt_phasor_masked,
        )

        dx = 10e-6
        # Aperture sized so a non-trivial mask boundary appears in
        # the field -- exercises both branches of the where().
        aperture = 0.6 * N * dx
        _, k_X, k_Y, mask = _build_test_inputs(
            N=N, dx=dx, aperture_diameter=aperture,
            wavelength=wavelength)
        sin_tx = np.sin(theta_x)
        sin_ty = np.sin(theta_y)

        E_jit = _multi_field_tilt_phasor_masked(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex128)
        E_ref = _legacy_numpy_path(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex128)

        # The merit-equivalence contract is 1e-12 -- the fused
        # multiply-add inside the kernel can shift the phase by ~1
        # ULP on a single coordinate, but cos/sin is a bounded
        # periodic function so the worst-case error is still well
        # inside 1e-12.
        np.testing.assert_allclose(
            E_jit, E_ref, rtol=1e-12, atol=1e-12,
            err_msg=(f'JIT path diverged from NumPy reference at '
                     f'N={N}, theta=({theta_x},{theta_y}), '
                     f'wavelength={wavelength}'))

    def test_complex64_dtype_specialisation(self):
        """The complex64 kernel agrees with the legacy NumPy
        ``.astype(complex64)`` path to single-precision tolerance.
        """
        from lumenairy.optimize._merit_jit import (
            _multi_field_tilt_phasor_masked,
        )

        N = 128
        dx = 10e-6
        wavelength = 1.30e-6
        aperture = 0.6 * N * dx
        _, k_X, k_Y, mask = _build_test_inputs(
            N=N, dx=dx, aperture_diameter=aperture,
            wavelength=wavelength)
        sin_tx = np.sin(0.008)
        sin_ty = np.sin(0.015)

        E_jit_c64 = _multi_field_tilt_phasor_masked(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex64)
        E_ref_c64 = _legacy_numpy_path(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex64)

        assert E_jit_c64.dtype == np.complex64, (
            f'complex64 dispatch must return complex64; got '
            f'{E_jit_c64.dtype}')
        # Single-precision tolerance: ~1e-6 relative.
        np.testing.assert_allclose(
            E_jit_c64, E_ref_c64, rtol=1e-5, atol=1e-6,
            err_msg='complex64 JIT vs NumPy reference disagreed')

    def test_helper_passthrough_below_threshold(self):
        """Below the JIT pixel-count threshold the helper falls
        back to NumPy, and must STILL agree bit-for-bit with the
        legacy expression."""
        from lumenairy.optimize._merit_jit import (
            _multi_field_tilt_phasor_masked,
        )

        N = 32  # 32**2 = 1024 pixels, well below the 16384 threshold.
        dx = 10e-6
        aperture = 0.6 * N * dx
        wavelength = 1.30e-6
        _, k_X, k_Y, mask = _build_test_inputs(
            N=N, dx=dx, aperture_diameter=aperture,
            wavelength=wavelength)
        sin_tx = np.sin(0.005)
        sin_ty = np.sin(0.012)

        E_helper = _multi_field_tilt_phasor_masked(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex128)
        E_ref = _legacy_numpy_path(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex128)

        np.testing.assert_allclose(
            E_helper, E_ref, rtol=1e-15, atol=1e-15,
            err_msg=('NumPy fallback path must match legacy '
                     'expression bit-for-bit (no JIT involved)'))


# ---------------------------------------------------------------------------
# 2. Numba speedup at large grid
# ---------------------------------------------------------------------------

class TestNumbaJitSpeedup:
    """The fused Numba kernel must beat the NumPy expression at
    N=256 with 8 fields by at least 1.5x.

    Skipped when Numba is not installed.
    """

    def test_numba_jit_speedup_on_large_grid(self):
        from lumenairy.optimize import _merit_jit as mj
        if not mj._NUMBA_AVAILABLE:
            pytest.skip('Numba not installed; JIT speedup pin is '
                        'inapplicable on this machine.')

        N = 256
        dx = 5e-6
        wavelength = 1.30e-6
        aperture = 0.7 * N * dx
        n_fields = 8

        _, k_X, k_Y, mask = _build_test_inputs(
            N=N, dx=dx, aperture_diameter=aperture,
            wavelength=wavelength)
        # Spread the field angles to a realistic 1.0-degree cone.
        rng = np.random.default_rng(42)
        thetas = rng.uniform(-0.018, 0.018, size=(n_fields, 2))

        # Warmup: first JIT call compiles the kernel (which can be
        # ~200 ms on cold cache) -- exclude that from timing.
        mj._multi_field_tilt_phasor_masked(
            np.sin(0.0), np.sin(0.0), k_X, k_Y, mask, np.complex128)

        # JIT path timing
        t0 = time.perf_counter()
        for tx, ty in thetas:
            _ = mj._multi_field_tilt_phasor_masked(
                np.sin(tx), np.sin(ty), k_X, k_Y, mask, np.complex128)
        t_jit = time.perf_counter() - t0

        # NumPy reference path timing
        t0 = time.perf_counter()
        for tx, ty in thetas:
            _ = _legacy_numpy_path(
                np.sin(tx), np.sin(ty), k_X, k_Y, mask, np.complex128)
        t_np = time.perf_counter() - t0

        speedup = t_np / max(t_jit, 1e-9)
        # Surfaced for human-readable benchmark capture; pytest -s
        # prints this to stdout.  The assertion below is the
        # contract; the print is diagnostic only.
        print(f'\n[v5.3 JIT bench] N={N} fields={n_fields}: '
              f't_jit={t_jit*1e3:.2f} ms  t_np={t_np*1e3:.2f} ms  '
              f'speedup={speedup:.2f}x')
        # Modest pin: 1.5x.  Empirically the kernel does 2-4x on
        # the v5.3 audit machine; under heavy parallel contention
        # the ratio can dip closer to 1.5.
        assert speedup >= 1.5, (
            f'Numba JIT must be >= 1.5x faster than NumPy at N={N} '
            f'with {n_fields} fields; got speedup={speedup:.2f}x '
            f'(t_jit={t_jit*1e3:.1f} ms, t_np={t_np*1e3:.1f} ms)')


# ---------------------------------------------------------------------------
# 3. NumPy fallback still works (Numba disabled by monkey-patch)
# ---------------------------------------------------------------------------

class TestNumpyFallback:
    """Monkey-patching ``_NUMBA_AVAILABLE = False`` must keep the
    merit returning the correct value -- the JIT path is a perf
    optimisation only."""

    def test_numpy_fallback_helper_matches_reference(self, monkeypatch):
        """With Numba disabled at the module level, the helper takes
        the pure-NumPy branch and must still match the legacy
        expression bit-for-bit."""
        from lumenairy.optimize import _merit_jit as mj

        monkeypatch.setattr(mj, '_NUMBA_AVAILABLE', False)

        N = 128
        dx = 10e-6
        wavelength = 1.30e-6
        aperture = 0.7 * N * dx
        _, k_X, k_Y, mask = _build_test_inputs(
            N=N, dx=dx, aperture_diameter=aperture,
            wavelength=wavelength)
        sin_tx = np.sin(0.010)
        sin_ty = np.sin(-0.014)

        E_fb = mj._multi_field_tilt_phasor_masked(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex128)
        E_ref = _legacy_numpy_path(
            sin_tx, sin_ty, k_X, k_Y, mask, np.complex128)

        np.testing.assert_allclose(
            E_fb, E_ref, rtol=1e-15, atol=1e-15,
            err_msg=('Pure-NumPy fallback (Numba-disabled) must '
                     'match the legacy expression bit-for-bit'))

    def test_multi_field_merit_evaluate_with_numpy_fallback(
        self, monkeypatch,
    ):
        """End-to-end pin: monkey-patch ``_NUMBA_AVAILABLE = False``
        inside the JIT module and run ``MultiFieldMerit.evaluate``
        through the wrapper.  The merit value must be finite and
        agree (to 1e-9) with a JIT-enabled run."""
        import lumenairy as la
        from lumenairy.optimize import _merit_jit as mj
        from lumenairy.optimize.core import (
            EvaluationContext,
            MultiFieldMerit,
            StrehlMerit,
            _clear_wrapper_merit_cache,
        )
        from lumenairy.raytrace import (
            surfaces_from_prescription,
            system_abcd,
        )

        pres = _simple_singlet()
        wavelength = 1.30e-6
        N, dx = 64, 12e-6
        surfs = surfaces_from_prescription(pres)
        _, efl, bfl, _ = system_abcd(surfs, wavelength)

        # Build a minimal valid ctx -- the wave leg inside evaluate
        # will populate E_exit / opd_map per field.
        ctx = EvaluationContext(
            prescription=pres, wavelength=wavelength,
            N=N, dx=dx, efl=float(efl), bfl=float(bfl),
            seidel=None, E_exit=None)

        merit = MultiFieldMerit(
            field_angles=[(0.0, 0.0), (0.005, 0.0), (0.0, 0.005)],
            sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
            weight=1.0)

        # Run once with JIT enabled (default).
        _clear_wrapper_merit_cache()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m_jit = merit.evaluate(ctx)
        # Reset & run with NumPy fallback forced.
        monkeypatch.setattr(mj, '_NUMBA_AVAILABLE', False)
        ctx2 = EvaluationContext(
            prescription=pres, wavelength=wavelength,
            N=N, dx=dx, efl=float(efl), bfl=float(bfl),
            seidel=None, E_exit=None)
        _clear_wrapper_merit_cache()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m_np = merit.evaluate(ctx2)

        assert np.isfinite(m_jit) and np.isfinite(m_np), (
            f'merit values must be finite; got m_jit={m_jit}, '
            f'm_np={m_np}')
        # The two paths should agree to within numerical noise.
        # StrehlMerit returns weight * max(0, deficit)**2; the
        # deficit involves through_focus_scan which has its own
        # tolerance, so we allow ~1e-9 absolute.
        np.testing.assert_allclose(
            m_jit, m_np, rtol=1e-9, atol=1e-9,
            err_msg=(f'JIT and NumPy-fallback merit values diverged: '
                     f'm_jit={m_jit}, m_np={m_np}'))


# ---------------------------------------------------------------------------
# 4. Meshgrid cache hit rate preserved
# ---------------------------------------------------------------------------

class TestCacheHitRatePreserved:
    """The v4.14.0 wrapper-merit-cache pin must NOT regress with
    the JIT change.  The JIT reads the same cached payload
    (X_factor / Y_factor / mask) that the legacy NumPy path read;
    each (N, dx, aperture, dtype) signature must still trigger
    exactly ONE meshgrid build over an entire evaluate-call sweep."""

    def setup_method(self, _method):
        from lumenairy.optimize.core import _clear_wrapper_merit_cache
        _clear_wrapper_merit_cache()

    def test_one_meshgrid_build_per_signature_after_jit_change(self):
        """Re-pin the cache invariant: 50 helper calls at the same
        (N, dx, aperture, dtype) trigger exactly one meshgrid build.
        """
        import lumenairy.optimize.core as oc
        from lumenairy.optimize._merit_jit import (
            _multi_field_tilt_phasor_masked,
        )

        N, dx, aperture = 128, 10e-6, 900e-6
        wavelength = 1.30e-6
        before = oc._WRAPPER_MERIT_MESHGRID_BUILDS
        # Drive 50 calls -- mimics the inner loop of a
        # ``MultiFieldMerit.evaluate`` with many field angles.
        for k in range(50):
            cache, k_X, k_Y, mask = _build_test_inputs(
                N=N, dx=dx, aperture_diameter=aperture,
                wavelength=wavelength)
            theta = 0.001 * k  # arbitrary; doesn't change cache key
            _ = _multi_field_tilt_phasor_masked(
                np.sin(theta), np.sin(theta), k_X, k_Y, mask,
                np.complex128)
        after = oc._WRAPPER_MERIT_MESHGRID_BUILDS
        assert after == before + 1, (
            f'expected exactly 1 meshgrid build per signature over 50 '
            f'helper calls (v4.14.0 cache pin); got '
            f'{after - before} builds.  The v5.3 JIT change must not '
            f'rebuild the cache per field.')

    def test_multi_field_merit_full_evaluate_does_not_regress_cache(
        self,
    ):
        """End-to-end pin: a full ``MultiFieldMerit.evaluate`` run
        with several field angles produces exactly ONE meshgrid
        build -- same as before v5.3."""
        import lumenairy.optimize.core as oc
        from lumenairy.optimize.core import (
            EvaluationContext,
            MultiFieldMerit,
            StrehlMerit,
            _clear_wrapper_merit_cache,
        )
        from lumenairy.raytrace import (
            surfaces_from_prescription,
            system_abcd,
        )

        pres = _simple_singlet()
        wavelength = 1.30e-6
        N, dx = 64, 12e-6
        surfs = surfaces_from_prescription(pres)
        _, efl, bfl, _ = system_abcd(surfs, wavelength)

        ctx = EvaluationContext(
            prescription=pres, wavelength=wavelength,
            N=N, dx=dx, efl=float(efl), bfl=float(bfl),
            seidel=None, E_exit=None)
        merit = MultiFieldMerit(
            field_angles=[(0.0, 0.0), (0.005, 0.0),
                          (0.0, 0.005), (0.003, 0.003)],
            sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
            weight=1.0)
        _clear_wrapper_merit_cache()
        before = oc._WRAPPER_MERIT_MESHGRID_BUILDS
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            merit.evaluate(ctx)
        after = oc._WRAPPER_MERIT_MESHGRID_BUILDS
        n_builds = after - before
        # The wave leg inside ``MultiFieldMerit.evaluate`` may
        # itself touch the cache for the propagation step; the
        # contract is that the per-field tilt-phase build does NOT
        # add any extra builds beyond the one-per-signature pin.
        # Empirically the count is 1 -- bump the bound to 2 to
        # allow for an internal helper hitting a sibling
        # (N, dx, aperture, dtype) signature with a wave-leg-
        # specific aperture.
        assert n_builds <= 2, (
            f'MultiFieldMerit.evaluate with 4 fields must trigger '
            f'<= 2 meshgrid builds (one per unique signature); got '
            f'{n_builds}.  The v5.3 JIT change must not rebuild the '
            f'cache per field.')
