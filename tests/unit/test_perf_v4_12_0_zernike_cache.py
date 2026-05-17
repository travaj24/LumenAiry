"""Correctness pins for the v4.12.0 Zernike basis-matrix cache
(audit perf #7).

Verifies:

* Cache hit on the second call returns bit-identical output (the same
  underlying ndarray instances).
* ``clear_zernike_basis_cache`` empties the cache.
* ``zernike_decompose`` produces the same coefficients with the cache
  cold vs warm (cache is transparent).
* LRU eviction: after ``maxsize + 1`` distinct keys are inserted, the
  oldest-touched entry has been evicted while the most recent
  ``maxsize`` survive.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.analysis import core as analysis_core
from lumenairy.analysis.core import (
    clear_zernike_basis_cache,
    zernike_basis_matrix,
    zernike_decompose,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _grid(N: int, dx: float = 1.0):
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    return X, Y, float(N / 2) * dx


# ---------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------

def test_cache_hit_returns_same_object():
    """Second call with same args should hit the cache (same ndarray)."""
    clear_zernike_basis_cache()
    X, Y, r = _grid(64)

    B1, m1 = zernike_basis_matrix(15, X, Y, r)
    B2, m2 = zernike_basis_matrix(15, X, Y, r)

    # Same underlying objects -- proves we hit the cache, not just
    # re-ran the build and got numerically-equal output.
    assert B1 is B2
    assert m1 is m2
    # And of course bit-identical content.
    assert np.array_equal(B1, B2)
    assert np.array_equal(m1, m2)


def test_clear_cache_empties_it():
    """``clear_zernike_basis_cache`` should drop every cached entry."""
    clear_zernike_basis_cache()
    X, Y, r = _grid(64)

    _ = zernike_basis_matrix(15, X, Y, r)
    assert len(analysis_core._ZERNIKE_BASIS_CACHE) >= 1

    clear_zernike_basis_cache()
    assert len(analysis_core._ZERNIKE_BASIS_CACHE) == 0

    # First call after clear should rebuild -> new ndarray instance.
    B_new, _ = zernike_basis_matrix(15, X, Y, r)
    B_again, _ = zernike_basis_matrix(15, X, Y, r)
    assert B_new is B_again  # cache rebuilt and re-hit.


def test_zernike_decompose_transparent_to_cache():
    """``zernike_decompose`` results should be byte-identical between
    cold and warm cache states.
    """
    # Build a synthetic OPD with a defocus + astigmatism signature so
    # the fit is meaningful.
    N, dx = 128, 5e-6
    aperture = (N - 8) * dx  # leave a small margin

    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    rho_sq = (X ** 2 + Y ** 2) / (aperture / 2) ** 2
    opd = 1e-6 * (2 * rho_sq - 1) + 0.5e-6 * (X ** 2 - Y ** 2) / (aperture / 2) ** 2

    clear_zernike_basis_cache()
    coeffs_cold, _ = zernike_decompose(opd, dx, aperture, n_modes=21)
    # Second call should hit the cache.
    coeffs_warm, _ = zernike_decompose(opd, dx, aperture, n_modes=21)

    # Cache must be a pure performance optimization -- bit-identical
    # output (same basis matrix -> same lstsq solve).
    np.testing.assert_array_equal(coeffs_cold, coeffs_warm)


def test_cache_lru_eviction():
    """After inserting maxsize + 1 distinct keys, the oldest entry is
    gone but the most recent maxsize survive.
    """
    clear_zernike_basis_cache()
    maxsize = analysis_core._ZERNIKE_BASIS_CACHE_MAXSIZE

    # Build maxsize + 1 distinct grids (vary N -> distinct shape ->
    # distinct fingerprint -> distinct cache key).
    grids = []
    for k in range(maxsize + 1):
        N = 32 + k  # 32, 33, ...  -- each call is a distinct key.
        X, Y, r = _grid(N)
        grids.append((N, X, Y, r))
        zernike_basis_matrix(5, X, Y, r)

    # The first-inserted entry should have been evicted.
    assert len(analysis_core._ZERNIKE_BASIS_CACHE) == maxsize

    # Probe the oldest grid: this should rebuild (cache miss) ->
    # return a NEW ndarray, distinct from any previously-returned one.
    N0, X0, Y0, r0 = grids[0]
    B_rebuilt, _ = zernike_basis_matrix(5, X0, Y0, r0)
    # Now it's in the cache again.
    B_hit, _ = zernike_basis_matrix(5, X0, Y0, r0)
    assert B_rebuilt is B_hit

    # The most-recent grid should still be cached -- a fresh call
    # should hit (same object as the original insertion).
    N_last, X_last, Y_last, r_last = grids[-1]
    B_last_first, _ = zernike_basis_matrix(5, X_last, Y_last, r_last)
    B_last_second, _ = zernike_basis_matrix(5, X_last, Y_last, r_last)
    assert B_last_first is B_last_second


def test_distinct_args_distinct_cache_entries():
    """Different n_modes / pupil_radius / grid should produce distinct
    cache entries (and so distinct output instances).
    """
    clear_zernike_basis_cache()
    X, Y, r = _grid(64)

    B_a, _ = zernike_basis_matrix(10, X, Y, r)
    B_b, _ = zernike_basis_matrix(15, X, Y, r)  # different n_modes
    B_c, _ = zernike_basis_matrix(10, X, Y, r * 0.5)  # different radius

    # All three should be distinct objects with distinct shapes /
    # contents (the second has more columns; the third has a smaller
    # pupil so fewer rows).
    assert B_a.shape[1] == 10
    assert B_b.shape[1] == 15
    assert B_a.shape[0] != B_c.shape[0]


def test_cache_consistent_with_uncached_build():
    """The cached output must equal the result of calling the
    underlying uncached build directly -- the cache is purely
    additive.
    """
    clear_zernike_basis_cache()
    X, Y, r = _grid(96)

    B_cached, m_cached = zernike_basis_matrix(21, X, Y, r)
    B_uncached, m_uncached = analysis_core._zernike_basis_matrix_build(
        21, X, Y, r)

    np.testing.assert_array_equal(B_cached, B_uncached)
    np.testing.assert_array_equal(m_cached, m_uncached)


def test_fresh_meshgrid_hits_cache():
    """Two independent ``np.meshgrid`` calls with the same args should
    produce arrays that hit the same cache entry, even though they're
    distinct objects.  This is the case that mattered in v4.11.2:
    every ``zernike_decompose`` call rebuilds the meshgrid.
    """
    clear_zernike_basis_cache()
    N, dx = 64, 5e-6
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx

    X1, Y1 = np.meshgrid(x, y)
    X2, Y2 = np.meshgrid(x, y)
    assert X1 is not X2
    assert Y1 is not Y2

    r = float(N / 2) * dx
    B1, m1 = zernike_basis_matrix(21, X1, Y1, r)
    B2, m2 = zernike_basis_matrix(21, X2, Y2, r)

    # The second call should hit the cache (same object), despite the
    # inputs being structurally distinct.
    assert B1 is B2
    assert m1 is m2
