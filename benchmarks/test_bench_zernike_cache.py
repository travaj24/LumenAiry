"""Benchmarks for the v4.12.0 Zernike basis-matrix cache (audit perf #7).

Three workload pins:

* ``zernike_basis_cold`` -- first call after cache clear.  Should
  match the pre-fix wall time (~22 ms on a typical laptop at 256x256,
  21 modes), since the cold path is the original build.
* ``zernike_basis_warm`` -- second-and-onwards call with the same
  args.  Should be in the low-microseconds range -- a dict lookup +
  tuple return, with no array build.
* ``zernike_decompose_10x`` -- ten ``zernike_decompose`` calls on the
  same grid.  With the cache, only the first pays the basis-build
  cost; the other nine just pay for the lstsq solve.  Run this both
  warm (default: cache pre-warmed in setup) and effectively-cold
  (cache cleared each iteration) for direct comparison.

The benchmarks are intentionally cheap (<10 s total) so they slot
into the normal ``test_bench_baseline.py`` cadence.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.analysis.core import (
    clear_zernike_basis_cache,
    zernike_basis_matrix,
    zernike_decompose,
)


# ---------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------

_N = 256
_DX = 1.0
_N_MODES = 21


def _make_grid(N: int = _N, dx: float = _DX):
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    return X, Y, float(N / 2) * dx


def _make_opd_grid(N: int = 256, dx: float = 5e-6):
    """Grid for the ``zernike_decompose`` benchmark.  256x256 is the
    same scale as ``test_bench_zernike_basis_21x_256`` so basis-build
    dominates and the cache savings are visible end-to-end (the
    lstsq solve only costs a few ms even at 21 modes)."""
    aperture = (N - 8) * dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    rho_sq = (X ** 2 + Y ** 2) / (aperture / 2) ** 2
    opd = (1e-6 * (2 * rho_sq - 1)
           + 0.5e-6 * (X ** 2 - Y ** 2) / (aperture / 2) ** 2)
    return opd, dx, aperture


# ---------------------------------------------------------------------
# Cold-cache: should match the pre-fix baseline.
# ---------------------------------------------------------------------

def test_bench_zernike_basis_cold(benchmark):
    """Cold-cache build cost.  The cache is cleared on every iteration
    so this measures the unoptimized path."""
    X, Y, r = _make_grid()

    def _cold():
        clear_zernike_basis_cache()
        return zernike_basis_matrix(_N_MODES, X, Y, r)

    benchmark(_cold)


# ---------------------------------------------------------------------
# Warm-cache: should be O(microseconds).
# ---------------------------------------------------------------------

def test_bench_zernike_basis_warm(benchmark):
    """Warm-cache hit.  We pre-build the basis once outside the
    benchmark so every timed iteration is a pure dict lookup."""
    X, Y, r = _make_grid()
    clear_zernike_basis_cache()
    # Pre-warm.
    _ = zernike_basis_matrix(_N_MODES, X, Y, r)

    benchmark(zernike_basis_matrix, _N_MODES, X, Y, r)


# ---------------------------------------------------------------------
# zernike_decompose: 10 calls on the same grid.
# ---------------------------------------------------------------------

def test_bench_zernike_decompose_10x_warm(benchmark):
    """Ten ``zernike_decompose`` calls on the same grid with the cache
    warmed once.  Should be roughly (build + 10 * solve) total -- the
    basis is shared across calls."""
    opd, dx, aperture = _make_opd_grid()
    # Pre-warm: first call builds the basis.
    clear_zernike_basis_cache()
    _ = zernike_decompose(opd, dx, aperture, n_modes=_N_MODES)

    def _ten():
        for _ in range(10):
            zernike_decompose(opd, dx, aperture, n_modes=_N_MODES)

    benchmark(_ten)


def test_bench_zernike_decompose_10x_cold(benchmark):
    """Ten ``zernike_decompose`` calls on the same grid with the cache
    cleared before each call.  This mimics the v4.11.2 behaviour --
    every call rebuilds the basis."""
    opd, dx, aperture = _make_opd_grid()

    def _ten_cold():
        for _ in range(10):
            clear_zernike_basis_cache()
            zernike_decompose(opd, dx, aperture, n_modes=_N_MODES)

    benchmark(_ten_cold)
