"""Correctness pin for the v4.13.0 Chebyshev arccos-hoist
(Tier-2 perf, audit group alpha task 5).

The pre-v4.13.0 ``surface_sag_chebyshev`` recomputed ``np.arccos(xn)``
and ``np.arccos(yn)`` inside the per-coefficient loop.  v4.13.0 hoists
both arccos calls outside the loop -- they depend only on the
normalised grid, not on the polynomial order (i, j).  This test pins
the optimised output to a reference inline computation at 10
coefficients on a 64x64 grid.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements.freeform import surface_sag_chebyshev


def _scalar_reference(X, Y, R, conic, cheb_coeffs, norm_x, norm_y):
    """Pre-v4.13.0 Chebyshev sag (arccos inside the loop), inlined
    here as the pin reference."""
    from lumenairy.elements.lenses import surface_sag_general
    h_sq = X ** 2 + Y ** 2
    sag = surface_sag_general(h_sq, R, conic)
    if cheb_coeffs:
        xn_raw = X / norm_x
        yn_raw = Y / norm_y
        outside = (np.abs(xn_raw) > 1.0) | (np.abs(yn_raw) > 1.0)
        xn = np.clip(xn_raw, -1, 1)
        yn = np.clip(yn_raw, -1, 1)
        departure = np.zeros_like(sag)
        for (i, j), c in cheb_coeffs.items():
            Ti = np.cos(i * np.arccos(xn))
            Tj = np.cos(j * np.arccos(yn))
            departure = departure + c * Ti * Tj
        sag = sag + np.where(outside, 0.0, departure)
    return sag


def _build_grid(N=64, half_extent=5e-3):
    x = np.linspace(-half_extent, half_extent, N)
    y = np.linspace(-half_extent, half_extent, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


def _10_coefficient_set():
    """A non-trivial, varied set of 10 Chebyshev coefficients spanning
    low and high polynomial orders."""
    return {
        (0, 0): 1e-7,
        (1, 0): 5e-7,
        (0, 1): -3e-7,
        (2, 1): 2e-8,
        (1, 2): -1e-8,
        (3, 0): 4e-9,
        (0, 4): 7e-9,
        (3, 3): -2e-9,
        (5, 2): 1e-9,
        (4, 5): -5e-10,
    }


def test_hoisted_arccos_matches_inline_reference_64x64():
    """64x64 grid, 10 coefficients, no base curvature."""
    X, Y = _build_grid(N=64)
    coeffs = _10_coefficient_set()
    norm_x = norm_y = 4e-3  # smaller than grid extent so some pixels are outside
    sag_hoisted = surface_sag_chebyshev(
        X, Y, R=np.inf, conic=0.0,
        cheb_coeffs=coeffs, norm_x=norm_x, norm_y=norm_y)
    sag_ref = _scalar_reference(
        X, Y, R=np.inf, conic=0.0,
        cheb_coeffs=coeffs, norm_x=norm_x, norm_y=norm_y)
    tol = 1e-12 * max(float(np.max(np.abs(sag_ref))), 1.0)
    assert np.max(np.abs(sag_hoisted - sag_ref)) <= tol, (
        f'sag mismatch: max diff '
        f'{np.max(np.abs(sag_hoisted - sag_ref)):.3e}')


def test_hoisted_arccos_matches_inline_reference_with_base_curvature():
    """Same pin but with a curved base sphere underneath the freeform
    departure -- exercises the surface_sag_general code path too."""
    X, Y = _build_grid(N=64)
    coeffs = _10_coefficient_set()
    sag_hoisted = surface_sag_chebyshev(
        X, Y, R=20e-3, conic=-0.5,
        cheb_coeffs=coeffs, norm_x=4e-3, norm_y=4e-3)
    sag_ref = _scalar_reference(
        X, Y, R=20e-3, conic=-0.5,
        cheb_coeffs=coeffs, norm_x=4e-3, norm_y=4e-3)
    tol = 1e-12 * max(float(np.max(np.abs(sag_ref))), 1.0)
    assert np.max(np.abs(sag_hoisted - sag_ref)) <= tol


def test_hoisted_arccos_outside_region_zeroed():
    """Pixels outside [-norm_x, norm_x] x [-norm_y, norm_y] must
    retain only the base-conic sag with zero freeform departure."""
    X, Y = _build_grid(N=64)
    coeffs = _10_coefficient_set()
    norm = 1e-3  # tight, ensures most of the 5 mm grid is outside
    sag = surface_sag_chebyshev(
        X, Y, R=np.inf, conic=0.0,
        cheb_coeffs=coeffs, norm_x=norm, norm_y=norm)
    outside = (np.abs(X) > norm) | (np.abs(Y) > norm)
    # Base sag with R=inf and conic=0 is identically zero -> all
    # outside pixels must be 0.
    assert np.allclose(sag[outside], 0.0, atol=1e-15)
