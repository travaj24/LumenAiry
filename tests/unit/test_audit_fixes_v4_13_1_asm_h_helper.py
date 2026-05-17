"""Regression pin for the v4.13.1 shared ``_build_asm_H_square``
helper (audit P2 #15).

Before v4.13.1 ``analysis.detector._build_asm_H_for_lenslet``
duplicated the angular-spectrum H + bandlimit construction from
``propagators.propagation``.  Drift risk: if the canonical propagator
flipped its bandlimit policy / fftshift convention / dtype handling
the detector copy silently diverged.

v4.13.1 consolidates both call sites into a single
``propagators.propagation._build_asm_H_square`` helper.  This test
pins the helper's output against a hand-built reference computed
inline so any future change to the H/bandlimit math has to be
explicit and signed off here.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.propagators.propagation import _build_asm_H_square


def _reference_H(N, dx, z, wavelength, dtype=np.complex128, bandlimit=True):
    """Hand-built reference following the centered convention.

    Matches the inline JAX-path build in
    :func:`angular_spectrum_propagate` for square grids: centered
    frequency axis, evanescent zeroing, Matsushima bandlimit (outer
    product of per-axis masks).
    """
    k = 2.0 * np.pi / wavelength
    fx = (np.arange(N, dtype=np.float64) - N / 2) / (N * dx)
    fy = fx
    kx_sq = (2 * np.pi * fx) ** 2
    ky_sq = (2 * np.pi * fy) ** 2
    kz_sq = k ** 2 - kx_sq[None, :] - ky_sq[:, None]
    prop = kz_sq > 0
    kz = np.where(prop, np.sqrt(np.where(prop, kz_sq, 0.0)), 0.0)
    H = np.where(prop, np.exp(1j * kz * z), 0.0).astype(dtype)
    if bandlimit and z != 0:
        L = N * dx
        f_max = L / (2 * wavelength * abs(z))
        bl_x = np.abs(fx) < f_max
        bl_y = np.abs(fy) < f_max
        mask = bl_x[None, :] & bl_y[:, None]
        H = H * mask.astype(dtype)
    return H


def test_helper_matches_inline_bandlimit_on_small_grid():
    """64x64 grid with bandlimit enabled: helper output == hand-built
    reference to within machine epsilon."""
    N, dx, z, lam = 64, 1.0e-6, 1.5e-3, 633e-9
    H_helper = _build_asm_H_square(N, dx, z, lam, bandlimit=True)
    H_ref = _reference_H(N, dx, z, lam, bandlimit=True)
    assert H_helper.shape == (N, N)
    assert H_helper.dtype == np.complex128
    diff = float(np.max(np.abs(H_helper - H_ref)))
    assert diff <= 1e-14, f"helper vs reference: max abs diff = {diff:.3e}"


def test_helper_matches_inline_bandlimit_off_small_grid():
    """64x64 grid with bandlimit disabled: helper output == hand-built
    reference to within machine epsilon."""
    N, dx, z, lam = 64, 1.0e-6, 1.5e-3, 633e-9
    H_helper = _build_asm_H_square(N, dx, z, lam, bandlimit=False)
    H_ref = _reference_H(N, dx, z, lam, bandlimit=False)
    diff = float(np.max(np.abs(H_helper - H_ref)))
    assert diff <= 1e-14, f"helper vs reference: max abs diff = {diff:.3e}"


def test_helper_matches_inline_larger_grid():
    """256x256 grid, NIR wavelength, longer propagation distance."""
    N, dx, z, lam = 256, 2.5e-6, 5.0e-3, 1.55e-6
    H_helper = _build_asm_H_square(N, dx, z, lam, bandlimit=True)
    H_ref = _reference_H(N, dx, z, lam, bandlimit=True)
    diff = float(np.max(np.abs(H_helper - H_ref)))
    assert diff <= 1e-14, f"helper vs reference: max abs diff = {diff:.3e}"


def test_helper_complex64_dtype_promotion():
    """complex64 request: helper returns complex64, magnitudes match
    reference to within float32 epsilon."""
    N, dx, z, lam = 64, 1.0e-6, 1.5e-3, 633e-9
    H_helper = _build_asm_H_square(
        N, dx, z, lam, dtype=np.complex64, bandlimit=True)
    H_ref = _reference_H(N, dx, z, lam, dtype=np.complex64, bandlimit=True)
    assert H_helper.dtype == np.complex64
    diff = float(np.max(np.abs(H_helper - H_ref)))
    assert diff <= 1e-6, f"complex64 helper vs reference: {diff:.3e}"


def test_helper_real_dtype_promoted_to_complex128():
    """Real dtype request must be promoted to complex128, not silently
    lose the imaginary part."""
    N, dx, z, lam = 32, 1.0e-6, 1.5e-3, 633e-9
    H_helper = _build_asm_H_square(
        N, dx, z, lam, dtype=np.float64, bandlimit=True)
    assert H_helper.dtype == np.complex128, (
        f"real dtype must be promoted; got {H_helper.dtype}")


def test_helper_matches_detector_callsite():
    """The Shack-Hartmann detector code path calls
    ``_build_asm_H_square(sa_pixels, dx, lenslet_focal, wavelength,
    dtype=E_batch.dtype, bandlimit=True)``.  Pin that exact signature
    pattern."""
    sa_pixels, dx, lenslet_focal, wavelength = 16, 5.0e-6, 4.0e-3, 633e-9
    H_helper = _build_asm_H_square(
        sa_pixels, dx, lenslet_focal, wavelength,
        dtype=np.complex128, bandlimit=True)
    H_ref = _reference_H(
        sa_pixels, dx, lenslet_focal, wavelength,
        dtype=np.complex128, bandlimit=True)
    assert H_helper.shape == (sa_pixels, sa_pixels)
    diff = float(np.max(np.abs(H_helper - H_ref)))
    assert diff <= 1e-14, f"detector-callsite helper drift: {diff:.3e}"


def test_helper_zero_z_short_circuits_to_unity():
    """z == 0 produces H = 1 everywhere (the propagator's zero-distance
    short-circuit; bandlimit is silently ignored at z=0 as the formula
    diverges)."""
    N, dx, lam = 32, 1.0e-6, 633e-9
    H = _build_asm_H_square(N, dx, 0.0, lam, bandlimit=True)
    # At z=0, exp(1j*kz*0) = 1 for all propagating modes, 0 for evanescent.
    # In a typical regime k > max(kx, ky) so propagating-mask is full,
    # H should be ~1.0 everywhere.
    k = 2.0 * np.pi / lam
    fx = (np.arange(N, dtype=np.float64) - N / 2) / (N * dx)
    kx_max = 2 * np.pi * np.max(np.abs(fx))
    # k must dominate kx_max for the unity-everywhere assertion.
    assert k > kx_max * np.sqrt(2), (
        "test setup: kx grid larger than k; pick a smaller N or dx.")
    assert np.allclose(H, 1.0 + 0.0j, atol=1e-14)
