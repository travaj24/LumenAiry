"""Regression tests for audit finding S3-7 (sources/core.py).

The source factories used to materialise two dense ``N x N`` float64
coordinate grids via ``np.meshgrid(x, y)`` (plus squared temporaries --
~1 GB transient at N=8192) where broadcast 1-D axes ``x[None, :]`` /
``y[:, None]`` give a bit-identical result at a fraction of the memory.
The fix replaced every such site with broadcast views.

These tests are an *independent* oracle: each rebuilds the field from a
full ``np.meshgrid`` (the pre-refactor primitive) and the documented
closed form, then asserts the factory output is BIT-IDENTICAL
(``np.array_equal``) to that reference.  All probes deliberately break
x<->y symmetry (``dx != dy``, ``x0 != y0``, asymmetric mode indices,
``Nx != Ny``) so a transposed-axis regression in the broadcast
replacement would flip the field and fail, not slip through.
"""
from __future__ import annotations

import numpy as np

import lumenairy as la
from lumenairy.sources.core import (
    _resolve_complex_dtype,
    _schell_phase_realizations,
    hermite_physicist,
    laguerre_generalized,
)

# Asymmetric grid so an x<->y axis swap is observable.
_N = 32
_DX = 5.0e-6
_DY = 3.0e-6
_LAM = 1.31e-6
_X0 = 6.0e-6
_Y0 = -3.0e-6


def _axes(N, dx, dy):
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    return x, y


def test_tilted_plane_wave_broadcast_bit_identical():
    ax, ay, amp = 0.02, -0.011, 1.7
    E, x, y = la.create_tilted_plane_wave(
        _N, _DX, _LAM, angle_x=ax, angle_y=ay, amplitude=amp, dy=_DY)
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / _LAM
    phase = k0 * (np.sin(ax) * X + np.sin(ay) * Y)
    ref = (amp * np.exp(1j * phase)).astype(_resolve_complex_dtype(None))
    assert np.array_equal(E, ref)


def test_point_source_broadcast_bit_identical():
    z0, amp = -2.0e-3, 1.3
    E, x, y = la.create_point_source(
        _N, _DX, _LAM, x0=_X0, y0=_Y0, z0=z0, amplitude=amp, dy=_DY)
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / _LAM
    r = np.sqrt((X - _X0) ** 2 + (Y - _Y0) ** 2 + z0 ** 2)
    r = np.maximum(r, 0.5 * np.sqrt(_DX * _DX + _DY * _DY))
    sign = -1.0 if z0 > 0.0 else 1.0
    ref = (amp * np.exp(1j * sign * k0 * r) / r).astype(
        _resolve_complex_dtype(None))
    assert np.array_equal(E, ref)


def test_top_hat_broadcast_bit_identical():
    diameter = 60.0e-6
    E, x, y = la.create_top_hat_beam(
        _N, _DX, _LAM, diameter=diameter, x0=_X0, y0=_Y0, dy=_DY)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - _X0) ** 2 + (Y - _Y0) ** 2)
    ref = np.where(r <= diameter / 2, 1.0, 0.0).astype(
        _resolve_complex_dtype(None))
    norm = np.sqrt(np.sum(np.abs(ref) ** 2) * _DX * _DY)
    if norm > 0:
        ref /= norm
    assert np.array_equal(E, ref)


def test_annular_broadcast_bit_identical():
    od, idm = 80.0e-6, 30.0e-6
    E, x, y = la.create_annular_beam(
        _N, _DX, _LAM, outer_diameter=od, inner_diameter=idm,
        x0=_X0, y0=_Y0, dy=_DY)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - _X0) ** 2 + (Y - _Y0) ** 2)
    ref = np.where((r <= od / 2) & (r >= idm / 2), 1.0, 0.0).astype(
        _resolve_complex_dtype(None))
    norm = np.sqrt(np.sum(np.abs(ref) ** 2) * _DX * _DY)
    if norm > 0:
        ref /= norm
    assert np.array_equal(E, ref)


def test_bessel_broadcast_bit_identical():
    from scipy.special import j0
    cone = 0.05
    E, x, y = la.create_bessel_beam(
        _N, _DX, _LAM, cone, x0=_X0, y0=_Y0, dy=_DY)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - _X0) ** 2 + (Y - _Y0) ** 2)
    k_r = 2 * np.pi / _LAM * np.sin(cone)
    ref = j0(k_r * r).astype(_resolve_complex_dtype(None))
    assert np.array_equal(E, ref)


def test_hermite_gauss_broadcast_bit_identical():
    # m != n makes the field sensitive to an x<->y swap.
    m, n, w0 = 2, 1, 25.0e-6
    E, x, y = la.create_hermite_gauss(
        _N, _DX, w0, _LAM, m=m, n=n, x0=_X0, y0=_Y0, dy=_DY)
    X, Y = np.meshgrid(x, y)
    u = np.sqrt(2) * (X - _X0) / w0
    v = np.sqrt(2) * (Y - _Y0) / w0
    Hm = hermite_physicist(m, u)
    Hn = hermite_physicist(n, v)
    gaussian = np.exp(-((X - _X0) ** 2 + (Y - _Y0) ** 2) / w0 ** 2)
    ref = (Hm * Hn * gaussian).astype(_resolve_complex_dtype(None))
    norm = np.sqrt(np.sum(np.abs(ref) ** 2) * _DX * _DY)
    if norm > 0:
        ref /= norm
    assert np.array_equal(E, ref)


def test_laguerre_gauss_broadcast_bit_identical():
    p, l, w0 = 1, 2, 25.0e-6
    E, x, y = la.create_laguerre_gauss(
        _N, _DX, w0, _LAM, p=p, l=l, x0=_X0, y0=_Y0, dy=_DY)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - _X0) ** 2 + (Y - _Y0) ** 2)
    theta = np.arctan2(Y - _Y0, X - _X0)
    rho = np.sqrt(2) * r / w0
    L = laguerre_generalized(p, abs(l), rho ** 2)
    gaussian = np.exp(-r ** 2 / w0 ** 2)
    ref = (rho ** abs(l) * L * gaussian
           * np.exp(1j * l * theta)).astype(_resolve_complex_dtype(None))
    norm = np.sqrt(np.sum(np.abs(ref) ** 2) * _DX * _DY)
    if norm > 0:
        ref /= norm
    assert np.array_equal(E, ref)


def _schell_phi_meshgrid_ref(*, Ny, Nx, dx, dy, sigma_g, nr, seed):
    """Independent meshgrid reconstruction of _schell_phase_realizations,
    mirroring the KX/KY grid site (sources/core.py) exactly."""
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    spec_filter = np.exp(-(KX * KX + KY * KY) * (sigma_g ** 2) / 4.0)
    mean_I = float(np.sum(np.abs(spec_filter) ** 2) / (Ny * Nx))
    norm = np.sqrt(mean_I) if mean_I > 0.0 else 1.0
    out = np.empty((nr, Ny, Nx), dtype=np.complex128)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    rng = np.random.default_rng(seed)
    for k in range(nr):
        w_re = rng.standard_normal((Ny, Nx))
        w_im = rng.standard_normal((Ny, Nx))
        W = (w_re + 1j * w_im) * inv_sqrt2
        out[k] = np.fft.ifft2(np.fft.fft2(W) * spec_filter) / norm
    return out


def test_schell_phase_realizations_kxky_bit_identical():
    # Rectangular grid + dx != dy: the KX/KY orientation is load-bearing.
    Ny, Nx, sigma_g, nr, seed = 6, 10, 20.0e-6, 3, 12345
    phi = _schell_phase_realizations(
        N=Nx, Ny=Ny, Nx=Nx, dx=_DX, dy=_DY,
        coherence_length=sigma_g, n_realizations=nr,
        rng=np.random.default_rng(seed))
    ref = _schell_phi_meshgrid_ref(
        Ny=Ny, Nx=Nx, dx=_DX, dy=_DY, sigma_g=sigma_g, nr=nr, seed=seed)
    assert np.array_equal(phi, ref)


def test_gaussian_schell_source_amp_bit_identical():
    # Exercises the intensity-envelope meshgrid site (amp = exp(-r^2/w0^2)).
    w0, sigma_g, nr, seed = 40.0e-6, 20.0e-6, 3, 777
    E_ens, dx, dy, lam = la.create_gaussian_schell_source(
        N=_N, dx=_DX, wavelength=_LAM, w0=w0, sigma_g=sigma_g,
        n_realizations=nr, dy=_DY, seed=seed, return_kind='ensemble')
    x, y = _axes(_N, _DX, _DY)
    X, Y = np.meshgrid(x, y)
    amp = np.exp(-(X * X + Y * Y) / (w0 ** 2))
    phi = _schell_phase_realizations(
        N=_N, Ny=_N, Nx=_N, dx=_DX, dy=_DY,
        coherence_length=sigma_g, n_realizations=nr,
        rng=np.random.default_rng(seed))
    ref = (amp[None, :, :] * phi).astype(_resolve_complex_dtype(None))
    assert np.array_equal(E_ens, ref)
