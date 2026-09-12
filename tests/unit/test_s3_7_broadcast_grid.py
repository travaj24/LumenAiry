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


def _schell_phi_meshgrid_ref(*, Ny, Nx, dx, dy, sigma_g, nr, seed,
                             Ny_p=None, Nx_p=None):
    """Independent meshgrid reconstruction of _schell_phase_realizations,
    mirroring the KX/KY grid site (sources/core.py) exactly.

    ``Ny_p`` / ``Nx_p`` are the PADDED dimensions the generator filters on
    (v5.46, audit Z2: an FFT filter is a circular convolution, so the noise
    is drawn and filtered on a grid >= 4 sigma_g larger per side and the
    central window cropped).  Omit them for the ``pad_sigma=0.0`` path, which
    filters on the bare grid.  The padded SIZE is not what this file is
    pinning -- the KX/KY ORIENTATION is -- so the reference takes it as given
    and rebuilds the frequency grid with a dense ``np.meshgrid``.
    """
    Ny_p = Ny if Ny_p is None else Ny_p
    Nx_p = Nx if Nx_p is None else Nx_p
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx_p, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny_p, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    spec_filter = np.exp(-(KX * KX + KY * KY) * (sigma_g ** 2) / 4.0)
    mean_I = float(np.sum(np.abs(spec_filter) ** 2) / (Ny_p * Nx_p))
    norm = np.sqrt(mean_I) if mean_I > 0.0 else 1.0
    off_y, off_x = (Ny_p - Ny) // 2, (Nx_p - Nx) // 2
    out = np.empty((nr, Ny, Nx), dtype=np.complex128)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    rng = np.random.default_rng(seed)
    for k in range(nr):
        w_re = rng.standard_normal((Ny_p, Nx_p))
        w_im = rng.standard_normal((Ny_p, Nx_p))
        W = (w_re + 1j * w_im) * inv_sqrt2
        phi = np.fft.ifft2(np.fft.fft2(W) * spec_filter) / norm
        out[k] = phi[off_y:off_y + Ny, off_x:off_x + Nx]
    return out


def test_schell_phase_realizations_kxky_bit_identical_unpadded():
    """``pad_sigma=0.0`` -- the bare-grid path (v5.46 escape hatch, and the
    <= v5.45 default).  Rectangular grid + dx != dy: the KX/KY orientation is
    load-bearing."""
    Ny, Nx, sigma_g, nr, seed = 6, 10, 20.0e-6, 3, 12345
    phi = _schell_phase_realizations(
        Ny=Ny, Nx=Nx, dx=_DX, dy=_DY,
        coherence_length=sigma_g, n_realizations=nr,
        rng=np.random.default_rng(seed), pad_sigma=0.0)
    ref = _schell_phi_meshgrid_ref(
        Ny=Ny, Nx=Nx, dx=_DX, dy=_DY, sigma_g=sigma_g, nr=nr, seed=seed)
    assert np.array_equal(phi, ref)


def test_schell_phase_realizations_kxky_bit_identical_padded():
    """The DEFAULT (anti-wrap padded) path, same orientation claim.

    v5.46 (audit Z2): the generator now draws and filters the noise on a grid
    padded by >= 4 sigma_g per side and crops the centre, so the reference
    has to be rebuilt on the padded frequency grid -- but the property this
    file exists to pin is unchanged: ``KX, KY = kx[None, :], ky[:, None]``
    must reproduce ``np.meshgrid(kx, ky)`` bit-for-bit on a grid where
    ``Ny != Nx`` and ``dx != dy``, so a transposed-axis regression fails
    rather than slipping through.

    ``sigma_g`` is small enough here that the pad is not capped, so the
    padded dimensions are ``next_fast_len(N + 2*ceil(4 sigma_g / d))``.
    """
    from scipy.fft import next_fast_len
    Ny, Nx, sigma_g, nr, seed = 6, 10, 4.0e-6, 3, 12345
    pad_y = int(np.ceil(4.0 * sigma_g / _DY))
    pad_x = int(np.ceil(4.0 * sigma_g / _DX))
    Ny_p, Nx_p = int(next_fast_len(Ny + 2 * pad_y)), \
        int(next_fast_len(Nx + 2 * pad_x))
    phi = _schell_phase_realizations(
        Ny=Ny, Nx=Nx, dx=_DX, dy=_DY,
        coherence_length=sigma_g, n_realizations=nr,
        rng=np.random.default_rng(seed))
    ref = _schell_phi_meshgrid_ref(
        Ny=Ny, Nx=Nx, dx=_DX, dy=_DY, sigma_g=sigma_g, nr=nr, seed=seed,
        Ny_p=Ny_p, Nx_p=Nx_p)
    assert np.array_equal(phi, ref)


def test_gaussian_beam_broadcast_bit_identical():
    """v5.46 (audit Z3): ``create_gaussian_beam`` was the LAST factory still
    building a dense ``np.meshgrid`` (peak/output 3.00x at complex128, 5.00x
    at complex64).  Same independent-oracle shape as its siblings above:
    rebuild from a full meshgrid and the documented closed form
    ``exp(-r^2 / w0^2)``, asserting bit-identity.  ``x0 != y0`` and
    ``dx != dy`` so a transposed-axis regression is observable, and the
    ``normalize='peak'`` divide (now in place) is exercised."""
    w0 = 17.0e-6
    E, x, y = la.create_gaussian_beam(
        _N, _DX, _LAM, w0=w0, x0=_X0, y0=_Y0, dy=_DY, normalize='peak')
    X, Y = np.meshgrid(*_axes(_N, _DX, _DY))
    sigma = w0 / np.sqrt(2.0)
    ref = np.exp(-((X - _X0) ** 2 + (Y - _Y0) ** 2) / (2 * sigma ** 2))
    ref = ref.astype(_resolve_complex_dtype(None))
    mx = float(np.abs(ref).max())
    if mx > 0:
        ref = ref / mx
    assert np.array_equal(E, ref)
    assert np.array_equal(x, _axes(_N, _DX, _DY)[0])
    assert np.array_equal(y, _axes(_N, _DX, _DY)[1])


def test_gaussian_schell_source_amp_bit_identical():
    # Exercises the intensity-envelope meshgrid site (amp = exp(-r^2/w0^2)).
    w0, sigma_g, nr, seed = 40.0e-6, 20.0e-6, 3, 777
    E_ens, dx, dy, lam = la.create_gaussian_schell_source(
        N=_N, dx=_DX, wavelength=_LAM, w0=w0, sigma_g=sigma_g,
        n_realizations=nr, dy=_DY, rng=seed, return_kind='ensemble')
    x, y = _axes(_N, _DX, _DY)
    X, Y = np.meshgrid(x, y)
    amp = np.exp(-(X * X + Y * Y) / (w0 ** 2))
    phi = _schell_phase_realizations(
        Ny=_N, Nx=_N, dx=_DX, dy=_DY,
        coherence_length=sigma_g, n_realizations=nr,
        rng=np.random.default_rng(seed))
    ref = (amp[None, :, :] * phi).astype(_resolve_complex_dtype(None))
    assert np.array_equal(E_ens, ref)
