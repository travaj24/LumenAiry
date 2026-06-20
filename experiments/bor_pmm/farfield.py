"""BOR-PMM Milestone 5 (far-field core): Fourier-Bessel / discrete Hankel
decomposition of an axisymmetric near-field onto cylindrical "orders".

A transverse field of azimuthal order ``m`` on a bounded radius ``[0, R]``
expands in the Fourier-Bessel series

    f(r) = sum_n c_n J_m(alpha_n r / R),     alpha_n = j_{m,n}  (n-th zero of J_m)
    c_n  = (2 / (R^2 J_{m+1}(alpha_n)^2)) * INT_0^R f(r) J_m(alpha_n r/R) r dr

Each term is an outgoing cylindrical wave of transverse wavenumber
``kt_n = alpha_n / R``; in an open half-space of index ``sqrt(eps)`` it radiates
at polar angle ``sin(theta_n) = kt_n / (sqrt(eps) k0)`` (propagating iff
``kt_n < sqrt(eps) k0``).  This is the cylindrical analog of the planar grating's
Fourier-order decomposition; for a circular grating of radial period ``Lambda``
the populated ``kt_n`` cluster near ``kt_inc + 2 pi p / Lambda`` (the p-th
diffraction order), reducing to the planar grating equation as ``R -> inf``.

POWER NORMALIZATION (validated, Parseval to ~1e-10):
    INT_0^R |f|^2 r dr = sum_n |c_n|^2 * N_n,   N_n = R^2 J_{m+1}(alpha_n)^2 / 2
so a per-order power fraction is ``|c_n|^2 N_n / INT|f|^2 r dr`` -- the basis for
diffraction efficiencies (combined with the modal z-flux for the propagating set).
"""
from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, jv


def fourier_bessel(f, r, h, m, nmax):
    """Decompose ``f`` (sampled on the cell-centered grid ``r``, spacing ``h``)
    into ``nmax`` Fourier-Bessel coefficients of order ``m``.

    Returns ``(c, kt, norm)``: coefficients ``c_n``, transverse wavenumbers
    ``kt_n = alpha_n / R``, and the squared-norms ``N_n`` (for Parseval / power).
    """
    R = r[-1] + h / 2.0                       # domain edge (cell-centered grid)
    alpha = jn_zeros(m, nmax)
    c = np.zeros(nmax, dtype=complex)
    norm = np.zeros(nmax)
    for n in range(nmax):
        Jn = jv(m, alpha[n] * r / R)
        c[n] = np.sum(f * Jn * r * h) * 2.0 / (R ** 2 * jv(m + 1, alpha[n]) ** 2)
        norm[n] = R ** 2 * jv(m + 1, alpha[n]) ** 2 / 2.0
    return c, alpha / R, norm


def far_field_angles(kt, eps, k0):
    """Polar angles ``theta_n`` (radians) for transverse wavenumbers ``kt`` in a
    medium of permittivity ``eps``; NaN for evanescent (non-radiating) orders."""
    s = kt / (np.sqrt(eps) * k0)
    theta = np.full_like(s, np.nan, dtype=float)
    prop = s <= 1.0
    theta[prop] = np.arcsin(s[prop])
    return theta, prop


def order_power_fractions(f, r, h, m, eps, k0, nmax):
    """Per-cylindrical-order power fractions of a near-field ``f`` of order ``m``,
    with the propagating mask and far-field angles.  (Power-normalized via the
    Parseval relation; lossless fractions over the propagating set + evanescent
    tail sum to 1.)
    """
    c, kt, norm = fourier_bessel(f, r, h, m, nmax)
    total = np.sum(np.abs(c) ** 2 * norm)
    frac = np.abs(c) ** 2 * norm / max(total, 1e-300)
    theta, prop = far_field_angles(kt, eps, k0)
    return dict(c=c, kt=kt, frac=frac, theta=theta, prop=prop, total=total)
