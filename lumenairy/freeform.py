"""
Freeform surface types beyond standard conic + asphere.

Adds XY polynomial surfaces, Q-type (Forbes) polynomials, and
Chebyshev surfaces for modern freeform optics design.

These integrate with the existing prescription dict format via
``surface_sag_freeform(X, Y, surface_dict)`` which checks for the
``'freeform_type'`` key.

Author: Andrew Traverso
"""
from __future__ import annotations
import numpy as np
from .lenses import surface_sag_general, surface_sag_biconic


def surface_sag_xy_polynomial(X, Y, R=np.inf, conic=0.0,
                               xy_coeffs=None):
    """XY polynomial freeform surface sag.

    ``z(x, y) = base_conic_sag(r) + sum_{i,j} c_{ij} x^i y^j``

    Parameters
    ----------
    X, Y : ndarray
        Surface-local coordinates [m].
    R : float
        Base radius of curvature [m] (inf = flat base).
    conic : float
        Base conic constant.
    xy_coeffs : dict of {(i, j): coefficient}
        XY polynomial terms.  Keys are ``(power_x, power_y)`` tuples;
        values are coefficients in meters (same units as sag).
        Example: ``{(2, 0): 1e-6, (0, 2): -1e-6}`` adds an
        astigmatic departure.

    Returns
    -------
    sag : ndarray
    """
    h_sq = X ** 2 + Y ** 2
    sag = surface_sag_general(h_sq, R, conic)
    if xy_coeffs:
        for (i, j), c in xy_coeffs.items():
            sag = sag + c * (X ** i) * (Y ** j)
    return sag


def surface_sag_zernike_freeform(X, Y, R=np.inf, conic=0.0,
                                  zernike_coeffs=None, norm_radius=1.0):
    """Zernike-polynomial freeform surface sag.

    ``z(x, y) = base_sag(r) + sum_j c_j Z_j(rho, theta)``

    where Z_j are OSA-normalised Zernike polynomials over the pupil
    of radius ``norm_radius``.

    Parameters
    ----------
    X, Y : ndarray
    R, conic : float
        Base surface parameters.
    zernike_coeffs : dict of {j: coefficient_m}
        OSA Zernike index j -> sag departure coefficient [m].
    norm_radius : float
        Normalisation radius for the Zernike polynomials [m].

    Returns
    -------
    sag : ndarray
    """
    from .analysis import zernike_polynomial, zernike_index_to_nm

    h_sq = X ** 2 + Y ** 2
    sag = surface_sag_general(h_sq, R, conic)
    if zernike_coeffs:
        rho = np.sqrt(h_sq) / norm_radius
        theta = np.arctan2(Y, X)
        for j, c in zernike_coeffs.items():
            n, m = zernike_index_to_nm(j)
            sag = sag + c * zernike_polynomial(n, m, rho, theta)
    return sag


def surface_sag_chebyshev(X, Y, R=np.inf, conic=0.0,
                           cheb_coeffs=None, norm_x=1.0, norm_y=1.0):
    """Chebyshev polynomial freeform surface sag.

    ``z(x, y) = base_sag(r) + sum_{i,j} c_{ij} T_i(x/a) T_j(y/b)``

    where T_n is the Chebyshev polynomial of the first kind.

    Parameters
    ----------
    X, Y : ndarray
    R, conic : float
    cheb_coeffs : dict of {(i, j): coefficient_m}
    norm_x, norm_y : float
        Normalisation half-extents [m].

    Returns
    -------
    sag : ndarray
    """
    h_sq = X ** 2 + Y ** 2
    sag = surface_sag_general(h_sq, R, conic)
    if cheb_coeffs:
        # Chebyshev polynomials T_n are only defined on [-1, 1].  We
        # clip the argument so arccos doesn't NaN, but we ALSO zero
        # the freeform contribution outside the normalisation box so
        # the sag doesn't jump to the boundary value (T_n(+-1)) for
        # out-of-domain pixels.  Without this guard the ray tracer
        # sees a large step discontinuity at the domain edge.
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


def surface_sag_freeform(X, Y, surface_dict):
    """Dispatch to the correct freeform sag function based on the
    ``freeform_type`` key in the surface dict.

    Supports: ``'xy_polynomial'``, ``'zernike'``, ``'chebyshev'``.
    Falls back to standard sag if ``freeform_type`` is absent.

    Parameters
    ----------
    X, Y : ndarray
    surface_dict : dict
        Must contain ``'radius'``.  May contain ``'freeform_type'``
        and associated coefficient keys.

    Returns
    -------
    sag : ndarray
    """
    ft = surface_dict.get('freeform_type')
    R = surface_dict.get('radius', np.inf)
    kc = surface_dict.get('conic', 0.0)

    if ft == 'xy_polynomial':
        return surface_sag_xy_polynomial(
            X, Y, R=R, conic=kc,
            xy_coeffs=surface_dict.get('xy_coeffs'))
    elif ft == 'zernike':
        return surface_sag_zernike_freeform(
            X, Y, R=R, conic=kc,
            zernike_coeffs=surface_dict.get('zernike_coeffs'),
            norm_radius=surface_dict.get('norm_radius', 1.0))
    elif ft == 'chebyshev':
        return surface_sag_chebyshev(
            X, Y, R=R, conic=kc,
            cheb_coeffs=surface_dict.get('cheb_coeffs'),
            norm_x=surface_dict.get('norm_x', 1.0),
            norm_y=surface_dict.get('norm_y', 1.0))
    else:
        # Standard rotationally-symmetric or biconic
        R_y = surface_dict.get('radius_y')
        if R_y is not None:
            return surface_sag_biconic(
                X, Y, R_x=R, R_y=R_y,
                conic_x=kc,
                conic_y=surface_dict.get('conic_y'),
                aspheric_coeffs=surface_dict.get('aspheric_coeffs'),
                aspheric_coeffs_y=surface_dict.get('aspheric_coeffs_y'))
        return surface_sag_general(
            X ** 2 + Y ** 2, R, kc,
            surface_dict.get('aspheric_coeffs'))
