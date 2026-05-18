"""
Freeform surface types beyond standard conic + asphere.

Adds XY polynomial surfaces, Q-type orthogonal polynomials, and
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
                               xy_coeffs=None,
                               norm_x=1.0, norm_y=1.0):
    """XY polynomial freeform surface sag.

    ``z(x, y) = base_conic_sag(r) + sum_{i,j} c_{ij} x^i y^j``  for
    ``|x| <= norm_x`` and ``|y| <= norm_y``; outside this box the
    freeform departure is zeroed.

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
    norm_x, norm_y : float, default 1.0
        Half-extents [m] of the rectangular domain over which the
        polynomial is valid.  Outside ``|x| <= norm_x``,
        ``|y| <= norm_y`` the freeform departure is zeroed (matching
        the Chebyshev branch's out-of-domain guard), so the raytracer
        does not see a discontinuous step at the aperture rim from
        high-order terms.  Defaults to a 1-m unit box, which keeps
        the polynomial evaluation in raw metres (identical to the
        pre-v4.14.2 behaviour) while still zeroing pathological
        out-of-domain pixels.  Set to the physical clear-aperture
        half-extent for a tighter guard.

    Returns
    -------
    sag : ndarray
    """
    # v4.14.3 (P1-NEW-11): reject non-positive ``norm_x`` / ``norm_y``.
    # A negative value (e.g. typo ``norm_x = -0.05`` on a 50 mm half-
    # aperture) makes ``outside = abs(X) > -0.05`` true at every pixel
    # and the freeform contribution becomes invisible -- a silent
    # data-loss bug.  Validate at function entry so the typo surfaces
    # at the call site, not as a baffling "freeform did nothing"
    # downstream.
    if not (np.isfinite(norm_x) and norm_x > 0):
        raise ValueError(
            f"surface_sag_xy_polynomial: norm_x must be a positive "
            f"finite half-extent [m]; got norm_x={norm_x!r}.  A "
            f"negative or zero value would make the out-of-domain "
            f"mask true everywhere and zero the freeform "
            f"contribution silently.")
    if not (np.isfinite(norm_y) and norm_y > 0):
        raise ValueError(
            f"surface_sag_xy_polynomial: norm_y must be a positive "
            f"finite half-extent [m]; got norm_y={norm_y!r}.")
    h_sq = X ** 2 + Y ** 2
    sag = surface_sag_general(h_sq, R, conic)
    if xy_coeffs:
        # v4.14.2 (P1-NEW-5): zero the polynomial departure outside
        # the (norm_x, norm_y) domain, matching the Chebyshev branch's
        # ``np.where(outside, 0.0, departure)`` pattern below.  Without
        # this guard a high-order coefficient ``(2, 0): 1e3`` on a
        # 50 mm half-grid produces a 2.5 m corner sag applied to
        # pixels outside the physical aperture, and downstream
        # raytracing sees a discontinuity at the rim.  The polynomial
        # is still evaluated in raw (X, Y) so the coefficient
        # semantics are unchanged from pre-v4.14.2; ``norm_x``/
        # ``norm_y`` only define the rectangular clip region.
        outside = (np.abs(X) > norm_x) | (np.abs(Y) > norm_y)
        departure = np.zeros_like(sag)
        for (i, j), c in xy_coeffs.items():
            departure = departure + c * (X ** i) * (Y ** j)
        sag = sag + np.where(outside, 0.0, departure)
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
    from ..analysis import zernike_polynomial, zernike_index_to_nm

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
    # v4.14.3 (P1-NEW-11): reject non-positive ``norm_x`` / ``norm_y``.
    # ``xn_raw = X / norm_x`` with ``norm_x < 0`` flips the polynomial
    # domain sign (so ``T_n(x/norm_x)`` evaluates on the mirror-imaged
    # input and the resulting sag silently has wrong parity for odd
    # n).  Validate at entry to match the ``surface_sag_xy_polynomial``
    # guard and surface typos at the callsite.
    if not (np.isfinite(norm_x) and norm_x > 0):
        raise ValueError(
            f"surface_sag_chebyshev: norm_x must be a positive "
            f"finite half-extent [m]; got norm_x={norm_x!r}.  A "
            f"negative or zero value would mirror-image the "
            f"polynomial domain and silently flip the sign of "
            f"odd-order Chebyshev terms.")
    if not (np.isfinite(norm_y) and norm_y > 0):
        raise ValueError(
            f"surface_sag_chebyshev: norm_y must be a positive "
            f"finite half-extent [m]; got norm_y={norm_y!r}.")
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
        # v4.13.0 (Tier-2 perf, audit group alpha): hoist arccos out
        # of the per-coefficient loop -- it depends only on the grid,
        # not on (i, j) -- and additionally cache T_i(xn) / T_j(yn) by
        # polynomial order.  At typical freeform coefficient counts
        # (8-32 terms) many (i, j) pairs share an i or j, so the cos
        # evaluations get reused across the dictionary.
        theta_x = np.arccos(xn)
        theta_y = np.arccos(yn)
        unique_i = {i for (i, _) in cheb_coeffs.keys()}
        unique_j = {j for (_, j) in cheb_coeffs.keys()}
        Ti_cache = {i: np.cos(i * theta_x) for i in unique_i}
        Tj_cache = {j: np.cos(j * theta_y) for j in unique_j}
        departure = np.zeros_like(sag)
        for (i, j), c in cheb_coeffs.items():
            departure = departure + c * Ti_cache[i] * Tj_cache[j]
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
            xy_coeffs=surface_dict.get('xy_coeffs'),
            norm_x=surface_dict.get('norm_x', 1.0),
            norm_y=surface_dict.get('norm_y', 1.0))
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
