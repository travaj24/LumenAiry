"""
Freeform surface types beyond standard conic + asphere.

Adds XY polynomial surfaces, Q-type orthogonal polynomials, Chebyshev
surfaces, and Forbes Q-bfs / Q-con polynomial bases for modern
freeform optics design.

These integrate with the existing prescription dict format via
``surface_sag_freeform(X, Y, surface_dict)`` which checks for the
``'freeform_type'`` key.

References
----------
* G. W. Forbes, "Shape specification for axially symmetric optical
  surfaces," Opt. Express 15(8), 5218-5226 (2007).  See Eq. 13 for the
  Q-bfs (best-fit-sphere-subtracted) polynomial expansion.
* G. W. Forbes, "Robust, efficient computational methods for axially
  symmetric optical aspheres," Opt. Express 18(13), 13851-13862
  (2010).  See Section 3 for the explicit 3-term recurrences for
  both Q-bfs (Sec. 3.2) and Q-con (Sec. 3.1) orthonormal bases.

Author: Andrew Traverso
"""
from __future__ import annotations

import numpy as np

# v5.2 (ROADMAP "Duplicate `_xp_of`" cleanup): consolidated wrapper.
from ..backend import array_namespace as _xp_of  # noqa: E402
from .lenses import surface_sag_biconic, surface_sag_general

# ============================================================================
# Forbes Q-type orthonormal polynomial evaluators (v4.15)
# ============================================================================
#
# Both Q-bfs (best-fit-sphere subtracted, Forbes 2007 Eq. 13) and Q-con
# (conic subtracted, Forbes 2010 Eq. 6) bases are orthonormal polynomial
# families in ``x = u^2`` over [0, 1], differing only by their
# orthogonality weight:
#
#   Q-bfs weight:  ``x (1 - x) dx``   -> shifted Jacobi P_m^(1, 1)(2x - 1)
#   Q-con weight:  ``x^2 dx``         -> shifted Jacobi P_m^(0, 2)(2x - 1)
#
# The orthonormalisation convention used by Forbes (and by this
# library) is::
#
#   int_0^1 Q_m(x) Q_n(x) w(x) dx = delta_{mn}
#
# Implementation uses the standard Jacobi 3-term recurrence
# (Abramowitz & Stegun 22.7.1 / Forbes 2010 Sec. 3) on the shifted
# coordinate ``t = 2x - 1`` and applies the closed-form L2 norm to
# convert the unnormalised Jacobi P_m^(alpha, beta) into the
# orthonormal Q_m.  The closed-form on [0, 1] is obtained from the
# standard A&S 22.2.1 norm on [-1, 1] divided by the Jacobian factor
# ``2^(alpha + beta + 1)`` that maps the weight to [0, 1]:
#
#   h_n_[0, 1] = G(n+alpha+1) G(n+beta+1)
#                / ((2n+alpha+beta+1) n! G(n+alpha+beta+1))
#
# so the orthonormalisation multiplier is ``c_n = 1 / sqrt(h_n_[0, 1])``.
# Specialisations:
#
# * Q-bfs (alpha = beta = 1): ``h_n_[0, 1] = (n+1) / ((2n+3)(n+2))``,
#   giving ``c_n = sqrt((2n+3)(n+2) / (n+1))`` -- note the
#   ``(n+1)^1`` denominator (no spurious factor of 8; no power of 2
#   on ``(n+1)``).
# * Q-con (alpha = 0, beta = 2): ``h_n_[0, 1] = 1 / (2n+3)``,
#   giving ``c_n = sqrt(2n+3)``.
#
# v4.15.1 (P3 doc-formula fix): pre-v4.15.1 versions of this comment
# block and the CHANGELOG stated the Q-bfs multiplier as
# ``sqrt((2n+3)(n+2) / (n+1)^2)`` (or equivalently the norm as
# ``sqrt(8(n+2) / ((2n+3)(n+1)^2))``).  Both forms are wrong on the
# ``(n+1)`` power and carry a spurious factor of 8 inherited from
# the unmapped [-1, 1] norm.  The implementation
# (``_jacobi_norm_factor`` below) is correct -- only the prose lied;
# the numerical orthonormality check in
# ``test_v4_15_agent_d.TestForbesQbfsOrthonormality`` and the new
# ``test_orthonormalizer_docstring_matches_code`` regression pin
# both verify the implementation, leaving this docstring block as
# the single source of truth for the closed-form formula.


def _jacobi_norm_factor(n, alpha, beta):
    """Return the orthonormalisation multiplier ``c_n`` such that
    ``Q_n(x) = c_n * P_n^(alpha, beta)(2x - 1)`` is orthonormal on
    [0, 1] with weight ``(1 - x)^alpha x^beta dx``.

    Derived from the standard Jacobi norm
    ``h_n = 2^(alpha+beta+1) / (2n+alpha+beta+1)
            * G(n+alpha+1) G(n+beta+1) / (n! G(n+alpha+beta+1))``
    on [-1, 1] (A&S 22.2.1), mapped to [0, 1] via ``t = 2x - 1`` which
    introduces a factor of ``2^(alpha+beta+1)`` in the denominator
    (weight transforms as ``(1-t)^alpha (1+t)^beta dt
    = 2^(alpha+beta+1) (1-x)^alpha x^beta dx``), yielding
    ``int_0^1 P_n^2 (1-x)^alpha x^beta dx = h_n / 2^(alpha+beta+1)``.
    """
    from math import exp, lgamma, log
    # h_n on [-1, 1]:
    log_h_n = ((alpha + beta + 1) * log(2.0)
               - log(2 * n + alpha + beta + 1)
               + lgamma(n + alpha + 1) + lgamma(n + beta + 1)
               - lgamma(n + 1) - lgamma(n + alpha + beta + 1))
    # Mapped to [0, 1] (weight picks up a factor 2^(alpha+beta+1)):
    log_norm_sq = log_h_n - (alpha + beta + 1) * log(2.0)
    # c_n = 1 / sqrt(int_0^1 P_n^2 w dx)
    return exp(-0.5 * log_norm_sq)


def _shifted_jacobi_eval(x, n, alpha, beta):
    """Evaluate the shifted Jacobi polynomial ``P_n^(alpha, beta)(2x - 1)``
    elementwise on an array ``x``.

    Uses the standard 3-term recurrence (A&S 22.7.1, equivalently
    Forbes 2010 Sec. 3 specialised to ``alpha = 1, beta = 1`` for Q-bfs
    and ``alpha = 0, beta = 2`` for Q-con).  The recurrence is:

    ``2(n+1)(n+alpha+beta+1)(2n+alpha+beta) P_{n+1}(t)
        = (2n+alpha+beta+1) * [(2n+alpha+beta)(2n+alpha+beta+2) t
                              + alpha^2 - beta^2] P_n(t)
          - 2 (n+alpha) (n+beta) (2n+alpha+beta+2) P_{n-1}(t)``

    starting from ``P_0(t) = 1``, ``P_1(t) = (1/2) [(alpha - beta)
    + (alpha + beta + 2) t]``.

    Parameters
    ----------
    x : ndarray
        Argument in ``[0, 1]``; the polynomial is evaluated at the
        shifted coordinate ``t = 2 x - 1`` in ``[-1, 1]``.
    n : int
        Polynomial order (non-negative).
    alpha, beta : float
        Jacobi parameters.

    Returns
    -------
    P_n : ndarray
        Same shape as ``x``.
    """
    xp = _xp_of(x)
    t = 2.0 * x - 1.0
    if n == 0:
        return xp.ones_like(t)
    if n == 1:
        return 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * t)
    P_prev = xp.ones_like(t)
    P_curr = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * t)
    for k in range(1, n):
        a1 = 2.0 * (k + 1) * (k + alpha + beta + 1) * (2 * k + alpha + beta)
        a2 = (2 * k + alpha + beta + 1) * (
            (2 * k + alpha + beta) * (2 * k + alpha + beta + 2)
        )
        a3 = (2 * k + alpha + beta + 1) * (alpha * alpha - beta * beta)
        a4 = 2.0 * (k + alpha) * (k + beta) * (2 * k + alpha + beta + 2)
        P_next = ((a2 * t + a3) * P_curr - a4 * P_prev) / a1
        P_prev = P_curr
        P_curr = P_next
    return P_curr


def _q_bfs_eval(x, n):
    """Orthonormal Forbes Q-bfs polynomial of order ``n`` evaluated on
    ``x = u^2`` (in ``[0, 1]``).  Forbes 2007 Eq. 13 basis;
    orthonormalised on ``[0, 1]`` with weight ``x (1 - x) dx``."""
    c_n = _jacobi_norm_factor(n, 1, 1)
    return c_n * _shifted_jacobi_eval(x, n, 1, 1)


def _q_con_eval(x, n):
    """Orthonormal Forbes Q-con polynomial of order ``n`` evaluated on
    ``x = u^2`` (in ``[0, 1]``).  Forbes 2010 Eq. 6 basis;
    orthonormalised on ``[0, 1]`` with weight ``x^2 dx``."""
    c_n = _jacobi_norm_factor(n, 0, 2)
    return c_n * _shifted_jacobi_eval(x, n, 0, 2)


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
    from ..analysis import zernike_index_to_nm, zernike_polynomial

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


def surface_sag_q_bfs(X, Y, *, radius, coefficients, r_max,
                       dx, dy=None, norm_x=1.0, norm_y=1.0):
    """Forbes Q-bfs (best-fit-sphere subtracted) freeform surface sag.

    Implements Eq. 13 of Forbes 2007 ("Shape specification for axially
    symmetric optical surfaces", Opt. Express 15(8), 5218-5226):

    ``z(r) = z_bfs(r) + u^2 (1 - u^2) * sum_{m=0}^{M} a_m * Q_m^bfs(u^2)``

    where ``u = r / r_max`` is the normalised radial coordinate (in
    ``[0, 1]`` over the clear aperture), ``z_bfs(r) = c r^2 / (1 +
    sqrt(1 - c^2 r^2))`` is the best-fit sphere with curvature
    ``c = 1 / radius``, and ``Q_m^bfs`` are the Forbes Q-bfs orthonormal
    polynomials.  The orthonormality convention is::

        int_0^1 Q_m^bfs(u^2) Q_n^bfs(u^2) u^2 (1 - u^2) d(u^2)
            = delta_{mn}

    The 3-term recurrence used to evaluate ``Q_m^bfs`` is the shifted
    Jacobi recurrence with ``(alpha, beta) = (1, 1)`` per Forbes 2010
    Section 3.2 ("Robust, efficient computational methods for axially
    symmetric optical aspheres", Opt. Express 18(13), 13851-13862).

    Domain clipping (v4.15.1 P1-F1-1 alignment).  The Q-bfs basis is
    defined on the *radial* coordinate ``u = r / r_max`` over the unit
    disc, so the PRIMARY clip is radial: pixels with ``r > r_max``
    have their freeform departure zeroed.  The ``norm_x`` / ``norm_y``
    kwargs act as a SECONDARY rectangular pupil-clip; both clips
    apply, so a pixel must lie inside *both* ``r <= r_max`` and
    ``|x| <= norm_x AND |y| <= norm_y`` to receive a non-zero
    departure.  Pre-v4.15.1 used only the rectangular box, which let
    corner pixels at ``(0.9 r_max, 0.9 r_max)`` through despite
    having ``r ~ 1.27 r_max`` outside the Forbes domain.

    Parameters
    ----------
    X, Y : ndarray
        Surface-local coordinates [m].
    radius : float
        Best-fit-sphere radius of curvature [m].  Use ``float('inf')``
        or ``np.inf`` for a flat base.
    coefficients : sequence of float
        ``a_m`` coefficients in Eq. 13 above, in meters (same units as
        sag).  ``coefficients[m]`` corresponds to ``a_m``.  An empty
        list, ``None``, or all-zero coefficients yields a pure
        best-fit-sphere sag.
    r_max : float
        Normalisation radius for ``u = r / r_max`` [m].  Typically the
        clear-aperture semi-diameter.  Must be positive and finite.
        Acts as the primary radial-domain clip: pixels with
        ``r > r_max`` are zeroed.
    dx : float
        Grid spacing in x [m].  Must be positive and finite.  Retained
        in the signature for consistency with other ``surface_sag_*``
        APIs and to allow callers to pass the grid spacing alongside
        ``X, Y``; not used in the sag computation itself.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    norm_x, norm_y : float, default 1.0
        Secondary rectangular pupil-clip half-extents [m]; pixels
        outside ``|x| <= norm_x`` or ``|y| <= norm_y`` are zeroed in
        addition to the radial ``r > r_max`` clip.  The v4.14.3
        ``P1-NEW-11`` negative-half-extent guard applies.  Default
        ``1.0 m`` is a non-restrictive box that lets the radial clip
        dominate when the caller cares only about the Forbes domain.

    Returns
    -------
    sag : ndarray
        Surface sag [m], same shape as ``X``.
    """
    # P1-NEW-11 convention (v4.14.3): negative / zero / non-finite
    # half-extents make the out-of-domain mask true everywhere and
    # silently zero the freeform; reject at the call site.
    if not (np.isfinite(norm_x) and norm_x > 0):
        raise ValueError(
            f"surface_sag_q_bfs: norm_x must be a positive finite "
            f"half-extent [m]; got norm_x={norm_x!r}.")
    if not (np.isfinite(norm_y) and norm_y > 0):
        raise ValueError(
            f"surface_sag_q_bfs: norm_y must be a positive finite "
            f"half-extent [m]; got norm_y={norm_y!r}.")
    if not (np.isfinite(r_max) and r_max > 0):
        raise ValueError(
            f"surface_sag_q_bfs: r_max must be a positive finite "
            f"normalisation radius [m]; got r_max={r_max!r}.")
    if not (np.isfinite(dx) and dx > 0):
        raise ValueError(
            f"surface_sag_q_bfs: dx must be a positive finite grid "
            f"spacing [m]; got dx={dx!r}.")
    if dy is None:
        dy = dx
    if not (np.isfinite(dy) and dy > 0):
        raise ValueError(
            f"surface_sag_q_bfs: dy must be a positive finite grid "
            f"spacing [m]; got dy={dy!r}.")
    xp = _xp_of(X, Y)
    h_sq = X ** 2 + Y ** 2
    # z_bfs: spherical sag (conic = 0).  Use the established
    # surface_sag_general path so NaN/edge handling matches the rest
    # of the library.
    sag = surface_sag_general(h_sq, radius, 0.0)
    if coefficients is None or len(coefficients) == 0:
        return sag
    # Skip the polynomial evaluation if all coefficients vanish --
    # then sag == z_bfs to machine precision (used by the zero-
    # coefficient regression pin).
    coeffs = list(coefficients)
    if all(c == 0.0 for c in coeffs):
        return sag
    # Normalised radial coordinate u = r / r_max; arg to Q_m is u^2.
    u_sq = h_sq / (r_max * r_max)
    # Forbes 2007 Eq. 13: prefactor u^2 (1 - u^2) on the polynomial
    # series.  Outside u = 1 (i.e., r > r_max) the prefactor goes
    # negative, but the rectangular out-of-domain mask below will
    # zero the contribution there anyway.
    prefactor = u_sq * (1.0 - u_sq)
    departure = xp.zeros_like(sag)
    for m, a_m in enumerate(coeffs):
        if a_m == 0.0:
            continue
        Q_m = _q_bfs_eval(u_sq, m)
        departure = departure + a_m * Q_m
    departure = prefactor * departure
    # v4.15.1 P1-F1-1: PRIMARY clip is radial (Forbes domain
    # ``u <= 1`` <=> ``r <= r_max``), SECONDARY clip is the rectangular
    # ``[-norm_x, norm_x] x [-norm_y, norm_y]`` pupil box.  Both clips
    # apply -- a pixel must lie inside both to keep its freeform
    # departure.  Pre-v4.15.1 used only the rectangular box, which
    # let pixels at ``(0.9 r_max, 0.9 r_max)`` (radial 1.27 r_max,
    # outside the Forbes domain) through with the Q-bfs prefactor
    # going through its sign-flip at ``u^2 = 1``.
    outside = (u_sq > 1.0) | (xp.abs(X) > norm_x) | (xp.abs(Y) > norm_y)
    sag = sag + xp.where(outside, xp.zeros((), dtype=departure.dtype),
                         departure)
    return sag


def surface_sag_q_con(X, Y, *, radius, conic, coefficients, r_max,
                       dx, dy=None, norm_x=1.0, norm_y=1.0):
    """Forbes Q-con (conic-subtracted) freeform surface sag.

    Implements Eq. 6 of Forbes 2010 ("Robust, efficient computational
    methods for axially symmetric optical aspheres", Opt. Express
    18(13), 13851-13862):

    ``z(r) = z_conic(r) + u^4 * sum_{m=0}^{M} a_m * Q_m^con(u^2)``

    where ``u = r / r_max``, ``z_conic(r) = c r^2 / (1 +
    sqrt(1 - (1 + k) c^2 r^2))`` is the base conic with curvature
    ``c = 1 / radius`` and conic constant ``k = conic``, and
    ``Q_m^con`` are the Forbes Q-con orthonormal polynomials.  The
    orthonormality convention is::

        int_0^1 Q_m^con(u^2) Q_n^con(u^2) u^4 d(u^2) = delta_{mn}

    The 3-term recurrence used to evaluate ``Q_m^con`` is the shifted
    Jacobi recurrence with ``(alpha, beta) = (0, 2)`` per Forbes 2010
    Section 3.1.

    Domain clipping (v4.15.1 P1-F1-1 alignment).  As for ``surface_sag_q_bfs``,
    the PRIMARY clip is radial (``r <= r_max``) because the Q-con
    basis is defined on the unit disc; the SECONDARY rectangular
    ``[-norm_x, norm_x] x [-norm_y, norm_y]`` pupil box also applies.
    Pre-v4.15.1 used only the rectangular box.  The v4.14.3
    ``P1-NEW-11`` negative-half-extent guard applies.

    Parameters
    ----------
    X, Y : ndarray
        Surface-local coordinates [m].
    radius : float
        Base conic radius of curvature [m].
    conic : float
        Base conic constant ``k``.
    coefficients : sequence of float
        ``a_m`` coefficients in Eq. 6 above, in meters.
        ``coefficients[m]`` corresponds to ``a_m``.
    r_max : float
        Normalisation radius for ``u = r / r_max`` [m].  Acts as the
        primary radial-domain clip: pixels with ``r > r_max`` are
        zeroed.
    dx : float
        Grid spacing in x [m].  Retained for API consistency.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    norm_x, norm_y : float, default 1.0
        Secondary rectangular pupil-clip half-extents [m]; pixels
        outside the box are zeroed in addition to the radial clip.

    Returns
    -------
    sag : ndarray
        Surface sag [m], same shape as ``X``.
    """
    if not (np.isfinite(norm_x) and norm_x > 0):
        raise ValueError(
            f"surface_sag_q_con: norm_x must be a positive finite "
            f"half-extent [m]; got norm_x={norm_x!r}.")
    if not (np.isfinite(norm_y) and norm_y > 0):
        raise ValueError(
            f"surface_sag_q_con: norm_y must be a positive finite "
            f"half-extent [m]; got norm_y={norm_y!r}.")
    if not (np.isfinite(r_max) and r_max > 0):
        raise ValueError(
            f"surface_sag_q_con: r_max must be a positive finite "
            f"normalisation radius [m]; got r_max={r_max!r}.")
    if not (np.isfinite(dx) and dx > 0):
        raise ValueError(
            f"surface_sag_q_con: dx must be a positive finite grid "
            f"spacing [m]; got dx={dx!r}.")
    if dy is None:
        dy = dx
    if not (np.isfinite(dy) and dy > 0):
        raise ValueError(
            f"surface_sag_q_con: dy must be a positive finite grid "
            f"spacing [m]; got dy={dy!r}.")
    xp = _xp_of(X, Y)
    h_sq = X ** 2 + Y ** 2
    # Base conic sag (with non-zero conic constant unlike Q-bfs):
    sag = surface_sag_general(h_sq, radius, conic)
    if coefficients is None or len(coefficients) == 0:
        return sag
    coeffs = list(coefficients)
    if all(c == 0.0 for c in coeffs):
        return sag
    u_sq = h_sq / (r_max * r_max)
    # Forbes 2010 Eq. 6: prefactor u^4 = (u^2)^2 on the polynomial
    # series.
    prefactor = u_sq * u_sq
    departure = xp.zeros_like(sag)
    for m, a_m in enumerate(coeffs):
        if a_m == 0.0:
            continue
        Q_m = _q_con_eval(u_sq, m)
        departure = departure + a_m * Q_m
    departure = prefactor * departure
    # v4.15.1 P1-F1-1: PRIMARY radial clip + SECONDARY rectangular box;
    # see surface_sag_q_bfs above for the rationale.
    outside = (u_sq > 1.0) | (xp.abs(X) > norm_x) | (xp.abs(Y) > norm_y)
    sag = sag + xp.where(outside, xp.zeros((), dtype=departure.dtype),
                         departure)
    return sag


def surface_sag_freeform(X, Y, surface_dict):
    """Dispatch to the correct freeform sag function based on the
    ``freeform_type`` key in the surface dict.

    Supports: ``'xy_polynomial'``, ``'zernike'``, ``'chebyshev'``,
    ``'q_bfs'`` (v4.15+), ``'q_con'`` (v4.15+).  Falls back to
    standard sag if ``freeform_type`` is absent.

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
    elif ft == 'q_bfs':
        # v4.15: Forbes Q-bfs polynomial freeform.  dx/dy are
        # required-by-signature; default to the grid stride implied
        # by X when not specified in the surface dict.
        _dx = surface_dict.get('dx')
        if _dx is None:
            # Best-effort fallback: derive from the X column stride.
            _dx = float(abs(X[0, 1] - X[0, 0])) if X.ndim == 2 \
                and X.shape[1] > 1 else 1.0
        # v4.15.1 P1-F1-2: require ``r_max`` explicitly for Q-bfs /
        # Q-con.  Pre-v4.15.1 silently defaulted to ``r_max=1.0``,
        # which for a user passing X/Y in metres meant the Forbes
        # domain shrank to a 1 m disc -- typically far larger than
        # the actual aperture, so the polynomial got evaluated on a
        # sub-pixel of the surface and the freeform departure was
        # effectively a no-op.  Raise so the unit-mismatch surfaces
        # at the call site rather than as silently-wrong sag.
        if 'r_max' not in surface_dict:
            raise TypeError(
                "surface_sag_freeform: 'r_max' is required when "
                "freeform_type='q_bfs' (the Forbes Q-bfs basis is "
                "defined on the radial coordinate u = r / r_max; "
                "pre-v4.15.1 silently defaulted to r_max=1.0 which "
                "caused a units mismatch when X/Y were passed in "
                "metres).  Pass r_max=<clear-aperture-semi-diameter> "
                "in the same units as X/Y.")
        return surface_sag_q_bfs(
            X, Y, radius=R,
            coefficients=surface_dict.get('q_bfs_coeffs', []),
            r_max=surface_dict['r_max'],
            dx=_dx,
            dy=surface_dict.get('dy'),
            norm_x=surface_dict.get('norm_x', 1.0),
            norm_y=surface_dict.get('norm_y', 1.0))
    elif ft == 'q_con':
        # v4.15: Forbes Q-con polynomial freeform.
        _dx = surface_dict.get('dx')
        if _dx is None:
            _dx = float(abs(X[0, 1] - X[0, 0])) if X.ndim == 2 \
                and X.shape[1] > 1 else 1.0
        # v4.15.1 P1-F1-2: see q_bfs branch above for rationale.
        if 'r_max' not in surface_dict:
            raise TypeError(
                "surface_sag_freeform: 'r_max' is required when "
                "freeform_type='q_con' (the Forbes Q-con basis is "
                "defined on the radial coordinate u = r / r_max; "
                "pre-v4.15.1 silently defaulted to r_max=1.0 which "
                "caused a units mismatch when X/Y were passed in "
                "metres).  Pass r_max=<clear-aperture-semi-diameter> "
                "in the same units as X/Y.")
        return surface_sag_q_con(
            X, Y, radius=R, conic=kc,
            coefficients=surface_dict.get('q_con_coeffs', []),
            r_max=surface_dict['r_max'],
            dx=_dx,
            dy=surface_dict.get('dy'),
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
