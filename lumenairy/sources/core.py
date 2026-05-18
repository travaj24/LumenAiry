"""
Beam source and mode generators for optical propagation simulations.

This module provides functions to create common laser beam profiles on a
discrete 2-D grid:

- Fundamental Gaussian beams (with optional GPU acceleration via CuPy)
- Hermite-Gaussian (HG_mn) modes
- Laguerre-Gaussian (LG_pl) modes

All fields are returned at the beam waist (flat phase) and are suitable for
use as input to angular-spectrum or other propagation routines.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import importlib.util as _importlib_util
CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None
cp = None  # populated lazily on first use


def _ensure_cupy_loaded():
    global cp
    if cp is None and CUPY_AVAILABLE:
        import cupy as _c
        cp = _c
    return cp is not None


def _validate_grid_params(
    N: Union[int, Tuple[int, int]],
    dx: float,
    wavelength: float,
    *,
    dy: Optional[float] = None,
    fn_name: str = 'create_*',
) -> None:
    """Validate grid-size + sample-spacing + wavelength for source factories.

    v4.14.2 (P1-NEW-10): Centralised at the top of ``sources/core.py``
    so every ``create_*`` factory raises identical, named errors on
    physically-impossible inputs.  Pre-v4.14.2 only the DOE family
    (``create_diffractive_lens``, ``create_kinoform``,
    ``create_fresnel_zone_plate``) validated these; the 10 factories
    in ``sources/core.py`` silently accepted ``N=0`` (empty grid,
    cryptic ``ValueError`` from ``meshgrid``), ``dx<=0`` (negative or
    zero pixel pitch produces a flipped or singular coordinate
    system), and ``wavelength<=0`` (divide-by-zero in ``k0 = 2*pi /
    wavelength``).

    Parameters
    ----------
    N : int or (Ny, Nx)
        Grid size.  Tuple form is for rectangular grids.
    dx : float
        Sample spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    dy : float, optional
        Sample spacing in y [m].  ``None`` is valid (the factory
        defaults to ``dy = dx``).
    fn_name : str
        Factory name, embedded into every error message so the
        caller knows which entry point raised.
    """
    # ``N`` -- positive integer, or a 2-tuple of positive integers.
    if isinstance(N, (tuple, list)):
        if len(N) != 2:
            raise ValueError(
                f"{fn_name}: N tuple form must be (Ny, Nx); got "
                f"length-{len(N)} sequence {N!r}.")
        Ny, Nx = N
        for label, n in (('Ny', Ny), ('Nx', Nx)):
            if not isinstance(n, (int, np.integer)):
                raise ValueError(
                    f"{fn_name}: {label} must be a positive integer, "
                    f"got {type(n).__name__} ({n!r}).")
            if int(n) <= 0:
                raise ValueError(
                    f"{fn_name}: {label} must be a positive integer, "
                    f"got {int(n)}.")
    else:
        if not isinstance(N, (int, np.integer)):
            raise ValueError(
                f"{fn_name}: N must be a positive integer "
                f"(or (Ny, Nx) tuple), got {type(N).__name__} ({N!r}).")
        if int(N) <= 0:
            raise ValueError(
                f"{fn_name}: N must be a positive integer, got {int(N)}.")

    # ``dx`` -- positive finite float.  ``np.float64`` ``bool`` etc.
    # all support comparison; only reject NaN / inf / non-numeric.
    try:
        dx_f = float(dx)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{fn_name}: dx must be a positive finite number [m], "
            f"got {dx!r} ({type(dx).__name__}).") from exc
    if not np.isfinite(dx_f) or dx_f <= 0.0:
        raise ValueError(
            f"{fn_name}: dx must be a positive finite number [m], "
            f"got {dx_f}.")

    # ``dy`` -- None (square grid) or positive finite float.
    if dy is not None:
        try:
            dy_f = float(dy)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{fn_name}: dy must be None or a positive finite "
                f"number [m], got {dy!r} ({type(dy).__name__}).") from exc
        if not np.isfinite(dy_f) or dy_f <= 0.0:
            raise ValueError(
                f"{fn_name}: dy must be None or a positive finite "
                f"number [m], got {dy_f}.")

    # ``wavelength`` -- positive finite float.
    try:
        wl_f = float(wavelength)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{fn_name}: wavelength must be a positive finite number "
            f"[m], got {wavelength!r} ({type(wavelength).__name__}).") from exc
    if not np.isfinite(wl_f) or wl_f <= 0.0:
        raise ValueError(
            f"{fn_name}: wavelength must be a positive finite number "
            f"[m], got {wl_f}.")


def _resolve_complex_dtype(dtype: Optional[Any]) -> np.dtype:
    """Resolve a user-provided dtype kwarg to a concrete complex dtype.

    Convention shared by every ``create_*`` factory in this module
    (4.8.1+):

    - ``dtype=None``       -> :data:`lumenairy.DEFAULT_COMPLEX_DTYPE`
      (the library-global default, controlled by
      :func:`set_default_complex_dtype` and
      :func:`lumenairy_context`).
    - ``dtype=np.complex64`` / ``dtype=np.complex128`` -> the given
      dtype, validated.
    - Anything else -> ``ValueError``.

    Centralised here so every factory honours the same precedence
    rules and the same input validation.  Before 4.8.1 the factories
    silently produced :class:`numpy.complex128` regardless of the
    library default, which broke memory budgeting for users who set
    :func:`set_default_complex_dtype` and then created a source.
    """
    from ..propagators.propagation import get_default_complex_dtype
    if dtype is None:
        return np.dtype(get_default_complex_dtype())
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError(
            f"dtype must be np.complex64 or np.complex128; got {dt!r}.")
    return dt


# ---------------------------------------------------------------------------
# Fundamental Gaussian beam
# ---------------------------------------------------------------------------

def create_gaussian_beam(
    N: Union[int, Tuple[int, int]],
    dx: float,
    wavelength: float,
    *,
    sigma: float,
    x0: float = 0,
    y0: float = 0,
    use_gpu: bool = False,
    dy: Optional[float] = None,
    normalize: str = 'peak',
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Gaussian beam field.

    Parameters
    ----------
    N : int or tuple
        Grid size. If int, creates an N x N grid. If tuple, interpreted as
        (Ny, Nx).
    dx : float
        Grid spacing in x [m].
    sigma : float
        Gaussian width parameter (field standard deviation) [m].
        The 1/e field amplitude radius is sigma * sqrt(2).
        The 1/e^2 intensity radius (beam waist w0) is also sigma * sqrt(2).
    wavelength : float, optional
        Reserved for future use (e.g. adding a spherical phase for a
        focused beam). Currently unused -- the returned field has flat phase.
    x0, y0 : float, default 0
        Center position of the beam [m].
    use_gpu : bool, default False
        If True and CuPy is available, create the arrays on the GPU.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    normalize : ``'peak'`` (default) / ``'power'`` / ``'none'``
        Output scaling.  ``'peak'`` (the historical default) returns
        a unit-peak amplitude field.  ``'power'`` returns a
        unit-integrated-power field (matching :func:`create_hermite_gauss`
        and :func:`create_laguerre_gauss` -- pass ``normalize='power'``
        whenever you want to chain or compare across the mode-family
        helpers).  ``'none'`` returns the raw ``exp(-r^2/(2 sigma^2))``
        without scaling.

    Returns
    -------
    E : ndarray, complex
        Gaussian beam field (Ny x Nx).
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    Signature is ``(N, dx, wavelength, *, sigma, ...)`` since 4.7.
    Prior to 4.7 the ordering was
    ``(N, dx, sigma, wavelength=None, ...)`` with positional ``sigma``;
    the new style places ``wavelength`` at the third positional
    slot (matching every other source factory) and makes
    ``sigma`` keyword-only.
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_gaussian_beam')
    if CUPY_AVAILABLE and use_gpu:
        # 4.10: pre-4.10 reached for module-level ``cp`` without first
        # calling _ensure_cupy_loaded(), so ``cp`` was still None and
        # the first GPU call raised AttributeError on xp.arange(...).
        _ensure_cupy_loaded()
        xp = cp
    else:
        xp = np
    if dy is None:
        dy = dx

    if isinstance(N, int):
        Ny, Nx = N, N
    else:
        Ny, Nx = N

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    # Gaussian amplitude: exp(-r^2 / (2 sigma^2))
    target_dtype = _resolve_complex_dtype(dtype)
    E = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    E = E.astype(target_dtype)

    if normalize == 'peak':
        pass  # already peak == 1 from exp(0) at the centre
    elif normalize == 'power':
        norm = xp.sqrt(xp.sum(xp.abs(E) ** 2) * dx * dy)
        if float(norm) > 0:
            E = E / norm
    elif normalize == 'none':
        pass
    else:
        raise ValueError(
            f"create_gaussian_beam: normalize must be one of "
            f"'peak', 'power', 'none'; got {normalize!r}.")

    return E, x, y


# ---------------------------------------------------------------------------
# Hermite-Gaussian modes
# ---------------------------------------------------------------------------

def hermite_physicist(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the physicist's Hermite polynomial H_n(x) via recurrence.

    Uses the three-term recurrence relation:

        H_0(x) = 1
        H_1(x) = 2x
        H_k(x) = 2x H_{k-1}(x) - 2(k-1) H_{k-2}(x)

    Parameters
    ----------
    n : int
        Polynomial order (>= 0).
    x : ndarray
        Points at which to evaluate H_n.

    Returns
    -------
    H_n : ndarray
        Values of the physicist's Hermite polynomial of order *n*.
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_prev2 = np.ones_like(x)
        H_prev1 = 2 * x
        for k in range(2, n + 1):
            H_curr = 2 * x * H_prev1 - 2 * (k - 1) * H_prev2
            H_prev2 = H_prev1
            H_prev1 = H_curr
        return H_curr


def create_hermite_gauss(
    N: Union[int, Tuple[int, int]],
    dx: float,
    w0: float,
    wavelength: float,
    m: int = 0,
    n: int = 0,
    x0: float = 0,
    y0: float = 0,
    dy: Optional[float] = None,
    normalize: str = 'power',
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Hermite-Gaussian (HG_mn) beam mode at the waist.

    Parameters
    ----------
    N : int or (Ny, Nx)
        Grid size.  If a scalar, an ``N x N`` square grid is built.  Pass
        a 2-tuple ``(Ny, Nx)`` for rectangular grids.
    dx : float
        Grid spacing in x [m].
    w0 : float
        Beam waist (1/e^2 intensity radius) [m].
    wavelength : float
        Wavelength [m]. Currently unused -- the field is returned at the
        waist with flat phase.
    m, n : int, default 0
        Transverse mode indices. HG_00 is the fundamental Gaussian.
    x0, y0 : float, default 0
        Beam center [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square pitch).
        Provide explicitly for rectangular-pitch grids so the
        Gaussian envelope isn't silently stretched along y.

    Returns
    -------
    E : ndarray, complex (Ny x Nx)
        Hermite-Gaussian mode field, power-normalised.
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    The (un-normalised) field is

        E_mn(x, y) = H_m(sqrt(2) x / w0) * H_n(sqrt(2) y / w0)
                      * exp(-(x^2 + y^2) / w0^2)

    where H_m is the physicist's Hermite polynomial of order m.

    **Normalisation note.**  When ``normalize='power'``, the returned
    field is power-normalised by **numerical integration over the
    grid** (``sum(|E|^2) * dx * dy = 1``).  This is grid-exact for any
    grid that fully contains the mode (typically ``L >= 4 w0``), but
    differs from the analytical normalisation
    ``N = 1 / sqrt(2^m m! 2^n n! pi w0^2 / 2)``
    used by :func:`lumenairy.propagators.asymptotic.fit_canonical_polynomials`
    and the modal-asymptotic propagators when the grid truncates the
    Gaussian tails.  For typical wave-optics grids the two normalisations
    agree to ~1e-6; for tight grids that clip at 2-3 w0 the discrepancy
    can grow.  Use the analytical asymptotic-module normalisation
    when chaining HG modes through the modal asymptotic propagator.
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_hermite_gauss')
    if dy is None:
        dy = dx
    if isinstance(N, (tuple, list)):
        Ny, Nx = int(N[0]), int(N[1])
    else:
        Ny = Nx = int(N)
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    u = np.sqrt(2) * (X - x0) / w0
    v = np.sqrt(2) * (Y - y0) / w0

    Hm = hermite_physicist(m, u)
    Hn = hermite_physicist(n, v)

    gaussian = np.exp(-((X - x0)**2 + (Y - y0)**2) / w0**2)
    E = (Hm * Hn * gaussian).astype(_resolve_complex_dtype(dtype))

    if normalize == 'power':
        norm = np.sqrt(np.sum(np.abs(E)**2) * dx * dy)
        if norm > 0:
            E /= norm
    elif normalize == 'peak':
        pk = float(np.abs(E).max())
        if pk > 0:
            E /= pk
    elif normalize == 'none':
        pass
    else:
        raise ValueError(
            f"create_hermite_gauss: normalize must be one of "
            f"'peak', 'power', 'none'; got {normalize!r}.")

    return E, x, y


# ---------------------------------------------------------------------------
# Laguerre-Gaussian modes
# ---------------------------------------------------------------------------

def laguerre_generalized(p: int, l_abs: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the generalized Laguerre polynomial L_p^l(x) via recurrence.

    Uses the three-term recurrence relation:

        L_0^l(x) = 1
        L_1^l(x) = 1 + l - x
        L_k^l(x) = ((2k - 1 + l - x) L_{k-1}^l(x)
                     - (k - 1 + l) L_{k-2}^l(x)) / k

    Parameters
    ----------
    p : int
        Polynomial order (radial index, >= 0).
    l_abs : int
        Associated (generalized) index (|l|, >= 0).
    x : ndarray
        Points at which to evaluate L_p^{l_abs}.

    Returns
    -------
    L_p : ndarray
        Values of the generalized Laguerre polynomial.
    """
    if p == 0:
        return np.ones_like(x)
    elif p == 1:
        return 1 + l_abs - x
    else:
        L_prev2 = np.ones_like(x)
        L_prev1 = 1 + l_abs - x
        for k in range(2, p + 1):
            L_curr = ((2 * k - 1 + l_abs - x) * L_prev1
                      - (k - 1 + l_abs) * L_prev2) / k
            L_prev2 = L_prev1
            L_prev1 = L_curr
        return L_curr


def create_laguerre_gauss(
    N: Union[int, Tuple[int, int]],
    dx: float,
    w0: float,
    wavelength: float,
    p: int = 0,
    l: int = 0,
    x0: float = 0,
    y0: float = 0,
    dy: Optional[float] = None,
    normalize: str = 'power',
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Laguerre-Gaussian (LG_pl) beam mode at the waist.

    Parameters
    ----------
    N : int or (Ny, Nx)
        Grid size.  Scalar = square grid; tuple = rectangular.
    dx : float
        Grid spacing in x [m].
    w0 : float
        Beam waist (1/e^2 intensity radius) [m].
    wavelength : float
        Wavelength [m]. Currently unused -- the field is returned at the
        waist with flat phase.
    p : int, default 0
        Radial index (number of radial nodes).
    l : int, default 0
        Azimuthal index (topological charge / orbital angular momentum).
        LG_00 is the fundamental Gaussian.
    x0, y0 : float, default 0
        Beam center [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E : ndarray, complex (Ny x Nx)
        Laguerre-Gaussian mode field, power-normalised.
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    The (un-normalised) field is

        E_pl(r, theta) = (r sqrt(2) / w0)^|l| * L_p^|l|(2 r^2 / w0^2)
                          * exp(-r^2 / w0^2) * exp(i l theta)

    where L_p^|l| is the generalized Laguerre polynomial.

    **Normalisation note.**  ``normalize='power'`` integrates over
    the grid numerically; same caveat as :func:`create_hermite_gauss`
    -- for grids that clip the mode tails the result differs from
    the analytical LG normalisation used in
    :mod:`lumenairy.propagators.asymptotic`.  For modal-asymptotic
    chains, prefer the analytical-normalised modes built inside the
    asymptotic propagator over passing this function's output
    through ``propagate_modal_asymptotic``.
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_laguerre_gauss')
    if dy is None:
        dy = dx
    if isinstance(N, (tuple, list)):
        Ny, Nx = int(N[0]), int(N[1])
    else:
        Ny = Nx = int(N)
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = np.arctan2(Y - y0, X - x0)

    rho = np.sqrt(2) * r / w0
    rho_sq = rho**2

    # Generalized Laguerre polynomial L_p^|l|
    L = laguerre_generalized(p, abs(l), rho_sq)

    gaussian = np.exp(-r**2 / w0**2)
    E = (rho**abs(l) * L * gaussian
         * np.exp(1j * l * theta)).astype(_resolve_complex_dtype(dtype))

    if normalize == 'power':
        norm = np.sqrt(np.sum(np.abs(E)**2) * dx * dy)
        if norm > 0:
            E /= norm
    elif normalize == 'peak':
        pk = float(np.abs(E).max())
        if pk > 0:
            E /= pk
    elif normalize == 'none':
        pass
    else:
        raise ValueError(
            f"create_laguerre_gauss: normalize must be one of "
            f"'peak', 'power', 'none'; got {normalize!r}.")

    return E, x, y


# ---------------------------------------------------------------------------
# Off-axis / tilted plane-wave sources
# ---------------------------------------------------------------------------

def create_tilted_plane_wave(
    N: int,
    dx: float,
    wavelength: float,
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    amplitude: float = 1.0,
    dy: Optional[float] = None,
    *,
    angle_x_deg: Optional[float] = None,
    angle_y_deg: Optional[float] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a tilted (off-axis) plane wave on an N x N grid.

    A tilted plane wave has a linear phase ramp across the pupil,
    representing a collimated beam arriving from a direction offset
    from the optical axis by ``angle_x`` (horizontal) and ``angle_y``
    (vertical).  This is the standard source for evaluating off-axis
    imaging performance -- pass it through ``apply_real_lens`` and
    compare the exit-pupil OPD or PSF to the on-axis case.

    Parameters
    ----------
    N : int
        Grid dimension (square N x N).
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    angle_x : float, default 0
        Field angle in the x-z plane [rad].  Positive = source
        tilted toward +x.
    angle_y : float, default 0
        Field angle in the y-z plane [rad].  Positive = source
        tilted toward +y.
    angle_x_deg, angle_y_deg : float, optional
        Same as ``angle_x`` / ``angle_y`` but expressed in degrees.
        When provided, these take precedence over the radian forms
        and provide a convenience for human-readable field-angle
        sweeps.  4.7+: the library is converging on ``*_deg`` as the
        canonical user-facing angle unit; the bare-radians
        ``angle_x`` / ``angle_y`` will become deprecated aliases in
        a future release.
    amplitude : float, default 1
        Uniform amplitude.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E : ndarray, complex, shape (N, N)
        Complex field on the grid.
    x, y : ndarray
        1-D coordinate arrays [m].
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_tilted_plane_wave')
    if angle_x_deg is not None:
        angle_x = float(np.radians(angle_x_deg))
    if angle_y_deg is not None:
        angle_y = float(np.radians(angle_y_deg))
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / wavelength
    phase = k0 * (np.sin(angle_x) * X + np.sin(angle_y) * Y)
    E = (amplitude * np.exp(1j * phase)).astype(_resolve_complex_dtype(dtype))
    return E, x, y


def create_point_source(
    N: int,
    dx: float,
    wavelength: float,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    amplitude: float = 1.0,
    dy: Optional[float] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a spherical wave from a point at ``(x0, y0, z0)``
    evaluated on the grid plane at z=0.

    Sign of ``z0`` selects diverging vs converging:

    - ``z0 < 0`` -- source is *before* the grid; the wavefront at z=0
      is *diverging* (positive curvature) and uses ``exp(+i k r)/r``.
    - ``z0 > 0`` -- target is *after* the grid; the wavefront at z=0
      is *converging* (negative curvature) and uses ``exp(-i k r)/r``.
      This is the field a perfect lens would put on the grid to focus
      to a point at axial distance ``z0`` past it.
    - ``z0 = 0`` -- 1/r singular profile in-plane (a diverging point
      source coincident with the grid); the central pixel is clamped
      to a finite floor.  Rarely the right tool; prefer a non-zero
      ``z0`` to get a well-defined spherical wavefront.

    The sign convention pairs with LumenAiry's ``exp(-i*omega*t)``
    time-harmonic convention: ``exp(+i k r)/r`` is an outgoing
    (diverging) spherical wave, ``exp(-i k r)/r`` is an incoming
    (converging) one.

    Parameters
    ----------
    N, dx, wavelength : usual
    x0, y0 : float
        Transverse position of the point [m].
    z0 : float
        Axial position of the point [m] relative to the grid plane.
        Negative = source before grid (diverging), positive = focus
        after grid (converging).
    amplitude : float
    dy : float, optional

    Returns
    -------
    E : ndarray, complex, shape (N, N)
    x, y : ndarray
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_point_source')
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / wavelength
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + z0 ** 2)
    # 4.10: warn when |z0| < dx -- the central pixel sits at the
    # spherical-wave singularity and any sub-pixel ``r`` is a pure
    # discretisation artefact.  The Fresnel-curvature representation
    # of a point source assumes |z0| >> dx; without that the result
    # is essentially a numerical singularity, not a physical field.
    if abs(z0) < dx:
        import warnings
        warnings.warn(
            f"create_point_source: |z0| = {abs(z0):.3e} m is comparable "
            f"to dx = {dx:.3e} m; the central pixel will dominate the "
            f"integrated power.  Use |z0| >> dx (typical: 10*dx or "
            f"more) for a meaningful Fresnel-curvature representation.",
            RuntimeWarning, stacklevel=2,
        )
    # 4.11.1 (H-PR-4): floor ``r`` at the local pixel half-diagonal
    # rather than 1e-30.  Pre-4.11.1 the |E| = amplitude / r evaluation
    # on the central pixel hit 1e30, dominating every downstream power
    # integral.  The half-diagonal floor caps |E_central| at
    # amplitude / (sqrt(dx**2+dy**2)/2), which is the largest distance
    # any sub-pixel point can have from the cell centroid -- physically
    # the right scale for a point source binned onto a finite grid.
    r_floor = 0.5 * np.sqrt(dx * dx + dy * dy)
    r = np.maximum(r, r_floor)
    # Sign convention under exp(-i*omega*t):
    #   z0 < 0 (source before grid)   -> diverging, exp(+i*k*r)/r
    #   z0 > 0 (focus after grid)     -> converging, exp(-i*k*r)/r
    #   z0 == 0 (in-plane singularity)-> use the diverging form
    sign = -1.0 if z0 > 0.0 else 1.0
    E = (amplitude * np.exp(1j * sign * k0 * r) / r).astype(
        _resolve_complex_dtype(dtype))
    return E, x, y


def create_multi_field_sources(
    N: int,
    dx: float,
    wavelength: float,
    field_angles: Sequence[Union[float, Tuple[float, float]]],
    amplitude: float = 1.0,
    dy: Optional[float] = None,
    dtype: Optional[Any] = None,
) -> Tuple[List[Tuple[np.ndarray, float, float]], np.ndarray, np.ndarray]:
    """Generate a list of tilted plane waves at the given field angles.

    Convenience wrapper around :func:`create_tilted_plane_wave` for
    setting up multi-field analyses.

    Parameters
    ----------
    N, dx, wavelength : usual
    field_angles : sequence of float or tuple
        Each element is either a scalar (y-tilt only) or a
        ``(angle_x, angle_y)`` tuple.
    amplitude : float
    dy : float, optional

    Returns
    -------
    sources : list of (E, angle_x, angle_y)
        One per field angle.  **Note the return shape differs from
        the scalar ``create_*`` helpers** (which return ``(E, x, y)``
        directly): this is a *list of tilted-plane-wave sources*,
        not a single field.
    x, y : ndarray
        Shared 1-D coordinate arrays.
    """
    sources = []
    x = y = None
    for a in field_angles:
        if isinstance(a, (list, tuple)):
            ax, ay = float(a[0]), float(a[1])
        else:
            ax, ay = 0.0, float(a)
        E, x, y = create_tilted_plane_wave(
            N, dx, wavelength, angle_x=ax, angle_y=ay,
            amplitude=amplitude, dy=dy, dtype=dtype)
        sources.append((E, ax, ay))
    return sources, x, y


# ---------------------------------------------------------------------------
# Extended source models (LED, fiber, top-hat, annular, Bessel)
# ---------------------------------------------------------------------------

def create_top_hat_beam(
    N: int,
    dx: float,
    wavelength: float,
    *,
    diameter: float,
    x0: float = 0,
    y0: float = 0,
    dy: Optional[float] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-intensity circular beam (top-hat / flat-top).

    Parameters
    ----------
    N, dx : int, float
    wavelength : float
        Reserved for future use (overlay of a spherical-phase term);
        currently does not affect the returned field.
    diameter : float
        Beam diameter [m].
    x0, y0 : float
        Center [m].

    Returns
    -------
    E, x, y : ndarray

    Notes
    -----
    Signature is ``(N, dx, wavelength, *, diameter, ...)`` since 4.7.
    Prior to 4.7 the ordering was
    ``(N, dx, diameter, wavelength=None, ...)``.
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_top_hat_beam')
    # 4.10: honour caller-supplied dy (anamorphic grid).  Pre-4.10
    # hard-coded dy = dx and used dx**2 for the area element, silently
    # ignoring caller-supplied dy on top-hat / annular / Bessel sources
    # only.  Defaults to dy = dx for back-compat.
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    E = np.where(r <= diameter / 2, 1.0, 0.0).astype(
        _resolve_complex_dtype(dtype))
    norm = np.sqrt(np.sum(np.abs(E) ** 2) * dx * dy)
    if norm > 0:
        E /= norm
    return E, x, y


def create_annular_beam(
    N: int,
    dx: float,
    wavelength: float,
    *,
    outer_diameter: float,
    inner_diameter: float,
    x0: float = 0,
    y0: float = 0,
    dy: Optional[float] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annular (donut) beam.

    Parameters
    ----------
    N, dx : int, float
    wavelength : float
        Reserved for future use.
    outer_diameter, inner_diameter : float [m]

    Returns
    -------
    E, x, y : ndarray

    Notes
    -----
    Signature is
    ``(N, dx, wavelength, *, outer_diameter, inner_diameter, ...)``
    since 4.7.
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_annular_beam')
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    E = np.where((r <= outer_diameter / 2) & (r >= inner_diameter / 2),
                  1.0, 0.0).astype(_resolve_complex_dtype(dtype))
    norm = np.sqrt(np.sum(np.abs(E) ** 2) * dx * dy)
    if norm > 0:
        E /= norm
    return E, x, y


def create_fiber_mode(
    N: int,
    dx: float,
    wavelength: float,
    *,
    mode_field_diameter: float,
    x0: float = 0,
    y0: float = 0,
    na: float = 0.12,
    dy: Optional[float] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-mode fiber output (Gaussian with NA-defined divergence).

    The mode-field diameter (MFD) is the 1/e^2 intensity diameter.
    The field is a Gaussian with w0 = MFD/2, and the NA is encoded
    in the far-field divergence angle.

    Parameters
    ----------
    N, dx : int, float
    wavelength : float [m]
    mode_field_diameter : float [m]
    x0, y0 : float
    na : float
        Fiber numerical aperture.  4.10: This argument is accepted
        but ONLY USED to emit a warning when NA > 0.2 (where the LP01
        mode departs significantly from a Gaussian and the
        MFD-Gaussian approximation breaks down by ~10 % in mode
        shape).  The near-field amplitude profile is determined
        entirely by ``mode_field_diameter``.  For high-NA fibres
        (PCF, multimode-near-cutoff) use a full LP01 mode solver
        externally and pass the result via ``Source.from_array``.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.  v4.13.2 added the
        keyword so :meth:`Source.fiber_mode` can pass an anamorphic
        ``dy`` through to the underlying Gaussian without erroring
        (the historical signature only accepted ``dx``, so a user-
        supplied ``dy=`` was silently squared via ``factory_kwargs``).

    Notes
    -----
    Signature is
    ``(N, dx, wavelength, *, mode_field_diameter, ...)`` since 4.7.

    Returns
    -------
    E, x, y : ndarray
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_fiber_mode')
    if na is not None and na > 0.2:
        import warnings
        warnings.warn(
            f"create_fiber_mode: NA={na:.3f} > 0.2 -- the LP01 mode is "
            "no longer well-approximated by a Gaussian at this NA "
            "(mode-shape error ~10%+).  For high-NA fibres use a "
            "full LP01 solver.",
            RuntimeWarning, stacklevel=2,
        )
    w0 = mode_field_diameter / 2.0
    sigma = w0 / np.sqrt(2)
    return create_gaussian_beam(N, dx, wavelength, sigma=sigma,
                                 x0=x0, y0=y0, dy=dy, dtype=dtype)


def create_led_source(
    N: int,
    dx: float,
    wavelength: Optional[float] = None,
    *args: Any,
    diameter: Optional[float] = None,
    divergence_angle: Optional[float] = None,
    dy: Optional[float] = None,
    x0: float = 0,
    y0: float = 0,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]], np.ndarray, np.ndarray]:
    """Lambertian LED source (incoherent; returns the intensity
    envelope as a complex field for use with partial-coherence
    imaging).

    The spatial extent is a uniform disk of given diameter; the
    angular extent (divergence) determines how many source angles
    to sample when using ``koehler_image`` or
    ``extended_source_image``.

    Parameters
    ----------
    N, dx : int, float
        Grid size + sample spacing.
    wavelength : float [m]
        Vacuum wavelength.  Now in the canonical 3rd positional slot
        (since v4.14.2) -- matches every other ``create_*`` factory in
        ``sources/core.py``.
    diameter : float [m]
        Emitting area diameter.  Keyword-only since v4.14.2.
    divergence_angle : float [rad]
        Half-angle of the emission cone.  Keyword-only since v4.14.2.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square grid).
        v4.14.2 added the kwarg so anamorphic grids can thread a
        distinct y-pitch through this factory like the rest of the
        v4.13.0+ Source family.
    x0, y0 : float, default 0
        Center of the emitting disk [m].
    dtype : optional
        Complex dtype (``np.complex64`` / ``np.complex128``); defaults
        to the library-global ``DEFAULT_COMPLEX_DTYPE``.

    Returns
    -------
    E : ndarray (complex)
        Amplitude envelope (uniform inside disk, zero outside).
    source_angles : list of (float, float)
        Suggested source angles for partial-coherence integration,
        covering the divergence cone with 37 angle samples (1 axial
        + 6 + 12 + 18 = 37 for ``n_ring = 3``).  4.10: docstring
        previously said "~21 samples" which underspecified the actual
        count; downstream callers allocating output arrays should size
        them at 37, not 21.
    x, y : ndarray
        1-D coordinate axes [m].

    Notes
    -----
    Signature is ``(N, dx, wavelength, *, diameter, divergence_angle,
    dy=None, x0=0, y0=0, dtype=None)`` since v4.14.2.  Pre-v4.14.2 the
    ordering was ``(N, dx, diameter, divergence_angle, wavelength,
    x0=0, y0=0, dtype=None)`` -- ``diameter`` and ``divergence_angle``
    were positional and the function neither accepted ``dy=`` nor a
    ``*`` keyword-only separator.  This broke the post-v4.7 convention
    of keyword-only physical parameters with ``wavelength`` in the
    canonical 3rd positional slot.

    The legacy positional form is still accepted for one release with a
    ``DeprecationWarning``.  Migrate to the keyword-only form before
    the deprecation grace period ends::

        # Old (deprecated, still works with a warning)
        E, angles, x, y = create_led_source(64, 16e-6, 100e-6, 0.3, 1.31e-6)

        # New (canonical)
        E, angles, x, y = create_led_source(
            64, 16e-6, 1.31e-6,
            diameter=100e-6, divergence_angle=0.3)
    """
    # v4.14.2 (P1-NEW-9): backward-compat shim for the pre-v4.14.2
    # positional ``(N, dx, diameter, divergence_angle, wavelength,
    # x0, y0, dtype)`` form.  We detect it via ``*args``: under the
    # new signature ``*args`` must be empty (everything after
    # ``wavelength`` is keyword-only).  If any positional surplus
    # arrives, treat the call as legacy:
    #   - ``wavelength`` (the 3rd positional under the new sig) is
    #     actually the legacy ``diameter``;
    #   - ``args[0]`` is the legacy ``divergence_angle``;
    #   - ``args[1]`` is the legacy ``wavelength``;
    #   - ``args[2..3]``, if present, are legacy ``x0``, ``y0``;
    #   - ``args[4]``, if present, is legacy ``dtype``.
    if args:
        import warnings
        warnings.warn(
            "create_led_source: positional call form "
            "``(N, dx, diameter, divergence_angle, wavelength, ...)`` "
            "is deprecated since v4.14.2.  Use the keyword-only form "
            "``create_led_source(N, dx, wavelength, *, diameter=..., "
            "divergence_angle=..., ...)`` instead.  The legacy "
            "positional form will be removed in a future release.",
            DeprecationWarning, stacklevel=2,
        )
        # Re-map legacy positionals.  ``wavelength`` is the 3rd
        # positional under the new sig but the legacy ``diameter``
        # under the old sig.
        _legacy_diameter = wavelength
        _legacy_divergence = args[0]
        _legacy_wavelength = args[1] if len(args) > 1 else None
        if _legacy_wavelength is None:
            raise TypeError(
                "create_led_source (legacy positional form): "
                "expected 5+ positional arguments "
                "``(N, dx, diameter, divergence_angle, wavelength, ...)``; "
                "got only 4.  Migrate to the new keyword-only form: "
                "``create_led_source(N, dx, wavelength, *, "
                "diameter=..., divergence_angle=...)``.")
        # Promote legacy positionals into the canonical kwargs, but do
        # not overwrite kwargs the caller explicitly supplied (that's
        # an unambiguous error).
        if diameter is not None:
            raise TypeError(
                "create_led_source: 'diameter' supplied both "
                "positionally (legacy form) and as a keyword.")
        if divergence_angle is not None:
            raise TypeError(
                "create_led_source: 'divergence_angle' supplied both "
                "positionally (legacy form) and as a keyword.")
        diameter = _legacy_diameter
        divergence_angle = _legacy_divergence
        wavelength = _legacy_wavelength
        # x0, y0, dtype are still positional-or-keyword under both
        # forms.  Pre-v4.14.2 they sat at positions 5, 6, 7 (after
        # wavelength); only consume them from ``args`` if the caller
        # actually passed positional surplus past wavelength.
        if len(args) > 2:
            x0 = args[2]
        if len(args) > 3:
            y0 = args[3]
        if len(args) > 4:
            dtype = args[4]
        if len(args) > 5:
            raise TypeError(
                "create_led_source (legacy positional form): too many "
                f"positional arguments; got {3 + len(args)} positional, "
                f"max 8 (legacy) or 3 (canonical, with kwargs).")

    # Validate the required keyword-only physical parameters
    # post-shim so the error path is the same for both call forms.
    if diameter is None:
        raise TypeError(
            "create_led_source: missing required keyword argument "
            "'diameter' (the LED emitting-area diameter in metres).")
    if divergence_angle is None:
        raise TypeError(
            "create_led_source: missing required keyword argument "
            "'divergence_angle' (the half-angle of the emission cone, "
            "radians).")
    if wavelength is None:
        raise TypeError(
            "create_led_source: missing required positional argument "
            "'wavelength' (vacuum wavelength in metres).")

    # v4.14.2 (P1-NEW-10): centralised input validation -- catches
    # N<=0, dx<=0, dy<=0, wavelength<=0, non-finite, etc., with a
    # clear error message that names the factory.
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_led_source')
    # Negative or zero diameter / divergence_angle would silently
    # produce a degenerate (empty / inside-out) source -- raise loud.
    if not (np.isfinite(diameter) and diameter > 0):
        raise ValueError(
            "create_led_source: diameter must be a positive finite "
            f"number [m], got {diameter}.")
    if not (np.isfinite(divergence_angle) and divergence_angle > 0):
        raise ValueError(
            "create_led_source: divergence_angle must be a positive "
            f"finite number [rad], got {divergence_angle}.")

    E, x, y = create_top_hat_beam(N, dx, wavelength, diameter=diameter,
                                   x0=x0, y0=y0, dy=dy, dtype=dtype)
    # Generate suggested source angles
    n_ring = 3
    angles = [(0.0, 0.0)]
    for ring in range(1, n_ring + 1):
        r = divergence_angle * ring / n_ring
        for k in range(6 * ring):
            theta = 2 * np.pi * k / (6 * ring)
            angles.append((r * np.cos(theta), r * np.sin(theta)))
    return E, angles, x, y


def create_bessel_beam(
    N: int,
    dx: float,
    wavelength: float,
    cone_angle: float,
    x0: float = 0,
    y0: float = 0,
    dy: Optional[float] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal Bessel beam (J_0 profile).

    Creates the field proportional to J_0(k_r * r) where
    k_r = k * sin(cone_angle).  This is an idealized non-diffracting
    beam; in practice it's produced by an axicon or annular aperture.

    Parameters
    ----------
    N, dx : int, float
    wavelength : float [m]
    cone_angle : float [rad]
        Half-angle of the Bessel cone.

    Returns
    -------
    E, x, y : ndarray
    """
    from scipy.special import j0

    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_bessel_beam')
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    k_r = 2 * np.pi / wavelength * np.sin(cone_angle)
    E = j0(k_r * r).astype(_resolve_complex_dtype(dtype))
    return E, x, y


# ---------------------------------------------------------------------------
# Source -- bundles E + dx + wavelength + source_point with chainable
# .propagate(...) and .from_X(...) factories
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as _dc_field
from typing import Optional as _Optional, Tuple as _Tuple


@dataclass
class Source:
    """A complex field on a regular grid with the metadata it needs to
    propagate through the rest of lumenairy.

    Bundles the four pieces of state every propagator wants -- ``E``,
    ``dx``, ``wavelength``, and (for asymptotic / aberration-tensor
    work) ``source_point`` -- so callers don't have to repeat them.

    Attributes
    ----------
    E : ndarray, complex
        Field on a (Ny, Nx) grid.
    dx : float
        Sample spacing along x [m].
    dy : float, optional
        Sample spacing along y [m].  v4.13.0 (audit L3): defaults to
        ``dx`` for square-grid sources, preserving back-compat for
        every existing caller.  Anamorphic grids (e.g. cylindrical
        beams sampled on rectangular pixels) can now thread a distinct
        y-pitch through the Source -> propagate -> PropagationResult
        chain without silently losing the metadata at the Source
        boundary.
    wavelength : float
        Vacuum wavelength [m].
    source_point : (float, float), default (0, 0)
        Object-plane location used by the asymptotic / LG-tensor
        propagators.  Ignored by ASM / GBD / HFPI / HF.
    name : str, optional
        Human-readable label, propagated to descendants for tracing.
    """
    E: 'object'  # numpy or cupy or jax ndarray
    dx: float
    wavelength: float
    source_point: _Tuple[float, float] = (0.0, 0.0)
    name: _Optional[str] = None
    # v4.13.0 (audit L3): anamorphic pitch on the y-axis.  ``None``
    # falls through to ``dx`` (square grid).  Placed last so existing
    # callers using positional args (``Source(E, dx, wavelength)``)
    # remain compatible.
    dy: _Optional[float] = None

    def __post_init__(self) -> None:
        # v4.13.0 (audit L3): default ``dy`` to ``dx`` so the
        # post-init attribute is always non-None for downstream code.
        if self.dy is None:
            self.dy = self.dx

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.E.shape[-2:])

    def __repr__(self) -> str:
        dy_part = (f", dy={self.dy:.3g}m"
                    if self.dy is not None and self.dy != self.dx else "")
        return (f"Source(shape={self.shape}, dx={self.dx:.3g}m{dy_part}, "
                f"wavelength={self.wavelength*1e9:.1f}nm, "
                f"source_point={self.source_point}, "
                f"name={self.name!r})")

    def propagate(self,
                   *,
                   method: str = 'auto',
                   z: _Optional[float] = None,
                   prescription: _Optional[dict] = None,
                   **kwargs) -> 'Source':
        """Propagate this Source via :func:`la.propagate` and return
        the result as a new Source on the (possibly resampled) output
        plane.

        ``z``, ``prescription``, ``method``, and any propagator-
        specific keyword arguments are forwarded to the dispatcher.
        ``dx`` defaults to the input spacing; pass ``output_dx`` to
        change it.  Wavelength, source_point, and name are inherited
        on the new Source (with ``->{method}`` appended to the name
        for trace-ability).
        """
        from ..propagators.dispatch import propagate
        E_out = propagate(
            self.E,
            wavelength=self.wavelength,
            dx=self.dx,
            z=z,
            prescription=prescription,
            method=method,
            **kwargs,
        )
        out_dx = kwargs.get('output_dx', self.dx) or self.dx
        # v4.13.0 audit P1-C: preserve the anamorphic y-pitch across
        # ``Source.propagate``.  Pre-fix the wrapped result advertised
        # ``dy == dx`` (via the ``Source.__post_init__`` default) even
        # when the underlying field carried a distinct y-pitch -- the
        # v4.13.0 L3 sweep added the ``dy`` field to ``Source`` but
        # missed threading it through this dispatcher and the 5
        # classmethod factories.  Use the caller's ``output_dy`` kwarg
        # when given (matches ``output_dx`` precedence), else fall
        # through to ``self.dy``.
        out_dy = kwargs.get('output_dy', self.dy) or self.dy
        new_name = (self.name or 'Source')
        new_name = f'{new_name}->{method}'
        return Source(
            E=E_out, dx=out_dx, dy=out_dy, wavelength=self.wavelength,
            source_point=self.source_point, name=new_name,
        )

    # -- Factories that wrap the existing create_X functions ----------

    # 4.11.2 (audit round-3): the classmethod factories below pass
    # ``**factory_kwargs`` through to the underlying ``create_*`` calls
    # so callers can configure ``dy=``, ``dtype=``, ``normalize=``,
    # ``use_gpu=``, etc. without having to call the bare function
    # directly.  Pre-4.11.2 these kwargs were not propagated, so
    # anamorphic grids and single-precision fields silently fell back
    # to the create_*'s defaults.

    @classmethod
    def gaussian(cls, w0: float, N: int, dx: float, wavelength: float,
                  *, x0: float = 0.0, y0: float = 0.0,
                  source_point: _Tuple[float, float] = (0.0, 0.0),
                  name: _Optional[str] = None,
                  use_gpu: bool = False,
                  **factory_kwargs) -> 'Source':
        """Gaussian beam at the waist.  ``w0`` is the 1/e^2 intensity
        radius.  Extra ``factory_kwargs`` (e.g. ``dy=``, ``dtype=``)
        are forwarded to :func:`create_gaussian_beam`."""
        sigma = w0 / np.sqrt(2)
        E, _, _ = create_gaussian_beam(
            N, dx, wavelength, sigma=sigma, x0=x0, y0=y0,
            use_gpu=use_gpu, **factory_kwargs)
        # v4.13.0 audit P1-C: preserve anamorphic ``dy`` on the
        # returned Source.  ``create_gaussian_beam`` already consumed
        # ``dy`` from ``factory_kwargs`` to build the field on the
        # anamorphic grid; the pre-fix ``cls(...)`` call omitted ``dy``
        # so the wrapped Source advertised ``dy == dx`` even when the
        # E-field was shaped on a rectangular pitch.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'Gaussian(w0={w0:.2g}m)')

    @classmethod
    def plane_wave(cls, N: int, dx: float, wavelength: float,
                    *, angle_x: float = 0.0, angle_y: float = 0.0,
                    amplitude: float = 1.0,
                    source_point: _Tuple[float, float] = (0.0, 0.0),
                    name: _Optional[str] = None,
                    **factory_kwargs) -> 'Source':
        """Tilted plane wave (uses ``create_tilted_plane_wave``)."""
        E, _, _ = create_tilted_plane_wave(
            N, dx, wavelength, angle_x=angle_x, angle_y=angle_y,
            amplitude=amplitude, **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or 'PlaneWave')

    @classmethod
    def point_source(cls, N: int, dx: float, wavelength: float,
                      *, x0: float = 0.0, y0: float = 0.0,
                      z0: float = 0.0, amplitude: float = 1.0,
                      name: _Optional[str] = None,
                      **factory_kwargs) -> 'Source':
        """Spherical wave from a point at ``(x0, y0, z0)``.

        ``z0 < 0`` -> diverging wavefront (source before grid);
        ``z0 > 0`` -> converging wavefront (focus after grid).
        See :func:`create_point_source` for the sign-convention
        details.
        """
        E, _, _ = create_point_source(
            N, dx, wavelength, x0=x0, y0=y0, z0=z0,
            amplitude=amplitude, **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=(float(x0), float(y0)),
                   name=name or 'PointSource')

    @classmethod
    def top_hat(cls, diameter: float, N: int, dx: float, wavelength: float,
                  *, x0: float = 0.0, y0: float = 0.0,
                  source_point: _Tuple[float, float] = (0.0, 0.0),
                  name: _Optional[str] = None,
                  **factory_kwargs) -> 'Source':
        """Uniform circular aperture beam."""
        E, _, _ = create_top_hat_beam(
            N, dx, wavelength, diameter=diameter, x0=x0, y0=y0,
            **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'TopHat(D={diameter:.2g}m)')

    @classmethod
    def fiber_mode(cls, mode_field_diameter: float, N: int, dx: float,
                    wavelength: float, *, x0: float = 0.0, y0: float = 0.0,
                    na: float = 0.12,
                    source_point: _Tuple[float, float] = (0.0, 0.0),
                    name: _Optional[str] = None,
                    **factory_kwargs) -> 'Source':
        """Single-mode fiber output."""
        E, _, _ = create_fiber_mode(
            N, dx, wavelength, mode_field_diameter=mode_field_diameter,
            x0=x0, y0=y0, na=na, **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'Fiber(MFD={mode_field_diameter:.2g}m)')

