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
    support_tuple_N: bool = False,
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
        Grid size.  Tuple form is only accepted when
        ``support_tuple_N=True``.
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
    support_tuple_N : bool, default False
        Whether the calling factory accepts ``(Ny, Nx)`` tuple-form
        ``N``.  Only the mode-family helpers
        (``create_gaussian_beam``, ``create_hermite_gauss``,
        ``create_laguerre_gauss``) unpack tuples internally; every
        other ``create_*`` factory in ``sources/core.py`` calls
        ``np.arange(N)`` on the raw input and crashes on tuple-N
        with an obscure ``np.arange`` ``TypeError``.  v4.14.3
        (P1-NEW-5): the validator now rejects tuple-N up-front
        with a named error when the factory doesn't support it,
        so callers get a clear message instead of an inscrutable
        downstream crash.
    """
    # ``N`` -- positive integer, or a 2-tuple of positive integers
    # (only when the calling factory unpacks tuples).
    #
    # v4.15 (P2-VAL-1 / v4.14.2 carryover): explicitly reject ``bool``.
    # ``isinstance(True, (int, np.integer))`` returns True so the
    # pre-v4.15 check accepted ``N=True`` / ``N=False`` as 1 / 0.
    # ``N=False`` then hit the ``int(N) <= 0`` guard with a confusing
    # "N=0" error; ``N=True`` (a Boolean grid size, plainly wrong)
    # silently produced a 1x1 grid.  Boolean ``N`` is almost certainly
    # a caller bug (passing ``N=large_grid_flag and 1024`` -> 1024 only
    # when flag is truthy; ``N=use_gpu and 256`` -> ``False`` when GPU
    # is off; etc.), so the loudest correct action is a TypeError.
    if isinstance(N, bool):
        raise TypeError(
            f"{fn_name}: N must be a positive integer, got bool ({N!r}).  "
            f"This is almost certainly a caller bug (e.g. "
            f"``N=flag and grid_size``); pass an explicit integer.")
    if isinstance(N, (tuple, list)):
        if not support_tuple_N:
            raise TypeError(
                f"{fn_name}: tuple-form N=(Ny, Nx) is not supported by "
                f"this factory; got {N!r}.  Pass a single positive "
                f"integer for a square N x N grid.  Tuple-form N is "
                f"currently supported only by create_gaussian_beam, "
                f"create_hermite_gauss, and create_laguerre_gauss.")
        if len(N) != 2:
            raise ValueError(
                f"{fn_name}: N tuple form must be (Ny, Nx); got "
                f"length-{len(N)} sequence {N!r}.")
        Ny, Nx = N
        for label, n in (('Ny', Ny), ('Nx', Nx)):
            # v4.15: same bool short-circuit per-axis.
            if isinstance(n, bool):
                raise TypeError(
                    f"{fn_name}: {label} must be a positive integer, "
                    f"got bool ({n!r}).")
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
            tuple_hint = (' (or (Ny, Nx) tuple)' if support_tuple_N else '')
            raise ValueError(
                f"{fn_name}: N must be a positive integer"
                f"{tuple_hint}, got {type(N).__name__} ({N!r}).")
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
                          fn_name='create_gaussian_beam',
                          support_tuple_N=True)
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
                          fn_name='create_hermite_gauss',
                          support_tuple_N=True)
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
                          fn_name='create_laguerre_gauss',
                          support_tuple_N=True)
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
    # v4.15.0 (P2-VAL-2 from v4.14.2 audit): centralised input
    # validation at the entry point so failures here name THIS
    # factory.  Pre-v4.15 the helper transitively validated through
    # ``create_tilted_plane_wave``, but the error message named the
    # internal callee rather than ``create_multi_field_sources``,
    # leaking an internal name in user-facing tracebacks.
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_multi_field_sources')
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
    # v4.14.3 (P1-NEW-10 / Agent B): reject non-physical MFD.
    # ``mode_field_diameter <= 0`` silently flips sigma's sign (yielding
    # an exponential field that grows away from the centre -- not a
    # mode but a numerical singularity) or hits divide-by-zero in
    # ``sigma = w0/sqrt(2)`` when MFD=0.  Raise loudly so the caller
    # sees a clear error instead of an inf/NaN-laced output array.
    try:
        mfd_f = float(mode_field_diameter)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"create_fiber_mode: mode_field_diameter must be a positive "
            f"finite number [m]; got {mode_field_diameter!r}.") from exc
    if not np.isfinite(mfd_f) or mfd_f <= 0.0:
        raise ValueError(
            f"create_fiber_mode: mode_field_diameter must be a positive "
            f"finite number [m]; got {mfd_f}.")
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
    #
    # v4.14.3 (P1-NEW-4 / Agent B): the bare ``*args`` collector is a
    # silent footgun if a user passes 5 positional arguments in the
    # NEW canonical order ``(N, dx, wavelength, diameter,
    # divergence_angle)``: the shim re-routes ``wavelength`` (e.g.
    # 633e-9) as ``diameter`` and ``divergence_angle`` (e.g. 0.3 rad)
    # as ``wavelength``, producing a 633 nm-wide LED with a 0.3 m
    # "wavelength" -- and only a misleading DeprecationWarning.  The
    # post-remap scale-inversion check below catches the canonical-
    # order mistake by spotting the diameter/wavelength magnitude
    # inversion that distinguishes the two call forms.  PEP 570
    # ``/`` was considered but does not gate the ``*args`` collector
    # so adds no safety here, while it would force every existing
    # kwarg-based caller (incl. the v4.14.2 audit test
    # infrastructure) to drop N/dx/wavelength out of kwargs.
    if args:
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
        # v4.14.3 (P1-NEW-4): scale-inversion sanity check.  In a
        # legitimate legacy call ``_legacy_diameter`` is an emitting-
        # area diameter (typically 10-1000 um, i.e. 1e-5..1e-3 m) and
        # ``_legacy_wavelength`` is a vacuum wavelength (1e-7..3e-6 m
        # over the UV-MWIR range).  If a user instead passes 5
        # positionals in the NEW canonical order, ``_legacy_diameter
        # = wavelength`` (1e-7..3e-6) and ``_legacy_wavelength =
        # divergence_angle`` (typically O(0.1) rad).  The flag
        # ``_legacy_wavelength > _legacy_diameter * 10`` separates
        # the two forms: legacy callers never feed a wavelength 10x
        # larger than the diameter (a 1 um LED at 10 um wavelength
        # is a thermal emitter, not an "LED"), but the canonical-
        # order mistake yields divergence/wavelength = 0.3/633e-9 ~
        # 5e5, which trips the check loudly.
        try:
            _diam_f = float(_legacy_diameter)
            _wl_f = float(_legacy_wavelength)
        except (TypeError, ValueError):
            _diam_f = _wl_f = None
        if (_diam_f is not None and _wl_f is not None
                and _diam_f > 0 and _wl_f > _diam_f * 10):
            raise TypeError(
                "create_led_source: detected scale-inverted positional "
                "arguments (apparent wavelength {:.3e} m > 10x apparent "
                "diameter {:.3e} m).  This usually means the call was "
                "made in the NEW canonical positional order "
                "``create_led_source(N, dx, wavelength, diameter, "
                "divergence_angle)``, which is rejected since v4.14.3: "
                "the canonical form requires ``diameter`` and "
                "``divergence_angle`` to be passed as keyword "
                "arguments.  Use ``create_led_source(N, dx, "
                "wavelength, diameter=..., divergence_angle=...)``."
                .format(_wl_f, _diam_f))
        # v4.15 (P2-DEP-1): route the legacy-positional warning through
        # the shared ``_deprecation.warn_deprecated_signature`` helper
        # instead of inline ``warnings.warn``.  Same DeprecationWarning
        # category, same message intent; the helper guarantees a
        # consistent format (``... is deprecated since v4.14.2, will be
        # removed in v5.0; use ...``) and pin-tested removal version.
        from .._deprecation import warn_deprecated_signature
        warn_deprecated_signature(
            function='create_led_source',
            old_signature=(
                'create_led_source(N, dx, diameter, divergence_angle, '
                'wavelength, ...)'),
            new_signature=(
                'create_led_source(N, dx, wavelength, *, diameter=..., '
                'divergence_angle=..., ...)'),
            version_added='4.14.2',
            version_removed='5.0',
            stacklevel=3,
        )
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
            # v4.15.0 (P3-MSG from v4.14.2 audit): rewrite the "max 8
            # (legacy) or 3 (canonical)" wording.  The canonical form
            # caps positionals at 3 (``N, dx, wavelength``) and the
            # legacy form caps at 8 (``N, dx, diameter,
            # divergence_angle, wavelength, x0, y0, dtype``); the
            # previous message conflated the two limits in a single
            # opaque sentence.  Spell them out separately.
            raise TypeError(
                "create_led_source: too many positional arguments "
                f"({3 + len(args)}).  The legacy positional form "
                "accepts at most 8 positionals: ``(N, dx, diameter, "
                "divergence_angle, wavelength, x0, y0, dtype)``.  "
                "The canonical form accepts at most 3 positionals: "
                "``(N, dx, wavelength)`` -- ``diameter``, "
                "``divergence_angle``, and the other physical "
                "parameters are keyword-only.  Migrate any extras "
                "to keyword arguments.")

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
    # v4.14.3 (P1-NEW-9 / Agent B): reject non-physical cone angles
    # that produce silent zero / evanescent fields.
    # ``cone_angle <= 0`` -> ``sin(theta) <= 0`` -> uniform DC field
    # mis-labelled "Bessel beam".
    # ``cone_angle == pi/2`` -> ``k_r = k0`` (grazing -- non-propagating).
    # ``cone_angle > pi/2`` -> ``sin(theta)`` decreases again, so any
    # value > pi/2 is aliased to the same physical k_r as some smaller
    # angle (e.g. 2*pi/3 -> sin=sqrt(3)/2 same as pi/3) BUT it
    # represents an evanescent ``k_r > k0`` regime once the user
    # presumed the literal angle.  Enforce the strictly-propagating
    # window ``(0, pi/2)``.
    try:
        cone_f = float(cone_angle)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"create_bessel_beam: cone_angle must be a finite number "
            f"in (0, pi/2) rad; got {cone_angle!r}.") from exc
    if not np.isfinite(cone_f) or not (0.0 < cone_f < np.pi / 2.0):
        raise ValueError(
            f"create_bessel_beam: cone_angle={cone_f} must be in "
            f"(0, pi/2) rad to produce a propagating Bessel beam.  "
            f"Got sin(cone_angle)={np.sin(cone_f):.3e}, "
            f"k_r/k0={np.sin(cone_f):.3e}.")
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
# Schell-model partial-coherence sources (v4.15, ROADMAP v4.16 #9)
# ---------------------------------------------------------------------------

def create_gaussian_schell_source(
    *,
    N: int,
    dx: float,
    wavelength: float,
    w0: float,
    sigma_g: float,
    n_realizations: int = 16,
    dy: Optional[float] = None,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatially-incoherent Gaussian-Schell beam.

    Returns the ensemble-averaged amplitude envelope (square root of
    the time-averaged intensity) of ``n_realizations`` independent
    Gaussian beams whose phases are draws from a band-limited random
    field with coherence length ``sigma_g``.  Suitable as the input
    "source" amplitude for partial-coherence imaging routines that
    treat the source as an ensemble of mutually-incoherent
    realizations.

    A Gaussian-Schell beam has Gaussian intensity profile (1/e^2
    radius ``w0``) and Gaussian coherence kernel
    ``mu(r1, r2) = exp(-|r1-r2|^2 / (2 sigma_g^2))``.  In the
    fully-coherent limit ``sigma_g -> infinity`` the beam reduces to a
    deterministic Gaussian with waist ``w0``; in the fully-incoherent
    limit ``sigma_g -> 0`` each pixel has independent phase.

    Parameters
    ----------
    N : int
        Square grid size (N x N).
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    w0 : float
        Gaussian 1/e^2 intensity radius of the beam envelope [m].
    sigma_g : float
        Gaussian transverse coherence length [m].  Must be > 0;
        ``sigma_g >> w0`` approaches the coherent limit.
    n_realizations : int, default 16
        Number of independent ensemble draws to average.  Larger
        values give a smoother intensity envelope at the cost of
        compute time.  Must be >= 1.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    seed : int, optional
        RNG seed for reproducibility.  ``None`` leaves the local
        Generator unseeded.
    dtype : optional
        Complex dtype.

    Returns
    -------
    E : ndarray (complex)
        Amplitude envelope (sqrt of the ensemble-averaged intensity),
        cast to the requested complex dtype.  The phase is the phase
        of the ensemble mean of E_realizations -- not physically
        meaningful in the partial-coherence limit but kept for
        downstream type-uniformity.
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_gaussian_schell_source')
    if not (np.isfinite(w0) and w0 > 0):
        raise ValueError(
            f"create_gaussian_schell_source: w0 must be a positive "
            f"finite number [m]; got {w0}.")
    if not (np.isfinite(sigma_g) and sigma_g > 0):
        raise ValueError(
            f"create_gaussian_schell_source: sigma_g must be a "
            f"positive finite number [m]; got {sigma_g}.")
    if not isinstance(n_realizations, (int, np.integer)) or \
            isinstance(n_realizations, bool) or int(n_realizations) < 1:
        raise ValueError(
            f"create_gaussian_schell_source: n_realizations must be a "
            f"positive integer; got {n_realizations!r}.")

    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    target_dtype = _resolve_complex_dtype(dtype)

    # Gaussian intensity envelope (amplitude is sqrt of intensity).
    sigma_int = w0 / 2.0  # 1/e^2 intensity radius -> Gaussian sigma
    amp = np.exp(-(X * X + Y * Y) / (2.0 * sigma_int ** 2))

    # Random-phase ensemble: each realization multiplies the deterministic
    # Gaussian amplitude by a unit-modulus random phase whose two-point
    # correlation is controlled by sigma_g.  Standard recipe: build a
    # white-noise array and convolve (multiplicatively in Fourier) with
    # a Gaussian kernel of width 1/sigma_g; the result has correlation
    # length sigma_g in the spatial domain.
    rng = np.random.default_rng(seed)
    # Gaussian filter in k-space whose spatial coherence length is sigma_g.
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    spec_filter = np.exp(-(KX * KX + KY * KY) * (sigma_g ** 2) / 2.0)

    intensity_sum = np.zeros_like(amp, dtype=np.float64)
    coherent_sum = np.zeros_like(amp, dtype=np.complex128)
    nr = int(n_realizations)
    for _ in range(nr):
        # White-noise random field (real-valued normal), filter, take phase.
        white = rng.standard_normal((N, N))
        filtered_k = np.fft.fft2(white) * spec_filter
        filtered = np.real(np.fft.ifft2(filtered_k))
        # Normalise to unit RMS so the phase scale is sigma_g-independent.
        rms = np.sqrt(np.mean(filtered ** 2))
        if rms > 0:
            filtered = filtered / rms
        phase = filtered  # treat as the random phase, modulo 2 pi
        E_real = amp * np.exp(1j * phase)
        intensity_sum += np.abs(E_real) ** 2
        coherent_sum += E_real

    intensity_mean = intensity_sum / nr
    # Use the phase of the coherent mean to keep the output complex-valued
    # (useful for type-uniformity); the amplitude is the ensemble-averaged
    # intensity envelope.
    coherent_phase = np.angle(coherent_sum / nr)
    E = (np.sqrt(intensity_mean) * np.exp(1j * coherent_phase)).astype(
        target_dtype)
    return E, x, y


def create_schell_model_source(
    *,
    N: int,
    dx: float,
    wavelength: float,
    intensity_profile: np.ndarray,
    coherence_length: float,
    n_realizations: int = 16,
    dy: Optional[float] = None,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic Schell-model source with user-supplied intensity profile.

    A Schell-model source factors the mutual coherence function into
    an intensity term and a translation-invariant coherence kernel:
    ``J(r1, r2) = sqrt(I(r1)) sqrt(I(r2)) mu(r1 - r2)``.  This factory
    samples ``n_realizations`` random-phase realizations consistent
    with a Gaussian coherence kernel of width ``coherence_length`` and
    the supplied ``intensity_profile``.

    Parameters
    ----------
    N : int
        Square grid size (N x N).
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    intensity_profile : ndarray (N, N), real
        Time-averaged intensity profile of the source (any units;
        normalisation is preserved through to the output).
    coherence_length : float
        Gaussian transverse coherence length [m]; must be > 0.
    n_realizations : int, default 16
        Number of independent ensemble draws.
    dy : float, optional
        Grid spacing in y [m].
    seed : int, optional
        RNG seed.
    dtype : optional
        Complex dtype.

    Returns
    -------
    E : ndarray (complex)
        Amplitude envelope of the ensemble-averaged intensity, cast to
        the requested complex dtype.
    x, y : ndarray
        1-D coordinate axes [m].
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_schell_model_source')
    if not (np.isfinite(coherence_length) and coherence_length > 0):
        raise ValueError(
            f"create_schell_model_source: coherence_length must be a "
            f"positive finite number [m]; got {coherence_length}.")
    if not isinstance(n_realizations, (int, np.integer)) or \
            isinstance(n_realizations, bool) or int(n_realizations) < 1:
        raise ValueError(
            f"create_schell_model_source: n_realizations must be a "
            f"positive integer; got {n_realizations!r}.")
    I = np.asarray(intensity_profile, dtype=float)
    if I.shape != (N, N):
        raise ValueError(
            f"create_schell_model_source: intensity_profile must have "
            f"shape (N, N) = ({N}, {N}); got {I.shape}.")
    if np.any(I < 0):
        raise ValueError(
            "create_schell_model_source: intensity_profile must be "
            "non-negative.")

    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    target_dtype = _resolve_complex_dtype(dtype)
    amp = np.sqrt(I)

    rng = np.random.default_rng(seed)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    spec_filter = np.exp(
        -(KX * KX + KY * KY) * (coherence_length ** 2) / 2.0)

    intensity_sum = np.zeros_like(amp, dtype=np.float64)
    coherent_sum = np.zeros_like(amp, dtype=np.complex128)
    nr = int(n_realizations)
    for _ in range(nr):
        white = rng.standard_normal((N, N))
        filtered_k = np.fft.fft2(white) * spec_filter
        filtered = np.real(np.fft.ifft2(filtered_k))
        rms = np.sqrt(np.mean(filtered ** 2))
        if rms > 0:
            filtered = filtered / rms
        phase = filtered
        E_real = amp * np.exp(1j * phase)
        intensity_sum += np.abs(E_real) ** 2
        coherent_sum += E_real

    intensity_mean = intensity_sum / nr
    coherent_phase = np.angle(coherent_sum / nr)
    E = (np.sqrt(intensity_mean) * np.exp(1j * coherent_phase)).astype(
        target_dtype)
    return E, x, y


def create_annular_incoherent_source(
    *,
    N: int,
    dx: float,
    wavelength: float,
    inner_radius: float,
    outer_radius: float,
    n_realizations: int = 16,
    dy: Optional[float] = None,
    seed: Optional[int] = None,
    dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annular (ring) source with non-zero source size for partial-
    coherence integration (v4.15, ROADMAP v4.16 #11).

    Distinct from :func:`create_annular_beam`, which returns the
    deterministic coherent annular field.  This factory samples
    ``n_realizations`` independent random-phase realizations whose
    common amplitude is the unit-norm annular indicator function;
    the returned ``E`` is the amplitude envelope (sqrt of the
    ensemble-averaged intensity).

    Parameters
    ----------
    N : int
        Square grid size (N x N).
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    inner_radius : float
        Annulus inner radius [m]; must be >= 0.
    outer_radius : float
        Annulus outer radius [m]; must be > inner_radius and finite.
    n_realizations : int, default 16
        Number of independent ensemble draws.
    dy : float, optional
        Grid spacing in y [m].
    seed : int, optional
        RNG seed.
    dtype : optional
        Complex dtype.

    Returns
    -------
    E : ndarray (complex)
        Amplitude envelope of the ensemble-averaged intensity.
    x, y : ndarray
        1-D coordinate axes [m].
    """
    _validate_grid_params(N, dx, wavelength, dy=dy,
                          fn_name='create_annular_incoherent_source')
    if not (np.isfinite(inner_radius) and inner_radius >= 0):
        raise ValueError(
            f"create_annular_incoherent_source: inner_radius must be a "
            f"non-negative finite number [m]; got {inner_radius}.")
    if not (np.isfinite(outer_radius) and outer_radius > 0):
        raise ValueError(
            f"create_annular_incoherent_source: outer_radius must be a "
            f"positive finite number [m]; got {outer_radius}.")
    if outer_radius <= inner_radius:
        raise ValueError(
            f"create_annular_incoherent_source: outer_radius "
            f"({outer_radius}) must be strictly greater than "
            f"inner_radius ({inner_radius}).")
    if not isinstance(n_realizations, (int, np.integer)) or \
            isinstance(n_realizations, bool) or int(n_realizations) < 1:
        raise ValueError(
            f"create_annular_incoherent_source: n_realizations must be "
            f"a positive integer; got {n_realizations!r}.")

    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    target_dtype = _resolve_complex_dtype(dtype)
    r = np.sqrt(X * X + Y * Y)
    mask = (r >= inner_radius) & (r <= outer_radius)
    amp = mask.astype(np.float64)
    # Normalise so the integrated power is 1, matching create_annular_beam.
    norm = np.sqrt(np.sum(amp ** 2) * dx * dy)
    if norm > 0:
        amp = amp / norm

    rng = np.random.default_rng(seed)
    intensity_sum = np.zeros_like(amp, dtype=np.float64)
    coherent_sum = np.zeros_like(amp, dtype=np.complex128)
    nr = int(n_realizations)
    for _ in range(nr):
        # Independent uniform phase per pixel (i.e. fully spatially
        # incoherent at the source plane -- the standard "non-zero
        # source size for partial-coherence integration" recipe).
        phase = rng.uniform(-np.pi, np.pi, (N, N))
        E_real = amp * np.exp(1j * phase)
        intensity_sum += np.abs(E_real) ** 2
        coherent_sum += E_real

    intensity_mean = intensity_sum / nr
    coherent_phase = np.angle(coherent_sum / nr)
    E = (np.sqrt(intensity_mean) * np.exp(1j * coherent_phase)).astype(
        target_dtype)
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

    # -----------------------------------------------------------------
    # v4.15 (ROADMAP v4.15 #2): size-arg normalisation on the 5
    # Source.* factory classmethods.
    #
    # Pre-v4.15 the 5 factories had inconsistent positional order:
    #   - ``Source.gaussian(w0, N, dx, wavelength)``  -- size first
    #   - ``Source.plane_wave(N, dx, wavelength)``    -- N first
    #   - ``Source.point_source(N, dx, wavelength)``  -- N first
    #   - ``Source.top_hat(diameter, N, dx, wavelength)`` -- size first
    #   - ``Source.fiber_mode(mfd, N, dx, wavelength)``  -- size first
    #
    # v4.15 picks the canonical order
    # ``Source.method(*, N, dx, wavelength, <size_kwargs>)`` (kwarg-only
    # with the ``*`` separator).  The legacy positional form is still
    # accepted for one release with a ``DeprecationWarning`` routed
    # through ``_deprecation.warn_deprecated_signature``; removal is
    # scheduled for v5.0.
    #
    # The three already-kwarg-only factories (``plane_wave``,
    # ``point_source``) keep their existing signature; the only change
    # for them is that they now appear under the canonical
    # ``Source.method(*, N, dx, wavelength, ...)`` umbrella in the
    # docs and the factory-validation parametrize list.
    # -----------------------------------------------------------------

    @classmethod
    def gaussian(cls, *args,
                  w0: _Optional[float] = None,
                  N: _Optional[int] = None,
                  dx: _Optional[float] = None,
                  wavelength: _Optional[float] = None,
                  x0: float = 0.0, y0: float = 0.0,
                  source_point: _Tuple[float, float] = (0.0, 0.0),
                  name: _Optional[str] = None,
                  use_gpu: bool = False,
                  **factory_kwargs) -> 'Source':
        """Gaussian beam at the waist.

        Canonical signature (v4.15+):
            ``Source.gaussian(*, N, dx, wavelength, w0, ...)``

        ``w0`` is the 1/e^2 intensity radius.  Extra
        ``factory_kwargs`` (e.g. ``dy=``, ``dtype=``) are forwarded to
        :func:`create_gaussian_beam`.

        Legacy signature (deprecated since v4.15, removal v5.0):
            ``Source.gaussian(w0, N, dx, wavelength, ...)``
        """
        # Legacy positional shim: pre-v4.15 callers passed
        # ``(w0, N, dx, wavelength)`` positionally.  Detect via *args
        # and emit a DeprecationWarning before remapping.
        if args:
            from .._deprecation import warn_deprecated_signature
            warn_deprecated_signature(
                function='Source.gaussian',
                old_signature='Source.gaussian(w0, N, dx, wavelength, ...)',
                new_signature=(
                    'Source.gaussian(*, N, dx, wavelength, w0, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=3,
            )
            if len(args) > 4:
                raise TypeError(
                    "Source.gaussian (legacy positional form): too many "
                    f"positional arguments; got {len(args)}, max 4 "
                    "(w0, N, dx, wavelength).")
            legacy = (None, None, None, None)
            legacy = args + legacy[len(args):]
            _l_w0, _l_N, _l_dx, _l_wl = legacy
            # Reject overlap with canonical kwargs.
            if w0 is not None and _l_w0 is not None:
                raise TypeError("Source.gaussian: 'w0' supplied both "
                                "positionally and as keyword.")
            if N is not None and _l_N is not None:
                raise TypeError("Source.gaussian: 'N' supplied both "
                                "positionally and as keyword.")
            if dx is not None and _l_dx is not None:
                raise TypeError("Source.gaussian: 'dx' supplied both "
                                "positionally and as keyword.")
            if wavelength is not None and _l_wl is not None:
                raise TypeError("Source.gaussian: 'wavelength' supplied "
                                "both positionally and as keyword.")
            if w0 is None:
                w0 = _l_w0
            if N is None:
                N = _l_N
            if dx is None:
                dx = _l_dx
            if wavelength is None:
                wavelength = _l_wl
        if w0 is None:
            raise TypeError(
                "Source.gaussian: missing required keyword argument "
                "'w0' (the 1/e^2 intensity radius in metres).")
        if N is None:
            raise TypeError(
                "Source.gaussian: missing required keyword argument 'N'.")
        if dx is None:
            raise TypeError(
                "Source.gaussian: missing required keyword argument 'dx'.")
        if wavelength is None:
            raise TypeError(
                "Source.gaussian: missing required keyword argument "
                "'wavelength'.")
        sigma = w0 / np.sqrt(2)
        E, _, _ = create_gaussian_beam(
            N, dx, wavelength, sigma=sigma, x0=x0, y0=y0,
            use_gpu=use_gpu, **factory_kwargs)
        # v4.13.0 audit P1-C: preserve anamorphic ``dy`` on the
        # returned Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'Gaussian(w0={w0:.2g}m)')

    @classmethod
    def plane_wave(cls, *args,
                    N: _Optional[int] = None,
                    dx: _Optional[float] = None,
                    wavelength: _Optional[float] = None,
                    angle_x: float = 0.0, angle_y: float = 0.0,
                    amplitude: float = 1.0,
                    source_point: _Tuple[float, float] = (0.0, 0.0),
                    name: _Optional[str] = None,
                    **factory_kwargs) -> 'Source':
        """Tilted plane wave (uses ``create_tilted_plane_wave``).

        Canonical signature (v4.15+):
            ``Source.plane_wave(*, N, dx, wavelength, ...)``

        Legacy signature (deprecated since v4.15, removal v5.0):
            ``Source.plane_wave(N, dx, wavelength, ...)``
        """
        if args:
            from .._deprecation import warn_deprecated_signature
            warn_deprecated_signature(
                function='Source.plane_wave',
                old_signature='Source.plane_wave(N, dx, wavelength, ...)',
                new_signature=(
                    'Source.plane_wave(*, N, dx, wavelength, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=3,
            )
            if len(args) > 3:
                raise TypeError(
                    "Source.plane_wave (legacy positional form): too "
                    f"many positional arguments; got {len(args)}, max 3 "
                    "(N, dx, wavelength).")
            legacy = (None, None, None)
            legacy = args + legacy[len(args):]
            _l_N, _l_dx, _l_wl = legacy
            if N is not None and _l_N is not None:
                raise TypeError("Source.plane_wave: 'N' supplied both "
                                "positionally and as keyword.")
            if dx is not None and _l_dx is not None:
                raise TypeError("Source.plane_wave: 'dx' supplied both "
                                "positionally and as keyword.")
            if wavelength is not None and _l_wl is not None:
                raise TypeError("Source.plane_wave: 'wavelength' "
                                "supplied both positionally and as keyword.")
            if N is None:
                N = _l_N
            if dx is None:
                dx = _l_dx
            if wavelength is None:
                wavelength = _l_wl
        if N is None:
            raise TypeError(
                "Source.plane_wave: missing required keyword argument 'N'.")
        if dx is None:
            raise TypeError(
                "Source.plane_wave: missing required keyword argument 'dx'.")
        if wavelength is None:
            raise TypeError(
                "Source.plane_wave: missing required keyword argument "
                "'wavelength'.")
        E, _, _ = create_tilted_plane_wave(
            N, dx, wavelength, angle_x=angle_x, angle_y=angle_y,
            amplitude=amplitude, **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or 'PlaneWave')

    @classmethod
    def point_source(cls, *args,
                      N: _Optional[int] = None,
                      dx: _Optional[float] = None,
                      wavelength: _Optional[float] = None,
                      x0: float = 0.0, y0: float = 0.0,
                      z0: float = 0.0, amplitude: float = 1.0,
                      name: _Optional[str] = None,
                      **factory_kwargs) -> 'Source':
        """Spherical wave from a point at ``(x0, y0, z0)``.

        Canonical signature (v4.15+):
            ``Source.point_source(*, N, dx, wavelength, ...)``

        Legacy signature (deprecated since v4.15, removal v5.0):
            ``Source.point_source(N, dx, wavelength, ...)``

        ``z0 < 0`` -> diverging wavefront (source before grid);
        ``z0 > 0`` -> converging wavefront (focus after grid).
        See :func:`create_point_source` for the sign-convention details.
        """
        if args:
            from .._deprecation import warn_deprecated_signature
            warn_deprecated_signature(
                function='Source.point_source',
                old_signature='Source.point_source(N, dx, wavelength, ...)',
                new_signature=(
                    'Source.point_source(*, N, dx, wavelength, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=3,
            )
            if len(args) > 3:
                raise TypeError(
                    "Source.point_source (legacy positional form): "
                    f"too many positional arguments; got {len(args)}, "
                    "max 3 (N, dx, wavelength).")
            legacy = (None, None, None)
            legacy = args + legacy[len(args):]
            _l_N, _l_dx, _l_wl = legacy
            if N is not None and _l_N is not None:
                raise TypeError("Source.point_source: 'N' supplied both "
                                "positionally and as keyword.")
            if dx is not None and _l_dx is not None:
                raise TypeError("Source.point_source: 'dx' supplied "
                                "both positionally and as keyword.")
            if wavelength is not None and _l_wl is not None:
                raise TypeError("Source.point_source: 'wavelength' "
                                "supplied both positionally and as keyword.")
            if N is None:
                N = _l_N
            if dx is None:
                dx = _l_dx
            if wavelength is None:
                wavelength = _l_wl
        if N is None:
            raise TypeError(
                "Source.point_source: missing required keyword argument 'N'.")
        if dx is None:
            raise TypeError(
                "Source.point_source: missing required keyword argument 'dx'.")
        if wavelength is None:
            raise TypeError(
                "Source.point_source: missing required keyword argument "
                "'wavelength'.")
        E, _, _ = create_point_source(
            N, dx, wavelength, x0=x0, y0=y0, z0=z0,
            amplitude=amplitude, **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=(float(x0), float(y0)),
                   name=name or 'PointSource')

    @classmethod
    def top_hat(cls, *args,
                  diameter: _Optional[float] = None,
                  N: _Optional[int] = None,
                  dx: _Optional[float] = None,
                  wavelength: _Optional[float] = None,
                  x0: float = 0.0, y0: float = 0.0,
                  source_point: _Tuple[float, float] = (0.0, 0.0),
                  name: _Optional[str] = None,
                  **factory_kwargs) -> 'Source':
        """Uniform circular aperture beam.

        Canonical signature (v4.15+):
            ``Source.top_hat(*, N, dx, wavelength, diameter, ...)``

        Legacy signature (deprecated since v4.15, removal v5.0):
            ``Source.top_hat(diameter, N, dx, wavelength, ...)``
        """
        if args:
            from .._deprecation import warn_deprecated_signature
            warn_deprecated_signature(
                function='Source.top_hat',
                old_signature='Source.top_hat(diameter, N, dx, wavelength, ...)',
                new_signature=(
                    'Source.top_hat(*, N, dx, wavelength, diameter, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=3,
            )
            if len(args) > 4:
                raise TypeError(
                    "Source.top_hat (legacy positional form): too many "
                    f"positional arguments; got {len(args)}, max 4 "
                    "(diameter, N, dx, wavelength).")
            legacy = (None, None, None, None)
            legacy = args + legacy[len(args):]
            _l_diameter, _l_N, _l_dx, _l_wl = legacy
            if diameter is not None and _l_diameter is not None:
                raise TypeError("Source.top_hat: 'diameter' supplied "
                                "both positionally and as keyword.")
            if N is not None and _l_N is not None:
                raise TypeError("Source.top_hat: 'N' supplied both "
                                "positionally and as keyword.")
            if dx is not None and _l_dx is not None:
                raise TypeError("Source.top_hat: 'dx' supplied both "
                                "positionally and as keyword.")
            if wavelength is not None and _l_wl is not None:
                raise TypeError("Source.top_hat: 'wavelength' supplied "
                                "both positionally and as keyword.")
            if diameter is None:
                diameter = _l_diameter
            if N is None:
                N = _l_N
            if dx is None:
                dx = _l_dx
            if wavelength is None:
                wavelength = _l_wl
        if diameter is None:
            raise TypeError(
                "Source.top_hat: missing required keyword argument "
                "'diameter' (beam diameter in metres).")
        if N is None:
            raise TypeError(
                "Source.top_hat: missing required keyword argument 'N'.")
        if dx is None:
            raise TypeError(
                "Source.top_hat: missing required keyword argument 'dx'.")
        if wavelength is None:
            raise TypeError(
                "Source.top_hat: missing required keyword argument "
                "'wavelength'.")
        E, _, _ = create_top_hat_beam(
            N, dx, wavelength, diameter=diameter, x0=x0, y0=y0,
            **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'TopHat(D={diameter:.2g}m)')

    @classmethod
    def fiber_mode(cls, *args,
                    mode_field_diameter: _Optional[float] = None,
                    N: _Optional[int] = None,
                    dx: _Optional[float] = None,
                    wavelength: _Optional[float] = None,
                    x0: float = 0.0, y0: float = 0.0,
                    na: float = 0.12,
                    source_point: _Tuple[float, float] = (0.0, 0.0),
                    name: _Optional[str] = None,
                    **factory_kwargs) -> 'Source':
        """Single-mode fiber output.

        Canonical signature (v4.15+):
            ``Source.fiber_mode(*, N, dx, wavelength,
            mode_field_diameter, ...)``

        Legacy signature (deprecated since v4.15, removal v5.0):
            ``Source.fiber_mode(mode_field_diameter, N, dx, wavelength,
            ...)``
        """
        if args:
            from .._deprecation import warn_deprecated_signature
            warn_deprecated_signature(
                function='Source.fiber_mode',
                old_signature=(
                    'Source.fiber_mode(mode_field_diameter, N, dx, '
                    'wavelength, ...)'),
                new_signature=(
                    'Source.fiber_mode(*, N, dx, wavelength, '
                    'mode_field_diameter, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=3,
            )
            if len(args) > 4:
                raise TypeError(
                    "Source.fiber_mode (legacy positional form): too "
                    f"many positional arguments; got {len(args)}, max "
                    "4 (mode_field_diameter, N, dx, wavelength).")
            legacy = (None, None, None, None)
            legacy = args + legacy[len(args):]
            _l_mfd, _l_N, _l_dx, _l_wl = legacy
            if mode_field_diameter is not None and _l_mfd is not None:
                raise TypeError(
                    "Source.fiber_mode: 'mode_field_diameter' supplied "
                    "both positionally and as keyword.")
            if N is not None and _l_N is not None:
                raise TypeError("Source.fiber_mode: 'N' supplied both "
                                "positionally and as keyword.")
            if dx is not None and _l_dx is not None:
                raise TypeError("Source.fiber_mode: 'dx' supplied both "
                                "positionally and as keyword.")
            if wavelength is not None and _l_wl is not None:
                raise TypeError("Source.fiber_mode: 'wavelength' "
                                "supplied both positionally and as keyword.")
            if mode_field_diameter is None:
                mode_field_diameter = _l_mfd
            if N is None:
                N = _l_N
            if dx is None:
                dx = _l_dx
            if wavelength is None:
                wavelength = _l_wl
        if mode_field_diameter is None:
            raise TypeError(
                "Source.fiber_mode: missing required keyword argument "
                "'mode_field_diameter' (the 1/e^2 intensity diameter "
                "in metres).")
        if N is None:
            raise TypeError(
                "Source.fiber_mode: missing required keyword argument 'N'.")
        if dx is None:
            raise TypeError(
                "Source.fiber_mode: missing required keyword argument 'dx'.")
        if wavelength is None:
            raise TypeError(
                "Source.fiber_mode: missing required keyword argument "
                "'wavelength'.")
        E, _, _ = create_fiber_mode(
            N, dx, wavelength, mode_field_diameter=mode_field_diameter,
            x0=x0, y0=y0, na=na, **factory_kwargs)
        # v4.13.0 audit P1-C: thread ``dy`` to the wrapped Source.
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'Fiber(MFD={mode_field_diameter:.2g}m)')

    # -----------------------------------------------------------------
    # v4.15 (ROADMAP v4.16 #9, #11): two new partial-coherence factories
    # for the Schell-model family and the annular-incoherent source.
    # -----------------------------------------------------------------

    @classmethod
    def gaussian_schell(cls, *, N: int, dx: float, wavelength: float,
                         w0: float, sigma_g: float,
                         n_realizations: int = 16,
                         source_point: _Tuple[float, float] = (0.0, 0.0),
                         name: _Optional[str] = None,
                         seed: _Optional[int] = None,
                         **factory_kwargs) -> 'Source':
        """Gaussian-Schell partial-coherence source.

        Wraps :func:`create_gaussian_schell_source`.  Returns a
        :class:`Source` whose ``E`` is the ensemble-averaged amplitude
        envelope (intensity sqrt) over ``n_realizations`` independent
        random-phase Gaussian draws -- suitable for downstream
        partial-coherence integration.
        """
        E, _, _ = create_gaussian_schell_source(
            N=N, dx=dx, wavelength=wavelength, w0=w0, sigma_g=sigma_g,
            n_realizations=n_realizations, seed=seed, **factory_kwargs)
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or (
                       f'GaussianSchell(w0={w0:.2g}m, '
                       f'sigma_g={sigma_g:.2g}m)'))

    @classmethod
    def schell_model(cls, *, N: int, dx: float, wavelength: float,
                      intensity_profile: np.ndarray,
                      coherence_length: float,
                      n_realizations: int = 16,
                      source_point: _Tuple[float, float] = (0.0, 0.0),
                      name: _Optional[str] = None,
                      seed: _Optional[int] = None,
                      **factory_kwargs) -> 'Source':
        """Generic Schell-model partial-coherence source.

        Wraps :func:`create_schell_model_source`.  Returns a
        :class:`Source` whose ``E`` is the ensemble-averaged amplitude
        envelope under a user-supplied intensity profile and Gaussian
        coherence kernel.
        """
        E, _, _ = create_schell_model_source(
            N=N, dx=dx, wavelength=wavelength,
            intensity_profile=intensity_profile,
            coherence_length=coherence_length,
            n_realizations=n_realizations, seed=seed,
            **factory_kwargs)
        return cls(E=E, dx=dx, dy=factory_kwargs.get('dy', dx),
                   wavelength=wavelength,
                   source_point=source_point,
                   name=name or (
                       f'SchellModel(lc={coherence_length:.2g}m)'))

