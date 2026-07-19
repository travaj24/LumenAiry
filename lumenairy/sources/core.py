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

import importlib.util as _importlib_util
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

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

    # v5.4.6 (audit F-39): reject non-physical scale parameters -- sigma=0
    # gives a divide-by-zero / NaN-laced field and sigma<0 silently
    # produces the same beam as |sigma|, both of which were accepted
    # without warning before.
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            f"create_gaussian_beam: sigma must be positive and finite, "
            f"got {sigma}.")
    # Gaussian amplitude: exp(-r^2 / (2 sigma^2))
    target_dtype = _resolve_complex_dtype(dtype)
    E = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    E = E.astype(target_dtype)

    if normalize == 'peak':
        # v5.4.6 (audit F-38): divide by the ACTUAL peak so the field is
        # unit-peak even when the beam centre (x0, y0) falls off the
        # sampled grid (where exp(0) is never sampled).  Strict no-op for
        # an on-grid peak (max is already 1).
        mx = float(xp.abs(E).max())
        if mx > 0:
            E = E / mx
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
    # SRC-2: w0 was unvalidated -- w0 = 0/NaN silently yields a NaN-laced
    # field and w0 < 0 a sign-flipped mode.
    if not (np.isfinite(w0) and w0 > 0):
        raise ValueError(
            f"create_hermite_gauss: w0 must be a positive finite beam "
            f"waist [m]; got {w0}.")
    if dy is None:
        dy = dx
    if isinstance(N, (tuple, list)):
        Ny, Nx = int(N[0]), int(N[1])
    else:
        Ny = Nx = int(N)
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid

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
    # SRC-2: w0 was unvalidated (see create_hermite_gauss).
    if not (np.isfinite(w0) and w0 > 0):
        raise ValueError(
            f"create_laguerre_gauss: w0 must be a positive finite beam "
            f"waist [m]; got {w0}.")
    if dy is None:
        dy = dx
    if isinstance(N, (tuple, list)):
        Ny, Nx = int(N[0]), int(N[1])
    else:
        Ny = Nx = int(N)
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid

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
    # v5.4.6 (audit F-40): kx = k0 sin(angle_x) and ky = k0 sin(angle_y) are
    # applied independently; if sin^2(ax) + sin^2(ay) > 1 the transverse
    # wavevector exceeds k0 and the wave is evanescent (non-propagating),
    # which a propagating-plane-wave construction cannot represent.
    _kt2 = float(np.sin(angle_x) ** 2 + np.sin(angle_y) ** 2)
    if _kt2 > 1.0:
        raise ValueError(
            f"create_tilted_plane_wave: sin^2(angle_x) + sin^2(angle_y) = "
            f"{_kt2:.4f} > 1 -- the transverse wavevector exceeds k0 and the "
            f"wave would be evanescent.  Reduce the tilt angle(s).")
    # v5.24.4 (audit S3-3): transverse-Nyquist (edge-NA) guard.  The
    # linear phase ramp advances by k0*sin(angle)*dx per pixel; once
    # |sin(angle)| exceeds lambda/(2*dx) that step passes pi/px, the
    # ramp is undersampled, and it aliases (folds) to a SMALLER
    # effective angle with no error raised.  Checked per-axis (x uses
    # dx, y uses dy) -- warning-only, no numerics change.
    _nyq_x = wavelength / (2.0 * dx)
    _nyq_y = wavelength / (2.0 * dy)
    _sx, _sy = abs(float(np.sin(angle_x))), abs(float(np.sin(angle_y)))
    if _sx > _nyq_x or _sy > _nyq_y:
        import warnings
        warnings.warn(
            f"create_tilted_plane_wave: |sin(angle)| exceeds the spatial "
            f"Nyquist limit lambda/(2*dx) (x: {_sx:.3f} vs {_nyq_x:.3f}; "
            f"y: {_sy:.3f} vs {_nyq_y:.3f}) -- the linear phase ramp is "
            f"undersampled and aliases to a smaller effective angle.  "
            f"Reduce the tilt angle or the grid spacing.",
            RuntimeWarning, stacklevel=2,
        )
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid
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
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid
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
    # v5.24.4 (audit S3-3): transverse-Nyquist (edge-NA) guard.  The
    # spherical-wave chirp's local transverse frequency at radius rho is
    # k0*(rho/r); the per-pixel phase step k0*(rho/r)*dx passes pi
    # (undersampled -> aliases) once the local-NA rho/sqrt(rho^2+z0^2)
    # exceeds the Nyquist limit lambda/(2*dx).  Unlike the tilted-plane-
    # wave ramp -- which aliases GLOBALLY the instant sin(angle) crosses
    # Nyquist (guarded exactly above) -- the spherical chirp aliases only
    # the pixels OUTSIDE the aliasing radius, a graded edge effect: at a
    # hairline crossing just the extreme-corner ring folds, which is
    # benign.  So warn only on CLEAR undersampling: edge local-NA past
    # _NYQ_MARGIN * Nyquist (the audit's flagged case is ~2.5x over;
    # a corner ring at ~1.3x is not worth an alarm).  Evaluated per-axis
    # at the transverse extreme (y-term dropped -> tightest per-axis
    # test; reproduces the audit edge-NA probe).  Warning-only, no
    # numerics change.
    _NYQ_MARGIN = 1.5
    _x_off = max(abs(float(x[0]) - x0), abs(float(x[-1]) - x0))
    _y_off = max(abs(float(y[0]) - y0), abs(float(y[-1]) - y0))
    _na_x = _x_off / np.sqrt(_x_off ** 2 + z0 ** 2) if (_x_off or z0) else 0.0
    _na_y = _y_off / np.sqrt(_y_off ** 2 + z0 ** 2) if (_y_off or z0) else 0.0
    _nyq_x = wavelength / (2.0 * dx)
    _nyq_y = wavelength / (2.0 * dy)
    if _na_x > _NYQ_MARGIN * _nyq_x or _na_y > _NYQ_MARGIN * _nyq_y:
        import warnings
        warnings.warn(
            f"create_point_source: spherical-wave edge local-NA clearly "
            f"exceeds the spatial Nyquist limit lambda/(2*dx) (x: "
            f"{_na_x:.3f} vs {_nyq_x:.3f}; y: {_na_y:.3f} vs {_nyq_y:.3f}) "
            f"-- the outer field aliases to a spurious smaller angle.  "
            f"Increase |z0| or dx, or reduce N.",
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
    # SRC-3: an empty field_angles silently returned (sources=[], x=None,
    # y=None) -- raise instead so callers do not propagate None axes.
    if len(field_angles) == 0:
        raise ValueError(
            "create_multi_field_sources: field_angles is empty; provide at "
            "least one field angle.")
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
    # SRC-2: diameter was unvalidated -- diameter <= 0 yields an all-zero
    # field, and the ``norm > 0`` guard then silently skips normalisation
    # (a silent zero source downstream).
    if not (np.isfinite(diameter) and diameter > 0):
        raise ValueError(
            f"create_top_hat_beam: diameter must be a positive finite "
            f"value [m]; got {diameter}.")
    # 4.10: honour caller-supplied dy (anamorphic grid).  Pre-4.10
    # hard-coded dy = dx and used dx**2 for the area element, silently
    # ignoring caller-supplied dy on top-hat / annular / Bessel sources
    # only.  Defaults to dy = dx for back-compat.
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid
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
    # SRC-2: neither diameter was validated and there was no inner < outer
    # check -- an inverted annulus silently returned the all-zero field.
    # Mirror the incoherent sibling create_annular_incoherent_source.
    if not (np.isfinite(inner_diameter) and inner_diameter >= 0):
        raise ValueError(
            f"create_annular_beam: inner_diameter must be a non-negative "
            f"finite value [m]; got {inner_diameter}.")
    if not (np.isfinite(outer_diameter) and outer_diameter > 0):
        raise ValueError(
            f"create_annular_beam: outer_diameter must be a positive finite "
            f"value [m]; got {outer_diameter}.")
    if not (outer_diameter > inner_diameter):
        raise ValueError(
            f"create_annular_beam: outer_diameter ({outer_diameter}) must be "
            f"strictly greater than inner_diameter ({inner_diameter}).")
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid
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
    """Single-mode fiber output (Gaussian near-field set by the
    mode-field diameter).

    The mode-field diameter (MFD) is the 1/e^2 intensity diameter.
    The near-field is a Gaussian with waist ``w0 = MFD/2``; its
    far-field divergence half-angle follows from ``w0`` and
    ``wavelength`` alone (``theta = wavelength/(pi*w0)``).  The ``na``
    argument does NOT set the divergence and does NOT affect the
    returned field -- it is advisory only and merely triggers the
    >0.2 Gaussian-approximation warning (see the ``na`` parameter
    note below).

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
    """Extended LED-style source (incoherent; returns the intensity
    envelope as a complex field for use with partial-coherence
    imaging).

    v5.4.6 (audit P3-11): the returned ``source_angles`` are sampled
    UNIFORMLY over the divergence cone (concentric rings), NOT Lambertian
    (cos-theta) weighted, and ``extended_source_image`` defaults every
    angle to unit weight -- so this models a uniform-RADIANCE disk in
    angle, not a true Lambertian (I ~ cos theta) emitter.  For a
    Lambertian profile, weight each returned angle by
    ``cos(sqrt(ax^2 + ay^2))`` in the downstream integration.

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
    # v5.4.7 (audit AUDIT_V5_4_6 gap #4): early grid validation, within the
    # meta-pin's body-head window (the full validation also runs after the
    # legacy shim resolves the wavelength; the shim never remaps N/dx).
    if not args and wavelength is not None:
        _validate_grid_params(N, dx, wavelength, dy=dy,
                              fn_name='create_led_source')
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
        # v4.15.3 (audit P1-NEW-F1-2 sweep): stacklevel=4 selects the
        # user-code frame.  Chain: warnings.warn (in _emit) [1] ->
        # _emit body [2] -> warn_deprecated_signature body [3] ->
        # create_led_source body [4] -> user code [5 = target].
        # Pre-v4.15.3 stacklevel=3 landed inside create_led_source
        # itself.
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
            stacklevel=4,
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
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    k_r = 2 * np.pi / wavelength * np.sin(cone_angle)
    E = j0(k_r * r).astype(_resolve_complex_dtype(dtype))
    return E, x, y


# ---------------------------------------------------------------------------
# Schell-model partial-coherence sources (v4.15, ROADMAP v4.16 #9)
#
# v4.15.1 (P0-NEW-2): redesigned to actually deliver partial coherence.
# The v4.15.0 release of these factories collapsed the ensemble to a
# single complex field ``E = sqrt(<I>) * exp(i * angle(<E>))`` before
# return; the resulting single field has fully-coherent MCF
# ``J(r1, r2) = E*(r1) * E(r2)``, NOT the documented Schell-model
# factorisation ``sqrt(I(r1) I(r2)) * mu(r1 - r2)``.  The audit
# (docs/audits/AUDIT_V4_15_0_2026_05_18.md, P0-NEW-2, 3-way confirmed)
# also flagged a unit-RMS phase normalisation that forced
# ``sigma_phi == 1`` regardless of ``sigma_g``, so neither the
# ``sigma_g -> 0`` (incoherent) nor the ``sigma_g -> infinity``
# (coherent) limit was recovered.
#
# The redesigned recipe (per Goodman, _Statistical Optics_, Sec 5.5 and
# Wolf 1982 JOSA 72 343) is:
#
#   1. For each realisation k = 1..n_realizations, draw circular-Gaussian
#      complex white noise W_k(r) on the grid.
#   2. Multiply by the Fourier-space filter
#      ``H(k) = exp(-|k|^2 * sigma_g^2 / 4)`` and inverse-FFT to obtain
#      the band-limited complex random field ``phi_k(r)``.
#   3. Normalise ``phi_k`` so ``<|phi_k(r)|^2>_r == 1`` (unit mean
#      intensity).  The two-point correlation of the resulting field is
#      ``<phi(r1) * conj(phi(r2))> = exp(-|r1-r2|^2 / (2 sigma_g^2))``
#      -- the canonical Gaussian Schell kernel.
#   4. The k-th ensemble realisation is ``E_k(r) = sqrt(I(r)) * phi_k(r)``
#      so the ensemble MCF
#      ``J(r1, r2) = <E(r1) * conj(E(r2))>
#                  = sqrt(I(r1) I(r2)) * exp(-|r1-r2|^2 / (2 sigma_g^2))``
#      which is the Schell-model factorisation with Gaussian kernel.
#
# Return contract: ``return_kind='ensemble'`` (default) yields the raw
# ``(n_realizations, Ny, Nx)`` complex ensemble plus ``dx``, ``dy``,
# ``wavelength``.  ``return_kind='mcf'`` yields a ``PartialCoherenceMCF``
# instance (full N^2 x N^2 storage for small N, coherent-mode
# decomposition for large N).  The v4.15.0 single-field return is gone
# -- user is sole consumer, no shim needed; the migration is documented
# in the release notes.
# ---------------------------------------------------------------------------

import dataclasses as _dataclasses


@_dataclasses.dataclass
class PartialCoherenceMCF:
    """Mutual coherence function ``J(r1, r2) = <E(r1) * conj(E(r2))>``
    for a partially-coherent source.

    Two storage forms are supported, selected automatically by
    :meth:`from_ensemble` based on grid size:

    * **Full** -- the dense ``(Ny*Nx, Ny*Nx)`` MCF matrix ``J_full``.
      Used when ``Ny*Nx <= max_full_N**2`` (default
      ``max_full_N == 64``, i.e. up to ``64 x 64`` grids).  Exact;
      every two-point query reads the matrix directly.
    * **Modal** -- a coherent-mode decomposition (Mandel & Wolf,
      _Optical Coherence and Quantum Optics_, Section 4.7;
      Wolf 1982 JOSA 72 343).  Stores the leading ``K`` eigenmodes
      ``modes[k, :, :]`` and eigenvalues ``eigenvalues[k]`` such that
      ``J(r1, r2) ~= sum_k lambda_k * mode_k(r1) * conj(mode_k(r2))``.
      ``K`` defaults to the smallest number of modes whose cumulative
      eigenvalue sum exceeds 99% of the total (the standard
      Karhunen-Loeve truncation threshold).  Used for large N where
      full storage is infeasible.

    The decomposition is computed via SVD of the ensemble matrix
    ``E_mat`` of shape ``(n_realizations, Ny*Nx)``: the eigenmodes of
    ``J = (1/K) * E_mat^H @ E_mat`` are the right-singular vectors of
    ``E_mat``, and the eigenvalues are ``s_i^2 / n_realizations``.
    This avoids ever materialising the dense MCF for large N.

    Attributes
    ----------
    shape : (int, int)
        Spatial shape ``(Ny, Nx)`` of each mode / of the underlying
        ensemble realisation.
    dx, dy : float
        Pixel pitches [m].
    wavelength : float
        Vacuum wavelength [m].
    J_full : ndarray (Ny*Nx, Ny*Nx), complex, optional
        Full MCF matrix; populated when the grid is small enough.
    modes : ndarray (K, Ny, Nx), complex, optional
        Leading-K coherent modes (orthonormal under the standard L2
        inner product on the grid); populated in modal storage.
    eigenvalues : ndarray (K,), real >= 0, optional
        Eigenvalues of ``J`` corresponding to ``modes``, in descending
        order; populated in modal storage.
    n_realizations : int, optional
        Number of ensemble realisations used to build the MCF.
        Informational only.

    Notes
    -----
    MCF-aware downstream propagators (Koehler-/Hopkins-style coherent-
    mode propagation, MCF transport through the system) are NOT in
    v4.15.1 scope and are deferred to v4.16+.  The current
    :class:`PartialCoherenceMCF` is consumable for inspection and
    analysis only (intensity, two-point coherence, coherent-mode
    extraction).
    """
    shape: Tuple[int, int]
    dx: float
    dy: float
    wavelength: float
    J_full: Optional[np.ndarray] = None
    modes: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    n_realizations: Optional[int] = None

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def from_ensemble(
        cls,
        ensemble: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
        max_full_N: int = 64,
        n_modes: Optional[int] = None,
        energy_threshold: float = 0.99,
    ) -> 'PartialCoherenceMCF':
        """Build an MCF object from an ``(n_realizations, Ny, Nx)`` ensemble.

        Parameters
        ----------
        ensemble : ndarray (n_realizations, Ny, Nx), complex
            Stack of independent random-phase realisations.
        dx, dy : float
            Pixel pitches [m].
        wavelength : float
            Vacuum wavelength [m].
        max_full_N : int, default 64
            Grids with ``Ny * Nx <= max_full_N**2`` get the dense
            ``J_full`` storage form; larger grids fall back to modal
            storage (no dense ``J``).
        n_modes : int, optional
            Override the truncation rank for modal storage.  ``None``
            uses ``energy_threshold`` to pick the smallest ``K`` with
            cumulative eigenvalue mass >= the threshold (default 99%).
        energy_threshold : float, default 0.99
            Truncation threshold for the cumulative eigenvalue ratio
            in modal storage.  Ignored when ``n_modes`` is explicit.
        """
        if not isinstance(ensemble, np.ndarray) or ensemble.ndim != 3:
            raise ValueError(
                "PartialCoherenceMCF.from_ensemble: ensemble must be a "
                "3-D ndarray of shape (n_realizations, Ny, Nx); got "
                f"shape {getattr(ensemble, 'shape', None)!r}.")
        nr, Ny, Nx = ensemble.shape
        if nr < 1:
            raise ValueError(
                "PartialCoherenceMCF.from_ensemble: ensemble must "
                f"contain at least 1 realisation; got {nr}.")
        N_pix = int(Ny) * int(Nx)
        # Flatten to (n_realizations, N_pix) for SVD / dense build.
        E_mat = ensemble.reshape(nr, N_pix).astype(np.complex128, copy=False)

        if N_pix <= int(max_full_N) ** 2:
            # Dense MCF J(r1, r2) = <E(r1) conj(E(r2))> (the documented
            # convention).  SRC-1: this is (1/nr) E_mat.T @ conj(E_mat) whose
            # [i, j] = <E(r_i) conj(E(r_j))>; the prior
            # ``E_mat.conj().T @ E_mat`` stored <conj(E(r_i)) E(r_j)> =
            # conj(J_doc), i.e. the CONJUGATE -- which flipped the coherence
            # phase for a complex-J (twisted/tilted) ensemble and disagreed
            # with the modal branch.  Hermitian, positive semi-definite.
            J_full = (E_mat.T @ E_mat.conj()) / float(nr)
            return cls(
                shape=(int(Ny), int(Nx)),
                dx=float(dx), dy=float(dy), wavelength=float(wavelength),
                J_full=J_full,
                modes=None, eigenvalues=None,
                n_realizations=int(nr),
            )

        # Modal storage via SVD of the ensemble matrix.
        # E_mat = U @ diag(s) @ Vh,  shape (nr, N_pix).
        # For J_doc(r1, r2) = <E(r1) conj(E(r2))> = (1/nr) E_mat.T @ conj(E_mat)
        # the eigenmodes are the UNCONJUGATED rows Vh[k, :] (SRC-1: the prior
        # comment said Vh[k, :].conj(), which matched the conjugated dense
        # build, not the documented convention), with eigenvalues s_k^2 / nr.
        # SVD with full_matrices=False yields at most ``min(nr, N_pix)``
        # singular values -- exactly the non-zero spectrum of J.
        U, s, Vh = np.linalg.svd(E_mat, full_matrices=False)
        eigvals = (s ** 2) / float(nr)
        # Truncation rank K.
        if n_modes is not None:
            K = int(n_modes)
            if K < 1 or K > eigvals.size:
                raise ValueError(
                    "PartialCoherenceMCF.from_ensemble: n_modes must be in "
                    f"[1, {eigvals.size}]; got {n_modes!r}.")
        else:
            total = float(np.sum(eigvals))
            if total <= 0.0:
                K = 1  # degenerate -- keep one mode
            else:
                cum = np.cumsum(eigvals) / total
                # First index where cumulative >= threshold; +1 for count.
                idx = int(np.searchsorted(cum, float(energy_threshold)))
                K = min(max(1, idx + 1), eigvals.size)
        # Each row of Vh is a right-singular vector of E_mat;
        # mode_k(r) corresponds to Vh[k, :] in the (Ny, Nx) layout.
        modes = Vh[:K, :].reshape(K, int(Ny), int(Nx)).astype(np.complex128)
        eigenvalues = eigvals[:K].astype(np.float64)
        return cls(
            shape=(int(Ny), int(Nx)),
            dx=float(dx), dy=float(dy), wavelength=float(wavelength),
            J_full=None,
            modes=modes, eigenvalues=eigenvalues,
            n_realizations=int(nr),
        )

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def intensity(self) -> np.ndarray:
        """Return the ``(Ny, Nx)`` intensity diagonal of ``J``.

        ``I(r) = J(r, r) = sum_k lambda_k * |mode_k(r)|^2`` in the
        modal form, or ``I = diag(J_full)`` reshaped to ``(Ny, Nx)``
        in the dense form.
        """
        Ny, Nx = self.shape
        if self.J_full is not None:
            return np.real(np.diag(self.J_full)).reshape(Ny, Nx)
        if self.modes is not None and self.eigenvalues is not None:
            # sum_k lambda_k * |phi_k|^2
            mod2 = (np.abs(self.modes) ** 2)  # (K, Ny, Nx)
            return np.tensordot(self.eigenvalues, mod2, axes=(0, 0))
        raise RuntimeError(
            "PartialCoherenceMCF: neither J_full nor (modes, eigenvalues) "
            "is populated; the object is in an inconsistent state.")

    def coherence_at(
        self,
        ix1: int, iy1: int,
        ix2: int, iy2: int,
    ) -> complex:
        """Return the complex coherence factor

            ``mu(r1, r2) = J(r1, r2) / sqrt(I(r1) * I(r2))``

        between the two grid points ``r1 = (ix1, iy1)`` and ``r2 =
        (ix2, iy2)``.  ``|mu|`` is bounded by 1 (Schwarz inequality);
        ``|mu| == 1`` is fully coherent, ``|mu| == 0`` fully incoherent.

        Note: the index convention is ``(ix, iy)`` (column, row), so
        ``(ix=0, iy=0)`` is the top-left pixel.
        """
        Ny, Nx = self.shape
        for label, ix, iy in (('1', ix1, iy1), ('2', ix2, iy2)):
            if not (0 <= int(ix) < Nx and 0 <= int(iy) < Ny):
                raise IndexError(
                    f"PartialCoherenceMCF.coherence_at: point {label} "
                    f"(ix={ix}, iy={iy}) is out of bounds for grid "
                    f"shape (Ny={Ny}, Nx={Nx}).")
        flat1 = int(iy1) * int(Nx) + int(ix1)
        flat2 = int(iy2) * int(Nx) + int(ix2)
        if self.J_full is not None:
            J12 = complex(self.J_full[flat1, flat2])
            I1 = float(np.real(self.J_full[flat1, flat1]))
            I2 = float(np.real(self.J_full[flat2, flat2]))
        else:
            if self.modes is None or self.eigenvalues is None:
                raise RuntimeError(
                    "PartialCoherenceMCF.coherence_at: object is in an "
                    "inconsistent state (no J_full and no modes).")
            modes_flat = self.modes.reshape(self.modes.shape[0], -1)
            phi1 = modes_flat[:, flat1]
            phi2 = modes_flat[:, flat2]
            J12 = complex(np.sum(self.eigenvalues * phi1 * np.conj(phi2)))
            I1 = float(np.sum(self.eigenvalues * (np.abs(phi1) ** 2)))
            I2 = float(np.sum(self.eigenvalues * (np.abs(phi2) ** 2)))
        denom = float(np.sqrt(I1 * I2))
        if denom <= 0.0:
            return 0.0 + 0.0j
        return J12 / denom

    def coherent_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(modes, eigenvalues)`` of the coherent-mode
        decomposition.

        In the modal storage form, returns the stored arrays directly.
        In the dense (``J_full``) form, computes the decomposition on
        demand via Hermitian eigendecomposition of ``J_full``.

        Returns
        -------
        modes : ndarray (K, Ny, Nx), complex
            Orthonormal coherent modes, ordered by descending
            eigenvalue.
        eigenvalues : ndarray (K,), real >= 0
            Eigenvalues of ``J``, descending.
        """
        Ny, Nx = self.shape
        if self.modes is not None and self.eigenvalues is not None:
            return self.modes, self.eigenvalues
        if self.J_full is None:
            raise RuntimeError(
                "PartialCoherenceMCF.coherent_modes: neither modal nor "
                "dense storage is populated.")
        # Hermitian eigendecomposition; eigh returns ascending.
        w, V = np.linalg.eigh(self.J_full)
        # Reverse to descending; clip tiny negatives from round-off.
        w = np.maximum(w[::-1], 0.0)
        V = V[:, ::-1]
        K = w.size
        modes = (V.T.reshape(K, Ny, Nx)).astype(np.complex128)
        return modes, w.astype(np.float64)


def _validate_return_kind(value: Any, fn_name: str) -> str:
    """Validate the ``return_kind`` kwarg shared by the 3 Schell
    factories.  Returns the canonicalised string."""
    if not isinstance(value, str):
        raise ValueError(
            f"{fn_name}: return_kind must be 'ensemble' or 'mcf'; got "
            f"{value!r} ({type(value).__name__}).")
    canon = value.lower().strip()
    if canon not in ('ensemble', 'mcf'):
        raise ValueError(
            f"{fn_name}: return_kind must be 'ensemble' or 'mcf'; got "
            f"{value!r}.")
    return canon


# v4.15.2 (P0-NEW-1) - v4.15.5: the 3 Schell factories changed default
# return shape from ``(E_2d, x, y)`` (v4.15.0) to
# ``(ensemble_3d, dx, dy, wavelength)`` (v4.15.1).  A one-release
# ``DeprecationWarning`` was emitted on the default path so pre-v4.15.0
# callers doing ``E, x, y = create_gaussian_schell_source(...)`` would
# see a loud heads-up rather than a propagation-time wrong-shape
# failure.  Detection used a per-module sentinel: when the caller left
# ``return_kind`` unset, the factory saw the sentinel and warned;
# explicit ``return_kind='ensemble'`` or ``'mcf'`` was silent.
#
# v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path warning
# is retired -- the v4.15.0 -> v4.15.1 transition has had four
# subsequent releases of exposure (v4.15.2 / v4.15.3 / v4.15.4 /
# v4.15.5 / v4.16.0).  The kwarg default is now plain ``'ensemble'``;
# the sentinel branch at each call site is preserved as a no-op for
# back-compat (callers explicitly forwarding ``_RETURN_KIND_UNSET``
# still resolve cleanly).  Sentinel + ``_warn_schell_return_kind_default``
# helper are slated for removal in v5.0.
#
# v4.15.3 (audit P2-NEW): the sentinel itself is a dedicated
# ``_SchellReturnKindUnsetSentinel(_Sentinel)`` subclass for
# consistency with the other sentinel patterns in the codebase
# (``_ZeroApertureMaskSentinel`` in ``optimize/core.py``,
# ``_AngleUnsetSentinel`` in ``elements/polarization.py``,
# ``_NoDefaultSentinel`` in ``_deprecation.py``).  The subclass-identity
# pattern lets downstream code do ``isinstance(x,
# _SchellReturnKindUnsetSentinel)`` without false positives from other
# bare ``_Sentinel`` instances that share the same registry.  Singleton
# instance still keyed by the ``_SCHELL_RETURN_KIND_UNSET`` registry
# name; pickle round-trip via ``_SENTINEL_REGISTRY`` lookup is
# unchanged.
from .._deprecation import _Sentinel as _DeprecationSentinel


class _SchellReturnKindUnsetSentinel(_DeprecationSentinel):
    """Singleton sentinel for "Schell ``return_kind`` argument was not
    explicitly passed by the caller".

    v4.15.3 (audit P2-NEW): dedicated subclass for consistency with
    :class:`_ZeroApertureMaskSentinel` (in ``optimize/core.py``),
    :class:`_AngleUnsetSentinel` (in ``elements/polarization.py``),
    :class:`_NoDefaultSentinel` (in ``_deprecation.py``).  No behaviour
    change relative to the pre-v4.15.3 bare ``_Sentinel``: the new
    subclass overrides nothing, the singleton is still keyed by the
    ``_SCHELL_RETURN_KIND_UNSET`` registry name, and the
    ``_SENTINEL_REGISTRY`` round-trip is identical.  The benefit is
    discoverability: ``isinstance(x, _SchellReturnKindUnsetSentinel)``
    works as a strict type-narrowing predicate without false positives
    from sibling bare ``_Sentinel`` instances.
    """
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__('_SCHELL_RETURN_KIND_UNSET')


_RETURN_KIND_UNSET: Any = _SchellReturnKindUnsetSentinel()
# v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the singleton + the
# subclass are retained as deprecated public-symbol attributes for
# back-compat with the v4.15.3 test pins
# (``tests/unit/test_v4_15_3_agent_b.py`` imports both names).  The
# subclass remains pickle-safe via the shared
# ``_SENTINEL_REGISTRY``; the singleton continues to compare
# ``False`` and stringify with the canonical registry name.  Targeted
# for removal in v5.0 alongside the rest of the deprecated-default
# scaffolding.


def _warn_schell_return_kind_default(fn_name: str) -> None:
    """Emit the v4.15.0 -> v4.15.1 return-shape ``DeprecationWarning``.

    v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path call
    sites no longer invoke this helper -- the 3 factories +
    ``Source.gaussian_schell`` / ``Source.schell_model`` classmethods
    now default ``return_kind='ensemble'`` directly, with no warning
    on the default path.  The helper itself is preserved for back-
    compat -- the v4.15.3 stacklevel meta-pin in
    ``tests/unit/test_v4_15_3_agent_b.py`` imports it -- and remains
    invocable by external callers who want to surface the warning
    manually (e.g. in custom Schell wrappers that still maintain the
    pre-v4.15.1 single-field return signature).

    Routed through the shared ``_deprecation.warn_deprecated_signature``
    helper so the warning message format matches the rest of the
    library and is silenceable with a single ``warnings.filterwarnings``
    incantation.  ``version_removed='5.0'`` documents the deprecation
    horizon (sentinel + helper are slated for removal in v5.0).

    Frame chain (innermost first), when invoked from a factory body:

      1. ``warnings.warn`` (inside ``_emit`` body)
      2. ``_emit`` -> ``warn_deprecated_signature``
      3. ``warn_deprecated_signature`` -> ``_warn_schell_return_kind_default``
      4. ``_warn_schell_return_kind_default`` -> factory body
      5. factory body -> user code  <-- target of stacklevel
    """
    from .._deprecation import warn_deprecated_signature
    warn_deprecated_signature(
        function=fn_name,
        old_signature=(
            f'{fn_name}(...)  # v4.15.0 returned (E_2d, x, y)'),
        new_signature=(
            f"{fn_name}(..., return_kind='ensemble')  "
            "# v4.15.1 default returns (ensemble_3d, dx, dy, "
            "wavelength); pass return_kind='ensemble' (preserve "
            "current 4-tuple) or return_kind='mcf' (opt into "
            "PartialCoherenceMCF) explicitly"),
        version_added='4.15.1',
        version_removed='5.0',
        stacklevel=5,
    )


def _schell_phase_realizations(
    *,
    Ny: int,
    Nx: int,
    dx: float,
    dy: float,
    coherence_length: float,
    n_realizations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate ``n_realizations`` band-limited complex random fields
    with the Gaussian Schell kernel as their two-point correlation.

    v5.24.x (audit S3-16): the vestigial ``N`` keyword was removed.  The
    grid is fully specified by ``Ny`` / ``Nx``; the old ``N`` argument
    duplicated ``Nx`` (callers passed ``N=Ny=Nx`` for the square-grid
    case) and was never read in the body -- a dead parameter that only
    invited an inconsistent (``N`` != ``Nx``) call.

    The recipe (Goodman, _Statistical Optics_, Sec 5.5):

    1. For each realisation, draw circular-Gaussian complex white
       noise ``W(r)`` on the grid (independent N(0, 1/2) real and
       imaginary parts).
    2. Multiply by the Fourier-space filter
       ``H(k) = exp(-|k|^2 * sigma_g^2 / 4)`` and inverse-FFT.  The
       resulting complex field ``phi(r)`` has spatial correlation
       ``<phi(r1) conj(phi(r2))> ~ exp(-|r1-r2|^2 / (2 sigma_g^2))``.
    3. Normalise by the DETERMINISTIC expected mean intensity
       ``phi <- phi / sqrt(E[<|phi|^2>])`` with
       ``E[<|phi|^2>] = sum_k |H(k)|^2 / (Ny*Nx)`` (Parseval), a single
       constant for the whole ensemble.  v5.4.6 (audit P3-10): this
       preserves unit mean intensity in EXPECTATION while keeping the
       two-point correlation exactly Gaussian-Schell; per-realisation
       normalisation (the pre-fix recipe) biases the empirical MCF.

    Returns
    -------
    phi : ndarray (n_realizations, Ny, Nx), complex128
        Unit-mean-intensity band-limited noise stack.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = kx[None, :], ky[:, None]  # S3-7: broadcast views, not a dense grid
    # Fourier-space Gaussian filter.  Variance of |phi(r)|^2 in real
    # space scales as integral of |H(k)|^2 dk; we re-normalise to
    # unit mean intensity per realisation below, so the absolute
    # amplitude of H here is irrelevant.
    sigma_g = float(coherence_length)
    spec_filter = np.exp(-(KX * KX + KY * KY) * (sigma_g ** 2) / 4.0)

    out = np.empty((int(n_realizations), int(Ny), int(Nx)),
                   dtype=np.complex128)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    # v5.4.6 (audit P3-10): normalise EVERY realisation by the single
    # DETERMINISTIC expected mean intensity, not by each realisation's own
    # random mean.  Dividing each realisation by its random mean is
    # non-linear in the ensemble and systematically biases the empirical
    # Gaussian-Schell two-point correlation (the bias does NOT average out
    # with more realisations).  By Parseval, E[mean(|Phi|^2)] =
    # sum_k |H(k)|^2 / (Ny*Nx).  Unit mean intensity is thus preserved IN
    # EXPECTATION while keeping <phi(r1) conj(phi(r2))> exactly Gaussian.
    _mean_I = float(np.sum(np.abs(spec_filter) ** 2) / (Ny * Nx))
    _norm = np.sqrt(_mean_I) if _mean_I > 0.0 else 1.0
    for k in range(int(n_realizations)):
        # Circular-Gaussian complex white noise.
        w_re = rng.standard_normal((Ny, Nx))
        w_im = rng.standard_normal((Ny, Nx))
        W = (w_re + 1j * w_im) * inv_sqrt2
        Phi = np.fft.ifft2(np.fft.fft2(W) * spec_filter)
        out[k] = Phi / _norm
    return out


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
    return_kind: Any = 'ensemble',
    max_full_N: int = 64,
    n_modes: Optional[int] = None,
    energy_threshold: float = 0.99,
) -> Union[Tuple[np.ndarray, float, float, float], 'PartialCoherenceMCF']:
    """Gaussian-Schell partial-coherence source.

    v4.15.1 (P0-NEW-2): redesigned to deliver actual partial coherence.
    Returns either the raw ``(n_realizations, Ny, Nx)`` complex
    ensemble (the canonical contract; caller averages intensities
    downstream) or a :class:`PartialCoherenceMCF` object exposing the
    mutual coherence function and its coherent-mode decomposition.

    A Gaussian-Schell beam has Gaussian intensity profile (1/e^2
    radius ``w0``) and Gaussian coherence kernel
    ``mu(r1, r2) = exp(-|r1-r2|^2 / (2 sigma_g^2))``.  In the
    fully-coherent limit ``sigma_g -> infinity`` the MCF approaches
    rank 1 (a deterministic Gaussian with waist ``w0``); in the
    fully-incoherent limit ``sigma_g -> 0`` the off-diagonal MCF
    decays to zero.

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
        ``sigma_g >> w0`` approaches the coherent limit;
        ``sigma_g << dx`` approaches the incoherent limit.
    n_realizations : int, default 16
        Number of independent ensemble draws.  Must be >= 1.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    seed : int, optional
        RNG seed for reproducibility.
    dtype : optional
        Complex dtype for the returned ensemble (default: library
        default).  Ignored when ``return_kind='mcf'`` -- the MCF
        always uses complex128 internally.
    return_kind : {'ensemble', 'mcf'}, default 'ensemble'
        ``'ensemble'`` (default): return ``(E_ensemble, dx, dy,
        wavelength)`` where ``E_ensemble`` is a ``(n_realizations, Ny,
        Nx)`` complex array.
        ``'mcf'``: return a :class:`PartialCoherenceMCF` instance.

        v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        ``DeprecationWarning`` (v4.15.2 -> v4.15.5) flagging the
        v4.15.0 -> v4.15.1 return-shape change is retired now that
        the new ensemble contract has had multiple releases of
        exposure.  The default is ``'ensemble'`` and the call is
        silent.  Callers can still inspect the sentinel via
        ``_RETURN_KIND_UNSET`` for back-compat (it is treated as
        ``'ensemble'``; sentinel + helper are slated for removal in
        v5.0).
    max_full_N : int, default 64
        Forwarded to :meth:`PartialCoherenceMCF.from_ensemble` when
        ``return_kind='mcf'``.  Grids with ``Ny * Nx > max_full_N**2``
        get the modal-storage form.
    n_modes : int, optional
        Forwarded to :meth:`PartialCoherenceMCF.from_ensemble` when
        ``return_kind='mcf'``.  ``None`` selects modes by
        ``energy_threshold``.
    energy_threshold : float, default 0.99
        Forwarded to :meth:`PartialCoherenceMCF.from_ensemble` when
        ``return_kind='mcf'`` and ``n_modes is None``: the smallest
        ``K`` with cumulative eigenvalue mass ``>= energy_threshold``
        is kept.  Accepted on the ``return_kind='ensemble'`` path for
        signature symmetry but a no-op there (no modal truncation
        occurs).  v4.15.2 (P2): added to close the kwarg-forwarding
        gap flagged by the v4.15.1 audit.

    Returns
    -------
    With ``return_kind='ensemble'`` (default):
        ``(E_ensemble, dx, dy, wavelength)`` where ``E_ensemble`` is a
        ``(n_realizations, Ny, Nx)`` complex array.
    With ``return_kind='mcf'``:
        :class:`PartialCoherenceMCF` instance.

    Notes
    -----
    The pre-v4.15.1 single-field return is gone; callers expecting
    ``(E_2d, x, y)`` must migrate.  See ``docs/release_notes/
    .release_notes_v4_15_1_agent_b.md`` for the migration guide.
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
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
    # ``DeprecationWarning`` (v4.15.2 -> v4.15.5) is retired now that
    # the v4.15.0 return-shape change has had multiple releases of
    # exposure.  The kwarg default is plain ``'ensemble'``; the
    # sentinel branch below is preserved as a no-op for back-compat
    # so callers explicitly passing ``return_kind=_RETURN_KIND_UNSET``
    # (rare; flagged by the v4.15.3 meta-pin) keep working.
    if return_kind is _RETURN_KIND_UNSET:
        return_kind = 'ensemble'
    rk = _validate_return_kind(
        return_kind, fn_name='create_gaussian_schell_source')

    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid
    target_dtype = _resolve_complex_dtype(dtype)

    # Gaussian intensity envelope.  amp = sqrt(I_target).  The 1/e^2
    # *intensity* radius is w0, so I = exp(-2 r^2 / w0^2) and
    # amp = exp(-r^2 / w0^2).
    amp = np.exp(-(X * X + Y * Y) / (w0 ** 2))

    rng = np.random.default_rng(seed)
    phi = _schell_phase_realizations(
        Ny=int(N), Nx=int(N), dx=float(dx), dy=float(dy),
        coherence_length=float(sigma_g),
        n_realizations=int(n_realizations), rng=rng)
    # E_k(r) = sqrt(I(r)) * phi_k(r).
    E_ensemble = (amp[None, :, :] * phi).astype(target_dtype)

    if rk == 'mcf':
        # v4.15.2 (P2): forward ``energy_threshold`` to close the
        # audit-flagged kwarg gap.  Truncation only fires here on
        # modal storage; the ``ensemble`` branch ignores the kwarg
        # by design (no truncation step there).
        return PartialCoherenceMCF.from_ensemble(
            E_ensemble,
            dx=float(dx), dy=float(dy), wavelength=float(wavelength),
            max_full_N=int(max_full_N), n_modes=n_modes,
            energy_threshold=float(energy_threshold))
    return E_ensemble, float(dx), float(dy), float(wavelength)


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
    return_kind: Any = 'ensemble',
    max_full_N: int = 64,
    n_modes: Optional[int] = None,
    energy_threshold: float = 0.99,
) -> Union[Tuple[np.ndarray, float, float, float], 'PartialCoherenceMCF']:
    """Generic Schell-model source with user-supplied intensity profile.

    v4.15.1 (P0-NEW-2): redesigned to deliver actual partial coherence.
    The MCF factorises as
    ``J(r1, r2) = sqrt(I(r1) I(r2)) * mu(r1 - r2)`` with the Gaussian
    coherence kernel
    ``mu(Delta_r) = exp(-|Delta_r|^2 / (2 coherence_length^2))``.

    Parameters
    ----------
    N : int
        Square grid size (N x N).
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    intensity_profile : ndarray (N, N), real >= 0
        Time-averaged intensity profile of the source.
    coherence_length : float
        Gaussian transverse coherence length [m]; must be > 0.
    n_realizations : int, default 16
        Number of independent ensemble draws.
    dy : float, optional
        Grid spacing in y [m].
    seed : int, optional
        RNG seed.
    dtype : optional
        Complex dtype for the ensemble form; ignored for ``'mcf'``.
    return_kind : {'ensemble', 'mcf'}, default 'ensemble'
        See :func:`create_gaussian_schell_source`.  v4.16.1 (audit
        item 6): the default-path ``DeprecationWarning`` is retired;
        the default is plain ``'ensemble'`` and the call is silent.
    max_full_N, n_modes
        See :meth:`PartialCoherenceMCF.from_ensemble`.
    energy_threshold : float, default 0.99
        See :func:`create_gaussian_schell_source`.  v4.15.2 (P2):
        forwarded to :meth:`PartialCoherenceMCF.from_ensemble` on the
        ``return_kind='mcf'`` path; ignored otherwise.

    Returns
    -------
    Same contract as :func:`create_gaussian_schell_source`.
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
    # v4.16.1: default-path warning retired; sentinel kept for back-
    # compat (see :func:`create_gaussian_schell_source` for the
    # rationale).
    if return_kind is _RETURN_KIND_UNSET:
        return_kind = 'ensemble'
    rk = _validate_return_kind(
        return_kind, fn_name='create_schell_model_source')

    if dy is None:
        dy = dx
    target_dtype = _resolve_complex_dtype(dtype)
    amp = np.sqrt(I)

    rng = np.random.default_rng(seed)
    phi = _schell_phase_realizations(
        Ny=int(N), Nx=int(N), dx=float(dx), dy=float(dy),
        coherence_length=float(coherence_length),
        n_realizations=int(n_realizations), rng=rng)
    E_ensemble = (amp[None, :, :] * phi).astype(target_dtype)

    if rk == 'mcf':
        # v4.15.2 (P2): forward ``energy_threshold`` -- see
        # :func:`create_gaussian_schell_source` for the rationale.
        return PartialCoherenceMCF.from_ensemble(
            E_ensemble,
            dx=float(dx), dy=float(dy), wavelength=float(wavelength),
            max_full_N=int(max_full_N), n_modes=n_modes,
            energy_threshold=float(energy_threshold))
    return E_ensemble, float(dx), float(dy), float(wavelength)


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
    return_kind: Any = 'ensemble',
    max_full_N: int = 64,
    n_modes: Optional[int] = None,
    energy_threshold: float = 0.99,
) -> Union[Tuple[np.ndarray, float, float, float], 'PartialCoherenceMCF']:
    """Annular (ring) source -- spatially-incoherent at the source
    plane (v4.15, ROADMAP v4.16 #11; v4.15.1 P0-NEW-2 redesign).

    Distinct from :func:`create_annular_beam`, which returns the
    deterministic coherent annular field.  This factory samples
    independent random-phase realisations whose common amplitude is
    the unit-norm annular indicator function; each pixel has its own
    independent uniform-(-pi, pi) phase (i.e. fully incoherent at the
    source plane, the classic "non-zero source size for partial-
    coherence integration" recipe).

    v4.15.1 (P0-NEW-2): the factory no longer collapses the ensemble
    into a single complex field; it returns either the raw ensemble or
    a :class:`PartialCoherenceMCF` (diagonal MCF reflecting the
    incoherent-source character).

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
    return_kind : {'ensemble', 'mcf'}, default 'ensemble'
        See :func:`create_gaussian_schell_source`.  v4.16.1 (audit
        item 6): the default-path ``DeprecationWarning`` is retired;
        the default is plain ``'ensemble'`` and the call is silent.
    max_full_N, n_modes
        See :meth:`PartialCoherenceMCF.from_ensemble`.
    energy_threshold : float, default 0.99
        See :func:`create_gaussian_schell_source`.  v4.15.2 (P2):
        forwarded to :meth:`PartialCoherenceMCF.from_ensemble` on the
        ``return_kind='mcf'`` path; ignored otherwise.

    Returns
    -------
    Same contract as :func:`create_gaussian_schell_source`.
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
    # v4.16.1: default-path warning retired; sentinel kept for back-
    # compat (see :func:`create_gaussian_schell_source` for the
    # rationale).
    if return_kind is _RETURN_KIND_UNSET:
        return_kind = 'ensemble'
    rk = _validate_return_kind(
        return_kind, fn_name='create_annular_incoherent_source')

    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = x[None, :], y[:, None]  # S3-7: broadcast views, not a dense N x N grid
    target_dtype = _resolve_complex_dtype(dtype)
    r = np.sqrt(X * X + Y * Y)
    mask = (r >= inner_radius) & (r <= outer_radius)
    amp = mask.astype(np.float64)
    # Normalise so the integrated power per realisation is 1.
    norm = np.sqrt(np.sum(amp ** 2) * dx * dy)
    if norm > 0:
        amp = amp / norm

    rng = np.random.default_rng(seed)
    nr = int(n_realizations)
    E_ensemble = np.empty((nr, int(N), int(N)), dtype=target_dtype)
    for k in range(nr):
        # Independent uniform-(-pi, pi) phase per pixel: the source
        # is spatially incoherent (coherence kernel == delta function
        # at the pixel scale).
        phase = rng.uniform(-np.pi, np.pi, (int(N), int(N)))
        E_ensemble[k] = (amp * np.exp(1j * phase)).astype(target_dtype)

    if rk == 'mcf':
        # v4.15.2 (P2): forward ``energy_threshold`` -- see
        # :func:`create_gaussian_schell_source` for the rationale.
        return PartialCoherenceMCF.from_ensemble(
            E_ensemble,
            dx=float(dx), dy=float(dy), wavelength=float(wavelength),
            max_full_N=int(max_full_N), n_modes=n_modes,
            energy_threshold=float(energy_threshold))
    return E_ensemble, float(dx), float(dy), float(wavelength)


# ---------------------------------------------------------------------------
# Source -- bundles E + dx + wavelength + source_point with chainable
# .propagate(...) and .from_X(...) factories
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional as _Optional
from typing import Tuple as _Tuple


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
    jones : object, optional
        v5.24.x (audit S3-17): OPTIONAL polarization / Jones channel.
        Defaults to ``None`` (scalar field -- the historical
        behaviour).  When set, it carries the vectorial polarization
        state alongside the scalar ``E`` so vectorial pipelines no
        longer have to bypass the ``Source`` container entirely.  This
        is a *minimal metadata channel*: the scalar propagators
        (:meth:`propagate`) do NOT act on it -- they transport ``E``
        and pass ``jones`` through UNCHANGED to the descendant Source
        so a downstream Jones-aware stage can pick it up.  Suggested
        conventions (not enforced, to avoid over-constraining callers):
        a length-2 complex vector ``[a_x, a_y]`` for a spatially
        uniform polarization, or a full vectorial field of shape
        ``(2, Ny, Nx)`` / ``(Ny, Nx, 2)``.  A future release may add a
        vectorial ``propagate`` that transforms this channel; until
        then it is pure carried metadata.
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
    # v5.24.x (audit S3-17): optional Jones/polarization channel.
    # Appended AFTER ``dy`` so every existing positional caller
    # (``Source(E, dx, wavelength[, source_point, name, dy])``) is
    # unaffected.  ``None`` == scalar field (the default).
    jones: 'object' = None

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
        # DS-1: route through ``return_result=True`` so pitch-CHANGING kernels
        # (fresnel / fraunhofer / sas, including the method='auto' far-field
        # selections at N_F < 0.1 / Q > 1) are handled correctly.  Their raw
        # return is a ``(E, dx_out, dy_out)`` tuple with an output pitch of
        # lambda*z/(N*dx); the previous code wrapped that tuple AS the field
        # (``Source.shape`` then raised) and kept the STALE input pitch.  The
        # PropagationResult carries the true field + output dx/dy (honouring an
        # explicit output_dx/output_dy in kwargs, and dx-unchanged for ASM).
        result = propagate(
            self.E,
            wavelength=self.wavelength,
            dx=self.dx,
            z=z,
            prescription=prescription,
            method=method,
            return_result=True,
            **kwargs,
        )
        out_dx = float(result.dx)
        # v4.13.0 audit P1-C (re-threaded under DS-1): preserve the anamorphic
        # y-pitch across ``Source.propagate``.  The dispatcher takes only a
        # single ``dx``; pitch-preserving kernels (asm/rs/hf/...) therefore
        # report ``result.dy == result.dx`` even when the input carried a
        # distinct y-pitch.  Precedence: an explicit ``output_dy`` kwarg wins;
        # otherwise a kernel that genuinely resampled y (``result.dy`` differs
        # from ``result.dx``) is honoured verbatim; otherwise keep the input's
        # (dy/dx) aspect by scaling ``self.dy`` by the same factor the x-pitch
        # changed (== 1 for pitch-preserving kernels, so ``self.dy`` survives).
        _out_dy_kw = kwargs.get('output_dy')
        _kern_dy = float(result.dy)
        if _out_dy_kw:
            out_dy = float(_out_dy_kw)
        elif _kern_dy != out_dx:
            out_dy = _kern_dy
        else:
            _scale = out_dx / self.dx if self.dx else 1.0
            out_dy = self.dy * _scale
        new_name = f"{self.name or 'Source'}->{method}"
        return Source(
            E=result.field, dx=out_dx, dy=out_dy,
            wavelength=self.wavelength,
            source_point=self.source_point, name=new_name,
            # v5.24.x (audit S3-17): the scalar propagators do not act
            # on the polarization channel; carry it through UNCHANGED
            # so a downstream Jones-aware stage still sees it.
            jones=self.jones,
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
            # v4.15.3 (audit P1-NEW-F1-2 sweep): stacklevel=4 -> user
            # code (chain: _emit [1] -> warn_deprecated_signature [2]
            # -> Source.gaussian classmethod body [3] -> user [4]).
            warn_deprecated_signature(
                function='Source.gaussian',
                old_signature='Source.gaussian(w0, N, dx, wavelength, ...)',
                new_signature=(
                    'Source.gaussian(*, N, dx, wavelength, w0, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=4,
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
            # v4.15.3 (audit P1-NEW-F1-2 sweep): stacklevel=4 -> user
            # code (chain: _emit [1] -> warn_deprecated_signature [2]
            # -> Source.plane_wave classmethod body [3] -> user [4]).
            warn_deprecated_signature(
                function='Source.plane_wave',
                old_signature='Source.plane_wave(N, dx, wavelength, ...)',
                new_signature=(
                    'Source.plane_wave(*, N, dx, wavelength, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=4,
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
            # v4.15.3 (audit P1-NEW-F1-2 sweep): stacklevel=4 -> user
            # code (chain: _emit [1] -> warn_deprecated_signature [2]
            # -> Source.point_source classmethod body [3] -> user [4]).
            warn_deprecated_signature(
                function='Source.point_source',
                old_signature='Source.point_source(N, dx, wavelength, ...)',
                new_signature=(
                    'Source.point_source(*, N, dx, wavelength, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=4,
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
            # v4.15.3 (audit P1-NEW-F1-2 sweep): stacklevel=4 -> user
            # code (chain: _emit [1] -> warn_deprecated_signature [2]
            # -> Source.top_hat classmethod body [3] -> user [4]).
            warn_deprecated_signature(
                function='Source.top_hat',
                old_signature='Source.top_hat(diameter, N, dx, wavelength, ...)',
                new_signature=(
                    'Source.top_hat(*, N, dx, wavelength, diameter, ...)'),
                version_added='4.15',
                version_removed='5.0',
                stacklevel=4,
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
            # v4.15.3 (audit P1-NEW-F1-2 sweep): stacklevel=4 -> user
            # code (chain: _emit [1] -> warn_deprecated_signature [2]
            # -> Source.fiber_mode classmethod body [3] -> user [4]).
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
                stacklevel=4,
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
                         return_kind: Any = 'ensemble',
                         **factory_kwargs) -> Any:
        """Gaussian-Schell partial-coherence source.

        Wraps :func:`create_gaussian_schell_source`.

        v4.15.2 (Agent E, AUDIT_V4_15_1 P2): the return type now
        matches the top-level factory's return-type convention
        verbatim -- ``return_kind='ensemble'`` yields the raw
        ``(ensemble, dx, dy, wavelength)`` 4-tuple (NOT a
        :class:`Source`-wrapped 3-D ensemble).  Pre-v4.15.2 this
        classmethod wrapped the 3-D ensemble inside a :class:`Source`
        whose ``E`` was 3-D, breaking the :class:`Source` contract
        (every other ``Source.*`` classmethod produces a 2-D field)
        and surprising downstream ``src.intensity()`` callers with
        broadcasting axis mismatches.

        v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        ``DeprecationWarning`` (v4.15.2 -> v4.15.5) for the v4.15.0
        -> v4.15.1 return-shape change is retired.  The default is
        plain ``'ensemble'`` and the call is silent.  The sentinel
        ``_RETURN_KIND_UNSET`` is preserved as a deprecated
        compatibility shim (treated as ``'ensemble'`` at the
        classmethod boundary); both sentinel + helper are slated for
        removal in v5.0.

        Return contract (v4.15.2+)
        --------------------------
        * ``return_kind='ensemble'`` (default): returns
          ``(E_ensemble, dx, dy, wavelength)`` -- the *same* 4-tuple
          as :func:`create_gaussian_schell_source`.  No
          :class:`Source` wrapping.  Downstream callers iterate over
          ``E_ensemble[k]`` and average intensities for the partial-
          coherence response, or wrap each realisation in a
          per-realisation :class:`Source` themselves when feeding a
          fully-coherent propagator chain.
        * ``return_kind='mcf'``: returns the bare
          :class:`PartialCoherenceMCF` object (NOT wrapped in a
          :class:`Source`, since the propagator pipeline currently
          consumes only a single complex field).

        Inconsistency note: this differs from :meth:`Source.gaussian`
        (which returns a :class:`Source` wrapping a 2-D field).  The
        inconsistency is honest -- Schell is partial-coherence,
        fundamentally different from coherent single-source, and
        wrapping a 3-D ensemble in a 2-D-field abstraction silently
        breaks the abstraction.  The MCF object is consumable for
        inspection / analysis only; MCF-aware downstream propagators
        are not in v4.15.x scope.

        2-D ``Source.E`` invariant break (intentional)
        ----------------------------------------------
        The 4-tuple return shape ``(E_ensemble, dx, dy, wavelength)``
        has ``E_ensemble.shape == (n_realizations, Ny, Nx)`` -- a
        3-D array, NOT the 2-D ``Source.E`` invariant the other
        ``Source.*`` classmethods uphold.  This is intentional --
        Schell is partial-coherence, fundamentally different from
        coherent single-source, and wrapping the ensemble inside a
        single :class:`Source` would silently break the 2-D
        abstraction.  Users who want a per-source iterator should
        unpack the ensemble explicitly:

            ens, _dx, _dy, _wl = Source.gaussian_schell(...)
            sources = [Source(E=ens[k], dx=_dx, dy=_dy,
                              wavelength=_wl) for k in range(len(ens))]

        A future ``Source.realizations()`` per-realization iterator
        is in scope for v4.16+ but is NOT shipped in v4.15.3.
        """
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        # DeprecationWarning is retired; the sentinel pass-through is
        # preserved so callers explicitly forwarding the sentinel
        # (rare; the v4.15.3 meta-pin path) still resolve cleanly.
        if return_kind is _RETURN_KIND_UNSET:
            return_kind = 'ensemble'
        result = create_gaussian_schell_source(
            N=N, dx=dx, wavelength=wavelength, w0=w0, sigma_g=sigma_g,
            n_realizations=n_realizations, seed=seed,
            return_kind=return_kind, **factory_kwargs)
        # v4.15.2 (Agent E): pass the factory's return value through
        # verbatim -- either the (ensemble, dx, dy, wavelength) tuple
        # or the bare PartialCoherenceMCF.  The pre-v4.15.2 path
        # wrapped the 3-D ensemble in a Source with a 3-D ``E``, which
        # broke the Source contract.  Downstream callers that want a
        # per-realisation Source wrap should do so explicitly:
        #     ens, _dx, _dy, _wl = Source.gaussian_schell(...)
        #     sources = [Source(E=ens[k], dx=_dx, dy=_dy,
        #                       wavelength=_wl) for k in range(len(ens))]
        # The ``source_point`` / ``name`` kwargs are accepted for API
        # symmetry with the coherent classmethods but are unused in
        # the ensemble-tuple return path.
        del source_point, name  # surface "unused" lint at the API
                                # boundary; v4.16+ may re-introduce a
                                # per-realisation wrapper that uses them.
        return result

    @classmethod
    def schell_model(cls, *, N: int, dx: float, wavelength: float,
                      intensity_profile: np.ndarray,
                      coherence_length: float,
                      n_realizations: int = 16,
                      source_point: _Tuple[float, float] = (0.0, 0.0),
                      name: _Optional[str] = None,
                      seed: _Optional[int] = None,
                      return_kind: Any = 'ensemble',
                      **factory_kwargs) -> Any:
        """Generic Schell-model partial-coherence source.

        Wraps :func:`create_schell_model_source`.  See
        :meth:`Source.gaussian_schell` for the v4.15.2 return-type
        convention (ensemble tuple by default, MCF object on
        ``return_kind='mcf'``; NOT a :class:`Source`-wrapped 3-D
        ensemble).

        v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        ``DeprecationWarning`` is retired in line with
        :meth:`Source.gaussian_schell`.  The default is plain
        ``'ensemble'`` and the call is silent.  The sentinel is
        preserved for back-compat (see :meth:`Source.gaussian_schell`
        docstring).

        2-D ``Source.E`` invariant break (intentional): the 4-tuple
        return has ``E_ensemble.shape == (n_realizations, Ny, Nx)``
        (3-D), breaking the 2-D-``E`` invariant other ``Source.*``
        classmethods uphold.  This is intentional -- Schell is
        partial-coherence, fundamentally different from coherent
        single-source.  See :meth:`Source.gaussian_schell` docstring
        for the full rationale and the v4.16+ ``Source.realizations()``
        per-realization-iterator plan.
        """
        # v4.16.1: default-path DeprecationWarning retired; sentinel
        # pass-through preserved (see Source.gaussian_schell).
        if return_kind is _RETURN_KIND_UNSET:
            return_kind = 'ensemble'
        result = create_schell_model_source(
            N=N, dx=dx, wavelength=wavelength,
            intensity_profile=intensity_profile,
            coherence_length=coherence_length,
            n_realizations=n_realizations, seed=seed,
            return_kind=return_kind, **factory_kwargs)
        # v4.15.2 (Agent E): same return-type contract as
        # ``Source.gaussian_schell`` -- pass the factory return through
        # verbatim; do not wrap the 3-D ensemble in a Source.
        del source_point, name
        return result

