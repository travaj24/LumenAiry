"""
Zernike polynomial decomposition of OPD / wavefront maps.

This submodule was carved out of ``lumenairy.analysis.core`` in v5.1.0
as part of the mechanical 6-file split (see ``ROADMAP.md`` v5.1
"Architecture / housekeeping").  All functions, signatures, and numerics
are unchanged -- the historical public API is preserved by a thin
re-export shell in ``lumenairy.analysis.core``.

OSA / ANSI single-index ordering is used throughout::

      j  |  (n, m)  |  classical name
     ---|----------|-----------------
      0  |  (0,  0) |  Piston
      1  |  (1, -1) |  Tilt  Y
      2  |  (1,  1) |  Tilt  X
      3  |  (2, -2) |  Oblique astigmatism
      4  |  (2,  0) |  Defocus
      5  |  (2,  2) |  Vertical astigmatism
      6  |  (3, -3) |  Oblique trefoil
      7  |  (3, -1) |  Vertical coma
      8  |  (3,  1) |  Horizontal coma
      9  |  (3,  3) |  Vertical trefoil
     10  |  (4, -4) |  Oblique quadrafoil
     11  |  (4, -2) |  Oblique secondary astigmatism
     12  |  (4,  0) |  Primary spherical
     13  |  (4,  2) |  Vertical secondary astigmatism
     14  |  (4,  4) |  Vertical quadrafoil
      ...

All Zernikes are normalised so that the rms of each mode over the
unit disk is 1.  Coefficients returned by :func:`zernike_decompose`
are therefore directly interpretable as RMS contributions in the
same units as the input OPD (meters if OPD is in meters).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    'zernike_index_to_nm',
    'zernike_nm_to_index',
    'zernike_polynomial',
    'zernike_basis_matrix',
    'zernike_decompose',
    'zernike_reconstruct',
    'clear_zernike_basis_cache',
    'astigmatism_mag_angle',
]


def zernike_index_to_nm(j: int) -> Tuple[int, int]:
    """Convert OSA single-index ``j`` to (n, m) Zernike indices."""
    j = int(j)
    if j < 0:
        raise ValueError(f"j must be >= 0, got {j}")
    # n = floor( (-1 + sqrt(1 + 8*j)) / 2 )
    n = int((-1 + np.sqrt(1 + 8 * j)) // 2)
    # Ensure n is large enough; guard floating-point edge
    while (n * (n + 2) - (j - n * (n + 1) // 2) * 2) < 0:
        n += 1
    m = 2 * j - n * (n + 2)
    return n, m


def zernike_nm_to_index(n: int, m: int) -> int:
    """Convert Zernike (n, m) to OSA single-index ``j``."""
    return (n * (n + 2) + m) // 2


def _zernike_radial(n, m, rho):
    """Radial polynomial R_n^m(rho) for rho in [0, 1].

    Computed via the explicit closed-form sum; stable and fast for
    ``n <= 20``.  Returns zero outside the unit disk.
    """
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    import math as _math
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        num = ((-1) ** s) * _math.factorial(n - s)
        den = (_math.factorial(s)
               * _math.factorial((n + m) // 2 - s)
               * _math.factorial((n - m) // 2 - s))
        R = R + (num / den) * rho ** (n - 2 * s)
    return R


def zernike_polynomial(
    n: int,
    m: int,
    rho: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Evaluate the OSA-normalised Zernike Z_n^m on pupil polar
    coordinates.

    Parameters
    ----------
    n : int
        Radial order (``n >= 0``).
    m : int
        Azimuthal order (``|m| <= n``, ``(n-m)`` even).
    rho : ndarray
        Normalised radius (0 outside pupil, 1 at pupil edge).
    theta : ndarray
        Azimuthal angle [rad], same shape as ``rho``.

    Returns
    -------
    Z : ndarray
        Zernike polynomial evaluated at (rho, theta), normalised so
        that the rms of ``Z`` over the unit disk equals 1.
    """
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    if (n - abs(m)) % 2 != 0 or abs(m) > n:
        raise ValueError(f"Invalid Zernike indices (n, m) = ({n}, {m})")
    # Normalisation constant (Noll 1976)
    if m == 0:
        N = np.sqrt(n + 1)
    else:
        N = np.sqrt(2 * (n + 1))
    R = _zernike_radial(n, m, rho)
    if m >= 0:
        angular = np.cos(m * theta)
    else:
        angular = np.sin(-m * theta)
    Z = N * R * angular
    # Zero outside pupil
    Z = np.where(rho <= 1.0, Z, 0.0)
    return Z


# ---------------------------------------------------------------------
# Zernike basis-matrix cache (audit perf #7, v4.12.0)
# ---------------------------------------------------------------------
#
# ``zernike_basis_matrix`` is hot during ``design_optimize`` runs:
# every Zernike-using merit term (RMSWavefrontMerit,
# MatchIdealThinLensMerit, MatchTargetOPDMerit, ZernikeCoefficientMerit,
# EvaluationContext.rms_wavefront_waves) rebuilds it on every call, and
# ``CompositeMerit`` evaluates several of these per ``evaluate(x)``.
# Across a finite-difference Jacobian sweep that's 60-100x rebuild of
# the same basis on the same grid -- ~20 ms each at 256x256, 21 modes.
#
# Cache strategy: content-fingerprint key.  The pupil grids X, Y are
# deterministic functions of (N, dx) -- so two distinct arrays with
# matching shape, dtype, and corner values are guaranteed (to within
# floating-point noise of the meshgrid arithmetic) to describe the
# same pupil.  Keying on the fingerprint (NOT on ``id()``) lets fresh
# ``np.meshgrid`` outputs across successive ``zernike_decompose`` /
# ``EvaluationContext.rms_wavefront_waves`` calls all hit the same
# cached basis.
#
# Trade-off: if a caller mutates an input array in place but does not
# change shape, dtype, or the first/last entries, the cache will
# still hit and return the stale basis.  This is highly unlikely in
# practice (pupil grids are rebuilt fresh each call) but documented
# here.  Use ``clear_zernike_basis_cache()`` to force a rebuild.
# ---------------------------------------------------------------------

_ZERNIKE_BASIS_CACHE: "OrderedDict[Any, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_ZERNIKE_BASIS_CACHE_MAXSIZE = 32
# v4.14.2 (P1-NEW-2 / Agent C): thread-safety lock for
# ``_ZERNIKE_BASIS_CACHE``.  Without this two threads racing through
# :func:`zernike_basis_matrix` could see a torn OrderedDict (``get`` ->
# ``__setitem__`` -> ``popitem`` is a read-modify-write sequence).
# Follows the ``_ASM_CACHE_LOCK`` precedent in
# :mod:`propagators.propagation`; the build itself (``_zernike_basis_
# matrix_build``) is pure-CPU numpy and re-entrant so it runs OUTSIDE
# the lock -- only the OrderedDict ops need guarding.
_ZERNIKE_BASIS_CACHE_LOCK = threading.Lock()


def _zernike_basis_cache_key(
    n_modes: int,
    X: np.ndarray,
    Y: np.ndarray,
    pupil_radius: float,
) -> Tuple[Any, ...]:
    """Build a (cheap) cache key for ``zernike_basis_matrix``.

    Uses a small content fingerprint -- shape, dtype, and the first +
    last entries of the grid -- as the cache key.  This intentionally
    *omits* object identity so that two distinct arrays produced by
    independent ``np.meshgrid`` calls (the common case in
    ``zernike_decompose`` and ``EvaluationContext.rms_wavefront_waves``)
    hit the same cache entry as long as their structure agrees.

    Trade-off: if a caller mutates the input arrays in place but does
    not change shape, dtype, or the corner values, the cache will
    still hit and return a stale basis.  This is unlikely in practice
    (pupil grids are rebuilt fresh each call) but documented for
    completeness.  Call ``clear_zernike_basis_cache()`` to force a
    rebuild.
    """
    # Coerce to ndarray for shape/dtype/corner access without copying.
    Xa = np.asarray(X)
    Ya = np.asarray(Y)
    return (
        int(n_modes),
        Xa.shape, Xa.dtype.str,
        Ya.shape, Ya.dtype.str,
        float(Xa.flat[0]), float(Xa.flat[-1]),
        float(Ya.flat[0]), float(Ya.flat[-1]),
        float(pupil_radius),
    )


def clear_zernike_basis_cache() -> None:
    """Drop every cached Zernike basis matrix.

    Useful when the caller has mutated a coordinate grid in place and
    wants the next ``zernike_basis_matrix`` call to recompute from
    scratch.  Also handy in unit tests that pin cache behaviour.
    """
    with _ZERNIKE_BASIS_CACHE_LOCK:
        _ZERNIKE_BASIS_CACHE.clear()


# v4.16.0 (ROADMAP #15): register the Zernike-basis clearer with the
# central registry at module-import time.  ``clear_asm_caches`` now
# walks the registry rather than enumerating clear calls by hand.
# Late-binding closure preserves ``mock.patch.object`` test semantic.
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'zernike_basis',
        lambda: getattr(_this_mod, 'clear_zernike_basis_cache')(),
    )
except ImportError:
    pass


def _zernike_basis_matrix_build(
    n_modes: int,
    X: np.ndarray,
    Y: np.ndarray,
    pupil_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uncached build of the Zernike basis matrix.  See
    :func:`zernike_basis_matrix` for the public, cached entry point.
    """
    r_sq = (X ** 2 + Y ** 2) / (pupil_radius ** 2)
    pupil_mask = r_sq <= 1.0
    rho = np.sqrt(r_sq[pupil_mask])
    theta = np.arctan2(Y[pupil_mask], X[pupil_mask])

    n_pixels = rho.size
    basis = np.empty((n_pixels, n_modes), dtype=np.float64)
    for j in range(n_modes):
        n, m = zernike_index_to_nm(j)
        basis[:, j] = zernike_polynomial(n, m, rho, theta)
    return basis, pupil_mask


def zernike_basis_matrix(
    n_modes: int,
    X: np.ndarray,
    Y: np.ndarray,
    pupil_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a design matrix of the first ``n_modes`` Zernike
    polynomials evaluated on the grid ``(X, Y)``.

    Returns a ``(N_pupil_pixels, n_modes)`` 2-D array where each row
    is one pupil pixel and each column is one Zernike mode.  Only
    pixels inside the pupil (``x^2 + y^2 <= pupil_radius^2``) are
    included in the returned matrix.

    Parameters
    ----------
    n_modes : int
    X, Y : ndarray
        Pupil-plane coordinate grids [m] (same shape).
    pupil_radius : float
        Pupil radius [m].

    Returns
    -------
    basis : ndarray, shape (N_pixels, n_modes)
    pupil_mask : ndarray of bool, same shape as X
        True where pixels are inside the pupil (= rows of ``basis``).

    Notes
    -----
    Results are cached (LRU, ``maxsize=32``) keyed on a small content
    fingerprint of the input grids (shape, dtype, first + last
    entries) plus ``n_modes`` and ``pupil_radius``.  Two structurally
    identical grids hit the same cache entry even if they are distinct
    array objects -- this is the common case under
    ``design_optimize``.  The same underlying arrays are returned on a
    cache hit -- treat the returned ``basis`` and ``pupil_mask`` as
    immutable; copy first if you need to modify them.  Call
    :func:`clear_zernike_basis_cache` to drop the cache (e.g. after
    mutating a grid in place).
    """
    key = _zernike_basis_cache_key(n_modes, X, Y, pupil_radius)
    with _ZERNIKE_BASIS_CACHE_LOCK:
        cached = _ZERNIKE_BASIS_CACHE.get(key)
        if cached is not None:
            # LRU bump: mark as most recently used.
            _ZERNIKE_BASIS_CACHE.move_to_end(key)
            return cached

    # Cache miss: build the basis OUTSIDE the lock (pure-CPU numpy,
    # re-entrant; lock-scope discipline keeps expensive work off the
    # critical section).  Two threads may double-build on a cold
    # cache for the same key -- benign waste, the second insert just
    # overwrites the first.
    basis, pupil_mask = _zernike_basis_matrix_build(
        n_modes, X, Y, pupil_radius)

    with _ZERNIKE_BASIS_CACHE_LOCK:
        _ZERNIKE_BASIS_CACHE[key] = (basis, pupil_mask)
        # Evict LRU entries until we're at or below the cap.
        while len(_ZERNIKE_BASIS_CACHE) > _ZERNIKE_BASIS_CACHE_MAXSIZE:
            _ZERNIKE_BASIS_CACHE.popitem(last=False)
    return basis, pupil_mask


# ----------------------------------------------------------------------
# v5.4 Phase 5 -- normalization & weighting per-mode rescaling helpers
# ----------------------------------------------------------------------
#
# The library's internal basis is OSA/ANSI orthonormal on the unit disk
# (see :func:`zernike_polynomial`).  ``zernike_decompose`` now accepts a
# ``normalization=`` kwarg by rescaling the OSA basis on a per-mode
# basis BEFORE the least-squares solve, then unscaling the recovered
# coefficients so callers receive coefficients expressed in the
# requested convention.  No new basis-matrix builds are needed.
#
# Conventions implemented
# -----------------------
# * ``'OSA'``       -- ANSI/OSA orthonormal (current behavior).
#                      Per-mode factor = 1.  Reference: ANSI Z80.28-2010.
# * ``'Noll'``      -- Noll 1976 (J. Opt. Soc. Am. 66, 207-211).  Same
#                      magnitudes as OSA, but the sign convention on
#                      the sine-like (m < 0) modes is opposite to OSA.
#                      Implemented as factor = -1 for m < 0, +1
#                      otherwise.  This matches the most-documented
#                      practical difference; if the caller really wants
#                      Noll's j-ordering as well they can permute
#                      after the fact.
# * ``'Standard'``  -- ANSI/Z80.28-2010 Born-Wolf form.  Algebraically
#                      identical to OSA (factor = 1).  Provided
#                      explicitly so dock dropdowns don't have to
#                      special-case the name.
# * ``'Fringe'``    -- Wyant / U-Arizona "Fringe" ordering uses
#                      single-index 1..37 with a DIFFERENT mode set
#                      and unnormalised peak-value-1 polynomials.
#                      Not implementable as a pure per-mode rescale of
#                      the OSA basis; ``NotImplementedError`` until a
#                      dedicated Fringe basis lands (planned: v5.5).
#
# The rescaling identity, for any per-mode factor ``s_j``:
#   if OSA basis is B and the requested basis is B' with
#   B'[:, j] = s_j * B[:, j], then a least-squares fit of OPD against
#   B' returns coefficients c'_j = c_j / s_j (where c_j are OSA
#   coefficients).  Equivalently we can solve against B (cached) and
#   divide: c'_j = c_j / s_j.  We do exactly that -- it avoids a
#   second basis-matrix build and a second cache key.
# ----------------------------------------------------------------------

_VALID_NORMALIZATIONS = ('OSA', 'Noll', 'Standard', 'Fringe')


def _normalization_factors(n_modes: int, normalization: str) -> np.ndarray:
    """Per-mode multiplicative rescale to go from OSA -> requested
    convention.

    Returns an array ``s`` of shape ``(n_modes,)`` such that, in the
    requested convention, the j-th Zernike polynomial equals
    ``s[j] * Z_j^{OSA}``.  Coefficients returned to the caller are
    divided by ``s`` so they represent expansion in the requested
    basis.

    Raises ``ValueError`` for unknown names and ``NotImplementedError``
    for conventions whose basis differs structurally from OSA (Fringe).
    """
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {normalization!r}; expected one of "
            f"{_VALID_NORMALIZATIONS}.")
    if normalization == 'Fringe':
        # Fringe uses a different mode set (37 polynomials with non-OSA
        # ordering and peak-value-1 instead of RMS-1).  Cannot express
        # as a per-mode scale of the OSA basis; needs a dedicated
        # Fringe basis builder, planned for v5.5.
        raise NotImplementedError(
            "normalization='Fringe' not yet supported; the Fringe "
            "convention uses a different mode ordering AND peak-value-1 "
            "(not RMS-1) polynomials, so it cannot be implemented as a "
            "per-mode rescale of the OSA basis.  Planned for v5.5; "
            "use 'OSA' or 'Noll' for now.")
    if normalization in ('OSA', 'Standard'):
        # ANSI/Z80.28-2010 ('Standard') is algebraically identical to
        # OSA/ANSI orthonormal -- both give N_n^m = sqrt(2(n+1)/
        # (1+delta_{m,0})).  No rescale required.
        return np.ones(n_modes, dtype=np.float64)
    # 'Noll': flip sign on sine-like (m < 0) modes vs OSA.  Magnitudes
    # are identical, so |c_j^{Noll}| = |c_j^{OSA}| and RMS aggregates
    # are unchanged; only signs on m<0 entries differ.
    s = np.ones(n_modes, dtype=np.float64)
    for j in range(n_modes):
        _n, m = zernike_index_to_nm(j)
        if m < 0:
            s[j] = -1.0
    return s


def zernike_decompose(
    opd_map: np.ndarray,
    dx: float,
    aperture: float,
    n_modes: int = 21,
    dy: Optional[float] = None,
    return_residual: bool = False,
    normalization: str = 'OSA',
    weighting: Optional[np.ndarray] = None,
) -> Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[str], np.ndarray, float]]:
    """Decompose a 2-D OPD map into Zernike coefficients using a
    numerically-stable Householder QR least-squares solve.

    Parameters
    ----------
    opd_map : ndarray (2-D, real)
        Optical path difference [m] over a grid.  Values outside the
        pupil may be ``NaN`` or 0; they are masked out before fitting.
    dx : float
        Grid spacing in x [m].
    aperture : float
        Clear aperture diameter [m].  Defines the pupil radius as
        ``aperture / 2``.
    n_modes : int, default 21
        Number of OSA-indexed Zernike modes to fit.  21 covers up
        through 5th-order spherical.  Higher = finer detail at the
        cost of ill-conditioning for sparsely-illuminated pupils.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    return_residual : bool, default False
        Also return the 2-D residual ``opd_map - reconstruction`` and
        its RMS.
    normalization : {'OSA', 'Noll', 'Standard', 'Fringe'}, default 'OSA'
        Zernike normalization convention.  ``'OSA'`` (default) and
        ``'Standard'`` (ANSI/Z80.28-2010 Born-Wolf form) are
        orthonormal on the unit disk and produce algebraically
        identical coefficients.  ``'Noll'`` (Noll 1976, J. Opt. Soc.
        Am. 66, 207-211) has the same magnitudes as OSA but the
        opposite sign convention on the sine-like (m < 0) modes --
        i.e. ``c_j^{Noll} = -c_j^{OSA}`` for any j whose OSA azimuthal
        order is negative; magnitudes and RMS-sums are unchanged.
        ``'Fringe'`` (Wyant / U-Arizona) raises
        ``NotImplementedError`` -- the Fringe convention uses a
        different mode set AND peak-value-1 polynomials, so it cannot
        be expressed as a per-mode rescale of the OSA basis (planned
        v5.5).
    weighting : ndarray of shape ``(Ny, Nx)``, optional
        Per-pixel weight applied during the least-squares solve.
        Each pupil row of the design matrix and the corresponding OPD
        sample is multiplied by ``sqrt(weighting)``, which is the
        canonical weighted-least-squares transformation (minimises
        ``sum_i w_i * (opd_i - sum_j c_j Z_j(x_i, y_i))**2``).  Useful
        when the pupil has obstructions or edge irregularities you
        want to suppress.  Defaults to ``None`` (uniform weight = 1
        on every pupil pixel; identical to v5.3.2).

    Returns
    -------
    coeffs : ndarray, shape (n_modes,)
        Fitted Zernike coefficients, units of ``opd_map`` (typically
        meters).  With OSA normalisation, the RMS wavefront error
        contributed by mode ``j`` is ``|coeffs[j]|``.
    names : list of str
        Human-readable name for each mode (e.g. ``'Defocus'``,
        ``'Primary spherical'``).
    residual : ndarray, optional
        ``opd_map - reconstruction``, 2-D.  Only returned when
        ``return_residual=True``.
    rms_residual : float, optional
        RMS of ``residual`` over the pupil, same units as
        ``opd_map``.  Only returned when ``return_residual=True``.

    Notes
    -----
    Uses ``scipy.linalg.lstsq(..., lapack_driver='gelsy')`` under the
    hood, which is a column-pivoted Householder QR with rank
    revelation.  This is more stable than the default SVD driver for
    ill-conditioned Zernike bases (common when the pupil is partially
    illuminated or when many modes are requested).

    Backward compatibility: calling ``zernike_decompose(opd, dx,
    aperture)`` with no ``normalization=`` / ``weighting=`` kwargs is
    bit-for-bit identical to v5.3.2 output.

    References
    ----------
    * Noll, R.J. (1976), "Zernike polynomials and atmospheric
      turbulence", J. Opt. Soc. Am. 66 (3), 207-211.
    * ANSI Z80.28-2010 "Ophthalmics -- Methods of reporting optical
      aberrations of eyes".
    * Mahajan, V.N. (2013), "Optical Imaging and Aberrations, Part
      III: Wavefront Analysis", SPIE Press.
    """
    if dy is None:
        dy = dx
    Ny, Nx = opd_map.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_pupil = 0.5 * aperture

    basis, pupil_mask = zernike_basis_matrix(n_modes, X, Y, r_pupil)

    # Validate / compute per-mode rescale.  Resolved BEFORE the
    # weighting branch so a 'Fringe' or 'Bogus' choice fails fast.
    norm_scale = _normalization_factors(n_modes, normalization)

    # Validate weighting shape (if any).  We accept either None (no
    # weighting, the historical default) or a 2-D array matching
    # ``opd_map``.  Negative or NaN weights are tolerated and treated
    # as "exclude this pixel" (sqrt(w<=0) -> 0 row contribution).
    weight_flat = None
    if weighting is not None:
        weighting = np.asarray(weighting)
        if weighting.shape != opd_map.shape:
            raise ValueError(
                f'weighting shape {weighting.shape} does not match '
                f'opd_map shape {opd_map.shape}.')
        weight_flat = weighting[pupil_mask].astype(np.float64, copy=False)
        # Mask non-positive / non-finite weights -- treat them like
        # outside-pupil so they drop out of the fit cleanly.
        weight_flat = np.where(
            np.isfinite(weight_flat) & (weight_flat > 0.0),
            weight_flat, 0.0)

    # Flatten the OPD to match basis rows
    opd_flat = opd_map[pupil_mask]
    # Drop NaN/inf rows from both sides
    finite = np.isfinite(opd_flat)
    if not finite.all():
        basis = basis[finite]
        opd_flat = opd_flat[finite]
        if weight_flat is not None:
            weight_flat = weight_flat[finite]
    # Drop zero-weight rows (uniform-fit path is unchanged because
    # ``weight_flat is None`` skips this branch).
    if weight_flat is not None:
        nonzero = weight_flat > 0.0
        if not nonzero.all():
            basis = basis[nonzero]
            opd_flat = opd_flat[nonzero]
            weight_flat = weight_flat[nonzero]
    if opd_flat.size < n_modes:
        raise ValueError(
            f'Not enough valid pupil samples ({opd_flat.size}) to fit '
            f'{n_modes} modes.  Check aperture/grid alignment or '
            f'reduce n_modes.')

    # Apply weighted-least-squares transformation: multiply each row
    # of (basis | opd) by sqrt(w_i).  Skipping this entirely when
    # ``weighting is None`` preserves bit-for-bit v5.3.2 output.
    if weight_flat is not None:
        sw = np.sqrt(weight_flat)
        basis = basis * sw[:, None]
        opd_flat = opd_flat * sw

    # Householder QR with column pivoting (gelsy driver)
    try:
        from scipy.linalg import lstsq as _slstsq
        coeffs, _residuals_sq, _rank, _sv = _slstsq(
            basis, opd_flat, lapack_driver='gelsy')
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # Fallback to numpy if scipy lstsq is unavailable
        # (ImportError) or rejects the inputs / diverges
        # (ValueError, LinAlgError).
        coeffs, *_ = np.linalg.lstsq(basis, opd_flat, rcond=None)

    # Rescale OSA coefficients into the requested convention.  For
    # 'OSA' and 'Standard' this is a no-op (factors are all 1.0).
    coeffs = coeffs / norm_scale

    names = [_zernike_classical_name(*zernike_index_to_nm(j))
             for j in range(n_modes)]

    if return_residual:
        # Reconstruct over the full pupil, not just the finite subset.
        # Reconstruction uses the OSA basis with OSA-equivalent
        # coefficients (coeffs * norm_scale) so the residual measures
        # the same wavefront regardless of the requested normalization.
        basis_full, _ = zernike_basis_matrix(
            n_modes, X, Y, r_pupil)
        recon_flat = basis_full @ (coeffs * norm_scale)
        reconstruction = np.zeros_like(opd_map)
        reconstruction[pupil_mask] = recon_flat
        residual = opd_map - reconstruction
        residual = np.where(pupil_mask, residual, np.nan)
        rms = float(np.sqrt(np.nanmean(residual ** 2)))
        return coeffs, names, residual, rms
    return coeffs, names


def zernike_reconstruct(
    coeffs: np.ndarray,
    dx: float,
    shape: Tuple[int, int],
    aperture: float,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Reconstruct a 2-D OPD map from Zernike coefficients.

    Inverse of :func:`zernike_decompose`: ``opd_map ≈ sum_j coeffs[j]
    * Z_j(x, y)`` inside the pupil, zero outside.

    Parameters
    ----------
    coeffs : ndarray, shape (n_modes,)
        Coefficients in OSA order.
    dx : float
        Grid spacing [m].
    shape : tuple (Ny, Nx)
        Output grid shape.
    aperture : float
        Pupil diameter [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    opd_map : ndarray, shape ``shape``
        Reconstructed OPD, zero outside the pupil.
    """
    if dy is None:
        dy = dx
    Ny, Nx = shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_pupil = 0.5 * aperture
    n_modes = len(coeffs)

    basis, pupil_mask = zernike_basis_matrix(n_modes, X, Y, r_pupil)
    recon_flat = basis @ np.asarray(coeffs, dtype=np.float64)
    opd_map = np.zeros(shape, dtype=np.float64)
    opd_map[pupil_mask] = recon_flat
    return opd_map


def _zernike_classical_name(n, m):
    """Return the classical name for a Zernike mode (n, m).

    Covers common aberration names; falls back to ``Z(n, m)`` for
    uncommon modes.
    """
    table = {
        (0, 0): 'Piston',
        (1, -1): 'Tilt Y',
        (1, 1): 'Tilt X',
        (2, -2): 'Oblique astigmatism',
        (2, 0): 'Defocus',
        (2, 2): 'Vertical astigmatism',
        (3, -3): 'Oblique trefoil',
        (3, -1): 'Vertical coma',
        (3, 1): 'Horizontal coma',
        (3, 3): 'Vertical trefoil',
        (4, -4): 'Oblique quadrafoil',
        (4, -2): 'Oblique secondary astigmatism',
        (4, 0): 'Primary spherical',
        (4, 2): 'Vertical secondary astigmatism',
        (4, 4): 'Vertical quadrafoil',
        (5, -1): 'Secondary vertical coma',
        (5, 1): 'Secondary horizontal coma',
        (6, 0): 'Secondary spherical',
        (8, 0): 'Tertiary spherical',
    }
    return table.get((n, m), f'Z({n}, {m})')


# ============================================================================
# v4.15.0 (C.4) -- Astigmatism magnitude + angle
# ============================================================================

def astigmatism_mag_angle(
    coeffs: Sequence[float],
) -> Tuple[float, float]:
    r"""Magnitude and orientation angle of primary astigmatism.

    Returns ``(|astig|, theta)`` from an OSA-indexed Zernike
    coefficient array as returned by :func:`zernike_decompose`.

    Convention (matches :func:`zernike_decompose` / OSA / ANSI):

    * ``coeffs[3]``  = Z(2, -2), "oblique astigmatism"   (= c3)
    * ``coeffs[5]``  = Z(2,  2), "vertical astigmatism"  (= c5)

    The magnitude is the quadrature sum, and the angle is the
    physical orientation of the principal axis of the astigmatism
    figure (per Mahajan, "Aberration Theory Made Simple", 2nd ed.,
    §8.2):

    .. math::
        |\mathrm{astig}| = \sqrt{c_3^2 + c_5^2}, \qquad
        \theta = \tfrac{1}{2} \, \mathrm{atan2}(c_3, c_5).

    Because astigmatism has C2 symmetry, the principal-axis angle
    ``theta`` is unique modulo :math:`\pi/2` (any whole-:math:`\pi/2`
    multiple maps the figure onto itself).  The formula returns a
    value in :math:`(-\pi/2,\, \pi/2]` -- the half-angle of a full
    :math:`(-\pi,\, \pi]` ``atan2`` range -- which spans a full
    period of the principal-axis ambiguity.  Callers who want a
    canonical-quadrant orientation should fold ``theta`` into
    ``[0, pi/2)`` by adding :math:`\pi/2` when ``theta < 0``.

    Parameters
    ----------
    coeffs : sequence of float
        OSA-indexed Zernike coefficients.  Must have length at least
        6 so that ``coeffs[3]`` and ``coeffs[5]`` are addressable.

    Returns
    -------
    magnitude : float
        ``sqrt(coeffs[3]**2 + coeffs[5]**2)``, same units as
        ``coeffs`` (e.g. metres of RMS wavefront error if ``coeffs``
        comes from a metres-OPD decomposition).
    theta : float
        Principal-axis orientation [rad].

    Raises
    ------
    ValueError
        If ``coeffs`` has fewer than 6 entries (cannot index c5).

    See Also
    --------
    zernike_decompose : Zernike fit producing the input array.

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.analysis import astigmatism_mag_angle
    >>> # Pure vertical astigmatism (c5 = +1, c3 = 0): theta = 0.
    >>> c = np.zeros(6); c[5] = 1.0
    >>> mag, theta = astigmatism_mag_angle(c)
    >>> bool(abs(mag - 1.0) < 1e-12 and abs(theta) < 1e-12)
    True
    >>> # Pure oblique astigmatism (c3 = +1, c5 = 0): theta = pi/4.
    >>> c = np.zeros(6); c[3] = 1.0
    >>> mag, theta = astigmatism_mag_angle(c)
    >>> bool(abs(mag - 1.0) < 1e-12 and abs(theta - np.pi/4) < 1e-12)
    True
    """
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    if coeffs_arr.ndim != 1:
        raise ValueError(
            f"astigmatism_mag_angle: coeffs must be 1-D; got shape "
            f"{coeffs_arr.shape!r}.")
    if coeffs_arr.size < 6:
        raise ValueError(
            f"astigmatism_mag_angle: need at least 6 coefficients "
            f"(OSA c5 = coeffs[5]); got length {coeffs_arr.size}.")
    c3 = float(coeffs_arr[3])  # OSA (n=2, m=-2): oblique astigmatism
    c5 = float(coeffs_arr[5])  # OSA (n=2, m=+2): vertical astigmatism
    magnitude = float(np.sqrt(c3 * c3 + c5 * c5))
    theta = 0.5 * float(np.arctan2(c3, c5))
    return magnitude, theta
