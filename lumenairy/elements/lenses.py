"""
Lens and focusing element phase screens.

This module provides phase-screen models for refractive focusing elements:
thin lenses (paraxial through aplanatic), thick singlets (spherical and
aspheric), multi-surface real lenses with split-step propagation through
glass, cylindrical lenses, axicons, and GRIN rod lenses.

All functions follow the exp(-i*omega*t) time convention and use SI meters
for spatial quantities.

Backends
--------
Most functions use NumPy by default.  ``apply_thin_lens``,
``apply_spherical_lens``, ``apply_aspheric_lens``, and
``apply_real_lens`` accept a *use_gpu* flag (or auto-dispatch when
the input is a CuPy array) and run entirely on the GPU when CuPy
is installed.  Per-surface FFT propagation, phase-screen multiply,
and aperture clipping all use the array-API ``xp.*`` namespace, so
NumPy and CuPy arrays follow the same code path.  At N=32768 with
a 16-surface prescription, the GPU path is typically 5-10x faster
than the threaded pyFFTW CPU path (NVIDIA RTX 4090 vs Ryzen 7950X).

``apply_real_lens_traced`` accepts ``use_gpu=True`` for its
Newton-inversion polynomial-fit stage; the inner ray trace and
amplitude leg can also be GPU-accelerated independently via
``amp_use_gpu=True``.

Author: Andrew Traverso
"""

from __future__ import annotations

# GPU backend ----------------------------------------------------------------
# Lazy: only check availability via find_spec, defer the actual import
# to first use.  Saves ~200 ms at lumenairy import time on machines
# with CuPy installed.
import importlib.util as _importlib_util
from typing import Dict, Optional, Tuple

import numpy as np

# Optional CuPy backend (lazy).  The availability probe, the first-use import
# and the isinstance test live in ONE place for the whole library
# (``backend/_optional.py``; audit 2026-09-11 TESTS-ARCH P2-9).
from ..backend._optional import (
    CUPY_AVAILABLE,
    ensure_cupy as _ensure_cupy,
    is_cupy_array as _optional_is_cupy_array,
)

# The grid-versus-aperture bookkeeping, re-exported so every
# ``lenses.check_grid_vs_apertures`` spelling resolves here.  It lives in a LEAF
# that imports nothing from ``lumenairy``, which is what lets ``_lens_traced``
# reach it without closing a module-level import cycle with this facade.
from ._lens_kernels import (  # noqa: F401 -- re-export, see the block below
    _collect_semi_diameters as _collect_semi_diameters,
    _warn_if_aperture_exceeds_grid as _warn_if_aperture_exceeds_grid,
    check_grid_vs_apertures as check_grid_vs_apertures,
    recommend_grid_for_prescription as recommend_grid_for_prescription,
)

cp = None  # this module's alias for the cupy module; see _ensure_cupy_loaded


def _ensure_cupy_loaded():
    """Load CuPy on first use; return True iff it is available.

    Keeps this module's ``cp`` alias populated because the GPU branches here
    -- and ``_lens_thin``'s PEP 562 ``cp`` forward, which reads
    ``_lenses_module.cp`` -- resolve the module-level name directly.
    """
    global cp
    if cp is None:
        cp = _ensure_cupy()
    return cp is not None


# Optional fused-expression backend ------------------------------------------
# numexpr evaluates array expressions in chunked, multi-threaded passes
# without materialising full N x N intermediates.  Used by apply_real_lens
# to fuse the ``E * exp(-1j*k0*opd)`` phase-screen multiply, which at
# N=32768 otherwise allocates three 17 GB complex128 temporaries.
NUMEXPR_AVAILABLE = _importlib_util.find_spec('numexpr') is not None
_ne = None  # populated by _ensure_numexpr_loaded() on first use


def _ensure_numexpr_loaded():
    global _ne
    if _ne is None and NUMEXPR_AVAILABLE:
        import numexpr as _n
        _ne = _n
    return _ne is not None

# Optional Numba JIT, LAZILY imported on first kernel use (audit P2-D: the eager
# ``import numba`` cost ~1.8 s of ``import lumenairy`` cold start).  Used by the
# fused polynomial-aspheric loop ``_aspheric_sag_accum_numba`` (3.2.14), which has
# a pure-NumPy fallback -- so numba is pulled in only when a caller actually hits
# that fast path AND numba is installed.  ``find_spec`` checks availability
# WITHOUT importing numba.
from ..backend._optional import (
    NUMBA_AVAILABLE as _OPTIONAL_NUMBA_AVAILABLE,
    numba_handles as _optional_numba_handles,
)

# The MODULE-LEVEL ``_NUMBA_AVAILABLE`` is load-bearing and stays a module
# attribute: it is read at CALL time and the test suite monkeypatches it to
# ``False`` to reach the pure-NumPy arm on a box where numba IS installed.  So
# the availability GATE is local while the import is shared
# (``backend/_optional.py``; audit 2026-09-11 TESTS-ARCH P2-9).
_NUMBA_AVAILABLE = _OPTIONAL_NUMBA_AVAILABLE
_numba = None                         # populated by _load_numba() on first use
_njit = None
_prange = None
_NUMBA_KERNELS: dict = {}             # kernel-name -> compiled fn (or None)


def _load_numba():
    """Import numba + njit/prange on first use; cache the handles.  Returns True
    iff numba is importable (False -> callers take the pure-NumPy fallback).

    Honours a monkeypatched module-level ``_NUMBA_AVAILABLE = False`` -- this
    library's spelling for "pretend the accelerator is absent" -- before
    consulting the shared loader."""
    global _numba, _njit, _prange
    if _numba is not None:
        return True
    if not _NUMBA_AVAILABLE:
        return False
    _numba, _njit, _prange = _optional_numba_handles()
    return _numba is not None


# The numexpr scaffold ABOVE (``NUMEXPR_AVAILABLE`` / ``_ne`` /
# ``_ensure_numexpr_loaded``) is NOT dead: ``lenses_maslov`` imports the
# flag and the loader from this module, and ``elements/__init__`` +
# ``lumenairy/__init__`` re-export ``NUMEXPR_AVAILABLE`` publicly.


def _is_cupy_array(x):
    """
    Reliable CuPy array check.  ``hasattr(x, 'device')`` used to be a
    duck-type test for a CuPy device array but broke in NumPy 2.x
    (``ndarray`` now exposes ``.device`` via the Array API standard),
    causing every NumPy array to get routed into the CuPy branch.

    ``_lens_thin`` asks ``backend._optional.is_cupy_array`` directly rather
    than delegating here -- it takes the same answer from the same helper,
    without the module-level import back into this file that a delegation
    would need.  The extra short-circuit below is the only difference, and it
    is about call cost, not about the answer.
    """
    if not CUPY_AVAILABLE:
        # Local short-circuit, not a delegation: this is the hot per-call
        # dispatch for the whole thin/spherical/aspheric family and the
        # CuPy-absent answer must stay one global read.
        return False
    if not _optional_is_cupy_array(x):
        return False
    _ensure_cupy_loaded()   # a True answer implies ``cp`` is live -- bind it
    return True



# ---------------------------------------------------------------------------
# Helper: general conic + aspheric surface sag
# ---------------------------------------------------------------------------

def _get_aspheric_sag_accum_numba():
    """Compile (once, on first call) and return the fused aspheric-sag numba
    kernel, or ``None`` if numba is unavailable.  Lazy so ``import lumenairy``
    never pays the numba import / compile cost (audit P2-D)."""
    if "aspheric_sag" in _NUMBA_KERNELS:
        return _NUMBA_KERNELS["aspheric_sag"]
    if not _load_numba():
        _NUMBA_KERNELS["aspheric_sag"] = None
        return None

    @_njit(cache=True, parallel=True, fastmath=True)
    def _aspheric_sag_accum_numba(h_sq, sag, powers, coeffs):
        """In-place accumulate sum_i coeff_i * h_sq**(power_i // 2) onto
        ``sag``.  Single fused pass over h_sq, no temporary arrays.

        ``h_sq`` and ``sag`` must be contiguous float64 arrays of the
        same shape.  ``powers`` is int32, ``coeffs`` is float64; both
        1-D and same length.
        """
        flat_h = h_sq.ravel()
        flat_s = sag.ravel()
        n = flat_h.size
        n_terms = powers.size
        for i in _prange(n):
            v = flat_h[i]
            acc = 0.0
            for j in range(n_terms):
                p = powers[j] // 2
                # h_sq^p via repeated squaring keeps the inner loop
                # branch-free (Numba unrolls small fixed-power loops).
                hp = 1.0
                for _ in range(p):
                    hp *= v
                acc += coeffs[j] * hp
            flat_s[i] += acc

    _NUMBA_KERNELS["aspheric_sag"] = _aspheric_sag_accum_numba
    return _aspheric_sag_accum_numba


def surface_sag_general(
    h_sq: np.ndarray,
    R: float,
    conic: float = 0.0,
    aspheric_coeffs: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Compute surface sag for a general conic + even-aspheric surface.

    This function is used by both the lens and mirror modules, so it is
    exported at module level (no leading underscore).

    Parameters
    ----------
    h_sq : ndarray
        Squared radial distance from the optical axis, x**2 + y**2  [m**2].
    R : float
        Radius of curvature [m].  Use ``float('inf')`` or ``np.inf`` for a
        flat surface.
    conic : float, optional
        Conic constant (default 0 = sphere, -1 = paraboloid, < -1 =
        hyperboloid, -1 < k < 0 = prolate ellipsoid, > 0 = oblate ellipsoid).
    aspheric_coeffs : dict or None, optional
        Even polynomial aspheric coefficients ``{power: coeff}``, e.g.
        ``{4: A4, 6: A6, 8: A8, 10: A10}``.  Each term contributes
        ``coeff * h_sq**(power // 2)`` to the sag.

    Returns
    -------
    sag : ndarray
        Signed surface sag (positive when R > 0).
    """
    # Array-API polymorphic: detect cupy vs numpy from the input.
    # This keeps the helper usable from both the CPU and GPU paths of
    # apply_real_lens without duplicating code.  Arithmetic
    # broadcasting (np.where, np.sqrt on cupy arrays) silently
    # converts to host, so we dispatch explicitly.
    xp = cp if _is_cupy_array(h_sq) else np

    # ``R = 0`` and ``R = nan`` are not surfaces: the conic expression divides
    # by ``R**2`` and then by ``R``, so both returned an all-NaN sag behind
    # anonymous numpy RuntimeWarnings ("divide by zero", "invalid value") that
    # name neither this function nor the offending key -- and a NaN radius
    # propagates silently all the way into the phase screen, where it zeroes
    # the whole field.  ``R = inf`` and ``R = None`` are the two spellings of a
    # FLAT surface and are handled below; anything else non-finite or zero is a
    # malformed prescription and gets the CONVENTIONS SS2 message.
    if R is not None and not np.isinf(R) and not (float(R) != 0.0
                                                  and np.isfinite(R)):
        _what = 'nan' if not np.isfinite(R) else '0'
        raise ValueError(
            f"surface_sag_general: radius R = {_what} is not a surface (the "
            f"conic sag divides by R).  Use R = np.inf or R = None for a FLAT "
            f"surface; a finite non-zero R for a curved one.")

    if R is not None and not np.isinf(R):
        # Conic sag: h^2 / (R * (1 + sqrt(1 - (1+k)*h^2/R^2)))
        # 4.10: outside the conic domain (norm >= 0.9999) the surface
        # is not defined.  A silent 0 sag there produces an
        # which produced an apparently-flat ring at the surface edge
        # for hyperbolic / oblate conics extending past the geometric
        # rim.  Return NaN instead so downstream consumers either mask
        # those pixels (via an aperture mask) or see the failure.
        #
        # Written as one in-place chain through a single scratch grid.  The
        # expression-per-line form allocated a fresh full grid for each of
        # ``norm``, ``denom_arg``, the ``sqrt``, the ``1 + ...``, the ``R * ...``,
        # the division and the final ``where`` -- 5.13 float64 grids at a
        # tracemalloc peak, 4.5x the wall clock of the identical arithmetic
        # written with ``out=`` (248 -> 55 ms at N = 2048), and 22 % of a
        # default three-surface ``apply_real_lens`` call.  Every operation and
        # its ORDER is unchanged, so the result is bit-identical; only the
        # temporaries are gone.  The domain mask stays a separate bool grid
        # (1/8 of a float64 one) because ``norm < 0.9999`` has to be taken
        # BEFORE ``norm`` is overwritten -- and it is taken in that sense and
        # then inverted, rather than as ``>= 0.9999``, so a NaN ``h_sq`` lands
        # on the INVALID side exactly where ``xp.where`` put it.
        sag = xp.multiply(h_sq, (1 + conic))
        sag = xp.divide(sag, R**2, out=sag)
        invalid = sag < 0.9999
        xp.logical_not(invalid, out=invalid)
        xp.subtract(1, sag, out=sag)
        sag[invalid] = 0.01
        xp.sqrt(sag, out=sag)
        xp.add(sag, 1, out=sag)
        xp.multiply(sag, R, out=sag)
        xp.divide(h_sq, sag, out=sag)
        sag[invalid] = xp.nan
        del invalid
    else:
        sag = xp.zeros_like(h_sq)

    if aspheric_coeffs:
        # Reject ODD powers HERE, at the
        # wave-optics sag entry point.  Both branches below evaluate
        # ``h_sq ** (power // 2)`` (the numba kernel's ``powers[j] // 2`` and
        # the NumPy fallback's ``power // 2``), so an odd power silently floors
        # to the NEXT-LOWER EVEN one -- a different surface, returned with no
        # diagnostic.  Measured pre-guard at ``{5: 1e6}``, h = 10 mm, flat base:
        # sag 0.01 m (== the ``{4: 1e6}`` sag, BIT-identical) against the true
        # 1.0e-4 m -- 100x -- with dz/dh 4.0 vs the true 0.05 (80x).  The same
        # ``{5: ...}`` fed through ``apply_real_lens`` returned a field
        # bit-identical to the ``{4: ...}`` lens.  ``Surface`` and the JAX
        # prescription path already reject it via the SAME shared checker; this
        # is the wave-optics path, which never builds a ``Surface``.
        # Import is function-local: ``lumenairy.raytrace.__init__`` pulls in
        # ``raytrace.surface``, which imports THIS module, so a module-level
        # ``from ..raytrace._conic_core import ...`` cycles at import time.
        # (Same deferred-import pattern as ``.._validation`` in _lens_thin.py.)
        from ..raytrace._conic_core import check_even_aspheric_powers
        check_even_aspheric_powers(aspheric_coeffs.keys(),
                                   fn_label='surface_sag_general')
        # 3.2.14: fused single-pass numba kernel when available.
        # Skips the per-term temporary array allocation that the
        # legacy NumPy fallback required (5 aspheric coeffs at N=4096
        # is ~640 MB of transient memory in that path).  CuPy stays
        # on the legacy path because numba targets host arrays.
        _sag_kernel = (_get_aspheric_sag_accum_numba()
                       if xp is np and _NUMBA_AVAILABLE else None)
        if (_sag_kernel is not None
                and h_sq.dtype == np.float64
                and sag.dtype == np.float64):
            powers_arr = np.fromiter(
                (int(p) for p in aspheric_coeffs.keys()), dtype=np.int32)
            coeffs_arr = np.fromiter(
                (float(c) for c in aspheric_coeffs.values()),
                dtype=np.float64)
            # The kernel accumulates through ``sag.ravel()``, which is a VIEW
            # only when ``sag`` is C-contiguous; for an F-ordered or transposed
            # array ``ravel()`` copies, the kernel adds the whole polynomial
            # into that copy and the copy is discarded -- the aspheric term
            # vanished silently and completely (measured 9.41e-6 m = 100 % of
            # the term).  ``sag`` inherits its memory order from ``h_sq``, so
            # any caller passing a non-C-contiguous ``h_sq`` hit it; every
            # in-repo caller happens not to, which is why it survived.  Make
            # the buffer contiguous, accumulate, and copy back when it was not
            # the same object.
            _sag_c = np.ascontiguousarray(sag)
            _sag_kernel(
                np.ascontiguousarray(h_sq), _sag_c, powers_arr, coeffs_arr)
            if _sag_c is not sag:
                sag[...] = _sag_c
            del _sag_c
        else:
            for power, coeff in aspheric_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

    return sag


# Keep the private alias so internal callers (apply_real_lens) can use either
# name without changing semantics.
_surface_sag_general = surface_sag_general


def surface_sag_biconic(
    X: np.ndarray,
    Y: np.ndarray,
    R_x: float,
    R_y: Optional[float] = None,
    conic_x: float = 0.0,
    conic_y: Optional[float] = None,
    aspheric_coeffs: Optional[Dict[int, float]] = None,
    aspheric_coeffs_y: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Biconic / cylindrical / toroidal surface sag.

    Generalises :func:`surface_sag_general` to surfaces that have
    different curvatures and conics along the x and y axes.  Covers:

    * **Biconic** (SEPARABLE per-axis form): independent R_x, R_y, K_x,
      K_y, where each axis contributes its own conic sag INDEPENDENTLY:

          z(x,y) = C_x*x² / (1 + sqrt(1 - (1+K_x)*C_x²*x²))
                 + C_y*y² / (1 + sqrt(1 - (1+K_y)*C_y²*y²))

      where C_x = 1/R_x, C_y = 1/R_y.

      RT-2 (AUDIT_RAYTRACE_CORE) -- deviation from Zemax "Biconic": this
      is the ``z = z_x(x) + z_y(y)`` SEPARABLE sum, NOT Zemax's biconic,
      which shares a SINGLE square root across both axes:

          z = (C_x*x² + C_y*y²)
              / (1 + sqrt(1 - (1+K_x)*C_x²*x² - (1+K_y)*C_y²*y²))

      The two agree paraxially (and exactly on either axis, y=0 or x=0)
      but diverge in the fourth-order cross-term, so an imported Zemax
      BICONIC surface is approximated at large aperture / off both axes.
      (``_surface_sag_derivatives_xy`` is exactly consistent with the
      SEPARABLE form used here, so the sag and its normals still agree
      internally.)
    * **Cylindrical**: pass ``R_y = inf`` (focusing in x only) or
      ``R_x = inf`` (focusing in y only).
    * **Toroidal** (approx., Zemax "Toroidal"): pass R_x (rotation-axis
      radius) and R_y (cross-section radius).
    * **Rotationally symmetric**: if ``R_y is None`` the function
      reduces to :func:`surface_sag_general` via h² = x² + y².

    Aspheric coefficients may be given per-axis for fully general
    anamorphic surfaces; ``aspheric_coeffs`` (x-axis) and
    ``aspheric_coeffs_y`` (y-axis) are separate dicts of
    ``{power: coeff}`` contributing ``coeff * h² ** (power // 2)``
    along each axis.

    Parameters
    ----------
    X, Y : ndarray
        Surface-local coordinates [m] (after any decenter/tilt).
        ``X`` and ``Y`` must have the same shape; meshgrid indexing is
        up to the caller.
    R_x : float
        Radius of curvature along x-axis [m] (``inf`` = flat in x).
    R_y : float, optional
        Radius of curvature along y-axis [m].  If ``None``, the surface
        is treated as rotationally symmetric with R = R_x (legacy).
    conic_x : float, default 0
        Conic constant along x.
    conic_y : float, optional
        Conic constant along y.  Defaults to ``conic_x`` if not given.
    aspheric_coeffs : dict or None
        Even-aspheric coefficients along x, ``{power: coeff}``.
    aspheric_coeffs_y : dict or None
        Even-aspheric coefficients along y.  If ``None`` and
        ``aspheric_coeffs`` is given, the x coefficients are reused for
        y (isotropic asphere).

    Returns
    -------
    sag : ndarray
        Signed surface sag, same shape as ``X`` / ``Y``.
    """
    # Detect array backend and use cupy ops when X/Y are device arrays.
    xp = cp if _is_cupy_array(X) else np
    X = xp.asarray(X)
    Y = xp.asarray(Y)

    if R_y is None:
        # Reduce to the rotationally-symmetric formula for backward
        # compatibility.  (Its own R-8 guard covers ``aspheric_coeffs``;
        # ``aspheric_coeffs_y`` is unread on this branch, as documented.)
        h_sq = X ** 2 + Y ** 2
        return surface_sag_general(h_sq, R_x, conic_x, aspheric_coeffs)

    if conic_y is None:
        conic_y = conic_x

    # Reject ODD powers on BOTH per-axis
    # coefficient dicts before ``_axis_sag`` floors them via
    # ``h_sq ** (power // 2)``.  Measured pre-guard at ``{3: 1e4}``, h = 10 mm,
    # flat base: sag 1.0 m -- BIT-identical to the ``{2: 1e4}`` sag -- against
    # the true 0.01 m, i.e. 100x, on the x AND the y coefficient set.  Same
    # shared checker (and message) as ``surface_sag_general`` /
    # ``raytrace.conic_sag``; see that site for the function-local-import note.
    from ..raytrace._conic_core import check_even_aspheric_powers
    if aspheric_coeffs:
        check_even_aspheric_powers(aspheric_coeffs.keys(),
                                   fn_label='surface_sag_biconic')
    if aspheric_coeffs_y:
        check_even_aspheric_powers(aspheric_coeffs_y.keys(),
                                   fn_label='surface_sag_biconic (aspheric_coeffs_y)')

    def _axis_sag(h_sq, R, K, asph):
        s = xp.zeros_like(h_sq)
        if R is not None and not np.isinf(R):
            norm = (1 + K) * h_sq / R ** 2
            valid = norm < 0.9999
            denom_arg = xp.where(valid, 1 - norm, 0.01)
            # Outside the conic domain
            # (norm >= 0.9999) the surface is not defined.  Return NaN
            # (not a silent 0.0 flat ring) so callers detect 'no real
            # surface', matching surface_sag_general.
            s = xp.where(
                valid,
                h_sq / (R * (1 + xp.sqrt(denom_arg))),
                xp.nan,
            )
        if asph:
            for power, coeff in asph.items():
                s = s + coeff * h_sq ** (power // 2)
        return s

    sag_x = _axis_sag(X ** 2, R_x, conic_x, aspheric_coeffs)
    sag_y = _axis_sag(Y ** 2, R_y, conic_y,
                      aspheric_coeffs_y if aspheric_coeffs_y is not None
                      else aspheric_coeffs)
    return sag_x + sag_y


# ---------------------------------------------------------------------------
# Grid-vs-aperture safety check
#
# The census, the public check, the recommendation and the warning live in
# ``_lens_kernels.py`` -- a LEAF that imports nothing from ``lumenairy``, so a
# family module can reach it without closing a module-level import cycle with
# this facade.  They are re-exported here (the import at the top of this file),
# so every ``lenses.check_grid_vs_apertures`` and
# ``from .lenses import _warn_if_aperture_exceeds_grid`` still resolves.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared Chebyshev helpers used by the Maslov and asymptotic propagators.
#
# The three Chebyshev Vandermonde helpers live in
# ``lumenairy/_math/chebyshev.py`` (see
# docs/history/lumenairy.elements.lenses.md).  The underscore-prefixed
# aliases below preserve every internal call site in this module --
# and every external import of the form
# ``from lumenairy.elements.lenses import _chebyshev_vandermonde`` --
# without forcing those callers to update their import paths.
# ---------------------------------------------------------------------------

from .._math.chebyshev import (  # noqa: F401 -- back-compat alias re-export (v5.2)
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,  # noqa: F401
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,  # noqa: F401
    chebyshev_vandermonde as _chebyshev_vandermonde,  # noqa: F401
)


def _multi_indices_total_degree(n_vars: int, max_order: int):
    """Enumerate multi-indices k with sum(k) <= max_order, as list of tuples."""
    out = []
    def recurse(prefix, remaining, depth):
        if depth == n_vars:
            out.append(tuple(prefix))
            return
        for k in range(remaining + 1):
            recurse(prefix + [k], remaining - k, depth + 1)
    recurse([], max_order, 0)
    return out


def _evaluate_polynomial_4d(coeffs: np.ndarray,
                              multi_indices,
                              u1: np.ndarray, u2: np.ndarray,
                              u3: np.ndarray, u4: np.ndarray,
                              max_order: int) -> np.ndarray:
    """
    Evaluate a 4-variable Chebyshev tensor-product polynomial in
    total-degree subspace at arbitrary (u1, u2, u3, u4) samples.

    Parameters
    ----------
    coeffs : ndarray, shape (M,)
        Polynomial coefficients in the same order as ``multi_indices``.
    multi_indices : list of 4-tuples
        Multi-indices (k1, k2, k3, k4) enumerating the basis.
    u1, u2, u3, u4 : ndarrays with identical shape
        Evaluation points in [-1, 1]^4.
    max_order : int
        Maximum individual index (== total-degree cap in this call).

    Returns
    -------
    value : ndarray with the broadcast shape of (u1, ..., u4)
    """
    # 3.5.6: vectorised over basis terms (no Python loop).  T_i has
    # shape (max_k+1, *u_shape); T_i[K_i] has shape (M, *u_shape);
    # product is (M, *u_shape); contracting over basis axis with
    # coeffs gives the result.  ~3x faster than the previous loop on
    # M=70 basis terms x 1024-pt u arrays.
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    basis = T1[K1] * T2[K2] * T3[K3] * T4[K4]
    return np.tensordot(np.asarray(coeffs, dtype=np.float64),
                         basis, axes=([0], [0]))


def _evaluate_polynomial_4d_and_grad34(coeffs: np.ndarray,
                                         multi_indices,
                                         u1, u2, u3, u4,
                                         max_order: int
                                         ) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """
    Evaluate the 4-variable polynomial at (u1, u2, u3, u4) and also its
    partial derivatives d/du3 and d/du4 (used for the Jacobian w.r.t.
    the v2 coordinates).

    Returns
    -------
    f, df_du3, df_du4
    """
    # 4.14.0 (Tier-2 perf, audit group): vectorised over basis terms,
    # mirroring the sibling ``_evaluate_polynomial_4d``.  The previous
    # version walked the M-term basis in a Python loop computing a
    # fused multiply-add over the evaluation grid; that loop overhead
    # is the dominant cost when this helper is hot inside the
    # asymptotic propagator's Newton iterations.
    #
    # The (M, *u_shape) basis tensors trade memory for three
    # tensordots; measured speedup vs the scalar loop at M=70 is
    # ~4.5x at an 8x8 grid, ~3.5x at 16x16, ~1.7x at 32x32 (and
    # marginally slower above ~48x48 where the (M, *u_shape)
    # intermediates spill L2 cache).  Typical
    # ``apply_real_lens_maslov`` callers run with n_v2=32, so the
    # 32x32 regime is the operational point; the cache-bound large-
    # grid regime is not exercised in production.
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    dT3 = _chebyshev_derivative_vandermonde(u3, max_order)
    dT4 = _chebyshev_derivative_vandermonde(u4, max_order)
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    # T12 has shape (M, *u_shape); reused for all three outputs.
    T12 = T1[K1] * T2[K2]
    c_arr = np.asarray(coeffs, dtype=np.float64)
    basis_f = T12 * T3[K3] * T4[K4]
    basis_d3 = T12 * dT3[K3] * T4[K4]
    basis_d4 = T12 * T3[K3] * dT4[K4]
    f = np.tensordot(c_arr, basis_f, axes=([0], [0]))
    df3 = np.tensordot(c_arr, basis_d3, axes=([0], [0]))
    df4 = np.tensordot(c_arr, basis_d4, axes=([0], [0]))
    return f, df3, df4


# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------

def _fit_normaliser(v: np.ndarray, pad: float = 0.05):
    """Return (center, half_range) such that (v - center)/half_range sits
    in [-(1-pad), (1-pad)].

    pad leaves a narrow margin so that mild extrapolation by the
    propagator is still bounded.
    """
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    center = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin) * (1.0 + pad)
    if half == 0.0:
        half = 1.0
    return center, half


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Maslov propagator -- moved to lumenairy.elements.lenses_maslov in v3.5.5
# to reduce lenses.py bloat.  Re-exported here for backwards-compatible
# `from lumenairy.elements.lenses import apply_real_lens_maslov` imports.
# ---------------------------------------------------------------------------

# Every name below is spelled ``X as X``.  That is not redundancy: it is the
# PEP 484 / mypy marker for a DELIBERATE re-export.  Without it
# ``mypy --strict`` (which sets ``no_implicit_reexport``) reports
# "Module ... does not explicitly export attribute X" for each of the 31
# names the package root re-imports from here, and this module -- a
# compatibility shell whose whole job is to keep the legacy import paths
# working -- would be the reason ``lumenairy/__init__.py`` cannot join the
# strict whitelist.  The alternative, an ``__all__`` on this 1 100-line
# module, would also change ``import *`` behaviour; this does not.
#
# ---------------------------------------------------------------------------
# JAX-traceable real-lens propagators moved to
# lumenairy.elements._lens_jax in v3.5.5.  Re-exported here so existing
# `from lumenairy.elements.lenses import apply_real_lens_traced_jax` /
# `apply_real_lens_maslov_jax` imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_jax import (  # noqa: E402
    apply_real_lens_maslov_jax as apply_real_lens_maslov_jax,
    apply_real_lens_traced_jax as apply_real_lens_traced_jax,
)

# ---------------------------------------------------------------------------
# Analytic split-step real-lens propagator moved to
# lumenairy.elements._lens_real in v3.5.5.  Re-exported here so
# existing `from lumenairy.elements.lenses import apply_real_lens`
# imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_real import (  # noqa: E402
    PreparedAnalyticLens as PreparedAnalyticLens,
    apply_real_lens as apply_real_lens,
    clear_pointwise_cos_grid_cache as clear_pointwise_cos_grid_cache,
    get_lens_sag_dtype as get_lens_sag_dtype,
    get_pointwise_cos_grid_cache_budget as get_pointwise_cos_grid_cache_budget,
    lens_sag_float32_opd_error as lens_sag_float32_opd_error,
    prepare_real_lens as prepare_real_lens,
    set_lens_sag_dtype as set_lens_sag_dtype,
    set_pointwise_cos_grid_cache_budget as set_pointwise_cos_grid_cache_budget,
)

# ---------------------------------------------------------------------------
# Thin-lens / single-element phase screens were moved to
# lumenairy.elements._lens_thin in v3.5.5.  Re-exported here so existing
# `from lumenairy.elements.lenses import apply_thin_lens` etc. continue
# to work.
# ---------------------------------------------------------------------------
from ._lens_thin import (  # noqa: E402
    apply_aspheric_lens as apply_aspheric_lens,
    apply_axicon as apply_axicon,
    apply_cylindrical_lens as apply_cylindrical_lens,
    apply_grin_lens as apply_grin_lens,
    apply_spherical_lens as apply_spherical_lens,
    apply_thin_lens as apply_thin_lens,
)

# ---------------------------------------------------------------------------
# Per-pixel ray-traced apply_real_lens variant moved to
# lumenairy.elements._lens_traced in v3.5.5.  Re-exported here so
# existing
#   from lumenairy.elements.lenses import apply_real_lens_traced
#   from lumenairy.elements.lenses import close_worker_pool
# imports continue to work.
# ---------------------------------------------------------------------------
from ._lens_traced import (  # noqa: E402
    PreparedTracedLens as PreparedTracedLens,
    TiltedCarrier as TiltedCarrier,
    apply_real_lens_traced as apply_real_lens_traced,
    apply_real_lens_traced_multi as apply_real_lens_traced_multi,
    apply_real_lens_traced_segmented as apply_real_lens_traced_segmented,
    close_worker_pool as close_worker_pool,
    get_lens_parallel_amp as get_lens_parallel_amp,
    prepare_real_lens_traced as prepare_real_lens_traced,
    set_lens_parallel_amp as set_lens_parallel_amp,
)
from ._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch as apply_real_lens_traced_multibranch,
)
from ._lens_traced_uniform import (  # noqa: E402
    apply_real_lens_traced_uniform as apply_real_lens_traced_uniform,
)
from .lenses_gbd import apply_real_lens_gbd as apply_real_lens_gbd  # noqa: E402
from .lenses_maslov import (  # noqa: E402
    apply_real_lens_maslov as apply_real_lens_maslov,
)
