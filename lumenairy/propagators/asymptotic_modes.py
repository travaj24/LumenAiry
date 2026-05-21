"""Hermite-Gauss / Laguerre-Gauss modal primitives for the
phase-space asymptotic propagator.

v5.1.0 file-split (Agent D, ROADMAP item):  extracted from
``lumenairy.propagators.asymptotic`` with NO public-API or physics
change.  Every name re-exports through the original module via
``from .asymptotic_modes import *`` so consumers that import
``from lumenairy.propagators.asymptotic import lg_polynomial``
continue to work unmodified.

Contents
--------

* :func:`lg_polynomial` / :func:`hg_polynomial` -- Cartesian
  polynomial coefficients of the LG and HG modes.
* :func:`evaluate_lg_mode` / :func:`evaluate_hg_mode` -- evaluate
  basis functions on a (x, y) grid.
* :func:`decompose_lg` / :func:`decompose_hg` -- project an
  arbitrary field onto the LG/HG basis.
* :func:`gaussian_moment_2d` / :func:`gaussian_moment_table_2d` --
  closed-form Wick-moment evaluation.
* :func:`lg_seidel_label` -- map LG ``(p, ell)`` to its Seidel
  aberration name.
* Module-level mode-stack caches
  (``_LG_MODE_STACK_CACHE`` / ``_HG_MODE_STACK_CACHE``) and their
  thread-safety locks.
"""

from __future__ import annotations

import functools
import math
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    'lg_polynomial',
    'clear_lg_polynomial_cache',
    'clear_lg_mode_stack_cache',
    'hg_polynomial',
    'evaluate_lg_mode',
    'evaluate_hg_mode',
    'decompose_lg',
    'decompose_hg',
    'lg_seidel_label',
    'gaussian_moment_2d',
    'gaussian_moment_table_2d',
]


# ===========================================================================
# Section 1 -- Laguerre-Gaussian and Hermite-Gaussian basis polynomials
# ===========================================================================

@functools.lru_cache(maxsize=256)
def _lg_polynomial_items(p: int, ell: int, w: float
                          ) -> Tuple[Tuple[Tuple[int, int], complex], ...]:
    """Immutable, hashable, cached representation of the LG polynomial.

    Returns a tuple of ``((i, j), c)`` pairs sorted by ``(i, j)``.  This
    is the canonical cache target -- the public :func:`lg_polynomial`
    wraps it in a fresh ``dict`` so callers can safely mutate the
    returned mapping without poisoning the cache.

    4.12.0 (perf #5): hoists the recursive build out of every
    per-pixel call in :func:`propagate_modal_asymptotic`.  Cache key is
    ``(p, ell, w)``; same modes evaluated repeatedly across a
    ``MultiWavelengthMerit`` sweep (one ``(p, ell)`` set, multiple
    pixel grids and beam waists) re-use the cached coefficients.
    """
    if p < 0:
        raise ValueError(f"LG radial index p must be >= 0, got {p}")
    if w <= 0:
        raise ValueError(f"LG waist w must be > 0, got {w}")
    abs_ell = abs(ell)
    s_sign = 1 if ell >= 0 else -1

    f = math.sqrt(2.0) / w
    # Standard waist-w normalisation:  N = sqrt(2 p! / (pi (p+|l|)! w^2)).
    # Verified by:  integral |LG_00|^2 dA = N^2 * pi w^2 / 2 = 1.
    N = math.sqrt(
        2.0 * math.factorial(p)
        / (math.pi * math.factorial(p + abs_ell) * (w * w))
    )

    coeffs: Dict[Tuple[int, int], complex] = {}
    for m in range(abs_ell + 1):
        binom_lm = math.comb(abs_ell, m)
        is_m = (1j * s_sign) ** m
        for k in range(p + 1):
            lag_coef = ((-1) ** k / math.factorial(k)
                        * math.comb(p + abs_ell, p - k))
            for j in range(k + 1):
                binom_kj = math.comb(k, j)
                i_x = (abs_ell - m) + 2 * j
                i_y = m + 2 * (k - j)
                c = (
                    N
                    * (f ** abs_ell)
                    * binom_lm
                    * is_m
                    * lag_coef
                    * (f ** (2 * k))
                    * binom_kj
                )
                key = (i_x, i_y)
                coeffs[key] = coeffs.get(key, 0.0 + 0.0j) + c
    # Return as an immutable tuple of items so the cache target is
    # hashable and callers cannot mutate the cached object.  Preserve
    # *insertion order* (not sorted-by-key) so the public-facing dict
    # iterates in the same order as the pre-v4.12.0 unhoisted build --
    # downstream poly algebra (``_polynomial_substitute_linear_2d``,
    # ``_polynomial_under_affine_shift``, ``_multiply_polys_2d``) walks
    # the dict in insertion order and accumulates sums whose
    # floating-point rounding order depends on it.  Sorting would
    # change a small handful of low-order digits and trip a few
    # tight-tolerance JAX-vs-NumPy comparisons in the validation
    # suite (tested against pre-v4.12.0 outputs at rel < 1e-3).
    return tuple(coeffs.items())


def lg_polynomial(p: int, ell: int, w: float) -> Dict[Tuple[int, int], complex]:
    """Cartesian polynomial coefficients of a Laguerre-Gaussian mode.

    The LG_{p,l} mode with waist ``w`` centred at the origin can be written
    as a polynomial in (x, y) times a shared Gaussian envelope::

        LG_{p,l}(x, y) = (sum_{i,j} c_{ij} x^i y^j) * exp(-(x^2 + y^2)/w^2)

    with the normalisation convention

        N_{p,l} = sqrt(2 * p! / (pi * (p + |l|)!)) / w

    so that the modes are orthonormal under the L^2 inner product
    ``<f, g> = integral f^* g  dx dy`` (no extra envelope factor).
    Pre-4.9 docstrings omitted the ``/ w`` factor; the code has
    always carried it correctly (see the implementation a few lines
    below).

    Parameters
    ----------
    p : int
        Radial index, p >= 0.
    ell : int
        Azimuthal index, any integer.  The angular dependence is
        ``exp(i*ell*phi)`` so positive ``ell`` rotates one way and
        negative ``ell`` the other.
    w : float
        Beam waist [m].

    Returns
    -------
    dict
        ``{(i, j): complex}`` mapping Cartesian monomial exponents to
        polynomial coefficients.  Total polynomial degree is
        ``|ell| + 2 p``.  A *fresh* dict is returned each call, safe
        to mutate.

    Notes
    -----
    The expansion is exact and finite (no truncation):  it follows from
    standard identities

        L_p^{|l|}(x) = sum_{k=0}^p (-1)^k / k! * binom(p + |l|, p - k) * x^k
        (x + i*s*y)^{|l|} = sum_{m=0}^{|l|} binom(|l|, m) (i*s)^m x^{|l|-m} y^m
        (x^2 + y^2)^k = sum_{j=0}^k binom(k, j) x^{2j} y^{2(k-j)}

    where ``s = sign(ell)``.  The LG mode is
    ``N * (sqrt(2)/w)^{|l|} * (x + i*s*y)^{|l|} * L_p^{|l|}(2 r^2/w^2)
    * exp(-r^2/w^2)``.

    Performance
    -----------
    4.12.0 (perf #5):  the recursive build is cached via
    :func:`functools.lru_cache` on the immutable inner helper
    :func:`_lg_polynomial_items` (``maxsize=256``).  This eliminates a
    per-pixel rebuild inside :func:`propagate_modal_asymptotic` and
    speeds up multi-mode aberration tensors / multi-wavelength
    chromatic merits by 3-10x on typical workloads.  Use
    :func:`clear_lg_polynomial_cache` to flush the cache if you need
    deterministic memory behaviour (e.g. parameter sweeps over many
    ``w`` values).
    """
    return dict(_lg_polynomial_items(p, ell, float(w)))


def clear_lg_polynomial_cache() -> None:
    """Clear the :func:`lg_polynomial` ``lru_cache``.

    Useful when sweeping over many ``w`` values (cache key includes
    ``w``) and the cache would otherwise grow until it evicts
    least-recently-used entries.  Safe to call at any time; subsequent
    :func:`lg_polynomial` calls will rebuild and re-cache.
    """
    _lg_polynomial_items.cache_clear()


# 4.14.0 (perf 1B): LG mode-stack cache for ``decompose_lg``.
#
# ``decompose_lg`` rebuilds N_modes complete (Ny, Nx) LG mode arrays from
# scratch and discards the shared ``exp(-(rx^2 + ry^2)/w^2)`` envelope
# between modes.  At p_max=3, ell_max=3 that is 28 N x N rebuilds for
# every call -- when the same ``(p_max, ell_max, w, cx, cy, grid)`` is
# evaluated repeatedly (e.g. across a wavelength sweep or aberration
# tensor build), this is wasteful.
#
# The cache below stores the **stack of conjugated modes**
# ``conj(LG_{p,l}(X, Y))`` as a single ``(N_modes, Ny, Nx)`` complex array
# keyed on the shape and basis parameters.  ``decompose_lg`` then collapses
# all 28 overlaps to a single ``np.einsum('mij,ij->m', modes_conj, field)``
# reduction.  Cleared explicitly via :func:`clear_lg_mode_stack_cache`.
_LG_MODE_STACK_CACHE: 'OrderedDict[Any, Tuple[Tuple[Tuple[int, int], ...], np.ndarray]]' = OrderedDict()
_LG_MODE_STACK_CACHE_MAX = 32
# v4.14.1 (P2-1): thread-safety lock for ``_LG_MODE_STACK_CACHE``.
# Concurrent ``design_optimize`` threads can race on the
# ``OrderedDict.get`` / ``move_to_end`` / ``popitem(last=False)``
# read-modify-write sequence in :func:`_lg_mode_conj_stack`.  Follows
# the ``_ASM_CACHE_LOCK`` precedent in :mod:`propagators.propagation`.
_LG_MODE_STACK_LOCK = threading.Lock()

_HG_MODE_STACK_CACHE: 'OrderedDict[Any, Tuple[Tuple[Tuple[int, int], ...], np.ndarray]]' = OrderedDict()
_HG_MODE_STACK_CACHE_MAX = 32
# v4.14.1 (P2-1): thread-safety lock for ``_HG_MODE_STACK_CACHE``; see
# ``_LG_MODE_STACK_LOCK``.
_HG_MODE_STACK_LOCK = threading.Lock()


def clear_lg_mode_stack_cache() -> None:
    """Clear the LG (and HG) mode-stack caches used by
    :func:`decompose_lg` / :func:`decompose_hg`.

    Each entry is a stack of ``(N_modes, Ny, Nx)`` complex arrays keyed
    on ``(p_max, ell_max, Ny, Nx, w, cx, cy, dx, dy, dtype_str)`` (LG) or
    ``(m_max, n_max, Ny, Nx, wx, wy, cx, cy, dx, dy, dtype_str)`` (HG).
    The ``dx, dy`` entries were added in v4.14.1 (P0-NEW-1) -- pre-v4.14.1
    keys captured only the grid shape, so two calls at the same N but
    different physical pitch silently collided on the cached entry.
    Safe to call at any time; subsequent decompose calls rebuild and
    re-cache.
    """
    with _LG_MODE_STACK_LOCK:
        _LG_MODE_STACK_CACHE.clear()
    with _HG_MODE_STACK_LOCK:
        _HG_MODE_STACK_CACHE.clear()


# v4.16.0 (ROADMAP #15): register the LG/HG mode-stack and LG
# polynomial clearers with the central registry at module-import
# time.  ``clear_asm_caches`` now walks the registry rather than
# enumerating clear calls by hand.
#
# Each registered entry is a *late-binding* lambda that re-resolves
# the clear-function from the module's current namespace at call
# time.  This preserves the pre-v4.16 ``mock.patch.object`` semantic:
# tests that monkey-patch ``analysis.core.clear_zernike_basis_cache``
# still observe their counter increment when ``clear_asm_caches``
# walks the registry.  The cost is one attribute lookup per cache
# per drain -- negligible vs. the actual cache-clear work.
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'lg_mode_stack',
        lambda: getattr(_this_mod, 'clear_lg_mode_stack_cache')(),
    )
    _register_cache_clearer(
        'lg_polynomial_items',
        lambda: getattr(_this_mod, 'clear_lg_polynomial_cache')(),
    )
except ImportError:
    # Defensive: if the registry module is unavailable (shouldn't be,
    # but a partial install or a reload sequence could expose it),
    # fall back to the v4.15 lazy-import fan-out in clear_asm_caches.
    pass


def _lg_mode_conj_stack(X: np.ndarray, Y: np.ndarray, w: float,
                         p_max: int, ell_max: int,
                         cx: float, cy: float,
                         dx: float, dy: float,
                         ) -> Tuple[Tuple[Tuple[int, int], ...], np.ndarray]:
    """Build / fetch the conjugated LG mode stack used by
    :func:`decompose_lg`.

    Returns ``(keys, stack)`` where ``keys`` is an ordered tuple of
    ``(p, ell)`` index pairs and ``stack`` is a complex
    ``(N_modes, Ny, Nx)`` array whose first axis is in the same order
    as ``keys``.  Each slice is ``np.conj(LG_{p, ell}(X, Y; w, cx, cy))``.

    Cache key includes the grid shape, the physical pitch ``(dx, dy)``,
    all basis parameters, and the dtype of the (X, Y) sample arrays so
    cached entries are only reused when the result would be bit-equal.

    v4.14.1 (P0-NEW-1):  ``dx, dy`` are included in the cache key.
    Pre-v4.14.1 keys captured only ``(Ny, Nx)``, so two calls with the
    same shape but different physical pitch (e.g. ``dx=1e-6`` then
    ``dx=2e-6`` at N=256) collided on the cache and the second call
    silently received the first call's modes evaluated against the
    second call's field.  Thread-safe via ``_LG_MODE_STACK_LOCK``.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Ny = int(X.shape[0])
    Nx = int(X.shape[1])
    dtype_str = str(np.result_type(X.dtype, Y.dtype, np.float64))
    cache_key = (
        int(p_max), int(ell_max), Ny, Nx,
        float(w), float(cx), float(cy),
        float(dx), float(dy), dtype_str,
    )
    with _LG_MODE_STACK_LOCK:
        cached = _LG_MODE_STACK_CACHE.get(cache_key)
        if cached is not None:
            # LRU touch: move to most-recent end.
            _LG_MODE_STACK_CACHE.move_to_end(cache_key)
            return cached

    rx = X - cx
    ry = Y - cy
    envelope = np.exp(-(rx * rx + ry * ry) / (w * w))
    keys: List[Tuple[int, int]] = []
    for p in range(p_max + 1):
        for ell in range(-ell_max, ell_max + 1):
            keys.append((p, ell))
    n_modes = len(keys)
    stack = np.empty((n_modes, Ny, Nx), dtype=np.complex128)
    for idx, (p, ell) in enumerate(keys):
        poly = lg_polynomial(p, ell, w)
        polynomial = _evaluate_poly2d(poly, rx, ry)
        # Conjugate once here so the einsum reduction is direct.
        stack[idx] = np.conj(polynomial * envelope)
    keys_t = tuple(keys)
    with _LG_MODE_STACK_LOCK:
        _LG_MODE_STACK_CACHE[cache_key] = (keys_t, stack)
        while len(_LG_MODE_STACK_CACHE) > _LG_MODE_STACK_CACHE_MAX:
            _LG_MODE_STACK_CACHE.popitem(last=False)
    return keys_t, stack


def _hg_mode_conj_stack(X: np.ndarray, Y: np.ndarray,
                         wx: float, wy: float,
                         m_max: int, n_max: int,
                         cx: float, cy: float,
                         dx: float, dy: float,
                         ) -> Tuple[Tuple[Tuple[int, int], ...], np.ndarray]:
    """Build / fetch the conjugated HG mode stack used by
    :func:`decompose_hg`.  See :func:`_lg_mode_conj_stack`.

    v4.14.1 (P0-NEW-1):  ``dx, dy`` are included in the cache key for
    the same reason as the LG variant -- same shape at different
    physical pitch must not collide.  Thread-safe via
    ``_HG_MODE_STACK_LOCK``.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Ny = int(X.shape[0])
    Nx = int(X.shape[1])
    dtype_str = str(np.result_type(X.dtype, Y.dtype, np.float64))
    cache_key = (
        int(m_max), int(n_max), Ny, Nx,
        float(wx), float(wy), float(cx), float(cy),
        float(dx), float(dy), dtype_str,
    )
    with _HG_MODE_STACK_LOCK:
        cached = _HG_MODE_STACK_CACHE.get(cache_key)
        if cached is not None:
            _HG_MODE_STACK_CACHE.move_to_end(cache_key)
            return cached

    rx = X - cx
    ry = Y - cy
    envelope = np.exp(-(rx * rx) / (wx * wx) - (ry * ry) / (wy * wy))
    keys: List[Tuple[int, int]] = []
    for mi in range(m_max + 1):
        for nj in range(n_max + 1):
            keys.append((mi, nj))
    n_modes_total = len(keys)
    stack = np.empty((n_modes_total, Ny, Nx), dtype=np.complex128)
    for idx, (mi, nj) in enumerate(keys):
        poly = hg_polynomial(mi, nj, wx, wy)
        polynomial = _evaluate_poly2d(poly, rx, ry)
        stack[idx] = np.conj(polynomial * envelope)
    keys_t = tuple(keys)
    with _HG_MODE_STACK_LOCK:
        _HG_MODE_STACK_CACHE[cache_key] = (keys_t, stack)
        while len(_HG_MODE_STACK_CACHE) > _HG_MODE_STACK_CACHE_MAX:
            _HG_MODE_STACK_CACHE.popitem(last=False)
    return keys_t, stack


def hg_polynomial(m: int, n: int, wx: float,
                  wy: Optional[float] = None
                  ) -> Dict[Tuple[int, int], complex]:
    """Cartesian polynomial coefficients of a Hermite-Gaussian mode.

    The HG_{m,n} mode with axis waists ``wx, wy`` centred at the origin is

        HG_{m,n}(x, y) = phi_m(x; wx) * phi_n(y; wy)

    with the 1-D physicist's-Hermite Gaussian basis function

        phi_k(u; w) = (2/(pi w^2))^{1/4} / sqrt(2^k k!)
                     * H_k(sqrt(2) u / w) * exp(-u^2 / w^2)

    Parameters
    ----------
    m, n : int
        x- and y-mode orders, both >= 0.
    wx : float
        Waist along x [m].
    wy : float, optional
        Waist along y [m].  Defaults to ``wx`` (round Gaussian).

    Returns
    -------
    dict
        ``{(i, j): complex}`` -- Cartesian polynomial coefficients
        such that ``HG_{m,n}(x, y) = (sum c_{ij} x^i y^j) *
        exp(-x^2/wx^2 - y^2/wy^2)``.

    Notes
    -----
    The basis is orthonormal:  ``int phi_m(x) phi_p(x) dx = delta_{mp}``.
    Total polynomial degree is ``m + n``.
    """
    if m < 0 or n < 0:
        raise ValueError(f"HG indices must be >= 0, got ({m}, {n})")
    if wx <= 0:
        raise ValueError(f"HG waist wx must be > 0, got {wx}")
    if wy is None:
        wy = wx
    if wy <= 0:
        raise ValueError(f"HG waist wy must be > 0, got {wy}")

    # 1-D Hermite polynomial coefficients of H_m(sqrt(2)*x/wx).
    # Build as polynomial coefficients in x (real, may have negative entries).
    def hermite_coeffs(k: int, alpha: float) -> Dict[int, float]:
        """Coefficients of H_k(alpha * x) as polynomial in x."""
        # H_0 = 1, H_1 = 2*x, H_{k+1}(z) = 2 z H_k(z) - 2 k H_{k-1}(z),
        # but here we have H_k(alpha x): substitute z = alpha x, then
        # rebuild polynomial in x.
        if k == 0:
            return {0: 1.0}
        if k == 1:
            return {1: 2.0 * alpha}
        prev2: Dict[int, float] = {0: 1.0}
        prev1: Dict[int, float] = {1: 2.0 * alpha}
        cur: Dict[int, float] = {}
        for kk in range(2, k + 1):
            # H_{kk}(alpha x) = 2 (alpha x) H_{kk-1}(alpha x)
            #                  - 2 (kk-1) H_{kk-2}(alpha x)
            cur = {}
            for power, coef in prev1.items():
                # 2 alpha x * coef * x^power = 2 alpha coef * x^(power+1)
                key = power + 1
                cur[key] = cur.get(key, 0.0) + 2.0 * alpha * coef
            for power, coef in prev2.items():
                cur[power] = cur.get(power, 0.0) - 2.0 * (kk - 1) * coef
            prev2, prev1 = prev1, cur
        return cur

    # 1-D normalisation:  N_k = (2/(pi w^2))^{1/4} / sqrt(2^k k!)
    Nx = ((2.0 / (math.pi * wx * wx)) ** 0.25
          / math.sqrt((2 ** m) * math.factorial(m)))
    Ny = ((2.0 / (math.pi * wy * wy)) ** 0.25
          / math.sqrt((2 ** n) * math.factorial(n)))

    Hx = hermite_coeffs(m, math.sqrt(2.0) / wx)   # in x
    Hy = hermite_coeffs(n, math.sqrt(2.0) / wy)   # in y

    coeffs: Dict[Tuple[int, int], complex] = {}
    for i, cx in Hx.items():
        for j, cy in Hy.items():
            key = (i, j)
            coeffs[key] = coeffs.get(key, 0.0 + 0.0j) + Nx * Ny * cx * cy
    return coeffs


def _evaluate_poly2d(coeffs: Dict[Tuple[int, int], complex],
                     x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a 2-D polynomial ``sum c_{ij} x^i y^j``.  Returns complex."""
    out = np.zeros_like(np.broadcast_to(x + 0j, np.broadcast(x, y).shape))
    out = np.array(out, dtype=np.complex128)
    if not coeffs:
        return out
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)
    # Pre-compute powers
    powers_x = [np.ones_like(x, dtype=np.float64)]
    for _ in range(max_i):
        powers_x.append(powers_x[-1] * x)
    powers_y = [np.ones_like(y, dtype=np.float64)]
    for _ in range(max_j):
        powers_y.append(powers_y[-1] * y)
    for (i, j), c in coeffs.items():
        out = out + c * powers_x[i] * powers_y[j]
    return out


def evaluate_lg_mode(p: int, ell: int, w: float,
                     x: np.ndarray, y: np.ndarray,
                     cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Evaluate LG_{p,l} on a 2-D grid.

    Parameters
    ----------
    p, ell : int
        Mode indices.
    w : float
        Waist [m].
    x, y : ndarray
        Cartesian sample points [m].  Shapes must broadcast.
    cx, cy : float, optional
        Mode centre [m].  Defaults to origin.

    Returns
    -------
    ndarray, complex
    """
    poly = lg_polynomial(p, ell, w)
    rx = x - cx
    ry = y - cy
    polynomial = _evaluate_poly2d(poly, rx, ry)
    envelope = np.exp(-(rx * rx + ry * ry) / (w * w))
    return polynomial * envelope


def evaluate_hg_mode(m: int, n: int, wx: float, wy: Optional[float],
                     x: np.ndarray, y: np.ndarray,
                     cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Evaluate HG_{m,n} on a 2-D grid (see ``hg_polynomial``)."""
    if wy is None:
        wy = wx
    poly = hg_polynomial(m, n, wx, wy)
    rx = x - cx
    ry = y - cy
    polynomial = _evaluate_poly2d(poly, rx, ry)
    envelope = np.exp(-(rx * rx) / (wx * wx) - (ry * ry) / (wy * wy))
    return polynomial * envelope


def lg_seidel_label(p: int, ell: int) -> str:
    """Map an LG output-mode index ``(p, ell)`` to its classical
    Seidel/Zernike aberration name.

    Used by tooling and diagnostics that
    want to report aberrations by name rather than by index.
    """
    abs_ell = abs(ell)
    table = {
        (0, 0): 'piston',
        (1, 0): 'defocus',
        (2, 0): 'spherical',
        (3, 0): 'higher_spherical',
        (0, 1): 'tilt',
        (1, 1): 'coma',
        (2, 1): 'higher_coma',
        (0, 2): 'astigmatism',
        (1, 2): 'higher_astigmatism',
        (0, 3): 'trefoil',
    }
    return table.get((p, abs_ell), f'p{p}_l{ell:+d}')


# ===========================================================================
# Section 2 -- Wick moments for 2-D complex-symmetric Gaussians
# ===========================================================================

def gaussian_moment_2d(a: int, b: int,
                       sigma: np.ndarray) -> complex:
    """Closed-form 2-D Gaussian moment ``<eta_x^a eta_y^b>_Sigma``.

    For a 2-D Gaussian with weight ``exp(-eta^T M eta)`` and complex-
    symmetric covariance ``Sigma == 0.5 * inv(M)``, this returns

        <eta_x^a eta_y^b>_Sigma = (1/Z) integral eta_x^a eta_y^b
                                       * exp(-eta^T M eta) d^2 eta

    where ``Z = pi / sqrt(det M)`` is the Gaussian normalisation.
    Vanishes by symmetry for ``a + b`` odd; otherwise evaluates the
    closed-form pair-counting sum.

    Parameters
    ----------
    a, b : int
        Non-negative integer exponents on eta_x, eta_y.
    sigma : ndarray, shape (2, 2)
        Covariance ``0.5 * inv(M)``.  Must be complex-symmetric.

    Returns
    -------
    complex

    Notes
    -----
    Wick contraction reduces the moment to a sum over balanced pair
    assignments.  The closed form has at most ``floor(min(a,b)/2) + 1``
    nonzero terms even though the naive enumeration would have
    ``(a + b - 1)!!`` pairings.
    """
    if a < 0 or b < 0:
        raise ValueError(f"Moment indices must be >= 0, got ({a}, {b})")
    if (a + b) % 2 != 0:
        return 0.0 + 0.0j

    s11 = complex(sigma[0, 0])
    s12 = complex(sigma[0, 1])
    s22 = complex(sigma[1, 1])

    # p_12 has the same parity as a (and as b, since a+b is even).
    p12_min = a % 2
    total = 0.0 + 0.0j
    fa = math.factorial(a)
    fb = math.factorial(b)
    for p12 in range(p12_min, min(a, b) + 1, 2):
        p11 = (a - p12) // 2
        p22 = (b - p12) // 2
        denom = (math.factorial(p11) * math.factorial(p12)
                 * math.factorial(p22) * (2 ** p11) * (2 ** p22))
        coef = (fa * fb) / denom
        total += (coef * (s11 ** p11) * (s12 ** p12) * (s22 ** p22))
    return total


def gaussian_moment_table_2d(M: np.ndarray, max_total_order: int
                              ) -> Dict[Tuple[int, int], complex]:
    """Pre-tabulate Gaussian moments up to a chosen total order.

    Used by the asymptotic propagator and aberration-tensor evaluator to
    amortise moment evaluation across many ``(n, m)`` mode pairs at the
    same output pixel:  the moments depend only on the covariance, not
    on the modal indices.

    Parameters
    ----------
    M : ndarray, shape (2, 2)
        Complex-symmetric quadratic form in ``exp(-eta^T M eta)``.
    max_total_order : int
        Build moments for all ``(a, b)`` with ``a + b <= max_total_order``.

    Returns
    -------
    dict
        ``{(a, b): <eta_x^a eta_y^b>_Sigma}`` for all valid index pairs.
    """
    if max_total_order < 0:
        raise ValueError(f"max_total_order must be >= 0, got {max_total_order}")
    if M.shape != (2, 2):
        raise ValueError(f"M must be 2x2, got shape {M.shape}")
    sigma = 0.5 * np.linalg.inv(M)
    table: Dict[Tuple[int, int], complex] = {}
    for total in range(max_total_order + 1):
        for a in range(total + 1):
            b = total - a
            table[(a, b)] = gaussian_moment_2d(a, b, sigma)
    return table


# ===========================================================================
# Section 7 -- LG/HG decomposition utilities
# ===========================================================================

def decompose_lg(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                 w: float, p_max: int, ell_max: int,
                 cx: float = 0.0, cy: float = 0.0
                 ) -> Dict[Tuple[int, int], complex]:
    """Project a complex field onto the Laguerre-Gaussian basis.

    Computes overlap integrals
        a_{p, ell} = integral conj(LG_{p, ell}(x, y)) * field(x, y) dx dy
    by trapezoidal quadrature on the supplied grid.

    Parameters
    ----------
    field : ndarray, complex, shape (Nx, Ny)
    x, y : ndarray, shape (Nx, Ny)
        Cartesian coordinates [m] (typically from meshgrid).
    w : float
        LG basis waist [m].
    p_max, ell_max : int
        Truncation:  retain p in [0, p_max], ell in [-ell_max, +ell_max].
    cx, cy : float, optional
        Basis centre.

    Returns
    -------
    dict
        ``{(p, ell): a_{p, ell}}``
    """
    # 4.10: accept both 1-D coordinate axes and 2-D meshgrids.  Pre-4.10
    # required 2-D meshgrids (np.diff with axis= calls), so the natural
    # pipeline of ``create_laguerre_gauss`` (returns 1-D x, y) into
    # ``decompose_lg`` raised IndexError on the np.diff call.
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing='xy')
        dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
        dy = float(np.mean(np.diff(y))) if y.size > 1 else 1.0
    else:
        if field.shape != x.shape or field.shape != y.shape:
            raise ValueError("field, x, y must have the same shape")
        X, Y = x, y
        dx = float(np.mean(np.diff(x, axis=1)[:, 0]))
        dy = float(np.mean(np.diff(y, axis=0)[0, :]))
    da = abs(dx * dy)
    # 4.14.0 (perf 1B): build the conjugated LG mode stack once per
    # (p_max, ell_max, shape, w, cx, cy, dtype) signature and collapse
    # all overlaps to a single ``einsum`` reduction.  Pre-v4.14.0 each
    # mode rebuilt the (Ny, Nx) array, recomputing the shared Gaussian
    # envelope ``exp(-(rx^2+ry^2)/w^2)`` 28 times for (p_max=3, ell_max=3).
    keys, modes_conj_stack = _lg_mode_conj_stack(
        X, Y, w, p_max, ell_max, cx, cy, dx, dy,
    )
    # Convert field to complex (cheap if already complex; required by einsum
    # since modes are complex).
    field_c = np.asarray(field)
    if not np.iscomplexobj(field_c):
        field_c = field_c.astype(np.complex128, copy=False)
    overlaps = np.einsum('mij,ij->m', modes_conj_stack, field_c) * da
    out: Dict[Tuple[int, int], complex] = {}
    for k, val in zip(keys, overlaps):
        out[k] = complex(val)
    return out


def decompose_hg(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                 wx: float, wy: Optional[float],
                 m_max: int, n_max: int,
                 cx: float = 0.0, cy: float = 0.0
                 ) -> Dict[Tuple[int, int], complex]:
    """Project a complex field onto the Hermite-Gaussian basis.  See
    ``decompose_lg`` for arguments."""
    if wy is None:
        wy = wx
    # 4.10: accept both 1-D coordinate axes and 2-D meshgrids (see
    # decompose_lg for rationale).
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing='xy')
        dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
        dy = float(np.mean(np.diff(y))) if y.size > 1 else 1.0
    else:
        if field.shape != x.shape or field.shape != y.shape:
            raise ValueError("field, x, y must have the same shape")
        X, Y = x, y
        dx = float(np.mean(np.diff(x, axis=1)[:, 0]))
        dy = float(np.mean(np.diff(y, axis=0)[0, :]))
    da = abs(dx * dy)
    # 4.14.0 (perf 1B): build the conjugated HG mode stack once per
    # signature and collapse all overlaps to one ``einsum``.  See
    # :func:`decompose_lg` for rationale.
    keys, modes_conj_stack = _hg_mode_conj_stack(
        X, Y, wx, wy, m_max, n_max, cx, cy, dx, dy,
    )
    field_c = np.asarray(field)
    if not np.iscomplexobj(field_c):
        field_c = field_c.astype(np.complex128, copy=False)
    overlaps = np.einsum('mij,ij->m', modes_conj_stack, field_c) * da
    out: Dict[Tuple[int, int], complex] = {}
    for k, val in zip(keys, overlaps):
        out[k] = complex(val)
    return out
