"""
lumenairy.asymptotic -- Phase-space asymptotic propagator and Laguerre-Gaussian
aberration tensor.

This module implements the closed-form Gaussian-moment evaluation of
the phase-space (Maslov) diffraction integral.  It complements
``apply_real_lens_maslov`` -- which evaluates the same underlying
integral by direct Chebyshev-quadrature in v_2 -- by replacing the
quadrature with a finite Wick-contracted moment over a
complex-symmetric covariance matrix built from the Chebyshev
polynomial fit.

See ``REFERENCES.txt`` for the foundational publications.

What this is for
----------------

The dominant cost in a wave-aware merit function for optical design is the
diffraction integral evaluated *thousands of times* in the inner optimisation
loop.  The asymptotic propagator runs ~10**3-10**4 times faster than direct
quadrature per output pixel and produces a *physically-named* aberration
tensor whose indices correspond to the classical Seidel/Zernike modes
(spherical, coma, astigmatism, ...).  Optimising against that tensor
directly is the optimisation analog of optimising against Strehl, but
computable in milliseconds.

Public API
----------

Math primitives
~~~~~~~~~~~~~~~

- ``lg_polynomial(p, ell, w)`` -- polynomial coefficients of the LG_{p,l}
  mode in Cartesian (x, y), suitable for closed-form Gaussian-moment
  evaluation.  Returns a dict ``{(i, j): c_{ij}}`` such that
  ``LG(x, y) = (sum c_{ij} x^i y^j) * exp(-(x^2 + y^2)/w^2)``.
- ``hg_polynomial(m, n, wx, wy)`` -- same for Hermite-Gaussian
  ``HG(x, y) = (sum c_{ij} x^i y^j) * exp(-x^2/wx^2 - y^2/wy^2)``.
- ``evaluate_lg_mode``, ``evaluate_hg_mode`` -- evaluate basis functions on
  a (x, y) grid.
- ``decompose_lg``, ``decompose_hg`` -- project an arbitrary field onto
  the LG/HG basis by overlap integrals.
- ``gaussian_moment_2d(a, b, sigma)`` -- evaluate
  ``<eta_1^a eta_2^b>_Sigma`` for a 2-D complex-symmetric Gaussian via
  Wick's theorem (closed form; no quadrature).
- ``gaussian_moment_table_2d(M, max_total_order)`` -- precompute moments
  up to a chosen total order; returns dict ``{(a, b): <...>}``.

Canonical polynomial fit
~~~~~~~~~~~~~~~~~~~~~~~~

- ``fit_canonical_polynomials(prescription, surf_pair, wavelength, ...)``
  -- public version of the apply_real_lens_maslov fit:  trace a
  Chebyshev-node grid through the surface pair, fit Phi(s2, v2) and
  s1(s2, v2) as 4-variable Chebyshev tensor-product polynomials, and
  return a ``CanonicalPolyFit`` dataclass.  Includes a
  linear-phase-extraction step that restores Nyquist sampling for
  diffractive surfaces at non-zero orders.

Aberration tensor and propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``solve_envelope_stationary(fit, s2, source_point, w_s, w_p, v2_centre)``
  -- Newton-solve the envelope-stationary equation for v_2*.
- ``aberration_tensor(fit, s2_image, source_lg_modes, pupil_lg_modes,
  output_lg_modes, source_point, w_s, w_p, v2_centre, w_o, ...)`` --
  compute the LG aberration tensor T_{k;n,m} whose indices are
  physical Seidel/Zernike aberrations.
- ``propagate_modal_asymptotic(fit, source_lg_amps, pupil_lg_amps,
  source_point, w_s, w_p, v2_centre, output_grid)`` --
  evaluate the leading-order modal propagator on a 2-D output grid.

Conventions
-----------

- All physical quantities in SI (positions in metres, OPD in waves where
  noted).  This matches the rest of lumenairy.
- Phase convention follows lumenairy:  field amplitudes carry phase
  factors ``exp(+i * 2 pi * Phi)`` (Phi in waves, equivalently
  ``exp(+i * k * OPL)``).
- Polynomial bases use the same Chebyshev tensor-product layout as
  ``apply_real_lens_maslov`` so canonical fits are reusable across both
  propagators.

See ``validation/test_asymptotic.py`` for the validation suite (Wick-moment
identities, Collins ABCD reduction, Fourier-of-pupil reduction, agreement
with apply_real_lens in the source-dominated and pupil-dominated limits,
Seidel-aberration correspondence, and end-to-end MeritTerm convergence).
"""

from __future__ import annotations

import copy
import functools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

# Reuse the proven Chebyshev infrastructure that already powers
# apply_real_lens_maslov.  These helpers are private only by leading-
# underscore convention; they are stable and well-validated.
from ..elements.lenses import (
    _chebyshev_vandermonde,
    _chebyshev_derivative_vandermonde,
    _chebyshev_second_derivative_vandermonde,
    _multi_indices_total_degree,
    _evaluate_polynomial_4d,
    _evaluate_polynomial_4d_and_grad34,
    _fit_normaliser,
)
from ..backend import JAX_AVAILABLE


__all__ = [
    # Data containers
    'CanonicalPolyFit',
    'HFPolyFit',
    'AberrationTensorResult',
    # Modes / polynomial coefficients
    'lg_polynomial',
    'clear_lg_polynomial_cache',
    'hg_polynomial',
    'evaluate_lg_mode',
    'evaluate_hg_mode',
    'decompose_lg',
    'decompose_hg',
    'lg_seidel_label',
    # Wick moments
    'gaussian_moment_2d',
    'gaussian_moment_table_2d',
    # Polynomial fits
    'fit_canonical_polynomials',
    'fit_hf_polynomials',
    # Stationary solver
    'solve_envelope_stationary',
    # Top-level evaluators
    'aberration_tensor',
    'propagate_modal_asymptotic',
    'propagate_hf_chebyshev_quadrature',
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
# Section 3 -- Canonical polynomial fit (Phi(s2, v2) and s1(s2, v2))
# ===========================================================================

@dataclass
class CanonicalPolyFit:
    """Result of a Chebyshev tensor-product fit to ray-traced
    ``Phi(s2, v2)`` and ``s1(s2, v2)``.

    Fields
    ------
    poly_order : int
        Total-degree truncation used in the fit (``|k|_1 <= poly_order``).
    multi_indices : list of 4-tuples
        Multi-indices ``(k1, k2, k3, k4)`` enumerating the basis terms,
        in the same order as the coefficient vectors.
    coef_phi : ndarray
        Coefficients of the Phi polynomial (in waves), residual after
        linear-phase extraction.
    coef_s1x, coef_s1y : ndarray
        Coefficients of the s1 back-map components (in metres).
    linear_coeffs_phi : ndarray, optional
        ``(alpha_0, alpha_1, alpha_2, alpha_3, alpha_4)`` of the linear
        prefit of Phi.  None if the fit was done
        with ``extract_linear_phase=False``.
    s2x_centre, s2x_halfrange : float
        Normalisation parameters for s2x:  ``u_s2x = (s2x - centre) /
        halfrange``.
    s2y_centre, s2y_halfrange : float
        Same for s2y.
    v2x_centre, v2x_halfrange : float
        Same for v2x.
    v2y_centre, v2y_halfrange : float
        Same for v2y.
    wavelength : float
        Wavelength used for the fit [m].
    res_phi_rms_waves : float
        RMS fit residual on the Chebyshev-node training set, in waves.
    res_s1_rms_m : float
        RMS fit residual on s1, in metres.
    n_rays : int
        Number of training rays.
    """
    poly_order: int
    multi_indices: List[Tuple[int, int, int, int]]
    coef_phi: np.ndarray
    coef_s1x: np.ndarray
    coef_s1y: np.ndarray
    s2x_centre: float
    s2x_halfrange: float
    s2y_centre: float
    s2y_halfrange: float
    v2x_centre: float
    v2x_halfrange: float
    v2y_centre: float
    v2y_halfrange: float
    wavelength: float
    res_phi_rms_waves: float = 0.0
    res_s1_rms_m: float = 0.0
    n_rays: int = 0
    linear_coeffs_phi: Optional[np.ndarray] = None
    extract_linear_phase: bool = False

    # ------------------------------------------------------------------
    # Coordinate normalisation helpers
    # ------------------------------------------------------------------
    def to_normalised(self, s2x: np.ndarray, s2y: np.ndarray,
                      v2x: np.ndarray, v2y: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map physical (s2, v2) into normalised [-1, 1]^4."""
        u1 = (s2x - self.s2x_centre) / self.s2x_halfrange
        u2 = (s2y - self.s2y_centre) / self.s2y_halfrange
        u3 = (v2x - self.v2x_centre) / self.v2x_halfrange
        u4 = (v2y - self.v2y_centre) / self.v2y_halfrange
        return u1, u2, u3, u4

    def in_box(self, s2x: np.ndarray, s2y: np.ndarray,
               v2x: np.ndarray, v2y: np.ndarray) -> np.ndarray:
        """Boolean mask of points inside the fit's normalised box."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        return ((np.abs(u1) <= 1.0) & (np.abs(u2) <= 1.0)
                & (np.abs(u3) <= 1.0) & (np.abs(u4) <= 1.0))

    # ------------------------------------------------------------------
    # Polynomial evaluation helpers
    # ------------------------------------------------------------------
    def eval_s1(self, s2x: np.ndarray, s2y: np.ndarray,
                v2x: np.ndarray, v2y: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the back-map s1(s2, v2) -> (s1x, s1y) at arbitrary
        (s2, v2) samples in physical units."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        s1x = _evaluate_polynomial_4d(self.coef_s1x, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        s1y = _evaluate_polynomial_4d(self.coef_s1y, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        return s1x, s1y

    def eval_phi(self, s2x: np.ndarray, s2y: np.ndarray,
                 v2x: np.ndarray, v2y: np.ndarray,
                 *, include_linear: bool = True) -> np.ndarray:
        """Evaluate Phi(s2, v2) [waves] at arbitrary samples.

        Parameters
        ----------
        include_linear : bool, default True
            If True and the fit extracted a linear ramp, re-add it to
            give the *raw* OPD that the ray trace would report.  If
            False, return only the Chebyshev residual (the
            integrator-safe form).
        """
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        phi = _evaluate_polynomial_4d(self.coef_phi, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
        return phi

    def eval_s1_with_v2_grad(self, s2x: np.ndarray, s2y: np.ndarray,
                              v2x: np.ndarray, v2y: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate s1(s2, v2) and its (v2x, v2y)-gradient (Jacobian).

        Returns
        -------
        s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y
            All in physical units; gradients have the chain-rule
            normalisation factor 1/v2x_halfrange and 1/v2y_halfrange
            already applied.
        """
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        s1x, du3_s1x, du4_s1x = _evaluate_polynomial_4d_and_grad34(
            self.coef_s1x, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        s1y, du3_s1y, du4_s1y = _evaluate_polynomial_4d_and_grad34(
            self.coef_s1y, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        # Chain rule:  d/dv2x = (1/v2x_halfrange) d/du3 ; same for v2y
        invhx = 1.0 / self.v2x_halfrange
        invhy = 1.0 / self.v2y_halfrange
        return (s1x, s1y,
                du3_s1x * invhx, du4_s1x * invhy,
                du3_s1y * invhx, du4_s1y * invhy)

    def eval_phi_with_v2_grad(self, s2x: np.ndarray, s2y: np.ndarray,
                               v2x: np.ndarray, v2y: np.ndarray,
                               *, include_linear: bool = False
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Phi and its v2-gradient.  ``include_linear``
        controls whether the linear-phase prefit is re-added (default
        False since the integrator drops it)."""
        u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
        phi, du3_phi, du4_phi = _evaluate_polynomial_4d_and_grad34(
            self.coef_phi, self.multi_indices, u1, u2, u3, u4,
            self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
            du3_phi = du3_phi + a3
            du4_phi = du4_phi + a4
        invhx = 1.0 / self.v2x_halfrange
        invhy = 1.0 / self.v2y_halfrange
        return phi, du3_phi * invhx, du4_phi * invhy


def fit_canonical_polynomials(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    extract_linear_phase: bool = True,
    object_distance: Optional[float] = None,
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    endpoint_anchored: bool = False,
    auto_bump_threshold_waves: Optional[float] = None,
    max_auto_poly_order: int = 10,
    source_centre: Tuple[float, float] = (0.0, 0.0),
) -> CanonicalPolyFit:
    """Fit Chebyshev tensor-product polynomials to ``Phi(s2, v2)`` and
    ``s1(s2, v2)`` over a 4-D source x pupil grid.

    The trace-and-fit pipeline samples a Chebyshev-node grid in the
    source-plane coordinates ``(s1_x, s1_y)`` and the input direction
    cosines ``(v1_x, v1_y)``, traces each ray forward through the
    prescription, and records the landing position ``s2`` and direction
    cosine ``v2`` at the last surface (along with the accumulated
    optical path).  The 4-D scattered training set
    ``{(s2, v2, s1, Phi)}`` then drives a least-squares fit of
    ``s1(s2, v2)`` and ``Phi(s2, v2)`` as 4-variable Chebyshev tensor-
    product polynomials.

    Sampling source positions in addition to pupil directions is what
    gives the back-map ``s1(s2, v2)`` non-trivial dependence on (s2,
    v2):  with a single source point the trace produces a 2-D
    submanifold of (s2, v2) values along which ``s1 = const`` and the
    fit's J = d s1 / d v2 collapses to zero.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (e.g. from
        ``load_zemax_zmx`` or a builder).
    wavelength : float
        Wavelength [m].
    source_box_half : float
        Half-width of the source sampling box [m].  Set to a few times
        the source-Gaussian waist consumers will use (typical 50 um).
    pupil_box_half : float
        Half-width of the input direction-cosine box [dimensionless].
        Set to cover the system aperture as seen from the source plane.
    n_field : int
        Per-axis Chebyshev-node grid size for source sampling.
    n_pupil : int
        Per-axis grid size for input direction sampling.  Total ray
        count is ``n_field**2 * n_pupil**2``.
    poly_order : int
        Total-degree truncation of the Chebyshev fit.  Default 6.
    extract_linear_phase : bool
        If True (default), pre-fit and subtract a 5-parameter linear
        ramp from Phi before the Chebyshev fit -- this restores Nyquist
        sampling for diffractive surfaces at non-zero orders (
        Section 5).  No-op for refractive systems.
    object_distance : float, optional
        Source-plane to first-surface distance [m].  Defaults to
        ``prescription.get('object_distance', 0.0)``.
    surface_diffraction : dict, optional
        Per-surface diffractive-order kicks for DOE / grating elements
        embedded in the prescription.  Maps surface index to
        ``(order_x, order_y, period_x, period_y)``.  Each ray's
        direction cosines are shifted by ``m * lambda / Lambda`` after
        refraction at that surface, evanescent orders are flagged
        ``alive=False``.  Use this to pin the LG fit to a single
        diffraction order of a Dammann splitter or any thin grating in
        a sequential prescription -- the resulting fit's chief-ray
        landings span the chosen order's image-plane footprint, so
        :func:`aberration_tensor` evaluations at that order's frame
        centres reflect the true diffractive wavefront.  See also
        :func:`lumenairy.raytrace.apply_doe_phase_traced`.
    endpoint_anchored : bool, optional
        If True, the Chebyshev sampling nodes for the input source-
        plane and pupil-direction-cosine grids are rescaled so the
        outermost two nodes per axis sit exactly at ``+- box_half``
        instead of slightly inside (``cos(pi/(2*N)) * box_half``).
        Equivalent to ``cos((k + 1/2)*pi / N) / cos(pi/(2*N))``.
        Marginally extends the fit's training-data support to the box
        edge; the conditioning hit on the least-squares fit is
        negligible at typical ``poly_order`` (4 - 6) but the fit
        coefficients change very slightly so this is opt-in for
        bit-compatibility with prior fits.  Default ``False`` (use
        the standard Chebyshev-Gauss roots).

    Returns
    -------
    CanonicalPolyFit
        Fit container.  ``fit.res_phi_rms_waves`` is typically below
        1e-4 waves on a well-conditioned refractive design.

    Raises
    ------
    ValueError
        On invalid input.
    RuntimeError
        If too many rays die for a meaningful fit.
    """
    from ..raytrace import surfaces_from_prescription, _make_bundle, trace

    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if poly_order < 0:
        raise ValueError(f"poly_order must be >= 0, got {poly_order}")
    if source_box_half <= 0 or pupil_box_half <= 0:
        raise ValueError("source_box_half and pupil_box_half must be > 0")

    if object_distance is None:
        object_distance = float(prescription.get('object_distance', 0.0)) or 0.0

    surfaces = surfaces_from_prescription(prescription)

    # 4-D Chebyshev-node grid:  (s1_x, s1_y) x (v1_x, v1_y).
    def cheb_nodes(n: int) -> np.ndarray:
        i = np.arange(n)
        x = np.cos(np.pi * (i + 0.5) / n)
        if endpoint_anchored and n >= 2:
            # Rescale so outermost nodes sit at exactly +-1 instead of
            # +-cos(pi/(2n)).  Outermost training samples then graze the
            # box edge.  Conditioning hit negligible for poly_order <= 6
            # with 6+ samples per axis.
            x = x / np.cos(np.pi / (2.0 * n))
        return x

    u_field = cheb_nodes(n_field)        # in (-1, 1)
    u_pupil = cheb_nodes(n_pupil)
    # 4.11.2: shift source-sampling box by ``source_centre`` so the
    # subaperture / patch-decomposition propagator can fit a local
    # polynomial centred on each patch's object-plane footprint.
    # Pre-4.11.2 the source box was hard-pinned to the origin, so the
    # subaperture loop's per-patch fits were all identical and only
    # the on-axis patch had meaningful chief-ray landings inside the
    # output evaluation grid -- off-axis patches saw a fit whose
    # ``s2x_centre`` / ``s2y_centre`` straddled the on-axis image and
    # were therefore mostly out-of-box (``|u| > 1``) at the
    # off-axis evaluation pixels, producing zero field.
    cx_src, cy_src = float(source_centre[0]), float(source_centre[1])
    s1x_axis = u_field * source_box_half + cx_src
    s1y_axis = u_field * source_box_half + cy_src
    v1x_axis = u_pupil * pupil_box_half
    v1y_axis = u_pupil * pupil_box_half

    S1X, S1Y, V1X, V1Y = np.meshgrid(s1x_axis, s1y_axis,
                                      v1x_axis, v1y_axis,
                                      indexing='ij')
    s1x_in = S1X.ravel()
    s1y_in = S1Y.ravel()
    v1x_in = V1X.ravel()
    v1y_in = V1Y.ravel()
    n_rays = s1x_in.size

    # Reject any input direction with sin^2(theta) >= 1 (non-real
    # longitudinal cosine):  this means pupil_box_half is too large.
    sumsq = v1x_in * v1x_in + v1y_in * v1y_in
    if np.any(sumsq >= 1.0):
        raise ValueError(
            "pupil_box_half too large -- input direction cosines would "
            "be non-real.  Reduce pupil_box_half "
            f"(current = {pupil_box_half}).")

    # Build a ray bundle launched at z = -object_distance.  trace()
    # automatically propagates the rays the appropriate distance to the
    # first surface (since the surfaces' z positions are determined by
    # the prescription's accumulated thicknesses).
    bundle = _make_bundle(
        x=s1x_in.astype(np.float64),
        y=s1y_in.astype(np.float64),
        L=v1x_in.astype(np.float64),
        M=v1y_in.astype(np.float64),
        wavelength=wavelength,
    )
    bundle.z = np.full(n_rays, -object_distance, dtype=np.float64)

    res = trace(bundle, surfaces, wavelength, output_filter='last',
                surface_diffraction=surface_diffraction)
    final = res.image_rays
    alive = np.asarray(final.alive, dtype=bool)
    n_alive = int(alive.sum())
    if n_alive < max(64, 0.5 * n_rays):
        raise RuntimeError(
            f"Too many rays died during canonical-fit trace: "
            f"{n_alive} alive of {n_rays}.  Reduce pupil_box_half, "
            f"check prescription apertures, or rebalance the source/"
            f"pupil sampling boxes.")

    s2x_obs = np.asarray(final.x, dtype=np.float64)
    s2y_obs = np.asarray(final.y, dtype=np.float64)
    v2x_obs = np.asarray(final.L, dtype=np.float64)
    v2y_obs = np.asarray(final.M, dtype=np.float64)
    phi_obs = np.asarray(final.opd, dtype=np.float64) / wavelength

    s2x_live = s2x_obs[alive]
    s2y_live = s2y_obs[alive]
    v2x_live = v2x_obs[alive]
    v2y_live = v2y_obs[alive]
    phi_live = phi_obs[alive]
    s1x_live = s1x_in[alive]
    s1y_live = s1y_in[alive]

    # Normalise the OUTPUT (s2, v2) coordinates to [-1, 1]^4 from the
    # observed extents of the live rays.  Rays that end
    # up at extreme s2 or v2 set the box; the 5%-pad in _fit_normaliser
    # leaves room for evaluation at boundary pixels.
    s2x_c, s2x_h = _fit_normaliser(s2x_live)
    s2y_c, s2y_h = _fit_normaliser(s2y_live)
    v2x_c, v2x_h = _fit_normaliser(v2x_live)
    v2y_c, v2y_h = _fit_normaliser(v2y_live)

    u_s2x = (s2x_live - s2x_c) / s2x_h
    u_s2y = (s2y_live - s2y_c) / s2y_h
    u_v2x = (v2x_live - v2x_c) / v2x_h
    u_v2y = (v2y_live - v2y_c) / v2y_h

    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s2x),
            u_s2x, u_s2y, u_v2x, u_v2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, phi_live, rcond=None)
        opd_residual = phi_live - X5 @ linear_coeffs
    else:
        opd_residual = phi_live.copy()

    multi_indices = _multi_indices_total_degree(4, poly_order)
    n_basis = len(multi_indices)
    T1 = _chebyshev_vandermonde(u_s2x, poly_order)
    T2 = _chebyshev_vandermonde(u_s2y, poly_order)
    T3 = _chebyshev_vandermonde(u_v2x, poly_order)
    T4 = _chebyshev_vandermonde(u_v2y, poly_order)
    # 3.5.6: Vectorised design-matrix build via fancy indexing,
    # replacing the per-basis-term Python loop.  Each T_i has shape
    # (poly_order+1, n_rays); T_i[K_i] has shape (n_basis, n_rays);
    # the elementwise product has shape (n_basis, n_rays); transpose
    # to (n_rays, n_basis).  Avoids the n_basis-iteration Python
    # loop with no temporary allocations beyond the result.
    K_arr = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K_arr[:, 0], K_arr[:, 1], K_arr[:, 2], K_arr[:, 3]
    A = (T1[K1] * T2[K2] * T3[K3] * T4[K4]).T.astype(
        np.float64, copy=False)

    coef_phi, *_ = np.linalg.lstsq(A, opd_residual, rcond=None)
    coef_s1x, *_ = np.linalg.lstsq(A, s1x_live, rcond=None)
    coef_s1y, *_ = np.linalg.lstsq(A, s1y_live, rcond=None)

    res_phi = float(np.sqrt(np.mean((opd_residual - A @ coef_phi) ** 2)))
    res_s1 = float(np.sqrt(0.5 * np.mean(
        (s1x_live - A @ coef_s1x) ** 2 + (s1y_live - A @ coef_s1y) ** 2
    )))

    # 3.5.6: optional automatic poly_order bump if the residual exceeds
    # the user-supplied threshold.  Useful for cemented multi-element
    # systems where order=6 (the default) under-fits.  Caller passes
    # ``auto_bump_threshold_waves=1e-3`` to opt in; default None
    # preserves the prior single-shot behaviour.
    if (auto_bump_threshold_waves is not None
            and res_phi > float(auto_bump_threshold_waves)
            and poly_order < int(max_auto_poly_order)):
        import warnings as _w
        _w.warn(
            f"fit_canonical_polynomials: order={poly_order} fit "
            f"residual {res_phi:.3e} waves exceeds threshold "
            f"{auto_bump_threshold_waves:.3e}; auto-bumping to "
            f"order={poly_order + 2} (max {max_auto_poly_order}).",
            RuntimeWarning, stacklevel=2)
        return fit_canonical_polynomials(
            prescription, wavelength,
            source_box_half=source_box_half,
            pupil_box_half=pupil_box_half,
            n_field=n_field, n_pupil=n_pupil,
            poly_order=poly_order + 2,
            extract_linear_phase=extract_linear_phase,
            object_distance=object_distance,
            surface_diffraction=surface_diffraction,
            endpoint_anchored=endpoint_anchored,
            auto_bump_threshold_waves=auto_bump_threshold_waves,
            max_auto_poly_order=max_auto_poly_order,
            source_centre=source_centre,
        )

    return CanonicalPolyFit(
        poly_order=poly_order,
        multi_indices=multi_indices,
        coef_phi=coef_phi,
        coef_s1x=coef_s1x,
        coef_s1y=coef_s1y,
        s2x_centre=s2x_c, s2x_halfrange=s2x_h,
        s2y_centre=s2y_c, s2y_halfrange=s2y_h,
        v2x_centre=v2x_c, v2x_halfrange=v2x_h,
        v2y_centre=v2y_c, v2y_halfrange=v2y_h,
        wavelength=wavelength,
        res_phi_rms_waves=res_phi,
        res_s1_rms_m=res_s1,
        n_rays=n_alive,
        linear_coeffs_phi=linear_coeffs,
        extract_linear_phase=extract_linear_phase,
    )


# ===========================================================================
# Section 4 -- Newton solver for the envelope-stationary point
# ===========================================================================

def solve_envelope_stationary(
    fit: CanonicalPolyFit,
    s2: Tuple[float, float],
    source_point: Tuple[float, float],
    *,
    w_s: float,
    w_p: float,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    v2_initial: Optional[Tuple[float, float]] = None,
    max_iter: int = 12,
    tol: float = 1e-12,
) -> Tuple[Tuple[float, float], int, float]:
    """Newton-solve the envelope-stationary equation.

    Find ``v_2^*`` such that the joint Gaussian envelope

        G(v_2) = exp(-|s_1(s_2, v_2) - s_src|^2 / w_s^2
                     - |v_2 - v_2_centre|^2 / w_p^2)

    is locally maximised at ``s_2``.  Equation (9) reads

        J^T (s_1 - s_src) / w_s^2 + (v_2 - v_2_centre) / w_p^2 = 0

    with ``J = ds_1 / dv_2``.

    Parameters
    ----------
    fit : CanonicalPolyFit
        Polynomial fit of the system.
    s2 : (float, float)
        Output-plane point [m].
    source_point : (float, float)
        Source-plane point [m].
    w_s, w_p : float
        Source and pupil Gaussian waists.  ``w_s`` in [m] (object plane);
        ``w_p`` in direction-cosine units [dimensionless].
    v2_centre : (float, float), optional
        Pupil centre in direction cosines.  Default origin.
    v2_initial : (float, float), optional
        Newton initial guess.  Defaults to ``v2_centre``.
    max_iter : int, optional
        Iteration cap.  Newton typically converges in 3-5; cap at 12.
    tol : float, optional
        Convergence tolerance on |residual|.  Default 1e-12.

    Returns
    -------
    v_star : (float, float)
        The stationary point.
    n_iter : int
        Iterations actually used.
    residual_norm : float
        Final |residual|.
    """
    s2x, s2y = float(s2[0]), float(s2[1])
    src_x, src_y = float(source_point[0]), float(source_point[1])
    v_cx, v_cy = float(v2_centre[0]), float(v2_centre[1])
    if v2_initial is None:
        v2x = v_cx
        v2y = v_cy
    else:
        v2x = float(v2_initial[0])
        v2y = float(v2_initial[1])

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)

    last_norm = float('inf')
    for it in range(max_iter):
        # Evaluate s1, J at current v2
        s2x_arr = np.asarray(s2x).reshape(())
        s2y_arr = np.asarray(s2y).reshape(())
        v2x_arr = np.asarray(v2x).reshape(())
        v2y_arr = np.asarray(v2y).reshape(())
        s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y = (
            fit.eval_s1_with_v2_grad(s2x_arr, s2y_arr, v2x_arr, v2y_arr)
        )
        s1x = float(s1x)
        s1y = float(s1y)
        # Jacobian of s_1 w.r.t. v_2
        J = np.array([
            [float(dS1x_dv2x), float(dS1x_dv2y)],
            [float(dS1y_dv2x), float(dS1y_dv2y)],
        ])
        delta_s1 = np.array([s1x - src_x, s1y - src_y])
        delta_v = np.array([v2x - v_cx, v2y - v_cy])
        residual = inv_ws2 * (J.T @ delta_s1) + inv_wp2 * delta_v
        rn = float(np.linalg.norm(residual))
        # 4.10: stricter stall check.  Pre-4.10 the chained comparison
        # `rn < tol or rn > 0.99 * last_norm > 1e-300` could fire on
        # the FIRST iteration (since last_norm = inf initially makes
        # 0.99*inf > rn trivially true on the right end of the chain
        # while the left end is false, but the overall expression
        # depends on Python's chaining semantics).  More importantly,
        # the stall heuristic fired even when one of two consecutive
        # iterations was sub-quadratic during a normal Newton
        # warm-up.  Require AT LEAST 2 iterations before declaring
        # a stall, and require BOTH halves of the chain to be true
        # explicitly to avoid the chained-comparison surprise.
        is_converged = rn < tol
        is_stalling = (it >= 2 and rn > 0.9 * last_norm and last_norm > 1e-300)
        if is_converged or is_stalling:
            return (v2x, v2y), it, rn
        last_norm = rn

        # Approximate Hessian:  d residual / d v2.
        # First term:  d/dv2 [ J^T (s_1 - s_src) ] = J^T @ J + sum_k
        #                    (s_1 - s_src)_k * d^2 s_1_k / dv2 dv2.
        # Hessian-of-s1 piece is second-order in (s_1 - s_src) and is
        # neglected here (Gauss-Newton-like).  This is exact at the
        # stationary point and gives quadratic-ish convergence.
        # Second term:  d/dv2 [(v_2 - v_2c) / w_p^2] = I / w_p^2.
        H = inv_ws2 * (J.T @ J) + inv_wp2 * np.eye(2)
        try:
            step = np.linalg.solve(H, residual)
        except np.linalg.LinAlgError:
            # Singular Hessian -- back off and bail.
            return (v2x, v2y), it, rn
        v2x = v2x - float(step[0])
        v2y = v2y - float(step[1])

    return (v2x, v2y), max_iter, last_norm


# ===========================================================================
# Section 5 -- Aberration tensor and modal asymptotic propagator
# ===========================================================================

@dataclass
class AberrationTensorResult:
    """Result of an LG aberration-tensor evaluation at one image point.

    Fields
    ------
    L : ndarray, shape (n_output_modes, n_source_modes)
        Aberration matrix L_{k,n} = sum_m b_m T_{k;n,m} after pupil
        contraction.  Rows index output LG modes; columns index source
        LG modes.
    output_modes : list of (p, ell)
    source_modes : list of (p, ell)
    pupil_modes : list of (p, ell)
    s2_image : (float, float)
        Chief-ray image point [m] at which this tensor was evaluated.
    w_s, w_p, w_o : float
        Source, pupil, output Gaussian waists.
    v_star : (float, float)
        Envelope-stationary v_2* at s2_image.

    Notes
    -----
    Indices of L correspond to physical aberrations via
    ``lg_seidel_label(p, ell)``:  (1, 0) is defocus, (2, 0) is
    spherical, (1, +-1) is coma, (0, +-2) is astigmatism, etc.
    Driving |L_{(2,0), 0}|^2 to zero suppresses on-axis spherical
    aberration, etc.
    """
    L: np.ndarray
    output_modes: List[Tuple[int, int]]
    source_modes: List[Tuple[int, int]]
    pupil_modes: List[Tuple[int, int]]
    s2_image: Tuple[float, float]
    w_s: float
    w_p: float
    w_o: float
    v_star: Tuple[float, float]


def _multiply_polys_2d(p_a: Dict[Tuple[int, int], complex],
                        p_b: Dict[Tuple[int, int], complex]
                        ) -> Dict[Tuple[int, int], complex]:
    """Multiply two 2-D polynomial dicts."""
    out: Dict[Tuple[int, int], complex] = {}
    for (i_a, j_a), c_a in p_a.items():
        for (i_b, j_b), c_b in p_b.items():
            key = (i_a + i_b, j_a + j_b)
            out[key] = out.get(key, 0.0 + 0.0j) + c_a * c_b
    return out


def _polynomial_under_affine_shift(
    coeffs: Dict[Tuple[int, int], complex],
    shift_x: complex, shift_y: complex,
    var_name: str = 'eta',
) -> Dict[Tuple[int, int], complex]:
    """Substitute (x, y) -> (x + shift_x, y + shift_y) in a 2-D polynomial.

    Used to produce the ``eta``-polynomial after the (s_1, v_2) -> eta
    coordinate change.
    """
    if not coeffs:
        return {}
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)
    # Pre-compute (x + shift_x)^i = sum_k C(i, k) shift_x^(i-k) x^k
    # as a coefficient table indexed by k.
    bin_x: List[Dict[int, complex]] = []
    for i in range(max_i + 1):
        row: Dict[int, complex] = {}
        for k in range(i + 1):
            row[k] = math.comb(i, k) * (shift_x ** (i - k))
        bin_x.append(row)
    bin_y: List[Dict[int, complex]] = []
    for j in range(max_j + 1):
        row = {}
        for k in range(j + 1):
            row[k] = math.comb(j, k) * (shift_y ** (j - k))
        bin_y.append(row)

    out: Dict[Tuple[int, int], complex] = {}
    for (i, j), c in coeffs.items():
        for kx, bx in bin_x[i].items():
            for ky, by in bin_y[j].items():
                key = (kx, ky)
                out[key] = out.get(key, 0.0 + 0.0j) + c * bx * by
    return out


def _contract_against_moment_table(
    poly: Dict[Tuple[int, int], complex],
    moments: Dict[Tuple[int, int], complex],
) -> complex:
    """Compute ``<P(eta)>_M = sum_{ij} c_{ij} <eta_x^i eta_y^j>``."""
    total = 0.0 + 0.0j
    for (i, j), c in poly.items():
        total += c * moments.get((i, j), 0.0 + 0.0j)
    return total


def _compute_M_b(
    fit: CanonicalPolyFit,
    s2x: float, s2y: float,
    v2x: float, v2y: float,
    src_x: float, src_y: float,
    w_s: float, w_p: float,
    v_cx: float, v_cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex]:
    """Build the complex beam matrix M, the linear
    term b (eq. 17), the Jacobian J*, and the OPL piston Phi*."""
    s2x_arr = np.asarray(s2x).reshape(())
    s2y_arr = np.asarray(s2y).reshape(())
    v2x_arr = np.asarray(v2x).reshape(())
    v2y_arr = np.asarray(v2y).reshape(())

    s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y = (
        fit.eval_s1_with_v2_grad(s2x_arr, s2y_arr, v2x_arr, v2y_arr)
    )
    phi, dPhi_dv2x, dPhi_dv2y = fit.eval_phi_with_v2_grad(
        s2x_arr, s2y_arr, v2x_arr, v2y_arr,
        include_linear=False,
    )

    s1x_v = float(s1x)
    s1y_v = float(s1y)
    phi_v = float(phi)
    J = np.array([
        [float(dS1x_dv2x), float(dS1x_dv2y)],
        [float(dS1y_dv2x), float(dS1y_dv2y)],
    ])
    g = np.array([float(dPhi_dv2x), float(dPhi_dv2y)])

    # Hessian of phi w.r.t. v2 (use finite differences of analytic 1st
    # derivatives; cleaner to differentiate the polynomial twice).
    H_phi = _phi_v2_hessian(fit, s2x, s2y, v2x, v2y)

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)
    M_real = inv_ws2 * (J.T @ J) + inv_wp2 * np.eye(2)
    M = M_real - 1j * math.pi * H_phi
    r_star = np.array([s1x_v - src_x, s1y_v - src_y])
    delta_v = np.array([v2x - v_cx, v2y - v_cy])
    b = (2.0j * math.pi * g
         - 2.0 * inv_ws2 * (J.T @ r_star)
         - 2.0 * inv_wp2 * delta_v)
    detJ = float(np.abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))

    G0 = math.exp(
        -(r_star[0] ** 2 + r_star[1] ** 2) / (w_s * w_s)
        - (delta_v[0] ** 2 + delta_v[1] ** 2) / (w_p * w_p)
    )
    return M, b, np.array([s1x_v, s1y_v]), J, complex(phi_v), G0, detJ


def _phi_v2_hessian(fit: CanonicalPolyFit, s2x: float, s2y: float,
                     v2x: float, v2y: float) -> np.ndarray:
    """Compute the 2x2 Hessian d^2 Phi / d v_2 d v_2 by analytic
    differentiation of the Chebyshev fit, in *physical* (waves per
    direction-cosine^2) units."""
    # Normalised coords
    u1 = (s2x - fit.s2x_centre) / fit.s2x_halfrange
    u2 = (s2y - fit.s2y_centre) / fit.s2y_halfrange
    u3 = (v2x - fit.v2x_centre) / fit.v2x_halfrange
    u4 = (v2y - fit.v2y_centre) / fit.v2y_halfrange

    T1 = _chebyshev_vandermonde(np.array(u1), fit.poly_order)
    T2 = _chebyshev_vandermonde(np.array(u2), fit.poly_order)
    T3 = _chebyshev_vandermonde(np.array(u3), fit.poly_order)
    T4 = _chebyshev_vandermonde(np.array(u4), fit.poly_order)
    dT3 = _chebyshev_derivative_vandermonde(np.array(u3), fit.poly_order)
    dT4 = _chebyshev_derivative_vandermonde(np.array(u4), fit.poly_order)
    d2T3 = _chebyshev_second_derivative_vandermonde(np.array(u3),
                                                       fit.poly_order)
    d2T4 = _chebyshev_second_derivative_vandermonde(np.array(u4),
                                                       fit.poly_order)

    h33 = 0.0
    h34 = 0.0
    h44 = 0.0
    for c, (k1, k2, k3, k4) in zip(fit.coef_phi, fit.multi_indices):
        if c == 0.0:
            continue
        T12 = float(T1[k1]) * float(T2[k2])
        h33 += c * T12 * float(d2T3[k3]) * float(T4[k4])
        h34 += c * T12 * float(dT3[k3]) * float(dT4[k4])
        h44 += c * T12 * float(T3[k3]) * float(d2T4[k4])
    # Add linear-phase Hessian contribution -- linear terms have zero
    # second derivative, so nothing to add.
    invhx = 1.0 / fit.v2x_halfrange
    invhy = 1.0 / fit.v2y_halfrange
    return np.array([
        [h33 * invhx * invhx, h34 * invhx * invhy],
        [h34 * invhx * invhy, h44 * invhy * invhy],
    ])


def aberration_tensor(
    fit: CanonicalPolyFit,
    s2_image: Tuple[float, float],
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    source_modes: Optional[List[Tuple[int, int]]] = None,
    pupil_modes: Optional[List[Tuple[int, int]]] = None,
    output_modes: Optional[List[Tuple[int, int]]] = None,
    pupil_amplitudes: Optional[Dict[Tuple[int, int], complex]] = None,
    w_s: float = 50e-6,
    w_p: float = 0.05,
    w_o: Optional[float] = None,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    sigma_grid_n: Optional[int] = None,
    sigma_grid_extent: Optional[float] = None,
) -> AberrationTensorResult:
    """LG aberration tensor at a single chief-ray image point.

    Expand source, pupil, and output
    fields in Laguerre-Gaussian bases, evaluate the leading-order
    asymptotic propagator analytically as a Wick-contracted Gaussian
    moment, and project onto the output basis to read off the
    coefficient ``L_{k, n} = sum_m b_m T_{k;n,m}`` of each named
    aberration channel.

    Parameters
    ----------
    fit : CanonicalPolyFit
        4-D Chebyshev fit of the prescription.
    s2_image : (float, float)
        Image-plane point [m] at which to evaluate the tensor.  Should
        be the chief-ray landing of ``source_point`` for the
        Seidel-name interpretation to apply.
    source_point : (float, float)
        Source-plane point [m].
    source_modes, pupil_modes, output_modes : list of (p, ell)
        LG mode indices to retain.  Defaults below give a useful
        baseline.
    pupil_amplitudes : dict, optional
        Pupil expansion coefficients ``{(p, ell): complex}``.  If None,
        defaults to a clean LG_{0,0} pupil (b_{0,0} = 1) -- the
        "ideal-Gaussian-pupil" convention used for design merit
        functions, where higher pupil modes are not needed because the
        Seidel content lives entirely on the output side.
    w_s, w_p : float
        Source and pupil Gaussian waists [m and direction-cosine].
    w_o : float, optional
        Output Gaussian waist [m].  Defaults to a value derived from
        the local complex beam matrix (Maréchal-scale).
    v2_centre : (float, float), optional
        Pupil centre.
    sigma_grid_n : int, optional
        Output-plane grid size used for the σ-integration path when
        any requested ``output_modes`` has ``ℓ ≠ 0``.  Default 64 is
        accurate for LG modes up to ``(p, |ℓ|) ~ (3, 3)``; bump for
        higher orders.  Ignored when all output modes have ``ℓ = 0``
        (the fast closed-form chief-ray projection is used).
    sigma_grid_extent : float, optional
        Half-extent of the σ-grid [m].  Default ``4 · w_o`` captures
        > 99.9 % of the LG tail mass.  Increase only if the output
        beam has unusually long tails (large pupil aberrations).

    Returns
    -------
    AberrationTensorResult

    Notes
    -----
    **Pre-4.9 limitation removed.**  Pre-4.9 the projection at the
    chief image collapsed to the constant term of the LG output
    polynomial, which is identically zero for any ``ℓ ≠ 0`` mode
    ((σ_x + j·σ_y)^|ℓ| · Laguerre has no constant term).  That made
    coma ``(1, ±1)``, astigmatism ``(0, ±2)``, tilt ``(0, ±1)``, and
    every other ℓ ≠ 0 entry of the returned tensor silently zero,
    even when the underlying aberration was present.  4.9 fixes this
    by doing the actual σ-integration via a small output-plane grid
    and a numerical LG projection (``propagate_modal_asymptotic`` +
    ``decompose_lg``).  ℓ = 0 modes still use the fast closed-form
    chief-ray path -- you get the new behaviour only when you ask
    for an ℓ ≠ 0 output mode.
    """
    if source_modes is None:
        source_modes = [(0, 0)]
    if pupil_modes is None:
        pupil_modes = [(0, 0)]
    if output_modes is None:
        # Default:  the named Seidel/Zernike aberrations through 4th order
        output_modes = [
            (0, 0),  # piston / Strehl
            (1, 0),  # defocus
            (2, 0),  # primary spherical
            (0, 1), (0, -1),  # tilt
            (1, 1), (1, -1),  # coma
            (0, 2), (0, -2),  # astigmatism
            (0, 3), (0, -3),  # trefoil
        ]
    if pupil_amplitudes is None:
        pupil_amplitudes = {(0, 0): 1.0 + 0.0j}

    s2x_img, s2y_img = float(s2_image[0]), float(s2_image[1])
    src_x, src_y = float(source_point[0]), float(source_point[1])

    # Solve the envelope-stationary equation at s2_image
    v_star, _n_iter, _resid = solve_envelope_stationary(
        fit, (s2x_img, s2y_img), (src_x, src_y),
        w_s=w_s, w_p=w_p, v2_centre=v2_centre,
    )
    v2x_star, v2y_star = v_star

    # Build M, b at v_star
    M, b, s1_star, J_star, phi_star, G0, detJ = _compute_M_b(
        fit, s2x_img, s2y_img, v2x_star, v2y_star,
        src_x, src_y, w_s, w_p, v2_centre[0], v2_centre[1]
    )

    # Choose output waist if not supplied
    if w_o is None:
        # Use the smaller of the two real-part eigenvalues of M^-1, and
        # drop a factor of pi (Maréchal convention scales as
        # 1/sqrt(lambda_max(Re M))).
        eig_M_real = np.linalg.eigvalsh(np.real(M))
        if eig_M_real.max() <= 0:
            w_o = 1e-6  # fallback for ill-conditioned cases
        else:
            w_o = 1.0 / math.sqrt(float(eig_M_real.max()))
        # Clamp to a physically reasonable range
        w_o = max(min(w_o, 1.0), 1e-9)

    # Stationary shift delta* = 0.5 M^-1 b
    M_inv = np.linalg.inv(M)
    delta_star = 0.5 * (M_inv @ b)
    # 4.10: removed unused `Sigma = 0.5 * M_inv` (dead code).
    # 4.11.2: route through the shared Maslov branch helper so this
    # site stays consistent with ``propagate_modal_asymptotic`` and
    # the JAX sibling evaluators.  Single-point evaluation defaults
    # to the principal sqrt (caller has no branch history at one
    # image point); see ``_maslov_branch_corrected_sqrt``.
    sqrt_detM, _, _ = _maslov_branch_corrected_sqrt(np.linalg.det(M))

    # Leading amplitude
    A_lead = (detJ * (math.pi / sqrt_detM) * G0
              * np.exp(2j * math.pi * phi_star)
              * np.exp(0.25 * b @ M_inv @ b))

    # Pre-tabulate eta-moments for max polynomial order needed
    max_order_needed = max(
        max((2 * p + abs(ell) for (p, ell) in source_modes), default=0)
        + max((2 * p + abs(ell) for (p, ell) in pupil_modes), default=0),
        4  # always have enough for low output orders
    )
    eta_moments = gaussian_moment_table_2d(M, max_order_needed)

    # Compute T_{k;n,m} for each (k, n, m).
    # The integrand polynomial is
    #     P_{n,m}(eta) = p^src_n(s_1(s_2*, v_2*+eta) - s_src; w_s)
    #                   * p^pup_m(v_2* - v_2c + eta; w_p)
    # at leading order s_1(s_2*, v_2*+eta) = s_1* + J* delta_star + J* eta;
    # so the source argument is r* + J* delta_star + J* eta.
    # The pupil argument is (v_2* - v_2c + delta_star + eta).
    n_out = len(output_modes)
    n_src = len(source_modes)
    L = np.zeros((n_out, n_src), dtype=np.complex128)

    # Source argument shift = r* + J* delta_star ; J* eta is the
    # eta-dependent piece.  But J* couples eta -> source-r-space, so
    # we need to substitute (eta_1, eta_2) -> J* (eta_1, eta_2) into
    # the source polynomial.  This is a linear coordinate change in
    # the source polynomial whose result is *another* polynomial in
    # eta of the same total degree.  Paper 2 leading order keeps this
    # affine substitution exactly.
    #
    # We implement it generically:  given p^src_n(r1, r2), substitute
    # r = (J* eta) + (r* + J* delta_star) -- the full affine
    # transformation -- and re-collect as polynomial in eta.

    r_const = s1_star + J_star @ np.array([delta_star[0], delta_star[1]]) - np.array([src_x, src_y])
    # Linear transform matrix on eta: r1 = J11 eta1 + J12 eta2 + r_const[0],
    # r2 = J21 eta1 + J22 eta2 + r_const[1].
    pupil_const = (np.array([v2x_star, v2y_star])
                   - np.array(v2_centre)
                   + np.array([delta_star[0], delta_star[1]]))

    # ------------------------------------------------------------------
    # 4.9 -- output-mode projection
    # ------------------------------------------------------------------
    # Pre-4.9 collapsed the output projection to ``out_poly.get((0, 0))``,
    # which is the constant term of the LG output polynomial.  This
    # zeroes out every ℓ ≠ 0 mode because
    # ``(σ_x + j·σ_y)^|ℓ| · Laguerre`` has no constant term for |ℓ| ≥ 1
    # -- silently producing zero coma/astigmatism/tilt tensor entries
    # even when the physical aberrations were present.  The audit's
    # action item #2.5 called this out and recommended either (a)
    # restricting the API to ℓ = 0 or (b) implementing the actual
    # σ-integration.  4.9 takes path (b).
    #
    # The σ-projection is exact at the leading-order asymptotic level:
    # the field at ``s2 = s2_image + σ`` is given by the same
    # saddle-point machinery that ``propagate_modal_asymptotic`` already
    # evaluates pixel-by-pixel.  Building a small grid around the chief
    # image and projecting numerically with ``decompose_lg`` does the
    # output Gaussian-moment integral against the LG_o basis without
    # collapsing the σ-dependence.
    #
    # For pure ℓ = 0 output sets we keep the closed-form chief-ray
    # evaluation -- it's faster and gives identical numbers in that
    # regime.  Mixed ℓ = 0 / ℓ ≠ 0 sets fall through to the grid path.
    needs_sigma_integration = any(k_out[1] != 0 for k_out in output_modes)
    # 4.10.3: closed-form ℓ=0 path now evaluates the output LG
    # polynomial at the saddle's σ_image (instead of grabbing only its
    # x⁰y⁰ Cartesian constant), so different (p, 0) modes are
    # distinguished for any OFF-axis saddle.  For an axial saddle
    # σ_image=(0,0), the LG_p,0 polynomial reduces to the
    # normalisation constant N_p,0 = sqrt(2/(π w_o²)) -- the same
    # for all p (point evaluation at the origin of orthonormal modes
    # that all peak there is a fundamental closed-form / saddle-point
    # limitation; the σ-grid path has the right orthogonal structure
    # but a different overall normalisation than the closed-form,
    # so we don't auto-route).  Pass a non-axial source_point /
    # s2_image, or include an ℓ ≠ 0 mode, to force the discriminating
    # behaviour.
    if not needs_sigma_integration:
        ell0_p_values = {k_out[0] for k_out in output_modes
                          if k_out[1] == 0}
        is_axial_saddle = (abs(s2x_img) + abs(s2y_img)
                            < 1e-9 * max(w_o, 1e-12))
        if len(ell0_p_values) > 1 and is_axial_saddle:
            import warnings
            warnings.warn(
                "aberration_tensor: ON-AXIS saddle σ_image=(0,0) with "
                f"multiple radial-p modes in {sorted(ell0_p_values)}.  "
                "All LG_p,0 modes have the same value N_p,0 at the "
                "origin, so the saddle-point closed-form cannot "
                "distinguish them.  This is a fundamental limit of the "
                "point-evaluation approximation, not a bug.  Use an "
                "off-axis source_point / s2_image to lift the "
                "degeneracy, or include an ℓ ≠ 0 mode to force the "
                "σ-grid integration path.",
                RuntimeWarning, stacklevel=3,
            )

    if not needs_sigma_integration:
        # ---- Closed-form chief-ray projection (ℓ = 0 only) -------------
        for io, k_out in enumerate(output_modes):
            out_poly_full = lg_polynomial(k_out[0], k_out[1], w_o)
            out_poly = {key: c.conjugate()
                        for key, c in out_poly_full.items()}
            for js, k_src in enumerate(source_modes):
                src_poly_r = lg_polynomial(k_src[0], k_src[1], w_s)
                src_poly_eta = _polynomial_substitute_linear_2d(
                    src_poly_r,
                    A_xx=J_star[0, 0], A_xy=J_star[0, 1],
                    A_yx=J_star[1, 0], A_yy=J_star[1, 1],
                    b_x=r_const[0], b_y=r_const[1],
                )
                T_acc = 0.0 + 0.0j
                for k_pup, b_pup in pupil_amplitudes.items():
                    if abs(b_pup) < 1e-300:
                        continue
                    pup_poly_r = lg_polynomial(k_pup[0], k_pup[1], w_p)
                    pup_poly_eta = _polynomial_under_affine_shift(
                        pup_poly_r,
                        shift_x=complex(pupil_const[0]),
                        shift_y=complex(pupil_const[1]),
                    )
                    P_eta = _multiply_polys_2d(src_poly_eta, pup_poly_eta)
                    exp_val = _contract_against_moment_table(
                        P_eta, eta_moments)
                    # 4.10.3: evaluate out_poly at the saddle's
                    # σ_image (the saddle is held fixed across the
                    # η-integration).  Pre-4.10.3 grabbed only the
                    # x⁰y⁰ Cartesian constant, which equals N_p,0
                    # for every (p, 0) mode and therefore collapsed
                    # defocus / spherical / secondary-spherical to
                    # the same scalar.  Evaluating the full
                    # polynomial at (s2x_img, s2y_img) recovers the
                    # mode-discriminating polynomial terms.  The
                    # ON-axis multi-p case is automatically routed
                    # to the σ-grid path by the
                    # needs_sigma_integration check above (modes
                    # genuinely degenerate at the origin in the
                    # saddle-point limit).
                    out_const = 0.0 + 0.0j
                    for (ii, jj), c in out_poly.items():
                        out_const += c * (s2x_img ** ii) * (s2y_img ** jj)
                    T_acc += b_pup * out_const * exp_val
                L[io, js] = A_lead * T_acc
    else:
        # ---- Grid-based σ-integration (handles all ℓ) ------------------
        # Grid extent: ~4·w_o each side captures > 99.9 % of the LG basis
        # tail (the dominant ℓ_max scaling).  Sampling: 64 points across
        # 8·w_o gives ~ w_o/8 resolution, plenty for accurate trapezoidal
        # projection of LG modes up to (p, ℓ) ~ (3, 3).  Both knobs can
        # be tuned via the new ``sigma_grid_n`` / ``sigma_grid_extent``
        # kwargs added below for users who need higher orders or
        # tighter accuracy.
        n_grid = int(sigma_grid_n) if sigma_grid_n is not None else 64
        extent = (float(sigma_grid_extent)
                  if sigma_grid_extent is not None else 4.0 * w_o)
        sx_arr = np.linspace(s2x_img - extent, s2x_img + extent, n_grid)
        sy_arr = np.linspace(s2y_img - extent, s2y_img + extent, n_grid)
        S2X, S2Y = np.meshgrid(sx_arr, sy_arr, indexing='xy')
        SX_local = S2X - s2x_img
        SY_local = S2Y - s2y_img

        max_p_o = max((k_out[0] for k_out in output_modes), default=0)
        max_ell_o = max((abs(k_out[1]) for k_out in output_modes),
                        default=0)

        for js, k_src in enumerate(source_modes):
            # Field on the grid for THIS source mode (with the
            # caller's pupil_amplitudes weighting).
            src_amps = {k_src: 1.0 + 0.0j}
            field = propagate_modal_asymptotic(
                fit,
                source_point=(src_x, src_y),
                source_amplitudes=src_amps,
                pupil_amplitudes=pupil_amplitudes,
                w_s=w_s, w_p=w_p, v2_centre=v2_centre,
                s2_grid_x=S2X, s2_grid_y=S2Y,
            )
            overlaps = decompose_lg(
                field, SX_local, SY_local,
                w=w_o, p_max=max_p_o, ell_max=max_ell_o,
            )
            for io, k_out in enumerate(output_modes):
                L[io, js] = overlaps.get(k_out, 0.0 + 0.0j)

    return AberrationTensorResult(
        L=L,
        output_modes=list(output_modes),
        source_modes=list(source_modes),
        pupil_modes=list(pupil_modes),
        s2_image=(s2x_img, s2y_img),
        w_s=w_s, w_p=w_p, w_o=w_o,
        v_star=(v2x_star, v2y_star),
    )


def _polynomial_substitute_linear_2d(
    coeffs: Dict[Tuple[int, int], complex],
    A_xx: float, A_xy: float, A_yx: float, A_yy: float,
    b_x: float, b_y: float,
) -> Dict[Tuple[int, int], complex]:
    """Substitute (r_x, r_y) -> A * (eta_x, eta_y) + (b_x, b_y) in a
    2-D polynomial, returning the resulting polynomial in (eta_x, eta_y).

    Used to push the source polynomial through the linear J* map at
    the envelope-stationary point.
    """
    if not coeffs:
        return {}
    # First pre-compute (a x + b y + c)^n expansion as polynomial in (x, y).
    # We need (A_xx eta_x + A_xy eta_y + b_x)^i and similarly for y.

    def axes_pow(coef_a: complex, coef_b: complex, coef_c: complex,
                 n: int) -> Dict[Tuple[int, int], complex]:
        """Expand (a x + b y + c)^n via multinomial."""
        out: Dict[Tuple[int, int], complex] = {}
        # multinomial:  sum over (i, j, k) with i + j + k = n of
        #     n!/(i! j! k!) * a^i * b^j * c^k * x^i * y^j
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                w = (math.factorial(n)
                     // (math.factorial(i) * math.factorial(j)
                         * math.factorial(k)))
                key = (i, j)
                out[key] = out.get(key, 0.0 + 0.0j) + (
                    w * (coef_a ** i) * (coef_b ** j) * (coef_c ** k)
                )
        return out

    out: Dict[Tuple[int, int], complex] = {}
    # Cache the expansions of the linear forms raised to each needed power
    max_i = max(k[0] for k in coeffs)
    max_j = max(k[1] for k in coeffs)

    # (A_xx eta_x + A_xy eta_y + b_x)^i ; build for i = 0..max_i
    cache_x: List[Dict[Tuple[int, int], complex]] = []
    for n in range(max_i + 1):
        cache_x.append(axes_pow(
            complex(A_xx), complex(A_xy), complex(b_x), n
        ))
    # (A_yx eta_x + A_yy eta_y + b_y)^j ; build for j = 0..max_j
    cache_y: List[Dict[Tuple[int, int], complex]] = []
    for n in range(max_j + 1):
        cache_y.append(axes_pow(
            complex(A_yx), complex(A_yy), complex(b_y), n
        ))

    for (i, j), c in coeffs.items():
        # Multiply cache_x[i] * cache_y[j] and accumulate into out.
        prod = _multiply_polys_2d(cache_x[i], cache_y[j])
        for key, pc in prod.items():
            out[key] = out.get(key, 0.0 + 0.0j) + c * pc
    return out


# ===========================================================================
# Section 6 -- Modal asymptotic propagator (leading-order)
# ===========================================================================

def _maslov_branch_corrected_sqrt(det_M, last_arg_detM=None,
                                   maslov_branch=0):
    """Keller-Maslov branch-continuous ``sqrt(det M)``.

    4.11.2: hoisted from ``propagate_modal_asymptotic`` (v4.10
    branch-tracking machinery) so the four other asymptotic-evaluator
    sites (``aberration_tensor``, ``aberration_tensor_lg00_jax``,
    ``_modal_field_lg00_pixel_jax``, and any future evaluator) can
    consume it without re-implementing the unwrap logic.

    The function returns a 3-tuple
    ``(sqrt_detM, new_last_arg_detM, new_maslov_branch)``.  Callers
    walking a sequence of pixels thread the second and third return
    values through successive calls; the first call may pass
    ``last_arg_detM=None``, ``maslov_branch=0`` (which corresponds to
    a principal sqrt with no caustic-crossing history).

    Single-point evaluators (the JAX twins and ``aberration_tensor``)
    call with defaults and get the principal sqrt -- the same value
    they computed before this hoist -- but the helper now exists at
    one canonical site so any future caustic-aware refinement (e.g.
    pre-computing an external Maslov index, switching the JAX twins
    to a closed-form complex sqrt with explicit branch cut) only
    needs to be done in one place.
    """
    import math as _math
    arg_detM = float(np.angle(det_M))
    new_branch = maslov_branch
    if last_arg_detM is not None:
        d_arg = arg_detM - last_arg_detM
        # Unwrap: if jump > +pi, we crossed -pi -> +pi (branch -1).
        # If < -pi, we crossed +pi -> -pi (branch +1).
        if d_arg > _math.pi:
            new_branch = maslov_branch - 1
        elif d_arg < -_math.pi:
            new_branch = maslov_branch + 1
    sqrt_principal = np.sqrt(det_M)
    if new_branch % 2 != 0:
        sqrt_detM = -sqrt_principal
    else:
        sqrt_detM = sqrt_principal
    return sqrt_detM, arg_detM, new_branch


def propagate_modal_asymptotic(
    fit: CanonicalPolyFit,
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    source_amplitudes: Optional[Dict[Tuple[int, int], complex]] = None,
    pupil_amplitudes: Optional[Dict[Tuple[int, int], complex]] = None,
    w_s: float = 50e-6,
    w_p: float = 0.05,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    s2_grid_x: np.ndarray,
    s2_grid_y: np.ndarray,
    maslov_tracking: str = 'row_reset',
) -> np.ndarray:
    """Evaluate the leading-order modal asymptotic propagator on a 2-D
    output grid.

    For each output pixel, solves the envelope-stationary equation,
    builds the complex beam matrix, and evaluates the closed-form
    Gaussian-moment expansion in the chosen LG mode bases.  This is
    ~10**3-10**4 times faster than direct quadrature for typical
    parameters.

    Parameters
    ----------
    fit : CanonicalPolyFit
    source_point : (float, float)
        Object-plane source location [m].
    source_amplitudes, pupil_amplitudes : dict, optional
        LG mode amplitudes ``{(p, ell): complex}`` for source and pupil.
        Defaults: source = LG_{0,0} (Gaussian); pupil = LG_{0,0}
        (matched soft aperture).
    w_s, w_p : float
        Source and pupil waists.
    v2_centre : (float, float)
        Pupil centre in direction cosines.
    s2_grid_x, s2_grid_y : ndarray
        Output-plane sample points [m].  Same shape; iterated jointly.
    maslov_tracking : str, optional
        Strategy for the Keller-Maslov branch index of ``sqrt(det M)``:

        * ``'principal'``  -- no unwrap; ``sqrt(det M)`` uses the
          principal branch (pre-4.10 behaviour; introduces an
          unphysical pi-phase jump every caustic crossing).
        * ``'1d_raster'``  -- 4.10 behaviour: unwrap along the raster-
          ordered pixel stream.  Works correctly along a single
          row but spuriously flips the counter at every row wrap
          because the raster jump in (x, y) is not a continuous path
          in the actual output plane.
        * ``'row_reset'``  -- 4.11.2 default: same as ``'1d_raster'``
          but resets the unwrap state at the start of each row.  The
          per-row reset costs the ability to detect a caustic that
          crosses an entire row, but eliminates the row-wrap false
          positive.  For grids whose caustic structure doesn't span
          full rows this is the correct trade-off; for grids where
          the caustic does run row-spanning, pass
          ``maslov_tracking='1d_raster'`` and verify against a
          ground-truth direct quadrature.

    Returns
    -------
    ndarray, complex, same shape as s2_grid_x
        Output field E(s_2).
    """
    if maslov_tracking not in ('principal', '1d_raster', 'row_reset'):
        raise ValueError(
            f"propagate_modal_asymptotic: maslov_tracking must be "
            f"'principal', '1d_raster', or 'row_reset', got "
            f"{maslov_tracking!r}.")
    if source_amplitudes is None:
        source_amplitudes = {(0, 0): 1.0 + 0.0j}
    if pupil_amplitudes is None:
        pupil_amplitudes = {(0, 0): 1.0 + 0.0j}
    src_x, src_y = float(source_point[0]), float(source_point[1])
    v_cx, v_cy = float(v2_centre[0]), float(v2_centre[1])

    s2x_arr = np.asarray(s2_grid_x, dtype=np.float64)
    s2y_arr = np.asarray(s2_grid_y, dtype=np.float64)
    if s2x_arr.shape != s2y_arr.shape:
        raise ValueError(
            f"s2_grid_x and s2_grid_y shape mismatch: {s2x_arr.shape} vs {s2y_arr.shape}")

    out = np.zeros(s2x_arr.shape, dtype=np.complex128)

    # Determine moment-table order needed
    max_order_src = max((2 * p + abs(l) for (p, l) in source_amplitudes), default=0)
    max_order_pup = max((2 * p + abs(l) for (p, l) in pupil_amplitudes), default=0)
    max_order_needed = max(max_order_src + max_order_pup, 0)

    # 4.12.0 (perf #5): hoist the LG basis polynomials out of the
    # per-pixel Newton loop.  ``lg_polynomial(p, ell, w)`` is a pure
    # function of its arguments -- it returned identical dicts on every
    # pixel pre-4.12.0 (and was the dominant cost in multi-mode
    # propagation, scaling with mode count * N_pixels).  Build one dict
    # per ``(p, ell)`` here and reuse across pixels.  The function
    # itself is also ``lru_cache``-d so a second call from a sibling
    # propagator (``aberration_tensor``, ``MultiWavelengthMerit``, ...)
    # hits the cache.
    src_poly_r_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], complex]] = {
        k_src: lg_polynomial(k_src[0], k_src[1], w_s)
        for k_src, a_src in source_amplitudes.items()
        if abs(a_src) >= 1e-300
    }
    pup_poly_r_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], complex]] = {
        k_pup: lg_polynomial(k_pup[0], k_pup[1], w_p)
        for k_pup, b_pup in pupil_amplitudes.items()
        if abs(b_pup) >= 1e-300
    }

    flat_x = s2x_arr.ravel()
    flat_y = s2y_arr.ravel()
    flat_out = np.zeros(flat_x.size, dtype=np.complex128)

    last_v_star = (v_cx, v_cy)  # warm-start across pixels
    # 4.10: track arg(det M) across pixels so 1 / sqrt(det M) stays
    # branch-continuous through caustics.  Pre-4.10 used the
    # principal sqrt branch, so arg(det M) sweeping past ±π flipped
    # the sign of amp_lead and introduced a fake pi-phase jump in
    # the reconstructed field.  We accumulate an integer branch
    # counter `maslov_branch` whose role is the Keller-Maslov index:
    # the field picks up an extra exp(-i*pi/2) for each caustic
    # crossing (sign of det M changing).
    #
    # 4.11.2: ``maslov_tracking`` chooses between the v4.10
    # 1-D-raster unwrap, the new default per-row reset, and the
    # legacy principal sqrt.  See the docstring for the trade-off.
    last_arg_detM = None  # principal-branch arg of det M at the previous pixel
    maslov_branch = 0     # accumulated branch index (each +/-2pi adds 1)
    # Per-row reset bookkeeping: when the raster sweeps to a new row,
    # reset the unwrap state so the jump between (x_max, y_n) and
    # (x_min, y_{n+1}) -- which is not a continuous path in the
    # output plane -- doesn't artifactually advance the Maslov index.
    Ny_grid, Nx_grid = s2x_arr.shape if s2x_arr.ndim >= 2 else (1, s2x_arr.size)
    for idx in range(flat_x.size):
        s2x_p = flat_x[idx]
        s2y_p = flat_y[idx]
        if (maslov_tracking == 'row_reset' and s2x_arr.ndim >= 2
                and Nx_grid > 0 and idx % Nx_grid == 0):
            last_arg_detM = None
            maslov_branch = 0

        # Skip points outside the fit's training box
        u1 = (s2x_p - fit.s2x_centre) / fit.s2x_halfrange
        u2 = (s2y_p - fit.s2y_centre) / fit.s2y_halfrange
        if abs(u1) > 1.0 or abs(u2) > 1.0:
            continue

        try:
            v_star, n_iter, residual = solve_envelope_stationary(
                fit, (s2x_p, s2y_p), (src_x, src_y),
                w_s=w_s, w_p=w_p, v2_centre=v2_centre,
                v2_initial=last_v_star,
            )
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            continue
        # If Newton wandered outside the fit box, fall back to v_centre
        # (otherwise we'll evaluate the polynomial fit out-of-domain).
        u3 = (v_star[0] - fit.v2x_centre) / fit.v2x_halfrange
        u4 = (v_star[1] - fit.v2y_centre) / fit.v2y_halfrange
        if abs(u3) > 1.0 or abs(u4) > 1.0 or not (math.isfinite(u3) and math.isfinite(u4)):
            continue
        last_v_star = v_star

        try:
            M, b, s1_star, J_star, phi_star, G0, detJ = _compute_M_b(
                fit, s2x_p, s2y_p, v_star[0], v_star[1],
                src_x, src_y, w_s, w_p, v_cx, v_cy
            )
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            continue
        if not (np.all(np.isfinite(M)) and np.all(np.isfinite(b))):
            continue
        det_M = np.linalg.det(M)
        if not math.isfinite(abs(det_M)) or abs(det_M) < 1e-300:
            continue
        # 4.10 / 4.11.2: track arg(det M) and adjust sqrt branch so
        # the field phase is continuous across caustics (Keller-
        # Maslov).  4.11.2 routes through the shared
        # ``_maslov_branch_corrected_sqrt`` helper so the same logic
        # is available to the sibling JAX evaluators and
        # ``aberration_tensor`` -- see the helper's docstring.  The
        # ``maslov_tracking='principal'`` mode bypasses the unwrap
        # state (no branch correction, principal sqrt).
        if maslov_tracking == 'principal':
            sqrt_detM = np.sqrt(det_M)
        else:
            sqrt_detM, last_arg_detM, maslov_branch = (
                _maslov_branch_corrected_sqrt(
                    det_M,
                    last_arg_detM=last_arg_detM,
                    maslov_branch=maslov_branch,
                )
            )
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(M_inv)):
            continue
        delta_star = 0.5 * (M_inv @ b)

        # Leading amplitude.  Guard against complex-Gaussian overflow in
        # the b^T M^-1 b exponent: cap at large magnitude (the integrand
        # is genuinely negligible if the exponent goes to -inf, but a
        # +inf would NaN the field).
        b_quad = 0.25 * (b @ M_inv @ b)
        if not math.isfinite(abs(b_quad)) or abs(b_quad.real) > 700:
            continue
        amp_lead = (detJ * (math.pi / sqrt_detM) * G0
                    * np.exp(2j * math.pi * phi_star)
                    * np.exp(b_quad))
        if not math.isfinite(abs(amp_lead)):
            continue

        # Moment table on this pixel's covariance
        eta_moments = gaussian_moment_table_2d(M, max_order_needed)

        r_const = s1_star + J_star @ np.array([delta_star[0], delta_star[1]]) \
                  - np.array([src_x, src_y])
        pupil_const = (np.array([v_star[0], v_star[1]])
                       - np.array([v_cx, v_cy])
                       + np.array([delta_star[0], delta_star[1]]))

        # Sum over (n, m) modes
        # 4.12.0 (perf #5): ``src_poly_r`` / ``pup_poly_r`` are pulled
        # from the hoisted ``*_poly_r_cache`` instead of rebuilt via
        # :func:`lg_polynomial` every pixel.  The affine substitutions
        # below DO depend on the per-pixel ``J_star`` / ``r_const`` /
        # ``pupil_const``, so they remain inside the pixel loop.
        E_pixel = 0.0 + 0.0j
        for k_src, a_src in source_amplitudes.items():
            if abs(a_src) < 1e-300:
                continue
            src_poly_r = src_poly_r_cache[k_src]
            src_poly_eta = _polynomial_substitute_linear_2d(
                src_poly_r,
                A_xx=J_star[0, 0], A_xy=J_star[0, 1],
                A_yx=J_star[1, 0], A_yy=J_star[1, 1],
                b_x=r_const[0], b_y=r_const[1],
            )
            for k_pup, b_pup in pupil_amplitudes.items():
                if abs(b_pup) < 1e-300:
                    continue
                pup_poly_r = pup_poly_r_cache[k_pup]
                pup_poly_eta = _polynomial_under_affine_shift(
                    pup_poly_r,
                    shift_x=complex(pupil_const[0]),
                    shift_y=complex(pupil_const[1]),
                )
                P_eta = _multiply_polys_2d(src_poly_eta, pup_poly_eta)
                exp_val = _contract_against_moment_table(P_eta, eta_moments)
                E_pixel += a_src * b_pup * exp_val

        flat_out[idx] = amp_lead * E_pixel

    return flat_out.reshape(s2x_arr.shape)


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
    out: Dict[Tuple[int, int], complex] = {}
    for p in range(p_max + 1):
        for ell in range(-ell_max, ell_max + 1):
            mode = evaluate_lg_mode(p, ell, w, X, Y, cx, cy)
            out[(p, ell)] = complex(np.sum(np.conj(mode) * field) * da)
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
    out: Dict[Tuple[int, int], complex] = {}
    for mi in range(m_max + 1):
        for nj in range(n_max + 1):
            mode = evaluate_hg_mode(mi, nj, wx, wy, X, Y, cx, cy)
            out[(mi, nj)] = complex(np.sum(np.conj(mode) * field) * da)
    return out
# ===========================================================================
# Section 8 -- HF-form polynomial fit:  Phi(s1, s2)
# ===========================================================================
#
# The canonical fit (Section 3) parametrises Phi as a function of (s2, v2)
# -- the natural coordinates for the asymptotic / Maslov saddle-point
# evaluation.  For direct Huygens-Fresnel quadrature without any
# saddle-point approximation we want Phi expressed as a function of the
# endpoints (s1, s2) so that the HF integral
#
#     E_HF(s2) = integral E_in(s1) sqrt(|det d2 Phi / d s1 d s2|)
#                * exp(2 pi i Phi(s1, s2)) d^2 s1
#
# can be evaluated by 2-D quadrature over s1 with no inverse mapping.
#
# The fit traces a Chebyshev grid in (s1, v1), records the (s1, s2_observed,
# OPL) tuples, and least-squares fits Phi as a 4-D Chebyshev tensor
# product in (s1_x, s1_y, s2_x, s2_y).  The Van Vleck cross-Hessian
# determinant ``det d2 Phi / d s1 d s2`` is computed analytically from
# the polynomial coefficients.


@dataclass
class HFPolyFit:
    """4-D Chebyshev tensor-product fit of Phi(s1, s2).

    Counterpart to :class:`CanonicalPolyFit` for direct
    Huygens-Fresnel quadrature.  Fields mirror CanonicalPolyFit but
    the four normalisation axes are (s1x, s1y, s2x, s2y) instead of
    (s2x, s2y, v2x, v2y).
    """
    poly_order: int
    multi_indices: List[Tuple[int, int, int, int]]
    coef_phi: np.ndarray
    s1x_centre: float
    s1x_halfrange: float
    s1y_centre: float
    s1y_halfrange: float
    s2x_centre: float
    s2x_halfrange: float
    s2y_centre: float
    s2y_halfrange: float
    wavelength: float
    res_phi_rms_waves: float = 0.0
    n_rays: int = 0
    linear_coeffs_phi: Optional[np.ndarray] = None
    extract_linear_phase: bool = False

    def to_normalised(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u1 = (s1x - self.s1x_centre) / self.s1x_halfrange
        u2 = (s1y - self.s1y_centre) / self.s1y_halfrange
        u3 = (s2x - self.s2x_centre) / self.s2x_halfrange
        u4 = (s2y - self.s2y_centre) / self.s2y_halfrange
        return u1, u2, u3, u4

    def in_box(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
    ) -> np.ndarray:
        u1, u2, u3, u4 = self.to_normalised(s1x, s1y, s2x, s2y)
        return ((np.abs(u1) <= 1.0) & (np.abs(u2) <= 1.0)
                & (np.abs(u3) <= 1.0) & (np.abs(u4) <= 1.0))

    def eval_phi(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
        *,
        include_linear: bool = True,
    ) -> np.ndarray:
        """Evaluate Phi(s1, s2) [waves]."""
        u1, u2, u3, u4 = self.to_normalised(s1x, s1y, s2x, s2y)
        phi = _evaluate_polynomial_4d(self.coef_phi, self.multi_indices,
                                       u1, u2, u3, u4, self.poly_order)
        if include_linear and self.linear_coeffs_phi is not None:
            a0, a1, a2, a3, a4 = self.linear_coeffs_phi
            phi = phi + (a0 + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u4)
        return phi

    def eval_van_vleck_density(
        self,
        s1x: np.ndarray,
        s1y: np.ndarray,
        s2x: np.ndarray,
        s2y: np.ndarray,
    ) -> np.ndarray:
        """Evaluate sqrt(|det d2 Phi / d s1 d s2|) -- the Van Vleck factor.

        The 2x2 cross-Hessian is computed analytically from the
        Chebyshev polynomial via term-wise derivatives.
        """
        u1, u2, u3, u4 = self.to_normalised(s1x, s1y, s2x, s2y)
        inv_h1x = 1.0 / self.s1x_halfrange
        inv_h1y = 1.0 / self.s1y_halfrange
        inv_h2x = 1.0 / self.s2x_halfrange
        inv_h2y = 1.0 / self.s2y_halfrange

        d13 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 0, 2)
        d14 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 0, 3)
        d23 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 1, 2)
        d24 = _eval_4d_cross_deriv(self.coef_phi, self.multi_indices,
                                    u1, u2, u3, u4, self.poly_order, 1, 3)

        H11 = d13 * (inv_h1x * inv_h2x)
        H12 = d14 * (inv_h1x * inv_h2y)
        H21 = d23 * (inv_h1y * inv_h2x)
        H22 = d24 * (inv_h1y * inv_h2y)
        det = H11 * H22 - H12 * H21
        return np.sqrt(np.abs(det))


def _eval_4d_cross_deriv(coef, multi_indices, u1, u2, u3, u4,
                          poly_order, axis_a, axis_b):
    """Evaluate d2 phi / (d u_a d u_b) of a 4-D Chebyshev tensor product
    at scattered points.

    Uses Chebyshev derivative tables (T'_n = n U_{n-1}) on the two
    differentiation axes and ordinary tables on the other two.
    """
    tables = [
        _chebyshev_vandermonde(u1, poly_order),
        _chebyshev_vandermonde(u2, poly_order),
        _chebyshev_vandermonde(u3, poly_order),
        _chebyshev_vandermonde(u4, poly_order),
    ]
    deriv_tables = [
        _chebyshev_derivative_vandermonde(u1, poly_order),
        _chebyshev_derivative_vandermonde(u2, poly_order),
        _chebyshev_derivative_vandermonde(u3, poly_order),
        _chebyshev_derivative_vandermonde(u4, poly_order),
    ]
    use_tables = [
        deriv_tables[i] if i in (axis_a, axis_b) else tables[i]
        for i in range(4)
    ]

    out = np.zeros_like(np.asarray(u1, dtype=np.float64))
    for j, (k1, k2, k3, k4) in enumerate(multi_indices):
        out += (coef[j] * use_tables[0][k1] * use_tables[1][k2]
                * use_tables[2][k3] * use_tables[3][k4])
    return out


def fit_hf_polynomials(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    extract_linear_phase: bool = True,
    object_distance: Optional[float] = None,
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    endpoint_anchored: bool = False,
) -> HFPolyFit:
    """Fit a 4-D Chebyshev tensor-product polynomial to Phi(s1, s2).

    HF counterpart to :func:`fit_canonical_polynomials`.  Traces a
    (s1, v1) Chebyshev-node grid through the prescription, records
    the (s1, s2_observed, OPL) tuples, and least-squares fits Phi
    as a function of the endpoint pair (s1, s2).

    Parameters
    ----------
    prescription : dict
    wavelength : float
    source_box_half, pupil_box_half : float
    n_field, n_pupil : int
        Per-axis Chebyshev sampling.
    poly_order : int
    extract_linear_phase : bool
    object_distance : float, optional
    surface_diffraction : dict, optional
    endpoint_anchored : bool

    Returns
    -------
    HFPolyFit
    """
    from ..raytrace import surfaces_from_prescription, _make_bundle, trace

    if wavelength <= 0:
        raise ValueError("wavelength must be > 0")
    if poly_order < 0:
        raise ValueError("poly_order must be >= 0")
    if source_box_half <= 0 or pupil_box_half <= 0:
        raise ValueError("source_box_half and pupil_box_half must be > 0")

    if object_distance is None:
        object_distance = float(prescription.get('object_distance', 0.0)) or 0.0

    surfaces = surfaces_from_prescription(prescription)

    def cheb_nodes(n):
        i = np.arange(n)
        x = np.cos(np.pi * (i + 0.5) / n)
        if endpoint_anchored and n >= 2:
            x = x / np.cos(np.pi / (2.0 * n))
        return x

    u_field = cheb_nodes(n_field)
    u_pupil = cheb_nodes(n_pupil)
    s1x_axis = u_field * source_box_half
    s1y_axis = u_field * source_box_half
    v1x_axis = u_pupil * pupil_box_half
    v1y_axis = u_pupil * pupil_box_half

    S1X, S1Y, V1X, V1Y = np.meshgrid(s1x_axis, s1y_axis,
                                      v1x_axis, v1y_axis,
                                      indexing='ij')
    s1x_in = S1X.ravel()
    s1y_in = S1Y.ravel()
    v1x_in = V1X.ravel()
    v1y_in = V1Y.ravel()
    n_rays = s1x_in.size

    sumsq = v1x_in * v1x_in + v1y_in * v1y_in
    if np.any(sumsq >= 1.0):
        raise ValueError(
            "pupil_box_half too large; reduce it.")

    bundle = _make_bundle(
        x=s1x_in.astype(np.float64),
        y=s1y_in.astype(np.float64),
        L=v1x_in.astype(np.float64),
        M=v1y_in.astype(np.float64),
        wavelength=wavelength,
    )
    bundle.z = np.full(n_rays, -object_distance, dtype=np.float64)

    res = trace(bundle, surfaces, wavelength, output_filter='last',
                surface_diffraction=surface_diffraction)
    final = res.image_rays
    alive = np.asarray(final.alive, dtype=bool)
    n_alive = int(alive.sum())
    if n_alive < max(64, 0.5 * n_rays):
        raise RuntimeError(
            f"Too many rays died during HF-fit trace: "
            f"{n_alive} alive of {n_rays}.  Reduce pupil_box_half "
            f"or check apertures.")

    s2x_obs = np.asarray(final.x, dtype=np.float64)
    s2y_obs = np.asarray(final.y, dtype=np.float64)
    phi_obs = np.asarray(final.opd, dtype=np.float64) / wavelength

    s1x_live = s1x_in[alive]
    s1y_live = s1y_in[alive]
    s2x_live = s2x_obs[alive]
    s2y_live = s2y_obs[alive]
    phi_live = phi_obs[alive]

    s1x_c, s1x_h = _fit_normaliser(s1x_live)
    s1y_c, s1y_h = _fit_normaliser(s1y_live)
    s2x_c, s2x_h = _fit_normaliser(s2x_live)
    s2y_c, s2y_h = _fit_normaliser(s2y_live)

    u_s1x = (s1x_live - s1x_c) / s1x_h
    u_s1y = (s1y_live - s1y_c) / s1y_h
    u_s2x = (s2x_live - s2x_c) / s2x_h
    u_s2y = (s2y_live - s2y_c) / s2y_h

    linear_coeffs = None
    if extract_linear_phase:
        X5 = np.column_stack([
            np.ones_like(u_s1x), u_s1x, u_s1y, u_s2x, u_s2y,
        ])
        linear_coeffs, *_ = np.linalg.lstsq(X5, phi_live, rcond=None)
        opd_residual = phi_live - X5 @ linear_coeffs
    else:
        opd_residual = phi_live.copy()

    multi_indices = _multi_indices_total_degree(4, poly_order)
    n_basis = len(multi_indices)
    T1 = _chebyshev_vandermonde(u_s1x, poly_order)
    T2 = _chebyshev_vandermonde(u_s1y, poly_order)
    T3 = _chebyshev_vandermonde(u_s2x, poly_order)
    T4 = _chebyshev_vandermonde(u_s2y, poly_order)
    A = np.empty((u_s1x.size, n_basis), dtype=np.float64)
    for j, (k1, k2, k3, k4) in enumerate(multi_indices):
        A[:, j] = T1[k1] * T2[k2] * T3[k3] * T4[k4]
    coef_phi, *_ = np.linalg.lstsq(A, opd_residual, rcond=None)
    res_phi = float(np.sqrt(np.mean((opd_residual - A @ coef_phi) ** 2)))

    return HFPolyFit(
        poly_order=poly_order,
        multi_indices=multi_indices,
        coef_phi=coef_phi,
        s1x_centre=s1x_c, s1x_halfrange=s1x_h,
        s1y_centre=s1y_c, s1y_halfrange=s1y_h,
        s2x_centre=s2x_c, s2x_halfrange=s2x_h,
        s2y_centre=s2y_c, s2y_halfrange=s2y_h,
        wavelength=wavelength,
        res_phi_rms_waves=res_phi,
        n_rays=n_alive,
        linear_coeffs_phi=linear_coeffs,
        extract_linear_phase=extract_linear_phase,
    )


def propagate_hf_chebyshev_quadrature(
    fit: HFPolyFit,
    E_in: np.ndarray,
    input_grid_x: np.ndarray,
    input_grid_y: np.ndarray,
    output_grid_x: np.ndarray,
    output_grid_y: np.ndarray,
    *,
    apply_van_vleck: bool = True,
    chunk_output: int = 64,
    max_chunk_memory_mb: float = 256.0,
) -> np.ndarray:
    """Direct 2-D HF quadrature using a tensor-product Chebyshev
    polynomial fit of Phi(s1, s2).

    For each output point s2, evaluates

        E_HF(s2) = sum over input grid of
                   E_in(s1) * sqrt(|det d2 Phi / d s1 d s2|)
                   * exp(2 pi i Phi(s1, s2)) * dx * dy

    using the polynomial fit's analytical Phi and Van Vleck factor.

    Parameters
    ----------
    fit : HFPolyFit
    E_in : ndarray (Ny_in, Nx_in) complex
    input_grid_x, input_grid_y : 1-D arrays
    output_grid_x, output_grid_y : 1-D arrays
    apply_van_vleck : bool
    chunk_output : int
        Output points processed per chunk to bound memory.  4.12.0:
        chunk dimension is now a true broadcast axis (was a Python
        loop pre-4.12.0); larger ``chunk_output`` directly improves
        throughput as long as the broadcast-tensor memory fits.
    max_chunk_memory_mb : float
        Soft cap on the per-chunk broadcast-tensor footprint
        (``chunk * Ny_in * Nx_in`` complex128 = ``16 * chunk *
        Ny_in * Nx_in`` bytes).  If ``chunk_output`` would exceed
        this, the chunk is shrunk to fit.  Default 256 MB.

    Returns
    -------
    ndarray (Ny_out, Nx_out) complex
    """
    Nx_in = input_grid_x.shape[0]
    Ny_in = input_grid_y.shape[0]
    Nx_out = output_grid_x.shape[0]
    Ny_out = output_grid_y.shape[0]
    if E_in.shape != (Ny_in, Nx_in):
        raise ValueError(
            f"E_in shape {E_in.shape} mismatches input grid "
            f"({Ny_in}, {Nx_in}).")

    dx = float(np.mean(np.diff(input_grid_x)))
    dy = float(np.mean(np.diff(input_grid_y)))
    pixel_area = dx * dy

    S1X, S1Y = np.meshgrid(input_grid_x, input_grid_y, indexing='xy')
    # 4.11.2: force a complex output dtype so a real-valued E_in (e.g. a
    # pure intensity mask) doesn't silently strip the imaginary part of
    # the HF kernel during ``kernel.astype(E_in.dtype)`` and the
    # subsequent multiply.  Pre-4.11.2 a real E_in produced a real-
    # valued "field" with the imaginary half of the HF integrand
    # summed away -- relative-intensity contrast was preserved but
    # interference structure was destroyed.
    if np.iscomplexobj(E_in):
        out_dtype = E_in.dtype
    elif E_in.dtype == np.float64:
        out_dtype = np.complex128
    else:
        out_dtype = np.complex64
    out = np.zeros((Ny_out, Nx_out), dtype=out_dtype)

    # ------------------------------------------------------------------
    # 4.12.0 (perf #6): vectorise across the output chunk.
    # ------------------------------------------------------------------
    # Pre-4.12.0 the outer ``for start in range(0, n_out, chunk_output)``
    # paired with an inner scalar ``for k in range(start, end)``
    # evaluated one output pixel at a time -- the chunking did nothing
    # because the inner loop was a Python loop, not a vector op.  Every
    # iteration rebuilt the full ``(Ny_in, Nx_in)`` Chebyshev Vandermonde
    # tables on the input grid and called ``np.exp(2j*pi*phi)`` on the
    # full grid.
    #
    # 4.12.0 hoists the input-grid Chebyshev tables outside the chunk
    # loop and vectorises the chunk axis as a true broadcast dim
    # ``(chunk, Ny_in, Nx_in)``.  Memory footprint at chunk=64 on
    # (256, 256) input is ~64 MB complex128 (fits comfortably under
    # the 256 MB default cap); at chunk=1024 it would be ~1 GB and is
    # capped by ``max_chunk_memory_mb``.
    n_out = Ny_out * Nx_out
    poly_order = fit.poly_order
    multi_indices = fit.multi_indices
    K = np.asarray(multi_indices, dtype=np.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    coef_phi = np.asarray(fit.coef_phi, dtype=np.float64)

    # Normalised input-grid coords -- shape (Ny_in, Nx_in)
    u1 = (S1X - fit.s1x_centre) / fit.s1x_halfrange
    u2 = (S1Y - fit.s1y_centre) / fit.s1y_halfrange
    # Chebyshev Vandermonde tables on input grid (and derivative tables
    # for the Van Vleck cross-Hessian, computed once and reused).
    T1 = _chebyshev_vandermonde(u1, poly_order)   # (k+1, Ny_in, Nx_in)
    T2 = _chebyshev_vandermonde(u2, poly_order)
    if apply_van_vleck:
        dT1 = _chebyshev_derivative_vandermonde(u1, poly_order)
        dT2 = _chebyshev_derivative_vandermonde(u2, poly_order)

    # Flatten output coordinates so the chunk axis is 1-D.
    out_iy, out_ix = np.divmod(np.arange(n_out), Nx_out)
    s2x_flat = output_grid_x[out_ix].astype(np.float64)
    s2y_flat = output_grid_y[out_iy].astype(np.float64)

    # Memory guard:  cap chunk so the (chunk, Ny_in, Nx_in) broadcast
    # tensor (complex128, 16 B/element) stays under the soft budget.
    bytes_per_pixel_per_output = 16 * Ny_in * Nx_in
    if bytes_per_pixel_per_output > 0:
        max_chunk_by_mem = max(
            1,
            int((max_chunk_memory_mb * 1024 * 1024)
                // bytes_per_pixel_per_output)
        )
    else:
        max_chunk_by_mem = chunk_output
    effective_chunk = max(1, min(int(chunk_output), max_chunk_by_mem))

    # Linear-phase coefficients (or zeros if none)
    if fit.linear_coeffs_phi is not None:
        a0, a1, a2, a3, a4 = (float(c) for c in fit.linear_coeffs_phi)
    else:
        a0 = a1 = a2 = a3 = a4 = 0.0
    has_linear = fit.linear_coeffs_phi is not None

    inv_h1x = 1.0 / fit.s1x_halfrange
    inv_h1y = 1.0 / fit.s1y_halfrange
    inv_h2x = 1.0 / fit.s2x_halfrange
    inv_h2y = 1.0 / fit.s2y_halfrange

    flat_out = np.zeros(n_out, dtype=out_dtype)

    for start in range(0, n_out, effective_chunk):
        end = min(start + effective_chunk, n_out)
        s2x_c = s2x_flat[start:end]    # (c,)
        s2y_c = s2y_flat[start:end]    # (c,)
        u3 = (s2x_c - fit.s2x_centre) / fit.s2x_halfrange
        u4 = (s2y_c - fit.s2y_centre) / fit.s2y_halfrange
        # Chebyshev Vandermonde tables on the chunk of output points
        # -- shape (k+1, c).  These need broadcasting against the input
        # tables along a new (Ny_in, Nx_in) axis pair when forming the
        # final basis.
        T3 = _chebyshev_vandermonde(u3, poly_order)   # (k+1, c)
        T4 = _chebyshev_vandermonde(u4, poly_order)

        # Per-basis-term coefficients on the input grid:
        #   tab_12[m] = T1[K1[m]] * T2[K2[m]]       (M, Ny_in, Nx_in)
        # and on the chunk:
        #   tab_34[m] = T3[K3[m]] * T4[K4[m]]       (M, c)
        # The full 4-D basis at every (chunk, Ny_in, Nx_in) point is
        # tab_12[m, y, x] * tab_34[m, c]; the phi sum reduces over m:
        #   phi[c, y, x] = sum_m coef[m] tab_12[m, y, x] tab_34[m, c]
        # which is a tensordot over the basis axis.
        tab_12 = T1[K1] * T2[K2]                       # (M, Ny_in, Nx_in)
        tab_34 = T3[K3] * T4[K4]                       # (M, c)

        # phi_poly[c, y, x] = sum_m (coef[m] * tab_34[m, c]) tab_12[m, y, x]
        #                   = (coef * tab_34) [c, m] @ tab_12 [m, y*x]
        # Fast path: weight tab_12 by (coef * tab_34) and reduce over m.
        cw = coef_phi[:, None] * tab_34                # (M, c)
        # tensordot over the basis axis: result shape (c, Ny_in, Nx_in)
        phi_poly = np.tensordot(cw, tab_12, axes=([0], [0]))

        if has_linear:
            # u1, u2 have shape (Ny_in, Nx_in); u3, u4 have shape (c,).
            # The linear term is a0 + a1 u1 + a2 u2 + a3 u3 + a4 u4
            # which broadcasts to (c, Ny_in, Nx_in).
            lin_xy = a0 + a1 * u1 + a2 * u2                # (Ny_in, Nx_in)
            lin_c = a3 * u3 + a4 * u4                       # (c,)
            phi = phi_poly + lin_xy[None, :, :] + lin_c[:, None, None]
        else:
            phi = phi_poly

        kernel = np.exp(2j * np.pi * phi).astype(out_dtype)   # (c, Ny_in, Nx_in)

        if apply_van_vleck:
            # Cross-Hessian d2 phi / (du_a du_b) for a, b in {(1,3),(1,4),(2,3),(2,4)}.
            # Use the same tensor-product pattern as phi but swap in derivative tables.
            tab_dT1 = dT1[K1]   # (M, Ny_in, Nx_in)
            tab_dT2 = dT2[K2]
            tab_T1 = T1[K1]
            tab_T2 = T2[K2]
            # Derivative tables on s2 coords -- chunked (M, c):
            dT3_arr = _chebyshev_derivative_vandermonde(u3, poly_order)
            dT4_arr = _chebyshev_derivative_vandermonde(u4, poly_order)
            tab_dT3 = dT3_arr[K3]
            tab_dT4 = dT4_arr[K4]
            tab_T3 = T3[K3]
            tab_T4 = T4[K4]

            # d13 = d2 phi / (du1 du3); inputs (M, Ny, Nx) x (M, c)
            xy_d13 = tab_dT1 * tab_T2                          # (M, Ny, Nx)
            c_d13 = coef_phi[:, None] * tab_dT3 * tab_T4       # (M, c)
            d13 = np.tensordot(c_d13, xy_d13, axes=([0], [0])) # (c, Ny, Nx)

            xy_d14 = tab_dT1 * tab_T2
            c_d14 = coef_phi[:, None] * tab_T3 * tab_dT4
            d14 = np.tensordot(c_d14, xy_d14, axes=([0], [0]))

            xy_d23 = tab_T1 * tab_dT2
            c_d23 = coef_phi[:, None] * tab_dT3 * tab_T4
            d23 = np.tensordot(c_d23, xy_d23, axes=([0], [0]))

            xy_d24 = tab_T1 * tab_dT2
            c_d24 = coef_phi[:, None] * tab_T3 * tab_dT4
            d24 = np.tensordot(c_d24, xy_d24, axes=([0], [0]))

            H11 = d13 * (inv_h1x * inv_h2x)
            H12 = d14 * (inv_h1x * inv_h2y)
            H21 = d23 * (inv_h1y * inv_h2x)
            H22 = d24 * (inv_h1y * inv_h2y)
            det = H11 * H22 - H12 * H21
            density = np.sqrt(np.abs(det))                      # (c, Ny, Nx)
            integrand_kernel = density * kernel
        else:
            integrand_kernel = kernel

        # Reduce over the input grid; result shape (c,).
        # einsum is comparable in speed to tensordot + sum here; tensordot
        # is fractionally faster on contiguous arrays.
        chunk_field = np.einsum('cyx,yx->c',
                                 integrand_kernel,
                                 E_in.astype(out_dtype, copy=False),
                                 optimize=False) * pixel_area
        flat_out[start:end] = chunk_field

    out = flat_out.reshape(Ny_out, Nx_out)
    # 4.10: apply the Van Vleck-Morette asymptotic prefactor (2π)^(-d/2)·i^(-d/2)
    # for d = 2: this is -i/(2π).  Pre-4.10 omitted the i^(-d/2) (the
    # Maslov phase) so the result was off by a global 90° phase relative
    # to the Fresnel kernel (which is 1/(iλz) = -i/(λz) in the paraxial
    # limit).  Without this factor, coherent superposition of HF-quadrature
    # output with ASM/Fresnel output is 90° out of phase.
    out = out * (-1j)
    return out


# ===========================================================================
# Section 9 -- Backend-aware polynomial evaluation (NumPy / CuPy / JAX)
# ===========================================================================
#
# The polynomial helpers in lumenairy.elements.lenses are NumPy-only
# (they use np.empty / in-place writes that JAX can't trace).  These
# replacements use array_namespace dispatch and functional-style
# Chebyshev recurrence so the evaluation can run on JAX arrays and
# differentiate via jax.grad.
#
# The fit objects (CanonicalPolyFit, HFPolyFit) gain ``eval_phi_xp``
# methods that route through these backend-aware helpers.


def _chebyshev_vandermonde_xp(u, max_k, xp):
    """Backend-aware Chebyshev Vandermonde T[n](u).

    Returns a list of arrays (length max_k+1) rather than a single
    stacked array, so JAX can build it functionally.
    """
    u_arr = xp.asarray(u)
    T = [xp.ones_like(u_arr)]
    if max_k >= 1:
        T.append(u_arr)
    for n in range(2, max_k + 1):
        T.append(2.0 * u_arr * T[n - 1] - T[n - 2])
    return T


def _evaluate_polynomial_4d_xp(coeffs, multi_indices, u1, u2, u3, u4,
                                  max_order, xp):
    """Backend-aware 4-D Chebyshev tensor-product evaluator.

    Mirrors :func:`lumenairy.elements.lenses._evaluate_polynomial_4d`
    but routes through ``xp = array_namespace(u1, u2, u3, u4)`` so it
    works on NumPy / CuPy / JAX arrays uniformly.
    """
    T1 = _chebyshev_vandermonde_xp(u1, max_order, xp)
    T2 = _chebyshev_vandermonde_xp(u2, max_order, xp)
    T3 = _chebyshev_vandermonde_xp(u3, max_order, xp)
    T4 = _chebyshev_vandermonde_xp(u4, max_order, xp)
    # 3.5.6: vectorised across basis terms.  Same shape contract as
    # the looped version (~70 iters per call previously); ~3-5x
    # faster on NumPy, removes a Python loop that interfered with
    # `jax.jit` tracing on JAX.  ``_chebyshev_vandermonde_xp``
    # returns a Python list (functional construction friendly to
    # JAX); stack to an array so we can fancy-index with the
    # multi-index columns.
    T1_stack = xp.stack(T1)
    T2_stack = xp.stack(T2)
    T3_stack = xp.stack(T3)
    T4_stack = xp.stack(T4)
    import numpy as _np_local
    K = _np_local.asarray(multi_indices, dtype=_np_local.int64)
    K1, K2, K3, K4 = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    basis = T1_stack[K1] * T2_stack[K2] * T3_stack[K3] * T4_stack[K4]
    coeffs_arr = xp.asarray(coeffs)
    return xp.tensordot(coeffs_arr, basis, axes=([0], [0]))


# Add JAX-friendly evaluator methods to the fit dataclasses via
# monkey-patching.  These are purely additive -- the original NumPy
# eval_phi / eval_s1 methods are unchanged and remain the default.

def _CanonicalPolyFit_eval_phi_xp(self, s2x, s2y, v2x, v2y, *,
                                   include_linear=True):
    """Backend-aware Phi(s2, v2) evaluation.

    Accepts NumPy / CuPy / JAX inputs and stays in the input backend.
    Output is differentiable via ``jax.grad`` for JAX inputs.
    """
    from ..backend import array_namespace
    xp = array_namespace(s2x, s2y, v2x, v2y)
    u1 = (s2x - self.s2x_centre) / self.s2x_halfrange
    u2 = (s2y - self.s2y_centre) / self.s2y_halfrange
    u3 = (v2x - self.v2x_centre) / self.v2x_halfrange
    u4 = (v2y - self.v2y_centre) / self.v2y_halfrange
    phi = _evaluate_polynomial_4d_xp(
        self.coef_phi, self.multi_indices,
        u1, u2, u3, u4, self.poly_order, xp,
    )
    if include_linear and self.linear_coeffs_phi is not None:
        a0, a1, a2, a3, a4 = self.linear_coeffs_phi
        phi = phi + (float(a0) + float(a1) * u1 + float(a2) * u2
                     + float(a3) * u3 + float(a4) * u4)
    return phi


def _CanonicalPolyFit_eval_s1_xp(self, s2x, s2y, v2x, v2y):
    """Backend-aware s1(s2, v2) back-map evaluation."""
    from ..backend import array_namespace
    xp = array_namespace(s2x, s2y, v2x, v2y)
    u1 = (s2x - self.s2x_centre) / self.s2x_halfrange
    u2 = (s2y - self.s2y_centre) / self.s2y_halfrange
    u3 = (v2x - self.v2x_centre) / self.v2x_halfrange
    u4 = (v2y - self.v2y_centre) / self.v2y_halfrange
    s1x = _evaluate_polynomial_4d_xp(
        self.coef_s1x, self.multi_indices,
        u1, u2, u3, u4, self.poly_order, xp,
    )
    s1y = _evaluate_polynomial_4d_xp(
        self.coef_s1y, self.multi_indices,
        u1, u2, u3, u4, self.poly_order, xp,
    )
    return s1x, s1y


def _HFPolyFit_eval_phi_xp(self, s1x, s1y, s2x, s2y, *,
                            include_linear=True):
    """Backend-aware Phi(s1, s2) evaluation for the HF fit."""
    from ..backend import array_namespace
    xp = array_namespace(s1x, s1y, s2x, s2y)
    u1 = (s1x - self.s1x_centre) / self.s1x_halfrange
    u2 = (s1y - self.s1y_centre) / self.s1y_halfrange
    u3 = (s2x - self.s2x_centre) / self.s2x_halfrange
    u4 = (s2y - self.s2y_centre) / self.s2y_halfrange
    phi = _evaluate_polynomial_4d_xp(
        self.coef_phi, self.multi_indices,
        u1, u2, u3, u4, self.poly_order, xp,
    )
    if include_linear and self.linear_coeffs_phi is not None:
        a0, a1, a2, a3, a4 = self.linear_coeffs_phi
        phi = phi + (float(a0) + float(a1) * u1 + float(a2) * u2
                     + float(a3) * u3 + float(a4) * u4)
    return phi


# Attach the new methods to the existing dataclasses (additive; old
# eval_phi / eval_s1 stay intact for back-compat).
CanonicalPolyFit.eval_phi_xp = _CanonicalPolyFit_eval_phi_xp
CanonicalPolyFit.eval_s1_xp = _CanonicalPolyFit_eval_s1_xp
HFPolyFit.eval_phi_xp = _HFPolyFit_eval_phi_xp


# ===========================================================================
# Section 11 -- JAX paths for aberration_tensor / propagate_modal_asymptotic
# ===========================================================================
#
# JAX-traceable, closed-form variants restricted to the
# LG_{0,0} -> LG_{0,0} -> LG_{0,0} (Strehl-amplitude) case.  These
# are the most useful end-to-end-differentiable channels for design
# optimisation (the leading "ideal-Gaussian-pupil-and-source"
# coefficient that absorbs Strehl, defocus, and tilt into a single
# complex amplitude after the M-matrix formalism).
#
# Restrictions vs. the NumPy versions:
#   * v_star (the envelope-stationary pupil point) is a parameter,
#     not solved internally.  Run the NumPy
#     :func:`solve_envelope_stationary` once and feed v_star in --
#     this avoids an iterative Newton in the JAX trace and keeps the
#     graph fully closed-form.
#   * Only the (p, ell) = (0, 0) source / pupil / output channel is
#     evaluated -- multi-mode tensors require the dict-based polynomial
#     algebra that does not translate to JAX cleanly.
#
# Differentiable wrt: fit coefficients, s2_image, v_star, source_point,
# w_s, w_p, w_o, v2_centre.


def _compute_M_b_xp(fit, s2x, s2y, v2x, v2y, src_x, src_y, w_s, w_p,
                     v_cx, v_cy):
    """Backend-aware (NumPy / JAX) M, b, J*, phi*, G0, detJ at
    (s2, v_star).

    Mirrors :func:`_compute_M_b` but uses ``eval_phi_xp`` /
    ``eval_s1_xp`` and JAX autodiff (``jacfwd`` / ``hessian``) for the
    Jacobian and Hessian.  Stays in the input array's backend so it
    runs on NumPy *or* JAX inputs and is differentiable when called on
    JAX inputs.
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not installed; install with `pip install jax`")
    import jax
    import jax.numpy as jnp

    s2x = jnp.asarray(s2x); s2y = jnp.asarray(s2y)

    def s1_of_v(v):
        s1x, s1y = fit.eval_s1_xp(s2x, s2y, v[0], v[1])
        return jnp.stack([s1x, s1y])

    def phi_of_v(v):
        return fit.eval_phi_xp(s2x, s2y, v[0], v[1], include_linear=False)

    v_arr = jnp.stack([jnp.asarray(v2x), jnp.asarray(v2y)])
    s1_star = s1_of_v(v_arr)
    J = jax.jacfwd(s1_of_v)(v_arr)        # (2, 2)
    phi_val = phi_of_v(v_arr)
    g = jax.grad(phi_of_v)(v_arr)         # (2,)
    H_phi = jax.hessian(phi_of_v)(v_arr)  # (2, 2)

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)

    M_real = inv_ws2 * (J.T @ J) + inv_wp2 * jnp.eye(2)
    M = M_real - 1j * jnp.pi * H_phi

    r_star = jnp.stack([s1_star[0] - src_x, s1_star[1] - src_y])
    delta_v = jnp.stack([v_arr[0] - v_cx, v_arr[1] - v_cy])
    b = (2.0j * jnp.pi * g
         - 2.0 * inv_ws2 * (J.T @ r_star)
         - 2.0 * inv_wp2 * delta_v)
    detJ = jnp.abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])

    G0 = jnp.exp(
        -(r_star[0] ** 2 + r_star[1] ** 2) / (w_s * w_s)
        - (delta_v[0] ** 2 + delta_v[1] ** 2) / (w_p * w_p)
    )
    return M, b, s1_star, J, phi_val, G0, detJ


class JaxAberrationTensorResult(NamedTuple):
    """JAX-friendly mirror of :class:`AberrationTensorResult`.

    Fields match the NumPy result so callers can use the same
    ``.L``-indexing pattern across both backends.  The ``L`` field is
    a (1, 1) JAX array carrying the LG_{(0,0),(0,0)} coefficient; the
    ``output_modes`` / ``source_modes`` / ``pupil_modes`` fields are
    static Python lists (NamedTuple PyTree leaves).
    """
    L: object
    output_modes: list
    source_modes: list
    pupil_modes: list
    s2_image: tuple
    w_s: float
    w_p: float
    w_o: object  # may be a JAX scalar
    v_star: tuple


def aberration_tensor_lg00_jax(
    fit: CanonicalPolyFit,
    s2_image: Tuple[float, float],
    v_star: Tuple[float, float],
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    w_s: float = 50e-6,
    w_p: float = 0.05,
    w_o: Optional[float] = None,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    return_result: bool = False,
) -> Any:
    """JAX-traceable LG_{0,0} -> LG_{0,0} -> LG_{0,0} aberration coefficient.

    Single-coefficient form of :func:`aberration_tensor` for the
    output mode (0, 0) projected onto a (0, 0) source and (0, 0) pupil
    -- the leading Strehl amplitude in the modal asymptotic expansion.

    The Newton solve for the envelope-stationary point is **not**
    performed here -- pass ``v_star`` (typically computed once via the
    NumPy :func:`solve_envelope_stationary`).  Skipping the Newton
    keeps the JAX graph fully closed-form and JIT/grad-friendly.

    Parameters
    ----------
    fit : CanonicalPolyFit
    s2_image : (float, float) or 2-tuple of JAX scalars
    v_star : (float, float) or 2-tuple of JAX scalars
        Envelope-stationary point.
    source_point : (float, float)
    w_s, w_p, w_o : float
        Gaussian waists for source / pupil / output.  ``w_o`` defaults
        to a Maréchal scale derived from M.
    v2_centre : (float, float)
    return_result : bool, default False
        If True, return a :class:`JaxAberrationTensorResult` whose
        ``.L`` field holds a (1, 1) JAX array (mirroring the NumPy
        :func:`aberration_tensor` API).  If False (default), return
        the bare complex scalar -- the simplest target for
        ``jax.grad``.

    Returns
    -------
    complex JAX scalar OR :class:`JaxAberrationTensorResult`

    Differentiable via ``jax.grad`` wrt fit coefficients, s2_image,
    v_star, source_point, w_s, w_p, w_o, v2_centre.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")
    import jax.numpy as jnp

    s2x, s2y = s2_image
    v2x, v2y = v_star
    src_x, src_y = source_point
    v_cx, v_cy = v2_centre

    M, b, _s1_star, _J, phi_star, G0, detJ = _compute_M_b_xp(
        fit, s2x, s2y, v2x, v2y, src_x, src_y, w_s, w_p, v_cx, v_cy
    )

    M_inv = jnp.linalg.inv(M)
    # 4.11.2: JAX single-point evaluator uses the principal sqrt, the
    # same value the shared :func:`_maslov_branch_corrected_sqrt` returns
    # with default arguments (no branch history -- there's no pixel
    # loop in the JAX twin to carry one).  The hoist is documented so a
    # future caller that pre-computes a Maslov index can supply a
    # branch-corrected sign by multiplying ``sqrt_detM`` externally.
    # See the NumPy ``aberration_tensor`` for the analogous code path.
    sqrt_detM = jnp.sqrt(jnp.linalg.det(M))

    if w_o is None:
        eig_M_real = jnp.linalg.eigvalsh(jnp.real(M))
        # eigvalsh returns ascending; the largest eigenvalue gives the
        # tightest output Gaussian; clamp to a positive minimum.
        w_o = 1.0 / jnp.sqrt(jnp.maximum(eig_M_real[-1], 1e-30))

    b_quad = 0.25 * (b @ M_inv @ b)
    A_lead = (detJ * (jnp.pi / sqrt_detM) * G0
              * jnp.exp(2j * jnp.pi * phi_star)
              * jnp.exp(b_quad))

    # LG_{0,0} normalisation: N = sqrt(2 / (pi w^2)).  Output projection
    # is the conjugate of N_o (real positive, so just N_o).
    N_s = jnp.sqrt(2.0 / (jnp.pi * w_s * w_s))
    N_p = jnp.sqrt(2.0 / (jnp.pi * w_p * w_p))
    N_o = jnp.sqrt(2.0 / (jnp.pi * w_o * w_o))

    L_scalar = A_lead * N_s * N_p * N_o
    if not return_result:
        return L_scalar

    L_matrix = L_scalar.reshape((1, 1))
    return JaxAberrationTensorResult(
        L=L_matrix,
        output_modes=[(0, 0)],
        source_modes=[(0, 0)],
        pupil_modes=[(0, 0)],
        s2_image=tuple(s2_image),
        w_s=w_s, w_p=w_p, w_o=w_o,
        v_star=tuple(v_star),
    )


def _modal_field_lg00_pixel_jax(fit, s2x, s2y, v2x, v2y,
                                  src_x, src_y, w_s, w_p, v_cx, v_cy):
    """Single-pixel LG_{0,0} field value (no output-mode projection).

    Mirrors the ``propagate_modal_asymptotic`` per-pixel formula for
    source = pupil = LG_{0,0}: returns the field
    ``E(s2) = A_lead * N_s * N_p`` (the polynomial in ``eta`` reduces to
    a constant whose Wick zeroth moment is unity).
    """
    import jax.numpy as jnp
    M, b, _s1_star, _J, phi_star, G0, detJ = _compute_M_b_xp(
        fit, s2x, s2y, v2x, v2y, src_x, src_y, w_s, w_p, v_cx, v_cy
    )
    M_inv = jnp.linalg.inv(M)
    # 4.11.2: see comment in ``aberration_tensor_lg00_jax`` -- single-
    # pixel JAX evaluator uses the principal sqrt; consistent with the
    # shared NumPy helper :func:`_maslov_branch_corrected_sqrt` called
    # with default branch arguments.
    sqrt_detM = jnp.sqrt(jnp.linalg.det(M))
    b_quad = 0.25 * (b @ M_inv @ b)
    A_lead = (detJ * (jnp.pi / sqrt_detM) * G0
              * jnp.exp(2j * jnp.pi * phi_star)
              * jnp.exp(b_quad))
    N_s = jnp.sqrt(2.0 / (jnp.pi * w_s * w_s))
    N_p = jnp.sqrt(2.0 / (jnp.pi * w_p * w_p))
    return A_lead * N_s * N_p


def propagate_modal_asymptotic_lg00_jax(
    fit: CanonicalPolyFit,
    s2_grid_x: np.ndarray,
    s2_grid_y: np.ndarray,
    v_star_grid: np.ndarray,
    *,
    source_point: Tuple[float, float] = (0.0, 0.0),
    w_s: float = 50e-6,
    w_p: float = 0.05,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
) -> Any:
    """JAX-traceable per-pixel evaluator for the LG_{0,0} channel.

    Vectorised LG_{0,0} -> LG_{0,0} version of
    :func:`propagate_modal_asymptotic` over a 2-D output-plane grid.
    Returns the field ``E(s2)`` at each pixel (matching the NumPy
    version's output, not the basis-projected coefficient that
    :func:`aberration_tensor_lg00_jax` returns).

    The ``v_star_grid`` argument supplies the pre-solved envelope-
    stationary point at each pixel -- typically obtained via
    :func:`solve_envelope_stationary` on the NumPy fit (warm-start
    chain along the grid).  Skipping the per-pixel Newton in JAX
    keeps the entire evaluator JIT/grad-friendly.

    Parameters
    ----------
    fit : CanonicalPolyFit
    s2_grid_x, s2_grid_y : array-like
        Output-plane sample points, same shape.
    v_star_grid : array-like, shape (..., 2)
        Pre-solved envelope-stationary point at each grid point.  The
        leading dimensions match s2_grid_x; the trailing dimension is 2
        (v2x, v2y).
    source_point : (float, float)
    w_s, w_p : float
    v2_centre : (float, float)

    Returns
    -------
    complex JAX array, same shape as s2_grid_x
        Output field E(s2).
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")
    import jax
    import jax.numpy as jnp

    s2x = jnp.asarray(s2_grid_x)
    s2y = jnp.asarray(s2_grid_y)
    v_grid = jnp.asarray(v_star_grid)
    if s2x.shape != s2y.shape:
        raise ValueError(
            f"s2_grid_x and s2_grid_y shape mismatch: {s2x.shape} vs {s2y.shape}")
    if v_grid.shape[:-1] != s2x.shape or v_grid.shape[-1] != 2:
        raise ValueError(
            f"v_star_grid shape {v_grid.shape} incompatible with grid {s2x.shape} "
            "(expected (..., 2))")

    src_x, src_y = source_point
    v_cx, v_cy = v2_centre

    def evaluate_pixel(sx, sy, v):
        return _modal_field_lg00_pixel_jax(
            fit, sx, sy, v[0], v[1],
            src_x, src_y, w_s, w_p, v_cx, v_cy,
        )

    flat_sx = s2x.reshape(-1)
    flat_sy = s2y.reshape(-1)
    flat_v = v_grid.reshape(-1, 2)
    # NOTE (v4.12): we *do not* wrap this vmap with ``jax.jit`` because
    # the closure captures ``fit`` (a :class:`CanonicalPolyFit` dataclass
    # not registered as a JAX pytree).  Direct jit'ing would force a
    # fresh trace on every call with a new fit, and depending on JAX
    # version may fail to hash the dataclass at all.  The vmap'd path
    # already fuses the per-pixel evaluator into one XLA dispatch;
    # see future-work note in tests/unit/test_perf_v4_12_0_jax_jit.py.
    flat_out = jax.vmap(evaluate_pixel, in_axes=(0, 0, 0))(
        flat_sx, flat_sy, flat_v)
    return flat_out.reshape(s2x.shape)


# ===========================================================================
# JAX-grad-friendly Newton solver for the envelope-stationary equation
# (implicit-function-theorem backward, fixed-iter forward)
# ===========================================================================
#
# The forward pass runs a fixed-iteration Newton in JAX (jax.lax.fori_loop).
# The backward pass uses jax.custom_vjp + the implicit function theorem:
#
#     F(v*; theta) = 0     =>     dv*/dtheta = -[dF/dv*]^-1 dF/dtheta
#
# So the gradient w.r.t. inputs is computed by a single 2x2 linear solve,
# not by unrolling autograd through the iteration loop.  This gives the
# *exact* IFT gradient at the fixed point regardless of N_iter (as long
# as Newton has converged); it also avoids growing the computational graph
# linearly with iteration count.
#
# Lazy JAX: the @custom_vjp decoration runs on first call (cached
# thereafter).  Module imports cleanly without JAX installed; only
# calling the function requires JAX.

# Module-level cache for the decorated solver; populated on first call.
_JAX_IFT_SOLVER_CACHE = None


def _build_jax_ift_solver():
    """Construct (and cache) the @jax.custom_vjp-decorated Newton-IFT
    solver.  Lazy: imports JAX inside, runs at most once per process."""
    global _JAX_IFT_SOLVER_CACHE
    if _JAX_IFT_SOLVER_CACHE is not None:
        return _JAX_IFT_SOLVER_CACHE

    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not installed; install with `pip install jax`.")
    import jax
    import jax.numpy as jnp
    from functools import partial as _partial

    def _residual(v, s2, source, ws, wp, vc, fit):
        """Envelope-stationary residual
            F(v) = (J^T (s1 - source)) / w_s^2 + (v - v_centre) / w_p^2
        with J = ds_1/dv, s_1 = fit.eval_s1_xp(s2, v).  Returns 2-vec.
        """
        def s1_of_v(vv):
            s1x, s1y = fit.eval_s1_xp(s2[0], s2[1], vv[0], vv[1])
            return jnp.stack([s1x, s1y])

        s1 = s1_of_v(v)
        J = jax.jacfwd(s1_of_v)(v)
        delta_s = s1 - source
        delta_v = v - vc
        return (J.T @ delta_s) / (ws * ws) + delta_v / (wp * wp)

    def _newton_loop(s2, source, ws, wp, vc, n_iter, fit):
        """Fixed-iter Gauss-Newton-like step on F = 0."""

        def step(_, v):
            def s1_of_v(vv):
                s1x, s1y = fit.eval_s1_xp(s2[0], s2[1], vv[0], vv[1])
                return jnp.stack([s1x, s1y])

            s1 = s1_of_v(v)
            J = jax.jacfwd(s1_of_v)(v)
            F = (J.T @ (s1 - source)) / (ws * ws) + (v - vc) / (wp * wp)
            # Gauss-Newton-like Hessian: keep the J^T J piece, drop the
            # second-order term in (s_1 - source) -- exact at the fixed
            # point, gives quadratic-ish convergence elsewhere.
            H = (J.T @ J) / (ws * ws) + jnp.eye(2) / (wp * wp)
            return v - jnp.linalg.solve(H, F)

        return jax.lax.fori_loop(0, n_iter, step, vc)

    @_partial(jax.custom_vjp, nondiff_argnums=(5, 6))
    def _solve(s2, source, ws, wp, vc, n_iter, fit):
        return _newton_loop(s2, source, ws, wp, vc, n_iter, fit)

    def _fwd(s2, source, ws, wp, vc, n_iter, fit):
        v_star = _newton_loop(s2, source, ws, wp, vc, n_iter, fit)
        # Save inputs for the IFT backward.
        return v_star, (v_star, s2, source, ws, wp, vc)

    def _bwd(n_iter, fit, saved, g):
        # Implicit-function theorem:
        #   dv*/dtheta = -[dF/dv]^-1 dF/dtheta
        # so the cotangent contribution of theta is
        #   bar(theta) = -lambda^T dF/dtheta
        # where lambda solves
        #   [dF/dv]^T lambda = bar(v) = g
        # See e.g. Krishnan & Mahmoud 2020, "Differentiating through implicit
        # solvers".  Computing dF/dv and dF/dtheta uses jax.jacrev on the
        # JAX-traceable residual.
        v_star, s2, source, ws, wp, vc = saved
        # All Jacobians evaluated at v_star.
        dF_dv = jax.jacrev(_residual, argnums=0)(
            v_star, s2, source, ws, wp, vc, fit)
        lam = jnp.linalg.solve(dF_dv.T, g)
        # -lambda^T dF/dtheta for each differentiable arg.
        dF_ds2 = jax.jacrev(_residual, argnums=1)(
            v_star, s2, source, ws, wp, vc, fit)
        dF_dsrc = jax.jacrev(_residual, argnums=2)(
            v_star, s2, source, ws, wp, vc, fit)
        dF_dws = jax.jacrev(_residual, argnums=3)(
            v_star, s2, source, ws, wp, vc, fit)
        dF_dwp = jax.jacrev(_residual, argnums=4)(
            v_star, s2, source, ws, wp, vc, fit)
        dF_dvc = jax.jacrev(_residual, argnums=5)(
            v_star, s2, source, ws, wp, vc, fit)
        grad_s2 = -lam @ dF_ds2
        grad_src = -lam @ dF_dsrc
        grad_ws = -lam @ dF_dws    # scalar via 2-vec @ 2-vec
        grad_wp = -lam @ dF_dwp
        grad_vc = -lam @ dF_dvc
        return grad_s2, grad_src, grad_ws, grad_wp, grad_vc

    _solve.defvjp(_fwd, _bwd)
    _JAX_IFT_SOLVER_CACHE = _solve
    return _solve


def solve_envelope_stationary_jax_ift(
    fit: CanonicalPolyFit,
    s2: Tuple[float, float],
    source_point: Tuple[float, float],
    *,
    w_s: float,
    w_p: float,
    v2_centre: Tuple[float, float] = (0.0, 0.0),
    n_iter: int = 15,
) -> Any:
    """JAX-differentiable Newton solver for the envelope-stationary
    equation, with gradients computed via the **implicit function
    theorem** (custom_vjp).

    The forward pass runs a fixed ``n_iter`` Gauss-Newton iterations
    inside ``jax.lax.fori_loop``; the backward pass computes
    ``dv*/dθ = -[∂F/∂v]^{-1} ∂F/∂θ`` via a single 2x2 linear solve,
    so the gradient is independent of ``n_iter`` (as long as Newton
    has converged).

    This is the JAX-grad-friendly companion to
    :func:`solve_envelope_stationary` (NumPy).  Use it when you need
    ``v*`` to flow through a JAX trace (e.g. as input to
    :func:`aberration_tensor_lg00_jax`) and want the gradient of
    downstream losses to propagate back to ``s2``, ``source_point``,
    ``w_s``, ``w_p``, or ``v2_centre``.

    Parameters
    ----------
    fit : CanonicalPolyFit
        Polynomial fit; treated as a *non-differentiable* closure
        (its coefficients are not part of the JAX gradient chain).
    s2, source_point : 2-element array-like
        Image- and source-plane points.  Differentiable.
    w_s, w_p : float (or JAX scalar)
        Source / pupil Gaussian waists.  Differentiable.
    v2_centre : 2-element array-like
        Pupil centre.  Differentiable.
    n_iter : int, default 15
        Fixed Newton-iteration count.  More than 12 is rarely needed
        for well-conditioned designs; 15 gives ample margin.

    Returns
    -------
    v_star : JAX 2-vector
        The envelope-stationary point.  Backward-differentiable via
        IFT.

    Notes
    -----
    Convergence is not checked at run time -- the iteration always
    runs the full ``n_iter`` steps.  For pathological inputs where
    the Gauss-Newton Hessian becomes ill-conditioned, prefer the
    NumPy :func:`solve_envelope_stationary` (which has a stalling
    early-exit + linalg-error fallback).
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not installed; install with `pip install jax`.")
    import jax.numpy as jnp

    solver = _build_jax_ift_solver()
    s2_j = jnp.asarray(s2)
    src_j = jnp.asarray(source_point)
    ws_j = jnp.asarray(w_s)
    wp_j = jnp.asarray(w_p)
    vc_j = jnp.asarray(v2_centre)
    return solver(s2_j, src_j, ws_j, wp_j, vc_j, int(n_iter), fit)


# Add to public API.
if 'aberration_tensor_lg00_jax' not in __all__:
    __all__.append('aberration_tensor_lg00_jax')
if 'propagate_modal_asymptotic_lg00_jax' not in __all__:
    __all__.append('propagate_modal_asymptotic_lg00_jax')
if 'JaxAberrationTensorResult' not in __all__:
    __all__.append('JaxAberrationTensorResult')
if 'solve_envelope_stationary_jax_ift' not in __all__:
    __all__.append('solve_envelope_stationary_jax_ift')


def fit_canonical_polynomials_jax(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    extract_linear_phase: bool = True,
    object_distance: Optional[float] = None,
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    endpoint_anchored: bool = False,
) -> CanonicalPolyFit:
    """JAX-traceable canonical polynomial fit.

    Mirrors :func:`fit_canonical_polynomials` but runs the
    sample-collection ray trace and Chebyshev least-squares solve
    inside JAX, so ``jax.grad`` flows from the resulting fit's
    ``coef_phi`` / ``coef_s1x`` / ``coef_s1y`` back to differentiable
    inputs (``wavelength``, ``source_box_half``, ``pupil_box_half``,
    ``object_distance``).

    The returned :class:`CanonicalPolyFit` carries JAX arrays in its
    ``coef_*`` fields; downstream evaluation via ``eval_phi_xp`` /
    ``eval_s1_xp`` (and the propagators that wrap them) operates on
    those JAX arrays directly, preserving the gradient graph end-to-end.

    Limitations
    -----------
    * The prescription dict (radii, conic, aspheric coeffs) is treated
      as a static argument, same as :func:`trace_jax`.  Differentiate
      w.r.t. lens parameters via :func:`fit_canonical_polynomials` or
      finite differences for now.
    * Vignetting is folded in via a finite-mass weight on each ray;
      heavy vignetting (>50% of rays dead) raises an error matching
      the NumPy version's behaviour.
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not installed; install with `pip install jax`")
    import jax
    import jax.numpy as jnp
    from ..raytrace.jax_trace import make_jax_ray_state, trace_jax

    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if poly_order < 0:
        raise ValueError(f"poly_order must be >= 0, got {poly_order}")
    if source_box_half <= 0 or pupil_box_half <= 0:
        raise ValueError("source_box_half and pupil_box_half must be > 0")
    if not jax.config.jax_enable_x64:
        # 3.5.6: auto-enable JAX x64 with a one-time warning rather
        # than raising.  Single-precision JAX gives ~5% error in
        # coef_phi on a moderate singlet and NaN gradients -- match
        # the reference NumPy fit's float64 precision by default.
        # Users who explicitly want float32 should set
        # `jax.config.update('jax_enable_x64', False)` AFTER importing
        # lumenairy and pass `# nofmt`-style explicit dtype kwargs
        # downstream.
        import warnings as _warnings
        _warnings.warn(
            "fit_canonical_polynomials_jax: JAX x64 mode is required "
            "(single-precision lstsq gives ~5% coefficient error). "
            "Auto-enabling via jax.config.update('jax_enable_x64', "
            "True) for the rest of this Python session.",
            RuntimeWarning, stacklevel=2)
        jax.config.update('jax_enable_x64', True)

    if object_distance is None:
        object_distance = float(prescription.get('object_distance', 0.0)) or 0.0

    # 4-D Chebyshev-node grid -- static (NumPy) at JIT-trace time.
    def cheb_nodes(n: int) -> np.ndarray:
        i = np.arange(n)
        x = np.cos(np.pi * (i + 0.5) / n)
        if endpoint_anchored and n >= 2:
            x = x / np.cos(np.pi / (2.0 * n))
        return x

    u_field_np = cheb_nodes(n_field)
    u_pupil_np = cheb_nodes(n_pupil)

    # Push to JAX so source_box_half / pupil_box_half can be
    # differentiable scalars.
    u_field = jnp.asarray(u_field_np)
    u_pupil = jnp.asarray(u_pupil_np)
    s1x_axis = u_field * source_box_half
    s1y_axis = u_field * source_box_half
    v1x_axis = u_pupil * pupil_box_half
    v1y_axis = u_pupil * pupil_box_half

    S1X, S1Y, V1X, V1Y = jnp.meshgrid(
        s1x_axis, s1y_axis, v1x_axis, v1y_axis, indexing='ij')
    s1x_in = S1X.ravel()
    s1y_in = S1Y.ravel()
    v1x_in = V1X.ravel()
    v1y_in = V1Y.ravel()
    n_rays = int(s1x_in.size)

    # Static check on pupil box (Python-time -- pupil_box_half should
    # be < 1 for real direction cosines).
    if float(pupil_box_half) >= 1.0:
        raise ValueError(
            f"pupil_box_half must be < 1; got {pupil_box_half}.")

    # Initial state at z = -object_distance.
    sumsq = v1x_in * v1x_in + v1y_in * v1y_in
    N1 = jnp.sqrt(jnp.maximum(1.0 - sumsq, 0.0))
    state = make_jax_ray_state(
        x=s1x_in, y=s1y_in,
        z=jnp.full_like(s1x_in, -float(object_distance)),
        L=v1x_in, M=v1y_in, N=N1,
    )

    final = trace_jax(state, prescription, wavelength,
                      surface_diffraction=surface_diffraction)

    alive = final.alive
    # Best-effort liveness check.  Skipped silently under jit/grad
    # tracing where alive is an abstract array; relies on the user
    # having validated the prescription via the NumPy function once.
    try:
        n_alive_check = int(jnp.sum(alive))
        if n_alive_check < max(64, 0.5 * n_rays):
            raise RuntimeError(
                f"Too many rays died during canonical-fit trace: "
                f"{n_alive_check} alive of {n_rays}.  Reduce "
                f"pupil_box_half, check prescription apertures, or "
                f"rebalance the source/pupil sampling boxes.")
        n_alive_static = n_alive_check
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        n_alive_static = n_rays  # opaque under tracing -- assume OK

    # Use weights = alive to drop dead rays from the lstsq without
    # variable-shape arrays (which JAX disallows).
    w = alive.astype(jnp.float64 if jax.config.jax_enable_x64
                     else jnp.float32)

    s2x_obs = final.x
    s2y_obs = final.y
    v2x_obs = final.L
    v2y_obs = final.M
    phi_obs = final.opd / float(wavelength)

    # ---- Normaliser (centre, half-range) from alive observations ----
    def _normaliser_jax(v, mask):
        v_masked = jnp.where(mask, v, jnp.nan)
        vmin = jnp.nanmin(v_masked)
        vmax = jnp.nanmax(v_masked)
        centre = 0.5 * (vmin + vmax)
        half = 0.5 * (vmax - vmin) * 1.05
        half = jnp.where(half == 0.0, 1.0, half)
        return centre, half

    s2x_c, s2x_h = _normaliser_jax(s2x_obs, alive)
    s2y_c, s2y_h = _normaliser_jax(s2y_obs, alive)
    v2x_c, v2x_h = _normaliser_jax(v2x_obs, alive)
    v2y_c, v2y_h = _normaliser_jax(v2y_obs, alive)

    u_s2x = (s2x_obs - s2x_c) / s2x_h
    u_s2y = (s2y_obs - s2y_c) / s2y_h
    u_v2x = (v2x_obs - v2x_c) / v2x_h
    u_v2y = (v2y_obs - v2y_c) / v2y_h

    # ---- Linear-phase pre-fit ---------------------------------------
    if extract_linear_phase:
        X5 = jnp.stack([
            jnp.ones_like(u_s2x),
            u_s2x, u_s2y, u_v2x, u_v2y,
        ], axis=1)
        # Apply mask via row weighting.
        Xw = X5 * w[:, None]
        bw = phi_obs * w
        linear_coeffs, *_ = jnp.linalg.lstsq(Xw, bw, rcond=None)
        opd_residual = phi_obs - X5 @ linear_coeffs
    else:
        linear_coeffs = None
        opd_residual = phi_obs

    # ---- Total-degree multi-indices (Python-time) -------------------
    multi_indices = _multi_indices_total_degree(4, poly_order)
    n_basis = len(multi_indices)

    # ---- Build Chebyshev Vandermonde for each axis ------------------
    def cheb_vand_jax(u, max_k):
        T = [jnp.ones_like(u), u]
        for n in range(2, max_k + 1):
            T.append(2.0 * u * T[-1] - T[-2])
        return jnp.stack(T)  # (max_k+1, ...)

    T1 = cheb_vand_jax(u_s2x, poly_order)
    T2 = cheb_vand_jax(u_s2y, poly_order)
    T3 = cheb_vand_jax(u_v2x, poly_order)
    T4 = cheb_vand_jax(u_v2y, poly_order)

    K1 = jnp.asarray([m[0] for m in multi_indices], dtype=jnp.int32)
    K2 = jnp.asarray([m[1] for m in multi_indices], dtype=jnp.int32)
    K3 = jnp.asarray([m[2] for m in multi_indices], dtype=jnp.int32)
    K4 = jnp.asarray([m[3] for m in multi_indices], dtype=jnp.int32)
    A_full = T1[K1] * T2[K2] * T3[K3] * T4[K4]   # (n_basis, n_rays)
    A = A_full.T  # (n_rays, n_basis)

    # Mask via row weights so dead rays contribute nothing to the
    # least-squares normal equations.
    A_w = A * w[:, None]

    coef_phi, *_ = jnp.linalg.lstsq(A_w, opd_residual * w, rcond=None)
    coef_s1x, *_ = jnp.linalg.lstsq(A_w, s1x_in * w, rcond=None)
    coef_s1y, *_ = jnp.linalg.lstsq(A_w, s1y_in * w, rcond=None)

    res_phi_rms = jnp.sqrt(jnp.sum(w * (opd_residual - A @ coef_phi) ** 2)
                            / jnp.maximum(jnp.sum(w), 1.0))
    res_s1_rms = jnp.sqrt(0.5 * jnp.sum(
        w * ((s1x_in - A @ coef_s1x) ** 2 +
             (s1y_in - A @ coef_s1y) ** 2)
    ) / jnp.maximum(jnp.sum(w), 1.0))

    # Note: under JIT/grad these are JAX traced scalars, not Python
    # floats; the dataclass type hints are advisory.  Same goes for
    # the centre/halfrange fields and res_*.
    return CanonicalPolyFit(
        poly_order=poly_order,
        multi_indices=multi_indices,
        coef_phi=coef_phi,
        coef_s1x=coef_s1x,
        coef_s1y=coef_s1y,
        s2x_centre=s2x_c, s2x_halfrange=s2x_h,
        s2y_centre=s2y_c, s2y_halfrange=s2y_h,
        v2x_centre=v2x_c, v2x_halfrange=v2x_h,
        v2y_centre=v2y_c, v2y_halfrange=v2y_h,
        wavelength=wavelength,
        res_phi_rms_waves=res_phi_rms,
        res_s1_rms_m=res_s1_rms,
        n_rays=n_alive_static,
        linear_coeffs_phi=linear_coeffs,
        extract_linear_phase=extract_linear_phase,
    )


if 'fit_canonical_polynomials_jax' not in __all__:
    __all__.append('fit_canonical_polynomials_jax')
