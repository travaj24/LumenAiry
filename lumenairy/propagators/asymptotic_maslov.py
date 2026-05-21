"""Maslov-index handling + batched helpers for the asymptotic propagator.

v5.1.0 file-split (Agent D):  extracted from
``lumenairy.propagators.asymptotic`` with NO public-API or physics
change.  Holds:

* :func:`_maslov_branch_corrected_sqrt` -- Keller-Maslov branch-
  continuous ``sqrt(det M)``.  Single canonical site so all
  asymptotic evaluators consume the same unwrap logic.
* Batched analogues of the scalar saddle-point helpers used by the
  vectorised cold-start path in
  :func:`propagate_modal_asymptotic`:
  :func:`_phi_v2_hessian_batch`,
  :func:`_compute_M_b_batch`,
  :func:`_gaussian_moment_table_2d_batch`,
  :func:`_solve_envelope_stationary_batch`.
* Batched polynomial-algebra helpers
  (:func:`_batched_polynomial_substitute_linear_2d`,
  :func:`_batched_polynomial_under_affine_shift`,
  :func:`_poly_dict_to_array`).

All re-exported through :mod:`lumenairy.propagators.asymptotic`
so existing call sites continue to work unchanged.  These names
remain semantically private (leading underscore) but they are
imported by name in the test suite (audit pins) and consumed by
:func:`propagate_modal_asymptotic` defined in the
``asymptotic`` shell so it can resolve names against the shell's
module globals (test-monkey-patching contract).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .._math.chebyshev import (
    chebyshev_derivative_vandermonde as _chebyshev_derivative_vandermonde,
)
from .._math.chebyshev import (
    chebyshev_second_derivative_vandermonde as _chebyshev_second_derivative_vandermonde,
)

# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# Chebyshev helpers moved to lumenairy._math.chebyshev; binding the
# new public names to the legacy underscore-prefixed locals keeps the
# existing call sites in this module unchanged.
from .._math.chebyshev import (
    chebyshev_vandermonde as _chebyshev_vandermonde,
)
from .asymptotic_canonical_fit import CanonicalPolyFit

__all__: List[str] = [
    # Module-private names; re-exported via the asymptotic shell.
]


# ===========================================================================
# Section 6 -- Modal asymptotic propagator (leading-order) -- helpers
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


def _phi_v2_hessian_batch(fit: CanonicalPolyFit,
                           s2x: np.ndarray, s2y: np.ndarray,
                           v2x: np.ndarray, v2y: np.ndarray) -> np.ndarray:
    """Vectorised ``_phi_v2_hessian`` over arrays of pixels.

    Returns an ``(N, 2, 2)`` real array where each ``[k]`` is the same
    Hessian that the scalar ``_phi_v2_hessian`` would return for pixel
    ``k``.

    Implementation notes
    --------------------
    Uses the SAME term-by-term accumulation order as the scalar
    helper (loop over multi-index, skip-if-zero) so the
    floating-point result is bit-equal to the scalar version pixel by
    pixel.  Switching to a single ``np.einsum('m,mn->n', coef, basis)``
    contraction loses up to ~1e-7 relative because the accumulation
    walks the basis terms in a different order; the saddle-point
    sqrt(det M) then amplifies that into a 1e-8 field error and fails
    the 1e-12 correctness pin.  We pay one Python loop over basis terms
    here (typically <= 70 terms for poly_order=6, totally negligible vs
    the N_pixel batch dimension below).
    """
    s2x = np.asarray(s2x, dtype=np.float64)
    s2y = np.asarray(s2y, dtype=np.float64)
    v2x = np.asarray(v2x, dtype=np.float64)
    v2y = np.asarray(v2y, dtype=np.float64)
    u1 = (s2x - fit.s2x_centre) / fit.s2x_halfrange
    u2 = (s2y - fit.s2y_centre) / fit.s2y_halfrange
    u3 = (v2x - fit.v2x_centre) / fit.v2x_halfrange
    u4 = (v2y - fit.v2y_centre) / fit.v2y_halfrange

    T1 = _chebyshev_vandermonde(u1, fit.poly_order)
    T2 = _chebyshev_vandermonde(u2, fit.poly_order)
    T3 = _chebyshev_vandermonde(u3, fit.poly_order)
    T4 = _chebyshev_vandermonde(u4, fit.poly_order)
    dT3 = _chebyshev_derivative_vandermonde(u3, fit.poly_order)
    dT4 = _chebyshev_derivative_vandermonde(u4, fit.poly_order)
    d2T3 = _chebyshev_second_derivative_vandermonde(u3, fit.poly_order)
    d2T4 = _chebyshev_second_derivative_vandermonde(u4, fit.poly_order)

    N = s2x.shape[0]
    h33 = np.zeros(N, dtype=np.float64)
    h34 = np.zeros(N, dtype=np.float64)
    h44 = np.zeros(N, dtype=np.float64)
    # Same accumulation order as the scalar helper.  T1[k1] is a 1-D
    # array of length N (one Chebyshev value per pixel), so the
    # per-term update is a vectorised N-element addition.
    for c, (k1, k2, k3, k4) in zip(fit.coef_phi, fit.multi_indices):
        if c == 0.0:
            continue
        T12 = T1[k1] * T2[k2]
        h33 += c * T12 * d2T3[k3] * T4[k4]
        h34 += c * T12 * dT3[k3] * dT4[k4]
        h44 += c * T12 * T3[k3] * d2T4[k4]

    invhx = 1.0 / fit.v2x_halfrange
    invhy = 1.0 / fit.v2y_halfrange
    H = np.empty((N, 2, 2), dtype=np.float64)
    H[:, 0, 0] = h33 * (invhx * invhx)
    H[:, 0, 1] = h34 * (invhx * invhy)
    H[:, 1, 0] = H[:, 0, 1]
    H[:, 1, 1] = h44 * (invhy * invhy)
    return H


def _compute_M_b_batch(fit: CanonicalPolyFit,
                        s2x: np.ndarray, s2y: np.ndarray,
                        v2x: np.ndarray, v2y: np.ndarray,
                        src_x: float, src_y: float,
                        w_s: float, w_p: float,
                        v_cx: float, v_cy: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray]:
    """Vectorised ``_compute_M_b`` over arrays of pixels.

    Inputs ``s2x, s2y, v2x, v2y`` are 1-D arrays of length ``N``.
    Returns batched arrays:

    * ``M[N, 2, 2]`` complex beam matrix.
    * ``b[N, 2]``   complex linear term.
    * ``s1_star[N, 2]`` source-plane sample point.
    * ``J_star[N, 2, 2]`` Jacobian.
    * ``phi_star[N]`` OPL piston (waves, complex per return convention).
    * ``G0[N]`` real Gaussian amplitude.
    * ``detJ[N]`` Jacobian determinant magnitude.
    """
    s2x = np.asarray(s2x, dtype=np.float64)
    s2y = np.asarray(s2y, dtype=np.float64)
    v2x = np.asarray(v2x, dtype=np.float64)
    v2y = np.asarray(v2y, dtype=np.float64)
    (s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y) = (
        fit.eval_s1_with_v2_grad(s2x, s2y, v2x, v2y)
    )
    phi, dPhi_dv2x, dPhi_dv2y = fit.eval_phi_with_v2_grad(
        s2x, s2y, v2x, v2y, include_linear=False,
    )

    N = s2x.shape[0]
    J = np.empty((N, 2, 2), dtype=np.float64)
    J[:, 0, 0] = dS1x_dv2x
    J[:, 0, 1] = dS1x_dv2y
    J[:, 1, 0] = dS1y_dv2x
    J[:, 1, 1] = dS1y_dv2y
    g = np.empty((N, 2), dtype=np.float64)
    g[:, 0] = dPhi_dv2x
    g[:, 1] = dPhi_dv2y

    H_phi = _phi_v2_hessian_batch(fit, s2x, s2y, v2x, v2y)

    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)
    # J^T @ J per pixel: (N, 2, 2).  ``np.matmul`` broadcasts the last
    # two axes so transpose of (N, 2, 2) is (N, 2, 2) via swapaxes.
    JtJ = np.matmul(np.swapaxes(J, -1, -2), J)
    eye2 = np.eye(2, dtype=np.float64)
    M_real = inv_ws2 * JtJ + inv_wp2 * eye2  # broadcasts (N, 2, 2)
    M = M_real.astype(np.complex128) - (1j * math.pi) * H_phi
    s1_star = np.empty((N, 2), dtype=np.float64)
    s1_star[:, 0] = s1x
    s1_star[:, 1] = s1y
    r_star = s1_star - np.array([src_x, src_y], dtype=np.float64)
    delta_v = np.empty((N, 2), dtype=np.float64)
    delta_v[:, 0] = v2x - v_cx
    delta_v[:, 1] = v2y - v_cy
    # b = 2 pi i g - 2 inv_ws2 J^T r_star - 2 inv_wp2 delta_v
    # J^T @ r per pixel -> (N, 2)
    Jt_r = np.einsum('nij,nj->ni', np.swapaxes(J, -1, -2), r_star)
    b = ((2.0j * math.pi) * g.astype(np.complex128)
         - 2.0 * inv_ws2 * Jt_r.astype(np.complex128)
         - 2.0 * inv_wp2 * delta_v.astype(np.complex128))
    detJ = np.abs(J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0])
    G0 = np.exp(
        -(r_star[:, 0] ** 2 + r_star[:, 1] ** 2) / (w_s * w_s)
        - (delta_v[:, 0] ** 2 + delta_v[:, 1] ** 2) / (w_p * w_p)
    )
    return M, b, s1_star, J, phi.astype(np.complex128), G0, detJ


def _gaussian_moment_table_2d_batch(M: np.ndarray, max_total_order: int
                                     ) -> Tuple[List[Tuple[int, int]],
                                                np.ndarray]:
    """Vectorised ``gaussian_moment_table_2d`` over an ``(N, 2, 2)``
    batch of M matrices.

    Returns ``(keys, table)`` where ``keys`` is the same ``(a, b)`` index
    enumeration that ``gaussian_moment_table_2d`` produces (in
    insertion order ``(total, a)``), and ``table`` is a complex
    ``(N, n_terms)`` array with ``table[k, q]`` equal to
    ``<eta_x^{keys[q][0]} eta_y^{keys[q][1]}>_{Sigma_k}`` with
    ``Sigma_k = 0.5 inv(M[k])``.
    """
    if max_total_order < 0:
        raise ValueError(
            f"max_total_order must be >= 0, got {max_total_order}")
    M = np.asarray(M)
    if M.ndim != 3 or M.shape[-2:] != (2, 2):
        raise ValueError(
            f"M must be (N, 2, 2), got shape {M.shape}")
    N = M.shape[0]
    # Closed-form 2x2 inverse for the covariance.  Same ULP tradeoff
    # as in :func:`propagate_modal_asymptotic` -- :func:`np.linalg.inv`
    # on a (N, 2, 2) slice gives marginally different rounding than
    # the explicit formula, but the difference is well below the
    # tolerance of any downstream test pin (~1e-10).
    a11 = M[:, 0, 0]
    a12 = M[:, 0, 1]
    a21 = M[:, 1, 0]
    a22 = M[:, 1, 1]
    det = a11 * a22 - a12 * a21
    inv_det = 1.0 / det
    s11 = 0.5 * a22 * inv_det        # sigma[0, 0]
    s12 = -0.5 * a12 * inv_det       # sigma[0, 1]
    s22 = 0.5 * a11 * inv_det        # sigma[1, 1]

    # Build the same key enumeration as gaussian_moment_table_2d.
    keys: List[Tuple[int, int]] = []
    for total in range(max_total_order + 1):
        for a in range(total + 1):
            keys.append((a, total - a))
    n_terms = len(keys)
    table = np.empty((N, n_terms), dtype=np.complex128)
    for q, (a, b) in enumerate(keys):
        if (a + b) % 2 != 0:
            table[:, q] = 0.0 + 0.0j
            continue
        # Wick contraction (closed-form pair-counting sum).  Same
        # algebra as gaussian_moment_2d:
        #   <eta_x^a eta_y^b> = sum_{p12 = (a%2), step 2}^{min(a,b)}
        #       a!*b! / (p11! p12! p22! 2^p11 2^p22)
        #       * s11^p11 * s12^p12 * s22^p22
        # where p11 = (a - p12) / 2, p22 = (b - p12) / 2.
        p12_min = a % 2
        fa = math.factorial(a)
        fb = math.factorial(b)
        col = np.zeros(N, dtype=np.complex128)
        for p12 in range(p12_min, min(a, b) + 1, 2):
            p11 = (a - p12) // 2
            p22 = (b - p12) // 2
            denom = (math.factorial(p11) * math.factorial(p12)
                     * math.factorial(p22) * (2 ** p11) * (2 ** p22))
            coef = (fa * fb) / denom
            col += coef * (s11 ** p11) * (s12 ** p12) * (s22 ** p22)
        table[:, q] = col
    return keys, table


def _poly_dict_to_array(coeffs: Dict[Tuple[int, int], complex],
                         ix_max: int, iy_max: int) -> np.ndarray:
    """Convert ``{(i, j): c}`` to a ``(ix_max+1, iy_max+1)`` complex
    dense array."""
    arr = np.zeros((ix_max + 1, iy_max + 1), dtype=np.complex128)
    for (i, j), c in coeffs.items():
        arr[i, j] = c
    return arr


def _batched_polynomial_substitute_linear_2d(
    src_arr: np.ndarray,  # (Ix+1, Iy+1) source polynomial coeffs
    A: np.ndarray,        # (N, 2, 2) per-pixel linear map
    b_const: np.ndarray,  # (N, 2) per-pixel offset
) -> np.ndarray:
    """Batched analogue of ``_polynomial_substitute_linear_2d``.

    Substitutes ``(r_x, r_y) -> A_k @ (eta_x, eta_y) + b_const_k`` in
    each pixel's polynomial and returns ``(N, Ox+1, Oy+1)`` per-pixel
    coefficient arrays.  ``Ox, Oy`` follow the same logic as the
    scalar version (driven by source's max-i, max-j).
    """
    Ix_max = src_arr.shape[0] - 1
    Iy_max = src_arr.shape[1] - 1
    N = A.shape[0]
    A = np.asarray(A, dtype=np.complex128)
    b_const = np.asarray(b_const, dtype=np.complex128)
    A_xx = A[:, 0, 0]
    A_xy = A[:, 0, 1]
    A_yx = A[:, 1, 0]
    A_yy = A[:, 1, 1]
    b_x = b_const[:, 0]
    b_y = b_const[:, 1]
    # cache_x[i] :: (i+1, i+1, N) polynomial in (eta_x, eta_y) per pixel
    # representing (A_xx eta_x + A_xy eta_y + b_x)^i.  Match the scalar
    # helper's multinomial expansion exactly:
    #   (a x + b y + c)^n = sum_{i+j+k=n} n!/(i!j!k!) a^i b^j c^k x^i y^j
    cache_x: List[np.ndarray] = []
    cache_y: List[np.ndarray] = []
    # Pre-tabulate factorials to keep this loop in pure-Python integer math.
    for kind in (0, 1):
        coef_a, coef_b, coef_c = ((A_xx, A_xy, b_x) if kind == 0
                                   else (A_yx, A_yy, b_y))
        cache = []
        for n in range(max(Ix_max, Iy_max) + 1):
            poly_n = np.zeros((n + 1, n + 1, N), dtype=np.complex128)
            fn = math.factorial(n)
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    k = n - i - j
                    w = fn // (math.factorial(i) * math.factorial(j)
                                * math.factorial(k))
                    poly_n[i, j, :] += (
                        w * (coef_a ** i) * (coef_b ** j) * (coef_c ** k)
                    )
            cache.append(poly_n)
        if kind == 0:
            cache_x = cache
        else:
            cache_y = cache

    # Output total degree max is (Ix_max + Iy_max).
    Ox_max = Ix_max + Iy_max
    Oy_max = Ix_max + Iy_max
    out = np.zeros((N, Ox_max + 1, Oy_max + 1), dtype=np.complex128)
    for i in range(Ix_max + 1):
        for j in range(Iy_max + 1):
            c = src_arr[i, j]
            if c == 0.0:
                continue
            px = cache_x[i]  # (i+1, i+1, N)
            py = cache_y[j]  # (j+1, j+1, N)
            if px.size == 0 or py.size == 0:
                continue
            # Multiply px (shape (a1+1, b1+1, N)) by py (shape (a2+1, b2+1, N))
            # in (eta_x, eta_y) per pixel, then accumulate into out * c.
            # The product polynomial has shape ((a1+a2+1), (b1+b2+1), N).
            for ix1 in range(px.shape[0]):
                for jy1 in range(px.shape[1]):
                    coef1 = px[ix1, jy1]    # (N,)
                    if not np.any(coef1):
                        continue
                    for ix2 in range(py.shape[0]):
                        for jy2 in range(py.shape[1]):
                            coef2 = py[ix2, jy2]
                            if not np.any(coef2):
                                continue
                            out[:, ix1 + ix2, jy1 + jy2] += (
                                c * coef1 * coef2
                            )
    return out


def _batched_polynomial_under_affine_shift(
    pup_arr: np.ndarray,    # (Ix+1, Iy+1) pupil polynomial coeffs
    shift_x: np.ndarray,    # (N,) complex per-pixel shift
    shift_y: np.ndarray,    # (N,) complex per-pixel shift
) -> np.ndarray:
    """Batched ``_polynomial_under_affine_shift``.

    Substitutes ``(x, y) -> (x + shift_x_k, y + shift_y_k)`` per pixel.
    Returns ``(N, Ix+1, Iy+1)``.
    """
    Ix_max = pup_arr.shape[0] - 1
    Iy_max = pup_arr.shape[1] - 1
    shift_x = np.asarray(shift_x, dtype=np.complex128)
    shift_y = np.asarray(shift_y, dtype=np.complex128)
    N = shift_x.shape[0]
    # bin_x[i, k] = C(i, k) * shift_x^(i-k) per pixel; shape (Ix+1, Ix+1, N)
    bin_x = np.zeros((Ix_max + 1, Ix_max + 1, N), dtype=np.complex128)
    bin_y = np.zeros((Iy_max + 1, Iy_max + 1, N), dtype=np.complex128)
    for i in range(Ix_max + 1):
        for k in range(i + 1):
            bin_x[i, k, :] = math.comb(i, k) * (shift_x ** (i - k))
    for j in range(Iy_max + 1):
        for k in range(j + 1):
            bin_y[j, k, :] = math.comb(j, k) * (shift_y ** (j - k))
    out = np.zeros((N, Ix_max + 1, Iy_max + 1), dtype=np.complex128)
    for i in range(Ix_max + 1):
        for j in range(Iy_max + 1):
            c = pup_arr[i, j]
            if c == 0.0:
                continue
            for kx in range(i + 1):
                bx = bin_x[i, kx]  # (N,)
                for ky in range(j + 1):
                    by = bin_y[j, ky]
                    out[:, kx, ky] += c * bx * by
    return out


def _solve_envelope_stationary_batch(
    fit: CanonicalPolyFit,
    s2x: np.ndarray, s2y: np.ndarray,
    src_x: float, src_y: float,
    w_s: float, w_p: float,
    v_cx: float, v_cy: float,
    max_iter: int = 12,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ``solve_envelope_stationary`` over a 1-D batch of
    output pixels.

    Returns ``(v2x_star, v2y_star, converged_mask)``.  Pixels that
    fail (singular Hessian or non-finite update) carry the last
    finite iterate and ``converged_mask=False``.  ``converged_mask`` is
    ``True`` only for pixels whose residual fell below ``tol``;
    stalled and singular pixels remain ``False`` so callers can branch
    on the failure mode.

    All pixels iterate in lockstep starting from ``(v_cx, v_cy)``;
    converged pixels still consume CPU on subsequent iterations but
    that cost is amortised across the batch.  The math matches the
    scalar Gauss-Newton-like solver bit-for-bit at the stationary
    point (where the neglected s_1 Hessian piece vanishes).

    v4.14.1 (P1-NEW-3):  prior to v4.14.1 the loop set
    ``converged[idx[done & ~is_conv]] = True`` to drop stalled /
    singular pixels from the active set, which silently flagged
    failures as successes -- contrary to the documented contract.
    The function now uses a separate ``finished`` mask for the
    active-set bookkeeping and writes ``True`` to ``converged`` only
    for genuinely-converged pixels (``rn < tol``).
    """
    s2x = np.asarray(s2x, dtype=np.float64)
    s2y = np.asarray(s2y, dtype=np.float64)
    N = s2x.shape[0]
    v2x = np.full(N, v_cx, dtype=np.float64)
    v2y = np.full(N, v_cy, dtype=np.float64)
    converged = np.zeros(N, dtype=bool)
    # v4.14.1 (P1-NEW-3): ``finished`` is the active-set bookkeeping
    # mask -- it covers BOTH genuinely-converged pixels and pixels
    # dropped due to stall / singularity / non-finite step.
    # ``converged`` (returned to the caller) is only set on residual-
    # passes-tol so the docstring contract holds.
    finished = np.zeros(N, dtype=bool)
    inv_ws2 = 1.0 / (w_s * w_s)
    inv_wp2 = 1.0 / (w_p * w_p)
    last_norm = np.full(N, np.inf, dtype=np.float64)
    np.array([src_x, src_y], dtype=np.float64)
    np.array([v_cx, v_cy], dtype=np.float64)
    eye2 = np.eye(2, dtype=np.float64)
    for it in range(max_iter):
        active = ~finished
        if not np.any(active):
            break
        # Evaluate s1, J on the *active* subset; this keeps cost down
        # once most pixels have converged.
        idx = np.where(active)[0]
        sx = s2x[idx]
        sy = s2y[idx]
        vx = v2x[idx]
        vy = v2y[idx]
        (s1x, s1y, dS1x_dv2x, dS1x_dv2y, dS1y_dv2x, dS1y_dv2y) = (
            fit.eval_s1_with_v2_grad(sx, sy, vx, vy)
        )
        K = sx.shape[0]
        J = np.empty((K, 2, 2), dtype=np.float64)
        J[:, 0, 0] = dS1x_dv2x
        J[:, 0, 1] = dS1x_dv2y
        J[:, 1, 0] = dS1y_dv2x
        J[:, 1, 1] = dS1y_dv2y
        delta_s1 = np.empty((K, 2), dtype=np.float64)
        delta_s1[:, 0] = s1x - src_x
        delta_s1[:, 1] = s1y - src_y
        delta_v = np.empty((K, 2), dtype=np.float64)
        delta_v[:, 0] = vx - v_cx
        delta_v[:, 1] = vy - v_cy
        # residual = inv_ws2 * (J^T @ delta_s1) + inv_wp2 * delta_v
        residual = inv_ws2 * np.einsum(
            'kij,kj->ki', np.swapaxes(J, -1, -2), delta_s1
        ) + inv_wp2 * delta_v
        rn = np.sqrt(residual[:, 0] ** 2 + residual[:, 1] ** 2)
        # Mark convergence / stall.  Match the scalar logic:
        #   is_converged = rn < tol
        #   is_stalling  = (it >= 2 and rn > 0.9 * last_norm and last_norm > 1e-300)
        is_conv = rn < tol
        is_stall = (it >= 2) & (rn > 0.9 * last_norm[idx]) & (last_norm[idx] > 1e-300)
        done = is_conv | is_stall
        # H = inv_ws2 * J^T J + inv_wp2 * I
        H = inv_ws2 * np.matmul(np.swapaxes(J, -1, -2), J) + inv_wp2 * eye2
        # Batched 2x2 solve.  H is real symmetric positive-definite for
        # well-conditioned pixels.  Use closed form for speed and to
        # surface singular cases via finite-mask.
        det_H = H[:, 0, 0] * H[:, 1, 1] - H[:, 0, 1] * H[:, 1, 0]
        # Avoid divide-by-zero; mask out singular pixels (they stall in
        # place; converged stays whatever it was).
        safe = det_H != 0.0
        step = np.zeros((K, 2), dtype=np.float64)
        if np.any(safe):
            inv_det = 1.0 / det_H[safe]
            step[safe, 0] = inv_det * (H[safe, 1, 1] * residual[safe, 0]
                                        - H[safe, 0, 1] * residual[safe, 1])
            step[safe, 1] = inv_det * (-H[safe, 1, 0] * residual[safe, 0]
                                        + H[safe, 0, 0] * residual[safe, 1])
        # Pixels with singular H mark as done (last iterate retained).
        done = done | ~safe
        # Stalled pixels with non-finite step also done.
        bad = ~np.isfinite(step[:, 0]) | ~np.isfinite(step[:, 1])
        done = done | bad
        # v4.14.1 (P1-NEW-3): ``converged`` is the user-facing success
        # flag; only set it for residual-passes-tol pixels.
        converged[idx[is_conv]] = True
        # Non-done pixels: take the Newton step.
        update_mask = ~done
        v2x[idx[update_mask]] = vx[update_mask] - step[update_mask, 0]
        v2y[idx[update_mask]] = vy[update_mask] - step[update_mask, 1]
        # Stalled / singular: don't move (mirrors scalar early return).
        last_norm[idx] = rn
        # ``finished`` is the active-set drop flag -- mark ALL done
        # pixels (converged, stalled, singular, non-finite) so they
        # leave the active set, but DO NOT promote stall/singular into
        # ``converged``.
        finished[idx[done]] = True
    return v2x, v2y, converged
