"""JAX-traceable twin of the NumPy asymptotic propagator path.

v5.1.0 file-split (Agent D):  extracted from
``lumenairy.propagators.asymptotic`` with NO public-API or physics
change.  Holds:

* Backend-aware Chebyshev evaluators
  (``_chebyshev_vandermonde_xp`` / ``_evaluate_polynomial_4d_xp``)
  used by :class:`CanonicalPolyFit` / :class:`HFPolyFit`'s
  ``eval_phi_xp`` / ``eval_s1_xp`` methods (attached here via
  monkey-patch at import time).
* The JAX path for :func:`aberration_tensor` restricted to the
  LG_{0,0} -> LG_{0,0} -> LG_{0,0} channel
  (:func:`aberration_tensor_lg00_jax`).
* The JAX vmap'd per-pixel modal propagator
  (:func:`propagate_modal_asymptotic_lg00_jax`).
* The JAX-grad-friendly Newton-IFT solver
  (:func:`solve_envelope_stationary_jax_ift`).
* The JAX-traceable canonical fit (:func:`fit_canonical_polynomials_jax`).
* The ``_JAX_IFT_SOLVER_CACHE`` singleton + companion lock for
  lazy first-call ``custom_vjp`` decoration.

All re-exported through :mod:`lumenairy.propagators.asymptotic` so
existing call sites continue to work unchanged.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

from ..backend import JAX_AVAILABLE
from ..elements.lenses import (
    _multi_indices_total_degree,
)
from .asymptotic_canonical_fit import CanonicalPolyFit, HFPolyFit


__all__ = [
    'JaxAberrationTensorResult',
    'aberration_tensor_lg00_jax',
    'propagate_modal_asymptotic_lg00_jax',
    'solve_envelope_stationary_jax_ift',
    'fit_canonical_polynomials_jax',
]


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
# v4.14.2: paired with ``_JAX_IFT_SOLVER_CACHE_LOCK`` per the cache-lock
# meta-pin requirement.  Race window is only on first concurrent call
# after import; subsequent calls are read-only.  The lock guards the
# build-and-publish race so two threads don't both run the JAX
# custom_vjp decoration.
_JAX_IFT_SOLVER_CACHE = None
_JAX_IFT_SOLVER_CACHE_LOCK = threading.Lock()


def _build_jax_ift_solver():
    """Construct (and cache) the @jax.custom_vjp-decorated Newton-IFT
    solver.  Lazy: imports JAX inside, runs at most once per process.

    v4.14.2: double-check locking pattern.  Fast path (no lock) for the
    common case where the cache is already populated; slow path
    (with lock) re-checks inside the lock and delegates the actual
    build to :func:`_build_jax_ift_solver_impl`.  Guards the
    build-and-publish race so two threads don't both run the JAX
    custom_vjp decoration on first concurrent call.
    """
    global _JAX_IFT_SOLVER_CACHE
    # Fast path -- common case after first call.
    if _JAX_IFT_SOLVER_CACHE is not None:
        return _JAX_IFT_SOLVER_CACHE
    # Slow path -- first call (or first concurrent call from a fresh
    # process).  Acquire the lock; re-check inside; build under lock.
    with _JAX_IFT_SOLVER_CACHE_LOCK:
        if _JAX_IFT_SOLVER_CACHE is None:
            _JAX_IFT_SOLVER_CACHE = _build_jax_ift_solver_impl()
        return _JAX_IFT_SOLVER_CACHE


def _build_jax_ift_solver_impl():
    """Worker for :func:`_build_jax_ift_solver` -- runs the actual JAX
    ``custom_vjp`` decoration.  Called under
    ``_JAX_IFT_SOLVER_CACHE_LOCK`` exactly once per process even
    under concurrent first calls (the wrapper handles the double-
    check + assignment).
    """
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
    # v4.14.2: the cache assignment moved to the wrapper
    # (_build_jax_ift_solver) so it happens atomically under the
    # lock.  Just return the decorated solver here.
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
