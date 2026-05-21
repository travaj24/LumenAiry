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

Module layout (v5.1.0 Agent D file-split, ROADMAP item)
-------------------------------------------------------

The implementation lives in five submodules:

* :mod:`.asymptotic_modes` -- LG/HG polynomials and mode-stack caches.
* :mod:`.asymptotic_canonical_fit` -- ``CanonicalPolyFit`` /
  ``HFPolyFit`` and their trace-and-fit pipelines plus
  ``solve_envelope_stationary`` and ``propagate_hf_chebyshev_quadrature``.
* :mod:`.asymptotic_aberration_tensor` -- ``AberrationTensorResult`` and
  ``aberration_tensor``.
* :mod:`.asymptotic_maslov` -- Maslov-branch ``sqrt(det M)`` helper and
  the batched-evaluator helpers consumed by ``propagate_modal_asymptotic``.
* :mod:`.asymptotic_jax_twin` -- JAX paths
  (``aberration_tensor_lg00_jax``, ``propagate_modal_asymptotic_lg00_jax``,
  ``solve_envelope_stationary_jax_ift``, ``fit_canonical_polynomials_jax``)
  plus the backend-aware Chebyshev evaluators.

The split is purely mechanical -- every previously-public name remains
importable from ``lumenairy.propagators.asymptotic`` and behaves
bit-for-bit identically.  ``propagate_modal_asymptotic`` is defined
*in this shell* (not in a submodule) so its body resolves
``_solve_envelope_stationary_batch`` and the batched helpers against
this module's globals.  This preserves the pre-v5.1.0 monkey-patch
contract relied upon by
``tests/unit/test_audit_fixes_v4_14_1_agent_a.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .asymptotic_aberration_tensor import (  # noqa: F401
    AberrationTensorResult,
    _compute_M_b,
    _contract_against_moment_table,
    _multiply_polys_2d,
    _phi_v2_hessian,
    _polynomial_substitute_linear_2d,
    _polynomial_under_affine_shift,
    aberration_tensor,
)
from .asymptotic_canonical_fit import (  # noqa: F401
    CanonicalPolyFit,
    HFPolyFit,
    _eval_4d_cross_deriv,
    fit_canonical_polynomials,
    fit_hf_polynomials,
    propagate_hf_chebyshev_quadrature,
    solve_envelope_stationary,
)
from .asymptotic_jax_twin import (  # noqa: F401
    _JAX_IFT_SOLVER_CACHE,
    _JAX_IFT_SOLVER_CACHE_LOCK,
    JaxAberrationTensorResult,
    _build_jax_ift_solver,
    _build_jax_ift_solver_impl,
    _chebyshev_vandermonde_xp,
    _compute_M_b_xp,
    _evaluate_polynomial_4d_xp,
    _modal_field_lg00_pixel_jax,
    aberration_tensor_lg00_jax,
    fit_canonical_polynomials_jax,
    propagate_modal_asymptotic_lg00_jax,
    solve_envelope_stationary_jax_ift,
)
from .asymptotic_maslov import (  # noqa: F401
    _batched_polynomial_substitute_linear_2d,
    _batched_polynomial_under_affine_shift,
    _compute_M_b_batch,
    _gaussian_moment_table_2d_batch,
    _maslov_branch_corrected_sqrt,
    _phi_v2_hessian_batch,
    _poly_dict_to_array,
    _solve_envelope_stationary_batch,
)

# Re-export submodule contents so existing call sites continue to
# work unchanged (``from lumenairy.propagators.asymptotic import X``).
from .asymptotic_modes import (  # noqa: F401
    _HG_MODE_STACK_CACHE,
    _HG_MODE_STACK_CACHE_MAX,
    _HG_MODE_STACK_LOCK,
    _LG_MODE_STACK_CACHE,
    _LG_MODE_STACK_CACHE_MAX,
    _LG_MODE_STACK_LOCK,
    _evaluate_poly2d,
    _hg_mode_conj_stack,
    _lg_mode_conj_stack,
    # Private helpers / caches referenced by tests + internal sites.
    _lg_polynomial_items,
    clear_lg_mode_stack_cache,
    clear_lg_polynomial_cache,
    decompose_hg,
    decompose_lg,
    evaluate_hg_mode,
    evaluate_lg_mode,
    gaussian_moment_2d,
    gaussian_moment_table_2d,
    hg_polynomial,
    # Public modal API
    lg_polynomial,
    lg_seidel_label,
)

__all__ = [
    # Data containers
    'CanonicalPolyFit',
    'HFPolyFit',
    'AberrationTensorResult',
    # Modes / polynomial coefficients
    'lg_polynomial',
    'clear_lg_polynomial_cache',
    'clear_lg_mode_stack_cache',
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
    # JAX paths
    'aberration_tensor_lg00_jax',
    'propagate_modal_asymptotic_lg00_jax',
    'JaxAberrationTensorResult',
    'solve_envelope_stationary_jax_ift',
    'fit_canonical_polynomials_jax',
]


# ===========================================================================
# Section 6 -- Modal asymptotic propagator (leading-order)
# ===========================================================================
#
# ``propagate_modal_asymptotic`` is kept in the shell (not moved to
# ``asymptotic_maslov``) so its globals are this module's globals.
# Pre-v5.1.0 ``tests/unit/test_audit_fixes_v4_14_1_agent_a.py`` (the
# v4.15 cold-start invariant pin) monkey-patches
# ``_asy._solve_envelope_stationary_batch`` on the shell module and
# expects ``propagate_modal_asymptotic`` to honour the patch.
# Moving the function to ``asymptotic_maslov.py`` would break this
# contract because Python resolves intra-module name lookups against
# the function's ``__globals__`` (the module the function is defined
# in) -- patches on the shell would no longer take effect.  Leaving
# the body here keeps the contract intact while still pushing the
# batched/Maslov machinery into ``asymptotic_maslov.py``.


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
          but resets the unwrap state at the start of each row.

    Returns
    -------
    ndarray, complex, same shape as s2_grid_x
        Output field E(s_2).

    Notes
    -----
    **v4.15 vectorisation closure.**  Prior to v4.15 this function
    used a per-pixel Python loop with a warm-started Newton chain.
    The chain landed in *wrong-saddle basins* near the grid edges,
    where the overflow guard ``|b_quad| > 700`` silently zeroed those
    pixels.  v4.15 switches to a single batched call into
    :func:`_solve_envelope_stationary_batch` followed by
    :func:`_compute_M_b_batch` and the batched polynomial-substitution
    helpers (cold-start Newton for every pixel from ``v2_centre``).
    The new body finds the physical saddle uniformly and produces
    strictly more non-zero pixels at grid edges -- physically MORE
    correct, but breaks the pre-v4.15 bit-equal pin against the
    warm-start reference (the v4.12.0 and v4.14.0 bit-equal pins
    were relaxed to property pins in the same v4.15 patch; see
    ``docs/audits/AUDIT_V4_14_2_2026_05_17.md`` Part 3.5 closure and
    ``docs/release_notes/.release_notes_v4_15_agent_a.md``).

    The Maslov branch-unwrap is still a sequential O(N) pass over
    the valid pixels (the unwrap is fundamentally serial; only the
    Newton + M/b + polynomial-substitution work is batched).
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

    # Determine moment-table order needed
    max_order_src = max(
        (2 * p + abs(l) for (p, l) in source_amplitudes), default=0)
    max_order_pup = max(
        (2 * p + abs(l) for (p, l) in pupil_amplitudes), default=0)
    max_order_needed = max(max_order_src + max_order_pup, 0)

    # 4.12.0 (perf #5): hoist the LG basis polynomials out of the
    # per-pixel Newton loop.  ``lg_polynomial(p, ell, w)`` is a pure
    # function of its arguments -- it returned identical dicts on every
    # pixel pre-4.12.0 (and was the dominant cost in multi-mode
    # propagation, scaling with mode count * N_pixels).
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
    N_pix = flat_x.size
    if N_pix == 0:
        return flat_out.reshape(s2x_arr.shape)

    Ny_grid, Nx_grid = (s2x_arr.shape if s2x_arr.ndim >= 2
                         else (1, s2x_arr.size))

    # ------------------------------------------------------------------
    # v4.15 (ROADMAP #1) -- batched cold-start Newton + batched M/b/poly
    # ------------------------------------------------------------------
    # Per-pixel ``continue`` from the scalar path becomes element-wise
    # ``valid`` masking here.

    # ---- Fit-box mask on s2 ------------------------------------------
    u1 = (flat_x - fit.s2x_centre) / fit.s2x_halfrange
    u2 = (flat_y - fit.s2y_centre) / fit.s2y_halfrange
    in_box_s2 = (np.abs(u1) <= 1.0) & (np.abs(u2) <= 1.0)

    # ---- Batched cold-start Newton solve ----------------------------
    try:
        v2x_star, v2y_star, _conv = _solve_envelope_stationary_batch(
            fit, flat_x, flat_y, src_x, src_y,
            w_s=w_s, w_p=w_p, v_cx=v_cx, v_cy=v_cy,
        )
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        return flat_out.reshape(s2x_arr.shape)

    # Replace any non-finite Newton outputs with v_centre to keep the
    # downstream M/b batch evaluation in-domain.
    bad_v = (~np.isfinite(v2x_star)) | (~np.isfinite(v2y_star))
    if np.any(bad_v):
        v2x_star = np.where(bad_v, v_cx, v2x_star)
        v2y_star = np.where(bad_v, v_cy, v2y_star)

    u3 = (v2x_star - fit.v2x_centre) / fit.v2x_halfrange
    u4 = (v2y_star - fit.v2y_centre) / fit.v2y_halfrange
    in_box_v = ((np.abs(u3) <= 1.0) & (np.abs(u4) <= 1.0)
                & (~bad_v))

    valid = in_box_s2 & in_box_v
    if not np.any(valid):
        return flat_out.reshape(s2x_arr.shape)

    # ---- Batched M, b, J*, phi*, G0, detJ ---------------------------
    try:
        M_all, b_all, s1_star_all, J_all, phi_star_all, G0_all, detJ_all = (
            _compute_M_b_batch(
                fit, flat_x, flat_y, v2x_star, v2y_star,
                src_x, src_y, w_s, w_p, v_cx, v_cy,
            )
        )
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        return flat_out.reshape(s2x_arr.shape)

    finite_M = np.all(np.isfinite(M_all.real) & np.isfinite(M_all.imag),
                      axis=(-2, -1))
    finite_b = np.all(np.isfinite(b_all.real) & np.isfinite(b_all.imag),
                      axis=-1)
    valid = valid & finite_M & finite_b
    if not np.any(valid):
        return flat_out.reshape(s2x_arr.shape)

    # ---- Closed-form 2x2 det / inverse with sentinel masking --------
    a00 = M_all[:, 0, 0]
    a01 = M_all[:, 0, 1]
    a10 = M_all[:, 1, 0]
    a11 = M_all[:, 1, 1]
    det_M = a00 * a11 - a01 * a10
    abs_det = np.abs(det_M)
    ok_det = np.isfinite(abs_det) & (abs_det >= 1e-300)
    valid = valid & ok_det
    if not np.any(valid):
        return flat_out.reshape(s2x_arr.shape)

    safe_det = np.where(ok_det, det_M, 1.0 + 0.0j)
    inv_det = 1.0 / safe_det
    M_inv_all = np.empty_like(M_all)
    M_inv_all[:, 0, 0] = a11 * inv_det
    M_inv_all[:, 0, 1] = -a01 * inv_det
    M_inv_all[:, 1, 0] = -a10 * inv_det
    M_inv_all[:, 1, 1] = a00 * inv_det
    finite_Minv = np.all(np.isfinite(M_inv_all.real)
                          & np.isfinite(M_inv_all.imag),
                          axis=(-2, -1))
    valid = valid & finite_Minv

    # ---- b_quad = 0.25 * b^T M^-1 b ---------------------------------
    Minv_b = np.einsum('nij,nj->ni', M_inv_all, b_all)
    b_quad = 0.25 * (b_all[:, 0] * Minv_b[:, 0]
                      + b_all[:, 1] * Minv_b[:, 1])
    delta_star_all = 0.5 * Minv_b
    ok_bquad = np.isfinite(np.abs(b_quad)) & (np.abs(b_quad.real) <= 700.0)
    valid = valid & ok_bquad
    if not np.any(valid):
        return flat_out.reshape(s2x_arr.shape)

    # ---- sqrt(det M) with Maslov-branch unwrap ----------------------
    if maslov_tracking == 'principal':
        sqrt_detM_all = np.sqrt(det_M)
    else:
        sqrt_detM_all = np.empty_like(det_M)
        last_arg = None
        branch_counter = 0
        for idx in range(N_pix):
            if (maslov_tracking == 'row_reset' and s2x_arr.ndim >= 2
                    and Nx_grid > 0 and idx % Nx_grid == 0):
                last_arg = None
                branch_counter = 0
            if not valid[idx]:
                sqrt_detM_all[idx] = np.sqrt(det_M[idx])
                continue
            sqrt_val, last_arg, branch_counter = (
                _maslov_branch_corrected_sqrt(
                    det_M[idx],
                    last_arg_detM=last_arg,
                    maslov_branch=branch_counter,
                )
            )
            sqrt_detM_all[idx] = sqrt_val

    # ---- amp_lead = detJ * pi/sqrt(det M) * G0 * exp(2*pi*i*phi) * exp(b_quad)
    # v4.14.2 dtype-aware sentinel (np.zeros((), dtype=...)) per
    # ``test_no_unguarded_zero_plus_zeroj_in_np_where`` meta-pin.
    sqrt_dtype = sqrt_detM_all.dtype
    bquad_dtype = b_quad.dtype
    phi_dtype = phi_star_all.dtype
    safe_sqrt = np.where(sqrt_detM_all != 0, sqrt_detM_all,
                          np.ones((), dtype=sqrt_dtype))
    safe_bquad = np.where(ok_bquad, b_quad,
                           np.zeros((), dtype=bquad_dtype))
    safe_phi = np.where(np.isfinite(phi_star_all), phi_star_all,
                         np.zeros((), dtype=phi_dtype))
    amp_lead_all = (detJ_all
                    * (math.pi / safe_sqrt)
                    * G0_all
                    * np.exp(2j * math.pi * safe_phi)
                    * np.exp(safe_bquad))
    ok_amp = np.isfinite(np.abs(amp_lead_all))
    valid = valid & ok_amp
    if not np.any(valid):
        return flat_out.reshape(s2x_arr.shape)

    # ---- Batched eta moment table -----------------------------------
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return flat_out.reshape(s2x_arr.shape)
    M_valid = M_all[valid_idx]
    moment_keys, moment_table_valid = _gaussian_moment_table_2d_batch(
        M_valid, max_order_needed,
    )
    if moment_keys:
        max_i_mom = max(k[0] for k in moment_keys)
        max_j_mom = max(k[1] for k in moment_keys)
    else:
        max_i_mom = 0
        max_j_mom = 0
    N_valid = valid_idx.size
    moment_arr = np.zeros((N_valid, max_i_mom + 1, max_j_mom + 1),
                           dtype=np.complex128)
    for q, (a, b_ord) in enumerate(moment_keys):
        moment_arr[:, a, b_ord] = moment_table_valid[:, q]

    # ---- r_const, pupil_const on valid subset -----------------------
    J_valid = J_all[valid_idx]
    delta_valid = delta_star_all[valid_idx]
    s1_valid = s1_star_all[valid_idx]
    v2x_valid = v2x_star[valid_idx]
    v2y_valid = v2y_star[valid_idx]
    J_dot_delta = np.einsum('nij,nj->ni', J_valid.astype(np.complex128),
                              delta_valid)
    src_vec = np.array([src_x, src_y], dtype=np.complex128)
    r_const = (s1_valid.astype(np.complex128) + J_dot_delta - src_vec)
    pupil_const = np.empty((N_valid, 2), dtype=np.complex128)
    pupil_const[:, 0] = v2x_valid + delta_valid[:, 0] - v_cx
    pupil_const[:, 1] = v2y_valid + delta_valid[:, 1] - v_cy

    # ---- Sum over LG mode pairs via batched polynomial helpers ------
    E_pixel = np.zeros(N_valid, dtype=np.complex128)
    src_arr_cache: Dict[Tuple[int, int], np.ndarray] = {}
    pup_arr_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for k_src in src_poly_r_cache:
        src_d = src_poly_r_cache[k_src]
        if not src_d:
            continue
        ix_max = max(k[0] for k in src_d)
        iy_max = max(k[1] for k in src_d)
        src_arr_cache[k_src] = _poly_dict_to_array(src_d, ix_max, iy_max)
    for k_pup in pup_poly_r_cache:
        pup_d = pup_poly_r_cache[k_pup]
        if not pup_d:
            continue
        ix_max = max(k[0] for k in pup_d)
        iy_max = max(k[1] for k in pup_d)
        pup_arr_cache[k_pup] = _poly_dict_to_array(pup_d, ix_max, iy_max)

    A_valid = J_valid.astype(np.complex128)  # (N_valid, 2, 2)

    for k_src, a_src in source_amplitudes.items():
        if abs(a_src) < 1e-300:
            continue
        if k_src not in src_arr_cache:
            continue
        src_arr = src_arr_cache[k_src]
        src_poly_eta_batch = _batched_polynomial_substitute_linear_2d(
            src_arr, A_valid, r_const,
        )
        for k_pup, b_pup in pupil_amplitudes.items():
            if abs(b_pup) < 1e-300:
                continue
            if k_pup not in pup_arr_cache:
                continue
            pup_arr = pup_arr_cache[k_pup]
            pup_poly_eta_batch = _batched_polynomial_under_affine_shift(
                pup_arr, pupil_const[:, 0], pupil_const[:, 1],
            )
            Ix_src = src_poly_eta_batch.shape[1] - 1
            Iy_src = src_poly_eta_batch.shape[2] - 1
            Ix_pup = pup_poly_eta_batch.shape[1] - 1
            Iy_pup = pup_poly_eta_batch.shape[2] - 1
            need_i = Ix_src + Ix_pup
            need_j = Iy_src + Iy_pup
            have_i = moment_arr.shape[1] - 1
            have_j = moment_arr.shape[2] - 1
            if need_i > have_i or need_j > have_j:
                padded = np.zeros(
                    (N_valid, max(have_i + 1, need_i + 1),
                     max(have_j + 1, need_j + 1)),
                    dtype=np.complex128,
                )
                padded[:, :have_i + 1, :have_j + 1] = moment_arr
                mom_used = padded
            else:
                mom_used = moment_arr
            exp_val = np.zeros(N_valid, dtype=np.complex128)
            for a in range(Ix_src + 1):
                for b_idx in range(Iy_src + 1):
                    src_coef = src_poly_eta_batch[:, a, b_idx]
                    if not np.any(src_coef):
                        continue
                    for c in range(Ix_pup + 1):
                        for d in range(Iy_pup + 1):
                            pup_coef = pup_poly_eta_batch[:, c, d]
                            if not np.any(pup_coef):
                                continue
                            exp_val += (src_coef * pup_coef
                                         * mom_used[:, a + c, b_idx + d])
            E_pixel += a_src * b_pup * exp_val

    # ---- Scatter back to the full output grid -----------------------
    flat_out[valid_idx] = amp_lead_all[valid_idx] * E_pixel
    finite_out = np.isfinite(flat_out.real) & np.isfinite(flat_out.imag)
    flat_out = np.where(finite_out, flat_out,
                         np.zeros((), dtype=flat_out.dtype))

    return flat_out.reshape(s2x_arr.shape)
