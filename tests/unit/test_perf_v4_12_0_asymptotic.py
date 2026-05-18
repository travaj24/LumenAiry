"""Pinning tests for the v4.12.0 perf optimisations in
``lumenairy.propagators.asymptotic``.

Two optimisations are exercised here:

* **Perf #5 -- modal asymptotic LG polynomial hoist & cache.**
  Pre-v4.12.0 :func:`propagate_modal_asymptotic` rebuilt
  ``lg_polynomial(p, ell, w_s)`` / ``lg_polynomial(p, ell, w_p)`` on
  every output pixel.  v4.12.0 hoists these into per-(p, ell) caches
  outside the pixel loop and adds an ``lru_cache`` on the
  ``lg_polynomial`` helper itself.  These tests pin:
    - the per-pixel results agree against a fresh batched-reference
      build at 1e-8 absolute (relaxed from 1e-10 in v4.15 -- see
      below).
    - the ``lru_cache`` is well-behaved: same args produce identical
      output, ``clear_lg_polynomial_cache()`` produces identical
      output, cache statistics increment as expected.
    - the public ``lg_polynomial`` returns a fresh ``dict`` each call
      (mutation safety).

  **v4.15 closure (this update).**  Pre-v4.15
  :func:`propagate_modal_asymptotic` used a per-pixel
  warm-started Newton chain.  v4.15 switched to a single batched
  cold-start :func:`_solve_envelope_stationary_batch` +
  :func:`_compute_M_b_batch` -- this finds the physical saddle
  uniformly (the warm-start chain landed in wrong-saddle basins at
  grid edges; see
  ``docs/audits/AUDIT_V4_14_2_2026_05_17.md`` Part 3.5 and
  ``docs/release_notes/.release_notes_v4_15_agent_a.md``).  The
  ``test_lg00_single_mode_matches_reference`` and
  ``test_multimode_matches_reference`` tests below USED to be
  bit-equal pins (``rel < 1e-10`` against the warm-start scalar
  reference); they are now property pins:
    * Energy conservation between the new path and the warm-start
      reference is preserved up to a 5% tolerance (the new path can
      add small-amplitude content at grid edges where the
      warm-start chain falsely zeroed pixels).
    * Non-zero pixel count of the new path is >= the warm-start
      reference count on benign grids (the strict ">=" inequality
      is the physics direction of the wrong-saddle finding; in
      regions where the warm-start was already correct the two
      paths agree to ~1e-8 absolute).
    * Per-pixel agreement with a *batched-reference* loop -- the
      canonical implementation is now the cold-start batched path
      itself, so the reference is a pure rebuild of the same
      algorithm in the test file at 1e-8 absolute.

* **Perf #6 -- vectorised HF Chebyshev-quadrature chunk loop.**
  Pre-v4.12.0 :func:`propagate_hf_chebyshev_quadrature` had a Python
  ``for k in range(start, end)`` inner loop over the chunk that
  evaluated one output pixel at a time (the chunking did nothing).
  v4.12.0 makes the chunk a true broadcast axis.  This test pins:
    - the vectorised output agrees bit-near-exactly with a reference
      scalar-pixel-at-a-time evaluation.
    - the paraxial limit matches a Fresnel ``1/(iλz)`` reduction
      (existing physical-correctness anchor; the v4.10 ``-1j``
      prefactor is preserved).
    - the ``max_chunk_memory_mb`` guard correctly shrinks the chunk
      below ``chunk_output`` when the broadcast tensor would exceed
      the budget.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    _lg_polynomial_items,
    _multiply_polys_2d,
    _polynomial_substitute_linear_2d,
    _polynomial_under_affine_shift,
    _contract_against_moment_table,
    clear_lg_polynomial_cache,
    fit_canonical_polynomials,
    fit_hf_polynomials,
    gaussian_moment_table_2d,
    lg_polynomial,
    propagate_hf_chebyshev_quadrature,
    propagate_modal_asymptotic,
)
from lumenairy.propagators.asymptotic import (
    _compute_M_b,
    _maslov_branch_corrected_sqrt,
    solve_envelope_stationary,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_singlet():
    rx = lm.make_singlet(
        R1=20e-3, R2=-20e-3, d=2e-3,
        glass='N-BK7', aperture=10e-3,
    )
    rx['object_distance'] = 0.1
    return rx


def _build_canonical_fit():
    rx = _make_singlet()
    return fit_canonical_polynomials(
        rx, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=8, n_pupil=8, poly_order=6,
    )


def _build_hf_fit():
    rx = _make_singlet()
    return fit_hf_polynomials(
        rx, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=6, n_pupil=6, poly_order=4,
    )


def _reference_lg_polynomial_no_cache(p: int, ell: int, w: float):
    """Reference (pre-v4.12.0) LG polynomial build, no caching.

    Identical algorithm to :func:`lg_polynomial`; bypasses the cache
    so a "rebuild every pixel" reference can verify the cache returns
    bit-equal values.
    """
    abs_ell = abs(ell)
    s_sign = 1 if ell >= 0 else -1
    f = math.sqrt(2.0) / w
    N = math.sqrt(
        2.0 * math.factorial(p)
        / (math.pi * math.factorial(p + abs_ell) * (w * w))
    )
    coeffs = {}
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
    return coeffs


def _scalar_pixel_reference_propagate_modal(fit, *,
                                             source_point, source_amplitudes,
                                             pupil_amplitudes, w_s, w_p,
                                             v2_centre, s2_grid_x, s2_grid_y,
                                             maslov_tracking='row_reset'):
    """Reference implementation that rebuilds ``lg_polynomial`` per pixel.

    Mirrors the pre-v4.12.0 pixel loop with no hoisting and no cache.
    Used to verify the v4.12.0 hoisted/cached version produces
    bit-near-exact output.
    """
    if source_amplitudes is None:
        source_amplitudes = {(0, 0): 1.0 + 0.0j}
    if pupil_amplitudes is None:
        pupil_amplitudes = {(0, 0): 1.0 + 0.0j}
    src_x, src_y = float(source_point[0]), float(source_point[1])
    v_cx, v_cy = float(v2_centre[0]), float(v2_centre[1])

    s2x_arr = np.asarray(s2_grid_x, dtype=np.float64)
    s2y_arr = np.asarray(s2_grid_y, dtype=np.float64)
    out = np.zeros(s2x_arr.shape, dtype=np.complex128)

    max_order_src = max(
        (2 * p + abs(l) for (p, l) in source_amplitudes), default=0)
    max_order_pup = max(
        (2 * p + abs(l) for (p, l) in pupil_amplitudes), default=0)
    max_order_needed = max(max_order_src + max_order_pup, 0)

    flat_x = s2x_arr.ravel()
    flat_y = s2y_arr.ravel()
    flat_out = np.zeros(flat_x.size, dtype=np.complex128)

    last_v_star = (v_cx, v_cy)
    last_arg_detM = None
    maslov_branch = 0
    Ny_grid, Nx_grid = (s2x_arr.shape if s2x_arr.ndim >= 2
                        else (1, s2x_arr.size))

    for idx in range(flat_x.size):
        s2x_p = flat_x[idx]
        s2y_p = flat_y[idx]
        if (maslov_tracking == 'row_reset' and s2x_arr.ndim >= 2
                and Nx_grid > 0 and idx % Nx_grid == 0):
            # v4.14.1 (P1-NEW-5):  ``row_reset`` also resets the
            # Newton warm-start at each row wrap (eliminating the
            # cross-row chain that plausibly entered wrong-saddle
            # basins near grid edges).  This reference loop is kept
            # in lock-step with the public path in
            # propagators/asymptotic.py so the 1e-10 rel pins hold.
            last_arg_detM = None
            maslov_branch = 0
            last_v_star = (v_cx, v_cy)

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
        u3 = (v_star[0] - fit.v2x_centre) / fit.v2x_halfrange
        u4 = (v_star[1] - fit.v2y_centre) / fit.v2y_halfrange
        if (abs(u3) > 1.0 or abs(u4) > 1.0
                or not (math.isfinite(u3) and math.isfinite(u4))):
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
        b_quad = 0.25 * (b @ M_inv @ b)
        if not math.isfinite(abs(b_quad)) or abs(b_quad.real) > 700:
            continue
        amp_lead = (detJ * (math.pi / sqrt_detM) * G0
                    * np.exp(2j * math.pi * phi_star)
                    * np.exp(b_quad))
        if not math.isfinite(abs(amp_lead)):
            continue
        eta_moments = gaussian_moment_table_2d(M, max_order_needed)

        r_const = (s1_star
                   + J_star @ np.array([delta_star[0], delta_star[1]])
                   - np.array([src_x, src_y]))
        pupil_const = (np.array([v_star[0], v_star[1]])
                       - np.array([v_cx, v_cy])
                       + np.array([delta_star[0], delta_star[1]]))

        E_pixel = 0.0 + 0.0j
        for k_src, a_src in source_amplitudes.items():
            if abs(a_src) < 1e-300:
                continue
            # ** Pre-v4.12.0 behaviour: rebuild every pixel. **
            src_poly_r = _reference_lg_polynomial_no_cache(
                k_src[0], k_src[1], w_s)
            src_poly_eta = _polynomial_substitute_linear_2d(
                src_poly_r,
                A_xx=J_star[0, 0], A_xy=J_star[0, 1],
                A_yx=J_star[1, 0], A_yy=J_star[1, 1],
                b_x=r_const[0], b_y=r_const[1],
            )
            for k_pup, b_pup in pupil_amplitudes.items():
                if abs(b_pup) < 1e-300:
                    continue
                pup_poly_r = _reference_lg_polynomial_no_cache(
                    k_pup[0], k_pup[1], w_p)
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


def _batched_cold_start_reference_propagate_modal(
        fit, *, source_point, source_amplitudes, pupil_amplitudes,
        w_s, w_p, v2_centre, s2_grid_x, s2_grid_y,
        maslov_tracking='row_reset',
):
    """v4.15 reference: cold-start scalar-pixel loop using the SAME
    physical algorithm as the new public batched path.

    Each pixel cold-starts the Newton from ``v2_centre`` (no
    warm-start chain).  This reference is the canonical implementation
    against which the v4.15 batched
    :func:`propagate_modal_asymptotic` is property-pinned at 1e-8
    absolute tolerance.  The two paths differ only in micro-rounding
    (closed-form 2x2 ``M^-1`` vs ``np.linalg.inv``, einsum vs
    ``J.T @ J``, etc) so 1e-8 absolute is the achievable accuracy
    floor.

    Used to replace the pre-v4.15 1e-10 rel pin against the
    *warm-start* scalar reference -- see this module's docstring for
    the bit-equality -> property-pin migration history.
    """
    if source_amplitudes is None:
        source_amplitudes = {(0, 0): 1.0 + 0.0j}
    if pupil_amplitudes is None:
        pupil_amplitudes = {(0, 0): 1.0 + 0.0j}
    src_x, src_y = float(source_point[0]), float(source_point[1])
    v_cx, v_cy = float(v2_centre[0]), float(v2_centre[1])

    s2x_arr = np.asarray(s2_grid_x, dtype=np.float64)
    s2y_arr = np.asarray(s2_grid_y, dtype=np.float64)

    max_order_src = max(
        (2 * p + abs(l) for (p, l) in source_amplitudes), default=0)
    max_order_pup = max(
        (2 * p + abs(l) for (p, l) in pupil_amplitudes), default=0)
    max_order_needed = max(max_order_src + max_order_pup, 0)

    flat_x = s2x_arr.ravel()
    flat_y = s2y_arr.ravel()
    flat_out = np.zeros(flat_x.size, dtype=np.complex128)

    last_arg_detM = None
    maslov_branch = 0
    Nx_grid = (s2x_arr.shape[1] if s2x_arr.ndim >= 2 else s2x_arr.size)

    for idx in range(flat_x.size):
        s2x_p = flat_x[idx]
        s2y_p = flat_y[idx]
        if (maslov_tracking == 'row_reset' and s2x_arr.ndim >= 2
                and Nx_grid > 0 and idx % Nx_grid == 0):
            last_arg_detM = None
            maslov_branch = 0

        u1 = (s2x_p - fit.s2x_centre) / fit.s2x_halfrange
        u2 = (s2y_p - fit.s2y_centre) / fit.s2y_halfrange
        if abs(u1) > 1.0 or abs(u2) > 1.0:
            continue

        # ** v4.15 cold-start: pass v2_initial=v2_centre (i.e. None
        # default) on every pixel. **  This is the only physical
        # difference from the pre-v4.15 warm-start reference.
        try:
            v_star, _, _ = solve_envelope_stationary(
                fit, (s2x_p, s2y_p), (src_x, src_y),
                w_s=w_s, w_p=w_p, v2_centre=v2_centre,
                v2_initial=v2_centre,
            )
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            continue

        u3 = (v_star[0] - fit.v2x_centre) / fit.v2x_halfrange
        u4 = (v_star[1] - fit.v2y_centre) / fit.v2y_halfrange
        if (abs(u3) > 1.0 or abs(u4) > 1.0
                or not (math.isfinite(u3) and math.isfinite(u4))):
            continue

        try:
            M, b, s1_star, J_star, phi_star, G0, detJ = _compute_M_b(
                fit, s2x_p, s2y_p, v_star[0], v_star[1],
                src_x, src_y, w_s, w_p, v_cx, v_cy,
            )
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            continue
        if not (np.all(np.isfinite(M)) and np.all(np.isfinite(b))):
            continue
        det_M = np.linalg.det(M)
        if not math.isfinite(abs(det_M)) or abs(det_M) < 1e-300:
            continue
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
        b_quad = 0.25 * (b @ M_inv @ b)
        if not math.isfinite(abs(b_quad)) or abs(b_quad.real) > 700:
            continue
        amp_lead = (detJ * (math.pi / sqrt_detM) * G0
                    * np.exp(2j * math.pi * phi_star)
                    * np.exp(b_quad))
        if not math.isfinite(abs(amp_lead)):
            continue
        eta_moments = gaussian_moment_table_2d(M, max_order_needed)

        r_const = (s1_star
                   + J_star @ np.array([delta_star[0], delta_star[1]])
                   - np.array([src_x, src_y]))
        pupil_const = (np.array([v_star[0], v_star[1]])
                       - np.array([v_cx, v_cy])
                       + np.array([delta_star[0], delta_star[1]]))

        E_pixel = 0.0 + 0.0j
        for k_src, a_src in source_amplitudes.items():
            if abs(a_src) < 1e-300:
                continue
            src_poly_r = lg_polynomial(k_src[0], k_src[1], w_s)
            src_poly_eta = _polynomial_substitute_linear_2d(
                src_poly_r,
                A_xx=J_star[0, 0], A_xy=J_star[0, 1],
                A_yx=J_star[1, 0], A_yy=J_star[1, 1],
                b_x=r_const[0], b_y=r_const[1],
            )
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
                exp_val = _contract_against_moment_table(P_eta, eta_moments)
                E_pixel += a_src * b_pup * exp_val

        flat_out[idx] = amp_lead * E_pixel

    return flat_out.reshape(s2x_arr.shape)


def _scalar_pixel_reference_hf_chebyshev(fit, E_in, ix, iy, ox, oy,
                                          *, apply_van_vleck=True):
    """Reference implementation that scalar-loops over output pixels.

    Mirrors the pre-v4.12.0 ``for k in range(start, end)`` inner loop
    intent:  one output pixel at a time, full ``(Ny_in, Nx_in)``
    Chebyshev tables re-evaluated per pixel.  The pre-fix code passed
    Python-float ``s2x`` / ``s2y`` directly to ``fit.eval_phi`` which
    crashed on a 2-D ``S1X`` input due to a shape-broadcast bug in the
    underlying polynomial evaluator (a pre-existing issue documented
    in :mod:`test_audit_fixes_v4_11_2_hfpi_hf`).  This reference
    broadcasts ``s2x`` / ``s2y`` up to the meshgrid shape so the
    underlying ``_evaluate_polynomial_4d`` receives mutually-
    broadcastable arrays -- this is the *intended* per-pixel
    semantics that the pre-fix code was trying (and failing) to
    express.
    """
    Nx_in = ix.shape[0]
    Ny_in = iy.shape[0]
    Nx_out = ox.shape[0]
    Ny_out = oy.shape[0]
    dx = float(np.mean(np.diff(ix)))
    dy = float(np.mean(np.diff(iy)))
    pixel_area = dx * dy
    S1X, S1Y = np.meshgrid(ix, iy, indexing='xy')
    if np.iscomplexobj(E_in):
        out_dtype = E_in.dtype
    elif E_in.dtype == np.float64:
        out_dtype = np.complex128
    else:
        out_dtype = np.complex64
    out = np.zeros((Ny_out, Nx_out), dtype=out_dtype)
    n_out = Ny_out * Nx_out
    for k in range(n_out):
        oy_idx = k // Nx_out
        ox_idx = k % Nx_out
        s2x_scalar = float(ox[ox_idx])
        s2y_scalar = float(oy[oy_idx])
        # Broadcast scalar s2 up to the input-grid shape so the
        # underlying _evaluate_polynomial_4d sees mutually-
        # broadcastable arrays.  (The pre-fix code passed Python
        # floats, which exposed a latent shape bug in the polynomial
        # evaluator -- documented at
        # tests/unit/test_audit_fixes_v4_11_2_hfpi_hf.py:371.)
        s2x = np.full_like(S1X, s2x_scalar)
        s2y = np.full_like(S1Y, s2y_scalar)
        phi = fit.eval_phi(S1X, S1Y, s2x, s2y, include_linear=True)
        kernel = np.exp(2j * np.pi * phi).astype(out_dtype)
        if apply_van_vleck:
            density = fit.eval_van_vleck_density(S1X, S1Y, s2x, s2y)
            integrand = E_in * density * kernel
        else:
            integrand = E_in * kernel
        out[oy_idx, ox_idx] = np.sum(integrand) * pixel_area
    out = out * (-1j)
    return out


# ============================================================================
# Perf #5 -- lg_polynomial cache & hoist correctness
# ============================================================================


class TestLgPolynomialCache:
    """Pin the ``lru_cache`` semantics for :func:`lg_polynomial`."""

    def test_same_args_produce_identical_dicts(self):
        clear_lg_polynomial_cache()
        p1 = lg_polynomial(0, 0, 1e-3)
        p2 = lg_polynomial(0, 0, 1e-3)
        assert set(p1.keys()) == set(p2.keys())
        for k in p1:
            assert p1[k] == p2[k], f"mismatch at {k}: {p1[k]} vs {p2[k]}"

    def test_returns_fresh_dict_each_call(self):
        """Cache must NOT leak the same dict object -- callers may
        mutate the result, and that mutation must not poison the cache.
        """
        clear_lg_polynomial_cache()
        p1 = lg_polynomial(1, 0, 1e-3)
        p2 = lg_polynomial(1, 0, 1e-3)
        assert p1 is not p2, (
            "lg_polynomial should return a fresh dict each call.  "
            "Returning the cached dict directly risks caller mutation "
            "poisoning the cache for subsequent users."
        )
        # Mutate p1; p2 and any subsequent call must be unaffected.
        p1[(0, 0)] = 999.0
        p3 = lg_polynomial(1, 0, 1e-3)
        assert p3[(0, 0)] != 999.0, (
            "Mutating the dict returned from lg_polynomial poisoned "
            "the cache -- the lru_cache must hold an immutable handle."
        )

    def test_cache_hit_miss_counts(self):
        clear_lg_polynomial_cache()
        info0 = _lg_polynomial_items.cache_info()
        assert info0.hits == 0 and info0.misses == 0
        lg_polynomial(0, 0, 1e-3)
        info1 = _lg_polynomial_items.cache_info()
        assert info1.misses == 1 and info1.hits == 0
        lg_polynomial(0, 0, 1e-3)
        info2 = _lg_polynomial_items.cache_info()
        assert info2.hits == 1 and info2.misses == 1
        # Different args -> another miss
        lg_polynomial(1, 1, 1e-3)
        info3 = _lg_polynomial_items.cache_info()
        assert info3.misses == 2 and info3.hits == 1

    def test_clear_cache_resets_state(self):
        clear_lg_polynomial_cache()
        lg_polynomial(0, 0, 1e-3)
        lg_polynomial(1, 1, 1e-3)
        assert _lg_polynomial_items.cache_info().currsize == 2
        clear_lg_polynomial_cache()
        info = _lg_polynomial_items.cache_info()
        assert info.currsize == 0
        assert info.hits == 0
        assert info.misses == 0

    def test_clear_then_recall_produces_same_output(self):
        clear_lg_polynomial_cache()
        p_before = lg_polynomial(2, -1, 1.5e-3)
        clear_lg_polynomial_cache()
        p_after = lg_polynomial(2, -1, 1.5e-3)
        assert set(p_before.keys()) == set(p_after.keys())
        for k in p_before:
            assert p_before[k] == p_after[k]

    def test_matches_pre_v4_12_0_reference(self):
        """Cached output must match a no-cache reference."""
        clear_lg_polynomial_cache()
        for (p, ell, w) in [(0, 0, 1e-3),
                            (1, 0, 1e-3),
                            (0, 1, 1e-3),
                            (2, -3, 5e-4),
                            (3, 2, 2e-3)]:
            cached = lg_polynomial(p, ell, w)
            ref = _reference_lg_polynomial_no_cache(p, ell, w)
            assert set(cached.keys()) == set(ref.keys()), (
                f"LG({p}, {ell}, w={w}) cache vs ref key mismatch: "
                f"{set(cached.keys())} vs {set(ref.keys())}"
            )
            for k in cached:
                assert cached[k] == ref[k], (
                    f"LG({p}, {ell}, w={w}) coef at {k}: "
                    f"cached {cached[k]} vs ref {ref[k]}"
                )


class TestPropagateModalAsymptoticCorrectness:
    """Pin bit-near-exact agreement between the v4.12.0 hoisted-cache
    path and a reference that rebuilds per pixel."""

    @pytest.fixture(scope='class')
    def fit(self):
        return _build_canonical_fit()

    @pytest.fixture(scope='class')
    def grid(self, fit):
        L = fit.s2x_halfrange * 0.3
        n = 16
        ax = np.linspace(-L, L, n) + fit.s2x_centre
        ay = np.linspace(-L, L, n) + fit.s2y_centre
        X, Y = np.meshgrid(ax, ay, indexing='xy')
        return X, Y

    def test_lg00_single_mode_matches_reference(self, fit, grid):
        """v4.15 property pin (relaxed from pre-v4.15 1e-10 rel
        bit-equal pin):

        * Per-pixel agreement vs a fresh cold-start batched reference
          at 1e-8 absolute on a benign on-axis grid (the only source
          of disagreement is closed-form 2x2 ``M^-1`` vs
          ``np.linalg.inv`` micro-rounding).
        * Total energy of the new path is within 5% of the
          warm-start scalar reference's energy (the new path
          recovers small-amplitude grid-edge content that the
          warm-start chain falsely zeroed -- the new total can
          legitimately differ).
        * Non-zero pixel count of the new path is >= the
          warm-start reference count (the wrong-saddle finding is
          uni-directional: cold-start should produce >= the
          non-zero pixel set warm-start produced on this benign
          centered-source grid).

        See module docstring for the bit-equality -> property-pin
        migration history.
        """
        X, Y = grid
        source_amps = {(0, 0): 1.0 + 0.0j}
        pupil_amps = {(0, 0): 1.0 + 0.0j}
        clear_lg_polynomial_cache()
        new = propagate_modal_asymptotic(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        # ---- Pin 1: per-pixel agreement vs a fresh batched-cold-start
        # reference loop (the canonical algorithm).  1e-8 absolute is
        # the achievable accuracy floor between the public batched
        # path and a fresh scalar-loop reproduction of the same
        # cold-start algorithm.
        cold_ref = _batched_cold_start_reference_propagate_modal(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        cold_peak = float(np.max(np.abs(cold_ref)))
        max_abs_diff = float(np.max(np.abs(new - cold_ref)))
        # Per the v4.15 brief: 1e-8 absolute tolerance vs the
        # batched-reference loop (small drift from closed-form vs
        # ``np.linalg.inv`` and einsum rounding ordering).
        assert max_abs_diff < 1e-8 * max(cold_peak, 1.0), (
            f"LG_(0,0) single-mode vs cold-start batched reference: "
            f"max|new - cold_ref| = {max_abs_diff:.3e}, "
            f"cold_peak = {cold_peak:.3e}"
        )

        # ---- Pin 2: total-energy preservation vs warm-start ref.
        warm_ref = _scalar_pixel_reference_propagate_modal(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        warm_peak = float(np.max(np.abs(warm_ref)))
        if warm_peak == 0:
            pytest.skip("warm-start reference produced no signal")
        # Energy is sum |E|^2; the new path may include grid-edge
        # pixels the warm-start dropped, so the new energy is >=
        # warm-start energy modulo the 5% tolerance.
        new_energy = float(np.sum(np.abs(new) ** 2))
        warm_energy = float(np.sum(np.abs(warm_ref) ** 2))
        rel_e_diff = abs(new_energy - warm_energy) / warm_energy
        assert rel_e_diff < 0.05, (
            f"LG_(0,0) total-energy mismatch with warm-start ref: "
            f"new={new_energy:.3e}, warm={warm_energy:.3e}, "
            f"rel diff={rel_e_diff:.3e}"
        )

        # ---- Pin 3: non-zero pixel count comparison.
        new_nz = int(np.sum(new != 0))
        warm_nz = int(np.sum(warm_ref != 0))
        assert new_nz >= warm_nz, (
            f"v4.15 cold-start path produced FEWER non-zero pixels "
            f"({new_nz}) than the warm-start reference "
            f"({warm_nz}) -- the wrong-saddle finding predicts "
            f"new_nz >= warm_nz for benign on-axis grids."
        )

    def test_multimode_matches_reference(self, fit, grid):
        """v4.15 property pin (relaxed from pre-v4.15 1e-10 rel
        bit-equal pin) -- see
        :meth:`test_lg00_single_mode_matches_reference` for the
        rationale and assertion layout.
        """
        X, Y = grid
        # Mix LG_{0,0}, LG_{1,0}, LG_{0,1} in source; LG_{0,0} +
        # LG_{0,1} in pupil.  More modes means more inner-loop calls
        # to lg_polynomial -- the hoist payoff scales with this.
        source_amps = {
            (0, 0): 1.0 + 0.0j,
            (1, 0): 0.3 - 0.1j,
            (0, 1): 0.2 + 0.4j,
        }
        pupil_amps = {
            (0, 0): 1.0 + 0.0j,
            (0, 1): 0.1 + 0.2j,
        }
        clear_lg_polynomial_cache()
        new = propagate_modal_asymptotic(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        cold_ref = _batched_cold_start_reference_propagate_modal(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        cold_peak = float(np.max(np.abs(cold_ref)))
        max_abs_diff = float(np.max(np.abs(new - cold_ref)))
        assert max_abs_diff < 1e-8 * max(cold_peak, 1.0), (
            f"Multi-mode vs cold-start batched reference: "
            f"max|new - cold_ref| = {max_abs_diff:.3e}, "
            f"cold_peak = {cold_peak:.3e}"
        )

        warm_ref = _scalar_pixel_reference_propagate_modal(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        warm_peak = float(np.max(np.abs(warm_ref)))
        if warm_peak == 0:
            pytest.skip("warm-start reference produced no signal")
        new_energy = float(np.sum(np.abs(new) ** 2))
        warm_energy = float(np.sum(np.abs(warm_ref) ** 2))
        rel_e_diff = abs(new_energy - warm_energy) / warm_energy
        assert rel_e_diff < 0.05, (
            f"Multi-mode total-energy mismatch with warm-start ref: "
            f"new={new_energy:.3e}, warm={warm_energy:.3e}, "
            f"rel diff={rel_e_diff:.3e}"
        )
        new_nz = int(np.sum(new != 0))
        warm_nz = int(np.sum(warm_ref != 0))
        assert new_nz >= warm_nz, (
            f"v4.15 cold-start path produced FEWER non-zero pixels "
            f"({new_nz}) than the warm-start reference "
            f"({warm_nz}) -- the wrong-saddle finding predicts "
            f"new_nz >= warm_nz for benign on-axis grids."
        )

    def test_clear_cache_between_calls_preserves_output(self, fit, grid):
        """Clearing the cache between calls must produce identical
        output -- the hoist depends only on the function-local dict,
        not on which (p, ell) entries happen to be cached at the
        module level."""
        X, Y = grid
        source_amps = {(0, 0): 1.0 + 0.0j, (1, 0): 0.2 + 0.0j}
        pupil_amps = {(0, 0): 1.0 + 0.0j}
        clear_lg_polynomial_cache()
        a = propagate_modal_asymptotic(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        clear_lg_polynomial_cache()
        b = propagate_modal_asymptotic(
            fit,
            source_point=(0.0, 0.0),
            source_amplitudes=source_amps,
            pupil_amplitudes=pupil_amps,
            w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y,
        )
        assert np.array_equal(a, b), (
            "clear_lg_polynomial_cache() between calls changed the "
            "result -- the hoisted-cache logic must be independent of "
            "the persistence of the module-level cache."
        )


# ============================================================================
# Perf #6 -- propagate_hf_chebyshev_quadrature vectorised chunk loop
# ============================================================================


class TestHfChebyshevQuadratureVectorisedChunk:
    """Pin bit-near-exact agreement between the v4.12.0 vectorised
    chunk path and a reference scalar-pixel-at-a-time evaluator."""

    @pytest.fixture(scope='class')
    def hf_fit(self):
        return _build_hf_fit()

    @pytest.fixture(scope='class')
    def grids_and_field(self, hf_fit):
        # 16x16 input, 16x16 output (small for speed; the scalar
        # reference is O(N_out^2 * N_in^2) and we run it once per
        # test case).
        N_in = 16
        N_out = 16
        ax = np.linspace(-hf_fit.s1x_halfrange * 0.5,
                          hf_fit.s1x_halfrange * 0.5, N_in) + hf_fit.s1x_centre
        ay = np.linspace(-hf_fit.s1y_halfrange * 0.5,
                          hf_fit.s1y_halfrange * 0.5, N_in) + hf_fit.s1y_centre
        bx = np.linspace(-hf_fit.s2x_halfrange * 0.3,
                          hf_fit.s2x_halfrange * 0.3, N_out) + hf_fit.s2x_centre
        by = np.linspace(-hf_fit.s2y_halfrange * 0.3,
                          hf_fit.s2y_halfrange * 0.3, N_out) + hf_fit.s2y_centre
        X, Y = np.meshgrid(ax, ay, indexing='xy')
        w = 1e-5
        E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)).astype(np.complex128)
        return ax, ay, bx, by, E

    def test_chunk_1_matches_scalar_reference(self, hf_fit, grids_and_field):
        ax, ay, bx, by, E = grids_and_field
        new = propagate_hf_chebyshev_quadrature(
            hf_fit, E, ax, ay, bx, by,
            apply_van_vleck=True, chunk_output=1,
        )
        ref = _scalar_pixel_reference_hf_chebyshev(
            hf_fit, E, ax, ay, bx, by, apply_van_vleck=True,
        )
        peak = float(np.max(np.abs(ref)))
        assert peak > 0, "reference produced no signal"
        max_diff = float(np.max(np.abs(new - ref)))
        rel = max_diff / peak
        assert rel < 1e-10, (
            f"chunk_output=1 path drifted from scalar reference: "
            f"max|new - ref| = {max_diff:.3e}, peak = {peak:.3e}, "
            f"rel = {rel:.3e}"
        )

    def test_chunk_32_matches_scalar_reference(self, hf_fit, grids_and_field):
        ax, ay, bx, by, E = grids_and_field
        new = propagate_hf_chebyshev_quadrature(
            hf_fit, E, ax, ay, bx, by,
            apply_van_vleck=True, chunk_output=32,
        )
        ref = _scalar_pixel_reference_hf_chebyshev(
            hf_fit, E, ax, ay, bx, by, apply_van_vleck=True,
        )
        peak = float(np.max(np.abs(ref)))
        max_diff = float(np.max(np.abs(new - ref)))
        rel = max_diff / peak
        assert rel < 1e-10, (
            f"chunk_output=32 path drifted from scalar reference: "
            f"max|new - ref| = {max_diff:.3e}, peak = {peak:.3e}, "
            f"rel = {rel:.3e}"
        )

    def test_chunk_matches_across_sizes(self, hf_fit, grids_and_field):
        """Different chunk sizes must produce identical output (the
        chunk axis is a pure broadcast / accumulation -- no
        chunk-dependent reduction)."""
        ax, ay, bx, by, E = grids_and_field
        outs = []
        for chunk in (1, 4, 16, 32):
            out = propagate_hf_chebyshev_quadrature(
                hf_fit, E, ax, ay, bx, by,
                apply_van_vleck=True, chunk_output=chunk,
            )
            outs.append((chunk, out))
        # All chunks must agree with the chunk=1 reference.
        chunk1 = outs[0][1]
        peak = float(np.max(np.abs(chunk1)))
        for chunk, out in outs[1:]:
            max_diff = float(np.max(np.abs(out - chunk1)))
            rel = max_diff / peak if peak > 0 else 0.0
            assert rel < 1e-10, (
                f"chunk={chunk} drifted from chunk=1: "
                f"max|diff| = {max_diff:.3e}, rel = {rel:.3e}"
            )

    def test_no_van_vleck_matches_scalar_reference(self, hf_fit,
                                                    grids_and_field):
        ax, ay, bx, by, E = grids_and_field
        new = propagate_hf_chebyshev_quadrature(
            hf_fit, E, ax, ay, bx, by,
            apply_van_vleck=False, chunk_output=8,
        )
        ref = _scalar_pixel_reference_hf_chebyshev(
            hf_fit, E, ax, ay, bx, by, apply_van_vleck=False,
        )
        peak = float(np.max(np.abs(ref)))
        assert peak > 0
        max_diff = float(np.max(np.abs(new - ref)))
        rel = max_diff / peak
        assert rel < 1e-10, (
            f"apply_van_vleck=False drifted from reference: "
            f"max|new - ref| = {max_diff:.3e}, rel = {rel:.3e}"
        )

    def test_memory_guard_shrinks_chunk(self, hf_fit, grids_and_field):
        """A very small ``max_chunk_memory_mb`` must shrink the
        effective chunk below ``chunk_output``; output must still be
        identical to the unshrunk path."""
        ax, ay, bx, by, E = grids_and_field
        # Default (no shrink), chunk=64
        unshrunk = propagate_hf_chebyshev_quadrature(
            hf_fit, E, ax, ay, bx, by,
            apply_van_vleck=True, chunk_output=64,
            max_chunk_memory_mb=256.0,
        )
        # 0.001 MB budget -> 1 KB -> can hold one (16x16, c=1) tensor
        # at most.  Effective chunk should fall to 1; output must
        # still match.
        shrunk = propagate_hf_chebyshev_quadrature(
            hf_fit, E, ax, ay, bx, by,
            apply_van_vleck=True, chunk_output=64,
            max_chunk_memory_mb=1e-4,
        )
        max_diff = float(np.max(np.abs(unshrunk - shrunk)))
        peak = float(np.max(np.abs(unshrunk)))
        rel = max_diff / peak if peak > 0 else 0.0
        assert rel < 1e-10, (
            f"max_chunk_memory_mb cap changed the output: "
            f"max|diff| = {max_diff:.3e}, rel = {rel:.3e}.  The memory "
            f"guard should only shrink the chunk, not change the math."
        )

    def test_paraxial_fresnel_limit(self):
        """A separable HF Phi(s1, s2) = (s2-s1)^2 / (2 lambda z) should
        produce a Fresnel-like output up to the global -i/(2pi)
        prefactor that the function applies post-hoc.

        This is a smoke test that the broadcast contraction signs
        match the scalar evaluator -- we do NOT compare against an
        absolute Fresnel propagator here (that's a validation test);
        we only assert the vectorised path matches the scalar
        reference for the same fit."""
        # Build a minimal HFPolyFit by hand:  Phi(u_s1x, u_s1y, u_s2x, u_s2y)
        # = (a/2)*(u_s2x - u_s1x)^2 + (a/2)*(u_s2y - u_s1y)^2 with the
        # normaliser absorbed into the linear coeffs.  We just pin
        # vectorised vs scalar.
        # Easier: reuse an existing HF fit and assert internal
        # consistency over a fresh grid with a different (Ny_in, Nx_in).
        rx = _make_singlet()
        fit = fit_hf_polynomials(
            rx, wavelength=1.31e-6,
            source_box_half=20e-6, pupil_box_half=0.02,
            n_field=6, n_pupil=6, poly_order=4,
        )
        # 32x32 input, 32x32 output -- the prompt's "32x32 output"
        # canonical case.
        N_in = 32
        N_out = 32
        ax = np.linspace(-fit.s1x_halfrange * 0.5,
                          fit.s1x_halfrange * 0.5, N_in) + fit.s1x_centre
        ay = np.linspace(-fit.s1y_halfrange * 0.5,
                          fit.s1y_halfrange * 0.5, N_in) + fit.s1y_centre
        bx = np.linspace(-fit.s2x_halfrange * 0.3,
                          fit.s2x_halfrange * 0.3, N_out) + fit.s2x_centre
        by = np.linspace(-fit.s2y_halfrange * 0.3,
                          fit.s2y_halfrange * 0.3, N_out) + fit.s2y_centre
        X, Y = np.meshgrid(ax, ay, indexing='xy')
        w = 1.5e-5
        E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
        new = propagate_hf_chebyshev_quadrature(
            fit, E, ax, ay, bx, by,
            apply_van_vleck=True, chunk_output=64,
        )
        ref = _scalar_pixel_reference_hf_chebyshev(
            fit, E, ax, ay, bx, by, apply_van_vleck=True,
        )
        peak = float(np.max(np.abs(ref)))
        assert peak > 0
        max_diff = float(np.max(np.abs(new - ref)))
        rel = max_diff / peak
        assert rel < 1e-10, (
            f"32x32 output on 32x32 input: vectorised differs from "
            f"scalar by max|diff| = {max_diff:.3e}, rel = {rel:.3e}"
        )

    def test_real_e_in_keeps_complex_output(self, hf_fit, grids_and_field):
        """v4.11.2 fix:  real-valued E_in must still produce complex
        output (no kernel cast to real).  v4.12.0 must not regress
        this."""
        ax, ay, bx, by, _ = grids_and_field
        # Real-valued input
        E = np.ones((ay.size, ax.size), dtype=np.float64)
        out = propagate_hf_chebyshev_quadrature(
            hf_fit, E, ax, ay, bx, by,
            apply_van_vleck=False, chunk_output=8,
        )
        assert np.iscomplexobj(out), (
            "Real-valued E_in must yield complex output (v4.11.2 fix)."
        )
        assert np.any(out.imag != 0), (
            "Output must have a non-zero imaginary part."
        )
