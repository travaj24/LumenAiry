"""Pinning tests for the v4.14.1 audit (Agent A scope --
``lumenairy.propagators.asymptotic``).

Four audit items are pinned:

* **P0-NEW-1** -- ``_lg_mode_conj_stack`` / ``_hg_mode_conj_stack``
  cache keys silently elided the physical pitch ``(dx, dy)``.  Two
  calls at the same shape ``(Ny, Nx)`` but different pitch (e.g.
  ``dx=1e-6`` then ``dx=2e-6`` at N=64) collided on the cached entry
  and the second call received the first call's modes evaluated
  against the second call's field.  v4.14.1 adds ``dx, dy`` to both
  cache keys.  The pinning tests exercise the collision scenario
  directly:  build at one pitch, build again at a different pitch
  with the same N, w, and centres, and confirm the second result
  reflects the new pitch (it would NOT if the cache key were stale).

* **P2-1** -- ``_LG_MODE_STACK_CACHE`` / ``_HG_MODE_STACK_CACHE``
  were bare ``OrderedDict`` instances without thread-safety.
  Concurrent ``design_optimize`` worker threads can race on the
  ``get`` / ``move_to_end`` / ``popitem`` read-modify-write sequence.
  v4.14.1 wraps every read-modify-write under ``_LG_MODE_STACK_LOCK``
  / ``_HG_MODE_STACK_LOCK`` (one lock per cache) following the
  ``_ASM_CACHE_LOCK`` precedent in
  :mod:`lumenairy.propagators.propagation`.  No bit-equal test --
  threading correctness is a smoke test:  spawn N threads, each
  calls ``decompose_lg`` / ``decompose_hg`` on the same input, no
  exceptions raised and all returned dicts agree.

* **P1-NEW-3** -- ``_solve_envelope_stationary_batch`` violated its
  own docstring contract by writing ``True`` to ``converged_mask``
  for pixels that failed (stalled or singular Hessian) instead of
  preserving ``False`` per the docstring.  v4.14.1 separates the
  active-set bookkeeping (a new local ``finished`` mask) from the
  user-facing ``converged`` flag; only residual-passes-tol pixels
  are marked ``True``.  Pinning test:  feed a duck-typed fit whose
  Jacobian is rank-deficient and use ``w_p = 1e300`` so the
  inv_wp2 piece of the Hessian collapses to zero -- this produces a
  singular Hessian, the solver drops the pixel, and the returned
  ``converged_mask`` must be ``False``.

* **P1-NEW-5** -- ``propagate_modal_asymptotic``'s ``row_reset``
  branch resets ``last_arg_detM`` and ``maslov_branch`` AND, as of
  v4.14.1, also resets the Newton warm-start ``last_v_star``.
  This eliminates the cross-row Newton chain that spans the
  discontinuous raster jump from (x_max, y_n) to (x_min,
  y_{n+1}) -- plausibly the v4.14.0 wrong-saddle-basin mechanism
  near grid edges (largest jump in s_2).  v4.14.1 implemented
  option (a) (the full reset) in coordination with updating the
  v4.14.0 bit-equal pin (``test_lg00_single_mode_bit_equal`` in
  ``test_audit_fixes_v4_14_0_agent_1.py``) so the reference loop
  resets ``last_v_star`` too.  The pinning test asserts the new
  behaviour:  ``last_v_star`` IS reset at row wrap.

Author:  Agent A -- v4.14.1.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.propagators import asymptotic as _asy
from lumenairy.propagators.asymptotic import (
    _HG_MODE_STACK_CACHE,
    _LG_MODE_STACK_CACHE,
    _hg_mode_conj_stack,
    _lg_mode_conj_stack,
    _solve_envelope_stationary_batch,
    clear_lg_mode_stack_cache,
    clear_lg_polynomial_cache,
    decompose_hg,
    decompose_lg,
    propagate_modal_asymptotic,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _meshgrid(N: int, half_extent: float) -> Tuple[np.ndarray, np.ndarray,
                                                    float, float]:
    """Build a square (N, N) meshgrid spanning [-half_extent, +half_extent]."""
    x = np.linspace(-half_extent, half_extent, N)
    y = np.linspace(-half_extent, half_extent, N)
    X, Y = np.meshgrid(x, y, indexing='xy')
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return X, Y, dx, dy


# ===========================================================================
# A.1 (P0-NEW-1) -- dx, dy in mode-stack cache keys
# ===========================================================================


class TestModeStackCacheKeyIncludesPitch:
    """Pin that ``_lg_mode_conj_stack`` / ``_hg_mode_conj_stack`` cache
    keys discriminate on physical pitch ``(dx, dy)``.

    The pre-v4.14.1 bug:  two calls at the same shape ``(Ny, Nx)`` but
    different ``dx, dy`` returned the cached modes from the first call,
    silently making the second call's decompose use the wrong modes.
    """

    def test_lg_mode_stack_cache_key_includes_pitch(self):
        """Build LG mode stack at (N=64, dx=1e-6) then at
        (N=64, dx=2e-6), same N, w, p_max, ell_max, cx, cy.  The
        second result must NOT be the cached first stack.
        """
        clear_lg_mode_stack_cache()
        clear_lg_polynomial_cache()
        N = 64
        # Same N, same waist, same centres -- only pitch differs.
        w = 20e-6
        cx = 0.0
        cy = 0.0
        # First call: pitch = 2*1e-6 / (N - 1)
        X1, Y1, dx1, dy1 = _meshgrid(N, half_extent=1e-6 * (N - 1) / 2.0)
        # Second call: pitch = 2*2e-6 / (N - 1)  -- twice as coarse
        X2, Y2, dx2, dy2 = _meshgrid(N, half_extent=2e-6 * (N - 1) / 2.0)
        # Sanity check the test setup itself.
        assert X1.shape == X2.shape == (N, N), 'shapes must match'
        assert not np.isclose(dx1, dx2), 'pitches must differ for this test'

        keys1, stack1 = _lg_mode_conj_stack(
            X1, Y1, w, p_max=2, ell_max=2, cx=cx, cy=cy, dx=dx1, dy=dy1,
        )
        # Pre-v4.14.1 bug:  with same N, same (w, cx, cy, p_max, ell_max)
        # but different pitch, the second call would hit the cached
        # stack1 (built on X1) instead of rebuilding on X2.  v4.14.1
        # includes ``dx, dy`` in the key so it MUST rebuild.
        keys2, stack2 = _lg_mode_conj_stack(
            X2, Y2, w, p_max=2, ell_max=2, cx=cx, cy=cy, dx=dx2, dy=dy2,
        )
        assert keys1 == keys2, '(p, ell) ordering should match'
        # The modes evaluated on a *physically wider* grid differ
        # noticeably from the modes evaluated on the original grid.
        # Both stacks include the LG_{0,0} mode (Gaussian); the
        # envelope at the corner pixel of the wider grid is
        # exp(-(2*1e-6)^2/w^2) vs exp(-(1e-6)^2/w^2) for the
        # narrower grid -- those differ by a factor of order
        # exp(-3e-12 / w^2).  Even at w=20e-6 the envelope at the
        # corner of the wider grid is ~7% smaller than the narrower
        # one.  Pick any pixel where the difference is well above
        # numerical noise.
        diff = np.max(np.abs(stack1 - stack2))
        # If the cache key was stale, stack1 == stack2 exactly (bit
        # for bit) because Python would have returned the SAME ndarray
        # object.  The new pinning test wants a clear physical signal,
        # not just identity inequality.
        assert diff > 1e-6, (
            f'modes did not rebuild at new pitch -- max abs diff '
            f'{diff:.3e}.  Cache key likely missing (dx, dy).'
        )
        # And rule out the most insidious failure mode:  the SAME
        # underlying object getting returned twice (bit-equal AND
        # is-identical).  After v4.14.1 these are independent
        # builds, hence different objects.
        assert stack1 is not stack2, (
            'cache returned the same object for two distinct pitches'
        )

    def test_hg_mode_stack_cache_key_includes_pitch(self):
        """Same as the LG test but for ``_hg_mode_conj_stack``."""
        clear_lg_mode_stack_cache()
        N = 64
        wx = 20e-6
        wy = 25e-6
        cx = 0.0
        cy = 0.0
        X1, Y1, dx1, dy1 = _meshgrid(N, half_extent=1e-6 * (N - 1) / 2.0)
        X2, Y2, dx2, dy2 = _meshgrid(N, half_extent=2e-6 * (N - 1) / 2.0)
        assert not np.isclose(dx1, dx2)
        keys1, stack1 = _hg_mode_conj_stack(
            X1, Y1, wx, wy, m_max=2, n_max=2, cx=cx, cy=cy,
            dx=dx1, dy=dy1,
        )
        keys2, stack2 = _hg_mode_conj_stack(
            X2, Y2, wx, wy, m_max=2, n_max=2, cx=cx, cy=cy,
            dx=dx2, dy=dy2,
        )
        assert keys1 == keys2
        diff = np.max(np.abs(stack1 - stack2))
        assert diff > 1e-6, (
            f'HG modes did not rebuild at new pitch -- diff {diff:.3e}'
        )
        assert stack1 is not stack2

    def test_lg_decompose_lg_no_cache_collision_at_different_pitch(self):
        """End-to-end:  ``decompose_lg`` must return overlaps consistent
        with each call's own grid pitch, not the cached previous pitch.

        We project a Gaussian centred at the origin onto the LG_{0, 0}
        mode and check that the recovered amplitude is ~1 in both
        cases -- which it would NOT be if the second call were running
        the first call's modes against the second call's field grid.
        """
        clear_lg_mode_stack_cache()
        clear_lg_polynomial_cache()
        N = 64
        w = 30e-6
        # First grid:  ~4.5*w extent.
        X1, Y1, dx1, dy1 = _meshgrid(N, half_extent=4.5 * w)
        # Second grid:  ~9*w extent -- same N, twice the pitch.
        X2, Y2, dx2, dy2 = _meshgrid(N, half_extent=9.0 * w)
        # Build the LG_{0, 0} amplitude=1 field on each grid.
        from lumenairy.propagators.asymptotic import evaluate_lg_mode
        F1 = evaluate_lg_mode(0, 0, w, X1, Y1)
        F2 = evaluate_lg_mode(0, 0, w, X2, Y2)
        out1 = decompose_lg(F1, X1, Y1, w, p_max=0, ell_max=0)
        out2 = decompose_lg(F2, X2, Y2, w, p_max=0, ell_max=0)
        # Both should recover amplitude ~= 1.0+0j for (0, 0).  If the
        # second call hit the cached modes from the first grid (the
        # P0-NEW-1 bug), the overlap would be wrong because the modes
        # evaluated at the *first* grid points would be projected
        # against a field sampled at the *second* grid points -- but
        # numpy einsum requires shape-match, so actually the
        # silent-wrong-physics manifests as a numerically incorrect
        # overlap rather than a shape error (the SHAPES match, since
        # only the pitch differs).
        assert abs(out1[(0, 0)] - (1.0 + 0.0j)) < 1e-3, (
            f'first-grid overlap {out1[(0, 0)]} not ~= 1.0'
        )
        assert abs(out2[(0, 0)] - (1.0 + 0.0j)) < 1e-3, (
            f'second-grid overlap {out2[(0, 0)]} not ~= 1.0 -- '
            f'P0-NEW-1 cache collision regression'
        )


# ===========================================================================
# A.2 (P2-1) -- thread-safety locks on LG / HG mode-stack caches
# ===========================================================================


class TestModeStackCacheLocks:
    """Smoke-test that ``_LG_MODE_STACK_LOCK`` / ``_HG_MODE_STACK_LOCK``
    prevent races on concurrent decompose calls.  This is infrastructure
    -- no precise contract to pin -- so we just confirm that N concurrent
    workers (a) do not raise and (b) return identical results.
    """

    def test_concurrent_decompose_lg_no_races(self):
        clear_lg_mode_stack_cache()
        clear_lg_polynomial_cache()
        N_threads = 4
        N_grid = 32
        w = 30e-6
        X, Y, _, _ = _meshgrid(N_grid, half_extent=4.0 * w)
        from lumenairy.propagators.asymptotic import evaluate_lg_mode
        field = evaluate_lg_mode(0, 0, w, X, Y)
        results: List[Dict[Tuple[int, int], complex]] = []
        errors: List[BaseException] = []
        lock = threading.Lock()

        def _worker() -> None:
            try:
                out = decompose_lg(field, X, Y, w, p_max=2, ell_max=2)
                with lock:
                    results.append(out)
            except BaseException as exc:  # noqa: BLE001 -- smoke test
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(N_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f'concurrent decompose_lg raised: {errors}'
        assert len(results) == N_threads
        # All workers ran the same input -- results must agree.
        ref = results[0]
        for k in ref:
            for r in results[1:]:
                assert abs(r[k] - ref[k]) < 1e-12, (
                    f'concurrent decompose_lg disagreement on mode {k}'
                )

    def test_concurrent_decompose_hg_no_races(self):
        clear_lg_mode_stack_cache()
        N_threads = 4
        N_grid = 32
        wx = 30e-6
        wy = 25e-6
        X, Y, _, _ = _meshgrid(N_grid, half_extent=4.0 * wx)
        from lumenairy.propagators.asymptotic import evaluate_hg_mode
        field = evaluate_hg_mode(0, 0, wx, wy, X, Y)
        results: List[Dict[Tuple[int, int], complex]] = []
        errors: List[BaseException] = []
        lock = threading.Lock()

        def _worker() -> None:
            try:
                out = decompose_hg(field, X, Y, wx, wy,
                                    m_max=2, n_max=2)
                with lock:
                    results.append(out)
            except BaseException as exc:  # noqa: BLE001 -- smoke test
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(N_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f'concurrent decompose_hg raised: {errors}'
        assert len(results) == N_threads
        ref = results[0]
        for k in ref:
            for r in results[1:]:
                assert abs(r[k] - ref[k]) < 1e-12


# ===========================================================================
# A.3 (P1-NEW-3) -- _solve_envelope_stationary_batch contract
# ===========================================================================


class TestSolveEnvelopeStationaryBatchContract:
    """Pin that pixels which fail (singular Hessian) end with
    ``converged_mask=False`` per the function's docstring.

    Pre-v4.14.1 the function set ``converged[idx[done & ~is_conv]] = True``
    to drop stalled / singular pixels from the active set, which
    silently flagged failures as successes -- contrary to the docstring
    contract.  v4.14.1 separates the active-set bookkeeping (a new
    local ``finished`` mask) from the user-facing ``converged`` flag.
    """

    def test_singular_hessian_pixels_flagged_not_converged(self):
        """Construct a duck-typed fit whose Jacobian is rank-deficient
        ([[1, 0], [0, 0]]) and use ``w_p = 1e300`` (so ``inv_wp2 -> 0``).

        Then the Hessian H = inv_ws2 * J^T J = inv_ws2 * [[1,0],[0,0]]
        is singular (det_H = 0) at the initial guess, so the solver
        marks the pixel as ``done`` via ``~safe`` and skips the update.
        The residual ``rn`` is non-zero (delta_s1 != 0 by construction
        and w_s is finite) so ``is_conv = False``.  After v4.14.1 the
        returned ``converged_mask`` is ``False`` for this pixel.
        Pre-v4.14.1 it was wrongly ``True``.
        """
        # Duck-typed fit:  only needs eval_s1_with_v2_grad.
        class _DegenerateFit:
            def eval_s1_with_v2_grad(self, sx, sy, vx, vy):
                # Constant s1 != source so delta_s1 is nonzero.
                # J = [[1, 0], [0, 0]] -- rank deficient.
                K = sx.shape[0]
                s1x = np.full(K, 0.5e-3, dtype=np.float64)
                s1y = np.full(K, 0.5e-3, dtype=np.float64)
                dS1x_dv2x = np.ones(K, dtype=np.float64)
                dS1x_dv2y = np.zeros(K, dtype=np.float64)
                dS1y_dv2x = np.zeros(K, dtype=np.float64)
                dS1y_dv2y = np.zeros(K, dtype=np.float64)
                return (s1x, s1y, dS1x_dv2x, dS1x_dv2y,
                        dS1y_dv2x, dS1y_dv2y)

        fit = _DegenerateFit()
        # One pixel.  Source at origin so delta_s1 = [0.5e-3, 0.5e-3].
        s2x = np.array([0.0], dtype=np.float64)
        s2y = np.array([0.0], dtype=np.float64)
        # w_p = 1e300 makes inv_wp2 = 1e-600 -> 0.0 in float64.
        v2x_star, v2y_star, conv = _solve_envelope_stationary_batch(
            fit, s2x, s2y,
            src_x=0.0, src_y=0.0,
            w_s=50e-6, w_p=1e300,
            v_cx=0.0, v_cy=0.0,
            max_iter=12, tol=1e-12,
        )
        # The Hessian is singular (det = 0); the pixel must drop with
        # converged_mask = False per the docstring contract.
        assert conv.shape == (1,)
        assert conv[0] == False, (  # noqa: E712 -- want explicit bool
            'singular-Hessian pixel must NOT be flagged converged'
            ' (P1-NEW-3 docstring contract)'
        )
        # And the last finite iterate is preserved (v_cx, v_cy) since
        # the singular pixel never updates.
        assert np.isfinite(v2x_star[0]) and np.isfinite(v2y_star[0])

    def test_genuinely_converged_pixel_still_marked_true(self):
        """Sanity:  a pixel that actually converges still receives
        ``converged_mask=True``.  We use a well-conditioned diagonal
        Jacobian and converge in one step.
        """
        class _WellPosedFit:
            def eval_s1_with_v2_grad(self, sx, sy, vx, vy):
                K = sx.shape[0]
                # s1 = (src_x, src_y) when v == v_centre => delta_s1 = 0
                # and starting at v_cx, v_cy => delta_v = 0 too.  Then
                # residual = 0 at iter 0 and the pixel converges on
                # the first iteration.
                s1x = np.zeros(K, dtype=np.float64)
                s1y = np.zeros(K, dtype=np.float64)
                dS1x_dv2x = np.ones(K, dtype=np.float64)
                dS1x_dv2y = np.zeros(K, dtype=np.float64)
                dS1y_dv2x = np.zeros(K, dtype=np.float64)
                dS1y_dv2y = np.ones(K, dtype=np.float64)
                return (s1x, s1y, dS1x_dv2x, dS1x_dv2y,
                        dS1y_dv2x, dS1y_dv2y)
        fit = _WellPosedFit()
        s2x = np.array([0.0], dtype=np.float64)
        s2y = np.array([0.0], dtype=np.float64)
        _, _, conv = _solve_envelope_stationary_batch(
            fit, s2x, s2y, src_x=0.0, src_y=0.0,
            w_s=50e-6, w_p=0.02,
            v_cx=0.0, v_cy=0.0,
        )
        assert conv[0] == True, (  # noqa: E712
            'well-posed pixel must still be flagged converged'
        )


# ===========================================================================
# A.4 (P1-NEW-5) -- row_reset resets the Newton warm-start too
# ===========================================================================


class TestRowResetResetsWarmStart:
    """Pin the row-reset warm-start contract.

    History:

    * **v4.14.1** closed P1-NEW-5 option (a):  the ``row_reset``
      branch reset the per-pixel Newton warm-start ``last_v_star``
      to the pupil centre at each row wrap, eliminating the
      cross-row chain that plausibly entered wrong-saddle basins
      near grid edges.  The v4.14.1 test patched the scalar
      :func:`solve_envelope_stationary` to inspect ``v2_initial``
      at the row-wrap pixel.

    * **v4.15** structurally retired the warm-start chain
      entirely:  the public :func:`propagate_modal_asymptotic`
      now routes through :func:`_solve_envelope_stationary_batch`
      with cold-start seeds for every pixel.  The v4.14.1
      semantic ("row wrap resets v2_initial to pupil centre") is
      now structurally guaranteed for ALL pixels in all three
      ``maslov_tracking`` modes -- the test below is updated to
      pin the stronger v4.15 invariant by spying on the batched
      solver.

    The companion pin for v4.15's row-reset Maslov-branch
    behaviour lives in
    :mod:`tests/unit/test_v4_15_agent_a.py::test_modal_asymptotic_row_reset_still_works`.
    """

    def test_row_reset_resets_warm_start(self, monkeypatch):
        """v4.15 invariant:  the public propagator routes through
        :func:`_solve_envelope_stationary_batch` (which structurally
        cold-starts every pixel at the pupil centre ``(v_cx, v_cy)``)
        and the scalar :func:`solve_envelope_stationary` -- the
        warm-start chain that pre-v4.15 carried ``last_v_star`` across
        pixels -- is no longer invoked at all by the public path.

        This pins the v4.15 stronger guarantee (cold-start for ALL
        pixels in ALL ``maslov_tracking`` modes) rather than the
        v4.14.1 narrower guarantee (warm-start reset only at row
        wrap).  Stronger pin: the warm-start chain is structurally
        absent, not just reset.
        """
        from lumenairy.propagators.asymptotic import (
            _solve_envelope_stationary_batch,
            fit_canonical_polynomials,
        )

        rx = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3,
        )
        rx['object_distance'] = 0.1
        fit = fit_canonical_polynomials(
            rx, wavelength=1.31e-6,
            source_box_half=20e-6, pupil_box_half=0.02,
            n_field=4, n_pupil=4, poly_order=3,
        )
        v_cx = 0.0
        v_cy = 0.0

        scalar_calls: List[Dict[str, Any]] = []
        original_scalar = _asy.solve_envelope_stationary

        def _spy_scalar(*args, **kwargs):
            scalar_calls.append({'args_len': len(args), 'kw_keys': sorted(kwargs)})
            return original_scalar(*args, **kwargs)

        batch_calls: List[Tuple[float, float]] = []
        original_batch = _solve_envelope_stationary_batch

        def _spy_batch(*args, **kwargs):
            cx = kwargs.get('v_cx') if 'v_cx' in kwargs else (
                args[7] if len(args) > 7 else None)
            cy = kwargs.get('v_cy') if 'v_cy' in kwargs else (
                args[8] if len(args) > 8 else None)
            batch_calls.append((float(cx), float(cy)))
            return original_batch(*args, **kwargs)

        monkeypatch.setattr(_asy, 'solve_envelope_stationary', _spy_scalar)
        monkeypatch.setattr(
            _asy, '_solve_envelope_stationary_batch', _spy_batch)

        s2x_axis = np.linspace(-10e-6, 10e-6, 3)
        s2y_axis = np.linspace(-10e-6, 10e-6, 3)
        S2X, S2Y = np.meshgrid(s2x_axis, s2y_axis, indexing='xy')

        for tracking in ('principal', '1d_raster', 'row_reset'):
            scalar_calls.clear()
            batch_calls.clear()
            _ = propagate_modal_asymptotic(
                fit,
                source_point=(0.0, 0.0),
                w_s=20e-6,
                w_p=0.02,
                v2_centre=(v_cx, v_cy),
                s2_grid_x=S2X,
                s2_grid_y=S2Y,
                maslov_tracking=tracking,
            )
            assert len(batch_calls) >= 1, (
                f'tracking={tracking!r}: batched solver never called '
                f'-- v4.15 public propagator must route through '
                f'_solve_envelope_stationary_batch'
            )
            assert len(scalar_calls) == 0, (
                f'tracking={tracking!r}: scalar solve_envelope_stationary '
                f'was called {len(scalar_calls)} times -- v4.15 deletes '
                f'the warm-start chain entirely; the scalar solver must '
                f'NOT be invoked by the public path in any tracking mode.'
            )
            for call_idx, (cx_seen, cy_seen) in enumerate(batch_calls):
                assert abs(cx_seen - v_cx) < 1e-30, (
                    f'v4.15 cold-start invariant: tracking={tracking!r}, '
                    f'call {call_idx}: batched solver received v_cx={cx_seen} '
                    f'instead of pupil-centre v_cx={v_cx}.'
                )
                assert abs(cy_seen - v_cy) < 1e-30, (
                    f'v4.15 cold-start invariant: tracking={tracking!r}, '
                    f'call {call_idx}: batched solver received v_cy={cy_seen} '
                    f'instead of pupil-centre v_cy={v_cy}.'
                )
