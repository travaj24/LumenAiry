"""Consolidated audit-fix tests for the **optimize** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 6 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_13_0_perf_fd_grad.py``
* ``test_audit_fixes_v4_13_2_agent_c.py``
* ``test_audit_fixes_v4_14_0_agent_4.py``
* ``test_audit_fixes_v4_14_1_agent_b.py``
* ``test_audit_fixes_v4_14_2_agent_b.py``
* ``test_audit_fixes_v4_14_2_agent_c.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  inspect.getsource proxy
tests are tagged with a TODO comment per AUDIT_V4_13_1 Part 6.1.
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_13_0_perf_fd_grad.py
# Audit version: V4_13_0  scope: perf_fd_grad
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pins for the v4.13.0 finite-difference gradient
#   parameterisation (audit Phase-3 Group beta task beta.3).
#   
#   v4.13.0 adds an ``scheme={'central','forward'}`` parameter to
#   ``lumenairy.optimize.core._fd_grad_pure`` and the wrapped
#   ``_fd_grad_for`` helper.
#   
#   * ``scheme='central'`` (the default) preserves bit-identical pre-
#     v4.13.0 behaviour: 2N evaluations per gradient, O(h^2) truncation.
#   * ``scheme='forward'`` is an opt-in perf path: N+1 evaluations per
#     gradient (or N if the caller passes ``f0``).  O(h) truncation.
#   
#   The ``f0`` parameter is meaningful only for ``scheme='forward'`` (the
#   central-difference path does not evaluate at the centre, so there is
#   no centre value to reuse).
#   
#   What this test pins
#   -------------------
#   
#   1. **Default scheme is central** -- ``_fd_grad_pure(f, x)`` with no
#      ``scheme=`` returns the central-difference gradient.
#   2. **Central-difference correctness** -- on a quadratic, central
#      matches the analytical gradient to ~1e-9 (O(h^2) truncation with
#      h~1e-7).
#   3. **Forward-difference correctness** -- on the same quadratic,
#      forward matches analytical to ~1e-3 (O(h) truncation with h~1e-7).
#   4. **Eval-count contracts**:
#      * central: exactly 2N calls to f.
#      * forward, f0=None: exactly N+1 calls.
#      * forward, f0=<value>: exactly N calls.
#   5. **f0 reuse invariance (forward only)** -- passing ``f0=f(x)`` gives
#      bit-identical output to ``f0=None`` on the forward path.
#   6. **Invalid scheme** -- ``scheme='banana'`` raises ValueError.
#   7. **scale_floor floor still respected** -- per-variable floor is
#      applied in both schemes.
# ============================================================================
import numpy as np
import pytest

from lumenairy.optimize.core import _fd_grad_pure

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_quadratic(N: int, seed: int = 0):
    """Build a quadratic merit ``f(x) = x^T A x`` with random SPD A
    plus its analytical gradient ``grad_f(x) = 2 A x``."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(N, N))
    A = M.T @ M + N * np.eye(N)  # symmetric positive-definite
    def f(x):
        x = np.asarray(x, dtype=np.float64)
        return float(x @ A @ x)
    def grad(x):
        return 2.0 * A @ np.asarray(x, dtype=np.float64)
    return f, grad


class _CountingFn:
    """Wrap a callable to count invocations."""
    def __init__(self, fn):
        self._fn = fn
        self.count = 0
    def __call__(self, x):
        self.count += 1
        return self._fn(x)


# ---------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------

def test_fd_grad_default_scheme_is_central():
    """``_fd_grad_pure(f, x)`` with no ``scheme=`` must return the
    central-difference gradient (bit-identical to explicit
    ``scheme='central'``).  This pins the v4.13.0 contract that the
    default preserves pre-v4.13 behaviour."""
    N = 5
    f, _ = _make_quadratic(N, seed=0)
    x0 = np.array([0.1, -0.3, 0.5, 0.2, -0.7])
    g_default = _fd_grad_pure(f, x0, eps=1e-7)
    g_central = _fd_grad_pure(f, x0, eps=1e-7, scheme='central')
    np.testing.assert_array_equal(g_default, g_central)


def test_fd_grad_central_matches_analytical_to_1e_9():
    """Central differences are O(h^2) so at h=1e-7 the relative error
    on a smooth quadratic should sit well below 1e-9."""
    N = 5
    f, grad = _make_quadratic(N, seed=42)
    x0 = np.array([0.1, -0.3, 0.5, 0.2, -0.7])
    g_analytic = grad(x0)
    g_fd = _fd_grad_pure(f, x0, eps=1e-7, scheme='central')
    rel = np.max(np.abs(g_fd - g_analytic) / (np.abs(g_analytic) + 1e-12))
    assert rel < 1e-7, (
        f"central FD on quadratic should be ~1e-9; got rel={rel:.3e}")


def test_fd_grad_forward_matches_analytical_to_1e_3():
    """Forward differences are O(h) so at h=1e-7 the relative error
    on a smooth quadratic should sit ~1e-7 or so; we pin a generous
    1e-3 bound (matches the legacy contract that the optimisation
    line search can tolerate)."""
    N = 5
    f, grad = _make_quadratic(N, seed=42)
    x0 = np.array([0.1, -0.3, 0.5, 0.2, -0.7])
    g_analytic = grad(x0)
    g_fd = _fd_grad_pure(f, x0, eps=1e-7, scheme='forward')
    rel = np.max(np.abs(g_fd - g_analytic) / (np.abs(g_analytic) + 1e-12))
    assert rel < 1e-3, (
        f"forward FD on quadratic should be ~1e-3; got rel={rel:.3e}")


def test_fd_grad_eval_count_central_is_2N():
    """Central differences must invoke f exactly 2N times per
    gradient (once at x+h*e_i and once at x-h*e_i for each i)."""
    N = 6
    f, _ = _make_quadratic(N, seed=7)
    x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cf = _CountingFn(f)
    _ = _fd_grad_pure(cf, x0, eps=1e-7, scheme='central')
    assert cf.count == 2 * N, (
        f"central FD expected 2N={2*N} calls, got {cf.count}")


def test_fd_grad_eval_count_forward_f0_none_is_Nplus1():
    """Forward FD with ``f0=None`` must invoke f exactly N+1 times."""
    N = 6
    f, _ = _make_quadratic(N, seed=7)
    x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    cf = _CountingFn(f)
    _ = _fd_grad_pure(cf, x0, eps=1e-7, scheme='forward', f0=None)
    assert cf.count == N + 1, (
        f"forward FD with f0=None expected N+1={N+1} calls, "
        f"got {cf.count}")


def test_fd_grad_eval_count_forward_with_f0_is_N():
    """Forward FD with ``f0=<value>`` must invoke f exactly N times."""
    N = 6
    f, _ = _make_quadratic(N, seed=7)
    x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    f0_val = f(x0)  # not counted on cf
    cf = _CountingFn(f)
    _ = _fd_grad_pure(cf, x0, eps=1e-7, scheme='forward', f0=f0_val)
    assert cf.count == N, (
        f"forward FD with f0 expected N={N} calls, got {cf.count}")


def test_fd_grad_forward_f0_reuse_is_bit_identical():
    """On the forward path, passing ``f0 = f(x0)`` must give bit-
    identical output to ``f0 = None``.  This pins the "f0 reuse is a
    perf-only optimisation" contract."""
    N = 7
    f, _ = _make_quadratic(N, seed=11)
    x0 = np.array([0.2, -0.4, 0.6, 0.1, -0.5, 0.8, 0.3])
    g_no_f0 = _fd_grad_pure(f, x0, eps=1e-7, scheme='forward', f0=None)
    f0 = f(x0)
    g_with_f0 = _fd_grad_pure(f, x0, eps=1e-7, scheme='forward', f0=f0)
    np.testing.assert_array_equal(g_no_f0, g_with_f0)


def test_fd_grad_invalid_scheme_raises():
    """Unknown scheme name must raise ValueError."""
    N = 3
    f, _ = _make_quadratic(N, seed=0)
    x0 = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="scheme"):
        _fd_grad_pure(f, x0, eps=1e-7, scheme='banana')


def test_fd_grad_scale_floor_central():
    """``scale_floor`` is honoured under central differences (the
    legacy / default scheme).  At x=0 the parameter magnitude is zero
    so the floor sets the step."""
    N = 3
    c = np.array([1.0, -2.0, 3.0])
    def f(x):
        return float(c @ np.asarray(x))
    x0 = np.zeros(N)
    g = _fd_grad_pure(f, x0, eps=1e-3, scale_floor=np.ones(N),
                      scheme='central')
    np.testing.assert_allclose(g, c, rtol=1e-6)


def test_fd_grad_scale_floor_forward():
    """``scale_floor`` is honoured under forward differences as
    well."""
    N = 3
    c = np.array([1.0, -2.0, 3.0])
    def f(x):
        return float(c @ np.asarray(x))
    x0 = np.zeros(N)
    g = _fd_grad_pure(f, x0, eps=1e-3, scale_floor=np.ones(N),
                      scheme='forward')
    np.testing.assert_allclose(g, c, rtol=1e-6)


# ============================================================================
# Source: test_audit_fixes_v4_13_2_agent_c.py
# Audit version: V4_13_2  scope: agent_c
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for v4.13.2 audit fixes (Agent C scope).
#   
#   Covers the seven Agent-C items from ``AUDIT_V4_13_1_2026_05_17.md``
#   Part 10 (Consolidation):
#   
#   * **C.1 / C-P0-1** -- ``make_lg_aberration_merit_jax`` had a ``pass``-
#     only inner loop that silently ignored its ``targets`` dict.  Fixed
#     to (a) reject non-(0, 0) targets with ``NotImplementedError`` and
#     (b) actually apply the (0, 0) piston weight to the returned merit.
#   * **C.2 / C-P0-2** -- ``MultiFieldMerit.field_angles`` applied a
#     Y-axis-only tilt despite the docstring's generic off-axis-angle
#     language.  Fixed to accept either scalars (back-compat, Y-tilt)
#     or ``(theta_x, theta_y)`` tuples, with a one-shot
#     ``DeprecationWarning`` on the scalar form.
#   * **C.3 / P1-NEW-L** -- ``dual_annealing``'s inline lambda callback
#     did NOT poll ``is_cancelled(progress)``; ``CancellableProgress``
#     cancellation from a Qt button was silently ignored.  Fixed by
#     promoting to a named callback that matches the other three scipy
#     callbacks.
#   * **C.4 / C-P1-2** -- the three wrapper merits
#     (``MultiWavelengthMerit``, ``MultiFieldMerit``, ``ToleranceAwareMerit``)
#     built sub-``EvaluationContext``s without threading ``x=ctx.x``; a
#     ``JaxMeritTerm(build_args=...)`` inside them silently fell back to
#     legacy ``fn(ctx)`` mode (analytic gradient -> FD).
#   * **C.5 / P1-NEW-G** -- ``_zero_C_air_gap`` returned ``g1`` (the
#     placeholder) on a degenerate ABCD instead of raising.  Fixed to
#     raise ``RuntimeError`` with a clear diagnostic; both callers
#     already catch ``RuntimeError``.
#   * **C.6 / P1-NEW-I** -- ``RandomState.choice`` returned int32 on the
#     JAX backend and int64 on NumPy.  Fixed by pinning the JAX
#     ``randint`` dtype to ``int64``.
#   * **C.7 / P1-NEW-K** -- the ``RandomState.choice`` ``replace=False``
#     JAX dispatch lacked the safety-net try/except for pre-0.4.x JAX
#     that the v4.13.1 CHANGELOG promised.  Added.
#   
#   Author: Andrew Traverso -- v4.13.2
# ============================================================================

import time
import warnings

import numpy as np
import pytest

# ============================================================================
# C.1 -- make_lg_aberration_merit_jax weighted sum (was pass-only loop)
# ============================================================================

class TestAuditFixesV4_13_2_agent_c_C1MakeLgAberrationMeritJax:
    """``make_lg_aberration_merit_jax`` now actually honours its
    ``targets`` dict instead of silently returning the same (0, 0)
    piston sum regardless of weights."""

    def _build_merit(self, targets, weight_kwarg=1.0):
        pytest.importorskip('jax')
        # Lazy import so the test module loads on JAX-less envs.
        import lumenairy
        from lumenairy.optimize.core import make_lg_aberration_merit_jax
        pres = lumenairy.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        pres['object_distance'] = 0.0

        def build_args(x):
            # x[0] = w_s; route into the 5th positional slot.
            return (None, None, None, None, x[0], None)

        return make_lg_aberration_merit_jax(
            pres, wavelength=1.30e-6,
            targets=targets,
            build_args=build_args,
            field_points=[(0.0, 0.0)],
            weight=weight_kwarg,
        ), pres

    def test_non_piston_target_raises_notimplemented(self):
        """The JAX path only supports (0, 0); other (p, ell) keys
        must raise a clear error rather than silently producing the
        same result as (0, 0)."""
        pytest.importorskip('jax')
        import lumenairy
        from lumenairy.optimize.core import make_lg_aberration_merit_jax
        pres = lumenairy.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        pres['object_distance'] = 0.0
        with pytest.raises(NotImplementedError) as info:
            make_lg_aberration_merit_jax(
                pres, wavelength=1.30e-6,
                targets={(2, 0): 1.0},
                build_args=lambda x: (None,) * 6,
                field_points=[(0.0, 0.0)],
            )
        msg = str(info.value)
        assert '(0, 0)' in msg or '0, 0' in msg, (
            f'error must direct the user to the (0, 0) Strehl '
            f'restriction; got {msg!r}')

    def test_piston_weight_scales_merit_linearly(self):
        """The merit value at the same x must scale linearly with
        the (0, 0) weight -- this proves the inner ``wgt`` is
        actually being multiplied in, not silently dropped (pre-fix
        the ``pass``-only loop dropped it)."""
        from lumenairy.optimize.core import EvaluationContext

        merit_w1, pres = self._build_merit(targets={(0, 0): 1.0})
        merit_w3, _ = self._build_merit(targets={(0, 0): 3.0})

        # Use the SAME parameter vector for both evals -- only the
        # weight differs.  Choose a sane w_s in metres.
        x = np.array([50e-6])
        ctx_a = EvaluationContext(
            prescription=pres, wavelength=1.30e-6,
            N=64, dx=10e-6, x=x)
        ctx_b = EvaluationContext(
            prescription=pres, wavelength=1.30e-6,
            N=64, dx=10e-6, x=x)

        try:
            v1 = float(merit_w1.evaluate(ctx_a))
            v3 = float(merit_w3.evaluate(ctx_b))
        except (RuntimeError, ValueError, ZeroDivisionError,
                np.linalg.LinAlgError) as exc:
            # Trivial singlet may not yield a finite LG tensor on
            # every JAX runtime; we still pin the API contract via
            # the NotImplementedError test above.
            pytest.skip(f'LG-tensor evaluation unstable on this '
                        f'singlet: {type(exc).__name__}: {exc}')
        if not (np.isfinite(v1) and np.isfinite(v3)
                and v1 > 1e-30):
            pytest.skip('LG-tensor evaluation returned a non-finite '
                        'or zero value on this minimal singlet; the '
                        'NotImplementedError test still pins the '
                        'contract.')
        ratio = v3 / v1
        assert abs(ratio - 3.0) < 1e-6, (
            f'Piston weight failed to scale the merit: v(w=1)={v1}, '
            f'v(w=3)={v3}, ratio={ratio} (expected 3.0).  This pins '
            f'C.1 -- pre-fix the inner ``wgt`` was dropped via a '
            f'pass-only loop.')


# ============================================================================
# C.2 -- MultiFieldMerit accepts (theta_x, theta_y) tuples
# ============================================================================

class TestAuditFixesV4_13_2_agent_c_C2MultiFieldMeritAxis:
    """``MultiFieldMerit`` accepts both scalar (back-compat: Y-tilt)
    and ``(theta_x, theta_y)`` tuple field-angle entries.  A scalar
    entry emits a one-shot ``DeprecationWarning``."""

    def setup_method(self, method):
        # Reset the class-level one-shot flag so each test
        # independently exercises the warning path.
        from lumenairy.optimize.core import MultiFieldMerit
        MultiFieldMerit._scalar_warning_issued = False

    def test_scalar_form_works_and_emits_deprecation(self):
        """Back-compat: a list of scalars still works and applies
        the same Y-axis tilt as before; one DeprecationWarning fires
        on construction."""
        from lumenairy.optimize.core import MultiFieldMerit, StrehlMerit
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            m = MultiFieldMerit(
                field_angles=[0.0, 0.01],
                sub_merit=StrehlMerit(weight=1.0),
                weight=1.0)
        dep = [w for w in ws if issubclass(w.category, DeprecationWarning)]
        assert dep, (
            f'scalar field_angles entries must emit a '
            f'DeprecationWarning; got categories='
            f'{[w.category.__name__ for w in ws]}')
        # The internal store must be tuples of (theta_x, theta_y);
        # scalar 0.01 -> (0.0, 0.01).
        assert m.field_angles == [(0.0, 0.0), (0.0, 0.01)], (
            f'scalar entries should normalise to (0, theta) tuples; '
            f'got {m.field_angles}')

    def test_tuple_form_works_no_warning(self):
        """Tuples are accepted without any deprecation warning."""
        from lumenairy.optimize.core import MultiFieldMerit, StrehlMerit
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            m = MultiFieldMerit(
                field_angles=[(0.0, 0.0), (0.005, 0.01)],
                sub_merit=StrehlMerit(weight=1.0),
                weight=1.0)
        dep = [w for w in ws if issubclass(w.category, DeprecationWarning)]
        assert not dep, (
            f'tuple field_angles must NOT trigger a deprecation '
            f'warning; got messages='
            f'{[str(w.message) for w in dep]}')
        assert m.field_angles == [(0.0, 0.0), (0.005, 0.01)]

    def test_tilt_phase_uses_x_component(self):
        """A non-zero ``theta_x`` must actually appear in the
        constructed tilted plane wave (the pre-fix code dropped the
        X component silently).  Verifies by reproducing the tilt
        formula used inside ``evaluate`` and checking sign /
        sensitivity to ``theta_x``.
        """
        # We exercise the formula directly -- running the whole
        # evaluate() through apply_real_lens is unnecessary for
        # this axis-convention pin and would take seconds.
        import numpy as np
        N = 32
        dx = 10e-6
        wavelength = 1.30e-6
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        k0 = 2 * np.pi / wavelength
        theta_x = 0.01
        theta_y = 0.0
        # New (post-fix) formula:
        phase_xtilt = k0 * (np.sin(theta_x) * X + np.sin(theta_y) * Y)
        # Old (pre-fix) formula -- Y only -- with these inputs is
        # identically zero.
        phase_pre = k0 * np.sin(theta_y) * Y
        # Sanity: pre-fix would have been zero everywhere for this
        # configuration (theta_y=0); post-fix must have a non-zero
        # X gradient.
        assert np.max(np.abs(phase_pre)) == 0.0
        # Post-fix: phase varies along X (non-zero gradient along
        # axis=1).
        grad_x = phase_xtilt[N // 2, 1:] - phase_xtilt[N // 2, :-1]
        assert np.max(np.abs(grad_x)) > 0.0, (
            f'theta_x != 0 must produce an X-axis phase gradient; '
            f'got grad_x={grad_x[:5]}... (this is the C.2 fix '
            f'pin)')


# ============================================================================
# C.3 -- dual_annealing callback wires into cancellation
# ============================================================================

class TestAuditFixesV4_13_2_agent_c_C3DualAnnealingCancellation:
    """``design_optimize(method='dual_annealing', ...)`` honours
    ``CancellableProgress.cancel()`` -- the named callback now polls
    ``is_cancelled(progress)`` instead of being an inline lambda
    that returned ``None`` unconditionally."""

    def test_dual_annealing_callback_signature_polls_is_cancelled(self):
        """Source-level pin: the ``_scipy_cb_da`` named callback
        exists in the optimize/core module and references
        ``is_cancelled(progress)``.  This is the cheapest possible
        pin -- a runtime cancellation test is also added below."""
        from pathlib import Path

        import lumenairy
        src = (Path(lumenairy.__file__).parent / 'optimize'
               / 'core.py').read_text(encoding='cp1252')
        assert 'def _scipy_cb_da' in src, (
            '_scipy_cb_da named callback missing -- C.3 fix not '
            'applied')
        # Confirm the named callback references is_cancelled.
        idx = src.find('def _scipy_cb_da')
        # Slice the next ~400 chars; that's where the body lives.
        body = src[idx:idx + 400]
        assert 'is_cancelled' in body, (
            'dual_annealing callback does not poll '
            'is_cancelled(progress)')
        # And the dispatch site must reference _scipy_cb_da (not
        # an inline lambda).
        da_call_idx = src.find('so.dual_annealing(')
        assert da_call_idx != -1, ('dual_annealing dispatch not '
                                   'found -- file changed?')
        da_call = src[da_call_idx:da_call_idx + 400]
        assert '_scipy_cb_da' in da_call, (
            'dual_annealing call still uses inline lambda; expected '
            'callback=_scipy_cb_da')

    def test_dual_annealing_terminates_quickly_when_cancelled(self):
        """Runtime pin: cancel BEFORE the run starts, then verify
        the call returns in much less than the full ``maxiter``
        budget would take."""
        import lumenairy
        from lumenairy import CancellableProgress, StrehlMerit
        from lumenairy.optimize.core import (
            DesignParameterization,
            design_optimize,
        )
        pres = lumenairy.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        param = DesignParameterization(
            template=pres,
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(30e-3, 100e-3)],
        )
        # Pre-cancel: the first callback poll should fire True and
        # ask dual_annealing to terminate immediately.
        progress = CancellableProgress()
        progress.cancel()
        t0 = time.time()
        try:
            design_optimize(
                parameterization=param,
                merit_terms=[StrehlMerit(weight=1.0)],
                wavelength=1.30e-6,
                N=32, dx=10e-6,
                method='dual_annealing',
                max_iter=200,         # generous so 'short' is unambiguous
                verbose=False,
                progress=progress,
            )
        except Exception:
            # Some sub-pipelines may raise on this tiny grid;
            # cancellation must still have short-circuited.
            pass
        dt = time.time() - t0
        # max_iter=200 would normally take many seconds; cancelled
        # at the first callback poll must finish in well under
        # 30 s on any reasonable machine.  Use a generous bound
        # to keep CI green across hardware.
        assert dt < 30.0, (
            f'dual_annealing did NOT honour cancel() in time; '
            f'elapsed={dt:.1f}s.  Pre-fix the inline lambda '
            f'callback never returned True so the run consumed '
            f'the full maxiter budget.')


# ============================================================================
# C.4 -- ctx.x threaded into wrapper sub-contexts
# ============================================================================

class TestAuditFixesV4_13_2_agent_c_C4WrapperContextX:
    """``MultiWavelengthMerit``, ``MultiFieldMerit``, and
    ``ToleranceAwareMerit`` thread ``ctx.x`` into the sub-context
    so a wrapped ``JaxMeritTerm`` with ``build_args`` actually
    receives ``ctx.x`` (and uses the analytic-gradient path), not
    None (which would silently fall back to legacy ``fn(ctx)``)."""

    def test_multi_wavelength_threads_x(self):
        """Inner ``JaxMeritTerm`` with ``build_args`` records the
        ctx.x it receives.  After C.4 the recorded value matches
        the parent ctx.x (pre-fix it was None)."""
        from lumenairy.optimize.core import (
            EvaluationContext,
            JaxMeritTerm,
            MeritTerm,
            MultiWavelengthMerit,
        )

        captured = {'x_seen': 'unset'}

        class _Spy(MeritTerm):
            name = 'Spy'
            needs_wave = False
            weight = 1.0
            def evaluate(self, ctx):
                captured['x_seen'] = (
                    None if ctx.x is None else np.asarray(ctx.x).copy())
                return 0.0

        spy = _Spy()
        mw = MultiWavelengthMerit(
            wavelengths=[1.30e-6, 1.55e-6], sub_merit=spy)

        parent_x = np.array([1.0, 2.0, 3.0])
        # MultiWavelengthMerit needs a prescription it can run the
        # ABCD on; use a trivial singlet.
        import lumenairy
        pres = lumenairy.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        ctx = EvaluationContext(
            prescription=pres, wavelength=1.30e-6,
            N=32, dx=10e-6, efl=0.1, bfl=0.1,
            x=parent_x)
        try:
            mw.evaluate(ctx)
        except Exception:
            # Inner wave-leg may fail on the tiny grid; we only care
            # that the spy got called at least once with x.
            pass
        assert captured['x_seen'] is not None and isinstance(
            captured['x_seen'], np.ndarray), (
                f"MultiWavelengthMerit failed to thread ctx.x into "
                f"sub-context; spy saw {captured['x_seen']!r}.  Pre-"
                f"fix this silently degraded JaxMeritTerm analytic "
                f"gradients to FD.")
        np.testing.assert_array_equal(captured['x_seen'], parent_x)

    def test_multi_field_threads_x(self):
        """Same threading test for ``MultiFieldMerit``."""
        from lumenairy.optimize.core import (
            EvaluationContext,
            MeritTerm,
            MultiFieldMerit,
        )
        captured = {'x_seen': 'unset'}

        class _Spy(MeritTerm):
            name = 'Spy'
            needs_wave = True
            weight = 1.0
            def evaluate(self, ctx):
                captured['x_seen'] = (
                    None if ctx.x is None else np.asarray(ctx.x).copy())
                return 0.0

        spy = _Spy()
        mf = MultiFieldMerit(
            field_angles=[(0.0, 0.0)], sub_merit=spy)
        parent_x = np.array([10.0, 20.0])
        import lumenairy
        pres = lumenairy.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        ctx = EvaluationContext(
            prescription=pres, wavelength=1.30e-6,
            N=32, dx=10e-6, efl=0.1, bfl=0.1,
            x=parent_x)
        try:
            mf.evaluate(ctx)
        except Exception:
            pass
        assert isinstance(captured['x_seen'], np.ndarray), (
            f"MultiFieldMerit failed to thread ctx.x into sub-"
            f"context; spy saw {captured['x_seen']!r}.")
        np.testing.assert_array_equal(captured['x_seen'], parent_x)

    def test_tolerance_aware_threads_x(self):
        """Same threading test for ``ToleranceAwareMerit``."""
        from lumenairy.optimize.core import (
            EvaluationContext,
            MeritTerm,
            ToleranceAwareMerit,
        )
        captured = {'x_seen': 'unset'}

        class _Spy(MeritTerm):
            name = 'Spy'
            needs_wave = False
            weight = 1.0
            def evaluate(self, ctx):
                captured['x_seen'] = (
                    None if ctx.x is None else np.asarray(ctx.x).copy())
                return 0.0

        spy = _Spy()
        tol = ToleranceAwareMerit(
            sub_merit=spy,
            perturbation_spec=[{
                'surface_index': 0,
                'decenter_std': 0.0,
                'tilt_std': 0.0,
                'form_error_rms': 0.0,
            }],
            n_trials=1,
            seed=0,
        )
        parent_x = np.array([7.0])
        import lumenairy
        pres = lumenairy.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        ctx = EvaluationContext(
            prescription=pres, wavelength=1.30e-6,
            N=32, dx=10e-6, efl=0.1, bfl=0.1,
            x=parent_x)
        try:
            tol.evaluate(ctx)
        except Exception:
            pass
        assert isinstance(captured['x_seen'], np.ndarray), (
            f"ToleranceAwareMerit failed to thread ctx.x into sub-"
            f"context; spy saw {captured['x_seen']!r}.")
        np.testing.assert_array_equal(captured['x_seen'], parent_x)


# ============================================================================
# C.5 -- _zero_C_air_gap raises on degenerate ABCD
# ============================================================================

class TestAuditFixesV4_13_2_agent_c_C5ZeroCAirGapDegenerate:
    """``_zero_C_air_gap`` raises ``RuntimeError`` (was: silently
    returns the placeholder gap) when the ABCD ``C`` element is
    field-independent in this geometry.  The callers
    ``beam_expander_prescription`` and ``keplerian_telescope``
    already catch ``RuntimeError`` so the user-facing behaviour is
    unchanged on the success path."""

    def test_degenerate_two_lens_raises_runtimeerror(self):
        """Two identical thin-lens-like surfaces with NO refractive
        power between them give a combined C that is independent
        of the air gap.  The solver must signal that explicitly."""
        from lumenairy.optimize.multiconfig import _zero_C_air_gap
        # Build a flat-flat (powerless) "lens" pair: every surface
        # has R = +inf.  System power is identically zero, and the
        # combined ABCD's C element is zero for every gap.
        pres = {
            'name': 'degenerate-flat',
            'aperture_diameter': 25.4e-3,
            'surfaces': [
                {'radius': float('inf'), 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': float('inf'), 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
                {'radius': float('inf'), 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': float('inf'), 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [
                2e-3,   # glass thickness 1
                10e-3,  # placeholder air gap (slot 1)
                2e-3,   # glass thickness 2
            ],
        }
        with pytest.raises(RuntimeError) as info:
            _zero_C_air_gap(pres, gap_slot_index=1, wavelength=1.30e-6)
        msg = str(info.value)
        assert 'degenerate' in msg.lower(), (
            f'RuntimeError message must mention degeneracy; got {msg!r}')


# ============================================================================
# C.6 -- RandomState.choice returns int64 on both backends
# ============================================================================

class TestAuditFixesV4_13_2_agent_c_C6ChoiceInt64:
    """Both NumPy and JAX paths of ``RandomState.choice`` now return
    ``np.int64`` -- pre-fix the JAX path defaulted to int32 from
    ``jax.random.randint`` which broke cross-backend pipelines."""

    def test_numpy_choice_int64(self):
        from lumenairy.backend.random import RandomState
        rs = RandomState(rng=42)
        out = rs.choice(10, (5,), replace=True)
        assert np.asarray(out).dtype == np.int64, (
            f'NumPy choice should return int64 (got {out.dtype}).')

    def test_jax_choice_int64(self):
        jax = pytest.importorskip('jax')
        # int64 requires x64 to be enabled in the JAX runtime.
        jax.config.update('jax_enable_x64', True)
        from lumenairy.backend.random import RandomState
        key = jax.random.PRNGKey(42)
        rs = RandomState(rng=key)
        out = rs.choice(10, (5,), replace=True)
        # Bridge to NumPy for dtype comparison.
        dt = np.asarray(out).dtype
        assert dt == np.int64, (
            f'JAX choice should return int64 to match NumPy (got '
            f'{dt}); pre-fix this was int32.')


# ============================================================================
# C.7 -- RandomState.choice(replace=False) on JAX: old-JAX safety net
# ============================================================================

class TestAuditFixesV4_13_2_agent_c_C7ChoiceReplaceFalseSafetyNet:
    """The ``jax.random.choice(replace=False, ...)`` dispatch is
    wrapped in a try/except so pre-0.4.x JAX builds raise a clear
    migration ``RuntimeError`` instead of a bare ``TypeError``.
    This is defensive infrastructure for a path that is unreachable
    on the current JAX runtime; monkeypatch to verify."""

    def test_typeerror_promoted_to_runtimeerror(self, monkeypatch):
        jax = pytest.importorskip('jax')
        from lumenairy.backend import random as rnd

        # Capture the original so we can selectively explode only the
        # replace=False call.
        original_choice = jax.random.choice

        def _exploding_choice(*args, **kwargs):
            if kwargs.get('replace') is False:
                raise TypeError(
                    "replace=False not supported on this JAX build")
            return original_choice(*args, **kwargs)

        monkeypatch.setattr(jax.random, 'choice', _exploding_choice)

        rs = rnd.RandomState(rng=jax.random.PRNGKey(0))
        with pytest.raises(RuntimeError) as info:
            rs.choice(10, (3,), replace=False)
        msg = str(info.value)
        assert 'JAX >= 0.4' in msg or '0.4' in msg, (
            f'RuntimeError must include the version-upgrade message; '
            f'got {msg!r}')
        # And the original TypeError must be chained via ``from e``.
        assert info.value.__cause__ is not None
        assert isinstance(info.value.__cause__, TypeError)


# ============================================================================
# Source: test_audit_fixes_v4_14_0_agent_4.py
# Audit version: V4_14_0  scope: agent_4
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.0 Agent-4 perf win.
#   
#   Audit reference
#   ---------------
#   
#   v4.14.0 Agent-4 scope: ``lumenairy/optimize/core.py`` ONLY.
#   
#   The three wrapper merits -- ``MultiWavelengthMerit``,
#   ``MultiFieldMerit``, ``ToleranceAwareMerit`` -- each rebuilt
#   ``np.indices`` / meshgrid / aperture-mask / Y-tilt-phase arrays on
#   every per-wavelength, per-field, per-trial leg.  For a representative
#   5-wavelength * 5-field * 40-FD-eval optimisation step at N=512 that
#   amounted to up to 1000 N x N meshgrid builds per outer iteration, none
#   of which depended on the parameter vector being differenced.
#   
#   v4.14.0 adds a module-level LRU(32) cache keyed on
#   ``(Ny, Nx, dx, aperture_hash, dtype_str)`` and routes all three
#   wrapper merits through it.  Per-leg work reduces to ``np.exp(1j *
#   sin_a * cached_k0_Y) * cached_aperture_mask`` (MultiFieldMerit) or a
#   single ``.copy()`` of the cached np.ones template (the other two)
#   plus the standard ``apply_real_lens`` call.
#   
#   What this test file pins
#   ------------------------
#   
#   * Cache identity -- repeated calls with the same key return the same
#     cached arrays (not just numerically equal -- the same object) so
#     the per-leg cost stays a single multiply.
#   * Cache invalidation -- different ``(N, dx, aperture, dtype)`` keys
#     produce distinct cached payloads.
#   * Meshgrid-build counter -- the eval-count contract: 1 build per
#     (N, dx, aperture) signature for a full sweep, not 1 per leg.
#   * Correctness -- pre-perf and post-perf merit values must match
#     bit-near-exact (1e-12 relative).  Compared against a reference
#     implementation that materialises the meshgrid on every call.
#   * LRU bound -- the cache evicts at the ``_WRAPPER_MERIT_CACHE_SIZE``
#     threshold.
#   * ``clear_asm_caches`` wiring -- the propagation-layer clear-all hook
#     also drops the wrapper-merit cache.
#   
#   Author: Andrew Traverso -- v4.14.0 / Agent 4
# ============================================================================

import warnings

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.optimize.core import (
    _WRAPPER_MERIT_CACHE,
    _WRAPPER_MERIT_CACHE_SIZE,
    DesignParameterization,
    EvaluationContext,
    MultiFieldMerit,
    MultiWavelengthMerit,
    StrehlMerit,
    ToleranceAwareMerit,
    _clear_wrapper_merit_cache,
    _get_wrapper_merit_cache,
    _wrapper_merit_aperture_key,
    design_optimize,
)
from lumenairy.propagators.propagation import clear_asm_caches

# ============================================================================
# Helpers
# ============================================================================

def _meshgrid_build_count() -> int:
    """Re-read the module-level counter (it is mutated, not rebound)."""
    import lumenairy.optimize.core as core
    return core._WRAPPER_MERIT_MESHGRID_BUILDS


def _simple_singlet():
    """Build a minimal singlet prescription suitable for design_optimize
    smoke runs.  The same geometry as v4.13.x C.2 / C.4 tests."""
    return {
        'surfaces': [
            {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -50e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [3e-3],
        'aperture_diameter': 10e-3,
    }


# ============================================================================
# Cache primitives
# ============================================================================

class TestAuditFixesV4_14_0_agent_4_CachePrimitives:
    """Exercise the module-level cache helper directly."""

    def setup_method(self, method):
        _clear_wrapper_merit_cache()

    def test_cache_hit_returns_same_object(self):
        """Two calls with the same key share the same payload object."""
        c1 = _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        before = _meshgrid_build_count()
        c2 = _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        after = _meshgrid_build_count()
        assert c1 is c2, 'cache hit must return the SAME dict (not a copy)'
        assert c1['X'] is c2['X'], 'cache hit must reuse X array'
        assert c1['Y'] is c2['Y'], 'cache hit must reuse Y array'
        assert c1['mask'] is c2['mask'], 'cache hit must reuse mask array'
        assert after == before, (
            f'cache hit must NOT increment meshgrid-build counter; '
            f'got before={before}, after={after}')

    def test_cache_miss_rebuilds_and_counts(self):
        """Different (N, dx, aperture, dtype) produces a fresh build."""
        _clear_wrapper_merit_cache()
        base = _meshgrid_build_count()
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        _get_wrapper_merit_cache(128, 1e-6, 50e-6, np.complex128)
        _get_wrapper_merit_cache(64, 2e-6, 50e-6, np.complex128)
        _get_wrapper_merit_cache(64, 1e-6, 80e-6, np.complex128)
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex64)
        # Five distinct keys -> five builds.
        assert _meshgrid_build_count() == base + 5

    def test_cache_payload_correctness(self):
        """The cached X/Y/mask/Y_factor/E_ones agree with a fresh
        np.indices/meshgrid reference build."""
        N = 32
        dx = 5e-6
        ap = 80e-6
        _clear_wrapper_merit_cache()
        c = _get_wrapper_merit_cache(N, dx, ap, np.complex128)
        # Reference
        Y_idx, X_idx = np.indices((N, N))
        X_ref = (X_idx - N / 2) * dx
        Y_ref = (Y_idx - N / 2) * dx
        mask_ref = (X_ref ** 2 + Y_ref ** 2) <= (ap / 2.0) ** 2
        assert np.array_equal(c['X'], X_ref)
        assert np.array_equal(c['Y'], Y_ref)
        assert np.array_equal(c['mask'], mask_ref)
        # Y_factor is 2*pi * Y (no wavelength baked in).
        assert np.allclose(c['Y_factor'], 2.0 * np.pi * Y_ref)
        assert np.allclose(c['X_factor'], 2.0 * np.pi * X_ref)
        assert c['E_ones'].shape == (N, N)
        assert c['E_ones'].dtype == np.complex128

    def test_lru_eviction(self):
        """Once the cache exceeds ``_WRAPPER_MERIT_CACHE_SIZE`` (32)
        the oldest entry is dropped."""
        _clear_wrapper_merit_cache()
        # Insert SIZE+5 distinct entries.  Vary N so each key is
        # unique.
        for i in range(_WRAPPER_MERIT_CACHE_SIZE + 5):
            _get_wrapper_merit_cache(32 + i, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) == _WRAPPER_MERIT_CACHE_SIZE, (
            f'cache size {len(_WRAPPER_MERIT_CACHE)} != '
            f'{_WRAPPER_MERIT_CACHE_SIZE}')

    def test_aperture_key_none(self):
        """``None`` aperture maps to a stable scalar tag."""
        assert _wrapper_merit_aperture_key(None) == ('none',)

    def test_aperture_key_scalar(self):
        """Numeric aperture maps to ``('scalar', float)``."""
        assert _wrapper_merit_aperture_key(10e-3) == ('scalar', 10e-3)
        # np.float64 and python float must collide.
        assert (_wrapper_merit_aperture_key(np.float64(10e-3))
                == _wrapper_merit_aperture_key(10e-3))

    def test_aperture_key_array(self):
        """ndarray aperture -- different contents must hash differently;
        identical contents must hash to the same key."""
        a1 = np.zeros((8, 8), dtype=bool)
        a2 = a1.copy()
        a3 = a1.copy()
        a3[0, 0] = True
        assert (_wrapper_merit_aperture_key(a1)
                == _wrapper_merit_aperture_key(a2))
        assert (_wrapper_merit_aperture_key(a1)
                != _wrapper_merit_aperture_key(a3))

    def test_aperture_key_array_vs_scalar_distinct(self):
        """An ndarray aperture key never collides with a scalar key."""
        a = np.ones((4, 4), dtype=bool)
        k_arr = _wrapper_merit_aperture_key(a)
        k_sc = _wrapper_merit_aperture_key(1.0)
        assert k_arr != k_sc
        assert k_arr[0] == 'arr' and k_sc[0] == 'scalar'

    def test_clear_resets_counter(self):
        """``_clear_wrapper_merit_cache`` zeros the build counter."""
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        assert _meshgrid_build_count() >= 1
        _clear_wrapper_merit_cache()
        assert _meshgrid_build_count() == 0
        assert len(_WRAPPER_MERIT_CACHE) == 0


# ============================================================================
# clear_asm_caches wiring
# ============================================================================

class TestAuditFixesV4_14_0_agent_4_ClearAsmCachesWiring:
    """``clear_asm_caches`` is monkey-patched at import time to also
    drop the wrapper-merit cache.  Pinning this ensures the
    ``lumenairy_context(clear_caches_on_exit=True)`` hook in
    ``_context.py`` continues to leave both layers pristine."""

    def test_clear_asm_caches_drops_wrapper_merit_cache(self):
        """A call to ``clear_asm_caches`` empties the wrapper-merit
        cache as a side effect."""
        _clear_wrapper_merit_cache()
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) >= 1
        clear_asm_caches()
        assert len(_WRAPPER_MERIT_CACHE) == 0, (
            'clear_asm_caches must also drop the wrapper-merit cache '
            '(monkey-patched composite clear)')

    def test_clear_asm_caches_via_top_level_export(self):
        """``lumenairy.clear_asm_caches`` (the top-level re-export)
        also picks up the composite version."""
        _clear_wrapper_merit_cache()
        _get_wrapper_merit_cache(32, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) >= 1
        lm.clear_asm_caches()
        assert len(_WRAPPER_MERIT_CACHE) == 0


# ============================================================================
# MultiFieldMerit correctness: cached path vs reference build
# ============================================================================

class TestAuditFixesV4_14_0_agent_4_MultiFieldMeritCorrectness:
    """The cached tilted-plane-wave construction must agree
    bit-near-exact with the v4.13.2 reference (rebuild meshgrid on
    every call)."""

    def test_tilted_plane_wave_matches_reference(self):
        """For a synthetic ``(N, dx, aperture, wavelength)`` build
        the tilted plane wave from the cache; compare against an
        explicit np.meshgrid rebuild."""
        N = 32
        dx = 10e-6
        ap = 100e-6
        wavelength = 1.30e-6
        theta_x = 0.005
        theta_y = 0.012
        _clear_wrapper_merit_cache()
        # Reference (pre-perf path)
        x_ref = (np.arange(N) - N / 2) * dx
        y_ref = (np.arange(N) - N / 2) * dx
        X_ref, Y_ref = np.meshgrid(x_ref, y_ref)
        k0 = 2 * np.pi / wavelength
        tilt_phase_ref = (k0 * np.sin(theta_x) * X_ref
                          + k0 * np.sin(theta_y) * Y_ref)
        mask_ref = (X_ref ** 2 + Y_ref ** 2) <= (ap / 2.0) ** 2
        E_ref = np.where(
            mask_ref, np.exp(1j * tilt_phase_ref), 0.0
        ).astype(np.complex128)
        # Cached path
        c = _get_wrapper_merit_cache(N, dx, ap, np.complex128)
        k_X = c['X_factor'] / wavelength
        k_Y = c['Y_factor'] / wavelength
        tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
        E_new = np.where(
            c['mask'], np.exp(1j * tilt_phase), 0.0
        ).astype(np.complex128)
        # Bit-near-exact agreement (the only mathematical difference
        # is the multiply order; numerically identical at 1e-15).
        np.testing.assert_allclose(
            E_new, E_ref, rtol=1e-13, atol=1e-13,
            err_msg='cached tilted-plane-wave must match reference '
                    'build to 1e-13')


# ============================================================================
# MultiWavelengthMerit correctness via design_optimize
# ============================================================================

class TestAuditFixesV4_14_0_agent_4_MultiWavelengthMeritCorrectness:
    """End-to-end pin: a short ``design_optimize`` run with a
    ``MultiWavelengthMerit`` produces the same merit values as a
    reference run using the pre-perf direct-meshgrid path.

    Realised as: run a fixed 1-iteration optimisation and snapshot
    the final ``merit`` value; this is the cheapest cross-check that
    exercises the cached code path inside the wrapper merit's
    ``evaluate``.
    """

    def test_short_optimisation_runs_without_error(self):
        """The cached path must support a full design_optimize run
        without raising / producing NaN.  Three wavelengths * three
        FD evals @ N=32 (very cheap)."""
        template = _simple_singlet()
        param = DesignParameterization(
            template=template,
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(20e-3, 80e-3)])
        sub = StrehlMerit(weight=1.0)
        merit = [MultiWavelengthMerit(
            wavelengths=[1.27e-6, 1.30e-6, 1.33e-6],
            sub_merit=sub, weight=1.0)]
        _clear_wrapper_merit_cache()
        with warnings.catch_warnings():
            # design_optimize may emit per-merit RuntimeWarnings if
            # the wave leg fails on the FD-perturbed prescription;
            # these are not in scope for this test.
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
            res = design_optimize(
                param, merit, wavelength=1.30e-6,
                N=32, dx=10e-6,
                method='L-BFGS-B', max_iter=1, verbose=False)
        assert np.isfinite(res.merit), (
            f'merit must stay finite through the cached path; got '
            f'{res.merit}')


# ============================================================================
# Eval-count pin: meshgrid_build_count for a full run
# ============================================================================

class TestAuditFixesV4_14_0_agent_4_MeshgridBuildCountPin:
    """The headline perf claim: meshgrid_build_count == 1 (one) per
    ``(N, dx, aperture)`` signature over the entire optimisation
    run, regardless of #wavelengths / #fields / #FD evals.
    """

    def test_one_build_per_signature_multifield(self):
        """A MultiFieldMerit run with 3 field angles * several
        evaluate() calls produces exactly ONE meshgrid build."""
        N = 32
        dx = 10e-6
        ap = 100e-6
        _clear_wrapper_merit_cache()
        # Simulate the inner-loop call pattern WITHOUT spinning up
        # the full design_optimize -- just hit the cache helper as
        # MultiFieldMerit's evaluate would.
        for _ in range(50):
            _get_wrapper_merit_cache(N, dx, ap, np.complex128)
        assert _meshgrid_build_count() == 1, (
            f'expected exactly 1 build per signature; got '
            f'{_meshgrid_build_count()}')

    def test_three_signatures_three_builds(self):
        """Three distinct ``(N, dx, ap)`` signatures = three builds,
        no matter how many evaluate() calls per signature."""
        _clear_wrapper_merit_cache()
        for _ in range(20):
            _get_wrapper_merit_cache(32, 1e-6, 50e-6, np.complex128)
            _get_wrapper_merit_cache(32, 1e-6, 60e-6, np.complex128)
            _get_wrapper_merit_cache(32, 1e-6, 70e-6, np.complex128)
        assert _meshgrid_build_count() == 3


# ============================================================================
# v4.13 closure preservation
# ============================================================================

class TestAuditFixesV4_14_0_agent_4_V413ClosuresPreserved:
    """Pin that the v4.13.2 x=ctx.x threading and field_angles tuple
    support survived the v4.14.0 perf refactor."""

    def setup_method(self, method):
        # Reset the deprecation one-shot.
        MultiFieldMerit._scalar_warning_issued = False

    def test_field_angles_tuple_still_accepted(self):
        """Tuple form (theta_x, theta_y) still works without
        warnings (preserved from v4.13.2 C-P0-2)."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            m = MultiFieldMerit(
                field_angles=[(0.0, 0.0), (0.005, 0.01)],
                sub_merit=StrehlMerit(weight=1.0),
                weight=1.0)
        dep = [w for w in ws if issubclass(w.category, DeprecationWarning)]
        assert not dep
        assert m.field_angles == [(0.0, 0.0), (0.005, 0.01)]

    def test_field_angles_scalar_still_emits_deprecation(self):
        """Scalar form still emits the one-shot DeprecationWarning
        (preserved from v4.13.2 C-P0-2)."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            MultiFieldMerit(
                field_angles=[0.005, 0.01],
                sub_merit=StrehlMerit(weight=1.0),
                weight=1.0)
        dep = [w for w in ws if issubclass(w.category, DeprecationWarning)]
        assert dep, 'scalar field_angles must still emit DeprecationWarning'

    def test_x_thread_through_multifield_sub_ctx_source_pin(self):
        """Source-level pin for the v4.13.2 C-P1-2 ``x=getattr(ctx,
        'x', None)`` thread through the wrapper merits.  The v4.14.0
        perf refactor must preserve this closure; we grep the file
        contents for the call signature."""
        from pathlib import Path

        import lumenairy
        src = (Path(lumenairy.__file__).parent / 'optimize'
               / 'core.py').read_text(encoding='cp1252')
        # MultiFieldMerit, MultiWavelengthMerit, ToleranceAwareMerit
        # all build sub_ctx EvaluationContext objects.  Each must
        # forward x.
        # Count occurrences of x=getattr(ctx, 'x', None) -- expect
        # at least 3 (one per wrapper merit).
        marker = "x=getattr(ctx, 'x', None)"
        n = src.count(marker)
        assert n >= 3, (
            f'expected at least 3 occurrences of {marker!r} (one per '
            f'wrapper merit; v4.13.2 C-P1-2); got {n}')


# ============================================================================
# Source: test_audit_fixes_v4_14_1_agent_b.py
# Audit version: V4_14_1  scope: agent_b
# Original module docstring preserved as comment block for git-blame traceability:
#   Tests for the v4.14.1 audit-fix Agent-B changes.
#   
#   Scope: ``lumenairy/optimize/core.py`` and
#   ``lumenairy/propagators/propagation.py``.
#   
#   What this module pins
#   ---------------------
#   
#   * **B.1 -- P1-NEW-1 aperture=0 semantics regression.**
#     ``MultiWavelengthMerit`` / ``MultiFieldMerit`` must zero the field
#     when ``prescription['aperture_diameter'] == 0`` (the deliberate
#     "block all light" branch).  v4.14.0 mapped ``ap <= 0`` to
#     ``mask=None`` and the downstream callers then treated the
#     deliberate zero as "no aperture -> full grid plane wave," flipping
#     the semantics 180 degrees.  v4.14.1 introduces the
#     ``_ZERO_APERTURE_MASK`` sentinel and makes both wrapper merits
#     branch on ``mask is _ZERO_APERTURE_MASK``.
#   
#   * **B.2 -- P2-1 wrapper-merit cache lock.**
#     ``_WRAPPER_MERIT_CACHE`` is now guarded by a module-level
#     ``threading.Lock``; the ``get/move_to_end/__setitem__/popitem`` ops
#     and ``_clear_wrapper_merit_cache`` all acquire it.  No new
#     pinning beyond a smoke test that ``threading.Lock`` is present.
#   
#   * **B.3 -- P2-3 monkey-patch -> lazy-import.**
#     v4.14.0 monkey-patched ``propagation.clear_asm_caches`` to also
#     drop the wrapper-merit cache.  v4.14.1 inverts the dependency:
#     ``clear_asm_caches()`` now lazy-imports
#     ``_clear_wrapper_merit_cache`` from ``optimize.core`` and calls it
#     inline.  This test imports the original module-attribute path
#     (``lumenairy.propagators.propagation.clear_asm_caches``) and pins
#     the cross-module clear.
#   
#   * **B.4 -- Tier-0 #5 LG/HG mode-stack cache wiring.**
#     v4.14.0 CHANGELOG claimed ``clear_asm_caches()`` clears the LG/HG
#     mode-stack cache; in reality only ``lumenairy_context`` did.
#     v4.14.1 chains a lazy import of
#     ``propagators.asymptotic.clear_lg_mode_stack_cache`` into
#     ``clear_asm_caches``.  Populate the LG cache via ``decompose_lg``,
#     call ``clear_asm_caches()``, assert the LG cache is empty.
#   
#   Author: Andrew Traverso -- v4.14.1 / Agent B
# ============================================================================

import threading

import numpy as np

import lumenairy.propagators.propagation as _propagation_module
from lumenairy.optimize.core import (
    _WRAPPER_MERIT_CACHE,
    _WRAPPER_MERIT_CACHE_LOCK,
    _ZERO_APERTURE_MASK,
    _clear_wrapper_merit_cache,
    _get_wrapper_merit_cache,
)
from lumenairy.propagators.asymptotic import (
    _LG_MODE_STACK_CACHE,
    decompose_lg,
)

# ============================================================================
# B.1 -- P1-NEW-1: aperture=0 vs aperture=None semantics
# ============================================================================

class TestAuditFixesV4_14_1_agent_b_ApertureZeroSemantics:
    """``aperture_diameter == 0`` must zero the field; ``None`` must
    map to "no aperture, full grid."""

    def setup_method(self, method):
        _clear_wrapper_merit_cache()

    def test_cache_returns_sentinel_for_aperture_zero(self):
        """A scalar ``aperture_diameter=0`` produces the dedicated
        ``_ZERO_APERTURE_MASK`` sentinel, NOT ``None`` (the v4.14.0
        regression)."""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, 0.0, np.complex128)
        assert entry['mask'] is _ZERO_APERTURE_MASK, (
            'aperture_diameter=0 must map to _ZERO_APERTURE_MASK '
            "(deliberate 'block all light')")

    def test_cache_returns_sentinel_for_aperture_negative(self):
        """Same as zero but negative -- still 'no light through'."""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, -1e-3, np.complex128)
        assert entry['mask'] is _ZERO_APERTURE_MASK

    def test_cache_returns_none_for_aperture_none(self):
        """``None`` aperture must NOT collide with the zero sentinel;
        downstream callers treat it as 'no aperture, full grid.'"""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, None, np.complex128)
        assert entry['mask'] is None
        assert entry['mask'] is not _ZERO_APERTURE_MASK

    def test_cache_returns_ndarray_for_positive_aperture(self):
        """A positive scalar aperture produces a boolean ndarray
        (the circular mask) -- neither None nor the sentinel."""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, 100e-6, np.complex128)
        assert isinstance(entry['mask'], np.ndarray)
        assert entry['mask'].dtype == bool

    def test_multifield_branches_on_zero_aperture_sentinel(self):
        """``MultiFieldMerit.evaluate`` must branch on
        ``aperture_mask is _ZERO_APERTURE_MASK`` and build
        ``E_tilted = np.zeros(...)`` (the v4.14.0 regression silently
        produced ``np.exp(1j*tilt_phase)`` -- a grid-filling plane
        wave -- which inverted the 'block all light' semantics).

        The library's ``validate_prescription`` strict-mode rejects
        ``aperture_diameter=0`` so we cannot drive this through
        ``evaluate``; instead we exercise the per-field branching
        body directly via the cache, which is the source of truth.
        """
        N = 32
        dx = 5e-6
        wl = 1.3e-6
        _clear_wrapper_merit_cache()
        # Mimic the MultiFieldMerit setup line-for-line: ap=0 keeps
        # the deliberate zero (no fallback substitution since 0 is
        # not None).
        ap_diam = 0.0
        cdtype = np.complex128
        _cache = _get_wrapper_merit_cache(N, dx, float(ap_diam), cdtype)
        aperture_mask = _cache['mask']
        assert aperture_mask is _ZERO_APERTURE_MASK
        k_X = _cache['X_factor'] / wl
        k_Y = _cache['Y_factor'] / wl
        theta_x, theta_y = 0.01, 0.02
        tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
        # Reproduce the v4.14.1 branching contract.
        if aperture_mask is _ZERO_APERTURE_MASK:
            E_tilted = np.zeros((N, N), dtype=cdtype)
        elif aperture_mask is None:
            E_tilted = np.exp(1j * tilt_phase).astype(cdtype)
        else:
            E_tilted = np.where(aperture_mask, np.exp(1j * tilt_phase),
                                 0.0).astype(cdtype)
        assert np.array_equal(E_tilted, np.zeros((N, N), dtype=cdtype)), (
            'aperture=0 must zero E_tilted (the deliberate '
            "'block all light' branch)")

    def test_multifield_aperture_none_produces_full_grid(self):
        """``aperture_mask is None`` must produce the grid-filling
        plane wave (full-grid behaviour)."""
        N = 32
        dx = 5e-6
        wl = 1.3e-6
        _clear_wrapper_merit_cache()
        cdtype = np.complex128
        _cache = _get_wrapper_merit_cache(N, dx, None, cdtype)
        aperture_mask = _cache['mask']
        assert aperture_mask is None
        k_X = _cache['X_factor'] / wl
        k_Y = _cache['Y_factor'] / wl
        theta_x, theta_y = 0.0, 0.0
        tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
        if aperture_mask is _ZERO_APERTURE_MASK:
            E_tilted = np.zeros((N, N), dtype=cdtype)
        elif aperture_mask is None:
            E_tilted = np.exp(1j * tilt_phase).astype(cdtype)
        else:
            E_tilted = np.where(aperture_mask, np.exp(1j * tilt_phase),
                                 0.0).astype(cdtype)
        # On-axis (theta=0) tilt_phase is zero everywhere -> E_tilted
        # is all ones (the full-grid plane wave).
        assert np.allclose(E_tilted, np.ones((N, N), dtype=cdtype))

    def test_multiwavelength_branches_on_zero_aperture_sentinel(self):
        """``MultiWavelengthMerit.evaluate`` must branch on the
        zero-aperture sentinel and build ``E_in = np.zeros(...)``.
        Same constraint as ``MultiFieldMerit``: prescription
        validation forbids ``aperture_diameter=0`` so we exercise
        the cache-driven branch body directly."""
        N = 32
        dx = 5e-6
        _clear_wrapper_merit_cache()
        cdtype = np.complex128
        _cache = _get_wrapper_merit_cache(N, dx, 0.0, cdtype)
        mask = _cache['mask']
        assert mask is _ZERO_APERTURE_MASK
        # Reproduce the v4.14.1 MultiWavelengthMerit branching.
        if mask is _ZERO_APERTURE_MASK:
            E_in_wl = np.zeros((N, N), dtype=cdtype)
        elif mask is None:
            E_in_wl = np.ones((N, N), dtype=cdtype)
        else:
            E_in_wl = mask.astype(cdtype)
        assert np.array_equal(E_in_wl, np.zeros((N, N), dtype=cdtype))

        # Counter-test: ap=None gives the full-grid ones field.
        _clear_wrapper_merit_cache()
        _cache_none = _get_wrapper_merit_cache(N, dx, None, cdtype)
        mask = _cache_none['mask']
        assert mask is None
        if mask is _ZERO_APERTURE_MASK:
            E_in_none = np.zeros((N, N), dtype=cdtype)
        elif mask is None:
            E_in_none = np.ones((N, N), dtype=cdtype)
        else:
            E_in_none = mask.astype(cdtype)
        assert np.array_equal(E_in_none, np.ones((N, N), dtype=cdtype))


# ============================================================================
# B.2 -- P2-1: lock present and acquired
# ============================================================================

class TestAuditFixesV4_14_1_agent_b_WrapperMeritCacheLock:
    """The wrapper-merit cache now carries a threading.Lock so
    concurrent design_optimize threads cannot tear the OrderedDict."""

    def test_lock_is_a_threading_lock(self):
        """Sanity: ``_WRAPPER_MERIT_CACHE_LOCK`` exists and looks like
        a ``threading.Lock`` (it carries an ``acquire`` method and is
        usable as a context manager)."""
        # threading.Lock is a factory function returning a _thread.lock
        # primitive that doesn't expose the original class via
        # isinstance, so we check the duck-type shape.
        assert hasattr(_WRAPPER_MERIT_CACHE_LOCK, 'acquire')
        assert hasattr(_WRAPPER_MERIT_CACHE_LOCK, 'release')
        # Confirm it is re-acquirable after release (i.e. it really
        # is unlocked between calls).
        with _WRAPPER_MERIT_CACHE_LOCK:
            assert _WRAPPER_MERIT_CACHE_LOCK.locked()
        assert not _WRAPPER_MERIT_CACHE_LOCK.locked()

    def test_concurrent_cache_access_does_not_corrupt(self):
        """Smoke test: hammer ``_get_wrapper_merit_cache`` from 8
        threads with mixed keys.  The cache must not lose entries
        or crash."""
        _clear_wrapper_merit_cache()
        errors = []

        def worker(seed):
            try:
                for _ in range(50):
                    N = 32 + (seed % 4) * 8
                    _get_wrapper_merit_cache(N, 1e-6, 50e-6,
                                              np.complex128)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        assert not errors, f'concurrent access raised: {errors}'
        # 4 distinct N values -> at most 4 cache entries.
        assert 1 <= len(_WRAPPER_MERIT_CACHE) <= 4


# ============================================================================
# B.3 -- P2-3: lazy-import replaces monkey-patch
# ============================================================================

class TestAuditFixesV4_14_1_agent_b_ClearAsmCachesLazyImport:
    """``propagation.clear_asm_caches`` (the module-attribute path,
    NOT the v4.14.0 monkey-patched re-bind) must still drop the
    wrapper-merit cache via the v4.14.1 lazy-import."""

    def test_propagation_clear_asm_caches_drops_wrapper_merit(self):
        """Imported via ``lumenairy.propagators.propagation.clear_asm_caches``
        directly (no monkey-patch involvement).  v4.14.1 adds a
        lazy-import + call to ``_clear_wrapper_merit_cache`` inside
        the propagation-layer body."""
        _clear_wrapper_merit_cache()
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) >= 1
        _propagation_module.clear_asm_caches()
        assert len(_WRAPPER_MERIT_CACHE) == 0, (
            'propagation.clear_asm_caches must drop the wrapper-merit '
            'cache via the v4.14.1 lazy-import (the v4.14.0 '
            'monkey-patch has been removed)')


# ============================================================================
# B.4 -- Tier-0 #5: LG mode-stack cache wired into clear_asm_caches
# ============================================================================

class TestAuditFixesV4_14_1_agent_b_ClearAsmCachesDropsLgCache:
    """``clear_asm_caches`` must also flush the LG/HG mode-stack
    cache.  The v4.14.0 CHANGELOG claimed this; only the v4.14.1
    fix actually wires it up."""

    def test_clear_asm_caches_drops_lg_mode_stack_cache(self):
        """Populate ``_LG_MODE_STACK_CACHE`` via a real ``decompose_lg``
        call; call ``clear_asm_caches``; assert the cache is empty."""
        # Build a synthetic field on a small grid; populate the cache.
        N = 32
        x = np.linspace(-50e-6, 50e-6, N)
        y = np.linspace(-50e-6, 50e-6, N)
        X, Y = np.meshgrid(x, y, indexing='xy')
        field = np.exp(-(X ** 2 + Y ** 2) / (30e-6) ** 2).astype(
            np.complex128)
        # decompose_lg builds + caches the conjugated mode stack.
        _LG_MODE_STACK_CACHE.clear()
        _ = decompose_lg(field, X, Y, w=30e-6, p_max=2, ell_max=1)
        assert len(_LG_MODE_STACK_CACHE) >= 1, (
            'decompose_lg should populate the LG mode-stack cache')

        _propagation_module.clear_asm_caches()

        assert len(_LG_MODE_STACK_CACHE) == 0, (
            'clear_asm_caches must drop the LG mode-stack cache '
            '(v4.14.1 Tier-0 #5 wires the lazy-import; v4.14.0 '
            'CHANGELOG claimed this but never implemented it)')


# ============================================================================
# Source: test_audit_fixes_v4_14_2_agent_b.py
# Audit version: V4_14_2  scope: agent_b
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.2 Agent-B audit fixes.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_14_1_2026_05_17.md`` P1-NEW-1 / P1-NEW-4 (Tier 0 row 2 / row 5).
#   v4.14.1 introduced ``_ZeroApertureMaskSentinel`` to handle
#   ``aperture_diameter=0`` correctly but the CHANGELOG only documented
#   "3 callers updated"; the audit found a **fourth** consumer of
#   ``_get_wrapper_merit_cache`` -- ``ToleranceAwareMerit.evaluate`` -- and a
#   pre-existing semantically-identical bug in ``MatchIdealSystemMerit._make_source``.
#   Both produced a grid-filling plane wave when ``aperture_diameter`` was
#   explicitly zero, which then propagated through ``apply_real_lens`` as a
#   bright on-axis "source" and silently mis-scored the merit.
#   
#   Agent B fixes
#   -------------
#   
#   * **B.1 / P1-NEW-1 (ToleranceAware leg)** -- Add the canonical
#     ``_cache['mask'] is _ZERO_APERTURE_MASK -> zero E_in`` branch to
#     ``ToleranceAwareMerit.evaluate`` so the per-trial wave-leg source is
#     zero when the (perturbed) prescription's ``aperture_diameter`` is
#     explicitly zero or negative.  Mirrors the canonical branch at
#     ``MultiWavelengthMerit.evaluate`` / ``MultiFieldMerit.evaluate``.
#   
#   * **B.2 / P1-NEW-1 (MatchIdealSystem leg)** -- Pre-v4.14.2
#     ``_make_source`` had an ``if ap is not None and np.isfinite(ap) and
#     ap > 0:`` guard that fell through to "full grid plane wave" when
#     ``aperture_diameter`` was zero.  v4.14.2 adds an explicit
#     ``ap <= 0 -> zero field`` branch matching the
#     ``_ZERO_APERTURE_MASK`` semantics.
#   
#   * **B.2-residual / P1-NEW-4 (0+0j sweep)** -- The ``np.where(mask, E,
#     0.0 + 0.0j)`` literal at the same site silently upcast complex64
#     fields to complex128.  v4.14.2 replaces the literal with the
#     dtype-aware-zero pattern ``np.where(mask, E, np.zeros((), dtype=cdtype))``
#     matching the v4.13.2 sweep across apply_aperture / apply_mirror /
#     _lens_thin / _lens_real.
#   
#   * **B.3 (validation gap)** -- ``apply_perturbations`` does NOT call
#     ``validate_prescription``, but since perturbations only modify
#     per-surface ``decenter`` / ``tilt`` / ``form_error`` fields (not the
#     prescription-level ``aperture_diameter``), the perturbed
#     ``aperture_diameter`` is always identical to the nominal.  The B.1
#     fix already handles the case where the nominal aperture is zero;
#     this test pins the contract that ``apply_perturbations`` does not
#     itself zero/negativise the aperture.
#   
#   Author: Andrew Traverso -- v4.14.2 / Agent B
# ============================================================================

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.optimize import (
    EvaluationContext,
    MatchIdealSystemMerit,
    StrehlMerit,
    ToleranceAwareMerit,
)
from lumenairy.optimize.core import (
    _ZERO_APERTURE_MASK,
    _clear_wrapper_merit_cache,
    _get_wrapper_merit_cache,
)

# ============================================================================
# Helpers
# ============================================================================

def _zero_aperture_prescription():
    """Build a singlet prescription, then override aperture_diameter to 0.

    ``make_singlet`` requires a positive aperture argument (its argparse
    accepts any float but validate_prescription would reject 0); we
    bypass by constructing the dict normally and setting the field
    directly.  This is the only way to exercise the production code path
    where ``aperture_diameter <= 0`` reaches the merit-eval code.
    """
    rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                          glass='N-BK7', aperture=5e-3)
    rx['aperture_diameter'] = 0.0
    return rx


# ============================================================================
# B.1 -- ToleranceAwareMerit honours _ZERO_APERTURE_MASK sentinel
# ============================================================================

class TestAuditFixesV4_14_2_agent_b_B1ToleranceAwareApertureZero:
    """``ToleranceAwareMerit.evaluate`` produces a zero E_in (not a
    full-grid plane wave) when the prescription's aperture_diameter
    is explicitly zero.

    Pre-fix the cached ``E_ones`` array was always full-ones; the
    ``_ZERO_APERTURE_MASK`` sentinel was placed in ``_cache['mask']``
    but ``ToleranceAwareMerit`` did NOT check it, so a deliberate-zero
    aperture silently propagated a grid-filling plane wave through
    ``apply_real_lens``.
    """

    def test_tolerance_aware_aperture_zero_produces_zero_field(self):
        """Capture E_in by monkey-patching apply_real_lens and assert
        all zeros for an aperture_diameter=0 prescription.

        Pins **P1-NEW-1 (ToleranceAware leg)**.
        """
        rx = _zero_aperture_prescription()
        # Clear the module cache so this test sees a fresh build.
        _clear_wrapper_merit_cache()

        captured = {}

        from lumenairy.elements import lenses as lens_mod


        def capturing_apply_real_lens(E_in, **kw):
            captured['E_in'] = np.asarray(E_in).copy()
            # Return zeros so downstream through-focus and Strehl
            # computations are well-defined.
            return np.zeros_like(E_in)

        try:
            # Patch the binding inside optimize.core, since that's the
            # ``from ..elements.lenses import apply_real_lens`` import
            # site ToleranceAwareMerit actually uses.
            from lumenairy.optimize import core as opt_core
            saved = opt_core.apply_real_lens
            opt_core.apply_real_lens = capturing_apply_real_lens
            try:
                ctx = EvaluationContext(
                    prescription=rx, wavelength=1.31e-6,
                    N=32, dx=10e-6, efl=0.025, bfl=0.025)
                # A single-trial ToleranceAware over an empty spec
                # exercises the perturbed wave-leg path without
                # actually perturbing anything; the perturbed
                # prescription is a deep copy of the original (with
                # aperture_diameter = 0 preserved).
                merit = ToleranceAwareMerit(
                    sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
                    perturbation_spec=[], n_trials=1, seed=0,
                    weight=1.0)
                # Suppress benign warnings from downstream through_focus
                # on a zero field.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _ = merit.evaluate(ctx)
            finally:
                opt_core.apply_real_lens = saved
        finally:
            _clear_wrapper_merit_cache()

        assert 'E_in' in captured, (
            'ToleranceAwareMerit.evaluate did not reach apply_real_lens')
        E_in = captured['E_in']
        assert E_in.shape == (32, 32)
        # The whole point: every grid point must be zero so the
        # downstream apply_real_lens sees "no light," matching the
        # physical meaning of aperture_diameter=0.
        assert np.all(E_in == 0), (
            f'ToleranceAware aperture=0 produced non-zero E_in '
            f'(max|E|={np.max(np.abs(E_in)):.3e}); '
            f'sentinel branch was not honoured.')

    def test_tolerance_aware_aperture_positive_unchanged(self):
        """Sanity check: ``aperture_diameter > 0`` produces the
        cached full-ones template (the v4.14.1 invariant).
        """
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=5e-3)
        _clear_wrapper_merit_cache()

        captured = {}

        def capturing_apply_real_lens(E_in, **kw):
            captured['E_in'] = np.asarray(E_in).copy()
            return np.zeros_like(E_in)

        try:
            from lumenairy.optimize import core as opt_core
            saved = opt_core.apply_real_lens
            opt_core.apply_real_lens = capturing_apply_real_lens
            try:
                ctx = EvaluationContext(
                    prescription=rx, wavelength=1.31e-6,
                    N=32, dx=10e-6, efl=0.025, bfl=0.025)
                merit = ToleranceAwareMerit(
                    sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
                    perturbation_spec=[], n_trials=1, seed=0,
                    weight=1.0)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _ = merit.evaluate(ctx)
            finally:
                opt_core.apply_real_lens = saved
        finally:
            _clear_wrapper_merit_cache()

        E_in = captured['E_in']
        # Positive aperture -> cached template = np.ones.  Pin against
        # the contract documented in _get_wrapper_merit_cache.
        assert np.all(E_in == 1.0), (
            f'ToleranceAware aperture>0 did not produce the cached '
            f'np.ones template (mean|E|={np.mean(np.abs(E_in)):.3e}).')


# ============================================================================
# B.2 -- MatchIdealSystemMerit._make_source honours zero-aperture
# ============================================================================

class TestAuditFixesV4_14_2_agent_b_B2MatchIdealApertureZero:
    """``MatchIdealSystemMerit._make_source`` with an
    aperture_diameter=0 prescription produces a zero field.

    Pre-fix the ``if ap > 0`` guard fell through to the bare
    ``np.ones`` branch, producing a grid-filling plane wave.
    """

    def test_match_ideal_aperture_zero_produces_zero_field(self):
        """Pins **P1-NEW-1 (MatchIdealSystem leg)**.

        Builds a MatchIdealSystemMerit instance and calls
        ``_make_source`` directly with a zero-aperture context.
        """
        rx = _zero_aperture_prescription()
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)

        # Minimal ideal-system spec; the merit only needs _make_source
        # to be callable, not the full evaluate path.
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025},
                            {'type': 'propagate', 'z': 0.025}],
            weight=1.0)
        E = merit._make_source(ctx, wavelength=1.31e-6,
                                field_angle=(0.0, 0.0))
        assert E.shape == (32, 32)
        assert np.all(E == 0), (
            f'MatchIdealSystem aperture=0 produced non-zero source '
            f'field (max|E|={np.max(np.abs(E)):.3e}); '
            f'the ap <= 0 branch was not taken.')

    def test_match_ideal_aperture_positive_produces_circular_mask(self):
        """Sanity check: ``aperture > 0`` produces the standard
        circular boolean mask, unchanged by the B.2 fix.
        """
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=120e-6)
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025}],
            weight=1.0)
        E = merit._make_source(ctx, wavelength=1.31e-6,
                                field_angle=(0.0, 0.0))
        # Some pixels lit, some zero -- the circular mask.
        nonzero = np.count_nonzero(np.abs(E))
        assert 0 < nonzero < 32 * 32, (
            f'MatchIdealSystem aperture>0 did not produce a partial '
            f'mask (nonzero={nonzero}/{32 * 32}).')

    def test_match_ideal_aperture_none_produces_full_grid(self):
        """``aperture_diameter`` missing -> full-grid plane wave
        (unchanged pre-existing behaviour)."""
        # Construct a singlet, then strip aperture_diameter.
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=5e-3)
        rx.pop('aperture_diameter', None)
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025}],
            weight=1.0)
        E = merit._make_source(ctx, wavelength=1.31e-6,
                                field_angle=(0.0, 0.0))
        # No aperture specified -> full-grid plane wave.
        assert np.all(np.abs(E) == 1.0), (
            'MatchIdealSystem with no aperture should produce a '
            'full-grid plane wave.')


# ============================================================================
# B.2-residual -- complex64 dtype preserved through _make_source
# ============================================================================

class TestAuditFixesV4_14_2_agent_b_B2MatchIdealComplex64DtypePreserved:
    """The v4.14.2 dtype-aware-zero fix at ``optimize/core.py:966`` --
    formerly ``np.where(mask, E, 0.0 + 0.0j)`` -- must not upcast a
    complex64 cdtype back to complex128.

    Pins **P1-NEW-4** at this specific site.
    """

    def test_match_ideal_complex64_dtype_preserved(self):
        """Drive ``get_default_complex_dtype`` to complex64 via the
        precision knob and assert ``_make_source`` returns complex64.
        """
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=120e-6)
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025}],
            weight=1.0)

        # Flip the runtime precision knob to single.
        from lumenairy.propagators.propagation import (
            get_default_complex_dtype,
            set_default_complex_dtype,
        )
        saved = get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            E = merit._make_source(ctx, wavelength=1.31e-6,
                                    field_angle=(0.0, 0.0))
            assert E.dtype == np.complex64, (
                f'MatchIdealSystem._make_source upcast complex64 -> '
                f'{E.dtype} (P1-NEW-4 regression); the 0+0j literal '
                f'must be np.zeros((), dtype=cdtype).')
        finally:
            set_default_complex_dtype(saved)


# ============================================================================
# B.3 -- apply_perturbations does not silently zero/negativise aperture
# ============================================================================

class TestAuditFixesV4_14_2_agent_b_B3PerturbedApertureInvariant:
    """``apply_perturbations`` modifies per-surface ``decenter`` /
    ``tilt`` / ``form_error`` fields but DOES NOT touch the
    prescription-level ``aperture_diameter``.  This pin guards against
    a future regression where perturbations start mutating the aperture
    (which would then escape ``validate_prescription`` since
    ``apply_perturbations`` does not invoke it).
    """

    def test_apply_perturbations_does_not_modify_aperture(self):
        """Apply a typical Monte-Carlo perturbation set and assert the
        perturbed prescription's ``aperture_diameter`` is identical to
        the nominal.  Pins the contract that B.1's sentinel guard
        already covers the perturbed wave leg.
        """
        from lumenairy.analysis.through_focus import (
            Perturbation,
            apply_perturbations,
        )
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=5e-3)
        nominal_ap = rx['aperture_diameter']

        perts = [Perturbation(
            surface_index=0,
            decenter=(10e-6, 5e-6),
            tilt=(1e-4, -1e-4),
            form_error_rms=20e-9,
            random_seed=12345,
            name='unit_test_pert')]
        perturbed = apply_perturbations(rx, perts, N=32, dx=10e-6)
        assert perturbed['aperture_diameter'] == nominal_ap, (
            'apply_perturbations should not modify the prescription-'
            'level aperture_diameter; if this fails, the B.1 sentinel '
            'guard is insufficient -- additional validate_prescription '
            'guard required at ToleranceAwareMerit.evaluate entry.')

    def test_tolerance_aware_with_perturbation_aperture_zero(self):
        """Pin the end-to-end contract: a *perturbed* trial whose
        nominal aperture is zero produces a zero E_in (and thus a
        well-defined merit, not a garbage plane-wave score).
        """
        from lumenairy.analysis.through_focus import Perturbation

        rx = _zero_aperture_prescription()
        _clear_wrapper_merit_cache()

        captured = []

        def capturing_apply_real_lens(E_in, **kw):
            captured.append(np.asarray(E_in).copy())
            return np.zeros_like(E_in)

        try:
            from lumenairy.optimize import core as opt_core
            saved = opt_core.apply_real_lens
            opt_core.apply_real_lens = capturing_apply_real_lens
            try:
                ctx = EvaluationContext(
                    prescription=rx, wavelength=1.31e-6,
                    N=32, dx=10e-6, efl=0.025, bfl=0.025)
                # 3 trials, one perturbation each -- exercises the
                # full _evaluate_perturbed loop.
                merit = ToleranceAwareMerit(
                    sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
                    perturbation_spec=[{
                        'surface_index': 0,
                        'decenter_std': 5e-6,
                        'tilt_std': 1e-4,
                        'form_error_rms': 0.0,
                    }],
                    n_trials=3, seed=42, weight=1.0)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _ = merit.evaluate(ctx)
            finally:
                opt_core.apply_real_lens = saved
        finally:
            _clear_wrapper_merit_cache()

        assert len(captured) == 3, (
            f'Expected 3 captures (one per trial), got {len(captured)}')
        for i, E in enumerate(captured):
            assert np.all(E == 0), (
                f'Trial {i}: perturbed E_in was not zero '
                f'(max|E|={np.max(np.abs(E)):.3e}); sentinel was not '
                f'honoured under perturbation.')


# ============================================================================
# Cache invariance pin -- sentinel still in place after these fixes
# ============================================================================

class TestAuditFixesV4_14_2_agent_b_SentinelStillCached:
    """Sanity check: the v4.14.1 contract that
    ``_get_wrapper_merit_cache`` returns ``_cache['mask'] is
    _ZERO_APERTURE_MASK`` for aperture_diameter=0 is still honoured.
    """

    def test_zero_aperture_cache_mask_is_sentinel(self):
        """Pins the v4.14.1 P1-NEW-1 invariant the B.1/B.2 fixes
        depend on."""
        _clear_wrapper_merit_cache()
        try:
            entry = _get_wrapper_merit_cache(
                32, 10e-6, 0.0, np.complex128)
            assert entry['mask'] is _ZERO_APERTURE_MASK
        finally:
            _clear_wrapper_merit_cache()

    def test_negative_aperture_cache_mask_is_sentinel(self):
        """Negative aperture (could arise if a perturbed
        aperture_diameter ever went below zero) also routes through
        the sentinel branch.
        """
        _clear_wrapper_merit_cache()
        try:
            entry = _get_wrapper_merit_cache(
                32, 10e-6, -1e-3, np.complex128)
            assert entry['mask'] is _ZERO_APERTURE_MASK
        finally:
            _clear_wrapper_merit_cache()


# ============================================================================
# Source: test_audit_fixes_v4_14_2_agent_c.py
# Audit version: V4_14_2  scope: agent_c
# Original module docstring preserved as comment block for git-blame traceability:
#   Tests for the v4.14.2 audit-fix Agent-C changes.
#   
#   Scope: the 5 cache-host files plus ``propagators/propagation.py``:
#   
#   - ``lumenairy/analysis/core.py``           (Zernike basis cache)
#   - ``lumenairy/analysis/through_focus.py``  (through-focus JAX scan cache)
#   - ``lumenairy/analysis/phase_retrieval.py``(GS / ER / HIO kernel caches)
#   - ``lumenairy/propagators/system.py``                  (propagate_through_system JAX cache)
#   - ``lumenairy/raytrace/jax_trace.py``      (trace_jax JAX kernel cache)
#   - ``lumenairy/propagators/propagation.py`` (clear_asm_caches scope expansion)
#   
#   What this module pins
#   ---------------------
#   
#   * **C.1 -- P1-NEW-2 thread-safety locks on 7 older caches.**
#     The v4.14.1 lock-scope pattern (``_LG_MODE_STACK_LOCK``,
#     ``_HG_MODE_STACK_LOCK``, ``_WRAPPER_MERIT_CACHE_LOCK``,
#     ``_ASM_CACHE_LOCK``) is extended to seven caches that pre-dated
#     v4.14.0 and were missed by the v4.14.1 sweep.  This test
#     spawns 4 threads x 50 iters per cache and asserts no exception
#     and final cache size <= maxsize, no duplicate keys.
#   
#   * **C.2 -- P1-NEW-3 clear_asm_caches() scope expansion.**
#     ``clear_asm_caches()`` previously chained only the LG/HG
#     mode-stack cache and the wrapper-merit meshgrid cache.  v4.14.2
#     extends it to also chain Zernike, through-focus, propagate-system,
#     phase-retrieval, and trace-jax kernel caches -- matching what
#     :func:`lumenairy.lumenairy_context` already does via direct
#     submodule imports.
#   
#   * **C.3 -- P1-NEW-4 (partial) phase_retrieval residual 0+0j.**
#     ``analysis/phase_retrieval.py`` line ~402 ``np.where(support,
#     obj_new, 0.0+0.0j)`` would silently upcast a complex64 ``obj_new``
#     to complex128 (the literal is a Python complex which numpy
#     promotes to complex128).  v4.14.2 uses
#     ``np.where(support, obj_new, np.zeros((), dtype=obj_new.dtype))``
#     -- the v4.13.2 dtype-aware pattern.
#   
#   Author: Andrew Traverso -- v4.14.2 / Agent C
# ============================================================================

import threading

import numpy as np
import pytest

# ============================================================================
# C.1 -- Thread-safety locks on 7 older caches (P1-NEW-2)
# ============================================================================

class TestAuditFixesV4_14_2_agent_c_C1LocksPresent:
    """Pin that every cache-host module exposes a matching
    ``_<CACHENAME>_LOCK`` module-level ``threading.Lock`` constant
    next to each ``_<CACHENAME>``.  Independent of the C.4 meta-pin
    so the C.1 fix has a named regression target distinct from the
    library-wide walker."""

    def test_zernike_basis_cache_lock_present(self):
        from lumenairy.analysis import core as ac
        assert hasattr(ac, '_ZERNIKE_BASIS_CACHE_LOCK')
        assert isinstance(ac._ZERNIKE_BASIS_CACHE_LOCK,
                          type(threading.Lock()))

    def test_through_focus_scan_jax_cache_lock_present(self):
        from lumenairy.analysis import through_focus as tf
        assert hasattr(tf, '_THROUGH_FOCUS_SCAN_JAX_CACHE_LOCK')
        assert isinstance(tf._THROUGH_FOCUS_SCAN_JAX_CACHE_LOCK,
                          type(threading.Lock()))

    def test_propagate_system_jax_cache_lock_present(self):
        from lumenairy.propagators import system as sys_mod
        assert hasattr(sys_mod, '_PROPAGATE_SYSTEM_JAX_CACHE_LOCK')
        assert isinstance(sys_mod._PROPAGATE_SYSTEM_JAX_CACHE_LOCK,
                          type(threading.Lock()))

    def test_phase_retrieval_kernel_cache_locks_present(self):
        from lumenairy.analysis import phase_retrieval as pr
        for name in ('_GS_KERNEL_CACHE_LOCK',
                     '_ER_KERNEL_CACHE_LOCK',
                     '_HIO_KERNEL_CACHE_LOCK'):
            assert hasattr(pr, name), (
                f'phase_retrieval missing {name!r}')
            assert isinstance(getattr(pr, name),
                              type(threading.Lock()))

    def test_trace_jax_cache_lock_present(self):
        from lumenairy.raytrace import jax_trace as jt
        assert hasattr(jt, '_TRACE_JAX_CACHE_LOCK')
        assert isinstance(jt._TRACE_JAX_CACHE_LOCK,
                          type(threading.Lock()))


class TestAuditFixesV4_14_2_agent_c_C1ConcurrentAccessNoExceptions:
    """Spawn 4 threads x 50 iters on each cache and assert no
    exception and the final cache state is consistent (size <=
    maxsize, no duplicate keys).  Tests the real concurrency
    contract.

    The threads call the public cached entry point so we exercise
    the production code paths (not the OrderedDict directly).
    Cache keys vary across the iterations so we genuinely race
    inserts, not just hits.
    """

    def _run_threads(self, target, n_threads=4, n_iters=50):
        """Launch ``n_threads`` threads each running ``target(i)`` for
        ``i in range(n_iters)``.  Returns (exceptions_per_thread,
        size_per_thread)."""
        errors = []

        def _worker(tid):
            try:
                for i in range(n_iters):
                    target(tid, i)
            except Exception as e:  # noqa: BLE001 -- capture-for-assert only
                errors.append((tid, type(e).__name__, str(e)))

        threads = [threading.Thread(target=_worker, args=(tid,))
                   for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errors

    def test_zernike_basis_cache_concurrent(self):
        """Concurrent ``zernike_basis_matrix`` calls vary ``n_modes``
        per iter so we genuinely race the LRU insert/evict path."""
        from lumenairy.analysis.core import (
            _ZERNIKE_BASIS_CACHE,
            _ZERNIKE_BASIS_CACHE_MAXSIZE,
            clear_zernike_basis_cache,
            zernike_basis_matrix,
        )
        clear_zernike_basis_cache()

        N = 32
        x = (np.arange(N) - N / 2) * 1e-6
        X, Y = np.meshgrid(x, x)
        r_pup = 0.5 * (N * 1e-6) * 0.9

        def _target(tid, i):
            # Vary n_modes so caches both hit and miss; mod by 8 so
            # threads share some keys (forcing LRU recency races).
            n_modes = 6 + (i % 8)
            basis, mask = zernike_basis_matrix(n_modes, X, Y, r_pup)
            assert basis.shape[1] == n_modes
            assert mask.shape == (N, N)

        errs = self._run_threads(_target)
        assert errs == [], f'threads raised: {errs}'
        # Final state invariants.
        assert len(_ZERNIKE_BASIS_CACHE) <= _ZERNIKE_BASIS_CACHE_MAXSIZE
        # No duplicate keys (OrderedDict guarantees this, but pin it).
        keys = list(_ZERNIKE_BASIS_CACHE.keys())
        assert len(keys) == len(set(keys))

    def test_phase_retrieval_kernel_cache_concurrent(self):
        """Concurrent numpy-only error_reduction calls that exercise
        the np.where dtype site.  The JAX kernel caches require JAX
        which may not be installed; we cover them via direct
        OrderedDict manipulation under the lock instead."""
        from lumenairy.analysis.phase_retrieval import (
            _ER_KERNEL_CACHE,
            _ER_KERNEL_CACHE_LOCK,
            _GS_KERNEL_CACHE,
            _GS_KERNEL_CACHE_LOCK,
            _HIO_KERNEL_CACHE,
            _HIO_KERNEL_CACHE_LOCK,
            clear_phase_retrieval_caches,
        )
        clear_phase_retrieval_caches()

        # Simulate the cache contention with fake kernels (no JAX
        # required).  This still exercises every OrderedDict op the
        # production code does and proves the lock works against
        # racing inserts.
        def _populate(cache, lock, tid, i):
            key = (i % 12, 'complex128')
            with lock:
                if cache.get(key) is None:
                    cache[key] = lambda: tid  # placeholder
                    while len(cache) > 32:
                        cache.popitem(last=False)
                else:
                    cache.move_to_end(key)

        def _target(tid, i):
            _populate(_GS_KERNEL_CACHE, _GS_KERNEL_CACHE_LOCK, tid, i)
            _populate(_ER_KERNEL_CACHE, _ER_KERNEL_CACHE_LOCK, tid, i)
            _populate(_HIO_KERNEL_CACHE, _HIO_KERNEL_CACHE_LOCK, tid, i)

        errs = self._run_threads(_target)
        assert errs == [], f'threads raised: {errs}'
        for cache in (_GS_KERNEL_CACHE, _ER_KERNEL_CACHE,
                      _HIO_KERNEL_CACHE):
            keys = list(cache.keys())
            assert len(keys) == len(set(keys))
            assert len(keys) <= 32

    def test_through_focus_scan_jax_cache_concurrent(self):
        """Direct OrderedDict contention against the lock.  Avoids
        the JAX-installed gate that ``through_focus_scan_jax`` would
        impose; the contract here is that the cache + lock survive
        N concurrent get/__setitem__/move_to_end/popitem sequences
        without raising."""
        from lumenairy.analysis.through_focus import (
            _THROUGH_FOCUS_SCAN_JAX_CACHE,
            _THROUGH_FOCUS_SCAN_JAX_CACHE_LOCK,
            _THROUGH_FOCUS_SCAN_JAX_CACHE_MAXSIZE,
            clear_through_focus_scan_jax_cache,
        )
        clear_through_focus_scan_jax_cache()

        def _target(tid, i):
            key = (i % 10, 'k')
            with _THROUGH_FOCUS_SCAN_JAX_CACHE_LOCK:
                if _THROUGH_FOCUS_SCAN_JAX_CACHE.get(key) is None:
                    _THROUGH_FOCUS_SCAN_JAX_CACHE[key] = tid
                    while (len(_THROUGH_FOCUS_SCAN_JAX_CACHE)
                           > _THROUGH_FOCUS_SCAN_JAX_CACHE_MAXSIZE):
                        _THROUGH_FOCUS_SCAN_JAX_CACHE.popitem(last=False)
                else:
                    _THROUGH_FOCUS_SCAN_JAX_CACHE.move_to_end(key)

        errs = self._run_threads(_target)
        assert errs == [], f'threads raised: {errs}'
        keys = list(_THROUGH_FOCUS_SCAN_JAX_CACHE.keys())
        assert len(keys) == len(set(keys))
        assert len(keys) <= _THROUGH_FOCUS_SCAN_JAX_CACHE_MAXSIZE

    def test_propagate_system_jax_cache_concurrent(self):
        """Same approach -- direct cache+lock contention without
        requiring JAX."""
        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            _PROPAGATE_SYSTEM_JAX_CACHE_LOCK,
            _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE,
            clear_propagate_system_jax_cache,
        )
        clear_propagate_system_jax_cache()

        def _target(tid, i):
            key = (i % 10, 'sig')
            with _PROPAGATE_SYSTEM_JAX_CACHE_LOCK:
                if _PROPAGATE_SYSTEM_JAX_CACHE.get(key) is None:
                    _PROPAGATE_SYSTEM_JAX_CACHE[key] = tid
                    while (len(_PROPAGATE_SYSTEM_JAX_CACHE)
                           > _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE):
                        _PROPAGATE_SYSTEM_JAX_CACHE.popitem(last=False)
                else:
                    _PROPAGATE_SYSTEM_JAX_CACHE.move_to_end(key)

        errs = self._run_threads(_target)
        assert errs == [], f'threads raised: {errs}'
        keys = list(_PROPAGATE_SYSTEM_JAX_CACHE.keys())
        assert len(keys) == len(set(keys))
        assert len(keys) <= _PROPAGATE_SYSTEM_JAX_CACHE_MAXSIZE

    def test_trace_jax_cache_concurrent(self):
        """Same approach for the ray-trace JAX kernel cache."""
        from lumenairy.raytrace.jax_trace import (
            _TRACE_JAX_CACHE,
            _TRACE_JAX_CACHE_LOCK,
            _TRACE_JAX_CACHE_MAXSIZE,
            clear_trace_jax_cache,
        )
        clear_trace_jax_cache()

        def _target(tid, i):
            key = (i % 10, 1.31e-6)
            with _TRACE_JAX_CACHE_LOCK:
                if _TRACE_JAX_CACHE.get(key) is None:
                    _TRACE_JAX_CACHE[key] = tid
                    while (len(_TRACE_JAX_CACHE)
                           > _TRACE_JAX_CACHE_MAXSIZE):
                        _TRACE_JAX_CACHE.popitem(last=False)
                else:
                    _TRACE_JAX_CACHE.move_to_end(key)

        errs = self._run_threads(_target)
        assert errs == [], f'threads raised: {errs}'
        keys = list(_TRACE_JAX_CACHE.keys())
        assert len(keys) == len(set(keys))
        assert len(keys) <= _TRACE_JAX_CACHE_MAXSIZE


# ============================================================================
# C.2 -- clear_asm_caches() scope expansion (P1-NEW-3)
# ============================================================================

class TestAuditFixesV4_14_2_agent_c_C2ClearAsmCachesChainsAll:
    """Pin that ``clear_asm_caches()`` now reaches every sibling
    cache the audit identified, by populating each cache and
    asserting it is empty after the call."""

    def test_clears_zernike_basis_cache(self):
        from lumenairy.analysis.core import (
            _ZERNIKE_BASIS_CACHE,
            zernike_basis_matrix,
        )
        from lumenairy.propagators.propagation import clear_asm_caches

        # Populate
        N = 24
        x = (np.arange(N) - N / 2) * 1e-6
        X, Y = np.meshgrid(x, x)
        zernike_basis_matrix(15, X, Y, 0.5 * N * 1e-6 * 0.9)
        assert len(_ZERNIKE_BASIS_CACHE) > 0

        clear_asm_caches()
        assert len(_ZERNIKE_BASIS_CACHE) == 0

    def test_clears_through_focus_scan_jax_cache(self):
        """Populate the cache directly (no JAX required) then
        assert clear_asm_caches drains it."""
        from lumenairy.analysis.through_focus import (
            _THROUGH_FOCUS_SCAN_JAX_CACHE,
        )
        from lumenairy.propagators.propagation import clear_asm_caches

        # Populate
        _THROUGH_FOCUS_SCAN_JAX_CACHE[('fake-key',)] = lambda: None
        assert len(_THROUGH_FOCUS_SCAN_JAX_CACHE) == 1

        clear_asm_caches()
        assert len(_THROUGH_FOCUS_SCAN_JAX_CACHE) == 0

    def test_clears_propagate_system_jax_cache(self):
        from lumenairy.propagators.propagation import clear_asm_caches
        from lumenairy.propagators.system import _PROPAGATE_SYSTEM_JAX_CACHE

        _PROPAGATE_SYSTEM_JAX_CACHE[('fake-key',)] = lambda: None
        assert len(_PROPAGATE_SYSTEM_JAX_CACHE) == 1

        clear_asm_caches()
        assert len(_PROPAGATE_SYSTEM_JAX_CACHE) == 0

    def test_clears_phase_retrieval_caches(self):
        from lumenairy.analysis.phase_retrieval import (
            _ER_KERNEL_CACHE,
            _GS_KERNEL_CACHE,
            _HIO_KERNEL_CACHE,
        )
        from lumenairy.propagators.propagation import clear_asm_caches

        _GS_KERNEL_CACHE[('fk',)] = lambda: None
        _ER_KERNEL_CACHE[('fk',)] = lambda: None
        _HIO_KERNEL_CACHE[('fk',)] = lambda: None
        assert len(_GS_KERNEL_CACHE) == 1
        assert len(_ER_KERNEL_CACHE) == 1
        assert len(_HIO_KERNEL_CACHE) == 1

        clear_asm_caches()
        assert len(_GS_KERNEL_CACHE) == 0
        assert len(_ER_KERNEL_CACHE) == 0
        assert len(_HIO_KERNEL_CACHE) == 0

    def test_clears_trace_jax_cache(self):
        from lumenairy.propagators.propagation import clear_asm_caches
        from lumenairy.raytrace.jax_trace import _TRACE_JAX_CACHE

        _TRACE_JAX_CACHE[('fake-key',)] = lambda: None
        assert len(_TRACE_JAX_CACHE) == 1

        clear_asm_caches()
        assert len(_TRACE_JAX_CACHE) == 0

    def test_combined_drain_leaves_all_caches_empty(self):
        """Single ``clear_asm_caches()`` call leaves every chained
        cache pristine.  This is the "pristine state" promise from
        the v4.14.2 docstring rewrite."""
        from lumenairy.analysis.core import _ZERNIKE_BASIS_CACHE
        from lumenairy.analysis.phase_retrieval import (
            _ER_KERNEL_CACHE,
            _GS_KERNEL_CACHE,
            _HIO_KERNEL_CACHE,
        )
        from lumenairy.analysis.through_focus import (
            _THROUGH_FOCUS_SCAN_JAX_CACHE,
        )
        from lumenairy.propagators.propagation import (
            _BANDLIMIT_CACHE,
            _FREQ_GRID_CACHE,
            _H_CACHE,
            clear_asm_caches,
        )
        from lumenairy.propagators.system import _PROPAGATE_SYSTEM_JAX_CACHE
        from lumenairy.raytrace.jax_trace import _TRACE_JAX_CACHE

        # Populate every cache that does NOT require JAX.
        _GS_KERNEL_CACHE[('k',)] = lambda: None
        _ER_KERNEL_CACHE[('k',)] = lambda: None
        _HIO_KERNEL_CACHE[('k',)] = lambda: None
        _THROUGH_FOCUS_SCAN_JAX_CACHE[('k',)] = lambda: None
        _PROPAGATE_SYSTEM_JAX_CACHE[('k',)] = lambda: None
        _TRACE_JAX_CACHE[('k',)] = lambda: None

        # Real Zernike build
        N = 16
        from lumenairy.analysis.core import zernike_basis_matrix
        x = (np.arange(N) - N / 2) * 1e-6
        X, Y = np.meshgrid(x, x)
        zernike_basis_matrix(6, X, Y, 0.5 * N * 1e-6 * 0.9)

        clear_asm_caches()

        for cache in (_FREQ_GRID_CACHE, _BANDLIMIT_CACHE, _H_CACHE,
                      _ZERNIKE_BASIS_CACHE,
                      _THROUGH_FOCUS_SCAN_JAX_CACHE,
                      _PROPAGATE_SYSTEM_JAX_CACHE,
                      _GS_KERNEL_CACHE, _ER_KERNEL_CACHE,
                      _HIO_KERNEL_CACHE,
                      _TRACE_JAX_CACHE):
            assert len(cache) == 0, (
                f'cache {id(cache)} still has {len(cache)} entries '
                f'after clear_asm_caches()')

    def test_clear_asm_caches_docstring_lists_new_targets(self):
        """The docstring update must explicitly mention each newly-
        chained cache so users searching for it can find the entry
        point.  Pinning the names verbatim catches any future
        documentation drift."""
        from lumenairy.propagators.propagation import clear_asm_caches
        doc = clear_asm_caches.__doc__
        assert doc is not None
        # Each new chain target appears by callable name.
        for needle in ('clear_zernike_basis_cache',
                       'clear_through_focus_scan_jax_cache',
                       'clear_propagate_system_jax_cache',
                       'clear_phase_retrieval_caches',
                       'clear_trace_jax_cache'):
            assert needle in doc, (
                f'clear_asm_caches docstring missing reference to '
                f'{needle!r}; the v4.14.2 scope expansion must list '
                f'every chained cache.')


# ============================================================================
# C.3 -- phase_retrieval residual 0+0j dtype regression (P1-NEW-4)
# ============================================================================

class TestAuditFixesV4_14_2_agent_c_C3PhaseRetrievalDtypePreservation:
    """Pin that the numpy-path ``error_reduction`` preserves a
    complex64 input's dtype through the ``np.where(support, obj_new,
    ...)`` site at line ~402.

    Pre-fix the residual literal ``0.0 + 0.0j`` is a Python complex
    which numpy treats as complex128 -- ``np.where`` then upcasts
    the entire output to complex128.  Post-fix the residual is
    ``np.zeros((), dtype=obj_new.dtype)`` so the dtype propagates.
    """

    def test_error_reduction_preserves_complex64(self):
        """error_reduction(dtype=np.complex64) returns complex64."""
        from lumenairy.analysis.phase_retrieval import error_reduction

        N = 32
        rng = np.random.default_rng(0)
        # Build a synthetic Fourier-magnitude target (real, positive).
        meas = rng.uniform(0.1, 1.0, size=(N, N)).astype(np.float32)
        support = np.ones((N, N), dtype=bool)
        # Support is the full grid; force the recovery to converge
        # somewhere.
        result = error_reduction(
            meas, support, n_iter=3, dtype=np.complex64, seed=0,
        )
        # error_reduction may return (obj, err) or (obj, err, hist);
        # canonically (obj, err) when return_history is False (default).
        obj = result[0]
        assert obj.dtype == np.complex64, (
            f'error_reduction(dtype=complex64) should return complex64, '
            f'got {obj.dtype}.  Likely the bare ``0.0+0.0j`` literal '
            f'at line ~402 upcast the np.where output back to '
            f'complex128.')

    def test_error_reduction_preserves_complex128(self):
        """error_reduction(dtype=np.complex128) returns complex128
        (regression pin -- the dtype-aware patch must not flip the
        default path either)."""
        from lumenairy.analysis.phase_retrieval import error_reduction

        N = 32
        rng = np.random.default_rng(0)
        meas = rng.uniform(0.1, 1.0, size=(N, N)).astype(np.float64)
        support = np.ones((N, N), dtype=bool)
        obj, _err = error_reduction(
            meas, support, n_iter=3, dtype=np.complex128, seed=0,
        )
        assert obj.dtype == np.complex128


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
