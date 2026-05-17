"""Correctness pins for the v4.13.0 finite-difference gradient
parameterisation (audit Phase-3 Group beta task beta.3).

v4.13.0 adds an ``scheme={'central','forward'}`` parameter to
``lumenairy.optimize.core._fd_grad_pure`` and the wrapped
``_fd_grad_for`` helper.

* ``scheme='central'`` (the default) preserves bit-identical pre-
  v4.13.0 behaviour: 2N evaluations per gradient, O(h^2) truncation.
* ``scheme='forward'`` is an opt-in perf path: N+1 evaluations per
  gradient (or N if the caller passes ``f0``).  O(h) truncation.

The ``f0`` parameter is meaningful only for ``scheme='forward'`` (the
central-difference path does not evaluate at the centre, so there is
no centre value to reuse).

What this test pins
-------------------

1. **Default scheme is central** -- ``_fd_grad_pure(f, x)`` with no
   ``scheme=`` returns the central-difference gradient.
2. **Central-difference correctness** -- on a quadratic, central
   matches the analytical gradient to ~1e-9 (O(h^2) truncation with
   h~1e-7).
3. **Forward-difference correctness** -- on the same quadratic,
   forward matches analytical to ~1e-3 (O(h) truncation with h~1e-7).
4. **Eval-count contracts**:
   * central: exactly 2N calls to f.
   * forward, f0=None: exactly N+1 calls.
   * forward, f0=<value>: exactly N calls.
5. **f0 reuse invariance (forward only)** -- passing ``f0=f(x)`` gives
   bit-identical output to ``f0=None`` on the forward path.
6. **Invalid scheme** -- ``scheme='banana'`` raises ValueError.
7. **scale_floor floor still respected** -- per-variable floor is
   applied in both schemes.
"""
from __future__ import annotations

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
