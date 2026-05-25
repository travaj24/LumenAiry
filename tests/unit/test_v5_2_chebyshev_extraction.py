"""v5.2 Shared Chebyshev helpers extraction -- pin tests.

ROADMAP v5.1 item (lines 221-225) called out an inverted-dependency
layering bug: ``propagators/asymptotic*.py`` and
``elements/lenses_maslov.py`` both imported the three private
Chebyshev Vandermonde helpers from ``elements/lenses.py``.  Those
helpers are pure math primitives -- they have no notion of optics --
so ``propagators/`` should not have to reach into ``elements/`` to
use them.

v5.2 extraction:

  Old:  lumenairy/elements/lenses.py:722
          def _chebyshev_vandermonde(u, max_k): ...
          def _chebyshev_derivative_vandermonde(u, max_k): ...
          def _chebyshev_second_derivative_vandermonde(u, max_k): ...

        lumenairy/propagators/asymptotic_jax_twin.py:65
          def _chebyshev_vandermonde_xp(u, max_k, xp): ...

  New:  lumenairy/_math/chebyshev.py
          chebyshev_vandermonde(u, max_k, xp=None)       # consolidated
          chebyshev_derivative_vandermonde(u, max_k)
          chebyshev_second_derivative_vandermonde(u, max_k)

        elements/lenses.py keeps underscore-prefixed back-compat
        aliases bound to the new public names so external callers
        (and the long-tail audit-fix tests) continue to work
        unchanged.

This file pins:

* The new module exists and exports the three public names.
* The Chebyshev recurrence is mathematically correct at well-known
  evaluation points (T_0..T_3 closed-form values).
* The derivative recurrence matches T_n'(u) = n * U_{n-1}(u) where
  U_n is the second-kind Chebyshev polynomial.
* The second-derivative recurrence matches the analytic form
  T''_n(u) = n * ((n+1) T_n(u) - U_n(u)) / (u^2 - 1) at interior
  points, where the closed form is non-singular.
* Back-compat: ``from lumenairy.elements.lenses import
  _chebyshev_vandermonde`` still works (the underscore-prefixed
  alias is preserved).
* The xp-dispatched path matches the NumPy default when xp=numpy.
* The xp-dispatched path is JAX-traceable when JAX is importable
  (dtype check + jax.jit roundtrip).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pin 1 -- module + public-name existence
# ---------------------------------------------------------------------------

def test_math_chebyshev_module_exists():
    mod = importlib.import_module('lumenairy._math.chebyshev')
    for name in (
        'chebyshev_vandermonde',
        'chebyshev_derivative_vandermonde',
        'chebyshev_second_derivative_vandermonde',
    ):
        assert hasattr(mod, name), (
            f"lumenairy._math.chebyshev missing public name {name!r}"
        )
    # v5.4 Phase 5 added ``chebyshev_fit_2d`` to the module (the
    # tested fit primitive promoted out of the UI dock).  This pin
    # accepts that addition; the original three Vandermonde helpers
    # remain mandatory above.
    assert sorted(mod.__all__) == [
        'chebyshev_derivative_vandermonde',
        'chebyshev_fit_2d',
        'chebyshev_second_derivative_vandermonde',
        'chebyshev_vandermonde',
    ]


# ---------------------------------------------------------------------------
# Pin 2 -- closed-form values of T_n(u) for n = 0..3
# ---------------------------------------------------------------------------
#
# T_0(u) = 1
# T_1(u) = u
# T_2(u) = 2 u^2 - 1
# T_3(u) = 4 u^3 - 3 u
#
# We evaluate at u = [-1, 0, 1] where the values are exact integers.

def test_chebyshev_values_at_endpoints_and_origin():
    from lumenairy._math.chebyshev import chebyshev_vandermonde
    u = np.array([-1.0, 0.0, 1.0])
    T = chebyshev_vandermonde(u, max_k=3)
    assert T.shape == (4, 3)
    # T_0 = 1
    np.testing.assert_allclose(T[0], [1.0, 1.0, 1.0])
    # T_1 = u
    np.testing.assert_allclose(T[1], [-1.0, 0.0, 1.0])
    # T_2 = 2u^2 - 1
    np.testing.assert_allclose(T[2], [1.0, -1.0, 1.0])
    # T_3 = 4u^3 - 3u
    np.testing.assert_allclose(T[3], [-1.0, 0.0, 1.0])


def test_chebyshev_values_random_points_match_polynomial():
    """T_n(u) at random points matches the explicit polynomial form."""
    from lumenairy._math.chebyshev import chebyshev_vandermonde
    rng = np.random.default_rng(0)
    u = rng.uniform(-1.0, 1.0, size=(7,))
    T = chebyshev_vandermonde(u, max_k=5)
    np.testing.assert_allclose(T[0], np.ones_like(u))
    np.testing.assert_allclose(T[1], u)
    np.testing.assert_allclose(T[2], 2.0 * u**2 - 1.0)
    np.testing.assert_allclose(T[3], 4.0 * u**3 - 3.0 * u)
    np.testing.assert_allclose(T[4], 8.0 * u**4 - 8.0 * u**2 + 1.0)
    np.testing.assert_allclose(T[5], 16.0 * u**5 - 20.0 * u**3 + 5.0 * u)


# ---------------------------------------------------------------------------
# Pin 3 -- derivative recurrence: T'_n(u) = n * U_{n-1}(u)
# ---------------------------------------------------------------------------
#
# U_n is the second-kind Chebyshev polynomial.  We build U[0..max_k-1]
# from its own 3-term recurrence and compare row-by-row.

def _second_kind(u: np.ndarray, max_k: int) -> np.ndarray:
    """Reference U_n(u) for n = 0..max_k via the standard recurrence."""
    U = np.empty((max_k + 1,) + u.shape, dtype=np.float64)
    U[0] = 1.0
    if max_k >= 1:
        U[1] = 2.0 * u
    for n in range(2, max_k + 1):
        U[n] = 2.0 * u * U[n - 1] - U[n - 2]
    return U


def test_chebyshev_derivative_matches_n_times_U_nminus1():
    from lumenairy._math.chebyshev import chebyshev_derivative_vandermonde
    rng = np.random.default_rng(1)
    u = rng.uniform(-0.95, 0.95, size=(11,))   # avoid endpoints
    max_k = 6
    Tp = chebyshev_derivative_vandermonde(u, max_k)
    assert Tp.shape == (max_k + 1, u.shape[0])
    # T'_0 = 0
    np.testing.assert_allclose(Tp[0], np.zeros_like(u))
    # T'_n(u) = n * U_{n-1}(u)
    U = _second_kind(u, max_k - 1)
    for n in range(1, max_k + 1):
        np.testing.assert_allclose(
            Tp[n], float(n) * U[n - 1],
            err_msg=f"T'_{n} mismatch vs n * U_{n - 1}",
        )


# ---------------------------------------------------------------------------
# Pin 4 -- second-derivative recurrence vs analytic closed form
# ---------------------------------------------------------------------------
#
# T''_n(u) = n * ((n+1) T_n(u) - U_n(u)) / (u^2 - 1)
#
# Singular at u = +/- 1; we evaluate at interior points only.

def test_chebyshev_second_derivative_matches_closed_form():
    from lumenairy._math.chebyshev import (
        chebyshev_second_derivative_vandermonde,
        chebyshev_vandermonde,
    )
    rng = np.random.default_rng(2)
    u = rng.uniform(-0.8, 0.8, size=(9,))
    max_k = 5
    Tpp = chebyshev_second_derivative_vandermonde(u, max_k)
    T = chebyshev_vandermonde(u, max_k)
    U = _second_kind(u, max_k)
    # T''_0 = 0, T''_1 = 0
    np.testing.assert_allclose(Tpp[0], np.zeros_like(u), atol=1e-14)
    np.testing.assert_allclose(Tpp[1], np.zeros_like(u), atol=1e-14)
    # T''_2 = 4 (constant)
    np.testing.assert_allclose(Tpp[2], 4.0 * np.ones_like(u), atol=1e-14)
    # General n: analytic form
    denom = u**2 - 1.0
    for n in range(2, max_k + 1):
        expected = float(n) * ((n + 1) * T[n] - U[n]) / denom
        np.testing.assert_allclose(
            Tpp[n], expected, rtol=1e-12, atol=1e-12,
            err_msg=f"T''_{n} mismatch vs analytic form",
        )


# ---------------------------------------------------------------------------
# Pin 5 -- back-compat aliases on elements.lenses
# ---------------------------------------------------------------------------
#
# The pre-v5.2 import path is the single most common form across the
# codebase (and the long-tail audit-fix tests).  We do NOT update those
# call sites in this extraction; instead we preserve underscore-prefixed
# aliases on ``lumenairy.elements.lenses``.  This pin enforces that the
# aliases still resolve and forward to the new canonical implementation.

def test_back_compat_alias_on_elements_lenses():
    from lumenairy._math.chebyshev import (
        chebyshev_derivative_vandermonde,
        chebyshev_second_derivative_vandermonde,
        chebyshev_vandermonde,
    )
    from lumenairy.elements.lenses import (
        _chebyshev_derivative_vandermonde,
        _chebyshev_second_derivative_vandermonde,
        _chebyshev_vandermonde,
    )
    # Identity: the aliases ARE the canonical functions (not wrappers).
    assert _chebyshev_vandermonde is chebyshev_vandermonde
    assert _chebyshev_derivative_vandermonde is chebyshev_derivative_vandermonde
    assert (_chebyshev_second_derivative_vandermonde
            is chebyshev_second_derivative_vandermonde)
    # The canonical module is _math.chebyshev, not elements.lenses.
    assert _chebyshev_vandermonde.__module__ == 'lumenairy._math.chebyshev'


def test_back_compat_alias_returns_same_value():
    """Sanity: the alias path produces bit-identical output."""
    from lumenairy._math.chebyshev import chebyshev_vandermonde as canonical
    from lumenairy.elements.lenses import _chebyshev_vandermonde as alias
    u = np.linspace(-1.0, 1.0, 17)
    np.testing.assert_array_equal(alias(u, 6), canonical(u, 6))


# ---------------------------------------------------------------------------
# Pin 6 -- xp-dispatched path: NumPy backend identical to default
# ---------------------------------------------------------------------------

def test_xp_default_equals_xp_numpy():
    from lumenairy._math.chebyshev import chebyshev_vandermonde
    u = np.linspace(-0.9, 0.9, 13)
    T_default = chebyshev_vandermonde(u, max_k=5)
    T_xp_np = chebyshev_vandermonde(u, max_k=5, xp=np)
    # Same shape contract -- stacked array, not list-of-arrays.
    assert T_default.shape == T_xp_np.shape == (6, 13)
    np.testing.assert_allclose(T_default, T_xp_np, rtol=0.0, atol=0.0)


def test_xp_alias_chebyshev_vandermonde_xp_matches_canonical():
    """The legacy ``_chebyshev_vandermonde_xp`` shim on the JAX-twin
    forwards to the new canonical implementation via the xp kwarg."""
    from lumenairy._math.chebyshev import chebyshev_vandermonde
    from lumenairy.propagators.asymptotic_jax_twin import (
        _chebyshev_vandermonde_xp,
    )
    u = np.linspace(-0.9, 0.9, 9)
    shim = _chebyshev_vandermonde_xp(u, 4, np)
    canon = chebyshev_vandermonde(u, 4, xp=np)
    np.testing.assert_array_equal(shim, canon)


# ---------------------------------------------------------------------------
# Pin 7 -- xp-dispatched path stays traceable on JAX (optional)
# ---------------------------------------------------------------------------

def test_xp_jax_traceable():
    """JAX backend: result is a jax array and survives jax.jit."""
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    from lumenairy._math.chebyshev import chebyshev_vandermonde

    u_jax = jnp.linspace(-0.8, 0.8, 7)
    T_jax = chebyshev_vandermonde(u_jax, max_k=4, xp=jnp)
    # dtype is a jax dtype (i.e. float, not object)
    assert jnp.issubdtype(T_jax.dtype, jnp.floating)
    assert T_jax.shape == (5, 7)

    # jit-compile -- if the impl had Python control flow on the array,
    # this would raise TracerBoolConversionError or similar.
    jitted = jax.jit(lambda u: chebyshev_vandermonde(u, max_k=4, xp=jnp))
    T_jit = jitted(u_jax)
    # Match the NumPy reference to floating tolerance.
    T_ref = chebyshev_vandermonde(np.asarray(u_jax), max_k=4)
    np.testing.assert_allclose(np.asarray(T_jit), T_ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Pin 8 -- v5.2.5 (AUDIT_V5_2_3 P3-F1-3): derivative + second-derivative
# helpers honor the ``xp=`` kwarg the same way ``chebyshev_vandermonde``
# does.  Pre-v5.2.5 they ignored / didn't accept ``xp=``; this pin
# enforces the symmetric API.
# ---------------------------------------------------------------------------

def test_chebyshev_derivative_vandermonde_xp_dispatch():
    """First-derivative helper: xp=np matches default; xp=jnp matches
    under jax.jit when JAX is importable."""
    from lumenairy._math.chebyshev import chebyshev_derivative_vandermonde
    u = np.linspace(-0.9, 0.9, 13)
    Tp_default = chebyshev_derivative_vandermonde(u, max_k=5)
    Tp_xp_np = chebyshev_derivative_vandermonde(u, max_k=5, xp=np)
    # xp=np must be bit-identical to the default branch.
    assert Tp_default.shape == Tp_xp_np.shape == (6, 13)
    np.testing.assert_array_equal(Tp_default, Tp_xp_np)

    # max_k = 0 edge case: single all-zero row.
    Tp0 = chebyshev_derivative_vandermonde(u, max_k=0, xp=np)
    assert Tp0.shape == (1, 13)
    np.testing.assert_array_equal(Tp0, np.zeros((1, 13)))

    # Optional JAX leg.
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    u_jax = jnp.linspace(-0.9, 0.9, 13)
    Tp_jax = chebyshev_derivative_vandermonde(u_jax, max_k=5, xp=jnp)
    assert jnp.issubdtype(Tp_jax.dtype, jnp.floating)
    assert Tp_jax.shape == (6, 13)
    # jit-compile round-trip.
    jitted = jax.jit(
        lambda u: chebyshev_derivative_vandermonde(u, max_k=5, xp=jnp)
    )
    Tp_jit = jitted(u_jax)
    # JAX defaults to float32 unless x64 is enabled; loosen tolerance
    # accordingly.  Derivative magnitudes scale as n * U_{n-1}, so the
    # absolute error grows with max_k.
    np.testing.assert_allclose(
        np.asarray(Tp_jit), Tp_default, rtol=1e-4, atol=1e-4,
    )


def test_chebyshev_second_derivative_vandermonde_xp_dispatch():
    """Second-derivative helper: xp=np matches default; xp=jnp matches
    under jax.jit when JAX is importable."""
    from lumenairy._math.chebyshev import (
        chebyshev_second_derivative_vandermonde,
    )
    u = np.linspace(-0.85, 0.85, 11)
    Tpp_default = chebyshev_second_derivative_vandermonde(u, max_k=5)
    Tpp_xp_np = chebyshev_second_derivative_vandermonde(u, max_k=5, xp=np)
    # xp=np must be bit-identical to the default branch.
    assert Tpp_default.shape == Tpp_xp_np.shape == (6, 11)
    np.testing.assert_array_equal(Tpp_default, Tpp_xp_np)

    # max_k < 2 edge cases: all-zero rows.
    Tpp0 = chebyshev_second_derivative_vandermonde(u, max_k=0, xp=np)
    assert Tpp0.shape == (1, 11)
    np.testing.assert_array_equal(Tpp0, np.zeros((1, 11)))
    Tpp1 = chebyshev_second_derivative_vandermonde(u, max_k=1, xp=np)
    assert Tpp1.shape == (2, 11)
    np.testing.assert_array_equal(Tpp1, np.zeros((2, 11)))

    # Optional JAX leg.
    jax = pytest.importorskip('jax')
    jnp = jax.numpy
    u_jax = jnp.linspace(-0.85, 0.85, 11)
    Tpp_jax = chebyshev_second_derivative_vandermonde(u_jax, max_k=5, xp=jnp)
    assert jnp.issubdtype(Tpp_jax.dtype, jnp.floating)
    assert Tpp_jax.shape == (6, 11)
    # jit-compile round-trip.
    jitted = jax.jit(
        lambda u: chebyshev_second_derivative_vandermonde(u, max_k=5, xp=jnp)
    )
    Tpp_jit = jitted(u_jax)
    # JAX defaults to float32 unless x64 is enabled; loosen tolerance
    # accordingly.  Second-derivative magnitudes scale faster than the
    # first derivative (O(n^3) at endpoints), so allow more slack.
    np.testing.assert_allclose(
        np.asarray(Tpp_jit), Tpp_default, rtol=1e-3, atol=1e-3,
    )
