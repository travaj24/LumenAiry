"""
Chebyshev Vandermonde tables -- shared math primitives.

This module hosts the three Chebyshev helpers that the canonical
asymptotic propagator and the Maslov-corrected real-lens kernel both
need.  They were originally inlined in
``lumenairy.elements.lenses`` (around line 722 in v5.1.x) because
the Maslov machinery used to live there; the asymptotic propagator
then imported them from ``elements/``, creating an inverted
dependency where ``propagators/`` reached into ``elements/`` for a
math primitive.

v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
The three helpers move here, into a propagator-free math root.
``elements/lenses.py`` and every ``propagators/asymptotic*.py``
consumer now imports from ``lumenairy._math.chebyshev`` directly.
Underscore-prefixed back-compat aliases are preserved at the old
import site so external callers keep working.

The first-kind Chebyshev polynomial T_n(u) satisfies::

    T_0(u) = 1
    T_1(u) = u
    T_{n+1}(u) = 2 u T_n(u) - T_{n-1}(u)

Its derivative obeys ``T_n'(u) = n U_{n-1}(u)`` where U is the
Chebyshev polynomial of the second kind (same recurrence with
seed ``U_0 = 1, U_1 = 2u``).  The second derivative is computed by
differentiating the T-recurrence directly to avoid the singular
``1/(u^2 - 1)`` factor at the endpoints.

All three helpers return a stacked array of shape
``(max_k + 1,) + u.shape`` indexed by polynomial order along axis 0.
The default backend is NumPy.  Pass ``xp=jax.numpy`` (or any other
array-API module) to evaluate on a non-NumPy backend; in that mode a
functional-style construction is used so ``jax.jit`` / ``jax.grad``
can trace through it.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = [
    'chebyshev_vandermonde',
    'chebyshev_derivative_vandermonde',
    'chebyshev_second_derivative_vandermonde',
]


def chebyshev_vandermonde(u: np.ndarray, max_k: int,
                          xp: Optional[Any] = None) -> np.ndarray:
    """
    Build the Chebyshev Vandermonde-like array T[n](u) for n = 0..max_k.

    Parameters
    ----------
    u : ndarray, any shape, values in [-1, 1]
    max_k : int
    xp : array-API module, optional
        Backend to evaluate on.  ``None`` (default) uses NumPy with the
        original in-place stacked-array construction.  Pass
        ``jax.numpy`` (or any other array-API module) to evaluate on a
        non-NumPy backend; a functional-style construction is used so
        the result is traceable / differentiable.

    Returns
    -------
    T : ndarray of shape (max_k+1,) + u.shape
        T[n] is T_n(u), computed by the standard 3-term recurrence.
    """
    if xp is None or xp is np:
        # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
        # original NumPy path, moved verbatim from elements/lenses.py.
        u = np.asarray(u)
        T = np.empty((max_k + 1,) + u.shape, dtype=np.float64)
        T[0] = 1.0
        if max_k >= 1:
            T[1] = u
        for n in range(2, max_k + 1):
            T[n] = 2.0 * u * T[n - 1] - T[n - 2]
        return T

    # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
    # xp-dispatched path, moved verbatim from propagators/asymptotic_jax_twin.py
    # (formerly ``_chebyshev_vandermonde_xp``).  Returns a stacked array
    # via xp.stack, so the caller sees the same shape contract as the
    # NumPy path regardless of backend.  Functional construction (no
    # in-place writes) keeps this jax.jit / jax.grad traceable.
    u_arr = xp.asarray(u)
    T = [xp.ones_like(u_arr)]
    if max_k >= 1:
        T.append(u_arr)
    for n in range(2, max_k + 1):
        T.append(2.0 * u_arr * T[n - 1] - T[n - 2])
    return xp.stack(T)


def chebyshev_derivative_vandermonde(u: np.ndarray, max_k: int
                                     ) -> np.ndarray:
    """
    Build T_n'(u) for n = 0..max_k.  Uses T_n'(x) = n * U_{n-1}(x),
    where U is the Chebyshev polynomial of the second kind.

    Returns
    -------
    Tp : ndarray of shape (max_k+1,) + u.shape
    """
    u = np.asarray(u)
    Tp = np.zeros((max_k + 1,) + u.shape, dtype=np.float64)
    if max_k < 1:
        return Tp
    # U_0(x) = 1, U_1(x) = 2x, U_{n+1} = 2x U_n - U_{n-1}
    U = np.empty((max_k + 1,) + u.shape, dtype=np.float64)
    U[0] = 1.0
    if max_k >= 1:
        U[1] = 2.0 * u
    for n in range(2, max_k + 1):
        U[n] = 2.0 * u * U[n - 1] - U[n - 2]
    # T_n'(x) = n * U_{n-1}(x)  for n >= 1
    for n in range(1, max_k + 1):
        Tp[n] = float(n) * U[n - 1]
    return Tp


def chebyshev_second_derivative_vandermonde(u: np.ndarray, max_k: int
                                            ) -> np.ndarray:
    """
    Build T_n''(u) for n = 0..max_k.

    T''_n(x) can be derived from T_n and U_n via the identity
        T''_n(x) = n * ((n+1) T_n(x) - U_n(x)) / (x^2 - 1)   (x != +/- 1)
    but this has singular denominators at the endpoints.  A more stable
    recurrence is obtained by differentiating the standard T recurrence
    once more:
        T''_0 = 0,  T''_1 = 0,  T''_2 = 4,
        T''_{n+1} = 2 x T''_n + 4 T'_n - T''_{n-1}

    Uses the same 3-term recurrence style as the first-derivative
    helper, so the cost is O(max_k) per evaluation point.

    Returns
    -------
    Tpp : ndarray of shape (max_k+1,) + u.shape
    """
    u = np.asarray(u)
    shape = u.shape
    Tpp = np.zeros((max_k + 1,) + shape, dtype=np.float64)
    if max_k < 2:
        return Tpp
    # We'll need T'_n to drive the recurrence
    Tp = chebyshev_derivative_vandermonde(u, max_k)
    # T''_0 = 0, T''_1 = 0, T''_2 = 4 (constant)
    Tpp[2] = 4.0 * np.ones(shape, dtype=np.float64)
    for n in range(2, max_k):
        Tpp[n + 1] = 2.0 * u * Tpp[n] + 4.0 * Tp[n] - Tpp[n - 1]
    return Tpp
