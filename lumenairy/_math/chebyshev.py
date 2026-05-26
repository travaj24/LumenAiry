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
    'chebyshev_fit_2d',
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


def chebyshev_derivative_vandermonde(u: np.ndarray, max_k: int,
                                     xp: Optional[Any] = None
                                     ) -> np.ndarray:
    """
    Build T_n'(u) for n = 0..max_k.  Uses T_n'(x) = n * U_{n-1}(x),
    where U is the Chebyshev polynomial of the second kind.

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
    Tp : ndarray of shape (max_k+1,) + u.shape
    """
    if xp is None or xp is np:
        # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
        # original NumPy path, moved verbatim from elements/lenses.py.
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

    # v5.2.5 (AUDIT_V5_2_3 P3-F1-3 chebyshev derivative xp dispatch):
    # xp-dispatched path mirrors chebyshev_vandermonde's JAX-friendly
    # functional construction.  Build U_0..U_{max_k} via list-append on
    # the 3-term recurrence, then form T'_n = n * U_{n-1} for n >= 1.
    # Returns a stacked array via xp.stack so the caller sees the same
    # shape contract as the NumPy path.  No in-place writes keeps this
    # jax.jit / jax.grad traceable.
    u_arr = xp.asarray(u)
    if max_k < 1:
        # Single zero row at order 0.
        return xp.stack([xp.zeros_like(u_arr)])
    # v5.3 (AUDIT_V5_2_5 P3-5): the ``if max_k >= 1`` guard above
    # was dead code -- we already early-returned at ``max_k < 1``
    # so by this point ``max_k >= 1`` is guaranteed.  Removed the
    # redundant check.
    # U_0(x) = 1, U_1(x) = 2x, U_{n+1} = 2x U_n - U_{n-1}
    U = [xp.ones_like(u_arr), 2.0 * u_arr]
    for n in range(2, max_k + 1):
        U.append(2.0 * u_arr * U[n - 1] - U[n - 2])
    # T'_0 = 0, T'_n = n * U_{n-1} for n >= 1.
    Tp_rows = [xp.zeros_like(u_arr)]
    for n in range(1, max_k + 1):
        Tp_rows.append(float(n) * U[n - 1])
    return xp.stack(Tp_rows)


def chebyshev_second_derivative_vandermonde(u: np.ndarray, max_k: int,
                                            xp: Optional[Any] = None
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
    Tpp : ndarray of shape (max_k+1,) + u.shape
    """
    if xp is None or xp is np:
        # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
        # original NumPy path, moved verbatim from elements/lenses.py.
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

    # v5.2.5 (AUDIT_V5_2_3 P3-F1-3 chebyshev derivative xp dispatch):
    # xp-dispatched path -- functional list-of-rows construction so
    # jax.jit / jax.grad can trace through it.  Recurrence is identical
    # to the NumPy branch: T''_0 = 0, T''_1 = 0, T''_2 = 4,
    # T''_{n+1} = 2 x T''_n + 4 T'_n - T''_{n-1}.
    u_arr = xp.asarray(u)
    if max_k < 2:
        # All-zero rows at orders 0..max_k.
        zero = xp.zeros_like(u_arr)
        rows = [zero for _ in range(max_k + 1)]
        return xp.stack(rows)
    # Need T'_n on the same backend to drive the recurrence.
    Tp = chebyshev_derivative_vandermonde(u_arr, max_k, xp=xp)
    zero = xp.zeros_like(u_arr)
    Tpp_rows = [zero, zero, 4.0 * xp.ones_like(u_arr)]
    for n in range(2, max_k):
        Tpp_rows.append(
            2.0 * u_arr * Tpp_rows[n] + 4.0 * Tp[n] - Tpp_rows[n - 1]
        )
    return xp.stack(Tpp_rows)


def chebyshev_fit_2d(x: np.ndarray,
                     y: np.ndarray,
                     z: np.ndarray,
                     n_max_x: int = 8,
                     n_max_y: int = 8,
                     *,
                     weight: Optional[np.ndarray] = None,
                     normalize_xy: bool = True,
                     return_residual: bool = False):
    """Fit a 2-D height-map z(x, y) to first-kind Chebyshev polynomials.

    Solves the least-squares problem

        z(x_p, y_p) ~ sum_{i=0..n_max_x, j=0..n_max_y} c_{ij}
                          T_i(x_p / a) T_j(y_p / b)

    for the coefficients ``c_{ij}`` and returns them in the
    ``{(i, j): c_ij}`` dict format consumed by
    ``lumenairy.elements.freeform.surface_sag_chebyshev``.

    v5.4 Phase 5: this helper was hoisted out of
    ``lumenairy/ui/chebyshev_fit_dock.py``'s inline P3-C
    implementation so the library now ships a single, tested fit
    primitive instead of leaving each UI to roll its own
    ``chebvander2d`` + ``lstsq`` block.

    Parameters
    ----------
    x, y : ndarray
        1-D coordinate arrays.  Must form a regular grid (the
        2-D meshgrid is built internally with ``indexing='xy'``).
    z : ndarray
        2-D height map of shape ``(len(y), len(x))``.
    n_max_x, n_max_y : int, optional
        Maximum polynomial degree in each dimension (default 8).
        The fit basis is the tensor product ``T_i(x) T_j(y)`` for
        ``0 <= i <= n_max_x`` and ``0 <= j <= n_max_y``.
    weight : ndarray, optional
        Optional 2-D weight / mask array of the same shape as ``z``.
        Pixels with ``weight <= 0`` are excluded from the LS solve;
        positive entries are interpreted as variance weights
        (``sqrt(w)`` rows are scaled into the normal equations).
        Non-finite ``z`` entries are also excluded automatically.
    normalize_xy : bool, optional
        If True (default), rescale ``x`` and ``y`` to the canonical
        Chebyshev domain ``[-1, 1]`` before evaluating the basis.
        Required for Chebyshev orthogonality on a finite aperture;
        set False only if the caller has already normalised ``x``,
        ``y`` themselves.
    return_residual : bool, optional
        If True, also return the residual ``z - z_fit`` evaluated on
        the original grid (NaN where ``weight <= 0`` so the masked
        regions are clearly excluded from PV / RMS statistics).

    Returns
    -------
    coeffs : dict of {(int, int): float}
        Sparse coefficient dictionary; entries with magnitude below
        ``1e-12`` are dropped.  v5.4.1 tightened this from ``1e-15``
        (ULP noise) to ``1e-12`` (matches the test-tolerance band;
        eliminates ~15 spurious near-zero coefficients on constant-z
        fits).  Keys are ``(i, j)`` with ``i`` the x-order and ``j``
        the y-order, matching the convention of
        ``surface_sag_chebyshev``.
    residual : ndarray, only if ``return_residual=True``
        ``z - z_fit`` on the original ``(len(y), len(x))`` grid.
    """
    # v5.4 Phase 5: lstsq-based 2-D Chebyshev fit promoted out of the
    # UI dock so external callers (notebook scripts, batch metrology
    # tools, regression tests) can share a single tested entry point.
    # Implementation uses numpy.polynomial.chebyshev.chebvander2d to
    # build the design matrix -- exactly the basis that
    # ``surface_sag_chebyshev`` evaluates -- so the emitted
    # ``{(i, j): c_ij}`` dict round-trips through the library's own
    # evaluator without an extra scaling step.
    from numpy.polynomial.chebyshev import chebval2d, chebvander2d

    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError(
            f"chebyshev_fit_2d: z must be 2-D; got shape {z.shape}.")
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"chebyshev_fit_2d: z shape {z.shape} must equal "
            f"(len(y), len(x)) = ({y.size}, {x.size}).")

    # Normalise coordinates to [-1, 1] so the Chebyshev basis is
    # orthogonal on the sampled aperture.  Guard against degenerate
    # zero-extent axes (would divide by zero below).
    if normalize_xy:
        x_span = float(x.max() - x.min())
        y_span = float(y.max() - y.min())
        if x_span == 0.0 or y_span == 0.0:
            raise ValueError(
                "chebyshev_fit_2d: cannot normalise -- x or y has "
                "zero extent.  Pass normalize_xy=False if you have "
                "already scaled the coordinates.")
        x_norm = 2.0 * (x - x.min()) / x_span - 1.0
        y_norm = 2.0 * (y - y.min()) / y_span - 1.0
    else:
        x_norm, y_norm = x, y

    X, Y = np.meshgrid(x_norm, y_norm, indexing='xy')
    # chebvander2d builds the design matrix of shape
    # (M, (n_x+1)*(n_y+1)) in row-major (i then j) order.
    V = chebvander2d(X.ravel(), Y.ravel(), [n_max_x, n_max_y])
    z_flat = z.ravel()

    # Drop non-finite samples up front so the LS solve never sees a
    # NaN; the caller's residual is filled with NaN at those pixels.
    finite = np.isfinite(z_flat)
    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        if w.shape != z.shape:
            raise ValueError(
                f"chebyshev_fit_2d: weight shape {w.shape} does not "
                f"match z shape {z.shape}.")
        w_flat = w.ravel()
        keep = finite & (w_flat > 0)
        sqrt_w = np.sqrt(w_flat[keep])
        V_solve = V[keep] * sqrt_w[:, None]
        z_solve = z_flat[keep] * sqrt_w
    else:
        keep = finite
        V_solve = V[keep]
        z_solve = z_flat[keep]

    n_terms = (n_max_x + 1) * (n_max_y + 1)
    if int(keep.sum()) < n_terms:
        raise ValueError(
            f"chebyshev_fit_2d: not enough valid samples "
            f"({int(keep.sum())}) to fit ({n_max_x}+1) x "
            f"({n_max_y}+1) = {n_terms} coefficients.")

    coeffs_flat, _, _, _ = np.linalg.lstsq(
        V_solve, z_solve, rcond=None)
    # chebvander2d packs columns in (i, j) row-major order, so the
    # natural reshape is (n_x+1, n_y+1) with axis 0 = x-order.
    coeffs_2d = coeffs_flat.reshape(n_max_x + 1, n_max_y + 1)

    # v5.4.1 (audit P3 #8): tighten prune threshold from 1e-15 (ULP
    # noise) to 1e-12 (matches the test-tolerance band; eliminates
    # ~15 spurious near-zero coefficients on constant-z fits).
    coeffs_dict = {
        (i, j): float(coeffs_2d[i, j])
        for i in range(n_max_x + 1)
        for j in range(n_max_y + 1)
        if abs(coeffs_2d[i, j]) > 1e-12
    }

    if return_residual:
        z_fit = chebval2d(X, Y, coeffs_2d).reshape(z.shape)
        residual = z - z_fit
        if weight is not None:
            # Match the dock convention: blank out excluded pixels
            # with NaN so PV / RMS statistics see only the fit
            # support.
            mask_2d = (np.asarray(weight, dtype=np.float64) > 0)
            residual = np.where(mask_2d, residual, np.nan)
        return coeffs_dict, residual
    return coeffs_dict
