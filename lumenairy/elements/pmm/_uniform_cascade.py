"""
lumenairy.elements.pmm._uniform_cascade -- the laterally-uniform-layer fast
path of :class:`~lumenairy.elements.pmm.stack.PMMStack`'s vertical cascade.

Two pieces, both exact rewrites (in exact arithmetic) of what the dense path
computes; in floating point they agree with it to the grid's round-off --
<= 5e-12 on Chebyshev / single-element grids, about 1e-6 at the default
geometric floor, where the dense path carries that much itself:

1. **Uniform-layer modes from the shared geometric eig.**  For a layer whose
   permittivity is the same isotropic scalar ``eps`` in every union cell, the
   coupled operator of
   :func:`~lumenairy.elements.pmm._core._sem_modes_tensor` collapses exactly to
   ``eps * I - blockdiag(Kx2, Kx2)`` (see
   :func:`~lumenairy.elements.pmm._core._uniform_geo_eig`), so ONE ``n x n`` eig
   of the eps-free geometric operator ``Kx2`` serves every uniform medium on a
   grid -- the half-spaces and every uniform layer -- instead of one ``2n x 2n``
   eig per layer.  The modes are those of
   :func:`~lumenairy.elements.pmm._core._sem_modes_uniform` (same forward
   selector, same small-eigenvalue clamp); only the eigenvector gauge differs
   from a raw ``2n`` eig, which is physically equivalent.

2. **Diagonal run collapse.**  Every uniform medium on one grid shares the
   eigenvector matrix ``W = blockdiag(w, w)``, so between two of them the
   interface mode-match ``a = Wb^-1 Wa`` is the identity and ``b = Vb^-1 Va``
   is DIAGONAL, and so is every propagation S-matrix.  A run of consecutive
   uniform media (the superstrate and substrate included) is therefore cascaded
   with elementwise arithmetic and enters the dense Redheffer cascade as one
   S-matrix.

Why ``b`` is diagonal.  With ``safe = where(|lam| < 1e-12, 1e-12, lam)`` (the
clamp :func:`_sem_modes_uniform` applies when it forms ``V``) and
``q2 = eps - mu`` (``mu`` the geometric spectrum), the modal partner of a
uniform medium is ``V = blockdiag(w, w) @ [[0, diag(D_y)], [diag(D_x), 0]]``
with ``D_x = -eps / safe[:n]`` (the ``E_x``-type columns) and
``D_y = q2 / safe[n:]`` (the ``E_y``-type columns), hence
``Vb^-1 Va = diag(D_x,a / D_x,b, D_y,a / D_y,b)``.  The interface S-matrix of
:func:`~lumenairy.elements.pmm._core._interface_smatrix` with ``a = 1`` is then
``(-(1-b)/(1+b), 2/(1+b), 2b/(1+b), (1-b)/(1+b))`` elementwise (its third block
``0.5 (apb - amb apb^-1 amb)`` simplified), propagation by ``k0 t`` multiplies
by ``X = exp(-lam k0 t)``, and the Redheffer star of two diagonal S-matrices is
the elementwise form of :func:`~lumenairy.elements.rcwa._core._redheffer_star`.
All three are the dense kernels' own formulas, run on the diagonals:
``_core._interface_from_ab``, ``rcwa._core._star_from_blocks`` and
``rcwa._core._propagation_scale`` take the array operations as arguments, so
this module carries no copy of them.

Every function returns ``None`` (the caller then takes the dense path) where the
elementwise form is not finite -- e.g. ``1 + b = 0`` -- so nothing is ever
returned that the dense path would not also produce.  The dense path's
``_guarded_inverse`` screens are trivially passed by a diagonal matrix, so no
guard is lost.

Public ``exp(-i w t)`` gauge throughout (the convention of the classical
PMMStack cascade).
"""
from __future__ import annotations

import operator

import numpy as np

from ...backend import is_jax_array
from ..rcwa._core import _propagation_scale, _star_from_blocks
from ._core import _interface_from_ab, _safe_lam

__all__ = []                      # private module

#: The label the mode-cut diagnostics record for a uniform LAYER solved from the
#: shared geometric eig (the half-spaces keep ``_sem_modes_uniform``'s default).
UNIFORM_LAYER_LABEL = "PMM modal solve (uniform layer)"

# The small-eigenvalue clamp is the dense modal solvers' own
# (``_core._LAM_CLAMP`` / ``_core._safe_lam``): the diagonal interface MUST use
# the same clamped value, or a uniform layer sitting on a Rayleigh anomaly of
# its own medium (``lam = 0``) divides by zero.


def _uniform_iso_eps(eps_u):
    """The scalar permittivity of a laterally uniform ISOTROPIC layer, else
    ``None``.

    ``eps_u`` is the layer's per-union-cell list of permittivities (``(3, 3)``
    tensors or scalars).  Returns ``e`` (a Python complex) when every entry is
    EXACTLY equal (``np.array_equal``) and equal to ``e * I3``; ``None`` for
    anything else -- anisotropic or off-diagonal tensors, a traced (JAX) or
    callable entry, an unresolved material key, a non-finite value.  Exact
    equality is deliberate: a layer the dense path would solve as patterned is
    never re-classified by a tolerance."""
    if eps_u is None or len(eps_u) == 0:
        return None
    first = None
    for e in eps_u:
        if callable(e) or isinstance(e, (str, bytes)) or is_jax_array(e):
            return None
        try:
            M = np.asarray(e, dtype=complex)
        except (TypeError, ValueError):
            return None
        if M.ndim == 0:
            M = M * np.eye(3, dtype=complex)
        if M.shape != (3, 3):
            return None
        if first is None:
            first = M
        elif not np.array_equal(M, first):
            return None
    e0 = first[0, 0]
    if not np.isfinite(e0):
        return None
    if not np.array_equal(first, e0 * np.eye(3, dtype=complex)):
        return None
    return complex(e0)


def _uniform_medium_diag(eps, lam, mu, xp=np):
    """The diagonal ``D`` (length ``2n``) of a uniform medium's modal partner
    relative to the shared eigenvector matrix (see the module docstring):
    ``D[:n] = -eps / safe[:n]``, ``D[n:] = (eps - mu) / safe[n:]``.

    ``lam`` is the medium's modal ``lam`` as returned by
    :func:`_sem_modes_uniform` (forward selector applied) and ``mu`` the
    geometric spectrum it was built from; ``eps - mu`` is the same ``q2`` that
    function multiplies into ``V``.  ``xp`` selects the array module (the
    JAX twin passes ``jax.numpy`` and a possibly traced ``eps``)."""
    n = mu.shape[0]
    if xp is not np:
        safe = _safe_lam(lam, xp)
        return xp.concatenate([-eps / safe[:n], (eps - mu) / safe[n:]])
    eps = complex(eps)
    safe = _safe_lam(lam)
    D = np.empty(2 * n, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        D[:n] = -eps / safe[:n]
        D[n:] = (eps - mu) / safe[n:]
    return D


def _diag_interface(Da, Db, xp=np):
    """Elementwise interface S-matrix from medium ``a`` into medium ``b`` (both
    uniform on one grid), or ``None`` when it is not finite (NumPy; a traced
    value cannot be screened, the JAX twin's caller runs no fallback)."""
    if xp is not np:
        b = Da / Db
        return _interface_from_ab(1.0 + b, 1.0 - b, xp.reciprocal,
                                  xp.multiply)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b = Da / Db
        # a = 1 between two uniform media on one grid: the dense formula,
        # elementwise (one implementation, _core._interface_from_ab)
        S = _interface_from_ab(1.0 + b, 1.0 - b, np.reciprocal, np.multiply)
    if not all(np.all(np.isfinite(x)) for x in S):
        return None
    return S


def _diag_propagate(S, lam, k0_t, xp=np):
    """``S`` starred with the propagation S-matrix of ``(lam, k0 t)``: the
    dense kernel's own formula (``rcwa._core._propagation_scale``) on the
    diagonals."""
    if xp is not np:
        return _propagation_scale(S, xp.exp(-lam * k0_t))
    with np.errstate(over="ignore", invalid="ignore"):
        X = np.exp(-lam * k0_t)
    return _propagation_scale(S, X)


def _elementwise_div(A, M, what):
    return A / M


def _diag_star(A, B, xp=np):
    """Elementwise Redheffer star of two diagonal S-matrices -- the dense
    kernel's own formula (``rcwa._core._star_from_blocks``) with elementwise
    products and division -- or ``None`` when a denominator vanishes or the
    result is not finite (NumPy only)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        S = _star_from_blocks(A, B, 1.0, operator.mul, _elementwise_div)
    if xp is not np:
        return S
    if not all(np.all(np.isfinite(x)) for x in S):
        return None
    return S


def _diag_run(D_list, lam_list, k0_t_list, xp=np):
    """The S-matrix of one RUN of consecutive uniform media, elementwise.

    ``D_list`` / ``lam_list`` hold the media in cascade order (``m >= 2``);
    ``k0_t_list[j]`` is ``k0 * thickness`` of medium ``j`` for ``j >= 1`` (the
    first medium's propagation belongs to the cascade BEFORE the run), or
    ``None`` for a medium that is not propagated through (the substrate).  The
    run is ``iface(0, 1), prop(1), iface(1, 2), prop(2), ...`` -- exactly the
    order the dense loop applies.  Returns the four diagonals, or ``None`` when
    any step is not finite."""
    S = _diag_interface(D_list[0], D_list[1], xp)
    if S is None:
        return None
    for j in range(1, len(D_list)):
        if j > 1:
            Si = _diag_interface(D_list[j - 1], D_list[j], xp)
            if Si is None:
                return None
            S = _diag_star(S, Si, xp)
            if S is None:
                return None
        if k0_t_list[j] is not None:
            S = _diag_propagate(S, lam_list[j], k0_t_list[j], xp)
    if xp is not np:
        return S
    if not all(np.all(np.isfinite(x)) for x in S):
        return None
    return S


def _cascade_vertical(media, k0_t, *, interface, star, propagation_star,
                      collapse=True, stats=None, xp=np):
    """The symmetric (``+/-q``) Redheffer cascade ``sup -> layers -> sub`` with
    every run of consecutive uniform media collapsed into one diagonal
    S-matrix.

    ``media`` is the cascade-ordered list ``[sup, layer_1, ..., layer_L, sub]``;
    each entry is ``(W, V, lam, D)`` with ``D`` the diagonal of
    :func:`_uniform_medium_diag` for a uniform medium on the SHARED basis, or
    ``None`` for a patterned layer.  ``k0_t[i]`` is ``k0 * thickness`` of
    ``layer_{i+1}``.  ``interface`` / ``star`` / ``propagation_star`` are the
    dense kernels the caller uses (passed in so this module carries no copy of
    them).  ``collapse=False`` reproduces the plain per-layer loop.

    A run whose elementwise cascade is not finite is cascaded by the dense
    kernels instead, layer by layer, so the result never depends on the fast
    path having succeeded.  ``stats`` (a dict, optional) counts the collapsed
    runs and the dense fallbacks for the tests.  ``xp`` selects the array
    module: the JAX twin runs this same cascade with ``jax.numpy`` (no
    finiteness screen under a trace, so no fallback there)."""
    nmed = len(media)

    def _k0t(j):                 # medium j's propagation (None for sub)
        return k0_t[j - 1] if 1 <= j <= nmed - 2 else None

    S = None

    def _append(Si):
        return Si if S is None else star(S, Si)

    j = 1
    while j < nmed:
        uni_pair = media[j - 1][3] is not None and media[j][3] is not None
        if collapse and uni_pair:
            end = j
            while end + 1 < nmed and media[end + 1][3] is not None:
                end += 1
            run = list(range(j - 1, end + 1))
            Sd = _diag_run([media[m][3] for m in run],
                           [media[m][2] for m in run],
                           [None] + [_k0t(m) for m in run[1:]], xp)
            if Sd is not None:
                S = _append(tuple(xp.diag(v) for v in Sd))
                if stats is not None:
                    stats["runs"] = stats.get("runs", 0) + 1
                    stats["collapsed_media"] = (stats.get("collapsed_media", 0)
                                                + len(run))
                j = end + 1
                continue
            if stats is not None:
                stats["fallback"] = stats.get("fallback", 0) + 1
            # dense fallback for this run, layer by layer
            for m in range(j, end + 1):
                Wa, Va = media[m - 1][0], media[m - 1][1]
                Wb, Vb, lam_b = media[m][0], media[m][1], media[m][2]
                S = _append(interface(Wa, Va, Wb, Vb))
                if _k0t(m) is not None:
                    S = propagation_star(S, lam_b, _k0t(m))
            j = end + 1
            continue
        Wa, Va = media[j - 1][0], media[j - 1][1]
        Wb, Vb, lam_b = media[j][0], media[j][1], media[j][2]
        S = _append(interface(Wa, Va, Wb, Vb))
        if _k0t(j) is not None:
            S = propagation_star(S, lam_b, _k0t(j))
        j += 1
    return S
